
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: 
4: Auxiliary functions for f2py2e.
5: 
6: Copyright 1999,2000 Pearu Peterson all rights reserved,
7: Pearu Peterson <pearu@ioc.ee>
8: Permission to use, modify, and distribute this software is given under the
9: terms of the NumPy (BSD style) LICENSE.
10: 
11: 
12: NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
13: $Date: 2005/07/24 19:01:55 $
14: Pearu Peterson
15: 
16: '''
17: from __future__ import division, absolute_import, print_function
18: 
19: import pprint
20: import sys
21: import types
22: from functools import reduce
23: 
24: from . import __version__
25: from . import cfuncs
26: 
27: __all__ = [
28:     'applyrules', 'debugcapi', 'dictappend', 'errmess', 'gentitle',
29:     'getargs2', 'getcallprotoargument', 'getcallstatement',
30:     'getfortranname', 'getpymethoddef', 'getrestdoc', 'getusercode',
31:     'getusercode1', 'hasbody', 'hascallstatement', 'hascommon',
32:     'hasexternals', 'hasinitvalue', 'hasnote', 'hasresultnote',
33:     'isallocatable', 'isarray', 'isarrayofstrings', 'iscomplex',
34:     'iscomplexarray', 'iscomplexfunction', 'iscomplexfunction_warn',
35:     'isdouble', 'isdummyroutine', 'isexternal', 'isfunction',
36:     'isfunction_wrap', 'isint1array', 'isinteger', 'isintent_aux',
37:     'isintent_c', 'isintent_callback', 'isintent_copy', 'isintent_dict',
38:     'isintent_hide', 'isintent_in', 'isintent_inout', 'isintent_inplace',
39:     'isintent_nothide', 'isintent_out', 'isintent_overwrite', 'islogical',
40:     'islogicalfunction', 'islong_complex', 'islong_double',
41:     'islong_doublefunction', 'islong_long', 'islong_longfunction',
42:     'ismodule', 'ismoduleroutine', 'isoptional', 'isprivate', 'isrequired',
43:     'isroutine', 'isscalar', 'issigned_long_longarray', 'isstring',
44:     'isstringarray', 'isstringfunction', 'issubroutine',
45:     'issubroutine_wrap', 'isthreadsafe', 'isunsigned', 'isunsigned_char',
46:     'isunsigned_chararray', 'isunsigned_long_long',
47:     'isunsigned_long_longarray', 'isunsigned_short',
48:     'isunsigned_shortarray', 'l_and', 'l_not', 'l_or', 'outmess',
49:     'replace', 'show', 'stripcomma', 'throw_error',
50: ]
51: 
52: 
53: f2py_version = __version__.version
54: 
55: 
56: errmess = sys.stderr.write
57: show = pprint.pprint
58: 
59: options = {}
60: debugoptions = []
61: wrapfuncs = 1
62: 
63: 
64: def outmess(t):
65:     if options.get('verbose', 1):
66:         sys.stdout.write(t)
67: 
68: 
69: def debugcapi(var):
70:     return 'capi' in debugoptions
71: 
72: 
73: def _isstring(var):
74:     return 'typespec' in var and var['typespec'] == 'character' and \
75:            not isexternal(var)
76: 
77: 
78: def isstring(var):
79:     return _isstring(var) and not isarray(var)
80: 
81: 
82: def ischaracter(var):
83:     return isstring(var) and 'charselector' not in var
84: 
85: 
86: def isstringarray(var):
87:     return isarray(var) and _isstring(var)
88: 
89: 
90: def isarrayofstrings(var):
91:     # leaving out '*' for now so that `character*(*) a(m)` and `character
92:     # a(m,*)` are treated differently. Luckily `character**` is illegal.
93:     return isstringarray(var) and var['dimension'][-1] == '(*)'
94: 
95: 
96: def isarray(var):
97:     return 'dimension' in var and not isexternal(var)
98: 
99: 
100: def isscalar(var):
101:     return not (isarray(var) or isstring(var) or isexternal(var))
102: 
103: 
104: def iscomplex(var):
105:     return isscalar(var) and \
106:            var.get('typespec') in ['complex', 'double complex']
107: 
108: 
109: def islogical(var):
110:     return isscalar(var) and var.get('typespec') == 'logical'
111: 
112: 
113: def isinteger(var):
114:     return isscalar(var) and var.get('typespec') == 'integer'
115: 
116: 
117: def isreal(var):
118:     return isscalar(var) and var.get('typespec') == 'real'
119: 
120: 
121: def get_kind(var):
122:     try:
123:         return var['kindselector']['*']
124:     except KeyError:
125:         try:
126:             return var['kindselector']['kind']
127:         except KeyError:
128:             pass
129: 
130: 
131: def islong_long(var):
132:     if not isscalar(var):
133:         return 0
134:     if var.get('typespec') not in ['integer', 'logical']:
135:         return 0
136:     return get_kind(var) == '8'
137: 
138: 
139: def isunsigned_char(var):
140:     if not isscalar(var):
141:         return 0
142:     if var.get('typespec') != 'integer':
143:         return 0
144:     return get_kind(var) == '-1'
145: 
146: 
147: def isunsigned_short(var):
148:     if not isscalar(var):
149:         return 0
150:     if var.get('typespec') != 'integer':
151:         return 0
152:     return get_kind(var) == '-2'
153: 
154: 
155: def isunsigned(var):
156:     if not isscalar(var):
157:         return 0
158:     if var.get('typespec') != 'integer':
159:         return 0
160:     return get_kind(var) == '-4'
161: 
162: 
163: def isunsigned_long_long(var):
164:     if not isscalar(var):
165:         return 0
166:     if var.get('typespec') != 'integer':
167:         return 0
168:     return get_kind(var) == '-8'
169: 
170: 
171: def isdouble(var):
172:     if not isscalar(var):
173:         return 0
174:     if not var.get('typespec') == 'real':
175:         return 0
176:     return get_kind(var) == '8'
177: 
178: 
179: def islong_double(var):
180:     if not isscalar(var):
181:         return 0
182:     if not var.get('typespec') == 'real':
183:         return 0
184:     return get_kind(var) == '16'
185: 
186: 
187: def islong_complex(var):
188:     if not iscomplex(var):
189:         return 0
190:     return get_kind(var) == '32'
191: 
192: 
193: def iscomplexarray(var):
194:     return isarray(var) and \
195:            var.get('typespec') in ['complex', 'double complex']
196: 
197: 
198: def isint1array(var):
199:     return isarray(var) and var.get('typespec') == 'integer' \
200:         and get_kind(var) == '1'
201: 
202: 
203: def isunsigned_chararray(var):
204:     return isarray(var) and var.get('typespec') in ['integer', 'logical']\
205:         and get_kind(var) == '-1'
206: 
207: 
208: def isunsigned_shortarray(var):
209:     return isarray(var) and var.get('typespec') in ['integer', 'logical']\
210:         and get_kind(var) == '-2'
211: 
212: 
213: def isunsignedarray(var):
214:     return isarray(var) and var.get('typespec') in ['integer', 'logical']\
215:         and get_kind(var) == '-4'
216: 
217: 
218: def isunsigned_long_longarray(var):
219:     return isarray(var) and var.get('typespec') in ['integer', 'logical']\
220:         and get_kind(var) == '-8'
221: 
222: 
223: def issigned_chararray(var):
224:     return isarray(var) and var.get('typespec') in ['integer', 'logical']\
225:         and get_kind(var) == '1'
226: 
227: 
228: def issigned_shortarray(var):
229:     return isarray(var) and var.get('typespec') in ['integer', 'logical']\
230:         and get_kind(var) == '2'
231: 
232: 
233: def issigned_array(var):
234:     return isarray(var) and var.get('typespec') in ['integer', 'logical']\
235:         and get_kind(var) == '4'
236: 
237: 
238: def issigned_long_longarray(var):
239:     return isarray(var) and var.get('typespec') in ['integer', 'logical']\
240:         and get_kind(var) == '8'
241: 
242: 
243: def isallocatable(var):
244:     return 'attrspec' in var and 'allocatable' in var['attrspec']
245: 
246: 
247: def ismutable(var):
248:     return not ('dimension' not in var or isstring(var))
249: 
250: 
251: def ismoduleroutine(rout):
252:     return 'modulename' in rout
253: 
254: 
255: def ismodule(rout):
256:     return 'block' in rout and 'module' == rout['block']
257: 
258: 
259: def isfunction(rout):
260:     return 'block' in rout and 'function' == rout['block']
261: 
262: def isfunction_wrap(rout):
263:     if isintent_c(rout):
264:         return 0
265:     return wrapfuncs and isfunction(rout) and (not isexternal(rout))
266: 
267: 
268: def issubroutine(rout):
269:     return 'block' in rout and 'subroutine' == rout['block']
270: 
271: 
272: def issubroutine_wrap(rout):
273:     if isintent_c(rout):
274:         return 0
275:     return issubroutine(rout) and hasassumedshape(rout)
276: 
277: 
278: def hasassumedshape(rout):
279:     if rout.get('hasassumedshape'):
280:         return True
281:     for a in rout['args']:
282:         for d in rout['vars'].get(a, {}).get('dimension', []):
283:             if d == ':':
284:                 rout['hasassumedshape'] = True
285:                 return True
286:     return False
287: 
288: 
289: def isroutine(rout):
290:     return isfunction(rout) or issubroutine(rout)
291: 
292: 
293: def islogicalfunction(rout):
294:     if not isfunction(rout):
295:         return 0
296:     if 'result' in rout:
297:         a = rout['result']
298:     else:
299:         a = rout['name']
300:     if a in rout['vars']:
301:         return islogical(rout['vars'][a])
302:     return 0
303: 
304: 
305: def islong_longfunction(rout):
306:     if not isfunction(rout):
307:         return 0
308:     if 'result' in rout:
309:         a = rout['result']
310:     else:
311:         a = rout['name']
312:     if a in rout['vars']:
313:         return islong_long(rout['vars'][a])
314:     return 0
315: 
316: 
317: def islong_doublefunction(rout):
318:     if not isfunction(rout):
319:         return 0
320:     if 'result' in rout:
321:         a = rout['result']
322:     else:
323:         a = rout['name']
324:     if a in rout['vars']:
325:         return islong_double(rout['vars'][a])
326:     return 0
327: 
328: 
329: def iscomplexfunction(rout):
330:     if not isfunction(rout):
331:         return 0
332:     if 'result' in rout:
333:         a = rout['result']
334:     else:
335:         a = rout['name']
336:     if a in rout['vars']:
337:         return iscomplex(rout['vars'][a])
338:     return 0
339: 
340: 
341: def iscomplexfunction_warn(rout):
342:     if iscomplexfunction(rout):
343:         outmess('''\
344:     **************************************************************
345:         Warning: code with a function returning complex value
346:         may not work correctly with your Fortran compiler.
347:         Run the following test before using it in your applications:
348:         $(f2py install dir)/test-site/{b/runme_scalar,e/runme}
349:         When using GNU gcc/g77 compilers, codes should work correctly.
350:     **************************************************************\n''')
351:         return 1
352:     return 0
353: 
354: 
355: def isstringfunction(rout):
356:     if not isfunction(rout):
357:         return 0
358:     if 'result' in rout:
359:         a = rout['result']
360:     else:
361:         a = rout['name']
362:     if a in rout['vars']:
363:         return isstring(rout['vars'][a])
364:     return 0
365: 
366: 
367: def hasexternals(rout):
368:     return 'externals' in rout and rout['externals']
369: 
370: 
371: def isthreadsafe(rout):
372:     return 'f2pyenhancements' in rout and \
373:            'threadsafe' in rout['f2pyenhancements']
374: 
375: 
376: def hasvariables(rout):
377:     return 'vars' in rout and rout['vars']
378: 
379: 
380: def isoptional(var):
381:     return ('attrspec' in var and 'optional' in var['attrspec'] and
382:             'required' not in var['attrspec']) and isintent_nothide(var)
383: 
384: 
385: def isexternal(var):
386:     return 'attrspec' in var and 'external' in var['attrspec']
387: 
388: 
389: def isrequired(var):
390:     return not isoptional(var) and isintent_nothide(var)
391: 
392: 
393: def isintent_in(var):
394:     if 'intent' not in var:
395:         return 1
396:     if 'hide' in var['intent']:
397:         return 0
398:     if 'inplace' in var['intent']:
399:         return 0
400:     if 'in' in var['intent']:
401:         return 1
402:     if 'out' in var['intent']:
403:         return 0
404:     if 'inout' in var['intent']:
405:         return 0
406:     if 'outin' in var['intent']:
407:         return 0
408:     return 1
409: 
410: 
411: def isintent_inout(var):
412:     return ('intent' in var and ('inout' in var['intent'] or
413:             'outin' in var['intent']) and 'in' not in var['intent'] and
414:             'hide' not in var['intent'] and 'inplace' not in var['intent'])
415: 
416: 
417: def isintent_out(var):
418:     return 'out' in var.get('intent', [])
419: 
420: 
421: def isintent_hide(var):
422:     return ('intent' in var and ('hide' in var['intent'] or
423:             ('out' in var['intent'] and 'in' not in var['intent'] and
424:                 (not l_or(isintent_inout, isintent_inplace)(var)))))
425: 
426: def isintent_nothide(var):
427:     return not isintent_hide(var)
428: 
429: 
430: def isintent_c(var):
431:     return 'c' in var.get('intent', [])
432: 
433: 
434: def isintent_cache(var):
435:     return 'cache' in var.get('intent', [])
436: 
437: 
438: def isintent_copy(var):
439:     return 'copy' in var.get('intent', [])
440: 
441: 
442: def isintent_overwrite(var):
443:     return 'overwrite' in var.get('intent', [])
444: 
445: 
446: def isintent_callback(var):
447:     return 'callback' in var.get('intent', [])
448: 
449: 
450: def isintent_inplace(var):
451:     return 'inplace' in var.get('intent', [])
452: 
453: 
454: def isintent_aux(var):
455:     return 'aux' in var.get('intent', [])
456: 
457: 
458: def isintent_aligned4(var):
459:     return 'aligned4' in var.get('intent', [])
460: 
461: 
462: def isintent_aligned8(var):
463:     return 'aligned8' in var.get('intent', [])
464: 
465: 
466: def isintent_aligned16(var):
467:     return 'aligned16' in var.get('intent', [])
468: 
469: isintent_dict = {isintent_in: 'INTENT_IN', isintent_inout: 'INTENT_INOUT',
470:                  isintent_out: 'INTENT_OUT', isintent_hide: 'INTENT_HIDE',
471:                  isintent_cache: 'INTENT_CACHE',
472:                  isintent_c: 'INTENT_C', isoptional: 'OPTIONAL',
473:                  isintent_inplace: 'INTENT_INPLACE',
474:                  isintent_aligned4: 'INTENT_ALIGNED4',
475:                  isintent_aligned8: 'INTENT_ALIGNED8',
476:                  isintent_aligned16: 'INTENT_ALIGNED16',
477:                  }
478: 
479: 
480: def isprivate(var):
481:     return 'attrspec' in var and 'private' in var['attrspec']
482: 
483: 
484: def hasinitvalue(var):
485:     return '=' in var
486: 
487: 
488: def hasinitvalueasstring(var):
489:     if not hasinitvalue(var):
490:         return 0
491:     return var['='][0] in ['"', "'"]
492: 
493: 
494: def hasnote(var):
495:     return 'note' in var
496: 
497: 
498: def hasresultnote(rout):
499:     if not isfunction(rout):
500:         return 0
501:     if 'result' in rout:
502:         a = rout['result']
503:     else:
504:         a = rout['name']
505:     if a in rout['vars']:
506:         return hasnote(rout['vars'][a])
507:     return 0
508: 
509: 
510: def hascommon(rout):
511:     return 'common' in rout
512: 
513: 
514: def containscommon(rout):
515:     if hascommon(rout):
516:         return 1
517:     if hasbody(rout):
518:         for b in rout['body']:
519:             if containscommon(b):
520:                 return 1
521:     return 0
522: 
523: 
524: def containsmodule(block):
525:     if ismodule(block):
526:         return 1
527:     if not hasbody(block):
528:         return 0
529:     for b in block['body']:
530:         if containsmodule(b):
531:             return 1
532:     return 0
533: 
534: 
535: def hasbody(rout):
536:     return 'body' in rout
537: 
538: 
539: def hascallstatement(rout):
540:     return getcallstatement(rout) is not None
541: 
542: 
543: def istrue(var):
544:     return 1
545: 
546: 
547: def isfalse(var):
548:     return 0
549: 
550: 
551: class F2PYError(Exception):
552:     pass
553: 
554: 
555: class throw_error:
556: 
557:     def __init__(self, mess):
558:         self.mess = mess
559: 
560:     def __call__(self, var):
561:         mess = '\n\n  var = %s\n  Message: %s\n' % (var, self.mess)
562:         raise F2PYError(mess)
563: 
564: 
565: def l_and(*f):
566:     l, l2 = 'lambda v', []
567:     for i in range(len(f)):
568:         l = '%s,f%d=f[%d]' % (l, i, i)
569:         l2.append('f%d(v)' % (i))
570:     return eval('%s:%s' % (l, ' and '.join(l2)))
571: 
572: 
573: def l_or(*f):
574:     l, l2 = 'lambda v', []
575:     for i in range(len(f)):
576:         l = '%s,f%d=f[%d]' % (l, i, i)
577:         l2.append('f%d(v)' % (i))
578:     return eval('%s:%s' % (l, ' or '.join(l2)))
579: 
580: 
581: def l_not(f):
582:     return eval('lambda v,f=f:not f(v)')
583: 
584: 
585: def isdummyroutine(rout):
586:     try:
587:         return rout['f2pyenhancements']['fortranname'] == ''
588:     except KeyError:
589:         return 0
590: 
591: 
592: def getfortranname(rout):
593:     try:
594:         name = rout['f2pyenhancements']['fortranname']
595:         if name == '':
596:             raise KeyError
597:         if not name:
598:             errmess('Failed to use fortranname from %s\n' %
599:                     (rout['f2pyenhancements']))
600:             raise KeyError
601:     except KeyError:
602:         name = rout['name']
603:     return name
604: 
605: 
606: def getmultilineblock(rout, blockname, comment=1, counter=0):
607:     try:
608:         r = rout['f2pyenhancements'].get(blockname)
609:     except KeyError:
610:         return
611:     if not r:
612:         return
613:     if counter > 0 and isinstance(r, str):
614:         return
615:     if isinstance(r, list):
616:         if counter >= len(r):
617:             return
618:         r = r[counter]
619:     if r[:3] == "'''":
620:         if comment:
621:             r = '\t/* start ' + blockname + \
622:                 ' multiline (' + repr(counter) + ') */\n' + r[3:]
623:         else:
624:             r = r[3:]
625:         if r[-3:] == "'''":
626:             if comment:
627:                 r = r[:-3] + '\n\t/* end multiline (' + repr(counter) + ')*/'
628:             else:
629:                 r = r[:-3]
630:         else:
631:             errmess("%s multiline block should end with `'''`: %s\n"
632:                     % (blockname, repr(r)))
633:     return r
634: 
635: 
636: def getcallstatement(rout):
637:     return getmultilineblock(rout, 'callstatement')
638: 
639: 
640: def getcallprotoargument(rout, cb_map={}):
641:     r = getmultilineblock(rout, 'callprotoargument', comment=0)
642:     if r:
643:         return r
644:     if hascallstatement(rout):
645:         outmess(
646:             'warning: callstatement is defined without callprotoargument\n')
647:         return
648:     from .capi_maps import getctype
649:     arg_types, arg_types2 = [], []
650:     if l_and(isstringfunction, l_not(isfunction_wrap))(rout):
651:         arg_types.extend(['char*', 'size_t'])
652:     for n in rout['args']:
653:         var = rout['vars'][n]
654:         if isintent_callback(var):
655:             continue
656:         if n in cb_map:
657:             ctype = cb_map[n] + '_typedef'
658:         else:
659:             ctype = getctype(var)
660:             if l_and(isintent_c, l_or(isscalar, iscomplex))(var):
661:                 pass
662:             elif isstring(var):
663:                 pass
664:             else:
665:                 ctype = ctype + '*'
666:             if isstring(var) or isarrayofstrings(var):
667:                 arg_types2.append('size_t')
668:         arg_types.append(ctype)
669: 
670:     proto_args = ','.join(arg_types + arg_types2)
671:     if not proto_args:
672:         proto_args = 'void'
673:     return proto_args
674: 
675: 
676: def getusercode(rout):
677:     return getmultilineblock(rout, 'usercode')
678: 
679: 
680: def getusercode1(rout):
681:     return getmultilineblock(rout, 'usercode', counter=1)
682: 
683: 
684: def getpymethoddef(rout):
685:     return getmultilineblock(rout, 'pymethoddef')
686: 
687: 
688: def getargs(rout):
689:     sortargs, args = [], []
690:     if 'args' in rout:
691:         args = rout['args']
692:         if 'sortvars' in rout:
693:             for a in rout['sortvars']:
694:                 if a in args:
695:                     sortargs.append(a)
696:             for a in args:
697:                 if a not in sortargs:
698:                     sortargs.append(a)
699:         else:
700:             sortargs = rout['args']
701:     return args, sortargs
702: 
703: 
704: def getargs2(rout):
705:     sortargs, args = [], rout.get('args', [])
706:     auxvars = [a for a in rout['vars'].keys() if isintent_aux(rout['vars'][a])
707:                and a not in args]
708:     args = auxvars + args
709:     if 'sortvars' in rout:
710:         for a in rout['sortvars']:
711:             if a in args:
712:                 sortargs.append(a)
713:         for a in args:
714:             if a not in sortargs:
715:                 sortargs.append(a)
716:     else:
717:         sortargs = auxvars + rout['args']
718:     return args, sortargs
719: 
720: 
721: def getrestdoc(rout):
722:     if 'f2pymultilines' not in rout:
723:         return None
724:     k = None
725:     if rout['block'] == 'python module':
726:         k = rout['block'], rout['name']
727:     return rout['f2pymultilines'].get(k, None)
728: 
729: 
730: def gentitle(name):
731:     l = (80 - len(name) - 6) // 2
732:     return '/*%s %s %s*/' % (l * '*', name, l * '*')
733: 
734: 
735: def flatlist(l):
736:     if isinstance(l, list):
737:         return reduce(lambda x, y, f=flatlist: x + f(y), l, [])
738:     return [l]
739: 
740: 
741: def stripcomma(s):
742:     if s and s[-1] == ',':
743:         return s[:-1]
744:     return s
745: 
746: 
747: def replace(str, d, defaultsep=''):
748:     if isinstance(d, list):
749:         return [replace(str, _m, defaultsep) for _m in d]
750:     if isinstance(str, list):
751:         return [replace(_m, d, defaultsep) for _m in str]
752:     for k in 2 * list(d.keys()):
753:         if k == 'separatorsfor':
754:             continue
755:         if 'separatorsfor' in d and k in d['separatorsfor']:
756:             sep = d['separatorsfor'][k]
757:         else:
758:             sep = defaultsep
759:         if isinstance(d[k], list):
760:             str = str.replace('#%s#' % (k), sep.join(flatlist(d[k])))
761:         else:
762:             str = str.replace('#%s#' % (k), d[k])
763:     return str
764: 
765: 
766: def dictappend(rd, ar):
767:     if isinstance(ar, list):
768:         for a in ar:
769:             rd = dictappend(rd, a)
770:         return rd
771:     for k in ar.keys():
772:         if k[0] == '_':
773:             continue
774:         if k in rd:
775:             if isinstance(rd[k], str):
776:                 rd[k] = [rd[k]]
777:             if isinstance(rd[k], list):
778:                 if isinstance(ar[k], list):
779:                     rd[k] = rd[k] + ar[k]
780:                 else:
781:                     rd[k].append(ar[k])
782:             elif isinstance(rd[k], dict):
783:                 if isinstance(ar[k], dict):
784:                     if k == 'separatorsfor':
785:                         for k1 in ar[k].keys():
786:                             if k1 not in rd[k]:
787:                                 rd[k][k1] = ar[k][k1]
788:                     else:
789:                         rd[k] = dictappend(rd[k], ar[k])
790:         else:
791:             rd[k] = ar[k]
792:     return rd
793: 
794: 
795: def applyrules(rules, d, var={}):
796:     ret = {}
797:     if isinstance(rules, list):
798:         for r in rules:
799:             rr = applyrules(r, d, var)
800:             ret = dictappend(ret, rr)
801:             if '_break' in rr:
802:                 break
803:         return ret
804:     if '_check' in rules and (not rules['_check'](var)):
805:         return ret
806:     if 'need' in rules:
807:         res = applyrules({'needs': rules['need']}, d, var)
808:         if 'needs' in res:
809:             cfuncs.append_needs(res['needs'])
810: 
811:     for k in rules.keys():
812:         if k == 'separatorsfor':
813:             ret[k] = rules[k]
814:             continue
815:         if isinstance(rules[k], str):
816:             ret[k] = replace(rules[k], d)
817:         elif isinstance(rules[k], list):
818:             ret[k] = []
819:             for i in rules[k]:
820:                 ar = applyrules({k: i}, d, var)
821:                 if k in ar:
822:                     ret[k].append(ar[k])
823:         elif k[0] == '_':
824:             continue
825:         elif isinstance(rules[k], dict):
826:             ret[k] = []
827:             for k1 in rules[k].keys():
828:                 if isinstance(k1, types.FunctionType) and k1(var):
829:                     if isinstance(rules[k][k1], list):
830:                         for i in rules[k][k1]:
831:                             if isinstance(i, dict):
832:                                 res = applyrules({'supertext': i}, d, var)
833:                                 if 'supertext' in res:
834:                                     i = res['supertext']
835:                                 else:
836:                                     i = ''
837:                             ret[k].append(replace(i, d))
838:                     else:
839:                         i = rules[k][k1]
840:                         if isinstance(i, dict):
841:                             res = applyrules({'supertext': i}, d)
842:                             if 'supertext' in res:
843:                                 i = res['supertext']
844:                             else:
845:                                 i = ''
846:                         ret[k].append(replace(i, d))
847:         else:
848:             errmess('applyrules: ignoring rule %s.\n' % repr(rules[k]))
849:         if isinstance(ret[k], list):
850:             if len(ret[k]) == 1:
851:                 ret[k] = ret[k][0]
852:             if ret[k] == []:
853:                 del ret[k]
854:     return ret
855: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_66686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, (-1)), 'str', '\n\nAuxiliary functions for f2py2e.\n\nCopyright 1999,2000 Pearu Peterson all rights reserved,\nPearu Peterson <pearu@ioc.ee>\nPermission to use, modify, and distribute this software is given under the\nterms of the NumPy (BSD style) LICENSE.\n\n\nNO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.\n$Date: 2005/07/24 19:01:55 $\nPearu Peterson\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import pprint' statement (line 19)
import pprint

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'pprint', pprint, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'import sys' statement (line 20)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'import types' statement (line 21)
import types

import_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from functools import reduce' statement (line 22)
from functools import reduce

import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'functools', None, module_type_store, ['reduce'], [reduce])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'from numpy.f2py import __version__' statement (line 24)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_66687 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py')

if (type(import_66687) is not StypyTypeError):

    if (import_66687 != 'pyd_module'):
        __import__(import_66687)
        sys_modules_66688 = sys.modules[import_66687]
        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py', sys_modules_66688.module_type_store, module_type_store, ['__version__'])
        nest_module(stypy.reporting.localization.Localization(__file__, 24, 0), __file__, sys_modules_66688, sys_modules_66688.module_type_store, module_type_store)
    else:
        from numpy.f2py import __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py', None, module_type_store, ['__version__'], [__version__])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'numpy.f2py', import_66687)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from numpy.f2py import cfuncs' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_66689 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py')

if (type(import_66689) is not StypyTypeError):

    if (import_66689 != 'pyd_module'):
        __import__(import_66689)
        sys_modules_66690 = sys.modules[import_66689]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py', sys_modules_66690.module_type_store, module_type_store, ['cfuncs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_66690, sys_modules_66690.module_type_store, module_type_store)
    else:
        from numpy.f2py import cfuncs

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py', None, module_type_store, ['cfuncs'], [cfuncs])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'numpy.f2py', import_66689)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


# Assigning a List to a Name (line 27):

# Assigning a List to a Name (line 27):
__all__ = ['applyrules', 'debugcapi', 'dictappend', 'errmess', 'gentitle', 'getargs2', 'getcallprotoargument', 'getcallstatement', 'getfortranname', 'getpymethoddef', 'getrestdoc', 'getusercode', 'getusercode1', 'hasbody', 'hascallstatement', 'hascommon', 'hasexternals', 'hasinitvalue', 'hasnote', 'hasresultnote', 'isallocatable', 'isarray', 'isarrayofstrings', 'iscomplex', 'iscomplexarray', 'iscomplexfunction', 'iscomplexfunction_warn', 'isdouble', 'isdummyroutine', 'isexternal', 'isfunction', 'isfunction_wrap', 'isint1array', 'isinteger', 'isintent_aux', 'isintent_c', 'isintent_callback', 'isintent_copy', 'isintent_dict', 'isintent_hide', 'isintent_in', 'isintent_inout', 'isintent_inplace', 'isintent_nothide', 'isintent_out', 'isintent_overwrite', 'islogical', 'islogicalfunction', 'islong_complex', 'islong_double', 'islong_doublefunction', 'islong_long', 'islong_longfunction', 'ismodule', 'ismoduleroutine', 'isoptional', 'isprivate', 'isrequired', 'isroutine', 'isscalar', 'issigned_long_longarray', 'isstring', 'isstringarray', 'isstringfunction', 'issubroutine', 'issubroutine_wrap', 'isthreadsafe', 'isunsigned', 'isunsigned_char', 'isunsigned_chararray', 'isunsigned_long_long', 'isunsigned_long_longarray', 'isunsigned_short', 'isunsigned_shortarray', 'l_and', 'l_not', 'l_or', 'outmess', 'replace', 'show', 'stripcomma', 'throw_error']
module_type_store.set_exportable_members(['applyrules', 'debugcapi', 'dictappend', 'errmess', 'gentitle', 'getargs2', 'getcallprotoargument', 'getcallstatement', 'getfortranname', 'getpymethoddef', 'getrestdoc', 'getusercode', 'getusercode1', 'hasbody', 'hascallstatement', 'hascommon', 'hasexternals', 'hasinitvalue', 'hasnote', 'hasresultnote', 'isallocatable', 'isarray', 'isarrayofstrings', 'iscomplex', 'iscomplexarray', 'iscomplexfunction', 'iscomplexfunction_warn', 'isdouble', 'isdummyroutine', 'isexternal', 'isfunction', 'isfunction_wrap', 'isint1array', 'isinteger', 'isintent_aux', 'isintent_c', 'isintent_callback', 'isintent_copy', 'isintent_dict', 'isintent_hide', 'isintent_in', 'isintent_inout', 'isintent_inplace', 'isintent_nothide', 'isintent_out', 'isintent_overwrite', 'islogical', 'islogicalfunction', 'islong_complex', 'islong_double', 'islong_doublefunction', 'islong_long', 'islong_longfunction', 'ismodule', 'ismoduleroutine', 'isoptional', 'isprivate', 'isrequired', 'isroutine', 'isscalar', 'issigned_long_longarray', 'isstring', 'isstringarray', 'isstringfunction', 'issubroutine', 'issubroutine_wrap', 'isthreadsafe', 'isunsigned', 'isunsigned_char', 'isunsigned_chararray', 'isunsigned_long_long', 'isunsigned_long_longarray', 'isunsigned_short', 'isunsigned_shortarray', 'l_and', 'l_not', 'l_or', 'outmess', 'replace', 'show', 'stripcomma', 'throw_error'])

# Obtaining an instance of the builtin type 'list' (line 27)
list_66691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
str_66692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'str', 'applyrules')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66692)
# Adding element type (line 27)
str_66693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 18), 'str', 'debugcapi')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66693)
# Adding element type (line 27)
str_66694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 31), 'str', 'dictappend')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66694)
# Adding element type (line 27)
str_66695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 45), 'str', 'errmess')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66695)
# Adding element type (line 27)
str_66696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 56), 'str', 'gentitle')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66696)
# Adding element type (line 27)
str_66697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'str', 'getargs2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66697)
# Adding element type (line 27)
str_66698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'str', 'getcallprotoargument')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66698)
# Adding element type (line 27)
str_66699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 40), 'str', 'getcallstatement')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66699)
# Adding element type (line 27)
str_66700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'str', 'getfortranname')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66700)
# Adding element type (line 27)
str_66701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 22), 'str', 'getpymethoddef')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66701)
# Adding element type (line 27)
str_66702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 40), 'str', 'getrestdoc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66702)
# Adding element type (line 27)
str_66703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 54), 'str', 'getusercode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66703)
# Adding element type (line 27)
str_66704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'str', 'getusercode1')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66704)
# Adding element type (line 27)
str_66705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'str', 'hasbody')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66705)
# Adding element type (line 27)
str_66706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'str', 'hascallstatement')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66706)
# Adding element type (line 27)
str_66707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 51), 'str', 'hascommon')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66707)
# Adding element type (line 27)
str_66708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'str', 'hasexternals')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66708)
# Adding element type (line 27)
str_66709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'str', 'hasinitvalue')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66709)
# Adding element type (line 27)
str_66710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 36), 'str', 'hasnote')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66710)
# Adding element type (line 27)
str_66711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 47), 'str', 'hasresultnote')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66711)
# Adding element type (line 27)
str_66712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'str', 'isallocatable')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66712)
# Adding element type (line 27)
str_66713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'str', 'isarray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66713)
# Adding element type (line 27)
str_66714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 32), 'str', 'isarrayofstrings')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66714)
# Adding element type (line 27)
str_66715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 52), 'str', 'iscomplex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66715)
# Adding element type (line 27)
str_66716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'str', 'iscomplexarray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66716)
# Adding element type (line 27)
str_66717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 22), 'str', 'iscomplexfunction')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66717)
# Adding element type (line 27)
str_66718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 43), 'str', 'iscomplexfunction_warn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66718)
# Adding element type (line 27)
str_66719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'str', 'isdouble')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66719)
# Adding element type (line 27)
str_66720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 16), 'str', 'isdummyroutine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66720)
# Adding element type (line 27)
str_66721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 34), 'str', 'isexternal')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66721)
# Adding element type (line 27)
str_66722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 48), 'str', 'isfunction')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66722)
# Adding element type (line 27)
str_66723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', 'isfunction_wrap')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66723)
# Adding element type (line 27)
str_66724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', 'isint1array')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66724)
# Adding element type (line 27)
str_66725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 38), 'str', 'isinteger')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66725)
# Adding element type (line 27)
str_66726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 51), 'str', 'isintent_aux')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66726)
# Adding element type (line 27)
str_66727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'str', 'isintent_c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66727)
# Adding element type (line 27)
str_66728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'str', 'isintent_callback')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66728)
# Adding element type (line 27)
str_66729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 39), 'str', 'isintent_copy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66729)
# Adding element type (line 27)
str_66730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 56), 'str', 'isintent_dict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66730)
# Adding element type (line 27)
str_66731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'str', 'isintent_hide')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66731)
# Adding element type (line 27)
str_66732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'str', 'isintent_in')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66732)
# Adding element type (line 27)
str_66733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 36), 'str', 'isintent_inout')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66733)
# Adding element type (line 27)
str_66734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 54), 'str', 'isintent_inplace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66734)
# Adding element type (line 27)
str_66735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 4), 'str', 'isintent_nothide')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66735)
# Adding element type (line 27)
str_66736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'str', 'isintent_out')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66736)
# Adding element type (line 27)
str_66737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 40), 'str', 'isintent_overwrite')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66737)
# Adding element type (line 27)
str_66738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 62), 'str', 'islogical')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66738)
# Adding element type (line 27)
str_66739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'str', 'islogicalfunction')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66739)
# Adding element type (line 27)
str_66740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'str', 'islong_complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66740)
# Adding element type (line 27)
str_66741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 43), 'str', 'islong_double')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66741)
# Adding element type (line 27)
str_66742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 4), 'str', 'islong_doublefunction')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66742)
# Adding element type (line 27)
str_66743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 29), 'str', 'islong_long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66743)
# Adding element type (line 27)
str_66744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 44), 'str', 'islong_longfunction')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66744)
# Adding element type (line 27)
str_66745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'str', 'ismodule')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66745)
# Adding element type (line 27)
str_66746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 16), 'str', 'ismoduleroutine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66746)
# Adding element type (line 27)
str_66747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 35), 'str', 'isoptional')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66747)
# Adding element type (line 27)
str_66748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 49), 'str', 'isprivate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66748)
# Adding element type (line 27)
str_66749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 62), 'str', 'isrequired')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66749)
# Adding element type (line 27)
str_66750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'str', 'isroutine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66750)
# Adding element type (line 27)
str_66751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 17), 'str', 'isscalar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66751)
# Adding element type (line 27)
str_66752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 29), 'str', 'issigned_long_longarray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66752)
# Adding element type (line 27)
str_66753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 56), 'str', 'isstring')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66753)
# Adding element type (line 27)
str_66754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 4), 'str', 'isstringarray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66754)
# Adding element type (line 27)
str_66755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'str', 'isstringfunction')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66755)
# Adding element type (line 27)
str_66756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 41), 'str', 'issubroutine')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66756)
# Adding element type (line 27)
str_66757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'str', 'issubroutine_wrap')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66757)
# Adding element type (line 27)
str_66758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'str', 'isthreadsafe')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66758)
# Adding element type (line 27)
str_66759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 41), 'str', 'isunsigned')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66759)
# Adding element type (line 27)
str_66760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 55), 'str', 'isunsigned_char')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66760)
# Adding element type (line 27)
str_66761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'str', 'isunsigned_chararray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66761)
# Adding element type (line 27)
str_66762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 28), 'str', 'isunsigned_long_long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66762)
# Adding element type (line 27)
str_66763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 4), 'str', 'isunsigned_long_longarray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66763)
# Adding element type (line 27)
str_66764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 33), 'str', 'isunsigned_short')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66764)
# Adding element type (line 27)
str_66765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 4), 'str', 'isunsigned_shortarray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66765)
# Adding element type (line 27)
str_66766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 29), 'str', 'l_and')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66766)
# Adding element type (line 27)
str_66767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 38), 'str', 'l_not')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66767)
# Adding element type (line 27)
str_66768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 47), 'str', 'l_or')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66768)
# Adding element type (line 27)
str_66769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 55), 'str', 'outmess')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66769)
# Adding element type (line 27)
str_66770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'str', 'replace')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66770)
# Adding element type (line 27)
str_66771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 15), 'str', 'show')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66771)
# Adding element type (line 27)
str_66772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'str', 'stripcomma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66772)
# Adding element type (line 27)
str_66773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'str', 'throw_error')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), list_66691, str_66773)

# Assigning a type to the variable '__all__' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), '__all__', list_66691)

# Assigning a Attribute to a Name (line 53):

# Assigning a Attribute to a Name (line 53):
# Getting the type of '__version__' (line 53)
version___66774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), '__version__')
# Obtaining the member 'version' of a type (line 53)
version_66775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 15), version___66774, 'version')
# Assigning a type to the variable 'f2py_version' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'f2py_version', version_66775)

# Assigning a Attribute to a Name (line 56):

# Assigning a Attribute to a Name (line 56):
# Getting the type of 'sys' (line 56)
sys_66776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 10), 'sys')
# Obtaining the member 'stderr' of a type (line 56)
stderr_66777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 10), sys_66776, 'stderr')
# Obtaining the member 'write' of a type (line 56)
write_66778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 10), stderr_66777, 'write')
# Assigning a type to the variable 'errmess' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'errmess', write_66778)

# Assigning a Attribute to a Name (line 57):

# Assigning a Attribute to a Name (line 57):
# Getting the type of 'pprint' (line 57)
pprint_66779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 7), 'pprint')
# Obtaining the member 'pprint' of a type (line 57)
pprint_66780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 7), pprint_66779, 'pprint')
# Assigning a type to the variable 'show' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'show', pprint_66780)

# Assigning a Dict to a Name (line 59):

# Assigning a Dict to a Name (line 59):

# Obtaining an instance of the builtin type 'dict' (line 59)
dict_66781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 59)

# Assigning a type to the variable 'options' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'options', dict_66781)

# Assigning a List to a Name (line 60):

# Assigning a List to a Name (line 60):

# Obtaining an instance of the builtin type 'list' (line 60)
list_66782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 60)

# Assigning a type to the variable 'debugoptions' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'debugoptions', list_66782)

# Assigning a Num to a Name (line 61):

# Assigning a Num to a Name (line 61):
int_66783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 12), 'int')
# Assigning a type to the variable 'wrapfuncs' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'wrapfuncs', int_66783)

@norecursion
def outmess(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'outmess'
    module_type_store = module_type_store.open_function_context('outmess', 64, 0, False)
    
    # Passed parameters checking function
    outmess.stypy_localization = localization
    outmess.stypy_type_of_self = None
    outmess.stypy_type_store = module_type_store
    outmess.stypy_function_name = 'outmess'
    outmess.stypy_param_names_list = ['t']
    outmess.stypy_varargs_param_name = None
    outmess.stypy_kwargs_param_name = None
    outmess.stypy_call_defaults = defaults
    outmess.stypy_call_varargs = varargs
    outmess.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'outmess', ['t'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'outmess', localization, ['t'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'outmess(...)' code ##################

    
    
    # Call to get(...): (line 65)
    # Processing the call arguments (line 65)
    str_66786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'str', 'verbose')
    int_66787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 30), 'int')
    # Processing the call keyword arguments (line 65)
    kwargs_66788 = {}
    # Getting the type of 'options' (line 65)
    options_66784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 7), 'options', False)
    # Obtaining the member 'get' of a type (line 65)
    get_66785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 7), options_66784, 'get')
    # Calling get(args, kwargs) (line 65)
    get_call_result_66789 = invoke(stypy.reporting.localization.Localization(__file__, 65, 7), get_66785, *[str_66786, int_66787], **kwargs_66788)
    
    # Testing the type of an if condition (line 65)
    if_condition_66790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 4), get_call_result_66789)
    # Assigning a type to the variable 'if_condition_66790' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'if_condition_66790', if_condition_66790)
    # SSA begins for if statement (line 65)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to write(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 't' (line 66)
    t_66794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 't', False)
    # Processing the call keyword arguments (line 66)
    kwargs_66795 = {}
    # Getting the type of 'sys' (line 66)
    sys_66791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'sys', False)
    # Obtaining the member 'stdout' of a type (line 66)
    stdout_66792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), sys_66791, 'stdout')
    # Obtaining the member 'write' of a type (line 66)
    write_66793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), stdout_66792, 'write')
    # Calling write(args, kwargs) (line 66)
    write_call_result_66796 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), write_66793, *[t_66794], **kwargs_66795)
    
    # SSA join for if statement (line 65)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'outmess(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'outmess' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_66797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66797)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'outmess'
    return stypy_return_type_66797

# Assigning a type to the variable 'outmess' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'outmess', outmess)

@norecursion
def debugcapi(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'debugcapi'
    module_type_store = module_type_store.open_function_context('debugcapi', 69, 0, False)
    
    # Passed parameters checking function
    debugcapi.stypy_localization = localization
    debugcapi.stypy_type_of_self = None
    debugcapi.stypy_type_store = module_type_store
    debugcapi.stypy_function_name = 'debugcapi'
    debugcapi.stypy_param_names_list = ['var']
    debugcapi.stypy_varargs_param_name = None
    debugcapi.stypy_kwargs_param_name = None
    debugcapi.stypy_call_defaults = defaults
    debugcapi.stypy_call_varargs = varargs
    debugcapi.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'debugcapi', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'debugcapi', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'debugcapi(...)' code ##################

    
    str_66798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 11), 'str', 'capi')
    # Getting the type of 'debugoptions' (line 70)
    debugoptions_66799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'debugoptions')
    # Applying the binary operator 'in' (line 70)
    result_contains_66800 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 11), 'in', str_66798, debugoptions_66799)
    
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', result_contains_66800)
    
    # ################# End of 'debugcapi(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'debugcapi' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_66801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66801)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'debugcapi'
    return stypy_return_type_66801

# Assigning a type to the variable 'debugcapi' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'debugcapi', debugcapi)

@norecursion
def _isstring(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_isstring'
    module_type_store = module_type_store.open_function_context('_isstring', 73, 0, False)
    
    # Passed parameters checking function
    _isstring.stypy_localization = localization
    _isstring.stypy_type_of_self = None
    _isstring.stypy_type_store = module_type_store
    _isstring.stypy_function_name = '_isstring'
    _isstring.stypy_param_names_list = ['var']
    _isstring.stypy_varargs_param_name = None
    _isstring.stypy_kwargs_param_name = None
    _isstring.stypy_call_defaults = defaults
    _isstring.stypy_call_varargs = varargs
    _isstring.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_isstring', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_isstring', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_isstring(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_66802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 11), 'str', 'typespec')
    # Getting the type of 'var' (line 74)
    var_66803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'var')
    # Applying the binary operator 'in' (line 74)
    result_contains_66804 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), 'in', str_66802, var_66803)
    
    
    
    # Obtaining the type of the subscript
    str_66805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 37), 'str', 'typespec')
    # Getting the type of 'var' (line 74)
    var_66806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 33), 'var')
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___66807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 33), var_66806, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_66808 = invoke(stypy.reporting.localization.Localization(__file__, 74, 33), getitem___66807, str_66805)
    
    str_66809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 52), 'str', 'character')
    # Applying the binary operator '==' (line 74)
    result_eq_66810 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 33), '==', subscript_call_result_66808, str_66809)
    
    # Applying the binary operator 'and' (line 74)
    result_and_keyword_66811 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), 'and', result_contains_66804, result_eq_66810)
    
    
    # Call to isexternal(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'var' (line 75)
    var_66813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'var', False)
    # Processing the call keyword arguments (line 75)
    kwargs_66814 = {}
    # Getting the type of 'isexternal' (line 75)
    isexternal_66812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'isexternal', False)
    # Calling isexternal(args, kwargs) (line 75)
    isexternal_call_result_66815 = invoke(stypy.reporting.localization.Localization(__file__, 75, 15), isexternal_66812, *[var_66813], **kwargs_66814)
    
    # Applying the 'not' unary operator (line 75)
    result_not__66816 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), 'not', isexternal_call_result_66815)
    
    # Applying the binary operator 'and' (line 74)
    result_and_keyword_66817 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), 'and', result_and_keyword_66811, result_not__66816)
    
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type', result_and_keyword_66817)
    
    # ################# End of '_isstring(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_isstring' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_66818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66818)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_isstring'
    return stypy_return_type_66818

# Assigning a type to the variable '_isstring' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), '_isstring', _isstring)

@norecursion
def isstring(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isstring'
    module_type_store = module_type_store.open_function_context('isstring', 78, 0, False)
    
    # Passed parameters checking function
    isstring.stypy_localization = localization
    isstring.stypy_type_of_self = None
    isstring.stypy_type_store = module_type_store
    isstring.stypy_function_name = 'isstring'
    isstring.stypy_param_names_list = ['var']
    isstring.stypy_varargs_param_name = None
    isstring.stypy_kwargs_param_name = None
    isstring.stypy_call_defaults = defaults
    isstring.stypy_call_varargs = varargs
    isstring.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isstring', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isstring', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isstring(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to _isstring(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'var' (line 79)
    var_66820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'var', False)
    # Processing the call keyword arguments (line 79)
    kwargs_66821 = {}
    # Getting the type of '_isstring' (line 79)
    _isstring_66819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), '_isstring', False)
    # Calling _isstring(args, kwargs) (line 79)
    _isstring_call_result_66822 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), _isstring_66819, *[var_66820], **kwargs_66821)
    
    
    
    # Call to isarray(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'var' (line 79)
    var_66824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 42), 'var', False)
    # Processing the call keyword arguments (line 79)
    kwargs_66825 = {}
    # Getting the type of 'isarray' (line 79)
    isarray_66823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 34), 'isarray', False)
    # Calling isarray(args, kwargs) (line 79)
    isarray_call_result_66826 = invoke(stypy.reporting.localization.Localization(__file__, 79, 34), isarray_66823, *[var_66824], **kwargs_66825)
    
    # Applying the 'not' unary operator (line 79)
    result_not__66827 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 30), 'not', isarray_call_result_66826)
    
    # Applying the binary operator 'and' (line 79)
    result_and_keyword_66828 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 11), 'and', _isstring_call_result_66822, result_not__66827)
    
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type', result_and_keyword_66828)
    
    # ################# End of 'isstring(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isstring' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_66829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66829)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isstring'
    return stypy_return_type_66829

# Assigning a type to the variable 'isstring' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'isstring', isstring)

@norecursion
def ischaracter(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ischaracter'
    module_type_store = module_type_store.open_function_context('ischaracter', 82, 0, False)
    
    # Passed parameters checking function
    ischaracter.stypy_localization = localization
    ischaracter.stypy_type_of_self = None
    ischaracter.stypy_type_store = module_type_store
    ischaracter.stypy_function_name = 'ischaracter'
    ischaracter.stypy_param_names_list = ['var']
    ischaracter.stypy_varargs_param_name = None
    ischaracter.stypy_kwargs_param_name = None
    ischaracter.stypy_call_defaults = defaults
    ischaracter.stypy_call_varargs = varargs
    ischaracter.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ischaracter', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ischaracter', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ischaracter(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isstring(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'var' (line 83)
    var_66831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'var', False)
    # Processing the call keyword arguments (line 83)
    kwargs_66832 = {}
    # Getting the type of 'isstring' (line 83)
    isstring_66830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'isstring', False)
    # Calling isstring(args, kwargs) (line 83)
    isstring_call_result_66833 = invoke(stypy.reporting.localization.Localization(__file__, 83, 11), isstring_66830, *[var_66831], **kwargs_66832)
    
    
    str_66834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 29), 'str', 'charselector')
    # Getting the type of 'var' (line 83)
    var_66835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 51), 'var')
    # Applying the binary operator 'notin' (line 83)
    result_contains_66836 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 29), 'notin', str_66834, var_66835)
    
    # Applying the binary operator 'and' (line 83)
    result_and_keyword_66837 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), 'and', isstring_call_result_66833, result_contains_66836)
    
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type', result_and_keyword_66837)
    
    # ################# End of 'ischaracter(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ischaracter' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_66838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66838)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ischaracter'
    return stypy_return_type_66838

# Assigning a type to the variable 'ischaracter' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'ischaracter', ischaracter)

@norecursion
def isstringarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isstringarray'
    module_type_store = module_type_store.open_function_context('isstringarray', 86, 0, False)
    
    # Passed parameters checking function
    isstringarray.stypy_localization = localization
    isstringarray.stypy_type_of_self = None
    isstringarray.stypy_type_store = module_type_store
    isstringarray.stypy_function_name = 'isstringarray'
    isstringarray.stypy_param_names_list = ['var']
    isstringarray.stypy_varargs_param_name = None
    isstringarray.stypy_kwargs_param_name = None
    isstringarray.stypy_call_defaults = defaults
    isstringarray.stypy_call_varargs = varargs
    isstringarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isstringarray', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isstringarray', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isstringarray(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isarray(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'var' (line 87)
    var_66840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'var', False)
    # Processing the call keyword arguments (line 87)
    kwargs_66841 = {}
    # Getting the type of 'isarray' (line 87)
    isarray_66839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'isarray', False)
    # Calling isarray(args, kwargs) (line 87)
    isarray_call_result_66842 = invoke(stypy.reporting.localization.Localization(__file__, 87, 11), isarray_66839, *[var_66840], **kwargs_66841)
    
    
    # Call to _isstring(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'var' (line 87)
    var_66844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 38), 'var', False)
    # Processing the call keyword arguments (line 87)
    kwargs_66845 = {}
    # Getting the type of '_isstring' (line 87)
    _isstring_66843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 28), '_isstring', False)
    # Calling _isstring(args, kwargs) (line 87)
    _isstring_call_result_66846 = invoke(stypy.reporting.localization.Localization(__file__, 87, 28), _isstring_66843, *[var_66844], **kwargs_66845)
    
    # Applying the binary operator 'and' (line 87)
    result_and_keyword_66847 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 11), 'and', isarray_call_result_66842, _isstring_call_result_66846)
    
    # Assigning a type to the variable 'stypy_return_type' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type', result_and_keyword_66847)
    
    # ################# End of 'isstringarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isstringarray' in the type store
    # Getting the type of 'stypy_return_type' (line 86)
    stypy_return_type_66848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66848)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isstringarray'
    return stypy_return_type_66848

# Assigning a type to the variable 'isstringarray' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'isstringarray', isstringarray)

@norecursion
def isarrayofstrings(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isarrayofstrings'
    module_type_store = module_type_store.open_function_context('isarrayofstrings', 90, 0, False)
    
    # Passed parameters checking function
    isarrayofstrings.stypy_localization = localization
    isarrayofstrings.stypy_type_of_self = None
    isarrayofstrings.stypy_type_store = module_type_store
    isarrayofstrings.stypy_function_name = 'isarrayofstrings'
    isarrayofstrings.stypy_param_names_list = ['var']
    isarrayofstrings.stypy_varargs_param_name = None
    isarrayofstrings.stypy_kwargs_param_name = None
    isarrayofstrings.stypy_call_defaults = defaults
    isarrayofstrings.stypy_call_varargs = varargs
    isarrayofstrings.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isarrayofstrings', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isarrayofstrings', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isarrayofstrings(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isstringarray(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'var' (line 93)
    var_66850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 25), 'var', False)
    # Processing the call keyword arguments (line 93)
    kwargs_66851 = {}
    # Getting the type of 'isstringarray' (line 93)
    isstringarray_66849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'isstringarray', False)
    # Calling isstringarray(args, kwargs) (line 93)
    isstringarray_call_result_66852 = invoke(stypy.reporting.localization.Localization(__file__, 93, 11), isstringarray_66849, *[var_66850], **kwargs_66851)
    
    
    
    # Obtaining the type of the subscript
    int_66853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 51), 'int')
    
    # Obtaining the type of the subscript
    str_66854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 38), 'str', 'dimension')
    # Getting the type of 'var' (line 93)
    var_66855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 34), 'var')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___66856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 34), var_66855, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_66857 = invoke(stypy.reporting.localization.Localization(__file__, 93, 34), getitem___66856, str_66854)
    
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___66858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 34), subscript_call_result_66857, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_66859 = invoke(stypy.reporting.localization.Localization(__file__, 93, 34), getitem___66858, int_66853)
    
    str_66860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 58), 'str', '(*)')
    # Applying the binary operator '==' (line 93)
    result_eq_66861 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 34), '==', subscript_call_result_66859, str_66860)
    
    # Applying the binary operator 'and' (line 93)
    result_and_keyword_66862 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 11), 'and', isstringarray_call_result_66852, result_eq_66861)
    
    # Assigning a type to the variable 'stypy_return_type' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type', result_and_keyword_66862)
    
    # ################# End of 'isarrayofstrings(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isarrayofstrings' in the type store
    # Getting the type of 'stypy_return_type' (line 90)
    stypy_return_type_66863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66863)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isarrayofstrings'
    return stypy_return_type_66863

# Assigning a type to the variable 'isarrayofstrings' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'isarrayofstrings', isarrayofstrings)

@norecursion
def isarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isarray'
    module_type_store = module_type_store.open_function_context('isarray', 96, 0, False)
    
    # Passed parameters checking function
    isarray.stypy_localization = localization
    isarray.stypy_type_of_self = None
    isarray.stypy_type_store = module_type_store
    isarray.stypy_function_name = 'isarray'
    isarray.stypy_param_names_list = ['var']
    isarray.stypy_varargs_param_name = None
    isarray.stypy_kwargs_param_name = None
    isarray.stypy_call_defaults = defaults
    isarray.stypy_call_varargs = varargs
    isarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isarray', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isarray', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isarray(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_66864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 11), 'str', 'dimension')
    # Getting the type of 'var' (line 97)
    var_66865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 26), 'var')
    # Applying the binary operator 'in' (line 97)
    result_contains_66866 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), 'in', str_66864, var_66865)
    
    
    
    # Call to isexternal(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'var' (line 97)
    var_66868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 49), 'var', False)
    # Processing the call keyword arguments (line 97)
    kwargs_66869 = {}
    # Getting the type of 'isexternal' (line 97)
    isexternal_66867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 38), 'isexternal', False)
    # Calling isexternal(args, kwargs) (line 97)
    isexternal_call_result_66870 = invoke(stypy.reporting.localization.Localization(__file__, 97, 38), isexternal_66867, *[var_66868], **kwargs_66869)
    
    # Applying the 'not' unary operator (line 97)
    result_not__66871 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 34), 'not', isexternal_call_result_66870)
    
    # Applying the binary operator 'and' (line 97)
    result_and_keyword_66872 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 11), 'and', result_contains_66866, result_not__66871)
    
    # Assigning a type to the variable 'stypy_return_type' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type', result_and_keyword_66872)
    
    # ################# End of 'isarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isarray' in the type store
    # Getting the type of 'stypy_return_type' (line 96)
    stypy_return_type_66873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66873)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isarray'
    return stypy_return_type_66873

# Assigning a type to the variable 'isarray' (line 96)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 0), 'isarray', isarray)

@norecursion
def isscalar(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isscalar'
    module_type_store = module_type_store.open_function_context('isscalar', 100, 0, False)
    
    # Passed parameters checking function
    isscalar.stypy_localization = localization
    isscalar.stypy_type_of_self = None
    isscalar.stypy_type_store = module_type_store
    isscalar.stypy_function_name = 'isscalar'
    isscalar.stypy_param_names_list = ['var']
    isscalar.stypy_varargs_param_name = None
    isscalar.stypy_kwargs_param_name = None
    isscalar.stypy_call_defaults = defaults
    isscalar.stypy_call_varargs = varargs
    isscalar.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isscalar', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isscalar', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isscalar(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    # Call to isarray(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'var' (line 101)
    var_66875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'var', False)
    # Processing the call keyword arguments (line 101)
    kwargs_66876 = {}
    # Getting the type of 'isarray' (line 101)
    isarray_66874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'isarray', False)
    # Calling isarray(args, kwargs) (line 101)
    isarray_call_result_66877 = invoke(stypy.reporting.localization.Localization(__file__, 101, 16), isarray_66874, *[var_66875], **kwargs_66876)
    
    
    # Call to isstring(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'var' (line 101)
    var_66879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 41), 'var', False)
    # Processing the call keyword arguments (line 101)
    kwargs_66880 = {}
    # Getting the type of 'isstring' (line 101)
    isstring_66878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 32), 'isstring', False)
    # Calling isstring(args, kwargs) (line 101)
    isstring_call_result_66881 = invoke(stypy.reporting.localization.Localization(__file__, 101, 32), isstring_66878, *[var_66879], **kwargs_66880)
    
    # Applying the binary operator 'or' (line 101)
    result_or_keyword_66882 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 16), 'or', isarray_call_result_66877, isstring_call_result_66881)
    
    # Call to isexternal(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'var' (line 101)
    var_66884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 60), 'var', False)
    # Processing the call keyword arguments (line 101)
    kwargs_66885 = {}
    # Getting the type of 'isexternal' (line 101)
    isexternal_66883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 49), 'isexternal', False)
    # Calling isexternal(args, kwargs) (line 101)
    isexternal_call_result_66886 = invoke(stypy.reporting.localization.Localization(__file__, 101, 49), isexternal_66883, *[var_66884], **kwargs_66885)
    
    # Applying the binary operator 'or' (line 101)
    result_or_keyword_66887 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 16), 'or', result_or_keyword_66882, isexternal_call_result_66886)
    
    # Applying the 'not' unary operator (line 101)
    result_not__66888 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), 'not', result_or_keyword_66887)
    
    # Assigning a type to the variable 'stypy_return_type' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type', result_not__66888)
    
    # ################# End of 'isscalar(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isscalar' in the type store
    # Getting the type of 'stypy_return_type' (line 100)
    stypy_return_type_66889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66889)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isscalar'
    return stypy_return_type_66889

# Assigning a type to the variable 'isscalar' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'isscalar', isscalar)

@norecursion
def iscomplex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iscomplex'
    module_type_store = module_type_store.open_function_context('iscomplex', 104, 0, False)
    
    # Passed parameters checking function
    iscomplex.stypy_localization = localization
    iscomplex.stypy_type_of_self = None
    iscomplex.stypy_type_store = module_type_store
    iscomplex.stypy_function_name = 'iscomplex'
    iscomplex.stypy_param_names_list = ['var']
    iscomplex.stypy_varargs_param_name = None
    iscomplex.stypy_kwargs_param_name = None
    iscomplex.stypy_call_defaults = defaults
    iscomplex.stypy_call_varargs = varargs
    iscomplex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iscomplex', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iscomplex', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iscomplex(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isscalar(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'var' (line 105)
    var_66891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'var', False)
    # Processing the call keyword arguments (line 105)
    kwargs_66892 = {}
    # Getting the type of 'isscalar' (line 105)
    isscalar_66890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 105)
    isscalar_call_result_66893 = invoke(stypy.reporting.localization.Localization(__file__, 105, 11), isscalar_66890, *[var_66891], **kwargs_66892)
    
    
    
    # Call to get(...): (line 106)
    # Processing the call arguments (line 106)
    str_66896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 19), 'str', 'typespec')
    # Processing the call keyword arguments (line 106)
    kwargs_66897 = {}
    # Getting the type of 'var' (line 106)
    var_66894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'var', False)
    # Obtaining the member 'get' of a type (line 106)
    get_66895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 11), var_66894, 'get')
    # Calling get(args, kwargs) (line 106)
    get_call_result_66898 = invoke(stypy.reporting.localization.Localization(__file__, 106, 11), get_66895, *[str_66896], **kwargs_66897)
    
    
    # Obtaining an instance of the builtin type 'list' (line 106)
    list_66899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 106)
    # Adding element type (line 106)
    str_66900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 35), 'str', 'complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 34), list_66899, str_66900)
    # Adding element type (line 106)
    str_66901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 46), 'str', 'double complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 34), list_66899, str_66901)
    
    # Applying the binary operator 'in' (line 106)
    result_contains_66902 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 11), 'in', get_call_result_66898, list_66899)
    
    # Applying the binary operator 'and' (line 105)
    result_and_keyword_66903 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 11), 'and', isscalar_call_result_66893, result_contains_66902)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', result_and_keyword_66903)
    
    # ################# End of 'iscomplex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iscomplex' in the type store
    # Getting the type of 'stypy_return_type' (line 104)
    stypy_return_type_66904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66904)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iscomplex'
    return stypy_return_type_66904

# Assigning a type to the variable 'iscomplex' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'iscomplex', iscomplex)

@norecursion
def islogical(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'islogical'
    module_type_store = module_type_store.open_function_context('islogical', 109, 0, False)
    
    # Passed parameters checking function
    islogical.stypy_localization = localization
    islogical.stypy_type_of_self = None
    islogical.stypy_type_store = module_type_store
    islogical.stypy_function_name = 'islogical'
    islogical.stypy_param_names_list = ['var']
    islogical.stypy_varargs_param_name = None
    islogical.stypy_kwargs_param_name = None
    islogical.stypy_call_defaults = defaults
    islogical.stypy_call_varargs = varargs
    islogical.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'islogical', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'islogical', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'islogical(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isscalar(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'var' (line 110)
    var_66906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'var', False)
    # Processing the call keyword arguments (line 110)
    kwargs_66907 = {}
    # Getting the type of 'isscalar' (line 110)
    isscalar_66905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 110)
    isscalar_call_result_66908 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), isscalar_66905, *[var_66906], **kwargs_66907)
    
    
    
    # Call to get(...): (line 110)
    # Processing the call arguments (line 110)
    str_66911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 37), 'str', 'typespec')
    # Processing the call keyword arguments (line 110)
    kwargs_66912 = {}
    # Getting the type of 'var' (line 110)
    var_66909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 29), 'var', False)
    # Obtaining the member 'get' of a type (line 110)
    get_66910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 29), var_66909, 'get')
    # Calling get(args, kwargs) (line 110)
    get_call_result_66913 = invoke(stypy.reporting.localization.Localization(__file__, 110, 29), get_66910, *[str_66911], **kwargs_66912)
    
    str_66914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 52), 'str', 'logical')
    # Applying the binary operator '==' (line 110)
    result_eq_66915 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 29), '==', get_call_result_66913, str_66914)
    
    # Applying the binary operator 'and' (line 110)
    result_and_keyword_66916 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 11), 'and', isscalar_call_result_66908, result_eq_66915)
    
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type', result_and_keyword_66916)
    
    # ################# End of 'islogical(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'islogical' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_66917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66917)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'islogical'
    return stypy_return_type_66917

# Assigning a type to the variable 'islogical' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'islogical', islogical)

@norecursion
def isinteger(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isinteger'
    module_type_store = module_type_store.open_function_context('isinteger', 113, 0, False)
    
    # Passed parameters checking function
    isinteger.stypy_localization = localization
    isinteger.stypy_type_of_self = None
    isinteger.stypy_type_store = module_type_store
    isinteger.stypy_function_name = 'isinteger'
    isinteger.stypy_param_names_list = ['var']
    isinteger.stypy_varargs_param_name = None
    isinteger.stypy_kwargs_param_name = None
    isinteger.stypy_call_defaults = defaults
    isinteger.stypy_call_varargs = varargs
    isinteger.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isinteger', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isinteger', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isinteger(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isscalar(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'var' (line 114)
    var_66919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 20), 'var', False)
    # Processing the call keyword arguments (line 114)
    kwargs_66920 = {}
    # Getting the type of 'isscalar' (line 114)
    isscalar_66918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 114)
    isscalar_call_result_66921 = invoke(stypy.reporting.localization.Localization(__file__, 114, 11), isscalar_66918, *[var_66919], **kwargs_66920)
    
    
    
    # Call to get(...): (line 114)
    # Processing the call arguments (line 114)
    str_66924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 37), 'str', 'typespec')
    # Processing the call keyword arguments (line 114)
    kwargs_66925 = {}
    # Getting the type of 'var' (line 114)
    var_66922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 29), 'var', False)
    # Obtaining the member 'get' of a type (line 114)
    get_66923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 29), var_66922, 'get')
    # Calling get(args, kwargs) (line 114)
    get_call_result_66926 = invoke(stypy.reporting.localization.Localization(__file__, 114, 29), get_66923, *[str_66924], **kwargs_66925)
    
    str_66927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 52), 'str', 'integer')
    # Applying the binary operator '==' (line 114)
    result_eq_66928 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 29), '==', get_call_result_66926, str_66927)
    
    # Applying the binary operator 'and' (line 114)
    result_and_keyword_66929 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 11), 'and', isscalar_call_result_66921, result_eq_66928)
    
    # Assigning a type to the variable 'stypy_return_type' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type', result_and_keyword_66929)
    
    # ################# End of 'isinteger(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isinteger' in the type store
    # Getting the type of 'stypy_return_type' (line 113)
    stypy_return_type_66930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66930)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isinteger'
    return stypy_return_type_66930

# Assigning a type to the variable 'isinteger' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'isinteger', isinteger)

@norecursion
def isreal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isreal'
    module_type_store = module_type_store.open_function_context('isreal', 117, 0, False)
    
    # Passed parameters checking function
    isreal.stypy_localization = localization
    isreal.stypy_type_of_self = None
    isreal.stypy_type_store = module_type_store
    isreal.stypy_function_name = 'isreal'
    isreal.stypy_param_names_list = ['var']
    isreal.stypy_varargs_param_name = None
    isreal.stypy_kwargs_param_name = None
    isreal.stypy_call_defaults = defaults
    isreal.stypy_call_varargs = varargs
    isreal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isreal', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isreal', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isreal(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isscalar(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'var' (line 118)
    var_66932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 20), 'var', False)
    # Processing the call keyword arguments (line 118)
    kwargs_66933 = {}
    # Getting the type of 'isscalar' (line 118)
    isscalar_66931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 118)
    isscalar_call_result_66934 = invoke(stypy.reporting.localization.Localization(__file__, 118, 11), isscalar_66931, *[var_66932], **kwargs_66933)
    
    
    
    # Call to get(...): (line 118)
    # Processing the call arguments (line 118)
    str_66937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 37), 'str', 'typespec')
    # Processing the call keyword arguments (line 118)
    kwargs_66938 = {}
    # Getting the type of 'var' (line 118)
    var_66935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 29), 'var', False)
    # Obtaining the member 'get' of a type (line 118)
    get_66936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 29), var_66935, 'get')
    # Calling get(args, kwargs) (line 118)
    get_call_result_66939 = invoke(stypy.reporting.localization.Localization(__file__, 118, 29), get_66936, *[str_66937], **kwargs_66938)
    
    str_66940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 52), 'str', 'real')
    # Applying the binary operator '==' (line 118)
    result_eq_66941 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 29), '==', get_call_result_66939, str_66940)
    
    # Applying the binary operator 'and' (line 118)
    result_and_keyword_66942 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 11), 'and', isscalar_call_result_66934, result_eq_66941)
    
    # Assigning a type to the variable 'stypy_return_type' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type', result_and_keyword_66942)
    
    # ################# End of 'isreal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isreal' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_66943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66943)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isreal'
    return stypy_return_type_66943

# Assigning a type to the variable 'isreal' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'isreal', isreal)

@norecursion
def get_kind(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_kind'
    module_type_store = module_type_store.open_function_context('get_kind', 121, 0, False)
    
    # Passed parameters checking function
    get_kind.stypy_localization = localization
    get_kind.stypy_type_of_self = None
    get_kind.stypy_type_store = module_type_store
    get_kind.stypy_function_name = 'get_kind'
    get_kind.stypy_param_names_list = ['var']
    get_kind.stypy_varargs_param_name = None
    get_kind.stypy_kwargs_param_name = None
    get_kind.stypy_call_defaults = defaults
    get_kind.stypy_call_varargs = varargs
    get_kind.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_kind', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_kind', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_kind(...)' code ##################

    
    
    # SSA begins for try-except statement (line 122)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    str_66944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 35), 'str', '*')
    
    # Obtaining the type of the subscript
    str_66945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 19), 'str', 'kindselector')
    # Getting the type of 'var' (line 123)
    var_66946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'var')
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___66947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 15), var_66946, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_66948 = invoke(stypy.reporting.localization.Localization(__file__, 123, 15), getitem___66947, str_66945)
    
    # Obtaining the member '__getitem__' of a type (line 123)
    getitem___66949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 15), subscript_call_result_66948, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 123)
    subscript_call_result_66950 = invoke(stypy.reporting.localization.Localization(__file__, 123, 15), getitem___66949, str_66944)
    
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', subscript_call_result_66950)
    # SSA branch for the except part of a try statement (line 122)
    # SSA branch for the except 'KeyError' branch of a try statement (line 122)
    module_type_store.open_ssa_branch('except')
    
    
    # SSA begins for try-except statement (line 125)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    str_66951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 39), 'str', 'kind')
    
    # Obtaining the type of the subscript
    str_66952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 23), 'str', 'kindselector')
    # Getting the type of 'var' (line 126)
    var_66953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 'var')
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___66954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 19), var_66953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 126)
    subscript_call_result_66955 = invoke(stypy.reporting.localization.Localization(__file__, 126, 19), getitem___66954, str_66952)
    
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___66956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 19), subscript_call_result_66955, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 126)
    subscript_call_result_66957 = invoke(stypy.reporting.localization.Localization(__file__, 126, 19), getitem___66956, str_66951)
    
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'stypy_return_type', subscript_call_result_66957)
    # SSA branch for the except part of a try statement (line 125)
    # SSA branch for the except 'KeyError' branch of a try statement (line 125)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 125)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 122)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_kind(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_kind' in the type store
    # Getting the type of 'stypy_return_type' (line 121)
    stypy_return_type_66958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66958)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_kind'
    return stypy_return_type_66958

# Assigning a type to the variable 'get_kind' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'get_kind', get_kind)

@norecursion
def islong_long(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'islong_long'
    module_type_store = module_type_store.open_function_context('islong_long', 131, 0, False)
    
    # Passed parameters checking function
    islong_long.stypy_localization = localization
    islong_long.stypy_type_of_self = None
    islong_long.stypy_type_store = module_type_store
    islong_long.stypy_function_name = 'islong_long'
    islong_long.stypy_param_names_list = ['var']
    islong_long.stypy_varargs_param_name = None
    islong_long.stypy_kwargs_param_name = None
    islong_long.stypy_call_defaults = defaults
    islong_long.stypy_call_varargs = varargs
    islong_long.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'islong_long', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'islong_long', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'islong_long(...)' code ##################

    
    
    
    # Call to isscalar(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'var' (line 132)
    var_66960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), 'var', False)
    # Processing the call keyword arguments (line 132)
    kwargs_66961 = {}
    # Getting the type of 'isscalar' (line 132)
    isscalar_66959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 132)
    isscalar_call_result_66962 = invoke(stypy.reporting.localization.Localization(__file__, 132, 11), isscalar_66959, *[var_66960], **kwargs_66961)
    
    # Applying the 'not' unary operator (line 132)
    result_not__66963 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 7), 'not', isscalar_call_result_66962)
    
    # Testing the type of an if condition (line 132)
    if_condition_66964 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 4), result_not__66963)
    # Assigning a type to the variable 'if_condition_66964' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'if_condition_66964', if_condition_66964)
    # SSA begins for if statement (line 132)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_66965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', int_66965)
    # SSA join for if statement (line 132)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to get(...): (line 134)
    # Processing the call arguments (line 134)
    str_66968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 15), 'str', 'typespec')
    # Processing the call keyword arguments (line 134)
    kwargs_66969 = {}
    # Getting the type of 'var' (line 134)
    var_66966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 7), 'var', False)
    # Obtaining the member 'get' of a type (line 134)
    get_66967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 7), var_66966, 'get')
    # Calling get(args, kwargs) (line 134)
    get_call_result_66970 = invoke(stypy.reporting.localization.Localization(__file__, 134, 7), get_66967, *[str_66968], **kwargs_66969)
    
    
    # Obtaining an instance of the builtin type 'list' (line 134)
    list_66971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 134)
    # Adding element type (line 134)
    str_66972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 35), 'str', 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 34), list_66971, str_66972)
    # Adding element type (line 134)
    str_66973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 46), 'str', 'logical')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 34), list_66971, str_66973)
    
    # Applying the binary operator 'notin' (line 134)
    result_contains_66974 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 7), 'notin', get_call_result_66970, list_66971)
    
    # Testing the type of an if condition (line 134)
    if_condition_66975 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 4), result_contains_66974)
    # Assigning a type to the variable 'if_condition_66975' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'if_condition_66975', if_condition_66975)
    # SSA begins for if statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_66976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'stypy_return_type', int_66976)
    # SSA join for if statement (line 134)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to get_kind(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'var' (line 136)
    var_66978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'var', False)
    # Processing the call keyword arguments (line 136)
    kwargs_66979 = {}
    # Getting the type of 'get_kind' (line 136)
    get_kind_66977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 136)
    get_kind_call_result_66980 = invoke(stypy.reporting.localization.Localization(__file__, 136, 11), get_kind_66977, *[var_66978], **kwargs_66979)
    
    str_66981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 28), 'str', '8')
    # Applying the binary operator '==' (line 136)
    result_eq_66982 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 11), '==', get_kind_call_result_66980, str_66981)
    
    # Assigning a type to the variable 'stypy_return_type' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type', result_eq_66982)
    
    # ################# End of 'islong_long(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'islong_long' in the type store
    # Getting the type of 'stypy_return_type' (line 131)
    stypy_return_type_66983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_66983)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'islong_long'
    return stypy_return_type_66983

# Assigning a type to the variable 'islong_long' (line 131)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'islong_long', islong_long)

@norecursion
def isunsigned_char(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isunsigned_char'
    module_type_store = module_type_store.open_function_context('isunsigned_char', 139, 0, False)
    
    # Passed parameters checking function
    isunsigned_char.stypy_localization = localization
    isunsigned_char.stypy_type_of_self = None
    isunsigned_char.stypy_type_store = module_type_store
    isunsigned_char.stypy_function_name = 'isunsigned_char'
    isunsigned_char.stypy_param_names_list = ['var']
    isunsigned_char.stypy_varargs_param_name = None
    isunsigned_char.stypy_kwargs_param_name = None
    isunsigned_char.stypy_call_defaults = defaults
    isunsigned_char.stypy_call_varargs = varargs
    isunsigned_char.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isunsigned_char', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isunsigned_char', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isunsigned_char(...)' code ##################

    
    
    
    # Call to isscalar(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'var' (line 140)
    var_66985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'var', False)
    # Processing the call keyword arguments (line 140)
    kwargs_66986 = {}
    # Getting the type of 'isscalar' (line 140)
    isscalar_66984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 140)
    isscalar_call_result_66987 = invoke(stypy.reporting.localization.Localization(__file__, 140, 11), isscalar_66984, *[var_66985], **kwargs_66986)
    
    # Applying the 'not' unary operator (line 140)
    result_not__66988 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 7), 'not', isscalar_call_result_66987)
    
    # Testing the type of an if condition (line 140)
    if_condition_66989 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 4), result_not__66988)
    # Assigning a type to the variable 'if_condition_66989' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'if_condition_66989', if_condition_66989)
    # SSA begins for if statement (line 140)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_66990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'stypy_return_type', int_66990)
    # SSA join for if statement (line 140)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to get(...): (line 142)
    # Processing the call arguments (line 142)
    str_66993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 15), 'str', 'typespec')
    # Processing the call keyword arguments (line 142)
    kwargs_66994 = {}
    # Getting the type of 'var' (line 142)
    var_66991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 7), 'var', False)
    # Obtaining the member 'get' of a type (line 142)
    get_66992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 7), var_66991, 'get')
    # Calling get(args, kwargs) (line 142)
    get_call_result_66995 = invoke(stypy.reporting.localization.Localization(__file__, 142, 7), get_66992, *[str_66993], **kwargs_66994)
    
    str_66996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 30), 'str', 'integer')
    # Applying the binary operator '!=' (line 142)
    result_ne_66997 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 7), '!=', get_call_result_66995, str_66996)
    
    # Testing the type of an if condition (line 142)
    if_condition_66998 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 4), result_ne_66997)
    # Assigning a type to the variable 'if_condition_66998' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'if_condition_66998', if_condition_66998)
    # SSA begins for if statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_66999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type', int_66999)
    # SSA join for if statement (line 142)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to get_kind(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'var' (line 144)
    var_67001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'var', False)
    # Processing the call keyword arguments (line 144)
    kwargs_67002 = {}
    # Getting the type of 'get_kind' (line 144)
    get_kind_67000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 144)
    get_kind_call_result_67003 = invoke(stypy.reporting.localization.Localization(__file__, 144, 11), get_kind_67000, *[var_67001], **kwargs_67002)
    
    str_67004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 28), 'str', '-1')
    # Applying the binary operator '==' (line 144)
    result_eq_67005 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 11), '==', get_kind_call_result_67003, str_67004)
    
    # Assigning a type to the variable 'stypy_return_type' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type', result_eq_67005)
    
    # ################# End of 'isunsigned_char(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isunsigned_char' in the type store
    # Getting the type of 'stypy_return_type' (line 139)
    stypy_return_type_67006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67006)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isunsigned_char'
    return stypy_return_type_67006

# Assigning a type to the variable 'isunsigned_char' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'isunsigned_char', isunsigned_char)

@norecursion
def isunsigned_short(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isunsigned_short'
    module_type_store = module_type_store.open_function_context('isunsigned_short', 147, 0, False)
    
    # Passed parameters checking function
    isunsigned_short.stypy_localization = localization
    isunsigned_short.stypy_type_of_self = None
    isunsigned_short.stypy_type_store = module_type_store
    isunsigned_short.stypy_function_name = 'isunsigned_short'
    isunsigned_short.stypy_param_names_list = ['var']
    isunsigned_short.stypy_varargs_param_name = None
    isunsigned_short.stypy_kwargs_param_name = None
    isunsigned_short.stypy_call_defaults = defaults
    isunsigned_short.stypy_call_varargs = varargs
    isunsigned_short.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isunsigned_short', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isunsigned_short', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isunsigned_short(...)' code ##################

    
    
    
    # Call to isscalar(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'var' (line 148)
    var_67008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'var', False)
    # Processing the call keyword arguments (line 148)
    kwargs_67009 = {}
    # Getting the type of 'isscalar' (line 148)
    isscalar_67007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 148)
    isscalar_call_result_67010 = invoke(stypy.reporting.localization.Localization(__file__, 148, 11), isscalar_67007, *[var_67008], **kwargs_67009)
    
    # Applying the 'not' unary operator (line 148)
    result_not__67011 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 7), 'not', isscalar_call_result_67010)
    
    # Testing the type of an if condition (line 148)
    if_condition_67012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), result_not__67011)
    # Assigning a type to the variable 'if_condition_67012' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_67012', if_condition_67012)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', int_67013)
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to get(...): (line 150)
    # Processing the call arguments (line 150)
    str_67016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 15), 'str', 'typespec')
    # Processing the call keyword arguments (line 150)
    kwargs_67017 = {}
    # Getting the type of 'var' (line 150)
    var_67014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 7), 'var', False)
    # Obtaining the member 'get' of a type (line 150)
    get_67015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 7), var_67014, 'get')
    # Calling get(args, kwargs) (line 150)
    get_call_result_67018 = invoke(stypy.reporting.localization.Localization(__file__, 150, 7), get_67015, *[str_67016], **kwargs_67017)
    
    str_67019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 30), 'str', 'integer')
    # Applying the binary operator '!=' (line 150)
    result_ne_67020 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 7), '!=', get_call_result_67018, str_67019)
    
    # Testing the type of an if condition (line 150)
    if_condition_67021 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 4), result_ne_67020)
    # Assigning a type to the variable 'if_condition_67021' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'if_condition_67021', if_condition_67021)
    # SSA begins for if statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'stypy_return_type', int_67022)
    # SSA join for if statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to get_kind(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'var' (line 152)
    var_67024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'var', False)
    # Processing the call keyword arguments (line 152)
    kwargs_67025 = {}
    # Getting the type of 'get_kind' (line 152)
    get_kind_67023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 152)
    get_kind_call_result_67026 = invoke(stypy.reporting.localization.Localization(__file__, 152, 11), get_kind_67023, *[var_67024], **kwargs_67025)
    
    str_67027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'str', '-2')
    # Applying the binary operator '==' (line 152)
    result_eq_67028 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 11), '==', get_kind_call_result_67026, str_67027)
    
    # Assigning a type to the variable 'stypy_return_type' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type', result_eq_67028)
    
    # ################# End of 'isunsigned_short(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isunsigned_short' in the type store
    # Getting the type of 'stypy_return_type' (line 147)
    stypy_return_type_67029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67029)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isunsigned_short'
    return stypy_return_type_67029

# Assigning a type to the variable 'isunsigned_short' (line 147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'isunsigned_short', isunsigned_short)

@norecursion
def isunsigned(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isunsigned'
    module_type_store = module_type_store.open_function_context('isunsigned', 155, 0, False)
    
    # Passed parameters checking function
    isunsigned.stypy_localization = localization
    isunsigned.stypy_type_of_self = None
    isunsigned.stypy_type_store = module_type_store
    isunsigned.stypy_function_name = 'isunsigned'
    isunsigned.stypy_param_names_list = ['var']
    isunsigned.stypy_varargs_param_name = None
    isunsigned.stypy_kwargs_param_name = None
    isunsigned.stypy_call_defaults = defaults
    isunsigned.stypy_call_varargs = varargs
    isunsigned.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isunsigned', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isunsigned', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isunsigned(...)' code ##################

    
    
    
    # Call to isscalar(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'var' (line 156)
    var_67031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'var', False)
    # Processing the call keyword arguments (line 156)
    kwargs_67032 = {}
    # Getting the type of 'isscalar' (line 156)
    isscalar_67030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 156)
    isscalar_call_result_67033 = invoke(stypy.reporting.localization.Localization(__file__, 156, 11), isscalar_67030, *[var_67031], **kwargs_67032)
    
    # Applying the 'not' unary operator (line 156)
    result_not__67034 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 7), 'not', isscalar_call_result_67033)
    
    # Testing the type of an if condition (line 156)
    if_condition_67035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 4), result_not__67034)
    # Assigning a type to the variable 'if_condition_67035' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'if_condition_67035', if_condition_67035)
    # SSA begins for if statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'stypy_return_type', int_67036)
    # SSA join for if statement (line 156)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to get(...): (line 158)
    # Processing the call arguments (line 158)
    str_67039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 15), 'str', 'typespec')
    # Processing the call keyword arguments (line 158)
    kwargs_67040 = {}
    # Getting the type of 'var' (line 158)
    var_67037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 7), 'var', False)
    # Obtaining the member 'get' of a type (line 158)
    get_67038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 7), var_67037, 'get')
    # Calling get(args, kwargs) (line 158)
    get_call_result_67041 = invoke(stypy.reporting.localization.Localization(__file__, 158, 7), get_67038, *[str_67039], **kwargs_67040)
    
    str_67042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 30), 'str', 'integer')
    # Applying the binary operator '!=' (line 158)
    result_ne_67043 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 7), '!=', get_call_result_67041, str_67042)
    
    # Testing the type of an if condition (line 158)
    if_condition_67044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 4), result_ne_67043)
    # Assigning a type to the variable 'if_condition_67044' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'if_condition_67044', if_condition_67044)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type', int_67045)
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to get_kind(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'var' (line 160)
    var_67047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'var', False)
    # Processing the call keyword arguments (line 160)
    kwargs_67048 = {}
    # Getting the type of 'get_kind' (line 160)
    get_kind_67046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 160)
    get_kind_call_result_67049 = invoke(stypy.reporting.localization.Localization(__file__, 160, 11), get_kind_67046, *[var_67047], **kwargs_67048)
    
    str_67050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 28), 'str', '-4')
    # Applying the binary operator '==' (line 160)
    result_eq_67051 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 11), '==', get_kind_call_result_67049, str_67050)
    
    # Assigning a type to the variable 'stypy_return_type' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type', result_eq_67051)
    
    # ################# End of 'isunsigned(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isunsigned' in the type store
    # Getting the type of 'stypy_return_type' (line 155)
    stypy_return_type_67052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67052)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isunsigned'
    return stypy_return_type_67052

# Assigning a type to the variable 'isunsigned' (line 155)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'isunsigned', isunsigned)

@norecursion
def isunsigned_long_long(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isunsigned_long_long'
    module_type_store = module_type_store.open_function_context('isunsigned_long_long', 163, 0, False)
    
    # Passed parameters checking function
    isunsigned_long_long.stypy_localization = localization
    isunsigned_long_long.stypy_type_of_self = None
    isunsigned_long_long.stypy_type_store = module_type_store
    isunsigned_long_long.stypy_function_name = 'isunsigned_long_long'
    isunsigned_long_long.stypy_param_names_list = ['var']
    isunsigned_long_long.stypy_varargs_param_name = None
    isunsigned_long_long.stypy_kwargs_param_name = None
    isunsigned_long_long.stypy_call_defaults = defaults
    isunsigned_long_long.stypy_call_varargs = varargs
    isunsigned_long_long.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isunsigned_long_long', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isunsigned_long_long', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isunsigned_long_long(...)' code ##################

    
    
    
    # Call to isscalar(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'var' (line 164)
    var_67054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'var', False)
    # Processing the call keyword arguments (line 164)
    kwargs_67055 = {}
    # Getting the type of 'isscalar' (line 164)
    isscalar_67053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 164)
    isscalar_call_result_67056 = invoke(stypy.reporting.localization.Localization(__file__, 164, 11), isscalar_67053, *[var_67054], **kwargs_67055)
    
    # Applying the 'not' unary operator (line 164)
    result_not__67057 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 7), 'not', isscalar_call_result_67056)
    
    # Testing the type of an if condition (line 164)
    if_condition_67058 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 4), result_not__67057)
    # Assigning a type to the variable 'if_condition_67058' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'if_condition_67058', if_condition_67058)
    # SSA begins for if statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'stypy_return_type', int_67059)
    # SSA join for if statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to get(...): (line 166)
    # Processing the call arguments (line 166)
    str_67062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 15), 'str', 'typespec')
    # Processing the call keyword arguments (line 166)
    kwargs_67063 = {}
    # Getting the type of 'var' (line 166)
    var_67060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 7), 'var', False)
    # Obtaining the member 'get' of a type (line 166)
    get_67061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 7), var_67060, 'get')
    # Calling get(args, kwargs) (line 166)
    get_call_result_67064 = invoke(stypy.reporting.localization.Localization(__file__, 166, 7), get_67061, *[str_67062], **kwargs_67063)
    
    str_67065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 30), 'str', 'integer')
    # Applying the binary operator '!=' (line 166)
    result_ne_67066 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 7), '!=', get_call_result_67064, str_67065)
    
    # Testing the type of an if condition (line 166)
    if_condition_67067 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 4), result_ne_67066)
    # Assigning a type to the variable 'if_condition_67067' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'if_condition_67067', if_condition_67067)
    # SSA begins for if statement (line 166)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'stypy_return_type', int_67068)
    # SSA join for if statement (line 166)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to get_kind(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'var' (line 168)
    var_67070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'var', False)
    # Processing the call keyword arguments (line 168)
    kwargs_67071 = {}
    # Getting the type of 'get_kind' (line 168)
    get_kind_67069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 168)
    get_kind_call_result_67072 = invoke(stypy.reporting.localization.Localization(__file__, 168, 11), get_kind_67069, *[var_67070], **kwargs_67071)
    
    str_67073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 28), 'str', '-8')
    # Applying the binary operator '==' (line 168)
    result_eq_67074 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 11), '==', get_kind_call_result_67072, str_67073)
    
    # Assigning a type to the variable 'stypy_return_type' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type', result_eq_67074)
    
    # ################# End of 'isunsigned_long_long(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isunsigned_long_long' in the type store
    # Getting the type of 'stypy_return_type' (line 163)
    stypy_return_type_67075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67075)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isunsigned_long_long'
    return stypy_return_type_67075

# Assigning a type to the variable 'isunsigned_long_long' (line 163)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 0), 'isunsigned_long_long', isunsigned_long_long)

@norecursion
def isdouble(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isdouble'
    module_type_store = module_type_store.open_function_context('isdouble', 171, 0, False)
    
    # Passed parameters checking function
    isdouble.stypy_localization = localization
    isdouble.stypy_type_of_self = None
    isdouble.stypy_type_store = module_type_store
    isdouble.stypy_function_name = 'isdouble'
    isdouble.stypy_param_names_list = ['var']
    isdouble.stypy_varargs_param_name = None
    isdouble.stypy_kwargs_param_name = None
    isdouble.stypy_call_defaults = defaults
    isdouble.stypy_call_varargs = varargs
    isdouble.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isdouble', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isdouble', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isdouble(...)' code ##################

    
    
    
    # Call to isscalar(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'var' (line 172)
    var_67077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'var', False)
    # Processing the call keyword arguments (line 172)
    kwargs_67078 = {}
    # Getting the type of 'isscalar' (line 172)
    isscalar_67076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 172)
    isscalar_call_result_67079 = invoke(stypy.reporting.localization.Localization(__file__, 172, 11), isscalar_67076, *[var_67077], **kwargs_67078)
    
    # Applying the 'not' unary operator (line 172)
    result_not__67080 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 7), 'not', isscalar_call_result_67079)
    
    # Testing the type of an if condition (line 172)
    if_condition_67081 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 4), result_not__67080)
    # Assigning a type to the variable 'if_condition_67081' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'if_condition_67081', if_condition_67081)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', int_67082)
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Call to get(...): (line 174)
    # Processing the call arguments (line 174)
    str_67085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 19), 'str', 'typespec')
    # Processing the call keyword arguments (line 174)
    kwargs_67086 = {}
    # Getting the type of 'var' (line 174)
    var_67083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 'var', False)
    # Obtaining the member 'get' of a type (line 174)
    get_67084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 11), var_67083, 'get')
    # Calling get(args, kwargs) (line 174)
    get_call_result_67087 = invoke(stypy.reporting.localization.Localization(__file__, 174, 11), get_67084, *[str_67085], **kwargs_67086)
    
    str_67088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 34), 'str', 'real')
    # Applying the binary operator '==' (line 174)
    result_eq_67089 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 11), '==', get_call_result_67087, str_67088)
    
    # Applying the 'not' unary operator (line 174)
    result_not__67090 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 7), 'not', result_eq_67089)
    
    # Testing the type of an if condition (line 174)
    if_condition_67091 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 4), result_not__67090)
    # Assigning a type to the variable 'if_condition_67091' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'if_condition_67091', if_condition_67091)
    # SSA begins for if statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', int_67092)
    # SSA join for if statement (line 174)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to get_kind(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'var' (line 176)
    var_67094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 20), 'var', False)
    # Processing the call keyword arguments (line 176)
    kwargs_67095 = {}
    # Getting the type of 'get_kind' (line 176)
    get_kind_67093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 176)
    get_kind_call_result_67096 = invoke(stypy.reporting.localization.Localization(__file__, 176, 11), get_kind_67093, *[var_67094], **kwargs_67095)
    
    str_67097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'str', '8')
    # Applying the binary operator '==' (line 176)
    result_eq_67098 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 11), '==', get_kind_call_result_67096, str_67097)
    
    # Assigning a type to the variable 'stypy_return_type' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type', result_eq_67098)
    
    # ################# End of 'isdouble(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isdouble' in the type store
    # Getting the type of 'stypy_return_type' (line 171)
    stypy_return_type_67099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67099)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isdouble'
    return stypy_return_type_67099

# Assigning a type to the variable 'isdouble' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'isdouble', isdouble)

@norecursion
def islong_double(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'islong_double'
    module_type_store = module_type_store.open_function_context('islong_double', 179, 0, False)
    
    # Passed parameters checking function
    islong_double.stypy_localization = localization
    islong_double.stypy_type_of_self = None
    islong_double.stypy_type_store = module_type_store
    islong_double.stypy_function_name = 'islong_double'
    islong_double.stypy_param_names_list = ['var']
    islong_double.stypy_varargs_param_name = None
    islong_double.stypy_kwargs_param_name = None
    islong_double.stypy_call_defaults = defaults
    islong_double.stypy_call_varargs = varargs
    islong_double.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'islong_double', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'islong_double', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'islong_double(...)' code ##################

    
    
    
    # Call to isscalar(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'var' (line 180)
    var_67101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 20), 'var', False)
    # Processing the call keyword arguments (line 180)
    kwargs_67102 = {}
    # Getting the type of 'isscalar' (line 180)
    isscalar_67100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'isscalar', False)
    # Calling isscalar(args, kwargs) (line 180)
    isscalar_call_result_67103 = invoke(stypy.reporting.localization.Localization(__file__, 180, 11), isscalar_67100, *[var_67101], **kwargs_67102)
    
    # Applying the 'not' unary operator (line 180)
    result_not__67104 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 7), 'not', isscalar_call_result_67103)
    
    # Testing the type of an if condition (line 180)
    if_condition_67105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 4), result_not__67104)
    # Assigning a type to the variable 'if_condition_67105' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'if_condition_67105', if_condition_67105)
    # SSA begins for if statement (line 180)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'stypy_return_type', int_67106)
    # SSA join for if statement (line 180)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    
    # Call to get(...): (line 182)
    # Processing the call arguments (line 182)
    str_67109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 19), 'str', 'typespec')
    # Processing the call keyword arguments (line 182)
    kwargs_67110 = {}
    # Getting the type of 'var' (line 182)
    var_67107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'var', False)
    # Obtaining the member 'get' of a type (line 182)
    get_67108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 11), var_67107, 'get')
    # Calling get(args, kwargs) (line 182)
    get_call_result_67111 = invoke(stypy.reporting.localization.Localization(__file__, 182, 11), get_67108, *[str_67109], **kwargs_67110)
    
    str_67112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 34), 'str', 'real')
    # Applying the binary operator '==' (line 182)
    result_eq_67113 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 11), '==', get_call_result_67111, str_67112)
    
    # Applying the 'not' unary operator (line 182)
    result_not__67114 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 7), 'not', result_eq_67113)
    
    # Testing the type of an if condition (line 182)
    if_condition_67115 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 4), result_not__67114)
    # Assigning a type to the variable 'if_condition_67115' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'if_condition_67115', if_condition_67115)
    # SSA begins for if statement (line 182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stypy_return_type', int_67116)
    # SSA join for if statement (line 182)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to get_kind(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'var' (line 184)
    var_67118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'var', False)
    # Processing the call keyword arguments (line 184)
    kwargs_67119 = {}
    # Getting the type of 'get_kind' (line 184)
    get_kind_67117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 184)
    get_kind_call_result_67120 = invoke(stypy.reporting.localization.Localization(__file__, 184, 11), get_kind_67117, *[var_67118], **kwargs_67119)
    
    str_67121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 28), 'str', '16')
    # Applying the binary operator '==' (line 184)
    result_eq_67122 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 11), '==', get_kind_call_result_67120, str_67121)
    
    # Assigning a type to the variable 'stypy_return_type' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type', result_eq_67122)
    
    # ################# End of 'islong_double(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'islong_double' in the type store
    # Getting the type of 'stypy_return_type' (line 179)
    stypy_return_type_67123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67123)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'islong_double'
    return stypy_return_type_67123

# Assigning a type to the variable 'islong_double' (line 179)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'islong_double', islong_double)

@norecursion
def islong_complex(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'islong_complex'
    module_type_store = module_type_store.open_function_context('islong_complex', 187, 0, False)
    
    # Passed parameters checking function
    islong_complex.stypy_localization = localization
    islong_complex.stypy_type_of_self = None
    islong_complex.stypy_type_store = module_type_store
    islong_complex.stypy_function_name = 'islong_complex'
    islong_complex.stypy_param_names_list = ['var']
    islong_complex.stypy_varargs_param_name = None
    islong_complex.stypy_kwargs_param_name = None
    islong_complex.stypy_call_defaults = defaults
    islong_complex.stypy_call_varargs = varargs
    islong_complex.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'islong_complex', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'islong_complex', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'islong_complex(...)' code ##################

    
    
    
    # Call to iscomplex(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'var' (line 188)
    var_67125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 21), 'var', False)
    # Processing the call keyword arguments (line 188)
    kwargs_67126 = {}
    # Getting the type of 'iscomplex' (line 188)
    iscomplex_67124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'iscomplex', False)
    # Calling iscomplex(args, kwargs) (line 188)
    iscomplex_call_result_67127 = invoke(stypy.reporting.localization.Localization(__file__, 188, 11), iscomplex_67124, *[var_67125], **kwargs_67126)
    
    # Applying the 'not' unary operator (line 188)
    result_not__67128 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 7), 'not', iscomplex_call_result_67127)
    
    # Testing the type of an if condition (line 188)
    if_condition_67129 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 4), result_not__67128)
    # Assigning a type to the variable 'if_condition_67129' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'if_condition_67129', if_condition_67129)
    # SSA begins for if statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'stypy_return_type', int_67130)
    # SSA join for if statement (line 188)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to get_kind(...): (line 190)
    # Processing the call arguments (line 190)
    # Getting the type of 'var' (line 190)
    var_67132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 20), 'var', False)
    # Processing the call keyword arguments (line 190)
    kwargs_67133 = {}
    # Getting the type of 'get_kind' (line 190)
    get_kind_67131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 190)
    get_kind_call_result_67134 = invoke(stypy.reporting.localization.Localization(__file__, 190, 11), get_kind_67131, *[var_67132], **kwargs_67133)
    
    str_67135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 28), 'str', '32')
    # Applying the binary operator '==' (line 190)
    result_eq_67136 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 11), '==', get_kind_call_result_67134, str_67135)
    
    # Assigning a type to the variable 'stypy_return_type' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'stypy_return_type', result_eq_67136)
    
    # ################# End of 'islong_complex(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'islong_complex' in the type store
    # Getting the type of 'stypy_return_type' (line 187)
    stypy_return_type_67137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67137)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'islong_complex'
    return stypy_return_type_67137

# Assigning a type to the variable 'islong_complex' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'islong_complex', islong_complex)

@norecursion
def iscomplexarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iscomplexarray'
    module_type_store = module_type_store.open_function_context('iscomplexarray', 193, 0, False)
    
    # Passed parameters checking function
    iscomplexarray.stypy_localization = localization
    iscomplexarray.stypy_type_of_self = None
    iscomplexarray.stypy_type_store = module_type_store
    iscomplexarray.stypy_function_name = 'iscomplexarray'
    iscomplexarray.stypy_param_names_list = ['var']
    iscomplexarray.stypy_varargs_param_name = None
    iscomplexarray.stypy_kwargs_param_name = None
    iscomplexarray.stypy_call_defaults = defaults
    iscomplexarray.stypy_call_varargs = varargs
    iscomplexarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iscomplexarray', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iscomplexarray', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iscomplexarray(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isarray(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'var' (line 194)
    var_67139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 19), 'var', False)
    # Processing the call keyword arguments (line 194)
    kwargs_67140 = {}
    # Getting the type of 'isarray' (line 194)
    isarray_67138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 11), 'isarray', False)
    # Calling isarray(args, kwargs) (line 194)
    isarray_call_result_67141 = invoke(stypy.reporting.localization.Localization(__file__, 194, 11), isarray_67138, *[var_67139], **kwargs_67140)
    
    
    
    # Call to get(...): (line 195)
    # Processing the call arguments (line 195)
    str_67144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 19), 'str', 'typespec')
    # Processing the call keyword arguments (line 195)
    kwargs_67145 = {}
    # Getting the type of 'var' (line 195)
    var_67142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 11), 'var', False)
    # Obtaining the member 'get' of a type (line 195)
    get_67143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 11), var_67142, 'get')
    # Calling get(args, kwargs) (line 195)
    get_call_result_67146 = invoke(stypy.reporting.localization.Localization(__file__, 195, 11), get_67143, *[str_67144], **kwargs_67145)
    
    
    # Obtaining an instance of the builtin type 'list' (line 195)
    list_67147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 195)
    # Adding element type (line 195)
    str_67148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 35), 'str', 'complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 34), list_67147, str_67148)
    # Adding element type (line 195)
    str_67149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 46), 'str', 'double complex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 195, 34), list_67147, str_67149)
    
    # Applying the binary operator 'in' (line 195)
    result_contains_67150 = python_operator(stypy.reporting.localization.Localization(__file__, 195, 11), 'in', get_call_result_67146, list_67147)
    
    # Applying the binary operator 'and' (line 194)
    result_and_keyword_67151 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 11), 'and', isarray_call_result_67141, result_contains_67150)
    
    # Assigning a type to the variable 'stypy_return_type' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'stypy_return_type', result_and_keyword_67151)
    
    # ################# End of 'iscomplexarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iscomplexarray' in the type store
    # Getting the type of 'stypy_return_type' (line 193)
    stypy_return_type_67152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67152)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iscomplexarray'
    return stypy_return_type_67152

# Assigning a type to the variable 'iscomplexarray' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'iscomplexarray', iscomplexarray)

@norecursion
def isint1array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isint1array'
    module_type_store = module_type_store.open_function_context('isint1array', 198, 0, False)
    
    # Passed parameters checking function
    isint1array.stypy_localization = localization
    isint1array.stypy_type_of_self = None
    isint1array.stypy_type_store = module_type_store
    isint1array.stypy_function_name = 'isint1array'
    isint1array.stypy_param_names_list = ['var']
    isint1array.stypy_varargs_param_name = None
    isint1array.stypy_kwargs_param_name = None
    isint1array.stypy_call_defaults = defaults
    isint1array.stypy_call_varargs = varargs
    isint1array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isint1array', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isint1array', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isint1array(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isarray(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'var' (line 199)
    var_67154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'var', False)
    # Processing the call keyword arguments (line 199)
    kwargs_67155 = {}
    # Getting the type of 'isarray' (line 199)
    isarray_67153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'isarray', False)
    # Calling isarray(args, kwargs) (line 199)
    isarray_call_result_67156 = invoke(stypy.reporting.localization.Localization(__file__, 199, 11), isarray_67153, *[var_67154], **kwargs_67155)
    
    
    
    # Call to get(...): (line 199)
    # Processing the call arguments (line 199)
    str_67159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 36), 'str', 'typespec')
    # Processing the call keyword arguments (line 199)
    kwargs_67160 = {}
    # Getting the type of 'var' (line 199)
    var_67157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'var', False)
    # Obtaining the member 'get' of a type (line 199)
    get_67158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 28), var_67157, 'get')
    # Calling get(args, kwargs) (line 199)
    get_call_result_67161 = invoke(stypy.reporting.localization.Localization(__file__, 199, 28), get_67158, *[str_67159], **kwargs_67160)
    
    str_67162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 51), 'str', 'integer')
    # Applying the binary operator '==' (line 199)
    result_eq_67163 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 28), '==', get_call_result_67161, str_67162)
    
    # Applying the binary operator 'and' (line 199)
    result_and_keyword_67164 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 11), 'and', isarray_call_result_67156, result_eq_67163)
    
    
    # Call to get_kind(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'var' (line 200)
    var_67166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 21), 'var', False)
    # Processing the call keyword arguments (line 200)
    kwargs_67167 = {}
    # Getting the type of 'get_kind' (line 200)
    get_kind_67165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 200)
    get_kind_call_result_67168 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), get_kind_67165, *[var_67166], **kwargs_67167)
    
    str_67169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 29), 'str', '1')
    # Applying the binary operator '==' (line 200)
    result_eq_67170 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 12), '==', get_kind_call_result_67168, str_67169)
    
    # Applying the binary operator 'and' (line 199)
    result_and_keyword_67171 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 11), 'and', result_and_keyword_67164, result_eq_67170)
    
    # Assigning a type to the variable 'stypy_return_type' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'stypy_return_type', result_and_keyword_67171)
    
    # ################# End of 'isint1array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isint1array' in the type store
    # Getting the type of 'stypy_return_type' (line 198)
    stypy_return_type_67172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67172)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isint1array'
    return stypy_return_type_67172

# Assigning a type to the variable 'isint1array' (line 198)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 0), 'isint1array', isint1array)

@norecursion
def isunsigned_chararray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isunsigned_chararray'
    module_type_store = module_type_store.open_function_context('isunsigned_chararray', 203, 0, False)
    
    # Passed parameters checking function
    isunsigned_chararray.stypy_localization = localization
    isunsigned_chararray.stypy_type_of_self = None
    isunsigned_chararray.stypy_type_store = module_type_store
    isunsigned_chararray.stypy_function_name = 'isunsigned_chararray'
    isunsigned_chararray.stypy_param_names_list = ['var']
    isunsigned_chararray.stypy_varargs_param_name = None
    isunsigned_chararray.stypy_kwargs_param_name = None
    isunsigned_chararray.stypy_call_defaults = defaults
    isunsigned_chararray.stypy_call_varargs = varargs
    isunsigned_chararray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isunsigned_chararray', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isunsigned_chararray', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isunsigned_chararray(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isarray(...): (line 204)
    # Processing the call arguments (line 204)
    # Getting the type of 'var' (line 204)
    var_67174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 19), 'var', False)
    # Processing the call keyword arguments (line 204)
    kwargs_67175 = {}
    # Getting the type of 'isarray' (line 204)
    isarray_67173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'isarray', False)
    # Calling isarray(args, kwargs) (line 204)
    isarray_call_result_67176 = invoke(stypy.reporting.localization.Localization(__file__, 204, 11), isarray_67173, *[var_67174], **kwargs_67175)
    
    
    
    # Call to get(...): (line 204)
    # Processing the call arguments (line 204)
    str_67179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 36), 'str', 'typespec')
    # Processing the call keyword arguments (line 204)
    kwargs_67180 = {}
    # Getting the type of 'var' (line 204)
    var_67177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 28), 'var', False)
    # Obtaining the member 'get' of a type (line 204)
    get_67178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 28), var_67177, 'get')
    # Calling get(args, kwargs) (line 204)
    get_call_result_67181 = invoke(stypy.reporting.localization.Localization(__file__, 204, 28), get_67178, *[str_67179], **kwargs_67180)
    
    
    # Obtaining an instance of the builtin type 'list' (line 204)
    list_67182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 204)
    # Adding element type (line 204)
    str_67183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 52), 'str', 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 51), list_67182, str_67183)
    # Adding element type (line 204)
    str_67184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 63), 'str', 'logical')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 51), list_67182, str_67184)
    
    # Applying the binary operator 'in' (line 204)
    result_contains_67185 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 28), 'in', get_call_result_67181, list_67182)
    
    # Applying the binary operator 'and' (line 204)
    result_and_keyword_67186 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 11), 'and', isarray_call_result_67176, result_contains_67185)
    
    
    # Call to get_kind(...): (line 205)
    # Processing the call arguments (line 205)
    # Getting the type of 'var' (line 205)
    var_67188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 21), 'var', False)
    # Processing the call keyword arguments (line 205)
    kwargs_67189 = {}
    # Getting the type of 'get_kind' (line 205)
    get_kind_67187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 205)
    get_kind_call_result_67190 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), get_kind_67187, *[var_67188], **kwargs_67189)
    
    str_67191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 29), 'str', '-1')
    # Applying the binary operator '==' (line 205)
    result_eq_67192 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 12), '==', get_kind_call_result_67190, str_67191)
    
    # Applying the binary operator 'and' (line 204)
    result_and_keyword_67193 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 11), 'and', result_and_keyword_67186, result_eq_67192)
    
    # Assigning a type to the variable 'stypy_return_type' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type', result_and_keyword_67193)
    
    # ################# End of 'isunsigned_chararray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isunsigned_chararray' in the type store
    # Getting the type of 'stypy_return_type' (line 203)
    stypy_return_type_67194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67194)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isunsigned_chararray'
    return stypy_return_type_67194

# Assigning a type to the variable 'isunsigned_chararray' (line 203)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 0), 'isunsigned_chararray', isunsigned_chararray)

@norecursion
def isunsigned_shortarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isunsigned_shortarray'
    module_type_store = module_type_store.open_function_context('isunsigned_shortarray', 208, 0, False)
    
    # Passed parameters checking function
    isunsigned_shortarray.stypy_localization = localization
    isunsigned_shortarray.stypy_type_of_self = None
    isunsigned_shortarray.stypy_type_store = module_type_store
    isunsigned_shortarray.stypy_function_name = 'isunsigned_shortarray'
    isunsigned_shortarray.stypy_param_names_list = ['var']
    isunsigned_shortarray.stypy_varargs_param_name = None
    isunsigned_shortarray.stypy_kwargs_param_name = None
    isunsigned_shortarray.stypy_call_defaults = defaults
    isunsigned_shortarray.stypy_call_varargs = varargs
    isunsigned_shortarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isunsigned_shortarray', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isunsigned_shortarray', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isunsigned_shortarray(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isarray(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'var' (line 209)
    var_67196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 19), 'var', False)
    # Processing the call keyword arguments (line 209)
    kwargs_67197 = {}
    # Getting the type of 'isarray' (line 209)
    isarray_67195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 11), 'isarray', False)
    # Calling isarray(args, kwargs) (line 209)
    isarray_call_result_67198 = invoke(stypy.reporting.localization.Localization(__file__, 209, 11), isarray_67195, *[var_67196], **kwargs_67197)
    
    
    
    # Call to get(...): (line 209)
    # Processing the call arguments (line 209)
    str_67201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 36), 'str', 'typespec')
    # Processing the call keyword arguments (line 209)
    kwargs_67202 = {}
    # Getting the type of 'var' (line 209)
    var_67199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 28), 'var', False)
    # Obtaining the member 'get' of a type (line 209)
    get_67200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 28), var_67199, 'get')
    # Calling get(args, kwargs) (line 209)
    get_call_result_67203 = invoke(stypy.reporting.localization.Localization(__file__, 209, 28), get_67200, *[str_67201], **kwargs_67202)
    
    
    # Obtaining an instance of the builtin type 'list' (line 209)
    list_67204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 209)
    # Adding element type (line 209)
    str_67205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 52), 'str', 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 51), list_67204, str_67205)
    # Adding element type (line 209)
    str_67206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 63), 'str', 'logical')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 51), list_67204, str_67206)
    
    # Applying the binary operator 'in' (line 209)
    result_contains_67207 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 28), 'in', get_call_result_67203, list_67204)
    
    # Applying the binary operator 'and' (line 209)
    result_and_keyword_67208 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 11), 'and', isarray_call_result_67198, result_contains_67207)
    
    
    # Call to get_kind(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'var' (line 210)
    var_67210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 21), 'var', False)
    # Processing the call keyword arguments (line 210)
    kwargs_67211 = {}
    # Getting the type of 'get_kind' (line 210)
    get_kind_67209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 210)
    get_kind_call_result_67212 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), get_kind_67209, *[var_67210], **kwargs_67211)
    
    str_67213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 29), 'str', '-2')
    # Applying the binary operator '==' (line 210)
    result_eq_67214 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 12), '==', get_kind_call_result_67212, str_67213)
    
    # Applying the binary operator 'and' (line 209)
    result_and_keyword_67215 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 11), 'and', result_and_keyword_67208, result_eq_67214)
    
    # Assigning a type to the variable 'stypy_return_type' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type', result_and_keyword_67215)
    
    # ################# End of 'isunsigned_shortarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isunsigned_shortarray' in the type store
    # Getting the type of 'stypy_return_type' (line 208)
    stypy_return_type_67216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67216)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isunsigned_shortarray'
    return stypy_return_type_67216

# Assigning a type to the variable 'isunsigned_shortarray' (line 208)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 0), 'isunsigned_shortarray', isunsigned_shortarray)

@norecursion
def isunsignedarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isunsignedarray'
    module_type_store = module_type_store.open_function_context('isunsignedarray', 213, 0, False)
    
    # Passed parameters checking function
    isunsignedarray.stypy_localization = localization
    isunsignedarray.stypy_type_of_self = None
    isunsignedarray.stypy_type_store = module_type_store
    isunsignedarray.stypy_function_name = 'isunsignedarray'
    isunsignedarray.stypy_param_names_list = ['var']
    isunsignedarray.stypy_varargs_param_name = None
    isunsignedarray.stypy_kwargs_param_name = None
    isunsignedarray.stypy_call_defaults = defaults
    isunsignedarray.stypy_call_varargs = varargs
    isunsignedarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isunsignedarray', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isunsignedarray', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isunsignedarray(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isarray(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'var' (line 214)
    var_67218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 19), 'var', False)
    # Processing the call keyword arguments (line 214)
    kwargs_67219 = {}
    # Getting the type of 'isarray' (line 214)
    isarray_67217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'isarray', False)
    # Calling isarray(args, kwargs) (line 214)
    isarray_call_result_67220 = invoke(stypy.reporting.localization.Localization(__file__, 214, 11), isarray_67217, *[var_67218], **kwargs_67219)
    
    
    
    # Call to get(...): (line 214)
    # Processing the call arguments (line 214)
    str_67223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 36), 'str', 'typespec')
    # Processing the call keyword arguments (line 214)
    kwargs_67224 = {}
    # Getting the type of 'var' (line 214)
    var_67221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'var', False)
    # Obtaining the member 'get' of a type (line 214)
    get_67222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 28), var_67221, 'get')
    # Calling get(args, kwargs) (line 214)
    get_call_result_67225 = invoke(stypy.reporting.localization.Localization(__file__, 214, 28), get_67222, *[str_67223], **kwargs_67224)
    
    
    # Obtaining an instance of the builtin type 'list' (line 214)
    list_67226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 214)
    # Adding element type (line 214)
    str_67227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 52), 'str', 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 51), list_67226, str_67227)
    # Adding element type (line 214)
    str_67228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 63), 'str', 'logical')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 51), list_67226, str_67228)
    
    # Applying the binary operator 'in' (line 214)
    result_contains_67229 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 28), 'in', get_call_result_67225, list_67226)
    
    # Applying the binary operator 'and' (line 214)
    result_and_keyword_67230 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 11), 'and', isarray_call_result_67220, result_contains_67229)
    
    
    # Call to get_kind(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'var' (line 215)
    var_67232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'var', False)
    # Processing the call keyword arguments (line 215)
    kwargs_67233 = {}
    # Getting the type of 'get_kind' (line 215)
    get_kind_67231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 215)
    get_kind_call_result_67234 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), get_kind_67231, *[var_67232], **kwargs_67233)
    
    str_67235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 29), 'str', '-4')
    # Applying the binary operator '==' (line 215)
    result_eq_67236 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 12), '==', get_kind_call_result_67234, str_67235)
    
    # Applying the binary operator 'and' (line 214)
    result_and_keyword_67237 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 11), 'and', result_and_keyword_67230, result_eq_67236)
    
    # Assigning a type to the variable 'stypy_return_type' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type', result_and_keyword_67237)
    
    # ################# End of 'isunsignedarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isunsignedarray' in the type store
    # Getting the type of 'stypy_return_type' (line 213)
    stypy_return_type_67238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67238)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isunsignedarray'
    return stypy_return_type_67238

# Assigning a type to the variable 'isunsignedarray' (line 213)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 0), 'isunsignedarray', isunsignedarray)

@norecursion
def isunsigned_long_longarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isunsigned_long_longarray'
    module_type_store = module_type_store.open_function_context('isunsigned_long_longarray', 218, 0, False)
    
    # Passed parameters checking function
    isunsigned_long_longarray.stypy_localization = localization
    isunsigned_long_longarray.stypy_type_of_self = None
    isunsigned_long_longarray.stypy_type_store = module_type_store
    isunsigned_long_longarray.stypy_function_name = 'isunsigned_long_longarray'
    isunsigned_long_longarray.stypy_param_names_list = ['var']
    isunsigned_long_longarray.stypy_varargs_param_name = None
    isunsigned_long_longarray.stypy_kwargs_param_name = None
    isunsigned_long_longarray.stypy_call_defaults = defaults
    isunsigned_long_longarray.stypy_call_varargs = varargs
    isunsigned_long_longarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isunsigned_long_longarray', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isunsigned_long_longarray', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isunsigned_long_longarray(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isarray(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'var' (line 219)
    var_67240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 19), 'var', False)
    # Processing the call keyword arguments (line 219)
    kwargs_67241 = {}
    # Getting the type of 'isarray' (line 219)
    isarray_67239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 11), 'isarray', False)
    # Calling isarray(args, kwargs) (line 219)
    isarray_call_result_67242 = invoke(stypy.reporting.localization.Localization(__file__, 219, 11), isarray_67239, *[var_67240], **kwargs_67241)
    
    
    
    # Call to get(...): (line 219)
    # Processing the call arguments (line 219)
    str_67245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 36), 'str', 'typespec')
    # Processing the call keyword arguments (line 219)
    kwargs_67246 = {}
    # Getting the type of 'var' (line 219)
    var_67243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'var', False)
    # Obtaining the member 'get' of a type (line 219)
    get_67244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), var_67243, 'get')
    # Calling get(args, kwargs) (line 219)
    get_call_result_67247 = invoke(stypy.reporting.localization.Localization(__file__, 219, 28), get_67244, *[str_67245], **kwargs_67246)
    
    
    # Obtaining an instance of the builtin type 'list' (line 219)
    list_67248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 219)
    # Adding element type (line 219)
    str_67249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 52), 'str', 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 51), list_67248, str_67249)
    # Adding element type (line 219)
    str_67250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 63), 'str', 'logical')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 51), list_67248, str_67250)
    
    # Applying the binary operator 'in' (line 219)
    result_contains_67251 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 28), 'in', get_call_result_67247, list_67248)
    
    # Applying the binary operator 'and' (line 219)
    result_and_keyword_67252 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 11), 'and', isarray_call_result_67242, result_contains_67251)
    
    
    # Call to get_kind(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'var' (line 220)
    var_67254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'var', False)
    # Processing the call keyword arguments (line 220)
    kwargs_67255 = {}
    # Getting the type of 'get_kind' (line 220)
    get_kind_67253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 220)
    get_kind_call_result_67256 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), get_kind_67253, *[var_67254], **kwargs_67255)
    
    str_67257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 29), 'str', '-8')
    # Applying the binary operator '==' (line 220)
    result_eq_67258 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 12), '==', get_kind_call_result_67256, str_67257)
    
    # Applying the binary operator 'and' (line 219)
    result_and_keyword_67259 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 11), 'and', result_and_keyword_67252, result_eq_67258)
    
    # Assigning a type to the variable 'stypy_return_type' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type', result_and_keyword_67259)
    
    # ################# End of 'isunsigned_long_longarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isunsigned_long_longarray' in the type store
    # Getting the type of 'stypy_return_type' (line 218)
    stypy_return_type_67260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67260)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isunsigned_long_longarray'
    return stypy_return_type_67260

# Assigning a type to the variable 'isunsigned_long_longarray' (line 218)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 0), 'isunsigned_long_longarray', isunsigned_long_longarray)

@norecursion
def issigned_chararray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'issigned_chararray'
    module_type_store = module_type_store.open_function_context('issigned_chararray', 223, 0, False)
    
    # Passed parameters checking function
    issigned_chararray.stypy_localization = localization
    issigned_chararray.stypy_type_of_self = None
    issigned_chararray.stypy_type_store = module_type_store
    issigned_chararray.stypy_function_name = 'issigned_chararray'
    issigned_chararray.stypy_param_names_list = ['var']
    issigned_chararray.stypy_varargs_param_name = None
    issigned_chararray.stypy_kwargs_param_name = None
    issigned_chararray.stypy_call_defaults = defaults
    issigned_chararray.stypy_call_varargs = varargs
    issigned_chararray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'issigned_chararray', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'issigned_chararray', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'issigned_chararray(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isarray(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'var' (line 224)
    var_67262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'var', False)
    # Processing the call keyword arguments (line 224)
    kwargs_67263 = {}
    # Getting the type of 'isarray' (line 224)
    isarray_67261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 11), 'isarray', False)
    # Calling isarray(args, kwargs) (line 224)
    isarray_call_result_67264 = invoke(stypy.reporting.localization.Localization(__file__, 224, 11), isarray_67261, *[var_67262], **kwargs_67263)
    
    
    
    # Call to get(...): (line 224)
    # Processing the call arguments (line 224)
    str_67267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 36), 'str', 'typespec')
    # Processing the call keyword arguments (line 224)
    kwargs_67268 = {}
    # Getting the type of 'var' (line 224)
    var_67265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 28), 'var', False)
    # Obtaining the member 'get' of a type (line 224)
    get_67266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 28), var_67265, 'get')
    # Calling get(args, kwargs) (line 224)
    get_call_result_67269 = invoke(stypy.reporting.localization.Localization(__file__, 224, 28), get_67266, *[str_67267], **kwargs_67268)
    
    
    # Obtaining an instance of the builtin type 'list' (line 224)
    list_67270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 224)
    # Adding element type (line 224)
    str_67271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 52), 'str', 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 51), list_67270, str_67271)
    # Adding element type (line 224)
    str_67272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 63), 'str', 'logical')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 224, 51), list_67270, str_67272)
    
    # Applying the binary operator 'in' (line 224)
    result_contains_67273 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 28), 'in', get_call_result_67269, list_67270)
    
    # Applying the binary operator 'and' (line 224)
    result_and_keyword_67274 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 11), 'and', isarray_call_result_67264, result_contains_67273)
    
    
    # Call to get_kind(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'var' (line 225)
    var_67276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 21), 'var', False)
    # Processing the call keyword arguments (line 225)
    kwargs_67277 = {}
    # Getting the type of 'get_kind' (line 225)
    get_kind_67275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 225)
    get_kind_call_result_67278 = invoke(stypy.reporting.localization.Localization(__file__, 225, 12), get_kind_67275, *[var_67276], **kwargs_67277)
    
    str_67279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 29), 'str', '1')
    # Applying the binary operator '==' (line 225)
    result_eq_67280 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 12), '==', get_kind_call_result_67278, str_67279)
    
    # Applying the binary operator 'and' (line 224)
    result_and_keyword_67281 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 11), 'and', result_and_keyword_67274, result_eq_67280)
    
    # Assigning a type to the variable 'stypy_return_type' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'stypy_return_type', result_and_keyword_67281)
    
    # ################# End of 'issigned_chararray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'issigned_chararray' in the type store
    # Getting the type of 'stypy_return_type' (line 223)
    stypy_return_type_67282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67282)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'issigned_chararray'
    return stypy_return_type_67282

# Assigning a type to the variable 'issigned_chararray' (line 223)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 0), 'issigned_chararray', issigned_chararray)

@norecursion
def issigned_shortarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'issigned_shortarray'
    module_type_store = module_type_store.open_function_context('issigned_shortarray', 228, 0, False)
    
    # Passed parameters checking function
    issigned_shortarray.stypy_localization = localization
    issigned_shortarray.stypy_type_of_self = None
    issigned_shortarray.stypy_type_store = module_type_store
    issigned_shortarray.stypy_function_name = 'issigned_shortarray'
    issigned_shortarray.stypy_param_names_list = ['var']
    issigned_shortarray.stypy_varargs_param_name = None
    issigned_shortarray.stypy_kwargs_param_name = None
    issigned_shortarray.stypy_call_defaults = defaults
    issigned_shortarray.stypy_call_varargs = varargs
    issigned_shortarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'issigned_shortarray', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'issigned_shortarray', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'issigned_shortarray(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isarray(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'var' (line 229)
    var_67284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 19), 'var', False)
    # Processing the call keyword arguments (line 229)
    kwargs_67285 = {}
    # Getting the type of 'isarray' (line 229)
    isarray_67283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 11), 'isarray', False)
    # Calling isarray(args, kwargs) (line 229)
    isarray_call_result_67286 = invoke(stypy.reporting.localization.Localization(__file__, 229, 11), isarray_67283, *[var_67284], **kwargs_67285)
    
    
    
    # Call to get(...): (line 229)
    # Processing the call arguments (line 229)
    str_67289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 36), 'str', 'typespec')
    # Processing the call keyword arguments (line 229)
    kwargs_67290 = {}
    # Getting the type of 'var' (line 229)
    var_67287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 28), 'var', False)
    # Obtaining the member 'get' of a type (line 229)
    get_67288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 28), var_67287, 'get')
    # Calling get(args, kwargs) (line 229)
    get_call_result_67291 = invoke(stypy.reporting.localization.Localization(__file__, 229, 28), get_67288, *[str_67289], **kwargs_67290)
    
    
    # Obtaining an instance of the builtin type 'list' (line 229)
    list_67292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 229)
    # Adding element type (line 229)
    str_67293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 52), 'str', 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 51), list_67292, str_67293)
    # Adding element type (line 229)
    str_67294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 63), 'str', 'logical')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 51), list_67292, str_67294)
    
    # Applying the binary operator 'in' (line 229)
    result_contains_67295 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 28), 'in', get_call_result_67291, list_67292)
    
    # Applying the binary operator 'and' (line 229)
    result_and_keyword_67296 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 11), 'and', isarray_call_result_67286, result_contains_67295)
    
    
    # Call to get_kind(...): (line 230)
    # Processing the call arguments (line 230)
    # Getting the type of 'var' (line 230)
    var_67298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 21), 'var', False)
    # Processing the call keyword arguments (line 230)
    kwargs_67299 = {}
    # Getting the type of 'get_kind' (line 230)
    get_kind_67297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 230)
    get_kind_call_result_67300 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), get_kind_67297, *[var_67298], **kwargs_67299)
    
    str_67301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 29), 'str', '2')
    # Applying the binary operator '==' (line 230)
    result_eq_67302 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 12), '==', get_kind_call_result_67300, str_67301)
    
    # Applying the binary operator 'and' (line 229)
    result_and_keyword_67303 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 11), 'and', result_and_keyword_67296, result_eq_67302)
    
    # Assigning a type to the variable 'stypy_return_type' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type', result_and_keyword_67303)
    
    # ################# End of 'issigned_shortarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'issigned_shortarray' in the type store
    # Getting the type of 'stypy_return_type' (line 228)
    stypy_return_type_67304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67304)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'issigned_shortarray'
    return stypy_return_type_67304

# Assigning a type to the variable 'issigned_shortarray' (line 228)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 0), 'issigned_shortarray', issigned_shortarray)

@norecursion
def issigned_array(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'issigned_array'
    module_type_store = module_type_store.open_function_context('issigned_array', 233, 0, False)
    
    # Passed parameters checking function
    issigned_array.stypy_localization = localization
    issigned_array.stypy_type_of_self = None
    issigned_array.stypy_type_store = module_type_store
    issigned_array.stypy_function_name = 'issigned_array'
    issigned_array.stypy_param_names_list = ['var']
    issigned_array.stypy_varargs_param_name = None
    issigned_array.stypy_kwargs_param_name = None
    issigned_array.stypy_call_defaults = defaults
    issigned_array.stypy_call_varargs = varargs
    issigned_array.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'issigned_array', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'issigned_array', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'issigned_array(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isarray(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'var' (line 234)
    var_67306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 19), 'var', False)
    # Processing the call keyword arguments (line 234)
    kwargs_67307 = {}
    # Getting the type of 'isarray' (line 234)
    isarray_67305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'isarray', False)
    # Calling isarray(args, kwargs) (line 234)
    isarray_call_result_67308 = invoke(stypy.reporting.localization.Localization(__file__, 234, 11), isarray_67305, *[var_67306], **kwargs_67307)
    
    
    
    # Call to get(...): (line 234)
    # Processing the call arguments (line 234)
    str_67311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 36), 'str', 'typespec')
    # Processing the call keyword arguments (line 234)
    kwargs_67312 = {}
    # Getting the type of 'var' (line 234)
    var_67309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 28), 'var', False)
    # Obtaining the member 'get' of a type (line 234)
    get_67310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 28), var_67309, 'get')
    # Calling get(args, kwargs) (line 234)
    get_call_result_67313 = invoke(stypy.reporting.localization.Localization(__file__, 234, 28), get_67310, *[str_67311], **kwargs_67312)
    
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_67314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    # Adding element type (line 234)
    str_67315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 52), 'str', 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 51), list_67314, str_67315)
    # Adding element type (line 234)
    str_67316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 63), 'str', 'logical')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 51), list_67314, str_67316)
    
    # Applying the binary operator 'in' (line 234)
    result_contains_67317 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 28), 'in', get_call_result_67313, list_67314)
    
    # Applying the binary operator 'and' (line 234)
    result_and_keyword_67318 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 11), 'and', isarray_call_result_67308, result_contains_67317)
    
    
    # Call to get_kind(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'var' (line 235)
    var_67320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 21), 'var', False)
    # Processing the call keyword arguments (line 235)
    kwargs_67321 = {}
    # Getting the type of 'get_kind' (line 235)
    get_kind_67319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 235)
    get_kind_call_result_67322 = invoke(stypy.reporting.localization.Localization(__file__, 235, 12), get_kind_67319, *[var_67320], **kwargs_67321)
    
    str_67323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 29), 'str', '4')
    # Applying the binary operator '==' (line 235)
    result_eq_67324 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 12), '==', get_kind_call_result_67322, str_67323)
    
    # Applying the binary operator 'and' (line 234)
    result_and_keyword_67325 = python_operator(stypy.reporting.localization.Localization(__file__, 234, 11), 'and', result_and_keyword_67318, result_eq_67324)
    
    # Assigning a type to the variable 'stypy_return_type' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type', result_and_keyword_67325)
    
    # ################# End of 'issigned_array(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'issigned_array' in the type store
    # Getting the type of 'stypy_return_type' (line 233)
    stypy_return_type_67326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67326)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'issigned_array'
    return stypy_return_type_67326

# Assigning a type to the variable 'issigned_array' (line 233)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 0), 'issigned_array', issigned_array)

@norecursion
def issigned_long_longarray(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'issigned_long_longarray'
    module_type_store = module_type_store.open_function_context('issigned_long_longarray', 238, 0, False)
    
    # Passed parameters checking function
    issigned_long_longarray.stypy_localization = localization
    issigned_long_longarray.stypy_type_of_self = None
    issigned_long_longarray.stypy_type_store = module_type_store
    issigned_long_longarray.stypy_function_name = 'issigned_long_longarray'
    issigned_long_longarray.stypy_param_names_list = ['var']
    issigned_long_longarray.stypy_varargs_param_name = None
    issigned_long_longarray.stypy_kwargs_param_name = None
    issigned_long_longarray.stypy_call_defaults = defaults
    issigned_long_longarray.stypy_call_varargs = varargs
    issigned_long_longarray.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'issigned_long_longarray', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'issigned_long_longarray', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'issigned_long_longarray(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isarray(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'var' (line 239)
    var_67328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 19), 'var', False)
    # Processing the call keyword arguments (line 239)
    kwargs_67329 = {}
    # Getting the type of 'isarray' (line 239)
    isarray_67327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 11), 'isarray', False)
    # Calling isarray(args, kwargs) (line 239)
    isarray_call_result_67330 = invoke(stypy.reporting.localization.Localization(__file__, 239, 11), isarray_67327, *[var_67328], **kwargs_67329)
    
    
    
    # Call to get(...): (line 239)
    # Processing the call arguments (line 239)
    str_67333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 36), 'str', 'typespec')
    # Processing the call keyword arguments (line 239)
    kwargs_67334 = {}
    # Getting the type of 'var' (line 239)
    var_67331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 28), 'var', False)
    # Obtaining the member 'get' of a type (line 239)
    get_67332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 28), var_67331, 'get')
    # Calling get(args, kwargs) (line 239)
    get_call_result_67335 = invoke(stypy.reporting.localization.Localization(__file__, 239, 28), get_67332, *[str_67333], **kwargs_67334)
    
    
    # Obtaining an instance of the builtin type 'list' (line 239)
    list_67336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 239)
    # Adding element type (line 239)
    str_67337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 52), 'str', 'integer')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 51), list_67336, str_67337)
    # Adding element type (line 239)
    str_67338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 63), 'str', 'logical')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 239, 51), list_67336, str_67338)
    
    # Applying the binary operator 'in' (line 239)
    result_contains_67339 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 28), 'in', get_call_result_67335, list_67336)
    
    # Applying the binary operator 'and' (line 239)
    result_and_keyword_67340 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 11), 'and', isarray_call_result_67330, result_contains_67339)
    
    
    # Call to get_kind(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'var' (line 240)
    var_67342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 21), 'var', False)
    # Processing the call keyword arguments (line 240)
    kwargs_67343 = {}
    # Getting the type of 'get_kind' (line 240)
    get_kind_67341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'get_kind', False)
    # Calling get_kind(args, kwargs) (line 240)
    get_kind_call_result_67344 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), get_kind_67341, *[var_67342], **kwargs_67343)
    
    str_67345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 29), 'str', '8')
    # Applying the binary operator '==' (line 240)
    result_eq_67346 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 12), '==', get_kind_call_result_67344, str_67345)
    
    # Applying the binary operator 'and' (line 239)
    result_and_keyword_67347 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 11), 'and', result_and_keyword_67340, result_eq_67346)
    
    # Assigning a type to the variable 'stypy_return_type' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type', result_and_keyword_67347)
    
    # ################# End of 'issigned_long_longarray(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'issigned_long_longarray' in the type store
    # Getting the type of 'stypy_return_type' (line 238)
    stypy_return_type_67348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67348)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'issigned_long_longarray'
    return stypy_return_type_67348

# Assigning a type to the variable 'issigned_long_longarray' (line 238)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 0), 'issigned_long_longarray', issigned_long_longarray)

@norecursion
def isallocatable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isallocatable'
    module_type_store = module_type_store.open_function_context('isallocatable', 243, 0, False)
    
    # Passed parameters checking function
    isallocatable.stypy_localization = localization
    isallocatable.stypy_type_of_self = None
    isallocatable.stypy_type_store = module_type_store
    isallocatable.stypy_function_name = 'isallocatable'
    isallocatable.stypy_param_names_list = ['var']
    isallocatable.stypy_varargs_param_name = None
    isallocatable.stypy_kwargs_param_name = None
    isallocatable.stypy_call_defaults = defaults
    isallocatable.stypy_call_varargs = varargs
    isallocatable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isallocatable', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isallocatable', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isallocatable(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_67349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 11), 'str', 'attrspec')
    # Getting the type of 'var' (line 244)
    var_67350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 25), 'var')
    # Applying the binary operator 'in' (line 244)
    result_contains_67351 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 11), 'in', str_67349, var_67350)
    
    
    str_67352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 33), 'str', 'allocatable')
    
    # Obtaining the type of the subscript
    str_67353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 54), 'str', 'attrspec')
    # Getting the type of 'var' (line 244)
    var_67354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 50), 'var')
    # Obtaining the member '__getitem__' of a type (line 244)
    getitem___67355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 50), var_67354, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 244)
    subscript_call_result_67356 = invoke(stypy.reporting.localization.Localization(__file__, 244, 50), getitem___67355, str_67353)
    
    # Applying the binary operator 'in' (line 244)
    result_contains_67357 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 33), 'in', str_67352, subscript_call_result_67356)
    
    # Applying the binary operator 'and' (line 244)
    result_and_keyword_67358 = python_operator(stypy.reporting.localization.Localization(__file__, 244, 11), 'and', result_contains_67351, result_contains_67357)
    
    # Assigning a type to the variable 'stypy_return_type' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'stypy_return_type', result_and_keyword_67358)
    
    # ################# End of 'isallocatable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isallocatable' in the type store
    # Getting the type of 'stypy_return_type' (line 243)
    stypy_return_type_67359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67359)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isallocatable'
    return stypy_return_type_67359

# Assigning a type to the variable 'isallocatable' (line 243)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'isallocatable', isallocatable)

@norecursion
def ismutable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ismutable'
    module_type_store = module_type_store.open_function_context('ismutable', 247, 0, False)
    
    # Passed parameters checking function
    ismutable.stypy_localization = localization
    ismutable.stypy_type_of_self = None
    ismutable.stypy_type_store = module_type_store
    ismutable.stypy_function_name = 'ismutable'
    ismutable.stypy_param_names_list = ['var']
    ismutable.stypy_varargs_param_name = None
    ismutable.stypy_kwargs_param_name = None
    ismutable.stypy_call_defaults = defaults
    ismutable.stypy_call_varargs = varargs
    ismutable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ismutable', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ismutable', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ismutable(...)' code ##################

    
    
    # Evaluating a boolean operation
    
    str_67360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 16), 'str', 'dimension')
    # Getting the type of 'var' (line 248)
    var_67361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 35), 'var')
    # Applying the binary operator 'notin' (line 248)
    result_contains_67362 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 16), 'notin', str_67360, var_67361)
    
    
    # Call to isstring(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'var' (line 248)
    var_67364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 51), 'var', False)
    # Processing the call keyword arguments (line 248)
    kwargs_67365 = {}
    # Getting the type of 'isstring' (line 248)
    isstring_67363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 42), 'isstring', False)
    # Calling isstring(args, kwargs) (line 248)
    isstring_call_result_67366 = invoke(stypy.reporting.localization.Localization(__file__, 248, 42), isstring_67363, *[var_67364], **kwargs_67365)
    
    # Applying the binary operator 'or' (line 248)
    result_or_keyword_67367 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 16), 'or', result_contains_67362, isstring_call_result_67366)
    
    # Applying the 'not' unary operator (line 248)
    result_not__67368 = python_operator(stypy.reporting.localization.Localization(__file__, 248, 11), 'not', result_or_keyword_67367)
    
    # Assigning a type to the variable 'stypy_return_type' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type', result_not__67368)
    
    # ################# End of 'ismutable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ismutable' in the type store
    # Getting the type of 'stypy_return_type' (line 247)
    stypy_return_type_67369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67369)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ismutable'
    return stypy_return_type_67369

# Assigning a type to the variable 'ismutable' (line 247)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 0), 'ismutable', ismutable)

@norecursion
def ismoduleroutine(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ismoduleroutine'
    module_type_store = module_type_store.open_function_context('ismoduleroutine', 251, 0, False)
    
    # Passed parameters checking function
    ismoduleroutine.stypy_localization = localization
    ismoduleroutine.stypy_type_of_self = None
    ismoduleroutine.stypy_type_store = module_type_store
    ismoduleroutine.stypy_function_name = 'ismoduleroutine'
    ismoduleroutine.stypy_param_names_list = ['rout']
    ismoduleroutine.stypy_varargs_param_name = None
    ismoduleroutine.stypy_kwargs_param_name = None
    ismoduleroutine.stypy_call_defaults = defaults
    ismoduleroutine.stypy_call_varargs = varargs
    ismoduleroutine.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ismoduleroutine', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ismoduleroutine', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ismoduleroutine(...)' code ##################

    
    str_67370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 11), 'str', 'modulename')
    # Getting the type of 'rout' (line 252)
    rout_67371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 27), 'rout')
    # Applying the binary operator 'in' (line 252)
    result_contains_67372 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 11), 'in', str_67370, rout_67371)
    
    # Assigning a type to the variable 'stypy_return_type' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'stypy_return_type', result_contains_67372)
    
    # ################# End of 'ismoduleroutine(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ismoduleroutine' in the type store
    # Getting the type of 'stypy_return_type' (line 251)
    stypy_return_type_67373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67373)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ismoduleroutine'
    return stypy_return_type_67373

# Assigning a type to the variable 'ismoduleroutine' (line 251)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 0), 'ismoduleroutine', ismoduleroutine)

@norecursion
def ismodule(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ismodule'
    module_type_store = module_type_store.open_function_context('ismodule', 255, 0, False)
    
    # Passed parameters checking function
    ismodule.stypy_localization = localization
    ismodule.stypy_type_of_self = None
    ismodule.stypy_type_store = module_type_store
    ismodule.stypy_function_name = 'ismodule'
    ismodule.stypy_param_names_list = ['rout']
    ismodule.stypy_varargs_param_name = None
    ismodule.stypy_kwargs_param_name = None
    ismodule.stypy_call_defaults = defaults
    ismodule.stypy_call_varargs = varargs
    ismodule.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ismodule', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ismodule', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ismodule(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_67374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 11), 'str', 'block')
    # Getting the type of 'rout' (line 256)
    rout_67375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 22), 'rout')
    # Applying the binary operator 'in' (line 256)
    result_contains_67376 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 11), 'in', str_67374, rout_67375)
    
    
    str_67377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 31), 'str', 'module')
    
    # Obtaining the type of the subscript
    str_67378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 48), 'str', 'block')
    # Getting the type of 'rout' (line 256)
    rout_67379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 43), 'rout')
    # Obtaining the member '__getitem__' of a type (line 256)
    getitem___67380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 43), rout_67379, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 256)
    subscript_call_result_67381 = invoke(stypy.reporting.localization.Localization(__file__, 256, 43), getitem___67380, str_67378)
    
    # Applying the binary operator '==' (line 256)
    result_eq_67382 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 31), '==', str_67377, subscript_call_result_67381)
    
    # Applying the binary operator 'and' (line 256)
    result_and_keyword_67383 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 11), 'and', result_contains_67376, result_eq_67382)
    
    # Assigning a type to the variable 'stypy_return_type' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type', result_and_keyword_67383)
    
    # ################# End of 'ismodule(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ismodule' in the type store
    # Getting the type of 'stypy_return_type' (line 255)
    stypy_return_type_67384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67384)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ismodule'
    return stypy_return_type_67384

# Assigning a type to the variable 'ismodule' (line 255)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 0), 'ismodule', ismodule)

@norecursion
def isfunction(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isfunction'
    module_type_store = module_type_store.open_function_context('isfunction', 259, 0, False)
    
    # Passed parameters checking function
    isfunction.stypy_localization = localization
    isfunction.stypy_type_of_self = None
    isfunction.stypy_type_store = module_type_store
    isfunction.stypy_function_name = 'isfunction'
    isfunction.stypy_param_names_list = ['rout']
    isfunction.stypy_varargs_param_name = None
    isfunction.stypy_kwargs_param_name = None
    isfunction.stypy_call_defaults = defaults
    isfunction.stypy_call_varargs = varargs
    isfunction.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isfunction', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isfunction', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isfunction(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_67385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 11), 'str', 'block')
    # Getting the type of 'rout' (line 260)
    rout_67386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 22), 'rout')
    # Applying the binary operator 'in' (line 260)
    result_contains_67387 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 11), 'in', str_67385, rout_67386)
    
    
    str_67388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 31), 'str', 'function')
    
    # Obtaining the type of the subscript
    str_67389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 50), 'str', 'block')
    # Getting the type of 'rout' (line 260)
    rout_67390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 45), 'rout')
    # Obtaining the member '__getitem__' of a type (line 260)
    getitem___67391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 45), rout_67390, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 260)
    subscript_call_result_67392 = invoke(stypy.reporting.localization.Localization(__file__, 260, 45), getitem___67391, str_67389)
    
    # Applying the binary operator '==' (line 260)
    result_eq_67393 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 31), '==', str_67388, subscript_call_result_67392)
    
    # Applying the binary operator 'and' (line 260)
    result_and_keyword_67394 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 11), 'and', result_contains_67387, result_eq_67393)
    
    # Assigning a type to the variable 'stypy_return_type' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type', result_and_keyword_67394)
    
    # ################# End of 'isfunction(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isfunction' in the type store
    # Getting the type of 'stypy_return_type' (line 259)
    stypy_return_type_67395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67395)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isfunction'
    return stypy_return_type_67395

# Assigning a type to the variable 'isfunction' (line 259)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'isfunction', isfunction)

@norecursion
def isfunction_wrap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isfunction_wrap'
    module_type_store = module_type_store.open_function_context('isfunction_wrap', 262, 0, False)
    
    # Passed parameters checking function
    isfunction_wrap.stypy_localization = localization
    isfunction_wrap.stypy_type_of_self = None
    isfunction_wrap.stypy_type_store = module_type_store
    isfunction_wrap.stypy_function_name = 'isfunction_wrap'
    isfunction_wrap.stypy_param_names_list = ['rout']
    isfunction_wrap.stypy_varargs_param_name = None
    isfunction_wrap.stypy_kwargs_param_name = None
    isfunction_wrap.stypy_call_defaults = defaults
    isfunction_wrap.stypy_call_varargs = varargs
    isfunction_wrap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isfunction_wrap', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isfunction_wrap', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isfunction_wrap(...)' code ##################

    
    
    # Call to isintent_c(...): (line 263)
    # Processing the call arguments (line 263)
    # Getting the type of 'rout' (line 263)
    rout_67397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 18), 'rout', False)
    # Processing the call keyword arguments (line 263)
    kwargs_67398 = {}
    # Getting the type of 'isintent_c' (line 263)
    isintent_c_67396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 7), 'isintent_c', False)
    # Calling isintent_c(args, kwargs) (line 263)
    isintent_c_call_result_67399 = invoke(stypy.reporting.localization.Localization(__file__, 263, 7), isintent_c_67396, *[rout_67397], **kwargs_67398)
    
    # Testing the type of an if condition (line 263)
    if_condition_67400 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 263, 4), isintent_c_call_result_67399)
    # Assigning a type to the variable 'if_condition_67400' (line 263)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'if_condition_67400', if_condition_67400)
    # SSA begins for if statement (line 263)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'stypy_return_type', int_67401)
    # SSA join for if statement (line 263)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Evaluating a boolean operation
    # Getting the type of 'wrapfuncs' (line 265)
    wrapfuncs_67402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'wrapfuncs')
    
    # Call to isfunction(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'rout' (line 265)
    rout_67404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 36), 'rout', False)
    # Processing the call keyword arguments (line 265)
    kwargs_67405 = {}
    # Getting the type of 'isfunction' (line 265)
    isfunction_67403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 25), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 265)
    isfunction_call_result_67406 = invoke(stypy.reporting.localization.Localization(__file__, 265, 25), isfunction_67403, *[rout_67404], **kwargs_67405)
    
    # Applying the binary operator 'and' (line 265)
    result_and_keyword_67407 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 11), 'and', wrapfuncs_67402, isfunction_call_result_67406)
    
    
    # Call to isexternal(...): (line 265)
    # Processing the call arguments (line 265)
    # Getting the type of 'rout' (line 265)
    rout_67409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 62), 'rout', False)
    # Processing the call keyword arguments (line 265)
    kwargs_67410 = {}
    # Getting the type of 'isexternal' (line 265)
    isexternal_67408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 51), 'isexternal', False)
    # Calling isexternal(args, kwargs) (line 265)
    isexternal_call_result_67411 = invoke(stypy.reporting.localization.Localization(__file__, 265, 51), isexternal_67408, *[rout_67409], **kwargs_67410)
    
    # Applying the 'not' unary operator (line 265)
    result_not__67412 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 47), 'not', isexternal_call_result_67411)
    
    # Applying the binary operator 'and' (line 265)
    result_and_keyword_67413 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 11), 'and', result_and_keyword_67407, result_not__67412)
    
    # Assigning a type to the variable 'stypy_return_type' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'stypy_return_type', result_and_keyword_67413)
    
    # ################# End of 'isfunction_wrap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isfunction_wrap' in the type store
    # Getting the type of 'stypy_return_type' (line 262)
    stypy_return_type_67414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67414)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isfunction_wrap'
    return stypy_return_type_67414

# Assigning a type to the variable 'isfunction_wrap' (line 262)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 0), 'isfunction_wrap', isfunction_wrap)

@norecursion
def issubroutine(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'issubroutine'
    module_type_store = module_type_store.open_function_context('issubroutine', 268, 0, False)
    
    # Passed parameters checking function
    issubroutine.stypy_localization = localization
    issubroutine.stypy_type_of_self = None
    issubroutine.stypy_type_store = module_type_store
    issubroutine.stypy_function_name = 'issubroutine'
    issubroutine.stypy_param_names_list = ['rout']
    issubroutine.stypy_varargs_param_name = None
    issubroutine.stypy_kwargs_param_name = None
    issubroutine.stypy_call_defaults = defaults
    issubroutine.stypy_call_varargs = varargs
    issubroutine.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'issubroutine', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'issubroutine', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'issubroutine(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_67415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 11), 'str', 'block')
    # Getting the type of 'rout' (line 269)
    rout_67416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'rout')
    # Applying the binary operator 'in' (line 269)
    result_contains_67417 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 11), 'in', str_67415, rout_67416)
    
    
    str_67418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 31), 'str', 'subroutine')
    
    # Obtaining the type of the subscript
    str_67419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 52), 'str', 'block')
    # Getting the type of 'rout' (line 269)
    rout_67420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 47), 'rout')
    # Obtaining the member '__getitem__' of a type (line 269)
    getitem___67421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 47), rout_67420, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 269)
    subscript_call_result_67422 = invoke(stypy.reporting.localization.Localization(__file__, 269, 47), getitem___67421, str_67419)
    
    # Applying the binary operator '==' (line 269)
    result_eq_67423 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 31), '==', str_67418, subscript_call_result_67422)
    
    # Applying the binary operator 'and' (line 269)
    result_and_keyword_67424 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 11), 'and', result_contains_67417, result_eq_67423)
    
    # Assigning a type to the variable 'stypy_return_type' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'stypy_return_type', result_and_keyword_67424)
    
    # ################# End of 'issubroutine(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'issubroutine' in the type store
    # Getting the type of 'stypy_return_type' (line 268)
    stypy_return_type_67425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67425)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'issubroutine'
    return stypy_return_type_67425

# Assigning a type to the variable 'issubroutine' (line 268)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 0), 'issubroutine', issubroutine)

@norecursion
def issubroutine_wrap(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'issubroutine_wrap'
    module_type_store = module_type_store.open_function_context('issubroutine_wrap', 272, 0, False)
    
    # Passed parameters checking function
    issubroutine_wrap.stypy_localization = localization
    issubroutine_wrap.stypy_type_of_self = None
    issubroutine_wrap.stypy_type_store = module_type_store
    issubroutine_wrap.stypy_function_name = 'issubroutine_wrap'
    issubroutine_wrap.stypy_param_names_list = ['rout']
    issubroutine_wrap.stypy_varargs_param_name = None
    issubroutine_wrap.stypy_kwargs_param_name = None
    issubroutine_wrap.stypy_call_defaults = defaults
    issubroutine_wrap.stypy_call_varargs = varargs
    issubroutine_wrap.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'issubroutine_wrap', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'issubroutine_wrap', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'issubroutine_wrap(...)' code ##################

    
    
    # Call to isintent_c(...): (line 273)
    # Processing the call arguments (line 273)
    # Getting the type of 'rout' (line 273)
    rout_67427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 18), 'rout', False)
    # Processing the call keyword arguments (line 273)
    kwargs_67428 = {}
    # Getting the type of 'isintent_c' (line 273)
    isintent_c_67426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 7), 'isintent_c', False)
    # Calling isintent_c(args, kwargs) (line 273)
    isintent_c_call_result_67429 = invoke(stypy.reporting.localization.Localization(__file__, 273, 7), isintent_c_67426, *[rout_67427], **kwargs_67428)
    
    # Testing the type of an if condition (line 273)
    if_condition_67430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 273, 4), isintent_c_call_result_67429)
    # Assigning a type to the variable 'if_condition_67430' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'if_condition_67430', if_condition_67430)
    # SSA begins for if statement (line 273)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'stypy_return_type', int_67431)
    # SSA join for if statement (line 273)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Evaluating a boolean operation
    
    # Call to issubroutine(...): (line 275)
    # Processing the call arguments (line 275)
    # Getting the type of 'rout' (line 275)
    rout_67433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 24), 'rout', False)
    # Processing the call keyword arguments (line 275)
    kwargs_67434 = {}
    # Getting the type of 'issubroutine' (line 275)
    issubroutine_67432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 11), 'issubroutine', False)
    # Calling issubroutine(args, kwargs) (line 275)
    issubroutine_call_result_67435 = invoke(stypy.reporting.localization.Localization(__file__, 275, 11), issubroutine_67432, *[rout_67433], **kwargs_67434)
    
    
    # Call to hasassumedshape(...): (line 275)
    # Processing the call arguments (line 275)
    # Getting the type of 'rout' (line 275)
    rout_67437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 50), 'rout', False)
    # Processing the call keyword arguments (line 275)
    kwargs_67438 = {}
    # Getting the type of 'hasassumedshape' (line 275)
    hasassumedshape_67436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 34), 'hasassumedshape', False)
    # Calling hasassumedshape(args, kwargs) (line 275)
    hasassumedshape_call_result_67439 = invoke(stypy.reporting.localization.Localization(__file__, 275, 34), hasassumedshape_67436, *[rout_67437], **kwargs_67438)
    
    # Applying the binary operator 'and' (line 275)
    result_and_keyword_67440 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 11), 'and', issubroutine_call_result_67435, hasassumedshape_call_result_67439)
    
    # Assigning a type to the variable 'stypy_return_type' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'stypy_return_type', result_and_keyword_67440)
    
    # ################# End of 'issubroutine_wrap(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'issubroutine_wrap' in the type store
    # Getting the type of 'stypy_return_type' (line 272)
    stypy_return_type_67441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67441)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'issubroutine_wrap'
    return stypy_return_type_67441

# Assigning a type to the variable 'issubroutine_wrap' (line 272)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 0), 'issubroutine_wrap', issubroutine_wrap)

@norecursion
def hasassumedshape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hasassumedshape'
    module_type_store = module_type_store.open_function_context('hasassumedshape', 278, 0, False)
    
    # Passed parameters checking function
    hasassumedshape.stypy_localization = localization
    hasassumedshape.stypy_type_of_self = None
    hasassumedshape.stypy_type_store = module_type_store
    hasassumedshape.stypy_function_name = 'hasassumedshape'
    hasassumedshape.stypy_param_names_list = ['rout']
    hasassumedshape.stypy_varargs_param_name = None
    hasassumedshape.stypy_kwargs_param_name = None
    hasassumedshape.stypy_call_defaults = defaults
    hasassumedshape.stypy_call_varargs = varargs
    hasassumedshape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hasassumedshape', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hasassumedshape', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hasassumedshape(...)' code ##################

    
    
    # Call to get(...): (line 279)
    # Processing the call arguments (line 279)
    str_67444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 16), 'str', 'hasassumedshape')
    # Processing the call keyword arguments (line 279)
    kwargs_67445 = {}
    # Getting the type of 'rout' (line 279)
    rout_67442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 7), 'rout', False)
    # Obtaining the member 'get' of a type (line 279)
    get_67443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 7), rout_67442, 'get')
    # Calling get(args, kwargs) (line 279)
    get_call_result_67446 = invoke(stypy.reporting.localization.Localization(__file__, 279, 7), get_67443, *[str_67444], **kwargs_67445)
    
    # Testing the type of an if condition (line 279)
    if_condition_67447 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 4), get_call_result_67446)
    # Assigning a type to the variable 'if_condition_67447' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'if_condition_67447', if_condition_67447)
    # SSA begins for if statement (line 279)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 280)
    True_67448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'stypy_return_type', True_67448)
    # SSA join for if statement (line 279)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    str_67449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 18), 'str', 'args')
    # Getting the type of 'rout' (line 281)
    rout_67450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 13), 'rout')
    # Obtaining the member '__getitem__' of a type (line 281)
    getitem___67451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 13), rout_67450, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 281)
    subscript_call_result_67452 = invoke(stypy.reporting.localization.Localization(__file__, 281, 13), getitem___67451, str_67449)
    
    # Testing the type of a for loop iterable (line 281)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 281, 4), subscript_call_result_67452)
    # Getting the type of the for loop variable (line 281)
    for_loop_var_67453 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 281, 4), subscript_call_result_67452)
    # Assigning a type to the variable 'a' (line 281)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'a', for_loop_var_67453)
    # SSA begins for a for statement (line 281)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to get(...): (line 282)
    # Processing the call arguments (line 282)
    str_67464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 45), 'str', 'dimension')
    
    # Obtaining an instance of the builtin type 'list' (line 282)
    list_67465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 58), 'list')
    # Adding type elements to the builtin type 'list' instance (line 282)
    
    # Processing the call keyword arguments (line 282)
    kwargs_67466 = {}
    
    # Call to get(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'a' (line 282)
    a_67459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 34), 'a', False)
    
    # Obtaining an instance of the builtin type 'dict' (line 282)
    dict_67460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 37), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 282)
    
    # Processing the call keyword arguments (line 282)
    kwargs_67461 = {}
    
    # Obtaining the type of the subscript
    str_67454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 22), 'str', 'vars')
    # Getting the type of 'rout' (line 282)
    rout_67455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 282)
    getitem___67456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 17), rout_67455, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 282)
    subscript_call_result_67457 = invoke(stypy.reporting.localization.Localization(__file__, 282, 17), getitem___67456, str_67454)
    
    # Obtaining the member 'get' of a type (line 282)
    get_67458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 17), subscript_call_result_67457, 'get')
    # Calling get(args, kwargs) (line 282)
    get_call_result_67462 = invoke(stypy.reporting.localization.Localization(__file__, 282, 17), get_67458, *[a_67459, dict_67460], **kwargs_67461)
    
    # Obtaining the member 'get' of a type (line 282)
    get_67463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 17), get_call_result_67462, 'get')
    # Calling get(args, kwargs) (line 282)
    get_call_result_67467 = invoke(stypy.reporting.localization.Localization(__file__, 282, 17), get_67463, *[str_67464, list_67465], **kwargs_67466)
    
    # Testing the type of a for loop iterable (line 282)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 282, 8), get_call_result_67467)
    # Getting the type of the for loop variable (line 282)
    for_loop_var_67468 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 282, 8), get_call_result_67467)
    # Assigning a type to the variable 'd' (line 282)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'd', for_loop_var_67468)
    # SSA begins for a for statement (line 282)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'd' (line 283)
    d_67469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 15), 'd')
    str_67470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 20), 'str', ':')
    # Applying the binary operator '==' (line 283)
    result_eq_67471 = python_operator(stypy.reporting.localization.Localization(__file__, 283, 15), '==', d_67469, str_67470)
    
    # Testing the type of an if condition (line 283)
    if_condition_67472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 283, 12), result_eq_67471)
    # Assigning a type to the variable 'if_condition_67472' (line 283)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'if_condition_67472', if_condition_67472)
    # SSA begins for if statement (line 283)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 284):
    
    # Assigning a Name to a Subscript (line 284):
    # Getting the type of 'True' (line 284)
    True_67473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 42), 'True')
    # Getting the type of 'rout' (line 284)
    rout_67474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'rout')
    str_67475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 21), 'str', 'hasassumedshape')
    # Storing an element on a container (line 284)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 16), rout_67474, (str_67475, True_67473))
    # Getting the type of 'True' (line 285)
    True_67476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 23), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 16), 'stypy_return_type', True_67476)
    # SSA join for if statement (line 283)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'False' (line 286)
    False_67477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'stypy_return_type', False_67477)
    
    # ################# End of 'hasassumedshape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hasassumedshape' in the type store
    # Getting the type of 'stypy_return_type' (line 278)
    stypy_return_type_67478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67478)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hasassumedshape'
    return stypy_return_type_67478

# Assigning a type to the variable 'hasassumedshape' (line 278)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 0), 'hasassumedshape', hasassumedshape)

@norecursion
def isroutine(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isroutine'
    module_type_store = module_type_store.open_function_context('isroutine', 289, 0, False)
    
    # Passed parameters checking function
    isroutine.stypy_localization = localization
    isroutine.stypy_type_of_self = None
    isroutine.stypy_type_store = module_type_store
    isroutine.stypy_function_name = 'isroutine'
    isroutine.stypy_param_names_list = ['rout']
    isroutine.stypy_varargs_param_name = None
    isroutine.stypy_kwargs_param_name = None
    isroutine.stypy_call_defaults = defaults
    isroutine.stypy_call_varargs = varargs
    isroutine.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isroutine', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isroutine', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isroutine(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Call to isfunction(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'rout' (line 290)
    rout_67480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 22), 'rout', False)
    # Processing the call keyword arguments (line 290)
    kwargs_67481 = {}
    # Getting the type of 'isfunction' (line 290)
    isfunction_67479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 11), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 290)
    isfunction_call_result_67482 = invoke(stypy.reporting.localization.Localization(__file__, 290, 11), isfunction_67479, *[rout_67480], **kwargs_67481)
    
    
    # Call to issubroutine(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'rout' (line 290)
    rout_67484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 44), 'rout', False)
    # Processing the call keyword arguments (line 290)
    kwargs_67485 = {}
    # Getting the type of 'issubroutine' (line 290)
    issubroutine_67483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 31), 'issubroutine', False)
    # Calling issubroutine(args, kwargs) (line 290)
    issubroutine_call_result_67486 = invoke(stypy.reporting.localization.Localization(__file__, 290, 31), issubroutine_67483, *[rout_67484], **kwargs_67485)
    
    # Applying the binary operator 'or' (line 290)
    result_or_keyword_67487 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 11), 'or', isfunction_call_result_67482, issubroutine_call_result_67486)
    
    # Assigning a type to the variable 'stypy_return_type' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type', result_or_keyword_67487)
    
    # ################# End of 'isroutine(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isroutine' in the type store
    # Getting the type of 'stypy_return_type' (line 289)
    stypy_return_type_67488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67488)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isroutine'
    return stypy_return_type_67488

# Assigning a type to the variable 'isroutine' (line 289)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 0), 'isroutine', isroutine)

@norecursion
def islogicalfunction(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'islogicalfunction'
    module_type_store = module_type_store.open_function_context('islogicalfunction', 293, 0, False)
    
    # Passed parameters checking function
    islogicalfunction.stypy_localization = localization
    islogicalfunction.stypy_type_of_self = None
    islogicalfunction.stypy_type_store = module_type_store
    islogicalfunction.stypy_function_name = 'islogicalfunction'
    islogicalfunction.stypy_param_names_list = ['rout']
    islogicalfunction.stypy_varargs_param_name = None
    islogicalfunction.stypy_kwargs_param_name = None
    islogicalfunction.stypy_call_defaults = defaults
    islogicalfunction.stypy_call_varargs = varargs
    islogicalfunction.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'islogicalfunction', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'islogicalfunction', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'islogicalfunction(...)' code ##################

    
    
    
    # Call to isfunction(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'rout' (line 294)
    rout_67490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'rout', False)
    # Processing the call keyword arguments (line 294)
    kwargs_67491 = {}
    # Getting the type of 'isfunction' (line 294)
    isfunction_67489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 11), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 294)
    isfunction_call_result_67492 = invoke(stypy.reporting.localization.Localization(__file__, 294, 11), isfunction_67489, *[rout_67490], **kwargs_67491)
    
    # Applying the 'not' unary operator (line 294)
    result_not__67493 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 7), 'not', isfunction_call_result_67492)
    
    # Testing the type of an if condition (line 294)
    if_condition_67494 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 4), result_not__67493)
    # Assigning a type to the variable 'if_condition_67494' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'if_condition_67494', if_condition_67494)
    # SSA begins for if statement (line 294)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'stypy_return_type', int_67495)
    # SSA join for if statement (line 294)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_67496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 7), 'str', 'result')
    # Getting the type of 'rout' (line 296)
    rout_67497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 'rout')
    # Applying the binary operator 'in' (line 296)
    result_contains_67498 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 7), 'in', str_67496, rout_67497)
    
    # Testing the type of an if condition (line 296)
    if_condition_67499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 4), result_contains_67498)
    # Assigning a type to the variable 'if_condition_67499' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'if_condition_67499', if_condition_67499)
    # SSA begins for if statement (line 296)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 297):
    
    # Assigning a Subscript to a Name (line 297):
    
    # Obtaining the type of the subscript
    str_67500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 17), 'str', 'result')
    # Getting the type of 'rout' (line 297)
    rout_67501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 297)
    getitem___67502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 12), rout_67501, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 297)
    subscript_call_result_67503 = invoke(stypy.reporting.localization.Localization(__file__, 297, 12), getitem___67502, str_67500)
    
    # Assigning a type to the variable 'a' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'a', subscript_call_result_67503)
    # SSA branch for the else part of an if statement (line 296)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 299):
    
    # Assigning a Subscript to a Name (line 299):
    
    # Obtaining the type of the subscript
    str_67504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 17), 'str', 'name')
    # Getting the type of 'rout' (line 299)
    rout_67505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 299)
    getitem___67506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), rout_67505, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 299)
    subscript_call_result_67507 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), getitem___67506, str_67504)
    
    # Assigning a type to the variable 'a' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'a', subscript_call_result_67507)
    # SSA join for if statement (line 296)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a' (line 300)
    a_67508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 7), 'a')
    
    # Obtaining the type of the subscript
    str_67509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 17), 'str', 'vars')
    # Getting the type of 'rout' (line 300)
    rout_67510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 300)
    getitem___67511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), rout_67510, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 300)
    subscript_call_result_67512 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), getitem___67511, str_67509)
    
    # Applying the binary operator 'in' (line 300)
    result_contains_67513 = python_operator(stypy.reporting.localization.Localization(__file__, 300, 7), 'in', a_67508, subscript_call_result_67512)
    
    # Testing the type of an if condition (line 300)
    if_condition_67514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 300, 4), result_contains_67513)
    # Assigning a type to the variable 'if_condition_67514' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'if_condition_67514', if_condition_67514)
    # SSA begins for if statement (line 300)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to islogical(...): (line 301)
    # Processing the call arguments (line 301)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 301)
    a_67516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 38), 'a', False)
    
    # Obtaining the type of the subscript
    str_67517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 30), 'str', 'vars')
    # Getting the type of 'rout' (line 301)
    rout_67518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 25), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 301)
    getitem___67519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 25), rout_67518, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 301)
    subscript_call_result_67520 = invoke(stypy.reporting.localization.Localization(__file__, 301, 25), getitem___67519, str_67517)
    
    # Obtaining the member '__getitem__' of a type (line 301)
    getitem___67521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 25), subscript_call_result_67520, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 301)
    subscript_call_result_67522 = invoke(stypy.reporting.localization.Localization(__file__, 301, 25), getitem___67521, a_67516)
    
    # Processing the call keyword arguments (line 301)
    kwargs_67523 = {}
    # Getting the type of 'islogical' (line 301)
    islogical_67515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 15), 'islogical', False)
    # Calling islogical(args, kwargs) (line 301)
    islogical_call_result_67524 = invoke(stypy.reporting.localization.Localization(__file__, 301, 15), islogical_67515, *[subscript_call_result_67522], **kwargs_67523)
    
    # Assigning a type to the variable 'stypy_return_type' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'stypy_return_type', islogical_call_result_67524)
    # SSA join for if statement (line 300)
    module_type_store = module_type_store.join_ssa_context()
    
    int_67525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'stypy_return_type', int_67525)
    
    # ################# End of 'islogicalfunction(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'islogicalfunction' in the type store
    # Getting the type of 'stypy_return_type' (line 293)
    stypy_return_type_67526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67526)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'islogicalfunction'
    return stypy_return_type_67526

# Assigning a type to the variable 'islogicalfunction' (line 293)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 0), 'islogicalfunction', islogicalfunction)

@norecursion
def islong_longfunction(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'islong_longfunction'
    module_type_store = module_type_store.open_function_context('islong_longfunction', 305, 0, False)
    
    # Passed parameters checking function
    islong_longfunction.stypy_localization = localization
    islong_longfunction.stypy_type_of_self = None
    islong_longfunction.stypy_type_store = module_type_store
    islong_longfunction.stypy_function_name = 'islong_longfunction'
    islong_longfunction.stypy_param_names_list = ['rout']
    islong_longfunction.stypy_varargs_param_name = None
    islong_longfunction.stypy_kwargs_param_name = None
    islong_longfunction.stypy_call_defaults = defaults
    islong_longfunction.stypy_call_varargs = varargs
    islong_longfunction.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'islong_longfunction', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'islong_longfunction', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'islong_longfunction(...)' code ##################

    
    
    
    # Call to isfunction(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'rout' (line 306)
    rout_67528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 22), 'rout', False)
    # Processing the call keyword arguments (line 306)
    kwargs_67529 = {}
    # Getting the type of 'isfunction' (line 306)
    isfunction_67527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 11), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 306)
    isfunction_call_result_67530 = invoke(stypy.reporting.localization.Localization(__file__, 306, 11), isfunction_67527, *[rout_67528], **kwargs_67529)
    
    # Applying the 'not' unary operator (line 306)
    result_not__67531 = python_operator(stypy.reporting.localization.Localization(__file__, 306, 7), 'not', isfunction_call_result_67530)
    
    # Testing the type of an if condition (line 306)
    if_condition_67532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 306, 4), result_not__67531)
    # Assigning a type to the variable 'if_condition_67532' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'if_condition_67532', if_condition_67532)
    # SSA begins for if statement (line 306)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'stypy_return_type', int_67533)
    # SSA join for if statement (line 306)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_67534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 7), 'str', 'result')
    # Getting the type of 'rout' (line 308)
    rout_67535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 19), 'rout')
    # Applying the binary operator 'in' (line 308)
    result_contains_67536 = python_operator(stypy.reporting.localization.Localization(__file__, 308, 7), 'in', str_67534, rout_67535)
    
    # Testing the type of an if condition (line 308)
    if_condition_67537 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 308, 4), result_contains_67536)
    # Assigning a type to the variable 'if_condition_67537' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'if_condition_67537', if_condition_67537)
    # SSA begins for if statement (line 308)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 309):
    
    # Assigning a Subscript to a Name (line 309):
    
    # Obtaining the type of the subscript
    str_67538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 17), 'str', 'result')
    # Getting the type of 'rout' (line 309)
    rout_67539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 309)
    getitem___67540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 12), rout_67539, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 309)
    subscript_call_result_67541 = invoke(stypy.reporting.localization.Localization(__file__, 309, 12), getitem___67540, str_67538)
    
    # Assigning a type to the variable 'a' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'a', subscript_call_result_67541)
    # SSA branch for the else part of an if statement (line 308)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 311):
    
    # Assigning a Subscript to a Name (line 311):
    
    # Obtaining the type of the subscript
    str_67542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 311, 17), 'str', 'name')
    # Getting the type of 'rout' (line 311)
    rout_67543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 311)
    getitem___67544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 12), rout_67543, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 311)
    subscript_call_result_67545 = invoke(stypy.reporting.localization.Localization(__file__, 311, 12), getitem___67544, str_67542)
    
    # Assigning a type to the variable 'a' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'a', subscript_call_result_67545)
    # SSA join for if statement (line 308)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a' (line 312)
    a_67546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 7), 'a')
    
    # Obtaining the type of the subscript
    str_67547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 17), 'str', 'vars')
    # Getting the type of 'rout' (line 312)
    rout_67548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___67549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 12), rout_67548, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_67550 = invoke(stypy.reporting.localization.Localization(__file__, 312, 12), getitem___67549, str_67547)
    
    # Applying the binary operator 'in' (line 312)
    result_contains_67551 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 7), 'in', a_67546, subscript_call_result_67550)
    
    # Testing the type of an if condition (line 312)
    if_condition_67552 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 4), result_contains_67551)
    # Assigning a type to the variable 'if_condition_67552' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'if_condition_67552', if_condition_67552)
    # SSA begins for if statement (line 312)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to islong_long(...): (line 313)
    # Processing the call arguments (line 313)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 313)
    a_67554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 40), 'a', False)
    
    # Obtaining the type of the subscript
    str_67555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 32), 'str', 'vars')
    # Getting the type of 'rout' (line 313)
    rout_67556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 27), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___67557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 27), rout_67556, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_67558 = invoke(stypy.reporting.localization.Localization(__file__, 313, 27), getitem___67557, str_67555)
    
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___67559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 27), subscript_call_result_67558, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_67560 = invoke(stypy.reporting.localization.Localization(__file__, 313, 27), getitem___67559, a_67554)
    
    # Processing the call keyword arguments (line 313)
    kwargs_67561 = {}
    # Getting the type of 'islong_long' (line 313)
    islong_long_67553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 15), 'islong_long', False)
    # Calling islong_long(args, kwargs) (line 313)
    islong_long_call_result_67562 = invoke(stypy.reporting.localization.Localization(__file__, 313, 15), islong_long_67553, *[subscript_call_result_67560], **kwargs_67561)
    
    # Assigning a type to the variable 'stypy_return_type' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'stypy_return_type', islong_long_call_result_67562)
    # SSA join for if statement (line 312)
    module_type_store = module_type_store.join_ssa_context()
    
    int_67563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 4), 'stypy_return_type', int_67563)
    
    # ################# End of 'islong_longfunction(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'islong_longfunction' in the type store
    # Getting the type of 'stypy_return_type' (line 305)
    stypy_return_type_67564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67564)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'islong_longfunction'
    return stypy_return_type_67564

# Assigning a type to the variable 'islong_longfunction' (line 305)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 0), 'islong_longfunction', islong_longfunction)

@norecursion
def islong_doublefunction(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'islong_doublefunction'
    module_type_store = module_type_store.open_function_context('islong_doublefunction', 317, 0, False)
    
    # Passed parameters checking function
    islong_doublefunction.stypy_localization = localization
    islong_doublefunction.stypy_type_of_self = None
    islong_doublefunction.stypy_type_store = module_type_store
    islong_doublefunction.stypy_function_name = 'islong_doublefunction'
    islong_doublefunction.stypy_param_names_list = ['rout']
    islong_doublefunction.stypy_varargs_param_name = None
    islong_doublefunction.stypy_kwargs_param_name = None
    islong_doublefunction.stypy_call_defaults = defaults
    islong_doublefunction.stypy_call_varargs = varargs
    islong_doublefunction.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'islong_doublefunction', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'islong_doublefunction', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'islong_doublefunction(...)' code ##################

    
    
    
    # Call to isfunction(...): (line 318)
    # Processing the call arguments (line 318)
    # Getting the type of 'rout' (line 318)
    rout_67566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 22), 'rout', False)
    # Processing the call keyword arguments (line 318)
    kwargs_67567 = {}
    # Getting the type of 'isfunction' (line 318)
    isfunction_67565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 318)
    isfunction_call_result_67568 = invoke(stypy.reporting.localization.Localization(__file__, 318, 11), isfunction_67565, *[rout_67566], **kwargs_67567)
    
    # Applying the 'not' unary operator (line 318)
    result_not__67569 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 7), 'not', isfunction_call_result_67568)
    
    # Testing the type of an if condition (line 318)
    if_condition_67570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 318, 4), result_not__67569)
    # Assigning a type to the variable 'if_condition_67570' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'if_condition_67570', if_condition_67570)
    # SSA begins for if statement (line 318)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'stypy_return_type', int_67571)
    # SSA join for if statement (line 318)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_67572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 7), 'str', 'result')
    # Getting the type of 'rout' (line 320)
    rout_67573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 19), 'rout')
    # Applying the binary operator 'in' (line 320)
    result_contains_67574 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 7), 'in', str_67572, rout_67573)
    
    # Testing the type of an if condition (line 320)
    if_condition_67575 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 4), result_contains_67574)
    # Assigning a type to the variable 'if_condition_67575' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'if_condition_67575', if_condition_67575)
    # SSA begins for if statement (line 320)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 321):
    
    # Assigning a Subscript to a Name (line 321):
    
    # Obtaining the type of the subscript
    str_67576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 17), 'str', 'result')
    # Getting the type of 'rout' (line 321)
    rout_67577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 321)
    getitem___67578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 12), rout_67577, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 321)
    subscript_call_result_67579 = invoke(stypy.reporting.localization.Localization(__file__, 321, 12), getitem___67578, str_67576)
    
    # Assigning a type to the variable 'a' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'a', subscript_call_result_67579)
    # SSA branch for the else part of an if statement (line 320)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 323):
    
    # Assigning a Subscript to a Name (line 323):
    
    # Obtaining the type of the subscript
    str_67580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 17), 'str', 'name')
    # Getting the type of 'rout' (line 323)
    rout_67581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 323)
    getitem___67582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), rout_67581, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 323)
    subscript_call_result_67583 = invoke(stypy.reporting.localization.Localization(__file__, 323, 12), getitem___67582, str_67580)
    
    # Assigning a type to the variable 'a' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'a', subscript_call_result_67583)
    # SSA join for if statement (line 320)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a' (line 324)
    a_67584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 7), 'a')
    
    # Obtaining the type of the subscript
    str_67585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 17), 'str', 'vars')
    # Getting the type of 'rout' (line 324)
    rout_67586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 324)
    getitem___67587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), rout_67586, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 324)
    subscript_call_result_67588 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), getitem___67587, str_67585)
    
    # Applying the binary operator 'in' (line 324)
    result_contains_67589 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 7), 'in', a_67584, subscript_call_result_67588)
    
    # Testing the type of an if condition (line 324)
    if_condition_67590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 324, 4), result_contains_67589)
    # Assigning a type to the variable 'if_condition_67590' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'if_condition_67590', if_condition_67590)
    # SSA begins for if statement (line 324)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to islong_double(...): (line 325)
    # Processing the call arguments (line 325)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 325)
    a_67592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 42), 'a', False)
    
    # Obtaining the type of the subscript
    str_67593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 34), 'str', 'vars')
    # Getting the type of 'rout' (line 325)
    rout_67594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 29), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___67595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 29), rout_67594, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_67596 = invoke(stypy.reporting.localization.Localization(__file__, 325, 29), getitem___67595, str_67593)
    
    # Obtaining the member '__getitem__' of a type (line 325)
    getitem___67597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 29), subscript_call_result_67596, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 325)
    subscript_call_result_67598 = invoke(stypy.reporting.localization.Localization(__file__, 325, 29), getitem___67597, a_67592)
    
    # Processing the call keyword arguments (line 325)
    kwargs_67599 = {}
    # Getting the type of 'islong_double' (line 325)
    islong_double_67591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 15), 'islong_double', False)
    # Calling islong_double(args, kwargs) (line 325)
    islong_double_call_result_67600 = invoke(stypy.reporting.localization.Localization(__file__, 325, 15), islong_double_67591, *[subscript_call_result_67598], **kwargs_67599)
    
    # Assigning a type to the variable 'stypy_return_type' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'stypy_return_type', islong_double_call_result_67600)
    # SSA join for if statement (line 324)
    module_type_store = module_type_store.join_ssa_context()
    
    int_67601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'stypy_return_type', int_67601)
    
    # ################# End of 'islong_doublefunction(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'islong_doublefunction' in the type store
    # Getting the type of 'stypy_return_type' (line 317)
    stypy_return_type_67602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67602)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'islong_doublefunction'
    return stypy_return_type_67602

# Assigning a type to the variable 'islong_doublefunction' (line 317)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 0), 'islong_doublefunction', islong_doublefunction)

@norecursion
def iscomplexfunction(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iscomplexfunction'
    module_type_store = module_type_store.open_function_context('iscomplexfunction', 329, 0, False)
    
    # Passed parameters checking function
    iscomplexfunction.stypy_localization = localization
    iscomplexfunction.stypy_type_of_self = None
    iscomplexfunction.stypy_type_store = module_type_store
    iscomplexfunction.stypy_function_name = 'iscomplexfunction'
    iscomplexfunction.stypy_param_names_list = ['rout']
    iscomplexfunction.stypy_varargs_param_name = None
    iscomplexfunction.stypy_kwargs_param_name = None
    iscomplexfunction.stypy_call_defaults = defaults
    iscomplexfunction.stypy_call_varargs = varargs
    iscomplexfunction.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iscomplexfunction', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iscomplexfunction', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iscomplexfunction(...)' code ##################

    
    
    
    # Call to isfunction(...): (line 330)
    # Processing the call arguments (line 330)
    # Getting the type of 'rout' (line 330)
    rout_67604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 22), 'rout', False)
    # Processing the call keyword arguments (line 330)
    kwargs_67605 = {}
    # Getting the type of 'isfunction' (line 330)
    isfunction_67603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 11), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 330)
    isfunction_call_result_67606 = invoke(stypy.reporting.localization.Localization(__file__, 330, 11), isfunction_67603, *[rout_67604], **kwargs_67605)
    
    # Applying the 'not' unary operator (line 330)
    result_not__67607 = python_operator(stypy.reporting.localization.Localization(__file__, 330, 7), 'not', isfunction_call_result_67606)
    
    # Testing the type of an if condition (line 330)
    if_condition_67608 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 330, 4), result_not__67607)
    # Assigning a type to the variable 'if_condition_67608' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 4), 'if_condition_67608', if_condition_67608)
    # SSA begins for if statement (line 330)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 331)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'stypy_return_type', int_67609)
    # SSA join for if statement (line 330)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_67610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 7), 'str', 'result')
    # Getting the type of 'rout' (line 332)
    rout_67611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 'rout')
    # Applying the binary operator 'in' (line 332)
    result_contains_67612 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 7), 'in', str_67610, rout_67611)
    
    # Testing the type of an if condition (line 332)
    if_condition_67613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 332, 4), result_contains_67612)
    # Assigning a type to the variable 'if_condition_67613' (line 332)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'if_condition_67613', if_condition_67613)
    # SSA begins for if statement (line 332)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 333):
    
    # Assigning a Subscript to a Name (line 333):
    
    # Obtaining the type of the subscript
    str_67614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 17), 'str', 'result')
    # Getting the type of 'rout' (line 333)
    rout_67615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___67616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), rout_67615, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_67617 = invoke(stypy.reporting.localization.Localization(__file__, 333, 12), getitem___67616, str_67614)
    
    # Assigning a type to the variable 'a' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'a', subscript_call_result_67617)
    # SSA branch for the else part of an if statement (line 332)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 335):
    
    # Assigning a Subscript to a Name (line 335):
    
    # Obtaining the type of the subscript
    str_67618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 17), 'str', 'name')
    # Getting the type of 'rout' (line 335)
    rout_67619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 335)
    getitem___67620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), rout_67619, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 335)
    subscript_call_result_67621 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), getitem___67620, str_67618)
    
    # Assigning a type to the variable 'a' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'a', subscript_call_result_67621)
    # SSA join for if statement (line 332)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a' (line 336)
    a_67622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 7), 'a')
    
    # Obtaining the type of the subscript
    str_67623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 17), 'str', 'vars')
    # Getting the type of 'rout' (line 336)
    rout_67624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 336)
    getitem___67625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), rout_67624, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 336)
    subscript_call_result_67626 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), getitem___67625, str_67623)
    
    # Applying the binary operator 'in' (line 336)
    result_contains_67627 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 7), 'in', a_67622, subscript_call_result_67626)
    
    # Testing the type of an if condition (line 336)
    if_condition_67628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 4), result_contains_67627)
    # Assigning a type to the variable 'if_condition_67628' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'if_condition_67628', if_condition_67628)
    # SSA begins for if statement (line 336)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to iscomplex(...): (line 337)
    # Processing the call arguments (line 337)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 337)
    a_67630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 38), 'a', False)
    
    # Obtaining the type of the subscript
    str_67631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 30), 'str', 'vars')
    # Getting the type of 'rout' (line 337)
    rout_67632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 25), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___67633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 25), rout_67632, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_67634 = invoke(stypy.reporting.localization.Localization(__file__, 337, 25), getitem___67633, str_67631)
    
    # Obtaining the member '__getitem__' of a type (line 337)
    getitem___67635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 25), subscript_call_result_67634, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 337)
    subscript_call_result_67636 = invoke(stypy.reporting.localization.Localization(__file__, 337, 25), getitem___67635, a_67630)
    
    # Processing the call keyword arguments (line 337)
    kwargs_67637 = {}
    # Getting the type of 'iscomplex' (line 337)
    iscomplex_67629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 15), 'iscomplex', False)
    # Calling iscomplex(args, kwargs) (line 337)
    iscomplex_call_result_67638 = invoke(stypy.reporting.localization.Localization(__file__, 337, 15), iscomplex_67629, *[subscript_call_result_67636], **kwargs_67637)
    
    # Assigning a type to the variable 'stypy_return_type' (line 337)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 8), 'stypy_return_type', iscomplex_call_result_67638)
    # SSA join for if statement (line 336)
    module_type_store = module_type_store.join_ssa_context()
    
    int_67639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 338)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'stypy_return_type', int_67639)
    
    # ################# End of 'iscomplexfunction(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iscomplexfunction' in the type store
    # Getting the type of 'stypy_return_type' (line 329)
    stypy_return_type_67640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67640)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iscomplexfunction'
    return stypy_return_type_67640

# Assigning a type to the variable 'iscomplexfunction' (line 329)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 0), 'iscomplexfunction', iscomplexfunction)

@norecursion
def iscomplexfunction_warn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'iscomplexfunction_warn'
    module_type_store = module_type_store.open_function_context('iscomplexfunction_warn', 341, 0, False)
    
    # Passed parameters checking function
    iscomplexfunction_warn.stypy_localization = localization
    iscomplexfunction_warn.stypy_type_of_self = None
    iscomplexfunction_warn.stypy_type_store = module_type_store
    iscomplexfunction_warn.stypy_function_name = 'iscomplexfunction_warn'
    iscomplexfunction_warn.stypy_param_names_list = ['rout']
    iscomplexfunction_warn.stypy_varargs_param_name = None
    iscomplexfunction_warn.stypy_kwargs_param_name = None
    iscomplexfunction_warn.stypy_call_defaults = defaults
    iscomplexfunction_warn.stypy_call_varargs = varargs
    iscomplexfunction_warn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'iscomplexfunction_warn', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'iscomplexfunction_warn', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'iscomplexfunction_warn(...)' code ##################

    
    
    # Call to iscomplexfunction(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'rout' (line 342)
    rout_67642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 25), 'rout', False)
    # Processing the call keyword arguments (line 342)
    kwargs_67643 = {}
    # Getting the type of 'iscomplexfunction' (line 342)
    iscomplexfunction_67641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 7), 'iscomplexfunction', False)
    # Calling iscomplexfunction(args, kwargs) (line 342)
    iscomplexfunction_call_result_67644 = invoke(stypy.reporting.localization.Localization(__file__, 342, 7), iscomplexfunction_67641, *[rout_67642], **kwargs_67643)
    
    # Testing the type of an if condition (line 342)
    if_condition_67645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 4), iscomplexfunction_call_result_67644)
    # Assigning a type to the variable 'if_condition_67645' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'if_condition_67645', if_condition_67645)
    # SSA begins for if statement (line 342)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 343)
    # Processing the call arguments (line 343)
    str_67647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, (-1)), 'str', '    **************************************************************\n        Warning: code with a function returning complex value\n        may not work correctly with your Fortran compiler.\n        Run the following test before using it in your applications:\n        $(f2py install dir)/test-site/{b/runme_scalar,e/runme}\n        When using GNU gcc/g77 compilers, codes should work correctly.\n    **************************************************************\n')
    # Processing the call keyword arguments (line 343)
    kwargs_67648 = {}
    # Getting the type of 'outmess' (line 343)
    outmess_67646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'outmess', False)
    # Calling outmess(args, kwargs) (line 343)
    outmess_call_result_67649 = invoke(stypy.reporting.localization.Localization(__file__, 343, 8), outmess_67646, *[str_67647], **kwargs_67648)
    
    int_67650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 8), 'stypy_return_type', int_67650)
    # SSA join for if statement (line 342)
    module_type_store = module_type_store.join_ssa_context()
    
    int_67651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'stypy_return_type', int_67651)
    
    # ################# End of 'iscomplexfunction_warn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'iscomplexfunction_warn' in the type store
    # Getting the type of 'stypy_return_type' (line 341)
    stypy_return_type_67652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67652)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'iscomplexfunction_warn'
    return stypy_return_type_67652

# Assigning a type to the variable 'iscomplexfunction_warn' (line 341)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 0), 'iscomplexfunction_warn', iscomplexfunction_warn)

@norecursion
def isstringfunction(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isstringfunction'
    module_type_store = module_type_store.open_function_context('isstringfunction', 355, 0, False)
    
    # Passed parameters checking function
    isstringfunction.stypy_localization = localization
    isstringfunction.stypy_type_of_self = None
    isstringfunction.stypy_type_store = module_type_store
    isstringfunction.stypy_function_name = 'isstringfunction'
    isstringfunction.stypy_param_names_list = ['rout']
    isstringfunction.stypy_varargs_param_name = None
    isstringfunction.stypy_kwargs_param_name = None
    isstringfunction.stypy_call_defaults = defaults
    isstringfunction.stypy_call_varargs = varargs
    isstringfunction.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isstringfunction', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isstringfunction', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isstringfunction(...)' code ##################

    
    
    
    # Call to isfunction(...): (line 356)
    # Processing the call arguments (line 356)
    # Getting the type of 'rout' (line 356)
    rout_67654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 22), 'rout', False)
    # Processing the call keyword arguments (line 356)
    kwargs_67655 = {}
    # Getting the type of 'isfunction' (line 356)
    isfunction_67653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 11), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 356)
    isfunction_call_result_67656 = invoke(stypy.reporting.localization.Localization(__file__, 356, 11), isfunction_67653, *[rout_67654], **kwargs_67655)
    
    # Applying the 'not' unary operator (line 356)
    result_not__67657 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 7), 'not', isfunction_call_result_67656)
    
    # Testing the type of an if condition (line 356)
    if_condition_67658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 4), result_not__67657)
    # Assigning a type to the variable 'if_condition_67658' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'if_condition_67658', if_condition_67658)
    # SSA begins for if statement (line 356)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'stypy_return_type', int_67659)
    # SSA join for if statement (line 356)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_67660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 7), 'str', 'result')
    # Getting the type of 'rout' (line 358)
    rout_67661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 19), 'rout')
    # Applying the binary operator 'in' (line 358)
    result_contains_67662 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 7), 'in', str_67660, rout_67661)
    
    # Testing the type of an if condition (line 358)
    if_condition_67663 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 358, 4), result_contains_67662)
    # Assigning a type to the variable 'if_condition_67663' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'if_condition_67663', if_condition_67663)
    # SSA begins for if statement (line 358)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 359):
    
    # Assigning a Subscript to a Name (line 359):
    
    # Obtaining the type of the subscript
    str_67664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 17), 'str', 'result')
    # Getting the type of 'rout' (line 359)
    rout_67665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 359)
    getitem___67666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 12), rout_67665, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 359)
    subscript_call_result_67667 = invoke(stypy.reporting.localization.Localization(__file__, 359, 12), getitem___67666, str_67664)
    
    # Assigning a type to the variable 'a' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'a', subscript_call_result_67667)
    # SSA branch for the else part of an if statement (line 358)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 361):
    
    # Assigning a Subscript to a Name (line 361):
    
    # Obtaining the type of the subscript
    str_67668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 17), 'str', 'name')
    # Getting the type of 'rout' (line 361)
    rout_67669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 361)
    getitem___67670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 12), rout_67669, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 361)
    subscript_call_result_67671 = invoke(stypy.reporting.localization.Localization(__file__, 361, 12), getitem___67670, str_67668)
    
    # Assigning a type to the variable 'a' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'a', subscript_call_result_67671)
    # SSA join for if statement (line 358)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a' (line 362)
    a_67672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 7), 'a')
    
    # Obtaining the type of the subscript
    str_67673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 362, 17), 'str', 'vars')
    # Getting the type of 'rout' (line 362)
    rout_67674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 362)
    getitem___67675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 12), rout_67674, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 362)
    subscript_call_result_67676 = invoke(stypy.reporting.localization.Localization(__file__, 362, 12), getitem___67675, str_67673)
    
    # Applying the binary operator 'in' (line 362)
    result_contains_67677 = python_operator(stypy.reporting.localization.Localization(__file__, 362, 7), 'in', a_67672, subscript_call_result_67676)
    
    # Testing the type of an if condition (line 362)
    if_condition_67678 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 362, 4), result_contains_67677)
    # Assigning a type to the variable 'if_condition_67678' (line 362)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 4), 'if_condition_67678', if_condition_67678)
    # SSA begins for if statement (line 362)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to isstring(...): (line 363)
    # Processing the call arguments (line 363)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 363)
    a_67680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 37), 'a', False)
    
    # Obtaining the type of the subscript
    str_67681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 29), 'str', 'vars')
    # Getting the type of 'rout' (line 363)
    rout_67682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 24), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 363)
    getitem___67683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 24), rout_67682, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 363)
    subscript_call_result_67684 = invoke(stypy.reporting.localization.Localization(__file__, 363, 24), getitem___67683, str_67681)
    
    # Obtaining the member '__getitem__' of a type (line 363)
    getitem___67685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 24), subscript_call_result_67684, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 363)
    subscript_call_result_67686 = invoke(stypy.reporting.localization.Localization(__file__, 363, 24), getitem___67685, a_67680)
    
    # Processing the call keyword arguments (line 363)
    kwargs_67687 = {}
    # Getting the type of 'isstring' (line 363)
    isstring_67679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 15), 'isstring', False)
    # Calling isstring(args, kwargs) (line 363)
    isstring_call_result_67688 = invoke(stypy.reporting.localization.Localization(__file__, 363, 15), isstring_67679, *[subscript_call_result_67686], **kwargs_67687)
    
    # Assigning a type to the variable 'stypy_return_type' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'stypy_return_type', isstring_call_result_67688)
    # SSA join for if statement (line 362)
    module_type_store = module_type_store.join_ssa_context()
    
    int_67689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'stypy_return_type', int_67689)
    
    # ################# End of 'isstringfunction(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isstringfunction' in the type store
    # Getting the type of 'stypy_return_type' (line 355)
    stypy_return_type_67690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67690)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isstringfunction'
    return stypy_return_type_67690

# Assigning a type to the variable 'isstringfunction' (line 355)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 0), 'isstringfunction', isstringfunction)

@norecursion
def hasexternals(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hasexternals'
    module_type_store = module_type_store.open_function_context('hasexternals', 367, 0, False)
    
    # Passed parameters checking function
    hasexternals.stypy_localization = localization
    hasexternals.stypy_type_of_self = None
    hasexternals.stypy_type_store = module_type_store
    hasexternals.stypy_function_name = 'hasexternals'
    hasexternals.stypy_param_names_list = ['rout']
    hasexternals.stypy_varargs_param_name = None
    hasexternals.stypy_kwargs_param_name = None
    hasexternals.stypy_call_defaults = defaults
    hasexternals.stypy_call_varargs = varargs
    hasexternals.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hasexternals', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hasexternals', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hasexternals(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_67691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 11), 'str', 'externals')
    # Getting the type of 'rout' (line 368)
    rout_67692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 26), 'rout')
    # Applying the binary operator 'in' (line 368)
    result_contains_67693 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 11), 'in', str_67691, rout_67692)
    
    
    # Obtaining the type of the subscript
    str_67694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 40), 'str', 'externals')
    # Getting the type of 'rout' (line 368)
    rout_67695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 35), 'rout')
    # Obtaining the member '__getitem__' of a type (line 368)
    getitem___67696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 35), rout_67695, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 368)
    subscript_call_result_67697 = invoke(stypy.reporting.localization.Localization(__file__, 368, 35), getitem___67696, str_67694)
    
    # Applying the binary operator 'and' (line 368)
    result_and_keyword_67698 = python_operator(stypy.reporting.localization.Localization(__file__, 368, 11), 'and', result_contains_67693, subscript_call_result_67697)
    
    # Assigning a type to the variable 'stypy_return_type' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type', result_and_keyword_67698)
    
    # ################# End of 'hasexternals(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hasexternals' in the type store
    # Getting the type of 'stypy_return_type' (line 367)
    stypy_return_type_67699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67699)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hasexternals'
    return stypy_return_type_67699

# Assigning a type to the variable 'hasexternals' (line 367)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 0), 'hasexternals', hasexternals)

@norecursion
def isthreadsafe(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isthreadsafe'
    module_type_store = module_type_store.open_function_context('isthreadsafe', 371, 0, False)
    
    # Passed parameters checking function
    isthreadsafe.stypy_localization = localization
    isthreadsafe.stypy_type_of_self = None
    isthreadsafe.stypy_type_store = module_type_store
    isthreadsafe.stypy_function_name = 'isthreadsafe'
    isthreadsafe.stypy_param_names_list = ['rout']
    isthreadsafe.stypy_varargs_param_name = None
    isthreadsafe.stypy_kwargs_param_name = None
    isthreadsafe.stypy_call_defaults = defaults
    isthreadsafe.stypy_call_varargs = varargs
    isthreadsafe.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isthreadsafe', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isthreadsafe', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isthreadsafe(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_67700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 11), 'str', 'f2pyenhancements')
    # Getting the type of 'rout' (line 372)
    rout_67701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 33), 'rout')
    # Applying the binary operator 'in' (line 372)
    result_contains_67702 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 11), 'in', str_67700, rout_67701)
    
    
    str_67703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 11), 'str', 'threadsafe')
    
    # Obtaining the type of the subscript
    str_67704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 32), 'str', 'f2pyenhancements')
    # Getting the type of 'rout' (line 373)
    rout_67705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 27), 'rout')
    # Obtaining the member '__getitem__' of a type (line 373)
    getitem___67706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 27), rout_67705, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 373)
    subscript_call_result_67707 = invoke(stypy.reporting.localization.Localization(__file__, 373, 27), getitem___67706, str_67704)
    
    # Applying the binary operator 'in' (line 373)
    result_contains_67708 = python_operator(stypy.reporting.localization.Localization(__file__, 373, 11), 'in', str_67703, subscript_call_result_67707)
    
    # Applying the binary operator 'and' (line 372)
    result_and_keyword_67709 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 11), 'and', result_contains_67702, result_contains_67708)
    
    # Assigning a type to the variable 'stypy_return_type' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'stypy_return_type', result_and_keyword_67709)
    
    # ################# End of 'isthreadsafe(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isthreadsafe' in the type store
    # Getting the type of 'stypy_return_type' (line 371)
    stypy_return_type_67710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67710)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isthreadsafe'
    return stypy_return_type_67710

# Assigning a type to the variable 'isthreadsafe' (line 371)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'isthreadsafe', isthreadsafe)

@norecursion
def hasvariables(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hasvariables'
    module_type_store = module_type_store.open_function_context('hasvariables', 376, 0, False)
    
    # Passed parameters checking function
    hasvariables.stypy_localization = localization
    hasvariables.stypy_type_of_self = None
    hasvariables.stypy_type_store = module_type_store
    hasvariables.stypy_function_name = 'hasvariables'
    hasvariables.stypy_param_names_list = ['rout']
    hasvariables.stypy_varargs_param_name = None
    hasvariables.stypy_kwargs_param_name = None
    hasvariables.stypy_call_defaults = defaults
    hasvariables.stypy_call_varargs = varargs
    hasvariables.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hasvariables', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hasvariables', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hasvariables(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_67711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 11), 'str', 'vars')
    # Getting the type of 'rout' (line 377)
    rout_67712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 21), 'rout')
    # Applying the binary operator 'in' (line 377)
    result_contains_67713 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 11), 'in', str_67711, rout_67712)
    
    
    # Obtaining the type of the subscript
    str_67714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 35), 'str', 'vars')
    # Getting the type of 'rout' (line 377)
    rout_67715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 30), 'rout')
    # Obtaining the member '__getitem__' of a type (line 377)
    getitem___67716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 30), rout_67715, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 377)
    subscript_call_result_67717 = invoke(stypy.reporting.localization.Localization(__file__, 377, 30), getitem___67716, str_67714)
    
    # Applying the binary operator 'and' (line 377)
    result_and_keyword_67718 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 11), 'and', result_contains_67713, subscript_call_result_67717)
    
    # Assigning a type to the variable 'stypy_return_type' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'stypy_return_type', result_and_keyword_67718)
    
    # ################# End of 'hasvariables(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hasvariables' in the type store
    # Getting the type of 'stypy_return_type' (line 376)
    stypy_return_type_67719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67719)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hasvariables'
    return stypy_return_type_67719

# Assigning a type to the variable 'hasvariables' (line 376)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 0), 'hasvariables', hasvariables)

@norecursion
def isoptional(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isoptional'
    module_type_store = module_type_store.open_function_context('isoptional', 380, 0, False)
    
    # Passed parameters checking function
    isoptional.stypy_localization = localization
    isoptional.stypy_type_of_self = None
    isoptional.stypy_type_store = module_type_store
    isoptional.stypy_function_name = 'isoptional'
    isoptional.stypy_param_names_list = ['var']
    isoptional.stypy_varargs_param_name = None
    isoptional.stypy_kwargs_param_name = None
    isoptional.stypy_call_defaults = defaults
    isoptional.stypy_call_varargs = varargs
    isoptional.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isoptional', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isoptional', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isoptional(...)' code ##################

    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    str_67720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 12), 'str', 'attrspec')
    # Getting the type of 'var' (line 381)
    var_67721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 26), 'var')
    # Applying the binary operator 'in' (line 381)
    result_contains_67722 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 12), 'in', str_67720, var_67721)
    
    
    str_67723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 34), 'str', 'optional')
    
    # Obtaining the type of the subscript
    str_67724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 381, 52), 'str', 'attrspec')
    # Getting the type of 'var' (line 381)
    var_67725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 48), 'var')
    # Obtaining the member '__getitem__' of a type (line 381)
    getitem___67726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 48), var_67725, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 381)
    subscript_call_result_67727 = invoke(stypy.reporting.localization.Localization(__file__, 381, 48), getitem___67726, str_67724)
    
    # Applying the binary operator 'in' (line 381)
    result_contains_67728 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 34), 'in', str_67723, subscript_call_result_67727)
    
    # Applying the binary operator 'and' (line 381)
    result_and_keyword_67729 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 12), 'and', result_contains_67722, result_contains_67728)
    
    str_67730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 12), 'str', 'required')
    
    # Obtaining the type of the subscript
    str_67731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 34), 'str', 'attrspec')
    # Getting the type of 'var' (line 382)
    var_67732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 30), 'var')
    # Obtaining the member '__getitem__' of a type (line 382)
    getitem___67733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 30), var_67732, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 382)
    subscript_call_result_67734 = invoke(stypy.reporting.localization.Localization(__file__, 382, 30), getitem___67733, str_67731)
    
    # Applying the binary operator 'notin' (line 382)
    result_contains_67735 = python_operator(stypy.reporting.localization.Localization(__file__, 382, 12), 'notin', str_67730, subscript_call_result_67734)
    
    # Applying the binary operator 'and' (line 381)
    result_and_keyword_67736 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 12), 'and', result_and_keyword_67729, result_contains_67735)
    
    
    # Call to isintent_nothide(...): (line 382)
    # Processing the call arguments (line 382)
    # Getting the type of 'var' (line 382)
    var_67738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 68), 'var', False)
    # Processing the call keyword arguments (line 382)
    kwargs_67739 = {}
    # Getting the type of 'isintent_nothide' (line 382)
    isintent_nothide_67737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 51), 'isintent_nothide', False)
    # Calling isintent_nothide(args, kwargs) (line 382)
    isintent_nothide_call_result_67740 = invoke(stypy.reporting.localization.Localization(__file__, 382, 51), isintent_nothide_67737, *[var_67738], **kwargs_67739)
    
    # Applying the binary operator 'and' (line 381)
    result_and_keyword_67741 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 11), 'and', result_and_keyword_67736, isintent_nothide_call_result_67740)
    
    # Assigning a type to the variable 'stypy_return_type' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type', result_and_keyword_67741)
    
    # ################# End of 'isoptional(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isoptional' in the type store
    # Getting the type of 'stypy_return_type' (line 380)
    stypy_return_type_67742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67742)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isoptional'
    return stypy_return_type_67742

# Assigning a type to the variable 'isoptional' (line 380)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 0), 'isoptional', isoptional)

@norecursion
def isexternal(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isexternal'
    module_type_store = module_type_store.open_function_context('isexternal', 385, 0, False)
    
    # Passed parameters checking function
    isexternal.stypy_localization = localization
    isexternal.stypy_type_of_self = None
    isexternal.stypy_type_store = module_type_store
    isexternal.stypy_function_name = 'isexternal'
    isexternal.stypy_param_names_list = ['var']
    isexternal.stypy_varargs_param_name = None
    isexternal.stypy_kwargs_param_name = None
    isexternal.stypy_call_defaults = defaults
    isexternal.stypy_call_varargs = varargs
    isexternal.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isexternal', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isexternal', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isexternal(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_67743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 11), 'str', 'attrspec')
    # Getting the type of 'var' (line 386)
    var_67744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 25), 'var')
    # Applying the binary operator 'in' (line 386)
    result_contains_67745 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 11), 'in', str_67743, var_67744)
    
    
    str_67746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 33), 'str', 'external')
    
    # Obtaining the type of the subscript
    str_67747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 51), 'str', 'attrspec')
    # Getting the type of 'var' (line 386)
    var_67748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 47), 'var')
    # Obtaining the member '__getitem__' of a type (line 386)
    getitem___67749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 47), var_67748, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 386)
    subscript_call_result_67750 = invoke(stypy.reporting.localization.Localization(__file__, 386, 47), getitem___67749, str_67747)
    
    # Applying the binary operator 'in' (line 386)
    result_contains_67751 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 33), 'in', str_67746, subscript_call_result_67750)
    
    # Applying the binary operator 'and' (line 386)
    result_and_keyword_67752 = python_operator(stypy.reporting.localization.Localization(__file__, 386, 11), 'and', result_contains_67745, result_contains_67751)
    
    # Assigning a type to the variable 'stypy_return_type' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'stypy_return_type', result_and_keyword_67752)
    
    # ################# End of 'isexternal(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isexternal' in the type store
    # Getting the type of 'stypy_return_type' (line 385)
    stypy_return_type_67753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67753)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isexternal'
    return stypy_return_type_67753

# Assigning a type to the variable 'isexternal' (line 385)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 0), 'isexternal', isexternal)

@norecursion
def isrequired(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isrequired'
    module_type_store = module_type_store.open_function_context('isrequired', 389, 0, False)
    
    # Passed parameters checking function
    isrequired.stypy_localization = localization
    isrequired.stypy_type_of_self = None
    isrequired.stypy_type_store = module_type_store
    isrequired.stypy_function_name = 'isrequired'
    isrequired.stypy_param_names_list = ['var']
    isrequired.stypy_varargs_param_name = None
    isrequired.stypy_kwargs_param_name = None
    isrequired.stypy_call_defaults = defaults
    isrequired.stypy_call_varargs = varargs
    isrequired.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isrequired', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isrequired', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isrequired(...)' code ##################

    
    # Evaluating a boolean operation
    
    
    # Call to isoptional(...): (line 390)
    # Processing the call arguments (line 390)
    # Getting the type of 'var' (line 390)
    var_67755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 26), 'var', False)
    # Processing the call keyword arguments (line 390)
    kwargs_67756 = {}
    # Getting the type of 'isoptional' (line 390)
    isoptional_67754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 15), 'isoptional', False)
    # Calling isoptional(args, kwargs) (line 390)
    isoptional_call_result_67757 = invoke(stypy.reporting.localization.Localization(__file__, 390, 15), isoptional_67754, *[var_67755], **kwargs_67756)
    
    # Applying the 'not' unary operator (line 390)
    result_not__67758 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 11), 'not', isoptional_call_result_67757)
    
    
    # Call to isintent_nothide(...): (line 390)
    # Processing the call arguments (line 390)
    # Getting the type of 'var' (line 390)
    var_67760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 52), 'var', False)
    # Processing the call keyword arguments (line 390)
    kwargs_67761 = {}
    # Getting the type of 'isintent_nothide' (line 390)
    isintent_nothide_67759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 35), 'isintent_nothide', False)
    # Calling isintent_nothide(args, kwargs) (line 390)
    isintent_nothide_call_result_67762 = invoke(stypy.reporting.localization.Localization(__file__, 390, 35), isintent_nothide_67759, *[var_67760], **kwargs_67761)
    
    # Applying the binary operator 'and' (line 390)
    result_and_keyword_67763 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 11), 'and', result_not__67758, isintent_nothide_call_result_67762)
    
    # Assigning a type to the variable 'stypy_return_type' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 4), 'stypy_return_type', result_and_keyword_67763)
    
    # ################# End of 'isrequired(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isrequired' in the type store
    # Getting the type of 'stypy_return_type' (line 389)
    stypy_return_type_67764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67764)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isrequired'
    return stypy_return_type_67764

# Assigning a type to the variable 'isrequired' (line 389)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 0), 'isrequired', isrequired)

@norecursion
def isintent_in(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_in'
    module_type_store = module_type_store.open_function_context('isintent_in', 393, 0, False)
    
    # Passed parameters checking function
    isintent_in.stypy_localization = localization
    isintent_in.stypy_type_of_self = None
    isintent_in.stypy_type_store = module_type_store
    isintent_in.stypy_function_name = 'isintent_in'
    isintent_in.stypy_param_names_list = ['var']
    isintent_in.stypy_varargs_param_name = None
    isintent_in.stypy_kwargs_param_name = None
    isintent_in.stypy_call_defaults = defaults
    isintent_in.stypy_call_varargs = varargs
    isintent_in.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_in', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_in', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_in(...)' code ##################

    
    
    str_67765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 7), 'str', 'intent')
    # Getting the type of 'var' (line 394)
    var_67766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 23), 'var')
    # Applying the binary operator 'notin' (line 394)
    result_contains_67767 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 7), 'notin', str_67765, var_67766)
    
    # Testing the type of an if condition (line 394)
    if_condition_67768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 4), result_contains_67767)
    # Assigning a type to the variable 'if_condition_67768' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'if_condition_67768', if_condition_67768)
    # SSA begins for if statement (line 394)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 395)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'stypy_return_type', int_67769)
    # SSA join for if statement (line 394)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_67770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 7), 'str', 'hide')
    
    # Obtaining the type of the subscript
    str_67771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 21), 'str', 'intent')
    # Getting the type of 'var' (line 396)
    var_67772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 17), 'var')
    # Obtaining the member '__getitem__' of a type (line 396)
    getitem___67773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 17), var_67772, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 396)
    subscript_call_result_67774 = invoke(stypy.reporting.localization.Localization(__file__, 396, 17), getitem___67773, str_67771)
    
    # Applying the binary operator 'in' (line 396)
    result_contains_67775 = python_operator(stypy.reporting.localization.Localization(__file__, 396, 7), 'in', str_67770, subscript_call_result_67774)
    
    # Testing the type of an if condition (line 396)
    if_condition_67776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 396, 4), result_contains_67775)
    # Assigning a type to the variable 'if_condition_67776' (line 396)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'if_condition_67776', if_condition_67776)
    # SSA begins for if statement (line 396)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'stypy_return_type', int_67777)
    # SSA join for if statement (line 396)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_67778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 7), 'str', 'inplace')
    
    # Obtaining the type of the subscript
    str_67779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 24), 'str', 'intent')
    # Getting the type of 'var' (line 398)
    var_67780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 20), 'var')
    # Obtaining the member '__getitem__' of a type (line 398)
    getitem___67781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 20), var_67780, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 398)
    subscript_call_result_67782 = invoke(stypy.reporting.localization.Localization(__file__, 398, 20), getitem___67781, str_67779)
    
    # Applying the binary operator 'in' (line 398)
    result_contains_67783 = python_operator(stypy.reporting.localization.Localization(__file__, 398, 7), 'in', str_67778, subscript_call_result_67782)
    
    # Testing the type of an if condition (line 398)
    if_condition_67784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 398, 4), result_contains_67783)
    # Assigning a type to the variable 'if_condition_67784' (line 398)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 4), 'if_condition_67784', if_condition_67784)
    # SSA begins for if statement (line 398)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'stypy_return_type', int_67785)
    # SSA join for if statement (line 398)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_67786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 7), 'str', 'in')
    
    # Obtaining the type of the subscript
    str_67787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 19), 'str', 'intent')
    # Getting the type of 'var' (line 400)
    var_67788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 15), 'var')
    # Obtaining the member '__getitem__' of a type (line 400)
    getitem___67789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 15), var_67788, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 400)
    subscript_call_result_67790 = invoke(stypy.reporting.localization.Localization(__file__, 400, 15), getitem___67789, str_67787)
    
    # Applying the binary operator 'in' (line 400)
    result_contains_67791 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 7), 'in', str_67786, subscript_call_result_67790)
    
    # Testing the type of an if condition (line 400)
    if_condition_67792 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 400, 4), result_contains_67791)
    # Assigning a type to the variable 'if_condition_67792' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'if_condition_67792', if_condition_67792)
    # SSA begins for if statement (line 400)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'stypy_return_type', int_67793)
    # SSA join for if statement (line 400)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_67794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 7), 'str', 'out')
    
    # Obtaining the type of the subscript
    str_67795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 20), 'str', 'intent')
    # Getting the type of 'var' (line 402)
    var_67796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 16), 'var')
    # Obtaining the member '__getitem__' of a type (line 402)
    getitem___67797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 16), var_67796, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 402)
    subscript_call_result_67798 = invoke(stypy.reporting.localization.Localization(__file__, 402, 16), getitem___67797, str_67795)
    
    # Applying the binary operator 'in' (line 402)
    result_contains_67799 = python_operator(stypy.reporting.localization.Localization(__file__, 402, 7), 'in', str_67794, subscript_call_result_67798)
    
    # Testing the type of an if condition (line 402)
    if_condition_67800 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 402, 4), result_contains_67799)
    # Assigning a type to the variable 'if_condition_67800' (line 402)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'if_condition_67800', if_condition_67800)
    # SSA begins for if statement (line 402)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 403, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'stypy_return_type', int_67801)
    # SSA join for if statement (line 402)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_67802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 7), 'str', 'inout')
    
    # Obtaining the type of the subscript
    str_67803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 404, 22), 'str', 'intent')
    # Getting the type of 'var' (line 404)
    var_67804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 18), 'var')
    # Obtaining the member '__getitem__' of a type (line 404)
    getitem___67805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 18), var_67804, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 404)
    subscript_call_result_67806 = invoke(stypy.reporting.localization.Localization(__file__, 404, 18), getitem___67805, str_67803)
    
    # Applying the binary operator 'in' (line 404)
    result_contains_67807 = python_operator(stypy.reporting.localization.Localization(__file__, 404, 7), 'in', str_67802, subscript_call_result_67806)
    
    # Testing the type of an if condition (line 404)
    if_condition_67808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 404, 4), result_contains_67807)
    # Assigning a type to the variable 'if_condition_67808' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'if_condition_67808', if_condition_67808)
    # SSA begins for if statement (line 404)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'stypy_return_type', int_67809)
    # SSA join for if statement (line 404)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_67810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 7), 'str', 'outin')
    
    # Obtaining the type of the subscript
    str_67811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 22), 'str', 'intent')
    # Getting the type of 'var' (line 406)
    var_67812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 18), 'var')
    # Obtaining the member '__getitem__' of a type (line 406)
    getitem___67813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 18), var_67812, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 406)
    subscript_call_result_67814 = invoke(stypy.reporting.localization.Localization(__file__, 406, 18), getitem___67813, str_67811)
    
    # Applying the binary operator 'in' (line 406)
    result_contains_67815 = python_operator(stypy.reporting.localization.Localization(__file__, 406, 7), 'in', str_67810, subscript_call_result_67814)
    
    # Testing the type of an if condition (line 406)
    if_condition_67816 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 406, 4), result_contains_67815)
    # Assigning a type to the variable 'if_condition_67816' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'if_condition_67816', if_condition_67816)
    # SSA begins for if statement (line 406)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_67817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 407, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 'stypy_return_type', int_67817)
    # SSA join for if statement (line 406)
    module_type_store = module_type_store.join_ssa_context()
    
    int_67818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'stypy_return_type', int_67818)
    
    # ################# End of 'isintent_in(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_in' in the type store
    # Getting the type of 'stypy_return_type' (line 393)
    stypy_return_type_67819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67819)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_in'
    return stypy_return_type_67819

# Assigning a type to the variable 'isintent_in' (line 393)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 0), 'isintent_in', isintent_in)

@norecursion
def isintent_inout(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_inout'
    module_type_store = module_type_store.open_function_context('isintent_inout', 411, 0, False)
    
    # Passed parameters checking function
    isintent_inout.stypy_localization = localization
    isintent_inout.stypy_type_of_self = None
    isintent_inout.stypy_type_store = module_type_store
    isintent_inout.stypy_function_name = 'isintent_inout'
    isintent_inout.stypy_param_names_list = ['var']
    isintent_inout.stypy_varargs_param_name = None
    isintent_inout.stypy_kwargs_param_name = None
    isintent_inout.stypy_call_defaults = defaults
    isintent_inout.stypy_call_varargs = varargs
    isintent_inout.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_inout', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_inout', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_inout(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_67820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 12), 'str', 'intent')
    # Getting the type of 'var' (line 412)
    var_67821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 24), 'var')
    # Applying the binary operator 'in' (line 412)
    result_contains_67822 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 12), 'in', str_67820, var_67821)
    
    
    # Evaluating a boolean operation
    
    str_67823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 33), 'str', 'inout')
    
    # Obtaining the type of the subscript
    str_67824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 48), 'str', 'intent')
    # Getting the type of 'var' (line 412)
    var_67825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 44), 'var')
    # Obtaining the member '__getitem__' of a type (line 412)
    getitem___67826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 44), var_67825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 412)
    subscript_call_result_67827 = invoke(stypy.reporting.localization.Localization(__file__, 412, 44), getitem___67826, str_67824)
    
    # Applying the binary operator 'in' (line 412)
    result_contains_67828 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 33), 'in', str_67823, subscript_call_result_67827)
    
    
    str_67829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 12), 'str', 'outin')
    
    # Obtaining the type of the subscript
    str_67830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 27), 'str', 'intent')
    # Getting the type of 'var' (line 413)
    var_67831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 23), 'var')
    # Obtaining the member '__getitem__' of a type (line 413)
    getitem___67832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 23), var_67831, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 413)
    subscript_call_result_67833 = invoke(stypy.reporting.localization.Localization(__file__, 413, 23), getitem___67832, str_67830)
    
    # Applying the binary operator 'in' (line 413)
    result_contains_67834 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 12), 'in', str_67829, subscript_call_result_67833)
    
    # Applying the binary operator 'or' (line 412)
    result_or_keyword_67835 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 33), 'or', result_contains_67828, result_contains_67834)
    
    # Applying the binary operator 'and' (line 412)
    result_and_keyword_67836 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 12), 'and', result_contains_67822, result_or_keyword_67835)
    
    str_67837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 42), 'str', 'in')
    
    # Obtaining the type of the subscript
    str_67838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 58), 'str', 'intent')
    # Getting the type of 'var' (line 413)
    var_67839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 54), 'var')
    # Obtaining the member '__getitem__' of a type (line 413)
    getitem___67840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 54), var_67839, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 413)
    subscript_call_result_67841 = invoke(stypy.reporting.localization.Localization(__file__, 413, 54), getitem___67840, str_67838)
    
    # Applying the binary operator 'notin' (line 413)
    result_contains_67842 = python_operator(stypy.reporting.localization.Localization(__file__, 413, 42), 'notin', str_67837, subscript_call_result_67841)
    
    # Applying the binary operator 'and' (line 412)
    result_and_keyword_67843 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 12), 'and', result_and_keyword_67836, result_contains_67842)
    
    str_67844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 12), 'str', 'hide')
    
    # Obtaining the type of the subscript
    str_67845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 30), 'str', 'intent')
    # Getting the type of 'var' (line 414)
    var_67846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 26), 'var')
    # Obtaining the member '__getitem__' of a type (line 414)
    getitem___67847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 26), var_67846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 414)
    subscript_call_result_67848 = invoke(stypy.reporting.localization.Localization(__file__, 414, 26), getitem___67847, str_67845)
    
    # Applying the binary operator 'notin' (line 414)
    result_contains_67849 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 12), 'notin', str_67844, subscript_call_result_67848)
    
    # Applying the binary operator 'and' (line 412)
    result_and_keyword_67850 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 12), 'and', result_and_keyword_67843, result_contains_67849)
    
    str_67851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 44), 'str', 'inplace')
    
    # Obtaining the type of the subscript
    str_67852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 414, 65), 'str', 'intent')
    # Getting the type of 'var' (line 414)
    var_67853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 61), 'var')
    # Obtaining the member '__getitem__' of a type (line 414)
    getitem___67854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 61), var_67853, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 414)
    subscript_call_result_67855 = invoke(stypy.reporting.localization.Localization(__file__, 414, 61), getitem___67854, str_67852)
    
    # Applying the binary operator 'notin' (line 414)
    result_contains_67856 = python_operator(stypy.reporting.localization.Localization(__file__, 414, 44), 'notin', str_67851, subscript_call_result_67855)
    
    # Applying the binary operator 'and' (line 412)
    result_and_keyword_67857 = python_operator(stypy.reporting.localization.Localization(__file__, 412, 12), 'and', result_and_keyword_67850, result_contains_67856)
    
    # Assigning a type to the variable 'stypy_return_type' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'stypy_return_type', result_and_keyword_67857)
    
    # ################# End of 'isintent_inout(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_inout' in the type store
    # Getting the type of 'stypy_return_type' (line 411)
    stypy_return_type_67858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67858)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_inout'
    return stypy_return_type_67858

# Assigning a type to the variable 'isintent_inout' (line 411)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 0), 'isintent_inout', isintent_inout)

@norecursion
def isintent_out(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_out'
    module_type_store = module_type_store.open_function_context('isintent_out', 417, 0, False)
    
    # Passed parameters checking function
    isintent_out.stypy_localization = localization
    isintent_out.stypy_type_of_self = None
    isintent_out.stypy_type_store = module_type_store
    isintent_out.stypy_function_name = 'isintent_out'
    isintent_out.stypy_param_names_list = ['var']
    isintent_out.stypy_varargs_param_name = None
    isintent_out.stypy_kwargs_param_name = None
    isintent_out.stypy_call_defaults = defaults
    isintent_out.stypy_call_varargs = varargs
    isintent_out.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_out', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_out', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_out(...)' code ##################

    
    str_67859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 11), 'str', 'out')
    
    # Call to get(...): (line 418)
    # Processing the call arguments (line 418)
    str_67862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 28), 'str', 'intent')
    
    # Obtaining an instance of the builtin type 'list' (line 418)
    list_67863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 418)
    
    # Processing the call keyword arguments (line 418)
    kwargs_67864 = {}
    # Getting the type of 'var' (line 418)
    var_67860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 20), 'var', False)
    # Obtaining the member 'get' of a type (line 418)
    get_67861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 20), var_67860, 'get')
    # Calling get(args, kwargs) (line 418)
    get_call_result_67865 = invoke(stypy.reporting.localization.Localization(__file__, 418, 20), get_67861, *[str_67862, list_67863], **kwargs_67864)
    
    # Applying the binary operator 'in' (line 418)
    result_contains_67866 = python_operator(stypy.reporting.localization.Localization(__file__, 418, 11), 'in', str_67859, get_call_result_67865)
    
    # Assigning a type to the variable 'stypy_return_type' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'stypy_return_type', result_contains_67866)
    
    # ################# End of 'isintent_out(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_out' in the type store
    # Getting the type of 'stypy_return_type' (line 417)
    stypy_return_type_67867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67867)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_out'
    return stypy_return_type_67867

# Assigning a type to the variable 'isintent_out' (line 417)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 417, 0), 'isintent_out', isintent_out)

@norecursion
def isintent_hide(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_hide'
    module_type_store = module_type_store.open_function_context('isintent_hide', 421, 0, False)
    
    # Passed parameters checking function
    isintent_hide.stypy_localization = localization
    isintent_hide.stypy_type_of_self = None
    isintent_hide.stypy_type_store = module_type_store
    isintent_hide.stypy_function_name = 'isintent_hide'
    isintent_hide.stypy_param_names_list = ['var']
    isintent_hide.stypy_varargs_param_name = None
    isintent_hide.stypy_kwargs_param_name = None
    isintent_hide.stypy_call_defaults = defaults
    isintent_hide.stypy_call_varargs = varargs
    isintent_hide.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_hide', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_hide', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_hide(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_67868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 12), 'str', 'intent')
    # Getting the type of 'var' (line 422)
    var_67869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 24), 'var')
    # Applying the binary operator 'in' (line 422)
    result_contains_67870 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 12), 'in', str_67868, var_67869)
    
    
    # Evaluating a boolean operation
    
    str_67871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 33), 'str', 'hide')
    
    # Obtaining the type of the subscript
    str_67872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 47), 'str', 'intent')
    # Getting the type of 'var' (line 422)
    var_67873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 43), 'var')
    # Obtaining the member '__getitem__' of a type (line 422)
    getitem___67874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 43), var_67873, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 422)
    subscript_call_result_67875 = invoke(stypy.reporting.localization.Localization(__file__, 422, 43), getitem___67874, str_67872)
    
    # Applying the binary operator 'in' (line 422)
    result_contains_67876 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 33), 'in', str_67871, subscript_call_result_67875)
    
    
    # Evaluating a boolean operation
    
    str_67877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 13), 'str', 'out')
    
    # Obtaining the type of the subscript
    str_67878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 26), 'str', 'intent')
    # Getting the type of 'var' (line 423)
    var_67879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 22), 'var')
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___67880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 22), var_67879, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 423)
    subscript_call_result_67881 = invoke(stypy.reporting.localization.Localization(__file__, 423, 22), getitem___67880, str_67878)
    
    # Applying the binary operator 'in' (line 423)
    result_contains_67882 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 13), 'in', str_67877, subscript_call_result_67881)
    
    
    str_67883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 40), 'str', 'in')
    
    # Obtaining the type of the subscript
    str_67884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 56), 'str', 'intent')
    # Getting the type of 'var' (line 423)
    var_67885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 52), 'var')
    # Obtaining the member '__getitem__' of a type (line 423)
    getitem___67886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 52), var_67885, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 423)
    subscript_call_result_67887 = invoke(stypy.reporting.localization.Localization(__file__, 423, 52), getitem___67886, str_67884)
    
    # Applying the binary operator 'notin' (line 423)
    result_contains_67888 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 40), 'notin', str_67883, subscript_call_result_67887)
    
    # Applying the binary operator 'and' (line 423)
    result_and_keyword_67889 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 13), 'and', result_contains_67882, result_contains_67888)
    
    
    # Call to (...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'var' (line 424)
    var_67895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 60), 'var', False)
    # Processing the call keyword arguments (line 424)
    kwargs_67896 = {}
    
    # Call to l_or(...): (line 424)
    # Processing the call arguments (line 424)
    # Getting the type of 'isintent_inout' (line 424)
    isintent_inout_67891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 26), 'isintent_inout', False)
    # Getting the type of 'isintent_inplace' (line 424)
    isintent_inplace_67892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 42), 'isintent_inplace', False)
    # Processing the call keyword arguments (line 424)
    kwargs_67893 = {}
    # Getting the type of 'l_or' (line 424)
    l_or_67890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 21), 'l_or', False)
    # Calling l_or(args, kwargs) (line 424)
    l_or_call_result_67894 = invoke(stypy.reporting.localization.Localization(__file__, 424, 21), l_or_67890, *[isintent_inout_67891, isintent_inplace_67892], **kwargs_67893)
    
    # Calling (args, kwargs) (line 424)
    _call_result_67897 = invoke(stypy.reporting.localization.Localization(__file__, 424, 21), l_or_call_result_67894, *[var_67895], **kwargs_67896)
    
    # Applying the 'not' unary operator (line 424)
    result_not__67898 = python_operator(stypy.reporting.localization.Localization(__file__, 424, 17), 'not', _call_result_67897)
    
    # Applying the binary operator 'and' (line 423)
    result_and_keyword_67899 = python_operator(stypy.reporting.localization.Localization(__file__, 423, 13), 'and', result_and_keyword_67889, result_not__67898)
    
    # Applying the binary operator 'or' (line 422)
    result_or_keyword_67900 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 33), 'or', result_contains_67876, result_and_keyword_67899)
    
    # Applying the binary operator 'and' (line 422)
    result_and_keyword_67901 = python_operator(stypy.reporting.localization.Localization(__file__, 422, 12), 'and', result_contains_67870, result_or_keyword_67900)
    
    # Assigning a type to the variable 'stypy_return_type' (line 422)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type', result_and_keyword_67901)
    
    # ################# End of 'isintent_hide(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_hide' in the type store
    # Getting the type of 'stypy_return_type' (line 421)
    stypy_return_type_67902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67902)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_hide'
    return stypy_return_type_67902

# Assigning a type to the variable 'isintent_hide' (line 421)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 421, 0), 'isintent_hide', isintent_hide)

@norecursion
def isintent_nothide(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_nothide'
    module_type_store = module_type_store.open_function_context('isintent_nothide', 426, 0, False)
    
    # Passed parameters checking function
    isintent_nothide.stypy_localization = localization
    isintent_nothide.stypy_type_of_self = None
    isintent_nothide.stypy_type_store = module_type_store
    isintent_nothide.stypy_function_name = 'isintent_nothide'
    isintent_nothide.stypy_param_names_list = ['var']
    isintent_nothide.stypy_varargs_param_name = None
    isintent_nothide.stypy_kwargs_param_name = None
    isintent_nothide.stypy_call_defaults = defaults
    isintent_nothide.stypy_call_varargs = varargs
    isintent_nothide.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_nothide', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_nothide', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_nothide(...)' code ##################

    
    
    # Call to isintent_hide(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'var' (line 427)
    var_67904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 29), 'var', False)
    # Processing the call keyword arguments (line 427)
    kwargs_67905 = {}
    # Getting the type of 'isintent_hide' (line 427)
    isintent_hide_67903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 15), 'isintent_hide', False)
    # Calling isintent_hide(args, kwargs) (line 427)
    isintent_hide_call_result_67906 = invoke(stypy.reporting.localization.Localization(__file__, 427, 15), isintent_hide_67903, *[var_67904], **kwargs_67905)
    
    # Applying the 'not' unary operator (line 427)
    result_not__67907 = python_operator(stypy.reporting.localization.Localization(__file__, 427, 11), 'not', isintent_hide_call_result_67906)
    
    # Assigning a type to the variable 'stypy_return_type' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'stypy_return_type', result_not__67907)
    
    # ################# End of 'isintent_nothide(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_nothide' in the type store
    # Getting the type of 'stypy_return_type' (line 426)
    stypy_return_type_67908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67908)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_nothide'
    return stypy_return_type_67908

# Assigning a type to the variable 'isintent_nothide' (line 426)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 0), 'isintent_nothide', isintent_nothide)

@norecursion
def isintent_c(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_c'
    module_type_store = module_type_store.open_function_context('isintent_c', 430, 0, False)
    
    # Passed parameters checking function
    isintent_c.stypy_localization = localization
    isintent_c.stypy_type_of_self = None
    isintent_c.stypy_type_store = module_type_store
    isintent_c.stypy_function_name = 'isintent_c'
    isintent_c.stypy_param_names_list = ['var']
    isintent_c.stypy_varargs_param_name = None
    isintent_c.stypy_kwargs_param_name = None
    isintent_c.stypy_call_defaults = defaults
    isintent_c.stypy_call_varargs = varargs
    isintent_c.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_c', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_c', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_c(...)' code ##################

    
    str_67909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 11), 'str', 'c')
    
    # Call to get(...): (line 431)
    # Processing the call arguments (line 431)
    str_67912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 26), 'str', 'intent')
    
    # Obtaining an instance of the builtin type 'list' (line 431)
    list_67913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 431)
    
    # Processing the call keyword arguments (line 431)
    kwargs_67914 = {}
    # Getting the type of 'var' (line 431)
    var_67910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 18), 'var', False)
    # Obtaining the member 'get' of a type (line 431)
    get_67911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 18), var_67910, 'get')
    # Calling get(args, kwargs) (line 431)
    get_call_result_67915 = invoke(stypy.reporting.localization.Localization(__file__, 431, 18), get_67911, *[str_67912, list_67913], **kwargs_67914)
    
    # Applying the binary operator 'in' (line 431)
    result_contains_67916 = python_operator(stypy.reporting.localization.Localization(__file__, 431, 11), 'in', str_67909, get_call_result_67915)
    
    # Assigning a type to the variable 'stypy_return_type' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'stypy_return_type', result_contains_67916)
    
    # ################# End of 'isintent_c(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_c' in the type store
    # Getting the type of 'stypy_return_type' (line 430)
    stypy_return_type_67917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67917)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_c'
    return stypy_return_type_67917

# Assigning a type to the variable 'isintent_c' (line 430)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 0), 'isintent_c', isintent_c)

@norecursion
def isintent_cache(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_cache'
    module_type_store = module_type_store.open_function_context('isintent_cache', 434, 0, False)
    
    # Passed parameters checking function
    isintent_cache.stypy_localization = localization
    isintent_cache.stypy_type_of_self = None
    isintent_cache.stypy_type_store = module_type_store
    isintent_cache.stypy_function_name = 'isintent_cache'
    isintent_cache.stypy_param_names_list = ['var']
    isintent_cache.stypy_varargs_param_name = None
    isintent_cache.stypy_kwargs_param_name = None
    isintent_cache.stypy_call_defaults = defaults
    isintent_cache.stypy_call_varargs = varargs
    isintent_cache.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_cache', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_cache', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_cache(...)' code ##################

    
    str_67918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 11), 'str', 'cache')
    
    # Call to get(...): (line 435)
    # Processing the call arguments (line 435)
    str_67921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 30), 'str', 'intent')
    
    # Obtaining an instance of the builtin type 'list' (line 435)
    list_67922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 435, 40), 'list')
    # Adding type elements to the builtin type 'list' instance (line 435)
    
    # Processing the call keyword arguments (line 435)
    kwargs_67923 = {}
    # Getting the type of 'var' (line 435)
    var_67919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 22), 'var', False)
    # Obtaining the member 'get' of a type (line 435)
    get_67920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 22), var_67919, 'get')
    # Calling get(args, kwargs) (line 435)
    get_call_result_67924 = invoke(stypy.reporting.localization.Localization(__file__, 435, 22), get_67920, *[str_67921, list_67922], **kwargs_67923)
    
    # Applying the binary operator 'in' (line 435)
    result_contains_67925 = python_operator(stypy.reporting.localization.Localization(__file__, 435, 11), 'in', str_67918, get_call_result_67924)
    
    # Assigning a type to the variable 'stypy_return_type' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'stypy_return_type', result_contains_67925)
    
    # ################# End of 'isintent_cache(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_cache' in the type store
    # Getting the type of 'stypy_return_type' (line 434)
    stypy_return_type_67926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67926)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_cache'
    return stypy_return_type_67926

# Assigning a type to the variable 'isintent_cache' (line 434)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 0), 'isintent_cache', isintent_cache)

@norecursion
def isintent_copy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_copy'
    module_type_store = module_type_store.open_function_context('isintent_copy', 438, 0, False)
    
    # Passed parameters checking function
    isintent_copy.stypy_localization = localization
    isintent_copy.stypy_type_of_self = None
    isintent_copy.stypy_type_store = module_type_store
    isintent_copy.stypy_function_name = 'isintent_copy'
    isintent_copy.stypy_param_names_list = ['var']
    isintent_copy.stypy_varargs_param_name = None
    isintent_copy.stypy_kwargs_param_name = None
    isintent_copy.stypy_call_defaults = defaults
    isintent_copy.stypy_call_varargs = varargs
    isintent_copy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_copy', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_copy', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_copy(...)' code ##################

    
    str_67927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 11), 'str', 'copy')
    
    # Call to get(...): (line 439)
    # Processing the call arguments (line 439)
    str_67930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 29), 'str', 'intent')
    
    # Obtaining an instance of the builtin type 'list' (line 439)
    list_67931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 439)
    
    # Processing the call keyword arguments (line 439)
    kwargs_67932 = {}
    # Getting the type of 'var' (line 439)
    var_67928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 21), 'var', False)
    # Obtaining the member 'get' of a type (line 439)
    get_67929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 21), var_67928, 'get')
    # Calling get(args, kwargs) (line 439)
    get_call_result_67933 = invoke(stypy.reporting.localization.Localization(__file__, 439, 21), get_67929, *[str_67930, list_67931], **kwargs_67932)
    
    # Applying the binary operator 'in' (line 439)
    result_contains_67934 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 11), 'in', str_67927, get_call_result_67933)
    
    # Assigning a type to the variable 'stypy_return_type' (line 439)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'stypy_return_type', result_contains_67934)
    
    # ################# End of 'isintent_copy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_copy' in the type store
    # Getting the type of 'stypy_return_type' (line 438)
    stypy_return_type_67935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67935)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_copy'
    return stypy_return_type_67935

# Assigning a type to the variable 'isintent_copy' (line 438)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 0), 'isintent_copy', isintent_copy)

@norecursion
def isintent_overwrite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_overwrite'
    module_type_store = module_type_store.open_function_context('isintent_overwrite', 442, 0, False)
    
    # Passed parameters checking function
    isintent_overwrite.stypy_localization = localization
    isintent_overwrite.stypy_type_of_self = None
    isintent_overwrite.stypy_type_store = module_type_store
    isintent_overwrite.stypy_function_name = 'isintent_overwrite'
    isintent_overwrite.stypy_param_names_list = ['var']
    isintent_overwrite.stypy_varargs_param_name = None
    isintent_overwrite.stypy_kwargs_param_name = None
    isintent_overwrite.stypy_call_defaults = defaults
    isintent_overwrite.stypy_call_varargs = varargs
    isintent_overwrite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_overwrite', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_overwrite', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_overwrite(...)' code ##################

    
    str_67936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 11), 'str', 'overwrite')
    
    # Call to get(...): (line 443)
    # Processing the call arguments (line 443)
    str_67939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 34), 'str', 'intent')
    
    # Obtaining an instance of the builtin type 'list' (line 443)
    list_67940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 443)
    
    # Processing the call keyword arguments (line 443)
    kwargs_67941 = {}
    # Getting the type of 'var' (line 443)
    var_67937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 26), 'var', False)
    # Obtaining the member 'get' of a type (line 443)
    get_67938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 26), var_67937, 'get')
    # Calling get(args, kwargs) (line 443)
    get_call_result_67942 = invoke(stypy.reporting.localization.Localization(__file__, 443, 26), get_67938, *[str_67939, list_67940], **kwargs_67941)
    
    # Applying the binary operator 'in' (line 443)
    result_contains_67943 = python_operator(stypy.reporting.localization.Localization(__file__, 443, 11), 'in', str_67936, get_call_result_67942)
    
    # Assigning a type to the variable 'stypy_return_type' (line 443)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 443, 4), 'stypy_return_type', result_contains_67943)
    
    # ################# End of 'isintent_overwrite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_overwrite' in the type store
    # Getting the type of 'stypy_return_type' (line 442)
    stypy_return_type_67944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67944)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_overwrite'
    return stypy_return_type_67944

# Assigning a type to the variable 'isintent_overwrite' (line 442)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 0), 'isintent_overwrite', isintent_overwrite)

@norecursion
def isintent_callback(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_callback'
    module_type_store = module_type_store.open_function_context('isintent_callback', 446, 0, False)
    
    # Passed parameters checking function
    isintent_callback.stypy_localization = localization
    isintent_callback.stypy_type_of_self = None
    isintent_callback.stypy_type_store = module_type_store
    isintent_callback.stypy_function_name = 'isintent_callback'
    isintent_callback.stypy_param_names_list = ['var']
    isintent_callback.stypy_varargs_param_name = None
    isintent_callback.stypy_kwargs_param_name = None
    isintent_callback.stypy_call_defaults = defaults
    isintent_callback.stypy_call_varargs = varargs
    isintent_callback.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_callback', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_callback', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_callback(...)' code ##################

    
    str_67945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 11), 'str', 'callback')
    
    # Call to get(...): (line 447)
    # Processing the call arguments (line 447)
    str_67948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 33), 'str', 'intent')
    
    # Obtaining an instance of the builtin type 'list' (line 447)
    list_67949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 447)
    
    # Processing the call keyword arguments (line 447)
    kwargs_67950 = {}
    # Getting the type of 'var' (line 447)
    var_67946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 25), 'var', False)
    # Obtaining the member 'get' of a type (line 447)
    get_67947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 25), var_67946, 'get')
    # Calling get(args, kwargs) (line 447)
    get_call_result_67951 = invoke(stypy.reporting.localization.Localization(__file__, 447, 25), get_67947, *[str_67948, list_67949], **kwargs_67950)
    
    # Applying the binary operator 'in' (line 447)
    result_contains_67952 = python_operator(stypy.reporting.localization.Localization(__file__, 447, 11), 'in', str_67945, get_call_result_67951)
    
    # Assigning a type to the variable 'stypy_return_type' (line 447)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 447, 4), 'stypy_return_type', result_contains_67952)
    
    # ################# End of 'isintent_callback(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_callback' in the type store
    # Getting the type of 'stypy_return_type' (line 446)
    stypy_return_type_67953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67953)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_callback'
    return stypy_return_type_67953

# Assigning a type to the variable 'isintent_callback' (line 446)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 0), 'isintent_callback', isintent_callback)

@norecursion
def isintent_inplace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_inplace'
    module_type_store = module_type_store.open_function_context('isintent_inplace', 450, 0, False)
    
    # Passed parameters checking function
    isintent_inplace.stypy_localization = localization
    isintent_inplace.stypy_type_of_self = None
    isintent_inplace.stypy_type_store = module_type_store
    isintent_inplace.stypy_function_name = 'isintent_inplace'
    isintent_inplace.stypy_param_names_list = ['var']
    isintent_inplace.stypy_varargs_param_name = None
    isintent_inplace.stypy_kwargs_param_name = None
    isintent_inplace.stypy_call_defaults = defaults
    isintent_inplace.stypy_call_varargs = varargs
    isintent_inplace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_inplace', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_inplace', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_inplace(...)' code ##################

    
    str_67954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 11), 'str', 'inplace')
    
    # Call to get(...): (line 451)
    # Processing the call arguments (line 451)
    str_67957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 32), 'str', 'intent')
    
    # Obtaining an instance of the builtin type 'list' (line 451)
    list_67958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 451)
    
    # Processing the call keyword arguments (line 451)
    kwargs_67959 = {}
    # Getting the type of 'var' (line 451)
    var_67955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 24), 'var', False)
    # Obtaining the member 'get' of a type (line 451)
    get_67956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 24), var_67955, 'get')
    # Calling get(args, kwargs) (line 451)
    get_call_result_67960 = invoke(stypy.reporting.localization.Localization(__file__, 451, 24), get_67956, *[str_67957, list_67958], **kwargs_67959)
    
    # Applying the binary operator 'in' (line 451)
    result_contains_67961 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 11), 'in', str_67954, get_call_result_67960)
    
    # Assigning a type to the variable 'stypy_return_type' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), 'stypy_return_type', result_contains_67961)
    
    # ################# End of 'isintent_inplace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_inplace' in the type store
    # Getting the type of 'stypy_return_type' (line 450)
    stypy_return_type_67962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67962)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_inplace'
    return stypy_return_type_67962

# Assigning a type to the variable 'isintent_inplace' (line 450)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 0), 'isintent_inplace', isintent_inplace)

@norecursion
def isintent_aux(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_aux'
    module_type_store = module_type_store.open_function_context('isintent_aux', 454, 0, False)
    
    # Passed parameters checking function
    isintent_aux.stypy_localization = localization
    isintent_aux.stypy_type_of_self = None
    isintent_aux.stypy_type_store = module_type_store
    isintent_aux.stypy_function_name = 'isintent_aux'
    isintent_aux.stypy_param_names_list = ['var']
    isintent_aux.stypy_varargs_param_name = None
    isintent_aux.stypy_kwargs_param_name = None
    isintent_aux.stypy_call_defaults = defaults
    isintent_aux.stypy_call_varargs = varargs
    isintent_aux.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_aux', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_aux', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_aux(...)' code ##################

    
    str_67963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 11), 'str', 'aux')
    
    # Call to get(...): (line 455)
    # Processing the call arguments (line 455)
    str_67966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 28), 'str', 'intent')
    
    # Obtaining an instance of the builtin type 'list' (line 455)
    list_67967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 455)
    
    # Processing the call keyword arguments (line 455)
    kwargs_67968 = {}
    # Getting the type of 'var' (line 455)
    var_67964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 20), 'var', False)
    # Obtaining the member 'get' of a type (line 455)
    get_67965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 20), var_67964, 'get')
    # Calling get(args, kwargs) (line 455)
    get_call_result_67969 = invoke(stypy.reporting.localization.Localization(__file__, 455, 20), get_67965, *[str_67966, list_67967], **kwargs_67968)
    
    # Applying the binary operator 'in' (line 455)
    result_contains_67970 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 11), 'in', str_67963, get_call_result_67969)
    
    # Assigning a type to the variable 'stypy_return_type' (line 455)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'stypy_return_type', result_contains_67970)
    
    # ################# End of 'isintent_aux(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_aux' in the type store
    # Getting the type of 'stypy_return_type' (line 454)
    stypy_return_type_67971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67971)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_aux'
    return stypy_return_type_67971

# Assigning a type to the variable 'isintent_aux' (line 454)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 0), 'isintent_aux', isintent_aux)

@norecursion
def isintent_aligned4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_aligned4'
    module_type_store = module_type_store.open_function_context('isintent_aligned4', 458, 0, False)
    
    # Passed parameters checking function
    isintent_aligned4.stypy_localization = localization
    isintent_aligned4.stypy_type_of_self = None
    isintent_aligned4.stypy_type_store = module_type_store
    isintent_aligned4.stypy_function_name = 'isintent_aligned4'
    isintent_aligned4.stypy_param_names_list = ['var']
    isintent_aligned4.stypy_varargs_param_name = None
    isintent_aligned4.stypy_kwargs_param_name = None
    isintent_aligned4.stypy_call_defaults = defaults
    isintent_aligned4.stypy_call_varargs = varargs
    isintent_aligned4.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_aligned4', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_aligned4', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_aligned4(...)' code ##################

    
    str_67972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 11), 'str', 'aligned4')
    
    # Call to get(...): (line 459)
    # Processing the call arguments (line 459)
    str_67975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 33), 'str', 'intent')
    
    # Obtaining an instance of the builtin type 'list' (line 459)
    list_67976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 459)
    
    # Processing the call keyword arguments (line 459)
    kwargs_67977 = {}
    # Getting the type of 'var' (line 459)
    var_67973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 25), 'var', False)
    # Obtaining the member 'get' of a type (line 459)
    get_67974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 25), var_67973, 'get')
    # Calling get(args, kwargs) (line 459)
    get_call_result_67978 = invoke(stypy.reporting.localization.Localization(__file__, 459, 25), get_67974, *[str_67975, list_67976], **kwargs_67977)
    
    # Applying the binary operator 'in' (line 459)
    result_contains_67979 = python_operator(stypy.reporting.localization.Localization(__file__, 459, 11), 'in', str_67972, get_call_result_67978)
    
    # Assigning a type to the variable 'stypy_return_type' (line 459)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 4), 'stypy_return_type', result_contains_67979)
    
    # ################# End of 'isintent_aligned4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_aligned4' in the type store
    # Getting the type of 'stypy_return_type' (line 458)
    stypy_return_type_67980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67980)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_aligned4'
    return stypy_return_type_67980

# Assigning a type to the variable 'isintent_aligned4' (line 458)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 0), 'isintent_aligned4', isintent_aligned4)

@norecursion
def isintent_aligned8(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_aligned8'
    module_type_store = module_type_store.open_function_context('isintent_aligned8', 462, 0, False)
    
    # Passed parameters checking function
    isintent_aligned8.stypy_localization = localization
    isintent_aligned8.stypy_type_of_self = None
    isintent_aligned8.stypy_type_store = module_type_store
    isintent_aligned8.stypy_function_name = 'isintent_aligned8'
    isintent_aligned8.stypy_param_names_list = ['var']
    isintent_aligned8.stypy_varargs_param_name = None
    isintent_aligned8.stypy_kwargs_param_name = None
    isintent_aligned8.stypy_call_defaults = defaults
    isintent_aligned8.stypy_call_varargs = varargs
    isintent_aligned8.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_aligned8', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_aligned8', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_aligned8(...)' code ##################

    
    str_67981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 11), 'str', 'aligned8')
    
    # Call to get(...): (line 463)
    # Processing the call arguments (line 463)
    str_67984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 33), 'str', 'intent')
    
    # Obtaining an instance of the builtin type 'list' (line 463)
    list_67985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 463)
    
    # Processing the call keyword arguments (line 463)
    kwargs_67986 = {}
    # Getting the type of 'var' (line 463)
    var_67982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 25), 'var', False)
    # Obtaining the member 'get' of a type (line 463)
    get_67983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 25), var_67982, 'get')
    # Calling get(args, kwargs) (line 463)
    get_call_result_67987 = invoke(stypy.reporting.localization.Localization(__file__, 463, 25), get_67983, *[str_67984, list_67985], **kwargs_67986)
    
    # Applying the binary operator 'in' (line 463)
    result_contains_67988 = python_operator(stypy.reporting.localization.Localization(__file__, 463, 11), 'in', str_67981, get_call_result_67987)
    
    # Assigning a type to the variable 'stypy_return_type' (line 463)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 463, 4), 'stypy_return_type', result_contains_67988)
    
    # ################# End of 'isintent_aligned8(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_aligned8' in the type store
    # Getting the type of 'stypy_return_type' (line 462)
    stypy_return_type_67989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67989)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_aligned8'
    return stypy_return_type_67989

# Assigning a type to the variable 'isintent_aligned8' (line 462)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 0), 'isintent_aligned8', isintent_aligned8)

@norecursion
def isintent_aligned16(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isintent_aligned16'
    module_type_store = module_type_store.open_function_context('isintent_aligned16', 466, 0, False)
    
    # Passed parameters checking function
    isintent_aligned16.stypy_localization = localization
    isintent_aligned16.stypy_type_of_self = None
    isintent_aligned16.stypy_type_store = module_type_store
    isintent_aligned16.stypy_function_name = 'isintent_aligned16'
    isintent_aligned16.stypy_param_names_list = ['var']
    isintent_aligned16.stypy_varargs_param_name = None
    isintent_aligned16.stypy_kwargs_param_name = None
    isintent_aligned16.stypy_call_defaults = defaults
    isintent_aligned16.stypy_call_varargs = varargs
    isintent_aligned16.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isintent_aligned16', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isintent_aligned16', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isintent_aligned16(...)' code ##################

    
    str_67990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 11), 'str', 'aligned16')
    
    # Call to get(...): (line 467)
    # Processing the call arguments (line 467)
    str_67993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 34), 'str', 'intent')
    
    # Obtaining an instance of the builtin type 'list' (line 467)
    list_67994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 467)
    
    # Processing the call keyword arguments (line 467)
    kwargs_67995 = {}
    # Getting the type of 'var' (line 467)
    var_67991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 26), 'var', False)
    # Obtaining the member 'get' of a type (line 467)
    get_67992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 26), var_67991, 'get')
    # Calling get(args, kwargs) (line 467)
    get_call_result_67996 = invoke(stypy.reporting.localization.Localization(__file__, 467, 26), get_67992, *[str_67993, list_67994], **kwargs_67995)
    
    # Applying the binary operator 'in' (line 467)
    result_contains_67997 = python_operator(stypy.reporting.localization.Localization(__file__, 467, 11), 'in', str_67990, get_call_result_67996)
    
    # Assigning a type to the variable 'stypy_return_type' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'stypy_return_type', result_contains_67997)
    
    # ################# End of 'isintent_aligned16(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isintent_aligned16' in the type store
    # Getting the type of 'stypy_return_type' (line 466)
    stypy_return_type_67998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_67998)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isintent_aligned16'
    return stypy_return_type_67998

# Assigning a type to the variable 'isintent_aligned16' (line 466)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 0), 'isintent_aligned16', isintent_aligned16)

# Assigning a Dict to a Name (line 469):

# Assigning a Dict to a Name (line 469):

# Obtaining an instance of the builtin type 'dict' (line 469)
dict_67999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 469)
# Adding element type (key, value) (line 469)
# Getting the type of 'isintent_in' (line 469)
isintent_in_68000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 17), 'isintent_in')
str_68001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 30), 'str', 'INTENT_IN')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 16), dict_67999, (isintent_in_68000, str_68001))
# Adding element type (key, value) (line 469)
# Getting the type of 'isintent_inout' (line 469)
isintent_inout_68002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 43), 'isintent_inout')
str_68003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 59), 'str', 'INTENT_INOUT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 16), dict_67999, (isintent_inout_68002, str_68003))
# Adding element type (key, value) (line 469)
# Getting the type of 'isintent_out' (line 470)
isintent_out_68004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 17), 'isintent_out')
str_68005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 31), 'str', 'INTENT_OUT')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 16), dict_67999, (isintent_out_68004, str_68005))
# Adding element type (key, value) (line 469)
# Getting the type of 'isintent_hide' (line 470)
isintent_hide_68006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 45), 'isintent_hide')
str_68007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 60), 'str', 'INTENT_HIDE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 16), dict_67999, (isintent_hide_68006, str_68007))
# Adding element type (key, value) (line 469)
# Getting the type of 'isintent_cache' (line 471)
isintent_cache_68008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 17), 'isintent_cache')
str_68009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 471, 33), 'str', 'INTENT_CACHE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 16), dict_67999, (isintent_cache_68008, str_68009))
# Adding element type (key, value) (line 469)
# Getting the type of 'isintent_c' (line 472)
isintent_c_68010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 17), 'isintent_c')
str_68011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 29), 'str', 'INTENT_C')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 16), dict_67999, (isintent_c_68010, str_68011))
# Adding element type (key, value) (line 469)
# Getting the type of 'isoptional' (line 472)
isoptional_68012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 41), 'isoptional')
str_68013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 53), 'str', 'OPTIONAL')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 16), dict_67999, (isoptional_68012, str_68013))
# Adding element type (key, value) (line 469)
# Getting the type of 'isintent_inplace' (line 473)
isintent_inplace_68014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 473, 17), 'isintent_inplace')
str_68015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, 35), 'str', 'INTENT_INPLACE')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 16), dict_67999, (isintent_inplace_68014, str_68015))
# Adding element type (key, value) (line 469)
# Getting the type of 'isintent_aligned4' (line 474)
isintent_aligned4_68016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 17), 'isintent_aligned4')
str_68017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 36), 'str', 'INTENT_ALIGNED4')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 16), dict_67999, (isintent_aligned4_68016, str_68017))
# Adding element type (key, value) (line 469)
# Getting the type of 'isintent_aligned8' (line 475)
isintent_aligned8_68018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 17), 'isintent_aligned8')
str_68019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 36), 'str', 'INTENT_ALIGNED8')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 16), dict_67999, (isintent_aligned8_68018, str_68019))
# Adding element type (key, value) (line 469)
# Getting the type of 'isintent_aligned16' (line 476)
isintent_aligned16_68020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 17), 'isintent_aligned16')
str_68021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 37), 'str', 'INTENT_ALIGNED16')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 16), dict_67999, (isintent_aligned16_68020, str_68021))

# Assigning a type to the variable 'isintent_dict' (line 469)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 0), 'isintent_dict', dict_67999)

@norecursion
def isprivate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isprivate'
    module_type_store = module_type_store.open_function_context('isprivate', 480, 0, False)
    
    # Passed parameters checking function
    isprivate.stypy_localization = localization
    isprivate.stypy_type_of_self = None
    isprivate.stypy_type_store = module_type_store
    isprivate.stypy_function_name = 'isprivate'
    isprivate.stypy_param_names_list = ['var']
    isprivate.stypy_varargs_param_name = None
    isprivate.stypy_kwargs_param_name = None
    isprivate.stypy_call_defaults = defaults
    isprivate.stypy_call_varargs = varargs
    isprivate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isprivate', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isprivate', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isprivate(...)' code ##################

    
    # Evaluating a boolean operation
    
    str_68022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 11), 'str', 'attrspec')
    # Getting the type of 'var' (line 481)
    var_68023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 25), 'var')
    # Applying the binary operator 'in' (line 481)
    result_contains_68024 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 11), 'in', str_68022, var_68023)
    
    
    str_68025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 33), 'str', 'private')
    
    # Obtaining the type of the subscript
    str_68026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 481, 50), 'str', 'attrspec')
    # Getting the type of 'var' (line 481)
    var_68027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 46), 'var')
    # Obtaining the member '__getitem__' of a type (line 481)
    getitem___68028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 46), var_68027, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 481)
    subscript_call_result_68029 = invoke(stypy.reporting.localization.Localization(__file__, 481, 46), getitem___68028, str_68026)
    
    # Applying the binary operator 'in' (line 481)
    result_contains_68030 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 33), 'in', str_68025, subscript_call_result_68029)
    
    # Applying the binary operator 'and' (line 481)
    result_and_keyword_68031 = python_operator(stypy.reporting.localization.Localization(__file__, 481, 11), 'and', result_contains_68024, result_contains_68030)
    
    # Assigning a type to the variable 'stypy_return_type' (line 481)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 4), 'stypy_return_type', result_and_keyword_68031)
    
    # ################# End of 'isprivate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isprivate' in the type store
    # Getting the type of 'stypy_return_type' (line 480)
    stypy_return_type_68032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68032)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isprivate'
    return stypy_return_type_68032

# Assigning a type to the variable 'isprivate' (line 480)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 0), 'isprivate', isprivate)

@norecursion
def hasinitvalue(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hasinitvalue'
    module_type_store = module_type_store.open_function_context('hasinitvalue', 484, 0, False)
    
    # Passed parameters checking function
    hasinitvalue.stypy_localization = localization
    hasinitvalue.stypy_type_of_self = None
    hasinitvalue.stypy_type_store = module_type_store
    hasinitvalue.stypy_function_name = 'hasinitvalue'
    hasinitvalue.stypy_param_names_list = ['var']
    hasinitvalue.stypy_varargs_param_name = None
    hasinitvalue.stypy_kwargs_param_name = None
    hasinitvalue.stypy_call_defaults = defaults
    hasinitvalue.stypy_call_varargs = varargs
    hasinitvalue.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hasinitvalue', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hasinitvalue', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hasinitvalue(...)' code ##################

    
    str_68033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 485, 11), 'str', '=')
    # Getting the type of 'var' (line 485)
    var_68034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 18), 'var')
    # Applying the binary operator 'in' (line 485)
    result_contains_68035 = python_operator(stypy.reporting.localization.Localization(__file__, 485, 11), 'in', str_68033, var_68034)
    
    # Assigning a type to the variable 'stypy_return_type' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'stypy_return_type', result_contains_68035)
    
    # ################# End of 'hasinitvalue(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hasinitvalue' in the type store
    # Getting the type of 'stypy_return_type' (line 484)
    stypy_return_type_68036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68036)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hasinitvalue'
    return stypy_return_type_68036

# Assigning a type to the variable 'hasinitvalue' (line 484)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 484, 0), 'hasinitvalue', hasinitvalue)

@norecursion
def hasinitvalueasstring(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hasinitvalueasstring'
    module_type_store = module_type_store.open_function_context('hasinitvalueasstring', 488, 0, False)
    
    # Passed parameters checking function
    hasinitvalueasstring.stypy_localization = localization
    hasinitvalueasstring.stypy_type_of_self = None
    hasinitvalueasstring.stypy_type_store = module_type_store
    hasinitvalueasstring.stypy_function_name = 'hasinitvalueasstring'
    hasinitvalueasstring.stypy_param_names_list = ['var']
    hasinitvalueasstring.stypy_varargs_param_name = None
    hasinitvalueasstring.stypy_kwargs_param_name = None
    hasinitvalueasstring.stypy_call_defaults = defaults
    hasinitvalueasstring.stypy_call_varargs = varargs
    hasinitvalueasstring.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hasinitvalueasstring', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hasinitvalueasstring', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hasinitvalueasstring(...)' code ##################

    
    
    
    # Call to hasinitvalue(...): (line 489)
    # Processing the call arguments (line 489)
    # Getting the type of 'var' (line 489)
    var_68038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 24), 'var', False)
    # Processing the call keyword arguments (line 489)
    kwargs_68039 = {}
    # Getting the type of 'hasinitvalue' (line 489)
    hasinitvalue_68037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 11), 'hasinitvalue', False)
    # Calling hasinitvalue(args, kwargs) (line 489)
    hasinitvalue_call_result_68040 = invoke(stypy.reporting.localization.Localization(__file__, 489, 11), hasinitvalue_68037, *[var_68038], **kwargs_68039)
    
    # Applying the 'not' unary operator (line 489)
    result_not__68041 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 7), 'not', hasinitvalue_call_result_68040)
    
    # Testing the type of an if condition (line 489)
    if_condition_68042 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 489, 4), result_not__68041)
    # Assigning a type to the variable 'if_condition_68042' (line 489)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'if_condition_68042', if_condition_68042)
    # SSA begins for if statement (line 489)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_68043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 490)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'stypy_return_type', int_68043)
    # SSA join for if statement (line 489)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    int_68044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 20), 'int')
    
    # Obtaining the type of the subscript
    str_68045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 15), 'str', '=')
    # Getting the type of 'var' (line 491)
    var_68046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 11), 'var')
    # Obtaining the member '__getitem__' of a type (line 491)
    getitem___68047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 11), var_68046, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 491)
    subscript_call_result_68048 = invoke(stypy.reporting.localization.Localization(__file__, 491, 11), getitem___68047, str_68045)
    
    # Obtaining the member '__getitem__' of a type (line 491)
    getitem___68049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 11), subscript_call_result_68048, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 491)
    subscript_call_result_68050 = invoke(stypy.reporting.localization.Localization(__file__, 491, 11), getitem___68049, int_68044)
    
    
    # Obtaining an instance of the builtin type 'list' (line 491)
    list_68051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 491)
    # Adding element type (line 491)
    str_68052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 27), 'str', '"')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 26), list_68051, str_68052)
    # Adding element type (line 491)
    str_68053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 32), 'str', "'")
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 26), list_68051, str_68053)
    
    # Applying the binary operator 'in' (line 491)
    result_contains_68054 = python_operator(stypy.reporting.localization.Localization(__file__, 491, 11), 'in', subscript_call_result_68050, list_68051)
    
    # Assigning a type to the variable 'stypy_return_type' (line 491)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 4), 'stypy_return_type', result_contains_68054)
    
    # ################# End of 'hasinitvalueasstring(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hasinitvalueasstring' in the type store
    # Getting the type of 'stypy_return_type' (line 488)
    stypy_return_type_68055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68055)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hasinitvalueasstring'
    return stypy_return_type_68055

# Assigning a type to the variable 'hasinitvalueasstring' (line 488)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 0), 'hasinitvalueasstring', hasinitvalueasstring)

@norecursion
def hasnote(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hasnote'
    module_type_store = module_type_store.open_function_context('hasnote', 494, 0, False)
    
    # Passed parameters checking function
    hasnote.stypy_localization = localization
    hasnote.stypy_type_of_self = None
    hasnote.stypy_type_store = module_type_store
    hasnote.stypy_function_name = 'hasnote'
    hasnote.stypy_param_names_list = ['var']
    hasnote.stypy_varargs_param_name = None
    hasnote.stypy_kwargs_param_name = None
    hasnote.stypy_call_defaults = defaults
    hasnote.stypy_call_varargs = varargs
    hasnote.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hasnote', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hasnote', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hasnote(...)' code ##################

    
    str_68056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 11), 'str', 'note')
    # Getting the type of 'var' (line 495)
    var_68057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 495, 21), 'var')
    # Applying the binary operator 'in' (line 495)
    result_contains_68058 = python_operator(stypy.reporting.localization.Localization(__file__, 495, 11), 'in', str_68056, var_68057)
    
    # Assigning a type to the variable 'stypy_return_type' (line 495)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 4), 'stypy_return_type', result_contains_68058)
    
    # ################# End of 'hasnote(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hasnote' in the type store
    # Getting the type of 'stypy_return_type' (line 494)
    stypy_return_type_68059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68059)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hasnote'
    return stypy_return_type_68059

# Assigning a type to the variable 'hasnote' (line 494)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 0), 'hasnote', hasnote)

@norecursion
def hasresultnote(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hasresultnote'
    module_type_store = module_type_store.open_function_context('hasresultnote', 498, 0, False)
    
    # Passed parameters checking function
    hasresultnote.stypy_localization = localization
    hasresultnote.stypy_type_of_self = None
    hasresultnote.stypy_type_store = module_type_store
    hasresultnote.stypy_function_name = 'hasresultnote'
    hasresultnote.stypy_param_names_list = ['rout']
    hasresultnote.stypy_varargs_param_name = None
    hasresultnote.stypy_kwargs_param_name = None
    hasresultnote.stypy_call_defaults = defaults
    hasresultnote.stypy_call_varargs = varargs
    hasresultnote.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hasresultnote', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hasresultnote', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hasresultnote(...)' code ##################

    
    
    
    # Call to isfunction(...): (line 499)
    # Processing the call arguments (line 499)
    # Getting the type of 'rout' (line 499)
    rout_68061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 22), 'rout', False)
    # Processing the call keyword arguments (line 499)
    kwargs_68062 = {}
    # Getting the type of 'isfunction' (line 499)
    isfunction_68060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 11), 'isfunction', False)
    # Calling isfunction(args, kwargs) (line 499)
    isfunction_call_result_68063 = invoke(stypy.reporting.localization.Localization(__file__, 499, 11), isfunction_68060, *[rout_68061], **kwargs_68062)
    
    # Applying the 'not' unary operator (line 499)
    result_not__68064 = python_operator(stypy.reporting.localization.Localization(__file__, 499, 7), 'not', isfunction_call_result_68063)
    
    # Testing the type of an if condition (line 499)
    if_condition_68065 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 499, 4), result_not__68064)
    # Assigning a type to the variable 'if_condition_68065' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'if_condition_68065', if_condition_68065)
    # SSA begins for if statement (line 499)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_68066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 500)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'stypy_return_type', int_68066)
    # SSA join for if statement (line 499)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_68067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 7), 'str', 'result')
    # Getting the type of 'rout' (line 501)
    rout_68068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 19), 'rout')
    # Applying the binary operator 'in' (line 501)
    result_contains_68069 = python_operator(stypy.reporting.localization.Localization(__file__, 501, 7), 'in', str_68067, rout_68068)
    
    # Testing the type of an if condition (line 501)
    if_condition_68070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 501, 4), result_contains_68069)
    # Assigning a type to the variable 'if_condition_68070' (line 501)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'if_condition_68070', if_condition_68070)
    # SSA begins for if statement (line 501)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 502):
    
    # Assigning a Subscript to a Name (line 502):
    
    # Obtaining the type of the subscript
    str_68071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 17), 'str', 'result')
    # Getting the type of 'rout' (line 502)
    rout_68072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 502)
    getitem___68073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 12), rout_68072, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 502)
    subscript_call_result_68074 = invoke(stypy.reporting.localization.Localization(__file__, 502, 12), getitem___68073, str_68071)
    
    # Assigning a type to the variable 'a' (line 502)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'a', subscript_call_result_68074)
    # SSA branch for the else part of an if statement (line 501)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 504):
    
    # Assigning a Subscript to a Name (line 504):
    
    # Obtaining the type of the subscript
    str_68075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 17), 'str', 'name')
    # Getting the type of 'rout' (line 504)
    rout_68076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 504)
    getitem___68077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 12), rout_68076, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 504)
    subscript_call_result_68078 = invoke(stypy.reporting.localization.Localization(__file__, 504, 12), getitem___68077, str_68075)
    
    # Assigning a type to the variable 'a' (line 504)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'a', subscript_call_result_68078)
    # SSA join for if statement (line 501)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'a' (line 505)
    a_68079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 7), 'a')
    
    # Obtaining the type of the subscript
    str_68080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 17), 'str', 'vars')
    # Getting the type of 'rout' (line 505)
    rout_68081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 505)
    getitem___68082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 12), rout_68081, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 505)
    subscript_call_result_68083 = invoke(stypy.reporting.localization.Localization(__file__, 505, 12), getitem___68082, str_68080)
    
    # Applying the binary operator 'in' (line 505)
    result_contains_68084 = python_operator(stypy.reporting.localization.Localization(__file__, 505, 7), 'in', a_68079, subscript_call_result_68083)
    
    # Testing the type of an if condition (line 505)
    if_condition_68085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 505, 4), result_contains_68084)
    # Assigning a type to the variable 'if_condition_68085' (line 505)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 505, 4), 'if_condition_68085', if_condition_68085)
    # SSA begins for if statement (line 505)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to hasnote(...): (line 506)
    # Processing the call arguments (line 506)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 506)
    a_68087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 36), 'a', False)
    
    # Obtaining the type of the subscript
    str_68088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 28), 'str', 'vars')
    # Getting the type of 'rout' (line 506)
    rout_68089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 23), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 506)
    getitem___68090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 23), rout_68089, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 506)
    subscript_call_result_68091 = invoke(stypy.reporting.localization.Localization(__file__, 506, 23), getitem___68090, str_68088)
    
    # Obtaining the member '__getitem__' of a type (line 506)
    getitem___68092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 23), subscript_call_result_68091, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 506)
    subscript_call_result_68093 = invoke(stypy.reporting.localization.Localization(__file__, 506, 23), getitem___68092, a_68087)
    
    # Processing the call keyword arguments (line 506)
    kwargs_68094 = {}
    # Getting the type of 'hasnote' (line 506)
    hasnote_68086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 15), 'hasnote', False)
    # Calling hasnote(args, kwargs) (line 506)
    hasnote_call_result_68095 = invoke(stypy.reporting.localization.Localization(__file__, 506, 15), hasnote_68086, *[subscript_call_result_68093], **kwargs_68094)
    
    # Assigning a type to the variable 'stypy_return_type' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 8), 'stypy_return_type', hasnote_call_result_68095)
    # SSA join for if statement (line 505)
    module_type_store = module_type_store.join_ssa_context()
    
    int_68096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 507)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'stypy_return_type', int_68096)
    
    # ################# End of 'hasresultnote(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hasresultnote' in the type store
    # Getting the type of 'stypy_return_type' (line 498)
    stypy_return_type_68097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68097)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hasresultnote'
    return stypy_return_type_68097

# Assigning a type to the variable 'hasresultnote' (line 498)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 0), 'hasresultnote', hasresultnote)

@norecursion
def hascommon(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hascommon'
    module_type_store = module_type_store.open_function_context('hascommon', 510, 0, False)
    
    # Passed parameters checking function
    hascommon.stypy_localization = localization
    hascommon.stypy_type_of_self = None
    hascommon.stypy_type_store = module_type_store
    hascommon.stypy_function_name = 'hascommon'
    hascommon.stypy_param_names_list = ['rout']
    hascommon.stypy_varargs_param_name = None
    hascommon.stypy_kwargs_param_name = None
    hascommon.stypy_call_defaults = defaults
    hascommon.stypy_call_varargs = varargs
    hascommon.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hascommon', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hascommon', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hascommon(...)' code ##################

    
    str_68098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 11), 'str', 'common')
    # Getting the type of 'rout' (line 511)
    rout_68099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 23), 'rout')
    # Applying the binary operator 'in' (line 511)
    result_contains_68100 = python_operator(stypy.reporting.localization.Localization(__file__, 511, 11), 'in', str_68098, rout_68099)
    
    # Assigning a type to the variable 'stypy_return_type' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'stypy_return_type', result_contains_68100)
    
    # ################# End of 'hascommon(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hascommon' in the type store
    # Getting the type of 'stypy_return_type' (line 510)
    stypy_return_type_68101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68101)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hascommon'
    return stypy_return_type_68101

# Assigning a type to the variable 'hascommon' (line 510)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 0), 'hascommon', hascommon)

@norecursion
def containscommon(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'containscommon'
    module_type_store = module_type_store.open_function_context('containscommon', 514, 0, False)
    
    # Passed parameters checking function
    containscommon.stypy_localization = localization
    containscommon.stypy_type_of_self = None
    containscommon.stypy_type_store = module_type_store
    containscommon.stypy_function_name = 'containscommon'
    containscommon.stypy_param_names_list = ['rout']
    containscommon.stypy_varargs_param_name = None
    containscommon.stypy_kwargs_param_name = None
    containscommon.stypy_call_defaults = defaults
    containscommon.stypy_call_varargs = varargs
    containscommon.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'containscommon', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'containscommon', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'containscommon(...)' code ##################

    
    
    # Call to hascommon(...): (line 515)
    # Processing the call arguments (line 515)
    # Getting the type of 'rout' (line 515)
    rout_68103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 17), 'rout', False)
    # Processing the call keyword arguments (line 515)
    kwargs_68104 = {}
    # Getting the type of 'hascommon' (line 515)
    hascommon_68102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 7), 'hascommon', False)
    # Calling hascommon(args, kwargs) (line 515)
    hascommon_call_result_68105 = invoke(stypy.reporting.localization.Localization(__file__, 515, 7), hascommon_68102, *[rout_68103], **kwargs_68104)
    
    # Testing the type of an if condition (line 515)
    if_condition_68106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 515, 4), hascommon_call_result_68105)
    # Assigning a type to the variable 'if_condition_68106' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'if_condition_68106', if_condition_68106)
    # SSA begins for if statement (line 515)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_68107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'stypy_return_type', int_68107)
    # SSA join for if statement (line 515)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to hasbody(...): (line 517)
    # Processing the call arguments (line 517)
    # Getting the type of 'rout' (line 517)
    rout_68109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 15), 'rout', False)
    # Processing the call keyword arguments (line 517)
    kwargs_68110 = {}
    # Getting the type of 'hasbody' (line 517)
    hasbody_68108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 7), 'hasbody', False)
    # Calling hasbody(args, kwargs) (line 517)
    hasbody_call_result_68111 = invoke(stypy.reporting.localization.Localization(__file__, 517, 7), hasbody_68108, *[rout_68109], **kwargs_68110)
    
    # Testing the type of an if condition (line 517)
    if_condition_68112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 4), hasbody_call_result_68111)
    # Assigning a type to the variable 'if_condition_68112' (line 517)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 4), 'if_condition_68112', if_condition_68112)
    # SSA begins for if statement (line 517)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    str_68113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 22), 'str', 'body')
    # Getting the type of 'rout' (line 518)
    rout_68114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 17), 'rout')
    # Obtaining the member '__getitem__' of a type (line 518)
    getitem___68115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 17), rout_68114, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 518)
    subscript_call_result_68116 = invoke(stypy.reporting.localization.Localization(__file__, 518, 17), getitem___68115, str_68113)
    
    # Testing the type of a for loop iterable (line 518)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 518, 8), subscript_call_result_68116)
    # Getting the type of the for loop variable (line 518)
    for_loop_var_68117 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 518, 8), subscript_call_result_68116)
    # Assigning a type to the variable 'b' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 8), 'b', for_loop_var_68117)
    # SSA begins for a for statement (line 518)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to containscommon(...): (line 519)
    # Processing the call arguments (line 519)
    # Getting the type of 'b' (line 519)
    b_68119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 30), 'b', False)
    # Processing the call keyword arguments (line 519)
    kwargs_68120 = {}
    # Getting the type of 'containscommon' (line 519)
    containscommon_68118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 15), 'containscommon', False)
    # Calling containscommon(args, kwargs) (line 519)
    containscommon_call_result_68121 = invoke(stypy.reporting.localization.Localization(__file__, 519, 15), containscommon_68118, *[b_68119], **kwargs_68120)
    
    # Testing the type of an if condition (line 519)
    if_condition_68122 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 519, 12), containscommon_call_result_68121)
    # Assigning a type to the variable 'if_condition_68122' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 12), 'if_condition_68122', if_condition_68122)
    # SSA begins for if statement (line 519)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_68123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 23), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 520)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 16), 'stypy_return_type', int_68123)
    # SSA join for if statement (line 519)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 517)
    module_type_store = module_type_store.join_ssa_context()
    
    int_68124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'stypy_return_type', int_68124)
    
    # ################# End of 'containscommon(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'containscommon' in the type store
    # Getting the type of 'stypy_return_type' (line 514)
    stypy_return_type_68125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68125)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'containscommon'
    return stypy_return_type_68125

# Assigning a type to the variable 'containscommon' (line 514)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 0), 'containscommon', containscommon)

@norecursion
def containsmodule(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'containsmodule'
    module_type_store = module_type_store.open_function_context('containsmodule', 524, 0, False)
    
    # Passed parameters checking function
    containsmodule.stypy_localization = localization
    containsmodule.stypy_type_of_self = None
    containsmodule.stypy_type_store = module_type_store
    containsmodule.stypy_function_name = 'containsmodule'
    containsmodule.stypy_param_names_list = ['block']
    containsmodule.stypy_varargs_param_name = None
    containsmodule.stypy_kwargs_param_name = None
    containsmodule.stypy_call_defaults = defaults
    containsmodule.stypy_call_varargs = varargs
    containsmodule.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'containsmodule', ['block'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'containsmodule', localization, ['block'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'containsmodule(...)' code ##################

    
    
    # Call to ismodule(...): (line 525)
    # Processing the call arguments (line 525)
    # Getting the type of 'block' (line 525)
    block_68127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'block', False)
    # Processing the call keyword arguments (line 525)
    kwargs_68128 = {}
    # Getting the type of 'ismodule' (line 525)
    ismodule_68126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 7), 'ismodule', False)
    # Calling ismodule(args, kwargs) (line 525)
    ismodule_call_result_68129 = invoke(stypy.reporting.localization.Localization(__file__, 525, 7), ismodule_68126, *[block_68127], **kwargs_68128)
    
    # Testing the type of an if condition (line 525)
    if_condition_68130 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 525, 4), ismodule_call_result_68129)
    # Assigning a type to the variable 'if_condition_68130' (line 525)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 4), 'if_condition_68130', if_condition_68130)
    # SSA begins for if statement (line 525)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_68131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 526, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 526)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'stypy_return_type', int_68131)
    # SSA join for if statement (line 525)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to hasbody(...): (line 527)
    # Processing the call arguments (line 527)
    # Getting the type of 'block' (line 527)
    block_68133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 19), 'block', False)
    # Processing the call keyword arguments (line 527)
    kwargs_68134 = {}
    # Getting the type of 'hasbody' (line 527)
    hasbody_68132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 11), 'hasbody', False)
    # Calling hasbody(args, kwargs) (line 527)
    hasbody_call_result_68135 = invoke(stypy.reporting.localization.Localization(__file__, 527, 11), hasbody_68132, *[block_68133], **kwargs_68134)
    
    # Applying the 'not' unary operator (line 527)
    result_not__68136 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 7), 'not', hasbody_call_result_68135)
    
    # Testing the type of an if condition (line 527)
    if_condition_68137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 527, 4), result_not__68136)
    # Assigning a type to the variable 'if_condition_68137' (line 527)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'if_condition_68137', if_condition_68137)
    # SSA begins for if statement (line 527)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_68138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 528)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 8), 'stypy_return_type', int_68138)
    # SSA join for if statement (line 527)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    str_68139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 19), 'str', 'body')
    # Getting the type of 'block' (line 529)
    block_68140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 13), 'block')
    # Obtaining the member '__getitem__' of a type (line 529)
    getitem___68141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 13), block_68140, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 529)
    subscript_call_result_68142 = invoke(stypy.reporting.localization.Localization(__file__, 529, 13), getitem___68141, str_68139)
    
    # Testing the type of a for loop iterable (line 529)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 529, 4), subscript_call_result_68142)
    # Getting the type of the for loop variable (line 529)
    for_loop_var_68143 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 529, 4), subscript_call_result_68142)
    # Assigning a type to the variable 'b' (line 529)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 4), 'b', for_loop_var_68143)
    # SSA begins for a for statement (line 529)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to containsmodule(...): (line 530)
    # Processing the call arguments (line 530)
    # Getting the type of 'b' (line 530)
    b_68145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 26), 'b', False)
    # Processing the call keyword arguments (line 530)
    kwargs_68146 = {}
    # Getting the type of 'containsmodule' (line 530)
    containsmodule_68144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 11), 'containsmodule', False)
    # Calling containsmodule(args, kwargs) (line 530)
    containsmodule_call_result_68147 = invoke(stypy.reporting.localization.Localization(__file__, 530, 11), containsmodule_68144, *[b_68145], **kwargs_68146)
    
    # Testing the type of an if condition (line 530)
    if_condition_68148 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 530, 8), containsmodule_call_result_68147)
    # Assigning a type to the variable 'if_condition_68148' (line 530)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 8), 'if_condition_68148', if_condition_68148)
    # SSA begins for if statement (line 530)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_68149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 19), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 531)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'stypy_return_type', int_68149)
    # SSA join for if statement (line 530)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    int_68150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 532)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 4), 'stypy_return_type', int_68150)
    
    # ################# End of 'containsmodule(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'containsmodule' in the type store
    # Getting the type of 'stypy_return_type' (line 524)
    stypy_return_type_68151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68151)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'containsmodule'
    return stypy_return_type_68151

# Assigning a type to the variable 'containsmodule' (line 524)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 0), 'containsmodule', containsmodule)

@norecursion
def hasbody(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hasbody'
    module_type_store = module_type_store.open_function_context('hasbody', 535, 0, False)
    
    # Passed parameters checking function
    hasbody.stypy_localization = localization
    hasbody.stypy_type_of_self = None
    hasbody.stypy_type_store = module_type_store
    hasbody.stypy_function_name = 'hasbody'
    hasbody.stypy_param_names_list = ['rout']
    hasbody.stypy_varargs_param_name = None
    hasbody.stypy_kwargs_param_name = None
    hasbody.stypy_call_defaults = defaults
    hasbody.stypy_call_varargs = varargs
    hasbody.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hasbody', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hasbody', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hasbody(...)' code ##################

    
    str_68152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 536, 11), 'str', 'body')
    # Getting the type of 'rout' (line 536)
    rout_68153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 21), 'rout')
    # Applying the binary operator 'in' (line 536)
    result_contains_68154 = python_operator(stypy.reporting.localization.Localization(__file__, 536, 11), 'in', str_68152, rout_68153)
    
    # Assigning a type to the variable 'stypy_return_type' (line 536)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'stypy_return_type', result_contains_68154)
    
    # ################# End of 'hasbody(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hasbody' in the type store
    # Getting the type of 'stypy_return_type' (line 535)
    stypy_return_type_68155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68155)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hasbody'
    return stypy_return_type_68155

# Assigning a type to the variable 'hasbody' (line 535)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 0), 'hasbody', hasbody)

@norecursion
def hascallstatement(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'hascallstatement'
    module_type_store = module_type_store.open_function_context('hascallstatement', 539, 0, False)
    
    # Passed parameters checking function
    hascallstatement.stypy_localization = localization
    hascallstatement.stypy_type_of_self = None
    hascallstatement.stypy_type_store = module_type_store
    hascallstatement.stypy_function_name = 'hascallstatement'
    hascallstatement.stypy_param_names_list = ['rout']
    hascallstatement.stypy_varargs_param_name = None
    hascallstatement.stypy_kwargs_param_name = None
    hascallstatement.stypy_call_defaults = defaults
    hascallstatement.stypy_call_varargs = varargs
    hascallstatement.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'hascallstatement', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'hascallstatement', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'hascallstatement(...)' code ##################

    
    
    # Call to getcallstatement(...): (line 540)
    # Processing the call arguments (line 540)
    # Getting the type of 'rout' (line 540)
    rout_68157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 28), 'rout', False)
    # Processing the call keyword arguments (line 540)
    kwargs_68158 = {}
    # Getting the type of 'getcallstatement' (line 540)
    getcallstatement_68156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 11), 'getcallstatement', False)
    # Calling getcallstatement(args, kwargs) (line 540)
    getcallstatement_call_result_68159 = invoke(stypy.reporting.localization.Localization(__file__, 540, 11), getcallstatement_68156, *[rout_68157], **kwargs_68158)
    
    # Getting the type of 'None' (line 540)
    None_68160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 41), 'None')
    # Applying the binary operator 'isnot' (line 540)
    result_is_not_68161 = python_operator(stypy.reporting.localization.Localization(__file__, 540, 11), 'isnot', getcallstatement_call_result_68159, None_68160)
    
    # Assigning a type to the variable 'stypy_return_type' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'stypy_return_type', result_is_not_68161)
    
    # ################# End of 'hascallstatement(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'hascallstatement' in the type store
    # Getting the type of 'stypy_return_type' (line 539)
    stypy_return_type_68162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68162)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'hascallstatement'
    return stypy_return_type_68162

# Assigning a type to the variable 'hascallstatement' (line 539)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 0), 'hascallstatement', hascallstatement)

@norecursion
def istrue(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'istrue'
    module_type_store = module_type_store.open_function_context('istrue', 543, 0, False)
    
    # Passed parameters checking function
    istrue.stypy_localization = localization
    istrue.stypy_type_of_self = None
    istrue.stypy_type_store = module_type_store
    istrue.stypy_function_name = 'istrue'
    istrue.stypy_param_names_list = ['var']
    istrue.stypy_varargs_param_name = None
    istrue.stypy_kwargs_param_name = None
    istrue.stypy_call_defaults = defaults
    istrue.stypy_call_varargs = varargs
    istrue.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'istrue', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'istrue', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'istrue(...)' code ##################

    int_68163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'stypy_return_type', int_68163)
    
    # ################# End of 'istrue(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'istrue' in the type store
    # Getting the type of 'stypy_return_type' (line 543)
    stypy_return_type_68164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68164)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'istrue'
    return stypy_return_type_68164

# Assigning a type to the variable 'istrue' (line 543)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 543, 0), 'istrue', istrue)

@norecursion
def isfalse(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isfalse'
    module_type_store = module_type_store.open_function_context('isfalse', 547, 0, False)
    
    # Passed parameters checking function
    isfalse.stypy_localization = localization
    isfalse.stypy_type_of_self = None
    isfalse.stypy_type_store = module_type_store
    isfalse.stypy_function_name = 'isfalse'
    isfalse.stypy_param_names_list = ['var']
    isfalse.stypy_varargs_param_name = None
    isfalse.stypy_kwargs_param_name = None
    isfalse.stypy_call_defaults = defaults
    isfalse.stypy_call_varargs = varargs
    isfalse.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isfalse', ['var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isfalse', localization, ['var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isfalse(...)' code ##################

    int_68165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 548)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 4), 'stypy_return_type', int_68165)
    
    # ################# End of 'isfalse(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isfalse' in the type store
    # Getting the type of 'stypy_return_type' (line 547)
    stypy_return_type_68166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68166)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isfalse'
    return stypy_return_type_68166

# Assigning a type to the variable 'isfalse' (line 547)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 0), 'isfalse', isfalse)
# Declaration of the 'F2PYError' class
# Getting the type of 'Exception' (line 551)
Exception_68167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 16), 'Exception')

class F2PYError(Exception_68167, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 551, 0, False)
        # Assigning a type to the variable 'self' (line 552)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'F2PYError.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'F2PYError' (line 551)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 0), 'F2PYError', F2PYError)
# Declaration of the 'throw_error' class

class throw_error:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 557, 4, False)
        # Assigning a type to the variable 'self' (line 558)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'throw_error.__init__', ['mess'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['mess'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 558):
        
        # Assigning a Name to a Attribute (line 558):
        # Getting the type of 'mess' (line 558)
        mess_68168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 20), 'mess')
        # Getting the type of 'self' (line 558)
        self_68169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'self')
        # Setting the type of the member 'mess' of a type (line 558)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 8), self_68169, 'mess', mess_68168)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 560, 4, False)
        # Assigning a type to the variable 'self' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        throw_error.__call__.__dict__.__setitem__('stypy_localization', localization)
        throw_error.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        throw_error.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        throw_error.__call__.__dict__.__setitem__('stypy_function_name', 'throw_error.__call__')
        throw_error.__call__.__dict__.__setitem__('stypy_param_names_list', ['var'])
        throw_error.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        throw_error.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        throw_error.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        throw_error.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        throw_error.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        throw_error.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'throw_error.__call__', ['var'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['var'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a BinOp to a Name (line 561):
        
        # Assigning a BinOp to a Name (line 561):
        str_68170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 15), 'str', '\n\n  var = %s\n  Message: %s\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 561)
        tuple_68171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 52), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 561)
        # Adding element type (line 561)
        # Getting the type of 'var' (line 561)
        var_68172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 52), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 52), tuple_68171, var_68172)
        # Adding element type (line 561)
        # Getting the type of 'self' (line 561)
        self_68173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 57), 'self')
        # Obtaining the member 'mess' of a type (line 561)
        mess_68174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 57), self_68173, 'mess')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 561, 52), tuple_68171, mess_68174)
        
        # Applying the binary operator '%' (line 561)
        result_mod_68175 = python_operator(stypy.reporting.localization.Localization(__file__, 561, 15), '%', str_68170, tuple_68171)
        
        # Assigning a type to the variable 'mess' (line 561)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'mess', result_mod_68175)
        
        # Call to F2PYError(...): (line 562)
        # Processing the call arguments (line 562)
        # Getting the type of 'mess' (line 562)
        mess_68177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 24), 'mess', False)
        # Processing the call keyword arguments (line 562)
        kwargs_68178 = {}
        # Getting the type of 'F2PYError' (line 562)
        F2PYError_68176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 14), 'F2PYError', False)
        # Calling F2PYError(args, kwargs) (line 562)
        F2PYError_call_result_68179 = invoke(stypy.reporting.localization.Localization(__file__, 562, 14), F2PYError_68176, *[mess_68177], **kwargs_68178)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 562, 8), F2PYError_call_result_68179, 'raise parameter', BaseException)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 560)
        stypy_return_type_68180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_68180)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_68180


# Assigning a type to the variable 'throw_error' (line 555)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 0), 'throw_error', throw_error)

@norecursion
def l_and(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'l_and'
    module_type_store = module_type_store.open_function_context('l_and', 565, 0, False)
    
    # Passed parameters checking function
    l_and.stypy_localization = localization
    l_and.stypy_type_of_self = None
    l_and.stypy_type_store = module_type_store
    l_and.stypy_function_name = 'l_and'
    l_and.stypy_param_names_list = []
    l_and.stypy_varargs_param_name = 'f'
    l_and.stypy_kwargs_param_name = None
    l_and.stypy_call_defaults = defaults
    l_and.stypy_call_varargs = varargs
    l_and.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'l_and', [], 'f', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'l_and', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'l_and(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 566):
    
    # Assigning a Str to a Name (line 566):
    str_68181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 12), 'str', 'lambda v')
    # Assigning a type to the variable 'tuple_assignment_66676' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'tuple_assignment_66676', str_68181)
    
    # Assigning a List to a Name (line 566):
    
    # Obtaining an instance of the builtin type 'list' (line 566)
    list_68182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 566)
    
    # Assigning a type to the variable 'tuple_assignment_66677' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'tuple_assignment_66677', list_68182)
    
    # Assigning a Name to a Name (line 566):
    # Getting the type of 'tuple_assignment_66676' (line 566)
    tuple_assignment_66676_68183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'tuple_assignment_66676')
    # Assigning a type to the variable 'l' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'l', tuple_assignment_66676_68183)
    
    # Assigning a Name to a Name (line 566):
    # Getting the type of 'tuple_assignment_66677' (line 566)
    tuple_assignment_66677_68184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 4), 'tuple_assignment_66677')
    # Assigning a type to the variable 'l2' (line 566)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 566, 7), 'l2', tuple_assignment_66677_68184)
    
    
    # Call to range(...): (line 567)
    # Processing the call arguments (line 567)
    
    # Call to len(...): (line 567)
    # Processing the call arguments (line 567)
    # Getting the type of 'f' (line 567)
    f_68187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 23), 'f', False)
    # Processing the call keyword arguments (line 567)
    kwargs_68188 = {}
    # Getting the type of 'len' (line 567)
    len_68186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 19), 'len', False)
    # Calling len(args, kwargs) (line 567)
    len_call_result_68189 = invoke(stypy.reporting.localization.Localization(__file__, 567, 19), len_68186, *[f_68187], **kwargs_68188)
    
    # Processing the call keyword arguments (line 567)
    kwargs_68190 = {}
    # Getting the type of 'range' (line 567)
    range_68185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 13), 'range', False)
    # Calling range(args, kwargs) (line 567)
    range_call_result_68191 = invoke(stypy.reporting.localization.Localization(__file__, 567, 13), range_68185, *[len_call_result_68189], **kwargs_68190)
    
    # Testing the type of a for loop iterable (line 567)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 567, 4), range_call_result_68191)
    # Getting the type of the for loop variable (line 567)
    for_loop_var_68192 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 567, 4), range_call_result_68191)
    # Assigning a type to the variable 'i' (line 567)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'i', for_loop_var_68192)
    # SSA begins for a for statement (line 567)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 568):
    
    # Assigning a BinOp to a Name (line 568):
    str_68193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 12), 'str', '%s,f%d=f[%d]')
    
    # Obtaining an instance of the builtin type 'tuple' (line 568)
    tuple_68194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 568)
    # Adding element type (line 568)
    # Getting the type of 'l' (line 568)
    l_68195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 30), 'l')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 30), tuple_68194, l_68195)
    # Adding element type (line 568)
    # Getting the type of 'i' (line 568)
    i_68196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 33), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 30), tuple_68194, i_68196)
    # Adding element type (line 568)
    # Getting the type of 'i' (line 568)
    i_68197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 36), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 568, 30), tuple_68194, i_68197)
    
    # Applying the binary operator '%' (line 568)
    result_mod_68198 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 12), '%', str_68193, tuple_68194)
    
    # Assigning a type to the variable 'l' (line 568)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'l', result_mod_68198)
    
    # Call to append(...): (line 569)
    # Processing the call arguments (line 569)
    str_68201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 569, 18), 'str', 'f%d(v)')
    # Getting the type of 'i' (line 569)
    i_68202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 30), 'i', False)
    # Applying the binary operator '%' (line 569)
    result_mod_68203 = python_operator(stypy.reporting.localization.Localization(__file__, 569, 18), '%', str_68201, i_68202)
    
    # Processing the call keyword arguments (line 569)
    kwargs_68204 = {}
    # Getting the type of 'l2' (line 569)
    l2_68199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'l2', False)
    # Obtaining the member 'append' of a type (line 569)
    append_68200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 8), l2_68199, 'append')
    # Calling append(args, kwargs) (line 569)
    append_call_result_68205 = invoke(stypy.reporting.localization.Localization(__file__, 569, 8), append_68200, *[result_mod_68203], **kwargs_68204)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to eval(...): (line 570)
    # Processing the call arguments (line 570)
    str_68207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 16), 'str', '%s:%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 570)
    tuple_68208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 570)
    # Adding element type (line 570)
    # Getting the type of 'l' (line 570)
    l_68209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 27), 'l', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 27), tuple_68208, l_68209)
    # Adding element type (line 570)
    
    # Call to join(...): (line 570)
    # Processing the call arguments (line 570)
    # Getting the type of 'l2' (line 570)
    l2_68212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 43), 'l2', False)
    # Processing the call keyword arguments (line 570)
    kwargs_68213 = {}
    str_68210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 30), 'str', ' and ')
    # Obtaining the member 'join' of a type (line 570)
    join_68211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 30), str_68210, 'join')
    # Calling join(args, kwargs) (line 570)
    join_call_result_68214 = invoke(stypy.reporting.localization.Localization(__file__, 570, 30), join_68211, *[l2_68212], **kwargs_68213)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 570, 27), tuple_68208, join_call_result_68214)
    
    # Applying the binary operator '%' (line 570)
    result_mod_68215 = python_operator(stypy.reporting.localization.Localization(__file__, 570, 16), '%', str_68207, tuple_68208)
    
    # Processing the call keyword arguments (line 570)
    kwargs_68216 = {}
    # Getting the type of 'eval' (line 570)
    eval_68206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 11), 'eval', False)
    # Calling eval(args, kwargs) (line 570)
    eval_call_result_68217 = invoke(stypy.reporting.localization.Localization(__file__, 570, 11), eval_68206, *[result_mod_68215], **kwargs_68216)
    
    # Assigning a type to the variable 'stypy_return_type' (line 570)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 4), 'stypy_return_type', eval_call_result_68217)
    
    # ################# End of 'l_and(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'l_and' in the type store
    # Getting the type of 'stypy_return_type' (line 565)
    stypy_return_type_68218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68218)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'l_and'
    return stypy_return_type_68218

# Assigning a type to the variable 'l_and' (line 565)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 0), 'l_and', l_and)

@norecursion
def l_or(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'l_or'
    module_type_store = module_type_store.open_function_context('l_or', 573, 0, False)
    
    # Passed parameters checking function
    l_or.stypy_localization = localization
    l_or.stypy_type_of_self = None
    l_or.stypy_type_store = module_type_store
    l_or.stypy_function_name = 'l_or'
    l_or.stypy_param_names_list = []
    l_or.stypy_varargs_param_name = 'f'
    l_or.stypy_kwargs_param_name = None
    l_or.stypy_call_defaults = defaults
    l_or.stypy_call_varargs = varargs
    l_or.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'l_or', [], 'f', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'l_or', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'l_or(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 574):
    
    # Assigning a Str to a Name (line 574):
    str_68219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 12), 'str', 'lambda v')
    # Assigning a type to the variable 'tuple_assignment_66678' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_assignment_66678', str_68219)
    
    # Assigning a List to a Name (line 574):
    
    # Obtaining an instance of the builtin type 'list' (line 574)
    list_68220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 574)
    
    # Assigning a type to the variable 'tuple_assignment_66679' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_assignment_66679', list_68220)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'tuple_assignment_66678' (line 574)
    tuple_assignment_66678_68221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_assignment_66678')
    # Assigning a type to the variable 'l' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'l', tuple_assignment_66678_68221)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'tuple_assignment_66679' (line 574)
    tuple_assignment_66679_68222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_assignment_66679')
    # Assigning a type to the variable 'l2' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 7), 'l2', tuple_assignment_66679_68222)
    
    
    # Call to range(...): (line 575)
    # Processing the call arguments (line 575)
    
    # Call to len(...): (line 575)
    # Processing the call arguments (line 575)
    # Getting the type of 'f' (line 575)
    f_68225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 23), 'f', False)
    # Processing the call keyword arguments (line 575)
    kwargs_68226 = {}
    # Getting the type of 'len' (line 575)
    len_68224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 19), 'len', False)
    # Calling len(args, kwargs) (line 575)
    len_call_result_68227 = invoke(stypy.reporting.localization.Localization(__file__, 575, 19), len_68224, *[f_68225], **kwargs_68226)
    
    # Processing the call keyword arguments (line 575)
    kwargs_68228 = {}
    # Getting the type of 'range' (line 575)
    range_68223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 13), 'range', False)
    # Calling range(args, kwargs) (line 575)
    range_call_result_68229 = invoke(stypy.reporting.localization.Localization(__file__, 575, 13), range_68223, *[len_call_result_68227], **kwargs_68228)
    
    # Testing the type of a for loop iterable (line 575)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 575, 4), range_call_result_68229)
    # Getting the type of the for loop variable (line 575)
    for_loop_var_68230 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 575, 4), range_call_result_68229)
    # Assigning a type to the variable 'i' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'i', for_loop_var_68230)
    # SSA begins for a for statement (line 575)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 576):
    
    # Assigning a BinOp to a Name (line 576):
    str_68231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 12), 'str', '%s,f%d=f[%d]')
    
    # Obtaining an instance of the builtin type 'tuple' (line 576)
    tuple_68232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 576)
    # Adding element type (line 576)
    # Getting the type of 'l' (line 576)
    l_68233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 30), 'l')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 30), tuple_68232, l_68233)
    # Adding element type (line 576)
    # Getting the type of 'i' (line 576)
    i_68234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 33), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 30), tuple_68232, i_68234)
    # Adding element type (line 576)
    # Getting the type of 'i' (line 576)
    i_68235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 36), 'i')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 576, 30), tuple_68232, i_68235)
    
    # Applying the binary operator '%' (line 576)
    result_mod_68236 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 12), '%', str_68231, tuple_68232)
    
    # Assigning a type to the variable 'l' (line 576)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'l', result_mod_68236)
    
    # Call to append(...): (line 577)
    # Processing the call arguments (line 577)
    str_68239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 18), 'str', 'f%d(v)')
    # Getting the type of 'i' (line 577)
    i_68240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 30), 'i', False)
    # Applying the binary operator '%' (line 577)
    result_mod_68241 = python_operator(stypy.reporting.localization.Localization(__file__, 577, 18), '%', str_68239, i_68240)
    
    # Processing the call keyword arguments (line 577)
    kwargs_68242 = {}
    # Getting the type of 'l2' (line 577)
    l2_68237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 8), 'l2', False)
    # Obtaining the member 'append' of a type (line 577)
    append_68238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 577, 8), l2_68237, 'append')
    # Calling append(args, kwargs) (line 577)
    append_call_result_68243 = invoke(stypy.reporting.localization.Localization(__file__, 577, 8), append_68238, *[result_mod_68241], **kwargs_68242)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to eval(...): (line 578)
    # Processing the call arguments (line 578)
    str_68245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 16), 'str', '%s:%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 578)
    tuple_68246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 578)
    # Adding element type (line 578)
    # Getting the type of 'l' (line 578)
    l_68247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 27), 'l', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 27), tuple_68246, l_68247)
    # Adding element type (line 578)
    
    # Call to join(...): (line 578)
    # Processing the call arguments (line 578)
    # Getting the type of 'l2' (line 578)
    l2_68250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 42), 'l2', False)
    # Processing the call keyword arguments (line 578)
    kwargs_68251 = {}
    str_68248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 30), 'str', ' or ')
    # Obtaining the member 'join' of a type (line 578)
    join_68249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 30), str_68248, 'join')
    # Calling join(args, kwargs) (line 578)
    join_call_result_68252 = invoke(stypy.reporting.localization.Localization(__file__, 578, 30), join_68249, *[l2_68250], **kwargs_68251)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 578, 27), tuple_68246, join_call_result_68252)
    
    # Applying the binary operator '%' (line 578)
    result_mod_68253 = python_operator(stypy.reporting.localization.Localization(__file__, 578, 16), '%', str_68245, tuple_68246)
    
    # Processing the call keyword arguments (line 578)
    kwargs_68254 = {}
    # Getting the type of 'eval' (line 578)
    eval_68244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 11), 'eval', False)
    # Calling eval(args, kwargs) (line 578)
    eval_call_result_68255 = invoke(stypy.reporting.localization.Localization(__file__, 578, 11), eval_68244, *[result_mod_68253], **kwargs_68254)
    
    # Assigning a type to the variable 'stypy_return_type' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'stypy_return_type', eval_call_result_68255)
    
    # ################# End of 'l_or(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'l_or' in the type store
    # Getting the type of 'stypy_return_type' (line 573)
    stypy_return_type_68256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68256)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'l_or'
    return stypy_return_type_68256

# Assigning a type to the variable 'l_or' (line 573)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 0), 'l_or', l_or)

@norecursion
def l_not(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'l_not'
    module_type_store = module_type_store.open_function_context('l_not', 581, 0, False)
    
    # Passed parameters checking function
    l_not.stypy_localization = localization
    l_not.stypy_type_of_self = None
    l_not.stypy_type_store = module_type_store
    l_not.stypy_function_name = 'l_not'
    l_not.stypy_param_names_list = ['f']
    l_not.stypy_varargs_param_name = None
    l_not.stypy_kwargs_param_name = None
    l_not.stypy_call_defaults = defaults
    l_not.stypy_call_varargs = varargs
    l_not.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'l_not', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'l_not', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'l_not(...)' code ##################

    
    # Call to eval(...): (line 582)
    # Processing the call arguments (line 582)
    str_68258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 582, 16), 'str', 'lambda v,f=f:not f(v)')
    # Processing the call keyword arguments (line 582)
    kwargs_68259 = {}
    # Getting the type of 'eval' (line 582)
    eval_68257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 11), 'eval', False)
    # Calling eval(args, kwargs) (line 582)
    eval_call_result_68260 = invoke(stypy.reporting.localization.Localization(__file__, 582, 11), eval_68257, *[str_68258], **kwargs_68259)
    
    # Assigning a type to the variable 'stypy_return_type' (line 582)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'stypy_return_type', eval_call_result_68260)
    
    # ################# End of 'l_not(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'l_not' in the type store
    # Getting the type of 'stypy_return_type' (line 581)
    stypy_return_type_68261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68261)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'l_not'
    return stypy_return_type_68261

# Assigning a type to the variable 'l_not' (line 581)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 0), 'l_not', l_not)

@norecursion
def isdummyroutine(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'isdummyroutine'
    module_type_store = module_type_store.open_function_context('isdummyroutine', 585, 0, False)
    
    # Passed parameters checking function
    isdummyroutine.stypy_localization = localization
    isdummyroutine.stypy_type_of_self = None
    isdummyroutine.stypy_type_store = module_type_store
    isdummyroutine.stypy_function_name = 'isdummyroutine'
    isdummyroutine.stypy_param_names_list = ['rout']
    isdummyroutine.stypy_varargs_param_name = None
    isdummyroutine.stypy_kwargs_param_name = None
    isdummyroutine.stypy_call_defaults = defaults
    isdummyroutine.stypy_call_varargs = varargs
    isdummyroutine.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'isdummyroutine', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'isdummyroutine', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'isdummyroutine(...)' code ##################

    
    
    # SSA begins for try-except statement (line 586)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Obtaining the type of the subscript
    str_68262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 40), 'str', 'fortranname')
    
    # Obtaining the type of the subscript
    str_68263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 20), 'str', 'f2pyenhancements')
    # Getting the type of 'rout' (line 587)
    rout_68264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 15), 'rout')
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___68265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 15), rout_68264, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_68266 = invoke(stypy.reporting.localization.Localization(__file__, 587, 15), getitem___68265, str_68263)
    
    # Obtaining the member '__getitem__' of a type (line 587)
    getitem___68267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 15), subscript_call_result_68266, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 587)
    subscript_call_result_68268 = invoke(stypy.reporting.localization.Localization(__file__, 587, 15), getitem___68267, str_68262)
    
    str_68269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 587, 58), 'str', '')
    # Applying the binary operator '==' (line 587)
    result_eq_68270 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 15), '==', subscript_call_result_68268, str_68269)
    
    # Assigning a type to the variable 'stypy_return_type' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'stypy_return_type', result_eq_68270)
    # SSA branch for the except part of a try statement (line 586)
    # SSA branch for the except 'KeyError' branch of a try statement (line 586)
    module_type_store.open_ssa_branch('except')
    int_68271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 589)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 589, 8), 'stypy_return_type', int_68271)
    # SSA join for try-except statement (line 586)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'isdummyroutine(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'isdummyroutine' in the type store
    # Getting the type of 'stypy_return_type' (line 585)
    stypy_return_type_68272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68272)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'isdummyroutine'
    return stypy_return_type_68272

# Assigning a type to the variable 'isdummyroutine' (line 585)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 0), 'isdummyroutine', isdummyroutine)

@norecursion
def getfortranname(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getfortranname'
    module_type_store = module_type_store.open_function_context('getfortranname', 592, 0, False)
    
    # Passed parameters checking function
    getfortranname.stypy_localization = localization
    getfortranname.stypy_type_of_self = None
    getfortranname.stypy_type_store = module_type_store
    getfortranname.stypy_function_name = 'getfortranname'
    getfortranname.stypy_param_names_list = ['rout']
    getfortranname.stypy_varargs_param_name = None
    getfortranname.stypy_kwargs_param_name = None
    getfortranname.stypy_call_defaults = defaults
    getfortranname.stypy_call_varargs = varargs
    getfortranname.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getfortranname', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getfortranname', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getfortranname(...)' code ##################

    
    
    # SSA begins for try-except statement (line 593)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 594):
    
    # Assigning a Subscript to a Name (line 594):
    
    # Obtaining the type of the subscript
    str_68273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 40), 'str', 'fortranname')
    
    # Obtaining the type of the subscript
    str_68274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 20), 'str', 'f2pyenhancements')
    # Getting the type of 'rout' (line 594)
    rout_68275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 15), 'rout')
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___68276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 15), rout_68275, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_68277 = invoke(stypy.reporting.localization.Localization(__file__, 594, 15), getitem___68276, str_68274)
    
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___68278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 15), subscript_call_result_68277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_68279 = invoke(stypy.reporting.localization.Localization(__file__, 594, 15), getitem___68278, str_68273)
    
    # Assigning a type to the variable 'name' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'name', subscript_call_result_68279)
    
    
    # Getting the type of 'name' (line 595)
    name_68280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 11), 'name')
    str_68281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 19), 'str', '')
    # Applying the binary operator '==' (line 595)
    result_eq_68282 = python_operator(stypy.reporting.localization.Localization(__file__, 595, 11), '==', name_68280, str_68281)
    
    # Testing the type of an if condition (line 595)
    if_condition_68283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 595, 8), result_eq_68282)
    # Assigning a type to the variable 'if_condition_68283' (line 595)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 8), 'if_condition_68283', if_condition_68283)
    # SSA begins for if statement (line 595)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'KeyError' (line 596)
    KeyError_68284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 18), 'KeyError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 596, 12), KeyError_68284, 'raise parameter', BaseException)
    # SSA join for if statement (line 595)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'name' (line 597)
    name_68285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 15), 'name')
    # Applying the 'not' unary operator (line 597)
    result_not__68286 = python_operator(stypy.reporting.localization.Localization(__file__, 597, 11), 'not', name_68285)
    
    # Testing the type of an if condition (line 597)
    if_condition_68287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 8), result_not__68286)
    # Assigning a type to the variable 'if_condition_68287' (line 597)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 8), 'if_condition_68287', if_condition_68287)
    # SSA begins for if statement (line 597)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to errmess(...): (line 598)
    # Processing the call arguments (line 598)
    str_68289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 20), 'str', 'Failed to use fortranname from %s\n')
    
    # Obtaining the type of the subscript
    str_68290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 26), 'str', 'f2pyenhancements')
    # Getting the type of 'rout' (line 599)
    rout_68291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 21), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 599)
    getitem___68292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 21), rout_68291, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 599)
    subscript_call_result_68293 = invoke(stypy.reporting.localization.Localization(__file__, 599, 21), getitem___68292, str_68290)
    
    # Applying the binary operator '%' (line 598)
    result_mod_68294 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 20), '%', str_68289, subscript_call_result_68293)
    
    # Processing the call keyword arguments (line 598)
    kwargs_68295 = {}
    # Getting the type of 'errmess' (line 598)
    errmess_68288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 12), 'errmess', False)
    # Calling errmess(args, kwargs) (line 598)
    errmess_call_result_68296 = invoke(stypy.reporting.localization.Localization(__file__, 598, 12), errmess_68288, *[result_mod_68294], **kwargs_68295)
    
    # Getting the type of 'KeyError' (line 600)
    KeyError_68297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 18), 'KeyError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 600, 12), KeyError_68297, 'raise parameter', BaseException)
    # SSA join for if statement (line 597)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 593)
    # SSA branch for the except 'KeyError' branch of a try statement (line 593)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Subscript to a Name (line 602):
    
    # Assigning a Subscript to a Name (line 602):
    
    # Obtaining the type of the subscript
    str_68298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 20), 'str', 'name')
    # Getting the type of 'rout' (line 602)
    rout_68299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 15), 'rout')
    # Obtaining the member '__getitem__' of a type (line 602)
    getitem___68300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 15), rout_68299, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 602)
    subscript_call_result_68301 = invoke(stypy.reporting.localization.Localization(__file__, 602, 15), getitem___68300, str_68298)
    
    # Assigning a type to the variable 'name' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 8), 'name', subscript_call_result_68301)
    # SSA join for try-except statement (line 593)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'name' (line 603)
    name_68302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 11), 'name')
    # Assigning a type to the variable 'stypy_return_type' (line 603)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 4), 'stypy_return_type', name_68302)
    
    # ################# End of 'getfortranname(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getfortranname' in the type store
    # Getting the type of 'stypy_return_type' (line 592)
    stypy_return_type_68303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68303)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getfortranname'
    return stypy_return_type_68303

# Assigning a type to the variable 'getfortranname' (line 592)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 0), 'getfortranname', getfortranname)

@norecursion
def getmultilineblock(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_68304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 47), 'int')
    int_68305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 58), 'int')
    defaults = [int_68304, int_68305]
    # Create a new context for function 'getmultilineblock'
    module_type_store = module_type_store.open_function_context('getmultilineblock', 606, 0, False)
    
    # Passed parameters checking function
    getmultilineblock.stypy_localization = localization
    getmultilineblock.stypy_type_of_self = None
    getmultilineblock.stypy_type_store = module_type_store
    getmultilineblock.stypy_function_name = 'getmultilineblock'
    getmultilineblock.stypy_param_names_list = ['rout', 'blockname', 'comment', 'counter']
    getmultilineblock.stypy_varargs_param_name = None
    getmultilineblock.stypy_kwargs_param_name = None
    getmultilineblock.stypy_call_defaults = defaults
    getmultilineblock.stypy_call_varargs = varargs
    getmultilineblock.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getmultilineblock', ['rout', 'blockname', 'comment', 'counter'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getmultilineblock', localization, ['rout', 'blockname', 'comment', 'counter'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getmultilineblock(...)' code ##################

    
    
    # SSA begins for try-except statement (line 607)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 608):
    
    # Assigning a Call to a Name (line 608):
    
    # Call to get(...): (line 608)
    # Processing the call arguments (line 608)
    # Getting the type of 'blockname' (line 608)
    blockname_68311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 41), 'blockname', False)
    # Processing the call keyword arguments (line 608)
    kwargs_68312 = {}
    
    # Obtaining the type of the subscript
    str_68306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 17), 'str', 'f2pyenhancements')
    # Getting the type of 'rout' (line 608)
    rout_68307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 608, 12), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 608)
    getitem___68308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 12), rout_68307, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 608)
    subscript_call_result_68309 = invoke(stypy.reporting.localization.Localization(__file__, 608, 12), getitem___68308, str_68306)
    
    # Obtaining the member 'get' of a type (line 608)
    get_68310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 608, 12), subscript_call_result_68309, 'get')
    # Calling get(args, kwargs) (line 608)
    get_call_result_68313 = invoke(stypy.reporting.localization.Localization(__file__, 608, 12), get_68310, *[blockname_68311], **kwargs_68312)
    
    # Assigning a type to the variable 'r' (line 608)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 8), 'r', get_call_result_68313)
    # SSA branch for the except part of a try statement (line 607)
    # SSA branch for the except 'KeyError' branch of a try statement (line 607)
    module_type_store.open_ssa_branch('except')
    # Assigning a type to the variable 'stypy_return_type' (line 610)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'stypy_return_type', types.NoneType)
    # SSA join for try-except statement (line 607)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'r' (line 611)
    r_68314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 11), 'r')
    # Applying the 'not' unary operator (line 611)
    result_not__68315 = python_operator(stypy.reporting.localization.Localization(__file__, 611, 7), 'not', r_68314)
    
    # Testing the type of an if condition (line 611)
    if_condition_68316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 4), result_not__68315)
    # Assigning a type to the variable 'if_condition_68316' (line 611)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 4), 'if_condition_68316', if_condition_68316)
    # SSA begins for if statement (line 611)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 612)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 611)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'counter' (line 613)
    counter_68317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 7), 'counter')
    int_68318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 613, 17), 'int')
    # Applying the binary operator '>' (line 613)
    result_gt_68319 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 7), '>', counter_68317, int_68318)
    
    
    # Call to isinstance(...): (line 613)
    # Processing the call arguments (line 613)
    # Getting the type of 'r' (line 613)
    r_68321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 34), 'r', False)
    # Getting the type of 'str' (line 613)
    str_68322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 37), 'str', False)
    # Processing the call keyword arguments (line 613)
    kwargs_68323 = {}
    # Getting the type of 'isinstance' (line 613)
    isinstance_68320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 23), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 613)
    isinstance_call_result_68324 = invoke(stypy.reporting.localization.Localization(__file__, 613, 23), isinstance_68320, *[r_68321, str_68322], **kwargs_68323)
    
    # Applying the binary operator 'and' (line 613)
    result_and_keyword_68325 = python_operator(stypy.reporting.localization.Localization(__file__, 613, 7), 'and', result_gt_68319, isinstance_call_result_68324)
    
    # Testing the type of an if condition (line 613)
    if_condition_68326 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 613, 4), result_and_keyword_68325)
    # Assigning a type to the variable 'if_condition_68326' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'if_condition_68326', if_condition_68326)
    # SSA begins for if statement (line 613)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 613)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 615)
    # Getting the type of 'list' (line 615)
    list_68327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 21), 'list')
    # Getting the type of 'r' (line 615)
    r_68328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 18), 'r')
    
    (may_be_68329, more_types_in_union_68330) = may_be_subtype(list_68327, r_68328)

    if may_be_68329:

        if more_types_in_union_68330:
            # Runtime conditional SSA (line 615)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'r' (line 615)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'r', remove_not_subtype_from_union(r_68328, list))
        
        
        # Getting the type of 'counter' (line 616)
        counter_68331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 11), 'counter')
        
        # Call to len(...): (line 616)
        # Processing the call arguments (line 616)
        # Getting the type of 'r' (line 616)
        r_68333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 26), 'r', False)
        # Processing the call keyword arguments (line 616)
        kwargs_68334 = {}
        # Getting the type of 'len' (line 616)
        len_68332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 22), 'len', False)
        # Calling len(args, kwargs) (line 616)
        len_call_result_68335 = invoke(stypy.reporting.localization.Localization(__file__, 616, 22), len_68332, *[r_68333], **kwargs_68334)
        
        # Applying the binary operator '>=' (line 616)
        result_ge_68336 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 11), '>=', counter_68331, len_call_result_68335)
        
        # Testing the type of an if condition (line 616)
        if_condition_68337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 616, 8), result_ge_68336)
        # Assigning a type to the variable 'if_condition_68337' (line 616)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'if_condition_68337', if_condition_68337)
        # SSA begins for if statement (line 616)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 617)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 616)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 618):
        
        # Assigning a Subscript to a Name (line 618):
        
        # Obtaining the type of the subscript
        # Getting the type of 'counter' (line 618)
        counter_68338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 14), 'counter')
        # Getting the type of 'r' (line 618)
        r_68339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), 'r')
        # Obtaining the member '__getitem__' of a type (line 618)
        getitem___68340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 12), r_68339, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 618)
        subscript_call_result_68341 = invoke(stypy.reporting.localization.Localization(__file__, 618, 12), getitem___68340, counter_68338)
        
        # Assigning a type to the variable 'r' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'r', subscript_call_result_68341)

        if more_types_in_union_68330:
            # SSA join for if statement (line 615)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    
    # Obtaining the type of the subscript
    int_68342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 10), 'int')
    slice_68343 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 619, 7), None, int_68342, None)
    # Getting the type of 'r' (line 619)
    r_68344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 7), 'r')
    # Obtaining the member '__getitem__' of a type (line 619)
    getitem___68345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 7), r_68344, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 619)
    subscript_call_result_68346 = invoke(stypy.reporting.localization.Localization(__file__, 619, 7), getitem___68345, slice_68343)
    
    str_68347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 16), 'str', "'''")
    # Applying the binary operator '==' (line 619)
    result_eq_68348 = python_operator(stypy.reporting.localization.Localization(__file__, 619, 7), '==', subscript_call_result_68346, str_68347)
    
    # Testing the type of an if condition (line 619)
    if_condition_68349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 619, 4), result_eq_68348)
    # Assigning a type to the variable 'if_condition_68349' (line 619)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 4), 'if_condition_68349', if_condition_68349)
    # SSA begins for if statement (line 619)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'comment' (line 620)
    comment_68350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 11), 'comment')
    # Testing the type of an if condition (line 620)
    if_condition_68351 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 8), comment_68350)
    # Assigning a type to the variable 'if_condition_68351' (line 620)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'if_condition_68351', if_condition_68351)
    # SSA begins for if statement (line 620)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 621):
    
    # Assigning a BinOp to a Name (line 621):
    str_68352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 16), 'str', '\t/* start ')
    # Getting the type of 'blockname' (line 621)
    blockname_68353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 32), 'blockname')
    # Applying the binary operator '+' (line 621)
    result_add_68354 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 16), '+', str_68352, blockname_68353)
    
    str_68355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 16), 'str', ' multiline (')
    # Applying the binary operator '+' (line 621)
    result_add_68356 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 42), '+', result_add_68354, str_68355)
    
    
    # Call to repr(...): (line 622)
    # Processing the call arguments (line 622)
    # Getting the type of 'counter' (line 622)
    counter_68358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 38), 'counter', False)
    # Processing the call keyword arguments (line 622)
    kwargs_68359 = {}
    # Getting the type of 'repr' (line 622)
    repr_68357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 33), 'repr', False)
    # Calling repr(args, kwargs) (line 622)
    repr_call_result_68360 = invoke(stypy.reporting.localization.Localization(__file__, 622, 33), repr_68357, *[counter_68358], **kwargs_68359)
    
    # Applying the binary operator '+' (line 622)
    result_add_68361 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 31), '+', result_add_68356, repr_call_result_68360)
    
    str_68362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 49), 'str', ') */\n')
    # Applying the binary operator '+' (line 622)
    result_add_68363 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 47), '+', result_add_68361, str_68362)
    
    
    # Obtaining the type of the subscript
    int_68364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 622, 62), 'int')
    slice_68365 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 622, 60), int_68364, None, None)
    # Getting the type of 'r' (line 622)
    r_68366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 60), 'r')
    # Obtaining the member '__getitem__' of a type (line 622)
    getitem___68367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 60), r_68366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 622)
    subscript_call_result_68368 = invoke(stypy.reporting.localization.Localization(__file__, 622, 60), getitem___68367, slice_68365)
    
    # Applying the binary operator '+' (line 622)
    result_add_68369 = python_operator(stypy.reporting.localization.Localization(__file__, 622, 58), '+', result_add_68363, subscript_call_result_68368)
    
    # Assigning a type to the variable 'r' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 12), 'r', result_add_68369)
    # SSA branch for the else part of an if statement (line 620)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 624):
    
    # Assigning a Subscript to a Name (line 624):
    
    # Obtaining the type of the subscript
    int_68370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 18), 'int')
    slice_68371 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 624, 16), int_68370, None, None)
    # Getting the type of 'r' (line 624)
    r_68372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 16), 'r')
    # Obtaining the member '__getitem__' of a type (line 624)
    getitem___68373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 16), r_68372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 624)
    subscript_call_result_68374 = invoke(stypy.reporting.localization.Localization(__file__, 624, 16), getitem___68373, slice_68371)
    
    # Assigning a type to the variable 'r' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 12), 'r', subscript_call_result_68374)
    # SSA join for if statement (line 620)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    int_68375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 13), 'int')
    slice_68376 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 625, 11), int_68375, None, None)
    # Getting the type of 'r' (line 625)
    r_68377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 11), 'r')
    # Obtaining the member '__getitem__' of a type (line 625)
    getitem___68378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 625, 11), r_68377, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 625)
    subscript_call_result_68379 = invoke(stypy.reporting.localization.Localization(__file__, 625, 11), getitem___68378, slice_68376)
    
    str_68380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 625, 21), 'str', "'''")
    # Applying the binary operator '==' (line 625)
    result_eq_68381 = python_operator(stypy.reporting.localization.Localization(__file__, 625, 11), '==', subscript_call_result_68379, str_68380)
    
    # Testing the type of an if condition (line 625)
    if_condition_68382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 625, 8), result_eq_68381)
    # Assigning a type to the variable 'if_condition_68382' (line 625)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'if_condition_68382', if_condition_68382)
    # SSA begins for if statement (line 625)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'comment' (line 626)
    comment_68383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 15), 'comment')
    # Testing the type of an if condition (line 626)
    if_condition_68384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 626, 12), comment_68383)
    # Assigning a type to the variable 'if_condition_68384' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 12), 'if_condition_68384', if_condition_68384)
    # SSA begins for if statement (line 626)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 627):
    
    # Assigning a BinOp to a Name (line 627):
    
    # Obtaining the type of the subscript
    int_68385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 23), 'int')
    slice_68386 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 627, 20), None, int_68385, None)
    # Getting the type of 'r' (line 627)
    r_68387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 20), 'r')
    # Obtaining the member '__getitem__' of a type (line 627)
    getitem___68388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 20), r_68387, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 627)
    subscript_call_result_68389 = invoke(stypy.reporting.localization.Localization(__file__, 627, 20), getitem___68388, slice_68386)
    
    str_68390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 29), 'str', '\n\t/* end multiline (')
    # Applying the binary operator '+' (line 627)
    result_add_68391 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 20), '+', subscript_call_result_68389, str_68390)
    
    
    # Call to repr(...): (line 627)
    # Processing the call arguments (line 627)
    # Getting the type of 'counter' (line 627)
    counter_68393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 61), 'counter', False)
    # Processing the call keyword arguments (line 627)
    kwargs_68394 = {}
    # Getting the type of 'repr' (line 627)
    repr_68392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 56), 'repr', False)
    # Calling repr(args, kwargs) (line 627)
    repr_call_result_68395 = invoke(stypy.reporting.localization.Localization(__file__, 627, 56), repr_68392, *[counter_68393], **kwargs_68394)
    
    # Applying the binary operator '+' (line 627)
    result_add_68396 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 54), '+', result_add_68391, repr_call_result_68395)
    
    str_68397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 627, 72), 'str', ')*/')
    # Applying the binary operator '+' (line 627)
    result_add_68398 = python_operator(stypy.reporting.localization.Localization(__file__, 627, 70), '+', result_add_68396, str_68397)
    
    # Assigning a type to the variable 'r' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 16), 'r', result_add_68398)
    # SSA branch for the else part of an if statement (line 626)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 629):
    
    # Assigning a Subscript to a Name (line 629):
    
    # Obtaining the type of the subscript
    int_68399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 23), 'int')
    slice_68400 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 629, 20), None, int_68399, None)
    # Getting the type of 'r' (line 629)
    r_68401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 20), 'r')
    # Obtaining the member '__getitem__' of a type (line 629)
    getitem___68402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 20), r_68401, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 629)
    subscript_call_result_68403 = invoke(stypy.reporting.localization.Localization(__file__, 629, 20), getitem___68402, slice_68400)
    
    # Assigning a type to the variable 'r' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 16), 'r', subscript_call_result_68403)
    # SSA join for if statement (line 626)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 625)
    module_type_store.open_ssa_branch('else')
    
    # Call to errmess(...): (line 631)
    # Processing the call arguments (line 631)
    str_68405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 20), 'str', "%s multiline block should end with `'''`: %s\n")
    
    # Obtaining an instance of the builtin type 'tuple' (line 632)
    tuple_68406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 632)
    # Adding element type (line 632)
    # Getting the type of 'blockname' (line 632)
    blockname_68407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 23), 'blockname', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 23), tuple_68406, blockname_68407)
    # Adding element type (line 632)
    
    # Call to repr(...): (line 632)
    # Processing the call arguments (line 632)
    # Getting the type of 'r' (line 632)
    r_68409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 39), 'r', False)
    # Processing the call keyword arguments (line 632)
    kwargs_68410 = {}
    # Getting the type of 'repr' (line 632)
    repr_68408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 34), 'repr', False)
    # Calling repr(args, kwargs) (line 632)
    repr_call_result_68411 = invoke(stypy.reporting.localization.Localization(__file__, 632, 34), repr_68408, *[r_68409], **kwargs_68410)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 632, 23), tuple_68406, repr_call_result_68411)
    
    # Applying the binary operator '%' (line 631)
    result_mod_68412 = python_operator(stypy.reporting.localization.Localization(__file__, 631, 20), '%', str_68405, tuple_68406)
    
    # Processing the call keyword arguments (line 631)
    kwargs_68413 = {}
    # Getting the type of 'errmess' (line 631)
    errmess_68404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 12), 'errmess', False)
    # Calling errmess(args, kwargs) (line 631)
    errmess_call_result_68414 = invoke(stypy.reporting.localization.Localization(__file__, 631, 12), errmess_68404, *[result_mod_68412], **kwargs_68413)
    
    # SSA join for if statement (line 625)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 619)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'r' (line 633)
    r_68415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 11), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 633)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 4), 'stypy_return_type', r_68415)
    
    # ################# End of 'getmultilineblock(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getmultilineblock' in the type store
    # Getting the type of 'stypy_return_type' (line 606)
    stypy_return_type_68416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68416)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getmultilineblock'
    return stypy_return_type_68416

# Assigning a type to the variable 'getmultilineblock' (line 606)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 0), 'getmultilineblock', getmultilineblock)

@norecursion
def getcallstatement(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getcallstatement'
    module_type_store = module_type_store.open_function_context('getcallstatement', 636, 0, False)
    
    # Passed parameters checking function
    getcallstatement.stypy_localization = localization
    getcallstatement.stypy_type_of_self = None
    getcallstatement.stypy_type_store = module_type_store
    getcallstatement.stypy_function_name = 'getcallstatement'
    getcallstatement.stypy_param_names_list = ['rout']
    getcallstatement.stypy_varargs_param_name = None
    getcallstatement.stypy_kwargs_param_name = None
    getcallstatement.stypy_call_defaults = defaults
    getcallstatement.stypy_call_varargs = varargs
    getcallstatement.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getcallstatement', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getcallstatement', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getcallstatement(...)' code ##################

    
    # Call to getmultilineblock(...): (line 637)
    # Processing the call arguments (line 637)
    # Getting the type of 'rout' (line 637)
    rout_68418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 29), 'rout', False)
    str_68419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 35), 'str', 'callstatement')
    # Processing the call keyword arguments (line 637)
    kwargs_68420 = {}
    # Getting the type of 'getmultilineblock' (line 637)
    getmultilineblock_68417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 11), 'getmultilineblock', False)
    # Calling getmultilineblock(args, kwargs) (line 637)
    getmultilineblock_call_result_68421 = invoke(stypy.reporting.localization.Localization(__file__, 637, 11), getmultilineblock_68417, *[rout_68418, str_68419], **kwargs_68420)
    
    # Assigning a type to the variable 'stypy_return_type' (line 637)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 637, 4), 'stypy_return_type', getmultilineblock_call_result_68421)
    
    # ################# End of 'getcallstatement(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getcallstatement' in the type store
    # Getting the type of 'stypy_return_type' (line 636)
    stypy_return_type_68422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68422)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getcallstatement'
    return stypy_return_type_68422

# Assigning a type to the variable 'getcallstatement' (line 636)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 0), 'getcallstatement', getcallstatement)

@norecursion
def getcallprotoargument(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'dict' (line 640)
    dict_68423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 38), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 640)
    
    defaults = [dict_68423]
    # Create a new context for function 'getcallprotoargument'
    module_type_store = module_type_store.open_function_context('getcallprotoargument', 640, 0, False)
    
    # Passed parameters checking function
    getcallprotoargument.stypy_localization = localization
    getcallprotoargument.stypy_type_of_self = None
    getcallprotoargument.stypy_type_store = module_type_store
    getcallprotoargument.stypy_function_name = 'getcallprotoargument'
    getcallprotoargument.stypy_param_names_list = ['rout', 'cb_map']
    getcallprotoargument.stypy_varargs_param_name = None
    getcallprotoargument.stypy_kwargs_param_name = None
    getcallprotoargument.stypy_call_defaults = defaults
    getcallprotoargument.stypy_call_varargs = varargs
    getcallprotoargument.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getcallprotoargument', ['rout', 'cb_map'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getcallprotoargument', localization, ['rout', 'cb_map'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getcallprotoargument(...)' code ##################

    
    # Assigning a Call to a Name (line 641):
    
    # Assigning a Call to a Name (line 641):
    
    # Call to getmultilineblock(...): (line 641)
    # Processing the call arguments (line 641)
    # Getting the type of 'rout' (line 641)
    rout_68425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 26), 'rout', False)
    str_68426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 32), 'str', 'callprotoargument')
    # Processing the call keyword arguments (line 641)
    int_68427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 641, 61), 'int')
    keyword_68428 = int_68427
    kwargs_68429 = {'comment': keyword_68428}
    # Getting the type of 'getmultilineblock' (line 641)
    getmultilineblock_68424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'getmultilineblock', False)
    # Calling getmultilineblock(args, kwargs) (line 641)
    getmultilineblock_call_result_68430 = invoke(stypy.reporting.localization.Localization(__file__, 641, 8), getmultilineblock_68424, *[rout_68425, str_68426], **kwargs_68429)
    
    # Assigning a type to the variable 'r' (line 641)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 4), 'r', getmultilineblock_call_result_68430)
    
    # Getting the type of 'r' (line 642)
    r_68431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 7), 'r')
    # Testing the type of an if condition (line 642)
    if_condition_68432 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 642, 4), r_68431)
    # Assigning a type to the variable 'if_condition_68432' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 4), 'if_condition_68432', if_condition_68432)
    # SSA begins for if statement (line 642)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'r' (line 643)
    r_68433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 15), 'r')
    # Assigning a type to the variable 'stypy_return_type' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'stypy_return_type', r_68433)
    # SSA join for if statement (line 642)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to hascallstatement(...): (line 644)
    # Processing the call arguments (line 644)
    # Getting the type of 'rout' (line 644)
    rout_68435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 24), 'rout', False)
    # Processing the call keyword arguments (line 644)
    kwargs_68436 = {}
    # Getting the type of 'hascallstatement' (line 644)
    hascallstatement_68434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 7), 'hascallstatement', False)
    # Calling hascallstatement(args, kwargs) (line 644)
    hascallstatement_call_result_68437 = invoke(stypy.reporting.localization.Localization(__file__, 644, 7), hascallstatement_68434, *[rout_68435], **kwargs_68436)
    
    # Testing the type of an if condition (line 644)
    if_condition_68438 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 644, 4), hascallstatement_call_result_68437)
    # Assigning a type to the variable 'if_condition_68438' (line 644)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 644, 4), 'if_condition_68438', if_condition_68438)
    # SSA begins for if statement (line 644)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to outmess(...): (line 645)
    # Processing the call arguments (line 645)
    str_68440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 646, 12), 'str', 'warning: callstatement is defined without callprotoargument\n')
    # Processing the call keyword arguments (line 645)
    kwargs_68441 = {}
    # Getting the type of 'outmess' (line 645)
    outmess_68439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'outmess', False)
    # Calling outmess(args, kwargs) (line 645)
    outmess_call_result_68442 = invoke(stypy.reporting.localization.Localization(__file__, 645, 8), outmess_68439, *[str_68440], **kwargs_68441)
    
    # Assigning a type to the variable 'stypy_return_type' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 644)
    module_type_store = module_type_store.join_ssa_context()
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 648, 4))
    
    # 'from numpy.f2py.capi_maps import getctype' statement (line 648)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_68443 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 648, 4), 'numpy.f2py.capi_maps')

    if (type(import_68443) is not StypyTypeError):

        if (import_68443 != 'pyd_module'):
            __import__(import_68443)
            sys_modules_68444 = sys.modules[import_68443]
            import_from_module(stypy.reporting.localization.Localization(__file__, 648, 4), 'numpy.f2py.capi_maps', sys_modules_68444.module_type_store, module_type_store, ['getctype'])
            nest_module(stypy.reporting.localization.Localization(__file__, 648, 4), __file__, sys_modules_68444, sys_modules_68444.module_type_store, module_type_store)
        else:
            from numpy.f2py.capi_maps import getctype

            import_from_module(stypy.reporting.localization.Localization(__file__, 648, 4), 'numpy.f2py.capi_maps', None, module_type_store, ['getctype'], [getctype])

    else:
        # Assigning a type to the variable 'numpy.f2py.capi_maps' (line 648)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 648, 4), 'numpy.f2py.capi_maps', import_68443)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Assigning a Tuple to a Tuple (line 649):
    
    # Assigning a List to a Name (line 649):
    
    # Obtaining an instance of the builtin type 'list' (line 649)
    list_68445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 649)
    
    # Assigning a type to the variable 'tuple_assignment_66680' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_assignment_66680', list_68445)
    
    # Assigning a List to a Name (line 649):
    
    # Obtaining an instance of the builtin type 'list' (line 649)
    list_68446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 649, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 649)
    
    # Assigning a type to the variable 'tuple_assignment_66681' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_assignment_66681', list_68446)
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_assignment_66680' (line 649)
    tuple_assignment_66680_68447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_assignment_66680')
    # Assigning a type to the variable 'arg_types' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'arg_types', tuple_assignment_66680_68447)
    
    # Assigning a Name to a Name (line 649):
    # Getting the type of 'tuple_assignment_66681' (line 649)
    tuple_assignment_66681_68448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 4), 'tuple_assignment_66681')
    # Assigning a type to the variable 'arg_types2' (line 649)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 649, 15), 'arg_types2', tuple_assignment_66681_68448)
    
    
    # Call to (...): (line 650)
    # Processing the call arguments (line 650)
    # Getting the type of 'rout' (line 650)
    rout_68457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 55), 'rout', False)
    # Processing the call keyword arguments (line 650)
    kwargs_68458 = {}
    
    # Call to l_and(...): (line 650)
    # Processing the call arguments (line 650)
    # Getting the type of 'isstringfunction' (line 650)
    isstringfunction_68450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 13), 'isstringfunction', False)
    
    # Call to l_not(...): (line 650)
    # Processing the call arguments (line 650)
    # Getting the type of 'isfunction_wrap' (line 650)
    isfunction_wrap_68452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 37), 'isfunction_wrap', False)
    # Processing the call keyword arguments (line 650)
    kwargs_68453 = {}
    # Getting the type of 'l_not' (line 650)
    l_not_68451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 31), 'l_not', False)
    # Calling l_not(args, kwargs) (line 650)
    l_not_call_result_68454 = invoke(stypy.reporting.localization.Localization(__file__, 650, 31), l_not_68451, *[isfunction_wrap_68452], **kwargs_68453)
    
    # Processing the call keyword arguments (line 650)
    kwargs_68455 = {}
    # Getting the type of 'l_and' (line 650)
    l_and_68449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 7), 'l_and', False)
    # Calling l_and(args, kwargs) (line 650)
    l_and_call_result_68456 = invoke(stypy.reporting.localization.Localization(__file__, 650, 7), l_and_68449, *[isstringfunction_68450, l_not_call_result_68454], **kwargs_68455)
    
    # Calling (args, kwargs) (line 650)
    _call_result_68459 = invoke(stypy.reporting.localization.Localization(__file__, 650, 7), l_and_call_result_68456, *[rout_68457], **kwargs_68458)
    
    # Testing the type of an if condition (line 650)
    if_condition_68460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 650, 4), _call_result_68459)
    # Assigning a type to the variable 'if_condition_68460' (line 650)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 650, 4), 'if_condition_68460', if_condition_68460)
    # SSA begins for if statement (line 650)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to extend(...): (line 651)
    # Processing the call arguments (line 651)
    
    # Obtaining an instance of the builtin type 'list' (line 651)
    list_68463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 651)
    # Adding element type (line 651)
    str_68464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 26), 'str', 'char*')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 651, 25), list_68463, str_68464)
    # Adding element type (line 651)
    str_68465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 651, 35), 'str', 'size_t')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 651, 25), list_68463, str_68465)
    
    # Processing the call keyword arguments (line 651)
    kwargs_68466 = {}
    # Getting the type of 'arg_types' (line 651)
    arg_types_68461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 651, 8), 'arg_types', False)
    # Obtaining the member 'extend' of a type (line 651)
    extend_68462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 651, 8), arg_types_68461, 'extend')
    # Calling extend(args, kwargs) (line 651)
    extend_call_result_68467 = invoke(stypy.reporting.localization.Localization(__file__, 651, 8), extend_68462, *[list_68463], **kwargs_68466)
    
    # SSA join for if statement (line 650)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    str_68468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 18), 'str', 'args')
    # Getting the type of 'rout' (line 652)
    rout_68469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 652, 13), 'rout')
    # Obtaining the member '__getitem__' of a type (line 652)
    getitem___68470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 652, 13), rout_68469, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 652)
    subscript_call_result_68471 = invoke(stypy.reporting.localization.Localization(__file__, 652, 13), getitem___68470, str_68468)
    
    # Testing the type of a for loop iterable (line 652)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 652, 4), subscript_call_result_68471)
    # Getting the type of the for loop variable (line 652)
    for_loop_var_68472 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 652, 4), subscript_call_result_68471)
    # Assigning a type to the variable 'n' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 4), 'n', for_loop_var_68472)
    # SSA begins for a for statement (line 652)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 653):
    
    # Assigning a Subscript to a Name (line 653):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 653)
    n_68473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 27), 'n')
    
    # Obtaining the type of the subscript
    str_68474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 653, 19), 'str', 'vars')
    # Getting the type of 'rout' (line 653)
    rout_68475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 14), 'rout')
    # Obtaining the member '__getitem__' of a type (line 653)
    getitem___68476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 14), rout_68475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 653)
    subscript_call_result_68477 = invoke(stypy.reporting.localization.Localization(__file__, 653, 14), getitem___68476, str_68474)
    
    # Obtaining the member '__getitem__' of a type (line 653)
    getitem___68478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 653, 14), subscript_call_result_68477, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 653)
    subscript_call_result_68479 = invoke(stypy.reporting.localization.Localization(__file__, 653, 14), getitem___68478, n_68473)
    
    # Assigning a type to the variable 'var' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'var', subscript_call_result_68479)
    
    
    # Call to isintent_callback(...): (line 654)
    # Processing the call arguments (line 654)
    # Getting the type of 'var' (line 654)
    var_68481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 29), 'var', False)
    # Processing the call keyword arguments (line 654)
    kwargs_68482 = {}
    # Getting the type of 'isintent_callback' (line 654)
    isintent_callback_68480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 11), 'isintent_callback', False)
    # Calling isintent_callback(args, kwargs) (line 654)
    isintent_callback_call_result_68483 = invoke(stypy.reporting.localization.Localization(__file__, 654, 11), isintent_callback_68480, *[var_68481], **kwargs_68482)
    
    # Testing the type of an if condition (line 654)
    if_condition_68484 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 654, 8), isintent_callback_call_result_68483)
    # Assigning a type to the variable 'if_condition_68484' (line 654)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 654, 8), 'if_condition_68484', if_condition_68484)
    # SSA begins for if statement (line 654)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 654)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'n' (line 656)
    n_68485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 11), 'n')
    # Getting the type of 'cb_map' (line 656)
    cb_map_68486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 656, 16), 'cb_map')
    # Applying the binary operator 'in' (line 656)
    result_contains_68487 = python_operator(stypy.reporting.localization.Localization(__file__, 656, 11), 'in', n_68485, cb_map_68486)
    
    # Testing the type of an if condition (line 656)
    if_condition_68488 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 656, 8), result_contains_68487)
    # Assigning a type to the variable 'if_condition_68488' (line 656)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 656, 8), 'if_condition_68488', if_condition_68488)
    # SSA begins for if statement (line 656)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 657):
    
    # Assigning a BinOp to a Name (line 657):
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 657)
    n_68489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 27), 'n')
    # Getting the type of 'cb_map' (line 657)
    cb_map_68490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 20), 'cb_map')
    # Obtaining the member '__getitem__' of a type (line 657)
    getitem___68491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 657, 20), cb_map_68490, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 657)
    subscript_call_result_68492 = invoke(stypy.reporting.localization.Localization(__file__, 657, 20), getitem___68491, n_68489)
    
    str_68493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 32), 'str', '_typedef')
    # Applying the binary operator '+' (line 657)
    result_add_68494 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 20), '+', subscript_call_result_68492, str_68493)
    
    # Assigning a type to the variable 'ctype' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'ctype', result_add_68494)
    # SSA branch for the else part of an if statement (line 656)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 659):
    
    # Assigning a Call to a Name (line 659):
    
    # Call to getctype(...): (line 659)
    # Processing the call arguments (line 659)
    # Getting the type of 'var' (line 659)
    var_68496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 29), 'var', False)
    # Processing the call keyword arguments (line 659)
    kwargs_68497 = {}
    # Getting the type of 'getctype' (line 659)
    getctype_68495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 659, 20), 'getctype', False)
    # Calling getctype(args, kwargs) (line 659)
    getctype_call_result_68498 = invoke(stypy.reporting.localization.Localization(__file__, 659, 20), getctype_68495, *[var_68496], **kwargs_68497)
    
    # Assigning a type to the variable 'ctype' (line 659)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 659, 12), 'ctype', getctype_call_result_68498)
    
    
    # Call to (...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'var' (line 660)
    var_68508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 60), 'var', False)
    # Processing the call keyword arguments (line 660)
    kwargs_68509 = {}
    
    # Call to l_and(...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'isintent_c' (line 660)
    isintent_c_68500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 21), 'isintent_c', False)
    
    # Call to l_or(...): (line 660)
    # Processing the call arguments (line 660)
    # Getting the type of 'isscalar' (line 660)
    isscalar_68502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 38), 'isscalar', False)
    # Getting the type of 'iscomplex' (line 660)
    iscomplex_68503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 48), 'iscomplex', False)
    # Processing the call keyword arguments (line 660)
    kwargs_68504 = {}
    # Getting the type of 'l_or' (line 660)
    l_or_68501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 33), 'l_or', False)
    # Calling l_or(args, kwargs) (line 660)
    l_or_call_result_68505 = invoke(stypy.reporting.localization.Localization(__file__, 660, 33), l_or_68501, *[isscalar_68502, iscomplex_68503], **kwargs_68504)
    
    # Processing the call keyword arguments (line 660)
    kwargs_68506 = {}
    # Getting the type of 'l_and' (line 660)
    l_and_68499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 15), 'l_and', False)
    # Calling l_and(args, kwargs) (line 660)
    l_and_call_result_68507 = invoke(stypy.reporting.localization.Localization(__file__, 660, 15), l_and_68499, *[isintent_c_68500, l_or_call_result_68505], **kwargs_68506)
    
    # Calling (args, kwargs) (line 660)
    _call_result_68510 = invoke(stypy.reporting.localization.Localization(__file__, 660, 15), l_and_call_result_68507, *[var_68508], **kwargs_68509)
    
    # Testing the type of an if condition (line 660)
    if_condition_68511 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 660, 12), _call_result_68510)
    # Assigning a type to the variable 'if_condition_68511' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 12), 'if_condition_68511', if_condition_68511)
    # SSA begins for if statement (line 660)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 660)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isstring(...): (line 662)
    # Processing the call arguments (line 662)
    # Getting the type of 'var' (line 662)
    var_68513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 26), 'var', False)
    # Processing the call keyword arguments (line 662)
    kwargs_68514 = {}
    # Getting the type of 'isstring' (line 662)
    isstring_68512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 662, 17), 'isstring', False)
    # Calling isstring(args, kwargs) (line 662)
    isstring_call_result_68515 = invoke(stypy.reporting.localization.Localization(__file__, 662, 17), isstring_68512, *[var_68513], **kwargs_68514)
    
    # Testing the type of an if condition (line 662)
    if_condition_68516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 662, 17), isstring_call_result_68515)
    # Assigning a type to the variable 'if_condition_68516' (line 662)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 662, 17), 'if_condition_68516', if_condition_68516)
    # SSA begins for if statement (line 662)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 662)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 665):
    
    # Assigning a BinOp to a Name (line 665):
    # Getting the type of 'ctype' (line 665)
    ctype_68517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 665, 24), 'ctype')
    str_68518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 665, 32), 'str', '*')
    # Applying the binary operator '+' (line 665)
    result_add_68519 = python_operator(stypy.reporting.localization.Localization(__file__, 665, 24), '+', ctype_68517, str_68518)
    
    # Assigning a type to the variable 'ctype' (line 665)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 665, 16), 'ctype', result_add_68519)
    # SSA join for if statement (line 662)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 660)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Call to isstring(...): (line 666)
    # Processing the call arguments (line 666)
    # Getting the type of 'var' (line 666)
    var_68521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 24), 'var', False)
    # Processing the call keyword arguments (line 666)
    kwargs_68522 = {}
    # Getting the type of 'isstring' (line 666)
    isstring_68520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 15), 'isstring', False)
    # Calling isstring(args, kwargs) (line 666)
    isstring_call_result_68523 = invoke(stypy.reporting.localization.Localization(__file__, 666, 15), isstring_68520, *[var_68521], **kwargs_68522)
    
    
    # Call to isarrayofstrings(...): (line 666)
    # Processing the call arguments (line 666)
    # Getting the type of 'var' (line 666)
    var_68525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 49), 'var', False)
    # Processing the call keyword arguments (line 666)
    kwargs_68526 = {}
    # Getting the type of 'isarrayofstrings' (line 666)
    isarrayofstrings_68524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 32), 'isarrayofstrings', False)
    # Calling isarrayofstrings(args, kwargs) (line 666)
    isarrayofstrings_call_result_68527 = invoke(stypy.reporting.localization.Localization(__file__, 666, 32), isarrayofstrings_68524, *[var_68525], **kwargs_68526)
    
    # Applying the binary operator 'or' (line 666)
    result_or_keyword_68528 = python_operator(stypy.reporting.localization.Localization(__file__, 666, 15), 'or', isstring_call_result_68523, isarrayofstrings_call_result_68527)
    
    # Testing the type of an if condition (line 666)
    if_condition_68529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 666, 12), result_or_keyword_68528)
    # Assigning a type to the variable 'if_condition_68529' (line 666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 12), 'if_condition_68529', if_condition_68529)
    # SSA begins for if statement (line 666)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 667)
    # Processing the call arguments (line 667)
    str_68532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 667, 34), 'str', 'size_t')
    # Processing the call keyword arguments (line 667)
    kwargs_68533 = {}
    # Getting the type of 'arg_types2' (line 667)
    arg_types2_68530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 16), 'arg_types2', False)
    # Obtaining the member 'append' of a type (line 667)
    append_68531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 16), arg_types2_68530, 'append')
    # Calling append(args, kwargs) (line 667)
    append_call_result_68534 = invoke(stypy.reporting.localization.Localization(__file__, 667, 16), append_68531, *[str_68532], **kwargs_68533)
    
    # SSA join for if statement (line 666)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 656)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 668)
    # Processing the call arguments (line 668)
    # Getting the type of 'ctype' (line 668)
    ctype_68537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 25), 'ctype', False)
    # Processing the call keyword arguments (line 668)
    kwargs_68538 = {}
    # Getting the type of 'arg_types' (line 668)
    arg_types_68535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 668, 8), 'arg_types', False)
    # Obtaining the member 'append' of a type (line 668)
    append_68536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 668, 8), arg_types_68535, 'append')
    # Calling append(args, kwargs) (line 668)
    append_call_result_68539 = invoke(stypy.reporting.localization.Localization(__file__, 668, 8), append_68536, *[ctype_68537], **kwargs_68538)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 670):
    
    # Assigning a Call to a Name (line 670):
    
    # Call to join(...): (line 670)
    # Processing the call arguments (line 670)
    # Getting the type of 'arg_types' (line 670)
    arg_types_68542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 26), 'arg_types', False)
    # Getting the type of 'arg_types2' (line 670)
    arg_types2_68543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 670, 38), 'arg_types2', False)
    # Applying the binary operator '+' (line 670)
    result_add_68544 = python_operator(stypy.reporting.localization.Localization(__file__, 670, 26), '+', arg_types_68542, arg_types2_68543)
    
    # Processing the call keyword arguments (line 670)
    kwargs_68545 = {}
    str_68540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 670, 17), 'str', ',')
    # Obtaining the member 'join' of a type (line 670)
    join_68541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 670, 17), str_68540, 'join')
    # Calling join(args, kwargs) (line 670)
    join_call_result_68546 = invoke(stypy.reporting.localization.Localization(__file__, 670, 17), join_68541, *[result_add_68544], **kwargs_68545)
    
    # Assigning a type to the variable 'proto_args' (line 670)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 670, 4), 'proto_args', join_call_result_68546)
    
    
    # Getting the type of 'proto_args' (line 671)
    proto_args_68547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 11), 'proto_args')
    # Applying the 'not' unary operator (line 671)
    result_not__68548 = python_operator(stypy.reporting.localization.Localization(__file__, 671, 7), 'not', proto_args_68547)
    
    # Testing the type of an if condition (line 671)
    if_condition_68549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 671, 4), result_not__68548)
    # Assigning a type to the variable 'if_condition_68549' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 4), 'if_condition_68549', if_condition_68549)
    # SSA begins for if statement (line 671)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 672):
    
    # Assigning a Str to a Name (line 672):
    str_68550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 21), 'str', 'void')
    # Assigning a type to the variable 'proto_args' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'proto_args', str_68550)
    # SSA join for if statement (line 671)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'proto_args' (line 673)
    proto_args_68551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 673, 11), 'proto_args')
    # Assigning a type to the variable 'stypy_return_type' (line 673)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 673, 4), 'stypy_return_type', proto_args_68551)
    
    # ################# End of 'getcallprotoargument(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getcallprotoargument' in the type store
    # Getting the type of 'stypy_return_type' (line 640)
    stypy_return_type_68552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68552)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getcallprotoargument'
    return stypy_return_type_68552

# Assigning a type to the variable 'getcallprotoargument' (line 640)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 0), 'getcallprotoargument', getcallprotoargument)

@norecursion
def getusercode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getusercode'
    module_type_store = module_type_store.open_function_context('getusercode', 676, 0, False)
    
    # Passed parameters checking function
    getusercode.stypy_localization = localization
    getusercode.stypy_type_of_self = None
    getusercode.stypy_type_store = module_type_store
    getusercode.stypy_function_name = 'getusercode'
    getusercode.stypy_param_names_list = ['rout']
    getusercode.stypy_varargs_param_name = None
    getusercode.stypy_kwargs_param_name = None
    getusercode.stypy_call_defaults = defaults
    getusercode.stypy_call_varargs = varargs
    getusercode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getusercode', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getusercode', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getusercode(...)' code ##################

    
    # Call to getmultilineblock(...): (line 677)
    # Processing the call arguments (line 677)
    # Getting the type of 'rout' (line 677)
    rout_68554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 29), 'rout', False)
    str_68555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 677, 35), 'str', 'usercode')
    # Processing the call keyword arguments (line 677)
    kwargs_68556 = {}
    # Getting the type of 'getmultilineblock' (line 677)
    getmultilineblock_68553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 677, 11), 'getmultilineblock', False)
    # Calling getmultilineblock(args, kwargs) (line 677)
    getmultilineblock_call_result_68557 = invoke(stypy.reporting.localization.Localization(__file__, 677, 11), getmultilineblock_68553, *[rout_68554, str_68555], **kwargs_68556)
    
    # Assigning a type to the variable 'stypy_return_type' (line 677)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 677, 4), 'stypy_return_type', getmultilineblock_call_result_68557)
    
    # ################# End of 'getusercode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getusercode' in the type store
    # Getting the type of 'stypy_return_type' (line 676)
    stypy_return_type_68558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 676, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68558)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getusercode'
    return stypy_return_type_68558

# Assigning a type to the variable 'getusercode' (line 676)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 0), 'getusercode', getusercode)

@norecursion
def getusercode1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getusercode1'
    module_type_store = module_type_store.open_function_context('getusercode1', 680, 0, False)
    
    # Passed parameters checking function
    getusercode1.stypy_localization = localization
    getusercode1.stypy_type_of_self = None
    getusercode1.stypy_type_store = module_type_store
    getusercode1.stypy_function_name = 'getusercode1'
    getusercode1.stypy_param_names_list = ['rout']
    getusercode1.stypy_varargs_param_name = None
    getusercode1.stypy_kwargs_param_name = None
    getusercode1.stypy_call_defaults = defaults
    getusercode1.stypy_call_varargs = varargs
    getusercode1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getusercode1', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getusercode1', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getusercode1(...)' code ##################

    
    # Call to getmultilineblock(...): (line 681)
    # Processing the call arguments (line 681)
    # Getting the type of 'rout' (line 681)
    rout_68560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 29), 'rout', False)
    str_68561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 35), 'str', 'usercode')
    # Processing the call keyword arguments (line 681)
    int_68562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 55), 'int')
    keyword_68563 = int_68562
    kwargs_68564 = {'counter': keyword_68563}
    # Getting the type of 'getmultilineblock' (line 681)
    getmultilineblock_68559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 11), 'getmultilineblock', False)
    # Calling getmultilineblock(args, kwargs) (line 681)
    getmultilineblock_call_result_68565 = invoke(stypy.reporting.localization.Localization(__file__, 681, 11), getmultilineblock_68559, *[rout_68560, str_68561], **kwargs_68564)
    
    # Assigning a type to the variable 'stypy_return_type' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 4), 'stypy_return_type', getmultilineblock_call_result_68565)
    
    # ################# End of 'getusercode1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getusercode1' in the type store
    # Getting the type of 'stypy_return_type' (line 680)
    stypy_return_type_68566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 680, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68566)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getusercode1'
    return stypy_return_type_68566

# Assigning a type to the variable 'getusercode1' (line 680)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 680, 0), 'getusercode1', getusercode1)

@norecursion
def getpymethoddef(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getpymethoddef'
    module_type_store = module_type_store.open_function_context('getpymethoddef', 684, 0, False)
    
    # Passed parameters checking function
    getpymethoddef.stypy_localization = localization
    getpymethoddef.stypy_type_of_self = None
    getpymethoddef.stypy_type_store = module_type_store
    getpymethoddef.stypy_function_name = 'getpymethoddef'
    getpymethoddef.stypy_param_names_list = ['rout']
    getpymethoddef.stypy_varargs_param_name = None
    getpymethoddef.stypy_kwargs_param_name = None
    getpymethoddef.stypy_call_defaults = defaults
    getpymethoddef.stypy_call_varargs = varargs
    getpymethoddef.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getpymethoddef', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getpymethoddef', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getpymethoddef(...)' code ##################

    
    # Call to getmultilineblock(...): (line 685)
    # Processing the call arguments (line 685)
    # Getting the type of 'rout' (line 685)
    rout_68568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 29), 'rout', False)
    str_68569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 35), 'str', 'pymethoddef')
    # Processing the call keyword arguments (line 685)
    kwargs_68570 = {}
    # Getting the type of 'getmultilineblock' (line 685)
    getmultilineblock_68567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 11), 'getmultilineblock', False)
    # Calling getmultilineblock(args, kwargs) (line 685)
    getmultilineblock_call_result_68571 = invoke(stypy.reporting.localization.Localization(__file__, 685, 11), getmultilineblock_68567, *[rout_68568, str_68569], **kwargs_68570)
    
    # Assigning a type to the variable 'stypy_return_type' (line 685)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'stypy_return_type', getmultilineblock_call_result_68571)
    
    # ################# End of 'getpymethoddef(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getpymethoddef' in the type store
    # Getting the type of 'stypy_return_type' (line 684)
    stypy_return_type_68572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68572)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getpymethoddef'
    return stypy_return_type_68572

# Assigning a type to the variable 'getpymethoddef' (line 684)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 0), 'getpymethoddef', getpymethoddef)

@norecursion
def getargs(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getargs'
    module_type_store = module_type_store.open_function_context('getargs', 688, 0, False)
    
    # Passed parameters checking function
    getargs.stypy_localization = localization
    getargs.stypy_type_of_self = None
    getargs.stypy_type_store = module_type_store
    getargs.stypy_function_name = 'getargs'
    getargs.stypy_param_names_list = ['rout']
    getargs.stypy_varargs_param_name = None
    getargs.stypy_kwargs_param_name = None
    getargs.stypy_call_defaults = defaults
    getargs.stypy_call_varargs = varargs
    getargs.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getargs', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getargs', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getargs(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 689):
    
    # Assigning a List to a Name (line 689):
    
    # Obtaining an instance of the builtin type 'list' (line 689)
    list_68573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 689)
    
    # Assigning a type to the variable 'tuple_assignment_66682' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'tuple_assignment_66682', list_68573)
    
    # Assigning a List to a Name (line 689):
    
    # Obtaining an instance of the builtin type 'list' (line 689)
    list_68574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 689)
    
    # Assigning a type to the variable 'tuple_assignment_66683' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'tuple_assignment_66683', list_68574)
    
    # Assigning a Name to a Name (line 689):
    # Getting the type of 'tuple_assignment_66682' (line 689)
    tuple_assignment_66682_68575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'tuple_assignment_66682')
    # Assigning a type to the variable 'sortargs' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'sortargs', tuple_assignment_66682_68575)
    
    # Assigning a Name to a Name (line 689):
    # Getting the type of 'tuple_assignment_66683' (line 689)
    tuple_assignment_66683_68576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'tuple_assignment_66683')
    # Assigning a type to the variable 'args' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 14), 'args', tuple_assignment_66683_68576)
    
    
    str_68577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 7), 'str', 'args')
    # Getting the type of 'rout' (line 690)
    rout_68578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 17), 'rout')
    # Applying the binary operator 'in' (line 690)
    result_contains_68579 = python_operator(stypy.reporting.localization.Localization(__file__, 690, 7), 'in', str_68577, rout_68578)
    
    # Testing the type of an if condition (line 690)
    if_condition_68580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 690, 4), result_contains_68579)
    # Assigning a type to the variable 'if_condition_68580' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'if_condition_68580', if_condition_68580)
    # SSA begins for if statement (line 690)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 691):
    
    # Assigning a Subscript to a Name (line 691):
    
    # Obtaining the type of the subscript
    str_68581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 691, 20), 'str', 'args')
    # Getting the type of 'rout' (line 691)
    rout_68582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 691, 15), 'rout')
    # Obtaining the member '__getitem__' of a type (line 691)
    getitem___68583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 691, 15), rout_68582, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 691)
    subscript_call_result_68584 = invoke(stypy.reporting.localization.Localization(__file__, 691, 15), getitem___68583, str_68581)
    
    # Assigning a type to the variable 'args' (line 691)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 691, 8), 'args', subscript_call_result_68584)
    
    
    str_68585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 11), 'str', 'sortvars')
    # Getting the type of 'rout' (line 692)
    rout_68586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 25), 'rout')
    # Applying the binary operator 'in' (line 692)
    result_contains_68587 = python_operator(stypy.reporting.localization.Localization(__file__, 692, 11), 'in', str_68585, rout_68586)
    
    # Testing the type of an if condition (line 692)
    if_condition_68588 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 692, 8), result_contains_68587)
    # Assigning a type to the variable 'if_condition_68588' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 8), 'if_condition_68588', if_condition_68588)
    # SSA begins for if statement (line 692)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    str_68589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 26), 'str', 'sortvars')
    # Getting the type of 'rout' (line 693)
    rout_68590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 21), 'rout')
    # Obtaining the member '__getitem__' of a type (line 693)
    getitem___68591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 21), rout_68590, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 693)
    subscript_call_result_68592 = invoke(stypy.reporting.localization.Localization(__file__, 693, 21), getitem___68591, str_68589)
    
    # Testing the type of a for loop iterable (line 693)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 693, 12), subscript_call_result_68592)
    # Getting the type of the for loop variable (line 693)
    for_loop_var_68593 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 693, 12), subscript_call_result_68592)
    # Assigning a type to the variable 'a' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 12), 'a', for_loop_var_68593)
    # SSA begins for a for statement (line 693)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a' (line 694)
    a_68594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 19), 'a')
    # Getting the type of 'args' (line 694)
    args_68595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 24), 'args')
    # Applying the binary operator 'in' (line 694)
    result_contains_68596 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 19), 'in', a_68594, args_68595)
    
    # Testing the type of an if condition (line 694)
    if_condition_68597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 694, 16), result_contains_68596)
    # Assigning a type to the variable 'if_condition_68597' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 16), 'if_condition_68597', if_condition_68597)
    # SSA begins for if statement (line 694)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 695)
    # Processing the call arguments (line 695)
    # Getting the type of 'a' (line 695)
    a_68600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 36), 'a', False)
    # Processing the call keyword arguments (line 695)
    kwargs_68601 = {}
    # Getting the type of 'sortargs' (line 695)
    sortargs_68598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 20), 'sortargs', False)
    # Obtaining the member 'append' of a type (line 695)
    append_68599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 20), sortargs_68598, 'append')
    # Calling append(args, kwargs) (line 695)
    append_call_result_68602 = invoke(stypy.reporting.localization.Localization(__file__, 695, 20), append_68599, *[a_68600], **kwargs_68601)
    
    # SSA join for if statement (line 694)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'args' (line 696)
    args_68603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 21), 'args')
    # Testing the type of a for loop iterable (line 696)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 696, 12), args_68603)
    # Getting the type of the for loop variable (line 696)
    for_loop_var_68604 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 696, 12), args_68603)
    # Assigning a type to the variable 'a' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 12), 'a', for_loop_var_68604)
    # SSA begins for a for statement (line 696)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a' (line 697)
    a_68605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 19), 'a')
    # Getting the type of 'sortargs' (line 697)
    sortargs_68606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 697, 28), 'sortargs')
    # Applying the binary operator 'notin' (line 697)
    result_contains_68607 = python_operator(stypy.reporting.localization.Localization(__file__, 697, 19), 'notin', a_68605, sortargs_68606)
    
    # Testing the type of an if condition (line 697)
    if_condition_68608 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 697, 16), result_contains_68607)
    # Assigning a type to the variable 'if_condition_68608' (line 697)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 697, 16), 'if_condition_68608', if_condition_68608)
    # SSA begins for if statement (line 697)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 698)
    # Processing the call arguments (line 698)
    # Getting the type of 'a' (line 698)
    a_68611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 36), 'a', False)
    # Processing the call keyword arguments (line 698)
    kwargs_68612 = {}
    # Getting the type of 'sortargs' (line 698)
    sortargs_68609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 20), 'sortargs', False)
    # Obtaining the member 'append' of a type (line 698)
    append_68610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 698, 20), sortargs_68609, 'append')
    # Calling append(args, kwargs) (line 698)
    append_call_result_68613 = invoke(stypy.reporting.localization.Localization(__file__, 698, 20), append_68610, *[a_68611], **kwargs_68612)
    
    # SSA join for if statement (line 697)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 692)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 700):
    
    # Assigning a Subscript to a Name (line 700):
    
    # Obtaining the type of the subscript
    str_68614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 700, 28), 'str', 'args')
    # Getting the type of 'rout' (line 700)
    rout_68615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 700, 23), 'rout')
    # Obtaining the member '__getitem__' of a type (line 700)
    getitem___68616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 700, 23), rout_68615, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 700)
    subscript_call_result_68617 = invoke(stypy.reporting.localization.Localization(__file__, 700, 23), getitem___68616, str_68614)
    
    # Assigning a type to the variable 'sortargs' (line 700)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 700, 12), 'sortargs', subscript_call_result_68617)
    # SSA join for if statement (line 692)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 690)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 701)
    tuple_68618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 701, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 701)
    # Adding element type (line 701)
    # Getting the type of 'args' (line 701)
    args_68619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 11), 'args')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 701, 11), tuple_68618, args_68619)
    # Adding element type (line 701)
    # Getting the type of 'sortargs' (line 701)
    sortargs_68620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 701, 17), 'sortargs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 701, 11), tuple_68618, sortargs_68620)
    
    # Assigning a type to the variable 'stypy_return_type' (line 701)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 701, 4), 'stypy_return_type', tuple_68618)
    
    # ################# End of 'getargs(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getargs' in the type store
    # Getting the type of 'stypy_return_type' (line 688)
    stypy_return_type_68621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68621)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getargs'
    return stypy_return_type_68621

# Assigning a type to the variable 'getargs' (line 688)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 0), 'getargs', getargs)

@norecursion
def getargs2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getargs2'
    module_type_store = module_type_store.open_function_context('getargs2', 704, 0, False)
    
    # Passed parameters checking function
    getargs2.stypy_localization = localization
    getargs2.stypy_type_of_self = None
    getargs2.stypy_type_store = module_type_store
    getargs2.stypy_function_name = 'getargs2'
    getargs2.stypy_param_names_list = ['rout']
    getargs2.stypy_varargs_param_name = None
    getargs2.stypy_kwargs_param_name = None
    getargs2.stypy_call_defaults = defaults
    getargs2.stypy_call_varargs = varargs
    getargs2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getargs2', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getargs2', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getargs2(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 705):
    
    # Assigning a List to a Name (line 705):
    
    # Obtaining an instance of the builtin type 'list' (line 705)
    list_68622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 705)
    
    # Assigning a type to the variable 'tuple_assignment_66684' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'tuple_assignment_66684', list_68622)
    
    # Assigning a Call to a Name (line 705):
    
    # Call to get(...): (line 705)
    # Processing the call arguments (line 705)
    str_68625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 34), 'str', 'args')
    
    # Obtaining an instance of the builtin type 'list' (line 705)
    list_68626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 705, 42), 'list')
    # Adding type elements to the builtin type 'list' instance (line 705)
    
    # Processing the call keyword arguments (line 705)
    kwargs_68627 = {}
    # Getting the type of 'rout' (line 705)
    rout_68623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 25), 'rout', False)
    # Obtaining the member 'get' of a type (line 705)
    get_68624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 705, 25), rout_68623, 'get')
    # Calling get(args, kwargs) (line 705)
    get_call_result_68628 = invoke(stypy.reporting.localization.Localization(__file__, 705, 25), get_68624, *[str_68625, list_68626], **kwargs_68627)
    
    # Assigning a type to the variable 'tuple_assignment_66685' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'tuple_assignment_66685', get_call_result_68628)
    
    # Assigning a Name to a Name (line 705):
    # Getting the type of 'tuple_assignment_66684' (line 705)
    tuple_assignment_66684_68629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'tuple_assignment_66684')
    # Assigning a type to the variable 'sortargs' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'sortargs', tuple_assignment_66684_68629)
    
    # Assigning a Name to a Name (line 705):
    # Getting the type of 'tuple_assignment_66685' (line 705)
    tuple_assignment_66685_68630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 4), 'tuple_assignment_66685')
    # Assigning a type to the variable 'args' (line 705)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 705, 14), 'args', tuple_assignment_66685_68630)
    
    # Assigning a ListComp to a Name (line 706):
    
    # Assigning a ListComp to a Name (line 706):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to keys(...): (line 706)
    # Processing the call keyword arguments (line 706)
    kwargs_68651 = {}
    
    # Obtaining the type of the subscript
    str_68646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 31), 'str', 'vars')
    # Getting the type of 'rout' (line 706)
    rout_68647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 26), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 706)
    getitem___68648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 26), rout_68647, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 706)
    subscript_call_result_68649 = invoke(stypy.reporting.localization.Localization(__file__, 706, 26), getitem___68648, str_68646)
    
    # Obtaining the member 'keys' of a type (line 706)
    keys_68650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 26), subscript_call_result_68649, 'keys')
    # Calling keys(args, kwargs) (line 706)
    keys_call_result_68652 = invoke(stypy.reporting.localization.Localization(__file__, 706, 26), keys_68650, *[], **kwargs_68651)
    
    comprehension_68653 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 15), keys_call_result_68652)
    # Assigning a type to the variable 'a' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 15), 'a', comprehension_68653)
    
    # Evaluating a boolean operation
    
    # Call to isintent_aux(...): (line 706)
    # Processing the call arguments (line 706)
    
    # Obtaining the type of the subscript
    # Getting the type of 'a' (line 706)
    a_68633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 75), 'a', False)
    
    # Obtaining the type of the subscript
    str_68634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 67), 'str', 'vars')
    # Getting the type of 'rout' (line 706)
    rout_68635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 62), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 706)
    getitem___68636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 62), rout_68635, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 706)
    subscript_call_result_68637 = invoke(stypy.reporting.localization.Localization(__file__, 706, 62), getitem___68636, str_68634)
    
    # Obtaining the member '__getitem__' of a type (line 706)
    getitem___68638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 706, 62), subscript_call_result_68637, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 706)
    subscript_call_result_68639 = invoke(stypy.reporting.localization.Localization(__file__, 706, 62), getitem___68638, a_68633)
    
    # Processing the call keyword arguments (line 706)
    kwargs_68640 = {}
    # Getting the type of 'isintent_aux' (line 706)
    isintent_aux_68632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 49), 'isintent_aux', False)
    # Calling isintent_aux(args, kwargs) (line 706)
    isintent_aux_call_result_68641 = invoke(stypy.reporting.localization.Localization(__file__, 706, 49), isintent_aux_68632, *[subscript_call_result_68639], **kwargs_68640)
    
    
    # Getting the type of 'a' (line 707)
    a_68642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 19), 'a')
    # Getting the type of 'args' (line 707)
    args_68643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 707, 28), 'args')
    # Applying the binary operator 'notin' (line 707)
    result_contains_68644 = python_operator(stypy.reporting.localization.Localization(__file__, 707, 19), 'notin', a_68642, args_68643)
    
    # Applying the binary operator 'and' (line 706)
    result_and_keyword_68645 = python_operator(stypy.reporting.localization.Localization(__file__, 706, 49), 'and', isintent_aux_call_result_68641, result_contains_68644)
    
    # Getting the type of 'a' (line 706)
    a_68631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 706, 15), 'a')
    list_68654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 706, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 706, 15), list_68654, a_68631)
    # Assigning a type to the variable 'auxvars' (line 706)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 706, 4), 'auxvars', list_68654)
    
    # Assigning a BinOp to a Name (line 708):
    
    # Assigning a BinOp to a Name (line 708):
    # Getting the type of 'auxvars' (line 708)
    auxvars_68655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 11), 'auxvars')
    # Getting the type of 'args' (line 708)
    args_68656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 708, 21), 'args')
    # Applying the binary operator '+' (line 708)
    result_add_68657 = python_operator(stypy.reporting.localization.Localization(__file__, 708, 11), '+', auxvars_68655, args_68656)
    
    # Assigning a type to the variable 'args' (line 708)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 708, 4), 'args', result_add_68657)
    
    
    str_68658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 709, 7), 'str', 'sortvars')
    # Getting the type of 'rout' (line 709)
    rout_68659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 709, 21), 'rout')
    # Applying the binary operator 'in' (line 709)
    result_contains_68660 = python_operator(stypy.reporting.localization.Localization(__file__, 709, 7), 'in', str_68658, rout_68659)
    
    # Testing the type of an if condition (line 709)
    if_condition_68661 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 709, 4), result_contains_68660)
    # Assigning a type to the variable 'if_condition_68661' (line 709)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 709, 4), 'if_condition_68661', if_condition_68661)
    # SSA begins for if statement (line 709)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining the type of the subscript
    str_68662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 710, 22), 'str', 'sortvars')
    # Getting the type of 'rout' (line 710)
    rout_68663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 710, 17), 'rout')
    # Obtaining the member '__getitem__' of a type (line 710)
    getitem___68664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 710, 17), rout_68663, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 710)
    subscript_call_result_68665 = invoke(stypy.reporting.localization.Localization(__file__, 710, 17), getitem___68664, str_68662)
    
    # Testing the type of a for loop iterable (line 710)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 710, 8), subscript_call_result_68665)
    # Getting the type of the for loop variable (line 710)
    for_loop_var_68666 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 710, 8), subscript_call_result_68665)
    # Assigning a type to the variable 'a' (line 710)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 710, 8), 'a', for_loop_var_68666)
    # SSA begins for a for statement (line 710)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a' (line 711)
    a_68667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 15), 'a')
    # Getting the type of 'args' (line 711)
    args_68668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 711, 20), 'args')
    # Applying the binary operator 'in' (line 711)
    result_contains_68669 = python_operator(stypy.reporting.localization.Localization(__file__, 711, 15), 'in', a_68667, args_68668)
    
    # Testing the type of an if condition (line 711)
    if_condition_68670 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 711, 12), result_contains_68669)
    # Assigning a type to the variable 'if_condition_68670' (line 711)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 711, 12), 'if_condition_68670', if_condition_68670)
    # SSA begins for if statement (line 711)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 712)
    # Processing the call arguments (line 712)
    # Getting the type of 'a' (line 712)
    a_68673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 32), 'a', False)
    # Processing the call keyword arguments (line 712)
    kwargs_68674 = {}
    # Getting the type of 'sortargs' (line 712)
    sortargs_68671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 712, 16), 'sortargs', False)
    # Obtaining the member 'append' of a type (line 712)
    append_68672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 712, 16), sortargs_68671, 'append')
    # Calling append(args, kwargs) (line 712)
    append_call_result_68675 = invoke(stypy.reporting.localization.Localization(__file__, 712, 16), append_68672, *[a_68673], **kwargs_68674)
    
    # SSA join for if statement (line 711)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'args' (line 713)
    args_68676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 713, 17), 'args')
    # Testing the type of a for loop iterable (line 713)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 713, 8), args_68676)
    # Getting the type of the for loop variable (line 713)
    for_loop_var_68677 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 713, 8), args_68676)
    # Assigning a type to the variable 'a' (line 713)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 713, 8), 'a', for_loop_var_68677)
    # SSA begins for a for statement (line 713)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'a' (line 714)
    a_68678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 15), 'a')
    # Getting the type of 'sortargs' (line 714)
    sortargs_68679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 714, 24), 'sortargs')
    # Applying the binary operator 'notin' (line 714)
    result_contains_68680 = python_operator(stypy.reporting.localization.Localization(__file__, 714, 15), 'notin', a_68678, sortargs_68679)
    
    # Testing the type of an if condition (line 714)
    if_condition_68681 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 714, 12), result_contains_68680)
    # Assigning a type to the variable 'if_condition_68681' (line 714)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 714, 12), 'if_condition_68681', if_condition_68681)
    # SSA begins for if statement (line 714)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 715)
    # Processing the call arguments (line 715)
    # Getting the type of 'a' (line 715)
    a_68684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 32), 'a', False)
    # Processing the call keyword arguments (line 715)
    kwargs_68685 = {}
    # Getting the type of 'sortargs' (line 715)
    sortargs_68682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 715, 16), 'sortargs', False)
    # Obtaining the member 'append' of a type (line 715)
    append_68683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 715, 16), sortargs_68682, 'append')
    # Calling append(args, kwargs) (line 715)
    append_call_result_68686 = invoke(stypy.reporting.localization.Localization(__file__, 715, 16), append_68683, *[a_68684], **kwargs_68685)
    
    # SSA join for if statement (line 714)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 709)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 717):
    
    # Assigning a BinOp to a Name (line 717):
    # Getting the type of 'auxvars' (line 717)
    auxvars_68687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 19), 'auxvars')
    
    # Obtaining the type of the subscript
    str_68688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 717, 34), 'str', 'args')
    # Getting the type of 'rout' (line 717)
    rout_68689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 717, 29), 'rout')
    # Obtaining the member '__getitem__' of a type (line 717)
    getitem___68690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 717, 29), rout_68689, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 717)
    subscript_call_result_68691 = invoke(stypy.reporting.localization.Localization(__file__, 717, 29), getitem___68690, str_68688)
    
    # Applying the binary operator '+' (line 717)
    result_add_68692 = python_operator(stypy.reporting.localization.Localization(__file__, 717, 19), '+', auxvars_68687, subscript_call_result_68691)
    
    # Assigning a type to the variable 'sortargs' (line 717)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 717, 8), 'sortargs', result_add_68692)
    # SSA join for if statement (line 709)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 718)
    tuple_68693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 718, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 718)
    # Adding element type (line 718)
    # Getting the type of 'args' (line 718)
    args_68694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 11), 'args')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 11), tuple_68693, args_68694)
    # Adding element type (line 718)
    # Getting the type of 'sortargs' (line 718)
    sortargs_68695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 718, 17), 'sortargs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 718, 11), tuple_68693, sortargs_68695)
    
    # Assigning a type to the variable 'stypy_return_type' (line 718)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 718, 4), 'stypy_return_type', tuple_68693)
    
    # ################# End of 'getargs2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getargs2' in the type store
    # Getting the type of 'stypy_return_type' (line 704)
    stypy_return_type_68696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68696)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getargs2'
    return stypy_return_type_68696

# Assigning a type to the variable 'getargs2' (line 704)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 0), 'getargs2', getargs2)

@norecursion
def getrestdoc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getrestdoc'
    module_type_store = module_type_store.open_function_context('getrestdoc', 721, 0, False)
    
    # Passed parameters checking function
    getrestdoc.stypy_localization = localization
    getrestdoc.stypy_type_of_self = None
    getrestdoc.stypy_type_store = module_type_store
    getrestdoc.stypy_function_name = 'getrestdoc'
    getrestdoc.stypy_param_names_list = ['rout']
    getrestdoc.stypy_varargs_param_name = None
    getrestdoc.stypy_kwargs_param_name = None
    getrestdoc.stypy_call_defaults = defaults
    getrestdoc.stypy_call_varargs = varargs
    getrestdoc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getrestdoc', ['rout'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getrestdoc', localization, ['rout'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getrestdoc(...)' code ##################

    
    
    str_68697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 722, 7), 'str', 'f2pymultilines')
    # Getting the type of 'rout' (line 722)
    rout_68698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 722, 31), 'rout')
    # Applying the binary operator 'notin' (line 722)
    result_contains_68699 = python_operator(stypy.reporting.localization.Localization(__file__, 722, 7), 'notin', str_68697, rout_68698)
    
    # Testing the type of an if condition (line 722)
    if_condition_68700 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 722, 4), result_contains_68699)
    # Assigning a type to the variable 'if_condition_68700' (line 722)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 722, 4), 'if_condition_68700', if_condition_68700)
    # SSA begins for if statement (line 722)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 723)
    None_68701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 723, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 723)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 723, 8), 'stypy_return_type', None_68701)
    # SSA join for if statement (line 722)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 724):
    
    # Assigning a Name to a Name (line 724):
    # Getting the type of 'None' (line 724)
    None_68702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 724, 8), 'None')
    # Assigning a type to the variable 'k' (line 724)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 724, 4), 'k', None_68702)
    
    
    
    # Obtaining the type of the subscript
    str_68703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 12), 'str', 'block')
    # Getting the type of 'rout' (line 725)
    rout_68704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 725, 7), 'rout')
    # Obtaining the member '__getitem__' of a type (line 725)
    getitem___68705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 725, 7), rout_68704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 725)
    subscript_call_result_68706 = invoke(stypy.reporting.localization.Localization(__file__, 725, 7), getitem___68705, str_68703)
    
    str_68707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 725, 24), 'str', 'python module')
    # Applying the binary operator '==' (line 725)
    result_eq_68708 = python_operator(stypy.reporting.localization.Localization(__file__, 725, 7), '==', subscript_call_result_68706, str_68707)
    
    # Testing the type of an if condition (line 725)
    if_condition_68709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 725, 4), result_eq_68708)
    # Assigning a type to the variable 'if_condition_68709' (line 725)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 725, 4), 'if_condition_68709', if_condition_68709)
    # SSA begins for if statement (line 725)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Name (line 726):
    
    # Assigning a Tuple to a Name (line 726):
    
    # Obtaining an instance of the builtin type 'tuple' (line 726)
    tuple_68710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 726)
    # Adding element type (line 726)
    
    # Obtaining the type of the subscript
    str_68711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 17), 'str', 'block')
    # Getting the type of 'rout' (line 726)
    rout_68712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 12), 'rout')
    # Obtaining the member '__getitem__' of a type (line 726)
    getitem___68713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 12), rout_68712, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 726)
    subscript_call_result_68714 = invoke(stypy.reporting.localization.Localization(__file__, 726, 12), getitem___68713, str_68711)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 726, 12), tuple_68710, subscript_call_result_68714)
    # Adding element type (line 726)
    
    # Obtaining the type of the subscript
    str_68715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 726, 32), 'str', 'name')
    # Getting the type of 'rout' (line 726)
    rout_68716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 726, 27), 'rout')
    # Obtaining the member '__getitem__' of a type (line 726)
    getitem___68717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 726, 27), rout_68716, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 726)
    subscript_call_result_68718 = invoke(stypy.reporting.localization.Localization(__file__, 726, 27), getitem___68717, str_68715)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 726, 12), tuple_68710, subscript_call_result_68718)
    
    # Assigning a type to the variable 'k' (line 726)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 726, 8), 'k', tuple_68710)
    # SSA join for if statement (line 725)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to get(...): (line 727)
    # Processing the call arguments (line 727)
    # Getting the type of 'k' (line 727)
    k_68724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 38), 'k', False)
    # Getting the type of 'None' (line 727)
    None_68725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 41), 'None', False)
    # Processing the call keyword arguments (line 727)
    kwargs_68726 = {}
    
    # Obtaining the type of the subscript
    str_68719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 727, 16), 'str', 'f2pymultilines')
    # Getting the type of 'rout' (line 727)
    rout_68720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 727, 11), 'rout', False)
    # Obtaining the member '__getitem__' of a type (line 727)
    getitem___68721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 11), rout_68720, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 727)
    subscript_call_result_68722 = invoke(stypy.reporting.localization.Localization(__file__, 727, 11), getitem___68721, str_68719)
    
    # Obtaining the member 'get' of a type (line 727)
    get_68723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 727, 11), subscript_call_result_68722, 'get')
    # Calling get(args, kwargs) (line 727)
    get_call_result_68727 = invoke(stypy.reporting.localization.Localization(__file__, 727, 11), get_68723, *[k_68724, None_68725], **kwargs_68726)
    
    # Assigning a type to the variable 'stypy_return_type' (line 727)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 727, 4), 'stypy_return_type', get_call_result_68727)
    
    # ################# End of 'getrestdoc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getrestdoc' in the type store
    # Getting the type of 'stypy_return_type' (line 721)
    stypy_return_type_68728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 721, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68728)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getrestdoc'
    return stypy_return_type_68728

# Assigning a type to the variable 'getrestdoc' (line 721)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 721, 0), 'getrestdoc', getrestdoc)

@norecursion
def gentitle(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gentitle'
    module_type_store = module_type_store.open_function_context('gentitle', 730, 0, False)
    
    # Passed parameters checking function
    gentitle.stypy_localization = localization
    gentitle.stypy_type_of_self = None
    gentitle.stypy_type_store = module_type_store
    gentitle.stypy_function_name = 'gentitle'
    gentitle.stypy_param_names_list = ['name']
    gentitle.stypy_varargs_param_name = None
    gentitle.stypy_kwargs_param_name = None
    gentitle.stypy_call_defaults = defaults
    gentitle.stypy_call_varargs = varargs
    gentitle.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gentitle', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gentitle', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gentitle(...)' code ##################

    
    # Assigning a BinOp to a Name (line 731):
    
    # Assigning a BinOp to a Name (line 731):
    int_68729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 9), 'int')
    
    # Call to len(...): (line 731)
    # Processing the call arguments (line 731)
    # Getting the type of 'name' (line 731)
    name_68731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 18), 'name', False)
    # Processing the call keyword arguments (line 731)
    kwargs_68732 = {}
    # Getting the type of 'len' (line 731)
    len_68730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 731, 14), 'len', False)
    # Calling len(args, kwargs) (line 731)
    len_call_result_68733 = invoke(stypy.reporting.localization.Localization(__file__, 731, 14), len_68730, *[name_68731], **kwargs_68732)
    
    # Applying the binary operator '-' (line 731)
    result_sub_68734 = python_operator(stypy.reporting.localization.Localization(__file__, 731, 9), '-', int_68729, len_call_result_68733)
    
    int_68735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 26), 'int')
    # Applying the binary operator '-' (line 731)
    result_sub_68736 = python_operator(stypy.reporting.localization.Localization(__file__, 731, 24), '-', result_sub_68734, int_68735)
    
    int_68737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 731, 32), 'int')
    # Applying the binary operator '//' (line 731)
    result_floordiv_68738 = python_operator(stypy.reporting.localization.Localization(__file__, 731, 8), '//', result_sub_68736, int_68737)
    
    # Assigning a type to the variable 'l' (line 731)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 731, 4), 'l', result_floordiv_68738)
    str_68739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 11), 'str', '/*%s %s %s*/')
    
    # Obtaining an instance of the builtin type 'tuple' (line 732)
    tuple_68740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 732)
    # Adding element type (line 732)
    # Getting the type of 'l' (line 732)
    l_68741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 29), 'l')
    str_68742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 33), 'str', '*')
    # Applying the binary operator '*' (line 732)
    result_mul_68743 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 29), '*', l_68741, str_68742)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 29), tuple_68740, result_mul_68743)
    # Adding element type (line 732)
    # Getting the type of 'name' (line 732)
    name_68744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 38), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 29), tuple_68740, name_68744)
    # Adding element type (line 732)
    # Getting the type of 'l' (line 732)
    l_68745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 732, 44), 'l')
    str_68746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 732, 48), 'str', '*')
    # Applying the binary operator '*' (line 732)
    result_mul_68747 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 44), '*', l_68745, str_68746)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 732, 29), tuple_68740, result_mul_68747)
    
    # Applying the binary operator '%' (line 732)
    result_mod_68748 = python_operator(stypy.reporting.localization.Localization(__file__, 732, 11), '%', str_68739, tuple_68740)
    
    # Assigning a type to the variable 'stypy_return_type' (line 732)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 732, 4), 'stypy_return_type', result_mod_68748)
    
    # ################# End of 'gentitle(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gentitle' in the type store
    # Getting the type of 'stypy_return_type' (line 730)
    stypy_return_type_68749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 730, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68749)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gentitle'
    return stypy_return_type_68749

# Assigning a type to the variable 'gentitle' (line 730)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 730, 0), 'gentitle', gentitle)

@norecursion
def flatlist(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'flatlist'
    module_type_store = module_type_store.open_function_context('flatlist', 735, 0, False)
    
    # Passed parameters checking function
    flatlist.stypy_localization = localization
    flatlist.stypy_type_of_self = None
    flatlist.stypy_type_store = module_type_store
    flatlist.stypy_function_name = 'flatlist'
    flatlist.stypy_param_names_list = ['l']
    flatlist.stypy_varargs_param_name = None
    flatlist.stypy_kwargs_param_name = None
    flatlist.stypy_call_defaults = defaults
    flatlist.stypy_call_varargs = varargs
    flatlist.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'flatlist', ['l'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'flatlist', localization, ['l'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'flatlist(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 736)
    # Getting the type of 'list' (line 736)
    list_68750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 21), 'list')
    # Getting the type of 'l' (line 736)
    l_68751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 736, 18), 'l')
    
    (may_be_68752, more_types_in_union_68753) = may_be_subtype(list_68750, l_68751)

    if may_be_68752:

        if more_types_in_union_68753:
            # Runtime conditional SSA (line 736)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'l' (line 736)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 736, 4), 'l', remove_not_subtype_from_union(l_68751, list))
        
        # Call to reduce(...): (line 737)
        # Processing the call arguments (line 737)

        @norecursion
        def _stypy_temp_lambda_23(localization, *varargs, **kwargs):
            global module_type_store
            # Getting the type of 'flatlist' (line 737)
            flatlist_68755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 37), 'flatlist', False)
            # Assign values to the parameters with defaults
            defaults = [flatlist_68755]
            # Create a new context for function '_stypy_temp_lambda_23'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_23', 737, 22, True)
            # Passed parameters checking function
            _stypy_temp_lambda_23.stypy_localization = localization
            _stypy_temp_lambda_23.stypy_type_of_self = None
            _stypy_temp_lambda_23.stypy_type_store = module_type_store
            _stypy_temp_lambda_23.stypy_function_name = '_stypy_temp_lambda_23'
            _stypy_temp_lambda_23.stypy_param_names_list = ['x', 'y', 'f']
            _stypy_temp_lambda_23.stypy_varargs_param_name = None
            _stypy_temp_lambda_23.stypy_kwargs_param_name = None
            _stypy_temp_lambda_23.stypy_call_defaults = defaults
            _stypy_temp_lambda_23.stypy_call_varargs = varargs
            _stypy_temp_lambda_23.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_23', ['x', 'y', 'f'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_23', ['x', 'y', 'f'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 737)
            x_68756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 47), 'x', False)
            
            # Call to f(...): (line 737)
            # Processing the call arguments (line 737)
            # Getting the type of 'y' (line 737)
            y_68758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 53), 'y', False)
            # Processing the call keyword arguments (line 737)
            kwargs_68759 = {}
            # Getting the type of 'f' (line 737)
            f_68757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 51), 'f', False)
            # Calling f(args, kwargs) (line 737)
            f_call_result_68760 = invoke(stypy.reporting.localization.Localization(__file__, 737, 51), f_68757, *[y_68758], **kwargs_68759)
            
            # Applying the binary operator '+' (line 737)
            result_add_68761 = python_operator(stypy.reporting.localization.Localization(__file__, 737, 47), '+', x_68756, f_call_result_68760)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 737)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 22), 'stypy_return_type', result_add_68761)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_23' in the type store
            # Getting the type of 'stypy_return_type' (line 737)
            stypy_return_type_68762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 22), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_68762)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_23'
            return stypy_return_type_68762

        # Assigning a type to the variable '_stypy_temp_lambda_23' (line 737)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 22), '_stypy_temp_lambda_23', _stypy_temp_lambda_23)
        # Getting the type of '_stypy_temp_lambda_23' (line 737)
        _stypy_temp_lambda_23_68763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 22), '_stypy_temp_lambda_23')
        # Getting the type of 'l' (line 737)
        l_68764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 57), 'l', False)
        
        # Obtaining an instance of the builtin type 'list' (line 737)
        list_68765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 737, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 737)
        
        # Processing the call keyword arguments (line 737)
        kwargs_68766 = {}
        # Getting the type of 'reduce' (line 737)
        reduce_68754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 737, 15), 'reduce', False)
        # Calling reduce(args, kwargs) (line 737)
        reduce_call_result_68767 = invoke(stypy.reporting.localization.Localization(__file__, 737, 15), reduce_68754, *[_stypy_temp_lambda_23_68763, l_68764, list_68765], **kwargs_68766)
        
        # Assigning a type to the variable 'stypy_return_type' (line 737)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 737, 8), 'stypy_return_type', reduce_call_result_68767)

        if more_types_in_union_68753:
            # SSA join for if statement (line 736)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'list' (line 738)
    list_68768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 738, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 738)
    # Adding element type (line 738)
    # Getting the type of 'l' (line 738)
    l_68769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 738, 12), 'l')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 738, 11), list_68768, l_68769)
    
    # Assigning a type to the variable 'stypy_return_type' (line 738)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 738, 4), 'stypy_return_type', list_68768)
    
    # ################# End of 'flatlist(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'flatlist' in the type store
    # Getting the type of 'stypy_return_type' (line 735)
    stypy_return_type_68770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 735, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68770)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'flatlist'
    return stypy_return_type_68770

# Assigning a type to the variable 'flatlist' (line 735)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 735, 0), 'flatlist', flatlist)

@norecursion
def stripcomma(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'stripcomma'
    module_type_store = module_type_store.open_function_context('stripcomma', 741, 0, False)
    
    # Passed parameters checking function
    stripcomma.stypy_localization = localization
    stripcomma.stypy_type_of_self = None
    stripcomma.stypy_type_store = module_type_store
    stripcomma.stypy_function_name = 'stripcomma'
    stripcomma.stypy_param_names_list = ['s']
    stripcomma.stypy_varargs_param_name = None
    stripcomma.stypy_kwargs_param_name = None
    stripcomma.stypy_call_defaults = defaults
    stripcomma.stypy_call_varargs = varargs
    stripcomma.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'stripcomma', ['s'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'stripcomma', localization, ['s'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'stripcomma(...)' code ##################

    
    
    # Evaluating a boolean operation
    # Getting the type of 's' (line 742)
    s_68771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 7), 's')
    
    
    # Obtaining the type of the subscript
    int_68772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 15), 'int')
    # Getting the type of 's' (line 742)
    s_68773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 742, 13), 's')
    # Obtaining the member '__getitem__' of a type (line 742)
    getitem___68774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 742, 13), s_68773, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 742)
    subscript_call_result_68775 = invoke(stypy.reporting.localization.Localization(__file__, 742, 13), getitem___68774, int_68772)
    
    str_68776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 742, 22), 'str', ',')
    # Applying the binary operator '==' (line 742)
    result_eq_68777 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 13), '==', subscript_call_result_68775, str_68776)
    
    # Applying the binary operator 'and' (line 742)
    result_and_keyword_68778 = python_operator(stypy.reporting.localization.Localization(__file__, 742, 7), 'and', s_68771, result_eq_68777)
    
    # Testing the type of an if condition (line 742)
    if_condition_68779 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 742, 4), result_and_keyword_68778)
    # Assigning a type to the variable 'if_condition_68779' (line 742)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 742, 4), 'if_condition_68779', if_condition_68779)
    # SSA begins for if statement (line 742)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_68780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 743, 18), 'int')
    slice_68781 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 743, 15), None, int_68780, None)
    # Getting the type of 's' (line 743)
    s_68782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 743, 15), 's')
    # Obtaining the member '__getitem__' of a type (line 743)
    getitem___68783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 743, 15), s_68782, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 743)
    subscript_call_result_68784 = invoke(stypy.reporting.localization.Localization(__file__, 743, 15), getitem___68783, slice_68781)
    
    # Assigning a type to the variable 'stypy_return_type' (line 743)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 743, 8), 'stypy_return_type', subscript_call_result_68784)
    # SSA join for if statement (line 742)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 's' (line 744)
    s_68785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 744, 11), 's')
    # Assigning a type to the variable 'stypy_return_type' (line 744)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 744, 4), 'stypy_return_type', s_68785)
    
    # ################# End of 'stripcomma(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'stripcomma' in the type store
    # Getting the type of 'stypy_return_type' (line 741)
    stypy_return_type_68786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 741, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68786)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'stripcomma'
    return stypy_return_type_68786

# Assigning a type to the variable 'stripcomma' (line 741)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 741, 0), 'stripcomma', stripcomma)

@norecursion
def replace(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_68787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 747, 31), 'str', '')
    defaults = [str_68787]
    # Create a new context for function 'replace'
    module_type_store = module_type_store.open_function_context('replace', 747, 0, False)
    
    # Passed parameters checking function
    replace.stypy_localization = localization
    replace.stypy_type_of_self = None
    replace.stypy_type_store = module_type_store
    replace.stypy_function_name = 'replace'
    replace.stypy_param_names_list = ['str', 'd', 'defaultsep']
    replace.stypy_varargs_param_name = None
    replace.stypy_kwargs_param_name = None
    replace.stypy_call_defaults = defaults
    replace.stypy_call_varargs = varargs
    replace.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'replace', ['str', 'd', 'defaultsep'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'replace', localization, ['str', 'd', 'defaultsep'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'replace(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 748)
    # Getting the type of 'list' (line 748)
    list_68788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 21), 'list')
    # Getting the type of 'd' (line 748)
    d_68789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 748, 18), 'd')
    
    (may_be_68790, more_types_in_union_68791) = may_be_subtype(list_68788, d_68789)

    if may_be_68790:

        if more_types_in_union_68791:
            # Runtime conditional SSA (line 748)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'd' (line 748)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 748, 4), 'd', remove_not_subtype_from_union(d_68789, list))
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'd' (line 749)
        d_68798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 55), 'd')
        comprehension_68799 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 16), d_68798)
        # Assigning a type to the variable '_m' (line 749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 16), '_m', comprehension_68799)
        
        # Call to replace(...): (line 749)
        # Processing the call arguments (line 749)
        # Getting the type of 'str' (line 749)
        str_68793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 24), 'str', False)
        # Getting the type of '_m' (line 749)
        _m_68794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 29), '_m', False)
        # Getting the type of 'defaultsep' (line 749)
        defaultsep_68795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 33), 'defaultsep', False)
        # Processing the call keyword arguments (line 749)
        kwargs_68796 = {}
        # Getting the type of 'replace' (line 749)
        replace_68792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 749, 16), 'replace', False)
        # Calling replace(args, kwargs) (line 749)
        replace_call_result_68797 = invoke(stypy.reporting.localization.Localization(__file__, 749, 16), replace_68792, *[str_68793, _m_68794, defaultsep_68795], **kwargs_68796)
        
        list_68800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 749, 16), list_68800, replace_call_result_68797)
        # Assigning a type to the variable 'stypy_return_type' (line 749)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 749, 8), 'stypy_return_type', list_68800)

        if more_types_in_union_68791:
            # SSA join for if statement (line 748)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 750)
    # Getting the type of 'list' (line 750)
    list_68801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 23), 'list')
    # Getting the type of 'str' (line 750)
    str_68802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 750, 18), 'str')
    
    (may_be_68803, more_types_in_union_68804) = may_be_subtype(list_68801, str_68802)

    if may_be_68803:

        if more_types_in_union_68804:
            # Runtime conditional SSA (line 750)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'str' (line 750)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 750, 4), 'str', remove_not_subtype_from_union(str_68802, list))
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'str' (line 751)
        str_68811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 53), 'str')
        comprehension_68812 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 751, 16), str_68811)
        # Assigning a type to the variable '_m' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 16), '_m', comprehension_68812)
        
        # Call to replace(...): (line 751)
        # Processing the call arguments (line 751)
        # Getting the type of '_m' (line 751)
        _m_68806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 24), '_m', False)
        # Getting the type of 'd' (line 751)
        d_68807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 28), 'd', False)
        # Getting the type of 'defaultsep' (line 751)
        defaultsep_68808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 31), 'defaultsep', False)
        # Processing the call keyword arguments (line 751)
        kwargs_68809 = {}
        # Getting the type of 'replace' (line 751)
        replace_68805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 751, 16), 'replace', False)
        # Calling replace(args, kwargs) (line 751)
        replace_call_result_68810 = invoke(stypy.reporting.localization.Localization(__file__, 751, 16), replace_68805, *[_m_68806, d_68807, defaultsep_68808], **kwargs_68809)
        
        list_68813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 751, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 751, 16), list_68813, replace_call_result_68810)
        # Assigning a type to the variable 'stypy_return_type' (line 751)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 751, 8), 'stypy_return_type', list_68813)

        if more_types_in_union_68804:
            # SSA join for if statement (line 750)
            module_type_store = module_type_store.join_ssa_context()


    
    
    int_68814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 752, 13), 'int')
    
    # Call to list(...): (line 752)
    # Processing the call arguments (line 752)
    
    # Call to keys(...): (line 752)
    # Processing the call keyword arguments (line 752)
    kwargs_68818 = {}
    # Getting the type of 'd' (line 752)
    d_68816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 22), 'd', False)
    # Obtaining the member 'keys' of a type (line 752)
    keys_68817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 752, 22), d_68816, 'keys')
    # Calling keys(args, kwargs) (line 752)
    keys_call_result_68819 = invoke(stypy.reporting.localization.Localization(__file__, 752, 22), keys_68817, *[], **kwargs_68818)
    
    # Processing the call keyword arguments (line 752)
    kwargs_68820 = {}
    # Getting the type of 'list' (line 752)
    list_68815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 752, 17), 'list', False)
    # Calling list(args, kwargs) (line 752)
    list_call_result_68821 = invoke(stypy.reporting.localization.Localization(__file__, 752, 17), list_68815, *[keys_call_result_68819], **kwargs_68820)
    
    # Applying the binary operator '*' (line 752)
    result_mul_68822 = python_operator(stypy.reporting.localization.Localization(__file__, 752, 13), '*', int_68814, list_call_result_68821)
    
    # Testing the type of a for loop iterable (line 752)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 752, 4), result_mul_68822)
    # Getting the type of the for loop variable (line 752)
    for_loop_var_68823 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 752, 4), result_mul_68822)
    # Assigning a type to the variable 'k' (line 752)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 752, 4), 'k', for_loop_var_68823)
    # SSA begins for a for statement (line 752)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'k' (line 753)
    k_68824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 11), 'k')
    str_68825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 753, 16), 'str', 'separatorsfor')
    # Applying the binary operator '==' (line 753)
    result_eq_68826 = python_operator(stypy.reporting.localization.Localization(__file__, 753, 11), '==', k_68824, str_68825)
    
    # Testing the type of an if condition (line 753)
    if_condition_68827 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 753, 8), result_eq_68826)
    # Assigning a type to the variable 'if_condition_68827' (line 753)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 8), 'if_condition_68827', if_condition_68827)
    # SSA begins for if statement (line 753)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 753)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    str_68828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 11), 'str', 'separatorsfor')
    # Getting the type of 'd' (line 755)
    d_68829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 30), 'd')
    # Applying the binary operator 'in' (line 755)
    result_contains_68830 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 11), 'in', str_68828, d_68829)
    
    
    # Getting the type of 'k' (line 755)
    k_68831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 36), 'k')
    
    # Obtaining the type of the subscript
    str_68832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 755, 43), 'str', 'separatorsfor')
    # Getting the type of 'd' (line 755)
    d_68833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 755, 41), 'd')
    # Obtaining the member '__getitem__' of a type (line 755)
    getitem___68834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 755, 41), d_68833, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 755)
    subscript_call_result_68835 = invoke(stypy.reporting.localization.Localization(__file__, 755, 41), getitem___68834, str_68832)
    
    # Applying the binary operator 'in' (line 755)
    result_contains_68836 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 36), 'in', k_68831, subscript_call_result_68835)
    
    # Applying the binary operator 'and' (line 755)
    result_and_keyword_68837 = python_operator(stypy.reporting.localization.Localization(__file__, 755, 11), 'and', result_contains_68830, result_contains_68836)
    
    # Testing the type of an if condition (line 755)
    if_condition_68838 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 755, 8), result_and_keyword_68837)
    # Assigning a type to the variable 'if_condition_68838' (line 755)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 755, 8), 'if_condition_68838', if_condition_68838)
    # SSA begins for if statement (line 755)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 756):
    
    # Assigning a Subscript to a Name (line 756):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 756)
    k_68839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 37), 'k')
    
    # Obtaining the type of the subscript
    str_68840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 756, 20), 'str', 'separatorsfor')
    # Getting the type of 'd' (line 756)
    d_68841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 756, 18), 'd')
    # Obtaining the member '__getitem__' of a type (line 756)
    getitem___68842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 18), d_68841, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 756)
    subscript_call_result_68843 = invoke(stypy.reporting.localization.Localization(__file__, 756, 18), getitem___68842, str_68840)
    
    # Obtaining the member '__getitem__' of a type (line 756)
    getitem___68844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 756, 18), subscript_call_result_68843, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 756)
    subscript_call_result_68845 = invoke(stypy.reporting.localization.Localization(__file__, 756, 18), getitem___68844, k_68839)
    
    # Assigning a type to the variable 'sep' (line 756)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 756, 12), 'sep', subscript_call_result_68845)
    # SSA branch for the else part of an if statement (line 755)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 758):
    
    # Assigning a Name to a Name (line 758):
    # Getting the type of 'defaultsep' (line 758)
    defaultsep_68846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 18), 'defaultsep')
    # Assigning a type to the variable 'sep' (line 758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 12), 'sep', defaultsep_68846)
    # SSA join for if statement (line 755)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 759)
    # Getting the type of 'list' (line 759)
    list_68847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 28), 'list')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 759)
    k_68848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 24), 'k')
    # Getting the type of 'd' (line 759)
    d_68849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 22), 'd')
    # Obtaining the member '__getitem__' of a type (line 759)
    getitem___68850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 759, 22), d_68849, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 759)
    subscript_call_result_68851 = invoke(stypy.reporting.localization.Localization(__file__, 759, 22), getitem___68850, k_68848)
    
    
    (may_be_68852, more_types_in_union_68853) = may_be_subtype(list_68847, subscript_call_result_68851)

    if may_be_68852:

        if more_types_in_union_68853:
            # Runtime conditional SSA (line 759)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 760):
        
        # Assigning a Call to a Name (line 760):
        
        # Call to replace(...): (line 760)
        # Processing the call arguments (line 760)
        str_68856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 760, 30), 'str', '#%s#')
        # Getting the type of 'k' (line 760)
        k_68857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 40), 'k', False)
        # Applying the binary operator '%' (line 760)
        result_mod_68858 = python_operator(stypy.reporting.localization.Localization(__file__, 760, 30), '%', str_68856, k_68857)
        
        
        # Call to join(...): (line 760)
        # Processing the call arguments (line 760)
        
        # Call to flatlist(...): (line 760)
        # Processing the call arguments (line 760)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 760)
        k_68862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 64), 'k', False)
        # Getting the type of 'd' (line 760)
        d_68863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 62), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 760)
        getitem___68864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 62), d_68863, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 760)
        subscript_call_result_68865 = invoke(stypy.reporting.localization.Localization(__file__, 760, 62), getitem___68864, k_68862)
        
        # Processing the call keyword arguments (line 760)
        kwargs_68866 = {}
        # Getting the type of 'flatlist' (line 760)
        flatlist_68861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 53), 'flatlist', False)
        # Calling flatlist(args, kwargs) (line 760)
        flatlist_call_result_68867 = invoke(stypy.reporting.localization.Localization(__file__, 760, 53), flatlist_68861, *[subscript_call_result_68865], **kwargs_68866)
        
        # Processing the call keyword arguments (line 760)
        kwargs_68868 = {}
        # Getting the type of 'sep' (line 760)
        sep_68859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 44), 'sep', False)
        # Obtaining the member 'join' of a type (line 760)
        join_68860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 44), sep_68859, 'join')
        # Calling join(args, kwargs) (line 760)
        join_call_result_68869 = invoke(stypy.reporting.localization.Localization(__file__, 760, 44), join_68860, *[flatlist_call_result_68867], **kwargs_68868)
        
        # Processing the call keyword arguments (line 760)
        kwargs_68870 = {}
        # Getting the type of 'str' (line 760)
        str_68854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 760, 18), 'str', False)
        # Obtaining the member 'replace' of a type (line 760)
        replace_68855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 760, 18), str_68854, 'replace')
        # Calling replace(args, kwargs) (line 760)
        replace_call_result_68871 = invoke(stypy.reporting.localization.Localization(__file__, 760, 18), replace_68855, *[result_mod_68858, join_call_result_68869], **kwargs_68870)
        
        # Assigning a type to the variable 'str' (line 760)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 12), 'str', replace_call_result_68871)

        if more_types_in_union_68853:
            # Runtime conditional SSA for else branch (line 759)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_68852) or more_types_in_union_68853):
        
        # Assigning a Call to a Name (line 762):
        
        # Assigning a Call to a Name (line 762):
        
        # Call to replace(...): (line 762)
        # Processing the call arguments (line 762)
        str_68874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 762, 30), 'str', '#%s#')
        # Getting the type of 'k' (line 762)
        k_68875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 40), 'k', False)
        # Applying the binary operator '%' (line 762)
        result_mod_68876 = python_operator(stypy.reporting.localization.Localization(__file__, 762, 30), '%', str_68874, k_68875)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 762)
        k_68877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 46), 'k', False)
        # Getting the type of 'd' (line 762)
        d_68878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 44), 'd', False)
        # Obtaining the member '__getitem__' of a type (line 762)
        getitem___68879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 44), d_68878, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 762)
        subscript_call_result_68880 = invoke(stypy.reporting.localization.Localization(__file__, 762, 44), getitem___68879, k_68877)
        
        # Processing the call keyword arguments (line 762)
        kwargs_68881 = {}
        # Getting the type of 'str' (line 762)
        str_68872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 18), 'str', False)
        # Obtaining the member 'replace' of a type (line 762)
        replace_68873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 18), str_68872, 'replace')
        # Calling replace(args, kwargs) (line 762)
        replace_call_result_68882 = invoke(stypy.reporting.localization.Localization(__file__, 762, 18), replace_68873, *[result_mod_68876, subscript_call_result_68880], **kwargs_68881)
        
        # Assigning a type to the variable 'str' (line 762)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 12), 'str', replace_call_result_68882)

        if (may_be_68852 and more_types_in_union_68853):
            # SSA join for if statement (line 759)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'str' (line 763)
    str_68883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 11), 'str')
    # Assigning a type to the variable 'stypy_return_type' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 4), 'stypy_return_type', str_68883)
    
    # ################# End of 'replace(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'replace' in the type store
    # Getting the type of 'stypy_return_type' (line 747)
    stypy_return_type_68884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 747, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68884)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'replace'
    return stypy_return_type_68884

# Assigning a type to the variable 'replace' (line 747)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 747, 0), 'replace', replace)

@norecursion
def dictappend(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dictappend'
    module_type_store = module_type_store.open_function_context('dictappend', 766, 0, False)
    
    # Passed parameters checking function
    dictappend.stypy_localization = localization
    dictappend.stypy_type_of_self = None
    dictappend.stypy_type_store = module_type_store
    dictappend.stypy_function_name = 'dictappend'
    dictappend.stypy_param_names_list = ['rd', 'ar']
    dictappend.stypy_varargs_param_name = None
    dictappend.stypy_kwargs_param_name = None
    dictappend.stypy_call_defaults = defaults
    dictappend.stypy_call_varargs = varargs
    dictappend.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dictappend', ['rd', 'ar'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dictappend', localization, ['rd', 'ar'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dictappend(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 767)
    # Getting the type of 'list' (line 767)
    list_68885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 22), 'list')
    # Getting the type of 'ar' (line 767)
    ar_68886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 18), 'ar')
    
    (may_be_68887, more_types_in_union_68888) = may_be_subtype(list_68885, ar_68886)

    if may_be_68887:

        if more_types_in_union_68888:
            # Runtime conditional SSA (line 767)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'ar' (line 767)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 4), 'ar', remove_not_subtype_from_union(ar_68886, list))
        
        # Getting the type of 'ar' (line 768)
        ar_68889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 17), 'ar')
        # Testing the type of a for loop iterable (line 768)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 768, 8), ar_68889)
        # Getting the type of the for loop variable (line 768)
        for_loop_var_68890 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 768, 8), ar_68889)
        # Assigning a type to the variable 'a' (line 768)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'a', for_loop_var_68890)
        # SSA begins for a for statement (line 768)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 769):
        
        # Assigning a Call to a Name (line 769):
        
        # Call to dictappend(...): (line 769)
        # Processing the call arguments (line 769)
        # Getting the type of 'rd' (line 769)
        rd_68892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 28), 'rd', False)
        # Getting the type of 'a' (line 769)
        a_68893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 32), 'a', False)
        # Processing the call keyword arguments (line 769)
        kwargs_68894 = {}
        # Getting the type of 'dictappend' (line 769)
        dictappend_68891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 17), 'dictappend', False)
        # Calling dictappend(args, kwargs) (line 769)
        dictappend_call_result_68895 = invoke(stypy.reporting.localization.Localization(__file__, 769, 17), dictappend_68891, *[rd_68892, a_68893], **kwargs_68894)
        
        # Assigning a type to the variable 'rd' (line 769)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 12), 'rd', dictappend_call_result_68895)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'rd' (line 770)
        rd_68896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 15), 'rd')
        # Assigning a type to the variable 'stypy_return_type' (line 770)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 8), 'stypy_return_type', rd_68896)

        if more_types_in_union_68888:
            # SSA join for if statement (line 767)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to keys(...): (line 771)
    # Processing the call keyword arguments (line 771)
    kwargs_68899 = {}
    # Getting the type of 'ar' (line 771)
    ar_68897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 771, 13), 'ar', False)
    # Obtaining the member 'keys' of a type (line 771)
    keys_68898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 771, 13), ar_68897, 'keys')
    # Calling keys(args, kwargs) (line 771)
    keys_call_result_68900 = invoke(stypy.reporting.localization.Localization(__file__, 771, 13), keys_68898, *[], **kwargs_68899)
    
    # Testing the type of a for loop iterable (line 771)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 771, 4), keys_call_result_68900)
    # Getting the type of the for loop variable (line 771)
    for_loop_var_68901 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 771, 4), keys_call_result_68900)
    # Assigning a type to the variable 'k' (line 771)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 771, 4), 'k', for_loop_var_68901)
    # SSA begins for a for statement (line 771)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    int_68902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 13), 'int')
    # Getting the type of 'k' (line 772)
    k_68903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 11), 'k')
    # Obtaining the member '__getitem__' of a type (line 772)
    getitem___68904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 772, 11), k_68903, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 772)
    subscript_call_result_68905 = invoke(stypy.reporting.localization.Localization(__file__, 772, 11), getitem___68904, int_68902)
    
    str_68906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 772, 19), 'str', '_')
    # Applying the binary operator '==' (line 772)
    result_eq_68907 = python_operator(stypy.reporting.localization.Localization(__file__, 772, 11), '==', subscript_call_result_68905, str_68906)
    
    # Testing the type of an if condition (line 772)
    if_condition_68908 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 772, 8), result_eq_68907)
    # Assigning a type to the variable 'if_condition_68908' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 8), 'if_condition_68908', if_condition_68908)
    # SSA begins for if statement (line 772)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 772)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'k' (line 774)
    k_68909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 11), 'k')
    # Getting the type of 'rd' (line 774)
    rd_68910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 774, 16), 'rd')
    # Applying the binary operator 'in' (line 774)
    result_contains_68911 = python_operator(stypy.reporting.localization.Localization(__file__, 774, 11), 'in', k_68909, rd_68910)
    
    # Testing the type of an if condition (line 774)
    if_condition_68912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 774, 8), result_contains_68911)
    # Assigning a type to the variable 'if_condition_68912' (line 774)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 774, 8), 'if_condition_68912', if_condition_68912)
    # SSA begins for if statement (line 774)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Type idiom detected: calculating its left and rigth part (line 775)
    # Getting the type of 'str' (line 775)
    str_68913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 33), 'str')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 775)
    k_68914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 29), 'k')
    # Getting the type of 'rd' (line 775)
    rd_68915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 775, 26), 'rd')
    # Obtaining the member '__getitem__' of a type (line 775)
    getitem___68916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 775, 26), rd_68915, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 775)
    subscript_call_result_68917 = invoke(stypy.reporting.localization.Localization(__file__, 775, 26), getitem___68916, k_68914)
    
    
    (may_be_68918, more_types_in_union_68919) = may_be_subtype(str_68913, subscript_call_result_68917)

    if may_be_68918:

        if more_types_in_union_68919:
            # Runtime conditional SSA (line 775)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Subscript (line 776):
        
        # Assigning a List to a Subscript (line 776):
        
        # Obtaining an instance of the builtin type 'list' (line 776)
        list_68920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 776, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 776)
        # Adding element type (line 776)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 776)
        k_68921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 28), 'k')
        # Getting the type of 'rd' (line 776)
        rd_68922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 25), 'rd')
        # Obtaining the member '__getitem__' of a type (line 776)
        getitem___68923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 776, 25), rd_68922, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 776)
        subscript_call_result_68924 = invoke(stypy.reporting.localization.Localization(__file__, 776, 25), getitem___68923, k_68921)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 776, 24), list_68920, subscript_call_result_68924)
        
        # Getting the type of 'rd' (line 776)
        rd_68925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 16), 'rd')
        # Getting the type of 'k' (line 776)
        k_68926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 776, 19), 'k')
        # Storing an element on a container (line 776)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 776, 16), rd_68925, (k_68926, list_68920))

        if more_types_in_union_68919:
            # SSA join for if statement (line 775)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 777)
    # Getting the type of 'list' (line 777)
    list_68927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 33), 'list')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 777)
    k_68928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 29), 'k')
    # Getting the type of 'rd' (line 777)
    rd_68929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 777, 26), 'rd')
    # Obtaining the member '__getitem__' of a type (line 777)
    getitem___68930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 777, 26), rd_68929, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 777)
    subscript_call_result_68931 = invoke(stypy.reporting.localization.Localization(__file__, 777, 26), getitem___68930, k_68928)
    
    
    (may_be_68932, more_types_in_union_68933) = may_be_subtype(list_68927, subscript_call_result_68931)

    if may_be_68932:

        if more_types_in_union_68933:
            # Runtime conditional SSA (line 777)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Type idiom detected: calculating its left and rigth part (line 778)
        # Getting the type of 'list' (line 778)
        list_68934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 37), 'list')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 778)
        k_68935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 33), 'k')
        # Getting the type of 'ar' (line 778)
        ar_68936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 778, 30), 'ar')
        # Obtaining the member '__getitem__' of a type (line 778)
        getitem___68937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 778, 30), ar_68936, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 778)
        subscript_call_result_68938 = invoke(stypy.reporting.localization.Localization(__file__, 778, 30), getitem___68937, k_68935)
        
        
        (may_be_68939, more_types_in_union_68940) = may_be_subtype(list_68934, subscript_call_result_68938)

        if may_be_68939:

            if more_types_in_union_68940:
                # Runtime conditional SSA (line 778)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Subscript (line 779):
            
            # Assigning a BinOp to a Subscript (line 779):
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 779)
            k_68941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 31), 'k')
            # Getting the type of 'rd' (line 779)
            rd_68942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 28), 'rd')
            # Obtaining the member '__getitem__' of a type (line 779)
            getitem___68943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 28), rd_68942, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 779)
            subscript_call_result_68944 = invoke(stypy.reporting.localization.Localization(__file__, 779, 28), getitem___68943, k_68941)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 779)
            k_68945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 39), 'k')
            # Getting the type of 'ar' (line 779)
            ar_68946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 36), 'ar')
            # Obtaining the member '__getitem__' of a type (line 779)
            getitem___68947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 779, 36), ar_68946, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 779)
            subscript_call_result_68948 = invoke(stypy.reporting.localization.Localization(__file__, 779, 36), getitem___68947, k_68945)
            
            # Applying the binary operator '+' (line 779)
            result_add_68949 = python_operator(stypy.reporting.localization.Localization(__file__, 779, 28), '+', subscript_call_result_68944, subscript_call_result_68948)
            
            # Getting the type of 'rd' (line 779)
            rd_68950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 20), 'rd')
            # Getting the type of 'k' (line 779)
            k_68951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 779, 23), 'k')
            # Storing an element on a container (line 779)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 779, 20), rd_68950, (k_68951, result_add_68949))

            if more_types_in_union_68940:
                # Runtime conditional SSA for else branch (line 778)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_68939) or more_types_in_union_68940):
            
            # Call to append(...): (line 781)
            # Processing the call arguments (line 781)
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 781)
            k_68957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 36), 'k', False)
            # Getting the type of 'ar' (line 781)
            ar_68958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 33), 'ar', False)
            # Obtaining the member '__getitem__' of a type (line 781)
            getitem___68959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 33), ar_68958, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 781)
            subscript_call_result_68960 = invoke(stypy.reporting.localization.Localization(__file__, 781, 33), getitem___68959, k_68957)
            
            # Processing the call keyword arguments (line 781)
            kwargs_68961 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 781)
            k_68952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 23), 'k', False)
            # Getting the type of 'rd' (line 781)
            rd_68953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 781, 20), 'rd', False)
            # Obtaining the member '__getitem__' of a type (line 781)
            getitem___68954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 20), rd_68953, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 781)
            subscript_call_result_68955 = invoke(stypy.reporting.localization.Localization(__file__, 781, 20), getitem___68954, k_68952)
            
            # Obtaining the member 'append' of a type (line 781)
            append_68956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 781, 20), subscript_call_result_68955, 'append')
            # Calling append(args, kwargs) (line 781)
            append_call_result_68962 = invoke(stypy.reporting.localization.Localization(__file__, 781, 20), append_68956, *[subscript_call_result_68960], **kwargs_68961)
            

            if (may_be_68939 and more_types_in_union_68940):
                # SSA join for if statement (line 778)
                module_type_store = module_type_store.join_ssa_context()


        

        if more_types_in_union_68933:
            # Runtime conditional SSA for else branch (line 777)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_68932) or more_types_in_union_68933):
        
        # Type idiom detected: calculating its left and rigth part (line 782)
        # Getting the type of 'dict' (line 782)
        dict_68963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 35), 'dict')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 782)
        k_68964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 31), 'k')
        # Getting the type of 'rd' (line 782)
        rd_68965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 28), 'rd')
        # Obtaining the member '__getitem__' of a type (line 782)
        getitem___68966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 782, 28), rd_68965, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 782)
        subscript_call_result_68967 = invoke(stypy.reporting.localization.Localization(__file__, 782, 28), getitem___68966, k_68964)
        
        
        (may_be_68968, more_types_in_union_68969) = may_be_subtype(dict_68963, subscript_call_result_68967)

        if may_be_68968:

            if more_types_in_union_68969:
                # Runtime conditional SSA (line 782)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 783)
            # Getting the type of 'dict' (line 783)
            dict_68970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 37), 'dict')
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 783)
            k_68971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 33), 'k')
            # Getting the type of 'ar' (line 783)
            ar_68972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 783, 30), 'ar')
            # Obtaining the member '__getitem__' of a type (line 783)
            getitem___68973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 783, 30), ar_68972, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 783)
            subscript_call_result_68974 = invoke(stypy.reporting.localization.Localization(__file__, 783, 30), getitem___68973, k_68971)
            
            
            (may_be_68975, more_types_in_union_68976) = may_be_subtype(dict_68970, subscript_call_result_68974)

            if may_be_68975:

                if more_types_in_union_68976:
                    # Runtime conditional SSA (line 783)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                
                # Getting the type of 'k' (line 784)
                k_68977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 784, 23), 'k')
                str_68978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 784, 28), 'str', 'separatorsfor')
                # Applying the binary operator '==' (line 784)
                result_eq_68979 = python_operator(stypy.reporting.localization.Localization(__file__, 784, 23), '==', k_68977, str_68978)
                
                # Testing the type of an if condition (line 784)
                if_condition_68980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 784, 20), result_eq_68979)
                # Assigning a type to the variable 'if_condition_68980' (line 784)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 784, 20), 'if_condition_68980', if_condition_68980)
                # SSA begins for if statement (line 784)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to keys(...): (line 785)
                # Processing the call keyword arguments (line 785)
                kwargs_68986 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 785)
                k_68981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 37), 'k', False)
                # Getting the type of 'ar' (line 785)
                ar_68982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 34), 'ar', False)
                # Obtaining the member '__getitem__' of a type (line 785)
                getitem___68983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 34), ar_68982, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 785)
                subscript_call_result_68984 = invoke(stypy.reporting.localization.Localization(__file__, 785, 34), getitem___68983, k_68981)
                
                # Obtaining the member 'keys' of a type (line 785)
                keys_68985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 34), subscript_call_result_68984, 'keys')
                # Calling keys(args, kwargs) (line 785)
                keys_call_result_68987 = invoke(stypy.reporting.localization.Localization(__file__, 785, 34), keys_68985, *[], **kwargs_68986)
                
                # Testing the type of a for loop iterable (line 785)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 785, 24), keys_call_result_68987)
                # Getting the type of the for loop variable (line 785)
                for_loop_var_68988 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 785, 24), keys_call_result_68987)
                # Assigning a type to the variable 'k1' (line 785)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 24), 'k1', for_loop_var_68988)
                # SSA begins for a for statement (line 785)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Getting the type of 'k1' (line 786)
                k1_68989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 31), 'k1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 786)
                k_68990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 44), 'k')
                # Getting the type of 'rd' (line 786)
                rd_68991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 41), 'rd')
                # Obtaining the member '__getitem__' of a type (line 786)
                getitem___68992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 786, 41), rd_68991, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 786)
                subscript_call_result_68993 = invoke(stypy.reporting.localization.Localization(__file__, 786, 41), getitem___68992, k_68990)
                
                # Applying the binary operator 'notin' (line 786)
                result_contains_68994 = python_operator(stypy.reporting.localization.Localization(__file__, 786, 31), 'notin', k1_68989, subscript_call_result_68993)
                
                # Testing the type of an if condition (line 786)
                if_condition_68995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 786, 28), result_contains_68994)
                # Assigning a type to the variable 'if_condition_68995' (line 786)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 28), 'if_condition_68995', if_condition_68995)
                # SSA begins for if statement (line 786)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Subscript (line 787):
                
                # Assigning a Subscript to a Subscript (line 787):
                
                # Obtaining the type of the subscript
                # Getting the type of 'k1' (line 787)
                k1_68996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 50), 'k1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 787)
                k_68997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 47), 'k')
                # Getting the type of 'ar' (line 787)
                ar_68998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 44), 'ar')
                # Obtaining the member '__getitem__' of a type (line 787)
                getitem___68999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 44), ar_68998, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 787)
                subscript_call_result_69000 = invoke(stypy.reporting.localization.Localization(__file__, 787, 44), getitem___68999, k_68997)
                
                # Obtaining the member '__getitem__' of a type (line 787)
                getitem___69001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 44), subscript_call_result_69000, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 787)
                subscript_call_result_69002 = invoke(stypy.reporting.localization.Localization(__file__, 787, 44), getitem___69001, k1_68996)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 787)
                k_69003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 35), 'k')
                # Getting the type of 'rd' (line 787)
                rd_69004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 32), 'rd')
                # Obtaining the member '__getitem__' of a type (line 787)
                getitem___69005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 787, 32), rd_69004, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 787)
                subscript_call_result_69006 = invoke(stypy.reporting.localization.Localization(__file__, 787, 32), getitem___69005, k_69003)
                
                # Getting the type of 'k1' (line 787)
                k1_69007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 787, 38), 'k1')
                # Storing an element on a container (line 787)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 787, 32), subscript_call_result_69006, (k1_69007, subscript_call_result_69002))
                # SSA join for if statement (line 786)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA branch for the else part of an if statement (line 784)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 789):
                
                # Assigning a Call to a Subscript (line 789):
                
                # Call to dictappend(...): (line 789)
                # Processing the call arguments (line 789)
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 789)
                k_69009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 46), 'k', False)
                # Getting the type of 'rd' (line 789)
                rd_69010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 43), 'rd', False)
                # Obtaining the member '__getitem__' of a type (line 789)
                getitem___69011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 43), rd_69010, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 789)
                subscript_call_result_69012 = invoke(stypy.reporting.localization.Localization(__file__, 789, 43), getitem___69011, k_69009)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 789)
                k_69013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 53), 'k', False)
                # Getting the type of 'ar' (line 789)
                ar_69014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 50), 'ar', False)
                # Obtaining the member '__getitem__' of a type (line 789)
                getitem___69015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 789, 50), ar_69014, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 789)
                subscript_call_result_69016 = invoke(stypy.reporting.localization.Localization(__file__, 789, 50), getitem___69015, k_69013)
                
                # Processing the call keyword arguments (line 789)
                kwargs_69017 = {}
                # Getting the type of 'dictappend' (line 789)
                dictappend_69008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 32), 'dictappend', False)
                # Calling dictappend(args, kwargs) (line 789)
                dictappend_call_result_69018 = invoke(stypy.reporting.localization.Localization(__file__, 789, 32), dictappend_69008, *[subscript_call_result_69012, subscript_call_result_69016], **kwargs_69017)
                
                # Getting the type of 'rd' (line 789)
                rd_69019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 24), 'rd')
                # Getting the type of 'k' (line 789)
                k_69020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 789, 27), 'k')
                # Storing an element on a container (line 789)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 789, 24), rd_69019, (k_69020, dictappend_call_result_69018))
                # SSA join for if statement (line 784)
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_68976:
                    # SSA join for if statement (line 783)
                    module_type_store = module_type_store.join_ssa_context()


            

            if more_types_in_union_68969:
                # SSA join for if statement (line 782)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_68932 and more_types_in_union_68933):
            # SSA join for if statement (line 777)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the else part of an if statement (line 774)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Subscript (line 791):
    
    # Assigning a Subscript to a Subscript (line 791):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 791)
    k_69021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 23), 'k')
    # Getting the type of 'ar' (line 791)
    ar_69022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 20), 'ar')
    # Obtaining the member '__getitem__' of a type (line 791)
    getitem___69023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 791, 20), ar_69022, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 791)
    subscript_call_result_69024 = invoke(stypy.reporting.localization.Localization(__file__, 791, 20), getitem___69023, k_69021)
    
    # Getting the type of 'rd' (line 791)
    rd_69025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 12), 'rd')
    # Getting the type of 'k' (line 791)
    k_69026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 791, 15), 'k')
    # Storing an element on a container (line 791)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 791, 12), rd_69025, (k_69026, subscript_call_result_69024))
    # SSA join for if statement (line 774)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'rd' (line 792)
    rd_69027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 11), 'rd')
    # Assigning a type to the variable 'stypy_return_type' (line 792)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 4), 'stypy_return_type', rd_69027)
    
    # ################# End of 'dictappend(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dictappend' in the type store
    # Getting the type of 'stypy_return_type' (line 766)
    stypy_return_type_69028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_69028)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dictappend'
    return stypy_return_type_69028

# Assigning a type to the variable 'dictappend' (line 766)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 0), 'dictappend', dictappend)

@norecursion
def applyrules(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'dict' (line 795)
    dict_69029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 29), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 795)
    
    defaults = [dict_69029]
    # Create a new context for function 'applyrules'
    module_type_store = module_type_store.open_function_context('applyrules', 795, 0, False)
    
    # Passed parameters checking function
    applyrules.stypy_localization = localization
    applyrules.stypy_type_of_self = None
    applyrules.stypy_type_store = module_type_store
    applyrules.stypy_function_name = 'applyrules'
    applyrules.stypy_param_names_list = ['rules', 'd', 'var']
    applyrules.stypy_varargs_param_name = None
    applyrules.stypy_kwargs_param_name = None
    applyrules.stypy_call_defaults = defaults
    applyrules.stypy_call_varargs = varargs
    applyrules.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'applyrules', ['rules', 'd', 'var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'applyrules', localization, ['rules', 'd', 'var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'applyrules(...)' code ##################

    
    # Assigning a Dict to a Name (line 796):
    
    # Assigning a Dict to a Name (line 796):
    
    # Obtaining an instance of the builtin type 'dict' (line 796)
    dict_69030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 796, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 796)
    
    # Assigning a type to the variable 'ret' (line 796)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 4), 'ret', dict_69030)
    
    # Type idiom detected: calculating its left and rigth part (line 797)
    # Getting the type of 'list' (line 797)
    list_69031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 25), 'list')
    # Getting the type of 'rules' (line 797)
    rules_69032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 797, 18), 'rules')
    
    (may_be_69033, more_types_in_union_69034) = may_be_subtype(list_69031, rules_69032)

    if may_be_69033:

        if more_types_in_union_69034:
            # Runtime conditional SSA (line 797)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'rules' (line 797)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 797, 4), 'rules', remove_not_subtype_from_union(rules_69032, list))
        
        # Getting the type of 'rules' (line 798)
        rules_69035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 17), 'rules')
        # Testing the type of a for loop iterable (line 798)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 798, 8), rules_69035)
        # Getting the type of the for loop variable (line 798)
        for_loop_var_69036 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 798, 8), rules_69035)
        # Assigning a type to the variable 'r' (line 798)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 8), 'r', for_loop_var_69036)
        # SSA begins for a for statement (line 798)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 799):
        
        # Assigning a Call to a Name (line 799):
        
        # Call to applyrules(...): (line 799)
        # Processing the call arguments (line 799)
        # Getting the type of 'r' (line 799)
        r_69038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 28), 'r', False)
        # Getting the type of 'd' (line 799)
        d_69039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 31), 'd', False)
        # Getting the type of 'var' (line 799)
        var_69040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 34), 'var', False)
        # Processing the call keyword arguments (line 799)
        kwargs_69041 = {}
        # Getting the type of 'applyrules' (line 799)
        applyrules_69037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 17), 'applyrules', False)
        # Calling applyrules(args, kwargs) (line 799)
        applyrules_call_result_69042 = invoke(stypy.reporting.localization.Localization(__file__, 799, 17), applyrules_69037, *[r_69038, d_69039, var_69040], **kwargs_69041)
        
        # Assigning a type to the variable 'rr' (line 799)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 12), 'rr', applyrules_call_result_69042)
        
        # Assigning a Call to a Name (line 800):
        
        # Assigning a Call to a Name (line 800):
        
        # Call to dictappend(...): (line 800)
        # Processing the call arguments (line 800)
        # Getting the type of 'ret' (line 800)
        ret_69044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 29), 'ret', False)
        # Getting the type of 'rr' (line 800)
        rr_69045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 34), 'rr', False)
        # Processing the call keyword arguments (line 800)
        kwargs_69046 = {}
        # Getting the type of 'dictappend' (line 800)
        dictappend_69043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 18), 'dictappend', False)
        # Calling dictappend(args, kwargs) (line 800)
        dictappend_call_result_69047 = invoke(stypy.reporting.localization.Localization(__file__, 800, 18), dictappend_69043, *[ret_69044, rr_69045], **kwargs_69046)
        
        # Assigning a type to the variable 'ret' (line 800)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 800, 12), 'ret', dictappend_call_result_69047)
        
        
        str_69048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 801, 15), 'str', '_break')
        # Getting the type of 'rr' (line 801)
        rr_69049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 801, 27), 'rr')
        # Applying the binary operator 'in' (line 801)
        result_contains_69050 = python_operator(stypy.reporting.localization.Localization(__file__, 801, 15), 'in', str_69048, rr_69049)
        
        # Testing the type of an if condition (line 801)
        if_condition_69051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 801, 12), result_contains_69050)
        # Assigning a type to the variable 'if_condition_69051' (line 801)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 801, 12), 'if_condition_69051', if_condition_69051)
        # SSA begins for if statement (line 801)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 801)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'ret' (line 803)
        ret_69052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 803)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 8), 'stypy_return_type', ret_69052)

        if more_types_in_union_69034:
            # SSA join for if statement (line 797)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    str_69053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 7), 'str', '_check')
    # Getting the type of 'rules' (line 804)
    rules_69054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 19), 'rules')
    # Applying the binary operator 'in' (line 804)
    result_contains_69055 = python_operator(stypy.reporting.localization.Localization(__file__, 804, 7), 'in', str_69053, rules_69054)
    
    
    
    # Call to (...): (line 804)
    # Processing the call arguments (line 804)
    # Getting the type of 'var' (line 804)
    var_69060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 50), 'var', False)
    # Processing the call keyword arguments (line 804)
    kwargs_69061 = {}
    
    # Obtaining the type of the subscript
    str_69056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 40), 'str', '_check')
    # Getting the type of 'rules' (line 804)
    rules_69057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 34), 'rules', False)
    # Obtaining the member '__getitem__' of a type (line 804)
    getitem___69058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 804, 34), rules_69057, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 804)
    subscript_call_result_69059 = invoke(stypy.reporting.localization.Localization(__file__, 804, 34), getitem___69058, str_69056)
    
    # Calling (args, kwargs) (line 804)
    _call_result_69062 = invoke(stypy.reporting.localization.Localization(__file__, 804, 34), subscript_call_result_69059, *[var_69060], **kwargs_69061)
    
    # Applying the 'not' unary operator (line 804)
    result_not__69063 = python_operator(stypy.reporting.localization.Localization(__file__, 804, 30), 'not', _call_result_69062)
    
    # Applying the binary operator 'and' (line 804)
    result_and_keyword_69064 = python_operator(stypy.reporting.localization.Localization(__file__, 804, 7), 'and', result_contains_69055, result_not__69063)
    
    # Testing the type of an if condition (line 804)
    if_condition_69065 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 804, 4), result_and_keyword_69064)
    # Assigning a type to the variable 'if_condition_69065' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'if_condition_69065', if_condition_69065)
    # SSA begins for if statement (line 804)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'ret' (line 805)
    ret_69066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 15), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 8), 'stypy_return_type', ret_69066)
    # SSA join for if statement (line 804)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_69067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 7), 'str', 'need')
    # Getting the type of 'rules' (line 806)
    rules_69068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 17), 'rules')
    # Applying the binary operator 'in' (line 806)
    result_contains_69069 = python_operator(stypy.reporting.localization.Localization(__file__, 806, 7), 'in', str_69067, rules_69068)
    
    # Testing the type of an if condition (line 806)
    if_condition_69070 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 806, 4), result_contains_69069)
    # Assigning a type to the variable 'if_condition_69070' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 4), 'if_condition_69070', if_condition_69070)
    # SSA begins for if statement (line 806)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 807):
    
    # Assigning a Call to a Name (line 807):
    
    # Call to applyrules(...): (line 807)
    # Processing the call arguments (line 807)
    
    # Obtaining an instance of the builtin type 'dict' (line 807)
    dict_69072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 25), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 807)
    # Adding element type (key, value) (line 807)
    str_69073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 26), 'str', 'needs')
    
    # Obtaining the type of the subscript
    str_69074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 807, 41), 'str', 'need')
    # Getting the type of 'rules' (line 807)
    rules_69075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 35), 'rules', False)
    # Obtaining the member '__getitem__' of a type (line 807)
    getitem___69076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 807, 35), rules_69075, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 807)
    subscript_call_result_69077 = invoke(stypy.reporting.localization.Localization(__file__, 807, 35), getitem___69076, str_69074)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 807, 25), dict_69072, (str_69073, subscript_call_result_69077))
    
    # Getting the type of 'd' (line 807)
    d_69078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 51), 'd', False)
    # Getting the type of 'var' (line 807)
    var_69079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 54), 'var', False)
    # Processing the call keyword arguments (line 807)
    kwargs_69080 = {}
    # Getting the type of 'applyrules' (line 807)
    applyrules_69071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 807, 14), 'applyrules', False)
    # Calling applyrules(args, kwargs) (line 807)
    applyrules_call_result_69081 = invoke(stypy.reporting.localization.Localization(__file__, 807, 14), applyrules_69071, *[dict_69072, d_69078, var_69079], **kwargs_69080)
    
    # Assigning a type to the variable 'res' (line 807)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 807, 8), 'res', applyrules_call_result_69081)
    
    
    str_69082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 11), 'str', 'needs')
    # Getting the type of 'res' (line 808)
    res_69083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 22), 'res')
    # Applying the binary operator 'in' (line 808)
    result_contains_69084 = python_operator(stypy.reporting.localization.Localization(__file__, 808, 11), 'in', str_69082, res_69083)
    
    # Testing the type of an if condition (line 808)
    if_condition_69085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 808, 8), result_contains_69084)
    # Assigning a type to the variable 'if_condition_69085' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 8), 'if_condition_69085', if_condition_69085)
    # SSA begins for if statement (line 808)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append_needs(...): (line 809)
    # Processing the call arguments (line 809)
    
    # Obtaining the type of the subscript
    str_69088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 809, 36), 'str', 'needs')
    # Getting the type of 'res' (line 809)
    res_69089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 32), 'res', False)
    # Obtaining the member '__getitem__' of a type (line 809)
    getitem___69090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 32), res_69089, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 809)
    subscript_call_result_69091 = invoke(stypy.reporting.localization.Localization(__file__, 809, 32), getitem___69090, str_69088)
    
    # Processing the call keyword arguments (line 809)
    kwargs_69092 = {}
    # Getting the type of 'cfuncs' (line 809)
    cfuncs_69086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 12), 'cfuncs', False)
    # Obtaining the member 'append_needs' of a type (line 809)
    append_needs_69087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 12), cfuncs_69086, 'append_needs')
    # Calling append_needs(args, kwargs) (line 809)
    append_needs_call_result_69093 = invoke(stypy.reporting.localization.Localization(__file__, 809, 12), append_needs_69087, *[subscript_call_result_69091], **kwargs_69092)
    
    # SSA join for if statement (line 808)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 806)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to keys(...): (line 811)
    # Processing the call keyword arguments (line 811)
    kwargs_69096 = {}
    # Getting the type of 'rules' (line 811)
    rules_69094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 811, 13), 'rules', False)
    # Obtaining the member 'keys' of a type (line 811)
    keys_69095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 811, 13), rules_69094, 'keys')
    # Calling keys(args, kwargs) (line 811)
    keys_call_result_69097 = invoke(stypy.reporting.localization.Localization(__file__, 811, 13), keys_69095, *[], **kwargs_69096)
    
    # Testing the type of a for loop iterable (line 811)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 811, 4), keys_call_result_69097)
    # Getting the type of the for loop variable (line 811)
    for_loop_var_69098 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 811, 4), keys_call_result_69097)
    # Assigning a type to the variable 'k' (line 811)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 811, 4), 'k', for_loop_var_69098)
    # SSA begins for a for statement (line 811)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'k' (line 812)
    k_69099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 812, 11), 'k')
    str_69100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 812, 16), 'str', 'separatorsfor')
    # Applying the binary operator '==' (line 812)
    result_eq_69101 = python_operator(stypy.reporting.localization.Localization(__file__, 812, 11), '==', k_69099, str_69100)
    
    # Testing the type of an if condition (line 812)
    if_condition_69102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 812, 8), result_eq_69101)
    # Assigning a type to the variable 'if_condition_69102' (line 812)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 812, 8), 'if_condition_69102', if_condition_69102)
    # SSA begins for if statement (line 812)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 813):
    
    # Assigning a Subscript to a Subscript (line 813):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 813)
    k_69103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 27), 'k')
    # Getting the type of 'rules' (line 813)
    rules_69104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 21), 'rules')
    # Obtaining the member '__getitem__' of a type (line 813)
    getitem___69105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 21), rules_69104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 813)
    subscript_call_result_69106 = invoke(stypy.reporting.localization.Localization(__file__, 813, 21), getitem___69105, k_69103)
    
    # Getting the type of 'ret' (line 813)
    ret_69107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 12), 'ret')
    # Getting the type of 'k' (line 813)
    k_69108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 16), 'k')
    # Storing an element on a container (line 813)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 813, 12), ret_69107, (k_69108, subscript_call_result_69106))
    # SSA join for if statement (line 812)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 815)
    # Getting the type of 'str' (line 815)
    str_69109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 32), 'str')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 815)
    k_69110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 28), 'k')
    # Getting the type of 'rules' (line 815)
    rules_69111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 22), 'rules')
    # Obtaining the member '__getitem__' of a type (line 815)
    getitem___69112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 22), rules_69111, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 815)
    subscript_call_result_69113 = invoke(stypy.reporting.localization.Localization(__file__, 815, 22), getitem___69112, k_69110)
    
    
    (may_be_69114, more_types_in_union_69115) = may_be_subtype(str_69109, subscript_call_result_69113)

    if may_be_69114:

        if more_types_in_union_69115:
            # Runtime conditional SSA (line 815)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Subscript (line 816):
        
        # Assigning a Call to a Subscript (line 816):
        
        # Call to replace(...): (line 816)
        # Processing the call arguments (line 816)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 816)
        k_69117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 35), 'k', False)
        # Getting the type of 'rules' (line 816)
        rules_69118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 29), 'rules', False)
        # Obtaining the member '__getitem__' of a type (line 816)
        getitem___69119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 816, 29), rules_69118, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 816)
        subscript_call_result_69120 = invoke(stypy.reporting.localization.Localization(__file__, 816, 29), getitem___69119, k_69117)
        
        # Getting the type of 'd' (line 816)
        d_69121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 39), 'd', False)
        # Processing the call keyword arguments (line 816)
        kwargs_69122 = {}
        # Getting the type of 'replace' (line 816)
        replace_69116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 21), 'replace', False)
        # Calling replace(args, kwargs) (line 816)
        replace_call_result_69123 = invoke(stypy.reporting.localization.Localization(__file__, 816, 21), replace_69116, *[subscript_call_result_69120, d_69121], **kwargs_69122)
        
        # Getting the type of 'ret' (line 816)
        ret_69124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 12), 'ret')
        # Getting the type of 'k' (line 816)
        k_69125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 816, 16), 'k')
        # Storing an element on a container (line 816)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 816, 12), ret_69124, (k_69125, replace_call_result_69123))

        if more_types_in_union_69115:
            # Runtime conditional SSA for else branch (line 815)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_69114) or more_types_in_union_69115):
        
        # Type idiom detected: calculating its left and rigth part (line 817)
        # Getting the type of 'list' (line 817)
        list_69126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 34), 'list')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 817)
        k_69127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 30), 'k')
        # Getting the type of 'rules' (line 817)
        rules_69128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 817, 24), 'rules')
        # Obtaining the member '__getitem__' of a type (line 817)
        getitem___69129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 817, 24), rules_69128, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 817)
        subscript_call_result_69130 = invoke(stypy.reporting.localization.Localization(__file__, 817, 24), getitem___69129, k_69127)
        
        
        (may_be_69131, more_types_in_union_69132) = may_be_subtype(list_69126, subscript_call_result_69130)

        if may_be_69131:

            if more_types_in_union_69132:
                # Runtime conditional SSA (line 817)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Subscript (line 818):
            
            # Assigning a List to a Subscript (line 818):
            
            # Obtaining an instance of the builtin type 'list' (line 818)
            list_69133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 21), 'list')
            # Adding type elements to the builtin type 'list' instance (line 818)
            
            # Getting the type of 'ret' (line 818)
            ret_69134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 12), 'ret')
            # Getting the type of 'k' (line 818)
            k_69135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 16), 'k')
            # Storing an element on a container (line 818)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 818, 12), ret_69134, (k_69135, list_69133))
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 819)
            k_69136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 27), 'k')
            # Getting the type of 'rules' (line 819)
            rules_69137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 819, 21), 'rules')
            # Obtaining the member '__getitem__' of a type (line 819)
            getitem___69138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 819, 21), rules_69137, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 819)
            subscript_call_result_69139 = invoke(stypy.reporting.localization.Localization(__file__, 819, 21), getitem___69138, k_69136)
            
            # Testing the type of a for loop iterable (line 819)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 819, 12), subscript_call_result_69139)
            # Getting the type of the for loop variable (line 819)
            for_loop_var_69140 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 819, 12), subscript_call_result_69139)
            # Assigning a type to the variable 'i' (line 819)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 819, 12), 'i', for_loop_var_69140)
            # SSA begins for a for statement (line 819)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 820):
            
            # Assigning a Call to a Name (line 820):
            
            # Call to applyrules(...): (line 820)
            # Processing the call arguments (line 820)
            
            # Obtaining an instance of the builtin type 'dict' (line 820)
            dict_69142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 820, 32), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 820)
            # Adding element type (key, value) (line 820)
            # Getting the type of 'k' (line 820)
            k_69143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 33), 'k', False)
            # Getting the type of 'i' (line 820)
            i_69144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 36), 'i', False)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 820, 32), dict_69142, (k_69143, i_69144))
            
            # Getting the type of 'd' (line 820)
            d_69145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 40), 'd', False)
            # Getting the type of 'var' (line 820)
            var_69146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 43), 'var', False)
            # Processing the call keyword arguments (line 820)
            kwargs_69147 = {}
            # Getting the type of 'applyrules' (line 820)
            applyrules_69141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 21), 'applyrules', False)
            # Calling applyrules(args, kwargs) (line 820)
            applyrules_call_result_69148 = invoke(stypy.reporting.localization.Localization(__file__, 820, 21), applyrules_69141, *[dict_69142, d_69145, var_69146], **kwargs_69147)
            
            # Assigning a type to the variable 'ar' (line 820)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 16), 'ar', applyrules_call_result_69148)
            
            
            # Getting the type of 'k' (line 821)
            k_69149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 19), 'k')
            # Getting the type of 'ar' (line 821)
            ar_69150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 24), 'ar')
            # Applying the binary operator 'in' (line 821)
            result_contains_69151 = python_operator(stypy.reporting.localization.Localization(__file__, 821, 19), 'in', k_69149, ar_69150)
            
            # Testing the type of an if condition (line 821)
            if_condition_69152 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 821, 16), result_contains_69151)
            # Assigning a type to the variable 'if_condition_69152' (line 821)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 821, 16), 'if_condition_69152', if_condition_69152)
            # SSA begins for if statement (line 821)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 822)
            # Processing the call arguments (line 822)
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 822)
            k_69158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 37), 'k', False)
            # Getting the type of 'ar' (line 822)
            ar_69159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 34), 'ar', False)
            # Obtaining the member '__getitem__' of a type (line 822)
            getitem___69160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 34), ar_69159, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 822)
            subscript_call_result_69161 = invoke(stypy.reporting.localization.Localization(__file__, 822, 34), getitem___69160, k_69158)
            
            # Processing the call keyword arguments (line 822)
            kwargs_69162 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 822)
            k_69153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 24), 'k', False)
            # Getting the type of 'ret' (line 822)
            ret_69154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 822, 20), 'ret', False)
            # Obtaining the member '__getitem__' of a type (line 822)
            getitem___69155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 20), ret_69154, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 822)
            subscript_call_result_69156 = invoke(stypy.reporting.localization.Localization(__file__, 822, 20), getitem___69155, k_69153)
            
            # Obtaining the member 'append' of a type (line 822)
            append_69157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 822, 20), subscript_call_result_69156, 'append')
            # Calling append(args, kwargs) (line 822)
            append_call_result_69163 = invoke(stypy.reporting.localization.Localization(__file__, 822, 20), append_69157, *[subscript_call_result_69161], **kwargs_69162)
            
            # SSA join for if statement (line 821)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_69132:
                # Runtime conditional SSA for else branch (line 817)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_69131) or more_types_in_union_69132):
            
            
            
            # Obtaining the type of the subscript
            int_69164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 15), 'int')
            # Getting the type of 'k' (line 823)
            k_69165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 823, 13), 'k')
            # Obtaining the member '__getitem__' of a type (line 823)
            getitem___69166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 823, 13), k_69165, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 823)
            subscript_call_result_69167 = invoke(stypy.reporting.localization.Localization(__file__, 823, 13), getitem___69166, int_69164)
            
            str_69168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 823, 21), 'str', '_')
            # Applying the binary operator '==' (line 823)
            result_eq_69169 = python_operator(stypy.reporting.localization.Localization(__file__, 823, 13), '==', subscript_call_result_69167, str_69168)
            
            # Testing the type of an if condition (line 823)
            if_condition_69170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 823, 13), result_eq_69169)
            # Assigning a type to the variable 'if_condition_69170' (line 823)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 823, 13), 'if_condition_69170', if_condition_69170)
            # SSA begins for if statement (line 823)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA branch for the else part of an if statement (line 823)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 825)
            # Getting the type of 'dict' (line 825)
            dict_69171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 34), 'dict')
            
            # Obtaining the type of the subscript
            # Getting the type of 'k' (line 825)
            k_69172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 30), 'k')
            # Getting the type of 'rules' (line 825)
            rules_69173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 24), 'rules')
            # Obtaining the member '__getitem__' of a type (line 825)
            getitem___69174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 24), rules_69173, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 825)
            subscript_call_result_69175 = invoke(stypy.reporting.localization.Localization(__file__, 825, 24), getitem___69174, k_69172)
            
            
            (may_be_69176, more_types_in_union_69177) = may_be_subtype(dict_69171, subscript_call_result_69175)

            if may_be_69176:

                if more_types_in_union_69177:
                    # Runtime conditional SSA (line 825)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a List to a Subscript (line 826):
                
                # Assigning a List to a Subscript (line 826):
                
                # Obtaining an instance of the builtin type 'list' (line 826)
                list_69178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 21), 'list')
                # Adding type elements to the builtin type 'list' instance (line 826)
                
                # Getting the type of 'ret' (line 826)
                ret_69179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 12), 'ret')
                # Getting the type of 'k' (line 826)
                k_69180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 16), 'k')
                # Storing an element on a container (line 826)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 826, 12), ret_69179, (k_69180, list_69178))
                
                
                # Call to keys(...): (line 827)
                # Processing the call keyword arguments (line 827)
                kwargs_69186 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 827)
                k_69181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 28), 'k', False)
                # Getting the type of 'rules' (line 827)
                rules_69182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 22), 'rules', False)
                # Obtaining the member '__getitem__' of a type (line 827)
                getitem___69183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 22), rules_69182, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 827)
                subscript_call_result_69184 = invoke(stypy.reporting.localization.Localization(__file__, 827, 22), getitem___69183, k_69181)
                
                # Obtaining the member 'keys' of a type (line 827)
                keys_69185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 22), subscript_call_result_69184, 'keys')
                # Calling keys(args, kwargs) (line 827)
                keys_call_result_69187 = invoke(stypy.reporting.localization.Localization(__file__, 827, 22), keys_69185, *[], **kwargs_69186)
                
                # Testing the type of a for loop iterable (line 827)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 827, 12), keys_call_result_69187)
                # Getting the type of the for loop variable (line 827)
                for_loop_var_69188 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 827, 12), keys_call_result_69187)
                # Assigning a type to the variable 'k1' (line 827)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 12), 'k1', for_loop_var_69188)
                # SSA begins for a for statement (line 827)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Evaluating a boolean operation
                
                # Call to isinstance(...): (line 828)
                # Processing the call arguments (line 828)
                # Getting the type of 'k1' (line 828)
                k1_69190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 30), 'k1', False)
                # Getting the type of 'types' (line 828)
                types_69191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 34), 'types', False)
                # Obtaining the member 'FunctionType' of a type (line 828)
                FunctionType_69192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 34), types_69191, 'FunctionType')
                # Processing the call keyword arguments (line 828)
                kwargs_69193 = {}
                # Getting the type of 'isinstance' (line 828)
                isinstance_69189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 19), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 828)
                isinstance_call_result_69194 = invoke(stypy.reporting.localization.Localization(__file__, 828, 19), isinstance_69189, *[k1_69190, FunctionType_69192], **kwargs_69193)
                
                
                # Call to k1(...): (line 828)
                # Processing the call arguments (line 828)
                # Getting the type of 'var' (line 828)
                var_69196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 61), 'var', False)
                # Processing the call keyword arguments (line 828)
                kwargs_69197 = {}
                # Getting the type of 'k1' (line 828)
                k1_69195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 58), 'k1', False)
                # Calling k1(args, kwargs) (line 828)
                k1_call_result_69198 = invoke(stypy.reporting.localization.Localization(__file__, 828, 58), k1_69195, *[var_69196], **kwargs_69197)
                
                # Applying the binary operator 'and' (line 828)
                result_and_keyword_69199 = python_operator(stypy.reporting.localization.Localization(__file__, 828, 19), 'and', isinstance_call_result_69194, k1_call_result_69198)
                
                # Testing the type of an if condition (line 828)
                if_condition_69200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 828, 16), result_and_keyword_69199)
                # Assigning a type to the variable 'if_condition_69200' (line 828)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 828, 16), 'if_condition_69200', if_condition_69200)
                # SSA begins for if statement (line 828)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Type idiom detected: calculating its left and rigth part (line 829)
                # Getting the type of 'list' (line 829)
                list_69201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 48), 'list')
                
                # Obtaining the type of the subscript
                # Getting the type of 'k1' (line 829)
                k1_69202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 43), 'k1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 829)
                k_69203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 40), 'k')
                # Getting the type of 'rules' (line 829)
                rules_69204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 829, 34), 'rules')
                # Obtaining the member '__getitem__' of a type (line 829)
                getitem___69205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 34), rules_69204, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 829)
                subscript_call_result_69206 = invoke(stypy.reporting.localization.Localization(__file__, 829, 34), getitem___69205, k_69203)
                
                # Obtaining the member '__getitem__' of a type (line 829)
                getitem___69207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 829, 34), subscript_call_result_69206, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 829)
                subscript_call_result_69208 = invoke(stypy.reporting.localization.Localization(__file__, 829, 34), getitem___69207, k1_69202)
                
                
                (may_be_69209, more_types_in_union_69210) = may_be_subtype(list_69201, subscript_call_result_69208)

                if may_be_69209:

                    if more_types_in_union_69210:
                        # Runtime conditional SSA (line 829)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'k1' (line 830)
                    k1_69211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 42), 'k1')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'k' (line 830)
                    k_69212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 39), 'k')
                    # Getting the type of 'rules' (line 830)
                    rules_69213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 33), 'rules')
                    # Obtaining the member '__getitem__' of a type (line 830)
                    getitem___69214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 33), rules_69213, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 830)
                    subscript_call_result_69215 = invoke(stypy.reporting.localization.Localization(__file__, 830, 33), getitem___69214, k_69212)
                    
                    # Obtaining the member '__getitem__' of a type (line 830)
                    getitem___69216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 33), subscript_call_result_69215, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 830)
                    subscript_call_result_69217 = invoke(stypy.reporting.localization.Localization(__file__, 830, 33), getitem___69216, k1_69211)
                    
                    # Testing the type of a for loop iterable (line 830)
                    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 830, 24), subscript_call_result_69217)
                    # Getting the type of the for loop variable (line 830)
                    for_loop_var_69218 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 830, 24), subscript_call_result_69217)
                    # Assigning a type to the variable 'i' (line 830)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 830, 24), 'i', for_loop_var_69218)
                    # SSA begins for a for statement (line 830)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Type idiom detected: calculating its left and rigth part (line 831)
                    # Getting the type of 'dict' (line 831)
                    dict_69219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 45), 'dict')
                    # Getting the type of 'i' (line 831)
                    i_69220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 831, 42), 'i')
                    
                    (may_be_69221, more_types_in_union_69222) = may_be_subtype(dict_69219, i_69220)

                    if may_be_69221:

                        if more_types_in_union_69222:
                            # Runtime conditional SSA (line 831)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'i' (line 831)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 831, 28), 'i', remove_not_subtype_from_union(i_69220, dict))
                        
                        # Assigning a Call to a Name (line 832):
                        
                        # Assigning a Call to a Name (line 832):
                        
                        # Call to applyrules(...): (line 832)
                        # Processing the call arguments (line 832)
                        
                        # Obtaining an instance of the builtin type 'dict' (line 832)
                        dict_69224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 49), 'dict')
                        # Adding type elements to the builtin type 'dict' instance (line 832)
                        # Adding element type (key, value) (line 832)
                        str_69225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 832, 50), 'str', 'supertext')
                        # Getting the type of 'i' (line 832)
                        i_69226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 63), 'i', False)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 832, 49), dict_69224, (str_69225, i_69226))
                        
                        # Getting the type of 'd' (line 832)
                        d_69227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 67), 'd', False)
                        # Getting the type of 'var' (line 832)
                        var_69228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 70), 'var', False)
                        # Processing the call keyword arguments (line 832)
                        kwargs_69229 = {}
                        # Getting the type of 'applyrules' (line 832)
                        applyrules_69223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 38), 'applyrules', False)
                        # Calling applyrules(args, kwargs) (line 832)
                        applyrules_call_result_69230 = invoke(stypy.reporting.localization.Localization(__file__, 832, 38), applyrules_69223, *[dict_69224, d_69227, var_69228], **kwargs_69229)
                        
                        # Assigning a type to the variable 'res' (line 832)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 32), 'res', applyrules_call_result_69230)
                        
                        
                        str_69231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 833, 35), 'str', 'supertext')
                        # Getting the type of 'res' (line 833)
                        res_69232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 833, 50), 'res')
                        # Applying the binary operator 'in' (line 833)
                        result_contains_69233 = python_operator(stypy.reporting.localization.Localization(__file__, 833, 35), 'in', str_69231, res_69232)
                        
                        # Testing the type of an if condition (line 833)
                        if_condition_69234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 833, 32), result_contains_69233)
                        # Assigning a type to the variable 'if_condition_69234' (line 833)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 833, 32), 'if_condition_69234', if_condition_69234)
                        # SSA begins for if statement (line 833)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Subscript to a Name (line 834):
                        
                        # Assigning a Subscript to a Name (line 834):
                        
                        # Obtaining the type of the subscript
                        str_69235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 834, 44), 'str', 'supertext')
                        # Getting the type of 'res' (line 834)
                        res_69236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 834, 40), 'res')
                        # Obtaining the member '__getitem__' of a type (line 834)
                        getitem___69237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 834, 40), res_69236, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 834)
                        subscript_call_result_69238 = invoke(stypy.reporting.localization.Localization(__file__, 834, 40), getitem___69237, str_69235)
                        
                        # Assigning a type to the variable 'i' (line 834)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 834, 36), 'i', subscript_call_result_69238)
                        # SSA branch for the else part of an if statement (line 833)
                        module_type_store.open_ssa_branch('else')
                        
                        # Assigning a Str to a Name (line 836):
                        
                        # Assigning a Str to a Name (line 836):
                        str_69239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 836, 40), 'str', '')
                        # Assigning a type to the variable 'i' (line 836)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 836, 36), 'i', str_69239)
                        # SSA join for if statement (line 833)
                        module_type_store = module_type_store.join_ssa_context()
                        

                        if more_types_in_union_69222:
                            # SSA join for if statement (line 831)
                            module_type_store = module_type_store.join_ssa_context()


                    
                    
                    # Call to append(...): (line 837)
                    # Processing the call arguments (line 837)
                    
                    # Call to replace(...): (line 837)
                    # Processing the call arguments (line 837)
                    # Getting the type of 'i' (line 837)
                    i_69246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 50), 'i', False)
                    # Getting the type of 'd' (line 837)
                    d_69247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 53), 'd', False)
                    # Processing the call keyword arguments (line 837)
                    kwargs_69248 = {}
                    # Getting the type of 'replace' (line 837)
                    replace_69245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 42), 'replace', False)
                    # Calling replace(args, kwargs) (line 837)
                    replace_call_result_69249 = invoke(stypy.reporting.localization.Localization(__file__, 837, 42), replace_69245, *[i_69246, d_69247], **kwargs_69248)
                    
                    # Processing the call keyword arguments (line 837)
                    kwargs_69250 = {}
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'k' (line 837)
                    k_69240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 32), 'k', False)
                    # Getting the type of 'ret' (line 837)
                    ret_69241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 837, 28), 'ret', False)
                    # Obtaining the member '__getitem__' of a type (line 837)
                    getitem___69242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 28), ret_69241, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 837)
                    subscript_call_result_69243 = invoke(stypy.reporting.localization.Localization(__file__, 837, 28), getitem___69242, k_69240)
                    
                    # Obtaining the member 'append' of a type (line 837)
                    append_69244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 837, 28), subscript_call_result_69243, 'append')
                    # Calling append(args, kwargs) (line 837)
                    append_call_result_69251 = invoke(stypy.reporting.localization.Localization(__file__, 837, 28), append_69244, *[replace_call_result_69249], **kwargs_69250)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()
                    

                    if more_types_in_union_69210:
                        # Runtime conditional SSA for else branch (line 829)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_69209) or more_types_in_union_69210):
                    
                    # Assigning a Subscript to a Name (line 839):
                    
                    # Assigning a Subscript to a Name (line 839):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'k1' (line 839)
                    k1_69252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 37), 'k1')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'k' (line 839)
                    k_69253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 34), 'k')
                    # Getting the type of 'rules' (line 839)
                    rules_69254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 839, 28), 'rules')
                    # Obtaining the member '__getitem__' of a type (line 839)
                    getitem___69255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 28), rules_69254, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 839)
                    subscript_call_result_69256 = invoke(stypy.reporting.localization.Localization(__file__, 839, 28), getitem___69255, k_69253)
                    
                    # Obtaining the member '__getitem__' of a type (line 839)
                    getitem___69257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 839, 28), subscript_call_result_69256, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 839)
                    subscript_call_result_69258 = invoke(stypy.reporting.localization.Localization(__file__, 839, 28), getitem___69257, k1_69252)
                    
                    # Assigning a type to the variable 'i' (line 839)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 839, 24), 'i', subscript_call_result_69258)
                    
                    # Type idiom detected: calculating its left and rigth part (line 840)
                    # Getting the type of 'dict' (line 840)
                    dict_69259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 41), 'dict')
                    # Getting the type of 'i' (line 840)
                    i_69260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 840, 38), 'i')
                    
                    (may_be_69261, more_types_in_union_69262) = may_be_subtype(dict_69259, i_69260)

                    if may_be_69261:

                        if more_types_in_union_69262:
                            # Runtime conditional SSA (line 840)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'i' (line 840)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 840, 24), 'i', remove_not_subtype_from_union(i_69260, dict))
                        
                        # Assigning a Call to a Name (line 841):
                        
                        # Assigning a Call to a Name (line 841):
                        
                        # Call to applyrules(...): (line 841)
                        # Processing the call arguments (line 841)
                        
                        # Obtaining an instance of the builtin type 'dict' (line 841)
                        dict_69264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 45), 'dict')
                        # Adding type elements to the builtin type 'dict' instance (line 841)
                        # Adding element type (key, value) (line 841)
                        str_69265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 841, 46), 'str', 'supertext')
                        # Getting the type of 'i' (line 841)
                        i_69266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 59), 'i', False)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 841, 45), dict_69264, (str_69265, i_69266))
                        
                        # Getting the type of 'd' (line 841)
                        d_69267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 63), 'd', False)
                        # Processing the call keyword arguments (line 841)
                        kwargs_69268 = {}
                        # Getting the type of 'applyrules' (line 841)
                        applyrules_69263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 841, 34), 'applyrules', False)
                        # Calling applyrules(args, kwargs) (line 841)
                        applyrules_call_result_69269 = invoke(stypy.reporting.localization.Localization(__file__, 841, 34), applyrules_69263, *[dict_69264, d_69267], **kwargs_69268)
                        
                        # Assigning a type to the variable 'res' (line 841)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 841, 28), 'res', applyrules_call_result_69269)
                        
                        
                        str_69270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 842, 31), 'str', 'supertext')
                        # Getting the type of 'res' (line 842)
                        res_69271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 842, 46), 'res')
                        # Applying the binary operator 'in' (line 842)
                        result_contains_69272 = python_operator(stypy.reporting.localization.Localization(__file__, 842, 31), 'in', str_69270, res_69271)
                        
                        # Testing the type of an if condition (line 842)
                        if_condition_69273 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 842, 28), result_contains_69272)
                        # Assigning a type to the variable 'if_condition_69273' (line 842)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 842, 28), 'if_condition_69273', if_condition_69273)
                        # SSA begins for if statement (line 842)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Subscript to a Name (line 843):
                        
                        # Assigning a Subscript to a Name (line 843):
                        
                        # Obtaining the type of the subscript
                        str_69274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 843, 40), 'str', 'supertext')
                        # Getting the type of 'res' (line 843)
                        res_69275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 843, 36), 'res')
                        # Obtaining the member '__getitem__' of a type (line 843)
                        getitem___69276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 843, 36), res_69275, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 843)
                        subscript_call_result_69277 = invoke(stypy.reporting.localization.Localization(__file__, 843, 36), getitem___69276, str_69274)
                        
                        # Assigning a type to the variable 'i' (line 843)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 843, 32), 'i', subscript_call_result_69277)
                        # SSA branch for the else part of an if statement (line 842)
                        module_type_store.open_ssa_branch('else')
                        
                        # Assigning a Str to a Name (line 845):
                        
                        # Assigning a Str to a Name (line 845):
                        str_69278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 845, 36), 'str', '')
                        # Assigning a type to the variable 'i' (line 845)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 845, 32), 'i', str_69278)
                        # SSA join for if statement (line 842)
                        module_type_store = module_type_store.join_ssa_context()
                        

                        if more_types_in_union_69262:
                            # SSA join for if statement (line 840)
                            module_type_store = module_type_store.join_ssa_context()


                    
                    
                    # Call to append(...): (line 846)
                    # Processing the call arguments (line 846)
                    
                    # Call to replace(...): (line 846)
                    # Processing the call arguments (line 846)
                    # Getting the type of 'i' (line 846)
                    i_69285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 46), 'i', False)
                    # Getting the type of 'd' (line 846)
                    d_69286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 49), 'd', False)
                    # Processing the call keyword arguments (line 846)
                    kwargs_69287 = {}
                    # Getting the type of 'replace' (line 846)
                    replace_69284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 38), 'replace', False)
                    # Calling replace(args, kwargs) (line 846)
                    replace_call_result_69288 = invoke(stypy.reporting.localization.Localization(__file__, 846, 38), replace_69284, *[i_69285, d_69286], **kwargs_69287)
                    
                    # Processing the call keyword arguments (line 846)
                    kwargs_69289 = {}
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'k' (line 846)
                    k_69279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 28), 'k', False)
                    # Getting the type of 'ret' (line 846)
                    ret_69280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 846, 24), 'ret', False)
                    # Obtaining the member '__getitem__' of a type (line 846)
                    getitem___69281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 24), ret_69280, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 846)
                    subscript_call_result_69282 = invoke(stypy.reporting.localization.Localization(__file__, 846, 24), getitem___69281, k_69279)
                    
                    # Obtaining the member 'append' of a type (line 846)
                    append_69283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 846, 24), subscript_call_result_69282, 'append')
                    # Calling append(args, kwargs) (line 846)
                    append_call_result_69290 = invoke(stypy.reporting.localization.Localization(__file__, 846, 24), append_69283, *[replace_call_result_69288], **kwargs_69289)
                    

                    if (may_be_69209 and more_types_in_union_69210):
                        # SSA join for if statement (line 829)
                        module_type_store = module_type_store.join_ssa_context()


                
                # SSA join for if statement (line 828)
                module_type_store = module_type_store.join_ssa_context()
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_69177:
                    # Runtime conditional SSA for else branch (line 825)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_69176) or more_types_in_union_69177):
                
                # Call to errmess(...): (line 848)
                # Processing the call arguments (line 848)
                str_69292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 848, 20), 'str', 'applyrules: ignoring rule %s.\n')
                
                # Call to repr(...): (line 848)
                # Processing the call arguments (line 848)
                
                # Obtaining the type of the subscript
                # Getting the type of 'k' (line 848)
                k_69294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 67), 'k', False)
                # Getting the type of 'rules' (line 848)
                rules_69295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 61), 'rules', False)
                # Obtaining the member '__getitem__' of a type (line 848)
                getitem___69296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 848, 61), rules_69295, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 848)
                subscript_call_result_69297 = invoke(stypy.reporting.localization.Localization(__file__, 848, 61), getitem___69296, k_69294)
                
                # Processing the call keyword arguments (line 848)
                kwargs_69298 = {}
                # Getting the type of 'repr' (line 848)
                repr_69293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 56), 'repr', False)
                # Calling repr(args, kwargs) (line 848)
                repr_call_result_69299 = invoke(stypy.reporting.localization.Localization(__file__, 848, 56), repr_69293, *[subscript_call_result_69297], **kwargs_69298)
                
                # Applying the binary operator '%' (line 848)
                result_mod_69300 = python_operator(stypy.reporting.localization.Localization(__file__, 848, 20), '%', str_69292, repr_call_result_69299)
                
                # Processing the call keyword arguments (line 848)
                kwargs_69301 = {}
                # Getting the type of 'errmess' (line 848)
                errmess_69291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 848, 12), 'errmess', False)
                # Calling errmess(args, kwargs) (line 848)
                errmess_call_result_69302 = invoke(stypy.reporting.localization.Localization(__file__, 848, 12), errmess_69291, *[result_mod_69300], **kwargs_69301)
                

                if (may_be_69176 and more_types_in_union_69177):
                    # SSA join for if statement (line 825)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 823)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_69131 and more_types_in_union_69132):
                # SSA join for if statement (line 817)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_69114 and more_types_in_union_69115):
            # SSA join for if statement (line 815)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 849)
    # Getting the type of 'list' (line 849)
    list_69303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 30), 'list')
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 849)
    k_69304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 26), 'k')
    # Getting the type of 'ret' (line 849)
    ret_69305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 849, 22), 'ret')
    # Obtaining the member '__getitem__' of a type (line 849)
    getitem___69306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 849, 22), ret_69305, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 849)
    subscript_call_result_69307 = invoke(stypy.reporting.localization.Localization(__file__, 849, 22), getitem___69306, k_69304)
    
    
    (may_be_69308, more_types_in_union_69309) = may_be_subtype(list_69303, subscript_call_result_69307)

    if may_be_69308:

        if more_types_in_union_69309:
            # Runtime conditional SSA (line 849)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        
        
        # Call to len(...): (line 850)
        # Processing the call arguments (line 850)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 850)
        k_69311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 23), 'k', False)
        # Getting the type of 'ret' (line 850)
        ret_69312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 19), 'ret', False)
        # Obtaining the member '__getitem__' of a type (line 850)
        getitem___69313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 850, 19), ret_69312, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 850)
        subscript_call_result_69314 = invoke(stypy.reporting.localization.Localization(__file__, 850, 19), getitem___69313, k_69311)
        
        # Processing the call keyword arguments (line 850)
        kwargs_69315 = {}
        # Getting the type of 'len' (line 850)
        len_69310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 850, 15), 'len', False)
        # Calling len(args, kwargs) (line 850)
        len_call_result_69316 = invoke(stypy.reporting.localization.Localization(__file__, 850, 15), len_69310, *[subscript_call_result_69314], **kwargs_69315)
        
        int_69317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 850, 30), 'int')
        # Applying the binary operator '==' (line 850)
        result_eq_69318 = python_operator(stypy.reporting.localization.Localization(__file__, 850, 15), '==', len_call_result_69316, int_69317)
        
        # Testing the type of an if condition (line 850)
        if_condition_69319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 850, 12), result_eq_69318)
        # Assigning a type to the variable 'if_condition_69319' (line 850)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 850, 12), 'if_condition_69319', if_condition_69319)
        # SSA begins for if statement (line 850)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 851):
        
        # Assigning a Subscript to a Subscript (line 851):
        
        # Obtaining the type of the subscript
        int_69320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, 32), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 851)
        k_69321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 29), 'k')
        # Getting the type of 'ret' (line 851)
        ret_69322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 25), 'ret')
        # Obtaining the member '__getitem__' of a type (line 851)
        getitem___69323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 25), ret_69322, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 851)
        subscript_call_result_69324 = invoke(stypy.reporting.localization.Localization(__file__, 851, 25), getitem___69323, k_69321)
        
        # Obtaining the member '__getitem__' of a type (line 851)
        getitem___69325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 851, 25), subscript_call_result_69324, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 851)
        subscript_call_result_69326 = invoke(stypy.reporting.localization.Localization(__file__, 851, 25), getitem___69325, int_69320)
        
        # Getting the type of 'ret' (line 851)
        ret_69327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 16), 'ret')
        # Getting the type of 'k' (line 851)
        k_69328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 851, 20), 'k')
        # Storing an element on a container (line 851)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 851, 16), ret_69327, (k_69328, subscript_call_result_69326))
        # SSA join for if statement (line 850)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 852)
        k_69329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 19), 'k')
        # Getting the type of 'ret' (line 852)
        ret_69330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 852, 15), 'ret')
        # Obtaining the member '__getitem__' of a type (line 852)
        getitem___69331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 852, 15), ret_69330, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 852)
        subscript_call_result_69332 = invoke(stypy.reporting.localization.Localization(__file__, 852, 15), getitem___69331, k_69329)
        
        
        # Obtaining an instance of the builtin type 'list' (line 852)
        list_69333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 852, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 852)
        
        # Applying the binary operator '==' (line 852)
        result_eq_69334 = python_operator(stypy.reporting.localization.Localization(__file__, 852, 15), '==', subscript_call_result_69332, list_69333)
        
        # Testing the type of an if condition (line 852)
        if_condition_69335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 852, 12), result_eq_69334)
        # Assigning a type to the variable 'if_condition_69335' (line 852)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 852, 12), 'if_condition_69335', if_condition_69335)
        # SSA begins for if statement (line 852)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Deleting a member
        # Getting the type of 'ret' (line 853)
        ret_69336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 20), 'ret')
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 853)
        k_69337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 24), 'k')
        # Getting the type of 'ret' (line 853)
        ret_69338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 853, 20), 'ret')
        # Obtaining the member '__getitem__' of a type (line 853)
        getitem___69339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 853, 20), ret_69338, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 853)
        subscript_call_result_69340 = invoke(stypy.reporting.localization.Localization(__file__, 853, 20), getitem___69339, k_69337)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 853, 16), ret_69336, subscript_call_result_69340)
        # SSA join for if statement (line 852)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_69309:
            # SSA join for if statement (line 849)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 854)
    ret_69341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 854, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 854)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 854, 4), 'stypy_return_type', ret_69341)
    
    # ################# End of 'applyrules(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'applyrules' in the type store
    # Getting the type of 'stypy_return_type' (line 795)
    stypy_return_type_69342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_69342)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'applyrules'
    return stypy_return_type_69342

# Assigning a type to the variable 'applyrules' (line 795)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 0), 'applyrules', applyrules)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
