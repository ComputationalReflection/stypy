
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: import sys
5: import types
6: import re
7: import warnings
8: 
9: from numpy.core.numerictypes import issubclass_, issubsctype, issubdtype
10: from numpy.core import ndarray, ufunc, asarray
11: 
12: # getargspec and formatargspec were removed in Python 3.6
13: from numpy.compat import getargspec, formatargspec
14: 
15: __all__ = [
16:     'issubclass_', 'issubsctype', 'issubdtype', 'deprecate',
17:     'deprecate_with_doc', 'get_include', 'info', 'source', 'who',
18:     'lookfor', 'byte_bounds', 'safe_eval'
19:     ]
20: 
21: def get_include():
22:     '''
23:     Return the directory that contains the NumPy \\*.h header files.
24: 
25:     Extension modules that need to compile against NumPy should use this
26:     function to locate the appropriate include directory.
27: 
28:     Notes
29:     -----
30:     When using ``distutils``, for example in ``setup.py``.
31:     ::
32: 
33:         import numpy as np
34:         ...
35:         Extension('extension_name', ...
36:                 include_dirs=[np.get_include()])
37:         ...
38: 
39:     '''
40:     import numpy
41:     if numpy.show_config is None:
42:         # running from numpy source directory
43:         d = os.path.join(os.path.dirname(numpy.__file__), 'core', 'include')
44:     else:
45:         # using installed numpy core headers
46:         import numpy.core as core
47:         d = os.path.join(os.path.dirname(core.__file__), 'include')
48:     return d
49: 
50: 
51: def _set_function_name(func, name):
52:     func.__name__ = name
53:     return func
54: 
55: 
56: class _Deprecate(object):
57:     '''
58:     Decorator class to deprecate old functions.
59: 
60:     Refer to `deprecate` for details.
61: 
62:     See Also
63:     --------
64:     deprecate
65: 
66:     '''
67: 
68:     def __init__(self, old_name=None, new_name=None, message=None):
69:         self.old_name = old_name
70:         self.new_name = new_name
71:         self.message = message
72: 
73:     def __call__(self, func, *args, **kwargs):
74:         '''
75:         Decorator call.  Refer to ``decorate``.
76: 
77:         '''
78:         old_name = self.old_name
79:         new_name = self.new_name
80:         message = self.message
81: 
82:         import warnings
83:         if old_name is None:
84:             try:
85:                 old_name = func.__name__
86:             except AttributeError:
87:                 old_name = func.__name__
88:         if new_name is None:
89:             depdoc = "`%s` is deprecated!" % old_name
90:         else:
91:             depdoc = "`%s` is deprecated, use `%s` instead!" % \
92:                      (old_name, new_name)
93: 
94:         if message is not None:
95:             depdoc += "\n" + message
96: 
97:         def newfunc(*args,**kwds):
98:             '''`arrayrange` is deprecated, use `arange` instead!'''
99:             warnings.warn(depdoc, DeprecationWarning)
100:             return func(*args, **kwds)
101: 
102:         newfunc = _set_function_name(newfunc, old_name)
103:         doc = func.__doc__
104:         if doc is None:
105:             doc = depdoc
106:         else:
107:             doc = '\n\n'.join([depdoc, doc])
108:         newfunc.__doc__ = doc
109:         try:
110:             d = func.__dict__
111:         except AttributeError:
112:             pass
113:         else:
114:             newfunc.__dict__.update(d)
115:         return newfunc
116: 
117: def deprecate(*args, **kwargs):
118:     '''
119:     Issues a DeprecationWarning, adds warning to `old_name`'s
120:     docstring, rebinds ``old_name.__name__`` and returns the new
121:     function object.
122: 
123:     This function may also be used as a decorator.
124: 
125:     Parameters
126:     ----------
127:     func : function
128:         The function to be deprecated.
129:     old_name : str, optional
130:         The name of the function to be deprecated. Default is None, in
131:         which case the name of `func` is used.
132:     new_name : str, optional
133:         The new name for the function. Default is None, in which case the
134:         deprecation message is that `old_name` is deprecated. If given, the
135:         deprecation message is that `old_name` is deprecated and `new_name`
136:         should be used instead.
137:     message : str, optional
138:         Additional explanation of the deprecation.  Displayed in the
139:         docstring after the warning.
140: 
141:     Returns
142:     -------
143:     old_func : function
144:         The deprecated function.
145: 
146:     Examples
147:     --------
148:     Note that ``olduint`` returns a value after printing Deprecation
149:     Warning:
150: 
151:     >>> olduint = np.deprecate(np.uint)
152:     >>> olduint(6)
153:     /usr/lib/python2.5/site-packages/numpy/lib/utils.py:114:
154:     DeprecationWarning: uint32 is deprecated
155:       warnings.warn(str1, DeprecationWarning)
156:     6
157: 
158:     '''
159:     # Deprecate may be run as a function or as a decorator
160:     # If run as a function, we initialise the decorator class
161:     # and execute its __call__ method.
162: 
163:     if args:
164:         fn = args[0]
165:         args = args[1:]
166: 
167:         # backward compatibility -- can be removed
168:         # after next release
169:         if 'newname' in kwargs:
170:             kwargs['new_name'] = kwargs.pop('newname')
171:         if 'oldname' in kwargs:
172:             kwargs['old_name'] = kwargs.pop('oldname')
173: 
174:         return _Deprecate(*args, **kwargs)(fn)
175:     else:
176:         return _Deprecate(*args, **kwargs)
177: 
178: deprecate_with_doc = lambda msg: _Deprecate(message=msg)
179: 
180: 
181: #--------------------------------------------
182: # Determine if two arrays can share memory
183: #--------------------------------------------
184: 
185: def byte_bounds(a):
186:     '''
187:     Returns pointers to the end-points of an array.
188: 
189:     Parameters
190:     ----------
191:     a : ndarray
192:         Input array. It must conform to the Python-side of the array
193:         interface.
194: 
195:     Returns
196:     -------
197:     (low, high) : tuple of 2 integers
198:         The first integer is the first byte of the array, the second
199:         integer is just past the last byte of the array.  If `a` is not
200:         contiguous it will not use every byte between the (`low`, `high`)
201:         values.
202: 
203:     Examples
204:     --------
205:     >>> I = np.eye(2, dtype='f'); I.dtype
206:     dtype('float32')
207:     >>> low, high = np.byte_bounds(I)
208:     >>> high - low == I.size*I.itemsize
209:     True
210:     >>> I = np.eye(2, dtype='G'); I.dtype
211:     dtype('complex192')
212:     >>> low, high = np.byte_bounds(I)
213:     >>> high - low == I.size*I.itemsize
214:     True
215: 
216:     '''
217:     ai = a.__array_interface__
218:     a_data = ai['data'][0]
219:     astrides = ai['strides']
220:     ashape = ai['shape']
221:     bytes_a = asarray(a).dtype.itemsize
222: 
223:     a_low = a_high = a_data
224:     if astrides is None:
225:         # contiguous case
226:         a_high += a.size * bytes_a
227:     else:
228:         for shape, stride in zip(ashape, astrides):
229:             if stride < 0:
230:                 a_low += (shape-1)*stride
231:             else:
232:                 a_high += (shape-1)*stride
233:         a_high += bytes_a
234:     return a_low, a_high
235: 
236: 
237: #-----------------------------------------------------------------------------
238: # Function for output and information on the variables used.
239: #-----------------------------------------------------------------------------
240: 
241: 
242: def who(vardict=None):
243:     '''
244:     Print the Numpy arrays in the given dictionary.
245: 
246:     If there is no dictionary passed in or `vardict` is None then returns
247:     Numpy arrays in the globals() dictionary (all Numpy arrays in the
248:     namespace).
249: 
250:     Parameters
251:     ----------
252:     vardict : dict, optional
253:         A dictionary possibly containing ndarrays.  Default is globals().
254: 
255:     Returns
256:     -------
257:     out : None
258:         Returns 'None'.
259: 
260:     Notes
261:     -----
262:     Prints out the name, shape, bytes and type of all of the ndarrays
263:     present in `vardict`.
264: 
265:     Examples
266:     --------
267:     >>> a = np.arange(10)
268:     >>> b = np.ones(20)
269:     >>> np.who()
270:     Name            Shape            Bytes            Type
271:     ===========================================================
272:     a               10               40               int32
273:     b               20               160              float64
274:     Upper bound on total bytes  =       200
275: 
276:     >>> d = {'x': np.arange(2.0), 'y': np.arange(3.0), 'txt': 'Some str',
277:     ... 'idx':5}
278:     >>> np.who(d)
279:     Name            Shape            Bytes            Type
280:     ===========================================================
281:     y               3                24               float64
282:     x               2                16               float64
283:     Upper bound on total bytes  =       40
284: 
285:     '''
286:     if vardict is None:
287:         frame = sys._getframe().f_back
288:         vardict = frame.f_globals
289:     sta = []
290:     cache = {}
291:     for name in vardict.keys():
292:         if isinstance(vardict[name], ndarray):
293:             var = vardict[name]
294:             idv = id(var)
295:             if idv in cache.keys():
296:                 namestr = name + " (%s)" % cache[idv]
297:                 original = 0
298:             else:
299:                 cache[idv] = name
300:                 namestr = name
301:                 original = 1
302:             shapestr = " x ".join(map(str, var.shape))
303:             bytestr = str(var.nbytes)
304:             sta.append([namestr, shapestr, bytestr, var.dtype.name,
305:                         original])
306: 
307:     maxname = 0
308:     maxshape = 0
309:     maxbyte = 0
310:     totalbytes = 0
311:     for k in range(len(sta)):
312:         val = sta[k]
313:         if maxname < len(val[0]):
314:             maxname = len(val[0])
315:         if maxshape < len(val[1]):
316:             maxshape = len(val[1])
317:         if maxbyte < len(val[2]):
318:             maxbyte = len(val[2])
319:         if val[4]:
320:             totalbytes += int(val[2])
321: 
322:     if len(sta) > 0:
323:         sp1 = max(10, maxname)
324:         sp2 = max(10, maxshape)
325:         sp3 = max(10, maxbyte)
326:         prval = "Name %s Shape %s Bytes %s Type" % (sp1*' ', sp2*' ', sp3*' ')
327:         print(prval + "\n" + "="*(len(prval)+5) + "\n")
328: 
329:     for k in range(len(sta)):
330:         val = sta[k]
331:         print("%s %s %s %s %s %s %s" % (val[0], ' '*(sp1-len(val[0])+4),
332:                                         val[1], ' '*(sp2-len(val[1])+5),
333:                                         val[2], ' '*(sp3-len(val[2])+5),
334:                                         val[3]))
335:     print("\nUpper bound on total bytes  =       %d" % totalbytes)
336:     return
337: 
338: #-----------------------------------------------------------------------------
339: 
340: 
341: # NOTE:  pydoc defines a help function which works simliarly to this
342: #  except it uses a pager to take over the screen.
343: 
344: # combine name and arguments and split to multiple lines of width
345: # characters.  End lines on a comma and begin argument list indented with
346: # the rest of the arguments.
347: def _split_line(name, arguments, width):
348:     firstwidth = len(name)
349:     k = firstwidth
350:     newstr = name
351:     sepstr = ", "
352:     arglist = arguments.split(sepstr)
353:     for argument in arglist:
354:         if k == firstwidth:
355:             addstr = ""
356:         else:
357:             addstr = sepstr
358:         k = k + len(argument) + len(addstr)
359:         if k > width:
360:             k = firstwidth + 1 + len(argument)
361:             newstr = newstr + ",\n" + " "*(firstwidth+2) + argument
362:         else:
363:             newstr = newstr + addstr + argument
364:     return newstr
365: 
366: _namedict = None
367: _dictlist = None
368: 
369: # Traverse all module directories underneath globals
370: # to see if something is defined
371: def _makenamedict(module='numpy'):
372:     module = __import__(module, globals(), locals(), [])
373:     thedict = {module.__name__:module.__dict__}
374:     dictlist = [module.__name__]
375:     totraverse = [module.__dict__]
376:     while True:
377:         if len(totraverse) == 0:
378:             break
379:         thisdict = totraverse.pop(0)
380:         for x in thisdict.keys():
381:             if isinstance(thisdict[x], types.ModuleType):
382:                 modname = thisdict[x].__name__
383:                 if modname not in dictlist:
384:                     moddict = thisdict[x].__dict__
385:                     dictlist.append(modname)
386:                     totraverse.append(moddict)
387:                     thedict[modname] = moddict
388:     return thedict, dictlist
389: 
390: 
391: def _info(obj, output=sys.stdout):
392:     '''Provide information about ndarray obj.
393: 
394:     Parameters
395:     ----------
396:     obj: ndarray
397:         Must be ndarray, not checked.
398:     output:
399:         Where printed output goes.
400: 
401:     Notes
402:     -----
403:     Copied over from the numarray module prior to its removal.
404:     Adapted somewhat as only numpy is an option now.
405: 
406:     Called by info.
407: 
408:     '''
409:     extra = ""
410:     tic = ""
411:     bp = lambda x: x
412:     cls = getattr(obj, '__class__', type(obj))
413:     nm = getattr(cls, '__name__', cls)
414:     strides = obj.strides
415:     endian = obj.dtype.byteorder
416: 
417:     print("class: ", nm, file=output)
418:     print("shape: ", obj.shape, file=output)
419:     print("strides: ", strides, file=output)
420:     print("itemsize: ", obj.itemsize, file=output)
421:     print("aligned: ", bp(obj.flags.aligned), file=output)
422:     print("contiguous: ", bp(obj.flags.contiguous), file=output)
423:     print("fortran: ", obj.flags.fortran, file=output)
424:     print(
425:         "data pointer: %s%s" % (hex(obj.ctypes._as_parameter_.value), extra),
426:         file=output
427:         )
428:     print("byteorder: ", end=' ', file=output)
429:     if endian in ['|', '=']:
430:         print("%s%s%s" % (tic, sys.byteorder, tic), file=output)
431:         byteswap = False
432:     elif endian == '>':
433:         print("%sbig%s" % (tic, tic), file=output)
434:         byteswap = sys.byteorder != "big"
435:     else:
436:         print("%slittle%s" % (tic, tic), file=output)
437:         byteswap = sys.byteorder != "little"
438:     print("byteswap: ", bp(byteswap), file=output)
439:     print("type: %s" % obj.dtype, file=output)
440: 
441: 
442: def info(object=None, maxwidth=76, output=sys.stdout, toplevel='numpy'):
443:     '''
444:     Get help information for a function, class, or module.
445: 
446:     Parameters
447:     ----------
448:     object : object or str, optional
449:         Input object or name to get information about. If `object` is a
450:         numpy object, its docstring is given. If it is a string, available
451:         modules are searched for matching objects.  If None, information
452:         about `info` itself is returned.
453:     maxwidth : int, optional
454:         Printing width.
455:     output : file like object, optional
456:         File like object that the output is written to, default is
457:         ``stdout``.  The object has to be opened in 'w' or 'a' mode.
458:     toplevel : str, optional
459:         Start search at this level.
460: 
461:     See Also
462:     --------
463:     source, lookfor
464: 
465:     Notes
466:     -----
467:     When used interactively with an object, ``np.info(obj)`` is equivalent
468:     to ``help(obj)`` on the Python prompt or ``obj?`` on the IPython
469:     prompt.
470: 
471:     Examples
472:     --------
473:     >>> np.info(np.polyval) # doctest: +SKIP
474:        polyval(p, x)
475:          Evaluate the polynomial p at x.
476:          ...
477: 
478:     When using a string for `object` it is possible to get multiple results.
479: 
480:     >>> np.info('fft') # doctest: +SKIP
481:          *** Found in numpy ***
482:     Core FFT routines
483:     ...
484:          *** Found in numpy.fft ***
485:      fft(a, n=None, axis=-1)
486:     ...
487:          *** Repeat reference found in numpy.fft.fftpack ***
488:          *** Total of 3 references found. ***
489: 
490:     '''
491:     global _namedict, _dictlist
492:     # Local import to speed up numpy's import time.
493:     import pydoc
494:     import inspect
495: 
496:     if (hasattr(object, '_ppimport_importer') or
497:            hasattr(object, '_ppimport_module')):
498:         object = object._ppimport_module
499:     elif hasattr(object, '_ppimport_attr'):
500:         object = object._ppimport_attr
501: 
502:     if object is None:
503:         info(info)
504:     elif isinstance(object, ndarray):
505:         _info(object, output=output)
506:     elif isinstance(object, str):
507:         if _namedict is None:
508:             _namedict, _dictlist = _makenamedict(toplevel)
509:         numfound = 0
510:         objlist = []
511:         for namestr in _dictlist:
512:             try:
513:                 obj = _namedict[namestr][object]
514:                 if id(obj) in objlist:
515:                     print("\n     "
516:                           "*** Repeat reference found in %s *** " % namestr,
517:                           file=output
518:                           )
519:                 else:
520:                     objlist.append(id(obj))
521:                     print("     *** Found in %s ***" % namestr, file=output)
522:                     info(obj)
523:                     print("-"*maxwidth, file=output)
524:                 numfound += 1
525:             except KeyError:
526:                 pass
527:         if numfound == 0:
528:             print("Help for %s not found." % object, file=output)
529:         else:
530:             print("\n     "
531:                   "*** Total of %d references found. ***" % numfound,
532:                   file=output
533:                   )
534: 
535:     elif inspect.isfunction(object):
536:         name = object.__name__
537:         arguments = formatargspec(*getargspec(object))
538: 
539:         if len(name+arguments) > maxwidth:
540:             argstr = _split_line(name, arguments, maxwidth)
541:         else:
542:             argstr = name + arguments
543: 
544:         print(" " + argstr + "\n", file=output)
545:         print(inspect.getdoc(object), file=output)
546: 
547:     elif inspect.isclass(object):
548:         name = object.__name__
549:         arguments = "()"
550:         try:
551:             if hasattr(object, '__init__'):
552:                 arguments = formatargspec(
553:                         *getargspec(object.__init__.__func__)
554:                         )
555:                 arglist = arguments.split(', ')
556:                 if len(arglist) > 1:
557:                     arglist[1] = "("+arglist[1]
558:                     arguments = ", ".join(arglist[1:])
559:         except:
560:             pass
561: 
562:         if len(name+arguments) > maxwidth:
563:             argstr = _split_line(name, arguments, maxwidth)
564:         else:
565:             argstr = name + arguments
566: 
567:         print(" " + argstr + "\n", file=output)
568:         doc1 = inspect.getdoc(object)
569:         if doc1 is None:
570:             if hasattr(object, '__init__'):
571:                 print(inspect.getdoc(object.__init__), file=output)
572:         else:
573:             print(inspect.getdoc(object), file=output)
574: 
575:         methods = pydoc.allmethods(object)
576:         if methods != []:
577:             print("\n\nMethods:\n", file=output)
578:             for meth in methods:
579:                 if meth[0] == '_':
580:                     continue
581:                 thisobj = getattr(object, meth, None)
582:                 if thisobj is not None:
583:                     methstr, other = pydoc.splitdoc(
584:                             inspect.getdoc(thisobj) or "None"
585:                             )
586:                 print("  %s  --  %s" % (meth, methstr), file=output)
587: 
588:     elif (sys.version_info[0] < 3
589:             and isinstance(object, types.InstanceType)):
590:         # check for __call__ method
591:         # types.InstanceType is the type of the instances of oldstyle classes
592:         print("Instance of class: ", object.__class__.__name__, file=output)
593:         print(file=output)
594:         if hasattr(object, '__call__'):
595:             arguments = formatargspec(
596:                     *getargspec(object.__call__.__func__)
597:                     )
598:             arglist = arguments.split(', ')
599:             if len(arglist) > 1:
600:                 arglist[1] = "("+arglist[1]
601:                 arguments = ", ".join(arglist[1:])
602:             else:
603:                 arguments = "()"
604: 
605:             if hasattr(object, 'name'):
606:                 name = "%s" % object.name
607:             else:
608:                 name = "<name>"
609:             if len(name+arguments) > maxwidth:
610:                 argstr = _split_line(name, arguments, maxwidth)
611:             else:
612:                 argstr = name + arguments
613: 
614:             print(" " + argstr + "\n", file=output)
615:             doc = inspect.getdoc(object.__call__)
616:             if doc is not None:
617:                 print(inspect.getdoc(object.__call__), file=output)
618:             print(inspect.getdoc(object), file=output)
619: 
620:         else:
621:             print(inspect.getdoc(object), file=output)
622: 
623:     elif inspect.ismethod(object):
624:         name = object.__name__
625:         arguments = formatargspec(
626:                 *getargspec(object.__func__)
627:                 )
628:         arglist = arguments.split(', ')
629:         if len(arglist) > 1:
630:             arglist[1] = "("+arglist[1]
631:             arguments = ", ".join(arglist[1:])
632:         else:
633:             arguments = "()"
634: 
635:         if len(name+arguments) > maxwidth:
636:             argstr = _split_line(name, arguments, maxwidth)
637:         else:
638:             argstr = name + arguments
639: 
640:         print(" " + argstr + "\n", file=output)
641:         print(inspect.getdoc(object), file=output)
642: 
643:     elif hasattr(object, '__doc__'):
644:         print(inspect.getdoc(object), file=output)
645: 
646: 
647: def source(object, output=sys.stdout):
648:     '''
649:     Print or write to a file the source code for a Numpy object.
650: 
651:     The source code is only returned for objects written in Python. Many
652:     functions and classes are defined in C and will therefore not return
653:     useful information.
654: 
655:     Parameters
656:     ----------
657:     object : numpy object
658:         Input object. This can be any object (function, class, module,
659:         ...).
660:     output : file object, optional
661:         If `output` not supplied then source code is printed to screen
662:         (sys.stdout).  File object must be created with either write 'w' or
663:         append 'a' modes.
664: 
665:     See Also
666:     --------
667:     lookfor, info
668: 
669:     Examples
670:     --------
671:     >>> np.source(np.interp)                        #doctest: +SKIP
672:     In file: /usr/lib/python2.6/dist-packages/numpy/lib/function_base.py
673:     def interp(x, xp, fp, left=None, right=None):
674:         \"\"\".... (full docstring printed)\"\"\"
675:         if isinstance(x, (float, int, number)):
676:             return compiled_interp([x], xp, fp, left, right).item()
677:         else:
678:             return compiled_interp(x, xp, fp, left, right)
679: 
680:     The source code is only returned for objects written in Python.
681: 
682:     >>> np.source(np.array)                         #doctest: +SKIP
683:     Not available for this object.
684: 
685:     '''
686:     # Local import to speed up numpy's import time.
687:     import inspect
688:     try:
689:         print("In file: %s\n" % inspect.getsourcefile(object), file=output)
690:         print(inspect.getsource(object), file=output)
691:     except:
692:         print("Not available for this object.", file=output)
693: 
694: 
695: # Cache for lookfor: {id(module): {name: (docstring, kind, index), ...}...}
696: # where kind: "func", "class", "module", "object"
697: # and index: index in breadth-first namespace traversal
698: _lookfor_caches = {}
699: 
700: # regexp whose match indicates that the string may contain a function
701: # signature
702: _function_signature_re = re.compile(r"[a-z0-9_]+\(.*[,=].*\)", re.I)
703: 
704: def lookfor(what, module=None, import_modules=True, regenerate=False,
705:             output=None):
706:     '''
707:     Do a keyword search on docstrings.
708: 
709:     A list of of objects that matched the search is displayed,
710:     sorted by relevance. All given keywords need to be found in the
711:     docstring for it to be returned as a result, but the order does
712:     not matter.
713: 
714:     Parameters
715:     ----------
716:     what : str
717:         String containing words to look for.
718:     module : str or list, optional
719:         Name of module(s) whose docstrings to go through.
720:     import_modules : bool, optional
721:         Whether to import sub-modules in packages. Default is True.
722:     regenerate : bool, optional
723:         Whether to re-generate the docstring cache. Default is False.
724:     output : file-like, optional
725:         File-like object to write the output to. If omitted, use a pager.
726: 
727:     See Also
728:     --------
729:     source, info
730: 
731:     Notes
732:     -----
733:     Relevance is determined only roughly, by checking if the keywords occur
734:     in the function name, at the start of a docstring, etc.
735: 
736:     Examples
737:     --------
738:     >>> np.lookfor('binary representation')
739:     Search results for 'binary representation'
740:     ------------------------------------------
741:     numpy.binary_repr
742:         Return the binary representation of the input number as a string.
743:     numpy.core.setup_common.long_double_representation
744:         Given a binary dump as given by GNU od -b, look for long double
745:     numpy.base_repr
746:         Return a string representation of a number in the given base system.
747:     ...
748: 
749:     '''
750:     import pydoc
751: 
752:     # Cache
753:     cache = _lookfor_generate_cache(module, import_modules, regenerate)
754: 
755:     # Search
756:     # XXX: maybe using a real stemming search engine would be better?
757:     found = []
758:     whats = str(what).lower().split()
759:     if not whats:
760:         return
761: 
762:     for name, (docstring, kind, index) in cache.items():
763:         if kind in ('module', 'object'):
764:             # don't show modules or objects
765:             continue
766:         ok = True
767:         doc = docstring.lower()
768:         for w in whats:
769:             if w not in doc:
770:                 ok = False
771:                 break
772:         if ok:
773:             found.append(name)
774: 
775:     # Relevance sort
776:     # XXX: this is full Harrison-Stetson heuristics now,
777:     # XXX: it probably could be improved
778: 
779:     kind_relevance = {'func': 1000, 'class': 1000,
780:                       'module': -1000, 'object': -1000}
781: 
782:     def relevance(name, docstr, kind, index):
783:         r = 0
784:         # do the keywords occur within the start of the docstring?
785:         first_doc = "\n".join(docstr.lower().strip().split("\n")[:3])
786:         r += sum([200 for w in whats if w in first_doc])
787:         # do the keywords occur in the function name?
788:         r += sum([30 for w in whats if w in name])
789:         # is the full name long?
790:         r += -len(name) * 5
791:         # is the object of bad type?
792:         r += kind_relevance.get(kind, -1000)
793:         # is the object deep in namespace hierarchy?
794:         r += -name.count('.') * 10
795:         r += max(-index / 100, -100)
796:         return r
797: 
798:     def relevance_value(a):
799:         return relevance(a, *cache[a])
800:     found.sort(key=relevance_value)
801: 
802:     # Pretty-print
803:     s = "Search results for '%s'" % (' '.join(whats))
804:     help_text = [s, "-"*len(s)]
805:     for name in found[::-1]:
806:         doc, kind, ix = cache[name]
807: 
808:         doclines = [line.strip() for line in doc.strip().split("\n")
809:                     if line.strip()]
810: 
811:         # find a suitable short description
812:         try:
813:             first_doc = doclines[0].strip()
814:             if _function_signature_re.search(first_doc):
815:                 first_doc = doclines[1].strip()
816:         except IndexError:
817:             first_doc = ""
818:         help_text.append("%s\n    %s" % (name, first_doc))
819: 
820:     if not found:
821:         help_text.append("Nothing found.")
822: 
823:     # Output
824:     if output is not None:
825:         output.write("\n".join(help_text))
826:     elif len(help_text) > 10:
827:         pager = pydoc.getpager()
828:         pager("\n".join(help_text))
829:     else:
830:         print("\n".join(help_text))
831: 
832: def _lookfor_generate_cache(module, import_modules, regenerate):
833:     '''
834:     Generate docstring cache for given module.
835: 
836:     Parameters
837:     ----------
838:     module : str, None, module
839:         Module for which to generate docstring cache
840:     import_modules : bool
841:         Whether to import sub-modules in packages.
842:     regenerate : bool
843:         Re-generate the docstring cache
844: 
845:     Returns
846:     -------
847:     cache : dict {obj_full_name: (docstring, kind, index), ...}
848:         Docstring cache for the module, either cached one (regenerate=False)
849:         or newly generated.
850: 
851:     '''
852:     global _lookfor_caches
853:     # Local import to speed up numpy's import time.
854:     import inspect
855: 
856:     if sys.version_info[0] >= 3:
857:         # In Python3 stderr, stdout are text files.
858:         from io import StringIO
859:     else:
860:         from StringIO import StringIO
861: 
862:     if module is None:
863:         module = "numpy"
864: 
865:     if isinstance(module, str):
866:         try:
867:             __import__(module)
868:         except ImportError:
869:             return {}
870:         module = sys.modules[module]
871:     elif isinstance(module, list) or isinstance(module, tuple):
872:         cache = {}
873:         for mod in module:
874:             cache.update(_lookfor_generate_cache(mod, import_modules,
875:                                                  regenerate))
876:         return cache
877: 
878:     if id(module) in _lookfor_caches and not regenerate:
879:         return _lookfor_caches[id(module)]
880: 
881:     # walk items and collect docstrings
882:     cache = {}
883:     _lookfor_caches[id(module)] = cache
884:     seen = {}
885:     index = 0
886:     stack = [(module.__name__, module)]
887:     while stack:
888:         name, item = stack.pop(0)
889:         if id(item) in seen:
890:             continue
891:         seen[id(item)] = True
892: 
893:         index += 1
894:         kind = "object"
895: 
896:         if inspect.ismodule(item):
897:             kind = "module"
898:             try:
899:                 _all = item.__all__
900:             except AttributeError:
901:                 _all = None
902: 
903:             # import sub-packages
904:             if import_modules and hasattr(item, '__path__'):
905:                 for pth in item.__path__:
906:                     for mod_path in os.listdir(pth):
907:                         this_py = os.path.join(pth, mod_path)
908:                         init_py = os.path.join(pth, mod_path, '__init__.py')
909:                         if (os.path.isfile(this_py) and
910:                                 mod_path.endswith('.py')):
911:                             to_import = mod_path[:-3]
912:                         elif os.path.isfile(init_py):
913:                             to_import = mod_path
914:                         else:
915:                             continue
916:                         if to_import == '__init__':
917:                             continue
918: 
919:                         try:
920:                             # Catch SystemExit, too
921:                             base_exc = BaseException
922:                         except NameError:
923:                             # Python 2.4 doesn't have BaseException
924:                             base_exc = Exception
925: 
926:                         try:
927:                             old_stdout = sys.stdout
928:                             old_stderr = sys.stderr
929:                             try:
930:                                 sys.stdout = StringIO()
931:                                 sys.stderr = StringIO()
932:                                 __import__("%s.%s" % (name, to_import))
933:                             finally:
934:                                 sys.stdout = old_stdout
935:                                 sys.stderr = old_stderr
936:                         except base_exc:
937:                             continue
938: 
939:             for n, v in _getmembers(item):
940:                 try:
941:                     item_name = getattr(v, '__name__', "%s.%s" % (name, n))
942:                     mod_name = getattr(v, '__module__', None)
943:                 except NameError:
944:                     # ref. SWIG's global cvars
945:                     #    NameError: Unknown C global variable
946:                     item_name = "%s.%s" % (name, n)
947:                     mod_name = None
948:                 if '.' not in item_name and mod_name:
949:                     item_name = "%s.%s" % (mod_name, item_name)
950: 
951:                 if not item_name.startswith(name + '.'):
952:                     # don't crawl "foreign" objects
953:                     if isinstance(v, ufunc):
954:                         # ... unless they are ufuncs
955:                         pass
956:                     else:
957:                         continue
958:                 elif not (inspect.ismodule(v) or _all is None or n in _all):
959:                     continue
960:                 stack.append(("%s.%s" % (name, n), v))
961:         elif inspect.isclass(item):
962:             kind = "class"
963:             for n, v in _getmembers(item):
964:                 stack.append(("%s.%s" % (name, n), v))
965:         elif hasattr(item, "__call__"):
966:             kind = "func"
967: 
968:         try:
969:             doc = inspect.getdoc(item)
970:         except NameError:
971:             # ref SWIG's NameError: Unknown C global variable
972:             doc = None
973:         if doc is not None:
974:             cache[name] = (doc, kind, index)
975: 
976:     return cache
977: 
978: def _getmembers(item):
979:     import inspect
980:     try:
981:         members = inspect.getmembers(item)
982:     except Exception:
983:         members = [(x, getattr(item, x)) for x in dir(item)
984:                    if hasattr(item, x)]
985:     return members
986: 
987: #-----------------------------------------------------------------------------
988: 
989: # The following SafeEval class and company are adapted from Michael Spencer's
990: # ASPN Python Cookbook recipe:
991: #   http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/364469
992: # Accordingly it is mostly Copyright 2006 by Michael Spencer.
993: # The recipe, like most of the other ASPN Python Cookbook recipes was made
994: # available under the Python license.
995: #   http://www.python.org/license
996: 
997: # It has been modified to:
998: #   * handle unary -/+
999: #   * support True/False/None
1000: #   * raise SyntaxError instead of a custom exception.
1001: 
1002: class SafeEval(object):
1003:     '''
1004:     Object to evaluate constant string expressions.
1005: 
1006:     This includes strings with lists, dicts and tuples using the abstract
1007:     syntax tree created by ``compiler.parse``.
1008: 
1009:     .. deprecated:: 1.10.0
1010: 
1011:     See Also
1012:     --------
1013:     safe_eval
1014: 
1015:     '''
1016:     def __init__(self):
1017:         # 2014-10-15, 1.10
1018:         warnings.warn("SafeEval is deprecated in 1.10 and will be removed.",
1019:                       DeprecationWarning)
1020: 
1021:     def visit(self, node):
1022:         cls = node.__class__
1023:         meth = getattr(self, 'visit' + cls.__name__, self.default)
1024:         return meth(node)
1025: 
1026:     def default(self, node):
1027:         raise SyntaxError("Unsupported source construct: %s"
1028:                           % node.__class__)
1029: 
1030:     def visitExpression(self, node):
1031:         return self.visit(node.body)
1032: 
1033:     def visitNum(self, node):
1034:         return node.n
1035: 
1036:     def visitStr(self, node):
1037:         return node.s
1038: 
1039:     def visitBytes(self, node):
1040:         return node.s
1041: 
1042:     def visitDict(self, node,**kw):
1043:         return dict([(self.visit(k), self.visit(v))
1044:                      for k, v in zip(node.keys, node.values)])
1045: 
1046:     def visitTuple(self, node):
1047:         return tuple([self.visit(i) for i in node.elts])
1048: 
1049:     def visitList(self, node):
1050:         return [self.visit(i) for i in node.elts]
1051: 
1052:     def visitUnaryOp(self, node):
1053:         import ast
1054:         if isinstance(node.op, ast.UAdd):
1055:             return +self.visit(node.operand)
1056:         elif isinstance(node.op, ast.USub):
1057:             return -self.visit(node.operand)
1058:         else:
1059:             raise SyntaxError("Unknown unary op: %r" % node.op)
1060: 
1061:     def visitName(self, node):
1062:         if node.id == 'False':
1063:             return False
1064:         elif node.id == 'True':
1065:             return True
1066:         elif node.id == 'None':
1067:             return None
1068:         else:
1069:             raise SyntaxError("Unknown name: %s" % node.id)
1070: 
1071:     def visitNameConstant(self, node):
1072:         return node.value
1073: 
1074: 
1075: def safe_eval(source):
1076:     '''
1077:     Protected string evaluation.
1078: 
1079:     Evaluate a string containing a Python literal expression without
1080:     allowing the execution of arbitrary non-literal code.
1081: 
1082:     Parameters
1083:     ----------
1084:     source : str
1085:         The string to evaluate.
1086: 
1087:     Returns
1088:     -------
1089:     obj : object
1090:        The result of evaluating `source`.
1091: 
1092:     Raises
1093:     ------
1094:     SyntaxError
1095:         If the code has invalid Python syntax, or if it contains
1096:         non-literal code.
1097: 
1098:     Examples
1099:     --------
1100:     >>> np.safe_eval('1')
1101:     1
1102:     >>> np.safe_eval('[1, 2, 3]')
1103:     [1, 2, 3]
1104:     >>> np.safe_eval('{"foo": ("bar", 10.0)}')
1105:     {'foo': ('bar', 10.0)}
1106: 
1107:     >>> np.safe_eval('import os')
1108:     Traceback (most recent call last):
1109:       ...
1110:     SyntaxError: invalid syntax
1111: 
1112:     >>> np.safe_eval('open("/home/user/.ssh/id_dsa").read()')
1113:     Traceback (most recent call last):
1114:       ...
1115:     SyntaxError: Unsupported source construct: compiler.ast.CallFunc
1116: 
1117:     '''
1118:     # Local import to speed up numpy's import time.
1119:     import ast
1120: 
1121:     return ast.literal_eval(source)
1122: #-----------------------------------------------------------------------------
1123: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import types' statement (line 5)
import types

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import re' statement (line 6)
import re

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import warnings' statement (line 7)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.core.numerictypes import issubclass_, issubsctype, issubdtype' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_128934 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core.numerictypes')

if (type(import_128934) is not StypyTypeError):

    if (import_128934 != 'pyd_module'):
        __import__(import_128934)
        sys_modules_128935 = sys.modules[import_128934]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core.numerictypes', sys_modules_128935.module_type_store, module_type_store, ['issubclass_', 'issubsctype', 'issubdtype'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_128935, sys_modules_128935.module_type_store, module_type_store)
    else:
        from numpy.core.numerictypes import issubclass_, issubsctype, issubdtype

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core.numerictypes', None, module_type_store, ['issubclass_', 'issubsctype', 'issubdtype'], [issubclass_, issubsctype, issubdtype])

else:
    # Assigning a type to the variable 'numpy.core.numerictypes' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core.numerictypes', import_128934)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.core import ndarray, ufunc, asarray' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_128936 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core')

if (type(import_128936) is not StypyTypeError):

    if (import_128936 != 'pyd_module'):
        __import__(import_128936)
        sys_modules_128937 = sys.modules[import_128936]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', sys_modules_128937.module_type_store, module_type_store, ['ndarray', 'ufunc', 'asarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_128937, sys_modules_128937.module_type_store, module_type_store)
    else:
        from numpy.core import ndarray, ufunc, asarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', None, module_type_store, ['ndarray', 'ufunc', 'asarray'], [ndarray, ufunc, asarray])

else:
    # Assigning a type to the variable 'numpy.core' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', import_128936)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from numpy.compat import getargspec, formatargspec' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_128938 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.compat')

if (type(import_128938) is not StypyTypeError):

    if (import_128938 != 'pyd_module'):
        __import__(import_128938)
        sys_modules_128939 = sys.modules[import_128938]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.compat', sys_modules_128939.module_type_store, module_type_store, ['getargspec', 'formatargspec'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_128939, sys_modules_128939.module_type_store, module_type_store)
    else:
        from numpy.compat import getargspec, formatargspec

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.compat', None, module_type_store, ['getargspec', 'formatargspec'], [getargspec, formatargspec])

else:
    # Assigning a type to the variable 'numpy.compat' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'numpy.compat', import_128938)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 15):

# Assigning a List to a Name (line 15):
__all__ = ['issubclass_', 'issubsctype', 'issubdtype', 'deprecate', 'deprecate_with_doc', 'get_include', 'info', 'source', 'who', 'lookfor', 'byte_bounds', 'safe_eval']
module_type_store.set_exportable_members(['issubclass_', 'issubsctype', 'issubdtype', 'deprecate', 'deprecate_with_doc', 'get_include', 'info', 'source', 'who', 'lookfor', 'byte_bounds', 'safe_eval'])

# Obtaining an instance of the builtin type 'list' (line 15)
list_128940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_128941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'str', 'issubclass_')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128940, str_128941)
# Adding element type (line 15)
str_128942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'str', 'issubsctype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128940, str_128942)
# Adding element type (line 15)
str_128943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 34), 'str', 'issubdtype')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128940, str_128943)
# Adding element type (line 15)
str_128944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 48), 'str', 'deprecate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128940, str_128944)
# Adding element type (line 15)
str_128945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', 'deprecate_with_doc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128940, str_128945)
# Adding element type (line 15)
str_128946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'str', 'get_include')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128940, str_128946)
# Adding element type (line 15)
str_128947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 41), 'str', 'info')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128940, str_128947)
# Adding element type (line 15)
str_128948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 49), 'str', 'source')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128940, str_128948)
# Adding element type (line 15)
str_128949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 59), 'str', 'who')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128940, str_128949)
# Adding element type (line 15)
str_128950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'str', 'lookfor')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128940, str_128950)
# Adding element type (line 15)
str_128951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 15), 'str', 'byte_bounds')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128940, str_128951)
# Adding element type (line 15)
str_128952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'str', 'safe_eval')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_128940, str_128952)

# Assigning a type to the variable '__all__' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '__all__', list_128940)

@norecursion
def get_include(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_include'
    module_type_store = module_type_store.open_function_context('get_include', 21, 0, False)
    
    # Passed parameters checking function
    get_include.stypy_localization = localization
    get_include.stypy_type_of_self = None
    get_include.stypy_type_store = module_type_store
    get_include.stypy_function_name = 'get_include'
    get_include.stypy_param_names_list = []
    get_include.stypy_varargs_param_name = None
    get_include.stypy_kwargs_param_name = None
    get_include.stypy_call_defaults = defaults
    get_include.stypy_call_varargs = varargs
    get_include.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_include', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_include', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_include(...)' code ##################

    str_128953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'str', "\n    Return the directory that contains the NumPy \\*.h header files.\n\n    Extension modules that need to compile against NumPy should use this\n    function to locate the appropriate include directory.\n\n    Notes\n    -----\n    When using ``distutils``, for example in ``setup.py``.\n    ::\n\n        import numpy as np\n        ...\n        Extension('extension_name', ...\n                include_dirs=[np.get_include()])\n        ...\n\n    ")
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 4))
    
    # 'import numpy' statement (line 40)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
    import_128954 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 40, 4), 'numpy')

    if (type(import_128954) is not StypyTypeError):

        if (import_128954 != 'pyd_module'):
            __import__(import_128954)
            sys_modules_128955 = sys.modules[import_128954]
            import_module(stypy.reporting.localization.Localization(__file__, 40, 4), 'numpy', sys_modules_128955.module_type_store, module_type_store)
        else:
            import numpy

            import_module(stypy.reporting.localization.Localization(__file__, 40, 4), 'numpy', numpy, module_type_store)

    else:
        # Assigning a type to the variable 'numpy' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'numpy', import_128954)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
    
    
    # Type idiom detected: calculating its left and rigth part (line 41)
    # Getting the type of 'numpy' (line 41)
    numpy_128956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 7), 'numpy')
    # Obtaining the member 'show_config' of a type (line 41)
    show_config_128957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 7), numpy_128956, 'show_config')
    # Getting the type of 'None' (line 41)
    None_128958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'None')
    
    (may_be_128959, more_types_in_union_128960) = may_be_none(show_config_128957, None_128958)

    if may_be_128959:

        if more_types_in_union_128960:
            # Runtime conditional SSA (line 41)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 43):
        
        # Assigning a Call to a Name (line 43):
        
        # Call to join(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Call to dirname(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'numpy' (line 43)
        numpy_128967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 41), 'numpy', False)
        # Obtaining the member '__file__' of a type (line 43)
        file___128968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 41), numpy_128967, '__file__')
        # Processing the call keyword arguments (line 43)
        kwargs_128969 = {}
        # Getting the type of 'os' (line 43)
        os_128964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 43)
        path_128965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 25), os_128964, 'path')
        # Obtaining the member 'dirname' of a type (line 43)
        dirname_128966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 25), path_128965, 'dirname')
        # Calling dirname(args, kwargs) (line 43)
        dirname_call_result_128970 = invoke(stypy.reporting.localization.Localization(__file__, 43, 25), dirname_128966, *[file___128968], **kwargs_128969)
        
        str_128971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 58), 'str', 'core')
        str_128972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 66), 'str', 'include')
        # Processing the call keyword arguments (line 43)
        kwargs_128973 = {}
        # Getting the type of 'os' (line 43)
        os_128961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'os', False)
        # Obtaining the member 'path' of a type (line 43)
        path_128962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), os_128961, 'path')
        # Obtaining the member 'join' of a type (line 43)
        join_128963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), path_128962, 'join')
        # Calling join(args, kwargs) (line 43)
        join_call_result_128974 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), join_128963, *[dirname_call_result_128970, str_128971, str_128972], **kwargs_128973)
        
        # Assigning a type to the variable 'd' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'd', join_call_result_128974)

        if more_types_in_union_128960:
            # Runtime conditional SSA for else branch (line 41)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_128959) or more_types_in_union_128960):
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 46, 8))
        
        # 'import numpy.core' statement (line 46)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
        import_128975 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 46, 8), 'numpy.core')

        if (type(import_128975) is not StypyTypeError):

            if (import_128975 != 'pyd_module'):
                __import__(import_128975)
                sys_modules_128976 = sys.modules[import_128975]
                import_module(stypy.reporting.localization.Localization(__file__, 46, 8), 'core', sys_modules_128976.module_type_store, module_type_store)
            else:
                import numpy.core as core

                import_module(stypy.reporting.localization.Localization(__file__, 46, 8), 'core', numpy.core, module_type_store)

        else:
            # Assigning a type to the variable 'numpy.core' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'numpy.core', import_128975)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')
        
        
        # Assigning a Call to a Name (line 47):
        
        # Assigning a Call to a Name (line 47):
        
        # Call to join(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to dirname(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'core' (line 47)
        core_128983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 41), 'core', False)
        # Obtaining the member '__file__' of a type (line 47)
        file___128984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 41), core_128983, '__file__')
        # Processing the call keyword arguments (line 47)
        kwargs_128985 = {}
        # Getting the type of 'os' (line 47)
        os_128980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 47)
        path_128981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), os_128980, 'path')
        # Obtaining the member 'dirname' of a type (line 47)
        dirname_128982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), path_128981, 'dirname')
        # Calling dirname(args, kwargs) (line 47)
        dirname_call_result_128986 = invoke(stypy.reporting.localization.Localization(__file__, 47, 25), dirname_128982, *[file___128984], **kwargs_128985)
        
        str_128987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 57), 'str', 'include')
        # Processing the call keyword arguments (line 47)
        kwargs_128988 = {}
        # Getting the type of 'os' (line 47)
        os_128977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'os', False)
        # Obtaining the member 'path' of a type (line 47)
        path_128978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), os_128977, 'path')
        # Obtaining the member 'join' of a type (line 47)
        join_128979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), path_128978, 'join')
        # Calling join(args, kwargs) (line 47)
        join_call_result_128989 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), join_128979, *[dirname_call_result_128986, str_128987], **kwargs_128988)
        
        # Assigning a type to the variable 'd' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'd', join_call_result_128989)

        if (may_be_128959 and more_types_in_union_128960):
            # SSA join for if statement (line 41)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'd' (line 48)
    d_128990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'd')
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type', d_128990)
    
    # ################# End of 'get_include(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_include' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_128991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128991)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_include'
    return stypy_return_type_128991

# Assigning a type to the variable 'get_include' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'get_include', get_include)

@norecursion
def _set_function_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_set_function_name'
    module_type_store = module_type_store.open_function_context('_set_function_name', 51, 0, False)
    
    # Passed parameters checking function
    _set_function_name.stypy_localization = localization
    _set_function_name.stypy_type_of_self = None
    _set_function_name.stypy_type_store = module_type_store
    _set_function_name.stypy_function_name = '_set_function_name'
    _set_function_name.stypy_param_names_list = ['func', 'name']
    _set_function_name.stypy_varargs_param_name = None
    _set_function_name.stypy_kwargs_param_name = None
    _set_function_name.stypy_call_defaults = defaults
    _set_function_name.stypy_call_varargs = varargs
    _set_function_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_set_function_name', ['func', 'name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_set_function_name', localization, ['func', 'name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_set_function_name(...)' code ##################

    
    # Assigning a Name to a Attribute (line 52):
    
    # Assigning a Name to a Attribute (line 52):
    # Getting the type of 'name' (line 52)
    name_128992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'name')
    # Getting the type of 'func' (line 52)
    func_128993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'func')
    # Setting the type of the member '__name__' of a type (line 52)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), func_128993, '__name__', name_128992)
    # Getting the type of 'func' (line 53)
    func_128994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'func')
    # Assigning a type to the variable 'stypy_return_type' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type', func_128994)
    
    # ################# End of '_set_function_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_set_function_name' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_128995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_128995)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_set_function_name'
    return stypy_return_type_128995

# Assigning a type to the variable '_set_function_name' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), '_set_function_name', _set_function_name)
# Declaration of the '_Deprecate' class

class _Deprecate(object, ):
    str_128996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, (-1)), 'str', '\n    Decorator class to deprecate old functions.\n\n    Refer to `deprecate` for details.\n\n    See Also\n    --------\n    deprecate\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 68)
        None_128997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 32), 'None')
        # Getting the type of 'None' (line 68)
        None_128998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 47), 'None')
        # Getting the type of 'None' (line 68)
        None_128999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 61), 'None')
        defaults = [None_128997, None_128998, None_128999]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Deprecate.__init__', ['old_name', 'new_name', 'message'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['old_name', 'new_name', 'message'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 69):
        
        # Assigning a Name to a Attribute (line 69):
        # Getting the type of 'old_name' (line 69)
        old_name_129000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'old_name')
        # Getting the type of 'self' (line 69)
        self_129001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'self')
        # Setting the type of the member 'old_name' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), self_129001, 'old_name', old_name_129000)
        
        # Assigning a Name to a Attribute (line 70):
        
        # Assigning a Name to a Attribute (line 70):
        # Getting the type of 'new_name' (line 70)
        new_name_129002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'new_name')
        # Getting the type of 'self' (line 70)
        self_129003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'self')
        # Setting the type of the member 'new_name' of a type (line 70)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 8), self_129003, 'new_name', new_name_129002)
        
        # Assigning a Name to a Attribute (line 71):
        
        # Assigning a Name to a Attribute (line 71):
        # Getting the type of 'message' (line 71)
        message_129004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'message')
        # Getting the type of 'self' (line 71)
        self_129005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'self')
        # Setting the type of the member 'message' of a type (line 71)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), self_129005, 'message', message_129004)
        
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
        module_type_store = module_type_store.open_function_context('__call__', 73, 4, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _Deprecate.__call__.__dict__.__setitem__('stypy_localization', localization)
        _Deprecate.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _Deprecate.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _Deprecate.__call__.__dict__.__setitem__('stypy_function_name', '_Deprecate.__call__')
        _Deprecate.__call__.__dict__.__setitem__('stypy_param_names_list', ['func'])
        _Deprecate.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        _Deprecate.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        _Deprecate.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _Deprecate.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _Deprecate.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _Deprecate.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_Deprecate.__call__', ['func'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_129006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, (-1)), 'str', '\n        Decorator call.  Refer to ``decorate``.\n\n        ')
        
        # Assigning a Attribute to a Name (line 78):
        
        # Assigning a Attribute to a Name (line 78):
        # Getting the type of 'self' (line 78)
        self_129007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'self')
        # Obtaining the member 'old_name' of a type (line 78)
        old_name_129008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 19), self_129007, 'old_name')
        # Assigning a type to the variable 'old_name' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'old_name', old_name_129008)
        
        # Assigning a Attribute to a Name (line 79):
        
        # Assigning a Attribute to a Name (line 79):
        # Getting the type of 'self' (line 79)
        self_129009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'self')
        # Obtaining the member 'new_name' of a type (line 79)
        new_name_129010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 19), self_129009, 'new_name')
        # Assigning a type to the variable 'new_name' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'new_name', new_name_129010)
        
        # Assigning a Attribute to a Name (line 80):
        
        # Assigning a Attribute to a Name (line 80):
        # Getting the type of 'self' (line 80)
        self_129011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'self')
        # Obtaining the member 'message' of a type (line 80)
        message_129012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 18), self_129011, 'message')
        # Assigning a type to the variable 'message' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'message', message_129012)
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 82, 8))
        
        # 'import warnings' statement (line 82)
        import warnings

        import_module(stypy.reporting.localization.Localization(__file__, 82, 8), 'warnings', warnings, module_type_store)
        
        
        # Type idiom detected: calculating its left and rigth part (line 83)
        # Getting the type of 'old_name' (line 83)
        old_name_129013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'old_name')
        # Getting the type of 'None' (line 83)
        None_129014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 23), 'None')
        
        (may_be_129015, more_types_in_union_129016) = may_be_none(old_name_129013, None_129014)

        if may_be_129015:

            if more_types_in_union_129016:
                # Runtime conditional SSA (line 83)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 84)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Attribute to a Name (line 85):
            
            # Assigning a Attribute to a Name (line 85):
            # Getting the type of 'func' (line 85)
            func_129017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'func')
            # Obtaining the member '__name__' of a type (line 85)
            name___129018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 27), func_129017, '__name__')
            # Assigning a type to the variable 'old_name' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'old_name', name___129018)
            # SSA branch for the except part of a try statement (line 84)
            # SSA branch for the except 'AttributeError' branch of a try statement (line 84)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Attribute to a Name (line 87):
            
            # Assigning a Attribute to a Name (line 87):
            # Getting the type of 'func' (line 87)
            func_129019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 27), 'func')
            # Obtaining the member '__name__' of a type (line 87)
            name___129020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 27), func_129019, '__name__')
            # Assigning a type to the variable 'old_name' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 16), 'old_name', name___129020)
            # SSA join for try-except statement (line 84)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_129016:
                # SSA join for if statement (line 83)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 88)
        # Getting the type of 'new_name' (line 88)
        new_name_129021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'new_name')
        # Getting the type of 'None' (line 88)
        None_129022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'None')
        
        (may_be_129023, more_types_in_union_129024) = may_be_none(new_name_129021, None_129022)

        if may_be_129023:

            if more_types_in_union_129024:
                # Runtime conditional SSA (line 88)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 89):
            
            # Assigning a BinOp to a Name (line 89):
            str_129025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'str', '`%s` is deprecated!')
            # Getting the type of 'old_name' (line 89)
            old_name_129026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 45), 'old_name')
            # Applying the binary operator '%' (line 89)
            result_mod_129027 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 21), '%', str_129025, old_name_129026)
            
            # Assigning a type to the variable 'depdoc' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'depdoc', result_mod_129027)

            if more_types_in_union_129024:
                # Runtime conditional SSA for else branch (line 88)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_129023) or more_types_in_union_129024):
            
            # Assigning a BinOp to a Name (line 91):
            
            # Assigning a BinOp to a Name (line 91):
            str_129028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 21), 'str', '`%s` is deprecated, use `%s` instead!')
            
            # Obtaining an instance of the builtin type 'tuple' (line 92)
            tuple_129029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 92)
            # Adding element type (line 92)
            # Getting the type of 'old_name' (line 92)
            old_name_129030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'old_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 22), tuple_129029, old_name_129030)
            # Adding element type (line 92)
            # Getting the type of 'new_name' (line 92)
            new_name_129031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'new_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 22), tuple_129029, new_name_129031)
            
            # Applying the binary operator '%' (line 91)
            result_mod_129032 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 21), '%', str_129028, tuple_129029)
            
            # Assigning a type to the variable 'depdoc' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'depdoc', result_mod_129032)

            if (may_be_129023 and more_types_in_union_129024):
                # SSA join for if statement (line 88)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 94)
        # Getting the type of 'message' (line 94)
        message_129033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'message')
        # Getting the type of 'None' (line 94)
        None_129034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 26), 'None')
        
        (may_be_129035, more_types_in_union_129036) = may_not_be_none(message_129033, None_129034)

        if may_be_129035:

            if more_types_in_union_129036:
                # Runtime conditional SSA (line 94)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'depdoc' (line 95)
            depdoc_129037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'depdoc')
            str_129038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 22), 'str', '\n')
            # Getting the type of 'message' (line 95)
            message_129039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 29), 'message')
            # Applying the binary operator '+' (line 95)
            result_add_129040 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 22), '+', str_129038, message_129039)
            
            # Applying the binary operator '+=' (line 95)
            result_iadd_129041 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 12), '+=', depdoc_129037, result_add_129040)
            # Assigning a type to the variable 'depdoc' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'depdoc', result_iadd_129041)
            

            if more_types_in_union_129036:
                # SSA join for if statement (line 94)
                module_type_store = module_type_store.join_ssa_context()


        

        @norecursion
        def newfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'newfunc'
            module_type_store = module_type_store.open_function_context('newfunc', 97, 8, False)
            
            # Passed parameters checking function
            newfunc.stypy_localization = localization
            newfunc.stypy_type_of_self = None
            newfunc.stypy_type_store = module_type_store
            newfunc.stypy_function_name = 'newfunc'
            newfunc.stypy_param_names_list = []
            newfunc.stypy_varargs_param_name = 'args'
            newfunc.stypy_kwargs_param_name = 'kwds'
            newfunc.stypy_call_defaults = defaults
            newfunc.stypy_call_varargs = varargs
            newfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'newfunc', [], 'args', 'kwds', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'newfunc', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'newfunc(...)' code ##################

            str_129042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 12), 'str', '`arrayrange` is deprecated, use `arange` instead!')
            
            # Call to warn(...): (line 99)
            # Processing the call arguments (line 99)
            # Getting the type of 'depdoc' (line 99)
            depdoc_129045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'depdoc', False)
            # Getting the type of 'DeprecationWarning' (line 99)
            DeprecationWarning_129046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'DeprecationWarning', False)
            # Processing the call keyword arguments (line 99)
            kwargs_129047 = {}
            # Getting the type of 'warnings' (line 99)
            warnings_129043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 99)
            warn_129044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), warnings_129043, 'warn')
            # Calling warn(args, kwargs) (line 99)
            warn_call_result_129048 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), warn_129044, *[depdoc_129045, DeprecationWarning_129046], **kwargs_129047)
            
            
            # Call to func(...): (line 100)
            # Getting the type of 'args' (line 100)
            args_129050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'args', False)
            # Processing the call keyword arguments (line 100)
            # Getting the type of 'kwds' (line 100)
            kwds_129051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 33), 'kwds', False)
            kwargs_129052 = {'kwds_129051': kwds_129051}
            # Getting the type of 'func' (line 100)
            func_129049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), 'func', False)
            # Calling func(args, kwargs) (line 100)
            func_call_result_129053 = invoke(stypy.reporting.localization.Localization(__file__, 100, 19), func_129049, *[args_129050], **kwargs_129052)
            
            # Assigning a type to the variable 'stypy_return_type' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'stypy_return_type', func_call_result_129053)
            
            # ################# End of 'newfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'newfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 97)
            stypy_return_type_129054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_129054)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'newfunc'
            return stypy_return_type_129054

        # Assigning a type to the variable 'newfunc' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'newfunc', newfunc)
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to _set_function_name(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'newfunc' (line 102)
        newfunc_129056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 37), 'newfunc', False)
        # Getting the type of 'old_name' (line 102)
        old_name_129057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 46), 'old_name', False)
        # Processing the call keyword arguments (line 102)
        kwargs_129058 = {}
        # Getting the type of '_set_function_name' (line 102)
        _set_function_name_129055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 18), '_set_function_name', False)
        # Calling _set_function_name(args, kwargs) (line 102)
        _set_function_name_call_result_129059 = invoke(stypy.reporting.localization.Localization(__file__, 102, 18), _set_function_name_129055, *[newfunc_129056, old_name_129057], **kwargs_129058)
        
        # Assigning a type to the variable 'newfunc' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'newfunc', _set_function_name_call_result_129059)
        
        # Assigning a Attribute to a Name (line 103):
        
        # Assigning a Attribute to a Name (line 103):
        # Getting the type of 'func' (line 103)
        func_129060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 14), 'func')
        # Obtaining the member '__doc__' of a type (line 103)
        doc___129061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 14), func_129060, '__doc__')
        # Assigning a type to the variable 'doc' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'doc', doc___129061)
        
        # Type idiom detected: calculating its left and rigth part (line 104)
        # Getting the type of 'doc' (line 104)
        doc_129062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'doc')
        # Getting the type of 'None' (line 104)
        None_129063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'None')
        
        (may_be_129064, more_types_in_union_129065) = may_be_none(doc_129062, None_129063)

        if may_be_129064:

            if more_types_in_union_129065:
                # Runtime conditional SSA (line 104)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 105):
            
            # Assigning a Name to a Name (line 105):
            # Getting the type of 'depdoc' (line 105)
            depdoc_129066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'depdoc')
            # Assigning a type to the variable 'doc' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'doc', depdoc_129066)

            if more_types_in_union_129065:
                # Runtime conditional SSA for else branch (line 104)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_129064) or more_types_in_union_129065):
            
            # Assigning a Call to a Name (line 107):
            
            # Assigning a Call to a Name (line 107):
            
            # Call to join(...): (line 107)
            # Processing the call arguments (line 107)
            
            # Obtaining an instance of the builtin type 'list' (line 107)
            list_129069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 30), 'list')
            # Adding type elements to the builtin type 'list' instance (line 107)
            # Adding element type (line 107)
            # Getting the type of 'depdoc' (line 107)
            depdoc_129070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 31), 'depdoc', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 30), list_129069, depdoc_129070)
            # Adding element type (line 107)
            # Getting the type of 'doc' (line 107)
            doc_129071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 39), 'doc', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 30), list_129069, doc_129071)
            
            # Processing the call keyword arguments (line 107)
            kwargs_129072 = {}
            str_129067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 18), 'str', '\n\n')
            # Obtaining the member 'join' of a type (line 107)
            join_129068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 18), str_129067, 'join')
            # Calling join(args, kwargs) (line 107)
            join_call_result_129073 = invoke(stypy.reporting.localization.Localization(__file__, 107, 18), join_129068, *[list_129069], **kwargs_129072)
            
            # Assigning a type to the variable 'doc' (line 107)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'doc', join_call_result_129073)

            if (may_be_129064 and more_types_in_union_129065):
                # SSA join for if statement (line 104)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 108):
        
        # Assigning a Name to a Attribute (line 108):
        # Getting the type of 'doc' (line 108)
        doc_129074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 26), 'doc')
        # Getting the type of 'newfunc' (line 108)
        newfunc_129075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'newfunc')
        # Setting the type of the member '__doc__' of a type (line 108)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), newfunc_129075, '__doc__', doc_129074)
        
        
        # SSA begins for try-except statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Attribute to a Name (line 110):
        
        # Assigning a Attribute to a Name (line 110):
        # Getting the type of 'func' (line 110)
        func_129076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'func')
        # Obtaining the member '__dict__' of a type (line 110)
        dict___129077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 16), func_129076, '__dict__')
        # Assigning a type to the variable 'd' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'd', dict___129077)
        # SSA branch for the except part of a try statement (line 109)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 109)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 109)
        module_type_store.open_ssa_branch('except else')
        
        # Call to update(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'd' (line 114)
        d_129081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 36), 'd', False)
        # Processing the call keyword arguments (line 114)
        kwargs_129082 = {}
        # Getting the type of 'newfunc' (line 114)
        newfunc_129078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'newfunc', False)
        # Obtaining the member '__dict__' of a type (line 114)
        dict___129079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), newfunc_129078, '__dict__')
        # Obtaining the member 'update' of a type (line 114)
        update_129080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), dict___129079, 'update')
        # Calling update(args, kwargs) (line 114)
        update_call_result_129083 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), update_129080, *[d_129081], **kwargs_129082)
        
        # SSA join for try-except statement (line 109)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'newfunc' (line 115)
        newfunc_129084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'newfunc')
        # Assigning a type to the variable 'stypy_return_type' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'stypy_return_type', newfunc_129084)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 73)
        stypy_return_type_129085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129085)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_129085


# Assigning a type to the variable '_Deprecate' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), '_Deprecate', _Deprecate)

@norecursion
def deprecate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'deprecate'
    module_type_store = module_type_store.open_function_context('deprecate', 117, 0, False)
    
    # Passed parameters checking function
    deprecate.stypy_localization = localization
    deprecate.stypy_type_of_self = None
    deprecate.stypy_type_store = module_type_store
    deprecate.stypy_function_name = 'deprecate'
    deprecate.stypy_param_names_list = []
    deprecate.stypy_varargs_param_name = 'args'
    deprecate.stypy_kwargs_param_name = 'kwargs'
    deprecate.stypy_call_defaults = defaults
    deprecate.stypy_call_varargs = varargs
    deprecate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'deprecate', [], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'deprecate', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'deprecate(...)' code ##################

    str_129086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, (-1)), 'str', "\n    Issues a DeprecationWarning, adds warning to `old_name`'s\n    docstring, rebinds ``old_name.__name__`` and returns the new\n    function object.\n\n    This function may also be used as a decorator.\n\n    Parameters\n    ----------\n    func : function\n        The function to be deprecated.\n    old_name : str, optional\n        The name of the function to be deprecated. Default is None, in\n        which case the name of `func` is used.\n    new_name : str, optional\n        The new name for the function. Default is None, in which case the\n        deprecation message is that `old_name` is deprecated. If given, the\n        deprecation message is that `old_name` is deprecated and `new_name`\n        should be used instead.\n    message : str, optional\n        Additional explanation of the deprecation.  Displayed in the\n        docstring after the warning.\n\n    Returns\n    -------\n    old_func : function\n        The deprecated function.\n\n    Examples\n    --------\n    Note that ``olduint`` returns a value after printing Deprecation\n    Warning:\n\n    >>> olduint = np.deprecate(np.uint)\n    >>> olduint(6)\n    /usr/lib/python2.5/site-packages/numpy/lib/utils.py:114:\n    DeprecationWarning: uint32 is deprecated\n      warnings.warn(str1, DeprecationWarning)\n    6\n\n    ")
    
    # Getting the type of 'args' (line 163)
    args_129087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 7), 'args')
    # Testing the type of an if condition (line 163)
    if_condition_129088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 4), args_129087)
    # Assigning a type to the variable 'if_condition_129088' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'if_condition_129088', if_condition_129088)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 164):
    
    # Assigning a Subscript to a Name (line 164):
    
    # Obtaining the type of the subscript
    int_129089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 18), 'int')
    # Getting the type of 'args' (line 164)
    args_129090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 13), 'args')
    # Obtaining the member '__getitem__' of a type (line 164)
    getitem___129091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 13), args_129090, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 164)
    subscript_call_result_129092 = invoke(stypy.reporting.localization.Localization(__file__, 164, 13), getitem___129091, int_129089)
    
    # Assigning a type to the variable 'fn' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'fn', subscript_call_result_129092)
    
    # Assigning a Subscript to a Name (line 165):
    
    # Assigning a Subscript to a Name (line 165):
    
    # Obtaining the type of the subscript
    int_129093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 20), 'int')
    slice_129094 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 165, 15), int_129093, None, None)
    # Getting the type of 'args' (line 165)
    args_129095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 15), 'args')
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___129096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 15), args_129095, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_129097 = invoke(stypy.reporting.localization.Localization(__file__, 165, 15), getitem___129096, slice_129094)
    
    # Assigning a type to the variable 'args' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'args', subscript_call_result_129097)
    
    
    str_129098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 11), 'str', 'newname')
    # Getting the type of 'kwargs' (line 169)
    kwargs_129099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'kwargs')
    # Applying the binary operator 'in' (line 169)
    result_contains_129100 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 11), 'in', str_129098, kwargs_129099)
    
    # Testing the type of an if condition (line 169)
    if_condition_129101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 8), result_contains_129100)
    # Assigning a type to the variable 'if_condition_129101' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'if_condition_129101', if_condition_129101)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 170):
    
    # Assigning a Call to a Subscript (line 170):
    
    # Call to pop(...): (line 170)
    # Processing the call arguments (line 170)
    str_129104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 44), 'str', 'newname')
    # Processing the call keyword arguments (line 170)
    kwargs_129105 = {}
    # Getting the type of 'kwargs' (line 170)
    kwargs_129102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 33), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 170)
    pop_129103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 33), kwargs_129102, 'pop')
    # Calling pop(args, kwargs) (line 170)
    pop_call_result_129106 = invoke(stypy.reporting.localization.Localization(__file__, 170, 33), pop_129103, *[str_129104], **kwargs_129105)
    
    # Getting the type of 'kwargs' (line 170)
    kwargs_129107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'kwargs')
    str_129108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 19), 'str', 'new_name')
    # Storing an element on a container (line 170)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 12), kwargs_129107, (str_129108, pop_call_result_129106))
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_129109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 11), 'str', 'oldname')
    # Getting the type of 'kwargs' (line 171)
    kwargs_129110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 24), 'kwargs')
    # Applying the binary operator 'in' (line 171)
    result_contains_129111 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 11), 'in', str_129109, kwargs_129110)
    
    # Testing the type of an if condition (line 171)
    if_condition_129112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 171, 8), result_contains_129111)
    # Assigning a type to the variable 'if_condition_129112' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'if_condition_129112', if_condition_129112)
    # SSA begins for if statement (line 171)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 172):
    
    # Assigning a Call to a Subscript (line 172):
    
    # Call to pop(...): (line 172)
    # Processing the call arguments (line 172)
    str_129115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 44), 'str', 'oldname')
    # Processing the call keyword arguments (line 172)
    kwargs_129116 = {}
    # Getting the type of 'kwargs' (line 172)
    kwargs_129113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 33), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 172)
    pop_129114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 33), kwargs_129113, 'pop')
    # Calling pop(args, kwargs) (line 172)
    pop_call_result_129117 = invoke(stypy.reporting.localization.Localization(__file__, 172, 33), pop_129114, *[str_129115], **kwargs_129116)
    
    # Getting the type of 'kwargs' (line 172)
    kwargs_129118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'kwargs')
    str_129119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 19), 'str', 'old_name')
    # Storing an element on a container (line 172)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 12), kwargs_129118, (str_129119, pop_call_result_129117))
    # SSA join for if statement (line 171)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to (...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'fn' (line 174)
    fn_129125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 43), 'fn', False)
    # Processing the call keyword arguments (line 174)
    kwargs_129126 = {}
    
    # Call to _Deprecate(...): (line 174)
    # Getting the type of 'args' (line 174)
    args_129121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 27), 'args', False)
    # Processing the call keyword arguments (line 174)
    # Getting the type of 'kwargs' (line 174)
    kwargs_129122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 35), 'kwargs', False)
    kwargs_129123 = {'kwargs_129122': kwargs_129122}
    # Getting the type of '_Deprecate' (line 174)
    _Deprecate_129120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), '_Deprecate', False)
    # Calling _Deprecate(args, kwargs) (line 174)
    _Deprecate_call_result_129124 = invoke(stypy.reporting.localization.Localization(__file__, 174, 15), _Deprecate_129120, *[args_129121], **kwargs_129123)
    
    # Calling (args, kwargs) (line 174)
    _call_result_129127 = invoke(stypy.reporting.localization.Localization(__file__, 174, 15), _Deprecate_call_result_129124, *[fn_129125], **kwargs_129126)
    
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'stypy_return_type', _call_result_129127)
    # SSA branch for the else part of an if statement (line 163)
    module_type_store.open_ssa_branch('else')
    
    # Call to _Deprecate(...): (line 176)
    # Getting the type of 'args' (line 176)
    args_129129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 27), 'args', False)
    # Processing the call keyword arguments (line 176)
    # Getting the type of 'kwargs' (line 176)
    kwargs_129130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 35), 'kwargs', False)
    kwargs_129131 = {'kwargs_129130': kwargs_129130}
    # Getting the type of '_Deprecate' (line 176)
    _Deprecate_129128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), '_Deprecate', False)
    # Calling _Deprecate(args, kwargs) (line 176)
    _Deprecate_call_result_129132 = invoke(stypy.reporting.localization.Localization(__file__, 176, 15), _Deprecate_129128, *[args_129129], **kwargs_129131)
    
    # Assigning a type to the variable 'stypy_return_type' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', _Deprecate_call_result_129132)
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'deprecate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'deprecate' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_129133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129133)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'deprecate'
    return stypy_return_type_129133

# Assigning a type to the variable 'deprecate' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'deprecate', deprecate)

# Assigning a Lambda to a Name (line 178):

# Assigning a Lambda to a Name (line 178):

@norecursion
def _stypy_temp_lambda_30(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_30'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_30', 178, 21, True)
    # Passed parameters checking function
    _stypy_temp_lambda_30.stypy_localization = localization
    _stypy_temp_lambda_30.stypy_type_of_self = None
    _stypy_temp_lambda_30.stypy_type_store = module_type_store
    _stypy_temp_lambda_30.stypy_function_name = '_stypy_temp_lambda_30'
    _stypy_temp_lambda_30.stypy_param_names_list = ['msg']
    _stypy_temp_lambda_30.stypy_varargs_param_name = None
    _stypy_temp_lambda_30.stypy_kwargs_param_name = None
    _stypy_temp_lambda_30.stypy_call_defaults = defaults
    _stypy_temp_lambda_30.stypy_call_varargs = varargs
    _stypy_temp_lambda_30.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_30', ['msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_30', ['msg'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    
    # Call to _Deprecate(...): (line 178)
    # Processing the call keyword arguments (line 178)
    # Getting the type of 'msg' (line 178)
    msg_129135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 52), 'msg', False)
    keyword_129136 = msg_129135
    kwargs_129137 = {'message': keyword_129136}
    # Getting the type of '_Deprecate' (line 178)
    _Deprecate_129134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 33), '_Deprecate', False)
    # Calling _Deprecate(args, kwargs) (line 178)
    _Deprecate_call_result_129138 = invoke(stypy.reporting.localization.Localization(__file__, 178, 33), _Deprecate_129134, *[], **kwargs_129137)
    
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'stypy_return_type', _Deprecate_call_result_129138)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_30' in the type store
    # Getting the type of 'stypy_return_type' (line 178)
    stypy_return_type_129139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129139)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_30'
    return stypy_return_type_129139

# Assigning a type to the variable '_stypy_temp_lambda_30' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), '_stypy_temp_lambda_30', _stypy_temp_lambda_30)
# Getting the type of '_stypy_temp_lambda_30' (line 178)
_stypy_temp_lambda_30_129140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), '_stypy_temp_lambda_30')
# Assigning a type to the variable 'deprecate_with_doc' (line 178)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 0), 'deprecate_with_doc', _stypy_temp_lambda_30_129140)

@norecursion
def byte_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'byte_bounds'
    module_type_store = module_type_store.open_function_context('byte_bounds', 185, 0, False)
    
    # Passed parameters checking function
    byte_bounds.stypy_localization = localization
    byte_bounds.stypy_type_of_self = None
    byte_bounds.stypy_type_store = module_type_store
    byte_bounds.stypy_function_name = 'byte_bounds'
    byte_bounds.stypy_param_names_list = ['a']
    byte_bounds.stypy_varargs_param_name = None
    byte_bounds.stypy_kwargs_param_name = None
    byte_bounds.stypy_call_defaults = defaults
    byte_bounds.stypy_call_varargs = varargs
    byte_bounds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'byte_bounds', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'byte_bounds', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'byte_bounds(...)' code ##################

    str_129141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, (-1)), 'str', "\n    Returns pointers to the end-points of an array.\n\n    Parameters\n    ----------\n    a : ndarray\n        Input array. It must conform to the Python-side of the array\n        interface.\n\n    Returns\n    -------\n    (low, high) : tuple of 2 integers\n        The first integer is the first byte of the array, the second\n        integer is just past the last byte of the array.  If `a` is not\n        contiguous it will not use every byte between the (`low`, `high`)\n        values.\n\n    Examples\n    --------\n    >>> I = np.eye(2, dtype='f'); I.dtype\n    dtype('float32')\n    >>> low, high = np.byte_bounds(I)\n    >>> high - low == I.size*I.itemsize\n    True\n    >>> I = np.eye(2, dtype='G'); I.dtype\n    dtype('complex192')\n    >>> low, high = np.byte_bounds(I)\n    >>> high - low == I.size*I.itemsize\n    True\n\n    ")
    
    # Assigning a Attribute to a Name (line 217):
    
    # Assigning a Attribute to a Name (line 217):
    # Getting the type of 'a' (line 217)
    a_129142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 9), 'a')
    # Obtaining the member '__array_interface__' of a type (line 217)
    array_interface___129143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 9), a_129142, '__array_interface__')
    # Assigning a type to the variable 'ai' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'ai', array_interface___129143)
    
    # Assigning a Subscript to a Name (line 218):
    
    # Assigning a Subscript to a Name (line 218):
    
    # Obtaining the type of the subscript
    int_129144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 24), 'int')
    
    # Obtaining the type of the subscript
    str_129145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 16), 'str', 'data')
    # Getting the type of 'ai' (line 218)
    ai_129146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 13), 'ai')
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___129147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 13), ai_129146, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_129148 = invoke(stypy.reporting.localization.Localization(__file__, 218, 13), getitem___129147, str_129145)
    
    # Obtaining the member '__getitem__' of a type (line 218)
    getitem___129149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 13), subscript_call_result_129148, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 218)
    subscript_call_result_129150 = invoke(stypy.reporting.localization.Localization(__file__, 218, 13), getitem___129149, int_129144)
    
    # Assigning a type to the variable 'a_data' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'a_data', subscript_call_result_129150)
    
    # Assigning a Subscript to a Name (line 219):
    
    # Assigning a Subscript to a Name (line 219):
    
    # Obtaining the type of the subscript
    str_129151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 18), 'str', 'strides')
    # Getting the type of 'ai' (line 219)
    ai_129152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 15), 'ai')
    # Obtaining the member '__getitem__' of a type (line 219)
    getitem___129153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 15), ai_129152, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 219)
    subscript_call_result_129154 = invoke(stypy.reporting.localization.Localization(__file__, 219, 15), getitem___129153, str_129151)
    
    # Assigning a type to the variable 'astrides' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'astrides', subscript_call_result_129154)
    
    # Assigning a Subscript to a Name (line 220):
    
    # Assigning a Subscript to a Name (line 220):
    
    # Obtaining the type of the subscript
    str_129155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 16), 'str', 'shape')
    # Getting the type of 'ai' (line 220)
    ai_129156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 13), 'ai')
    # Obtaining the member '__getitem__' of a type (line 220)
    getitem___129157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 13), ai_129156, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 220)
    subscript_call_result_129158 = invoke(stypy.reporting.localization.Localization(__file__, 220, 13), getitem___129157, str_129155)
    
    # Assigning a type to the variable 'ashape' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'ashape', subscript_call_result_129158)
    
    # Assigning a Attribute to a Name (line 221):
    
    # Assigning a Attribute to a Name (line 221):
    
    # Call to asarray(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'a' (line 221)
    a_129160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'a', False)
    # Processing the call keyword arguments (line 221)
    kwargs_129161 = {}
    # Getting the type of 'asarray' (line 221)
    asarray_129159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 14), 'asarray', False)
    # Calling asarray(args, kwargs) (line 221)
    asarray_call_result_129162 = invoke(stypy.reporting.localization.Localization(__file__, 221, 14), asarray_129159, *[a_129160], **kwargs_129161)
    
    # Obtaining the member 'dtype' of a type (line 221)
    dtype_129163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 14), asarray_call_result_129162, 'dtype')
    # Obtaining the member 'itemsize' of a type (line 221)
    itemsize_129164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 14), dtype_129163, 'itemsize')
    # Assigning a type to the variable 'bytes_a' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'bytes_a', itemsize_129164)
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Name to a Name (line 223):
    # Getting the type of 'a_data' (line 223)
    a_data_129165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'a_data')
    # Assigning a type to the variable 'a_high' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'a_high', a_data_129165)
    
    # Assigning a Name to a Name (line 223):
    # Getting the type of 'a_high' (line 223)
    a_high_129166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'a_high')
    # Assigning a type to the variable 'a_low' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'a_low', a_high_129166)
    
    # Type idiom detected: calculating its left and rigth part (line 224)
    # Getting the type of 'astrides' (line 224)
    astrides_129167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 7), 'astrides')
    # Getting the type of 'None' (line 224)
    None_129168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'None')
    
    (may_be_129169, more_types_in_union_129170) = may_be_none(astrides_129167, None_129168)

    if may_be_129169:

        if more_types_in_union_129170:
            # Runtime conditional SSA (line 224)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'a_high' (line 226)
        a_high_129171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'a_high')
        # Getting the type of 'a' (line 226)
        a_129172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 18), 'a')
        # Obtaining the member 'size' of a type (line 226)
        size_129173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 18), a_129172, 'size')
        # Getting the type of 'bytes_a' (line 226)
        bytes_a_129174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 27), 'bytes_a')
        # Applying the binary operator '*' (line 226)
        result_mul_129175 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 18), '*', size_129173, bytes_a_129174)
        
        # Applying the binary operator '+=' (line 226)
        result_iadd_129176 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 8), '+=', a_high_129171, result_mul_129175)
        # Assigning a type to the variable 'a_high' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'a_high', result_iadd_129176)
        

        if more_types_in_union_129170:
            # Runtime conditional SSA for else branch (line 224)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_129169) or more_types_in_union_129170):
        
        
        # Call to zip(...): (line 228)
        # Processing the call arguments (line 228)
        # Getting the type of 'ashape' (line 228)
        ashape_129178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 33), 'ashape', False)
        # Getting the type of 'astrides' (line 228)
        astrides_129179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 41), 'astrides', False)
        # Processing the call keyword arguments (line 228)
        kwargs_129180 = {}
        # Getting the type of 'zip' (line 228)
        zip_129177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 29), 'zip', False)
        # Calling zip(args, kwargs) (line 228)
        zip_call_result_129181 = invoke(stypy.reporting.localization.Localization(__file__, 228, 29), zip_129177, *[ashape_129178, astrides_129179], **kwargs_129180)
        
        # Testing the type of a for loop iterable (line 228)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 228, 8), zip_call_result_129181)
        # Getting the type of the for loop variable (line 228)
        for_loop_var_129182 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 228, 8), zip_call_result_129181)
        # Assigning a type to the variable 'shape' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'shape', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 8), for_loop_var_129182))
        # Assigning a type to the variable 'stride' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'stride', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 8), for_loop_var_129182))
        # SSA begins for a for statement (line 228)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'stride' (line 229)
        stride_129183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'stride')
        int_129184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 24), 'int')
        # Applying the binary operator '<' (line 229)
        result_lt_129185 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 15), '<', stride_129183, int_129184)
        
        # Testing the type of an if condition (line 229)
        if_condition_129186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 12), result_lt_129185)
        # Assigning a type to the variable 'if_condition_129186' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'if_condition_129186', if_condition_129186)
        # SSA begins for if statement (line 229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'a_low' (line 230)
        a_low_129187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'a_low')
        # Getting the type of 'shape' (line 230)
        shape_129188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 26), 'shape')
        int_129189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 32), 'int')
        # Applying the binary operator '-' (line 230)
        result_sub_129190 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 26), '-', shape_129188, int_129189)
        
        # Getting the type of 'stride' (line 230)
        stride_129191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 35), 'stride')
        # Applying the binary operator '*' (line 230)
        result_mul_129192 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 25), '*', result_sub_129190, stride_129191)
        
        # Applying the binary operator '+=' (line 230)
        result_iadd_129193 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 16), '+=', a_low_129187, result_mul_129192)
        # Assigning a type to the variable 'a_low' (line 230)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'a_low', result_iadd_129193)
        
        # SSA branch for the else part of an if statement (line 229)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'a_high' (line 232)
        a_high_129194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'a_high')
        # Getting the type of 'shape' (line 232)
        shape_129195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 27), 'shape')
        int_129196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 33), 'int')
        # Applying the binary operator '-' (line 232)
        result_sub_129197 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 27), '-', shape_129195, int_129196)
        
        # Getting the type of 'stride' (line 232)
        stride_129198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 36), 'stride')
        # Applying the binary operator '*' (line 232)
        result_mul_129199 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 26), '*', result_sub_129197, stride_129198)
        
        # Applying the binary operator '+=' (line 232)
        result_iadd_129200 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 16), '+=', a_high_129194, result_mul_129199)
        # Assigning a type to the variable 'a_high' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 16), 'a_high', result_iadd_129200)
        
        # SSA join for if statement (line 229)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'a_high' (line 233)
        a_high_129201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'a_high')
        # Getting the type of 'bytes_a' (line 233)
        bytes_a_129202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 18), 'bytes_a')
        # Applying the binary operator '+=' (line 233)
        result_iadd_129203 = python_operator(stypy.reporting.localization.Localization(__file__, 233, 8), '+=', a_high_129201, bytes_a_129202)
        # Assigning a type to the variable 'a_high' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'a_high', result_iadd_129203)
        

        if (may_be_129169 and more_types_in_union_129170):
            # SSA join for if statement (line 224)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 234)
    tuple_129204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 234)
    # Adding element type (line 234)
    # Getting the type of 'a_low' (line 234)
    a_low_129205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 11), 'a_low')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 11), tuple_129204, a_low_129205)
    # Adding element type (line 234)
    # Getting the type of 'a_high' (line 234)
    a_high_129206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 18), 'a_high')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 11), tuple_129204, a_high_129206)
    
    # Assigning a type to the variable 'stypy_return_type' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type', tuple_129204)
    
    # ################# End of 'byte_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'byte_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 185)
    stypy_return_type_129207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129207)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'byte_bounds'
    return stypy_return_type_129207

# Assigning a type to the variable 'byte_bounds' (line 185)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 0), 'byte_bounds', byte_bounds)

@norecursion
def who(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 242)
    None_129208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'None')
    defaults = [None_129208]
    # Create a new context for function 'who'
    module_type_store = module_type_store.open_function_context('who', 242, 0, False)
    
    # Passed parameters checking function
    who.stypy_localization = localization
    who.stypy_type_of_self = None
    who.stypy_type_store = module_type_store
    who.stypy_function_name = 'who'
    who.stypy_param_names_list = ['vardict']
    who.stypy_varargs_param_name = None
    who.stypy_kwargs_param_name = None
    who.stypy_call_defaults = defaults
    who.stypy_call_varargs = varargs
    who.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'who', ['vardict'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'who', localization, ['vardict'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'who(...)' code ##################

    str_129209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, (-1)), 'str', "\n    Print the Numpy arrays in the given dictionary.\n\n    If there is no dictionary passed in or `vardict` is None then returns\n    Numpy arrays in the globals() dictionary (all Numpy arrays in the\n    namespace).\n\n    Parameters\n    ----------\n    vardict : dict, optional\n        A dictionary possibly containing ndarrays.  Default is globals().\n\n    Returns\n    -------\n    out : None\n        Returns 'None'.\n\n    Notes\n    -----\n    Prints out the name, shape, bytes and type of all of the ndarrays\n    present in `vardict`.\n\n    Examples\n    --------\n    >>> a = np.arange(10)\n    >>> b = np.ones(20)\n    >>> np.who()\n    Name            Shape            Bytes            Type\n    ===========================================================\n    a               10               40               int32\n    b               20               160              float64\n    Upper bound on total bytes  =       200\n\n    >>> d = {'x': np.arange(2.0), 'y': np.arange(3.0), 'txt': 'Some str',\n    ... 'idx':5}\n    >>> np.who(d)\n    Name            Shape            Bytes            Type\n    ===========================================================\n    y               3                24               float64\n    x               2                16               float64\n    Upper bound on total bytes  =       40\n\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 286)
    # Getting the type of 'vardict' (line 286)
    vardict_129210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 7), 'vardict')
    # Getting the type of 'None' (line 286)
    None_129211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 18), 'None')
    
    (may_be_129212, more_types_in_union_129213) = may_be_none(vardict_129210, None_129211)

    if may_be_129212:

        if more_types_in_union_129213:
            # Runtime conditional SSA (line 286)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 287):
        
        # Assigning a Attribute to a Name (line 287):
        
        # Call to _getframe(...): (line 287)
        # Processing the call keyword arguments (line 287)
        kwargs_129216 = {}
        # Getting the type of 'sys' (line 287)
        sys_129214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'sys', False)
        # Obtaining the member '_getframe' of a type (line 287)
        _getframe_129215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 16), sys_129214, '_getframe')
        # Calling _getframe(args, kwargs) (line 287)
        _getframe_call_result_129217 = invoke(stypy.reporting.localization.Localization(__file__, 287, 16), _getframe_129215, *[], **kwargs_129216)
        
        # Obtaining the member 'f_back' of a type (line 287)
        f_back_129218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 16), _getframe_call_result_129217, 'f_back')
        # Assigning a type to the variable 'frame' (line 287)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'frame', f_back_129218)
        
        # Assigning a Attribute to a Name (line 288):
        
        # Assigning a Attribute to a Name (line 288):
        # Getting the type of 'frame' (line 288)
        frame_129219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'frame')
        # Obtaining the member 'f_globals' of a type (line 288)
        f_globals_129220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 18), frame_129219, 'f_globals')
        # Assigning a type to the variable 'vardict' (line 288)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'vardict', f_globals_129220)

        if more_types_in_union_129213:
            # SSA join for if statement (line 286)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a List to a Name (line 289):
    
    # Assigning a List to a Name (line 289):
    
    # Obtaining an instance of the builtin type 'list' (line 289)
    list_129221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 10), 'list')
    # Adding type elements to the builtin type 'list' instance (line 289)
    
    # Assigning a type to the variable 'sta' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'sta', list_129221)
    
    # Assigning a Dict to a Name (line 290):
    
    # Assigning a Dict to a Name (line 290):
    
    # Obtaining an instance of the builtin type 'dict' (line 290)
    dict_129222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 290)
    
    # Assigning a type to the variable 'cache' (line 290)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'cache', dict_129222)
    
    
    # Call to keys(...): (line 291)
    # Processing the call keyword arguments (line 291)
    kwargs_129225 = {}
    # Getting the type of 'vardict' (line 291)
    vardict_129223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'vardict', False)
    # Obtaining the member 'keys' of a type (line 291)
    keys_129224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 16), vardict_129223, 'keys')
    # Calling keys(args, kwargs) (line 291)
    keys_call_result_129226 = invoke(stypy.reporting.localization.Localization(__file__, 291, 16), keys_129224, *[], **kwargs_129225)
    
    # Testing the type of a for loop iterable (line 291)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 291, 4), keys_call_result_129226)
    # Getting the type of the for loop variable (line 291)
    for_loop_var_129227 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 291, 4), keys_call_result_129226)
    # Assigning a type to the variable 'name' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'name', for_loop_var_129227)
    # SSA begins for a for statement (line 291)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to isinstance(...): (line 292)
    # Processing the call arguments (line 292)
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 292)
    name_129229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'name', False)
    # Getting the type of 'vardict' (line 292)
    vardict_129230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 22), 'vardict', False)
    # Obtaining the member '__getitem__' of a type (line 292)
    getitem___129231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 22), vardict_129230, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 292)
    subscript_call_result_129232 = invoke(stypy.reporting.localization.Localization(__file__, 292, 22), getitem___129231, name_129229)
    
    # Getting the type of 'ndarray' (line 292)
    ndarray_129233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 37), 'ndarray', False)
    # Processing the call keyword arguments (line 292)
    kwargs_129234 = {}
    # Getting the type of 'isinstance' (line 292)
    isinstance_129228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 292)
    isinstance_call_result_129235 = invoke(stypy.reporting.localization.Localization(__file__, 292, 11), isinstance_129228, *[subscript_call_result_129232, ndarray_129233], **kwargs_129234)
    
    # Testing the type of an if condition (line 292)
    if_condition_129236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 292, 8), isinstance_call_result_129235)
    # Assigning a type to the variable 'if_condition_129236' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'if_condition_129236', if_condition_129236)
    # SSA begins for if statement (line 292)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 293):
    
    # Assigning a Subscript to a Name (line 293):
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 293)
    name_129237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 26), 'name')
    # Getting the type of 'vardict' (line 293)
    vardict_129238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 18), 'vardict')
    # Obtaining the member '__getitem__' of a type (line 293)
    getitem___129239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 18), vardict_129238, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 293)
    subscript_call_result_129240 = invoke(stypy.reporting.localization.Localization(__file__, 293, 18), getitem___129239, name_129237)
    
    # Assigning a type to the variable 'var' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 12), 'var', subscript_call_result_129240)
    
    # Assigning a Call to a Name (line 294):
    
    # Assigning a Call to a Name (line 294):
    
    # Call to id(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'var' (line 294)
    var_129242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 21), 'var', False)
    # Processing the call keyword arguments (line 294)
    kwargs_129243 = {}
    # Getting the type of 'id' (line 294)
    id_129241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 18), 'id', False)
    # Calling id(args, kwargs) (line 294)
    id_call_result_129244 = invoke(stypy.reporting.localization.Localization(__file__, 294, 18), id_129241, *[var_129242], **kwargs_129243)
    
    # Assigning a type to the variable 'idv' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'idv', id_call_result_129244)
    
    
    # Getting the type of 'idv' (line 295)
    idv_129245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'idv')
    
    # Call to keys(...): (line 295)
    # Processing the call keyword arguments (line 295)
    kwargs_129248 = {}
    # Getting the type of 'cache' (line 295)
    cache_129246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 22), 'cache', False)
    # Obtaining the member 'keys' of a type (line 295)
    keys_129247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 22), cache_129246, 'keys')
    # Calling keys(args, kwargs) (line 295)
    keys_call_result_129249 = invoke(stypy.reporting.localization.Localization(__file__, 295, 22), keys_129247, *[], **kwargs_129248)
    
    # Applying the binary operator 'in' (line 295)
    result_contains_129250 = python_operator(stypy.reporting.localization.Localization(__file__, 295, 15), 'in', idv_129245, keys_call_result_129249)
    
    # Testing the type of an if condition (line 295)
    if_condition_129251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 295, 12), result_contains_129250)
    # Assigning a type to the variable 'if_condition_129251' (line 295)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'if_condition_129251', if_condition_129251)
    # SSA begins for if statement (line 295)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 296):
    
    # Assigning a BinOp to a Name (line 296):
    # Getting the type of 'name' (line 296)
    name_129252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 26), 'name')
    str_129253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 33), 'str', ' (%s)')
    
    # Obtaining the type of the subscript
    # Getting the type of 'idv' (line 296)
    idv_129254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 49), 'idv')
    # Getting the type of 'cache' (line 296)
    cache_129255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 43), 'cache')
    # Obtaining the member '__getitem__' of a type (line 296)
    getitem___129256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 43), cache_129255, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 296)
    subscript_call_result_129257 = invoke(stypy.reporting.localization.Localization(__file__, 296, 43), getitem___129256, idv_129254)
    
    # Applying the binary operator '%' (line 296)
    result_mod_129258 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 33), '%', str_129253, subscript_call_result_129257)
    
    # Applying the binary operator '+' (line 296)
    result_add_129259 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 26), '+', name_129252, result_mod_129258)
    
    # Assigning a type to the variable 'namestr' (line 296)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'namestr', result_add_129259)
    
    # Assigning a Num to a Name (line 297):
    
    # Assigning a Num to a Name (line 297):
    int_129260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 27), 'int')
    # Assigning a type to the variable 'original' (line 297)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'original', int_129260)
    # SSA branch for the else part of an if statement (line 295)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 299):
    
    # Assigning a Name to a Subscript (line 299):
    # Getting the type of 'name' (line 299)
    name_129261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 29), 'name')
    # Getting the type of 'cache' (line 299)
    cache_129262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'cache')
    # Getting the type of 'idv' (line 299)
    idv_129263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 22), 'idv')
    # Storing an element on a container (line 299)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 299, 16), cache_129262, (idv_129263, name_129261))
    
    # Assigning a Name to a Name (line 300):
    
    # Assigning a Name to a Name (line 300):
    # Getting the type of 'name' (line 300)
    name_129264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 26), 'name')
    # Assigning a type to the variable 'namestr' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'namestr', name_129264)
    
    # Assigning a Num to a Name (line 301):
    
    # Assigning a Num to a Name (line 301):
    int_129265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 27), 'int')
    # Assigning a type to the variable 'original' (line 301)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'original', int_129265)
    # SSA join for if statement (line 295)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 302):
    
    # Assigning a Call to a Name (line 302):
    
    # Call to join(...): (line 302)
    # Processing the call arguments (line 302)
    
    # Call to map(...): (line 302)
    # Processing the call arguments (line 302)
    # Getting the type of 'str' (line 302)
    str_129269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 38), 'str', False)
    # Getting the type of 'var' (line 302)
    var_129270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 43), 'var', False)
    # Obtaining the member 'shape' of a type (line 302)
    shape_129271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 43), var_129270, 'shape')
    # Processing the call keyword arguments (line 302)
    kwargs_129272 = {}
    # Getting the type of 'map' (line 302)
    map_129268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 34), 'map', False)
    # Calling map(args, kwargs) (line 302)
    map_call_result_129273 = invoke(stypy.reporting.localization.Localization(__file__, 302, 34), map_129268, *[str_129269, shape_129271], **kwargs_129272)
    
    # Processing the call keyword arguments (line 302)
    kwargs_129274 = {}
    str_129266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 23), 'str', ' x ')
    # Obtaining the member 'join' of a type (line 302)
    join_129267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 23), str_129266, 'join')
    # Calling join(args, kwargs) (line 302)
    join_call_result_129275 = invoke(stypy.reporting.localization.Localization(__file__, 302, 23), join_129267, *[map_call_result_129273], **kwargs_129274)
    
    # Assigning a type to the variable 'shapestr' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'shapestr', join_call_result_129275)
    
    # Assigning a Call to a Name (line 303):
    
    # Assigning a Call to a Name (line 303):
    
    # Call to str(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'var' (line 303)
    var_129277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 26), 'var', False)
    # Obtaining the member 'nbytes' of a type (line 303)
    nbytes_129278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 26), var_129277, 'nbytes')
    # Processing the call keyword arguments (line 303)
    kwargs_129279 = {}
    # Getting the type of 'str' (line 303)
    str_129276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 22), 'str', False)
    # Calling str(args, kwargs) (line 303)
    str_call_result_129280 = invoke(stypy.reporting.localization.Localization(__file__, 303, 22), str_129276, *[nbytes_129278], **kwargs_129279)
    
    # Assigning a type to the variable 'bytestr' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 12), 'bytestr', str_call_result_129280)
    
    # Call to append(...): (line 304)
    # Processing the call arguments (line 304)
    
    # Obtaining an instance of the builtin type 'list' (line 304)
    list_129283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 304)
    # Adding element type (line 304)
    # Getting the type of 'namestr' (line 304)
    namestr_129284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 24), 'namestr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 23), list_129283, namestr_129284)
    # Adding element type (line 304)
    # Getting the type of 'shapestr' (line 304)
    shapestr_129285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 33), 'shapestr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 23), list_129283, shapestr_129285)
    # Adding element type (line 304)
    # Getting the type of 'bytestr' (line 304)
    bytestr_129286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 43), 'bytestr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 23), list_129283, bytestr_129286)
    # Adding element type (line 304)
    # Getting the type of 'var' (line 304)
    var_129287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 52), 'var', False)
    # Obtaining the member 'dtype' of a type (line 304)
    dtype_129288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 52), var_129287, 'dtype')
    # Obtaining the member 'name' of a type (line 304)
    name_129289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 52), dtype_129288, 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 23), list_129283, name_129289)
    # Adding element type (line 304)
    # Getting the type of 'original' (line 305)
    original_129290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 24), 'original', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 304, 23), list_129283, original_129290)
    
    # Processing the call keyword arguments (line 304)
    kwargs_129291 = {}
    # Getting the type of 'sta' (line 304)
    sta_129281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'sta', False)
    # Obtaining the member 'append' of a type (line 304)
    append_129282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), sta_129281, 'append')
    # Calling append(args, kwargs) (line 304)
    append_call_result_129292 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), append_129282, *[list_129283], **kwargs_129291)
    
    # SSA join for if statement (line 292)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 307):
    
    # Assigning a Num to a Name (line 307):
    int_129293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 14), 'int')
    # Assigning a type to the variable 'maxname' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'maxname', int_129293)
    
    # Assigning a Num to a Name (line 308):
    
    # Assigning a Num to a Name (line 308):
    int_129294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 15), 'int')
    # Assigning a type to the variable 'maxshape' (line 308)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'maxshape', int_129294)
    
    # Assigning a Num to a Name (line 309):
    
    # Assigning a Num to a Name (line 309):
    int_129295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 14), 'int')
    # Assigning a type to the variable 'maxbyte' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'maxbyte', int_129295)
    
    # Assigning a Num to a Name (line 310):
    
    # Assigning a Num to a Name (line 310):
    int_129296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 17), 'int')
    # Assigning a type to the variable 'totalbytes' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'totalbytes', int_129296)
    
    
    # Call to range(...): (line 311)
    # Processing the call arguments (line 311)
    
    # Call to len(...): (line 311)
    # Processing the call arguments (line 311)
    # Getting the type of 'sta' (line 311)
    sta_129299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 23), 'sta', False)
    # Processing the call keyword arguments (line 311)
    kwargs_129300 = {}
    # Getting the type of 'len' (line 311)
    len_129298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 19), 'len', False)
    # Calling len(args, kwargs) (line 311)
    len_call_result_129301 = invoke(stypy.reporting.localization.Localization(__file__, 311, 19), len_129298, *[sta_129299], **kwargs_129300)
    
    # Processing the call keyword arguments (line 311)
    kwargs_129302 = {}
    # Getting the type of 'range' (line 311)
    range_129297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 13), 'range', False)
    # Calling range(args, kwargs) (line 311)
    range_call_result_129303 = invoke(stypy.reporting.localization.Localization(__file__, 311, 13), range_129297, *[len_call_result_129301], **kwargs_129302)
    
    # Testing the type of a for loop iterable (line 311)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 311, 4), range_call_result_129303)
    # Getting the type of the for loop variable (line 311)
    for_loop_var_129304 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 311, 4), range_call_result_129303)
    # Assigning a type to the variable 'k' (line 311)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 4), 'k', for_loop_var_129304)
    # SSA begins for a for statement (line 311)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 312):
    
    # Assigning a Subscript to a Name (line 312):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 312)
    k_129305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 18), 'k')
    # Getting the type of 'sta' (line 312)
    sta_129306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 14), 'sta')
    # Obtaining the member '__getitem__' of a type (line 312)
    getitem___129307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 14), sta_129306, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 312)
    subscript_call_result_129308 = invoke(stypy.reporting.localization.Localization(__file__, 312, 14), getitem___129307, k_129305)
    
    # Assigning a type to the variable 'val' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'val', subscript_call_result_129308)
    
    
    # Getting the type of 'maxname' (line 313)
    maxname_129309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 11), 'maxname')
    
    # Call to len(...): (line 313)
    # Processing the call arguments (line 313)
    
    # Obtaining the type of the subscript
    int_129311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 29), 'int')
    # Getting the type of 'val' (line 313)
    val_129312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 25), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 313)
    getitem___129313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 25), val_129312, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 313)
    subscript_call_result_129314 = invoke(stypy.reporting.localization.Localization(__file__, 313, 25), getitem___129313, int_129311)
    
    # Processing the call keyword arguments (line 313)
    kwargs_129315 = {}
    # Getting the type of 'len' (line 313)
    len_129310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 21), 'len', False)
    # Calling len(args, kwargs) (line 313)
    len_call_result_129316 = invoke(stypy.reporting.localization.Localization(__file__, 313, 21), len_129310, *[subscript_call_result_129314], **kwargs_129315)
    
    # Applying the binary operator '<' (line 313)
    result_lt_129317 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 11), '<', maxname_129309, len_call_result_129316)
    
    # Testing the type of an if condition (line 313)
    if_condition_129318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 8), result_lt_129317)
    # Assigning a type to the variable 'if_condition_129318' (line 313)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'if_condition_129318', if_condition_129318)
    # SSA begins for if statement (line 313)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 314):
    
    # Assigning a Call to a Name (line 314):
    
    # Call to len(...): (line 314)
    # Processing the call arguments (line 314)
    
    # Obtaining the type of the subscript
    int_129320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 30), 'int')
    # Getting the type of 'val' (line 314)
    val_129321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 26), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 314)
    getitem___129322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 26), val_129321, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 314)
    subscript_call_result_129323 = invoke(stypy.reporting.localization.Localization(__file__, 314, 26), getitem___129322, int_129320)
    
    # Processing the call keyword arguments (line 314)
    kwargs_129324 = {}
    # Getting the type of 'len' (line 314)
    len_129319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 22), 'len', False)
    # Calling len(args, kwargs) (line 314)
    len_call_result_129325 = invoke(stypy.reporting.localization.Localization(__file__, 314, 22), len_129319, *[subscript_call_result_129323], **kwargs_129324)
    
    # Assigning a type to the variable 'maxname' (line 314)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'maxname', len_call_result_129325)
    # SSA join for if statement (line 313)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'maxshape' (line 315)
    maxshape_129326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 11), 'maxshape')
    
    # Call to len(...): (line 315)
    # Processing the call arguments (line 315)
    
    # Obtaining the type of the subscript
    int_129328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 30), 'int')
    # Getting the type of 'val' (line 315)
    val_129329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 26), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 315)
    getitem___129330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 26), val_129329, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 315)
    subscript_call_result_129331 = invoke(stypy.reporting.localization.Localization(__file__, 315, 26), getitem___129330, int_129328)
    
    # Processing the call keyword arguments (line 315)
    kwargs_129332 = {}
    # Getting the type of 'len' (line 315)
    len_129327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 22), 'len', False)
    # Calling len(args, kwargs) (line 315)
    len_call_result_129333 = invoke(stypy.reporting.localization.Localization(__file__, 315, 22), len_129327, *[subscript_call_result_129331], **kwargs_129332)
    
    # Applying the binary operator '<' (line 315)
    result_lt_129334 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 11), '<', maxshape_129326, len_call_result_129333)
    
    # Testing the type of an if condition (line 315)
    if_condition_129335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 315, 8), result_lt_129334)
    # Assigning a type to the variable 'if_condition_129335' (line 315)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'if_condition_129335', if_condition_129335)
    # SSA begins for if statement (line 315)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 316):
    
    # Assigning a Call to a Name (line 316):
    
    # Call to len(...): (line 316)
    # Processing the call arguments (line 316)
    
    # Obtaining the type of the subscript
    int_129337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 316, 31), 'int')
    # Getting the type of 'val' (line 316)
    val_129338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 27), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 316)
    getitem___129339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 27), val_129338, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 316)
    subscript_call_result_129340 = invoke(stypy.reporting.localization.Localization(__file__, 316, 27), getitem___129339, int_129337)
    
    # Processing the call keyword arguments (line 316)
    kwargs_129341 = {}
    # Getting the type of 'len' (line 316)
    len_129336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 'len', False)
    # Calling len(args, kwargs) (line 316)
    len_call_result_129342 = invoke(stypy.reporting.localization.Localization(__file__, 316, 23), len_129336, *[subscript_call_result_129340], **kwargs_129341)
    
    # Assigning a type to the variable 'maxshape' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'maxshape', len_call_result_129342)
    # SSA join for if statement (line 315)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'maxbyte' (line 317)
    maxbyte_129343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 11), 'maxbyte')
    
    # Call to len(...): (line 317)
    # Processing the call arguments (line 317)
    
    # Obtaining the type of the subscript
    int_129345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 29), 'int')
    # Getting the type of 'val' (line 317)
    val_129346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 25), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 317)
    getitem___129347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 25), val_129346, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 317)
    subscript_call_result_129348 = invoke(stypy.reporting.localization.Localization(__file__, 317, 25), getitem___129347, int_129345)
    
    # Processing the call keyword arguments (line 317)
    kwargs_129349 = {}
    # Getting the type of 'len' (line 317)
    len_129344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 21), 'len', False)
    # Calling len(args, kwargs) (line 317)
    len_call_result_129350 = invoke(stypy.reporting.localization.Localization(__file__, 317, 21), len_129344, *[subscript_call_result_129348], **kwargs_129349)
    
    # Applying the binary operator '<' (line 317)
    result_lt_129351 = python_operator(stypy.reporting.localization.Localization(__file__, 317, 11), '<', maxbyte_129343, len_call_result_129350)
    
    # Testing the type of an if condition (line 317)
    if_condition_129352 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 317, 8), result_lt_129351)
    # Assigning a type to the variable 'if_condition_129352' (line 317)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 317, 8), 'if_condition_129352', if_condition_129352)
    # SSA begins for if statement (line 317)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 318):
    
    # Assigning a Call to a Name (line 318):
    
    # Call to len(...): (line 318)
    # Processing the call arguments (line 318)
    
    # Obtaining the type of the subscript
    int_129354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 30), 'int')
    # Getting the type of 'val' (line 318)
    val_129355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 26), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 318)
    getitem___129356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 26), val_129355, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 318)
    subscript_call_result_129357 = invoke(stypy.reporting.localization.Localization(__file__, 318, 26), getitem___129356, int_129354)
    
    # Processing the call keyword arguments (line 318)
    kwargs_129358 = {}
    # Getting the type of 'len' (line 318)
    len_129353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 22), 'len', False)
    # Calling len(args, kwargs) (line 318)
    len_call_result_129359 = invoke(stypy.reporting.localization.Localization(__file__, 318, 22), len_129353, *[subscript_call_result_129357], **kwargs_129358)
    
    # Assigning a type to the variable 'maxbyte' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'maxbyte', len_call_result_129359)
    # SSA join for if statement (line 317)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    int_129360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 15), 'int')
    # Getting the type of 'val' (line 319)
    val_129361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 11), 'val')
    # Obtaining the member '__getitem__' of a type (line 319)
    getitem___129362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 11), val_129361, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 319)
    subscript_call_result_129363 = invoke(stypy.reporting.localization.Localization(__file__, 319, 11), getitem___129362, int_129360)
    
    # Testing the type of an if condition (line 319)
    if_condition_129364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 319, 8), subscript_call_result_129363)
    # Assigning a type to the variable 'if_condition_129364' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'if_condition_129364', if_condition_129364)
    # SSA begins for if statement (line 319)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'totalbytes' (line 320)
    totalbytes_129365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'totalbytes')
    
    # Call to int(...): (line 320)
    # Processing the call arguments (line 320)
    
    # Obtaining the type of the subscript
    int_129367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 34), 'int')
    # Getting the type of 'val' (line 320)
    val_129368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 30), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 320)
    getitem___129369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 30), val_129368, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 320)
    subscript_call_result_129370 = invoke(stypy.reporting.localization.Localization(__file__, 320, 30), getitem___129369, int_129367)
    
    # Processing the call keyword arguments (line 320)
    kwargs_129371 = {}
    # Getting the type of 'int' (line 320)
    int_129366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 26), 'int', False)
    # Calling int(args, kwargs) (line 320)
    int_call_result_129372 = invoke(stypy.reporting.localization.Localization(__file__, 320, 26), int_129366, *[subscript_call_result_129370], **kwargs_129371)
    
    # Applying the binary operator '+=' (line 320)
    result_iadd_129373 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 12), '+=', totalbytes_129365, int_call_result_129372)
    # Assigning a type to the variable 'totalbytes' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'totalbytes', result_iadd_129373)
    
    # SSA join for if statement (line 319)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 322)
    # Processing the call arguments (line 322)
    # Getting the type of 'sta' (line 322)
    sta_129375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 11), 'sta', False)
    # Processing the call keyword arguments (line 322)
    kwargs_129376 = {}
    # Getting the type of 'len' (line 322)
    len_129374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 7), 'len', False)
    # Calling len(args, kwargs) (line 322)
    len_call_result_129377 = invoke(stypy.reporting.localization.Localization(__file__, 322, 7), len_129374, *[sta_129375], **kwargs_129376)
    
    int_129378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 18), 'int')
    # Applying the binary operator '>' (line 322)
    result_gt_129379 = python_operator(stypy.reporting.localization.Localization(__file__, 322, 7), '>', len_call_result_129377, int_129378)
    
    # Testing the type of an if condition (line 322)
    if_condition_129380 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 322, 4), result_gt_129379)
    # Assigning a type to the variable 'if_condition_129380' (line 322)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 322, 4), 'if_condition_129380', if_condition_129380)
    # SSA begins for if statement (line 322)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 323):
    
    # Assigning a Call to a Name (line 323):
    
    # Call to max(...): (line 323)
    # Processing the call arguments (line 323)
    int_129382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 323, 18), 'int')
    # Getting the type of 'maxname' (line 323)
    maxname_129383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 22), 'maxname', False)
    # Processing the call keyword arguments (line 323)
    kwargs_129384 = {}
    # Getting the type of 'max' (line 323)
    max_129381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 14), 'max', False)
    # Calling max(args, kwargs) (line 323)
    max_call_result_129385 = invoke(stypy.reporting.localization.Localization(__file__, 323, 14), max_129381, *[int_129382, maxname_129383], **kwargs_129384)
    
    # Assigning a type to the variable 'sp1' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'sp1', max_call_result_129385)
    
    # Assigning a Call to a Name (line 324):
    
    # Assigning a Call to a Name (line 324):
    
    # Call to max(...): (line 324)
    # Processing the call arguments (line 324)
    int_129387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 18), 'int')
    # Getting the type of 'maxshape' (line 324)
    maxshape_129388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 22), 'maxshape', False)
    # Processing the call keyword arguments (line 324)
    kwargs_129389 = {}
    # Getting the type of 'max' (line 324)
    max_129386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 14), 'max', False)
    # Calling max(args, kwargs) (line 324)
    max_call_result_129390 = invoke(stypy.reporting.localization.Localization(__file__, 324, 14), max_129386, *[int_129387, maxshape_129388], **kwargs_129389)
    
    # Assigning a type to the variable 'sp2' (line 324)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'sp2', max_call_result_129390)
    
    # Assigning a Call to a Name (line 325):
    
    # Assigning a Call to a Name (line 325):
    
    # Call to max(...): (line 325)
    # Processing the call arguments (line 325)
    int_129392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 325, 18), 'int')
    # Getting the type of 'maxbyte' (line 325)
    maxbyte_129393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 22), 'maxbyte', False)
    # Processing the call keyword arguments (line 325)
    kwargs_129394 = {}
    # Getting the type of 'max' (line 325)
    max_129391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 14), 'max', False)
    # Calling max(args, kwargs) (line 325)
    max_call_result_129395 = invoke(stypy.reporting.localization.Localization(__file__, 325, 14), max_129391, *[int_129392, maxbyte_129393], **kwargs_129394)
    
    # Assigning a type to the variable 'sp3' (line 325)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'sp3', max_call_result_129395)
    
    # Assigning a BinOp to a Name (line 326):
    
    # Assigning a BinOp to a Name (line 326):
    str_129396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 16), 'str', 'Name %s Shape %s Bytes %s Type')
    
    # Obtaining an instance of the builtin type 'tuple' (line 326)
    tuple_129397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 52), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 326)
    # Adding element type (line 326)
    # Getting the type of 'sp1' (line 326)
    sp1_129398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 52), 'sp1')
    str_129399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 56), 'str', ' ')
    # Applying the binary operator '*' (line 326)
    result_mul_129400 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 52), '*', sp1_129398, str_129399)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 52), tuple_129397, result_mul_129400)
    # Adding element type (line 326)
    # Getting the type of 'sp2' (line 326)
    sp2_129401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 61), 'sp2')
    str_129402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 65), 'str', ' ')
    # Applying the binary operator '*' (line 326)
    result_mul_129403 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 61), '*', sp2_129401, str_129402)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 52), tuple_129397, result_mul_129403)
    # Adding element type (line 326)
    # Getting the type of 'sp3' (line 326)
    sp3_129404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 70), 'sp3')
    str_129405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 74), 'str', ' ')
    # Applying the binary operator '*' (line 326)
    result_mul_129406 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 70), '*', sp3_129404, str_129405)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 52), tuple_129397, result_mul_129406)
    
    # Applying the binary operator '%' (line 326)
    result_mod_129407 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 16), '%', str_129396, tuple_129397)
    
    # Assigning a type to the variable 'prval' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'prval', result_mod_129407)
    
    # Call to print(...): (line 327)
    # Processing the call arguments (line 327)
    # Getting the type of 'prval' (line 327)
    prval_129409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 14), 'prval', False)
    str_129410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 22), 'str', '\n')
    # Applying the binary operator '+' (line 327)
    result_add_129411 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 14), '+', prval_129409, str_129410)
    
    str_129412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 29), 'str', '=')
    
    # Call to len(...): (line 327)
    # Processing the call arguments (line 327)
    # Getting the type of 'prval' (line 327)
    prval_129414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 38), 'prval', False)
    # Processing the call keyword arguments (line 327)
    kwargs_129415 = {}
    # Getting the type of 'len' (line 327)
    len_129413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 34), 'len', False)
    # Calling len(args, kwargs) (line 327)
    len_call_result_129416 = invoke(stypy.reporting.localization.Localization(__file__, 327, 34), len_129413, *[prval_129414], **kwargs_129415)
    
    int_129417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 45), 'int')
    # Applying the binary operator '+' (line 327)
    result_add_129418 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 34), '+', len_call_result_129416, int_129417)
    
    # Applying the binary operator '*' (line 327)
    result_mul_129419 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 29), '*', str_129412, result_add_129418)
    
    # Applying the binary operator '+' (line 327)
    result_add_129420 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 27), '+', result_add_129411, result_mul_129419)
    
    str_129421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 50), 'str', '\n')
    # Applying the binary operator '+' (line 327)
    result_add_129422 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 48), '+', result_add_129420, str_129421)
    
    # Processing the call keyword arguments (line 327)
    kwargs_129423 = {}
    # Getting the type of 'print' (line 327)
    print_129408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'print', False)
    # Calling print(args, kwargs) (line 327)
    print_call_result_129424 = invoke(stypy.reporting.localization.Localization(__file__, 327, 8), print_129408, *[result_add_129422], **kwargs_129423)
    
    # SSA join for if statement (line 322)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to range(...): (line 329)
    # Processing the call arguments (line 329)
    
    # Call to len(...): (line 329)
    # Processing the call arguments (line 329)
    # Getting the type of 'sta' (line 329)
    sta_129427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 23), 'sta', False)
    # Processing the call keyword arguments (line 329)
    kwargs_129428 = {}
    # Getting the type of 'len' (line 329)
    len_129426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 19), 'len', False)
    # Calling len(args, kwargs) (line 329)
    len_call_result_129429 = invoke(stypy.reporting.localization.Localization(__file__, 329, 19), len_129426, *[sta_129427], **kwargs_129428)
    
    # Processing the call keyword arguments (line 329)
    kwargs_129430 = {}
    # Getting the type of 'range' (line 329)
    range_129425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 13), 'range', False)
    # Calling range(args, kwargs) (line 329)
    range_call_result_129431 = invoke(stypy.reporting.localization.Localization(__file__, 329, 13), range_129425, *[len_call_result_129429], **kwargs_129430)
    
    # Testing the type of a for loop iterable (line 329)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 329, 4), range_call_result_129431)
    # Getting the type of the for loop variable (line 329)
    for_loop_var_129432 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 329, 4), range_call_result_129431)
    # Assigning a type to the variable 'k' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'k', for_loop_var_129432)
    # SSA begins for a for statement (line 329)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 330):
    
    # Assigning a Subscript to a Name (line 330):
    
    # Obtaining the type of the subscript
    # Getting the type of 'k' (line 330)
    k_129433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 18), 'k')
    # Getting the type of 'sta' (line 330)
    sta_129434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 14), 'sta')
    # Obtaining the member '__getitem__' of a type (line 330)
    getitem___129435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 14), sta_129434, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 330)
    subscript_call_result_129436 = invoke(stypy.reporting.localization.Localization(__file__, 330, 14), getitem___129435, k_129433)
    
    # Assigning a type to the variable 'val' (line 330)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'val', subscript_call_result_129436)
    
    # Call to print(...): (line 331)
    # Processing the call arguments (line 331)
    str_129438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 14), 'str', '%s %s %s %s %s %s %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 331)
    tuple_129439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 40), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 331)
    # Adding element type (line 331)
    
    # Obtaining the type of the subscript
    int_129440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 44), 'int')
    # Getting the type of 'val' (line 331)
    val_129441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 40), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___129442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 40), val_129441, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 331)
    subscript_call_result_129443 = invoke(stypy.reporting.localization.Localization(__file__, 331, 40), getitem___129442, int_129440)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 40), tuple_129439, subscript_call_result_129443)
    # Adding element type (line 331)
    str_129444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 48), 'str', ' ')
    # Getting the type of 'sp1' (line 331)
    sp1_129445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 53), 'sp1', False)
    
    # Call to len(...): (line 331)
    # Processing the call arguments (line 331)
    
    # Obtaining the type of the subscript
    int_129447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 65), 'int')
    # Getting the type of 'val' (line 331)
    val_129448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 61), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 331)
    getitem___129449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 61), val_129448, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 331)
    subscript_call_result_129450 = invoke(stypy.reporting.localization.Localization(__file__, 331, 61), getitem___129449, int_129447)
    
    # Processing the call keyword arguments (line 331)
    kwargs_129451 = {}
    # Getting the type of 'len' (line 331)
    len_129446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 57), 'len', False)
    # Calling len(args, kwargs) (line 331)
    len_call_result_129452 = invoke(stypy.reporting.localization.Localization(__file__, 331, 57), len_129446, *[subscript_call_result_129450], **kwargs_129451)
    
    # Applying the binary operator '-' (line 331)
    result_sub_129453 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 53), '-', sp1_129445, len_call_result_129452)
    
    int_129454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 69), 'int')
    # Applying the binary operator '+' (line 331)
    result_add_129455 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 68), '+', result_sub_129453, int_129454)
    
    # Applying the binary operator '*' (line 331)
    result_mul_129456 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 48), '*', str_129444, result_add_129455)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 40), tuple_129439, result_mul_129456)
    # Adding element type (line 331)
    
    # Obtaining the type of the subscript
    int_129457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 44), 'int')
    # Getting the type of 'val' (line 332)
    val_129458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 40), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 332)
    getitem___129459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 40), val_129458, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 332)
    subscript_call_result_129460 = invoke(stypy.reporting.localization.Localization(__file__, 332, 40), getitem___129459, int_129457)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 40), tuple_129439, subscript_call_result_129460)
    # Adding element type (line 331)
    str_129461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 48), 'str', ' ')
    # Getting the type of 'sp2' (line 332)
    sp2_129462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 53), 'sp2', False)
    
    # Call to len(...): (line 332)
    # Processing the call arguments (line 332)
    
    # Obtaining the type of the subscript
    int_129464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 65), 'int')
    # Getting the type of 'val' (line 332)
    val_129465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 61), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 332)
    getitem___129466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 61), val_129465, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 332)
    subscript_call_result_129467 = invoke(stypy.reporting.localization.Localization(__file__, 332, 61), getitem___129466, int_129464)
    
    # Processing the call keyword arguments (line 332)
    kwargs_129468 = {}
    # Getting the type of 'len' (line 332)
    len_129463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 57), 'len', False)
    # Calling len(args, kwargs) (line 332)
    len_call_result_129469 = invoke(stypy.reporting.localization.Localization(__file__, 332, 57), len_129463, *[subscript_call_result_129467], **kwargs_129468)
    
    # Applying the binary operator '-' (line 332)
    result_sub_129470 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 53), '-', sp2_129462, len_call_result_129469)
    
    int_129471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 332, 69), 'int')
    # Applying the binary operator '+' (line 332)
    result_add_129472 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 68), '+', result_sub_129470, int_129471)
    
    # Applying the binary operator '*' (line 332)
    result_mul_129473 = python_operator(stypy.reporting.localization.Localization(__file__, 332, 48), '*', str_129461, result_add_129472)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 40), tuple_129439, result_mul_129473)
    # Adding element type (line 331)
    
    # Obtaining the type of the subscript
    int_129474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 44), 'int')
    # Getting the type of 'val' (line 333)
    val_129475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 40), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___129476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 40), val_129475, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_129477 = invoke(stypy.reporting.localization.Localization(__file__, 333, 40), getitem___129476, int_129474)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 40), tuple_129439, subscript_call_result_129477)
    # Adding element type (line 331)
    str_129478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 48), 'str', ' ')
    # Getting the type of 'sp3' (line 333)
    sp3_129479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 53), 'sp3', False)
    
    # Call to len(...): (line 333)
    # Processing the call arguments (line 333)
    
    # Obtaining the type of the subscript
    int_129481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 65), 'int')
    # Getting the type of 'val' (line 333)
    val_129482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 61), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 333)
    getitem___129483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 61), val_129482, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 333)
    subscript_call_result_129484 = invoke(stypy.reporting.localization.Localization(__file__, 333, 61), getitem___129483, int_129481)
    
    # Processing the call keyword arguments (line 333)
    kwargs_129485 = {}
    # Getting the type of 'len' (line 333)
    len_129480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 57), 'len', False)
    # Calling len(args, kwargs) (line 333)
    len_call_result_129486 = invoke(stypy.reporting.localization.Localization(__file__, 333, 57), len_129480, *[subscript_call_result_129484], **kwargs_129485)
    
    # Applying the binary operator '-' (line 333)
    result_sub_129487 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 53), '-', sp3_129479, len_call_result_129486)
    
    int_129488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 333, 69), 'int')
    # Applying the binary operator '+' (line 333)
    result_add_129489 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 68), '+', result_sub_129487, int_129488)
    
    # Applying the binary operator '*' (line 333)
    result_mul_129490 = python_operator(stypy.reporting.localization.Localization(__file__, 333, 48), '*', str_129478, result_add_129489)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 40), tuple_129439, result_mul_129490)
    # Adding element type (line 331)
    
    # Obtaining the type of the subscript
    int_129491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 44), 'int')
    # Getting the type of 'val' (line 334)
    val_129492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 40), 'val', False)
    # Obtaining the member '__getitem__' of a type (line 334)
    getitem___129493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 40), val_129492, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 334)
    subscript_call_result_129494 = invoke(stypy.reporting.localization.Localization(__file__, 334, 40), getitem___129493, int_129491)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 331, 40), tuple_129439, subscript_call_result_129494)
    
    # Applying the binary operator '%' (line 331)
    result_mod_129495 = python_operator(stypy.reporting.localization.Localization(__file__, 331, 14), '%', str_129438, tuple_129439)
    
    # Processing the call keyword arguments (line 331)
    kwargs_129496 = {}
    # Getting the type of 'print' (line 331)
    print_129437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'print', False)
    # Calling print(args, kwargs) (line 331)
    print_call_result_129497 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), print_129437, *[result_mod_129495], **kwargs_129496)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 335)
    # Processing the call arguments (line 335)
    str_129499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 10), 'str', '\nUpper bound on total bytes  =       %d')
    # Getting the type of 'totalbytes' (line 335)
    totalbytes_129500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 55), 'totalbytes', False)
    # Applying the binary operator '%' (line 335)
    result_mod_129501 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 10), '%', str_129499, totalbytes_129500)
    
    # Processing the call keyword arguments (line 335)
    kwargs_129502 = {}
    # Getting the type of 'print' (line 335)
    print_129498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'print', False)
    # Calling print(args, kwargs) (line 335)
    print_call_result_129503 = invoke(stypy.reporting.localization.Localization(__file__, 335, 4), print_129498, *[result_mod_129501], **kwargs_129502)
    
    # Assigning a type to the variable 'stypy_return_type' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of 'who(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'who' in the type store
    # Getting the type of 'stypy_return_type' (line 242)
    stypy_return_type_129504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129504)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'who'
    return stypy_return_type_129504

# Assigning a type to the variable 'who' (line 242)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 0), 'who', who)

@norecursion
def _split_line(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_split_line'
    module_type_store = module_type_store.open_function_context('_split_line', 347, 0, False)
    
    # Passed parameters checking function
    _split_line.stypy_localization = localization
    _split_line.stypy_type_of_self = None
    _split_line.stypy_type_store = module_type_store
    _split_line.stypy_function_name = '_split_line'
    _split_line.stypy_param_names_list = ['name', 'arguments', 'width']
    _split_line.stypy_varargs_param_name = None
    _split_line.stypy_kwargs_param_name = None
    _split_line.stypy_call_defaults = defaults
    _split_line.stypy_call_varargs = varargs
    _split_line.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_split_line', ['name', 'arguments', 'width'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_split_line', localization, ['name', 'arguments', 'width'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_split_line(...)' code ##################

    
    # Assigning a Call to a Name (line 348):
    
    # Assigning a Call to a Name (line 348):
    
    # Call to len(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'name' (line 348)
    name_129506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 21), 'name', False)
    # Processing the call keyword arguments (line 348)
    kwargs_129507 = {}
    # Getting the type of 'len' (line 348)
    len_129505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 17), 'len', False)
    # Calling len(args, kwargs) (line 348)
    len_call_result_129508 = invoke(stypy.reporting.localization.Localization(__file__, 348, 17), len_129505, *[name_129506], **kwargs_129507)
    
    # Assigning a type to the variable 'firstwidth' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'firstwidth', len_call_result_129508)
    
    # Assigning a Name to a Name (line 349):
    
    # Assigning a Name to a Name (line 349):
    # Getting the type of 'firstwidth' (line 349)
    firstwidth_129509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 8), 'firstwidth')
    # Assigning a type to the variable 'k' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'k', firstwidth_129509)
    
    # Assigning a Name to a Name (line 350):
    
    # Assigning a Name to a Name (line 350):
    # Getting the type of 'name' (line 350)
    name_129510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 13), 'name')
    # Assigning a type to the variable 'newstr' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'newstr', name_129510)
    
    # Assigning a Str to a Name (line 351):
    
    # Assigning a Str to a Name (line 351):
    str_129511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 13), 'str', ', ')
    # Assigning a type to the variable 'sepstr' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'sepstr', str_129511)
    
    # Assigning a Call to a Name (line 352):
    
    # Assigning a Call to a Name (line 352):
    
    # Call to split(...): (line 352)
    # Processing the call arguments (line 352)
    # Getting the type of 'sepstr' (line 352)
    sepstr_129514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 30), 'sepstr', False)
    # Processing the call keyword arguments (line 352)
    kwargs_129515 = {}
    # Getting the type of 'arguments' (line 352)
    arguments_129512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 14), 'arguments', False)
    # Obtaining the member 'split' of a type (line 352)
    split_129513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 14), arguments_129512, 'split')
    # Calling split(args, kwargs) (line 352)
    split_call_result_129516 = invoke(stypy.reporting.localization.Localization(__file__, 352, 14), split_129513, *[sepstr_129514], **kwargs_129515)
    
    # Assigning a type to the variable 'arglist' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'arglist', split_call_result_129516)
    
    # Getting the type of 'arglist' (line 353)
    arglist_129517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 20), 'arglist')
    # Testing the type of a for loop iterable (line 353)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 353, 4), arglist_129517)
    # Getting the type of the for loop variable (line 353)
    for_loop_var_129518 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 353, 4), arglist_129517)
    # Assigning a type to the variable 'argument' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'argument', for_loop_var_129518)
    # SSA begins for a for statement (line 353)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'k' (line 354)
    k_129519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 11), 'k')
    # Getting the type of 'firstwidth' (line 354)
    firstwidth_129520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 16), 'firstwidth')
    # Applying the binary operator '==' (line 354)
    result_eq_129521 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 11), '==', k_129519, firstwidth_129520)
    
    # Testing the type of an if condition (line 354)
    if_condition_129522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 354, 8), result_eq_129521)
    # Assigning a type to the variable 'if_condition_129522' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'if_condition_129522', if_condition_129522)
    # SSA begins for if statement (line 354)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 355):
    
    # Assigning a Str to a Name (line 355):
    str_129523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 21), 'str', '')
    # Assigning a type to the variable 'addstr' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'addstr', str_129523)
    # SSA branch for the else part of an if statement (line 354)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 357):
    
    # Assigning a Name to a Name (line 357):
    # Getting the type of 'sepstr' (line 357)
    sepstr_129524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 21), 'sepstr')
    # Assigning a type to the variable 'addstr' (line 357)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 12), 'addstr', sepstr_129524)
    # SSA join for if statement (line 354)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 358):
    
    # Assigning a BinOp to a Name (line 358):
    # Getting the type of 'k' (line 358)
    k_129525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'k')
    
    # Call to len(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'argument' (line 358)
    argument_129527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 20), 'argument', False)
    # Processing the call keyword arguments (line 358)
    kwargs_129528 = {}
    # Getting the type of 'len' (line 358)
    len_129526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 16), 'len', False)
    # Calling len(args, kwargs) (line 358)
    len_call_result_129529 = invoke(stypy.reporting.localization.Localization(__file__, 358, 16), len_129526, *[argument_129527], **kwargs_129528)
    
    # Applying the binary operator '+' (line 358)
    result_add_129530 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 12), '+', k_129525, len_call_result_129529)
    
    
    # Call to len(...): (line 358)
    # Processing the call arguments (line 358)
    # Getting the type of 'addstr' (line 358)
    addstr_129532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 36), 'addstr', False)
    # Processing the call keyword arguments (line 358)
    kwargs_129533 = {}
    # Getting the type of 'len' (line 358)
    len_129531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 32), 'len', False)
    # Calling len(args, kwargs) (line 358)
    len_call_result_129534 = invoke(stypy.reporting.localization.Localization(__file__, 358, 32), len_129531, *[addstr_129532], **kwargs_129533)
    
    # Applying the binary operator '+' (line 358)
    result_add_129535 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 30), '+', result_add_129530, len_call_result_129534)
    
    # Assigning a type to the variable 'k' (line 358)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 8), 'k', result_add_129535)
    
    
    # Getting the type of 'k' (line 359)
    k_129536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 11), 'k')
    # Getting the type of 'width' (line 359)
    width_129537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 15), 'width')
    # Applying the binary operator '>' (line 359)
    result_gt_129538 = python_operator(stypy.reporting.localization.Localization(__file__, 359, 11), '>', k_129536, width_129537)
    
    # Testing the type of an if condition (line 359)
    if_condition_129539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 359, 8), result_gt_129538)
    # Assigning a type to the variable 'if_condition_129539' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 8), 'if_condition_129539', if_condition_129539)
    # SSA begins for if statement (line 359)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 360):
    
    # Assigning a BinOp to a Name (line 360):
    # Getting the type of 'firstwidth' (line 360)
    firstwidth_129540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 16), 'firstwidth')
    int_129541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 29), 'int')
    # Applying the binary operator '+' (line 360)
    result_add_129542 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 16), '+', firstwidth_129540, int_129541)
    
    
    # Call to len(...): (line 360)
    # Processing the call arguments (line 360)
    # Getting the type of 'argument' (line 360)
    argument_129544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 37), 'argument', False)
    # Processing the call keyword arguments (line 360)
    kwargs_129545 = {}
    # Getting the type of 'len' (line 360)
    len_129543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 33), 'len', False)
    # Calling len(args, kwargs) (line 360)
    len_call_result_129546 = invoke(stypy.reporting.localization.Localization(__file__, 360, 33), len_129543, *[argument_129544], **kwargs_129545)
    
    # Applying the binary operator '+' (line 360)
    result_add_129547 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 31), '+', result_add_129542, len_call_result_129546)
    
    # Assigning a type to the variable 'k' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'k', result_add_129547)
    
    # Assigning a BinOp to a Name (line 361):
    
    # Assigning a BinOp to a Name (line 361):
    # Getting the type of 'newstr' (line 361)
    newstr_129548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 21), 'newstr')
    str_129549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 30), 'str', ',\n')
    # Applying the binary operator '+' (line 361)
    result_add_129550 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 21), '+', newstr_129548, str_129549)
    
    str_129551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 38), 'str', ' ')
    # Getting the type of 'firstwidth' (line 361)
    firstwidth_129552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 43), 'firstwidth')
    int_129553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 54), 'int')
    # Applying the binary operator '+' (line 361)
    result_add_129554 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 43), '+', firstwidth_129552, int_129553)
    
    # Applying the binary operator '*' (line 361)
    result_mul_129555 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 38), '*', str_129551, result_add_129554)
    
    # Applying the binary operator '+' (line 361)
    result_add_129556 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 36), '+', result_add_129550, result_mul_129555)
    
    # Getting the type of 'argument' (line 361)
    argument_129557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 59), 'argument')
    # Applying the binary operator '+' (line 361)
    result_add_129558 = python_operator(stypy.reporting.localization.Localization(__file__, 361, 57), '+', result_add_129556, argument_129557)
    
    # Assigning a type to the variable 'newstr' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 12), 'newstr', result_add_129558)
    # SSA branch for the else part of an if statement (line 359)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a BinOp to a Name (line 363):
    
    # Assigning a BinOp to a Name (line 363):
    # Getting the type of 'newstr' (line 363)
    newstr_129559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 21), 'newstr')
    # Getting the type of 'addstr' (line 363)
    addstr_129560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 30), 'addstr')
    # Applying the binary operator '+' (line 363)
    result_add_129561 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 21), '+', newstr_129559, addstr_129560)
    
    # Getting the type of 'argument' (line 363)
    argument_129562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 39), 'argument')
    # Applying the binary operator '+' (line 363)
    result_add_129563 = python_operator(stypy.reporting.localization.Localization(__file__, 363, 37), '+', result_add_129561, argument_129562)
    
    # Assigning a type to the variable 'newstr' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'newstr', result_add_129563)
    # SSA join for if statement (line 359)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'newstr' (line 364)
    newstr_129564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 11), 'newstr')
    # Assigning a type to the variable 'stypy_return_type' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'stypy_return_type', newstr_129564)
    
    # ################# End of '_split_line(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_split_line' in the type store
    # Getting the type of 'stypy_return_type' (line 347)
    stypy_return_type_129565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129565)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_split_line'
    return stypy_return_type_129565

# Assigning a type to the variable '_split_line' (line 347)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 0), '_split_line', _split_line)

# Assigning a Name to a Name (line 366):

# Assigning a Name to a Name (line 366):
# Getting the type of 'None' (line 366)
None_129566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 12), 'None')
# Assigning a type to the variable '_namedict' (line 366)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 0), '_namedict', None_129566)

# Assigning a Name to a Name (line 367):

# Assigning a Name to a Name (line 367):
# Getting the type of 'None' (line 367)
None_129567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 12), 'None')
# Assigning a type to the variable '_dictlist' (line 367)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 0), '_dictlist', None_129567)

@norecursion
def _makenamedict(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_129568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 25), 'str', 'numpy')
    defaults = [str_129568]
    # Create a new context for function '_makenamedict'
    module_type_store = module_type_store.open_function_context('_makenamedict', 371, 0, False)
    
    # Passed parameters checking function
    _makenamedict.stypy_localization = localization
    _makenamedict.stypy_type_of_self = None
    _makenamedict.stypy_type_store = module_type_store
    _makenamedict.stypy_function_name = '_makenamedict'
    _makenamedict.stypy_param_names_list = ['module']
    _makenamedict.stypy_varargs_param_name = None
    _makenamedict.stypy_kwargs_param_name = None
    _makenamedict.stypy_call_defaults = defaults
    _makenamedict.stypy_call_varargs = varargs
    _makenamedict.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_makenamedict', ['module'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_makenamedict', localization, ['module'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_makenamedict(...)' code ##################

    
    # Assigning a Call to a Name (line 372):
    
    # Assigning a Call to a Name (line 372):
    
    # Call to __import__(...): (line 372)
    # Processing the call arguments (line 372)
    # Getting the type of 'module' (line 372)
    module_129570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 24), 'module', False)
    
    # Call to globals(...): (line 372)
    # Processing the call keyword arguments (line 372)
    kwargs_129572 = {}
    # Getting the type of 'globals' (line 372)
    globals_129571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 32), 'globals', False)
    # Calling globals(args, kwargs) (line 372)
    globals_call_result_129573 = invoke(stypy.reporting.localization.Localization(__file__, 372, 32), globals_129571, *[], **kwargs_129572)
    
    
    # Call to locals(...): (line 372)
    # Processing the call keyword arguments (line 372)
    kwargs_129575 = {}
    # Getting the type of 'locals' (line 372)
    locals_129574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 43), 'locals', False)
    # Calling locals(args, kwargs) (line 372)
    locals_call_result_129576 = invoke(stypy.reporting.localization.Localization(__file__, 372, 43), locals_129574, *[], **kwargs_129575)
    
    
    # Obtaining an instance of the builtin type 'list' (line 372)
    list_129577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 372)
    
    # Processing the call keyword arguments (line 372)
    kwargs_129578 = {}
    # Getting the type of '__import__' (line 372)
    import___129569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 13), '__import__', False)
    # Calling __import__(args, kwargs) (line 372)
    import___call_result_129579 = invoke(stypy.reporting.localization.Localization(__file__, 372, 13), import___129569, *[module_129570, globals_call_result_129573, locals_call_result_129576, list_129577], **kwargs_129578)
    
    # Assigning a type to the variable 'module' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'module', import___call_result_129579)
    
    # Assigning a Dict to a Name (line 373):
    
    # Assigning a Dict to a Name (line 373):
    
    # Obtaining an instance of the builtin type 'dict' (line 373)
    dict_129580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 373)
    # Adding element type (key, value) (line 373)
    # Getting the type of 'module' (line 373)
    module_129581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 15), 'module')
    # Obtaining the member '__name__' of a type (line 373)
    name___129582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 15), module_129581, '__name__')
    # Getting the type of 'module' (line 373)
    module_129583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 31), 'module')
    # Obtaining the member '__dict__' of a type (line 373)
    dict___129584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 31), module_129583, '__dict__')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 373, 14), dict_129580, (name___129582, dict___129584))
    
    # Assigning a type to the variable 'thedict' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'thedict', dict_129580)
    
    # Assigning a List to a Name (line 374):
    
    # Assigning a List to a Name (line 374):
    
    # Obtaining an instance of the builtin type 'list' (line 374)
    list_129585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 374)
    # Adding element type (line 374)
    # Getting the type of 'module' (line 374)
    module_129586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 16), 'module')
    # Obtaining the member '__name__' of a type (line 374)
    name___129587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 16), module_129586, '__name__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 374, 15), list_129585, name___129587)
    
    # Assigning a type to the variable 'dictlist' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'dictlist', list_129585)
    
    # Assigning a List to a Name (line 375):
    
    # Assigning a List to a Name (line 375):
    
    # Obtaining an instance of the builtin type 'list' (line 375)
    list_129588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 375)
    # Adding element type (line 375)
    # Getting the type of 'module' (line 375)
    module_129589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 18), 'module')
    # Obtaining the member '__dict__' of a type (line 375)
    dict___129590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 18), module_129589, '__dict__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 17), list_129588, dict___129590)
    
    # Assigning a type to the variable 'totraverse' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'totraverse', list_129588)
    
    # Getting the type of 'True' (line 376)
    True_129591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 10), 'True')
    # Testing the type of an if condition (line 376)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 376, 4), True_129591)
    # SSA begins for while statement (line 376)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    
    # Call to len(...): (line 377)
    # Processing the call arguments (line 377)
    # Getting the type of 'totraverse' (line 377)
    totraverse_129593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 15), 'totraverse', False)
    # Processing the call keyword arguments (line 377)
    kwargs_129594 = {}
    # Getting the type of 'len' (line 377)
    len_129592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 11), 'len', False)
    # Calling len(args, kwargs) (line 377)
    len_call_result_129595 = invoke(stypy.reporting.localization.Localization(__file__, 377, 11), len_129592, *[totraverse_129593], **kwargs_129594)
    
    int_129596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 30), 'int')
    # Applying the binary operator '==' (line 377)
    result_eq_129597 = python_operator(stypy.reporting.localization.Localization(__file__, 377, 11), '==', len_call_result_129595, int_129596)
    
    # Testing the type of an if condition (line 377)
    if_condition_129598 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 377, 8), result_eq_129597)
    # Assigning a type to the variable 'if_condition_129598' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'if_condition_129598', if_condition_129598)
    # SSA begins for if statement (line 377)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 377)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 379):
    
    # Assigning a Call to a Name (line 379):
    
    # Call to pop(...): (line 379)
    # Processing the call arguments (line 379)
    int_129601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 34), 'int')
    # Processing the call keyword arguments (line 379)
    kwargs_129602 = {}
    # Getting the type of 'totraverse' (line 379)
    totraverse_129599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 19), 'totraverse', False)
    # Obtaining the member 'pop' of a type (line 379)
    pop_129600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 19), totraverse_129599, 'pop')
    # Calling pop(args, kwargs) (line 379)
    pop_call_result_129603 = invoke(stypy.reporting.localization.Localization(__file__, 379, 19), pop_129600, *[int_129601], **kwargs_129602)
    
    # Assigning a type to the variable 'thisdict' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'thisdict', pop_call_result_129603)
    
    
    # Call to keys(...): (line 380)
    # Processing the call keyword arguments (line 380)
    kwargs_129606 = {}
    # Getting the type of 'thisdict' (line 380)
    thisdict_129604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 17), 'thisdict', False)
    # Obtaining the member 'keys' of a type (line 380)
    keys_129605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 17), thisdict_129604, 'keys')
    # Calling keys(args, kwargs) (line 380)
    keys_call_result_129607 = invoke(stypy.reporting.localization.Localization(__file__, 380, 17), keys_129605, *[], **kwargs_129606)
    
    # Testing the type of a for loop iterable (line 380)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 380, 8), keys_call_result_129607)
    # Getting the type of the for loop variable (line 380)
    for_loop_var_129608 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 380, 8), keys_call_result_129607)
    # Assigning a type to the variable 'x' (line 380)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 380, 8), 'x', for_loop_var_129608)
    # SSA begins for a for statement (line 380)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to isinstance(...): (line 381)
    # Processing the call arguments (line 381)
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 381)
    x_129610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 35), 'x', False)
    # Getting the type of 'thisdict' (line 381)
    thisdict_129611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 26), 'thisdict', False)
    # Obtaining the member '__getitem__' of a type (line 381)
    getitem___129612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 26), thisdict_129611, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 381)
    subscript_call_result_129613 = invoke(stypy.reporting.localization.Localization(__file__, 381, 26), getitem___129612, x_129610)
    
    # Getting the type of 'types' (line 381)
    types_129614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 39), 'types', False)
    # Obtaining the member 'ModuleType' of a type (line 381)
    ModuleType_129615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 39), types_129614, 'ModuleType')
    # Processing the call keyword arguments (line 381)
    kwargs_129616 = {}
    # Getting the type of 'isinstance' (line 381)
    isinstance_129609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 381)
    isinstance_call_result_129617 = invoke(stypy.reporting.localization.Localization(__file__, 381, 15), isinstance_129609, *[subscript_call_result_129613, ModuleType_129615], **kwargs_129616)
    
    # Testing the type of an if condition (line 381)
    if_condition_129618 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 381, 12), isinstance_call_result_129617)
    # Assigning a type to the variable 'if_condition_129618' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'if_condition_129618', if_condition_129618)
    # SSA begins for if statement (line 381)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 382):
    
    # Assigning a Attribute to a Name (line 382):
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 382)
    x_129619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 35), 'x')
    # Getting the type of 'thisdict' (line 382)
    thisdict_129620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 26), 'thisdict')
    # Obtaining the member '__getitem__' of a type (line 382)
    getitem___129621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 26), thisdict_129620, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 382)
    subscript_call_result_129622 = invoke(stypy.reporting.localization.Localization(__file__, 382, 26), getitem___129621, x_129619)
    
    # Obtaining the member '__name__' of a type (line 382)
    name___129623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 26), subscript_call_result_129622, '__name__')
    # Assigning a type to the variable 'modname' (line 382)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 16), 'modname', name___129623)
    
    
    # Getting the type of 'modname' (line 383)
    modname_129624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 'modname')
    # Getting the type of 'dictlist' (line 383)
    dictlist_129625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 34), 'dictlist')
    # Applying the binary operator 'notin' (line 383)
    result_contains_129626 = python_operator(stypy.reporting.localization.Localization(__file__, 383, 19), 'notin', modname_129624, dictlist_129625)
    
    # Testing the type of an if condition (line 383)
    if_condition_129627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 16), result_contains_129626)
    # Assigning a type to the variable 'if_condition_129627' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 16), 'if_condition_129627', if_condition_129627)
    # SSA begins for if statement (line 383)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 384):
    
    # Assigning a Attribute to a Name (line 384):
    
    # Obtaining the type of the subscript
    # Getting the type of 'x' (line 384)
    x_129628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 39), 'x')
    # Getting the type of 'thisdict' (line 384)
    thisdict_129629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 30), 'thisdict')
    # Obtaining the member '__getitem__' of a type (line 384)
    getitem___129630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 30), thisdict_129629, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 384)
    subscript_call_result_129631 = invoke(stypy.reporting.localization.Localization(__file__, 384, 30), getitem___129630, x_129628)
    
    # Obtaining the member '__dict__' of a type (line 384)
    dict___129632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 30), subscript_call_result_129631, '__dict__')
    # Assigning a type to the variable 'moddict' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'moddict', dict___129632)
    
    # Call to append(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'modname' (line 385)
    modname_129635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 36), 'modname', False)
    # Processing the call keyword arguments (line 385)
    kwargs_129636 = {}
    # Getting the type of 'dictlist' (line 385)
    dictlist_129633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), 'dictlist', False)
    # Obtaining the member 'append' of a type (line 385)
    append_129634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 20), dictlist_129633, 'append')
    # Calling append(args, kwargs) (line 385)
    append_call_result_129637 = invoke(stypy.reporting.localization.Localization(__file__, 385, 20), append_129634, *[modname_129635], **kwargs_129636)
    
    
    # Call to append(...): (line 386)
    # Processing the call arguments (line 386)
    # Getting the type of 'moddict' (line 386)
    moddict_129640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 38), 'moddict', False)
    # Processing the call keyword arguments (line 386)
    kwargs_129641 = {}
    # Getting the type of 'totraverse' (line 386)
    totraverse_129638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 20), 'totraverse', False)
    # Obtaining the member 'append' of a type (line 386)
    append_129639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 386, 20), totraverse_129638, 'append')
    # Calling append(args, kwargs) (line 386)
    append_call_result_129642 = invoke(stypy.reporting.localization.Localization(__file__, 386, 20), append_129639, *[moddict_129640], **kwargs_129641)
    
    
    # Assigning a Name to a Subscript (line 387):
    
    # Assigning a Name to a Subscript (line 387):
    # Getting the type of 'moddict' (line 387)
    moddict_129643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 39), 'moddict')
    # Getting the type of 'thedict' (line 387)
    thedict_129644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 20), 'thedict')
    # Getting the type of 'modname' (line 387)
    modname_129645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 28), 'modname')
    # Storing an element on a container (line 387)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 387, 20), thedict_129644, (modname_129645, moddict_129643))
    # SSA join for if statement (line 383)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 381)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 376)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 388)
    tuple_129646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 388)
    # Adding element type (line 388)
    # Getting the type of 'thedict' (line 388)
    thedict_129647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 11), 'thedict')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 11), tuple_129646, thedict_129647)
    # Adding element type (line 388)
    # Getting the type of 'dictlist' (line 388)
    dictlist_129648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'dictlist')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 388, 11), tuple_129646, dictlist_129648)
    
    # Assigning a type to the variable 'stypy_return_type' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'stypy_return_type', tuple_129646)
    
    # ################# End of '_makenamedict(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_makenamedict' in the type store
    # Getting the type of 'stypy_return_type' (line 371)
    stypy_return_type_129649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129649)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_makenamedict'
    return stypy_return_type_129649

# Assigning a type to the variable '_makenamedict' (line 371)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), '_makenamedict', _makenamedict)

@norecursion
def _info(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'sys' (line 391)
    sys_129650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 22), 'sys')
    # Obtaining the member 'stdout' of a type (line 391)
    stdout_129651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 22), sys_129650, 'stdout')
    defaults = [stdout_129651]
    # Create a new context for function '_info'
    module_type_store = module_type_store.open_function_context('_info', 391, 0, False)
    
    # Passed parameters checking function
    _info.stypy_localization = localization
    _info.stypy_type_of_self = None
    _info.stypy_type_store = module_type_store
    _info.stypy_function_name = '_info'
    _info.stypy_param_names_list = ['obj', 'output']
    _info.stypy_varargs_param_name = None
    _info.stypy_kwargs_param_name = None
    _info.stypy_call_defaults = defaults
    _info.stypy_call_varargs = varargs
    _info.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_info', ['obj', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_info', localization, ['obj', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_info(...)' code ##################

    str_129652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 408, (-1)), 'str', 'Provide information about ndarray obj.\n\n    Parameters\n    ----------\n    obj: ndarray\n        Must be ndarray, not checked.\n    output:\n        Where printed output goes.\n\n    Notes\n    -----\n    Copied over from the numarray module prior to its removal.\n    Adapted somewhat as only numpy is an option now.\n\n    Called by info.\n\n    ')
    
    # Assigning a Str to a Name (line 409):
    
    # Assigning a Str to a Name (line 409):
    str_129653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 12), 'str', '')
    # Assigning a type to the variable 'extra' (line 409)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'extra', str_129653)
    
    # Assigning a Str to a Name (line 410):
    
    # Assigning a Str to a Name (line 410):
    str_129654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 10), 'str', '')
    # Assigning a type to the variable 'tic' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'tic', str_129654)
    
    # Assigning a Lambda to a Name (line 411):
    
    # Assigning a Lambda to a Name (line 411):

    @norecursion
    def _stypy_temp_lambda_31(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_31'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_31', 411, 9, True)
        # Passed parameters checking function
        _stypy_temp_lambda_31.stypy_localization = localization
        _stypy_temp_lambda_31.stypy_type_of_self = None
        _stypy_temp_lambda_31.stypy_type_store = module_type_store
        _stypy_temp_lambda_31.stypy_function_name = '_stypy_temp_lambda_31'
        _stypy_temp_lambda_31.stypy_param_names_list = ['x']
        _stypy_temp_lambda_31.stypy_varargs_param_name = None
        _stypy_temp_lambda_31.stypy_kwargs_param_name = None
        _stypy_temp_lambda_31.stypy_call_defaults = defaults
        _stypy_temp_lambda_31.stypy_call_varargs = varargs
        _stypy_temp_lambda_31.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_31', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_31', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        # Getting the type of 'x' (line 411)
        x_129655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 19), 'x')
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 9), 'stypy_return_type', x_129655)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_31' in the type store
        # Getting the type of 'stypy_return_type' (line 411)
        stypy_return_type_129656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 9), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_129656)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_31'
        return stypy_return_type_129656

    # Assigning a type to the variable '_stypy_temp_lambda_31' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 9), '_stypy_temp_lambda_31', _stypy_temp_lambda_31)
    # Getting the type of '_stypy_temp_lambda_31' (line 411)
    _stypy_temp_lambda_31_129657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 9), '_stypy_temp_lambda_31')
    # Assigning a type to the variable 'bp' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'bp', _stypy_temp_lambda_31_129657)
    
    # Assigning a Call to a Name (line 412):
    
    # Assigning a Call to a Name (line 412):
    
    # Call to getattr(...): (line 412)
    # Processing the call arguments (line 412)
    # Getting the type of 'obj' (line 412)
    obj_129659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 18), 'obj', False)
    str_129660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 412, 23), 'str', '__class__')
    
    # Call to type(...): (line 412)
    # Processing the call arguments (line 412)
    # Getting the type of 'obj' (line 412)
    obj_129662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 41), 'obj', False)
    # Processing the call keyword arguments (line 412)
    kwargs_129663 = {}
    # Getting the type of 'type' (line 412)
    type_129661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 36), 'type', False)
    # Calling type(args, kwargs) (line 412)
    type_call_result_129664 = invoke(stypy.reporting.localization.Localization(__file__, 412, 36), type_129661, *[obj_129662], **kwargs_129663)
    
    # Processing the call keyword arguments (line 412)
    kwargs_129665 = {}
    # Getting the type of 'getattr' (line 412)
    getattr_129658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 10), 'getattr', False)
    # Calling getattr(args, kwargs) (line 412)
    getattr_call_result_129666 = invoke(stypy.reporting.localization.Localization(__file__, 412, 10), getattr_129658, *[obj_129659, str_129660, type_call_result_129664], **kwargs_129665)
    
    # Assigning a type to the variable 'cls' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'cls', getattr_call_result_129666)
    
    # Assigning a Call to a Name (line 413):
    
    # Assigning a Call to a Name (line 413):
    
    # Call to getattr(...): (line 413)
    # Processing the call arguments (line 413)
    # Getting the type of 'cls' (line 413)
    cls_129668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 17), 'cls', False)
    str_129669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 22), 'str', '__name__')
    # Getting the type of 'cls' (line 413)
    cls_129670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 34), 'cls', False)
    # Processing the call keyword arguments (line 413)
    kwargs_129671 = {}
    # Getting the type of 'getattr' (line 413)
    getattr_129667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 9), 'getattr', False)
    # Calling getattr(args, kwargs) (line 413)
    getattr_call_result_129672 = invoke(stypy.reporting.localization.Localization(__file__, 413, 9), getattr_129667, *[cls_129668, str_129669, cls_129670], **kwargs_129671)
    
    # Assigning a type to the variable 'nm' (line 413)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 4), 'nm', getattr_call_result_129672)
    
    # Assigning a Attribute to a Name (line 414):
    
    # Assigning a Attribute to a Name (line 414):
    # Getting the type of 'obj' (line 414)
    obj_129673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 14), 'obj')
    # Obtaining the member 'strides' of a type (line 414)
    strides_129674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 14), obj_129673, 'strides')
    # Assigning a type to the variable 'strides' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'strides', strides_129674)
    
    # Assigning a Attribute to a Name (line 415):
    
    # Assigning a Attribute to a Name (line 415):
    # Getting the type of 'obj' (line 415)
    obj_129675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 13), 'obj')
    # Obtaining the member 'dtype' of a type (line 415)
    dtype_129676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 13), obj_129675, 'dtype')
    # Obtaining the member 'byteorder' of a type (line 415)
    byteorder_129677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 13), dtype_129676, 'byteorder')
    # Assigning a type to the variable 'endian' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'endian', byteorder_129677)
    
    # Call to print(...): (line 417)
    # Processing the call arguments (line 417)
    str_129679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 417, 10), 'str', 'class: ')
    # Getting the type of 'nm' (line 417)
    nm_129680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 21), 'nm', False)
    # Processing the call keyword arguments (line 417)
    # Getting the type of 'output' (line 417)
    output_129681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 30), 'output', False)
    keyword_129682 = output_129681
    kwargs_129683 = {'file': keyword_129682}
    # Getting the type of 'print' (line 417)
    print_129678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'print', False)
    # Calling print(args, kwargs) (line 417)
    print_call_result_129684 = invoke(stypy.reporting.localization.Localization(__file__, 417, 4), print_129678, *[str_129679, nm_129680], **kwargs_129683)
    
    
    # Call to print(...): (line 418)
    # Processing the call arguments (line 418)
    str_129686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 10), 'str', 'shape: ')
    # Getting the type of 'obj' (line 418)
    obj_129687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 21), 'obj', False)
    # Obtaining the member 'shape' of a type (line 418)
    shape_129688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 21), obj_129687, 'shape')
    # Processing the call keyword arguments (line 418)
    # Getting the type of 'output' (line 418)
    output_129689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 37), 'output', False)
    keyword_129690 = output_129689
    kwargs_129691 = {'file': keyword_129690}
    # Getting the type of 'print' (line 418)
    print_129685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'print', False)
    # Calling print(args, kwargs) (line 418)
    print_call_result_129692 = invoke(stypy.reporting.localization.Localization(__file__, 418, 4), print_129685, *[str_129686, shape_129688], **kwargs_129691)
    
    
    # Call to print(...): (line 419)
    # Processing the call arguments (line 419)
    str_129694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 10), 'str', 'strides: ')
    # Getting the type of 'strides' (line 419)
    strides_129695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 23), 'strides', False)
    # Processing the call keyword arguments (line 419)
    # Getting the type of 'output' (line 419)
    output_129696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 37), 'output', False)
    keyword_129697 = output_129696
    kwargs_129698 = {'file': keyword_129697}
    # Getting the type of 'print' (line 419)
    print_129693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'print', False)
    # Calling print(args, kwargs) (line 419)
    print_call_result_129699 = invoke(stypy.reporting.localization.Localization(__file__, 419, 4), print_129693, *[str_129694, strides_129695], **kwargs_129698)
    
    
    # Call to print(...): (line 420)
    # Processing the call arguments (line 420)
    str_129701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 10), 'str', 'itemsize: ')
    # Getting the type of 'obj' (line 420)
    obj_129702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 24), 'obj', False)
    # Obtaining the member 'itemsize' of a type (line 420)
    itemsize_129703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 24), obj_129702, 'itemsize')
    # Processing the call keyword arguments (line 420)
    # Getting the type of 'output' (line 420)
    output_129704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 43), 'output', False)
    keyword_129705 = output_129704
    kwargs_129706 = {'file': keyword_129705}
    # Getting the type of 'print' (line 420)
    print_129700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'print', False)
    # Calling print(args, kwargs) (line 420)
    print_call_result_129707 = invoke(stypy.reporting.localization.Localization(__file__, 420, 4), print_129700, *[str_129701, itemsize_129703], **kwargs_129706)
    
    
    # Call to print(...): (line 421)
    # Processing the call arguments (line 421)
    str_129709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 10), 'str', 'aligned: ')
    
    # Call to bp(...): (line 421)
    # Processing the call arguments (line 421)
    # Getting the type of 'obj' (line 421)
    obj_129711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 26), 'obj', False)
    # Obtaining the member 'flags' of a type (line 421)
    flags_129712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 26), obj_129711, 'flags')
    # Obtaining the member 'aligned' of a type (line 421)
    aligned_129713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 26), flags_129712, 'aligned')
    # Processing the call keyword arguments (line 421)
    kwargs_129714 = {}
    # Getting the type of 'bp' (line 421)
    bp_129710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 23), 'bp', False)
    # Calling bp(args, kwargs) (line 421)
    bp_call_result_129715 = invoke(stypy.reporting.localization.Localization(__file__, 421, 23), bp_129710, *[aligned_129713], **kwargs_129714)
    
    # Processing the call keyword arguments (line 421)
    # Getting the type of 'output' (line 421)
    output_129716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 51), 'output', False)
    keyword_129717 = output_129716
    kwargs_129718 = {'file': keyword_129717}
    # Getting the type of 'print' (line 421)
    print_129708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 4), 'print', False)
    # Calling print(args, kwargs) (line 421)
    print_call_result_129719 = invoke(stypy.reporting.localization.Localization(__file__, 421, 4), print_129708, *[str_129709, bp_call_result_129715], **kwargs_129718)
    
    
    # Call to print(...): (line 422)
    # Processing the call arguments (line 422)
    str_129721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 10), 'str', 'contiguous: ')
    
    # Call to bp(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'obj' (line 422)
    obj_129723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 29), 'obj', False)
    # Obtaining the member 'flags' of a type (line 422)
    flags_129724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 29), obj_129723, 'flags')
    # Obtaining the member 'contiguous' of a type (line 422)
    contiguous_129725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 29), flags_129724, 'contiguous')
    # Processing the call keyword arguments (line 422)
    kwargs_129726 = {}
    # Getting the type of 'bp' (line 422)
    bp_129722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 26), 'bp', False)
    # Calling bp(args, kwargs) (line 422)
    bp_call_result_129727 = invoke(stypy.reporting.localization.Localization(__file__, 422, 26), bp_129722, *[contiguous_129725], **kwargs_129726)
    
    # Processing the call keyword arguments (line 422)
    # Getting the type of 'output' (line 422)
    output_129728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 57), 'output', False)
    keyword_129729 = output_129728
    kwargs_129730 = {'file': keyword_129729}
    # Getting the type of 'print' (line 422)
    print_129720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'print', False)
    # Calling print(args, kwargs) (line 422)
    print_call_result_129731 = invoke(stypy.reporting.localization.Localization(__file__, 422, 4), print_129720, *[str_129721, bp_call_result_129727], **kwargs_129730)
    
    
    # Call to print(...): (line 423)
    # Processing the call arguments (line 423)
    str_129733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 10), 'str', 'fortran: ')
    # Getting the type of 'obj' (line 423)
    obj_129734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 23), 'obj', False)
    # Obtaining the member 'flags' of a type (line 423)
    flags_129735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 23), obj_129734, 'flags')
    # Obtaining the member 'fortran' of a type (line 423)
    fortran_129736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 23), flags_129735, 'fortran')
    # Processing the call keyword arguments (line 423)
    # Getting the type of 'output' (line 423)
    output_129737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 47), 'output', False)
    keyword_129738 = output_129737
    kwargs_129739 = {'file': keyword_129738}
    # Getting the type of 'print' (line 423)
    print_129732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'print', False)
    # Calling print(args, kwargs) (line 423)
    print_call_result_129740 = invoke(stypy.reporting.localization.Localization(__file__, 423, 4), print_129732, *[str_129733, fortran_129736], **kwargs_129739)
    
    
    # Call to print(...): (line 424)
    # Processing the call arguments (line 424)
    str_129742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 8), 'str', 'data pointer: %s%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 425)
    tuple_129743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 425)
    # Adding element type (line 425)
    
    # Call to hex(...): (line 425)
    # Processing the call arguments (line 425)
    # Getting the type of 'obj' (line 425)
    obj_129745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 36), 'obj', False)
    # Obtaining the member 'ctypes' of a type (line 425)
    ctypes_129746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 36), obj_129745, 'ctypes')
    # Obtaining the member '_as_parameter_' of a type (line 425)
    _as_parameter__129747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 36), ctypes_129746, '_as_parameter_')
    # Obtaining the member 'value' of a type (line 425)
    value_129748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 36), _as_parameter__129747, 'value')
    # Processing the call keyword arguments (line 425)
    kwargs_129749 = {}
    # Getting the type of 'hex' (line 425)
    hex_129744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 32), 'hex', False)
    # Calling hex(args, kwargs) (line 425)
    hex_call_result_129750 = invoke(stypy.reporting.localization.Localization(__file__, 425, 32), hex_129744, *[value_129748], **kwargs_129749)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 32), tuple_129743, hex_call_result_129750)
    # Adding element type (line 425)
    # Getting the type of 'extra' (line 425)
    extra_129751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 70), 'extra', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 425, 32), tuple_129743, extra_129751)
    
    # Applying the binary operator '%' (line 425)
    result_mod_129752 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 8), '%', str_129742, tuple_129743)
    
    # Processing the call keyword arguments (line 424)
    # Getting the type of 'output' (line 426)
    output_129753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 13), 'output', False)
    keyword_129754 = output_129753
    kwargs_129755 = {'file': keyword_129754}
    # Getting the type of 'print' (line 424)
    print_129741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'print', False)
    # Calling print(args, kwargs) (line 424)
    print_call_result_129756 = invoke(stypy.reporting.localization.Localization(__file__, 424, 4), print_129741, *[result_mod_129752], **kwargs_129755)
    
    
    # Call to print(...): (line 428)
    # Processing the call arguments (line 428)
    str_129758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 10), 'str', 'byteorder: ')
    # Processing the call keyword arguments (line 428)
    str_129759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 428, 29), 'str', ' ')
    keyword_129760 = str_129759
    # Getting the type of 'output' (line 428)
    output_129761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 39), 'output', False)
    keyword_129762 = output_129761
    kwargs_129763 = {'end': keyword_129760, 'file': keyword_129762}
    # Getting the type of 'print' (line 428)
    print_129757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'print', False)
    # Calling print(args, kwargs) (line 428)
    print_call_result_129764 = invoke(stypy.reporting.localization.Localization(__file__, 428, 4), print_129757, *[str_129758], **kwargs_129763)
    
    
    
    # Getting the type of 'endian' (line 429)
    endian_129765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 7), 'endian')
    
    # Obtaining an instance of the builtin type 'list' (line 429)
    list_129766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 429)
    # Adding element type (line 429)
    str_129767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 18), 'str', '|')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 17), list_129766, str_129767)
    # Adding element type (line 429)
    str_129768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 23), 'str', '=')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 429, 17), list_129766, str_129768)
    
    # Applying the binary operator 'in' (line 429)
    result_contains_129769 = python_operator(stypy.reporting.localization.Localization(__file__, 429, 7), 'in', endian_129765, list_129766)
    
    # Testing the type of an if condition (line 429)
    if_condition_129770 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 429, 4), result_contains_129769)
    # Assigning a type to the variable 'if_condition_129770' (line 429)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 4), 'if_condition_129770', if_condition_129770)
    # SSA begins for if statement (line 429)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 430)
    # Processing the call arguments (line 430)
    str_129772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 14), 'str', '%s%s%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 430)
    tuple_129773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 430)
    # Adding element type (line 430)
    # Getting the type of 'tic' (line 430)
    tic_129774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 26), 'tic', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 26), tuple_129773, tic_129774)
    # Adding element type (line 430)
    # Getting the type of 'sys' (line 430)
    sys_129775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 31), 'sys', False)
    # Obtaining the member 'byteorder' of a type (line 430)
    byteorder_129776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 31), sys_129775, 'byteorder')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 26), tuple_129773, byteorder_129776)
    # Adding element type (line 430)
    # Getting the type of 'tic' (line 430)
    tic_129777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 46), 'tic', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 430, 26), tuple_129773, tic_129777)
    
    # Applying the binary operator '%' (line 430)
    result_mod_129778 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 14), '%', str_129772, tuple_129773)
    
    # Processing the call keyword arguments (line 430)
    # Getting the type of 'output' (line 430)
    output_129779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 57), 'output', False)
    keyword_129780 = output_129779
    kwargs_129781 = {'file': keyword_129780}
    # Getting the type of 'print' (line 430)
    print_129771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'print', False)
    # Calling print(args, kwargs) (line 430)
    print_call_result_129782 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), print_129771, *[result_mod_129778], **kwargs_129781)
    
    
    # Assigning a Name to a Name (line 431):
    
    # Assigning a Name to a Name (line 431):
    # Getting the type of 'False' (line 431)
    False_129783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 19), 'False')
    # Assigning a type to the variable 'byteswap' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'byteswap', False_129783)
    # SSA branch for the else part of an if statement (line 429)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'endian' (line 432)
    endian_129784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 9), 'endian')
    str_129785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 19), 'str', '>')
    # Applying the binary operator '==' (line 432)
    result_eq_129786 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 9), '==', endian_129784, str_129785)
    
    # Testing the type of an if condition (line 432)
    if_condition_129787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 432, 9), result_eq_129786)
    # Assigning a type to the variable 'if_condition_129787' (line 432)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 9), 'if_condition_129787', if_condition_129787)
    # SSA begins for if statement (line 432)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 433)
    # Processing the call arguments (line 433)
    str_129789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 14), 'str', '%sbig%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 433)
    tuple_129790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 433)
    # Adding element type (line 433)
    # Getting the type of 'tic' (line 433)
    tic_129791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 27), 'tic', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 27), tuple_129790, tic_129791)
    # Adding element type (line 433)
    # Getting the type of 'tic' (line 433)
    tic_129792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 32), 'tic', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 27), tuple_129790, tic_129792)
    
    # Applying the binary operator '%' (line 433)
    result_mod_129793 = python_operator(stypy.reporting.localization.Localization(__file__, 433, 14), '%', str_129789, tuple_129790)
    
    # Processing the call keyword arguments (line 433)
    # Getting the type of 'output' (line 433)
    output_129794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 43), 'output', False)
    keyword_129795 = output_129794
    kwargs_129796 = {'file': keyword_129795}
    # Getting the type of 'print' (line 433)
    print_129788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'print', False)
    # Calling print(args, kwargs) (line 433)
    print_call_result_129797 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), print_129788, *[result_mod_129793], **kwargs_129796)
    
    
    # Assigning a Compare to a Name (line 434):
    
    # Assigning a Compare to a Name (line 434):
    
    # Getting the type of 'sys' (line 434)
    sys_129798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 19), 'sys')
    # Obtaining the member 'byteorder' of a type (line 434)
    byteorder_129799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 19), sys_129798, 'byteorder')
    str_129800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 36), 'str', 'big')
    # Applying the binary operator '!=' (line 434)
    result_ne_129801 = python_operator(stypy.reporting.localization.Localization(__file__, 434, 19), '!=', byteorder_129799, str_129800)
    
    # Assigning a type to the variable 'byteswap' (line 434)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'byteswap', result_ne_129801)
    # SSA branch for the else part of an if statement (line 432)
    module_type_store.open_ssa_branch('else')
    
    # Call to print(...): (line 436)
    # Processing the call arguments (line 436)
    str_129803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 14), 'str', '%slittle%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 436)
    tuple_129804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 436, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 436)
    # Adding element type (line 436)
    # Getting the type of 'tic' (line 436)
    tic_129805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 30), 'tic', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 30), tuple_129804, tic_129805)
    # Adding element type (line 436)
    # Getting the type of 'tic' (line 436)
    tic_129806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 35), 'tic', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 436, 30), tuple_129804, tic_129806)
    
    # Applying the binary operator '%' (line 436)
    result_mod_129807 = python_operator(stypy.reporting.localization.Localization(__file__, 436, 14), '%', str_129803, tuple_129804)
    
    # Processing the call keyword arguments (line 436)
    # Getting the type of 'output' (line 436)
    output_129808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 46), 'output', False)
    keyword_129809 = output_129808
    kwargs_129810 = {'file': keyword_129809}
    # Getting the type of 'print' (line 436)
    print_129802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'print', False)
    # Calling print(args, kwargs) (line 436)
    print_call_result_129811 = invoke(stypy.reporting.localization.Localization(__file__, 436, 8), print_129802, *[result_mod_129807], **kwargs_129810)
    
    
    # Assigning a Compare to a Name (line 437):
    
    # Assigning a Compare to a Name (line 437):
    
    # Getting the type of 'sys' (line 437)
    sys_129812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 19), 'sys')
    # Obtaining the member 'byteorder' of a type (line 437)
    byteorder_129813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 19), sys_129812, 'byteorder')
    str_129814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 36), 'str', 'little')
    # Applying the binary operator '!=' (line 437)
    result_ne_129815 = python_operator(stypy.reporting.localization.Localization(__file__, 437, 19), '!=', byteorder_129813, str_129814)
    
    # Assigning a type to the variable 'byteswap' (line 437)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'byteswap', result_ne_129815)
    # SSA join for if statement (line 432)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 429)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 438)
    # Processing the call arguments (line 438)
    str_129817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 10), 'str', 'byteswap: ')
    
    # Call to bp(...): (line 438)
    # Processing the call arguments (line 438)
    # Getting the type of 'byteswap' (line 438)
    byteswap_129819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 27), 'byteswap', False)
    # Processing the call keyword arguments (line 438)
    kwargs_129820 = {}
    # Getting the type of 'bp' (line 438)
    bp_129818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 24), 'bp', False)
    # Calling bp(args, kwargs) (line 438)
    bp_call_result_129821 = invoke(stypy.reporting.localization.Localization(__file__, 438, 24), bp_129818, *[byteswap_129819], **kwargs_129820)
    
    # Processing the call keyword arguments (line 438)
    # Getting the type of 'output' (line 438)
    output_129822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 43), 'output', False)
    keyword_129823 = output_129822
    kwargs_129824 = {'file': keyword_129823}
    # Getting the type of 'print' (line 438)
    print_129816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 4), 'print', False)
    # Calling print(args, kwargs) (line 438)
    print_call_result_129825 = invoke(stypy.reporting.localization.Localization(__file__, 438, 4), print_129816, *[str_129817, bp_call_result_129821], **kwargs_129824)
    
    
    # Call to print(...): (line 439)
    # Processing the call arguments (line 439)
    str_129827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 10), 'str', 'type: %s')
    # Getting the type of 'obj' (line 439)
    obj_129828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 23), 'obj', False)
    # Obtaining the member 'dtype' of a type (line 439)
    dtype_129829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 23), obj_129828, 'dtype')
    # Applying the binary operator '%' (line 439)
    result_mod_129830 = python_operator(stypy.reporting.localization.Localization(__file__, 439, 10), '%', str_129827, dtype_129829)
    
    # Processing the call keyword arguments (line 439)
    # Getting the type of 'output' (line 439)
    output_129831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 39), 'output', False)
    keyword_129832 = output_129831
    kwargs_129833 = {'file': keyword_129832}
    # Getting the type of 'print' (line 439)
    print_129826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 4), 'print', False)
    # Calling print(args, kwargs) (line 439)
    print_call_result_129834 = invoke(stypy.reporting.localization.Localization(__file__, 439, 4), print_129826, *[result_mod_129830], **kwargs_129833)
    
    
    # ################# End of '_info(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_info' in the type store
    # Getting the type of 'stypy_return_type' (line 391)
    stypy_return_type_129835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129835)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_info'
    return stypy_return_type_129835

# Assigning a type to the variable '_info' (line 391)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 0), '_info', _info)

@norecursion
def info(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 442)
    None_129836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 16), 'None')
    int_129837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 31), 'int')
    # Getting the type of 'sys' (line 442)
    sys_129838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 42), 'sys')
    # Obtaining the member 'stdout' of a type (line 442)
    stdout_129839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 42), sys_129838, 'stdout')
    str_129840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 63), 'str', 'numpy')
    defaults = [None_129836, int_129837, stdout_129839, str_129840]
    # Create a new context for function 'info'
    module_type_store = module_type_store.open_function_context('info', 442, 0, False)
    
    # Passed parameters checking function
    info.stypy_localization = localization
    info.stypy_type_of_self = None
    info.stypy_type_store = module_type_store
    info.stypy_function_name = 'info'
    info.stypy_param_names_list = ['object', 'maxwidth', 'output', 'toplevel']
    info.stypy_varargs_param_name = None
    info.stypy_kwargs_param_name = None
    info.stypy_call_defaults = defaults
    info.stypy_call_varargs = varargs
    info.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'info', ['object', 'maxwidth', 'output', 'toplevel'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'info', localization, ['object', 'maxwidth', 'output', 'toplevel'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'info(...)' code ##################

    str_129841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, (-1)), 'str', "\n    Get help information for a function, class, or module.\n\n    Parameters\n    ----------\n    object : object or str, optional\n        Input object or name to get information about. If `object` is a\n        numpy object, its docstring is given. If it is a string, available\n        modules are searched for matching objects.  If None, information\n        about `info` itself is returned.\n    maxwidth : int, optional\n        Printing width.\n    output : file like object, optional\n        File like object that the output is written to, default is\n        ``stdout``.  The object has to be opened in 'w' or 'a' mode.\n    toplevel : str, optional\n        Start search at this level.\n\n    See Also\n    --------\n    source, lookfor\n\n    Notes\n    -----\n    When used interactively with an object, ``np.info(obj)`` is equivalent\n    to ``help(obj)`` on the Python prompt or ``obj?`` on the IPython\n    prompt.\n\n    Examples\n    --------\n    >>> np.info(np.polyval) # doctest: +SKIP\n       polyval(p, x)\n         Evaluate the polynomial p at x.\n         ...\n\n    When using a string for `object` it is possible to get multiple results.\n\n    >>> np.info('fft') # doctest: +SKIP\n         *** Found in numpy ***\n    Core FFT routines\n    ...\n         *** Found in numpy.fft ***\n     fft(a, n=None, axis=-1)\n    ...\n         *** Repeat reference found in numpy.fft.fftpack ***\n         *** Total of 3 references found. ***\n\n    ")
    # Marking variables as global (line 491)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 491, 4), '_namedict')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 491, 4), '_dictlist')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 493, 4))
    
    # 'import pydoc' statement (line 493)
    import pydoc

    import_module(stypy.reporting.localization.Localization(__file__, 493, 4), 'pydoc', pydoc, module_type_store)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 494, 4))
    
    # 'import inspect' statement (line 494)
    import inspect

    import_module(stypy.reporting.localization.Localization(__file__, 494, 4), 'inspect', inspect, module_type_store)
    
    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 496)
    # Processing the call arguments (line 496)
    # Getting the type of 'object' (line 496)
    object_129843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 16), 'object', False)
    str_129844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 24), 'str', '_ppimport_importer')
    # Processing the call keyword arguments (line 496)
    kwargs_129845 = {}
    # Getting the type of 'hasattr' (line 496)
    hasattr_129842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 8), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 496)
    hasattr_call_result_129846 = invoke(stypy.reporting.localization.Localization(__file__, 496, 8), hasattr_129842, *[object_129843, str_129844], **kwargs_129845)
    
    
    # Call to hasattr(...): (line 497)
    # Processing the call arguments (line 497)
    # Getting the type of 'object' (line 497)
    object_129848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 19), 'object', False)
    str_129849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 27), 'str', '_ppimport_module')
    # Processing the call keyword arguments (line 497)
    kwargs_129850 = {}
    # Getting the type of 'hasattr' (line 497)
    hasattr_129847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 11), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 497)
    hasattr_call_result_129851 = invoke(stypy.reporting.localization.Localization(__file__, 497, 11), hasattr_129847, *[object_129848, str_129849], **kwargs_129850)
    
    # Applying the binary operator 'or' (line 496)
    result_or_keyword_129852 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 8), 'or', hasattr_call_result_129846, hasattr_call_result_129851)
    
    # Testing the type of an if condition (line 496)
    if_condition_129853 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 496, 4), result_or_keyword_129852)
    # Assigning a type to the variable 'if_condition_129853' (line 496)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'if_condition_129853', if_condition_129853)
    # SSA begins for if statement (line 496)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 498):
    
    # Assigning a Attribute to a Name (line 498):
    # Getting the type of 'object' (line 498)
    object_129854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 17), 'object')
    # Obtaining the member '_ppimport_module' of a type (line 498)
    _ppimport_module_129855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 17), object_129854, '_ppimport_module')
    # Assigning a type to the variable 'object' (line 498)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'object', _ppimport_module_129855)
    # SSA branch for the else part of an if statement (line 496)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 499)
    str_129856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 25), 'str', '_ppimport_attr')
    # Getting the type of 'object' (line 499)
    object_129857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 17), 'object')
    
    (may_be_129858, more_types_in_union_129859) = may_provide_member(str_129856, object_129857)

    if may_be_129858:

        if more_types_in_union_129859:
            # Runtime conditional SSA (line 499)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'object' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 9), 'object', remove_not_member_provider_from_union(object_129857, '_ppimport_attr'))
        
        # Assigning a Attribute to a Name (line 500):
        
        # Assigning a Attribute to a Name (line 500):
        # Getting the type of 'object' (line 500)
        object_129860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 17), 'object')
        # Obtaining the member '_ppimport_attr' of a type (line 500)
        _ppimport_attr_129861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 17), object_129860, '_ppimport_attr')
        # Assigning a type to the variable 'object' (line 500)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 500, 8), 'object', _ppimport_attr_129861)

        if more_types_in_union_129859:
            # SSA join for if statement (line 499)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 496)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 502)
    # Getting the type of 'object' (line 502)
    object_129862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 7), 'object')
    # Getting the type of 'None' (line 502)
    None_129863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 17), 'None')
    
    (may_be_129864, more_types_in_union_129865) = may_be_none(object_129862, None_129863)

    if may_be_129864:

        if more_types_in_union_129865:
            # Runtime conditional SSA (line 502)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to info(...): (line 503)
        # Processing the call arguments (line 503)
        # Getting the type of 'info' (line 503)
        info_129867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 13), 'info', False)
        # Processing the call keyword arguments (line 503)
        kwargs_129868 = {}
        # Getting the type of 'info' (line 503)
        info_129866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'info', False)
        # Calling info(args, kwargs) (line 503)
        info_call_result_129869 = invoke(stypy.reporting.localization.Localization(__file__, 503, 8), info_129866, *[info_129867], **kwargs_129868)
        

        if more_types_in_union_129865:
            # Runtime conditional SSA for else branch (line 502)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_129864) or more_types_in_union_129865):
        
        
        # Call to isinstance(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 'object' (line 504)
        object_129871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 20), 'object', False)
        # Getting the type of 'ndarray' (line 504)
        ndarray_129872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 28), 'ndarray', False)
        # Processing the call keyword arguments (line 504)
        kwargs_129873 = {}
        # Getting the type of 'isinstance' (line 504)
        isinstance_129870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 9), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 504)
        isinstance_call_result_129874 = invoke(stypy.reporting.localization.Localization(__file__, 504, 9), isinstance_129870, *[object_129871, ndarray_129872], **kwargs_129873)
        
        # Testing the type of an if condition (line 504)
        if_condition_129875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 504, 9), isinstance_call_result_129874)
        # Assigning a type to the variable 'if_condition_129875' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 9), 'if_condition_129875', if_condition_129875)
        # SSA begins for if statement (line 504)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to _info(...): (line 505)
        # Processing the call arguments (line 505)
        # Getting the type of 'object' (line 505)
        object_129877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 14), 'object', False)
        # Processing the call keyword arguments (line 505)
        # Getting the type of 'output' (line 505)
        output_129878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 29), 'output', False)
        keyword_129879 = output_129878
        kwargs_129880 = {'output': keyword_129879}
        # Getting the type of '_info' (line 505)
        _info_129876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), '_info', False)
        # Calling _info(args, kwargs) (line 505)
        _info_call_result_129881 = invoke(stypy.reporting.localization.Localization(__file__, 505, 8), _info_129876, *[object_129877], **kwargs_129880)
        
        # SSA branch for the else part of an if statement (line 504)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 506)
        # Getting the type of 'str' (line 506)
        str_129882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 28), 'str')
        # Getting the type of 'object' (line 506)
        object_129883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 20), 'object')
        
        (may_be_129884, more_types_in_union_129885) = may_be_subtype(str_129882, object_129883)

        if may_be_129884:

            if more_types_in_union_129885:
                # Runtime conditional SSA (line 506)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'object' (line 506)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 9), 'object', remove_not_subtype_from_union(object_129883, str))
            
            # Type idiom detected: calculating its left and rigth part (line 507)
            # Getting the type of '_namedict' (line 507)
            _namedict_129886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 11), '_namedict')
            # Getting the type of 'None' (line 507)
            None_129887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 24), 'None')
            
            (may_be_129888, more_types_in_union_129889) = may_be_none(_namedict_129886, None_129887)

            if may_be_129888:

                if more_types_in_union_129889:
                    # Runtime conditional SSA (line 507)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Tuple (line 508):
                
                # Assigning a Call to a Name:
                
                # Call to _makenamedict(...): (line 508)
                # Processing the call arguments (line 508)
                # Getting the type of 'toplevel' (line 508)
                toplevel_129891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 49), 'toplevel', False)
                # Processing the call keyword arguments (line 508)
                kwargs_129892 = {}
                # Getting the type of '_makenamedict' (line 508)
                _makenamedict_129890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 35), '_makenamedict', False)
                # Calling _makenamedict(args, kwargs) (line 508)
                _makenamedict_call_result_129893 = invoke(stypy.reporting.localization.Localization(__file__, 508, 35), _makenamedict_129890, *[toplevel_129891], **kwargs_129892)
                
                # Assigning a type to the variable 'call_assignment_128922' (line 508)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'call_assignment_128922', _makenamedict_call_result_129893)
                
                # Assigning a Call to a Name (line 508):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_129896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 12), 'int')
                # Processing the call keyword arguments
                kwargs_129897 = {}
                # Getting the type of 'call_assignment_128922' (line 508)
                call_assignment_128922_129894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'call_assignment_128922', False)
                # Obtaining the member '__getitem__' of a type (line 508)
                getitem___129895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 12), call_assignment_128922_129894, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_129898 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129895, *[int_129896], **kwargs_129897)
                
                # Assigning a type to the variable 'call_assignment_128923' (line 508)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'call_assignment_128923', getitem___call_result_129898)
                
                # Assigning a Name to a Name (line 508):
                # Getting the type of 'call_assignment_128923' (line 508)
                call_assignment_128923_129899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'call_assignment_128923')
                # Assigning a type to the variable '_namedict' (line 508)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), '_namedict', call_assignment_128923_129899)
                
                # Assigning a Call to a Name (line 508):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_129902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 12), 'int')
                # Processing the call keyword arguments
                kwargs_129903 = {}
                # Getting the type of 'call_assignment_128922' (line 508)
                call_assignment_128922_129900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'call_assignment_128922', False)
                # Obtaining the member '__getitem__' of a type (line 508)
                getitem___129901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 12), call_assignment_128922_129900, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_129904 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___129901, *[int_129902], **kwargs_129903)
                
                # Assigning a type to the variable 'call_assignment_128924' (line 508)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'call_assignment_128924', getitem___call_result_129904)
                
                # Assigning a Name to a Name (line 508):
                # Getting the type of 'call_assignment_128924' (line 508)
                call_assignment_128924_129905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 12), 'call_assignment_128924')
                # Assigning a type to the variable '_dictlist' (line 508)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 23), '_dictlist', call_assignment_128924_129905)

                if more_types_in_union_129889:
                    # SSA join for if statement (line 507)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Num to a Name (line 509):
            
            # Assigning a Num to a Name (line 509):
            int_129906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 509, 19), 'int')
            # Assigning a type to the variable 'numfound' (line 509)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'numfound', int_129906)
            
            # Assigning a List to a Name (line 510):
            
            # Assigning a List to a Name (line 510):
            
            # Obtaining an instance of the builtin type 'list' (line 510)
            list_129907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 18), 'list')
            # Adding type elements to the builtin type 'list' instance (line 510)
            
            # Assigning a type to the variable 'objlist' (line 510)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'objlist', list_129907)
            
            # Getting the type of '_dictlist' (line 511)
            _dictlist_129908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 23), '_dictlist')
            # Testing the type of a for loop iterable (line 511)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 511, 8), _dictlist_129908)
            # Getting the type of the for loop variable (line 511)
            for_loop_var_129909 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 511, 8), _dictlist_129908)
            # Assigning a type to the variable 'namestr' (line 511)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'namestr', for_loop_var_129909)
            # SSA begins for a for statement (line 511)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # SSA begins for try-except statement (line 512)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Subscript to a Name (line 513):
            
            # Assigning a Subscript to a Name (line 513):
            
            # Obtaining the type of the subscript
            # Getting the type of 'object' (line 513)
            object_129910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 41), 'object')
            
            # Obtaining the type of the subscript
            # Getting the type of 'namestr' (line 513)
            namestr_129911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 32), 'namestr')
            # Getting the type of '_namedict' (line 513)
            _namedict_129912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 22), '_namedict')
            # Obtaining the member '__getitem__' of a type (line 513)
            getitem___129913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 22), _namedict_129912, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 513)
            subscript_call_result_129914 = invoke(stypy.reporting.localization.Localization(__file__, 513, 22), getitem___129913, namestr_129911)
            
            # Obtaining the member '__getitem__' of a type (line 513)
            getitem___129915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 22), subscript_call_result_129914, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 513)
            subscript_call_result_129916 = invoke(stypy.reporting.localization.Localization(__file__, 513, 22), getitem___129915, object_129910)
            
            # Assigning a type to the variable 'obj' (line 513)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 16), 'obj', subscript_call_result_129916)
            
            
            
            # Call to id(...): (line 514)
            # Processing the call arguments (line 514)
            # Getting the type of 'obj' (line 514)
            obj_129918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 22), 'obj', False)
            # Processing the call keyword arguments (line 514)
            kwargs_129919 = {}
            # Getting the type of 'id' (line 514)
            id_129917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 19), 'id', False)
            # Calling id(args, kwargs) (line 514)
            id_call_result_129920 = invoke(stypy.reporting.localization.Localization(__file__, 514, 19), id_129917, *[obj_129918], **kwargs_129919)
            
            # Getting the type of 'objlist' (line 514)
            objlist_129921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 30), 'objlist')
            # Applying the binary operator 'in' (line 514)
            result_contains_129922 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 19), 'in', id_call_result_129920, objlist_129921)
            
            # Testing the type of an if condition (line 514)
            if_condition_129923 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 514, 16), result_contains_129922)
            # Assigning a type to the variable 'if_condition_129923' (line 514)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 16), 'if_condition_129923', if_condition_129923)
            # SSA begins for if statement (line 514)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to print(...): (line 515)
            # Processing the call arguments (line 515)
            str_129925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 515, 26), 'str', '\n     *** Repeat reference found in %s *** ')
            # Getting the type of 'namestr' (line 516)
            namestr_129926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 68), 'namestr', False)
            # Applying the binary operator '%' (line 515)
            result_mod_129927 = python_operator(stypy.reporting.localization.Localization(__file__, 515, 26), '%', str_129925, namestr_129926)
            
            # Processing the call keyword arguments (line 515)
            # Getting the type of 'output' (line 517)
            output_129928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 31), 'output', False)
            keyword_129929 = output_129928
            kwargs_129930 = {'file': keyword_129929}
            # Getting the type of 'print' (line 515)
            print_129924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 20), 'print', False)
            # Calling print(args, kwargs) (line 515)
            print_call_result_129931 = invoke(stypy.reporting.localization.Localization(__file__, 515, 20), print_129924, *[result_mod_129927], **kwargs_129930)
            
            # SSA branch for the else part of an if statement (line 514)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 520)
            # Processing the call arguments (line 520)
            
            # Call to id(...): (line 520)
            # Processing the call arguments (line 520)
            # Getting the type of 'obj' (line 520)
            obj_129935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 38), 'obj', False)
            # Processing the call keyword arguments (line 520)
            kwargs_129936 = {}
            # Getting the type of 'id' (line 520)
            id_129934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 35), 'id', False)
            # Calling id(args, kwargs) (line 520)
            id_call_result_129937 = invoke(stypy.reporting.localization.Localization(__file__, 520, 35), id_129934, *[obj_129935], **kwargs_129936)
            
            # Processing the call keyword arguments (line 520)
            kwargs_129938 = {}
            # Getting the type of 'objlist' (line 520)
            objlist_129932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 20), 'objlist', False)
            # Obtaining the member 'append' of a type (line 520)
            append_129933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 20), objlist_129932, 'append')
            # Calling append(args, kwargs) (line 520)
            append_call_result_129939 = invoke(stypy.reporting.localization.Localization(__file__, 520, 20), append_129933, *[id_call_result_129937], **kwargs_129938)
            
            
            # Call to print(...): (line 521)
            # Processing the call arguments (line 521)
            str_129941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 26), 'str', '     *** Found in %s ***')
            # Getting the type of 'namestr' (line 521)
            namestr_129942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 55), 'namestr', False)
            # Applying the binary operator '%' (line 521)
            result_mod_129943 = python_operator(stypy.reporting.localization.Localization(__file__, 521, 26), '%', str_129941, namestr_129942)
            
            # Processing the call keyword arguments (line 521)
            # Getting the type of 'output' (line 521)
            output_129944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 69), 'output', False)
            keyword_129945 = output_129944
            kwargs_129946 = {'file': keyword_129945}
            # Getting the type of 'print' (line 521)
            print_129940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 20), 'print', False)
            # Calling print(args, kwargs) (line 521)
            print_call_result_129947 = invoke(stypy.reporting.localization.Localization(__file__, 521, 20), print_129940, *[result_mod_129943], **kwargs_129946)
            
            
            # Call to info(...): (line 522)
            # Processing the call arguments (line 522)
            # Getting the type of 'obj' (line 522)
            obj_129949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 25), 'obj', False)
            # Processing the call keyword arguments (line 522)
            kwargs_129950 = {}
            # Getting the type of 'info' (line 522)
            info_129948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 20), 'info', False)
            # Calling info(args, kwargs) (line 522)
            info_call_result_129951 = invoke(stypy.reporting.localization.Localization(__file__, 522, 20), info_129948, *[obj_129949], **kwargs_129950)
            
            
            # Call to print(...): (line 523)
            # Processing the call arguments (line 523)
            str_129953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 523, 26), 'str', '-')
            # Getting the type of 'maxwidth' (line 523)
            maxwidth_129954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 30), 'maxwidth', False)
            # Applying the binary operator '*' (line 523)
            result_mul_129955 = python_operator(stypy.reporting.localization.Localization(__file__, 523, 26), '*', str_129953, maxwidth_129954)
            
            # Processing the call keyword arguments (line 523)
            # Getting the type of 'output' (line 523)
            output_129956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 45), 'output', False)
            keyword_129957 = output_129956
            kwargs_129958 = {'file': keyword_129957}
            # Getting the type of 'print' (line 523)
            print_129952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 20), 'print', False)
            # Calling print(args, kwargs) (line 523)
            print_call_result_129959 = invoke(stypy.reporting.localization.Localization(__file__, 523, 20), print_129952, *[result_mul_129955], **kwargs_129958)
            
            # SSA join for if statement (line 514)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'numfound' (line 524)
            numfound_129960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'numfound')
            int_129961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 28), 'int')
            # Applying the binary operator '+=' (line 524)
            result_iadd_129962 = python_operator(stypy.reporting.localization.Localization(__file__, 524, 16), '+=', numfound_129960, int_129961)
            # Assigning a type to the variable 'numfound' (line 524)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 524, 16), 'numfound', result_iadd_129962)
            
            # SSA branch for the except part of a try statement (line 512)
            # SSA branch for the except 'KeyError' branch of a try statement (line 512)
            module_type_store.open_ssa_branch('except')
            pass
            # SSA join for try-except statement (line 512)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Getting the type of 'numfound' (line 527)
            numfound_129963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 11), 'numfound')
            int_129964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 527, 23), 'int')
            # Applying the binary operator '==' (line 527)
            result_eq_129965 = python_operator(stypy.reporting.localization.Localization(__file__, 527, 11), '==', numfound_129963, int_129964)
            
            # Testing the type of an if condition (line 527)
            if_condition_129966 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 527, 8), result_eq_129965)
            # Assigning a type to the variable 'if_condition_129966' (line 527)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'if_condition_129966', if_condition_129966)
            # SSA begins for if statement (line 527)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to print(...): (line 528)
            # Processing the call arguments (line 528)
            str_129968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 18), 'str', 'Help for %s not found.')
            # Getting the type of 'object' (line 528)
            object_129969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 45), 'object', False)
            # Applying the binary operator '%' (line 528)
            result_mod_129970 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 18), '%', str_129968, object_129969)
            
            # Processing the call keyword arguments (line 528)
            # Getting the type of 'output' (line 528)
            output_129971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 58), 'output', False)
            keyword_129972 = output_129971
            kwargs_129973 = {'file': keyword_129972}
            # Getting the type of 'print' (line 528)
            print_129967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'print', False)
            # Calling print(args, kwargs) (line 528)
            print_call_result_129974 = invoke(stypy.reporting.localization.Localization(__file__, 528, 12), print_129967, *[result_mod_129970], **kwargs_129973)
            
            # SSA branch for the else part of an if statement (line 527)
            module_type_store.open_ssa_branch('else')
            
            # Call to print(...): (line 530)
            # Processing the call arguments (line 530)
            str_129976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 18), 'str', '\n     *** Total of %d references found. ***')
            # Getting the type of 'numfound' (line 531)
            numfound_129977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 60), 'numfound', False)
            # Applying the binary operator '%' (line 530)
            result_mod_129978 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 18), '%', str_129976, numfound_129977)
            
            # Processing the call keyword arguments (line 530)
            # Getting the type of 'output' (line 532)
            output_129979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 23), 'output', False)
            keyword_129980 = output_129979
            kwargs_129981 = {'file': keyword_129980}
            # Getting the type of 'print' (line 530)
            print_129975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 12), 'print', False)
            # Calling print(args, kwargs) (line 530)
            print_call_result_129982 = invoke(stypy.reporting.localization.Localization(__file__, 530, 12), print_129975, *[result_mod_129978], **kwargs_129981)
            
            # SSA join for if statement (line 527)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_129885:
                # Runtime conditional SSA for else branch (line 506)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_129884) or more_types_in_union_129885):
            # Assigning a type to the variable 'object' (line 506)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 9), 'object', remove_subtype_from_union(object_129883, str))
            
            
            # Call to isfunction(...): (line 535)
            # Processing the call arguments (line 535)
            # Getting the type of 'object' (line 535)
            object_129985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 28), 'object', False)
            # Processing the call keyword arguments (line 535)
            kwargs_129986 = {}
            # Getting the type of 'inspect' (line 535)
            inspect_129983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 9), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 535)
            isfunction_129984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 9), inspect_129983, 'isfunction')
            # Calling isfunction(args, kwargs) (line 535)
            isfunction_call_result_129987 = invoke(stypy.reporting.localization.Localization(__file__, 535, 9), isfunction_129984, *[object_129985], **kwargs_129986)
            
            # Testing the type of an if condition (line 535)
            if_condition_129988 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 9), isfunction_call_result_129987)
            # Assigning a type to the variable 'if_condition_129988' (line 535)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 9), 'if_condition_129988', if_condition_129988)
            # SSA begins for if statement (line 535)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 536):
            
            # Assigning a Attribute to a Name (line 536):
            # Getting the type of 'object' (line 536)
            object_129989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 15), 'object')
            # Obtaining the member '__name__' of a type (line 536)
            name___129990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 536, 15), object_129989, '__name__')
            # Assigning a type to the variable 'name' (line 536)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 8), 'name', name___129990)
            
            # Assigning a Call to a Name (line 537):
            
            # Assigning a Call to a Name (line 537):
            
            # Call to formatargspec(...): (line 537)
            
            # Call to getargspec(...): (line 537)
            # Processing the call arguments (line 537)
            # Getting the type of 'object' (line 537)
            object_129993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 46), 'object', False)
            # Processing the call keyword arguments (line 537)
            kwargs_129994 = {}
            # Getting the type of 'getargspec' (line 537)
            getargspec_129992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 35), 'getargspec', False)
            # Calling getargspec(args, kwargs) (line 537)
            getargspec_call_result_129995 = invoke(stypy.reporting.localization.Localization(__file__, 537, 35), getargspec_129992, *[object_129993], **kwargs_129994)
            
            # Processing the call keyword arguments (line 537)
            kwargs_129996 = {}
            # Getting the type of 'formatargspec' (line 537)
            formatargspec_129991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 20), 'formatargspec', False)
            # Calling formatargspec(args, kwargs) (line 537)
            formatargspec_call_result_129997 = invoke(stypy.reporting.localization.Localization(__file__, 537, 20), formatargspec_129991, *[getargspec_call_result_129995], **kwargs_129996)
            
            # Assigning a type to the variable 'arguments' (line 537)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'arguments', formatargspec_call_result_129997)
            
            
            
            # Call to len(...): (line 539)
            # Processing the call arguments (line 539)
            # Getting the type of 'name' (line 539)
            name_129999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 15), 'name', False)
            # Getting the type of 'arguments' (line 539)
            arguments_130000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 20), 'arguments', False)
            # Applying the binary operator '+' (line 539)
            result_add_130001 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 15), '+', name_129999, arguments_130000)
            
            # Processing the call keyword arguments (line 539)
            kwargs_130002 = {}
            # Getting the type of 'len' (line 539)
            len_129998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 11), 'len', False)
            # Calling len(args, kwargs) (line 539)
            len_call_result_130003 = invoke(stypy.reporting.localization.Localization(__file__, 539, 11), len_129998, *[result_add_130001], **kwargs_130002)
            
            # Getting the type of 'maxwidth' (line 539)
            maxwidth_130004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 33), 'maxwidth')
            # Applying the binary operator '>' (line 539)
            result_gt_130005 = python_operator(stypy.reporting.localization.Localization(__file__, 539, 11), '>', len_call_result_130003, maxwidth_130004)
            
            # Testing the type of an if condition (line 539)
            if_condition_130006 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 539, 8), result_gt_130005)
            # Assigning a type to the variable 'if_condition_130006' (line 539)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'if_condition_130006', if_condition_130006)
            # SSA begins for if statement (line 539)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 540):
            
            # Assigning a Call to a Name (line 540):
            
            # Call to _split_line(...): (line 540)
            # Processing the call arguments (line 540)
            # Getting the type of 'name' (line 540)
            name_130008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 33), 'name', False)
            # Getting the type of 'arguments' (line 540)
            arguments_130009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 39), 'arguments', False)
            # Getting the type of 'maxwidth' (line 540)
            maxwidth_130010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 50), 'maxwidth', False)
            # Processing the call keyword arguments (line 540)
            kwargs_130011 = {}
            # Getting the type of '_split_line' (line 540)
            _split_line_130007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 21), '_split_line', False)
            # Calling _split_line(args, kwargs) (line 540)
            _split_line_call_result_130012 = invoke(stypy.reporting.localization.Localization(__file__, 540, 21), _split_line_130007, *[name_130008, arguments_130009, maxwidth_130010], **kwargs_130011)
            
            # Assigning a type to the variable 'argstr' (line 540)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 12), 'argstr', _split_line_call_result_130012)
            # SSA branch for the else part of an if statement (line 539)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 542):
            
            # Assigning a BinOp to a Name (line 542):
            # Getting the type of 'name' (line 542)
            name_130013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 21), 'name')
            # Getting the type of 'arguments' (line 542)
            arguments_130014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 28), 'arguments')
            # Applying the binary operator '+' (line 542)
            result_add_130015 = python_operator(stypy.reporting.localization.Localization(__file__, 542, 21), '+', name_130013, arguments_130014)
            
            # Assigning a type to the variable 'argstr' (line 542)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 12), 'argstr', result_add_130015)
            # SSA join for if statement (line 539)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to print(...): (line 544)
            # Processing the call arguments (line 544)
            str_130017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 14), 'str', ' ')
            # Getting the type of 'argstr' (line 544)
            argstr_130018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 20), 'argstr', False)
            # Applying the binary operator '+' (line 544)
            result_add_130019 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 14), '+', str_130017, argstr_130018)
            
            str_130020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 29), 'str', '\n')
            # Applying the binary operator '+' (line 544)
            result_add_130021 = python_operator(stypy.reporting.localization.Localization(__file__, 544, 27), '+', result_add_130019, str_130020)
            
            # Processing the call keyword arguments (line 544)
            # Getting the type of 'output' (line 544)
            output_130022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 40), 'output', False)
            keyword_130023 = output_130022
            kwargs_130024 = {'file': keyword_130023}
            # Getting the type of 'print' (line 544)
            print_130016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'print', False)
            # Calling print(args, kwargs) (line 544)
            print_call_result_130025 = invoke(stypy.reporting.localization.Localization(__file__, 544, 8), print_130016, *[result_add_130021], **kwargs_130024)
            
            
            # Call to print(...): (line 545)
            # Processing the call arguments (line 545)
            
            # Call to getdoc(...): (line 545)
            # Processing the call arguments (line 545)
            # Getting the type of 'object' (line 545)
            object_130029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 29), 'object', False)
            # Processing the call keyword arguments (line 545)
            kwargs_130030 = {}
            # Getting the type of 'inspect' (line 545)
            inspect_130027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 14), 'inspect', False)
            # Obtaining the member 'getdoc' of a type (line 545)
            getdoc_130028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 14), inspect_130027, 'getdoc')
            # Calling getdoc(args, kwargs) (line 545)
            getdoc_call_result_130031 = invoke(stypy.reporting.localization.Localization(__file__, 545, 14), getdoc_130028, *[object_130029], **kwargs_130030)
            
            # Processing the call keyword arguments (line 545)
            # Getting the type of 'output' (line 545)
            output_130032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 43), 'output', False)
            keyword_130033 = output_130032
            kwargs_130034 = {'file': keyword_130033}
            # Getting the type of 'print' (line 545)
            print_130026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'print', False)
            # Calling print(args, kwargs) (line 545)
            print_call_result_130035 = invoke(stypy.reporting.localization.Localization(__file__, 545, 8), print_130026, *[getdoc_call_result_130031], **kwargs_130034)
            
            # SSA branch for the else part of an if statement (line 535)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to isclass(...): (line 547)
            # Processing the call arguments (line 547)
            # Getting the type of 'object' (line 547)
            object_130038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 25), 'object', False)
            # Processing the call keyword arguments (line 547)
            kwargs_130039 = {}
            # Getting the type of 'inspect' (line 547)
            inspect_130036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 9), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 547)
            isclass_130037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 9), inspect_130036, 'isclass')
            # Calling isclass(args, kwargs) (line 547)
            isclass_call_result_130040 = invoke(stypy.reporting.localization.Localization(__file__, 547, 9), isclass_130037, *[object_130038], **kwargs_130039)
            
            # Testing the type of an if condition (line 547)
            if_condition_130041 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 547, 9), isclass_call_result_130040)
            # Assigning a type to the variable 'if_condition_130041' (line 547)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 9), 'if_condition_130041', if_condition_130041)
            # SSA begins for if statement (line 547)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 548):
            
            # Assigning a Attribute to a Name (line 548):
            # Getting the type of 'object' (line 548)
            object_130042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 15), 'object')
            # Obtaining the member '__name__' of a type (line 548)
            name___130043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 15), object_130042, '__name__')
            # Assigning a type to the variable 'name' (line 548)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 548, 8), 'name', name___130043)
            
            # Assigning a Str to a Name (line 549):
            
            # Assigning a Str to a Name (line 549):
            str_130044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 549, 20), 'str', '()')
            # Assigning a type to the variable 'arguments' (line 549)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'arguments', str_130044)
            
            
            # SSA begins for try-except statement (line 550)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Type idiom detected: calculating its left and rigth part (line 551)
            str_130045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 551, 31), 'str', '__init__')
            # Getting the type of 'object' (line 551)
            object_130046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 23), 'object')
            
            (may_be_130047, more_types_in_union_130048) = may_provide_member(str_130045, object_130046)

            if may_be_130047:

                if more_types_in_union_130048:
                    # Runtime conditional SSA (line 551)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'object' (line 551)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'object', remove_not_member_provider_from_union(object_130046, '__init__'))
                
                # Assigning a Call to a Name (line 552):
                
                # Assigning a Call to a Name (line 552):
                
                # Call to formatargspec(...): (line 552)
                
                # Call to getargspec(...): (line 553)
                # Processing the call arguments (line 553)
                # Getting the type of 'object' (line 553)
                object_130051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 36), 'object', False)
                # Obtaining the member '__init__' of a type (line 553)
                init___130052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 36), object_130051, '__init__')
                # Obtaining the member '__func__' of a type (line 553)
                func___130053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 553, 36), init___130052, '__func__')
                # Processing the call keyword arguments (line 553)
                kwargs_130054 = {}
                # Getting the type of 'getargspec' (line 553)
                getargspec_130050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 25), 'getargspec', False)
                # Calling getargspec(args, kwargs) (line 553)
                getargspec_call_result_130055 = invoke(stypy.reporting.localization.Localization(__file__, 553, 25), getargspec_130050, *[func___130053], **kwargs_130054)
                
                # Processing the call keyword arguments (line 552)
                kwargs_130056 = {}
                # Getting the type of 'formatargspec' (line 552)
                formatargspec_130049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 28), 'formatargspec', False)
                # Calling formatargspec(args, kwargs) (line 552)
                formatargspec_call_result_130057 = invoke(stypy.reporting.localization.Localization(__file__, 552, 28), formatargspec_130049, *[getargspec_call_result_130055], **kwargs_130056)
                
                # Assigning a type to the variable 'arguments' (line 552)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 16), 'arguments', formatargspec_call_result_130057)
                
                # Assigning a Call to a Name (line 555):
                
                # Assigning a Call to a Name (line 555):
                
                # Call to split(...): (line 555)
                # Processing the call arguments (line 555)
                str_130060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 555, 42), 'str', ', ')
                # Processing the call keyword arguments (line 555)
                kwargs_130061 = {}
                # Getting the type of 'arguments' (line 555)
                arguments_130058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 26), 'arguments', False)
                # Obtaining the member 'split' of a type (line 555)
                split_130059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 26), arguments_130058, 'split')
                # Calling split(args, kwargs) (line 555)
                split_call_result_130062 = invoke(stypy.reporting.localization.Localization(__file__, 555, 26), split_130059, *[str_130060], **kwargs_130061)
                
                # Assigning a type to the variable 'arglist' (line 555)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 16), 'arglist', split_call_result_130062)
                
                
                
                # Call to len(...): (line 556)
                # Processing the call arguments (line 556)
                # Getting the type of 'arglist' (line 556)
                arglist_130064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 23), 'arglist', False)
                # Processing the call keyword arguments (line 556)
                kwargs_130065 = {}
                # Getting the type of 'len' (line 556)
                len_130063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 19), 'len', False)
                # Calling len(args, kwargs) (line 556)
                len_call_result_130066 = invoke(stypy.reporting.localization.Localization(__file__, 556, 19), len_130063, *[arglist_130064], **kwargs_130065)
                
                int_130067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 34), 'int')
                # Applying the binary operator '>' (line 556)
                result_gt_130068 = python_operator(stypy.reporting.localization.Localization(__file__, 556, 19), '>', len_call_result_130066, int_130067)
                
                # Testing the type of an if condition (line 556)
                if_condition_130069 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 556, 16), result_gt_130068)
                # Assigning a type to the variable 'if_condition_130069' (line 556)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'if_condition_130069', if_condition_130069)
                # SSA begins for if statement (line 556)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Subscript (line 557):
                
                # Assigning a BinOp to a Subscript (line 557):
                str_130070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 33), 'str', '(')
                
                # Obtaining the type of the subscript
                int_130071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 45), 'int')
                # Getting the type of 'arglist' (line 557)
                arglist_130072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 37), 'arglist')
                # Obtaining the member '__getitem__' of a type (line 557)
                getitem___130073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 37), arglist_130072, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 557)
                subscript_call_result_130074 = invoke(stypy.reporting.localization.Localization(__file__, 557, 37), getitem___130073, int_130071)
                
                # Applying the binary operator '+' (line 557)
                result_add_130075 = python_operator(stypy.reporting.localization.Localization(__file__, 557, 33), '+', str_130070, subscript_call_result_130074)
                
                # Getting the type of 'arglist' (line 557)
                arglist_130076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 20), 'arglist')
                int_130077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 28), 'int')
                # Storing an element on a container (line 557)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 557, 20), arglist_130076, (int_130077, result_add_130075))
                
                # Assigning a Call to a Name (line 558):
                
                # Assigning a Call to a Name (line 558):
                
                # Call to join(...): (line 558)
                # Processing the call arguments (line 558)
                
                # Obtaining the type of the subscript
                int_130080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 50), 'int')
                slice_130081 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 558, 42), int_130080, None, None)
                # Getting the type of 'arglist' (line 558)
                arglist_130082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 42), 'arglist', False)
                # Obtaining the member '__getitem__' of a type (line 558)
                getitem___130083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 42), arglist_130082, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 558)
                subscript_call_result_130084 = invoke(stypy.reporting.localization.Localization(__file__, 558, 42), getitem___130083, slice_130081)
                
                # Processing the call keyword arguments (line 558)
                kwargs_130085 = {}
                str_130078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 32), 'str', ', ')
                # Obtaining the member 'join' of a type (line 558)
                join_130079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 32), str_130078, 'join')
                # Calling join(args, kwargs) (line 558)
                join_call_result_130086 = invoke(stypy.reporting.localization.Localization(__file__, 558, 32), join_130079, *[subscript_call_result_130084], **kwargs_130085)
                
                # Assigning a type to the variable 'arguments' (line 558)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 20), 'arguments', join_call_result_130086)
                # SSA join for if statement (line 556)
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_130048:
                    # SSA join for if statement (line 551)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA branch for the except part of a try statement (line 550)
            # SSA branch for the except '<any exception>' branch of a try statement (line 550)
            module_type_store.open_ssa_branch('except')
            pass
            # SSA join for try-except statement (line 550)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            
            # Call to len(...): (line 562)
            # Processing the call arguments (line 562)
            # Getting the type of 'name' (line 562)
            name_130088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 15), 'name', False)
            # Getting the type of 'arguments' (line 562)
            arguments_130089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 20), 'arguments', False)
            # Applying the binary operator '+' (line 562)
            result_add_130090 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 15), '+', name_130088, arguments_130089)
            
            # Processing the call keyword arguments (line 562)
            kwargs_130091 = {}
            # Getting the type of 'len' (line 562)
            len_130087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 11), 'len', False)
            # Calling len(args, kwargs) (line 562)
            len_call_result_130092 = invoke(stypy.reporting.localization.Localization(__file__, 562, 11), len_130087, *[result_add_130090], **kwargs_130091)
            
            # Getting the type of 'maxwidth' (line 562)
            maxwidth_130093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 33), 'maxwidth')
            # Applying the binary operator '>' (line 562)
            result_gt_130094 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 11), '>', len_call_result_130092, maxwidth_130093)
            
            # Testing the type of an if condition (line 562)
            if_condition_130095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 8), result_gt_130094)
            # Assigning a type to the variable 'if_condition_130095' (line 562)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'if_condition_130095', if_condition_130095)
            # SSA begins for if statement (line 562)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 563):
            
            # Assigning a Call to a Name (line 563):
            
            # Call to _split_line(...): (line 563)
            # Processing the call arguments (line 563)
            # Getting the type of 'name' (line 563)
            name_130097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 33), 'name', False)
            # Getting the type of 'arguments' (line 563)
            arguments_130098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 39), 'arguments', False)
            # Getting the type of 'maxwidth' (line 563)
            maxwidth_130099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 50), 'maxwidth', False)
            # Processing the call keyword arguments (line 563)
            kwargs_130100 = {}
            # Getting the type of '_split_line' (line 563)
            _split_line_130096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 21), '_split_line', False)
            # Calling _split_line(args, kwargs) (line 563)
            _split_line_call_result_130101 = invoke(stypy.reporting.localization.Localization(__file__, 563, 21), _split_line_130096, *[name_130097, arguments_130098, maxwidth_130099], **kwargs_130100)
            
            # Assigning a type to the variable 'argstr' (line 563)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'argstr', _split_line_call_result_130101)
            # SSA branch for the else part of an if statement (line 562)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 565):
            
            # Assigning a BinOp to a Name (line 565):
            # Getting the type of 'name' (line 565)
            name_130102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 21), 'name')
            # Getting the type of 'arguments' (line 565)
            arguments_130103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 28), 'arguments')
            # Applying the binary operator '+' (line 565)
            result_add_130104 = python_operator(stypy.reporting.localization.Localization(__file__, 565, 21), '+', name_130102, arguments_130103)
            
            # Assigning a type to the variable 'argstr' (line 565)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'argstr', result_add_130104)
            # SSA join for if statement (line 562)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to print(...): (line 567)
            # Processing the call arguments (line 567)
            str_130106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 14), 'str', ' ')
            # Getting the type of 'argstr' (line 567)
            argstr_130107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 20), 'argstr', False)
            # Applying the binary operator '+' (line 567)
            result_add_130108 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 14), '+', str_130106, argstr_130107)
            
            str_130109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 567, 29), 'str', '\n')
            # Applying the binary operator '+' (line 567)
            result_add_130110 = python_operator(stypy.reporting.localization.Localization(__file__, 567, 27), '+', result_add_130108, str_130109)
            
            # Processing the call keyword arguments (line 567)
            # Getting the type of 'output' (line 567)
            output_130111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 40), 'output', False)
            keyword_130112 = output_130111
            kwargs_130113 = {'file': keyword_130112}
            # Getting the type of 'print' (line 567)
            print_130105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 8), 'print', False)
            # Calling print(args, kwargs) (line 567)
            print_call_result_130114 = invoke(stypy.reporting.localization.Localization(__file__, 567, 8), print_130105, *[result_add_130110], **kwargs_130113)
            
            
            # Assigning a Call to a Name (line 568):
            
            # Assigning a Call to a Name (line 568):
            
            # Call to getdoc(...): (line 568)
            # Processing the call arguments (line 568)
            # Getting the type of 'object' (line 568)
            object_130117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 30), 'object', False)
            # Processing the call keyword arguments (line 568)
            kwargs_130118 = {}
            # Getting the type of 'inspect' (line 568)
            inspect_130115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 15), 'inspect', False)
            # Obtaining the member 'getdoc' of a type (line 568)
            getdoc_130116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 15), inspect_130115, 'getdoc')
            # Calling getdoc(args, kwargs) (line 568)
            getdoc_call_result_130119 = invoke(stypy.reporting.localization.Localization(__file__, 568, 15), getdoc_130116, *[object_130117], **kwargs_130118)
            
            # Assigning a type to the variable 'doc1' (line 568)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'doc1', getdoc_call_result_130119)
            
            # Type idiom detected: calculating its left and rigth part (line 569)
            # Getting the type of 'doc1' (line 569)
            doc1_130120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 11), 'doc1')
            # Getting the type of 'None' (line 569)
            None_130121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 19), 'None')
            
            (may_be_130122, more_types_in_union_130123) = may_be_none(doc1_130120, None_130121)

            if may_be_130122:

                if more_types_in_union_130123:
                    # Runtime conditional SSA (line 569)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Type idiom detected: calculating its left and rigth part (line 570)
                str_130124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 31), 'str', '__init__')
                # Getting the type of 'object' (line 570)
                object_130125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 23), 'object')
                
                (may_be_130126, more_types_in_union_130127) = may_provide_member(str_130124, object_130125)

                if may_be_130126:

                    if more_types_in_union_130127:
                        # Runtime conditional SSA (line 570)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'object' (line 570)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 570, 12), 'object', remove_not_member_provider_from_union(object_130125, '__init__'))
                    
                    # Call to print(...): (line 571)
                    # Processing the call arguments (line 571)
                    
                    # Call to getdoc(...): (line 571)
                    # Processing the call arguments (line 571)
                    # Getting the type of 'object' (line 571)
                    object_130131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 37), 'object', False)
                    # Obtaining the member '__init__' of a type (line 571)
                    init___130132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 37), object_130131, '__init__')
                    # Processing the call keyword arguments (line 571)
                    kwargs_130133 = {}
                    # Getting the type of 'inspect' (line 571)
                    inspect_130129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 22), 'inspect', False)
                    # Obtaining the member 'getdoc' of a type (line 571)
                    getdoc_130130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 22), inspect_130129, 'getdoc')
                    # Calling getdoc(args, kwargs) (line 571)
                    getdoc_call_result_130134 = invoke(stypy.reporting.localization.Localization(__file__, 571, 22), getdoc_130130, *[init___130132], **kwargs_130133)
                    
                    # Processing the call keyword arguments (line 571)
                    # Getting the type of 'output' (line 571)
                    output_130135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 60), 'output', False)
                    keyword_130136 = output_130135
                    kwargs_130137 = {'file': keyword_130136}
                    # Getting the type of 'print' (line 571)
                    print_130128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 16), 'print', False)
                    # Calling print(args, kwargs) (line 571)
                    print_call_result_130138 = invoke(stypy.reporting.localization.Localization(__file__, 571, 16), print_130128, *[getdoc_call_result_130134], **kwargs_130137)
                    

                    if more_types_in_union_130127:
                        # SSA join for if statement (line 570)
                        module_type_store = module_type_store.join_ssa_context()


                

                if more_types_in_union_130123:
                    # Runtime conditional SSA for else branch (line 569)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_130122) or more_types_in_union_130123):
                
                # Call to print(...): (line 573)
                # Processing the call arguments (line 573)
                
                # Call to getdoc(...): (line 573)
                # Processing the call arguments (line 573)
                # Getting the type of 'object' (line 573)
                object_130142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 33), 'object', False)
                # Processing the call keyword arguments (line 573)
                kwargs_130143 = {}
                # Getting the type of 'inspect' (line 573)
                inspect_130140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 18), 'inspect', False)
                # Obtaining the member 'getdoc' of a type (line 573)
                getdoc_130141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 18), inspect_130140, 'getdoc')
                # Calling getdoc(args, kwargs) (line 573)
                getdoc_call_result_130144 = invoke(stypy.reporting.localization.Localization(__file__, 573, 18), getdoc_130141, *[object_130142], **kwargs_130143)
                
                # Processing the call keyword arguments (line 573)
                # Getting the type of 'output' (line 573)
                output_130145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 47), 'output', False)
                keyword_130146 = output_130145
                kwargs_130147 = {'file': keyword_130146}
                # Getting the type of 'print' (line 573)
                print_130139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 12), 'print', False)
                # Calling print(args, kwargs) (line 573)
                print_call_result_130148 = invoke(stypy.reporting.localization.Localization(__file__, 573, 12), print_130139, *[getdoc_call_result_130144], **kwargs_130147)
                

                if (may_be_130122 and more_types_in_union_130123):
                    # SSA join for if statement (line 569)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Assigning a Call to a Name (line 575):
            
            # Assigning a Call to a Name (line 575):
            
            # Call to allmethods(...): (line 575)
            # Processing the call arguments (line 575)
            # Getting the type of 'object' (line 575)
            object_130151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 35), 'object', False)
            # Processing the call keyword arguments (line 575)
            kwargs_130152 = {}
            # Getting the type of 'pydoc' (line 575)
            pydoc_130149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 18), 'pydoc', False)
            # Obtaining the member 'allmethods' of a type (line 575)
            allmethods_130150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 18), pydoc_130149, 'allmethods')
            # Calling allmethods(args, kwargs) (line 575)
            allmethods_call_result_130153 = invoke(stypy.reporting.localization.Localization(__file__, 575, 18), allmethods_130150, *[object_130151], **kwargs_130152)
            
            # Assigning a type to the variable 'methods' (line 575)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 8), 'methods', allmethods_call_result_130153)
            
            
            # Getting the type of 'methods' (line 576)
            methods_130154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 11), 'methods')
            
            # Obtaining an instance of the builtin type 'list' (line 576)
            list_130155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 22), 'list')
            # Adding type elements to the builtin type 'list' instance (line 576)
            
            # Applying the binary operator '!=' (line 576)
            result_ne_130156 = python_operator(stypy.reporting.localization.Localization(__file__, 576, 11), '!=', methods_130154, list_130155)
            
            # Testing the type of an if condition (line 576)
            if_condition_130157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 576, 8), result_ne_130156)
            # Assigning a type to the variable 'if_condition_130157' (line 576)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'if_condition_130157', if_condition_130157)
            # SSA begins for if statement (line 576)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to print(...): (line 577)
            # Processing the call arguments (line 577)
            str_130159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 577, 18), 'str', '\n\nMethods:\n')
            # Processing the call keyword arguments (line 577)
            # Getting the type of 'output' (line 577)
            output_130160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 41), 'output', False)
            keyword_130161 = output_130160
            kwargs_130162 = {'file': keyword_130161}
            # Getting the type of 'print' (line 577)
            print_130158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 12), 'print', False)
            # Calling print(args, kwargs) (line 577)
            print_call_result_130163 = invoke(stypy.reporting.localization.Localization(__file__, 577, 12), print_130158, *[str_130159], **kwargs_130162)
            
            
            # Getting the type of 'methods' (line 578)
            methods_130164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 24), 'methods')
            # Testing the type of a for loop iterable (line 578)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 578, 12), methods_130164)
            # Getting the type of the for loop variable (line 578)
            for_loop_var_130165 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 578, 12), methods_130164)
            # Assigning a type to the variable 'meth' (line 578)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 12), 'meth', for_loop_var_130165)
            # SSA begins for a for statement (line 578)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            
            # Obtaining the type of the subscript
            int_130166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 24), 'int')
            # Getting the type of 'meth' (line 579)
            meth_130167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 19), 'meth')
            # Obtaining the member '__getitem__' of a type (line 579)
            getitem___130168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 19), meth_130167, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 579)
            subscript_call_result_130169 = invoke(stypy.reporting.localization.Localization(__file__, 579, 19), getitem___130168, int_130166)
            
            str_130170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 30), 'str', '_')
            # Applying the binary operator '==' (line 579)
            result_eq_130171 = python_operator(stypy.reporting.localization.Localization(__file__, 579, 19), '==', subscript_call_result_130169, str_130170)
            
            # Testing the type of an if condition (line 579)
            if_condition_130172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 579, 16), result_eq_130171)
            # Assigning a type to the variable 'if_condition_130172' (line 579)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 16), 'if_condition_130172', if_condition_130172)
            # SSA begins for if statement (line 579)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 579)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Name (line 581):
            
            # Assigning a Call to a Name (line 581):
            
            # Call to getattr(...): (line 581)
            # Processing the call arguments (line 581)
            # Getting the type of 'object' (line 581)
            object_130174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 34), 'object', False)
            # Getting the type of 'meth' (line 581)
            meth_130175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 42), 'meth', False)
            # Getting the type of 'None' (line 581)
            None_130176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 48), 'None', False)
            # Processing the call keyword arguments (line 581)
            kwargs_130177 = {}
            # Getting the type of 'getattr' (line 581)
            getattr_130173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 26), 'getattr', False)
            # Calling getattr(args, kwargs) (line 581)
            getattr_call_result_130178 = invoke(stypy.reporting.localization.Localization(__file__, 581, 26), getattr_130173, *[object_130174, meth_130175, None_130176], **kwargs_130177)
            
            # Assigning a type to the variable 'thisobj' (line 581)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 16), 'thisobj', getattr_call_result_130178)
            
            # Type idiom detected: calculating its left and rigth part (line 582)
            # Getting the type of 'thisobj' (line 582)
            thisobj_130179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'thisobj')
            # Getting the type of 'None' (line 582)
            None_130180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 34), 'None')
            
            (may_be_130181, more_types_in_union_130182) = may_not_be_none(thisobj_130179, None_130180)

            if may_be_130181:

                if more_types_in_union_130182:
                    # Runtime conditional SSA (line 582)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Tuple (line 583):
                
                # Assigning a Call to a Name:
                
                # Call to splitdoc(...): (line 583)
                # Processing the call arguments (line 583)
                
                # Evaluating a boolean operation
                
                # Call to getdoc(...): (line 584)
                # Processing the call arguments (line 584)
                # Getting the type of 'thisobj' (line 584)
                thisobj_130187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 43), 'thisobj', False)
                # Processing the call keyword arguments (line 584)
                kwargs_130188 = {}
                # Getting the type of 'inspect' (line 584)
                inspect_130185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 28), 'inspect', False)
                # Obtaining the member 'getdoc' of a type (line 584)
                getdoc_130186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 28), inspect_130185, 'getdoc')
                # Calling getdoc(args, kwargs) (line 584)
                getdoc_call_result_130189 = invoke(stypy.reporting.localization.Localization(__file__, 584, 28), getdoc_130186, *[thisobj_130187], **kwargs_130188)
                
                str_130190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 55), 'str', 'None')
                # Applying the binary operator 'or' (line 584)
                result_or_keyword_130191 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 28), 'or', getdoc_call_result_130189, str_130190)
                
                # Processing the call keyword arguments (line 583)
                kwargs_130192 = {}
                # Getting the type of 'pydoc' (line 583)
                pydoc_130183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 37), 'pydoc', False)
                # Obtaining the member 'splitdoc' of a type (line 583)
                splitdoc_130184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 37), pydoc_130183, 'splitdoc')
                # Calling splitdoc(args, kwargs) (line 583)
                splitdoc_call_result_130193 = invoke(stypy.reporting.localization.Localization(__file__, 583, 37), splitdoc_130184, *[result_or_keyword_130191], **kwargs_130192)
                
                # Assigning a type to the variable 'call_assignment_128925' (line 583)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'call_assignment_128925', splitdoc_call_result_130193)
                
                # Assigning a Call to a Name (line 583):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_130196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 20), 'int')
                # Processing the call keyword arguments
                kwargs_130197 = {}
                # Getting the type of 'call_assignment_128925' (line 583)
                call_assignment_128925_130194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'call_assignment_128925', False)
                # Obtaining the member '__getitem__' of a type (line 583)
                getitem___130195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 20), call_assignment_128925_130194, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_130198 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___130195, *[int_130196], **kwargs_130197)
                
                # Assigning a type to the variable 'call_assignment_128926' (line 583)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'call_assignment_128926', getitem___call_result_130198)
                
                # Assigning a Name to a Name (line 583):
                # Getting the type of 'call_assignment_128926' (line 583)
                call_assignment_128926_130199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'call_assignment_128926')
                # Assigning a type to the variable 'methstr' (line 583)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'methstr', call_assignment_128926_130199)
                
                # Assigning a Call to a Name (line 583):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_130202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 583, 20), 'int')
                # Processing the call keyword arguments
                kwargs_130203 = {}
                # Getting the type of 'call_assignment_128925' (line 583)
                call_assignment_128925_130200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'call_assignment_128925', False)
                # Obtaining the member '__getitem__' of a type (line 583)
                getitem___130201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 20), call_assignment_128925_130200, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_130204 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___130201, *[int_130202], **kwargs_130203)
                
                # Assigning a type to the variable 'call_assignment_128927' (line 583)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'call_assignment_128927', getitem___call_result_130204)
                
                # Assigning a Name to a Name (line 583):
                # Getting the type of 'call_assignment_128927' (line 583)
                call_assignment_128927_130205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 20), 'call_assignment_128927')
                # Assigning a type to the variable 'other' (line 583)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 583, 29), 'other', call_assignment_128927_130205)

                if more_types_in_union_130182:
                    # SSA join for if statement (line 582)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to print(...): (line 586)
            # Processing the call arguments (line 586)
            str_130207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 22), 'str', '  %s  --  %s')
            
            # Obtaining an instance of the builtin type 'tuple' (line 586)
            tuple_130208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 586, 40), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 586)
            # Adding element type (line 586)
            # Getting the type of 'meth' (line 586)
            meth_130209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 40), 'meth', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 40), tuple_130208, meth_130209)
            # Adding element type (line 586)
            # Getting the type of 'methstr' (line 586)
            methstr_130210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 46), 'methstr', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 586, 40), tuple_130208, methstr_130210)
            
            # Applying the binary operator '%' (line 586)
            result_mod_130211 = python_operator(stypy.reporting.localization.Localization(__file__, 586, 22), '%', str_130207, tuple_130208)
            
            # Processing the call keyword arguments (line 586)
            # Getting the type of 'output' (line 586)
            output_130212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 61), 'output', False)
            keyword_130213 = output_130212
            kwargs_130214 = {'file': keyword_130213}
            # Getting the type of 'print' (line 586)
            print_130206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 16), 'print', False)
            # Calling print(args, kwargs) (line 586)
            print_call_result_130215 = invoke(stypy.reporting.localization.Localization(__file__, 586, 16), print_130206, *[result_mod_130211], **kwargs_130214)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 576)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 547)
            module_type_store.open_ssa_branch('else')
            
            
            # Evaluating a boolean operation
            
            
            # Obtaining the type of the subscript
            int_130216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 27), 'int')
            # Getting the type of 'sys' (line 588)
            sys_130217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 10), 'sys')
            # Obtaining the member 'version_info' of a type (line 588)
            version_info_130218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 10), sys_130217, 'version_info')
            # Obtaining the member '__getitem__' of a type (line 588)
            getitem___130219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 10), version_info_130218, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 588)
            subscript_call_result_130220 = invoke(stypy.reporting.localization.Localization(__file__, 588, 10), getitem___130219, int_130216)
            
            int_130221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 32), 'int')
            # Applying the binary operator '<' (line 588)
            result_lt_130222 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 10), '<', subscript_call_result_130220, int_130221)
            
            
            # Call to isinstance(...): (line 589)
            # Processing the call arguments (line 589)
            # Getting the type of 'object' (line 589)
            object_130224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 27), 'object', False)
            # Getting the type of 'types' (line 589)
            types_130225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 35), 'types', False)
            # Obtaining the member 'InstanceType' of a type (line 589)
            InstanceType_130226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 35), types_130225, 'InstanceType')
            # Processing the call keyword arguments (line 589)
            kwargs_130227 = {}
            # Getting the type of 'isinstance' (line 589)
            isinstance_130223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 589)
            isinstance_call_result_130228 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), isinstance_130223, *[object_130224, InstanceType_130226], **kwargs_130227)
            
            # Applying the binary operator 'and' (line 588)
            result_and_keyword_130229 = python_operator(stypy.reporting.localization.Localization(__file__, 588, 10), 'and', result_lt_130222, isinstance_call_result_130228)
            
            # Testing the type of an if condition (line 588)
            if_condition_130230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 588, 9), result_and_keyword_130229)
            # Assigning a type to the variable 'if_condition_130230' (line 588)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 588, 9), 'if_condition_130230', if_condition_130230)
            # SSA begins for if statement (line 588)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to print(...): (line 592)
            # Processing the call arguments (line 592)
            str_130232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 14), 'str', 'Instance of class: ')
            # Getting the type of 'object' (line 592)
            object_130233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 37), 'object', False)
            # Obtaining the member '__class__' of a type (line 592)
            class___130234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 37), object_130233, '__class__')
            # Obtaining the member '__name__' of a type (line 592)
            name___130235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 37), class___130234, '__name__')
            # Processing the call keyword arguments (line 592)
            # Getting the type of 'output' (line 592)
            output_130236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 69), 'output', False)
            keyword_130237 = output_130236
            kwargs_130238 = {'file': keyword_130237}
            # Getting the type of 'print' (line 592)
            print_130231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'print', False)
            # Calling print(args, kwargs) (line 592)
            print_call_result_130239 = invoke(stypy.reporting.localization.Localization(__file__, 592, 8), print_130231, *[str_130232, name___130235], **kwargs_130238)
            
            
            # Call to print(...): (line 593)
            # Processing the call keyword arguments (line 593)
            # Getting the type of 'output' (line 593)
            output_130241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 19), 'output', False)
            keyword_130242 = output_130241
            kwargs_130243 = {'file': keyword_130242}
            # Getting the type of 'print' (line 593)
            print_130240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 8), 'print', False)
            # Calling print(args, kwargs) (line 593)
            print_call_result_130244 = invoke(stypy.reporting.localization.Localization(__file__, 593, 8), print_130240, *[], **kwargs_130243)
            
            
            # Type idiom detected: calculating its left and rigth part (line 594)
            str_130245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 594, 27), 'str', '__call__')
            # Getting the type of 'object' (line 594)
            object_130246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 19), 'object')
            
            (may_be_130247, more_types_in_union_130248) = may_provide_member(str_130245, object_130246)

            if may_be_130247:

                if more_types_in_union_130248:
                    # Runtime conditional SSA (line 594)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'object' (line 594)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'object', remove_not_member_provider_from_union(object_130246, '__call__'))
                
                # Assigning a Call to a Name (line 595):
                
                # Assigning a Call to a Name (line 595):
                
                # Call to formatargspec(...): (line 595)
                
                # Call to getargspec(...): (line 596)
                # Processing the call arguments (line 596)
                # Getting the type of 'object' (line 596)
                object_130251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 32), 'object', False)
                # Obtaining the member '__call__' of a type (line 596)
                call___130252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 32), object_130251, '__call__')
                # Obtaining the member '__func__' of a type (line 596)
                func___130253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 32), call___130252, '__func__')
                # Processing the call keyword arguments (line 596)
                kwargs_130254 = {}
                # Getting the type of 'getargspec' (line 596)
                getargspec_130250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 21), 'getargspec', False)
                # Calling getargspec(args, kwargs) (line 596)
                getargspec_call_result_130255 = invoke(stypy.reporting.localization.Localization(__file__, 596, 21), getargspec_130250, *[func___130253], **kwargs_130254)
                
                # Processing the call keyword arguments (line 595)
                kwargs_130256 = {}
                # Getting the type of 'formatargspec' (line 595)
                formatargspec_130249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 24), 'formatargspec', False)
                # Calling formatargspec(args, kwargs) (line 595)
                formatargspec_call_result_130257 = invoke(stypy.reporting.localization.Localization(__file__, 595, 24), formatargspec_130249, *[getargspec_call_result_130255], **kwargs_130256)
                
                # Assigning a type to the variable 'arguments' (line 595)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 595, 12), 'arguments', formatargspec_call_result_130257)
                
                # Assigning a Call to a Name (line 598):
                
                # Assigning a Call to a Name (line 598):
                
                # Call to split(...): (line 598)
                # Processing the call arguments (line 598)
                str_130260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 38), 'str', ', ')
                # Processing the call keyword arguments (line 598)
                kwargs_130261 = {}
                # Getting the type of 'arguments' (line 598)
                arguments_130258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 22), 'arguments', False)
                # Obtaining the member 'split' of a type (line 598)
                split_130259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 22), arguments_130258, 'split')
                # Calling split(args, kwargs) (line 598)
                split_call_result_130262 = invoke(stypy.reporting.localization.Localization(__file__, 598, 22), split_130259, *[str_130260], **kwargs_130261)
                
                # Assigning a type to the variable 'arglist' (line 598)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 12), 'arglist', split_call_result_130262)
                
                
                
                # Call to len(...): (line 599)
                # Processing the call arguments (line 599)
                # Getting the type of 'arglist' (line 599)
                arglist_130264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 19), 'arglist', False)
                # Processing the call keyword arguments (line 599)
                kwargs_130265 = {}
                # Getting the type of 'len' (line 599)
                len_130263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 15), 'len', False)
                # Calling len(args, kwargs) (line 599)
                len_call_result_130266 = invoke(stypy.reporting.localization.Localization(__file__, 599, 15), len_130263, *[arglist_130264], **kwargs_130265)
                
                int_130267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 599, 30), 'int')
                # Applying the binary operator '>' (line 599)
                result_gt_130268 = python_operator(stypy.reporting.localization.Localization(__file__, 599, 15), '>', len_call_result_130266, int_130267)
                
                # Testing the type of an if condition (line 599)
                if_condition_130269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 599, 12), result_gt_130268)
                # Assigning a type to the variable 'if_condition_130269' (line 599)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 12), 'if_condition_130269', if_condition_130269)
                # SSA begins for if statement (line 599)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Subscript (line 600):
                
                # Assigning a BinOp to a Subscript (line 600):
                str_130270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 29), 'str', '(')
                
                # Obtaining the type of the subscript
                int_130271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 41), 'int')
                # Getting the type of 'arglist' (line 600)
                arglist_130272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 33), 'arglist')
                # Obtaining the member '__getitem__' of a type (line 600)
                getitem___130273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 33), arglist_130272, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 600)
                subscript_call_result_130274 = invoke(stypy.reporting.localization.Localization(__file__, 600, 33), getitem___130273, int_130271)
                
                # Applying the binary operator '+' (line 600)
                result_add_130275 = python_operator(stypy.reporting.localization.Localization(__file__, 600, 29), '+', str_130270, subscript_call_result_130274)
                
                # Getting the type of 'arglist' (line 600)
                arglist_130276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'arglist')
                int_130277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 24), 'int')
                # Storing an element on a container (line 600)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 600, 16), arglist_130276, (int_130277, result_add_130275))
                
                # Assigning a Call to a Name (line 601):
                
                # Assigning a Call to a Name (line 601):
                
                # Call to join(...): (line 601)
                # Processing the call arguments (line 601)
                
                # Obtaining the type of the subscript
                int_130280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 46), 'int')
                slice_130281 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 601, 38), int_130280, None, None)
                # Getting the type of 'arglist' (line 601)
                arglist_130282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 38), 'arglist', False)
                # Obtaining the member '__getitem__' of a type (line 601)
                getitem___130283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 38), arglist_130282, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 601)
                subscript_call_result_130284 = invoke(stypy.reporting.localization.Localization(__file__, 601, 38), getitem___130283, slice_130281)
                
                # Processing the call keyword arguments (line 601)
                kwargs_130285 = {}
                str_130278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 28), 'str', ', ')
                # Obtaining the member 'join' of a type (line 601)
                join_130279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 28), str_130278, 'join')
                # Calling join(args, kwargs) (line 601)
                join_call_result_130286 = invoke(stypy.reporting.localization.Localization(__file__, 601, 28), join_130279, *[subscript_call_result_130284], **kwargs_130285)
                
                # Assigning a type to the variable 'arguments' (line 601)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 16), 'arguments', join_call_result_130286)
                # SSA branch for the else part of an if statement (line 599)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Str to a Name (line 603):
                
                # Assigning a Str to a Name (line 603):
                str_130287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 28), 'str', '()')
                # Assigning a type to the variable 'arguments' (line 603)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 603, 16), 'arguments', str_130287)
                # SSA join for if statement (line 599)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Type idiom detected: calculating its left and rigth part (line 605)
                str_130288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 605, 31), 'str', 'name')
                # Getting the type of 'object' (line 605)
                object_130289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 23), 'object')
                
                (may_be_130290, more_types_in_union_130291) = may_provide_member(str_130288, object_130289)

                if may_be_130290:

                    if more_types_in_union_130291:
                        # Runtime conditional SSA (line 605)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'object' (line 605)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'object', remove_not_member_provider_from_union(object_130289, 'name'))
                    
                    # Assigning a BinOp to a Name (line 606):
                    
                    # Assigning a BinOp to a Name (line 606):
                    str_130292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 23), 'str', '%s')
                    # Getting the type of 'object' (line 606)
                    object_130293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 30), 'object')
                    # Obtaining the member 'name' of a type (line 606)
                    name_130294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 30), object_130293, 'name')
                    # Applying the binary operator '%' (line 606)
                    result_mod_130295 = python_operator(stypy.reporting.localization.Localization(__file__, 606, 23), '%', str_130292, name_130294)
                    
                    # Assigning a type to the variable 'name' (line 606)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 16), 'name', result_mod_130295)

                    if more_types_in_union_130291:
                        # Runtime conditional SSA for else branch (line 605)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_130290) or more_types_in_union_130291):
                    # Assigning a type to the variable 'object' (line 605)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'object', remove_member_provider_from_union(object_130289, 'name'))
                    
                    # Assigning a Str to a Name (line 608):
                    
                    # Assigning a Str to a Name (line 608):
                    str_130296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 608, 23), 'str', '<name>')
                    # Assigning a type to the variable 'name' (line 608)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 608, 16), 'name', str_130296)

                    if (may_be_130290 and more_types_in_union_130291):
                        # SSA join for if statement (line 605)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                
                
                # Call to len(...): (line 609)
                # Processing the call arguments (line 609)
                # Getting the type of 'name' (line 609)
                name_130298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 19), 'name', False)
                # Getting the type of 'arguments' (line 609)
                arguments_130299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 24), 'arguments', False)
                # Applying the binary operator '+' (line 609)
                result_add_130300 = python_operator(stypy.reporting.localization.Localization(__file__, 609, 19), '+', name_130298, arguments_130299)
                
                # Processing the call keyword arguments (line 609)
                kwargs_130301 = {}
                # Getting the type of 'len' (line 609)
                len_130297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 15), 'len', False)
                # Calling len(args, kwargs) (line 609)
                len_call_result_130302 = invoke(stypy.reporting.localization.Localization(__file__, 609, 15), len_130297, *[result_add_130300], **kwargs_130301)
                
                # Getting the type of 'maxwidth' (line 609)
                maxwidth_130303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 37), 'maxwidth')
                # Applying the binary operator '>' (line 609)
                result_gt_130304 = python_operator(stypy.reporting.localization.Localization(__file__, 609, 15), '>', len_call_result_130302, maxwidth_130303)
                
                # Testing the type of an if condition (line 609)
                if_condition_130305 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 609, 12), result_gt_130304)
                # Assigning a type to the variable 'if_condition_130305' (line 609)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 12), 'if_condition_130305', if_condition_130305)
                # SSA begins for if statement (line 609)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 610):
                
                # Assigning a Call to a Name (line 610):
                
                # Call to _split_line(...): (line 610)
                # Processing the call arguments (line 610)
                # Getting the type of 'name' (line 610)
                name_130307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 37), 'name', False)
                # Getting the type of 'arguments' (line 610)
                arguments_130308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 43), 'arguments', False)
                # Getting the type of 'maxwidth' (line 610)
                maxwidth_130309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 54), 'maxwidth', False)
                # Processing the call keyword arguments (line 610)
                kwargs_130310 = {}
                # Getting the type of '_split_line' (line 610)
                _split_line_130306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 25), '_split_line', False)
                # Calling _split_line(args, kwargs) (line 610)
                _split_line_call_result_130311 = invoke(stypy.reporting.localization.Localization(__file__, 610, 25), _split_line_130306, *[name_130307, arguments_130308, maxwidth_130309], **kwargs_130310)
                
                # Assigning a type to the variable 'argstr' (line 610)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 16), 'argstr', _split_line_call_result_130311)
                # SSA branch for the else part of an if statement (line 609)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a BinOp to a Name (line 612):
                
                # Assigning a BinOp to a Name (line 612):
                # Getting the type of 'name' (line 612)
                name_130312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 25), 'name')
                # Getting the type of 'arguments' (line 612)
                arguments_130313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 32), 'arguments')
                # Applying the binary operator '+' (line 612)
                result_add_130314 = python_operator(stypy.reporting.localization.Localization(__file__, 612, 25), '+', name_130312, arguments_130313)
                
                # Assigning a type to the variable 'argstr' (line 612)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 16), 'argstr', result_add_130314)
                # SSA join for if statement (line 609)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Call to print(...): (line 614)
                # Processing the call arguments (line 614)
                str_130316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 18), 'str', ' ')
                # Getting the type of 'argstr' (line 614)
                argstr_130317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 24), 'argstr', False)
                # Applying the binary operator '+' (line 614)
                result_add_130318 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 18), '+', str_130316, argstr_130317)
                
                str_130319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 33), 'str', '\n')
                # Applying the binary operator '+' (line 614)
                result_add_130320 = python_operator(stypy.reporting.localization.Localization(__file__, 614, 31), '+', result_add_130318, str_130319)
                
                # Processing the call keyword arguments (line 614)
                # Getting the type of 'output' (line 614)
                output_130321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 44), 'output', False)
                keyword_130322 = output_130321
                kwargs_130323 = {'file': keyword_130322}
                # Getting the type of 'print' (line 614)
                print_130315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 12), 'print', False)
                # Calling print(args, kwargs) (line 614)
                print_call_result_130324 = invoke(stypy.reporting.localization.Localization(__file__, 614, 12), print_130315, *[result_add_130320], **kwargs_130323)
                
                
                # Assigning a Call to a Name (line 615):
                
                # Assigning a Call to a Name (line 615):
                
                # Call to getdoc(...): (line 615)
                # Processing the call arguments (line 615)
                # Getting the type of 'object' (line 615)
                object_130327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 33), 'object', False)
                # Obtaining the member '__call__' of a type (line 615)
                call___130328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 33), object_130327, '__call__')
                # Processing the call keyword arguments (line 615)
                kwargs_130329 = {}
                # Getting the type of 'inspect' (line 615)
                inspect_130325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 18), 'inspect', False)
                # Obtaining the member 'getdoc' of a type (line 615)
                getdoc_130326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 18), inspect_130325, 'getdoc')
                # Calling getdoc(args, kwargs) (line 615)
                getdoc_call_result_130330 = invoke(stypy.reporting.localization.Localization(__file__, 615, 18), getdoc_130326, *[call___130328], **kwargs_130329)
                
                # Assigning a type to the variable 'doc' (line 615)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'doc', getdoc_call_result_130330)
                
                # Type idiom detected: calculating its left and rigth part (line 616)
                # Getting the type of 'doc' (line 616)
                doc_130331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'doc')
                # Getting the type of 'None' (line 616)
                None_130332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 26), 'None')
                
                (may_be_130333, more_types_in_union_130334) = may_not_be_none(doc_130331, None_130332)

                if may_be_130333:

                    if more_types_in_union_130334:
                        # Runtime conditional SSA (line 616)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Call to print(...): (line 617)
                    # Processing the call arguments (line 617)
                    
                    # Call to getdoc(...): (line 617)
                    # Processing the call arguments (line 617)
                    # Getting the type of 'object' (line 617)
                    object_130338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 37), 'object', False)
                    # Obtaining the member '__call__' of a type (line 617)
                    call___130339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 37), object_130338, '__call__')
                    # Processing the call keyword arguments (line 617)
                    kwargs_130340 = {}
                    # Getting the type of 'inspect' (line 617)
                    inspect_130336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 22), 'inspect', False)
                    # Obtaining the member 'getdoc' of a type (line 617)
                    getdoc_130337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 617, 22), inspect_130336, 'getdoc')
                    # Calling getdoc(args, kwargs) (line 617)
                    getdoc_call_result_130341 = invoke(stypy.reporting.localization.Localization(__file__, 617, 22), getdoc_130337, *[call___130339], **kwargs_130340)
                    
                    # Processing the call keyword arguments (line 617)
                    # Getting the type of 'output' (line 617)
                    output_130342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 60), 'output', False)
                    keyword_130343 = output_130342
                    kwargs_130344 = {'file': keyword_130343}
                    # Getting the type of 'print' (line 617)
                    print_130335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 16), 'print', False)
                    # Calling print(args, kwargs) (line 617)
                    print_call_result_130345 = invoke(stypy.reporting.localization.Localization(__file__, 617, 16), print_130335, *[getdoc_call_result_130341], **kwargs_130344)
                    

                    if more_types_in_union_130334:
                        # SSA join for if statement (line 616)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                # Call to print(...): (line 618)
                # Processing the call arguments (line 618)
                
                # Call to getdoc(...): (line 618)
                # Processing the call arguments (line 618)
                # Getting the type of 'object' (line 618)
                object_130349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 33), 'object', False)
                # Processing the call keyword arguments (line 618)
                kwargs_130350 = {}
                # Getting the type of 'inspect' (line 618)
                inspect_130347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 18), 'inspect', False)
                # Obtaining the member 'getdoc' of a type (line 618)
                getdoc_130348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 18), inspect_130347, 'getdoc')
                # Calling getdoc(args, kwargs) (line 618)
                getdoc_call_result_130351 = invoke(stypy.reporting.localization.Localization(__file__, 618, 18), getdoc_130348, *[object_130349], **kwargs_130350)
                
                # Processing the call keyword arguments (line 618)
                # Getting the type of 'output' (line 618)
                output_130352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 47), 'output', False)
                keyword_130353 = output_130352
                kwargs_130354 = {'file': keyword_130353}
                # Getting the type of 'print' (line 618)
                print_130346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 12), 'print', False)
                # Calling print(args, kwargs) (line 618)
                print_call_result_130355 = invoke(stypy.reporting.localization.Localization(__file__, 618, 12), print_130346, *[getdoc_call_result_130351], **kwargs_130354)
                

                if more_types_in_union_130248:
                    # Runtime conditional SSA for else branch (line 594)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_130247) or more_types_in_union_130248):
                # Assigning a type to the variable 'object' (line 594)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 8), 'object', remove_member_provider_from_union(object_130246, '__call__'))
                
                # Call to print(...): (line 621)
                # Processing the call arguments (line 621)
                
                # Call to getdoc(...): (line 621)
                # Processing the call arguments (line 621)
                # Getting the type of 'object' (line 621)
                object_130359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 33), 'object', False)
                # Processing the call keyword arguments (line 621)
                kwargs_130360 = {}
                # Getting the type of 'inspect' (line 621)
                inspect_130357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 18), 'inspect', False)
                # Obtaining the member 'getdoc' of a type (line 621)
                getdoc_130358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 18), inspect_130357, 'getdoc')
                # Calling getdoc(args, kwargs) (line 621)
                getdoc_call_result_130361 = invoke(stypy.reporting.localization.Localization(__file__, 621, 18), getdoc_130358, *[object_130359], **kwargs_130360)
                
                # Processing the call keyword arguments (line 621)
                # Getting the type of 'output' (line 621)
                output_130362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 47), 'output', False)
                keyword_130363 = output_130362
                kwargs_130364 = {'file': keyword_130363}
                # Getting the type of 'print' (line 621)
                print_130356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 12), 'print', False)
                # Calling print(args, kwargs) (line 621)
                print_call_result_130365 = invoke(stypy.reporting.localization.Localization(__file__, 621, 12), print_130356, *[getdoc_call_result_130361], **kwargs_130364)
                

                if (may_be_130247 and more_types_in_union_130248):
                    # SSA join for if statement (line 594)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA branch for the else part of an if statement (line 588)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to ismethod(...): (line 623)
            # Processing the call arguments (line 623)
            # Getting the type of 'object' (line 623)
            object_130368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 26), 'object', False)
            # Processing the call keyword arguments (line 623)
            kwargs_130369 = {}
            # Getting the type of 'inspect' (line 623)
            inspect_130366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 9), 'inspect', False)
            # Obtaining the member 'ismethod' of a type (line 623)
            ismethod_130367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 9), inspect_130366, 'ismethod')
            # Calling ismethod(args, kwargs) (line 623)
            ismethod_call_result_130370 = invoke(stypy.reporting.localization.Localization(__file__, 623, 9), ismethod_130367, *[object_130368], **kwargs_130369)
            
            # Testing the type of an if condition (line 623)
            if_condition_130371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 623, 9), ismethod_call_result_130370)
            # Assigning a type to the variable 'if_condition_130371' (line 623)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 9), 'if_condition_130371', if_condition_130371)
            # SSA begins for if statement (line 623)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 624):
            
            # Assigning a Attribute to a Name (line 624):
            # Getting the type of 'object' (line 624)
            object_130372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 15), 'object')
            # Obtaining the member '__name__' of a type (line 624)
            name___130373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 15), object_130372, '__name__')
            # Assigning a type to the variable 'name' (line 624)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 8), 'name', name___130373)
            
            # Assigning a Call to a Name (line 625):
            
            # Assigning a Call to a Name (line 625):
            
            # Call to formatargspec(...): (line 625)
            
            # Call to getargspec(...): (line 626)
            # Processing the call arguments (line 626)
            # Getting the type of 'object' (line 626)
            object_130376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 28), 'object', False)
            # Obtaining the member '__func__' of a type (line 626)
            func___130377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 28), object_130376, '__func__')
            # Processing the call keyword arguments (line 626)
            kwargs_130378 = {}
            # Getting the type of 'getargspec' (line 626)
            getargspec_130375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 17), 'getargspec', False)
            # Calling getargspec(args, kwargs) (line 626)
            getargspec_call_result_130379 = invoke(stypy.reporting.localization.Localization(__file__, 626, 17), getargspec_130375, *[func___130377], **kwargs_130378)
            
            # Processing the call keyword arguments (line 625)
            kwargs_130380 = {}
            # Getting the type of 'formatargspec' (line 625)
            formatargspec_130374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 20), 'formatargspec', False)
            # Calling formatargspec(args, kwargs) (line 625)
            formatargspec_call_result_130381 = invoke(stypy.reporting.localization.Localization(__file__, 625, 20), formatargspec_130374, *[getargspec_call_result_130379], **kwargs_130380)
            
            # Assigning a type to the variable 'arguments' (line 625)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 625, 8), 'arguments', formatargspec_call_result_130381)
            
            # Assigning a Call to a Name (line 628):
            
            # Assigning a Call to a Name (line 628):
            
            # Call to split(...): (line 628)
            # Processing the call arguments (line 628)
            str_130384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 34), 'str', ', ')
            # Processing the call keyword arguments (line 628)
            kwargs_130385 = {}
            # Getting the type of 'arguments' (line 628)
            arguments_130382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 18), 'arguments', False)
            # Obtaining the member 'split' of a type (line 628)
            split_130383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 18), arguments_130382, 'split')
            # Calling split(args, kwargs) (line 628)
            split_call_result_130386 = invoke(stypy.reporting.localization.Localization(__file__, 628, 18), split_130383, *[str_130384], **kwargs_130385)
            
            # Assigning a type to the variable 'arglist' (line 628)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'arglist', split_call_result_130386)
            
            
            
            # Call to len(...): (line 629)
            # Processing the call arguments (line 629)
            # Getting the type of 'arglist' (line 629)
            arglist_130388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 15), 'arglist', False)
            # Processing the call keyword arguments (line 629)
            kwargs_130389 = {}
            # Getting the type of 'len' (line 629)
            len_130387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 11), 'len', False)
            # Calling len(args, kwargs) (line 629)
            len_call_result_130390 = invoke(stypy.reporting.localization.Localization(__file__, 629, 11), len_130387, *[arglist_130388], **kwargs_130389)
            
            int_130391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 629, 26), 'int')
            # Applying the binary operator '>' (line 629)
            result_gt_130392 = python_operator(stypy.reporting.localization.Localization(__file__, 629, 11), '>', len_call_result_130390, int_130391)
            
            # Testing the type of an if condition (line 629)
            if_condition_130393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 629, 8), result_gt_130392)
            # Assigning a type to the variable 'if_condition_130393' (line 629)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'if_condition_130393', if_condition_130393)
            # SSA begins for if statement (line 629)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Subscript (line 630):
            
            # Assigning a BinOp to a Subscript (line 630):
            str_130394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 25), 'str', '(')
            
            # Obtaining the type of the subscript
            int_130395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 37), 'int')
            # Getting the type of 'arglist' (line 630)
            arglist_130396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 29), 'arglist')
            # Obtaining the member '__getitem__' of a type (line 630)
            getitem___130397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 29), arglist_130396, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 630)
            subscript_call_result_130398 = invoke(stypy.reporting.localization.Localization(__file__, 630, 29), getitem___130397, int_130395)
            
            # Applying the binary operator '+' (line 630)
            result_add_130399 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 25), '+', str_130394, subscript_call_result_130398)
            
            # Getting the type of 'arglist' (line 630)
            arglist_130400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 12), 'arglist')
            int_130401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 20), 'int')
            # Storing an element on a container (line 630)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 630, 12), arglist_130400, (int_130401, result_add_130399))
            
            # Assigning a Call to a Name (line 631):
            
            # Assigning a Call to a Name (line 631):
            
            # Call to join(...): (line 631)
            # Processing the call arguments (line 631)
            
            # Obtaining the type of the subscript
            int_130404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 42), 'int')
            slice_130405 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 631, 34), int_130404, None, None)
            # Getting the type of 'arglist' (line 631)
            arglist_130406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 34), 'arglist', False)
            # Obtaining the member '__getitem__' of a type (line 631)
            getitem___130407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 34), arglist_130406, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 631)
            subscript_call_result_130408 = invoke(stypy.reporting.localization.Localization(__file__, 631, 34), getitem___130407, slice_130405)
            
            # Processing the call keyword arguments (line 631)
            kwargs_130409 = {}
            str_130402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 24), 'str', ', ')
            # Obtaining the member 'join' of a type (line 631)
            join_130403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 24), str_130402, 'join')
            # Calling join(args, kwargs) (line 631)
            join_call_result_130410 = invoke(stypy.reporting.localization.Localization(__file__, 631, 24), join_130403, *[subscript_call_result_130408], **kwargs_130409)
            
            # Assigning a type to the variable 'arguments' (line 631)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 12), 'arguments', join_call_result_130410)
            # SSA branch for the else part of an if statement (line 629)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Str to a Name (line 633):
            
            # Assigning a Str to a Name (line 633):
            str_130411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 24), 'str', '()')
            # Assigning a type to the variable 'arguments' (line 633)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 633, 12), 'arguments', str_130411)
            # SSA join for if statement (line 629)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            
            # Call to len(...): (line 635)
            # Processing the call arguments (line 635)
            # Getting the type of 'name' (line 635)
            name_130413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 15), 'name', False)
            # Getting the type of 'arguments' (line 635)
            arguments_130414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 20), 'arguments', False)
            # Applying the binary operator '+' (line 635)
            result_add_130415 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 15), '+', name_130413, arguments_130414)
            
            # Processing the call keyword arguments (line 635)
            kwargs_130416 = {}
            # Getting the type of 'len' (line 635)
            len_130412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 11), 'len', False)
            # Calling len(args, kwargs) (line 635)
            len_call_result_130417 = invoke(stypy.reporting.localization.Localization(__file__, 635, 11), len_130412, *[result_add_130415], **kwargs_130416)
            
            # Getting the type of 'maxwidth' (line 635)
            maxwidth_130418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 33), 'maxwidth')
            # Applying the binary operator '>' (line 635)
            result_gt_130419 = python_operator(stypy.reporting.localization.Localization(__file__, 635, 11), '>', len_call_result_130417, maxwidth_130418)
            
            # Testing the type of an if condition (line 635)
            if_condition_130420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 635, 8), result_gt_130419)
            # Assigning a type to the variable 'if_condition_130420' (line 635)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'if_condition_130420', if_condition_130420)
            # SSA begins for if statement (line 635)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 636):
            
            # Assigning a Call to a Name (line 636):
            
            # Call to _split_line(...): (line 636)
            # Processing the call arguments (line 636)
            # Getting the type of 'name' (line 636)
            name_130422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 33), 'name', False)
            # Getting the type of 'arguments' (line 636)
            arguments_130423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 39), 'arguments', False)
            # Getting the type of 'maxwidth' (line 636)
            maxwidth_130424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 50), 'maxwidth', False)
            # Processing the call keyword arguments (line 636)
            kwargs_130425 = {}
            # Getting the type of '_split_line' (line 636)
            _split_line_130421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 21), '_split_line', False)
            # Calling _split_line(args, kwargs) (line 636)
            _split_line_call_result_130426 = invoke(stypy.reporting.localization.Localization(__file__, 636, 21), _split_line_130421, *[name_130422, arguments_130423, maxwidth_130424], **kwargs_130425)
            
            # Assigning a type to the variable 'argstr' (line 636)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 636, 12), 'argstr', _split_line_call_result_130426)
            # SSA branch for the else part of an if statement (line 635)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 638):
            
            # Assigning a BinOp to a Name (line 638):
            # Getting the type of 'name' (line 638)
            name_130427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 21), 'name')
            # Getting the type of 'arguments' (line 638)
            arguments_130428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 28), 'arguments')
            # Applying the binary operator '+' (line 638)
            result_add_130429 = python_operator(stypy.reporting.localization.Localization(__file__, 638, 21), '+', name_130427, arguments_130428)
            
            # Assigning a type to the variable 'argstr' (line 638)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 12), 'argstr', result_add_130429)
            # SSA join for if statement (line 635)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to print(...): (line 640)
            # Processing the call arguments (line 640)
            str_130431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 14), 'str', ' ')
            # Getting the type of 'argstr' (line 640)
            argstr_130432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 20), 'argstr', False)
            # Applying the binary operator '+' (line 640)
            result_add_130433 = python_operator(stypy.reporting.localization.Localization(__file__, 640, 14), '+', str_130431, argstr_130432)
            
            str_130434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 640, 29), 'str', '\n')
            # Applying the binary operator '+' (line 640)
            result_add_130435 = python_operator(stypy.reporting.localization.Localization(__file__, 640, 27), '+', result_add_130433, str_130434)
            
            # Processing the call keyword arguments (line 640)
            # Getting the type of 'output' (line 640)
            output_130436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 40), 'output', False)
            keyword_130437 = output_130436
            kwargs_130438 = {'file': keyword_130437}
            # Getting the type of 'print' (line 640)
            print_130430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'print', False)
            # Calling print(args, kwargs) (line 640)
            print_call_result_130439 = invoke(stypy.reporting.localization.Localization(__file__, 640, 8), print_130430, *[result_add_130435], **kwargs_130438)
            
            
            # Call to print(...): (line 641)
            # Processing the call arguments (line 641)
            
            # Call to getdoc(...): (line 641)
            # Processing the call arguments (line 641)
            # Getting the type of 'object' (line 641)
            object_130443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 29), 'object', False)
            # Processing the call keyword arguments (line 641)
            kwargs_130444 = {}
            # Getting the type of 'inspect' (line 641)
            inspect_130441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 14), 'inspect', False)
            # Obtaining the member 'getdoc' of a type (line 641)
            getdoc_130442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 14), inspect_130441, 'getdoc')
            # Calling getdoc(args, kwargs) (line 641)
            getdoc_call_result_130445 = invoke(stypy.reporting.localization.Localization(__file__, 641, 14), getdoc_130442, *[object_130443], **kwargs_130444)
            
            # Processing the call keyword arguments (line 641)
            # Getting the type of 'output' (line 641)
            output_130446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 43), 'output', False)
            keyword_130447 = output_130446
            kwargs_130448 = {'file': keyword_130447}
            # Getting the type of 'print' (line 641)
            print_130440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'print', False)
            # Calling print(args, kwargs) (line 641)
            print_call_result_130449 = invoke(stypy.reporting.localization.Localization(__file__, 641, 8), print_130440, *[getdoc_call_result_130445], **kwargs_130448)
            
            # SSA branch for the else part of an if statement (line 623)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 643)
            str_130450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 25), 'str', '__doc__')
            # Getting the type of 'object' (line 643)
            object_130451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 17), 'object')
            
            (may_be_130452, more_types_in_union_130453) = may_provide_member(str_130450, object_130451)

            if may_be_130452:

                if more_types_in_union_130453:
                    # Runtime conditional SSA (line 643)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'object' (line 643)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 9), 'object', remove_not_member_provider_from_union(object_130451, '__doc__'))
                
                # Call to print(...): (line 644)
                # Processing the call arguments (line 644)
                
                # Call to getdoc(...): (line 644)
                # Processing the call arguments (line 644)
                # Getting the type of 'object' (line 644)
                object_130457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 29), 'object', False)
                # Processing the call keyword arguments (line 644)
                kwargs_130458 = {}
                # Getting the type of 'inspect' (line 644)
                inspect_130455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 14), 'inspect', False)
                # Obtaining the member 'getdoc' of a type (line 644)
                getdoc_130456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 644, 14), inspect_130455, 'getdoc')
                # Calling getdoc(args, kwargs) (line 644)
                getdoc_call_result_130459 = invoke(stypy.reporting.localization.Localization(__file__, 644, 14), getdoc_130456, *[object_130457], **kwargs_130458)
                
                # Processing the call keyword arguments (line 644)
                # Getting the type of 'output' (line 644)
                output_130460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 43), 'output', False)
                keyword_130461 = output_130460
                kwargs_130462 = {'file': keyword_130461}
                # Getting the type of 'print' (line 644)
                print_130454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 644, 8), 'print', False)
                # Calling print(args, kwargs) (line 644)
                print_call_result_130463 = invoke(stypy.reporting.localization.Localization(__file__, 644, 8), print_130454, *[getdoc_call_result_130459], **kwargs_130462)
                

                if more_types_in_union_130453:
                    # SSA join for if statement (line 643)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 623)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 588)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 547)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 535)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_129884 and more_types_in_union_129885):
                # SSA join for if statement (line 506)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 504)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_129864 and more_types_in_union_129865):
            # SSA join for if statement (line 502)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'info(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'info' in the type store
    # Getting the type of 'stypy_return_type' (line 442)
    stypy_return_type_130464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_130464)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'info'
    return stypy_return_type_130464

# Assigning a type to the variable 'info' (line 442)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 0), 'info', info)

@norecursion
def source(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'sys' (line 647)
    sys_130465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 26), 'sys')
    # Obtaining the member 'stdout' of a type (line 647)
    stdout_130466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 647, 26), sys_130465, 'stdout')
    defaults = [stdout_130466]
    # Create a new context for function 'source'
    module_type_store = module_type_store.open_function_context('source', 647, 0, False)
    
    # Passed parameters checking function
    source.stypy_localization = localization
    source.stypy_type_of_self = None
    source.stypy_type_store = module_type_store
    source.stypy_function_name = 'source'
    source.stypy_param_names_list = ['object', 'output']
    source.stypy_varargs_param_name = None
    source.stypy_kwargs_param_name = None
    source.stypy_call_defaults = defaults
    source.stypy_call_varargs = varargs
    source.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'source', ['object', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'source', localization, ['object', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'source(...)' code ##################

    str_130467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, (-1)), 'str', '\n    Print or write to a file the source code for a Numpy object.\n\n    The source code is only returned for objects written in Python. Many\n    functions and classes are defined in C and will therefore not return\n    useful information.\n\n    Parameters\n    ----------\n    object : numpy object\n        Input object. This can be any object (function, class, module,\n        ...).\n    output : file object, optional\n        If `output` not supplied then source code is printed to screen\n        (sys.stdout).  File object must be created with either write \'w\' or\n        append \'a\' modes.\n\n    See Also\n    --------\n    lookfor, info\n\n    Examples\n    --------\n    >>> np.source(np.interp)                        #doctest: +SKIP\n    In file: /usr/lib/python2.6/dist-packages/numpy/lib/function_base.py\n    def interp(x, xp, fp, left=None, right=None):\n        """.... (full docstring printed)"""\n        if isinstance(x, (float, int, number)):\n            return compiled_interp([x], xp, fp, left, right).item()\n        else:\n            return compiled_interp(x, xp, fp, left, right)\n\n    The source code is only returned for objects written in Python.\n\n    >>> np.source(np.array)                         #doctest: +SKIP\n    Not available for this object.\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 687, 4))
    
    # 'import inspect' statement (line 687)
    import inspect

    import_module(stypy.reporting.localization.Localization(__file__, 687, 4), 'inspect', inspect, module_type_store)
    
    
    
    # SSA begins for try-except statement (line 688)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to print(...): (line 689)
    # Processing the call arguments (line 689)
    str_130469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 14), 'str', 'In file: %s\n')
    
    # Call to getsourcefile(...): (line 689)
    # Processing the call arguments (line 689)
    # Getting the type of 'object' (line 689)
    object_130472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 54), 'object', False)
    # Processing the call keyword arguments (line 689)
    kwargs_130473 = {}
    # Getting the type of 'inspect' (line 689)
    inspect_130470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 32), 'inspect', False)
    # Obtaining the member 'getsourcefile' of a type (line 689)
    getsourcefile_130471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 689, 32), inspect_130470, 'getsourcefile')
    # Calling getsourcefile(args, kwargs) (line 689)
    getsourcefile_call_result_130474 = invoke(stypy.reporting.localization.Localization(__file__, 689, 32), getsourcefile_130471, *[object_130472], **kwargs_130473)
    
    # Applying the binary operator '%' (line 689)
    result_mod_130475 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 14), '%', str_130469, getsourcefile_call_result_130474)
    
    # Processing the call keyword arguments (line 689)
    # Getting the type of 'output' (line 689)
    output_130476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 68), 'output', False)
    keyword_130477 = output_130476
    kwargs_130478 = {'file': keyword_130477}
    # Getting the type of 'print' (line 689)
    print_130468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 8), 'print', False)
    # Calling print(args, kwargs) (line 689)
    print_call_result_130479 = invoke(stypy.reporting.localization.Localization(__file__, 689, 8), print_130468, *[result_mod_130475], **kwargs_130478)
    
    
    # Call to print(...): (line 690)
    # Processing the call arguments (line 690)
    
    # Call to getsource(...): (line 690)
    # Processing the call arguments (line 690)
    # Getting the type of 'object' (line 690)
    object_130483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 32), 'object', False)
    # Processing the call keyword arguments (line 690)
    kwargs_130484 = {}
    # Getting the type of 'inspect' (line 690)
    inspect_130481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 14), 'inspect', False)
    # Obtaining the member 'getsource' of a type (line 690)
    getsource_130482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 690, 14), inspect_130481, 'getsource')
    # Calling getsource(args, kwargs) (line 690)
    getsource_call_result_130485 = invoke(stypy.reporting.localization.Localization(__file__, 690, 14), getsource_130482, *[object_130483], **kwargs_130484)
    
    # Processing the call keyword arguments (line 690)
    # Getting the type of 'output' (line 690)
    output_130486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 46), 'output', False)
    keyword_130487 = output_130486
    kwargs_130488 = {'file': keyword_130487}
    # Getting the type of 'print' (line 690)
    print_130480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 8), 'print', False)
    # Calling print(args, kwargs) (line 690)
    print_call_result_130489 = invoke(stypy.reporting.localization.Localization(__file__, 690, 8), print_130480, *[getsource_call_result_130485], **kwargs_130488)
    
    # SSA branch for the except part of a try statement (line 688)
    # SSA branch for the except '<any exception>' branch of a try statement (line 688)
    module_type_store.open_ssa_branch('except')
    
    # Call to print(...): (line 692)
    # Processing the call arguments (line 692)
    str_130491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 692, 14), 'str', 'Not available for this object.')
    # Processing the call keyword arguments (line 692)
    # Getting the type of 'output' (line 692)
    output_130492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 53), 'output', False)
    keyword_130493 = output_130492
    kwargs_130494 = {'file': keyword_130493}
    # Getting the type of 'print' (line 692)
    print_130490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 8), 'print', False)
    # Calling print(args, kwargs) (line 692)
    print_call_result_130495 = invoke(stypy.reporting.localization.Localization(__file__, 692, 8), print_130490, *[str_130491], **kwargs_130494)
    
    # SSA join for try-except statement (line 688)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'source(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'source' in the type store
    # Getting the type of 'stypy_return_type' (line 647)
    stypy_return_type_130496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_130496)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'source'
    return stypy_return_type_130496

# Assigning a type to the variable 'source' (line 647)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 0), 'source', source)

# Assigning a Dict to a Name (line 698):

# Assigning a Dict to a Name (line 698):

# Obtaining an instance of the builtin type 'dict' (line 698)
dict_130497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 698)

# Assigning a type to the variable '_lookfor_caches' (line 698)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 0), '_lookfor_caches', dict_130497)

# Assigning a Call to a Name (line 702):

# Assigning a Call to a Name (line 702):

# Call to compile(...): (line 702)
# Processing the call arguments (line 702)
str_130500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 702, 36), 'str', '[a-z0-9_]+\\(.*[,=].*\\)')
# Getting the type of 're' (line 702)
re_130501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 63), 're', False)
# Obtaining the member 'I' of a type (line 702)
I_130502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 63), re_130501, 'I')
# Processing the call keyword arguments (line 702)
kwargs_130503 = {}
# Getting the type of 're' (line 702)
re_130498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 702, 25), 're', False)
# Obtaining the member 'compile' of a type (line 702)
compile_130499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 702, 25), re_130498, 'compile')
# Calling compile(args, kwargs) (line 702)
compile_call_result_130504 = invoke(stypy.reporting.localization.Localization(__file__, 702, 25), compile_130499, *[str_130500, I_130502], **kwargs_130503)

# Assigning a type to the variable '_function_signature_re' (line 702)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 702, 0), '_function_signature_re', compile_call_result_130504)

@norecursion
def lookfor(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 704)
    None_130505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 25), 'None')
    # Getting the type of 'True' (line 704)
    True_130506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 46), 'True')
    # Getting the type of 'False' (line 704)
    False_130507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 63), 'False')
    # Getting the type of 'None' (line 705)
    None_130508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 705, 19), 'None')
    defaults = [None_130505, True_130506, False_130507, None_130508]
    # Create a new context for function 'lookfor'
    module_type_store = module_type_store.open_function_context('lookfor', 704, 0, False)
    
    # Passed parameters checking function
    lookfor.stypy_localization = localization
    lookfor.stypy_type_of_self = None
    lookfor.stypy_type_store = module_type_store
    lookfor.stypy_function_name = 'lookfor'
    lookfor.stypy_param_names_list = ['what', 'module', 'import_modules', 'regenerate', 'output']
    lookfor.stypy_varargs_param_name = None
    lookfor.stypy_kwargs_param_name = None
    lookfor.stypy_call_defaults = defaults
    lookfor.stypy_call_varargs = varargs
    lookfor.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lookfor', ['what', 'module', 'import_modules', 'regenerate', 'output'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lookfor', localization, ['what', 'module', 'import_modules', 'regenerate', 'output'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lookfor(...)' code ##################

    str_130509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 749, (-1)), 'str', "\n    Do a keyword search on docstrings.\n\n    A list of of objects that matched the search is displayed,\n    sorted by relevance. All given keywords need to be found in the\n    docstring for it to be returned as a result, but the order does\n    not matter.\n\n    Parameters\n    ----------\n    what : str\n        String containing words to look for.\n    module : str or list, optional\n        Name of module(s) whose docstrings to go through.\n    import_modules : bool, optional\n        Whether to import sub-modules in packages. Default is True.\n    regenerate : bool, optional\n        Whether to re-generate the docstring cache. Default is False.\n    output : file-like, optional\n        File-like object to write the output to. If omitted, use a pager.\n\n    See Also\n    --------\n    source, info\n\n    Notes\n    -----\n    Relevance is determined only roughly, by checking if the keywords occur\n    in the function name, at the start of a docstring, etc.\n\n    Examples\n    --------\n    >>> np.lookfor('binary representation')\n    Search results for 'binary representation'\n    ------------------------------------------\n    numpy.binary_repr\n        Return the binary representation of the input number as a string.\n    numpy.core.setup_common.long_double_representation\n        Given a binary dump as given by GNU od -b, look for long double\n    numpy.base_repr\n        Return a string representation of a number in the given base system.\n    ...\n\n    ")
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 750, 4))
    
    # 'import pydoc' statement (line 750)
    import pydoc

    import_module(stypy.reporting.localization.Localization(__file__, 750, 4), 'pydoc', pydoc, module_type_store)
    
    
    # Assigning a Call to a Name (line 753):
    
    # Assigning a Call to a Name (line 753):
    
    # Call to _lookfor_generate_cache(...): (line 753)
    # Processing the call arguments (line 753)
    # Getting the type of 'module' (line 753)
    module_130511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 36), 'module', False)
    # Getting the type of 'import_modules' (line 753)
    import_modules_130512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 44), 'import_modules', False)
    # Getting the type of 'regenerate' (line 753)
    regenerate_130513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 60), 'regenerate', False)
    # Processing the call keyword arguments (line 753)
    kwargs_130514 = {}
    # Getting the type of '_lookfor_generate_cache' (line 753)
    _lookfor_generate_cache_130510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 753, 12), '_lookfor_generate_cache', False)
    # Calling _lookfor_generate_cache(args, kwargs) (line 753)
    _lookfor_generate_cache_call_result_130515 = invoke(stypy.reporting.localization.Localization(__file__, 753, 12), _lookfor_generate_cache_130510, *[module_130511, import_modules_130512, regenerate_130513], **kwargs_130514)
    
    # Assigning a type to the variable 'cache' (line 753)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 753, 4), 'cache', _lookfor_generate_cache_call_result_130515)
    
    # Assigning a List to a Name (line 757):
    
    # Assigning a List to a Name (line 757):
    
    # Obtaining an instance of the builtin type 'list' (line 757)
    list_130516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 757, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 757)
    
    # Assigning a type to the variable 'found' (line 757)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 757, 4), 'found', list_130516)
    
    # Assigning a Call to a Name (line 758):
    
    # Assigning a Call to a Name (line 758):
    
    # Call to split(...): (line 758)
    # Processing the call keyword arguments (line 758)
    kwargs_130525 = {}
    
    # Call to lower(...): (line 758)
    # Processing the call keyword arguments (line 758)
    kwargs_130522 = {}
    
    # Call to str(...): (line 758)
    # Processing the call arguments (line 758)
    # Getting the type of 'what' (line 758)
    what_130518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 16), 'what', False)
    # Processing the call keyword arguments (line 758)
    kwargs_130519 = {}
    # Getting the type of 'str' (line 758)
    str_130517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 758, 12), 'str', False)
    # Calling str(args, kwargs) (line 758)
    str_call_result_130520 = invoke(stypy.reporting.localization.Localization(__file__, 758, 12), str_130517, *[what_130518], **kwargs_130519)
    
    # Obtaining the member 'lower' of a type (line 758)
    lower_130521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 12), str_call_result_130520, 'lower')
    # Calling lower(args, kwargs) (line 758)
    lower_call_result_130523 = invoke(stypy.reporting.localization.Localization(__file__, 758, 12), lower_130521, *[], **kwargs_130522)
    
    # Obtaining the member 'split' of a type (line 758)
    split_130524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 758, 12), lower_call_result_130523, 'split')
    # Calling split(args, kwargs) (line 758)
    split_call_result_130526 = invoke(stypy.reporting.localization.Localization(__file__, 758, 12), split_130524, *[], **kwargs_130525)
    
    # Assigning a type to the variable 'whats' (line 758)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 758, 4), 'whats', split_call_result_130526)
    
    
    # Getting the type of 'whats' (line 759)
    whats_130527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 759, 11), 'whats')
    # Applying the 'not' unary operator (line 759)
    result_not__130528 = python_operator(stypy.reporting.localization.Localization(__file__, 759, 7), 'not', whats_130527)
    
    # Testing the type of an if condition (line 759)
    if_condition_130529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 759, 4), result_not__130528)
    # Assigning a type to the variable 'if_condition_130529' (line 759)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 759, 4), 'if_condition_130529', if_condition_130529)
    # SSA begins for if statement (line 759)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 760)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 760, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 759)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to items(...): (line 762)
    # Processing the call keyword arguments (line 762)
    kwargs_130532 = {}
    # Getting the type of 'cache' (line 762)
    cache_130530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 762, 42), 'cache', False)
    # Obtaining the member 'items' of a type (line 762)
    items_130531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 762, 42), cache_130530, 'items')
    # Calling items(args, kwargs) (line 762)
    items_call_result_130533 = invoke(stypy.reporting.localization.Localization(__file__, 762, 42), items_130531, *[], **kwargs_130532)
    
    # Testing the type of a for loop iterable (line 762)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 762, 4), items_call_result_130533)
    # Getting the type of the for loop variable (line 762)
    for_loop_var_130534 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 762, 4), items_call_result_130533)
    # Assigning a type to the variable 'name' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 762, 4), for_loop_var_130534))
    # Assigning a type to the variable 'docstring' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 4), 'docstring', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 762, 4), for_loop_var_130534))
    # Assigning a type to the variable 'kind' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 4), 'kind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 762, 4), for_loop_var_130534))
    # Assigning a type to the variable 'index' (line 762)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 762, 4), 'index', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 762, 4), for_loop_var_130534))
    # SSA begins for a for statement (line 762)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'kind' (line 763)
    kind_130535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 763, 11), 'kind')
    
    # Obtaining an instance of the builtin type 'tuple' (line 763)
    tuple_130536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 763)
    # Adding element type (line 763)
    str_130537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 20), 'str', 'module')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 20), tuple_130536, str_130537)
    # Adding element type (line 763)
    str_130538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 763, 30), 'str', 'object')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 763, 20), tuple_130536, str_130538)
    
    # Applying the binary operator 'in' (line 763)
    result_contains_130539 = python_operator(stypy.reporting.localization.Localization(__file__, 763, 11), 'in', kind_130535, tuple_130536)
    
    # Testing the type of an if condition (line 763)
    if_condition_130540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 763, 8), result_contains_130539)
    # Assigning a type to the variable 'if_condition_130540' (line 763)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 763, 8), 'if_condition_130540', if_condition_130540)
    # SSA begins for if statement (line 763)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 763)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 766):
    
    # Assigning a Name to a Name (line 766):
    # Getting the type of 'True' (line 766)
    True_130541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 766, 13), 'True')
    # Assigning a type to the variable 'ok' (line 766)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 766, 8), 'ok', True_130541)
    
    # Assigning a Call to a Name (line 767):
    
    # Assigning a Call to a Name (line 767):
    
    # Call to lower(...): (line 767)
    # Processing the call keyword arguments (line 767)
    kwargs_130544 = {}
    # Getting the type of 'docstring' (line 767)
    docstring_130542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 767, 14), 'docstring', False)
    # Obtaining the member 'lower' of a type (line 767)
    lower_130543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 767, 14), docstring_130542, 'lower')
    # Calling lower(args, kwargs) (line 767)
    lower_call_result_130545 = invoke(stypy.reporting.localization.Localization(__file__, 767, 14), lower_130543, *[], **kwargs_130544)
    
    # Assigning a type to the variable 'doc' (line 767)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 767, 8), 'doc', lower_call_result_130545)
    
    # Getting the type of 'whats' (line 768)
    whats_130546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 768, 17), 'whats')
    # Testing the type of a for loop iterable (line 768)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 768, 8), whats_130546)
    # Getting the type of the for loop variable (line 768)
    for_loop_var_130547 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 768, 8), whats_130546)
    # Assigning a type to the variable 'w' (line 768)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 768, 8), 'w', for_loop_var_130547)
    # SSA begins for a for statement (line 768)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'w' (line 769)
    w_130548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 15), 'w')
    # Getting the type of 'doc' (line 769)
    doc_130549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 769, 24), 'doc')
    # Applying the binary operator 'notin' (line 769)
    result_contains_130550 = python_operator(stypy.reporting.localization.Localization(__file__, 769, 15), 'notin', w_130548, doc_130549)
    
    # Testing the type of an if condition (line 769)
    if_condition_130551 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 769, 12), result_contains_130550)
    # Assigning a type to the variable 'if_condition_130551' (line 769)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 769, 12), 'if_condition_130551', if_condition_130551)
    # SSA begins for if statement (line 769)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 770):
    
    # Assigning a Name to a Name (line 770):
    # Getting the type of 'False' (line 770)
    False_130552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 770, 21), 'False')
    # Assigning a type to the variable 'ok' (line 770)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 770, 16), 'ok', False_130552)
    # SSA join for if statement (line 769)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ok' (line 772)
    ok_130553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 772, 11), 'ok')
    # Testing the type of an if condition (line 772)
    if_condition_130554 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 772, 8), ok_130553)
    # Assigning a type to the variable 'if_condition_130554' (line 772)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 772, 8), 'if_condition_130554', if_condition_130554)
    # SSA begins for if statement (line 772)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 773)
    # Processing the call arguments (line 773)
    # Getting the type of 'name' (line 773)
    name_130557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 25), 'name', False)
    # Processing the call keyword arguments (line 773)
    kwargs_130558 = {}
    # Getting the type of 'found' (line 773)
    found_130555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 773, 12), 'found', False)
    # Obtaining the member 'append' of a type (line 773)
    append_130556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 773, 12), found_130555, 'append')
    # Calling append(args, kwargs) (line 773)
    append_call_result_130559 = invoke(stypy.reporting.localization.Localization(__file__, 773, 12), append_130556, *[name_130557], **kwargs_130558)
    
    # SSA join for if statement (line 772)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 779):
    
    # Assigning a Dict to a Name (line 779):
    
    # Obtaining an instance of the builtin type 'dict' (line 779)
    dict_130560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 21), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 779)
    # Adding element type (key, value) (line 779)
    str_130561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 22), 'str', 'func')
    int_130562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 30), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 779, 21), dict_130560, (str_130561, int_130562))
    # Adding element type (key, value) (line 779)
    str_130563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 36), 'str', 'class')
    int_130564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 779, 45), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 779, 21), dict_130560, (str_130563, int_130564))
    # Adding element type (key, value) (line 779)
    str_130565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 22), 'str', 'module')
    int_130566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 32), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 779, 21), dict_130560, (str_130565, int_130566))
    # Adding element type (key, value) (line 779)
    str_130567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 39), 'str', 'object')
    int_130568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 780, 49), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 779, 21), dict_130560, (str_130567, int_130568))
    
    # Assigning a type to the variable 'kind_relevance' (line 779)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 779, 4), 'kind_relevance', dict_130560)

    @norecursion
    def relevance(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'relevance'
        module_type_store = module_type_store.open_function_context('relevance', 782, 4, False)
        
        # Passed parameters checking function
        relevance.stypy_localization = localization
        relevance.stypy_type_of_self = None
        relevance.stypy_type_store = module_type_store
        relevance.stypy_function_name = 'relevance'
        relevance.stypy_param_names_list = ['name', 'docstr', 'kind', 'index']
        relevance.stypy_varargs_param_name = None
        relevance.stypy_kwargs_param_name = None
        relevance.stypy_call_defaults = defaults
        relevance.stypy_call_varargs = varargs
        relevance.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'relevance', ['name', 'docstr', 'kind', 'index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'relevance', localization, ['name', 'docstr', 'kind', 'index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'relevance(...)' code ##################

        
        # Assigning a Num to a Name (line 783):
        
        # Assigning a Num to a Name (line 783):
        int_130569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 783, 12), 'int')
        # Assigning a type to the variable 'r' (line 783)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 783, 8), 'r', int_130569)
        
        # Assigning a Call to a Name (line 785):
        
        # Assigning a Call to a Name (line 785):
        
        # Call to join(...): (line 785)
        # Processing the call arguments (line 785)
        
        # Obtaining the type of the subscript
        int_130572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 66), 'int')
        slice_130573 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 785, 30), None, int_130572, None)
        
        # Call to split(...): (line 785)
        # Processing the call arguments (line 785)
        str_130582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 59), 'str', '\n')
        # Processing the call keyword arguments (line 785)
        kwargs_130583 = {}
        
        # Call to strip(...): (line 785)
        # Processing the call keyword arguments (line 785)
        kwargs_130579 = {}
        
        # Call to lower(...): (line 785)
        # Processing the call keyword arguments (line 785)
        kwargs_130576 = {}
        # Getting the type of 'docstr' (line 785)
        docstr_130574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 785, 30), 'docstr', False)
        # Obtaining the member 'lower' of a type (line 785)
        lower_130575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 30), docstr_130574, 'lower')
        # Calling lower(args, kwargs) (line 785)
        lower_call_result_130577 = invoke(stypy.reporting.localization.Localization(__file__, 785, 30), lower_130575, *[], **kwargs_130576)
        
        # Obtaining the member 'strip' of a type (line 785)
        strip_130578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 30), lower_call_result_130577, 'strip')
        # Calling strip(args, kwargs) (line 785)
        strip_call_result_130580 = invoke(stypy.reporting.localization.Localization(__file__, 785, 30), strip_130578, *[], **kwargs_130579)
        
        # Obtaining the member 'split' of a type (line 785)
        split_130581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 30), strip_call_result_130580, 'split')
        # Calling split(args, kwargs) (line 785)
        split_call_result_130584 = invoke(stypy.reporting.localization.Localization(__file__, 785, 30), split_130581, *[str_130582], **kwargs_130583)
        
        # Obtaining the member '__getitem__' of a type (line 785)
        getitem___130585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 30), split_call_result_130584, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 785)
        subscript_call_result_130586 = invoke(stypy.reporting.localization.Localization(__file__, 785, 30), getitem___130585, slice_130573)
        
        # Processing the call keyword arguments (line 785)
        kwargs_130587 = {}
        str_130570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 785, 20), 'str', '\n')
        # Obtaining the member 'join' of a type (line 785)
        join_130571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 785, 20), str_130570, 'join')
        # Calling join(args, kwargs) (line 785)
        join_call_result_130588 = invoke(stypy.reporting.localization.Localization(__file__, 785, 20), join_130571, *[subscript_call_result_130586], **kwargs_130587)
        
        # Assigning a type to the variable 'first_doc' (line 785)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 785, 8), 'first_doc', join_call_result_130588)
        
        # Getting the type of 'r' (line 786)
        r_130589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'r')
        
        # Call to sum(...): (line 786)
        # Processing the call arguments (line 786)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'whats' (line 786)
        whats_130595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 31), 'whats', False)
        comprehension_130596 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 786, 18), whats_130595)
        # Assigning a type to the variable 'w' (line 786)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 18), 'w', comprehension_130596)
        
        # Getting the type of 'w' (line 786)
        w_130592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 40), 'w', False)
        # Getting the type of 'first_doc' (line 786)
        first_doc_130593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 45), 'first_doc', False)
        # Applying the binary operator 'in' (line 786)
        result_contains_130594 = python_operator(stypy.reporting.localization.Localization(__file__, 786, 40), 'in', w_130592, first_doc_130593)
        
        int_130591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 18), 'int')
        list_130597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 786, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 786, 18), list_130597, int_130591)
        # Processing the call keyword arguments (line 786)
        kwargs_130598 = {}
        # Getting the type of 'sum' (line 786)
        sum_130590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 786, 13), 'sum', False)
        # Calling sum(args, kwargs) (line 786)
        sum_call_result_130599 = invoke(stypy.reporting.localization.Localization(__file__, 786, 13), sum_130590, *[list_130597], **kwargs_130598)
        
        # Applying the binary operator '+=' (line 786)
        result_iadd_130600 = python_operator(stypy.reporting.localization.Localization(__file__, 786, 8), '+=', r_130589, sum_call_result_130599)
        # Assigning a type to the variable 'r' (line 786)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 786, 8), 'r', result_iadd_130600)
        
        
        # Getting the type of 'r' (line 788)
        r_130601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'r')
        
        # Call to sum(...): (line 788)
        # Processing the call arguments (line 788)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'whats' (line 788)
        whats_130607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 30), 'whats', False)
        comprehension_130608 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 18), whats_130607)
        # Assigning a type to the variable 'w' (line 788)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 18), 'w', comprehension_130608)
        
        # Getting the type of 'w' (line 788)
        w_130604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 39), 'w', False)
        # Getting the type of 'name' (line 788)
        name_130605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 44), 'name', False)
        # Applying the binary operator 'in' (line 788)
        result_contains_130606 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 39), 'in', w_130604, name_130605)
        
        int_130603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 18), 'int')
        list_130609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 788, 18), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 788, 18), list_130609, int_130603)
        # Processing the call keyword arguments (line 788)
        kwargs_130610 = {}
        # Getting the type of 'sum' (line 788)
        sum_130602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 788, 13), 'sum', False)
        # Calling sum(args, kwargs) (line 788)
        sum_call_result_130611 = invoke(stypy.reporting.localization.Localization(__file__, 788, 13), sum_130602, *[list_130609], **kwargs_130610)
        
        # Applying the binary operator '+=' (line 788)
        result_iadd_130612 = python_operator(stypy.reporting.localization.Localization(__file__, 788, 8), '+=', r_130601, sum_call_result_130611)
        # Assigning a type to the variable 'r' (line 788)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 788, 8), 'r', result_iadd_130612)
        
        
        # Getting the type of 'r' (line 790)
        r_130613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'r')
        
        
        # Call to len(...): (line 790)
        # Processing the call arguments (line 790)
        # Getting the type of 'name' (line 790)
        name_130615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 18), 'name', False)
        # Processing the call keyword arguments (line 790)
        kwargs_130616 = {}
        # Getting the type of 'len' (line 790)
        len_130614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 790, 14), 'len', False)
        # Calling len(args, kwargs) (line 790)
        len_call_result_130617 = invoke(stypy.reporting.localization.Localization(__file__, 790, 14), len_130614, *[name_130615], **kwargs_130616)
        
        # Applying the 'usub' unary operator (line 790)
        result___neg___130618 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 13), 'usub', len_call_result_130617)
        
        int_130619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 790, 26), 'int')
        # Applying the binary operator '*' (line 790)
        result_mul_130620 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 13), '*', result___neg___130618, int_130619)
        
        # Applying the binary operator '+=' (line 790)
        result_iadd_130621 = python_operator(stypy.reporting.localization.Localization(__file__, 790, 8), '+=', r_130613, result_mul_130620)
        # Assigning a type to the variable 'r' (line 790)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 790, 8), 'r', result_iadd_130621)
        
        
        # Getting the type of 'r' (line 792)
        r_130622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 8), 'r')
        
        # Call to get(...): (line 792)
        # Processing the call arguments (line 792)
        # Getting the type of 'kind' (line 792)
        kind_130625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 32), 'kind', False)
        int_130626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 792, 38), 'int')
        # Processing the call keyword arguments (line 792)
        kwargs_130627 = {}
        # Getting the type of 'kind_relevance' (line 792)
        kind_relevance_130623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 792, 13), 'kind_relevance', False)
        # Obtaining the member 'get' of a type (line 792)
        get_130624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 792, 13), kind_relevance_130623, 'get')
        # Calling get(args, kwargs) (line 792)
        get_call_result_130628 = invoke(stypy.reporting.localization.Localization(__file__, 792, 13), get_130624, *[kind_130625, int_130626], **kwargs_130627)
        
        # Applying the binary operator '+=' (line 792)
        result_iadd_130629 = python_operator(stypy.reporting.localization.Localization(__file__, 792, 8), '+=', r_130622, get_call_result_130628)
        # Assigning a type to the variable 'r' (line 792)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 792, 8), 'r', result_iadd_130629)
        
        
        # Getting the type of 'r' (line 794)
        r_130630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'r')
        
        
        # Call to count(...): (line 794)
        # Processing the call arguments (line 794)
        str_130633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 25), 'str', '.')
        # Processing the call keyword arguments (line 794)
        kwargs_130634 = {}
        # Getting the type of 'name' (line 794)
        name_130631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 794, 14), 'name', False)
        # Obtaining the member 'count' of a type (line 794)
        count_130632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 794, 14), name_130631, 'count')
        # Calling count(args, kwargs) (line 794)
        count_call_result_130635 = invoke(stypy.reporting.localization.Localization(__file__, 794, 14), count_130632, *[str_130633], **kwargs_130634)
        
        # Applying the 'usub' unary operator (line 794)
        result___neg___130636 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 13), 'usub', count_call_result_130635)
        
        int_130637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 794, 32), 'int')
        # Applying the binary operator '*' (line 794)
        result_mul_130638 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 13), '*', result___neg___130636, int_130637)
        
        # Applying the binary operator '+=' (line 794)
        result_iadd_130639 = python_operator(stypy.reporting.localization.Localization(__file__, 794, 8), '+=', r_130630, result_mul_130638)
        # Assigning a type to the variable 'r' (line 794)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 794, 8), 'r', result_iadd_130639)
        
        
        # Getting the type of 'r' (line 795)
        r_130640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 8), 'r')
        
        # Call to max(...): (line 795)
        # Processing the call arguments (line 795)
        
        # Getting the type of 'index' (line 795)
        index_130642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 18), 'index', False)
        # Applying the 'usub' unary operator (line 795)
        result___neg___130643 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 17), 'usub', index_130642)
        
        int_130644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 26), 'int')
        # Applying the binary operator 'div' (line 795)
        result_div_130645 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 17), 'div', result___neg___130643, int_130644)
        
        int_130646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 795, 31), 'int')
        # Processing the call keyword arguments (line 795)
        kwargs_130647 = {}
        # Getting the type of 'max' (line 795)
        max_130641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 795, 13), 'max', False)
        # Calling max(args, kwargs) (line 795)
        max_call_result_130648 = invoke(stypy.reporting.localization.Localization(__file__, 795, 13), max_130641, *[result_div_130645, int_130646], **kwargs_130647)
        
        # Applying the binary operator '+=' (line 795)
        result_iadd_130649 = python_operator(stypy.reporting.localization.Localization(__file__, 795, 8), '+=', r_130640, max_call_result_130648)
        # Assigning a type to the variable 'r' (line 795)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 795, 8), 'r', result_iadd_130649)
        
        # Getting the type of 'r' (line 796)
        r_130650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 796, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 796)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 796, 8), 'stypy_return_type', r_130650)
        
        # ################# End of 'relevance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'relevance' in the type store
        # Getting the type of 'stypy_return_type' (line 782)
        stypy_return_type_130651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 782, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130651)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'relevance'
        return stypy_return_type_130651

    # Assigning a type to the variable 'relevance' (line 782)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 782, 4), 'relevance', relevance)

    @norecursion
    def relevance_value(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'relevance_value'
        module_type_store = module_type_store.open_function_context('relevance_value', 798, 4, False)
        
        # Passed parameters checking function
        relevance_value.stypy_localization = localization
        relevance_value.stypy_type_of_self = None
        relevance_value.stypy_type_store = module_type_store
        relevance_value.stypy_function_name = 'relevance_value'
        relevance_value.stypy_param_names_list = ['a']
        relevance_value.stypy_varargs_param_name = None
        relevance_value.stypy_kwargs_param_name = None
        relevance_value.stypy_call_defaults = defaults
        relevance_value.stypy_call_varargs = varargs
        relevance_value.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'relevance_value', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'relevance_value', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'relevance_value(...)' code ##################

        
        # Call to relevance(...): (line 799)
        # Processing the call arguments (line 799)
        # Getting the type of 'a' (line 799)
        a_130653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 25), 'a', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'a' (line 799)
        a_130654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 35), 'a', False)
        # Getting the type of 'cache' (line 799)
        cache_130655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 29), 'cache', False)
        # Obtaining the member '__getitem__' of a type (line 799)
        getitem___130656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 799, 29), cache_130655, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 799)
        subscript_call_result_130657 = invoke(stypy.reporting.localization.Localization(__file__, 799, 29), getitem___130656, a_130654)
        
        # Processing the call keyword arguments (line 799)
        kwargs_130658 = {}
        # Getting the type of 'relevance' (line 799)
        relevance_130652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 799, 15), 'relevance', False)
        # Calling relevance(args, kwargs) (line 799)
        relevance_call_result_130659 = invoke(stypy.reporting.localization.Localization(__file__, 799, 15), relevance_130652, *[a_130653, subscript_call_result_130657], **kwargs_130658)
        
        # Assigning a type to the variable 'stypy_return_type' (line 799)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 799, 8), 'stypy_return_type', relevance_call_result_130659)
        
        # ################# End of 'relevance_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'relevance_value' in the type store
        # Getting the type of 'stypy_return_type' (line 798)
        stypy_return_type_130660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_130660)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'relevance_value'
        return stypy_return_type_130660

    # Assigning a type to the variable 'relevance_value' (line 798)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 798, 4), 'relevance_value', relevance_value)
    
    # Call to sort(...): (line 800)
    # Processing the call keyword arguments (line 800)
    # Getting the type of 'relevance_value' (line 800)
    relevance_value_130663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 19), 'relevance_value', False)
    keyword_130664 = relevance_value_130663
    kwargs_130665 = {'key': keyword_130664}
    # Getting the type of 'found' (line 800)
    found_130661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 800, 4), 'found', False)
    # Obtaining the member 'sort' of a type (line 800)
    sort_130662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 800, 4), found_130661, 'sort')
    # Calling sort(args, kwargs) (line 800)
    sort_call_result_130666 = invoke(stypy.reporting.localization.Localization(__file__, 800, 4), sort_130662, *[], **kwargs_130665)
    
    
    # Assigning a BinOp to a Name (line 803):
    
    # Assigning a BinOp to a Name (line 803):
    str_130667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 8), 'str', "Search results for '%s'")
    
    # Call to join(...): (line 803)
    # Processing the call arguments (line 803)
    # Getting the type of 'whats' (line 803)
    whats_130670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 803, 46), 'whats', False)
    # Processing the call keyword arguments (line 803)
    kwargs_130671 = {}
    str_130668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 803, 37), 'str', ' ')
    # Obtaining the member 'join' of a type (line 803)
    join_130669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 803, 37), str_130668, 'join')
    # Calling join(args, kwargs) (line 803)
    join_call_result_130672 = invoke(stypy.reporting.localization.Localization(__file__, 803, 37), join_130669, *[whats_130670], **kwargs_130671)
    
    # Applying the binary operator '%' (line 803)
    result_mod_130673 = python_operator(stypy.reporting.localization.Localization(__file__, 803, 8), '%', str_130667, join_call_result_130672)
    
    # Assigning a type to the variable 's' (line 803)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 803, 4), 's', result_mod_130673)
    
    # Assigning a List to a Name (line 804):
    
    # Assigning a List to a Name (line 804):
    
    # Obtaining an instance of the builtin type 'list' (line 804)
    list_130674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 804)
    # Adding element type (line 804)
    # Getting the type of 's' (line 804)
    s_130675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 17), 's')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 804, 16), list_130674, s_130675)
    # Adding element type (line 804)
    str_130676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 804, 20), 'str', '-')
    
    # Call to len(...): (line 804)
    # Processing the call arguments (line 804)
    # Getting the type of 's' (line 804)
    s_130678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 28), 's', False)
    # Processing the call keyword arguments (line 804)
    kwargs_130679 = {}
    # Getting the type of 'len' (line 804)
    len_130677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 804, 24), 'len', False)
    # Calling len(args, kwargs) (line 804)
    len_call_result_130680 = invoke(stypy.reporting.localization.Localization(__file__, 804, 24), len_130677, *[s_130678], **kwargs_130679)
    
    # Applying the binary operator '*' (line 804)
    result_mul_130681 = python_operator(stypy.reporting.localization.Localization(__file__, 804, 20), '*', str_130676, len_call_result_130680)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 804, 16), list_130674, result_mul_130681)
    
    # Assigning a type to the variable 'help_text' (line 804)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 804, 4), 'help_text', list_130674)
    
    
    # Obtaining the type of the subscript
    int_130682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 805, 24), 'int')
    slice_130683 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 805, 16), None, None, int_130682)
    # Getting the type of 'found' (line 805)
    found_130684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 805, 16), 'found')
    # Obtaining the member '__getitem__' of a type (line 805)
    getitem___130685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 805, 16), found_130684, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 805)
    subscript_call_result_130686 = invoke(stypy.reporting.localization.Localization(__file__, 805, 16), getitem___130685, slice_130683)
    
    # Testing the type of a for loop iterable (line 805)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 805, 4), subscript_call_result_130686)
    # Getting the type of the for loop variable (line 805)
    for_loop_var_130687 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 805, 4), subscript_call_result_130686)
    # Assigning a type to the variable 'name' (line 805)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 805, 4), 'name', for_loop_var_130687)
    # SSA begins for a for statement (line 805)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Tuple (line 806):
    
    # Assigning a Subscript to a Name (line 806):
    
    # Obtaining the type of the subscript
    int_130688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 806)
    name_130689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 30), 'name')
    # Getting the type of 'cache' (line 806)
    cache_130690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 24), 'cache')
    # Obtaining the member '__getitem__' of a type (line 806)
    getitem___130691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 24), cache_130690, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 806)
    subscript_call_result_130692 = invoke(stypy.reporting.localization.Localization(__file__, 806, 24), getitem___130691, name_130689)
    
    # Obtaining the member '__getitem__' of a type (line 806)
    getitem___130693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 8), subscript_call_result_130692, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 806)
    subscript_call_result_130694 = invoke(stypy.reporting.localization.Localization(__file__, 806, 8), getitem___130693, int_130688)
    
    # Assigning a type to the variable 'tuple_var_assignment_128928' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'tuple_var_assignment_128928', subscript_call_result_130694)
    
    # Assigning a Subscript to a Name (line 806):
    
    # Obtaining the type of the subscript
    int_130695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 806)
    name_130696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 30), 'name')
    # Getting the type of 'cache' (line 806)
    cache_130697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 24), 'cache')
    # Obtaining the member '__getitem__' of a type (line 806)
    getitem___130698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 24), cache_130697, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 806)
    subscript_call_result_130699 = invoke(stypy.reporting.localization.Localization(__file__, 806, 24), getitem___130698, name_130696)
    
    # Obtaining the member '__getitem__' of a type (line 806)
    getitem___130700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 8), subscript_call_result_130699, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 806)
    subscript_call_result_130701 = invoke(stypy.reporting.localization.Localization(__file__, 806, 8), getitem___130700, int_130695)
    
    # Assigning a type to the variable 'tuple_var_assignment_128929' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'tuple_var_assignment_128929', subscript_call_result_130701)
    
    # Assigning a Subscript to a Name (line 806):
    
    # Obtaining the type of the subscript
    int_130702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 806, 8), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'name' (line 806)
    name_130703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 30), 'name')
    # Getting the type of 'cache' (line 806)
    cache_130704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 24), 'cache')
    # Obtaining the member '__getitem__' of a type (line 806)
    getitem___130705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 24), cache_130704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 806)
    subscript_call_result_130706 = invoke(stypy.reporting.localization.Localization(__file__, 806, 24), getitem___130705, name_130703)
    
    # Obtaining the member '__getitem__' of a type (line 806)
    getitem___130707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 806, 8), subscript_call_result_130706, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 806)
    subscript_call_result_130708 = invoke(stypy.reporting.localization.Localization(__file__, 806, 8), getitem___130707, int_130702)
    
    # Assigning a type to the variable 'tuple_var_assignment_128930' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'tuple_var_assignment_128930', subscript_call_result_130708)
    
    # Assigning a Name to a Name (line 806):
    # Getting the type of 'tuple_var_assignment_128928' (line 806)
    tuple_var_assignment_128928_130709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'tuple_var_assignment_128928')
    # Assigning a type to the variable 'doc' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'doc', tuple_var_assignment_128928_130709)
    
    # Assigning a Name to a Name (line 806):
    # Getting the type of 'tuple_var_assignment_128929' (line 806)
    tuple_var_assignment_128929_130710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'tuple_var_assignment_128929')
    # Assigning a type to the variable 'kind' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 13), 'kind', tuple_var_assignment_128929_130710)
    
    # Assigning a Name to a Name (line 806):
    # Getting the type of 'tuple_var_assignment_128930' (line 806)
    tuple_var_assignment_128930_130711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 806, 8), 'tuple_var_assignment_128930')
    # Assigning a type to the variable 'ix' (line 806)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 806, 19), 'ix', tuple_var_assignment_128930_130711)
    
    # Assigning a ListComp to a Name (line 808):
    
    # Assigning a ListComp to a Name (line 808):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 808)
    # Processing the call arguments (line 808)
    str_130725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 63), 'str', '\n')
    # Processing the call keyword arguments (line 808)
    kwargs_130726 = {}
    
    # Call to strip(...): (line 808)
    # Processing the call keyword arguments (line 808)
    kwargs_130722 = {}
    # Getting the type of 'doc' (line 808)
    doc_130720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 45), 'doc', False)
    # Obtaining the member 'strip' of a type (line 808)
    strip_130721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 45), doc_130720, 'strip')
    # Calling strip(args, kwargs) (line 808)
    strip_call_result_130723 = invoke(stypy.reporting.localization.Localization(__file__, 808, 45), strip_130721, *[], **kwargs_130722)
    
    # Obtaining the member 'split' of a type (line 808)
    split_130724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 45), strip_call_result_130723, 'split')
    # Calling split(args, kwargs) (line 808)
    split_call_result_130727 = invoke(stypy.reporting.localization.Localization(__file__, 808, 45), split_130724, *[str_130725], **kwargs_130726)
    
    comprehension_130728 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 808, 20), split_call_result_130727)
    # Assigning a type to the variable 'line' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 20), 'line', comprehension_130728)
    
    # Call to strip(...): (line 809)
    # Processing the call keyword arguments (line 809)
    kwargs_130718 = {}
    # Getting the type of 'line' (line 809)
    line_130716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 809, 23), 'line', False)
    # Obtaining the member 'strip' of a type (line 809)
    strip_130717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 809, 23), line_130716, 'strip')
    # Calling strip(args, kwargs) (line 809)
    strip_call_result_130719 = invoke(stypy.reporting.localization.Localization(__file__, 809, 23), strip_130717, *[], **kwargs_130718)
    
    
    # Call to strip(...): (line 808)
    # Processing the call keyword arguments (line 808)
    kwargs_130714 = {}
    # Getting the type of 'line' (line 808)
    line_130712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 808, 20), 'line', False)
    # Obtaining the member 'strip' of a type (line 808)
    strip_130713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 808, 20), line_130712, 'strip')
    # Calling strip(args, kwargs) (line 808)
    strip_call_result_130715 = invoke(stypy.reporting.localization.Localization(__file__, 808, 20), strip_130713, *[], **kwargs_130714)
    
    list_130729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 808, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 808, 20), list_130729, strip_call_result_130715)
    # Assigning a type to the variable 'doclines' (line 808)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 808, 8), 'doclines', list_130729)
    
    
    # SSA begins for try-except statement (line 812)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 813):
    
    # Assigning a Call to a Name (line 813):
    
    # Call to strip(...): (line 813)
    # Processing the call keyword arguments (line 813)
    kwargs_130735 = {}
    
    # Obtaining the type of the subscript
    int_130730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 813, 33), 'int')
    # Getting the type of 'doclines' (line 813)
    doclines_130731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 813, 24), 'doclines', False)
    # Obtaining the member '__getitem__' of a type (line 813)
    getitem___130732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 24), doclines_130731, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 813)
    subscript_call_result_130733 = invoke(stypy.reporting.localization.Localization(__file__, 813, 24), getitem___130732, int_130730)
    
    # Obtaining the member 'strip' of a type (line 813)
    strip_130734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 813, 24), subscript_call_result_130733, 'strip')
    # Calling strip(args, kwargs) (line 813)
    strip_call_result_130736 = invoke(stypy.reporting.localization.Localization(__file__, 813, 24), strip_130734, *[], **kwargs_130735)
    
    # Assigning a type to the variable 'first_doc' (line 813)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 813, 12), 'first_doc', strip_call_result_130736)
    
    
    # Call to search(...): (line 814)
    # Processing the call arguments (line 814)
    # Getting the type of 'first_doc' (line 814)
    first_doc_130739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 45), 'first_doc', False)
    # Processing the call keyword arguments (line 814)
    kwargs_130740 = {}
    # Getting the type of '_function_signature_re' (line 814)
    _function_signature_re_130737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 814, 15), '_function_signature_re', False)
    # Obtaining the member 'search' of a type (line 814)
    search_130738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 814, 15), _function_signature_re_130737, 'search')
    # Calling search(args, kwargs) (line 814)
    search_call_result_130741 = invoke(stypy.reporting.localization.Localization(__file__, 814, 15), search_130738, *[first_doc_130739], **kwargs_130740)
    
    # Testing the type of an if condition (line 814)
    if_condition_130742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 814, 12), search_call_result_130741)
    # Assigning a type to the variable 'if_condition_130742' (line 814)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 814, 12), 'if_condition_130742', if_condition_130742)
    # SSA begins for if statement (line 814)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 815):
    
    # Assigning a Call to a Name (line 815):
    
    # Call to strip(...): (line 815)
    # Processing the call keyword arguments (line 815)
    kwargs_130748 = {}
    
    # Obtaining the type of the subscript
    int_130743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 815, 37), 'int')
    # Getting the type of 'doclines' (line 815)
    doclines_130744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 815, 28), 'doclines', False)
    # Obtaining the member '__getitem__' of a type (line 815)
    getitem___130745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 28), doclines_130744, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 815)
    subscript_call_result_130746 = invoke(stypy.reporting.localization.Localization(__file__, 815, 28), getitem___130745, int_130743)
    
    # Obtaining the member 'strip' of a type (line 815)
    strip_130747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 815, 28), subscript_call_result_130746, 'strip')
    # Calling strip(args, kwargs) (line 815)
    strip_call_result_130749 = invoke(stypy.reporting.localization.Localization(__file__, 815, 28), strip_130747, *[], **kwargs_130748)
    
    # Assigning a type to the variable 'first_doc' (line 815)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 815, 16), 'first_doc', strip_call_result_130749)
    # SSA join for if statement (line 814)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 812)
    # SSA branch for the except 'IndexError' branch of a try statement (line 812)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Str to a Name (line 817):
    
    # Assigning a Str to a Name (line 817):
    str_130750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 817, 24), 'str', '')
    # Assigning a type to the variable 'first_doc' (line 817)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 817, 12), 'first_doc', str_130750)
    # SSA join for try-except statement (line 812)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 818)
    # Processing the call arguments (line 818)
    str_130753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 25), 'str', '%s\n    %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 818)
    tuple_130754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 818, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 818)
    # Adding element type (line 818)
    # Getting the type of 'name' (line 818)
    name_130755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 41), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 818, 41), tuple_130754, name_130755)
    # Adding element type (line 818)
    # Getting the type of 'first_doc' (line 818)
    first_doc_130756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 47), 'first_doc', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 818, 41), tuple_130754, first_doc_130756)
    
    # Applying the binary operator '%' (line 818)
    result_mod_130757 = python_operator(stypy.reporting.localization.Localization(__file__, 818, 25), '%', str_130753, tuple_130754)
    
    # Processing the call keyword arguments (line 818)
    kwargs_130758 = {}
    # Getting the type of 'help_text' (line 818)
    help_text_130751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 818, 8), 'help_text', False)
    # Obtaining the member 'append' of a type (line 818)
    append_130752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 818, 8), help_text_130751, 'append')
    # Calling append(args, kwargs) (line 818)
    append_call_result_130759 = invoke(stypy.reporting.localization.Localization(__file__, 818, 8), append_130752, *[result_mod_130757], **kwargs_130758)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'found' (line 820)
    found_130760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 820, 11), 'found')
    # Applying the 'not' unary operator (line 820)
    result_not__130761 = python_operator(stypy.reporting.localization.Localization(__file__, 820, 7), 'not', found_130760)
    
    # Testing the type of an if condition (line 820)
    if_condition_130762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 820, 4), result_not__130761)
    # Assigning a type to the variable 'if_condition_130762' (line 820)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 820, 4), 'if_condition_130762', if_condition_130762)
    # SSA begins for if statement (line 820)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 821)
    # Processing the call arguments (line 821)
    str_130765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 821, 25), 'str', 'Nothing found.')
    # Processing the call keyword arguments (line 821)
    kwargs_130766 = {}
    # Getting the type of 'help_text' (line 821)
    help_text_130763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 821, 8), 'help_text', False)
    # Obtaining the member 'append' of a type (line 821)
    append_130764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 821, 8), help_text_130763, 'append')
    # Calling append(args, kwargs) (line 821)
    append_call_result_130767 = invoke(stypy.reporting.localization.Localization(__file__, 821, 8), append_130764, *[str_130765], **kwargs_130766)
    
    # SSA join for if statement (line 820)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 824)
    # Getting the type of 'output' (line 824)
    output_130768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 4), 'output')
    # Getting the type of 'None' (line 824)
    None_130769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 824, 21), 'None')
    
    (may_be_130770, more_types_in_union_130771) = may_not_be_none(output_130768, None_130769)

    if may_be_130770:

        if more_types_in_union_130771:
            # Runtime conditional SSA (line 824)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to write(...): (line 825)
        # Processing the call arguments (line 825)
        
        # Call to join(...): (line 825)
        # Processing the call arguments (line 825)
        # Getting the type of 'help_text' (line 825)
        help_text_130776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 31), 'help_text', False)
        # Processing the call keyword arguments (line 825)
        kwargs_130777 = {}
        str_130774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 825, 21), 'str', '\n')
        # Obtaining the member 'join' of a type (line 825)
        join_130775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 21), str_130774, 'join')
        # Calling join(args, kwargs) (line 825)
        join_call_result_130778 = invoke(stypy.reporting.localization.Localization(__file__, 825, 21), join_130775, *[help_text_130776], **kwargs_130777)
        
        # Processing the call keyword arguments (line 825)
        kwargs_130779 = {}
        # Getting the type of 'output' (line 825)
        output_130772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 825, 8), 'output', False)
        # Obtaining the member 'write' of a type (line 825)
        write_130773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 825, 8), output_130772, 'write')
        # Calling write(args, kwargs) (line 825)
        write_call_result_130780 = invoke(stypy.reporting.localization.Localization(__file__, 825, 8), write_130773, *[join_call_result_130778], **kwargs_130779)
        

        if more_types_in_union_130771:
            # Runtime conditional SSA for else branch (line 824)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_130770) or more_types_in_union_130771):
        
        
        
        # Call to len(...): (line 826)
        # Processing the call arguments (line 826)
        # Getting the type of 'help_text' (line 826)
        help_text_130782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 13), 'help_text', False)
        # Processing the call keyword arguments (line 826)
        kwargs_130783 = {}
        # Getting the type of 'len' (line 826)
        len_130781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 826, 9), 'len', False)
        # Calling len(args, kwargs) (line 826)
        len_call_result_130784 = invoke(stypy.reporting.localization.Localization(__file__, 826, 9), len_130781, *[help_text_130782], **kwargs_130783)
        
        int_130785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 826, 26), 'int')
        # Applying the binary operator '>' (line 826)
        result_gt_130786 = python_operator(stypy.reporting.localization.Localization(__file__, 826, 9), '>', len_call_result_130784, int_130785)
        
        # Testing the type of an if condition (line 826)
        if_condition_130787 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 826, 9), result_gt_130786)
        # Assigning a type to the variable 'if_condition_130787' (line 826)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 826, 9), 'if_condition_130787', if_condition_130787)
        # SSA begins for if statement (line 826)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 827):
        
        # Assigning a Call to a Name (line 827):
        
        # Call to getpager(...): (line 827)
        # Processing the call keyword arguments (line 827)
        kwargs_130790 = {}
        # Getting the type of 'pydoc' (line 827)
        pydoc_130788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 827, 16), 'pydoc', False)
        # Obtaining the member 'getpager' of a type (line 827)
        getpager_130789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 827, 16), pydoc_130788, 'getpager')
        # Calling getpager(args, kwargs) (line 827)
        getpager_call_result_130791 = invoke(stypy.reporting.localization.Localization(__file__, 827, 16), getpager_130789, *[], **kwargs_130790)
        
        # Assigning a type to the variable 'pager' (line 827)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 827, 8), 'pager', getpager_call_result_130791)
        
        # Call to pager(...): (line 828)
        # Processing the call arguments (line 828)
        
        # Call to join(...): (line 828)
        # Processing the call arguments (line 828)
        # Getting the type of 'help_text' (line 828)
        help_text_130795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 24), 'help_text', False)
        # Processing the call keyword arguments (line 828)
        kwargs_130796 = {}
        str_130793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 828, 14), 'str', '\n')
        # Obtaining the member 'join' of a type (line 828)
        join_130794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 828, 14), str_130793, 'join')
        # Calling join(args, kwargs) (line 828)
        join_call_result_130797 = invoke(stypy.reporting.localization.Localization(__file__, 828, 14), join_130794, *[help_text_130795], **kwargs_130796)
        
        # Processing the call keyword arguments (line 828)
        kwargs_130798 = {}
        # Getting the type of 'pager' (line 828)
        pager_130792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 828, 8), 'pager', False)
        # Calling pager(args, kwargs) (line 828)
        pager_call_result_130799 = invoke(stypy.reporting.localization.Localization(__file__, 828, 8), pager_130792, *[join_call_result_130797], **kwargs_130798)
        
        # SSA branch for the else part of an if statement (line 826)
        module_type_store.open_ssa_branch('else')
        
        # Call to print(...): (line 830)
        # Processing the call arguments (line 830)
        
        # Call to join(...): (line 830)
        # Processing the call arguments (line 830)
        # Getting the type of 'help_text' (line 830)
        help_text_130803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 24), 'help_text', False)
        # Processing the call keyword arguments (line 830)
        kwargs_130804 = {}
        str_130801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 830, 14), 'str', '\n')
        # Obtaining the member 'join' of a type (line 830)
        join_130802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 830, 14), str_130801, 'join')
        # Calling join(args, kwargs) (line 830)
        join_call_result_130805 = invoke(stypy.reporting.localization.Localization(__file__, 830, 14), join_130802, *[help_text_130803], **kwargs_130804)
        
        # Processing the call keyword arguments (line 830)
        kwargs_130806 = {}
        # Getting the type of 'print' (line 830)
        print_130800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 830, 8), 'print', False)
        # Calling print(args, kwargs) (line 830)
        print_call_result_130807 = invoke(stypy.reporting.localization.Localization(__file__, 830, 8), print_130800, *[join_call_result_130805], **kwargs_130806)
        
        # SSA join for if statement (line 826)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_130770 and more_types_in_union_130771):
            # SSA join for if statement (line 824)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'lookfor(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lookfor' in the type store
    # Getting the type of 'stypy_return_type' (line 704)
    stypy_return_type_130808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 704, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_130808)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lookfor'
    return stypy_return_type_130808

# Assigning a type to the variable 'lookfor' (line 704)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 704, 0), 'lookfor', lookfor)

@norecursion
def _lookfor_generate_cache(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_lookfor_generate_cache'
    module_type_store = module_type_store.open_function_context('_lookfor_generate_cache', 832, 0, False)
    
    # Passed parameters checking function
    _lookfor_generate_cache.stypy_localization = localization
    _lookfor_generate_cache.stypy_type_of_self = None
    _lookfor_generate_cache.stypy_type_store = module_type_store
    _lookfor_generate_cache.stypy_function_name = '_lookfor_generate_cache'
    _lookfor_generate_cache.stypy_param_names_list = ['module', 'import_modules', 'regenerate']
    _lookfor_generate_cache.stypy_varargs_param_name = None
    _lookfor_generate_cache.stypy_kwargs_param_name = None
    _lookfor_generate_cache.stypy_call_defaults = defaults
    _lookfor_generate_cache.stypy_call_varargs = varargs
    _lookfor_generate_cache.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_lookfor_generate_cache', ['module', 'import_modules', 'regenerate'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_lookfor_generate_cache', localization, ['module', 'import_modules', 'regenerate'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_lookfor_generate_cache(...)' code ##################

    str_130809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 851, (-1)), 'str', '\n    Generate docstring cache for given module.\n\n    Parameters\n    ----------\n    module : str, None, module\n        Module for which to generate docstring cache\n    import_modules : bool\n        Whether to import sub-modules in packages.\n    regenerate : bool\n        Re-generate the docstring cache\n\n    Returns\n    -------\n    cache : dict {obj_full_name: (docstring, kind, index), ...}\n        Docstring cache for the module, either cached one (regenerate=False)\n        or newly generated.\n\n    ')
    # Marking variables as global (line 852)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 852, 4), '_lookfor_caches')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 854, 4))
    
    # 'import inspect' statement (line 854)
    import inspect

    import_module(stypy.reporting.localization.Localization(__file__, 854, 4), 'inspect', inspect, module_type_store)
    
    
    
    
    # Obtaining the type of the subscript
    int_130810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 24), 'int')
    # Getting the type of 'sys' (line 856)
    sys_130811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 856, 7), 'sys')
    # Obtaining the member 'version_info' of a type (line 856)
    version_info_130812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 7), sys_130811, 'version_info')
    # Obtaining the member '__getitem__' of a type (line 856)
    getitem___130813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 856, 7), version_info_130812, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 856)
    subscript_call_result_130814 = invoke(stypy.reporting.localization.Localization(__file__, 856, 7), getitem___130813, int_130810)
    
    int_130815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 856, 30), 'int')
    # Applying the binary operator '>=' (line 856)
    result_ge_130816 = python_operator(stypy.reporting.localization.Localization(__file__, 856, 7), '>=', subscript_call_result_130814, int_130815)
    
    # Testing the type of an if condition (line 856)
    if_condition_130817 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 856, 4), result_ge_130816)
    # Assigning a type to the variable 'if_condition_130817' (line 856)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 856, 4), 'if_condition_130817', if_condition_130817)
    # SSA begins for if statement (line 856)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 858, 8))
    
    # 'from io import StringIO' statement (line 858)
    from io import StringIO

    import_from_module(stypy.reporting.localization.Localization(__file__, 858, 8), 'io', None, module_type_store, ['StringIO'], [StringIO])
    
    # SSA branch for the else part of an if statement (line 856)
    module_type_store.open_ssa_branch('else')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 860, 8))
    
    # 'from StringIO import StringIO' statement (line 860)
    from StringIO import StringIO

    import_from_module(stypy.reporting.localization.Localization(__file__, 860, 8), 'StringIO', None, module_type_store, ['StringIO'], [StringIO])
    
    # SSA join for if statement (line 856)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 862)
    # Getting the type of 'module' (line 862)
    module_130818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 7), 'module')
    # Getting the type of 'None' (line 862)
    None_130819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 862, 17), 'None')
    
    (may_be_130820, more_types_in_union_130821) = may_be_none(module_130818, None_130819)

    if may_be_130820:

        if more_types_in_union_130821:
            # Runtime conditional SSA (line 862)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 863):
        
        # Assigning a Str to a Name (line 863):
        str_130822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 863, 17), 'str', 'numpy')
        # Assigning a type to the variable 'module' (line 863)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 863, 8), 'module', str_130822)

        if more_types_in_union_130821:
            # SSA join for if statement (line 862)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 865)
    # Getting the type of 'str' (line 865)
    str_130823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 26), 'str')
    # Getting the type of 'module' (line 865)
    module_130824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 865, 18), 'module')
    
    (may_be_130825, more_types_in_union_130826) = may_be_subtype(str_130823, module_130824)

    if may_be_130825:

        if more_types_in_union_130826:
            # Runtime conditional SSA (line 865)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'module' (line 865)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 4), 'module', remove_not_subtype_from_union(module_130824, str))
        
        
        # SSA begins for try-except statement (line 866)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to __import__(...): (line 867)
        # Processing the call arguments (line 867)
        # Getting the type of 'module' (line 867)
        module_130828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 23), 'module', False)
        # Processing the call keyword arguments (line 867)
        kwargs_130829 = {}
        # Getting the type of '__import__' (line 867)
        import___130827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 867, 12), '__import__', False)
        # Calling __import__(args, kwargs) (line 867)
        import___call_result_130830 = invoke(stypy.reporting.localization.Localization(__file__, 867, 12), import___130827, *[module_130828], **kwargs_130829)
        
        # SSA branch for the except part of a try statement (line 866)
        # SSA branch for the except 'ImportError' branch of a try statement (line 866)
        module_type_store.open_ssa_branch('except')
        
        # Obtaining an instance of the builtin type 'dict' (line 869)
        dict_130831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 869, 19), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 869)
        
        # Assigning a type to the variable 'stypy_return_type' (line 869)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 869, 12), 'stypy_return_type', dict_130831)
        # SSA join for try-except statement (line 866)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 870):
        
        # Assigning a Subscript to a Name (line 870):
        
        # Obtaining the type of the subscript
        # Getting the type of 'module' (line 870)
        module_130832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 29), 'module')
        # Getting the type of 'sys' (line 870)
        sys_130833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 870, 17), 'sys')
        # Obtaining the member 'modules' of a type (line 870)
        modules_130834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 17), sys_130833, 'modules')
        # Obtaining the member '__getitem__' of a type (line 870)
        getitem___130835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 870, 17), modules_130834, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 870)
        subscript_call_result_130836 = invoke(stypy.reporting.localization.Localization(__file__, 870, 17), getitem___130835, module_130832)
        
        # Assigning a type to the variable 'module' (line 870)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 870, 8), 'module', subscript_call_result_130836)

        if more_types_in_union_130826:
            # Runtime conditional SSA for else branch (line 865)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_130825) or more_types_in_union_130826):
        # Assigning a type to the variable 'module' (line 865)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 865, 4), 'module', remove_subtype_from_union(module_130824, str))
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 871)
        # Processing the call arguments (line 871)
        # Getting the type of 'module' (line 871)
        module_130838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 20), 'module', False)
        # Getting the type of 'list' (line 871)
        list_130839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 28), 'list', False)
        # Processing the call keyword arguments (line 871)
        kwargs_130840 = {}
        # Getting the type of 'isinstance' (line 871)
        isinstance_130837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 9), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 871)
        isinstance_call_result_130841 = invoke(stypy.reporting.localization.Localization(__file__, 871, 9), isinstance_130837, *[module_130838, list_130839], **kwargs_130840)
        
        
        # Call to isinstance(...): (line 871)
        # Processing the call arguments (line 871)
        # Getting the type of 'module' (line 871)
        module_130843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 48), 'module', False)
        # Getting the type of 'tuple' (line 871)
        tuple_130844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 56), 'tuple', False)
        # Processing the call keyword arguments (line 871)
        kwargs_130845 = {}
        # Getting the type of 'isinstance' (line 871)
        isinstance_130842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 871, 37), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 871)
        isinstance_call_result_130846 = invoke(stypy.reporting.localization.Localization(__file__, 871, 37), isinstance_130842, *[module_130843, tuple_130844], **kwargs_130845)
        
        # Applying the binary operator 'or' (line 871)
        result_or_keyword_130847 = python_operator(stypy.reporting.localization.Localization(__file__, 871, 9), 'or', isinstance_call_result_130841, isinstance_call_result_130846)
        
        # Testing the type of an if condition (line 871)
        if_condition_130848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 871, 9), result_or_keyword_130847)
        # Assigning a type to the variable 'if_condition_130848' (line 871)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 871, 9), 'if_condition_130848', if_condition_130848)
        # SSA begins for if statement (line 871)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Dict to a Name (line 872):
        
        # Assigning a Dict to a Name (line 872):
        
        # Obtaining an instance of the builtin type 'dict' (line 872)
        dict_130849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 872, 16), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 872)
        
        # Assigning a type to the variable 'cache' (line 872)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 872, 8), 'cache', dict_130849)
        
        # Getting the type of 'module' (line 873)
        module_130850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 873, 19), 'module')
        # Testing the type of a for loop iterable (line 873)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 873, 8), module_130850)
        # Getting the type of the for loop variable (line 873)
        for_loop_var_130851 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 873, 8), module_130850)
        # Assigning a type to the variable 'mod' (line 873)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 873, 8), 'mod', for_loop_var_130851)
        # SSA begins for a for statement (line 873)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to update(...): (line 874)
        # Processing the call arguments (line 874)
        
        # Call to _lookfor_generate_cache(...): (line 874)
        # Processing the call arguments (line 874)
        # Getting the type of 'mod' (line 874)
        mod_130855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 49), 'mod', False)
        # Getting the type of 'import_modules' (line 874)
        import_modules_130856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 54), 'import_modules', False)
        # Getting the type of 'regenerate' (line 875)
        regenerate_130857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 875, 49), 'regenerate', False)
        # Processing the call keyword arguments (line 874)
        kwargs_130858 = {}
        # Getting the type of '_lookfor_generate_cache' (line 874)
        _lookfor_generate_cache_130854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 25), '_lookfor_generate_cache', False)
        # Calling _lookfor_generate_cache(args, kwargs) (line 874)
        _lookfor_generate_cache_call_result_130859 = invoke(stypy.reporting.localization.Localization(__file__, 874, 25), _lookfor_generate_cache_130854, *[mod_130855, import_modules_130856, regenerate_130857], **kwargs_130858)
        
        # Processing the call keyword arguments (line 874)
        kwargs_130860 = {}
        # Getting the type of 'cache' (line 874)
        cache_130852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 874, 12), 'cache', False)
        # Obtaining the member 'update' of a type (line 874)
        update_130853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 874, 12), cache_130852, 'update')
        # Calling update(args, kwargs) (line 874)
        update_call_result_130861 = invoke(stypy.reporting.localization.Localization(__file__, 874, 12), update_130853, *[_lookfor_generate_cache_call_result_130859], **kwargs_130860)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'cache' (line 876)
        cache_130862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 876, 15), 'cache')
        # Assigning a type to the variable 'stypy_return_type' (line 876)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 876, 8), 'stypy_return_type', cache_130862)
        # SSA join for if statement (line 871)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_130825 and more_types_in_union_130826):
            # SSA join for if statement (line 865)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    
    # Call to id(...): (line 878)
    # Processing the call arguments (line 878)
    # Getting the type of 'module' (line 878)
    module_130864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 10), 'module', False)
    # Processing the call keyword arguments (line 878)
    kwargs_130865 = {}
    # Getting the type of 'id' (line 878)
    id_130863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 7), 'id', False)
    # Calling id(args, kwargs) (line 878)
    id_call_result_130866 = invoke(stypy.reporting.localization.Localization(__file__, 878, 7), id_130863, *[module_130864], **kwargs_130865)
    
    # Getting the type of '_lookfor_caches' (line 878)
    _lookfor_caches_130867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 21), '_lookfor_caches')
    # Applying the binary operator 'in' (line 878)
    result_contains_130868 = python_operator(stypy.reporting.localization.Localization(__file__, 878, 7), 'in', id_call_result_130866, _lookfor_caches_130867)
    
    
    # Getting the type of 'regenerate' (line 878)
    regenerate_130869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 878, 45), 'regenerate')
    # Applying the 'not' unary operator (line 878)
    result_not__130870 = python_operator(stypy.reporting.localization.Localization(__file__, 878, 41), 'not', regenerate_130869)
    
    # Applying the binary operator 'and' (line 878)
    result_and_keyword_130871 = python_operator(stypy.reporting.localization.Localization(__file__, 878, 7), 'and', result_contains_130868, result_not__130870)
    
    # Testing the type of an if condition (line 878)
    if_condition_130872 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 878, 4), result_and_keyword_130871)
    # Assigning a type to the variable 'if_condition_130872' (line 878)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 878, 4), 'if_condition_130872', if_condition_130872)
    # SSA begins for if statement (line 878)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    
    # Call to id(...): (line 879)
    # Processing the call arguments (line 879)
    # Getting the type of 'module' (line 879)
    module_130874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 34), 'module', False)
    # Processing the call keyword arguments (line 879)
    kwargs_130875 = {}
    # Getting the type of 'id' (line 879)
    id_130873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 31), 'id', False)
    # Calling id(args, kwargs) (line 879)
    id_call_result_130876 = invoke(stypy.reporting.localization.Localization(__file__, 879, 31), id_130873, *[module_130874], **kwargs_130875)
    
    # Getting the type of '_lookfor_caches' (line 879)
    _lookfor_caches_130877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 879, 15), '_lookfor_caches')
    # Obtaining the member '__getitem__' of a type (line 879)
    getitem___130878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 879, 15), _lookfor_caches_130877, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 879)
    subscript_call_result_130879 = invoke(stypy.reporting.localization.Localization(__file__, 879, 15), getitem___130878, id_call_result_130876)
    
    # Assigning a type to the variable 'stypy_return_type' (line 879)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 879, 8), 'stypy_return_type', subscript_call_result_130879)
    # SSA join for if statement (line 878)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Dict to a Name (line 882):
    
    # Assigning a Dict to a Name (line 882):
    
    # Obtaining an instance of the builtin type 'dict' (line 882)
    dict_130880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 882, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 882)
    
    # Assigning a type to the variable 'cache' (line 882)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 882, 4), 'cache', dict_130880)
    
    # Assigning a Name to a Subscript (line 883):
    
    # Assigning a Name to a Subscript (line 883):
    # Getting the type of 'cache' (line 883)
    cache_130881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 34), 'cache')
    # Getting the type of '_lookfor_caches' (line 883)
    _lookfor_caches_130882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 4), '_lookfor_caches')
    
    # Call to id(...): (line 883)
    # Processing the call arguments (line 883)
    # Getting the type of 'module' (line 883)
    module_130884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 23), 'module', False)
    # Processing the call keyword arguments (line 883)
    kwargs_130885 = {}
    # Getting the type of 'id' (line 883)
    id_130883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 883, 20), 'id', False)
    # Calling id(args, kwargs) (line 883)
    id_call_result_130886 = invoke(stypy.reporting.localization.Localization(__file__, 883, 20), id_130883, *[module_130884], **kwargs_130885)
    
    # Storing an element on a container (line 883)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 883, 4), _lookfor_caches_130882, (id_call_result_130886, cache_130881))
    
    # Assigning a Dict to a Name (line 884):
    
    # Assigning a Dict to a Name (line 884):
    
    # Obtaining an instance of the builtin type 'dict' (line 884)
    dict_130887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 884, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 884)
    
    # Assigning a type to the variable 'seen' (line 884)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 884, 4), 'seen', dict_130887)
    
    # Assigning a Num to a Name (line 885):
    
    # Assigning a Num to a Name (line 885):
    int_130888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 885, 12), 'int')
    # Assigning a type to the variable 'index' (line 885)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 885, 4), 'index', int_130888)
    
    # Assigning a List to a Name (line 886):
    
    # Assigning a List to a Name (line 886):
    
    # Obtaining an instance of the builtin type 'list' (line 886)
    list_130889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 886)
    # Adding element type (line 886)
    
    # Obtaining an instance of the builtin type 'tuple' (line 886)
    tuple_130890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 886, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 886)
    # Adding element type (line 886)
    # Getting the type of 'module' (line 886)
    module_130891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 14), 'module')
    # Obtaining the member '__name__' of a type (line 886)
    name___130892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 886, 14), module_130891, '__name__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 14), tuple_130890, name___130892)
    # Adding element type (line 886)
    # Getting the type of 'module' (line 886)
    module_130893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 886, 31), 'module')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 14), tuple_130890, module_130893)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 886, 12), list_130889, tuple_130890)
    
    # Assigning a type to the variable 'stack' (line 886)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 886, 4), 'stack', list_130889)
    
    # Getting the type of 'stack' (line 887)
    stack_130894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 887, 10), 'stack')
    # Testing the type of an if condition (line 887)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 887, 4), stack_130894)
    # SSA begins for while statement (line 887)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 888):
    
    # Assigning a Call to a Name:
    
    # Call to pop(...): (line 888)
    # Processing the call arguments (line 888)
    int_130897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 31), 'int')
    # Processing the call keyword arguments (line 888)
    kwargs_130898 = {}
    # Getting the type of 'stack' (line 888)
    stack_130895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 21), 'stack', False)
    # Obtaining the member 'pop' of a type (line 888)
    pop_130896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 21), stack_130895, 'pop')
    # Calling pop(args, kwargs) (line 888)
    pop_call_result_130899 = invoke(stypy.reporting.localization.Localization(__file__, 888, 21), pop_130896, *[int_130897], **kwargs_130898)
    
    # Assigning a type to the variable 'call_assignment_128931' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'call_assignment_128931', pop_call_result_130899)
    
    # Assigning a Call to a Name (line 888):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_130902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 8), 'int')
    # Processing the call keyword arguments
    kwargs_130903 = {}
    # Getting the type of 'call_assignment_128931' (line 888)
    call_assignment_128931_130900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'call_assignment_128931', False)
    # Obtaining the member '__getitem__' of a type (line 888)
    getitem___130901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 8), call_assignment_128931_130900, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_130904 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___130901, *[int_130902], **kwargs_130903)
    
    # Assigning a type to the variable 'call_assignment_128932' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'call_assignment_128932', getitem___call_result_130904)
    
    # Assigning a Name to a Name (line 888):
    # Getting the type of 'call_assignment_128932' (line 888)
    call_assignment_128932_130905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'call_assignment_128932')
    # Assigning a type to the variable 'name' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'name', call_assignment_128932_130905)
    
    # Assigning a Call to a Name (line 888):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_130908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 888, 8), 'int')
    # Processing the call keyword arguments
    kwargs_130909 = {}
    # Getting the type of 'call_assignment_128931' (line 888)
    call_assignment_128931_130906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'call_assignment_128931', False)
    # Obtaining the member '__getitem__' of a type (line 888)
    getitem___130907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 888, 8), call_assignment_128931_130906, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_130910 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___130907, *[int_130908], **kwargs_130909)
    
    # Assigning a type to the variable 'call_assignment_128933' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'call_assignment_128933', getitem___call_result_130910)
    
    # Assigning a Name to a Name (line 888):
    # Getting the type of 'call_assignment_128933' (line 888)
    call_assignment_128933_130911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 888, 8), 'call_assignment_128933')
    # Assigning a type to the variable 'item' (line 888)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 888, 14), 'item', call_assignment_128933_130911)
    
    
    
    # Call to id(...): (line 889)
    # Processing the call arguments (line 889)
    # Getting the type of 'item' (line 889)
    item_130913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 14), 'item', False)
    # Processing the call keyword arguments (line 889)
    kwargs_130914 = {}
    # Getting the type of 'id' (line 889)
    id_130912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 11), 'id', False)
    # Calling id(args, kwargs) (line 889)
    id_call_result_130915 = invoke(stypy.reporting.localization.Localization(__file__, 889, 11), id_130912, *[item_130913], **kwargs_130914)
    
    # Getting the type of 'seen' (line 889)
    seen_130916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 889, 23), 'seen')
    # Applying the binary operator 'in' (line 889)
    result_contains_130917 = python_operator(stypy.reporting.localization.Localization(__file__, 889, 11), 'in', id_call_result_130915, seen_130916)
    
    # Testing the type of an if condition (line 889)
    if_condition_130918 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 889, 8), result_contains_130917)
    # Assigning a type to the variable 'if_condition_130918' (line 889)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 889, 8), 'if_condition_130918', if_condition_130918)
    # SSA begins for if statement (line 889)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 889)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 891):
    
    # Assigning a Name to a Subscript (line 891):
    # Getting the type of 'True' (line 891)
    True_130919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 25), 'True')
    # Getting the type of 'seen' (line 891)
    seen_130920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 8), 'seen')
    
    # Call to id(...): (line 891)
    # Processing the call arguments (line 891)
    # Getting the type of 'item' (line 891)
    item_130922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 16), 'item', False)
    # Processing the call keyword arguments (line 891)
    kwargs_130923 = {}
    # Getting the type of 'id' (line 891)
    id_130921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 891, 13), 'id', False)
    # Calling id(args, kwargs) (line 891)
    id_call_result_130924 = invoke(stypy.reporting.localization.Localization(__file__, 891, 13), id_130921, *[item_130922], **kwargs_130923)
    
    # Storing an element on a container (line 891)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 891, 8), seen_130920, (id_call_result_130924, True_130919))
    
    # Getting the type of 'index' (line 893)
    index_130925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 893, 8), 'index')
    int_130926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 893, 17), 'int')
    # Applying the binary operator '+=' (line 893)
    result_iadd_130927 = python_operator(stypy.reporting.localization.Localization(__file__, 893, 8), '+=', index_130925, int_130926)
    # Assigning a type to the variable 'index' (line 893)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 893, 8), 'index', result_iadd_130927)
    
    
    # Assigning a Str to a Name (line 894):
    
    # Assigning a Str to a Name (line 894):
    str_130928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 894, 15), 'str', 'object')
    # Assigning a type to the variable 'kind' (line 894)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 894, 8), 'kind', str_130928)
    
    
    # Call to ismodule(...): (line 896)
    # Processing the call arguments (line 896)
    # Getting the type of 'item' (line 896)
    item_130931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 28), 'item', False)
    # Processing the call keyword arguments (line 896)
    kwargs_130932 = {}
    # Getting the type of 'inspect' (line 896)
    inspect_130929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 896, 11), 'inspect', False)
    # Obtaining the member 'ismodule' of a type (line 896)
    ismodule_130930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 896, 11), inspect_130929, 'ismodule')
    # Calling ismodule(args, kwargs) (line 896)
    ismodule_call_result_130933 = invoke(stypy.reporting.localization.Localization(__file__, 896, 11), ismodule_130930, *[item_130931], **kwargs_130932)
    
    # Testing the type of an if condition (line 896)
    if_condition_130934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 896, 8), ismodule_call_result_130933)
    # Assigning a type to the variable 'if_condition_130934' (line 896)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 896, 8), 'if_condition_130934', if_condition_130934)
    # SSA begins for if statement (line 896)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 897):
    
    # Assigning a Str to a Name (line 897):
    str_130935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 897, 19), 'str', 'module')
    # Assigning a type to the variable 'kind' (line 897)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 897, 12), 'kind', str_130935)
    
    
    # SSA begins for try-except statement (line 898)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 899):
    
    # Assigning a Attribute to a Name (line 899):
    # Getting the type of 'item' (line 899)
    item_130936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 899, 23), 'item')
    # Obtaining the member '__all__' of a type (line 899)
    all___130937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 899, 23), item_130936, '__all__')
    # Assigning a type to the variable '_all' (line 899)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 899, 16), '_all', all___130937)
    # SSA branch for the except part of a try statement (line 898)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 898)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 901):
    
    # Assigning a Name to a Name (line 901):
    # Getting the type of 'None' (line 901)
    None_130938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 901, 23), 'None')
    # Assigning a type to the variable '_all' (line 901)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 901, 16), '_all', None_130938)
    # SSA join for try-except statement (line 898)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'import_modules' (line 904)
    import_modules_130939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 15), 'import_modules')
    
    # Call to hasattr(...): (line 904)
    # Processing the call arguments (line 904)
    # Getting the type of 'item' (line 904)
    item_130941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 42), 'item', False)
    str_130942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 904, 48), 'str', '__path__')
    # Processing the call keyword arguments (line 904)
    kwargs_130943 = {}
    # Getting the type of 'hasattr' (line 904)
    hasattr_130940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 904, 34), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 904)
    hasattr_call_result_130944 = invoke(stypy.reporting.localization.Localization(__file__, 904, 34), hasattr_130940, *[item_130941, str_130942], **kwargs_130943)
    
    # Applying the binary operator 'and' (line 904)
    result_and_keyword_130945 = python_operator(stypy.reporting.localization.Localization(__file__, 904, 15), 'and', import_modules_130939, hasattr_call_result_130944)
    
    # Testing the type of an if condition (line 904)
    if_condition_130946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 904, 12), result_and_keyword_130945)
    # Assigning a type to the variable 'if_condition_130946' (line 904)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 904, 12), 'if_condition_130946', if_condition_130946)
    # SSA begins for if statement (line 904)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'item' (line 905)
    item_130947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 905, 27), 'item')
    # Obtaining the member '__path__' of a type (line 905)
    path___130948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 905, 27), item_130947, '__path__')
    # Testing the type of a for loop iterable (line 905)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 905, 16), path___130948)
    # Getting the type of the for loop variable (line 905)
    for_loop_var_130949 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 905, 16), path___130948)
    # Assigning a type to the variable 'pth' (line 905)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 905, 16), 'pth', for_loop_var_130949)
    # SSA begins for a for statement (line 905)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to listdir(...): (line 906)
    # Processing the call arguments (line 906)
    # Getting the type of 'pth' (line 906)
    pth_130952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 47), 'pth', False)
    # Processing the call keyword arguments (line 906)
    kwargs_130953 = {}
    # Getting the type of 'os' (line 906)
    os_130950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 906, 36), 'os', False)
    # Obtaining the member 'listdir' of a type (line 906)
    listdir_130951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 906, 36), os_130950, 'listdir')
    # Calling listdir(args, kwargs) (line 906)
    listdir_call_result_130954 = invoke(stypy.reporting.localization.Localization(__file__, 906, 36), listdir_130951, *[pth_130952], **kwargs_130953)
    
    # Testing the type of a for loop iterable (line 906)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 906, 20), listdir_call_result_130954)
    # Getting the type of the for loop variable (line 906)
    for_loop_var_130955 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 906, 20), listdir_call_result_130954)
    # Assigning a type to the variable 'mod_path' (line 906)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 906, 20), 'mod_path', for_loop_var_130955)
    # SSA begins for a for statement (line 906)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 907):
    
    # Assigning a Call to a Name (line 907):
    
    # Call to join(...): (line 907)
    # Processing the call arguments (line 907)
    # Getting the type of 'pth' (line 907)
    pth_130959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 47), 'pth', False)
    # Getting the type of 'mod_path' (line 907)
    mod_path_130960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 52), 'mod_path', False)
    # Processing the call keyword arguments (line 907)
    kwargs_130961 = {}
    # Getting the type of 'os' (line 907)
    os_130956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 907, 34), 'os', False)
    # Obtaining the member 'path' of a type (line 907)
    path_130957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 34), os_130956, 'path')
    # Obtaining the member 'join' of a type (line 907)
    join_130958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 907, 34), path_130957, 'join')
    # Calling join(args, kwargs) (line 907)
    join_call_result_130962 = invoke(stypy.reporting.localization.Localization(__file__, 907, 34), join_130958, *[pth_130959, mod_path_130960], **kwargs_130961)
    
    # Assigning a type to the variable 'this_py' (line 907)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 907, 24), 'this_py', join_call_result_130962)
    
    # Assigning a Call to a Name (line 908):
    
    # Assigning a Call to a Name (line 908):
    
    # Call to join(...): (line 908)
    # Processing the call arguments (line 908)
    # Getting the type of 'pth' (line 908)
    pth_130966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 47), 'pth', False)
    # Getting the type of 'mod_path' (line 908)
    mod_path_130967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 52), 'mod_path', False)
    str_130968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 908, 62), 'str', '__init__.py')
    # Processing the call keyword arguments (line 908)
    kwargs_130969 = {}
    # Getting the type of 'os' (line 908)
    os_130963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 908, 34), 'os', False)
    # Obtaining the member 'path' of a type (line 908)
    path_130964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 34), os_130963, 'path')
    # Obtaining the member 'join' of a type (line 908)
    join_130965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 908, 34), path_130964, 'join')
    # Calling join(args, kwargs) (line 908)
    join_call_result_130970 = invoke(stypy.reporting.localization.Localization(__file__, 908, 34), join_130965, *[pth_130966, mod_path_130967, str_130968], **kwargs_130969)
    
    # Assigning a type to the variable 'init_py' (line 908)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 908, 24), 'init_py', join_call_result_130970)
    
    
    # Evaluating a boolean operation
    
    # Call to isfile(...): (line 909)
    # Processing the call arguments (line 909)
    # Getting the type of 'this_py' (line 909)
    this_py_130974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 43), 'this_py', False)
    # Processing the call keyword arguments (line 909)
    kwargs_130975 = {}
    # Getting the type of 'os' (line 909)
    os_130971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 909, 28), 'os', False)
    # Obtaining the member 'path' of a type (line 909)
    path_130972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 28), os_130971, 'path')
    # Obtaining the member 'isfile' of a type (line 909)
    isfile_130973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 909, 28), path_130972, 'isfile')
    # Calling isfile(args, kwargs) (line 909)
    isfile_call_result_130976 = invoke(stypy.reporting.localization.Localization(__file__, 909, 28), isfile_130973, *[this_py_130974], **kwargs_130975)
    
    
    # Call to endswith(...): (line 910)
    # Processing the call arguments (line 910)
    str_130979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 910, 50), 'str', '.py')
    # Processing the call keyword arguments (line 910)
    kwargs_130980 = {}
    # Getting the type of 'mod_path' (line 910)
    mod_path_130977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 910, 32), 'mod_path', False)
    # Obtaining the member 'endswith' of a type (line 910)
    endswith_130978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 910, 32), mod_path_130977, 'endswith')
    # Calling endswith(args, kwargs) (line 910)
    endswith_call_result_130981 = invoke(stypy.reporting.localization.Localization(__file__, 910, 32), endswith_130978, *[str_130979], **kwargs_130980)
    
    # Applying the binary operator 'and' (line 909)
    result_and_keyword_130982 = python_operator(stypy.reporting.localization.Localization(__file__, 909, 28), 'and', isfile_call_result_130976, endswith_call_result_130981)
    
    # Testing the type of an if condition (line 909)
    if_condition_130983 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 909, 24), result_and_keyword_130982)
    # Assigning a type to the variable 'if_condition_130983' (line 909)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 909, 24), 'if_condition_130983', if_condition_130983)
    # SSA begins for if statement (line 909)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 911):
    
    # Assigning a Subscript to a Name (line 911):
    
    # Obtaining the type of the subscript
    int_130984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 911, 50), 'int')
    slice_130985 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 911, 40), None, int_130984, None)
    # Getting the type of 'mod_path' (line 911)
    mod_path_130986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 911, 40), 'mod_path')
    # Obtaining the member '__getitem__' of a type (line 911)
    getitem___130987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 911, 40), mod_path_130986, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 911)
    subscript_call_result_130988 = invoke(stypy.reporting.localization.Localization(__file__, 911, 40), getitem___130987, slice_130985)
    
    # Assigning a type to the variable 'to_import' (line 911)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 911, 28), 'to_import', subscript_call_result_130988)
    # SSA branch for the else part of an if statement (line 909)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isfile(...): (line 912)
    # Processing the call arguments (line 912)
    # Getting the type of 'init_py' (line 912)
    init_py_130992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 44), 'init_py', False)
    # Processing the call keyword arguments (line 912)
    kwargs_130993 = {}
    # Getting the type of 'os' (line 912)
    os_130989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 912, 29), 'os', False)
    # Obtaining the member 'path' of a type (line 912)
    path_130990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 29), os_130989, 'path')
    # Obtaining the member 'isfile' of a type (line 912)
    isfile_130991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 912, 29), path_130990, 'isfile')
    # Calling isfile(args, kwargs) (line 912)
    isfile_call_result_130994 = invoke(stypy.reporting.localization.Localization(__file__, 912, 29), isfile_130991, *[init_py_130992], **kwargs_130993)
    
    # Testing the type of an if condition (line 912)
    if_condition_130995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 912, 29), isfile_call_result_130994)
    # Assigning a type to the variable 'if_condition_130995' (line 912)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 912, 29), 'if_condition_130995', if_condition_130995)
    # SSA begins for if statement (line 912)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 913):
    
    # Assigning a Name to a Name (line 913):
    # Getting the type of 'mod_path' (line 913)
    mod_path_130996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 913, 40), 'mod_path')
    # Assigning a type to the variable 'to_import' (line 913)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 913, 28), 'to_import', mod_path_130996)
    # SSA branch for the else part of an if statement (line 912)
    module_type_store.open_ssa_branch('else')
    # SSA join for if statement (line 912)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 909)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'to_import' (line 916)
    to_import_130997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 916, 27), 'to_import')
    str_130998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 916, 40), 'str', '__init__')
    # Applying the binary operator '==' (line 916)
    result_eq_130999 = python_operator(stypy.reporting.localization.Localization(__file__, 916, 27), '==', to_import_130997, str_130998)
    
    # Testing the type of an if condition (line 916)
    if_condition_131000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 916, 24), result_eq_130999)
    # Assigning a type to the variable 'if_condition_131000' (line 916)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 916, 24), 'if_condition_131000', if_condition_131000)
    # SSA begins for if statement (line 916)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 916)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 919)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Name to a Name (line 921):
    
    # Assigning a Name to a Name (line 921):
    # Getting the type of 'BaseException' (line 921)
    BaseException_131001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 921, 39), 'BaseException')
    # Assigning a type to the variable 'base_exc' (line 921)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 921, 28), 'base_exc', BaseException_131001)
    # SSA branch for the except part of a try statement (line 919)
    # SSA branch for the except 'NameError' branch of a try statement (line 919)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 924):
    
    # Assigning a Name to a Name (line 924):
    # Getting the type of 'Exception' (line 924)
    Exception_131002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 924, 39), 'Exception')
    # Assigning a type to the variable 'base_exc' (line 924)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 924, 28), 'base_exc', Exception_131002)
    # SSA join for try-except statement (line 919)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 926)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Attribute to a Name (line 927):
    
    # Assigning a Attribute to a Name (line 927):
    # Getting the type of 'sys' (line 927)
    sys_131003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 927, 41), 'sys')
    # Obtaining the member 'stdout' of a type (line 927)
    stdout_131004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 927, 41), sys_131003, 'stdout')
    # Assigning a type to the variable 'old_stdout' (line 927)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 927, 28), 'old_stdout', stdout_131004)
    
    # Assigning a Attribute to a Name (line 928):
    
    # Assigning a Attribute to a Name (line 928):
    # Getting the type of 'sys' (line 928)
    sys_131005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 928, 41), 'sys')
    # Obtaining the member 'stderr' of a type (line 928)
    stderr_131006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 928, 41), sys_131005, 'stderr')
    # Assigning a type to the variable 'old_stderr' (line 928)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 928, 28), 'old_stderr', stderr_131006)
    
    # Try-finally block (line 929)
    
    # Assigning a Call to a Attribute (line 930):
    
    # Assigning a Call to a Attribute (line 930):
    
    # Call to StringIO(...): (line 930)
    # Processing the call keyword arguments (line 930)
    kwargs_131008 = {}
    # Getting the type of 'StringIO' (line 930)
    StringIO_131007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 45), 'StringIO', False)
    # Calling StringIO(args, kwargs) (line 930)
    StringIO_call_result_131009 = invoke(stypy.reporting.localization.Localization(__file__, 930, 45), StringIO_131007, *[], **kwargs_131008)
    
    # Getting the type of 'sys' (line 930)
    sys_131010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 930, 32), 'sys')
    # Setting the type of the member 'stdout' of a type (line 930)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 930, 32), sys_131010, 'stdout', StringIO_call_result_131009)
    
    # Assigning a Call to a Attribute (line 931):
    
    # Assigning a Call to a Attribute (line 931):
    
    # Call to StringIO(...): (line 931)
    # Processing the call keyword arguments (line 931)
    kwargs_131012 = {}
    # Getting the type of 'StringIO' (line 931)
    StringIO_131011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 45), 'StringIO', False)
    # Calling StringIO(args, kwargs) (line 931)
    StringIO_call_result_131013 = invoke(stypy.reporting.localization.Localization(__file__, 931, 45), StringIO_131011, *[], **kwargs_131012)
    
    # Getting the type of 'sys' (line 931)
    sys_131014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 931, 32), 'sys')
    # Setting the type of the member 'stderr' of a type (line 931)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 931, 32), sys_131014, 'stderr', StringIO_call_result_131013)
    
    # Call to __import__(...): (line 932)
    # Processing the call arguments (line 932)
    str_131016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 43), 'str', '%s.%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 932)
    tuple_131017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 932, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 932)
    # Adding element type (line 932)
    # Getting the type of 'name' (line 932)
    name_131018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 54), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 54), tuple_131017, name_131018)
    # Adding element type (line 932)
    # Getting the type of 'to_import' (line 932)
    to_import_131019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 60), 'to_import', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 932, 54), tuple_131017, to_import_131019)
    
    # Applying the binary operator '%' (line 932)
    result_mod_131020 = python_operator(stypy.reporting.localization.Localization(__file__, 932, 43), '%', str_131016, tuple_131017)
    
    # Processing the call keyword arguments (line 932)
    kwargs_131021 = {}
    # Getting the type of '__import__' (line 932)
    import___131015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 932, 32), '__import__', False)
    # Calling __import__(args, kwargs) (line 932)
    import___call_result_131022 = invoke(stypy.reporting.localization.Localization(__file__, 932, 32), import___131015, *[result_mod_131020], **kwargs_131021)
    
    
    # finally branch of the try-finally block (line 929)
    
    # Assigning a Name to a Attribute (line 934):
    
    # Assigning a Name to a Attribute (line 934):
    # Getting the type of 'old_stdout' (line 934)
    old_stdout_131023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 45), 'old_stdout')
    # Getting the type of 'sys' (line 934)
    sys_131024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 934, 32), 'sys')
    # Setting the type of the member 'stdout' of a type (line 934)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 934, 32), sys_131024, 'stdout', old_stdout_131023)
    
    # Assigning a Name to a Attribute (line 935):
    
    # Assigning a Name to a Attribute (line 935):
    # Getting the type of 'old_stderr' (line 935)
    old_stderr_131025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 45), 'old_stderr')
    # Getting the type of 'sys' (line 935)
    sys_131026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 935, 32), 'sys')
    # Setting the type of the member 'stderr' of a type (line 935)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 935, 32), sys_131026, 'stderr', old_stderr_131025)
    
    # SSA branch for the except part of a try statement (line 926)
    # SSA branch for the except 'base_exc' branch of a try statement (line 926)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 926)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 904)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to _getmembers(...): (line 939)
    # Processing the call arguments (line 939)
    # Getting the type of 'item' (line 939)
    item_131028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 36), 'item', False)
    # Processing the call keyword arguments (line 939)
    kwargs_131029 = {}
    # Getting the type of '_getmembers' (line 939)
    _getmembers_131027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 939, 24), '_getmembers', False)
    # Calling _getmembers(args, kwargs) (line 939)
    _getmembers_call_result_131030 = invoke(stypy.reporting.localization.Localization(__file__, 939, 24), _getmembers_131027, *[item_131028], **kwargs_131029)
    
    # Testing the type of a for loop iterable (line 939)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 939, 12), _getmembers_call_result_131030)
    # Getting the type of the for loop variable (line 939)
    for_loop_var_131031 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 939, 12), _getmembers_call_result_131030)
    # Assigning a type to the variable 'n' (line 939)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 12), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 939, 12), for_loop_var_131031))
    # Assigning a type to the variable 'v' (line 939)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 939, 12), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 939, 12), for_loop_var_131031))
    # SSA begins for a for statement (line 939)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 940)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 941):
    
    # Assigning a Call to a Name (line 941):
    
    # Call to getattr(...): (line 941)
    # Processing the call arguments (line 941)
    # Getting the type of 'v' (line 941)
    v_131033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 40), 'v', False)
    str_131034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 43), 'str', '__name__')
    str_131035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 55), 'str', '%s.%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 941)
    tuple_131036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, 66), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 941)
    # Adding element type (line 941)
    # Getting the type of 'name' (line 941)
    name_131037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 66), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 941, 66), tuple_131036, name_131037)
    # Adding element type (line 941)
    # Getting the type of 'n' (line 941)
    n_131038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 72), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 941, 66), tuple_131036, n_131038)
    
    # Applying the binary operator '%' (line 941)
    result_mod_131039 = python_operator(stypy.reporting.localization.Localization(__file__, 941, 55), '%', str_131035, tuple_131036)
    
    # Processing the call keyword arguments (line 941)
    kwargs_131040 = {}
    # Getting the type of 'getattr' (line 941)
    getattr_131032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 941, 32), 'getattr', False)
    # Calling getattr(args, kwargs) (line 941)
    getattr_call_result_131041 = invoke(stypy.reporting.localization.Localization(__file__, 941, 32), getattr_131032, *[v_131033, str_131034, result_mod_131039], **kwargs_131040)
    
    # Assigning a type to the variable 'item_name' (line 941)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 941, 20), 'item_name', getattr_call_result_131041)
    
    # Assigning a Call to a Name (line 942):
    
    # Assigning a Call to a Name (line 942):
    
    # Call to getattr(...): (line 942)
    # Processing the call arguments (line 942)
    # Getting the type of 'v' (line 942)
    v_131043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 39), 'v', False)
    str_131044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 942, 42), 'str', '__module__')
    # Getting the type of 'None' (line 942)
    None_131045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 56), 'None', False)
    # Processing the call keyword arguments (line 942)
    kwargs_131046 = {}
    # Getting the type of 'getattr' (line 942)
    getattr_131042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 942, 31), 'getattr', False)
    # Calling getattr(args, kwargs) (line 942)
    getattr_call_result_131047 = invoke(stypy.reporting.localization.Localization(__file__, 942, 31), getattr_131042, *[v_131043, str_131044, None_131045], **kwargs_131046)
    
    # Assigning a type to the variable 'mod_name' (line 942)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 942, 20), 'mod_name', getattr_call_result_131047)
    # SSA branch for the except part of a try statement (line 940)
    # SSA branch for the except 'NameError' branch of a try statement (line 940)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a BinOp to a Name (line 946):
    
    # Assigning a BinOp to a Name (line 946):
    str_131048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 32), 'str', '%s.%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 946)
    tuple_131049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 946, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 946)
    # Adding element type (line 946)
    # Getting the type of 'name' (line 946)
    name_131050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 43), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 946, 43), tuple_131049, name_131050)
    # Adding element type (line 946)
    # Getting the type of 'n' (line 946)
    n_131051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 946, 49), 'n')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 946, 43), tuple_131049, n_131051)
    
    # Applying the binary operator '%' (line 946)
    result_mod_131052 = python_operator(stypy.reporting.localization.Localization(__file__, 946, 32), '%', str_131048, tuple_131049)
    
    # Assigning a type to the variable 'item_name' (line 946)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 946, 20), 'item_name', result_mod_131052)
    
    # Assigning a Name to a Name (line 947):
    
    # Assigning a Name to a Name (line 947):
    # Getting the type of 'None' (line 947)
    None_131053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 31), 'None')
    # Assigning a type to the variable 'mod_name' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 20), 'mod_name', None_131053)
    # SSA join for try-except statement (line 940)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    str_131054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 948, 19), 'str', '.')
    # Getting the type of 'item_name' (line 948)
    item_name_131055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 30), 'item_name')
    # Applying the binary operator 'notin' (line 948)
    result_contains_131056 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 19), 'notin', str_131054, item_name_131055)
    
    # Getting the type of 'mod_name' (line 948)
    mod_name_131057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 44), 'mod_name')
    # Applying the binary operator 'and' (line 948)
    result_and_keyword_131058 = python_operator(stypy.reporting.localization.Localization(__file__, 948, 19), 'and', result_contains_131056, mod_name_131057)
    
    # Testing the type of an if condition (line 948)
    if_condition_131059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 948, 16), result_and_keyword_131058)
    # Assigning a type to the variable 'if_condition_131059' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 16), 'if_condition_131059', if_condition_131059)
    # SSA begins for if statement (line 948)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 949):
    
    # Assigning a BinOp to a Name (line 949):
    str_131060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 949, 32), 'str', '%s.%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 949)
    tuple_131061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 949, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 949)
    # Adding element type (line 949)
    # Getting the type of 'mod_name' (line 949)
    mod_name_131062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 43), 'mod_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 949, 43), tuple_131061, mod_name_131062)
    # Adding element type (line 949)
    # Getting the type of 'item_name' (line 949)
    item_name_131063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 949, 53), 'item_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 949, 43), tuple_131061, item_name_131063)
    
    # Applying the binary operator '%' (line 949)
    result_mod_131064 = python_operator(stypy.reporting.localization.Localization(__file__, 949, 32), '%', str_131060, tuple_131061)
    
    # Assigning a type to the variable 'item_name' (line 949)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 949, 20), 'item_name', result_mod_131064)
    # SSA join for if statement (line 948)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to startswith(...): (line 951)
    # Processing the call arguments (line 951)
    # Getting the type of 'name' (line 951)
    name_131067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 44), 'name', False)
    str_131068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 51), 'str', '.')
    # Applying the binary operator '+' (line 951)
    result_add_131069 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 44), '+', name_131067, str_131068)
    
    # Processing the call keyword arguments (line 951)
    kwargs_131070 = {}
    # Getting the type of 'item_name' (line 951)
    item_name_131065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 23), 'item_name', False)
    # Obtaining the member 'startswith' of a type (line 951)
    startswith_131066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 23), item_name_131065, 'startswith')
    # Calling startswith(args, kwargs) (line 951)
    startswith_call_result_131071 = invoke(stypy.reporting.localization.Localization(__file__, 951, 23), startswith_131066, *[result_add_131069], **kwargs_131070)
    
    # Applying the 'not' unary operator (line 951)
    result_not__131072 = python_operator(stypy.reporting.localization.Localization(__file__, 951, 19), 'not', startswith_call_result_131071)
    
    # Testing the type of an if condition (line 951)
    if_condition_131073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 951, 16), result_not__131072)
    # Assigning a type to the variable 'if_condition_131073' (line 951)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 16), 'if_condition_131073', if_condition_131073)
    # SSA begins for if statement (line 951)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to isinstance(...): (line 953)
    # Processing the call arguments (line 953)
    # Getting the type of 'v' (line 953)
    v_131075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 34), 'v', False)
    # Getting the type of 'ufunc' (line 953)
    ufunc_131076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 37), 'ufunc', False)
    # Processing the call keyword arguments (line 953)
    kwargs_131077 = {}
    # Getting the type of 'isinstance' (line 953)
    isinstance_131074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 953, 23), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 953)
    isinstance_call_result_131078 = invoke(stypy.reporting.localization.Localization(__file__, 953, 23), isinstance_131074, *[v_131075, ufunc_131076], **kwargs_131077)
    
    # Testing the type of an if condition (line 953)
    if_condition_131079 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 953, 20), isinstance_call_result_131078)
    # Assigning a type to the variable 'if_condition_131079' (line 953)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 953, 20), 'if_condition_131079', if_condition_131079)
    # SSA begins for if statement (line 953)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 953)
    module_type_store.open_ssa_branch('else')
    # SSA join for if statement (line 953)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 951)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Evaluating a boolean operation
    
    # Call to ismodule(...): (line 958)
    # Processing the call arguments (line 958)
    # Getting the type of 'v' (line 958)
    v_131082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 43), 'v', False)
    # Processing the call keyword arguments (line 958)
    kwargs_131083 = {}
    # Getting the type of 'inspect' (line 958)
    inspect_131080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 26), 'inspect', False)
    # Obtaining the member 'ismodule' of a type (line 958)
    ismodule_131081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 958, 26), inspect_131080, 'ismodule')
    # Calling ismodule(args, kwargs) (line 958)
    ismodule_call_result_131084 = invoke(stypy.reporting.localization.Localization(__file__, 958, 26), ismodule_131081, *[v_131082], **kwargs_131083)
    
    
    # Getting the type of '_all' (line 958)
    _all_131085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 49), '_all')
    # Getting the type of 'None' (line 958)
    None_131086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 57), 'None')
    # Applying the binary operator 'is' (line 958)
    result_is__131087 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 49), 'is', _all_131085, None_131086)
    
    # Applying the binary operator 'or' (line 958)
    result_or_keyword_131088 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 26), 'or', ismodule_call_result_131084, result_is__131087)
    
    # Getting the type of 'n' (line 958)
    n_131089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 65), 'n')
    # Getting the type of '_all' (line 958)
    _all_131090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 958, 70), '_all')
    # Applying the binary operator 'in' (line 958)
    result_contains_131091 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 65), 'in', n_131089, _all_131090)
    
    # Applying the binary operator 'or' (line 958)
    result_or_keyword_131092 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 26), 'or', result_or_keyword_131088, result_contains_131091)
    
    # Applying the 'not' unary operator (line 958)
    result_not__131093 = python_operator(stypy.reporting.localization.Localization(__file__, 958, 21), 'not', result_or_keyword_131092)
    
    # Testing the type of an if condition (line 958)
    if_condition_131094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 958, 21), result_not__131093)
    # Assigning a type to the variable 'if_condition_131094' (line 958)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 958, 21), 'if_condition_131094', if_condition_131094)
    # SSA begins for if statement (line 958)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 958)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 951)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 960)
    # Processing the call arguments (line 960)
    
    # Obtaining an instance of the builtin type 'tuple' (line 960)
    tuple_131097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 960)
    # Adding element type (line 960)
    str_131098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 30), 'str', '%s.%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 960)
    tuple_131099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 960)
    # Adding element type (line 960)
    # Getting the type of 'name' (line 960)
    name_131100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 41), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 41), tuple_131099, name_131100)
    # Adding element type (line 960)
    # Getting the type of 'n' (line 960)
    n_131101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 47), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 41), tuple_131099, n_131101)
    
    # Applying the binary operator '%' (line 960)
    result_mod_131102 = python_operator(stypy.reporting.localization.Localization(__file__, 960, 30), '%', str_131098, tuple_131099)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 30), tuple_131097, result_mod_131102)
    # Adding element type (line 960)
    # Getting the type of 'v' (line 960)
    v_131103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 51), 'v', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 960, 30), tuple_131097, v_131103)
    
    # Processing the call keyword arguments (line 960)
    kwargs_131104 = {}
    # Getting the type of 'stack' (line 960)
    stack_131095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 16), 'stack', False)
    # Obtaining the member 'append' of a type (line 960)
    append_131096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 960, 16), stack_131095, 'append')
    # Calling append(args, kwargs) (line 960)
    append_call_result_131105 = invoke(stypy.reporting.localization.Localization(__file__, 960, 16), append_131096, *[tuple_131097], **kwargs_131104)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 896)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isclass(...): (line 961)
    # Processing the call arguments (line 961)
    # Getting the type of 'item' (line 961)
    item_131108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 29), 'item', False)
    # Processing the call keyword arguments (line 961)
    kwargs_131109 = {}
    # Getting the type of 'inspect' (line 961)
    inspect_131106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 13), 'inspect', False)
    # Obtaining the member 'isclass' of a type (line 961)
    isclass_131107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 961, 13), inspect_131106, 'isclass')
    # Calling isclass(args, kwargs) (line 961)
    isclass_call_result_131110 = invoke(stypy.reporting.localization.Localization(__file__, 961, 13), isclass_131107, *[item_131108], **kwargs_131109)
    
    # Testing the type of an if condition (line 961)
    if_condition_131111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 961, 13), isclass_call_result_131110)
    # Assigning a type to the variable 'if_condition_131111' (line 961)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 961, 13), 'if_condition_131111', if_condition_131111)
    # SSA begins for if statement (line 961)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 962):
    
    # Assigning a Str to a Name (line 962):
    str_131112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 962, 19), 'str', 'class')
    # Assigning a type to the variable 'kind' (line 962)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 962, 12), 'kind', str_131112)
    
    
    # Call to _getmembers(...): (line 963)
    # Processing the call arguments (line 963)
    # Getting the type of 'item' (line 963)
    item_131114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 36), 'item', False)
    # Processing the call keyword arguments (line 963)
    kwargs_131115 = {}
    # Getting the type of '_getmembers' (line 963)
    _getmembers_131113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 963, 24), '_getmembers', False)
    # Calling _getmembers(args, kwargs) (line 963)
    _getmembers_call_result_131116 = invoke(stypy.reporting.localization.Localization(__file__, 963, 24), _getmembers_131113, *[item_131114], **kwargs_131115)
    
    # Testing the type of a for loop iterable (line 963)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 963, 12), _getmembers_call_result_131116)
    # Getting the type of the for loop variable (line 963)
    for_loop_var_131117 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 963, 12), _getmembers_call_result_131116)
    # Assigning a type to the variable 'n' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 12), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 963, 12), for_loop_var_131117))
    # Assigning a type to the variable 'v' (line 963)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 963, 12), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 963, 12), for_loop_var_131117))
    # SSA begins for a for statement (line 963)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 964)
    # Processing the call arguments (line 964)
    
    # Obtaining an instance of the builtin type 'tuple' (line 964)
    tuple_131120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 964)
    # Adding element type (line 964)
    str_131121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 30), 'str', '%s.%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 964)
    tuple_131122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 964, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 964)
    # Adding element type (line 964)
    # Getting the type of 'name' (line 964)
    name_131123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 41), 'name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 41), tuple_131122, name_131123)
    # Adding element type (line 964)
    # Getting the type of 'n' (line 964)
    n_131124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 47), 'n', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 41), tuple_131122, n_131124)
    
    # Applying the binary operator '%' (line 964)
    result_mod_131125 = python_operator(stypy.reporting.localization.Localization(__file__, 964, 30), '%', str_131121, tuple_131122)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 30), tuple_131120, result_mod_131125)
    # Adding element type (line 964)
    # Getting the type of 'v' (line 964)
    v_131126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 51), 'v', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 964, 30), tuple_131120, v_131126)
    
    # Processing the call keyword arguments (line 964)
    kwargs_131127 = {}
    # Getting the type of 'stack' (line 964)
    stack_131118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 16), 'stack', False)
    # Obtaining the member 'append' of a type (line 964)
    append_131119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 964, 16), stack_131118, 'append')
    # Calling append(args, kwargs) (line 964)
    append_call_result_131128 = invoke(stypy.reporting.localization.Localization(__file__, 964, 16), append_131119, *[tuple_131120], **kwargs_131127)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 961)
    module_type_store.open_ssa_branch('else')
    
    # Type idiom detected: calculating its left and rigth part (line 965)
    str_131129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 27), 'str', '__call__')
    # Getting the type of 'item' (line 965)
    item_131130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 21), 'item')
    
    (may_be_131131, more_types_in_union_131132) = may_provide_member(str_131129, item_131130)

    if may_be_131131:

        if more_types_in_union_131132:
            # Runtime conditional SSA (line 965)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'item' (line 965)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 13), 'item', remove_not_member_provider_from_union(item_131130, '__call__'))
        
        # Assigning a Str to a Name (line 966):
        
        # Assigning a Str to a Name (line 966):
        str_131133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 966, 19), 'str', 'func')
        # Assigning a type to the variable 'kind' (line 966)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 966, 12), 'kind', str_131133)

        if more_types_in_union_131132:
            # SSA join for if statement (line 965)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 961)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 896)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 968)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 969):
    
    # Assigning a Call to a Name (line 969):
    
    # Call to getdoc(...): (line 969)
    # Processing the call arguments (line 969)
    # Getting the type of 'item' (line 969)
    item_131136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 33), 'item', False)
    # Processing the call keyword arguments (line 969)
    kwargs_131137 = {}
    # Getting the type of 'inspect' (line 969)
    inspect_131134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 969, 18), 'inspect', False)
    # Obtaining the member 'getdoc' of a type (line 969)
    getdoc_131135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 969, 18), inspect_131134, 'getdoc')
    # Calling getdoc(args, kwargs) (line 969)
    getdoc_call_result_131138 = invoke(stypy.reporting.localization.Localization(__file__, 969, 18), getdoc_131135, *[item_131136], **kwargs_131137)
    
    # Assigning a type to the variable 'doc' (line 969)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 969, 12), 'doc', getdoc_call_result_131138)
    # SSA branch for the except part of a try statement (line 968)
    # SSA branch for the except 'NameError' branch of a try statement (line 968)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 972):
    
    # Assigning a Name to a Name (line 972):
    # Getting the type of 'None' (line 972)
    None_131139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 972, 18), 'None')
    # Assigning a type to the variable 'doc' (line 972)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 972, 12), 'doc', None_131139)
    # SSA join for try-except statement (line 968)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 973)
    # Getting the type of 'doc' (line 973)
    doc_131140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 8), 'doc')
    # Getting the type of 'None' (line 973)
    None_131141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 22), 'None')
    
    (may_be_131142, more_types_in_union_131143) = may_not_be_none(doc_131140, None_131141)

    if may_be_131142:

        if more_types_in_union_131143:
            # Runtime conditional SSA (line 973)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Tuple to a Subscript (line 974):
        
        # Assigning a Tuple to a Subscript (line 974):
        
        # Obtaining an instance of the builtin type 'tuple' (line 974)
        tuple_131144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 27), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 974)
        # Adding element type (line 974)
        # Getting the type of 'doc' (line 974)
        doc_131145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 27), 'doc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 974, 27), tuple_131144, doc_131145)
        # Adding element type (line 974)
        # Getting the type of 'kind' (line 974)
        kind_131146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 32), 'kind')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 974, 27), tuple_131144, kind_131146)
        # Adding element type (line 974)
        # Getting the type of 'index' (line 974)
        index_131147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 38), 'index')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 974, 27), tuple_131144, index_131147)
        
        # Getting the type of 'cache' (line 974)
        cache_131148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 12), 'cache')
        # Getting the type of 'name' (line 974)
        name_131149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 18), 'name')
        # Storing an element on a container (line 974)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 974, 12), cache_131148, (name_131149, tuple_131144))

        if more_types_in_union_131143:
            # SSA join for if statement (line 973)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for while statement (line 887)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'cache' (line 976)
    cache_131150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 11), 'cache')
    # Assigning a type to the variable 'stypy_return_type' (line 976)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 4), 'stypy_return_type', cache_131150)
    
    # ################# End of '_lookfor_generate_cache(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_lookfor_generate_cache' in the type store
    # Getting the type of 'stypy_return_type' (line 832)
    stypy_return_type_131151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 832, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_131151)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_lookfor_generate_cache'
    return stypy_return_type_131151

# Assigning a type to the variable '_lookfor_generate_cache' (line 832)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 832, 0), '_lookfor_generate_cache', _lookfor_generate_cache)

@norecursion
def _getmembers(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_getmembers'
    module_type_store = module_type_store.open_function_context('_getmembers', 978, 0, False)
    
    # Passed parameters checking function
    _getmembers.stypy_localization = localization
    _getmembers.stypy_type_of_self = None
    _getmembers.stypy_type_store = module_type_store
    _getmembers.stypy_function_name = '_getmembers'
    _getmembers.stypy_param_names_list = ['item']
    _getmembers.stypy_varargs_param_name = None
    _getmembers.stypy_kwargs_param_name = None
    _getmembers.stypy_call_defaults = defaults
    _getmembers.stypy_call_varargs = varargs
    _getmembers.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_getmembers', ['item'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_getmembers', localization, ['item'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_getmembers(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 979, 4))
    
    # 'import inspect' statement (line 979)
    import inspect

    import_module(stypy.reporting.localization.Localization(__file__, 979, 4), 'inspect', inspect, module_type_store)
    
    
    
    # SSA begins for try-except statement (line 980)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 981):
    
    # Assigning a Call to a Name (line 981):
    
    # Call to getmembers(...): (line 981)
    # Processing the call arguments (line 981)
    # Getting the type of 'item' (line 981)
    item_131154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 37), 'item', False)
    # Processing the call keyword arguments (line 981)
    kwargs_131155 = {}
    # Getting the type of 'inspect' (line 981)
    inspect_131152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 18), 'inspect', False)
    # Obtaining the member 'getmembers' of a type (line 981)
    getmembers_131153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 981, 18), inspect_131152, 'getmembers')
    # Calling getmembers(args, kwargs) (line 981)
    getmembers_call_result_131156 = invoke(stypy.reporting.localization.Localization(__file__, 981, 18), getmembers_131153, *[item_131154], **kwargs_131155)
    
    # Assigning a type to the variable 'members' (line 981)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 981, 8), 'members', getmembers_call_result_131156)
    # SSA branch for the except part of a try statement (line 980)
    # SSA branch for the except 'Exception' branch of a try statement (line 980)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a ListComp to a Name (line 983):
    
    # Assigning a ListComp to a Name (line 983):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to dir(...): (line 983)
    # Processing the call arguments (line 983)
    # Getting the type of 'item' (line 983)
    item_131170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 54), 'item', False)
    # Processing the call keyword arguments (line 983)
    kwargs_131171 = {}
    # Getting the type of 'dir' (line 983)
    dir_131169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 50), 'dir', False)
    # Calling dir(args, kwargs) (line 983)
    dir_call_result_131172 = invoke(stypy.reporting.localization.Localization(__file__, 983, 50), dir_131169, *[item_131170], **kwargs_131171)
    
    comprehension_131173 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 983, 19), dir_call_result_131172)
    # Assigning a type to the variable 'x' (line 983)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 983, 19), 'x', comprehension_131173)
    
    # Call to hasattr(...): (line 984)
    # Processing the call arguments (line 984)
    # Getting the type of 'item' (line 984)
    item_131165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 30), 'item', False)
    # Getting the type of 'x' (line 984)
    x_131166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 36), 'x', False)
    # Processing the call keyword arguments (line 984)
    kwargs_131167 = {}
    # Getting the type of 'hasattr' (line 984)
    hasattr_131164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 22), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 984)
    hasattr_call_result_131168 = invoke(stypy.reporting.localization.Localization(__file__, 984, 22), hasattr_131164, *[item_131165, x_131166], **kwargs_131167)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 983)
    tuple_131157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 983, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 983)
    # Adding element type (line 983)
    # Getting the type of 'x' (line 983)
    x_131158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 20), 'x')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 983, 20), tuple_131157, x_131158)
    # Adding element type (line 983)
    
    # Call to getattr(...): (line 983)
    # Processing the call arguments (line 983)
    # Getting the type of 'item' (line 983)
    item_131160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 31), 'item', False)
    # Getting the type of 'x' (line 983)
    x_131161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 37), 'x', False)
    # Processing the call keyword arguments (line 983)
    kwargs_131162 = {}
    # Getting the type of 'getattr' (line 983)
    getattr_131159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 983, 23), 'getattr', False)
    # Calling getattr(args, kwargs) (line 983)
    getattr_call_result_131163 = invoke(stypy.reporting.localization.Localization(__file__, 983, 23), getattr_131159, *[item_131160, x_131161], **kwargs_131162)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 983, 20), tuple_131157, getattr_call_result_131163)
    
    list_131174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 983, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 983, 19), list_131174, tuple_131157)
    # Assigning a type to the variable 'members' (line 983)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 983, 8), 'members', list_131174)
    # SSA join for try-except statement (line 980)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'members' (line 985)
    members_131175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 985, 11), 'members')
    # Assigning a type to the variable 'stypy_return_type' (line 985)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 985, 4), 'stypy_return_type', members_131175)
    
    # ################# End of '_getmembers(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_getmembers' in the type store
    # Getting the type of 'stypy_return_type' (line 978)
    stypy_return_type_131176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 978, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_131176)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_getmembers'
    return stypy_return_type_131176

# Assigning a type to the variable '_getmembers' (line 978)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 978, 0), '_getmembers', _getmembers)
# Declaration of the 'SafeEval' class

class SafeEval(object, ):
    str_131177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1015, (-1)), 'str', '\n    Object to evaluate constant string expressions.\n\n    This includes strings with lists, dicts and tuples using the abstract\n    syntax tree created by ``compiler.parse``.\n\n    .. deprecated:: 1.10.0\n\n    See Also\n    --------\n    safe_eval\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1016, 4, False)
        # Assigning a type to the variable 'self' (line 1017)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1017, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to warn(...): (line 1018)
        # Processing the call arguments (line 1018)
        str_131180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1018, 22), 'str', 'SafeEval is deprecated in 1.10 and will be removed.')
        # Getting the type of 'DeprecationWarning' (line 1019)
        DeprecationWarning_131181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1019, 22), 'DeprecationWarning', False)
        # Processing the call keyword arguments (line 1018)
        kwargs_131182 = {}
        # Getting the type of 'warnings' (line 1018)
        warnings_131178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1018, 8), 'warnings', False)
        # Obtaining the member 'warn' of a type (line 1018)
        warn_131179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1018, 8), warnings_131178, 'warn')
        # Calling warn(args, kwargs) (line 1018)
        warn_call_result_131183 = invoke(stypy.reporting.localization.Localization(__file__, 1018, 8), warn_131179, *[str_131180, DeprecationWarning_131181], **kwargs_131182)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def visit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit'
        module_type_store = module_type_store.open_function_context('visit', 1021, 4, False)
        # Assigning a type to the variable 'self' (line 1022)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1022, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SafeEval.visit.__dict__.__setitem__('stypy_localization', localization)
        SafeEval.visit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SafeEval.visit.__dict__.__setitem__('stypy_type_store', module_type_store)
        SafeEval.visit.__dict__.__setitem__('stypy_function_name', 'SafeEval.visit')
        SafeEval.visit.__dict__.__setitem__('stypy_param_names_list', ['node'])
        SafeEval.visit.__dict__.__setitem__('stypy_varargs_param_name', None)
        SafeEval.visit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SafeEval.visit.__dict__.__setitem__('stypy_call_defaults', defaults)
        SafeEval.visit.__dict__.__setitem__('stypy_call_varargs', varargs)
        SafeEval.visit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SafeEval.visit.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.visit', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit(...)' code ##################

        
        # Assigning a Attribute to a Name (line 1022):
        
        # Assigning a Attribute to a Name (line 1022):
        # Getting the type of 'node' (line 1022)
        node_131184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1022, 14), 'node')
        # Obtaining the member '__class__' of a type (line 1022)
        class___131185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1022, 14), node_131184, '__class__')
        # Assigning a type to the variable 'cls' (line 1022)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1022, 8), 'cls', class___131185)
        
        # Assigning a Call to a Name (line 1023):
        
        # Assigning a Call to a Name (line 1023):
        
        # Call to getattr(...): (line 1023)
        # Processing the call arguments (line 1023)
        # Getting the type of 'self' (line 1023)
        self_131187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 23), 'self', False)
        str_131188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1023, 29), 'str', 'visit')
        # Getting the type of 'cls' (line 1023)
        cls_131189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 39), 'cls', False)
        # Obtaining the member '__name__' of a type (line 1023)
        name___131190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1023, 39), cls_131189, '__name__')
        # Applying the binary operator '+' (line 1023)
        result_add_131191 = python_operator(stypy.reporting.localization.Localization(__file__, 1023, 29), '+', str_131188, name___131190)
        
        # Getting the type of 'self' (line 1023)
        self_131192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 53), 'self', False)
        # Obtaining the member 'default' of a type (line 1023)
        default_131193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1023, 53), self_131192, 'default')
        # Processing the call keyword arguments (line 1023)
        kwargs_131194 = {}
        # Getting the type of 'getattr' (line 1023)
        getattr_131186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1023, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 1023)
        getattr_call_result_131195 = invoke(stypy.reporting.localization.Localization(__file__, 1023, 15), getattr_131186, *[self_131187, result_add_131191, default_131193], **kwargs_131194)
        
        # Assigning a type to the variable 'meth' (line 1023)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1023, 8), 'meth', getattr_call_result_131195)
        
        # Call to meth(...): (line 1024)
        # Processing the call arguments (line 1024)
        # Getting the type of 'node' (line 1024)
        node_131197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1024, 20), 'node', False)
        # Processing the call keyword arguments (line 1024)
        kwargs_131198 = {}
        # Getting the type of 'meth' (line 1024)
        meth_131196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1024, 15), 'meth', False)
        # Calling meth(args, kwargs) (line 1024)
        meth_call_result_131199 = invoke(stypy.reporting.localization.Localization(__file__, 1024, 15), meth_131196, *[node_131197], **kwargs_131198)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1024)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1024, 8), 'stypy_return_type', meth_call_result_131199)
        
        # ################# End of 'visit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit' in the type store
        # Getting the type of 'stypy_return_type' (line 1021)
        stypy_return_type_131200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1021, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131200)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit'
        return stypy_return_type_131200


    @norecursion
    def default(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'default'
        module_type_store = module_type_store.open_function_context('default', 1026, 4, False)
        # Assigning a type to the variable 'self' (line 1027)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1027, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SafeEval.default.__dict__.__setitem__('stypy_localization', localization)
        SafeEval.default.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SafeEval.default.__dict__.__setitem__('stypy_type_store', module_type_store)
        SafeEval.default.__dict__.__setitem__('stypy_function_name', 'SafeEval.default')
        SafeEval.default.__dict__.__setitem__('stypy_param_names_list', ['node'])
        SafeEval.default.__dict__.__setitem__('stypy_varargs_param_name', None)
        SafeEval.default.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SafeEval.default.__dict__.__setitem__('stypy_call_defaults', defaults)
        SafeEval.default.__dict__.__setitem__('stypy_call_varargs', varargs)
        SafeEval.default.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SafeEval.default.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.default', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'default', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'default(...)' code ##################

        
        # Call to SyntaxError(...): (line 1027)
        # Processing the call arguments (line 1027)
        str_131202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1027, 26), 'str', 'Unsupported source construct: %s')
        # Getting the type of 'node' (line 1028)
        node_131203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1028, 28), 'node', False)
        # Obtaining the member '__class__' of a type (line 1028)
        class___131204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1028, 28), node_131203, '__class__')
        # Applying the binary operator '%' (line 1027)
        result_mod_131205 = python_operator(stypy.reporting.localization.Localization(__file__, 1027, 26), '%', str_131202, class___131204)
        
        # Processing the call keyword arguments (line 1027)
        kwargs_131206 = {}
        # Getting the type of 'SyntaxError' (line 1027)
        SyntaxError_131201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1027, 14), 'SyntaxError', False)
        # Calling SyntaxError(args, kwargs) (line 1027)
        SyntaxError_call_result_131207 = invoke(stypy.reporting.localization.Localization(__file__, 1027, 14), SyntaxError_131201, *[result_mod_131205], **kwargs_131206)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1027, 8), SyntaxError_call_result_131207, 'raise parameter', BaseException)
        
        # ################# End of 'default(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'default' in the type store
        # Getting the type of 'stypy_return_type' (line 1026)
        stypy_return_type_131208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1026, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131208)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'default'
        return stypy_return_type_131208


    @norecursion
    def visitExpression(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visitExpression'
        module_type_store = module_type_store.open_function_context('visitExpression', 1030, 4, False)
        # Assigning a type to the variable 'self' (line 1031)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SafeEval.visitExpression.__dict__.__setitem__('stypy_localization', localization)
        SafeEval.visitExpression.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SafeEval.visitExpression.__dict__.__setitem__('stypy_type_store', module_type_store)
        SafeEval.visitExpression.__dict__.__setitem__('stypy_function_name', 'SafeEval.visitExpression')
        SafeEval.visitExpression.__dict__.__setitem__('stypy_param_names_list', ['node'])
        SafeEval.visitExpression.__dict__.__setitem__('stypy_varargs_param_name', None)
        SafeEval.visitExpression.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SafeEval.visitExpression.__dict__.__setitem__('stypy_call_defaults', defaults)
        SafeEval.visitExpression.__dict__.__setitem__('stypy_call_varargs', varargs)
        SafeEval.visitExpression.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SafeEval.visitExpression.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.visitExpression', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visitExpression', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visitExpression(...)' code ##################

        
        # Call to visit(...): (line 1031)
        # Processing the call arguments (line 1031)
        # Getting the type of 'node' (line 1031)
        node_131211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 26), 'node', False)
        # Obtaining the member 'body' of a type (line 1031)
        body_131212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1031, 26), node_131211, 'body')
        # Processing the call keyword arguments (line 1031)
        kwargs_131213 = {}
        # Getting the type of 'self' (line 1031)
        self_131209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1031, 15), 'self', False)
        # Obtaining the member 'visit' of a type (line 1031)
        visit_131210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1031, 15), self_131209, 'visit')
        # Calling visit(args, kwargs) (line 1031)
        visit_call_result_131214 = invoke(stypy.reporting.localization.Localization(__file__, 1031, 15), visit_131210, *[body_131212], **kwargs_131213)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1031)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1031, 8), 'stypy_return_type', visit_call_result_131214)
        
        # ################# End of 'visitExpression(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visitExpression' in the type store
        # Getting the type of 'stypy_return_type' (line 1030)
        stypy_return_type_131215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1030, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131215)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visitExpression'
        return stypy_return_type_131215


    @norecursion
    def visitNum(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visitNum'
        module_type_store = module_type_store.open_function_context('visitNum', 1033, 4, False)
        # Assigning a type to the variable 'self' (line 1034)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SafeEval.visitNum.__dict__.__setitem__('stypy_localization', localization)
        SafeEval.visitNum.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SafeEval.visitNum.__dict__.__setitem__('stypy_type_store', module_type_store)
        SafeEval.visitNum.__dict__.__setitem__('stypy_function_name', 'SafeEval.visitNum')
        SafeEval.visitNum.__dict__.__setitem__('stypy_param_names_list', ['node'])
        SafeEval.visitNum.__dict__.__setitem__('stypy_varargs_param_name', None)
        SafeEval.visitNum.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SafeEval.visitNum.__dict__.__setitem__('stypy_call_defaults', defaults)
        SafeEval.visitNum.__dict__.__setitem__('stypy_call_varargs', varargs)
        SafeEval.visitNum.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SafeEval.visitNum.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.visitNum', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visitNum', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visitNum(...)' code ##################

        # Getting the type of 'node' (line 1034)
        node_131216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1034, 15), 'node')
        # Obtaining the member 'n' of a type (line 1034)
        n_131217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1034, 15), node_131216, 'n')
        # Assigning a type to the variable 'stypy_return_type' (line 1034)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1034, 8), 'stypy_return_type', n_131217)
        
        # ################# End of 'visitNum(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visitNum' in the type store
        # Getting the type of 'stypy_return_type' (line 1033)
        stypy_return_type_131218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1033, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131218)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visitNum'
        return stypy_return_type_131218


    @norecursion
    def visitStr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visitStr'
        module_type_store = module_type_store.open_function_context('visitStr', 1036, 4, False)
        # Assigning a type to the variable 'self' (line 1037)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SafeEval.visitStr.__dict__.__setitem__('stypy_localization', localization)
        SafeEval.visitStr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SafeEval.visitStr.__dict__.__setitem__('stypy_type_store', module_type_store)
        SafeEval.visitStr.__dict__.__setitem__('stypy_function_name', 'SafeEval.visitStr')
        SafeEval.visitStr.__dict__.__setitem__('stypy_param_names_list', ['node'])
        SafeEval.visitStr.__dict__.__setitem__('stypy_varargs_param_name', None)
        SafeEval.visitStr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SafeEval.visitStr.__dict__.__setitem__('stypy_call_defaults', defaults)
        SafeEval.visitStr.__dict__.__setitem__('stypy_call_varargs', varargs)
        SafeEval.visitStr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SafeEval.visitStr.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.visitStr', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visitStr', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visitStr(...)' code ##################

        # Getting the type of 'node' (line 1037)
        node_131219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1037, 15), 'node')
        # Obtaining the member 's' of a type (line 1037)
        s_131220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1037, 15), node_131219, 's')
        # Assigning a type to the variable 'stypy_return_type' (line 1037)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1037, 8), 'stypy_return_type', s_131220)
        
        # ################# End of 'visitStr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visitStr' in the type store
        # Getting the type of 'stypy_return_type' (line 1036)
        stypy_return_type_131221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1036, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131221)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visitStr'
        return stypy_return_type_131221


    @norecursion
    def visitBytes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visitBytes'
        module_type_store = module_type_store.open_function_context('visitBytes', 1039, 4, False)
        # Assigning a type to the variable 'self' (line 1040)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1040, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SafeEval.visitBytes.__dict__.__setitem__('stypy_localization', localization)
        SafeEval.visitBytes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SafeEval.visitBytes.__dict__.__setitem__('stypy_type_store', module_type_store)
        SafeEval.visitBytes.__dict__.__setitem__('stypy_function_name', 'SafeEval.visitBytes')
        SafeEval.visitBytes.__dict__.__setitem__('stypy_param_names_list', ['node'])
        SafeEval.visitBytes.__dict__.__setitem__('stypy_varargs_param_name', None)
        SafeEval.visitBytes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SafeEval.visitBytes.__dict__.__setitem__('stypy_call_defaults', defaults)
        SafeEval.visitBytes.__dict__.__setitem__('stypy_call_varargs', varargs)
        SafeEval.visitBytes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SafeEval.visitBytes.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.visitBytes', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visitBytes', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visitBytes(...)' code ##################

        # Getting the type of 'node' (line 1040)
        node_131222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1040, 15), 'node')
        # Obtaining the member 's' of a type (line 1040)
        s_131223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1040, 15), node_131222, 's')
        # Assigning a type to the variable 'stypy_return_type' (line 1040)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1040, 8), 'stypy_return_type', s_131223)
        
        # ################# End of 'visitBytes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visitBytes' in the type store
        # Getting the type of 'stypy_return_type' (line 1039)
        stypy_return_type_131224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1039, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131224)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visitBytes'
        return stypy_return_type_131224


    @norecursion
    def visitDict(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visitDict'
        module_type_store = module_type_store.open_function_context('visitDict', 1042, 4, False)
        # Assigning a type to the variable 'self' (line 1043)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1043, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SafeEval.visitDict.__dict__.__setitem__('stypy_localization', localization)
        SafeEval.visitDict.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SafeEval.visitDict.__dict__.__setitem__('stypy_type_store', module_type_store)
        SafeEval.visitDict.__dict__.__setitem__('stypy_function_name', 'SafeEval.visitDict')
        SafeEval.visitDict.__dict__.__setitem__('stypy_param_names_list', ['node'])
        SafeEval.visitDict.__dict__.__setitem__('stypy_varargs_param_name', None)
        SafeEval.visitDict.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        SafeEval.visitDict.__dict__.__setitem__('stypy_call_defaults', defaults)
        SafeEval.visitDict.__dict__.__setitem__('stypy_call_varargs', varargs)
        SafeEval.visitDict.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SafeEval.visitDict.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.visitDict', ['node'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visitDict', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visitDict(...)' code ##################

        
        # Call to dict(...): (line 1043)
        # Processing the call arguments (line 1043)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to zip(...): (line 1044)
        # Processing the call arguments (line 1044)
        # Getting the type of 'node' (line 1044)
        node_131238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 37), 'node', False)
        # Obtaining the member 'keys' of a type (line 1044)
        keys_131239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 37), node_131238, 'keys')
        # Getting the type of 'node' (line 1044)
        node_131240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 48), 'node', False)
        # Obtaining the member 'values' of a type (line 1044)
        values_131241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1044, 48), node_131240, 'values')
        # Processing the call keyword arguments (line 1044)
        kwargs_131242 = {}
        # Getting the type of 'zip' (line 1044)
        zip_131237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1044, 33), 'zip', False)
        # Calling zip(args, kwargs) (line 1044)
        zip_call_result_131243 = invoke(stypy.reporting.localization.Localization(__file__, 1044, 33), zip_131237, *[keys_131239, values_131241], **kwargs_131242)
        
        comprehension_131244 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 21), zip_call_result_131243)
        # Assigning a type to the variable 'k' (line 1043)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1043, 21), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 21), comprehension_131244))
        # Assigning a type to the variable 'v' (line 1043)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1043, 21), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 21), comprehension_131244))
        
        # Obtaining an instance of the builtin type 'tuple' (line 1043)
        tuple_131226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 1043)
        # Adding element type (line 1043)
        
        # Call to visit(...): (line 1043)
        # Processing the call arguments (line 1043)
        # Getting the type of 'k' (line 1043)
        k_131229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 33), 'k', False)
        # Processing the call keyword arguments (line 1043)
        kwargs_131230 = {}
        # Getting the type of 'self' (line 1043)
        self_131227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 22), 'self', False)
        # Obtaining the member 'visit' of a type (line 1043)
        visit_131228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 22), self_131227, 'visit')
        # Calling visit(args, kwargs) (line 1043)
        visit_call_result_131231 = invoke(stypy.reporting.localization.Localization(__file__, 1043, 22), visit_131228, *[k_131229], **kwargs_131230)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 22), tuple_131226, visit_call_result_131231)
        # Adding element type (line 1043)
        
        # Call to visit(...): (line 1043)
        # Processing the call arguments (line 1043)
        # Getting the type of 'v' (line 1043)
        v_131234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 48), 'v', False)
        # Processing the call keyword arguments (line 1043)
        kwargs_131235 = {}
        # Getting the type of 'self' (line 1043)
        self_131232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 37), 'self', False)
        # Obtaining the member 'visit' of a type (line 1043)
        visit_131233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1043, 37), self_131232, 'visit')
        # Calling visit(args, kwargs) (line 1043)
        visit_call_result_131236 = invoke(stypy.reporting.localization.Localization(__file__, 1043, 37), visit_131233, *[v_131234], **kwargs_131235)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 22), tuple_131226, visit_call_result_131236)
        
        list_131245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1043, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1043, 21), list_131245, tuple_131226)
        # Processing the call keyword arguments (line 1043)
        kwargs_131246 = {}
        # Getting the type of 'dict' (line 1043)
        dict_131225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1043, 15), 'dict', False)
        # Calling dict(args, kwargs) (line 1043)
        dict_call_result_131247 = invoke(stypy.reporting.localization.Localization(__file__, 1043, 15), dict_131225, *[list_131245], **kwargs_131246)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1043)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1043, 8), 'stypy_return_type', dict_call_result_131247)
        
        # ################# End of 'visitDict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visitDict' in the type store
        # Getting the type of 'stypy_return_type' (line 1042)
        stypy_return_type_131248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1042, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131248)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visitDict'
        return stypy_return_type_131248


    @norecursion
    def visitTuple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visitTuple'
        module_type_store = module_type_store.open_function_context('visitTuple', 1046, 4, False)
        # Assigning a type to the variable 'self' (line 1047)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1047, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SafeEval.visitTuple.__dict__.__setitem__('stypy_localization', localization)
        SafeEval.visitTuple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SafeEval.visitTuple.__dict__.__setitem__('stypy_type_store', module_type_store)
        SafeEval.visitTuple.__dict__.__setitem__('stypy_function_name', 'SafeEval.visitTuple')
        SafeEval.visitTuple.__dict__.__setitem__('stypy_param_names_list', ['node'])
        SafeEval.visitTuple.__dict__.__setitem__('stypy_varargs_param_name', None)
        SafeEval.visitTuple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SafeEval.visitTuple.__dict__.__setitem__('stypy_call_defaults', defaults)
        SafeEval.visitTuple.__dict__.__setitem__('stypy_call_varargs', varargs)
        SafeEval.visitTuple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SafeEval.visitTuple.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.visitTuple', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visitTuple', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visitTuple(...)' code ##################

        
        # Call to tuple(...): (line 1047)
        # Processing the call arguments (line 1047)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'node' (line 1047)
        node_131255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 45), 'node', False)
        # Obtaining the member 'elts' of a type (line 1047)
        elts_131256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1047, 45), node_131255, 'elts')
        comprehension_131257 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1047, 22), elts_131256)
        # Assigning a type to the variable 'i' (line 1047)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1047, 22), 'i', comprehension_131257)
        
        # Call to visit(...): (line 1047)
        # Processing the call arguments (line 1047)
        # Getting the type of 'i' (line 1047)
        i_131252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 33), 'i', False)
        # Processing the call keyword arguments (line 1047)
        kwargs_131253 = {}
        # Getting the type of 'self' (line 1047)
        self_131250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 22), 'self', False)
        # Obtaining the member 'visit' of a type (line 1047)
        visit_131251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1047, 22), self_131250, 'visit')
        # Calling visit(args, kwargs) (line 1047)
        visit_call_result_131254 = invoke(stypy.reporting.localization.Localization(__file__, 1047, 22), visit_131251, *[i_131252], **kwargs_131253)
        
        list_131258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1047, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1047, 22), list_131258, visit_call_result_131254)
        # Processing the call keyword arguments (line 1047)
        kwargs_131259 = {}
        # Getting the type of 'tuple' (line 1047)
        tuple_131249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1047, 15), 'tuple', False)
        # Calling tuple(args, kwargs) (line 1047)
        tuple_call_result_131260 = invoke(stypy.reporting.localization.Localization(__file__, 1047, 15), tuple_131249, *[list_131258], **kwargs_131259)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1047)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1047, 8), 'stypy_return_type', tuple_call_result_131260)
        
        # ################# End of 'visitTuple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visitTuple' in the type store
        # Getting the type of 'stypy_return_type' (line 1046)
        stypy_return_type_131261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1046, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visitTuple'
        return stypy_return_type_131261


    @norecursion
    def visitList(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visitList'
        module_type_store = module_type_store.open_function_context('visitList', 1049, 4, False)
        # Assigning a type to the variable 'self' (line 1050)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1050, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SafeEval.visitList.__dict__.__setitem__('stypy_localization', localization)
        SafeEval.visitList.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SafeEval.visitList.__dict__.__setitem__('stypy_type_store', module_type_store)
        SafeEval.visitList.__dict__.__setitem__('stypy_function_name', 'SafeEval.visitList')
        SafeEval.visitList.__dict__.__setitem__('stypy_param_names_list', ['node'])
        SafeEval.visitList.__dict__.__setitem__('stypy_varargs_param_name', None)
        SafeEval.visitList.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SafeEval.visitList.__dict__.__setitem__('stypy_call_defaults', defaults)
        SafeEval.visitList.__dict__.__setitem__('stypy_call_varargs', varargs)
        SafeEval.visitList.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SafeEval.visitList.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.visitList', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visitList', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visitList(...)' code ##################

        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'node' (line 1050)
        node_131267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 39), 'node')
        # Obtaining the member 'elts' of a type (line 1050)
        elts_131268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 39), node_131267, 'elts')
        comprehension_131269 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1050, 16), elts_131268)
        # Assigning a type to the variable 'i' (line 1050)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1050, 16), 'i', comprehension_131269)
        
        # Call to visit(...): (line 1050)
        # Processing the call arguments (line 1050)
        # Getting the type of 'i' (line 1050)
        i_131264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 27), 'i', False)
        # Processing the call keyword arguments (line 1050)
        kwargs_131265 = {}
        # Getting the type of 'self' (line 1050)
        self_131262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1050, 16), 'self', False)
        # Obtaining the member 'visit' of a type (line 1050)
        visit_131263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1050, 16), self_131262, 'visit')
        # Calling visit(args, kwargs) (line 1050)
        visit_call_result_131266 = invoke(stypy.reporting.localization.Localization(__file__, 1050, 16), visit_131263, *[i_131264], **kwargs_131265)
        
        list_131270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1050, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 1050, 16), list_131270, visit_call_result_131266)
        # Assigning a type to the variable 'stypy_return_type' (line 1050)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1050, 8), 'stypy_return_type', list_131270)
        
        # ################# End of 'visitList(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visitList' in the type store
        # Getting the type of 'stypy_return_type' (line 1049)
        stypy_return_type_131271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1049, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131271)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visitList'
        return stypy_return_type_131271


    @norecursion
    def visitUnaryOp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visitUnaryOp'
        module_type_store = module_type_store.open_function_context('visitUnaryOp', 1052, 4, False)
        # Assigning a type to the variable 'self' (line 1053)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1053, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SafeEval.visitUnaryOp.__dict__.__setitem__('stypy_localization', localization)
        SafeEval.visitUnaryOp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SafeEval.visitUnaryOp.__dict__.__setitem__('stypy_type_store', module_type_store)
        SafeEval.visitUnaryOp.__dict__.__setitem__('stypy_function_name', 'SafeEval.visitUnaryOp')
        SafeEval.visitUnaryOp.__dict__.__setitem__('stypy_param_names_list', ['node'])
        SafeEval.visitUnaryOp.__dict__.__setitem__('stypy_varargs_param_name', None)
        SafeEval.visitUnaryOp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SafeEval.visitUnaryOp.__dict__.__setitem__('stypy_call_defaults', defaults)
        SafeEval.visitUnaryOp.__dict__.__setitem__('stypy_call_varargs', varargs)
        SafeEval.visitUnaryOp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SafeEval.visitUnaryOp.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.visitUnaryOp', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visitUnaryOp', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visitUnaryOp(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1053, 8))
        
        # 'import ast' statement (line 1053)
        import ast

        import_module(stypy.reporting.localization.Localization(__file__, 1053, 8), 'ast', ast, module_type_store)
        
        
        
        # Call to isinstance(...): (line 1054)
        # Processing the call arguments (line 1054)
        # Getting the type of 'node' (line 1054)
        node_131273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 22), 'node', False)
        # Obtaining the member 'op' of a type (line 1054)
        op_131274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 22), node_131273, 'op')
        # Getting the type of 'ast' (line 1054)
        ast_131275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 31), 'ast', False)
        # Obtaining the member 'UAdd' of a type (line 1054)
        UAdd_131276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1054, 31), ast_131275, 'UAdd')
        # Processing the call keyword arguments (line 1054)
        kwargs_131277 = {}
        # Getting the type of 'isinstance' (line 1054)
        isinstance_131272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1054, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 1054)
        isinstance_call_result_131278 = invoke(stypy.reporting.localization.Localization(__file__, 1054, 11), isinstance_131272, *[op_131274, UAdd_131276], **kwargs_131277)
        
        # Testing the type of an if condition (line 1054)
        if_condition_131279 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1054, 8), isinstance_call_result_131278)
        # Assigning a type to the variable 'if_condition_131279' (line 1054)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1054, 8), 'if_condition_131279', if_condition_131279)
        # SSA begins for if statement (line 1054)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to visit(...): (line 1055)
        # Processing the call arguments (line 1055)
        # Getting the type of 'node' (line 1055)
        node_131282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 31), 'node', False)
        # Obtaining the member 'operand' of a type (line 1055)
        operand_131283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1055, 31), node_131282, 'operand')
        # Processing the call keyword arguments (line 1055)
        kwargs_131284 = {}
        # Getting the type of 'self' (line 1055)
        self_131280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1055, 20), 'self', False)
        # Obtaining the member 'visit' of a type (line 1055)
        visit_131281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1055, 20), self_131280, 'visit')
        # Calling visit(args, kwargs) (line 1055)
        visit_call_result_131285 = invoke(stypy.reporting.localization.Localization(__file__, 1055, 20), visit_131281, *[operand_131283], **kwargs_131284)
        
        # Applying the 'uadd' unary operator (line 1055)
        result___pos___131286 = python_operator(stypy.reporting.localization.Localization(__file__, 1055, 19), 'uadd', visit_call_result_131285)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1055)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1055, 12), 'stypy_return_type', result___pos___131286)
        # SSA branch for the else part of an if statement (line 1054)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinstance(...): (line 1056)
        # Processing the call arguments (line 1056)
        # Getting the type of 'node' (line 1056)
        node_131288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 24), 'node', False)
        # Obtaining the member 'op' of a type (line 1056)
        op_131289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 24), node_131288, 'op')
        # Getting the type of 'ast' (line 1056)
        ast_131290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 33), 'ast', False)
        # Obtaining the member 'USub' of a type (line 1056)
        USub_131291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1056, 33), ast_131290, 'USub')
        # Processing the call keyword arguments (line 1056)
        kwargs_131292 = {}
        # Getting the type of 'isinstance' (line 1056)
        isinstance_131287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1056, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 1056)
        isinstance_call_result_131293 = invoke(stypy.reporting.localization.Localization(__file__, 1056, 13), isinstance_131287, *[op_131289, USub_131291], **kwargs_131292)
        
        # Testing the type of an if condition (line 1056)
        if_condition_131294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1056, 13), isinstance_call_result_131293)
        # Assigning a type to the variable 'if_condition_131294' (line 1056)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1056, 13), 'if_condition_131294', if_condition_131294)
        # SSA begins for if statement (line 1056)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to visit(...): (line 1057)
        # Processing the call arguments (line 1057)
        # Getting the type of 'node' (line 1057)
        node_131297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 31), 'node', False)
        # Obtaining the member 'operand' of a type (line 1057)
        operand_131298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1057, 31), node_131297, 'operand')
        # Processing the call keyword arguments (line 1057)
        kwargs_131299 = {}
        # Getting the type of 'self' (line 1057)
        self_131295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1057, 20), 'self', False)
        # Obtaining the member 'visit' of a type (line 1057)
        visit_131296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1057, 20), self_131295, 'visit')
        # Calling visit(args, kwargs) (line 1057)
        visit_call_result_131300 = invoke(stypy.reporting.localization.Localization(__file__, 1057, 20), visit_131296, *[operand_131298], **kwargs_131299)
        
        # Applying the 'usub' unary operator (line 1057)
        result___neg___131301 = python_operator(stypy.reporting.localization.Localization(__file__, 1057, 19), 'usub', visit_call_result_131300)
        
        # Assigning a type to the variable 'stypy_return_type' (line 1057)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1057, 12), 'stypy_return_type', result___neg___131301)
        # SSA branch for the else part of an if statement (line 1056)
        module_type_store.open_ssa_branch('else')
        
        # Call to SyntaxError(...): (line 1059)
        # Processing the call arguments (line 1059)
        str_131303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1059, 30), 'str', 'Unknown unary op: %r')
        # Getting the type of 'node' (line 1059)
        node_131304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 55), 'node', False)
        # Obtaining the member 'op' of a type (line 1059)
        op_131305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1059, 55), node_131304, 'op')
        # Applying the binary operator '%' (line 1059)
        result_mod_131306 = python_operator(stypy.reporting.localization.Localization(__file__, 1059, 30), '%', str_131303, op_131305)
        
        # Processing the call keyword arguments (line 1059)
        kwargs_131307 = {}
        # Getting the type of 'SyntaxError' (line 1059)
        SyntaxError_131302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1059, 18), 'SyntaxError', False)
        # Calling SyntaxError(args, kwargs) (line 1059)
        SyntaxError_call_result_131308 = invoke(stypy.reporting.localization.Localization(__file__, 1059, 18), SyntaxError_131302, *[result_mod_131306], **kwargs_131307)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1059, 12), SyntaxError_call_result_131308, 'raise parameter', BaseException)
        # SSA join for if statement (line 1056)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1054)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'visitUnaryOp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visitUnaryOp' in the type store
        # Getting the type of 'stypy_return_type' (line 1052)
        stypy_return_type_131309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1052, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131309)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visitUnaryOp'
        return stypy_return_type_131309


    @norecursion
    def visitName(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visitName'
        module_type_store = module_type_store.open_function_context('visitName', 1061, 4, False)
        # Assigning a type to the variable 'self' (line 1062)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1062, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SafeEval.visitName.__dict__.__setitem__('stypy_localization', localization)
        SafeEval.visitName.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SafeEval.visitName.__dict__.__setitem__('stypy_type_store', module_type_store)
        SafeEval.visitName.__dict__.__setitem__('stypy_function_name', 'SafeEval.visitName')
        SafeEval.visitName.__dict__.__setitem__('stypy_param_names_list', ['node'])
        SafeEval.visitName.__dict__.__setitem__('stypy_varargs_param_name', None)
        SafeEval.visitName.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SafeEval.visitName.__dict__.__setitem__('stypy_call_defaults', defaults)
        SafeEval.visitName.__dict__.__setitem__('stypy_call_varargs', varargs)
        SafeEval.visitName.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SafeEval.visitName.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.visitName', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visitName', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visitName(...)' code ##################

        
        
        # Getting the type of 'node' (line 1062)
        node_131310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1062, 11), 'node')
        # Obtaining the member 'id' of a type (line 1062)
        id_131311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1062, 11), node_131310, 'id')
        str_131312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1062, 22), 'str', 'False')
        # Applying the binary operator '==' (line 1062)
        result_eq_131313 = python_operator(stypy.reporting.localization.Localization(__file__, 1062, 11), '==', id_131311, str_131312)
        
        # Testing the type of an if condition (line 1062)
        if_condition_131314 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1062, 8), result_eq_131313)
        # Assigning a type to the variable 'if_condition_131314' (line 1062)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1062, 8), 'if_condition_131314', if_condition_131314)
        # SSA begins for if statement (line 1062)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 1063)
        False_131315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1063, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 1063)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1063, 12), 'stypy_return_type', False_131315)
        # SSA branch for the else part of an if statement (line 1062)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'node' (line 1064)
        node_131316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1064, 13), 'node')
        # Obtaining the member 'id' of a type (line 1064)
        id_131317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1064, 13), node_131316, 'id')
        str_131318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1064, 24), 'str', 'True')
        # Applying the binary operator '==' (line 1064)
        result_eq_131319 = python_operator(stypy.reporting.localization.Localization(__file__, 1064, 13), '==', id_131317, str_131318)
        
        # Testing the type of an if condition (line 1064)
        if_condition_131320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1064, 13), result_eq_131319)
        # Assigning a type to the variable 'if_condition_131320' (line 1064)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1064, 13), 'if_condition_131320', if_condition_131320)
        # SSA begins for if statement (line 1064)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 1065)
        True_131321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1065, 19), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 1065)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1065, 12), 'stypy_return_type', True_131321)
        # SSA branch for the else part of an if statement (line 1064)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'node' (line 1066)
        node_131322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1066, 13), 'node')
        # Obtaining the member 'id' of a type (line 1066)
        id_131323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1066, 13), node_131322, 'id')
        str_131324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1066, 24), 'str', 'None')
        # Applying the binary operator '==' (line 1066)
        result_eq_131325 = python_operator(stypy.reporting.localization.Localization(__file__, 1066, 13), '==', id_131323, str_131324)
        
        # Testing the type of an if condition (line 1066)
        if_condition_131326 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 1066, 13), result_eq_131325)
        # Assigning a type to the variable 'if_condition_131326' (line 1066)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1066, 13), 'if_condition_131326', if_condition_131326)
        # SSA begins for if statement (line 1066)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 1067)
        None_131327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1067, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 1067)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1067, 12), 'stypy_return_type', None_131327)
        # SSA branch for the else part of an if statement (line 1066)
        module_type_store.open_ssa_branch('else')
        
        # Call to SyntaxError(...): (line 1069)
        # Processing the call arguments (line 1069)
        str_131329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1069, 30), 'str', 'Unknown name: %s')
        # Getting the type of 'node' (line 1069)
        node_131330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 51), 'node', False)
        # Obtaining the member 'id' of a type (line 1069)
        id_131331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1069, 51), node_131330, 'id')
        # Applying the binary operator '%' (line 1069)
        result_mod_131332 = python_operator(stypy.reporting.localization.Localization(__file__, 1069, 30), '%', str_131329, id_131331)
        
        # Processing the call keyword arguments (line 1069)
        kwargs_131333 = {}
        # Getting the type of 'SyntaxError' (line 1069)
        SyntaxError_131328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1069, 18), 'SyntaxError', False)
        # Calling SyntaxError(args, kwargs) (line 1069)
        SyntaxError_call_result_131334 = invoke(stypy.reporting.localization.Localization(__file__, 1069, 18), SyntaxError_131328, *[result_mod_131332], **kwargs_131333)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 1069, 12), SyntaxError_call_result_131334, 'raise parameter', BaseException)
        # SSA join for if statement (line 1066)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1064)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 1062)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'visitName(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visitName' in the type store
        # Getting the type of 'stypy_return_type' (line 1061)
        stypy_return_type_131335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1061, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131335)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visitName'
        return stypy_return_type_131335


    @norecursion
    def visitNameConstant(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visitNameConstant'
        module_type_store = module_type_store.open_function_context('visitNameConstant', 1071, 4, False)
        # Assigning a type to the variable 'self' (line 1072)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1072, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SafeEval.visitNameConstant.__dict__.__setitem__('stypy_localization', localization)
        SafeEval.visitNameConstant.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SafeEval.visitNameConstant.__dict__.__setitem__('stypy_type_store', module_type_store)
        SafeEval.visitNameConstant.__dict__.__setitem__('stypy_function_name', 'SafeEval.visitNameConstant')
        SafeEval.visitNameConstant.__dict__.__setitem__('stypy_param_names_list', ['node'])
        SafeEval.visitNameConstant.__dict__.__setitem__('stypy_varargs_param_name', None)
        SafeEval.visitNameConstant.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SafeEval.visitNameConstant.__dict__.__setitem__('stypy_call_defaults', defaults)
        SafeEval.visitNameConstant.__dict__.__setitem__('stypy_call_varargs', varargs)
        SafeEval.visitNameConstant.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SafeEval.visitNameConstant.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SafeEval.visitNameConstant', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visitNameConstant', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visitNameConstant(...)' code ##################

        # Getting the type of 'node' (line 1072)
        node_131336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1072, 15), 'node')
        # Obtaining the member 'value' of a type (line 1072)
        value_131337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1072, 15), node_131336, 'value')
        # Assigning a type to the variable 'stypy_return_type' (line 1072)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1072, 8), 'stypy_return_type', value_131337)
        
        # ################# End of 'visitNameConstant(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visitNameConstant' in the type store
        # Getting the type of 'stypy_return_type' (line 1071)
        stypy_return_type_131338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1071, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_131338)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visitNameConstant'
        return stypy_return_type_131338


# Assigning a type to the variable 'SafeEval' (line 1002)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1002, 0), 'SafeEval', SafeEval)

@norecursion
def safe_eval(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'safe_eval'
    module_type_store = module_type_store.open_function_context('safe_eval', 1075, 0, False)
    
    # Passed parameters checking function
    safe_eval.stypy_localization = localization
    safe_eval.stypy_type_of_self = None
    safe_eval.stypy_type_store = module_type_store
    safe_eval.stypy_function_name = 'safe_eval'
    safe_eval.stypy_param_names_list = ['source']
    safe_eval.stypy_varargs_param_name = None
    safe_eval.stypy_kwargs_param_name = None
    safe_eval.stypy_call_defaults = defaults
    safe_eval.stypy_call_varargs = varargs
    safe_eval.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'safe_eval', ['source'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'safe_eval', localization, ['source'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'safe_eval(...)' code ##################

    str_131339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1117, (-1)), 'str', '\n    Protected string evaluation.\n\n    Evaluate a string containing a Python literal expression without\n    allowing the execution of arbitrary non-literal code.\n\n    Parameters\n    ----------\n    source : str\n        The string to evaluate.\n\n    Returns\n    -------\n    obj : object\n       The result of evaluating `source`.\n\n    Raises\n    ------\n    SyntaxError\n        If the code has invalid Python syntax, or if it contains\n        non-literal code.\n\n    Examples\n    --------\n    >>> np.safe_eval(\'1\')\n    1\n    >>> np.safe_eval(\'[1, 2, 3]\')\n    [1, 2, 3]\n    >>> np.safe_eval(\'{"foo": ("bar", 10.0)}\')\n    {\'foo\': (\'bar\', 10.0)}\n\n    >>> np.safe_eval(\'import os\')\n    Traceback (most recent call last):\n      ...\n    SyntaxError: invalid syntax\n\n    >>> np.safe_eval(\'open("/home/user/.ssh/id_dsa").read()\')\n    Traceback (most recent call last):\n      ...\n    SyntaxError: Unsupported source construct: compiler.ast.CallFunc\n\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1119, 4))
    
    # 'import ast' statement (line 1119)
    import ast

    import_module(stypy.reporting.localization.Localization(__file__, 1119, 4), 'ast', ast, module_type_store)
    
    
    # Call to literal_eval(...): (line 1121)
    # Processing the call arguments (line 1121)
    # Getting the type of 'source' (line 1121)
    source_131342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 28), 'source', False)
    # Processing the call keyword arguments (line 1121)
    kwargs_131343 = {}
    # Getting the type of 'ast' (line 1121)
    ast_131340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1121, 11), 'ast', False)
    # Obtaining the member 'literal_eval' of a type (line 1121)
    literal_eval_131341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 1121, 11), ast_131340, 'literal_eval')
    # Calling literal_eval(args, kwargs) (line 1121)
    literal_eval_call_result_131344 = invoke(stypy.reporting.localization.Localization(__file__, 1121, 11), literal_eval_131341, *[source_131342], **kwargs_131343)
    
    # Assigning a type to the variable 'stypy_return_type' (line 1121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1121, 4), 'stypy_return_type', literal_eval_call_result_131344)
    
    # ################# End of 'safe_eval(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'safe_eval' in the type store
    # Getting the type of 'stypy_return_type' (line 1075)
    stypy_return_type_131345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 1075, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_131345)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'safe_eval'
    return stypy_return_type_131345

# Assigning a type to the variable 'safe_eval' (line 1075)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1075, 0), 'safe_eval', safe_eval)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
