
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Functions copypasted from newer versions of numpy.
2: 
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: import warnings
7: import sys
8: from warnings import WarningMessage
9: import re
10: from functools import wraps
11: import numpy as np
12: 
13: from scipy._lib._version import NumpyVersion
14: 
15: 
16: if NumpyVersion(np.__version__) > '1.7.0.dev':
17:     _assert_warns = np.testing.assert_warns
18: else:
19:     def _assert_warns(warning_class, func, *args, **kw):
20:         r'''
21:         Fail unless the given callable throws the specified warning.
22: 
23:         This definition is copypasted from numpy 1.9.0.dev.
24:         The version in earlier numpy returns None.
25: 
26:         Parameters
27:         ----------
28:         warning_class : class
29:             The class defining the warning that `func` is expected to throw.
30:         func : callable
31:             The callable to test.
32:         *args : Arguments
33:             Arguments passed to `func`.
34:         **kwargs : Kwargs
35:             Keyword arguments passed to `func`.
36: 
37:         Returns
38:         -------
39:         The value returned by `func`.
40: 
41:         '''
42:         with warnings.catch_warnings(record=True) as l:
43:             warnings.simplefilter('always')
44:             result = func(*args, **kw)
45:             if not len(l) > 0:
46:                 raise AssertionError("No warning raised when calling %s"
47:                         % func.__name__)
48:             if not l[0].category is warning_class:
49:                 raise AssertionError("First warning for %s is not a "
50:                         "%s( is %s)" % (func.__name__, warning_class, l[0]))
51:         return result
52: 
53: 
54: if NumpyVersion(np.__version__) >= '1.10.0':
55:     from numpy import broadcast_to
56: else:
57:     # Definition of `broadcast_to` from numpy 1.10.0.
58: 
59:     def _maybe_view_as_subclass(original_array, new_array):
60:         if type(original_array) is not type(new_array):
61:             # if input was an ndarray subclass and subclasses were OK,
62:             # then view the result as that subclass.
63:             new_array = new_array.view(type=type(original_array))
64:             # Since we have done something akin to a view from original_array, we
65:             # should let the subclass finalize (if it has it implemented, i.e., is
66:             # not None).
67:             if new_array.__array_finalize__:
68:                 new_array.__array_finalize__(original_array)
69:         return new_array
70: 
71:     def _broadcast_to(array, shape, subok, readonly):
72:         shape = tuple(shape) if np.iterable(shape) else (shape,)
73:         array = np.array(array, copy=False, subok=subok)
74:         if not shape and array.shape:
75:             raise ValueError('cannot broadcast a non-scalar to a scalar array')
76:         if any(size < 0 for size in shape):
77:             raise ValueError('all elements of broadcast shape must be non-'
78:                              'negative')
79:         broadcast = np.nditer(
80:             (array,), flags=['multi_index', 'refs_ok', 'zerosize_ok'],
81:             op_flags=['readonly'], itershape=shape, order='C').itviews[0]
82:         result = _maybe_view_as_subclass(array, broadcast)
83:         if not readonly and array.flags.writeable:
84:             result.flags.writeable = True
85:         return result
86: 
87:     def broadcast_to(array, shape, subok=False):
88:         return _broadcast_to(array, shape, subok=subok, readonly=True)
89: 
90: 
91: if NumpyVersion(np.__version__) >= '1.9.0':
92:     from numpy import unique
93: else:
94:     # the return_counts keyword was added in 1.9.0
95:     def unique(ar, return_index=False, return_inverse=False, return_counts=False):
96:         '''
97:         Find the unique elements of an array.
98: 
99:         Returns the sorted unique elements of an array. There are three optional
100:         outputs in addition to the unique elements: the indices of the input array
101:         that give the unique values, the indices of the unique array that
102:         reconstruct the input array, and the number of times each unique value
103:         comes up in the input array.
104: 
105:         Parameters
106:         ----------
107:         ar : array_like
108:             Input array. This will be flattened if it is not already 1-D.
109:         return_index : bool, optional
110:             If True, also return the indices of `ar` that result in the unique
111:             array.
112:         return_inverse : bool, optional
113:             If True, also return the indices of the unique array that can be used
114:             to reconstruct `ar`.
115:         return_counts : bool, optional
116:             If True, also return the number of times each unique value comes up
117:             in `ar`.
118: 
119:             .. versionadded:: 1.9.0
120: 
121:         Returns
122:         -------
123:         unique : ndarray
124:             The sorted unique values.
125:         unique_indices : ndarray, optional
126:             The indices of the first occurrences of the unique values in the
127:             (flattened) original array. Only provided if `return_index` is True.
128:         unique_inverse : ndarray, optional
129:             The indices to reconstruct the (flattened) original array from the
130:             unique array. Only provided if `return_inverse` is True.
131:         unique_counts : ndarray, optional
132:             The number of times each of the unique values comes up in the
133:             original array. Only provided if `return_counts` is True.
134: 
135:             .. versionadded:: 1.9.0
136: 
137:         Notes
138:         -----
139:         Taken over from numpy 1.12.0-dev (c8408bf9c).  Omitted examples,
140:         see numpy documentation for those.
141: 
142:         '''
143:         ar = np.asanyarray(ar).flatten()
144: 
145:         optional_indices = return_index or return_inverse
146:         optional_returns = optional_indices or return_counts
147: 
148:         if ar.size == 0:
149:             if not optional_returns:
150:                 ret = ar
151:             else:
152:                 ret = (ar,)
153:                 if return_index:
154:                     ret += (np.empty(0, np.bool),)
155:                 if return_inverse:
156:                     ret += (np.empty(0, np.bool),)
157:                 if return_counts:
158:                     ret += (np.empty(0, np.intp),)
159:             return ret
160: 
161:         if optional_indices:
162:             perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
163:             aux = ar[perm]
164:         else:
165:             ar.sort()
166:             aux = ar
167:         flag = np.concatenate(([True], aux[1:] != aux[:-1]))
168: 
169:         if not optional_returns:
170:             ret = aux[flag]
171:         else:
172:             ret = (aux[flag],)
173:             if return_index:
174:                 ret += (perm[flag],)
175:             if return_inverse:
176:                 iflag = np.cumsum(flag) - 1
177:                 inv_idx = np.empty(ar.shape, dtype=np.intp)
178:                 inv_idx[perm] = iflag
179:                 ret += (inv_idx,)
180:             if return_counts:
181:                 idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
182:                 ret += (np.diff(idx),)
183:         return ret
184: 
185: 
186: if NumpyVersion(np.__version__) > '1.12.0.dev':
187:     polyvalfromroots = np.polynomial.polynomial.polyvalfromroots
188: else:
189:     def polyvalfromroots(x, r, tensor=True):
190:         r'''
191:         Evaluate a polynomial specified by its roots at points x.
192: 
193:         This function is copypasted from numpy 1.12.0.dev.
194: 
195:         If `r` is of length `N`, this function returns the value
196: 
197:         .. math:: p(x) = \prod_{n=1}^{N} (x - r_n)
198: 
199:         The parameter `x` is converted to an array only if it is a tuple or a
200:         list, otherwise it is treated as a scalar. In either case, either `x`
201:         or its elements must support multiplication and addition both with
202:         themselves and with the elements of `r`.
203: 
204:         If `r` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
205:         `r` is multidimensional, then the shape of the result depends on the
206:         value of `tensor`. If `tensor is ``True`` the shape will be r.shape[1:]
207:         + x.shape; that is, each polynomial is evaluated at every value of `x`.
208:         If `tensor` is ``False``, the shape will be r.shape[1:]; that is, each
209:         polynomial is evaluated only for the corresponding broadcast value of
210:         `x`. Note that scalars have shape (,).
211: 
212:         Parameters
213:         ----------
214:         x : array_like, compatible object
215:             If `x` is a list or tuple, it is converted to an ndarray, otherwise
216:             it is left unchanged and treated as a scalar. In either case, `x`
217:             or its elements must support addition and multiplication with with
218:             themselves and with the elements of `r`.
219:         r : array_like
220:             Array of roots. If `r` is multidimensional the first index is the
221:             root index, while the remaining indices enumerate multiple
222:             polynomials. For instance, in the two dimensional case the roots of
223:             each polynomial may be thought of as stored in the columns of `r`.
224:         tensor : boolean, optional
225:             If True, the shape of the roots array is extended with ones on the
226:             right, one for each dimension of `x`. Scalars have dimension 0 for
227:             this action. The result is that every column of coefficients in `r`
228:             is evaluated for every element of `x`. If False, `x` is broadcast
229:             over the columns of `r` for the evaluation.  This keyword is useful
230:             when `r` is multidimensional. The default value is True.
231: 
232:         Returns
233:         -------
234:         values : ndarray, compatible object
235:             The shape of the returned array is described above.
236: 
237:         See Also
238:         --------
239:         polyroots, polyfromroots, polyval
240: 
241:         Examples
242:         --------
243:         >>> from numpy.polynomial.polynomial import polyvalfromroots
244:         >>> polyvalfromroots(1, [1,2,3])
245:         0.0
246:         >>> a = np.arange(4).reshape(2,2)
247:         >>> a
248:         array([[0, 1],
249:                [2, 3]])
250:         >>> polyvalfromroots(a, [-1, 0, 1])
251:         array([[ -0.,   0.],
252:                [  6.,  24.]])
253:         >>> r = np.arange(-2, 2).reshape(2,2) # multidimensional coefficients
254:         >>> r # each column of r defines one polynomial
255:         array([[-2, -1],
256:                [ 0,  1]])
257:         >>> b = [-2, 1]
258:         >>> polyvalfromroots(b, r, tensor=True)
259:         array([[-0.,  3.],
260:                [ 3., 0.]])
261:         >>> polyvalfromroots(b, r, tensor=False)
262:         array([-0.,  0.])
263:         '''
264:         r = np.array(r, ndmin=1, copy=0)
265:         if r.dtype.char in '?bBhHiIlLqQpP':
266:             r = r.astype(np.double)
267:         if isinstance(x, (tuple, list)):
268:             x = np.asarray(x)
269:         if isinstance(x, np.ndarray):
270:             if tensor:
271:                 r = r.reshape(r.shape + (1,)*x.ndim)
272:             elif x.ndim >= r.ndim:
273:                 raise ValueError("x.ndim must be < r.ndim when tensor == "
274:                                  "False")
275:         return np.prod(x - r, axis=0)
276: 
277: 
278: try:
279:     from numpy.testing import suppress_warnings
280: except ImportError:
281:     class suppress_warnings(object):
282:         '''
283:         Context manager and decorator doing much the same as
284:         ``warnings.catch_warnings``.
285: 
286:         However, it also provides a filter mechanism to work around
287:         http://bugs.python.org/issue4180.
288: 
289:         This bug causes Python before 3.4 to not reliably show warnings again
290:         after they have been ignored once (even within catch_warnings). It
291:         means that no "ignore" filter can be used easily, since following
292:         tests might need to see the warning. Additionally it allows easier
293:         specificity for testing warnings and can be nested.
294: 
295:         Parameters
296:         ----------
297:         forwarding_rule : str, optional
298:             One of "always", "once", "module", or "location". Analogous to
299:             the usual warnings module filter mode, it is useful to reduce
300:             noise mostly on the outmost level. Unsuppressed and unrecorded
301:             warnings will be forwarded based on this rule. Defaults to "always".
302:             "location" is equivalent to the warnings "default", match by exact
303:             location the warning warning originated from.
304: 
305:         Notes
306:         -----
307:         Filters added inside the context manager will be discarded again
308:         when leaving it. Upon entering all filters defined outside a
309:         context will be applied automatically.
310: 
311:         When a recording filter is added, matching warnings are stored in the
312:         ``log`` attribute as well as in the list returned by ``record``.
313: 
314:         If filters are added and the ``module`` keyword is given, the
315:         warning registry of this module will additionally be cleared when
316:         applying it, entering the context, or exiting it. This could cause
317:         warnings to appear a second time after leaving the context if they
318:         were configured to be printed once (default) and were already
319:         printed before the context was entered.
320: 
321:         Nesting this context manager will work as expected when the
322:         forwarding rule is "always" (default). Unfiltered and unrecorded
323:         warnings will be passed out and be matched by the outer level.
324:         On the outmost level they will be printed (or caught by another
325:         warnings context). The forwarding rule argument can modify this
326:         behaviour.
327: 
328:         Like ``catch_warnings`` this context manager is not threadsafe.
329: 
330:         Examples
331:         --------
332:         >>> with suppress_warnings() as sup:
333:         ...     sup.filter(DeprecationWarning, "Some text")
334:         ...     sup.filter(module=np.ma.core)
335:         ...     log = sup.record(FutureWarning, "Does this occur?")
336:         ...     command_giving_warnings()
337:         ...     # The FutureWarning was given once, the filtered warnings were
338:         ...     # ignored. All other warnings abide outside settings (may be
339:         ...     # printed/error)
340:         ...     assert_(len(log) == 1)
341:         ...     assert_(len(sup.log) == 1)  # also stored in log attribute
342: 
343:         Or as a decorator:
344: 
345:         >>> sup = suppress_warnings()
346:         >>> sup.filter(module=np.ma.core)  # module must match exact
347:         >>> @sup
348:         >>> def some_function():
349:         ...     # do something which causes a warning in np.ma.core
350:         ...     pass
351:         '''
352:         def __init__(self, forwarding_rule="always"):
353:             self._entered = False
354: 
355:             # Suppressions are either instance or defined inside one with block:
356:             self._suppressions = []
357: 
358:             if forwarding_rule not in {"always", "module", "once", "location"}:
359:                 raise ValueError("unsupported forwarding rule.")
360:             self._forwarding_rule = forwarding_rule
361: 
362:         def _clear_registries(self):
363:             if hasattr(warnings, "_filters_mutated"):
364:                 # clearing the registry should not be necessary on new pythons,
365:                 # instead the filters should be mutated.
366:                 warnings._filters_mutated()
367:                 return
368:             # Simply clear the registry, this should normally be harmless,
369:             # note that on new pythons it would be invalidated anyway.
370:             for module in self._tmp_modules:
371:                 if hasattr(module, "__warningregistry__"):
372:                     module.__warningregistry__.clear()
373: 
374:         def _filter(self, category=Warning, message="", module=None, record=False):
375:             if record:
376:                 record = []  # The log where to store warnings
377:             else:
378:                 record = None
379:             if self._entered:
380:                 if module is None:
381:                     warnings.filterwarnings(
382:                         "always", category=category, message=message)
383:                 else:
384:                     module_regex = module.__name__.replace('.', r'\.') + '$'
385:                     warnings.filterwarnings(
386:                         "always", category=category, message=message,
387:                         module=module_regex)
388:                     self._tmp_modules.add(module)
389:                     self._clear_registries()
390: 
391:                 self._tmp_suppressions.append(
392:                     (category, message, re.compile(message, re.I), module, record))
393:             else:
394:                 self._suppressions.append(
395:                     (category, message, re.compile(message, re.I), module, record))
396: 
397:             return record
398: 
399:         def filter(self, category=Warning, message="", module=None):
400:             '''
401:             Add a new suppressing filter or apply it if the state is entered.
402: 
403:             Parameters
404:             ----------
405:             category : class, optional
406:                 Warning class to filter
407:             message : string, optional
408:                 Regular expression matching the warning message.
409:             module : module, optional
410:                 Module to filter for. Note that the module (and its file)
411:                 must match exactly and cannot be a submodule. This may make
412:                 it unreliable for external modules.
413: 
414:             Notes
415:             -----
416:             When added within a context, filters are only added inside
417:             the context and will be forgotten when the context is exited.
418:             '''
419:             self._filter(category=category, message=message, module=module,
420:                          record=False)
421: 
422:         def record(self, category=Warning, message="", module=None):
423:             '''
424:             Append a new recording filter or apply it if the state is entered.
425: 
426:             All warnings matching will be appended to the ``log`` attribute.
427: 
428:             Parameters
429:             ----------
430:             category : class, optional
431:                 Warning class to filter
432:             message : string, optional
433:                 Regular expression matching the warning message.
434:             module : module, optional
435:                 Module to filter for. Note that the module (and its file)
436:                 must match exactly and cannot be a submodule. This may make
437:                 it unreliable for external modules.
438: 
439:             Returns
440:             -------
441:             log : list
442:                 A list which will be filled with all matched warnings.
443: 
444:             Notes
445:             -----
446:             When added within a context, filters are only added inside
447:             the context and will be forgotten when the context is exited.
448:             '''
449:             return self._filter(category=category, message=message, module=module,
450:                                 record=True)
451: 
452:         def __enter__(self):
453:             if self._entered:
454:                 raise RuntimeError("cannot enter suppress_warnings twice.")
455: 
456:             self._orig_show = warnings.showwarning
457:             self._filters = warnings.filters
458:             warnings.filters = self._filters[:]
459: 
460:             self._entered = True
461:             self._tmp_suppressions = []
462:             self._tmp_modules = set()
463:             self._forwarded = set()
464: 
465:             self.log = []  # reset global log (no need to keep same list)
466: 
467:             for cat, mess, _, mod, log in self._suppressions:
468:                 if log is not None:
469:                     del log[:]  # clear the log
470:                 if mod is None:
471:                     warnings.filterwarnings(
472:                         "always", category=cat, message=mess)
473:                 else:
474:                     module_regex = mod.__name__.replace('.', r'\.') + '$'
475:                     warnings.filterwarnings(
476:                         "always", category=cat, message=mess,
477:                         module=module_regex)
478:                     self._tmp_modules.add(mod)
479:             warnings.showwarning = self._showwarning
480:             self._clear_registries()
481: 
482:             return self
483: 
484:         def __exit__(self, *exc_info):
485:             warnings.showwarning = self._orig_show
486:             warnings.filters = self._filters
487:             self._clear_registries()
488:             self._entered = False
489:             del self._orig_show
490:             del self._filters
491: 
492:         def _showwarning(self, message, category, filename, lineno,
493:                          *args, **kwargs):
494:             use_warnmsg = kwargs.pop("use_warnmsg", None)
495:             for cat, _, pattern, mod, rec in (
496:                     self._suppressions + self._tmp_suppressions)[::-1]:
497:                 if (issubclass(category, cat) and
498:                         pattern.match(message.args[0]) is not None):
499:                     if mod is None:
500:                         # Message and category match, either recorded or ignored
501:                         if rec is not None:
502:                             msg = WarningMessage(message, category, filename,
503:                                                  lineno, **kwargs)
504:                             self.log.append(msg)
505:                             rec.append(msg)
506:                         return
507:                     # Use startswith, because warnings strips the c or o from
508:                     # .pyc/.pyo files.
509:                     elif mod.__file__.startswith(filename):
510:                         # The message and module (filename) match
511:                         if rec is not None:
512:                             msg = WarningMessage(message, category, filename,
513:                                                  lineno, **kwargs)
514:                             self.log.append(msg)
515:                             rec.append(msg)
516:                         return
517: 
518:             # There is no filter in place, so pass to the outside handler
519:             # unless we should only pass it once
520:             if self._forwarding_rule == "always":
521:                 if use_warnmsg is None:
522:                     self._orig_show(message, category, filename, lineno,
523:                                     *args, **kwargs)
524:                 else:
525:                     self._orig_showmsg(use_warnmsg)
526:                 return
527: 
528:             if self._forwarding_rule == "once":
529:                 signature = (message.args, category)
530:             elif self._forwarding_rule == "module":
531:                 signature = (message.args, category, filename)
532:             elif self._forwarding_rule == "location":
533:                 signature = (message.args, category, filename, lineno)
534: 
535:             if signature in self._forwarded:
536:                 return
537:             self._forwarded.add(signature)
538:             if use_warnmsg is None:
539:                 self._orig_show(message, category, filename, lineno, *args,
540:                                 **kwargs)
541:             else:
542:                 self._orig_showmsg(use_warnmsg)
543: 
544:         def __call__(self, func):
545:             '''
546:             Function decorator to apply certain suppressions to a whole
547:             function.
548:             '''
549:             @wraps(func)
550:             def new_func(*args, **kwargs):
551:                 with self:
552:                     return func(*args, **kwargs)
553: 
554:             return new_func
555: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_708636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'Functions copypasted from newer versions of numpy.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import warnings' statement (line 6)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import sys' statement (line 7)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from warnings import WarningMessage' statement (line 8)
try:
    from warnings import WarningMessage

except:
    WarningMessage = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'warnings', None, module_type_store, ['WarningMessage'], [WarningMessage])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import re' statement (line 9)
import re

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from functools import wraps' statement (line 10)
try:
    from functools import wraps

except:
    wraps = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'functools', None, module_type_store, ['wraps'], [wraps])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
import_708637 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_708637) is not StypyTypeError):

    if (import_708637 != 'pyd_module'):
        __import__(import_708637)
        sys_modules_708638 = sys.modules[import_708637]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_708638.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_708637)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from scipy._lib._version import NumpyVersion' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
import_708639 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._version')

if (type(import_708639) is not StypyTypeError):

    if (import_708639 != 'pyd_module'):
        __import__(import_708639)
        sys_modules_708640 = sys.modules[import_708639]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._version', sys_modules_708640.module_type_store, module_type_store, ['NumpyVersion'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_708640, sys_modules_708640.module_type_store, module_type_store)
    else:
        from scipy._lib._version import NumpyVersion

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._version', None, module_type_store, ['NumpyVersion'], [NumpyVersion])

else:
    # Assigning a type to the variable 'scipy._lib._version' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'scipy._lib._version', import_708639)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')




# Call to NumpyVersion(...): (line 16)
# Processing the call arguments (line 16)
# Getting the type of 'np' (line 16)
np_708642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'np', False)
# Obtaining the member '__version__' of a type (line 16)
version___708643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 16), np_708642, '__version__')
# Processing the call keyword arguments (line 16)
kwargs_708644 = {}
# Getting the type of 'NumpyVersion' (line 16)
NumpyVersion_708641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 3), 'NumpyVersion', False)
# Calling NumpyVersion(args, kwargs) (line 16)
NumpyVersion_call_result_708645 = invoke(stypy.reporting.localization.Localization(__file__, 16, 3), NumpyVersion_708641, *[version___708643], **kwargs_708644)

str_708646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 34), 'str', '1.7.0.dev')
# Applying the binary operator '>' (line 16)
result_gt_708647 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 3), '>', NumpyVersion_call_result_708645, str_708646)

# Testing the type of an if condition (line 16)
if_condition_708648 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 16, 0), result_gt_708647)
# Assigning a type to the variable 'if_condition_708648' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'if_condition_708648', if_condition_708648)
# SSA begins for if statement (line 16)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Attribute to a Name (line 17):
# Getting the type of 'np' (line 17)
np_708649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'np')
# Obtaining the member 'testing' of a type (line 17)
testing_708650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 20), np_708649, 'testing')
# Obtaining the member 'assert_warns' of a type (line 17)
assert_warns_708651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 20), testing_708650, 'assert_warns')
# Assigning a type to the variable '_assert_warns' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), '_assert_warns', assert_warns_708651)
# SSA branch for the else part of an if statement (line 16)
module_type_store.open_ssa_branch('else')

@norecursion
def _assert_warns(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_assert_warns'
    module_type_store = module_type_store.open_function_context('_assert_warns', 19, 4, False)
    
    # Passed parameters checking function
    _assert_warns.stypy_localization = localization
    _assert_warns.stypy_type_of_self = None
    _assert_warns.stypy_type_store = module_type_store
    _assert_warns.stypy_function_name = '_assert_warns'
    _assert_warns.stypy_param_names_list = ['warning_class', 'func']
    _assert_warns.stypy_varargs_param_name = 'args'
    _assert_warns.stypy_kwargs_param_name = 'kw'
    _assert_warns.stypy_call_defaults = defaults
    _assert_warns.stypy_call_varargs = varargs
    _assert_warns.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_assert_warns', ['warning_class', 'func'], 'args', 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_assert_warns', localization, ['warning_class', 'func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_assert_warns(...)' code ##################

    str_708652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, (-1)), 'str', '\n        Fail unless the given callable throws the specified warning.\n\n        This definition is copypasted from numpy 1.9.0.dev.\n        The version in earlier numpy returns None.\n\n        Parameters\n        ----------\n        warning_class : class\n            The class defining the warning that `func` is expected to throw.\n        func : callable\n            The callable to test.\n        *args : Arguments\n            Arguments passed to `func`.\n        **kwargs : Kwargs\n            Keyword arguments passed to `func`.\n\n        Returns\n        -------\n        The value returned by `func`.\n\n        ')
    
    # Call to catch_warnings(...): (line 42)
    # Processing the call keyword arguments (line 42)
    # Getting the type of 'True' (line 42)
    True_708655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 44), 'True', False)
    keyword_708656 = True_708655
    kwargs_708657 = {'record': keyword_708656}
    # Getting the type of 'warnings' (line 42)
    warnings_708653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'warnings', False)
    # Obtaining the member 'catch_warnings' of a type (line 42)
    catch_warnings_708654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), warnings_708653, 'catch_warnings')
    # Calling catch_warnings(args, kwargs) (line 42)
    catch_warnings_call_result_708658 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), catch_warnings_708654, *[], **kwargs_708657)
    
    with_708659 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 42, 13), catch_warnings_call_result_708658, 'with parameter', '__enter__', '__exit__')

    if with_708659:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 42)
        enter___708660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), catch_warnings_call_result_708658, '__enter__')
        with_enter_708661 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), enter___708660)
        # Assigning a type to the variable 'l' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'l', with_enter_708661)
        
        # Call to simplefilter(...): (line 43)
        # Processing the call arguments (line 43)
        str_708664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'str', 'always')
        # Processing the call keyword arguments (line 43)
        kwargs_708665 = {}
        # Getting the type of 'warnings' (line 43)
        warnings_708662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'warnings', False)
        # Obtaining the member 'simplefilter' of a type (line 43)
        simplefilter_708663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), warnings_708662, 'simplefilter')
        # Calling simplefilter(args, kwargs) (line 43)
        simplefilter_call_result_708666 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), simplefilter_708663, *[str_708664], **kwargs_708665)
        
        
        # Assigning a Call to a Name (line 44):
        
        # Call to func(...): (line 44)
        # Getting the type of 'args' (line 44)
        args_708668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'args', False)
        # Processing the call keyword arguments (line 44)
        # Getting the type of 'kw' (line 44)
        kw_708669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'kw', False)
        kwargs_708670 = {'kw_708669': kw_708669}
        # Getting the type of 'func' (line 44)
        func_708667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'func', False)
        # Calling func(args, kwargs) (line 44)
        func_call_result_708671 = invoke(stypy.reporting.localization.Localization(__file__, 44, 21), func_708667, *[args_708668], **kwargs_708670)
        
        # Assigning a type to the variable 'result' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'result', func_call_result_708671)
        
        
        
        
        # Call to len(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'l' (line 45)
        l_708673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'l', False)
        # Processing the call keyword arguments (line 45)
        kwargs_708674 = {}
        # Getting the type of 'len' (line 45)
        len_708672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'len', False)
        # Calling len(args, kwargs) (line 45)
        len_call_result_708675 = invoke(stypy.reporting.localization.Localization(__file__, 45, 19), len_708672, *[l_708673], **kwargs_708674)
        
        int_708676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 28), 'int')
        # Applying the binary operator '>' (line 45)
        result_gt_708677 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 19), '>', len_call_result_708675, int_708676)
        
        # Applying the 'not' unary operator (line 45)
        result_not__708678 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 15), 'not', result_gt_708677)
        
        # Testing the type of an if condition (line 45)
        if_condition_708679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 12), result_not__708678)
        # Assigning a type to the variable 'if_condition_708679' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'if_condition_708679', if_condition_708679)
        # SSA begins for if statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to AssertionError(...): (line 46)
        # Processing the call arguments (line 46)
        str_708681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 37), 'str', 'No warning raised when calling %s')
        # Getting the type of 'func' (line 47)
        func_708682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'func', False)
        # Obtaining the member '__name__' of a type (line 47)
        name___708683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 26), func_708682, '__name__')
        # Applying the binary operator '%' (line 46)
        result_mod_708684 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 37), '%', str_708681, name___708683)
        
        # Processing the call keyword arguments (line 46)
        kwargs_708685 = {}
        # Getting the type of 'AssertionError' (line 46)
        AssertionError_708680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'AssertionError', False)
        # Calling AssertionError(args, kwargs) (line 46)
        AssertionError_call_result_708686 = invoke(stypy.reporting.localization.Localization(__file__, 46, 22), AssertionError_708680, *[result_mod_708684], **kwargs_708685)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 46, 16), AssertionError_call_result_708686, 'raise parameter', BaseException)
        # SSA join for if statement (line 45)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        
        # Obtaining the type of the subscript
        int_708687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 21), 'int')
        # Getting the type of 'l' (line 48)
        l_708688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'l')
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___708689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), l_708688, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_708690 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), getitem___708689, int_708687)
        
        # Obtaining the member 'category' of a type (line 48)
        category_708691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), subscript_call_result_708690, 'category')
        # Getting the type of 'warning_class' (line 48)
        warning_class_708692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 36), 'warning_class')
        # Applying the binary operator 'is' (line 48)
        result_is__708693 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 19), 'is', category_708691, warning_class_708692)
        
        # Applying the 'not' unary operator (line 48)
        result_not__708694 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 15), 'not', result_is__708693)
        
        # Testing the type of an if condition (line 48)
        if_condition_708695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 12), result_not__708694)
        # Assigning a type to the variable 'if_condition_708695' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'if_condition_708695', if_condition_708695)
        # SSA begins for if statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to AssertionError(...): (line 49)
        # Processing the call arguments (line 49)
        str_708697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 37), 'str', 'First warning for %s is not a %s( is %s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 50)
        tuple_708698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 50)
        # Adding element type (line 50)
        # Getting the type of 'func' (line 50)
        func_708699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 40), 'func', False)
        # Obtaining the member '__name__' of a type (line 50)
        name___708700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 40), func_708699, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 40), tuple_708698, name___708700)
        # Adding element type (line 50)
        # Getting the type of 'warning_class' (line 50)
        warning_class_708701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 55), 'warning_class', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 40), tuple_708698, warning_class_708701)
        # Adding element type (line 50)
        
        # Obtaining the type of the subscript
        int_708702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 72), 'int')
        # Getting the type of 'l' (line 50)
        l_708703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 70), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___708704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 70), l_708703, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_708705 = invoke(stypy.reporting.localization.Localization(__file__, 50, 70), getitem___708704, int_708702)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 40), tuple_708698, subscript_call_result_708705)
        
        # Applying the binary operator '%' (line 49)
        result_mod_708706 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 37), '%', str_708697, tuple_708698)
        
        # Processing the call keyword arguments (line 49)
        kwargs_708707 = {}
        # Getting the type of 'AssertionError' (line 49)
        AssertionError_708696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'AssertionError', False)
        # Calling AssertionError(args, kwargs) (line 49)
        AssertionError_call_result_708708 = invoke(stypy.reporting.localization.Localization(__file__, 49, 22), AssertionError_708696, *[result_mod_708706], **kwargs_708707)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 49, 16), AssertionError_call_result_708708, 'raise parameter', BaseException)
        # SSA join for if statement (line 48)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 42)
        exit___708709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), catch_warnings_call_result_708658, '__exit__')
        with_exit_708710 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), exit___708709, None, None, None)

    # Getting the type of 'result' (line 51)
    result_708711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', result_708711)
    
    # ################# End of '_assert_warns(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_assert_warns' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_708712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708712)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_assert_warns'
    return stypy_return_type_708712

# Assigning a type to the variable '_assert_warns' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), '_assert_warns', _assert_warns)
# SSA join for if statement (line 16)
module_type_store = module_type_store.join_ssa_context()




# Call to NumpyVersion(...): (line 54)
# Processing the call arguments (line 54)
# Getting the type of 'np' (line 54)
np_708714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'np', False)
# Obtaining the member '__version__' of a type (line 54)
version___708715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), np_708714, '__version__')
# Processing the call keyword arguments (line 54)
kwargs_708716 = {}
# Getting the type of 'NumpyVersion' (line 54)
NumpyVersion_708713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 3), 'NumpyVersion', False)
# Calling NumpyVersion(args, kwargs) (line 54)
NumpyVersion_call_result_708717 = invoke(stypy.reporting.localization.Localization(__file__, 54, 3), NumpyVersion_708713, *[version___708715], **kwargs_708716)

str_708718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 35), 'str', '1.10.0')
# Applying the binary operator '>=' (line 54)
result_ge_708719 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 3), '>=', NumpyVersion_call_result_708717, str_708718)

# Testing the type of an if condition (line 54)
if_condition_708720 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 0), result_ge_708719)
# Assigning a type to the variable 'if_condition_708720' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'if_condition_708720', if_condition_708720)
# SSA begins for if statement (line 54)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 55, 4))

# 'from numpy import broadcast_to' statement (line 55)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
import_708721 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 55, 4), 'numpy')

if (type(import_708721) is not StypyTypeError):

    if (import_708721 != 'pyd_module'):
        __import__(import_708721)
        sys_modules_708722 = sys.modules[import_708721]
        import_from_module(stypy.reporting.localization.Localization(__file__, 55, 4), 'numpy', sys_modules_708722.module_type_store, module_type_store, ['broadcast_to'])
        nest_module(stypy.reporting.localization.Localization(__file__, 55, 4), __file__, sys_modules_708722, sys_modules_708722.module_type_store, module_type_store)
    else:
        from numpy import broadcast_to

        import_from_module(stypy.reporting.localization.Localization(__file__, 55, 4), 'numpy', None, module_type_store, ['broadcast_to'], [broadcast_to])

else:
    # Assigning a type to the variable 'numpy' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'numpy', import_708721)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')

# SSA branch for the else part of an if statement (line 54)
module_type_store.open_ssa_branch('else')

@norecursion
def _maybe_view_as_subclass(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_maybe_view_as_subclass'
    module_type_store = module_type_store.open_function_context('_maybe_view_as_subclass', 59, 4, False)
    
    # Passed parameters checking function
    _maybe_view_as_subclass.stypy_localization = localization
    _maybe_view_as_subclass.stypy_type_of_self = None
    _maybe_view_as_subclass.stypy_type_store = module_type_store
    _maybe_view_as_subclass.stypy_function_name = '_maybe_view_as_subclass'
    _maybe_view_as_subclass.stypy_param_names_list = ['original_array', 'new_array']
    _maybe_view_as_subclass.stypy_varargs_param_name = None
    _maybe_view_as_subclass.stypy_kwargs_param_name = None
    _maybe_view_as_subclass.stypy_call_defaults = defaults
    _maybe_view_as_subclass.stypy_call_varargs = varargs
    _maybe_view_as_subclass.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_maybe_view_as_subclass', ['original_array', 'new_array'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_maybe_view_as_subclass', localization, ['original_array', 'new_array'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_maybe_view_as_subclass(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 60)
    # Getting the type of 'original_array' (line 60)
    original_array_708723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'original_array')
    
    # Call to type(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'new_array' (line 60)
    new_array_708725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'new_array', False)
    # Processing the call keyword arguments (line 60)
    kwargs_708726 = {}
    # Getting the type of 'type' (line 60)
    type_708724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 39), 'type', False)
    # Calling type(args, kwargs) (line 60)
    type_call_result_708727 = invoke(stypy.reporting.localization.Localization(__file__, 60, 39), type_708724, *[new_array_708725], **kwargs_708726)
    
    
    (may_be_708728, more_types_in_union_708729) = may_not_be_type(original_array_708723, type_call_result_708727)

    if may_be_708728:

        if more_types_in_union_708729:
            # Runtime conditional SSA (line 60)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'original_array' (line 60)
        original_array_708730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'original_array')
        # Assigning a type to the variable 'original_array' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'original_array', remove_type_from_union(original_array_708730, type_call_result_708727))
        
        # Assigning a Call to a Name (line 63):
        
        # Call to view(...): (line 63)
        # Processing the call keyword arguments (line 63)
        
        # Call to type(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'original_array' (line 63)
        original_array_708734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 49), 'original_array', False)
        # Processing the call keyword arguments (line 63)
        kwargs_708735 = {}
        # Getting the type of 'type' (line 63)
        type_708733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 44), 'type', False)
        # Calling type(args, kwargs) (line 63)
        type_call_result_708736 = invoke(stypy.reporting.localization.Localization(__file__, 63, 44), type_708733, *[original_array_708734], **kwargs_708735)
        
        keyword_708737 = type_call_result_708736
        kwargs_708738 = {'type': keyword_708737}
        # Getting the type of 'new_array' (line 63)
        new_array_708731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'new_array', False)
        # Obtaining the member 'view' of a type (line 63)
        view_708732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 24), new_array_708731, 'view')
        # Calling view(args, kwargs) (line 63)
        view_call_result_708739 = invoke(stypy.reporting.localization.Localization(__file__, 63, 24), view_708732, *[], **kwargs_708738)
        
        # Assigning a type to the variable 'new_array' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'new_array', view_call_result_708739)
        
        # Getting the type of 'new_array' (line 67)
        new_array_708740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'new_array')
        # Obtaining the member '__array_finalize__' of a type (line 67)
        array_finalize___708741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 15), new_array_708740, '__array_finalize__')
        # Testing the type of an if condition (line 67)
        if_condition_708742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 12), array_finalize___708741)
        # Assigning a type to the variable 'if_condition_708742' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'if_condition_708742', if_condition_708742)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __array_finalize__(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'original_array' (line 68)
        original_array_708745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 45), 'original_array', False)
        # Processing the call keyword arguments (line 68)
        kwargs_708746 = {}
        # Getting the type of 'new_array' (line 68)
        new_array_708743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'new_array', False)
        # Obtaining the member '__array_finalize__' of a type (line 68)
        array_finalize___708744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), new_array_708743, '__array_finalize__')
        # Calling __array_finalize__(args, kwargs) (line 68)
        array_finalize___call_result_708747 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), array_finalize___708744, *[original_array_708745], **kwargs_708746)
        
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_708729:
            # SSA join for if statement (line 60)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'new_array' (line 69)
    new_array_708748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'new_array')
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', new_array_708748)
    
    # ################# End of '_maybe_view_as_subclass(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_maybe_view_as_subclass' in the type store
    # Getting the type of 'stypy_return_type' (line 59)
    stypy_return_type_708749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708749)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_maybe_view_as_subclass'
    return stypy_return_type_708749

# Assigning a type to the variable '_maybe_view_as_subclass' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), '_maybe_view_as_subclass', _maybe_view_as_subclass)

@norecursion
def _broadcast_to(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_broadcast_to'
    module_type_store = module_type_store.open_function_context('_broadcast_to', 71, 4, False)
    
    # Passed parameters checking function
    _broadcast_to.stypy_localization = localization
    _broadcast_to.stypy_type_of_self = None
    _broadcast_to.stypy_type_store = module_type_store
    _broadcast_to.stypy_function_name = '_broadcast_to'
    _broadcast_to.stypy_param_names_list = ['array', 'shape', 'subok', 'readonly']
    _broadcast_to.stypy_varargs_param_name = None
    _broadcast_to.stypy_kwargs_param_name = None
    _broadcast_to.stypy_call_defaults = defaults
    _broadcast_to.stypy_call_varargs = varargs
    _broadcast_to.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_broadcast_to', ['array', 'shape', 'subok', 'readonly'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_broadcast_to', localization, ['array', 'shape', 'subok', 'readonly'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_broadcast_to(...)' code ##################

    
    # Assigning a IfExp to a Name (line 72):
    
    
    # Call to iterable(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'shape' (line 72)
    shape_708752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 44), 'shape', False)
    # Processing the call keyword arguments (line 72)
    kwargs_708753 = {}
    # Getting the type of 'np' (line 72)
    np_708750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'np', False)
    # Obtaining the member 'iterable' of a type (line 72)
    iterable_708751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 32), np_708750, 'iterable')
    # Calling iterable(args, kwargs) (line 72)
    iterable_call_result_708754 = invoke(stypy.reporting.localization.Localization(__file__, 72, 32), iterable_708751, *[shape_708752], **kwargs_708753)
    
    # Testing the type of an if expression (line 72)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 16), iterable_call_result_708754)
    # SSA begins for if expression (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to tuple(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'shape' (line 72)
    shape_708756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'shape', False)
    # Processing the call keyword arguments (line 72)
    kwargs_708757 = {}
    # Getting the type of 'tuple' (line 72)
    tuple_708755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 72)
    tuple_call_result_708758 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), tuple_708755, *[shape_708756], **kwargs_708757)
    
    # SSA branch for the else part of an if expression (line 72)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 72)
    tuple_708759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 72)
    # Adding element type (line 72)
    # Getting the type of 'shape' (line 72)
    shape_708760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 57), 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 57), tuple_708759, shape_708760)
    
    # SSA join for if expression (line 72)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_708761 = union_type.UnionType.add(tuple_call_result_708758, tuple_708759)
    
    # Assigning a type to the variable 'shape' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'shape', if_exp_708761)
    
    # Assigning a Call to a Name (line 73):
    
    # Call to array(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'array' (line 73)
    array_708764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 25), 'array', False)
    # Processing the call keyword arguments (line 73)
    # Getting the type of 'False' (line 73)
    False_708765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 37), 'False', False)
    keyword_708766 = False_708765
    # Getting the type of 'subok' (line 73)
    subok_708767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 50), 'subok', False)
    keyword_708768 = subok_708767
    kwargs_708769 = {'subok': keyword_708768, 'copy': keyword_708766}
    # Getting the type of 'np' (line 73)
    np_708762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'np', False)
    # Obtaining the member 'array' of a type (line 73)
    array_708763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), np_708762, 'array')
    # Calling array(args, kwargs) (line 73)
    array_call_result_708770 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), array_708763, *[array_708764], **kwargs_708769)
    
    # Assigning a type to the variable 'array' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'array', array_call_result_708770)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'shape' (line 74)
    shape_708771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'shape')
    # Applying the 'not' unary operator (line 74)
    result_not__708772 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), 'not', shape_708771)
    
    # Getting the type of 'array' (line 74)
    array_708773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'array')
    # Obtaining the member 'shape' of a type (line 74)
    shape_708774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 25), array_708773, 'shape')
    # Applying the binary operator 'and' (line 74)
    result_and_keyword_708775 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 11), 'and', result_not__708772, shape_708774)
    
    # Testing the type of an if condition (line 74)
    if_condition_708776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 8), result_and_keyword_708775)
    # Assigning a type to the variable 'if_condition_708776' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'if_condition_708776', if_condition_708776)
    # SSA begins for if statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 75)
    # Processing the call arguments (line 75)
    str_708778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 29), 'str', 'cannot broadcast a non-scalar to a scalar array')
    # Processing the call keyword arguments (line 75)
    kwargs_708779 = {}
    # Getting the type of 'ValueError' (line 75)
    ValueError_708777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 75)
    ValueError_call_result_708780 = invoke(stypy.reporting.localization.Localization(__file__, 75, 18), ValueError_708777, *[str_708778], **kwargs_708779)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 75, 12), ValueError_call_result_708780, 'raise parameter', BaseException)
    # SSA join for if statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 76)
    # Processing the call arguments (line 76)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 76, 15, True)
    # Calculating comprehension expression
    # Getting the type of 'shape' (line 76)
    shape_708785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 36), 'shape', False)
    comprehension_708786 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 15), shape_708785)
    # Assigning a type to the variable 'size' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'size', comprehension_708786)
    
    # Getting the type of 'size' (line 76)
    size_708782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'size', False)
    int_708783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 22), 'int')
    # Applying the binary operator '<' (line 76)
    result_lt_708784 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 15), '<', size_708782, int_708783)
    
    list_708787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 15), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 15), list_708787, result_lt_708784)
    # Processing the call keyword arguments (line 76)
    kwargs_708788 = {}
    # Getting the type of 'any' (line 76)
    any_708781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'any', False)
    # Calling any(args, kwargs) (line 76)
    any_call_result_708789 = invoke(stypy.reporting.localization.Localization(__file__, 76, 11), any_708781, *[list_708787], **kwargs_708788)
    
    # Testing the type of an if condition (line 76)
    if_condition_708790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), any_call_result_708789)
    # Assigning a type to the variable 'if_condition_708790' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_708790', if_condition_708790)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 77)
    # Processing the call arguments (line 77)
    str_708792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 29), 'str', 'all elements of broadcast shape must be non-negative')
    # Processing the call keyword arguments (line 77)
    kwargs_708793 = {}
    # Getting the type of 'ValueError' (line 77)
    ValueError_708791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 77)
    ValueError_call_result_708794 = invoke(stypy.reporting.localization.Localization(__file__, 77, 18), ValueError_708791, *[str_708792], **kwargs_708793)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 77, 12), ValueError_call_result_708794, 'raise parameter', BaseException)
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 79):
    
    # Obtaining the type of the subscript
    int_708795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 71), 'int')
    
    # Call to nditer(...): (line 79)
    # Processing the call arguments (line 79)
    
    # Obtaining an instance of the builtin type 'tuple' (line 80)
    tuple_708798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 80)
    # Adding element type (line 80)
    # Getting the type of 'array' (line 80)
    array_708799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'array', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 13), tuple_708798, array_708799)
    
    # Processing the call keyword arguments (line 79)
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_708800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    # Adding element type (line 80)
    str_708801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 29), 'str', 'multi_index')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 28), list_708800, str_708801)
    # Adding element type (line 80)
    str_708802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 44), 'str', 'refs_ok')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 28), list_708800, str_708802)
    # Adding element type (line 80)
    str_708803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 55), 'str', 'zerosize_ok')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 28), list_708800, str_708803)
    
    keyword_708804 = list_708800
    
    # Obtaining an instance of the builtin type 'list' (line 81)
    list_708805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 81)
    # Adding element type (line 81)
    str_708806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 22), 'str', 'readonly')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 21), list_708805, str_708806)
    
    keyword_708807 = list_708805
    # Getting the type of 'shape' (line 81)
    shape_708808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 45), 'shape', False)
    keyword_708809 = shape_708808
    str_708810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 58), 'str', 'C')
    keyword_708811 = str_708810
    kwargs_708812 = {'itershape': keyword_708809, 'op_flags': keyword_708807, 'flags': keyword_708804, 'order': keyword_708811}
    # Getting the type of 'np' (line 79)
    np_708796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'np', False)
    # Obtaining the member 'nditer' of a type (line 79)
    nditer_708797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 20), np_708796, 'nditer')
    # Calling nditer(args, kwargs) (line 79)
    nditer_call_result_708813 = invoke(stypy.reporting.localization.Localization(__file__, 79, 20), nditer_708797, *[tuple_708798], **kwargs_708812)
    
    # Obtaining the member 'itviews' of a type (line 79)
    itviews_708814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 20), nditer_call_result_708813, 'itviews')
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___708815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 20), itviews_708814, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_708816 = invoke(stypy.reporting.localization.Localization(__file__, 79, 20), getitem___708815, int_708795)
    
    # Assigning a type to the variable 'broadcast' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'broadcast', subscript_call_result_708816)
    
    # Assigning a Call to a Name (line 82):
    
    # Call to _maybe_view_as_subclass(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'array' (line 82)
    array_708818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 41), 'array', False)
    # Getting the type of 'broadcast' (line 82)
    broadcast_708819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 48), 'broadcast', False)
    # Processing the call keyword arguments (line 82)
    kwargs_708820 = {}
    # Getting the type of '_maybe_view_as_subclass' (line 82)
    _maybe_view_as_subclass_708817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 17), '_maybe_view_as_subclass', False)
    # Calling _maybe_view_as_subclass(args, kwargs) (line 82)
    _maybe_view_as_subclass_call_result_708821 = invoke(stypy.reporting.localization.Localization(__file__, 82, 17), _maybe_view_as_subclass_708817, *[array_708818, broadcast_708819], **kwargs_708820)
    
    # Assigning a type to the variable 'result' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'result', _maybe_view_as_subclass_call_result_708821)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'readonly' (line 83)
    readonly_708822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'readonly')
    # Applying the 'not' unary operator (line 83)
    result_not__708823 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), 'not', readonly_708822)
    
    # Getting the type of 'array' (line 83)
    array_708824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 28), 'array')
    # Obtaining the member 'flags' of a type (line 83)
    flags_708825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 28), array_708824, 'flags')
    # Obtaining the member 'writeable' of a type (line 83)
    writeable_708826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 28), flags_708825, 'writeable')
    # Applying the binary operator 'and' (line 83)
    result_and_keyword_708827 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), 'and', result_not__708823, writeable_708826)
    
    # Testing the type of an if condition (line 83)
    if_condition_708828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 8), result_and_keyword_708827)
    # Assigning a type to the variable 'if_condition_708828' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'if_condition_708828', if_condition_708828)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Attribute (line 84):
    # Getting the type of 'True' (line 84)
    True_708829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 37), 'True')
    # Getting the type of 'result' (line 84)
    result_708830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'result')
    # Obtaining the member 'flags' of a type (line 84)
    flags_708831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), result_708830, 'flags')
    # Setting the type of the member 'writeable' of a type (line 84)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), flags_708831, 'writeable', True_708829)
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 85)
    result_708832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', result_708832)
    
    # ################# End of '_broadcast_to(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_broadcast_to' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_708833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708833)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_broadcast_to'
    return stypy_return_type_708833

# Assigning a type to the variable '_broadcast_to' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), '_broadcast_to', _broadcast_to)

@norecursion
def broadcast_to(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 87)
    False_708834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 41), 'False')
    defaults = [False_708834]
    # Create a new context for function 'broadcast_to'
    module_type_store = module_type_store.open_function_context('broadcast_to', 87, 4, False)
    
    # Passed parameters checking function
    broadcast_to.stypy_localization = localization
    broadcast_to.stypy_type_of_self = None
    broadcast_to.stypy_type_store = module_type_store
    broadcast_to.stypy_function_name = 'broadcast_to'
    broadcast_to.stypy_param_names_list = ['array', 'shape', 'subok']
    broadcast_to.stypy_varargs_param_name = None
    broadcast_to.stypy_kwargs_param_name = None
    broadcast_to.stypy_call_defaults = defaults
    broadcast_to.stypy_call_varargs = varargs
    broadcast_to.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'broadcast_to', ['array', 'shape', 'subok'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'broadcast_to', localization, ['array', 'shape', 'subok'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'broadcast_to(...)' code ##################

    
    # Call to _broadcast_to(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'array' (line 88)
    array_708836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'array', False)
    # Getting the type of 'shape' (line 88)
    shape_708837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 36), 'shape', False)
    # Processing the call keyword arguments (line 88)
    # Getting the type of 'subok' (line 88)
    subok_708838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 49), 'subok', False)
    keyword_708839 = subok_708838
    # Getting the type of 'True' (line 88)
    True_708840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 65), 'True', False)
    keyword_708841 = True_708840
    kwargs_708842 = {'subok': keyword_708839, 'readonly': keyword_708841}
    # Getting the type of '_broadcast_to' (line 88)
    _broadcast_to_708835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), '_broadcast_to', False)
    # Calling _broadcast_to(args, kwargs) (line 88)
    _broadcast_to_call_result_708843 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), _broadcast_to_708835, *[array_708836, shape_708837], **kwargs_708842)
    
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', _broadcast_to_call_result_708843)
    
    # ################# End of 'broadcast_to(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'broadcast_to' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_708844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708844)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'broadcast_to'
    return stypy_return_type_708844

# Assigning a type to the variable 'broadcast_to' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'broadcast_to', broadcast_to)
# SSA join for if statement (line 54)
module_type_store = module_type_store.join_ssa_context()




# Call to NumpyVersion(...): (line 91)
# Processing the call arguments (line 91)
# Getting the type of 'np' (line 91)
np_708846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'np', False)
# Obtaining the member '__version__' of a type (line 91)
version___708847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 16), np_708846, '__version__')
# Processing the call keyword arguments (line 91)
kwargs_708848 = {}
# Getting the type of 'NumpyVersion' (line 91)
NumpyVersion_708845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 3), 'NumpyVersion', False)
# Calling NumpyVersion(args, kwargs) (line 91)
NumpyVersion_call_result_708849 = invoke(stypy.reporting.localization.Localization(__file__, 91, 3), NumpyVersion_708845, *[version___708847], **kwargs_708848)

str_708850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 35), 'str', '1.9.0')
# Applying the binary operator '>=' (line 91)
result_ge_708851 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 3), '>=', NumpyVersion_call_result_708849, str_708850)

# Testing the type of an if condition (line 91)
if_condition_708852 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 0), result_ge_708851)
# Assigning a type to the variable 'if_condition_708852' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'if_condition_708852', if_condition_708852)
# SSA begins for if statement (line 91)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 92, 4))

# 'from numpy import unique' statement (line 92)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
import_708853 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 92, 4), 'numpy')

if (type(import_708853) is not StypyTypeError):

    if (import_708853 != 'pyd_module'):
        __import__(import_708853)
        sys_modules_708854 = sys.modules[import_708853]
        import_from_module(stypy.reporting.localization.Localization(__file__, 92, 4), 'numpy', sys_modules_708854.module_type_store, module_type_store, ['unique'])
        nest_module(stypy.reporting.localization.Localization(__file__, 92, 4), __file__, sys_modules_708854, sys_modules_708854.module_type_store, module_type_store)
    else:
        from numpy import unique

        import_from_module(stypy.reporting.localization.Localization(__file__, 92, 4), 'numpy', None, module_type_store, ['unique'], [unique])

else:
    # Assigning a type to the variable 'numpy' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'numpy', import_708853)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')

# SSA branch for the else part of an if statement (line 91)
module_type_store.open_ssa_branch('else')

@norecursion
def unique(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 95)
    False_708855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 32), 'False')
    # Getting the type of 'False' (line 95)
    False_708856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 54), 'False')
    # Getting the type of 'False' (line 95)
    False_708857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 75), 'False')
    defaults = [False_708855, False_708856, False_708857]
    # Create a new context for function 'unique'
    module_type_store = module_type_store.open_function_context('unique', 95, 4, False)
    
    # Passed parameters checking function
    unique.stypy_localization = localization
    unique.stypy_type_of_self = None
    unique.stypy_type_store = module_type_store
    unique.stypy_function_name = 'unique'
    unique.stypy_param_names_list = ['ar', 'return_index', 'return_inverse', 'return_counts']
    unique.stypy_varargs_param_name = None
    unique.stypy_kwargs_param_name = None
    unique.stypy_call_defaults = defaults
    unique.stypy_call_varargs = varargs
    unique.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unique', ['ar', 'return_index', 'return_inverse', 'return_counts'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unique', localization, ['ar', 'return_index', 'return_inverse', 'return_counts'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unique(...)' code ##################

    str_708858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, (-1)), 'str', '\n        Find the unique elements of an array.\n\n        Returns the sorted unique elements of an array. There are three optional\n        outputs in addition to the unique elements: the indices of the input array\n        that give the unique values, the indices of the unique array that\n        reconstruct the input array, and the number of times each unique value\n        comes up in the input array.\n\n        Parameters\n        ----------\n        ar : array_like\n            Input array. This will be flattened if it is not already 1-D.\n        return_index : bool, optional\n            If True, also return the indices of `ar` that result in the unique\n            array.\n        return_inverse : bool, optional\n            If True, also return the indices of the unique array that can be used\n            to reconstruct `ar`.\n        return_counts : bool, optional\n            If True, also return the number of times each unique value comes up\n            in `ar`.\n\n            .. versionadded:: 1.9.0\n\n        Returns\n        -------\n        unique : ndarray\n            The sorted unique values.\n        unique_indices : ndarray, optional\n            The indices of the first occurrences of the unique values in the\n            (flattened) original array. Only provided if `return_index` is True.\n        unique_inverse : ndarray, optional\n            The indices to reconstruct the (flattened) original array from the\n            unique array. Only provided if `return_inverse` is True.\n        unique_counts : ndarray, optional\n            The number of times each of the unique values comes up in the\n            original array. Only provided if `return_counts` is True.\n\n            .. versionadded:: 1.9.0\n\n        Notes\n        -----\n        Taken over from numpy 1.12.0-dev (c8408bf9c).  Omitted examples,\n        see numpy documentation for those.\n\n        ')
    
    # Assigning a Call to a Name (line 143):
    
    # Call to flatten(...): (line 143)
    # Processing the call keyword arguments (line 143)
    kwargs_708865 = {}
    
    # Call to asanyarray(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'ar' (line 143)
    ar_708861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 27), 'ar', False)
    # Processing the call keyword arguments (line 143)
    kwargs_708862 = {}
    # Getting the type of 'np' (line 143)
    np_708859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 13), 'np', False)
    # Obtaining the member 'asanyarray' of a type (line 143)
    asanyarray_708860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 13), np_708859, 'asanyarray')
    # Calling asanyarray(args, kwargs) (line 143)
    asanyarray_call_result_708863 = invoke(stypy.reporting.localization.Localization(__file__, 143, 13), asanyarray_708860, *[ar_708861], **kwargs_708862)
    
    # Obtaining the member 'flatten' of a type (line 143)
    flatten_708864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 13), asanyarray_call_result_708863, 'flatten')
    # Calling flatten(args, kwargs) (line 143)
    flatten_call_result_708866 = invoke(stypy.reporting.localization.Localization(__file__, 143, 13), flatten_708864, *[], **kwargs_708865)
    
    # Assigning a type to the variable 'ar' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'ar', flatten_call_result_708866)
    
    # Assigning a BoolOp to a Name (line 145):
    
    # Evaluating a boolean operation
    # Getting the type of 'return_index' (line 145)
    return_index_708867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 27), 'return_index')
    # Getting the type of 'return_inverse' (line 145)
    return_inverse_708868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 43), 'return_inverse')
    # Applying the binary operator 'or' (line 145)
    result_or_keyword_708869 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 27), 'or', return_index_708867, return_inverse_708868)
    
    # Assigning a type to the variable 'optional_indices' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'optional_indices', result_or_keyword_708869)
    
    # Assigning a BoolOp to a Name (line 146):
    
    # Evaluating a boolean operation
    # Getting the type of 'optional_indices' (line 146)
    optional_indices_708870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'optional_indices')
    # Getting the type of 'return_counts' (line 146)
    return_counts_708871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 47), 'return_counts')
    # Applying the binary operator 'or' (line 146)
    result_or_keyword_708872 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 27), 'or', optional_indices_708870, return_counts_708871)
    
    # Assigning a type to the variable 'optional_returns' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'optional_returns', result_or_keyword_708872)
    
    
    # Getting the type of 'ar' (line 148)
    ar_708873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'ar')
    # Obtaining the member 'size' of a type (line 148)
    size_708874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 11), ar_708873, 'size')
    int_708875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 22), 'int')
    # Applying the binary operator '==' (line 148)
    result_eq_708876 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 11), '==', size_708874, int_708875)
    
    # Testing the type of an if condition (line 148)
    if_condition_708877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 8), result_eq_708876)
    # Assigning a type to the variable 'if_condition_708877' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'if_condition_708877', if_condition_708877)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'optional_returns' (line 149)
    optional_returns_708878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 19), 'optional_returns')
    # Applying the 'not' unary operator (line 149)
    result_not__708879 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 15), 'not', optional_returns_708878)
    
    # Testing the type of an if condition (line 149)
    if_condition_708880 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 12), result_not__708879)
    # Assigning a type to the variable 'if_condition_708880' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'if_condition_708880', if_condition_708880)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'ar' (line 150)
    ar_708881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 22), 'ar')
    # Assigning a type to the variable 'ret' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'ret', ar_708881)
    # SSA branch for the else part of an if statement (line 149)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Name (line 152):
    
    # Obtaining an instance of the builtin type 'tuple' (line 152)
    tuple_708882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 152)
    # Adding element type (line 152)
    # Getting the type of 'ar' (line 152)
    ar_708883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 'ar')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 23), tuple_708882, ar_708883)
    
    # Assigning a type to the variable 'ret' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'ret', tuple_708882)
    
    # Getting the type of 'return_index' (line 153)
    return_index_708884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'return_index')
    # Testing the type of an if condition (line 153)
    if_condition_708885 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 16), return_index_708884)
    # Assigning a type to the variable 'if_condition_708885' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'if_condition_708885', if_condition_708885)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ret' (line 154)
    ret_708886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'ret')
    
    # Obtaining an instance of the builtin type 'tuple' (line 154)
    tuple_708887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 154)
    # Adding element type (line 154)
    
    # Call to empty(...): (line 154)
    # Processing the call arguments (line 154)
    int_708890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 37), 'int')
    # Getting the type of 'np' (line 154)
    np_708891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 40), 'np', False)
    # Obtaining the member 'bool' of a type (line 154)
    bool_708892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 40), np_708891, 'bool')
    # Processing the call keyword arguments (line 154)
    kwargs_708893 = {}
    # Getting the type of 'np' (line 154)
    np_708888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 28), 'np', False)
    # Obtaining the member 'empty' of a type (line 154)
    empty_708889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 28), np_708888, 'empty')
    # Calling empty(args, kwargs) (line 154)
    empty_call_result_708894 = invoke(stypy.reporting.localization.Localization(__file__, 154, 28), empty_708889, *[int_708890, bool_708892], **kwargs_708893)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 28), tuple_708887, empty_call_result_708894)
    
    # Applying the binary operator '+=' (line 154)
    result_iadd_708895 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 20), '+=', ret_708886, tuple_708887)
    # Assigning a type to the variable 'ret' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'ret', result_iadd_708895)
    
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_inverse' (line 155)
    return_inverse_708896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'return_inverse')
    # Testing the type of an if condition (line 155)
    if_condition_708897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 16), return_inverse_708896)
    # Assigning a type to the variable 'if_condition_708897' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'if_condition_708897', if_condition_708897)
    # SSA begins for if statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ret' (line 156)
    ret_708898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'ret')
    
    # Obtaining an instance of the builtin type 'tuple' (line 156)
    tuple_708899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 156)
    # Adding element type (line 156)
    
    # Call to empty(...): (line 156)
    # Processing the call arguments (line 156)
    int_708902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 37), 'int')
    # Getting the type of 'np' (line 156)
    np_708903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 40), 'np', False)
    # Obtaining the member 'bool' of a type (line 156)
    bool_708904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 40), np_708903, 'bool')
    # Processing the call keyword arguments (line 156)
    kwargs_708905 = {}
    # Getting the type of 'np' (line 156)
    np_708900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 28), 'np', False)
    # Obtaining the member 'empty' of a type (line 156)
    empty_708901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 28), np_708900, 'empty')
    # Calling empty(args, kwargs) (line 156)
    empty_call_result_708906 = invoke(stypy.reporting.localization.Localization(__file__, 156, 28), empty_708901, *[int_708902, bool_708904], **kwargs_708905)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 28), tuple_708899, empty_call_result_708906)
    
    # Applying the binary operator '+=' (line 156)
    result_iadd_708907 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 20), '+=', ret_708898, tuple_708899)
    # Assigning a type to the variable 'ret' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 20), 'ret', result_iadd_708907)
    
    # SSA join for if statement (line 155)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_counts' (line 157)
    return_counts_708908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 19), 'return_counts')
    # Testing the type of an if condition (line 157)
    if_condition_708909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 16), return_counts_708908)
    # Assigning a type to the variable 'if_condition_708909' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'if_condition_708909', if_condition_708909)
    # SSA begins for if statement (line 157)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ret' (line 158)
    ret_708910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'ret')
    
    # Obtaining an instance of the builtin type 'tuple' (line 158)
    tuple_708911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 158)
    # Adding element type (line 158)
    
    # Call to empty(...): (line 158)
    # Processing the call arguments (line 158)
    int_708914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 37), 'int')
    # Getting the type of 'np' (line 158)
    np_708915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 40), 'np', False)
    # Obtaining the member 'intp' of a type (line 158)
    intp_708916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 40), np_708915, 'intp')
    # Processing the call keyword arguments (line 158)
    kwargs_708917 = {}
    # Getting the type of 'np' (line 158)
    np_708912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'np', False)
    # Obtaining the member 'empty' of a type (line 158)
    empty_708913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 28), np_708912, 'empty')
    # Calling empty(args, kwargs) (line 158)
    empty_call_result_708918 = invoke(stypy.reporting.localization.Localization(__file__, 158, 28), empty_708913, *[int_708914, intp_708916], **kwargs_708917)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 28), tuple_708911, empty_call_result_708918)
    
    # Applying the binary operator '+=' (line 158)
    result_iadd_708919 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 20), '+=', ret_708910, tuple_708911)
    # Assigning a type to the variable 'ret' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 20), 'ret', result_iadd_708919)
    
    # SSA join for if statement (line 157)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 159)
    ret_708920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'stypy_return_type', ret_708920)
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'optional_indices' (line 161)
    optional_indices_708921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 11), 'optional_indices')
    # Testing the type of an if condition (line 161)
    if_condition_708922 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 8), optional_indices_708921)
    # Assigning a type to the variable 'if_condition_708922' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'if_condition_708922', if_condition_708922)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 162):
    
    # Call to argsort(...): (line 162)
    # Processing the call keyword arguments (line 162)
    
    # Getting the type of 'return_index' (line 162)
    return_index_708925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 50), 'return_index', False)
    # Testing the type of an if expression (line 162)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 35), return_index_708925)
    # SSA begins for if expression (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    str_708926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 35), 'str', 'mergesort')
    # SSA branch for the else part of an if expression (line 162)
    module_type_store.open_ssa_branch('if expression else')
    str_708927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 68), 'str', 'quicksort')
    # SSA join for if expression (line 162)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_708928 = union_type.UnionType.add(str_708926, str_708927)
    
    keyword_708929 = if_exp_708928
    kwargs_708930 = {'kind': keyword_708929}
    # Getting the type of 'ar' (line 162)
    ar_708923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'ar', False)
    # Obtaining the member 'argsort' of a type (line 162)
    argsort_708924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 19), ar_708923, 'argsort')
    # Calling argsort(args, kwargs) (line 162)
    argsort_call_result_708931 = invoke(stypy.reporting.localization.Localization(__file__, 162, 19), argsort_708924, *[], **kwargs_708930)
    
    # Assigning a type to the variable 'perm' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'perm', argsort_call_result_708931)
    
    # Assigning a Subscript to a Name (line 163):
    
    # Obtaining the type of the subscript
    # Getting the type of 'perm' (line 163)
    perm_708932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 21), 'perm')
    # Getting the type of 'ar' (line 163)
    ar_708933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 18), 'ar')
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___708934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 18), ar_708933, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_708935 = invoke(stypy.reporting.localization.Localization(__file__, 163, 18), getitem___708934, perm_708932)
    
    # Assigning a type to the variable 'aux' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'aux', subscript_call_result_708935)
    # SSA branch for the else part of an if statement (line 161)
    module_type_store.open_ssa_branch('else')
    
    # Call to sort(...): (line 165)
    # Processing the call keyword arguments (line 165)
    kwargs_708938 = {}
    # Getting the type of 'ar' (line 165)
    ar_708936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'ar', False)
    # Obtaining the member 'sort' of a type (line 165)
    sort_708937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), ar_708936, 'sort')
    # Calling sort(args, kwargs) (line 165)
    sort_call_result_708939 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), sort_708937, *[], **kwargs_708938)
    
    
    # Assigning a Name to a Name (line 166):
    # Getting the type of 'ar' (line 166)
    ar_708940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 18), 'ar')
    # Assigning a type to the variable 'aux' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'aux', ar_708940)
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 167):
    
    # Call to concatenate(...): (line 167)
    # Processing the call arguments (line 167)
    
    # Obtaining an instance of the builtin type 'tuple' (line 167)
    tuple_708943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 167)
    # Adding element type (line 167)
    
    # Obtaining an instance of the builtin type 'list' (line 167)
    list_708944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 167)
    # Adding element type (line 167)
    # Getting the type of 'True' (line 167)
    True_708945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 32), 'True', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 31), list_708944, True_708945)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 31), tuple_708943, list_708944)
    # Adding element type (line 167)
    
    
    # Obtaining the type of the subscript
    int_708946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 43), 'int')
    slice_708947 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 167, 39), int_708946, None, None)
    # Getting the type of 'aux' (line 167)
    aux_708948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'aux', False)
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___708949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 39), aux_708948, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_708950 = invoke(stypy.reporting.localization.Localization(__file__, 167, 39), getitem___708949, slice_708947)
    
    
    # Obtaining the type of the subscript
    int_708951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 55), 'int')
    slice_708952 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 167, 50), None, int_708951, None)
    # Getting the type of 'aux' (line 167)
    aux_708953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 50), 'aux', False)
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___708954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 50), aux_708953, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_708955 = invoke(stypy.reporting.localization.Localization(__file__, 167, 50), getitem___708954, slice_708952)
    
    # Applying the binary operator '!=' (line 167)
    result_ne_708956 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 39), '!=', subscript_call_result_708950, subscript_call_result_708955)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 31), tuple_708943, result_ne_708956)
    
    # Processing the call keyword arguments (line 167)
    kwargs_708957 = {}
    # Getting the type of 'np' (line 167)
    np_708941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 15), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 167)
    concatenate_708942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 15), np_708941, 'concatenate')
    # Calling concatenate(args, kwargs) (line 167)
    concatenate_call_result_708958 = invoke(stypy.reporting.localization.Localization(__file__, 167, 15), concatenate_708942, *[tuple_708943], **kwargs_708957)
    
    # Assigning a type to the variable 'flag' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'flag', concatenate_call_result_708958)
    
    
    # Getting the type of 'optional_returns' (line 169)
    optional_returns_708959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'optional_returns')
    # Applying the 'not' unary operator (line 169)
    result_not__708960 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 11), 'not', optional_returns_708959)
    
    # Testing the type of an if condition (line 169)
    if_condition_708961 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 8), result_not__708960)
    # Assigning a type to the variable 'if_condition_708961' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'if_condition_708961', if_condition_708961)
    # SSA begins for if statement (line 169)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 170):
    
    # Obtaining the type of the subscript
    # Getting the type of 'flag' (line 170)
    flag_708962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 22), 'flag')
    # Getting the type of 'aux' (line 170)
    aux_708963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'aux')
    # Obtaining the member '__getitem__' of a type (line 170)
    getitem___708964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 18), aux_708963, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 170)
    subscript_call_result_708965 = invoke(stypy.reporting.localization.Localization(__file__, 170, 18), getitem___708964, flag_708962)
    
    # Assigning a type to the variable 'ret' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'ret', subscript_call_result_708965)
    # SSA branch for the else part of an if statement (line 169)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Tuple to a Name (line 172):
    
    # Obtaining an instance of the builtin type 'tuple' (line 172)
    tuple_708966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 172)
    # Adding element type (line 172)
    
    # Obtaining the type of the subscript
    # Getting the type of 'flag' (line 172)
    flag_708967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 23), 'flag')
    # Getting the type of 'aux' (line 172)
    aux_708968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 19), 'aux')
    # Obtaining the member '__getitem__' of a type (line 172)
    getitem___708969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 19), aux_708968, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 172)
    subscript_call_result_708970 = invoke(stypy.reporting.localization.Localization(__file__, 172, 19), getitem___708969, flag_708967)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 19), tuple_708966, subscript_call_result_708970)
    
    # Assigning a type to the variable 'ret' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'ret', tuple_708966)
    
    # Getting the type of 'return_index' (line 173)
    return_index_708971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'return_index')
    # Testing the type of an if condition (line 173)
    if_condition_708972 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 12), return_index_708971)
    # Assigning a type to the variable 'if_condition_708972' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'if_condition_708972', if_condition_708972)
    # SSA begins for if statement (line 173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'ret' (line 174)
    ret_708973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'ret')
    
    # Obtaining an instance of the builtin type 'tuple' (line 174)
    tuple_708974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 174)
    # Adding element type (line 174)
    
    # Obtaining the type of the subscript
    # Getting the type of 'flag' (line 174)
    flag_708975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 29), 'flag')
    # Getting the type of 'perm' (line 174)
    perm_708976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'perm')
    # Obtaining the member '__getitem__' of a type (line 174)
    getitem___708977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 24), perm_708976, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 174)
    subscript_call_result_708978 = invoke(stypy.reporting.localization.Localization(__file__, 174, 24), getitem___708977, flag_708975)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 24), tuple_708974, subscript_call_result_708978)
    
    # Applying the binary operator '+=' (line 174)
    result_iadd_708979 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 16), '+=', ret_708973, tuple_708974)
    # Assigning a type to the variable 'ret' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'ret', result_iadd_708979)
    
    # SSA join for if statement (line 173)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_inverse' (line 175)
    return_inverse_708980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'return_inverse')
    # Testing the type of an if condition (line 175)
    if_condition_708981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 12), return_inverse_708980)
    # Assigning a type to the variable 'if_condition_708981' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'if_condition_708981', if_condition_708981)
    # SSA begins for if statement (line 175)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 176):
    
    # Call to cumsum(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'flag' (line 176)
    flag_708984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 34), 'flag', False)
    # Processing the call keyword arguments (line 176)
    kwargs_708985 = {}
    # Getting the type of 'np' (line 176)
    np_708982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'np', False)
    # Obtaining the member 'cumsum' of a type (line 176)
    cumsum_708983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 24), np_708982, 'cumsum')
    # Calling cumsum(args, kwargs) (line 176)
    cumsum_call_result_708986 = invoke(stypy.reporting.localization.Localization(__file__, 176, 24), cumsum_708983, *[flag_708984], **kwargs_708985)
    
    int_708987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 42), 'int')
    # Applying the binary operator '-' (line 176)
    result_sub_708988 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 24), '-', cumsum_call_result_708986, int_708987)
    
    # Assigning a type to the variable 'iflag' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'iflag', result_sub_708988)
    
    # Assigning a Call to a Name (line 177):
    
    # Call to empty(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'ar' (line 177)
    ar_708991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'ar', False)
    # Obtaining the member 'shape' of a type (line 177)
    shape_708992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 35), ar_708991, 'shape')
    # Processing the call keyword arguments (line 177)
    # Getting the type of 'np' (line 177)
    np_708993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 51), 'np', False)
    # Obtaining the member 'intp' of a type (line 177)
    intp_708994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 51), np_708993, 'intp')
    keyword_708995 = intp_708994
    kwargs_708996 = {'dtype': keyword_708995}
    # Getting the type of 'np' (line 177)
    np_708989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 26), 'np', False)
    # Obtaining the member 'empty' of a type (line 177)
    empty_708990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 26), np_708989, 'empty')
    # Calling empty(args, kwargs) (line 177)
    empty_call_result_708997 = invoke(stypy.reporting.localization.Localization(__file__, 177, 26), empty_708990, *[shape_708992], **kwargs_708996)
    
    # Assigning a type to the variable 'inv_idx' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'inv_idx', empty_call_result_708997)
    
    # Assigning a Name to a Subscript (line 178):
    # Getting the type of 'iflag' (line 178)
    iflag_708998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 32), 'iflag')
    # Getting the type of 'inv_idx' (line 178)
    inv_idx_708999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 16), 'inv_idx')
    # Getting the type of 'perm' (line 178)
    perm_709000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 24), 'perm')
    # Storing an element on a container (line 178)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 16), inv_idx_708999, (perm_709000, iflag_708998))
    
    # Getting the type of 'ret' (line 179)
    ret_709001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'ret')
    
    # Obtaining an instance of the builtin type 'tuple' (line 179)
    tuple_709002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 179)
    # Adding element type (line 179)
    # Getting the type of 'inv_idx' (line 179)
    inv_idx_709003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'inv_idx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 24), tuple_709002, inv_idx_709003)
    
    # Applying the binary operator '+=' (line 179)
    result_iadd_709004 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 16), '+=', ret_709001, tuple_709002)
    # Assigning a type to the variable 'ret' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'ret', result_iadd_709004)
    
    # SSA join for if statement (line 175)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'return_counts' (line 180)
    return_counts_709005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 15), 'return_counts')
    # Testing the type of an if condition (line 180)
    if_condition_709006 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 12), return_counts_709005)
    # Assigning a type to the variable 'if_condition_709006' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'if_condition_709006', if_condition_709006)
    # SSA begins for if statement (line 180)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 181):
    
    # Call to concatenate(...): (line 181)
    # Processing the call arguments (line 181)
    
    # Call to nonzero(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'flag' (line 181)
    flag_709011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 48), 'flag', False)
    # Processing the call keyword arguments (line 181)
    kwargs_709012 = {}
    # Getting the type of 'np' (line 181)
    np_709009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 37), 'np', False)
    # Obtaining the member 'nonzero' of a type (line 181)
    nonzero_709010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 37), np_709009, 'nonzero')
    # Calling nonzero(args, kwargs) (line 181)
    nonzero_call_result_709013 = invoke(stypy.reporting.localization.Localization(__file__, 181, 37), nonzero_709010, *[flag_709011], **kwargs_709012)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 181)
    tuple_709014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 181)
    # Adding element type (line 181)
    
    # Obtaining an instance of the builtin type 'list' (line 181)
    list_709015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 181)
    # Adding element type (line 181)
    # Getting the type of 'ar' (line 181)
    ar_709016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 58), 'ar', False)
    # Obtaining the member 'size' of a type (line 181)
    size_709017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 58), ar_709016, 'size')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 57), list_709015, size_709017)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 57), tuple_709014, list_709015)
    
    # Applying the binary operator '+' (line 181)
    result_add_709018 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 37), '+', nonzero_call_result_709013, tuple_709014)
    
    # Processing the call keyword arguments (line 181)
    kwargs_709019 = {}
    # Getting the type of 'np' (line 181)
    np_709007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 22), 'np', False)
    # Obtaining the member 'concatenate' of a type (line 181)
    concatenate_709008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 22), np_709007, 'concatenate')
    # Calling concatenate(args, kwargs) (line 181)
    concatenate_call_result_709020 = invoke(stypy.reporting.localization.Localization(__file__, 181, 22), concatenate_709008, *[result_add_709018], **kwargs_709019)
    
    # Assigning a type to the variable 'idx' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'idx', concatenate_call_result_709020)
    
    # Getting the type of 'ret' (line 182)
    ret_709021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'ret')
    
    # Obtaining an instance of the builtin type 'tuple' (line 182)
    tuple_709022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 182)
    # Adding element type (line 182)
    
    # Call to diff(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'idx' (line 182)
    idx_709025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 32), 'idx', False)
    # Processing the call keyword arguments (line 182)
    kwargs_709026 = {}
    # Getting the type of 'np' (line 182)
    np_709023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 24), 'np', False)
    # Obtaining the member 'diff' of a type (line 182)
    diff_709024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 24), np_709023, 'diff')
    # Calling diff(args, kwargs) (line 182)
    diff_call_result_709027 = invoke(stypy.reporting.localization.Localization(__file__, 182, 24), diff_709024, *[idx_709025], **kwargs_709026)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 24), tuple_709022, diff_call_result_709027)
    
    # Applying the binary operator '+=' (line 182)
    result_iadd_709028 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 16), '+=', ret_709021, tuple_709022)
    # Assigning a type to the variable 'ret' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 16), 'ret', result_iadd_709028)
    
    # SSA join for if statement (line 180)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 169)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 183)
    ret_709029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stypy_return_type', ret_709029)
    
    # ################# End of 'unique(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unique' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_709030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_709030)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unique'
    return stypy_return_type_709030

# Assigning a type to the variable 'unique' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'unique', unique)
# SSA join for if statement (line 91)
module_type_store = module_type_store.join_ssa_context()




# Call to NumpyVersion(...): (line 186)
# Processing the call arguments (line 186)
# Getting the type of 'np' (line 186)
np_709032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'np', False)
# Obtaining the member '__version__' of a type (line 186)
version___709033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 16), np_709032, '__version__')
# Processing the call keyword arguments (line 186)
kwargs_709034 = {}
# Getting the type of 'NumpyVersion' (line 186)
NumpyVersion_709031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 3), 'NumpyVersion', False)
# Calling NumpyVersion(args, kwargs) (line 186)
NumpyVersion_call_result_709035 = invoke(stypy.reporting.localization.Localization(__file__, 186, 3), NumpyVersion_709031, *[version___709033], **kwargs_709034)

str_709036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 34), 'str', '1.12.0.dev')
# Applying the binary operator '>' (line 186)
result_gt_709037 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 3), '>', NumpyVersion_call_result_709035, str_709036)

# Testing the type of an if condition (line 186)
if_condition_709038 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 0), result_gt_709037)
# Assigning a type to the variable 'if_condition_709038' (line 186)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'if_condition_709038', if_condition_709038)
# SSA begins for if statement (line 186)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Attribute to a Name (line 187):
# Getting the type of 'np' (line 187)
np_709039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'np')
# Obtaining the member 'polynomial' of a type (line 187)
polynomial_709040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 23), np_709039, 'polynomial')
# Obtaining the member 'polynomial' of a type (line 187)
polynomial_709041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 23), polynomial_709040, 'polynomial')
# Obtaining the member 'polyvalfromroots' of a type (line 187)
polyvalfromroots_709042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 23), polynomial_709041, 'polyvalfromroots')
# Assigning a type to the variable 'polyvalfromroots' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'polyvalfromroots', polyvalfromroots_709042)
# SSA branch for the else part of an if statement (line 186)
module_type_store.open_ssa_branch('else')

@norecursion
def polyvalfromroots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 189)
    True_709043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 38), 'True')
    defaults = [True_709043]
    # Create a new context for function 'polyvalfromroots'
    module_type_store = module_type_store.open_function_context('polyvalfromroots', 189, 4, False)
    
    # Passed parameters checking function
    polyvalfromroots.stypy_localization = localization
    polyvalfromroots.stypy_type_of_self = None
    polyvalfromroots.stypy_type_store = module_type_store
    polyvalfromroots.stypy_function_name = 'polyvalfromroots'
    polyvalfromroots.stypy_param_names_list = ['x', 'r', 'tensor']
    polyvalfromroots.stypy_varargs_param_name = None
    polyvalfromroots.stypy_kwargs_param_name = None
    polyvalfromroots.stypy_call_defaults = defaults
    polyvalfromroots.stypy_call_varargs = varargs
    polyvalfromroots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'polyvalfromroots', ['x', 'r', 'tensor'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'polyvalfromroots', localization, ['x', 'r', 'tensor'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'polyvalfromroots(...)' code ##################

    str_709044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 263, (-1)), 'str', '\n        Evaluate a polynomial specified by its roots at points x.\n\n        This function is copypasted from numpy 1.12.0.dev.\n\n        If `r` is of length `N`, this function returns the value\n\n        .. math:: p(x) = \\prod_{n=1}^{N} (x - r_n)\n\n        The parameter `x` is converted to an array only if it is a tuple or a\n        list, otherwise it is treated as a scalar. In either case, either `x`\n        or its elements must support multiplication and addition both with\n        themselves and with the elements of `r`.\n\n        If `r` is a 1-D array, then `p(x)` will have the same shape as `x`.  If\n        `r` is multidimensional, then the shape of the result depends on the\n        value of `tensor`. If `tensor is ``True`` the shape will be r.shape[1:]\n        + x.shape; that is, each polynomial is evaluated at every value of `x`.\n        If `tensor` is ``False``, the shape will be r.shape[1:]; that is, each\n        polynomial is evaluated only for the corresponding broadcast value of\n        `x`. Note that scalars have shape (,).\n\n        Parameters\n        ----------\n        x : array_like, compatible object\n            If `x` is a list or tuple, it is converted to an ndarray, otherwise\n            it is left unchanged and treated as a scalar. In either case, `x`\n            or its elements must support addition and multiplication with with\n            themselves and with the elements of `r`.\n        r : array_like\n            Array of roots. If `r` is multidimensional the first index is the\n            root index, while the remaining indices enumerate multiple\n            polynomials. For instance, in the two dimensional case the roots of\n            each polynomial may be thought of as stored in the columns of `r`.\n        tensor : boolean, optional\n            If True, the shape of the roots array is extended with ones on the\n            right, one for each dimension of `x`. Scalars have dimension 0 for\n            this action. The result is that every column of coefficients in `r`\n            is evaluated for every element of `x`. If False, `x` is broadcast\n            over the columns of `r` for the evaluation.  This keyword is useful\n            when `r` is multidimensional. The default value is True.\n\n        Returns\n        -------\n        values : ndarray, compatible object\n            The shape of the returned array is described above.\n\n        See Also\n        --------\n        polyroots, polyfromroots, polyval\n\n        Examples\n        --------\n        >>> from numpy.polynomial.polynomial import polyvalfromroots\n        >>> polyvalfromroots(1, [1,2,3])\n        0.0\n        >>> a = np.arange(4).reshape(2,2)\n        >>> a\n        array([[0, 1],\n               [2, 3]])\n        >>> polyvalfromroots(a, [-1, 0, 1])\n        array([[ -0.,   0.],\n               [  6.,  24.]])\n        >>> r = np.arange(-2, 2).reshape(2,2) # multidimensional coefficients\n        >>> r # each column of r defines one polynomial\n        array([[-2, -1],\n               [ 0,  1]])\n        >>> b = [-2, 1]\n        >>> polyvalfromroots(b, r, tensor=True)\n        array([[-0.,  3.],\n               [ 3., 0.]])\n        >>> polyvalfromroots(b, r, tensor=False)\n        array([-0.,  0.])\n        ')
    
    # Assigning a Call to a Name (line 264):
    
    # Call to array(...): (line 264)
    # Processing the call arguments (line 264)
    # Getting the type of 'r' (line 264)
    r_709047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 21), 'r', False)
    # Processing the call keyword arguments (line 264)
    int_709048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 30), 'int')
    keyword_709049 = int_709048
    int_709050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 38), 'int')
    keyword_709051 = int_709050
    kwargs_709052 = {'copy': keyword_709051, 'ndmin': keyword_709049}
    # Getting the type of 'np' (line 264)
    np_709045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'np', False)
    # Obtaining the member 'array' of a type (line 264)
    array_709046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), np_709045, 'array')
    # Calling array(args, kwargs) (line 264)
    array_call_result_709053 = invoke(stypy.reporting.localization.Localization(__file__, 264, 12), array_709046, *[r_709047], **kwargs_709052)
    
    # Assigning a type to the variable 'r' (line 264)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'r', array_call_result_709053)
    
    
    # Getting the type of 'r' (line 265)
    r_709054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'r')
    # Obtaining the member 'dtype' of a type (line 265)
    dtype_709055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), r_709054, 'dtype')
    # Obtaining the member 'char' of a type (line 265)
    char_709056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), dtype_709055, 'char')
    str_709057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 27), 'str', '?bBhHiIlLqQpP')
    # Applying the binary operator 'in' (line 265)
    result_contains_709058 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 11), 'in', char_709056, str_709057)
    
    # Testing the type of an if condition (line 265)
    if_condition_709059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), result_contains_709058)
    # Assigning a type to the variable 'if_condition_709059' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_709059', if_condition_709059)
    # SSA begins for if statement (line 265)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 266):
    
    # Call to astype(...): (line 266)
    # Processing the call arguments (line 266)
    # Getting the type of 'np' (line 266)
    np_709062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 25), 'np', False)
    # Obtaining the member 'double' of a type (line 266)
    double_709063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 25), np_709062, 'double')
    # Processing the call keyword arguments (line 266)
    kwargs_709064 = {}
    # Getting the type of 'r' (line 266)
    r_709060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'r', False)
    # Obtaining the member 'astype' of a type (line 266)
    astype_709061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 16), r_709060, 'astype')
    # Calling astype(args, kwargs) (line 266)
    astype_call_result_709065 = invoke(stypy.reporting.localization.Localization(__file__, 266, 16), astype_709061, *[double_709063], **kwargs_709064)
    
    # Assigning a type to the variable 'r' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'r', astype_call_result_709065)
    # SSA join for if statement (line 265)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 267)
    # Processing the call arguments (line 267)
    # Getting the type of 'x' (line 267)
    x_709067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 22), 'x', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 267)
    tuple_709068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 267)
    # Adding element type (line 267)
    # Getting the type of 'tuple' (line 267)
    tuple_709069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 26), 'tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 26), tuple_709068, tuple_709069)
    # Adding element type (line 267)
    # Getting the type of 'list' (line 267)
    list_709070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 33), 'list', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 267, 26), tuple_709068, list_709070)
    
    # Processing the call keyword arguments (line 267)
    kwargs_709071 = {}
    # Getting the type of 'isinstance' (line 267)
    isinstance_709066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 267)
    isinstance_call_result_709072 = invoke(stypy.reporting.localization.Localization(__file__, 267, 11), isinstance_709066, *[x_709067, tuple_709068], **kwargs_709071)
    
    # Testing the type of an if condition (line 267)
    if_condition_709073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 8), isinstance_call_result_709072)
    # Assigning a type to the variable 'if_condition_709073' (line 267)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'if_condition_709073', if_condition_709073)
    # SSA begins for if statement (line 267)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 268):
    
    # Call to asarray(...): (line 268)
    # Processing the call arguments (line 268)
    # Getting the type of 'x' (line 268)
    x_709076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 27), 'x', False)
    # Processing the call keyword arguments (line 268)
    kwargs_709077 = {}
    # Getting the type of 'np' (line 268)
    np_709074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 16), 'np', False)
    # Obtaining the member 'asarray' of a type (line 268)
    asarray_709075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 16), np_709074, 'asarray')
    # Calling asarray(args, kwargs) (line 268)
    asarray_call_result_709078 = invoke(stypy.reporting.localization.Localization(__file__, 268, 16), asarray_709075, *[x_709076], **kwargs_709077)
    
    # Assigning a type to the variable 'x' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'x', asarray_call_result_709078)
    # SSA join for if statement (line 267)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isinstance(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'x' (line 269)
    x_709080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'x', False)
    # Getting the type of 'np' (line 269)
    np_709081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 25), 'np', False)
    # Obtaining the member 'ndarray' of a type (line 269)
    ndarray_709082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 25), np_709081, 'ndarray')
    # Processing the call keyword arguments (line 269)
    kwargs_709083 = {}
    # Getting the type of 'isinstance' (line 269)
    isinstance_709079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 269)
    isinstance_call_result_709084 = invoke(stypy.reporting.localization.Localization(__file__, 269, 11), isinstance_709079, *[x_709080, ndarray_709082], **kwargs_709083)
    
    # Testing the type of an if condition (line 269)
    if_condition_709085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 8), isinstance_call_result_709084)
    # Assigning a type to the variable 'if_condition_709085' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'if_condition_709085', if_condition_709085)
    # SSA begins for if statement (line 269)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'tensor' (line 270)
    tensor_709086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 15), 'tensor')
    # Testing the type of an if condition (line 270)
    if_condition_709087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 12), tensor_709086)
    # Assigning a type to the variable 'if_condition_709087' (line 270)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'if_condition_709087', if_condition_709087)
    # SSA begins for if statement (line 270)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 271):
    
    # Call to reshape(...): (line 271)
    # Processing the call arguments (line 271)
    # Getting the type of 'r' (line 271)
    r_709090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 30), 'r', False)
    # Obtaining the member 'shape' of a type (line 271)
    shape_709091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 30), r_709090, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 271)
    tuple_709092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 41), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 271)
    # Adding element type (line 271)
    int_709093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 41), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 41), tuple_709092, int_709093)
    
    # Getting the type of 'x' (line 271)
    x_709094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 45), 'x', False)
    # Obtaining the member 'ndim' of a type (line 271)
    ndim_709095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 45), x_709094, 'ndim')
    # Applying the binary operator '*' (line 271)
    result_mul_709096 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 40), '*', tuple_709092, ndim_709095)
    
    # Applying the binary operator '+' (line 271)
    result_add_709097 = python_operator(stypy.reporting.localization.Localization(__file__, 271, 30), '+', shape_709091, result_mul_709096)
    
    # Processing the call keyword arguments (line 271)
    kwargs_709098 = {}
    # Getting the type of 'r' (line 271)
    r_709088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 20), 'r', False)
    # Obtaining the member 'reshape' of a type (line 271)
    reshape_709089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 20), r_709088, 'reshape')
    # Calling reshape(args, kwargs) (line 271)
    reshape_call_result_709099 = invoke(stypy.reporting.localization.Localization(__file__, 271, 20), reshape_709089, *[result_add_709097], **kwargs_709098)
    
    # Assigning a type to the variable 'r' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'r', reshape_call_result_709099)
    # SSA branch for the else part of an if statement (line 270)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'x' (line 272)
    x_709100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'x')
    # Obtaining the member 'ndim' of a type (line 272)
    ndim_709101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 17), x_709100, 'ndim')
    # Getting the type of 'r' (line 272)
    r_709102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 27), 'r')
    # Obtaining the member 'ndim' of a type (line 272)
    ndim_709103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 27), r_709102, 'ndim')
    # Applying the binary operator '>=' (line 272)
    result_ge_709104 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 17), '>=', ndim_709101, ndim_709103)
    
    # Testing the type of an if condition (line 272)
    if_condition_709105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 272, 17), result_ge_709104)
    # Assigning a type to the variable 'if_condition_709105' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 17), 'if_condition_709105', if_condition_709105)
    # SSA begins for if statement (line 272)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 273)
    # Processing the call arguments (line 273)
    str_709107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 33), 'str', 'x.ndim must be < r.ndim when tensor == False')
    # Processing the call keyword arguments (line 273)
    kwargs_709108 = {}
    # Getting the type of 'ValueError' (line 273)
    ValueError_709106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 273)
    ValueError_call_result_709109 = invoke(stypy.reporting.localization.Localization(__file__, 273, 22), ValueError_709106, *[str_709107], **kwargs_709108)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 273, 16), ValueError_call_result_709109, 'raise parameter', BaseException)
    # SSA join for if statement (line 272)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 270)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 269)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to prod(...): (line 275)
    # Processing the call arguments (line 275)
    # Getting the type of 'x' (line 275)
    x_709112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 23), 'x', False)
    # Getting the type of 'r' (line 275)
    r_709113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 27), 'r', False)
    # Applying the binary operator '-' (line 275)
    result_sub_709114 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 23), '-', x_709112, r_709113)
    
    # Processing the call keyword arguments (line 275)
    int_709115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 35), 'int')
    keyword_709116 = int_709115
    kwargs_709117 = {'axis': keyword_709116}
    # Getting the type of 'np' (line 275)
    np_709110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 15), 'np', False)
    # Obtaining the member 'prod' of a type (line 275)
    prod_709111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 15), np_709110, 'prod')
    # Calling prod(args, kwargs) (line 275)
    prod_call_result_709118 = invoke(stypy.reporting.localization.Localization(__file__, 275, 15), prod_709111, *[result_sub_709114], **kwargs_709117)
    
    # Assigning a type to the variable 'stypy_return_type' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'stypy_return_type', prod_call_result_709118)
    
    # ################# End of 'polyvalfromroots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'polyvalfromroots' in the type store
    # Getting the type of 'stypy_return_type' (line 189)
    stypy_return_type_709119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_709119)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'polyvalfromroots'
    return stypy_return_type_709119

# Assigning a type to the variable 'polyvalfromroots' (line 189)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'polyvalfromroots', polyvalfromroots)
# SSA join for if statement (line 186)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 278)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 279, 4))

# 'from numpy.testing import suppress_warnings' statement (line 279)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
import_709120 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 279, 4), 'numpy.testing')

if (type(import_709120) is not StypyTypeError):

    if (import_709120 != 'pyd_module'):
        __import__(import_709120)
        sys_modules_709121 = sys.modules[import_709120]
        import_from_module(stypy.reporting.localization.Localization(__file__, 279, 4), 'numpy.testing', sys_modules_709121.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 279, 4), __file__, sys_modules_709121, sys_modules_709121.module_type_store, module_type_store)
    else:
        from numpy.testing import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 279, 4), 'numpy.testing', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'numpy.testing' (line 279)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'numpy.testing', import_709120)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')

# SSA branch for the except part of a try statement (line 278)
# SSA branch for the except 'ImportError' branch of a try statement (line 278)
module_type_store.open_ssa_branch('except')
# Declaration of the 'suppress_warnings' class

class suppress_warnings(object, ):
    str_709122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, (-1)), 'str', '\n        Context manager and decorator doing much the same as\n        ``warnings.catch_warnings``.\n\n        However, it also provides a filter mechanism to work around\n        http://bugs.python.org/issue4180.\n\n        This bug causes Python before 3.4 to not reliably show warnings again\n        after they have been ignored once (even within catch_warnings). It\n        means that no "ignore" filter can be used easily, since following\n        tests might need to see the warning. Additionally it allows easier\n        specificity for testing warnings and can be nested.\n\n        Parameters\n        ----------\n        forwarding_rule : str, optional\n            One of "always", "once", "module", or "location". Analogous to\n            the usual warnings module filter mode, it is useful to reduce\n            noise mostly on the outmost level. Unsuppressed and unrecorded\n            warnings will be forwarded based on this rule. Defaults to "always".\n            "location" is equivalent to the warnings "default", match by exact\n            location the warning warning originated from.\n\n        Notes\n        -----\n        Filters added inside the context manager will be discarded again\n        when leaving it. Upon entering all filters defined outside a\n        context will be applied automatically.\n\n        When a recording filter is added, matching warnings are stored in the\n        ``log`` attribute as well as in the list returned by ``record``.\n\n        If filters are added and the ``module`` keyword is given, the\n        warning registry of this module will additionally be cleared when\n        applying it, entering the context, or exiting it. This could cause\n        warnings to appear a second time after leaving the context if they\n        were configured to be printed once (default) and were already\n        printed before the context was entered.\n\n        Nesting this context manager will work as expected when the\n        forwarding rule is "always" (default). Unfiltered and unrecorded\n        warnings will be passed out and be matched by the outer level.\n        On the outmost level they will be printed (or caught by another\n        warnings context). The forwarding rule argument can modify this\n        behaviour.\n\n        Like ``catch_warnings`` this context manager is not threadsafe.\n\n        Examples\n        --------\n        >>> with suppress_warnings() as sup:\n        ...     sup.filter(DeprecationWarning, "Some text")\n        ...     sup.filter(module=np.ma.core)\n        ...     log = sup.record(FutureWarning, "Does this occur?")\n        ...     command_giving_warnings()\n        ...     # The FutureWarning was given once, the filtered warnings were\n        ...     # ignored. All other warnings abide outside settings (may be\n        ...     # printed/error)\n        ...     assert_(len(log) == 1)\n        ...     assert_(len(sup.log) == 1)  # also stored in log attribute\n\n        Or as a decorator:\n\n        >>> sup = suppress_warnings()\n        >>> sup.filter(module=np.ma.core)  # module must match exact\n        >>> @sup\n        >>> def some_function():\n        ...     # do something which causes a warning in np.ma.core\n        ...     pass\n        ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_709123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 352, 43), 'str', 'always')
        defaults = [str_709123]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 352, 8, False)
        # Assigning a type to the variable 'self' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'suppress_warnings.__init__', ['forwarding_rule'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['forwarding_rule'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 353):
        # Getting the type of 'False' (line 353)
        False_709124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 28), 'False')
        # Getting the type of 'self' (line 353)
        self_709125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 12), 'self')
        # Setting the type of the member '_entered' of a type (line 353)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 353, 12), self_709125, '_entered', False_709124)
        
        # Assigning a List to a Attribute (line 356):
        
        # Obtaining an instance of the builtin type 'list' (line 356)
        list_709126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 356)
        
        # Getting the type of 'self' (line 356)
        self_709127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'self')
        # Setting the type of the member '_suppressions' of a type (line 356)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 12), self_709127, '_suppressions', list_709126)
        
        
        # Getting the type of 'forwarding_rule' (line 358)
        forwarding_rule_709128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 15), 'forwarding_rule')
        
        # Obtaining an instance of the builtin type 'set' (line 358)
        set_709129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 38), 'set')
        # Adding type elements to the builtin type 'set' instance (line 358)
        # Adding element type (line 358)
        str_709130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 39), 'str', 'always')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 38), set_709129, str_709130)
        # Adding element type (line 358)
        str_709131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 49), 'str', 'module')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 38), set_709129, str_709131)
        # Adding element type (line 358)
        str_709132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 59), 'str', 'once')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 38), set_709129, str_709132)
        # Adding element type (line 358)
        str_709133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 67), 'str', 'location')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 358, 38), set_709129, str_709133)
        
        # Applying the binary operator 'notin' (line 358)
        result_contains_709134 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 15), 'notin', forwarding_rule_709128, set_709129)
        
        # Testing the type of an if condition (line 358)
        if_condition_709135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 358, 12), result_contains_709134)
        # Assigning a type to the variable 'if_condition_709135' (line 358)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'if_condition_709135', if_condition_709135)
        # SSA begins for if statement (line 358)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 359)
        # Processing the call arguments (line 359)
        str_709137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 33), 'str', 'unsupported forwarding rule.')
        # Processing the call keyword arguments (line 359)
        kwargs_709138 = {}
        # Getting the type of 'ValueError' (line 359)
        ValueError_709136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 359)
        ValueError_call_result_709139 = invoke(stypy.reporting.localization.Localization(__file__, 359, 22), ValueError_709136, *[str_709137], **kwargs_709138)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 359, 16), ValueError_call_result_709139, 'raise parameter', BaseException)
        # SSA join for if statement (line 358)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 360):
        # Getting the type of 'forwarding_rule' (line 360)
        forwarding_rule_709140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 36), 'forwarding_rule')
        # Getting the type of 'self' (line 360)
        self_709141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'self')
        # Setting the type of the member '_forwarding_rule' of a type (line 360)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), self_709141, '_forwarding_rule', forwarding_rule_709140)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _clear_registries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_clear_registries'
        module_type_store = module_type_store.open_function_context('_clear_registries', 362, 8, False)
        # Assigning a type to the variable 'self' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        suppress_warnings._clear_registries.__dict__.__setitem__('stypy_localization', localization)
        suppress_warnings._clear_registries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        suppress_warnings._clear_registries.__dict__.__setitem__('stypy_type_store', module_type_store)
        suppress_warnings._clear_registries.__dict__.__setitem__('stypy_function_name', 'suppress_warnings._clear_registries')
        suppress_warnings._clear_registries.__dict__.__setitem__('stypy_param_names_list', [])
        suppress_warnings._clear_registries.__dict__.__setitem__('stypy_varargs_param_name', None)
        suppress_warnings._clear_registries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        suppress_warnings._clear_registries.__dict__.__setitem__('stypy_call_defaults', defaults)
        suppress_warnings._clear_registries.__dict__.__setitem__('stypy_call_varargs', varargs)
        suppress_warnings._clear_registries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        suppress_warnings._clear_registries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'suppress_warnings._clear_registries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_clear_registries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_clear_registries(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 363)
        str_709142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 33), 'str', '_filters_mutated')
        # Getting the type of 'warnings' (line 363)
        warnings_709143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 23), 'warnings')
        
        (may_be_709144, more_types_in_union_709145) = may_provide_member(str_709142, warnings_709143)

        if may_be_709144:

            if more_types_in_union_709145:
                # Runtime conditional SSA (line 363)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'warnings' (line 363)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 12), 'warnings', remove_not_member_provider_from_union(warnings_709143, '_filters_mutated'))
            
            # Call to _filters_mutated(...): (line 366)
            # Processing the call keyword arguments (line 366)
            kwargs_709148 = {}
            # Getting the type of 'warnings' (line 366)
            warnings_709146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 'warnings', False)
            # Obtaining the member '_filters_mutated' of a type (line 366)
            _filters_mutated_709147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 366, 16), warnings_709146, '_filters_mutated')
            # Calling _filters_mutated(args, kwargs) (line 366)
            _filters_mutated_call_result_709149 = invoke(stypy.reporting.localization.Localization(__file__, 366, 16), _filters_mutated_709147, *[], **kwargs_709148)
            
            # Assigning a type to the variable 'stypy_return_type' (line 367)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 16), 'stypy_return_type', types.NoneType)

            if more_types_in_union_709145:
                # SSA join for if statement (line 363)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'self' (line 370)
        self_709150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 26), 'self')
        # Obtaining the member '_tmp_modules' of a type (line 370)
        _tmp_modules_709151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 26), self_709150, '_tmp_modules')
        # Testing the type of a for loop iterable (line 370)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 370, 12), _tmp_modules_709151)
        # Getting the type of the for loop variable (line 370)
        for_loop_var_709152 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 370, 12), _tmp_modules_709151)
        # Assigning a type to the variable 'module' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'module', for_loop_var_709152)
        # SSA begins for a for statement (line 370)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 371)
        str_709153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 35), 'str', '__warningregistry__')
        # Getting the type of 'module' (line 371)
        module_709154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 27), 'module')
        
        (may_be_709155, more_types_in_union_709156) = may_provide_member(str_709153, module_709154)

        if may_be_709155:

            if more_types_in_union_709156:
                # Runtime conditional SSA (line 371)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'module' (line 371)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 16), 'module', remove_not_member_provider_from_union(module_709154, '__warningregistry__'))
            
            # Call to clear(...): (line 372)
            # Processing the call keyword arguments (line 372)
            kwargs_709160 = {}
            # Getting the type of 'module' (line 372)
            module_709157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'module', False)
            # Obtaining the member '__warningregistry__' of a type (line 372)
            warningregistry___709158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 20), module_709157, '__warningregistry__')
            # Obtaining the member 'clear' of a type (line 372)
            clear_709159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 20), warningregistry___709158, 'clear')
            # Calling clear(args, kwargs) (line 372)
            clear_call_result_709161 = invoke(stypy.reporting.localization.Localization(__file__, 372, 20), clear_709159, *[], **kwargs_709160)
            

            if more_types_in_union_709156:
                # SSA join for if statement (line 371)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_clear_registries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_clear_registries' in the type store
        # Getting the type of 'stypy_return_type' (line 362)
        stypy_return_type_709162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709162)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_clear_registries'
        return stypy_return_type_709162


    @norecursion
    def _filter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'Warning' (line 374)
        Warning_709163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 35), 'Warning')
        str_709164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 52), 'str', '')
        # Getting the type of 'None' (line 374)
        None_709165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 63), 'None')
        # Getting the type of 'False' (line 374)
        False_709166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 76), 'False')
        defaults = [Warning_709163, str_709164, None_709165, False_709166]
        # Create a new context for function '_filter'
        module_type_store = module_type_store.open_function_context('_filter', 374, 8, False)
        # Assigning a type to the variable 'self' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        suppress_warnings._filter.__dict__.__setitem__('stypy_localization', localization)
        suppress_warnings._filter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        suppress_warnings._filter.__dict__.__setitem__('stypy_type_store', module_type_store)
        suppress_warnings._filter.__dict__.__setitem__('stypy_function_name', 'suppress_warnings._filter')
        suppress_warnings._filter.__dict__.__setitem__('stypy_param_names_list', ['category', 'message', 'module', 'record'])
        suppress_warnings._filter.__dict__.__setitem__('stypy_varargs_param_name', None)
        suppress_warnings._filter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        suppress_warnings._filter.__dict__.__setitem__('stypy_call_defaults', defaults)
        suppress_warnings._filter.__dict__.__setitem__('stypy_call_varargs', varargs)
        suppress_warnings._filter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        suppress_warnings._filter.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'suppress_warnings._filter', ['category', 'message', 'module', 'record'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_filter', localization, ['category', 'message', 'module', 'record'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_filter(...)' code ##################

        
        # Getting the type of 'record' (line 375)
        record_709167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 15), 'record')
        # Testing the type of an if condition (line 375)
        if_condition_709168 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 12), record_709167)
        # Assigning a type to the variable 'if_condition_709168' (line 375)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 12), 'if_condition_709168', if_condition_709168)
        # SSA begins for if statement (line 375)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Name (line 376):
        
        # Obtaining an instance of the builtin type 'list' (line 376)
        list_709169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 376)
        
        # Assigning a type to the variable 'record' (line 376)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 16), 'record', list_709169)
        # SSA branch for the else part of an if statement (line 375)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 378):
        # Getting the type of 'None' (line 378)
        None_709170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 25), 'None')
        # Assigning a type to the variable 'record' (line 378)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 16), 'record', None_709170)
        # SSA join for if statement (line 375)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 379)
        self_709171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 15), 'self')
        # Obtaining the member '_entered' of a type (line 379)
        _entered_709172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 15), self_709171, '_entered')
        # Testing the type of an if condition (line 379)
        if_condition_709173 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 12), _entered_709172)
        # Assigning a type to the variable 'if_condition_709173' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'if_condition_709173', if_condition_709173)
        # SSA begins for if statement (line 379)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 380)
        # Getting the type of 'module' (line 380)
        module_709174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 19), 'module')
        # Getting the type of 'None' (line 380)
        None_709175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 29), 'None')
        
        (may_be_709176, more_types_in_union_709177) = may_be_none(module_709174, None_709175)

        if may_be_709176:

            if more_types_in_union_709177:
                # Runtime conditional SSA (line 380)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to filterwarnings(...): (line 381)
            # Processing the call arguments (line 381)
            str_709180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 24), 'str', 'always')
            # Processing the call keyword arguments (line 381)
            # Getting the type of 'category' (line 382)
            category_709181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 43), 'category', False)
            keyword_709182 = category_709181
            # Getting the type of 'message' (line 382)
            message_709183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 61), 'message', False)
            keyword_709184 = message_709183
            kwargs_709185 = {'category': keyword_709182, 'message': keyword_709184}
            # Getting the type of 'warnings' (line 381)
            warnings_709178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 20), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 381)
            filterwarnings_709179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 20), warnings_709178, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 381)
            filterwarnings_call_result_709186 = invoke(stypy.reporting.localization.Localization(__file__, 381, 20), filterwarnings_709179, *[str_709180], **kwargs_709185)
            

            if more_types_in_union_709177:
                # Runtime conditional SSA for else branch (line 380)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_709176) or more_types_in_union_709177):
            
            # Assigning a BinOp to a Name (line 384):
            
            # Call to replace(...): (line 384)
            # Processing the call arguments (line 384)
            str_709190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 59), 'str', '.')
            str_709191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 64), 'str', '\\.')
            # Processing the call keyword arguments (line 384)
            kwargs_709192 = {}
            # Getting the type of 'module' (line 384)
            module_709187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 35), 'module', False)
            # Obtaining the member '__name__' of a type (line 384)
            name___709188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 35), module_709187, '__name__')
            # Obtaining the member 'replace' of a type (line 384)
            replace_709189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 35), name___709188, 'replace')
            # Calling replace(args, kwargs) (line 384)
            replace_call_result_709193 = invoke(stypy.reporting.localization.Localization(__file__, 384, 35), replace_709189, *[str_709190, str_709191], **kwargs_709192)
            
            str_709194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 73), 'str', '$')
            # Applying the binary operator '+' (line 384)
            result_add_709195 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 35), '+', replace_call_result_709193, str_709194)
            
            # Assigning a type to the variable 'module_regex' (line 384)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 20), 'module_regex', result_add_709195)
            
            # Call to filterwarnings(...): (line 385)
            # Processing the call arguments (line 385)
            str_709198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 386, 24), 'str', 'always')
            # Processing the call keyword arguments (line 385)
            # Getting the type of 'category' (line 386)
            category_709199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 43), 'category', False)
            keyword_709200 = category_709199
            # Getting the type of 'message' (line 386)
            message_709201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 61), 'message', False)
            keyword_709202 = message_709201
            # Getting the type of 'module_regex' (line 387)
            module_regex_709203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 31), 'module_regex', False)
            keyword_709204 = module_regex_709203
            kwargs_709205 = {'category': keyword_709200, 'message': keyword_709202, 'module': keyword_709204}
            # Getting the type of 'warnings' (line 385)
            warnings_709196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 20), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 385)
            filterwarnings_709197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 20), warnings_709196, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 385)
            filterwarnings_call_result_709206 = invoke(stypy.reporting.localization.Localization(__file__, 385, 20), filterwarnings_709197, *[str_709198], **kwargs_709205)
            
            
            # Call to add(...): (line 388)
            # Processing the call arguments (line 388)
            # Getting the type of 'module' (line 388)
            module_709210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 42), 'module', False)
            # Processing the call keyword arguments (line 388)
            kwargs_709211 = {}
            # Getting the type of 'self' (line 388)
            self_709207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 20), 'self', False)
            # Obtaining the member '_tmp_modules' of a type (line 388)
            _tmp_modules_709208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 20), self_709207, '_tmp_modules')
            # Obtaining the member 'add' of a type (line 388)
            add_709209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 20), _tmp_modules_709208, 'add')
            # Calling add(args, kwargs) (line 388)
            add_call_result_709212 = invoke(stypy.reporting.localization.Localization(__file__, 388, 20), add_709209, *[module_709210], **kwargs_709211)
            
            
            # Call to _clear_registries(...): (line 389)
            # Processing the call keyword arguments (line 389)
            kwargs_709215 = {}
            # Getting the type of 'self' (line 389)
            self_709213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 20), 'self', False)
            # Obtaining the member '_clear_registries' of a type (line 389)
            _clear_registries_709214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 20), self_709213, '_clear_registries')
            # Calling _clear_registries(args, kwargs) (line 389)
            _clear_registries_call_result_709216 = invoke(stypy.reporting.localization.Localization(__file__, 389, 20), _clear_registries_709214, *[], **kwargs_709215)
            

            if (may_be_709176 and more_types_in_union_709177):
                # SSA join for if statement (line 380)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to append(...): (line 391)
        # Processing the call arguments (line 391)
        
        # Obtaining an instance of the builtin type 'tuple' (line 392)
        tuple_709220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 392)
        # Adding element type (line 392)
        # Getting the type of 'category' (line 392)
        category_709221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 21), 'category', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 21), tuple_709220, category_709221)
        # Adding element type (line 392)
        # Getting the type of 'message' (line 392)
        message_709222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 31), 'message', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 21), tuple_709220, message_709222)
        # Adding element type (line 392)
        
        # Call to compile(...): (line 392)
        # Processing the call arguments (line 392)
        # Getting the type of 'message' (line 392)
        message_709225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 51), 'message', False)
        # Getting the type of 're' (line 392)
        re_709226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 60), 're', False)
        # Obtaining the member 'I' of a type (line 392)
        I_709227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 60), re_709226, 'I')
        # Processing the call keyword arguments (line 392)
        kwargs_709228 = {}
        # Getting the type of 're' (line 392)
        re_709223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 40), 're', False)
        # Obtaining the member 'compile' of a type (line 392)
        compile_709224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 40), re_709223, 'compile')
        # Calling compile(args, kwargs) (line 392)
        compile_call_result_709229 = invoke(stypy.reporting.localization.Localization(__file__, 392, 40), compile_709224, *[message_709225, I_709227], **kwargs_709228)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 21), tuple_709220, compile_call_result_709229)
        # Adding element type (line 392)
        # Getting the type of 'module' (line 392)
        module_709230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 67), 'module', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 21), tuple_709220, module_709230)
        # Adding element type (line 392)
        # Getting the type of 'record' (line 392)
        record_709231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 75), 'record', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 392, 21), tuple_709220, record_709231)
        
        # Processing the call keyword arguments (line 391)
        kwargs_709232 = {}
        # Getting the type of 'self' (line 391)
        self_709217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 16), 'self', False)
        # Obtaining the member '_tmp_suppressions' of a type (line 391)
        _tmp_suppressions_709218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 16), self_709217, '_tmp_suppressions')
        # Obtaining the member 'append' of a type (line 391)
        append_709219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 16), _tmp_suppressions_709218, 'append')
        # Calling append(args, kwargs) (line 391)
        append_call_result_709233 = invoke(stypy.reporting.localization.Localization(__file__, 391, 16), append_709219, *[tuple_709220], **kwargs_709232)
        
        # SSA branch for the else part of an if statement (line 379)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 394)
        # Processing the call arguments (line 394)
        
        # Obtaining an instance of the builtin type 'tuple' (line 395)
        tuple_709237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 395)
        # Adding element type (line 395)
        # Getting the type of 'category' (line 395)
        category_709238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 21), 'category', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 21), tuple_709237, category_709238)
        # Adding element type (line 395)
        # Getting the type of 'message' (line 395)
        message_709239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 31), 'message', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 21), tuple_709237, message_709239)
        # Adding element type (line 395)
        
        # Call to compile(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'message' (line 395)
        message_709242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 51), 'message', False)
        # Getting the type of 're' (line 395)
        re_709243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 60), 're', False)
        # Obtaining the member 'I' of a type (line 395)
        I_709244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 60), re_709243, 'I')
        # Processing the call keyword arguments (line 395)
        kwargs_709245 = {}
        # Getting the type of 're' (line 395)
        re_709240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 40), 're', False)
        # Obtaining the member 'compile' of a type (line 395)
        compile_709241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 40), re_709240, 'compile')
        # Calling compile(args, kwargs) (line 395)
        compile_call_result_709246 = invoke(stypy.reporting.localization.Localization(__file__, 395, 40), compile_709241, *[message_709242, I_709244], **kwargs_709245)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 21), tuple_709237, compile_call_result_709246)
        # Adding element type (line 395)
        # Getting the type of 'module' (line 395)
        module_709247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 67), 'module', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 21), tuple_709237, module_709247)
        # Adding element type (line 395)
        # Getting the type of 'record' (line 395)
        record_709248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 75), 'record', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 21), tuple_709237, record_709248)
        
        # Processing the call keyword arguments (line 394)
        kwargs_709249 = {}
        # Getting the type of 'self' (line 394)
        self_709234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 16), 'self', False)
        # Obtaining the member '_suppressions' of a type (line 394)
        _suppressions_709235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 16), self_709234, '_suppressions')
        # Obtaining the member 'append' of a type (line 394)
        append_709236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 16), _suppressions_709235, 'append')
        # Calling append(args, kwargs) (line 394)
        append_call_result_709250 = invoke(stypy.reporting.localization.Localization(__file__, 394, 16), append_709236, *[tuple_709237], **kwargs_709249)
        
        # SSA join for if statement (line 379)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'record' (line 397)
        record_709251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 'record')
        # Assigning a type to the variable 'stypy_return_type' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'stypy_return_type', record_709251)
        
        # ################# End of '_filter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_filter' in the type store
        # Getting the type of 'stypy_return_type' (line 374)
        stypy_return_type_709252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709252)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_filter'
        return stypy_return_type_709252


    @norecursion
    def filter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'Warning' (line 399)
        Warning_709253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 34), 'Warning')
        str_709254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 51), 'str', '')
        # Getting the type of 'None' (line 399)
        None_709255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 62), 'None')
        defaults = [Warning_709253, str_709254, None_709255]
        # Create a new context for function 'filter'
        module_type_store = module_type_store.open_function_context('filter', 399, 8, False)
        # Assigning a type to the variable 'self' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        suppress_warnings.filter.__dict__.__setitem__('stypy_localization', localization)
        suppress_warnings.filter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        suppress_warnings.filter.__dict__.__setitem__('stypy_type_store', module_type_store)
        suppress_warnings.filter.__dict__.__setitem__('stypy_function_name', 'suppress_warnings.filter')
        suppress_warnings.filter.__dict__.__setitem__('stypy_param_names_list', ['category', 'message', 'module'])
        suppress_warnings.filter.__dict__.__setitem__('stypy_varargs_param_name', None)
        suppress_warnings.filter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        suppress_warnings.filter.__dict__.__setitem__('stypy_call_defaults', defaults)
        suppress_warnings.filter.__dict__.__setitem__('stypy_call_varargs', varargs)
        suppress_warnings.filter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        suppress_warnings.filter.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'suppress_warnings.filter', ['category', 'message', 'module'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'filter', localization, ['category', 'message', 'module'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'filter(...)' code ##################

        str_709256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, (-1)), 'str', '\n            Add a new suppressing filter or apply it if the state is entered.\n\n            Parameters\n            ----------\n            category : class, optional\n                Warning class to filter\n            message : string, optional\n                Regular expression matching the warning message.\n            module : module, optional\n                Module to filter for. Note that the module (and its file)\n                must match exactly and cannot be a submodule. This may make\n                it unreliable for external modules.\n\n            Notes\n            -----\n            When added within a context, filters are only added inside\n            the context and will be forgotten when the context is exited.\n            ')
        
        # Call to _filter(...): (line 419)
        # Processing the call keyword arguments (line 419)
        # Getting the type of 'category' (line 419)
        category_709259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 34), 'category', False)
        keyword_709260 = category_709259
        # Getting the type of 'message' (line 419)
        message_709261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 52), 'message', False)
        keyword_709262 = message_709261
        # Getting the type of 'module' (line 419)
        module_709263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 68), 'module', False)
        keyword_709264 = module_709263
        # Getting the type of 'False' (line 420)
        False_709265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 32), 'False', False)
        keyword_709266 = False_709265
        kwargs_709267 = {'category': keyword_709260, 'record': keyword_709266, 'message': keyword_709262, 'module': keyword_709264}
        # Getting the type of 'self' (line 419)
        self_709257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'self', False)
        # Obtaining the member '_filter' of a type (line 419)
        _filter_709258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 12), self_709257, '_filter')
        # Calling _filter(args, kwargs) (line 419)
        _filter_call_result_709268 = invoke(stypy.reporting.localization.Localization(__file__, 419, 12), _filter_709258, *[], **kwargs_709267)
        
        
        # ################# End of 'filter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'filter' in the type store
        # Getting the type of 'stypy_return_type' (line 399)
        stypy_return_type_709269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709269)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'filter'
        return stypy_return_type_709269


    @norecursion
    def record(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'Warning' (line 422)
        Warning_709270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 34), 'Warning')
        str_709271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 422, 51), 'str', '')
        # Getting the type of 'None' (line 422)
        None_709272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 62), 'None')
        defaults = [Warning_709270, str_709271, None_709272]
        # Create a new context for function 'record'
        module_type_store = module_type_store.open_function_context('record', 422, 8, False)
        # Assigning a type to the variable 'self' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        suppress_warnings.record.__dict__.__setitem__('stypy_localization', localization)
        suppress_warnings.record.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        suppress_warnings.record.__dict__.__setitem__('stypy_type_store', module_type_store)
        suppress_warnings.record.__dict__.__setitem__('stypy_function_name', 'suppress_warnings.record')
        suppress_warnings.record.__dict__.__setitem__('stypy_param_names_list', ['category', 'message', 'module'])
        suppress_warnings.record.__dict__.__setitem__('stypy_varargs_param_name', None)
        suppress_warnings.record.__dict__.__setitem__('stypy_kwargs_param_name', None)
        suppress_warnings.record.__dict__.__setitem__('stypy_call_defaults', defaults)
        suppress_warnings.record.__dict__.__setitem__('stypy_call_varargs', varargs)
        suppress_warnings.record.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        suppress_warnings.record.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'suppress_warnings.record', ['category', 'message', 'module'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'record', localization, ['category', 'message', 'module'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'record(...)' code ##################

        str_709273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 448, (-1)), 'str', '\n            Append a new recording filter or apply it if the state is entered.\n\n            All warnings matching will be appended to the ``log`` attribute.\n\n            Parameters\n            ----------\n            category : class, optional\n                Warning class to filter\n            message : string, optional\n                Regular expression matching the warning message.\n            module : module, optional\n                Module to filter for. Note that the module (and its file)\n                must match exactly and cannot be a submodule. This may make\n                it unreliable for external modules.\n\n            Returns\n            -------\n            log : list\n                A list which will be filled with all matched warnings.\n\n            Notes\n            -----\n            When added within a context, filters are only added inside\n            the context and will be forgotten when the context is exited.\n            ')
        
        # Call to _filter(...): (line 449)
        # Processing the call keyword arguments (line 449)
        # Getting the type of 'category' (line 449)
        category_709276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 41), 'category', False)
        keyword_709277 = category_709276
        # Getting the type of 'message' (line 449)
        message_709278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 59), 'message', False)
        keyword_709279 = message_709278
        # Getting the type of 'module' (line 449)
        module_709280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 75), 'module', False)
        keyword_709281 = module_709280
        # Getting the type of 'True' (line 450)
        True_709282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 39), 'True', False)
        keyword_709283 = True_709282
        kwargs_709284 = {'category': keyword_709277, 'record': keyword_709283, 'message': keyword_709279, 'module': keyword_709281}
        # Getting the type of 'self' (line 449)
        self_709274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 19), 'self', False)
        # Obtaining the member '_filter' of a type (line 449)
        _filter_709275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 449, 19), self_709274, '_filter')
        # Calling _filter(args, kwargs) (line 449)
        _filter_call_result_709285 = invoke(stypy.reporting.localization.Localization(__file__, 449, 19), _filter_709275, *[], **kwargs_709284)
        
        # Assigning a type to the variable 'stypy_return_type' (line 449)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 12), 'stypy_return_type', _filter_call_result_709285)
        
        # ################# End of 'record(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'record' in the type store
        # Getting the type of 'stypy_return_type' (line 422)
        stypy_return_type_709286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709286)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'record'
        return stypy_return_type_709286


    @norecursion
    def __enter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__enter__'
        module_type_store = module_type_store.open_function_context('__enter__', 452, 8, False)
        # Assigning a type to the variable 'self' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        suppress_warnings.__enter__.__dict__.__setitem__('stypy_localization', localization)
        suppress_warnings.__enter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        suppress_warnings.__enter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        suppress_warnings.__enter__.__dict__.__setitem__('stypy_function_name', 'suppress_warnings.__enter__')
        suppress_warnings.__enter__.__dict__.__setitem__('stypy_param_names_list', [])
        suppress_warnings.__enter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        suppress_warnings.__enter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        suppress_warnings.__enter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        suppress_warnings.__enter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        suppress_warnings.__enter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        suppress_warnings.__enter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'suppress_warnings.__enter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__enter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__enter__(...)' code ##################

        
        # Getting the type of 'self' (line 453)
        self_709287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 15), 'self')
        # Obtaining the member '_entered' of a type (line 453)
        _entered_709288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 15), self_709287, '_entered')
        # Testing the type of an if condition (line 453)
        if_condition_709289 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 453, 12), _entered_709288)
        # Assigning a type to the variable 'if_condition_709289' (line 453)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 453, 12), 'if_condition_709289', if_condition_709289)
        # SSA begins for if statement (line 453)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 454)
        # Processing the call arguments (line 454)
        str_709291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 35), 'str', 'cannot enter suppress_warnings twice.')
        # Processing the call keyword arguments (line 454)
        kwargs_709292 = {}
        # Getting the type of 'RuntimeError' (line 454)
        RuntimeError_709290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 22), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 454)
        RuntimeError_call_result_709293 = invoke(stypy.reporting.localization.Localization(__file__, 454, 22), RuntimeError_709290, *[str_709291], **kwargs_709292)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 454, 16), RuntimeError_call_result_709293, 'raise parameter', BaseException)
        # SSA join for if statement (line 453)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 456):
        # Getting the type of 'warnings' (line 456)
        warnings_709294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 30), 'warnings')
        # Obtaining the member 'showwarning' of a type (line 456)
        showwarning_709295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 30), warnings_709294, 'showwarning')
        # Getting the type of 'self' (line 456)
        self_709296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'self')
        # Setting the type of the member '_orig_show' of a type (line 456)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 12), self_709296, '_orig_show', showwarning_709295)
        
        # Assigning a Attribute to a Attribute (line 457):
        # Getting the type of 'warnings' (line 457)
        warnings_709297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 28), 'warnings')
        # Obtaining the member 'filters' of a type (line 457)
        filters_709298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 28), warnings_709297, 'filters')
        # Getting the type of 'self' (line 457)
        self_709299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'self')
        # Setting the type of the member '_filters' of a type (line 457)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 12), self_709299, '_filters', filters_709298)
        
        # Assigning a Subscript to a Attribute (line 458):
        
        # Obtaining the type of the subscript
        slice_709300 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 458, 31), None, None, None)
        # Getting the type of 'self' (line 458)
        self_709301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 31), 'self')
        # Obtaining the member '_filters' of a type (line 458)
        _filters_709302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 31), self_709301, '_filters')
        # Obtaining the member '__getitem__' of a type (line 458)
        getitem___709303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 31), _filters_709302, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 458)
        subscript_call_result_709304 = invoke(stypy.reporting.localization.Localization(__file__, 458, 31), getitem___709303, slice_709300)
        
        # Getting the type of 'warnings' (line 458)
        warnings_709305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'warnings')
        # Setting the type of the member 'filters' of a type (line 458)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 12), warnings_709305, 'filters', subscript_call_result_709304)
        
        # Assigning a Name to a Attribute (line 460):
        # Getting the type of 'True' (line 460)
        True_709306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 28), 'True')
        # Getting the type of 'self' (line 460)
        self_709307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'self')
        # Setting the type of the member '_entered' of a type (line 460)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 12), self_709307, '_entered', True_709306)
        
        # Assigning a List to a Attribute (line 461):
        
        # Obtaining an instance of the builtin type 'list' (line 461)
        list_709308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 461)
        
        # Getting the type of 'self' (line 461)
        self_709309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'self')
        # Setting the type of the member '_tmp_suppressions' of a type (line 461)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 12), self_709309, '_tmp_suppressions', list_709308)
        
        # Assigning a Call to a Attribute (line 462):
        
        # Call to set(...): (line 462)
        # Processing the call keyword arguments (line 462)
        kwargs_709311 = {}
        # Getting the type of 'set' (line 462)
        set_709310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 32), 'set', False)
        # Calling set(args, kwargs) (line 462)
        set_call_result_709312 = invoke(stypy.reporting.localization.Localization(__file__, 462, 32), set_709310, *[], **kwargs_709311)
        
        # Getting the type of 'self' (line 462)
        self_709313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'self')
        # Setting the type of the member '_tmp_modules' of a type (line 462)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 12), self_709313, '_tmp_modules', set_call_result_709312)
        
        # Assigning a Call to a Attribute (line 463):
        
        # Call to set(...): (line 463)
        # Processing the call keyword arguments (line 463)
        kwargs_709315 = {}
        # Getting the type of 'set' (line 463)
        set_709314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 30), 'set', False)
        # Calling set(args, kwargs) (line 463)
        set_call_result_709316 = invoke(stypy.reporting.localization.Localization(__file__, 463, 30), set_709314, *[], **kwargs_709315)
        
        # Getting the type of 'self' (line 463)
        self_709317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 463, 12), 'self')
        # Setting the type of the member '_forwarded' of a type (line 463)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 463, 12), self_709317, '_forwarded', set_call_result_709316)
        
        # Assigning a List to a Attribute (line 465):
        
        # Obtaining an instance of the builtin type 'list' (line 465)
        list_709318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 465)
        
        # Getting the type of 'self' (line 465)
        self_709319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 12), 'self')
        # Setting the type of the member 'log' of a type (line 465)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 12), self_709319, 'log', list_709318)
        
        # Getting the type of 'self' (line 467)
        self_709320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 42), 'self')
        # Obtaining the member '_suppressions' of a type (line 467)
        _suppressions_709321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 42), self_709320, '_suppressions')
        # Testing the type of a for loop iterable (line 467)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 467, 12), _suppressions_709321)
        # Getting the type of the for loop variable (line 467)
        for_loop_var_709322 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 467, 12), _suppressions_709321)
        # Assigning a type to the variable 'cat' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'cat', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 12), for_loop_var_709322))
        # Assigning a type to the variable 'mess' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'mess', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 12), for_loop_var_709322))
        # Assigning a type to the variable '_' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), '_', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 12), for_loop_var_709322))
        # Assigning a type to the variable 'mod' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'mod', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 12), for_loop_var_709322))
        # Assigning a type to the variable 'log' (line 467)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 12), 'log', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 467, 12), for_loop_var_709322))
        # SSA begins for a for statement (line 467)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 468)
        # Getting the type of 'log' (line 468)
        log_709323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 16), 'log')
        # Getting the type of 'None' (line 468)
        None_709324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 30), 'None')
        
        (may_be_709325, more_types_in_union_709326) = may_not_be_none(log_709323, None_709324)

        if may_be_709325:

            if more_types_in_union_709326:
                # Runtime conditional SSA (line 468)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Deleting a member
            # Getting the type of 'log' (line 469)
            log_709327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 24), 'log')
            
            # Obtaining the type of the subscript
            slice_709328 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 469, 24), None, None, None)
            # Getting the type of 'log' (line 469)
            log_709329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 24), 'log')
            # Obtaining the member '__getitem__' of a type (line 469)
            getitem___709330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 24), log_709329, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 469)
            subscript_call_result_709331 = invoke(stypy.reporting.localization.Localization(__file__, 469, 24), getitem___709330, slice_709328)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 469, 20), log_709327, subscript_call_result_709331)

            if more_types_in_union_709326:
                # SSA join for if statement (line 468)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 470)
        # Getting the type of 'mod' (line 470)
        mod_709332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 19), 'mod')
        # Getting the type of 'None' (line 470)
        None_709333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 26), 'None')
        
        (may_be_709334, more_types_in_union_709335) = may_be_none(mod_709332, None_709333)

        if may_be_709334:

            if more_types_in_union_709335:
                # Runtime conditional SSA (line 470)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to filterwarnings(...): (line 471)
            # Processing the call arguments (line 471)
            str_709338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 24), 'str', 'always')
            # Processing the call keyword arguments (line 471)
            # Getting the type of 'cat' (line 472)
            cat_709339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 43), 'cat', False)
            keyword_709340 = cat_709339
            # Getting the type of 'mess' (line 472)
            mess_709341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 56), 'mess', False)
            keyword_709342 = mess_709341
            kwargs_709343 = {'category': keyword_709340, 'message': keyword_709342}
            # Getting the type of 'warnings' (line 471)
            warnings_709336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 20), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 471)
            filterwarnings_709337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 20), warnings_709336, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 471)
            filterwarnings_call_result_709344 = invoke(stypy.reporting.localization.Localization(__file__, 471, 20), filterwarnings_709337, *[str_709338], **kwargs_709343)
            

            if more_types_in_union_709335:
                # Runtime conditional SSA for else branch (line 470)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_709334) or more_types_in_union_709335):
            
            # Assigning a BinOp to a Name (line 474):
            
            # Call to replace(...): (line 474)
            # Processing the call arguments (line 474)
            str_709348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 56), 'str', '.')
            str_709349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 61), 'str', '\\.')
            # Processing the call keyword arguments (line 474)
            kwargs_709350 = {}
            # Getting the type of 'mod' (line 474)
            mod_709345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 35), 'mod', False)
            # Obtaining the member '__name__' of a type (line 474)
            name___709346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 35), mod_709345, '__name__')
            # Obtaining the member 'replace' of a type (line 474)
            replace_709347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 35), name___709346, 'replace')
            # Calling replace(args, kwargs) (line 474)
            replace_call_result_709351 = invoke(stypy.reporting.localization.Localization(__file__, 474, 35), replace_709347, *[str_709348, str_709349], **kwargs_709350)
            
            str_709352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 70), 'str', '$')
            # Applying the binary operator '+' (line 474)
            result_add_709353 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 35), '+', replace_call_result_709351, str_709352)
            
            # Assigning a type to the variable 'module_regex' (line 474)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 20), 'module_regex', result_add_709353)
            
            # Call to filterwarnings(...): (line 475)
            # Processing the call arguments (line 475)
            str_709356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 476, 24), 'str', 'always')
            # Processing the call keyword arguments (line 475)
            # Getting the type of 'cat' (line 476)
            cat_709357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 43), 'cat', False)
            keyword_709358 = cat_709357
            # Getting the type of 'mess' (line 476)
            mess_709359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 56), 'mess', False)
            keyword_709360 = mess_709359
            # Getting the type of 'module_regex' (line 477)
            module_regex_709361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 31), 'module_regex', False)
            keyword_709362 = module_regex_709361
            kwargs_709363 = {'category': keyword_709358, 'message': keyword_709360, 'module': keyword_709362}
            # Getting the type of 'warnings' (line 475)
            warnings_709354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 20), 'warnings', False)
            # Obtaining the member 'filterwarnings' of a type (line 475)
            filterwarnings_709355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 20), warnings_709354, 'filterwarnings')
            # Calling filterwarnings(args, kwargs) (line 475)
            filterwarnings_call_result_709364 = invoke(stypy.reporting.localization.Localization(__file__, 475, 20), filterwarnings_709355, *[str_709356], **kwargs_709363)
            
            
            # Call to add(...): (line 478)
            # Processing the call arguments (line 478)
            # Getting the type of 'mod' (line 478)
            mod_709368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 42), 'mod', False)
            # Processing the call keyword arguments (line 478)
            kwargs_709369 = {}
            # Getting the type of 'self' (line 478)
            self_709365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 20), 'self', False)
            # Obtaining the member '_tmp_modules' of a type (line 478)
            _tmp_modules_709366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 20), self_709365, '_tmp_modules')
            # Obtaining the member 'add' of a type (line 478)
            add_709367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 478, 20), _tmp_modules_709366, 'add')
            # Calling add(args, kwargs) (line 478)
            add_call_result_709370 = invoke(stypy.reporting.localization.Localization(__file__, 478, 20), add_709367, *[mod_709368], **kwargs_709369)
            

            if (may_be_709334 and more_types_in_union_709335):
                # SSA join for if statement (line 470)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 479):
        # Getting the type of 'self' (line 479)
        self_709371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 35), 'self')
        # Obtaining the member '_showwarning' of a type (line 479)
        _showwarning_709372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 35), self_709371, '_showwarning')
        # Getting the type of 'warnings' (line 479)
        warnings_709373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'warnings')
        # Setting the type of the member 'showwarning' of a type (line 479)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 12), warnings_709373, 'showwarning', _showwarning_709372)
        
        # Call to _clear_registries(...): (line 480)
        # Processing the call keyword arguments (line 480)
        kwargs_709376 = {}
        # Getting the type of 'self' (line 480)
        self_709374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 12), 'self', False)
        # Obtaining the member '_clear_registries' of a type (line 480)
        _clear_registries_709375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 12), self_709374, '_clear_registries')
        # Calling _clear_registries(args, kwargs) (line 480)
        _clear_registries_call_result_709377 = invoke(stypy.reporting.localization.Localization(__file__, 480, 12), _clear_registries_709375, *[], **kwargs_709376)
        
        # Getting the type of 'self' (line 482)
        self_709378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 482)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 12), 'stypy_return_type', self_709378)
        
        # ################# End of '__enter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__enter__' in the type store
        # Getting the type of 'stypy_return_type' (line 452)
        stypy_return_type_709379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709379)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__enter__'
        return stypy_return_type_709379


    @norecursion
    def __exit__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__exit__'
        module_type_store = module_type_store.open_function_context('__exit__', 484, 8, False)
        # Assigning a type to the variable 'self' (line 485)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        suppress_warnings.__exit__.__dict__.__setitem__('stypy_localization', localization)
        suppress_warnings.__exit__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        suppress_warnings.__exit__.__dict__.__setitem__('stypy_type_store', module_type_store)
        suppress_warnings.__exit__.__dict__.__setitem__('stypy_function_name', 'suppress_warnings.__exit__')
        suppress_warnings.__exit__.__dict__.__setitem__('stypy_param_names_list', [])
        suppress_warnings.__exit__.__dict__.__setitem__('stypy_varargs_param_name', 'exc_info')
        suppress_warnings.__exit__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        suppress_warnings.__exit__.__dict__.__setitem__('stypy_call_defaults', defaults)
        suppress_warnings.__exit__.__dict__.__setitem__('stypy_call_varargs', varargs)
        suppress_warnings.__exit__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        suppress_warnings.__exit__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'suppress_warnings.__exit__', [], 'exc_info', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__exit__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__exit__(...)' code ##################

        
        # Assigning a Attribute to a Attribute (line 485):
        # Getting the type of 'self' (line 485)
        self_709380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 35), 'self')
        # Obtaining the member '_orig_show' of a type (line 485)
        _orig_show_709381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 35), self_709380, '_orig_show')
        # Getting the type of 'warnings' (line 485)
        warnings_709382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 12), 'warnings')
        # Setting the type of the member 'showwarning' of a type (line 485)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 485, 12), warnings_709382, 'showwarning', _orig_show_709381)
        
        # Assigning a Attribute to a Attribute (line 486):
        # Getting the type of 'self' (line 486)
        self_709383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 31), 'self')
        # Obtaining the member '_filters' of a type (line 486)
        _filters_709384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 31), self_709383, '_filters')
        # Getting the type of 'warnings' (line 486)
        warnings_709385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 12), 'warnings')
        # Setting the type of the member 'filters' of a type (line 486)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 486, 12), warnings_709385, 'filters', _filters_709384)
        
        # Call to _clear_registries(...): (line 487)
        # Processing the call keyword arguments (line 487)
        kwargs_709388 = {}
        # Getting the type of 'self' (line 487)
        self_709386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 12), 'self', False)
        # Obtaining the member '_clear_registries' of a type (line 487)
        _clear_registries_709387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 12), self_709386, '_clear_registries')
        # Calling _clear_registries(args, kwargs) (line 487)
        _clear_registries_call_result_709389 = invoke(stypy.reporting.localization.Localization(__file__, 487, 12), _clear_registries_709387, *[], **kwargs_709388)
        
        
        # Assigning a Name to a Attribute (line 488):
        # Getting the type of 'False' (line 488)
        False_709390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 28), 'False')
        # Getting the type of 'self' (line 488)
        self_709391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 12), 'self')
        # Setting the type of the member '_entered' of a type (line 488)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 12), self_709391, '_entered', False_709390)
        # Deleting a member
        # Getting the type of 'self' (line 489)
        self_709392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'self')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 489, 12), self_709392, '_orig_show')
        # Deleting a member
        # Getting the type of 'self' (line 490)
        self_709393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'self')
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 490, 12), self_709393, '_filters')
        
        # ################# End of '__exit__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__exit__' in the type store
        # Getting the type of 'stypy_return_type' (line 484)
        stypy_return_type_709394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 484, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709394)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__exit__'
        return stypy_return_type_709394


    @norecursion
    def _showwarning(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_showwarning'
        module_type_store = module_type_store.open_function_context('_showwarning', 492, 8, False)
        # Assigning a type to the variable 'self' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        suppress_warnings._showwarning.__dict__.__setitem__('stypy_localization', localization)
        suppress_warnings._showwarning.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        suppress_warnings._showwarning.__dict__.__setitem__('stypy_type_store', module_type_store)
        suppress_warnings._showwarning.__dict__.__setitem__('stypy_function_name', 'suppress_warnings._showwarning')
        suppress_warnings._showwarning.__dict__.__setitem__('stypy_param_names_list', ['message', 'category', 'filename', 'lineno'])
        suppress_warnings._showwarning.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        suppress_warnings._showwarning.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        suppress_warnings._showwarning.__dict__.__setitem__('stypy_call_defaults', defaults)
        suppress_warnings._showwarning.__dict__.__setitem__('stypy_call_varargs', varargs)
        suppress_warnings._showwarning.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        suppress_warnings._showwarning.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'suppress_warnings._showwarning', ['message', 'category', 'filename', 'lineno'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_showwarning', localization, ['message', 'category', 'filename', 'lineno'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_showwarning(...)' code ##################

        
        # Assigning a Call to a Name (line 494):
        
        # Call to pop(...): (line 494)
        # Processing the call arguments (line 494)
        str_709397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 37), 'str', 'use_warnmsg')
        # Getting the type of 'None' (line 494)
        None_709398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 52), 'None', False)
        # Processing the call keyword arguments (line 494)
        kwargs_709399 = {}
        # Getting the type of 'kwargs' (line 494)
        kwargs_709395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 26), 'kwargs', False)
        # Obtaining the member 'pop' of a type (line 494)
        pop_709396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 26), kwargs_709395, 'pop')
        # Calling pop(args, kwargs) (line 494)
        pop_call_result_709400 = invoke(stypy.reporting.localization.Localization(__file__, 494, 26), pop_709396, *[str_709397, None_709398], **kwargs_709399)
        
        # Assigning a type to the variable 'use_warnmsg' (line 494)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 494, 12), 'use_warnmsg', pop_call_result_709400)
        
        
        # Obtaining the type of the subscript
        int_709401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 67), 'int')
        slice_709402 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 496, 20), None, None, int_709401)
        # Getting the type of 'self' (line 496)
        self_709403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 20), 'self')
        # Obtaining the member '_suppressions' of a type (line 496)
        _suppressions_709404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 20), self_709403, '_suppressions')
        # Getting the type of 'self' (line 496)
        self_709405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 41), 'self')
        # Obtaining the member '_tmp_suppressions' of a type (line 496)
        _tmp_suppressions_709406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 41), self_709405, '_tmp_suppressions')
        # Applying the binary operator '+' (line 496)
        result_add_709407 = python_operator(stypy.reporting.localization.Localization(__file__, 496, 20), '+', _suppressions_709404, _tmp_suppressions_709406)
        
        # Obtaining the member '__getitem__' of a type (line 496)
        getitem___709408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 496, 20), result_add_709407, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 496)
        subscript_call_result_709409 = invoke(stypy.reporting.localization.Localization(__file__, 496, 20), getitem___709408, slice_709402)
        
        # Testing the type of a for loop iterable (line 495)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 495, 12), subscript_call_result_709409)
        # Getting the type of the for loop variable (line 495)
        for_loop_var_709410 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 495, 12), subscript_call_result_709409)
        # Assigning a type to the variable 'cat' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'cat', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 12), for_loop_var_709410))
        # Assigning a type to the variable '_' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), '_', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 12), for_loop_var_709410))
        # Assigning a type to the variable 'pattern' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'pattern', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 12), for_loop_var_709410))
        # Assigning a type to the variable 'mod' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'mod', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 12), for_loop_var_709410))
        # Assigning a type to the variable 'rec' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 12), 'rec', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 495, 12), for_loop_var_709410))
        # SSA begins for a for statement (line 495)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        # Call to issubclass(...): (line 497)
        # Processing the call arguments (line 497)
        # Getting the type of 'category' (line 497)
        category_709412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 31), 'category', False)
        # Getting the type of 'cat' (line 497)
        cat_709413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 41), 'cat', False)
        # Processing the call keyword arguments (line 497)
        kwargs_709414 = {}
        # Getting the type of 'issubclass' (line 497)
        issubclass_709411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 20), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 497)
        issubclass_call_result_709415 = invoke(stypy.reporting.localization.Localization(__file__, 497, 20), issubclass_709411, *[category_709412, cat_709413], **kwargs_709414)
        
        
        
        # Call to match(...): (line 498)
        # Processing the call arguments (line 498)
        
        # Obtaining the type of the subscript
        int_709418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 51), 'int')
        # Getting the type of 'message' (line 498)
        message_709419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 38), 'message', False)
        # Obtaining the member 'args' of a type (line 498)
        args_709420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 38), message_709419, 'args')
        # Obtaining the member '__getitem__' of a type (line 498)
        getitem___709421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 38), args_709420, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 498)
        subscript_call_result_709422 = invoke(stypy.reporting.localization.Localization(__file__, 498, 38), getitem___709421, int_709418)
        
        # Processing the call keyword arguments (line 498)
        kwargs_709423 = {}
        # Getting the type of 'pattern' (line 498)
        pattern_709416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 24), 'pattern', False)
        # Obtaining the member 'match' of a type (line 498)
        match_709417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 24), pattern_709416, 'match')
        # Calling match(args, kwargs) (line 498)
        match_call_result_709424 = invoke(stypy.reporting.localization.Localization(__file__, 498, 24), match_709417, *[subscript_call_result_709422], **kwargs_709423)
        
        # Getting the type of 'None' (line 498)
        None_709425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 62), 'None')
        # Applying the binary operator 'isnot' (line 498)
        result_is_not_709426 = python_operator(stypy.reporting.localization.Localization(__file__, 498, 24), 'isnot', match_call_result_709424, None_709425)
        
        # Applying the binary operator 'and' (line 497)
        result_and_keyword_709427 = python_operator(stypy.reporting.localization.Localization(__file__, 497, 20), 'and', issubclass_call_result_709415, result_is_not_709426)
        
        # Testing the type of an if condition (line 497)
        if_condition_709428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 497, 16), result_and_keyword_709427)
        # Assigning a type to the variable 'if_condition_709428' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 16), 'if_condition_709428', if_condition_709428)
        # SSA begins for if statement (line 497)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 499)
        # Getting the type of 'mod' (line 499)
        mod_709429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 23), 'mod')
        # Getting the type of 'None' (line 499)
        None_709430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 30), 'None')
        
        (may_be_709431, more_types_in_union_709432) = may_be_none(mod_709429, None_709430)

        if may_be_709431:

            if more_types_in_union_709432:
                # Runtime conditional SSA (line 499)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 501)
            # Getting the type of 'rec' (line 501)
            rec_709433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 24), 'rec')
            # Getting the type of 'None' (line 501)
            None_709434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 38), 'None')
            
            (may_be_709435, more_types_in_union_709436) = may_not_be_none(rec_709433, None_709434)

            if may_be_709435:

                if more_types_in_union_709436:
                    # Runtime conditional SSA (line 501)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Name (line 502):
                
                # Call to WarningMessage(...): (line 502)
                # Processing the call arguments (line 502)
                # Getting the type of 'message' (line 502)
                message_709438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 49), 'message', False)
                # Getting the type of 'category' (line 502)
                category_709439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 58), 'category', False)
                # Getting the type of 'filename' (line 502)
                filename_709440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 68), 'filename', False)
                # Getting the type of 'lineno' (line 503)
                lineno_709441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 49), 'lineno', False)
                # Processing the call keyword arguments (line 502)
                # Getting the type of 'kwargs' (line 503)
                kwargs_709442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 59), 'kwargs', False)
                kwargs_709443 = {'kwargs_709442': kwargs_709442}
                # Getting the type of 'WarningMessage' (line 502)
                WarningMessage_709437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 34), 'WarningMessage', False)
                # Calling WarningMessage(args, kwargs) (line 502)
                WarningMessage_call_result_709444 = invoke(stypy.reporting.localization.Localization(__file__, 502, 34), WarningMessage_709437, *[message_709438, category_709439, filename_709440, lineno_709441], **kwargs_709443)
                
                # Assigning a type to the variable 'msg' (line 502)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 28), 'msg', WarningMessage_call_result_709444)
                
                # Call to append(...): (line 504)
                # Processing the call arguments (line 504)
                # Getting the type of 'msg' (line 504)
                msg_709448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 44), 'msg', False)
                # Processing the call keyword arguments (line 504)
                kwargs_709449 = {}
                # Getting the type of 'self' (line 504)
                self_709445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 28), 'self', False)
                # Obtaining the member 'log' of a type (line 504)
                log_709446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 28), self_709445, 'log')
                # Obtaining the member 'append' of a type (line 504)
                append_709447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 28), log_709446, 'append')
                # Calling append(args, kwargs) (line 504)
                append_call_result_709450 = invoke(stypy.reporting.localization.Localization(__file__, 504, 28), append_709447, *[msg_709448], **kwargs_709449)
                
                
                # Call to append(...): (line 505)
                # Processing the call arguments (line 505)
                # Getting the type of 'msg' (line 505)
                msg_709453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 39), 'msg', False)
                # Processing the call keyword arguments (line 505)
                kwargs_709454 = {}
                # Getting the type of 'rec' (line 505)
                rec_709451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 28), 'rec', False)
                # Obtaining the member 'append' of a type (line 505)
                append_709452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 28), rec_709451, 'append')
                # Calling append(args, kwargs) (line 505)
                append_call_result_709455 = invoke(stypy.reporting.localization.Localization(__file__, 505, 28), append_709452, *[msg_709453], **kwargs_709454)
                

                if more_types_in_union_709436:
                    # SSA join for if statement (line 501)
                    module_type_store = module_type_store.join_ssa_context()


            
            # Assigning a type to the variable 'stypy_return_type' (line 506)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 24), 'stypy_return_type', types.NoneType)

            if more_types_in_union_709432:
                # Runtime conditional SSA for else branch (line 499)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_709431) or more_types_in_union_709432):
            
            
            # Call to startswith(...): (line 509)
            # Processing the call arguments (line 509)
            # Getting the type of 'filename' (line 509)
            filename_709459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 49), 'filename', False)
            # Processing the call keyword arguments (line 509)
            kwargs_709460 = {}
            # Getting the type of 'mod' (line 509)
            mod_709456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 25), 'mod', False)
            # Obtaining the member '__file__' of a type (line 509)
            file___709457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 25), mod_709456, '__file__')
            # Obtaining the member 'startswith' of a type (line 509)
            startswith_709458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 25), file___709457, 'startswith')
            # Calling startswith(args, kwargs) (line 509)
            startswith_call_result_709461 = invoke(stypy.reporting.localization.Localization(__file__, 509, 25), startswith_709458, *[filename_709459], **kwargs_709460)
            
            # Testing the type of an if condition (line 509)
            if_condition_709462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 509, 25), startswith_call_result_709461)
            # Assigning a type to the variable 'if_condition_709462' (line 509)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 25), 'if_condition_709462', if_condition_709462)
            # SSA begins for if statement (line 509)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Type idiom detected: calculating its left and rigth part (line 511)
            # Getting the type of 'rec' (line 511)
            rec_709463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 24), 'rec')
            # Getting the type of 'None' (line 511)
            None_709464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 38), 'None')
            
            (may_be_709465, more_types_in_union_709466) = may_not_be_none(rec_709463, None_709464)

            if may_be_709465:

                if more_types_in_union_709466:
                    # Runtime conditional SSA (line 511)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                # Assigning a Call to a Name (line 512):
                
                # Call to WarningMessage(...): (line 512)
                # Processing the call arguments (line 512)
                # Getting the type of 'message' (line 512)
                message_709468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 49), 'message', False)
                # Getting the type of 'category' (line 512)
                category_709469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 58), 'category', False)
                # Getting the type of 'filename' (line 512)
                filename_709470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 68), 'filename', False)
                # Getting the type of 'lineno' (line 513)
                lineno_709471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 49), 'lineno', False)
                # Processing the call keyword arguments (line 512)
                # Getting the type of 'kwargs' (line 513)
                kwargs_709472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 59), 'kwargs', False)
                kwargs_709473 = {'kwargs_709472': kwargs_709472}
                # Getting the type of 'WarningMessage' (line 512)
                WarningMessage_709467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 34), 'WarningMessage', False)
                # Calling WarningMessage(args, kwargs) (line 512)
                WarningMessage_call_result_709474 = invoke(stypy.reporting.localization.Localization(__file__, 512, 34), WarningMessage_709467, *[message_709468, category_709469, filename_709470, lineno_709471], **kwargs_709473)
                
                # Assigning a type to the variable 'msg' (line 512)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 512, 28), 'msg', WarningMessage_call_result_709474)
                
                # Call to append(...): (line 514)
                # Processing the call arguments (line 514)
                # Getting the type of 'msg' (line 514)
                msg_709478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 44), 'msg', False)
                # Processing the call keyword arguments (line 514)
                kwargs_709479 = {}
                # Getting the type of 'self' (line 514)
                self_709475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 28), 'self', False)
                # Obtaining the member 'log' of a type (line 514)
                log_709476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 28), self_709475, 'log')
                # Obtaining the member 'append' of a type (line 514)
                append_709477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 28), log_709476, 'append')
                # Calling append(args, kwargs) (line 514)
                append_call_result_709480 = invoke(stypy.reporting.localization.Localization(__file__, 514, 28), append_709477, *[msg_709478], **kwargs_709479)
                
                
                # Call to append(...): (line 515)
                # Processing the call arguments (line 515)
                # Getting the type of 'msg' (line 515)
                msg_709483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 39), 'msg', False)
                # Processing the call keyword arguments (line 515)
                kwargs_709484 = {}
                # Getting the type of 'rec' (line 515)
                rec_709481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 28), 'rec', False)
                # Obtaining the member 'append' of a type (line 515)
                append_709482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 28), rec_709481, 'append')
                # Calling append(args, kwargs) (line 515)
                append_call_result_709485 = invoke(stypy.reporting.localization.Localization(__file__, 515, 28), append_709482, *[msg_709483], **kwargs_709484)
                

                if more_types_in_union_709466:
                    # SSA join for if statement (line 511)
                    module_type_store = module_type_store.join_ssa_context()


            
            # Assigning a type to the variable 'stypy_return_type' (line 516)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 24), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 509)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_709431 and more_types_in_union_709432):
                # SSA join for if statement (line 499)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 497)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 520)
        self_709486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 15), 'self')
        # Obtaining the member '_forwarding_rule' of a type (line 520)
        _forwarding_rule_709487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 15), self_709486, '_forwarding_rule')
        str_709488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 520, 40), 'str', 'always')
        # Applying the binary operator '==' (line 520)
        result_eq_709489 = python_operator(stypy.reporting.localization.Localization(__file__, 520, 15), '==', _forwarding_rule_709487, str_709488)
        
        # Testing the type of an if condition (line 520)
        if_condition_709490 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 520, 12), result_eq_709489)
        # Assigning a type to the variable 'if_condition_709490' (line 520)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 520, 12), 'if_condition_709490', if_condition_709490)
        # SSA begins for if statement (line 520)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Type idiom detected: calculating its left and rigth part (line 521)
        # Getting the type of 'use_warnmsg' (line 521)
        use_warnmsg_709491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 19), 'use_warnmsg')
        # Getting the type of 'None' (line 521)
        None_709492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 34), 'None')
        
        (may_be_709493, more_types_in_union_709494) = may_be_none(use_warnmsg_709491, None_709492)

        if may_be_709493:

            if more_types_in_union_709494:
                # Runtime conditional SSA (line 521)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _orig_show(...): (line 522)
            # Processing the call arguments (line 522)
            # Getting the type of 'message' (line 522)
            message_709497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 36), 'message', False)
            # Getting the type of 'category' (line 522)
            category_709498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 45), 'category', False)
            # Getting the type of 'filename' (line 522)
            filename_709499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 55), 'filename', False)
            # Getting the type of 'lineno' (line 522)
            lineno_709500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 65), 'lineno', False)
            # Getting the type of 'args' (line 523)
            args_709501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 37), 'args', False)
            # Processing the call keyword arguments (line 522)
            # Getting the type of 'kwargs' (line 523)
            kwargs_709502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 45), 'kwargs', False)
            kwargs_709503 = {'kwargs_709502': kwargs_709502}
            # Getting the type of 'self' (line 522)
            self_709495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 20), 'self', False)
            # Obtaining the member '_orig_show' of a type (line 522)
            _orig_show_709496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 20), self_709495, '_orig_show')
            # Calling _orig_show(args, kwargs) (line 522)
            _orig_show_call_result_709504 = invoke(stypy.reporting.localization.Localization(__file__, 522, 20), _orig_show_709496, *[message_709497, category_709498, filename_709499, lineno_709500, args_709501], **kwargs_709503)
            

            if more_types_in_union_709494:
                # Runtime conditional SSA for else branch (line 521)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_709493) or more_types_in_union_709494):
            
            # Call to _orig_showmsg(...): (line 525)
            # Processing the call arguments (line 525)
            # Getting the type of 'use_warnmsg' (line 525)
            use_warnmsg_709507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 39), 'use_warnmsg', False)
            # Processing the call keyword arguments (line 525)
            kwargs_709508 = {}
            # Getting the type of 'self' (line 525)
            self_709505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 20), 'self', False)
            # Obtaining the member '_orig_showmsg' of a type (line 525)
            _orig_showmsg_709506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 525, 20), self_709505, '_orig_showmsg')
            # Calling _orig_showmsg(args, kwargs) (line 525)
            _orig_showmsg_call_result_709509 = invoke(stypy.reporting.localization.Localization(__file__, 525, 20), _orig_showmsg_709506, *[use_warnmsg_709507], **kwargs_709508)
            

            if (may_be_709493 and more_types_in_union_709494):
                # SSA join for if statement (line 521)
                module_type_store = module_type_store.join_ssa_context()


        
        # Assigning a type to the variable 'stypy_return_type' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 16), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 520)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 528)
        self_709510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 15), 'self')
        # Obtaining the member '_forwarding_rule' of a type (line 528)
        _forwarding_rule_709511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 15), self_709510, '_forwarding_rule')
        str_709512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 40), 'str', 'once')
        # Applying the binary operator '==' (line 528)
        result_eq_709513 = python_operator(stypy.reporting.localization.Localization(__file__, 528, 15), '==', _forwarding_rule_709511, str_709512)
        
        # Testing the type of an if condition (line 528)
        if_condition_709514 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 528, 12), result_eq_709513)
        # Assigning a type to the variable 'if_condition_709514' (line 528)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 528, 12), 'if_condition_709514', if_condition_709514)
        # SSA begins for if statement (line 528)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 529):
        
        # Obtaining an instance of the builtin type 'tuple' (line 529)
        tuple_709515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 529, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 529)
        # Adding element type (line 529)
        # Getting the type of 'message' (line 529)
        message_709516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 29), 'message')
        # Obtaining the member 'args' of a type (line 529)
        args_709517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 529, 29), message_709516, 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 29), tuple_709515, args_709517)
        # Adding element type (line 529)
        # Getting the type of 'category' (line 529)
        category_709518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 529, 43), 'category')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 529, 29), tuple_709515, category_709518)
        
        # Assigning a type to the variable 'signature' (line 529)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 529, 16), 'signature', tuple_709515)
        # SSA branch for the else part of an if statement (line 528)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 530)
        self_709519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 17), 'self')
        # Obtaining the member '_forwarding_rule' of a type (line 530)
        _forwarding_rule_709520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 530, 17), self_709519, '_forwarding_rule')
        str_709521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 530, 42), 'str', 'module')
        # Applying the binary operator '==' (line 530)
        result_eq_709522 = python_operator(stypy.reporting.localization.Localization(__file__, 530, 17), '==', _forwarding_rule_709520, str_709521)
        
        # Testing the type of an if condition (line 530)
        if_condition_709523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 530, 17), result_eq_709522)
        # Assigning a type to the variable 'if_condition_709523' (line 530)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 17), 'if_condition_709523', if_condition_709523)
        # SSA begins for if statement (line 530)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 531):
        
        # Obtaining an instance of the builtin type 'tuple' (line 531)
        tuple_709524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 531, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 531)
        # Adding element type (line 531)
        # Getting the type of 'message' (line 531)
        message_709525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 29), 'message')
        # Obtaining the member 'args' of a type (line 531)
        args_709526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 29), message_709525, 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 29), tuple_709524, args_709526)
        # Adding element type (line 531)
        # Getting the type of 'category' (line 531)
        category_709527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 43), 'category')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 29), tuple_709524, category_709527)
        # Adding element type (line 531)
        # Getting the type of 'filename' (line 531)
        filename_709528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 53), 'filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 531, 29), tuple_709524, filename_709528)
        
        # Assigning a type to the variable 'signature' (line 531)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 16), 'signature', tuple_709524)
        # SSA branch for the else part of an if statement (line 530)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 532)
        self_709529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 17), 'self')
        # Obtaining the member '_forwarding_rule' of a type (line 532)
        _forwarding_rule_709530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 17), self_709529, '_forwarding_rule')
        str_709531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 42), 'str', 'location')
        # Applying the binary operator '==' (line 532)
        result_eq_709532 = python_operator(stypy.reporting.localization.Localization(__file__, 532, 17), '==', _forwarding_rule_709530, str_709531)
        
        # Testing the type of an if condition (line 532)
        if_condition_709533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 532, 17), result_eq_709532)
        # Assigning a type to the variable 'if_condition_709533' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 17), 'if_condition_709533', if_condition_709533)
        # SSA begins for if statement (line 532)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Tuple to a Name (line 533):
        
        # Obtaining an instance of the builtin type 'tuple' (line 533)
        tuple_709534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 533)
        # Adding element type (line 533)
        # Getting the type of 'message' (line 533)
        message_709535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 29), 'message')
        # Obtaining the member 'args' of a type (line 533)
        args_709536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 29), message_709535, 'args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 29), tuple_709534, args_709536)
        # Adding element type (line 533)
        # Getting the type of 'category' (line 533)
        category_709537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 43), 'category')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 29), tuple_709534, category_709537)
        # Adding element type (line 533)
        # Getting the type of 'filename' (line 533)
        filename_709538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 53), 'filename')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 29), tuple_709534, filename_709538)
        # Adding element type (line 533)
        # Getting the type of 'lineno' (line 533)
        lineno_709539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 63), 'lineno')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 533, 29), tuple_709534, lineno_709539)
        
        # Assigning a type to the variable 'signature' (line 533)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 16), 'signature', tuple_709534)
        # SSA join for if statement (line 532)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 530)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 528)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'signature' (line 535)
        signature_709540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 15), 'signature')
        # Getting the type of 'self' (line 535)
        self_709541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 535, 28), 'self')
        # Obtaining the member '_forwarded' of a type (line 535)
        _forwarded_709542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 535, 28), self_709541, '_forwarded')
        # Applying the binary operator 'in' (line 535)
        result_contains_709543 = python_operator(stypy.reporting.localization.Localization(__file__, 535, 15), 'in', signature_709540, _forwarded_709542)
        
        # Testing the type of an if condition (line 535)
        if_condition_709544 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 535, 12), result_contains_709543)
        # Assigning a type to the variable 'if_condition_709544' (line 535)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 535, 12), 'if_condition_709544', if_condition_709544)
        # SSA begins for if statement (line 535)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 536)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 535)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to add(...): (line 537)
        # Processing the call arguments (line 537)
        # Getting the type of 'signature' (line 537)
        signature_709548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 32), 'signature', False)
        # Processing the call keyword arguments (line 537)
        kwargs_709549 = {}
        # Getting the type of 'self' (line 537)
        self_709545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'self', False)
        # Obtaining the member '_forwarded' of a type (line 537)
        _forwarded_709546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), self_709545, '_forwarded')
        # Obtaining the member 'add' of a type (line 537)
        add_709547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), _forwarded_709546, 'add')
        # Calling add(args, kwargs) (line 537)
        add_call_result_709550 = invoke(stypy.reporting.localization.Localization(__file__, 537, 12), add_709547, *[signature_709548], **kwargs_709549)
        
        
        # Type idiom detected: calculating its left and rigth part (line 538)
        # Getting the type of 'use_warnmsg' (line 538)
        use_warnmsg_709551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 15), 'use_warnmsg')
        # Getting the type of 'None' (line 538)
        None_709552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 30), 'None')
        
        (may_be_709553, more_types_in_union_709554) = may_be_none(use_warnmsg_709551, None_709552)

        if may_be_709553:

            if more_types_in_union_709554:
                # Runtime conditional SSA (line 538)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to _orig_show(...): (line 539)
            # Processing the call arguments (line 539)
            # Getting the type of 'message' (line 539)
            message_709557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 32), 'message', False)
            # Getting the type of 'category' (line 539)
            category_709558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 41), 'category', False)
            # Getting the type of 'filename' (line 539)
            filename_709559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 51), 'filename', False)
            # Getting the type of 'lineno' (line 539)
            lineno_709560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 61), 'lineno', False)
            # Getting the type of 'args' (line 539)
            args_709561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 70), 'args', False)
            # Processing the call keyword arguments (line 539)
            # Getting the type of 'kwargs' (line 540)
            kwargs_709562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 34), 'kwargs', False)
            kwargs_709563 = {'kwargs_709562': kwargs_709562}
            # Getting the type of 'self' (line 539)
            self_709555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 16), 'self', False)
            # Obtaining the member '_orig_show' of a type (line 539)
            _orig_show_709556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 16), self_709555, '_orig_show')
            # Calling _orig_show(args, kwargs) (line 539)
            _orig_show_call_result_709564 = invoke(stypy.reporting.localization.Localization(__file__, 539, 16), _orig_show_709556, *[message_709557, category_709558, filename_709559, lineno_709560, args_709561], **kwargs_709563)
            

            if more_types_in_union_709554:
                # Runtime conditional SSA for else branch (line 538)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_709553) or more_types_in_union_709554):
            
            # Call to _orig_showmsg(...): (line 542)
            # Processing the call arguments (line 542)
            # Getting the type of 'use_warnmsg' (line 542)
            use_warnmsg_709567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 35), 'use_warnmsg', False)
            # Processing the call keyword arguments (line 542)
            kwargs_709568 = {}
            # Getting the type of 'self' (line 542)
            self_709565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'self', False)
            # Obtaining the member '_orig_showmsg' of a type (line 542)
            _orig_showmsg_709566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 16), self_709565, '_orig_showmsg')
            # Calling _orig_showmsg(args, kwargs) (line 542)
            _orig_showmsg_call_result_709569 = invoke(stypy.reporting.localization.Localization(__file__, 542, 16), _orig_showmsg_709566, *[use_warnmsg_709567], **kwargs_709568)
            

            if (may_be_709553 and more_types_in_union_709554):
                # SSA join for if statement (line 538)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of '_showwarning(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_showwarning' in the type store
        # Getting the type of 'stypy_return_type' (line 492)
        stypy_return_type_709570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 492, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709570)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_showwarning'
        return stypy_return_type_709570


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 544, 8, False)
        # Assigning a type to the variable 'self' (line 545)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        suppress_warnings.__call__.__dict__.__setitem__('stypy_localization', localization)
        suppress_warnings.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        suppress_warnings.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        suppress_warnings.__call__.__dict__.__setitem__('stypy_function_name', 'suppress_warnings.__call__')
        suppress_warnings.__call__.__dict__.__setitem__('stypy_param_names_list', ['func'])
        suppress_warnings.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        suppress_warnings.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        suppress_warnings.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        suppress_warnings.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        suppress_warnings.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        suppress_warnings.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'suppress_warnings.__call__', ['func'], None, None, defaults, varargs, kwargs)

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

        str_709571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, (-1)), 'str', '\n            Function decorator to apply certain suppressions to a whole\n            function.\n            ')

        @norecursion
        def new_func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'new_func'
            module_type_store = module_type_store.open_function_context('new_func', 549, 12, False)
            
            # Passed parameters checking function
            new_func.stypy_localization = localization
            new_func.stypy_type_of_self = None
            new_func.stypy_type_store = module_type_store
            new_func.stypy_function_name = 'new_func'
            new_func.stypy_param_names_list = []
            new_func.stypy_varargs_param_name = 'args'
            new_func.stypy_kwargs_param_name = 'kwargs'
            new_func.stypy_call_defaults = defaults
            new_func.stypy_call_varargs = varargs
            new_func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'new_func', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'new_func', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'new_func(...)' code ##################

            # Getting the type of 'self' (line 551)
            self_709572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 21), 'self')
            with_709573 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 551, 21), self_709572, 'with parameter', '__enter__', '__exit__')

            if with_709573:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 551)
                enter___709574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 21), self_709572, '__enter__')
                with_enter_709575 = invoke(stypy.reporting.localization.Localization(__file__, 551, 21), enter___709574)
                
                # Call to func(...): (line 552)
                # Getting the type of 'args' (line 552)
                args_709577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 33), 'args', False)
                # Processing the call keyword arguments (line 552)
                # Getting the type of 'kwargs' (line 552)
                kwargs_709578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 41), 'kwargs', False)
                kwargs_709579 = {'kwargs_709578': kwargs_709578}
                # Getting the type of 'func' (line 552)
                func_709576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 27), 'func', False)
                # Calling func(args, kwargs) (line 552)
                func_call_result_709580 = invoke(stypy.reporting.localization.Localization(__file__, 552, 27), func_709576, *[args_709577], **kwargs_709579)
                
                # Assigning a type to the variable 'stypy_return_type' (line 552)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 20), 'stypy_return_type', func_call_result_709580)
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 551)
                exit___709581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 21), self_709572, '__exit__')
                with_exit_709582 = invoke(stypy.reporting.localization.Localization(__file__, 551, 21), exit___709581, None, None, None)

            
            # ################# End of 'new_func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'new_func' in the type store
            # Getting the type of 'stypy_return_type' (line 549)
            stypy_return_type_709583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_709583)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'new_func'
            return stypy_return_type_709583

        # Assigning a type to the variable 'new_func' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 12), 'new_func', new_func)
        # Getting the type of 'new_func' (line 554)
        new_func_709584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 19), 'new_func')
        # Assigning a type to the variable 'stypy_return_type' (line 554)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 12), 'stypy_return_type', new_func_709584)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 544)
        stypy_return_type_709585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709585)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_709585


# Assigning a type to the variable 'suppress_warnings' (line 281)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'suppress_warnings', suppress_warnings)
# SSA join for try-except statement (line 278)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
