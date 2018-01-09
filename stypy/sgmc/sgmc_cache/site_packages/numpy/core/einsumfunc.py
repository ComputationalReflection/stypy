
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Implementation of optimized einsum.
3: 
4: '''
5: from __future__ import division, absolute_import, print_function
6: 
7: from numpy.core.multiarray import c_einsum
8: from numpy.core.numeric import asarray, asanyarray, result_type
9: 
10: __all__ = ['einsum', 'einsum_path']
11: 
12: einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
13: einsum_symbols_set = set(einsum_symbols)
14: 
15: 
16: def _compute_size_by_dict(indices, idx_dict):
17:     '''
18:     Computes the product of the elements in indices based on the dictionary
19:     idx_dict.
20: 
21:     Parameters
22:     ----------
23:     indices : iterable
24:         Indices to base the product on.
25:     idx_dict : dictionary
26:         Dictionary of index sizes
27: 
28:     Returns
29:     -------
30:     ret : int
31:         The resulting product.
32: 
33:     Examples
34:     --------
35:     >>> _compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})
36:     90
37: 
38:     '''
39:     ret = 1
40:     for i in indices:
41:         ret *= idx_dict[i]
42:     return ret
43: 
44: 
45: def _find_contraction(positions, input_sets, output_set):
46:     '''
47:     Finds the contraction for a given set of input and output sets.
48: 
49:     Parameters
50:     ----------
51:     positions : iterable
52:         Integer positions of terms used in the contraction.
53:     input_sets : list
54:         List of sets that represent the lhs side of the einsum subscript
55:     output_set : set
56:         Set that represents the rhs side of the overall einsum subscript
57: 
58:     Returns
59:     -------
60:     new_result : set
61:         The indices of the resulting contraction
62:     remaining : list
63:         List of sets that have not been contracted, the new set is appended to
64:         the end of this list
65:     idx_removed : set
66:         Indices removed from the entire contraction
67:     idx_contraction : set
68:         The indices used in the current contraction
69: 
70:     Examples
71:     --------
72: 
73:     # A simple dot product test case
74:     >>> pos = (0, 1)
75:     >>> isets = [set('ab'), set('bc')]
76:     >>> oset = set('ac')
77:     >>> _find_contraction(pos, isets, oset)
78:     ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})
79: 
80:     # A more complex case with additional terms in the contraction
81:     >>> pos = (0, 2)
82:     >>> isets = [set('abd'), set('ac'), set('bdc')]
83:     >>> oset = set('ac')
84:     >>> _find_contraction(pos, isets, oset)
85:     ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})
86:     '''
87: 
88:     idx_contract = set()
89:     idx_remain = output_set.copy()
90:     remaining = []
91:     for ind, value in enumerate(input_sets):
92:         if ind in positions:
93:             idx_contract |= value
94:         else:
95:             remaining.append(value)
96:             idx_remain |= value
97: 
98:     new_result = idx_remain & idx_contract
99:     idx_removed = (idx_contract - new_result)
100:     remaining.append(new_result)
101: 
102:     return (new_result, remaining, idx_removed, idx_contract)
103: 
104: 
105: def _optimal_path(input_sets, output_set, idx_dict, memory_limit):
106:     '''
107:     Computes all possible pair contractions, sieves the results based
108:     on ``memory_limit`` and returns the lowest cost path. This algorithm
109:     scales factorial with respect to the elements in the list ``input_sets``.
110: 
111:     Parameters
112:     ----------
113:     input_sets : list
114:         List of sets that represent the lhs side of the einsum subscript
115:     output_set : set
116:         Set that represents the rhs side of the overall einsum subscript
117:     idx_dict : dictionary
118:         Dictionary of index sizes
119:     memory_limit : int
120:         The maximum number of elements in a temporary array
121: 
122:     Returns
123:     -------
124:     path : list
125:         The optimal contraction order within the memory limit constraint.
126: 
127:     Examples
128:     --------
129:     >>> isets = [set('abd'), set('ac'), set('bdc')]
130:     >>> oset = set('')
131:     >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
132:     >>> _path__optimal_path(isets, oset, idx_sizes, 5000)
133:     [(0, 2), (0, 1)]
134:     '''
135: 
136:     full_results = [(0, [], input_sets)]
137:     for iteration in range(len(input_sets) - 1):
138:         iter_results = []
139: 
140:         # Compute all unique pairs
141:         comb_iter = []
142:         for x in range(len(input_sets) - iteration):
143:             for y in range(x + 1, len(input_sets) - iteration):
144:                 comb_iter.append((x, y))
145: 
146:         for curr in full_results:
147:             cost, positions, remaining = curr
148:             for con in comb_iter:
149: 
150:                 # Find the contraction
151:                 cont = _find_contraction(con, remaining, output_set)
152:                 new_result, new_input_sets, idx_removed, idx_contract = cont
153: 
154:                 # Sieve the results based on memory_limit
155:                 new_size = _compute_size_by_dict(new_result, idx_dict)
156:                 if new_size > memory_limit:
157:                     continue
158: 
159:                 # Find cost
160:                 new_cost = _compute_size_by_dict(idx_contract, idx_dict)
161:                 if idx_removed:
162:                     new_cost *= 2
163: 
164:                 # Build (total_cost, positions, indices_remaining)
165:                 new_cost += cost
166:                 new_pos = positions + [con]
167:                 iter_results.append((new_cost, new_pos, new_input_sets))
168: 
169:         # Update list to iterate over
170:         full_results = iter_results
171: 
172:     # If we have not found anything return single einsum contraction
173:     if len(full_results) == 0:
174:         return [tuple(range(len(input_sets)))]
175: 
176:     path = min(full_results, key=lambda x: x[0])[1]
177:     return path
178: 
179: 
180: def _greedy_path(input_sets, output_set, idx_dict, memory_limit):
181:     '''
182:     Finds the path by contracting the best pair until the input list is
183:     exhausted. The best pair is found by minimizing the tuple
184:     ``(-prod(indices_removed), cost)``.  What this amounts to is prioritizing
185:     matrix multiplication or inner product operations, then Hadamard like
186:     operations, and finally outer operations. Outer products are limited by
187:     ``memory_limit``. This algorithm scales cubically with respect to the
188:     number of elements in the list ``input_sets``.
189: 
190:     Parameters
191:     ----------
192:     input_sets : list
193:         List of sets that represent the lhs side of the einsum subscript
194:     output_set : set
195:         Set that represents the rhs side of the overall einsum subscript
196:     idx_dict : dictionary
197:         Dictionary of index sizes
198:     memory_limit_limit : int
199:         The maximum number of elements in a temporary array
200: 
201:     Returns
202:     -------
203:     path : list
204:         The greedy contraction order within the memory limit constraint.
205: 
206:     Examples
207:     --------
208:     >>> isets = [set('abd'), set('ac'), set('bdc')]
209:     >>> oset = set('')
210:     >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
211:     >>> _path__greedy_path(isets, oset, idx_sizes, 5000)
212:     [(0, 2), (0, 1)]
213:     '''
214: 
215:     if len(input_sets) == 1:
216:         return [(0,)]
217: 
218:     path = []
219:     for iteration in range(len(input_sets) - 1):
220:         iteration_results = []
221:         comb_iter = []
222: 
223:         # Compute all unique pairs
224:         for x in range(len(input_sets)):
225:             for y in range(x + 1, len(input_sets)):
226:                 comb_iter.append((x, y))
227: 
228:         for positions in comb_iter:
229: 
230:             # Find the contraction
231:             contract = _find_contraction(positions, input_sets, output_set)
232:             idx_result, new_input_sets, idx_removed, idx_contract = contract
233: 
234:             # Sieve the results based on memory_limit
235:             if _compute_size_by_dict(idx_result, idx_dict) > memory_limit:
236:                 continue
237: 
238:             # Build sort tuple
239:             removed_size = _compute_size_by_dict(idx_removed, idx_dict)
240:             cost = _compute_size_by_dict(idx_contract, idx_dict)
241:             sort = (-removed_size, cost)
242: 
243:             # Add contraction to possible choices
244:             iteration_results.append([sort, positions, new_input_sets])
245: 
246:         # If we did not find a new contraction contract remaining
247:         if len(iteration_results) == 0:
248:             path.append(tuple(range(len(input_sets))))
249:             break
250: 
251:         # Sort based on first index
252:         best = min(iteration_results, key=lambda x: x[0])
253:         path.append(best[1])
254:         input_sets = best[2]
255: 
256:     return path
257: 
258: 
259: def _parse_einsum_input(operands):
260:     '''
261:     A reproduction of einsum c side einsum parsing in python.
262: 
263:     Returns
264:     -------
265:     input_strings : str
266:         Parsed input strings
267:     output_string : str
268:         Parsed output string
269:     operands : list of array_like
270:         The operands to use in the numpy contraction
271: 
272:     Examples
273:     --------
274:     The operand list is simplified to reduce printing:
275: 
276:     >>> a = np.random.rand(4, 4)
277:     >>> b = np.random.rand(4, 4, 4)
278:     >>> __parse_einsum_input(('...a,...a->...', a, b))
279:     ('za,xza', 'xz', [a, b])
280: 
281:     >>> __parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))
282:     ('za,xza', 'xz', [a, b])
283:     '''
284: 
285:     if len(operands) == 0:
286:         raise ValueError("No input operands")
287: 
288:     if isinstance(operands[0], str):
289:         subscripts = operands[0].replace(" ", "")
290:         operands = [asanyarray(v) for v in operands[1:]]
291: 
292:         # Ensure all characters are valid
293:         for s in subscripts:
294:             if s in '.,->':
295:                 continue
296:             if s not in einsum_symbols:
297:                 raise ValueError("Character %s is not a valid symbol." % s)
298: 
299:     else:
300:         tmp_operands = list(operands)
301:         operand_list = []
302:         subscript_list = []
303:         for p in range(len(operands) // 2):
304:             operand_list.append(tmp_operands.pop(0))
305:             subscript_list.append(tmp_operands.pop(0))
306: 
307:         output_list = tmp_operands[-1] if len(tmp_operands) else None
308:         operands = [asanyarray(v) for v in operand_list]
309:         subscripts = ""
310:         last = len(subscript_list) - 1
311:         for num, sub in enumerate(subscript_list):
312:             for s in sub:
313:                 if s is Ellipsis:
314:                     subscripts += "..."
315:                 elif isinstance(s, int):
316:                     subscripts += einsum_symbols[s]
317:                 else:
318:                     raise TypeError("For this input type lists must contain "
319:                                     "either int or Ellipsis")
320:             if num != last:
321:                 subscripts += ","
322: 
323:         if output_list is not None:
324:             subscripts += "->"
325:             for s in output_list:
326:                 if s is Ellipsis:
327:                     subscripts += "..."
328:                 elif isinstance(s, int):
329:                     subscripts += einsum_symbols[s]
330:                 else:
331:                     raise TypeError("For this input type lists must contain "
332:                                     "either int or Ellipsis")
333:     # Check for proper "->"
334:     if ("-" in subscripts) or (">" in subscripts):
335:         invalid = (subscripts.count("-") > 1) or (subscripts.count(">") > 1)
336:         if invalid or (subscripts.count("->") != 1):
337:             raise ValueError("Subscripts can only contain one '->'.")
338: 
339:     # Parse ellipses
340:     if "." in subscripts:
341:         used = subscripts.replace(".", "").replace(",", "").replace("->", "")
342:         unused = list(einsum_symbols_set - set(used))
343:         ellipse_inds = "".join(unused)
344:         longest = 0
345: 
346:         if "->" in subscripts:
347:             input_tmp, output_sub = subscripts.split("->")
348:             split_subscripts = input_tmp.split(",")
349:             out_sub = True
350:         else:
351:             split_subscripts = subscripts.split(',')
352:             out_sub = False
353: 
354:         for num, sub in enumerate(split_subscripts):
355:             if "." in sub:
356:                 if (sub.count(".") != 3) or (sub.count("...") != 1):
357:                     raise ValueError("Invalid Ellipses.")
358: 
359:                 # Take into account numerical values
360:                 if operands[num].shape == ():
361:                     ellipse_count = 0
362:                 else:
363:                     ellipse_count = max(operands[num].ndim, 1)
364:                     ellipse_count -= (len(sub) - 3)
365: 
366:                 if ellipse_count > longest:
367:                     longest = ellipse_count
368: 
369:                 if ellipse_count < 0:
370:                     raise ValueError("Ellipses lengths do not match.")
371:                 elif ellipse_count == 0:
372:                     split_subscripts[num] = sub.replace('...', '')
373:                 else:
374:                     rep_inds = ellipse_inds[-ellipse_count:]
375:                     split_subscripts[num] = sub.replace('...', rep_inds)
376: 
377:         subscripts = ",".join(split_subscripts)
378:         if longest == 0:
379:             out_ellipse = ""
380:         else:
381:             out_ellipse = ellipse_inds[-longest:]
382: 
383:         if out_sub:
384:             subscripts += "->" + output_sub.replace("...", out_ellipse)
385:         else:
386:             # Special care for outputless ellipses
387:             output_subscript = ""
388:             tmp_subscripts = subscripts.replace(",", "")
389:             for s in sorted(set(tmp_subscripts)):
390:                 if s not in (einsum_symbols):
391:                     raise ValueError("Character %s is not a valid symbol." % s)
392:                 if tmp_subscripts.count(s) == 1:
393:                     output_subscript += s
394:             normal_inds = ''.join(sorted(set(output_subscript) -
395:                                          set(out_ellipse)))
396: 
397:             subscripts += "->" + out_ellipse + normal_inds
398: 
399:     # Build output string if does not exist
400:     if "->" in subscripts:
401:         input_subscripts, output_subscript = subscripts.split("->")
402:     else:
403:         input_subscripts = subscripts
404:         # Build output subscripts
405:         tmp_subscripts = subscripts.replace(",", "")
406:         output_subscript = ""
407:         for s in sorted(set(tmp_subscripts)):
408:             if s not in einsum_symbols:
409:                 raise ValueError("Character %s is not a valid symbol." % s)
410:             if tmp_subscripts.count(s) == 1:
411:                 output_subscript += s
412: 
413:     # Make sure output subscripts are in the input
414:     for char in output_subscript:
415:         if char not in input_subscripts:
416:             raise ValueError("Output character %s did not appear in the input"
417:                              % char)
418: 
419:     # Make sure number operands is equivalent to the number of terms
420:     if len(input_subscripts.split(',')) != len(operands):
421:         raise ValueError("Number of einsum subscripts must be equal to the "
422:                          "number of operands.")
423: 
424:     return (input_subscripts, output_subscript, operands)
425: 
426: 
427: def einsum_path(*operands, **kwargs):
428:     '''
429:     einsum_path(subscripts, *operands, optimize='greedy')
430: 
431:     Evaluates the lowest cost contraction order for an einsum expression by
432:     considering the creation of intermediate arrays.
433: 
434:     Parameters
435:     ----------
436:     subscripts : str
437:         Specifies the subscripts for summation.
438:     *operands : list of array_like
439:         These are the arrays for the operation.
440:     optimize : {bool, list, tuple, 'greedy', 'optimal'}
441:         Choose the type of path. If a tuple is provided, the second argument is
442:         assumed to be the maximum intermediate size created. If only a single
443:         argument is provided the largest input or output array size is used
444:         as a maximum intermediate size.
445: 
446:         * if a list is given that starts with ``einsum_path``, uses this as the
447:           contraction path
448:         * if False no optimization is taken
449:         * if True defaults to the 'greedy' algorithm
450:         * 'optimal' An algorithm that combinatorially explores all possible
451:           ways of contracting the listed tensors and choosest the least costly
452:           path. Scales exponentially with the number of terms in the
453:           contraction.
454:         * 'greedy' An algorithm that chooses the best pair contraction
455:           at each step. Effectively, this algorithm searches the largest inner,
456:           Hadamard, and then outer products at each step. Scales cubically with
457:           the number of terms in the contraction. Equivalent to the 'optimal'
458:           path for most contractions.
459: 
460:         Default is 'greedy'.
461: 
462:     Returns
463:     -------
464:     path : list of tuples
465:         A list representation of the einsum path.
466:     string_repr : str
467:         A printable representation of the einsum path.
468: 
469:     Notes
470:     -----
471:     The resulting path indicates which terms of the input contraction should be
472:     contracted first, the result of this contraction is then appended to the
473:     end of the contraction list. This list can then be iterated over until all
474:     intermediate contractions are complete.
475: 
476:     See Also
477:     --------
478:     einsum, linalg.multi_dot
479: 
480:     Examples
481:     --------
482: 
483:     We can begin with a chain dot example. In this case, it is optimal to
484:     contract the ``b`` and ``c`` tensors first as reprsented by the first
485:     element of the path ``(1, 2)``. The resulting tensor is added to the end
486:     of the contraction and the remaining contraction ``(0, 1)`` is then
487:     completed.
488: 
489:     >>> a = np.random.rand(2, 2)
490:     >>> b = np.random.rand(2, 5)
491:     >>> c = np.random.rand(5, 2)
492:     >>> path_info = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')
493:     >>> print(path_info[0])
494:     ['einsum_path', (1, 2), (0, 1)]
495:     >>> print(path_info[1])
496:       Complete contraction:  ij,jk,kl->il
497:              Naive scaling:  4
498:          Optimized scaling:  3
499:           Naive FLOP count:  1.600e+02
500:       Optimized FLOP count:  5.600e+01
501:        Theoretical speedup:  2.857
502:       Largest intermediate:  4.000e+00 elements
503:     -------------------------------------------------------------------------
504:     scaling                  current                                remaining
505:     -------------------------------------------------------------------------
506:        3                   kl,jk->jl                                ij,jl->il
507:        3                   jl,ij->il                                   il->il
508: 
509: 
510:     A more complex index transformation example.
511: 
512:     >>> I = np.random.rand(10, 10, 10, 10)
513:     >>> C = np.random.rand(10, 10)
514:     >>> path_info = np.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C,
515:                                    optimize='greedy')
516: 
517:     >>> print(path_info[0])
518:     ['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)]
519:     >>> print(path_info[1])
520:       Complete contraction:  ea,fb,abcd,gc,hd->efgh
521:              Naive scaling:  8
522:          Optimized scaling:  5
523:           Naive FLOP count:  8.000e+08
524:       Optimized FLOP count:  8.000e+05
525:        Theoretical speedup:  1000.000
526:       Largest intermediate:  1.000e+04 elements
527:     --------------------------------------------------------------------------
528:     scaling                  current                                remaining
529:     --------------------------------------------------------------------------
530:        5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh
531:        5               bcde,fb->cdef                         gc,hd,cdef->efgh
532:        5               cdef,gc->defg                            hd,defg->efgh
533:        5               defg,hd->efgh                               efgh->efgh
534:     '''
535: 
536:     # Make sure all keywords are valid
537:     valid_contract_kwargs = ['optimize', 'einsum_call']
538:     unknown_kwargs = [k for (k, v) in kwargs.items() if k
539:                       not in valid_contract_kwargs]
540:     if len(unknown_kwargs):
541:         raise TypeError("Did not understand the following kwargs:"
542:                         " %s" % unknown_kwargs)
543: 
544:     # Figure out what the path really is
545:     path_type = kwargs.pop('optimize', False)
546:     if path_type is True:
547:         path_type = 'greedy'
548:     if path_type is None:
549:         path_type = False
550: 
551:     memory_limit = None
552: 
553:     # No optimization or a named path algorithm
554:     if (path_type is False) or isinstance(path_type, str):
555:         pass
556: 
557:     # Given an explicit path
558:     elif len(path_type) and (path_type[0] == 'einsum_path'):
559:         pass
560: 
561:     # Path tuple with memory limit
562:     elif ((len(path_type) == 2) and isinstance(path_type[0], str) and
563:             isinstance(path_type[1], (int, float))):
564:         memory_limit = int(path_type[1])
565:         path_type = path_type[0]
566: 
567:     else:
568:         raise TypeError("Did not understand the path: %s" % str(path_type))
569: 
570:     # Hidden option, only einsum should call this
571:     einsum_call_arg = kwargs.pop("einsum_call", False)
572: 
573:     # Python side parsing
574:     input_subscripts, output_subscript, operands = _parse_einsum_input(operands)
575:     subscripts = input_subscripts + '->' + output_subscript
576: 
577:     # Build a few useful list and sets
578:     input_list = input_subscripts.split(',')
579:     input_sets = [set(x) for x in input_list]
580:     output_set = set(output_subscript)
581:     indices = set(input_subscripts.replace(',', ''))
582: 
583:     # Get length of each unique dimension and ensure all dimensions are correct
584:     dimension_dict = {}
585:     for tnum, term in enumerate(input_list):
586:         sh = operands[tnum].shape
587:         if len(sh) != len(term):
588:             raise ValueError("Einstein sum subscript %s does not contain the "
589:                              "correct number of indices for operand %d.",
590:                              input_subscripts[tnum], tnum)
591:         for cnum, char in enumerate(term):
592:             dim = sh[cnum]
593:             if char in dimension_dict.keys():
594:                 if dimension_dict[char] != dim:
595:                     raise ValueError("Size of label '%s' for operand %d does "
596:                                      "not match previous terms.", char, tnum)
597:             else:
598:                 dimension_dict[char] = dim
599: 
600:     # Compute size of each input array plus the output array
601:     size_list = []
602:     for term in input_list + [output_subscript]:
603:         size_list.append(_compute_size_by_dict(term, dimension_dict))
604:     max_size = max(size_list)
605: 
606:     if memory_limit is None:
607:         memory_arg = max_size
608:     else:
609:         memory_arg = memory_limit
610: 
611:     # Compute naive cost
612:     # This isnt quite right, need to look into exactly how einsum does this
613:     naive_cost = _compute_size_by_dict(indices, dimension_dict)
614:     indices_in_input = input_subscripts.replace(',', '')
615:     mult = max(len(input_list) - 1, 1)
616:     if (len(indices_in_input) - len(set(indices_in_input))):
617:         mult *= 2
618:     naive_cost *= mult
619: 
620:     # Compute the path
621:     if (path_type is False) or (len(input_list) in [1, 2]) or (indices == output_set):
622:         # Nothing to be optimized, leave it to einsum
623:         path = [tuple(range(len(input_list)))]
624:     elif path_type == "greedy":
625:         # Maximum memory should be at most out_size for this algorithm
626:         memory_arg = min(memory_arg, max_size)
627:         path = _greedy_path(input_sets, output_set, dimension_dict, memory_arg)
628:     elif path_type == "optimal":
629:         path = _optimal_path(input_sets, output_set, dimension_dict, memory_arg)
630:     elif path_type[0] == 'einsum_path':
631:         path = path_type[1:]
632:     else:
633:         raise KeyError("Path name %s not found", path_type)
634: 
635:     cost_list, scale_list, size_list, contraction_list = [], [], [], []
636: 
637:     # Build contraction tuple (positions, gemm, einsum_str, remaining)
638:     for cnum, contract_inds in enumerate(path):
639:         # Make sure we remove inds from right to left
640:         contract_inds = tuple(sorted(list(contract_inds), reverse=True))
641: 
642:         contract = _find_contraction(contract_inds, input_sets, output_set)
643:         out_inds, input_sets, idx_removed, idx_contract = contract
644: 
645:         cost = _compute_size_by_dict(idx_contract, dimension_dict)
646:         if idx_removed:
647:             cost *= 2
648:         cost_list.append(cost)
649:         scale_list.append(len(idx_contract))
650:         size_list.append(_compute_size_by_dict(out_inds, dimension_dict))
651: 
652:         tmp_inputs = []
653:         for x in contract_inds:
654:             tmp_inputs.append(input_list.pop(x))
655: 
656:         # Last contraction
657:         if (cnum - len(path)) == -1:
658:             idx_result = output_subscript
659:         else:
660:             sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
661:             idx_result = "".join([x[1] for x in sorted(sort_result)])
662: 
663:         input_list.append(idx_result)
664:         einsum_str = ",".join(tmp_inputs) + "->" + idx_result
665: 
666:         contraction = (contract_inds, idx_removed, einsum_str, input_list[:])
667:         contraction_list.append(contraction)
668: 
669:     opt_cost = sum(cost_list) + 1
670: 
671:     if einsum_call_arg:
672:         return (operands, contraction_list)
673: 
674:     # Return the path along with a nice string representation
675:     overall_contraction = input_subscripts + "->" + output_subscript
676:     header = ("scaling", "current", "remaining")
677: 
678:     speedup = naive_cost / opt_cost
679:     max_i = max(size_list)
680: 
681:     path_print  = "  Complete contraction:  %s\n" % overall_contraction
682:     path_print += "         Naive scaling:  %d\n" % len(indices)
683:     path_print += "     Optimized scaling:  %d\n" % max(scale_list)
684:     path_print += "      Naive FLOP count:  %.3e\n" % naive_cost
685:     path_print += "  Optimized FLOP count:  %.3e\n" % opt_cost
686:     path_print += "   Theoretical speedup:  %3.3f\n" % speedup
687:     path_print += "  Largest intermediate:  %.3e elements\n" % max_i
688:     path_print += "-" * 74 + "\n"
689:     path_print += "%6s %24s %40s\n" % header
690:     path_print += "-" * 74
691: 
692:     for n, contraction in enumerate(contraction_list):
693:         inds, idx_rm, einsum_str, remaining = contraction
694:         remaining_str = ",".join(remaining) + "->" + output_subscript
695:         path_run = (scale_list[n], einsum_str, remaining_str)
696:         path_print += "\n%4d    %24s %40s" % path_run
697: 
698:     path = ['einsum_path'] + path
699:     return (path, path_print)
700: 
701: 
702: # Rewrite einsum to handle different cases
703: def einsum(*operands, **kwargs):
704:     '''
705:     einsum(subscripts, *operands, out=None, dtype=None, order='K',
706:            casting='safe', optimize=False)
707: 
708:     Evaluates the Einstein summation convention on the operands.
709: 
710:     Using the Einstein summation convention, many common multi-dimensional
711:     array operations can be represented in a simple fashion.  This function
712:     provides a way to compute such summations. The best way to understand this
713:     function is to try the examples below, which show how many common NumPy
714:     functions can be implemented as calls to `einsum`.
715: 
716:     Parameters
717:     ----------
718:     subscripts : str
719:         Specifies the subscripts for summation.
720:     operands : list of array_like
721:         These are the arrays for the operation.
722:     out : {ndarray, None}, optional
723:         If provided, the calculation is done into this array.
724:     dtype : {data-type, None}, optional
725:         If provided, forces the calculation to use the data type specified.
726:         Note that you may have to also give a more liberal `casting`
727:         parameter to allow the conversions. Default is None.
728:     order : {'C', 'F', 'A', 'K'}, optional
729:         Controls the memory layout of the output. 'C' means it should
730:         be C contiguous. 'F' means it should be Fortran contiguous,
731:         'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
732:         'K' means it should be as close to the layout as the inputs as
733:         is possible, including arbitrarily permuted axes.
734:         Default is 'K'.
735:     casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
736:         Controls what kind of data casting may occur.  Setting this to
737:         'unsafe' is not recommended, as it can adversely affect accumulations.
738: 
739:           * 'no' means the data types should not be cast at all.
740:           * 'equiv' means only byte-order changes are allowed.
741:           * 'safe' means only casts which can preserve values are allowed.
742:           * 'same_kind' means only safe casts or casts within a kind,
743:             like float64 to float32, are allowed.
744:           * 'unsafe' means any data conversions may be done.
745: 
746:         Default is 'safe'.
747:     optimize : {False, True, 'greedy', 'optimal'}, optional
748:         Controls if intermediate optimization should occur. No optimization
749:         will occur if False and True will default to the 'greedy' algorithm.
750:         Also accepts an explicit contraction list from the ``np.einsum_path``
751:         function. See ``np.einsum_path`` for more details. Default is False.
752: 
753:     Returns
754:     -------
755:     output : ndarray
756:         The calculation based on the Einstein summation convention.
757: 
758:     See Also
759:     --------
760:     einsum_path, dot, inner, outer, tensordot, linalg.multi_dot
761: 
762:     Notes
763:     -----
764:     .. versionadded:: 1.6.0
765: 
766:     The subscripts string is a comma-separated list of subscript labels,
767:     where each label refers to a dimension of the corresponding operand.
768:     Repeated subscripts labels in one operand take the diagonal.  For example,
769:     ``np.einsum('ii', a)`` is equivalent to ``np.trace(a)``.
770: 
771:     Whenever a label is repeated, it is summed, so ``np.einsum('i,i', a, b)``
772:     is equivalent to ``np.inner(a,b)``.  If a label appears only once,
773:     it is not summed, so ``np.einsum('i', a)`` produces a view of ``a``
774:     with no changes.
775: 
776:     The order of labels in the output is by default alphabetical.  This
777:     means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
778:     ``np.einsum('ji', a)`` takes its transpose.
779: 
780:     The output can be controlled by specifying output subscript labels
781:     as well.  This specifies the label order, and allows summing to
782:     be disallowed or forced when desired.  The call ``np.einsum('i->', a)``
783:     is like ``np.sum(a, axis=-1)``, and ``np.einsum('ii->i', a)``
784:     is like ``np.diag(a)``.  The difference is that `einsum` does not
785:     allow broadcasting by default.
786: 
787:     To enable and control broadcasting, use an ellipsis.  Default
788:     NumPy-style broadcasting is done by adding an ellipsis
789:     to the left of each term, like ``np.einsum('...ii->...i', a)``.
790:     To take the trace along the first and last axes,
791:     you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix
792:     product with the left-most indices instead of rightmost, you can do
793:     ``np.einsum('ij...,jk...->ik...', a, b)``.
794: 
795:     When there is only one operand, no axes are summed, and no output
796:     parameter is provided, a view into the operand is returned instead
797:     of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``
798:     produces a view.
799: 
800:     An alternative way to provide the subscripts and operands is as
801:     ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``. The examples
802:     below have corresponding `einsum` calls with the two parameter methods.
803: 
804:     .. versionadded:: 1.10.0
805: 
806:     Views returned from einsum are now writeable whenever the input array
807:     is writeable. For example, ``np.einsum('ijk...->kji...', a)`` will now
808:     have the same effect as ``np.swapaxes(a, 0, 2)`` and
809:     ``np.einsum('ii->i', a)`` will return a writeable view of the diagonal
810:     of a 2D array.
811: 
812:     .. versionadded:: 1.12.0
813: 
814:     Added the ``optimize`` argument which will optimize the contraction order
815:     of an einsum expression. For a contraction with three or more operands this
816:     can greatly increase the computational efficiency at the cost of a larger
817:     memory footprint during computation.
818: 
819:     See ``np.einsum_path`` for more details.
820: 
821:     Examples
822:     --------
823:     >>> a = np.arange(25).reshape(5,5)
824:     >>> b = np.arange(5)
825:     >>> c = np.arange(6).reshape(2,3)
826: 
827:     >>> np.einsum('ii', a)
828:     60
829:     >>> np.einsum(a, [0,0])
830:     60
831:     >>> np.trace(a)
832:     60
833: 
834:     >>> np.einsum('ii->i', a)
835:     array([ 0,  6, 12, 18, 24])
836:     >>> np.einsum(a, [0,0], [0])
837:     array([ 0,  6, 12, 18, 24])
838:     >>> np.diag(a)
839:     array([ 0,  6, 12, 18, 24])
840: 
841:     >>> np.einsum('ij,j', a, b)
842:     array([ 30,  80, 130, 180, 230])
843:     >>> np.einsum(a, [0,1], b, [1])
844:     array([ 30,  80, 130, 180, 230])
845:     >>> np.dot(a, b)
846:     array([ 30,  80, 130, 180, 230])
847:     >>> np.einsum('...j,j', a, b)
848:     array([ 30,  80, 130, 180, 230])
849: 
850:     >>> np.einsum('ji', c)
851:     array([[0, 3],
852:            [1, 4],
853:            [2, 5]])
854:     >>> np.einsum(c, [1,0])
855:     array([[0, 3],
856:            [1, 4],
857:            [2, 5]])
858:     >>> c.T
859:     array([[0, 3],
860:            [1, 4],
861:            [2, 5]])
862: 
863:     >>> np.einsum('..., ...', 3, c)
864:     array([[ 0,  3,  6],
865:            [ 9, 12, 15]])
866:     >>> np.einsum(',ij', 3, C)
867:     array([[ 0,  3,  6],
868:            [ 9, 12, 15]])
869:     >>> np.einsum(3, [Ellipsis], c, [Ellipsis])
870:     array([[ 0,  3,  6],
871:            [ 9, 12, 15]])
872:     >>> np.multiply(3, c)
873:     array([[ 0,  3,  6],
874:            [ 9, 12, 15]])
875: 
876:     >>> np.einsum('i,i', b, b)
877:     30
878:     >>> np.einsum(b, [0], b, [0])
879:     30
880:     >>> np.inner(b,b)
881:     30
882: 
883:     >>> np.einsum('i,j', np.arange(2)+1, b)
884:     array([[0, 1, 2, 3, 4],
885:            [0, 2, 4, 6, 8]])
886:     >>> np.einsum(np.arange(2)+1, [0], b, [1])
887:     array([[0, 1, 2, 3, 4],
888:            [0, 2, 4, 6, 8]])
889:     >>> np.outer(np.arange(2)+1, b)
890:     array([[0, 1, 2, 3, 4],
891:            [0, 2, 4, 6, 8]])
892: 
893:     >>> np.einsum('i...->...', a)
894:     array([50, 55, 60, 65, 70])
895:     >>> np.einsum(a, [0,Ellipsis], [Ellipsis])
896:     array([50, 55, 60, 65, 70])
897:     >>> np.sum(a, axis=0)
898:     array([50, 55, 60, 65, 70])
899: 
900:     >>> a = np.arange(60.).reshape(3,4,5)
901:     >>> b = np.arange(24.).reshape(4,3,2)
902:     >>> np.einsum('ijk,jil->kl', a, b)
903:     array([[ 4400.,  4730.],
904:            [ 4532.,  4874.],
905:            [ 4664.,  5018.],
906:            [ 4796.,  5162.],
907:            [ 4928.,  5306.]])
908:     >>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])
909:     array([[ 4400.,  4730.],
910:            [ 4532.,  4874.],
911:            [ 4664.,  5018.],
912:            [ 4796.,  5162.],
913:            [ 4928.,  5306.]])
914:     >>> np.tensordot(a,b, axes=([1,0],[0,1]))
915:     array([[ 4400.,  4730.],
916:            [ 4532.,  4874.],
917:            [ 4664.,  5018.],
918:            [ 4796.,  5162.],
919:            [ 4928.,  5306.]])
920: 
921:     >>> a = np.arange(6).reshape((3,2))
922:     >>> b = np.arange(12).reshape((4,3))
923:     >>> np.einsum('ki,jk->ij', a, b)
924:     array([[10, 28, 46, 64],
925:            [13, 40, 67, 94]])
926:     >>> np.einsum('ki,...k->i...', a, b)
927:     array([[10, 28, 46, 64],
928:            [13, 40, 67, 94]])
929:     >>> np.einsum('k...,jk', a, b)
930:     array([[10, 28, 46, 64],
931:            [13, 40, 67, 94]])
932: 
933:     >>> # since version 1.10.0
934:     >>> a = np.zeros((3, 3))
935:     >>> np.einsum('ii->i', a)[:] = 1
936:     >>> a
937:     array([[ 1.,  0.,  0.],
938:            [ 0.,  1.,  0.],
939:            [ 0.,  0.,  1.]])
940: 
941:     '''
942: 
943:     # Grab non-einsum kwargs
944:     optimize_arg = kwargs.pop('optimize', False)
945: 
946:     # If no optimization, run pure einsum
947:     if optimize_arg is False:
948:         return c_einsum(*operands, **kwargs)
949: 
950:     valid_einsum_kwargs = ['out', 'dtype', 'order', 'casting']
951:     einsum_kwargs = {k: v for (k, v) in kwargs.items() if
952:                      k in valid_einsum_kwargs}
953: 
954:     # Make sure all keywords are valid
955:     valid_contract_kwargs = ['optimize'] + valid_einsum_kwargs
956:     unknown_kwargs = [k for (k, v) in kwargs.items() if
957:                       k not in valid_contract_kwargs]
958: 
959:     if len(unknown_kwargs):
960:         raise TypeError("Did not understand the following kwargs: %s"
961:                         % unknown_kwargs)
962: 
963:     # Special handeling if out is specified
964:     specified_out = False
965:     out_array = einsum_kwargs.pop('out', None)
966:     if out_array is not None:
967:         specified_out = True
968: 
969:     # Build the contraction list and operand
970:     operands, contraction_list = einsum_path(*operands, optimize=optimize_arg,
971:                                              einsum_call=True)
972:     # Start contraction loop
973:     for num, contraction in enumerate(contraction_list):
974:         inds, idx_rm, einsum_str, remaining = contraction
975:         tmp_operands = []
976:         for x in inds:
977:             tmp_operands.append(operands.pop(x))
978: 
979:         # If out was specified
980:         if specified_out and ((num + 1) == len(contraction_list)):
981:             einsum_kwargs["out"] = out_array
982: 
983:         # Do the contraction
984:         new_view = c_einsum(einsum_str, *tmp_operands, **einsum_kwargs)
985: 
986:         # Append new items and derefernce what we can
987:         operands.append(new_view)
988:         del tmp_operands, new_view
989: 
990:     if specified_out:
991:         return out_array
992:     else:
993:         return operands[0]
994: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nImplementation of optimized einsum.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.core.multiarray import c_einsum' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_38 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.multiarray')

if (type(import_38) is not StypyTypeError):

    if (import_38 != 'pyd_module'):
        __import__(import_38)
        sys_modules_39 = sys.modules[import_38]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.multiarray', sys_modules_39.module_type_store, module_type_store, ['c_einsum'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_39, sys_modules_39.module_type_store, module_type_store)
    else:
        from numpy.core.multiarray import c_einsum

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.multiarray', None, module_type_store, ['c_einsum'], [c_einsum])

else:
    # Assigning a type to the variable 'numpy.core.multiarray' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.core.multiarray', import_38)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.core.numeric import asarray, asanyarray, result_type' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_40 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.numeric')

if (type(import_40) is not StypyTypeError):

    if (import_40 != 'pyd_module'):
        __import__(import_40)
        sys_modules_41 = sys.modules[import_40]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.numeric', sys_modules_41.module_type_store, module_type_store, ['asarray', 'asanyarray', 'result_type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_41, sys_modules_41.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import asarray, asanyarray, result_type

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.numeric', None, module_type_store, ['asarray', 'asanyarray', 'result_type'], [asarray, asanyarray, result_type])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.numeric', import_40)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


# Assigning a List to a Name (line 10):

# Assigning a List to a Name (line 10):
__all__ = ['einsum', 'einsum_path']
module_type_store.set_exportable_members(['einsum', 'einsum_path'])

# Obtaining an instance of the builtin type 'list' (line 10)
list_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 11), 'str', 'einsum')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_42, str_43)
# Adding element type (line 10)
str_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'str', 'einsum_path')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 10), list_42, str_44)

# Assigning a type to the variable '__all__' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), '__all__', list_42)

# Assigning a Str to a Name (line 12):

# Assigning a Str to a Name (line 12):
str_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'str', 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
# Assigning a type to the variable 'einsum_symbols' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'einsum_symbols', str_45)

# Assigning a Call to a Name (line 13):

# Assigning a Call to a Name (line 13):

# Call to set(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'einsum_symbols' (line 13)
einsum_symbols_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 25), 'einsum_symbols', False)
# Processing the call keyword arguments (line 13)
kwargs_48 = {}
# Getting the type of 'set' (line 13)
set_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 21), 'set', False)
# Calling set(args, kwargs) (line 13)
set_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 13, 21), set_46, *[einsum_symbols_47], **kwargs_48)

# Assigning a type to the variable 'einsum_symbols_set' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'einsum_symbols_set', set_call_result_49)

@norecursion
def _compute_size_by_dict(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_compute_size_by_dict'
    module_type_store = module_type_store.open_function_context('_compute_size_by_dict', 16, 0, False)
    
    # Passed parameters checking function
    _compute_size_by_dict.stypy_localization = localization
    _compute_size_by_dict.stypy_type_of_self = None
    _compute_size_by_dict.stypy_type_store = module_type_store
    _compute_size_by_dict.stypy_function_name = '_compute_size_by_dict'
    _compute_size_by_dict.stypy_param_names_list = ['indices', 'idx_dict']
    _compute_size_by_dict.stypy_varargs_param_name = None
    _compute_size_by_dict.stypy_kwargs_param_name = None
    _compute_size_by_dict.stypy_call_defaults = defaults
    _compute_size_by_dict.stypy_call_varargs = varargs
    _compute_size_by_dict.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_compute_size_by_dict', ['indices', 'idx_dict'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_compute_size_by_dict', localization, ['indices', 'idx_dict'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_compute_size_by_dict(...)' code ##################

    str_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'str', "\n    Computes the product of the elements in indices based on the dictionary\n    idx_dict.\n\n    Parameters\n    ----------\n    indices : iterable\n        Indices to base the product on.\n    idx_dict : dictionary\n        Dictionary of index sizes\n\n    Returns\n    -------\n    ret : int\n        The resulting product.\n\n    Examples\n    --------\n    >>> _compute_size_by_dict('abbc', {'a': 2, 'b':3, 'c':5})\n    90\n\n    ")
    
    # Assigning a Num to a Name (line 39):
    
    # Assigning a Num to a Name (line 39):
    int_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 10), 'int')
    # Assigning a type to the variable 'ret' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'ret', int_51)
    
    # Getting the type of 'indices' (line 40)
    indices_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'indices')
    # Testing the type of a for loop iterable (line 40)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 4), indices_52)
    # Getting the type of the for loop variable (line 40)
    for_loop_var_53 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 4), indices_52)
    # Assigning a type to the variable 'i' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'i', for_loop_var_53)
    # SSA begins for a for statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'ret' (line 41)
    ret_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'ret')
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 41)
    i_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 24), 'i')
    # Getting the type of 'idx_dict' (line 41)
    idx_dict_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'idx_dict')
    # Obtaining the member '__getitem__' of a type (line 41)
    getitem___57 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 15), idx_dict_56, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 41)
    subscript_call_result_58 = invoke(stypy.reporting.localization.Localization(__file__, 41, 15), getitem___57, i_55)
    
    # Applying the binary operator '*=' (line 41)
    result_imul_59 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 8), '*=', ret_54, subscript_call_result_58)
    # Assigning a type to the variable 'ret' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'ret', result_imul_59)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'ret' (line 42)
    ret_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'ret')
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type', ret_60)
    
    # ################# End of '_compute_size_by_dict(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_compute_size_by_dict' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_61)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_compute_size_by_dict'
    return stypy_return_type_61

# Assigning a type to the variable '_compute_size_by_dict' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), '_compute_size_by_dict', _compute_size_by_dict)

@norecursion
def _find_contraction(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_find_contraction'
    module_type_store = module_type_store.open_function_context('_find_contraction', 45, 0, False)
    
    # Passed parameters checking function
    _find_contraction.stypy_localization = localization
    _find_contraction.stypy_type_of_self = None
    _find_contraction.stypy_type_store = module_type_store
    _find_contraction.stypy_function_name = '_find_contraction'
    _find_contraction.stypy_param_names_list = ['positions', 'input_sets', 'output_set']
    _find_contraction.stypy_varargs_param_name = None
    _find_contraction.stypy_kwargs_param_name = None
    _find_contraction.stypy_call_defaults = defaults
    _find_contraction.stypy_call_varargs = varargs
    _find_contraction.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_find_contraction', ['positions', 'input_sets', 'output_set'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_find_contraction', localization, ['positions', 'input_sets', 'output_set'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_find_contraction(...)' code ##################

    str_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, (-1)), 'str', "\n    Finds the contraction for a given set of input and output sets.\n\n    Parameters\n    ----------\n    positions : iterable\n        Integer positions of terms used in the contraction.\n    input_sets : list\n        List of sets that represent the lhs side of the einsum subscript\n    output_set : set\n        Set that represents the rhs side of the overall einsum subscript\n\n    Returns\n    -------\n    new_result : set\n        The indices of the resulting contraction\n    remaining : list\n        List of sets that have not been contracted, the new set is appended to\n        the end of this list\n    idx_removed : set\n        Indices removed from the entire contraction\n    idx_contraction : set\n        The indices used in the current contraction\n\n    Examples\n    --------\n\n    # A simple dot product test case\n    >>> pos = (0, 1)\n    >>> isets = [set('ab'), set('bc')]\n    >>> oset = set('ac')\n    >>> _find_contraction(pos, isets, oset)\n    ({'a', 'c'}, [{'a', 'c'}], {'b'}, {'a', 'b', 'c'})\n\n    # A more complex case with additional terms in the contraction\n    >>> pos = (0, 2)\n    >>> isets = [set('abd'), set('ac'), set('bdc')]\n    >>> oset = set('ac')\n    >>> _find_contraction(pos, isets, oset)\n    ({'a', 'c'}, [{'a', 'c'}, {'a', 'c'}], {'b', 'd'}, {'a', 'b', 'c', 'd'})\n    ")
    
    # Assigning a Call to a Name (line 88):
    
    # Assigning a Call to a Name (line 88):
    
    # Call to set(...): (line 88)
    # Processing the call keyword arguments (line 88)
    kwargs_64 = {}
    # Getting the type of 'set' (line 88)
    set_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'set', False)
    # Calling set(args, kwargs) (line 88)
    set_call_result_65 = invoke(stypy.reporting.localization.Localization(__file__, 88, 19), set_63, *[], **kwargs_64)
    
    # Assigning a type to the variable 'idx_contract' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'idx_contract', set_call_result_65)
    
    # Assigning a Call to a Name (line 89):
    
    # Assigning a Call to a Name (line 89):
    
    # Call to copy(...): (line 89)
    # Processing the call keyword arguments (line 89)
    kwargs_68 = {}
    # Getting the type of 'output_set' (line 89)
    output_set_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'output_set', False)
    # Obtaining the member 'copy' of a type (line 89)
    copy_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 17), output_set_66, 'copy')
    # Calling copy(args, kwargs) (line 89)
    copy_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 89, 17), copy_67, *[], **kwargs_68)
    
    # Assigning a type to the variable 'idx_remain' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'idx_remain', copy_call_result_69)
    
    # Assigning a List to a Name (line 90):
    
    # Assigning a List to a Name (line 90):
    
    # Obtaining an instance of the builtin type 'list' (line 90)
    list_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 90)
    
    # Assigning a type to the variable 'remaining' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'remaining', list_70)
    
    
    # Call to enumerate(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'input_sets' (line 91)
    input_sets_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'input_sets', False)
    # Processing the call keyword arguments (line 91)
    kwargs_73 = {}
    # Getting the type of 'enumerate' (line 91)
    enumerate_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 91)
    enumerate_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 91, 22), enumerate_71, *[input_sets_72], **kwargs_73)
    
    # Testing the type of a for loop iterable (line 91)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 91, 4), enumerate_call_result_74)
    # Getting the type of the for loop variable (line 91)
    for_loop_var_75 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 91, 4), enumerate_call_result_74)
    # Assigning a type to the variable 'ind' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'ind', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 4), for_loop_var_75))
    # Assigning a type to the variable 'value' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 4), for_loop_var_75))
    # SSA begins for a for statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'ind' (line 92)
    ind_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'ind')
    # Getting the type of 'positions' (line 92)
    positions_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 18), 'positions')
    # Applying the binary operator 'in' (line 92)
    result_contains_78 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 11), 'in', ind_76, positions_77)
    
    # Testing the type of an if condition (line 92)
    if_condition_79 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 8), result_contains_78)
    # Assigning a type to the variable 'if_condition_79' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'if_condition_79', if_condition_79)
    # SSA begins for if statement (line 92)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'idx_contract' (line 93)
    idx_contract_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'idx_contract')
    # Getting the type of 'value' (line 93)
    value_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'value')
    # Applying the binary operator '|=' (line 93)
    result_ior_82 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 12), '|=', idx_contract_80, value_81)
    # Assigning a type to the variable 'idx_contract' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'idx_contract', result_ior_82)
    
    # SSA branch for the else part of an if statement (line 92)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'value' (line 95)
    value_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 29), 'value', False)
    # Processing the call keyword arguments (line 95)
    kwargs_86 = {}
    # Getting the type of 'remaining' (line 95)
    remaining_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'remaining', False)
    # Obtaining the member 'append' of a type (line 95)
    append_84 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), remaining_83, 'append')
    # Calling append(args, kwargs) (line 95)
    append_call_result_87 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), append_84, *[value_85], **kwargs_86)
    
    
    # Getting the type of 'idx_remain' (line 96)
    idx_remain_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'idx_remain')
    # Getting the type of 'value' (line 96)
    value_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'value')
    # Applying the binary operator '|=' (line 96)
    result_ior_90 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 12), '|=', idx_remain_88, value_89)
    # Assigning a type to the variable 'idx_remain' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'idx_remain', result_ior_90)
    
    # SSA join for if statement (line 92)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 98):
    
    # Assigning a BinOp to a Name (line 98):
    # Getting the type of 'idx_remain' (line 98)
    idx_remain_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 'idx_remain')
    # Getting the type of 'idx_contract' (line 98)
    idx_contract_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 'idx_contract')
    # Applying the binary operator '&' (line 98)
    result_and__93 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 17), '&', idx_remain_91, idx_contract_92)
    
    # Assigning a type to the variable 'new_result' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'new_result', result_and__93)
    
    # Assigning a BinOp to a Name (line 99):
    
    # Assigning a BinOp to a Name (line 99):
    # Getting the type of 'idx_contract' (line 99)
    idx_contract_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'idx_contract')
    # Getting the type of 'new_result' (line 99)
    new_result_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'new_result')
    # Applying the binary operator '-' (line 99)
    result_sub_96 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 19), '-', idx_contract_94, new_result_95)
    
    # Assigning a type to the variable 'idx_removed' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'idx_removed', result_sub_96)
    
    # Call to append(...): (line 100)
    # Processing the call arguments (line 100)
    # Getting the type of 'new_result' (line 100)
    new_result_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 21), 'new_result', False)
    # Processing the call keyword arguments (line 100)
    kwargs_100 = {}
    # Getting the type of 'remaining' (line 100)
    remaining_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'remaining', False)
    # Obtaining the member 'append' of a type (line 100)
    append_98 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 4), remaining_97, 'append')
    # Calling append(args, kwargs) (line 100)
    append_call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 100, 4), append_98, *[new_result_99], **kwargs_100)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 102)
    tuple_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 102)
    # Adding element type (line 102)
    # Getting the type of 'new_result' (line 102)
    new_result_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'new_result')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 12), tuple_102, new_result_103)
    # Adding element type (line 102)
    # Getting the type of 'remaining' (line 102)
    remaining_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 24), 'remaining')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 12), tuple_102, remaining_104)
    # Adding element type (line 102)
    # Getting the type of 'idx_removed' (line 102)
    idx_removed_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 35), 'idx_removed')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 12), tuple_102, idx_removed_105)
    # Adding element type (line 102)
    # Getting the type of 'idx_contract' (line 102)
    idx_contract_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 48), 'idx_contract')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 12), tuple_102, idx_contract_106)
    
    # Assigning a type to the variable 'stypy_return_type' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type', tuple_102)
    
    # ################# End of '_find_contraction(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_find_contraction' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_107)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_find_contraction'
    return stypy_return_type_107

# Assigning a type to the variable '_find_contraction' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), '_find_contraction', _find_contraction)

@norecursion
def _optimal_path(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_optimal_path'
    module_type_store = module_type_store.open_function_context('_optimal_path', 105, 0, False)
    
    # Passed parameters checking function
    _optimal_path.stypy_localization = localization
    _optimal_path.stypy_type_of_self = None
    _optimal_path.stypy_type_store = module_type_store
    _optimal_path.stypy_function_name = '_optimal_path'
    _optimal_path.stypy_param_names_list = ['input_sets', 'output_set', 'idx_dict', 'memory_limit']
    _optimal_path.stypy_varargs_param_name = None
    _optimal_path.stypy_kwargs_param_name = None
    _optimal_path.stypy_call_defaults = defaults
    _optimal_path.stypy_call_varargs = varargs
    _optimal_path.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_optimal_path', ['input_sets', 'output_set', 'idx_dict', 'memory_limit'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_optimal_path', localization, ['input_sets', 'output_set', 'idx_dict', 'memory_limit'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_optimal_path(...)' code ##################

    str_108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, (-1)), 'str', "\n    Computes all possible pair contractions, sieves the results based\n    on ``memory_limit`` and returns the lowest cost path. This algorithm\n    scales factorial with respect to the elements in the list ``input_sets``.\n\n    Parameters\n    ----------\n    input_sets : list\n        List of sets that represent the lhs side of the einsum subscript\n    output_set : set\n        Set that represents the rhs side of the overall einsum subscript\n    idx_dict : dictionary\n        Dictionary of index sizes\n    memory_limit : int\n        The maximum number of elements in a temporary array\n\n    Returns\n    -------\n    path : list\n        The optimal contraction order within the memory limit constraint.\n\n    Examples\n    --------\n    >>> isets = [set('abd'), set('ac'), set('bdc')]\n    >>> oset = set('')\n    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}\n    >>> _path__optimal_path(isets, oset, idx_sizes, 5000)\n    [(0, 2), (0, 1)]\n    ")
    
    # Assigning a List to a Name (line 136):
    
    # Assigning a List to a Name (line 136):
    
    # Obtaining an instance of the builtin type 'list' (line 136)
    list_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 136)
    # Adding element type (line 136)
    
    # Obtaining an instance of the builtin type 'tuple' (line 136)
    tuple_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 136)
    # Adding element type (line 136)
    int_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 21), tuple_110, int_111)
    # Adding element type (line 136)
    
    # Obtaining an instance of the builtin type 'list' (line 136)
    list_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 136)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 21), tuple_110, list_112)
    # Adding element type (line 136)
    # Getting the type of 'input_sets' (line 136)
    input_sets_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 28), 'input_sets')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 21), tuple_110, input_sets_113)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 136, 19), list_109, tuple_110)
    
    # Assigning a type to the variable 'full_results' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'full_results', list_109)
    
    
    # Call to range(...): (line 137)
    # Processing the call arguments (line 137)
    
    # Call to len(...): (line 137)
    # Processing the call arguments (line 137)
    # Getting the type of 'input_sets' (line 137)
    input_sets_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 31), 'input_sets', False)
    # Processing the call keyword arguments (line 137)
    kwargs_117 = {}
    # Getting the type of 'len' (line 137)
    len_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'len', False)
    # Calling len(args, kwargs) (line 137)
    len_call_result_118 = invoke(stypy.reporting.localization.Localization(__file__, 137, 27), len_115, *[input_sets_116], **kwargs_117)
    
    int_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 45), 'int')
    # Applying the binary operator '-' (line 137)
    result_sub_120 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 27), '-', len_call_result_118, int_119)
    
    # Processing the call keyword arguments (line 137)
    kwargs_121 = {}
    # Getting the type of 'range' (line 137)
    range_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 21), 'range', False)
    # Calling range(args, kwargs) (line 137)
    range_call_result_122 = invoke(stypy.reporting.localization.Localization(__file__, 137, 21), range_114, *[result_sub_120], **kwargs_121)
    
    # Testing the type of a for loop iterable (line 137)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 4), range_call_result_122)
    # Getting the type of the for loop variable (line 137)
    for_loop_var_123 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 4), range_call_result_122)
    # Assigning a type to the variable 'iteration' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'iteration', for_loop_var_123)
    # SSA begins for a for statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a List to a Name (line 138):
    
    # Assigning a List to a Name (line 138):
    
    # Obtaining an instance of the builtin type 'list' (line 138)
    list_124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 138)
    
    # Assigning a type to the variable 'iter_results' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'iter_results', list_124)
    
    # Assigning a List to a Name (line 141):
    
    # Assigning a List to a Name (line 141):
    
    # Obtaining an instance of the builtin type 'list' (line 141)
    list_125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 141)
    
    # Assigning a type to the variable 'comb_iter' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'comb_iter', list_125)
    
    
    # Call to range(...): (line 142)
    # Processing the call arguments (line 142)
    
    # Call to len(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'input_sets' (line 142)
    input_sets_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 'input_sets', False)
    # Processing the call keyword arguments (line 142)
    kwargs_129 = {}
    # Getting the type of 'len' (line 142)
    len_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 23), 'len', False)
    # Calling len(args, kwargs) (line 142)
    len_call_result_130 = invoke(stypy.reporting.localization.Localization(__file__, 142, 23), len_127, *[input_sets_128], **kwargs_129)
    
    # Getting the type of 'iteration' (line 142)
    iteration_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 41), 'iteration', False)
    # Applying the binary operator '-' (line 142)
    result_sub_132 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 23), '-', len_call_result_130, iteration_131)
    
    # Processing the call keyword arguments (line 142)
    kwargs_133 = {}
    # Getting the type of 'range' (line 142)
    range_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 17), 'range', False)
    # Calling range(args, kwargs) (line 142)
    range_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 142, 17), range_126, *[result_sub_132], **kwargs_133)
    
    # Testing the type of a for loop iterable (line 142)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 142, 8), range_call_result_134)
    # Getting the type of the for loop variable (line 142)
    for_loop_var_135 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 142, 8), range_call_result_134)
    # Assigning a type to the variable 'x' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'x', for_loop_var_135)
    # SSA begins for a for statement (line 142)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'x' (line 143)
    x_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 27), 'x', False)
    int_138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 31), 'int')
    # Applying the binary operator '+' (line 143)
    result_add_139 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 27), '+', x_137, int_138)
    
    
    # Call to len(...): (line 143)
    # Processing the call arguments (line 143)
    # Getting the type of 'input_sets' (line 143)
    input_sets_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 38), 'input_sets', False)
    # Processing the call keyword arguments (line 143)
    kwargs_142 = {}
    # Getting the type of 'len' (line 143)
    len_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 34), 'len', False)
    # Calling len(args, kwargs) (line 143)
    len_call_result_143 = invoke(stypy.reporting.localization.Localization(__file__, 143, 34), len_140, *[input_sets_141], **kwargs_142)
    
    # Getting the type of 'iteration' (line 143)
    iteration_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 52), 'iteration', False)
    # Applying the binary operator '-' (line 143)
    result_sub_145 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 34), '-', len_call_result_143, iteration_144)
    
    # Processing the call keyword arguments (line 143)
    kwargs_146 = {}
    # Getting the type of 'range' (line 143)
    range_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 21), 'range', False)
    # Calling range(args, kwargs) (line 143)
    range_call_result_147 = invoke(stypy.reporting.localization.Localization(__file__, 143, 21), range_136, *[result_add_139, result_sub_145], **kwargs_146)
    
    # Testing the type of a for loop iterable (line 143)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 143, 12), range_call_result_147)
    # Getting the type of the for loop variable (line 143)
    for_loop_var_148 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 143, 12), range_call_result_147)
    # Assigning a type to the variable 'y' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'y', for_loop_var_148)
    # SSA begins for a for statement (line 143)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 144)
    # Processing the call arguments (line 144)
    
    # Obtaining an instance of the builtin type 'tuple' (line 144)
    tuple_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 144)
    # Adding element type (line 144)
    # Getting the type of 'x' (line 144)
    x_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 34), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 34), tuple_151, x_152)
    # Adding element type (line 144)
    # Getting the type of 'y' (line 144)
    y_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 37), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 34), tuple_151, y_153)
    
    # Processing the call keyword arguments (line 144)
    kwargs_154 = {}
    # Getting the type of 'comb_iter' (line 144)
    comb_iter_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'comb_iter', False)
    # Obtaining the member 'append' of a type (line 144)
    append_150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), comb_iter_149, 'append')
    # Calling append(args, kwargs) (line 144)
    append_call_result_155 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), append_150, *[tuple_151], **kwargs_154)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'full_results' (line 146)
    full_results_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'full_results')
    # Testing the type of a for loop iterable (line 146)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 146, 8), full_results_156)
    # Getting the type of the for loop variable (line 146)
    for_loop_var_157 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 146, 8), full_results_156)
    # Assigning a type to the variable 'curr' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'curr', for_loop_var_157)
    # SSA begins for a for statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Tuple (line 147):
    
    # Assigning a Subscript to a Name (line 147):
    
    # Obtaining the type of the subscript
    int_158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 12), 'int')
    # Getting the type of 'curr' (line 147)
    curr_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 41), 'curr')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), curr_159, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_161 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), getitem___160, int_158)
    
    # Assigning a type to the variable 'tuple_var_assignment_1' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'tuple_var_assignment_1', subscript_call_result_161)
    
    # Assigning a Subscript to a Name (line 147):
    
    # Obtaining the type of the subscript
    int_162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 12), 'int')
    # Getting the type of 'curr' (line 147)
    curr_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 41), 'curr')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), curr_163, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_165 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), getitem___164, int_162)
    
    # Assigning a type to the variable 'tuple_var_assignment_2' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'tuple_var_assignment_2', subscript_call_result_165)
    
    # Assigning a Subscript to a Name (line 147):
    
    # Obtaining the type of the subscript
    int_166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 12), 'int')
    # Getting the type of 'curr' (line 147)
    curr_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 41), 'curr')
    # Obtaining the member '__getitem__' of a type (line 147)
    getitem___168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), curr_167, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 147)
    subscript_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), getitem___168, int_166)
    
    # Assigning a type to the variable 'tuple_var_assignment_3' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'tuple_var_assignment_3', subscript_call_result_169)
    
    # Assigning a Name to a Name (line 147):
    # Getting the type of 'tuple_var_assignment_1' (line 147)
    tuple_var_assignment_1_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'tuple_var_assignment_1')
    # Assigning a type to the variable 'cost' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'cost', tuple_var_assignment_1_170)
    
    # Assigning a Name to a Name (line 147):
    # Getting the type of 'tuple_var_assignment_2' (line 147)
    tuple_var_assignment_2_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'tuple_var_assignment_2')
    # Assigning a type to the variable 'positions' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 18), 'positions', tuple_var_assignment_2_171)
    
    # Assigning a Name to a Name (line 147):
    # Getting the type of 'tuple_var_assignment_3' (line 147)
    tuple_var_assignment_3_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'tuple_var_assignment_3')
    # Assigning a type to the variable 'remaining' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 29), 'remaining', tuple_var_assignment_3_172)
    
    # Getting the type of 'comb_iter' (line 148)
    comb_iter_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'comb_iter')
    # Testing the type of a for loop iterable (line 148)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 148, 12), comb_iter_173)
    # Getting the type of the for loop variable (line 148)
    for_loop_var_174 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 148, 12), comb_iter_173)
    # Assigning a type to the variable 'con' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'con', for_loop_var_174)
    # SSA begins for a for statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 151):
    
    # Assigning a Call to a Name (line 151):
    
    # Call to _find_contraction(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'con' (line 151)
    con_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 41), 'con', False)
    # Getting the type of 'remaining' (line 151)
    remaining_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 46), 'remaining', False)
    # Getting the type of 'output_set' (line 151)
    output_set_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 57), 'output_set', False)
    # Processing the call keyword arguments (line 151)
    kwargs_179 = {}
    # Getting the type of '_find_contraction' (line 151)
    _find_contraction_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), '_find_contraction', False)
    # Calling _find_contraction(args, kwargs) (line 151)
    _find_contraction_call_result_180 = invoke(stypy.reporting.localization.Localization(__file__, 151, 23), _find_contraction_175, *[con_176, remaining_177, output_set_178], **kwargs_179)
    
    # Assigning a type to the variable 'cont' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 16), 'cont', _find_contraction_call_result_180)
    
    # Assigning a Name to a Tuple (line 152):
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    int_181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 16), 'int')
    # Getting the type of 'cont' (line 152)
    cont_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 72), 'cont')
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), cont_182, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_184 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), getitem___183, int_181)
    
    # Assigning a type to the variable 'tuple_var_assignment_4' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'tuple_var_assignment_4', subscript_call_result_184)
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    int_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 16), 'int')
    # Getting the type of 'cont' (line 152)
    cont_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 72), 'cont')
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), cont_186, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_188 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), getitem___187, int_185)
    
    # Assigning a type to the variable 'tuple_var_assignment_5' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'tuple_var_assignment_5', subscript_call_result_188)
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    int_189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 16), 'int')
    # Getting the type of 'cont' (line 152)
    cont_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 72), 'cont')
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), cont_190, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_192 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), getitem___191, int_189)
    
    # Assigning a type to the variable 'tuple_var_assignment_6' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'tuple_var_assignment_6', subscript_call_result_192)
    
    # Assigning a Subscript to a Name (line 152):
    
    # Obtaining the type of the subscript
    int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 16), 'int')
    # Getting the type of 'cont' (line 152)
    cont_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 72), 'cont')
    # Obtaining the member '__getitem__' of a type (line 152)
    getitem___195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 16), cont_194, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
    subscript_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 152, 16), getitem___195, int_193)
    
    # Assigning a type to the variable 'tuple_var_assignment_7' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'tuple_var_assignment_7', subscript_call_result_196)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_var_assignment_4' (line 152)
    tuple_var_assignment_4_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'tuple_var_assignment_4')
    # Assigning a type to the variable 'new_result' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'new_result', tuple_var_assignment_4_197)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_var_assignment_5' (line 152)
    tuple_var_assignment_5_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'tuple_var_assignment_5')
    # Assigning a type to the variable 'new_input_sets' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 'new_input_sets', tuple_var_assignment_5_198)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_var_assignment_6' (line 152)
    tuple_var_assignment_6_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'tuple_var_assignment_6')
    # Assigning a type to the variable 'idx_removed' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 44), 'idx_removed', tuple_var_assignment_6_199)
    
    # Assigning a Name to a Name (line 152):
    # Getting the type of 'tuple_var_assignment_7' (line 152)
    tuple_var_assignment_7_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'tuple_var_assignment_7')
    # Assigning a type to the variable 'idx_contract' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 57), 'idx_contract', tuple_var_assignment_7_200)
    
    # Assigning a Call to a Name (line 155):
    
    # Assigning a Call to a Name (line 155):
    
    # Call to _compute_size_by_dict(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'new_result' (line 155)
    new_result_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 49), 'new_result', False)
    # Getting the type of 'idx_dict' (line 155)
    idx_dict_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 61), 'idx_dict', False)
    # Processing the call keyword arguments (line 155)
    kwargs_204 = {}
    # Getting the type of '_compute_size_by_dict' (line 155)
    _compute_size_by_dict_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 27), '_compute_size_by_dict', False)
    # Calling _compute_size_by_dict(args, kwargs) (line 155)
    _compute_size_by_dict_call_result_205 = invoke(stypy.reporting.localization.Localization(__file__, 155, 27), _compute_size_by_dict_201, *[new_result_202, idx_dict_203], **kwargs_204)
    
    # Assigning a type to the variable 'new_size' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'new_size', _compute_size_by_dict_call_result_205)
    
    
    # Getting the type of 'new_size' (line 156)
    new_size_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'new_size')
    # Getting the type of 'memory_limit' (line 156)
    memory_limit_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 30), 'memory_limit')
    # Applying the binary operator '>' (line 156)
    result_gt_208 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 19), '>', new_size_206, memory_limit_207)
    
    # Testing the type of an if condition (line 156)
    if_condition_209 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 16), result_gt_208)
    # Assigning a type to the variable 'if_condition_209' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'if_condition_209', if_condition_209)
    # SSA begins for if statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 156)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 160):
    
    # Assigning a Call to a Name (line 160):
    
    # Call to _compute_size_by_dict(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'idx_contract' (line 160)
    idx_contract_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 49), 'idx_contract', False)
    # Getting the type of 'idx_dict' (line 160)
    idx_dict_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 63), 'idx_dict', False)
    # Processing the call keyword arguments (line 160)
    kwargs_213 = {}
    # Getting the type of '_compute_size_by_dict' (line 160)
    _compute_size_by_dict_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 27), '_compute_size_by_dict', False)
    # Calling _compute_size_by_dict(args, kwargs) (line 160)
    _compute_size_by_dict_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 160, 27), _compute_size_by_dict_210, *[idx_contract_211, idx_dict_212], **kwargs_213)
    
    # Assigning a type to the variable 'new_cost' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'new_cost', _compute_size_by_dict_call_result_214)
    
    # Getting the type of 'idx_removed' (line 161)
    idx_removed_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'idx_removed')
    # Testing the type of an if condition (line 161)
    if_condition_216 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 16), idx_removed_215)
    # Assigning a type to the variable 'if_condition_216' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'if_condition_216', if_condition_216)
    # SSA begins for if statement (line 161)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'new_cost' (line 162)
    new_cost_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'new_cost')
    int_218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 32), 'int')
    # Applying the binary operator '*=' (line 162)
    result_imul_219 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 20), '*=', new_cost_217, int_218)
    # Assigning a type to the variable 'new_cost' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'new_cost', result_imul_219)
    
    # SSA join for if statement (line 161)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'new_cost' (line 165)
    new_cost_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'new_cost')
    # Getting the type of 'cost' (line 165)
    cost_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 'cost')
    # Applying the binary operator '+=' (line 165)
    result_iadd_222 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 16), '+=', new_cost_220, cost_221)
    # Assigning a type to the variable 'new_cost' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'new_cost', result_iadd_222)
    
    
    # Assigning a BinOp to a Name (line 166):
    
    # Assigning a BinOp to a Name (line 166):
    # Getting the type of 'positions' (line 166)
    positions_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 26), 'positions')
    
    # Obtaining an instance of the builtin type 'list' (line 166)
    list_224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 166)
    # Adding element type (line 166)
    # Getting the type of 'con' (line 166)
    con_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 39), 'con')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 166, 38), list_224, con_225)
    
    # Applying the binary operator '+' (line 166)
    result_add_226 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 26), '+', positions_223, list_224)
    
    # Assigning a type to the variable 'new_pos' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'new_pos', result_add_226)
    
    # Call to append(...): (line 167)
    # Processing the call arguments (line 167)
    
    # Obtaining an instance of the builtin type 'tuple' (line 167)
    tuple_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 37), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 167)
    # Adding element type (line 167)
    # Getting the type of 'new_cost' (line 167)
    new_cost_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 37), 'new_cost', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 37), tuple_229, new_cost_230)
    # Adding element type (line 167)
    # Getting the type of 'new_pos' (line 167)
    new_pos_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 47), 'new_pos', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 37), tuple_229, new_pos_231)
    # Adding element type (line 167)
    # Getting the type of 'new_input_sets' (line 167)
    new_input_sets_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 56), 'new_input_sets', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 37), tuple_229, new_input_sets_232)
    
    # Processing the call keyword arguments (line 167)
    kwargs_233 = {}
    # Getting the type of 'iter_results' (line 167)
    iter_results_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'iter_results', False)
    # Obtaining the member 'append' of a type (line 167)
    append_228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), iter_results_227, 'append')
    # Calling append(args, kwargs) (line 167)
    append_call_result_234 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), append_228, *[tuple_229], **kwargs_233)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 170):
    
    # Assigning a Name to a Name (line 170):
    # Getting the type of 'iter_results' (line 170)
    iter_results_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'iter_results')
    # Assigning a type to the variable 'full_results' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'full_results', iter_results_235)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'full_results' (line 173)
    full_results_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'full_results', False)
    # Processing the call keyword arguments (line 173)
    kwargs_238 = {}
    # Getting the type of 'len' (line 173)
    len_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 7), 'len', False)
    # Calling len(args, kwargs) (line 173)
    len_call_result_239 = invoke(stypy.reporting.localization.Localization(__file__, 173, 7), len_236, *[full_results_237], **kwargs_238)
    
    int_240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 28), 'int')
    # Applying the binary operator '==' (line 173)
    result_eq_241 = python_operator(stypy.reporting.localization.Localization(__file__, 173, 7), '==', len_call_result_239, int_240)
    
    # Testing the type of an if condition (line 173)
    if_condition_242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 173, 4), result_eq_241)
    # Assigning a type to the variable 'if_condition_242' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'if_condition_242', if_condition_242)
    # SSA begins for if statement (line 173)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 174)
    list_243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 174)
    # Adding element type (line 174)
    
    # Call to tuple(...): (line 174)
    # Processing the call arguments (line 174)
    
    # Call to range(...): (line 174)
    # Processing the call arguments (line 174)
    
    # Call to len(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'input_sets' (line 174)
    input_sets_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 32), 'input_sets', False)
    # Processing the call keyword arguments (line 174)
    kwargs_248 = {}
    # Getting the type of 'len' (line 174)
    len_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 28), 'len', False)
    # Calling len(args, kwargs) (line 174)
    len_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 174, 28), len_246, *[input_sets_247], **kwargs_248)
    
    # Processing the call keyword arguments (line 174)
    kwargs_250 = {}
    # Getting the type of 'range' (line 174)
    range_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 22), 'range', False)
    # Calling range(args, kwargs) (line 174)
    range_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 174, 22), range_245, *[len_call_result_249], **kwargs_250)
    
    # Processing the call keyword arguments (line 174)
    kwargs_252 = {}
    # Getting the type of 'tuple' (line 174)
    tuple_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 174)
    tuple_call_result_253 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), tuple_244, *[range_call_result_251], **kwargs_252)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 174, 15), list_243, tuple_call_result_253)
    
    # Assigning a type to the variable 'stypy_return_type' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'stypy_return_type', list_243)
    # SSA join for if statement (line 173)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 176):
    
    # Assigning a Subscript to a Name (line 176):
    
    # Obtaining the type of the subscript
    int_254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 49), 'int')
    
    # Call to min(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'full_results' (line 176)
    full_results_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'full_results', False)
    # Processing the call keyword arguments (line 176)

    @norecursion
    def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_1'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 176, 33, True)
        # Passed parameters checking function
        _stypy_temp_lambda_1.stypy_localization = localization
        _stypy_temp_lambda_1.stypy_type_of_self = None
        _stypy_temp_lambda_1.stypy_type_store = module_type_store
        _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
        _stypy_temp_lambda_1.stypy_param_names_list = ['x']
        _stypy_temp_lambda_1.stypy_varargs_param_name = None
        _stypy_temp_lambda_1.stypy_kwargs_param_name = None
        _stypy_temp_lambda_1.stypy_call_defaults = defaults
        _stypy_temp_lambda_1.stypy_call_varargs = varargs
        _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_1', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Obtaining the type of the subscript
        int_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 45), 'int')
        # Getting the type of 'x' (line 176)
        x_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 43), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 176)
        getitem___259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 43), x_258, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 176)
        subscript_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 176, 43), getitem___259, int_257)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), 'stypy_return_type', subscript_call_result_260)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_1' in the type store
        # Getting the type of 'stypy_return_type' (line 176)
        stypy_return_type_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_1'
        return stypy_return_type_261

    # Assigning a type to the variable '_stypy_temp_lambda_1' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
    # Getting the type of '_stypy_temp_lambda_1' (line 176)
    _stypy_temp_lambda_1_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), '_stypy_temp_lambda_1')
    keyword_263 = _stypy_temp_lambda_1_262
    kwargs_264 = {'key': keyword_263}
    # Getting the type of 'min' (line 176)
    min_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'min', False)
    # Calling min(args, kwargs) (line 176)
    min_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 176, 11), min_255, *[full_results_256], **kwargs_264)
    
    # Obtaining the member '__getitem__' of a type (line 176)
    getitem___266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 11), min_call_result_265, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 176)
    subscript_call_result_267 = invoke(stypy.reporting.localization.Localization(__file__, 176, 11), getitem___266, int_254)
    
    # Assigning a type to the variable 'path' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'path', subscript_call_result_267)
    # Getting the type of 'path' (line 177)
    path_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'path')
    # Assigning a type to the variable 'stypy_return_type' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'stypy_return_type', path_268)
    
    # ################# End of '_optimal_path(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_optimal_path' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_269)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_optimal_path'
    return stypy_return_type_269

# Assigning a type to the variable '_optimal_path' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), '_optimal_path', _optimal_path)

@norecursion
def _greedy_path(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_greedy_path'
    module_type_store = module_type_store.open_function_context('_greedy_path', 180, 0, False)
    
    # Passed parameters checking function
    _greedy_path.stypy_localization = localization
    _greedy_path.stypy_type_of_self = None
    _greedy_path.stypy_type_store = module_type_store
    _greedy_path.stypy_function_name = '_greedy_path'
    _greedy_path.stypy_param_names_list = ['input_sets', 'output_set', 'idx_dict', 'memory_limit']
    _greedy_path.stypy_varargs_param_name = None
    _greedy_path.stypy_kwargs_param_name = None
    _greedy_path.stypy_call_defaults = defaults
    _greedy_path.stypy_call_varargs = varargs
    _greedy_path.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_greedy_path', ['input_sets', 'output_set', 'idx_dict', 'memory_limit'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_greedy_path', localization, ['input_sets', 'output_set', 'idx_dict', 'memory_limit'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_greedy_path(...)' code ##################

    str_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, (-1)), 'str', "\n    Finds the path by contracting the best pair until the input list is\n    exhausted. The best pair is found by minimizing the tuple\n    ``(-prod(indices_removed), cost)``.  What this amounts to is prioritizing\n    matrix multiplication or inner product operations, then Hadamard like\n    operations, and finally outer operations. Outer products are limited by\n    ``memory_limit``. This algorithm scales cubically with respect to the\n    number of elements in the list ``input_sets``.\n\n    Parameters\n    ----------\n    input_sets : list\n        List of sets that represent the lhs side of the einsum subscript\n    output_set : set\n        Set that represents the rhs side of the overall einsum subscript\n    idx_dict : dictionary\n        Dictionary of index sizes\n    memory_limit_limit : int\n        The maximum number of elements in a temporary array\n\n    Returns\n    -------\n    path : list\n        The greedy contraction order within the memory limit constraint.\n\n    Examples\n    --------\n    >>> isets = [set('abd'), set('ac'), set('bdc')]\n    >>> oset = set('')\n    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}\n    >>> _path__greedy_path(isets, oset, idx_sizes, 5000)\n    [(0, 2), (0, 1)]\n    ")
    
    
    
    # Call to len(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'input_sets' (line 215)
    input_sets_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'input_sets', False)
    # Processing the call keyword arguments (line 215)
    kwargs_273 = {}
    # Getting the type of 'len' (line 215)
    len_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 7), 'len', False)
    # Calling len(args, kwargs) (line 215)
    len_call_result_274 = invoke(stypy.reporting.localization.Localization(__file__, 215, 7), len_271, *[input_sets_272], **kwargs_273)
    
    int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 26), 'int')
    # Applying the binary operator '==' (line 215)
    result_eq_276 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 7), '==', len_call_result_274, int_275)
    
    # Testing the type of an if condition (line 215)
    if_condition_277 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 4), result_eq_276)
    # Assigning a type to the variable 'if_condition_277' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'if_condition_277', if_condition_277)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'list' (line 216)
    list_278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 216)
    # Adding element type (line 216)
    
    # Obtaining an instance of the builtin type 'tuple' (line 216)
    tuple_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 216)
    # Adding element type (line 216)
    int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 17), tuple_279, int_280)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 216, 15), list_278, tuple_279)
    
    # Assigning a type to the variable 'stypy_return_type' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'stypy_return_type', list_278)
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 218):
    
    # Assigning a List to a Name (line 218):
    
    # Obtaining an instance of the builtin type 'list' (line 218)
    list_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 218)
    
    # Assigning a type to the variable 'path' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'path', list_281)
    
    
    # Call to range(...): (line 219)
    # Processing the call arguments (line 219)
    
    # Call to len(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'input_sets' (line 219)
    input_sets_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 31), 'input_sets', False)
    # Processing the call keyword arguments (line 219)
    kwargs_285 = {}
    # Getting the type of 'len' (line 219)
    len_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 27), 'len', False)
    # Calling len(args, kwargs) (line 219)
    len_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 219, 27), len_283, *[input_sets_284], **kwargs_285)
    
    int_287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 45), 'int')
    # Applying the binary operator '-' (line 219)
    result_sub_288 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 27), '-', len_call_result_286, int_287)
    
    # Processing the call keyword arguments (line 219)
    kwargs_289 = {}
    # Getting the type of 'range' (line 219)
    range_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'range', False)
    # Calling range(args, kwargs) (line 219)
    range_call_result_290 = invoke(stypy.reporting.localization.Localization(__file__, 219, 21), range_282, *[result_sub_288], **kwargs_289)
    
    # Testing the type of a for loop iterable (line 219)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 219, 4), range_call_result_290)
    # Getting the type of the for loop variable (line 219)
    for_loop_var_291 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 219, 4), range_call_result_290)
    # Assigning a type to the variable 'iteration' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'iteration', for_loop_var_291)
    # SSA begins for a for statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a List to a Name (line 220):
    
    # Assigning a List to a Name (line 220):
    
    # Obtaining an instance of the builtin type 'list' (line 220)
    list_292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 220)
    
    # Assigning a type to the variable 'iteration_results' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'iteration_results', list_292)
    
    # Assigning a List to a Name (line 221):
    
    # Assigning a List to a Name (line 221):
    
    # Obtaining an instance of the builtin type 'list' (line 221)
    list_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 221)
    
    # Assigning a type to the variable 'comb_iter' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'comb_iter', list_293)
    
    
    # Call to range(...): (line 224)
    # Processing the call arguments (line 224)
    
    # Call to len(...): (line 224)
    # Processing the call arguments (line 224)
    # Getting the type of 'input_sets' (line 224)
    input_sets_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'input_sets', False)
    # Processing the call keyword arguments (line 224)
    kwargs_297 = {}
    # Getting the type of 'len' (line 224)
    len_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 23), 'len', False)
    # Calling len(args, kwargs) (line 224)
    len_call_result_298 = invoke(stypy.reporting.localization.Localization(__file__, 224, 23), len_295, *[input_sets_296], **kwargs_297)
    
    # Processing the call keyword arguments (line 224)
    kwargs_299 = {}
    # Getting the type of 'range' (line 224)
    range_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 17), 'range', False)
    # Calling range(args, kwargs) (line 224)
    range_call_result_300 = invoke(stypy.reporting.localization.Localization(__file__, 224, 17), range_294, *[len_call_result_298], **kwargs_299)
    
    # Testing the type of a for loop iterable (line 224)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 224, 8), range_call_result_300)
    # Getting the type of the for loop variable (line 224)
    for_loop_var_301 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 224, 8), range_call_result_300)
    # Assigning a type to the variable 'x' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'x', for_loop_var_301)
    # SSA begins for a for statement (line 224)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'x' (line 225)
    x_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 27), 'x', False)
    int_304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 31), 'int')
    # Applying the binary operator '+' (line 225)
    result_add_305 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 27), '+', x_303, int_304)
    
    
    # Call to len(...): (line 225)
    # Processing the call arguments (line 225)
    # Getting the type of 'input_sets' (line 225)
    input_sets_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 38), 'input_sets', False)
    # Processing the call keyword arguments (line 225)
    kwargs_308 = {}
    # Getting the type of 'len' (line 225)
    len_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 34), 'len', False)
    # Calling len(args, kwargs) (line 225)
    len_call_result_309 = invoke(stypy.reporting.localization.Localization(__file__, 225, 34), len_306, *[input_sets_307], **kwargs_308)
    
    # Processing the call keyword arguments (line 225)
    kwargs_310 = {}
    # Getting the type of 'range' (line 225)
    range_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 21), 'range', False)
    # Calling range(args, kwargs) (line 225)
    range_call_result_311 = invoke(stypy.reporting.localization.Localization(__file__, 225, 21), range_302, *[result_add_305, len_call_result_309], **kwargs_310)
    
    # Testing the type of a for loop iterable (line 225)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 225, 12), range_call_result_311)
    # Getting the type of the for loop variable (line 225)
    for_loop_var_312 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 225, 12), range_call_result_311)
    # Assigning a type to the variable 'y' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 12), 'y', for_loop_var_312)
    # SSA begins for a for statement (line 225)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 226)
    # Processing the call arguments (line 226)
    
    # Obtaining an instance of the builtin type 'tuple' (line 226)
    tuple_315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 226)
    # Adding element type (line 226)
    # Getting the type of 'x' (line 226)
    x_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 34), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 34), tuple_315, x_316)
    # Adding element type (line 226)
    # Getting the type of 'y' (line 226)
    y_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 37), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 34), tuple_315, y_317)
    
    # Processing the call keyword arguments (line 226)
    kwargs_318 = {}
    # Getting the type of 'comb_iter' (line 226)
    comb_iter_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'comb_iter', False)
    # Obtaining the member 'append' of a type (line 226)
    append_314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 16), comb_iter_313, 'append')
    # Calling append(args, kwargs) (line 226)
    append_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 226, 16), append_314, *[tuple_315], **kwargs_318)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'comb_iter' (line 228)
    comb_iter_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 25), 'comb_iter')
    # Testing the type of a for loop iterable (line 228)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 228, 8), comb_iter_320)
    # Getting the type of the for loop variable (line 228)
    for_loop_var_321 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 228, 8), comb_iter_320)
    # Assigning a type to the variable 'positions' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'positions', for_loop_var_321)
    # SSA begins for a for statement (line 228)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 231):
    
    # Assigning a Call to a Name (line 231):
    
    # Call to _find_contraction(...): (line 231)
    # Processing the call arguments (line 231)
    # Getting the type of 'positions' (line 231)
    positions_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 41), 'positions', False)
    # Getting the type of 'input_sets' (line 231)
    input_sets_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 52), 'input_sets', False)
    # Getting the type of 'output_set' (line 231)
    output_set_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 64), 'output_set', False)
    # Processing the call keyword arguments (line 231)
    kwargs_326 = {}
    # Getting the type of '_find_contraction' (line 231)
    _find_contraction_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), '_find_contraction', False)
    # Calling _find_contraction(args, kwargs) (line 231)
    _find_contraction_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 231, 23), _find_contraction_322, *[positions_323, input_sets_324, output_set_325], **kwargs_326)
    
    # Assigning a type to the variable 'contract' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'contract', _find_contraction_call_result_327)
    
    # Assigning a Name to a Tuple (line 232):
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 12), 'int')
    # Getting the type of 'contract' (line 232)
    contract_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 68), 'contract')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), contract_329, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), getitem___330, int_328)
    
    # Assigning a type to the variable 'tuple_var_assignment_8' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'tuple_var_assignment_8', subscript_call_result_331)
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 12), 'int')
    # Getting the type of 'contract' (line 232)
    contract_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 68), 'contract')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), contract_333, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_335 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), getitem___334, int_332)
    
    # Assigning a type to the variable 'tuple_var_assignment_9' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'tuple_var_assignment_9', subscript_call_result_335)
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 12), 'int')
    # Getting the type of 'contract' (line 232)
    contract_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 68), 'contract')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), contract_337, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_339 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), getitem___338, int_336)
    
    # Assigning a type to the variable 'tuple_var_assignment_10' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'tuple_var_assignment_10', subscript_call_result_339)
    
    # Assigning a Subscript to a Name (line 232):
    
    # Obtaining the type of the subscript
    int_340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 12), 'int')
    # Getting the type of 'contract' (line 232)
    contract_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 68), 'contract')
    # Obtaining the member '__getitem__' of a type (line 232)
    getitem___342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), contract_341, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 232)
    subscript_call_result_343 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), getitem___342, int_340)
    
    # Assigning a type to the variable 'tuple_var_assignment_11' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'tuple_var_assignment_11', subscript_call_result_343)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_8' (line 232)
    tuple_var_assignment_8_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'tuple_var_assignment_8')
    # Assigning a type to the variable 'idx_result' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'idx_result', tuple_var_assignment_8_344)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_9' (line 232)
    tuple_var_assignment_9_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'tuple_var_assignment_9')
    # Assigning a type to the variable 'new_input_sets' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'new_input_sets', tuple_var_assignment_9_345)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_10' (line 232)
    tuple_var_assignment_10_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'tuple_var_assignment_10')
    # Assigning a type to the variable 'idx_removed' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 40), 'idx_removed', tuple_var_assignment_10_346)
    
    # Assigning a Name to a Name (line 232):
    # Getting the type of 'tuple_var_assignment_11' (line 232)
    tuple_var_assignment_11_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'tuple_var_assignment_11')
    # Assigning a type to the variable 'idx_contract' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 53), 'idx_contract', tuple_var_assignment_11_347)
    
    
    
    # Call to _compute_size_by_dict(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'idx_result' (line 235)
    idx_result_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 37), 'idx_result', False)
    # Getting the type of 'idx_dict' (line 235)
    idx_dict_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 49), 'idx_dict', False)
    # Processing the call keyword arguments (line 235)
    kwargs_351 = {}
    # Getting the type of '_compute_size_by_dict' (line 235)
    _compute_size_by_dict_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), '_compute_size_by_dict', False)
    # Calling _compute_size_by_dict(args, kwargs) (line 235)
    _compute_size_by_dict_call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 235, 15), _compute_size_by_dict_348, *[idx_result_349, idx_dict_350], **kwargs_351)
    
    # Getting the type of 'memory_limit' (line 235)
    memory_limit_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 61), 'memory_limit')
    # Applying the binary operator '>' (line 235)
    result_gt_354 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 15), '>', _compute_size_by_dict_call_result_352, memory_limit_353)
    
    # Testing the type of an if condition (line 235)
    if_condition_355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 12), result_gt_354)
    # Assigning a type to the variable 'if_condition_355' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 12), 'if_condition_355', if_condition_355)
    # SSA begins for if statement (line 235)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 235)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 239):
    
    # Assigning a Call to a Name (line 239):
    
    # Call to _compute_size_by_dict(...): (line 239)
    # Processing the call arguments (line 239)
    # Getting the type of 'idx_removed' (line 239)
    idx_removed_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 49), 'idx_removed', False)
    # Getting the type of 'idx_dict' (line 239)
    idx_dict_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 62), 'idx_dict', False)
    # Processing the call keyword arguments (line 239)
    kwargs_359 = {}
    # Getting the type of '_compute_size_by_dict' (line 239)
    _compute_size_by_dict_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), '_compute_size_by_dict', False)
    # Calling _compute_size_by_dict(args, kwargs) (line 239)
    _compute_size_by_dict_call_result_360 = invoke(stypy.reporting.localization.Localization(__file__, 239, 27), _compute_size_by_dict_356, *[idx_removed_357, idx_dict_358], **kwargs_359)
    
    # Assigning a type to the variable 'removed_size' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'removed_size', _compute_size_by_dict_call_result_360)
    
    # Assigning a Call to a Name (line 240):
    
    # Assigning a Call to a Name (line 240):
    
    # Call to _compute_size_by_dict(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'idx_contract' (line 240)
    idx_contract_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 41), 'idx_contract', False)
    # Getting the type of 'idx_dict' (line 240)
    idx_dict_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 55), 'idx_dict', False)
    # Processing the call keyword arguments (line 240)
    kwargs_364 = {}
    # Getting the type of '_compute_size_by_dict' (line 240)
    _compute_size_by_dict_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), '_compute_size_by_dict', False)
    # Calling _compute_size_by_dict(args, kwargs) (line 240)
    _compute_size_by_dict_call_result_365 = invoke(stypy.reporting.localization.Localization(__file__, 240, 19), _compute_size_by_dict_361, *[idx_contract_362, idx_dict_363], **kwargs_364)
    
    # Assigning a type to the variable 'cost' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'cost', _compute_size_by_dict_call_result_365)
    
    # Assigning a Tuple to a Name (line 241):
    
    # Assigning a Tuple to a Name (line 241):
    
    # Obtaining an instance of the builtin type 'tuple' (line 241)
    tuple_366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 241)
    # Adding element type (line 241)
    
    # Getting the type of 'removed_size' (line 241)
    removed_size_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'removed_size')
    # Applying the 'usub' unary operator (line 241)
    result___neg___368 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 20), 'usub', removed_size_367)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 20), tuple_366, result___neg___368)
    # Adding element type (line 241)
    # Getting the type of 'cost' (line 241)
    cost_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 35), 'cost')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 241, 20), tuple_366, cost_369)
    
    # Assigning a type to the variable 'sort' (line 241)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'sort', tuple_366)
    
    # Call to append(...): (line 244)
    # Processing the call arguments (line 244)
    
    # Obtaining an instance of the builtin type 'list' (line 244)
    list_372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 37), 'list')
    # Adding type elements to the builtin type 'list' instance (line 244)
    # Adding element type (line 244)
    # Getting the type of 'sort' (line 244)
    sort_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 38), 'sort', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 37), list_372, sort_373)
    # Adding element type (line 244)
    # Getting the type of 'positions' (line 244)
    positions_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 44), 'positions', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 37), list_372, positions_374)
    # Adding element type (line 244)
    # Getting the type of 'new_input_sets' (line 244)
    new_input_sets_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 55), 'new_input_sets', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 244, 37), list_372, new_input_sets_375)
    
    # Processing the call keyword arguments (line 244)
    kwargs_376 = {}
    # Getting the type of 'iteration_results' (line 244)
    iteration_results_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'iteration_results', False)
    # Obtaining the member 'append' of a type (line 244)
    append_371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), iteration_results_370, 'append')
    # Calling append(args, kwargs) (line 244)
    append_call_result_377 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), append_371, *[list_372], **kwargs_376)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'iteration_results' (line 247)
    iteration_results_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 15), 'iteration_results', False)
    # Processing the call keyword arguments (line 247)
    kwargs_380 = {}
    # Getting the type of 'len' (line 247)
    len_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 11), 'len', False)
    # Calling len(args, kwargs) (line 247)
    len_call_result_381 = invoke(stypy.reporting.localization.Localization(__file__, 247, 11), len_378, *[iteration_results_379], **kwargs_380)
    
    int_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 37), 'int')
    # Applying the binary operator '==' (line 247)
    result_eq_383 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 11), '==', len_call_result_381, int_382)
    
    # Testing the type of an if condition (line 247)
    if_condition_384 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 247, 8), result_eq_383)
    # Assigning a type to the variable 'if_condition_384' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'if_condition_384', if_condition_384)
    # SSA begins for if statement (line 247)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 248)
    # Processing the call arguments (line 248)
    
    # Call to tuple(...): (line 248)
    # Processing the call arguments (line 248)
    
    # Call to range(...): (line 248)
    # Processing the call arguments (line 248)
    
    # Call to len(...): (line 248)
    # Processing the call arguments (line 248)
    # Getting the type of 'input_sets' (line 248)
    input_sets_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 40), 'input_sets', False)
    # Processing the call keyword arguments (line 248)
    kwargs_391 = {}
    # Getting the type of 'len' (line 248)
    len_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 36), 'len', False)
    # Calling len(args, kwargs) (line 248)
    len_call_result_392 = invoke(stypy.reporting.localization.Localization(__file__, 248, 36), len_389, *[input_sets_390], **kwargs_391)
    
    # Processing the call keyword arguments (line 248)
    kwargs_393 = {}
    # Getting the type of 'range' (line 248)
    range_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 30), 'range', False)
    # Calling range(args, kwargs) (line 248)
    range_call_result_394 = invoke(stypy.reporting.localization.Localization(__file__, 248, 30), range_388, *[len_call_result_392], **kwargs_393)
    
    # Processing the call keyword arguments (line 248)
    kwargs_395 = {}
    # Getting the type of 'tuple' (line 248)
    tuple_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'tuple', False)
    # Calling tuple(args, kwargs) (line 248)
    tuple_call_result_396 = invoke(stypy.reporting.localization.Localization(__file__, 248, 24), tuple_387, *[range_call_result_394], **kwargs_395)
    
    # Processing the call keyword arguments (line 248)
    kwargs_397 = {}
    # Getting the type of 'path' (line 248)
    path_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'path', False)
    # Obtaining the member 'append' of a type (line 248)
    append_386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 12), path_385, 'append')
    # Calling append(args, kwargs) (line 248)
    append_call_result_398 = invoke(stypy.reporting.localization.Localization(__file__, 248, 12), append_386, *[tuple_call_result_396], **kwargs_397)
    
    # SSA join for if statement (line 247)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 252):
    
    # Assigning a Call to a Name (line 252):
    
    # Call to min(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'iteration_results' (line 252)
    iteration_results_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 19), 'iteration_results', False)
    # Processing the call keyword arguments (line 252)

    @norecursion
    def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_2'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 252, 42, True)
        # Passed parameters checking function
        _stypy_temp_lambda_2.stypy_localization = localization
        _stypy_temp_lambda_2.stypy_type_of_self = None
        _stypy_temp_lambda_2.stypy_type_store = module_type_store
        _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
        _stypy_temp_lambda_2.stypy_param_names_list = ['x']
        _stypy_temp_lambda_2.stypy_varargs_param_name = None
        _stypy_temp_lambda_2.stypy_kwargs_param_name = None
        _stypy_temp_lambda_2.stypy_call_defaults = defaults
        _stypy_temp_lambda_2.stypy_call_varargs = varargs
        _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_2', ['x'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Obtaining the type of the subscript
        int_401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 54), 'int')
        # Getting the type of 'x' (line 252)
        x_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 52), 'x', False)
        # Obtaining the member '__getitem__' of a type (line 252)
        getitem___403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 52), x_402, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 252)
        subscript_call_result_404 = invoke(stypy.reporting.localization.Localization(__file__, 252, 52), getitem___403, int_401)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 42), 'stypy_return_type', subscript_call_result_404)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_2' in the type store
        # Getting the type of 'stypy_return_type' (line 252)
        stypy_return_type_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 42), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_405)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_2'
        return stypy_return_type_405

    # Assigning a type to the variable '_stypy_temp_lambda_2' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 42), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
    # Getting the type of '_stypy_temp_lambda_2' (line 252)
    _stypy_temp_lambda_2_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 42), '_stypy_temp_lambda_2')
    keyword_407 = _stypy_temp_lambda_2_406
    kwargs_408 = {'key': keyword_407}
    # Getting the type of 'min' (line 252)
    min_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 15), 'min', False)
    # Calling min(args, kwargs) (line 252)
    min_call_result_409 = invoke(stypy.reporting.localization.Localization(__file__, 252, 15), min_399, *[iteration_results_400], **kwargs_408)
    
    # Assigning a type to the variable 'best' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'best', min_call_result_409)
    
    # Call to append(...): (line 253)
    # Processing the call arguments (line 253)
    
    # Obtaining the type of the subscript
    int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 25), 'int')
    # Getting the type of 'best' (line 253)
    best_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 20), 'best', False)
    # Obtaining the member '__getitem__' of a type (line 253)
    getitem___414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 20), best_413, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 253)
    subscript_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 253, 20), getitem___414, int_412)
    
    # Processing the call keyword arguments (line 253)
    kwargs_416 = {}
    # Getting the type of 'path' (line 253)
    path_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'path', False)
    # Obtaining the member 'append' of a type (line 253)
    append_411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 8), path_410, 'append')
    # Calling append(args, kwargs) (line 253)
    append_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 253, 8), append_411, *[subscript_call_result_415], **kwargs_416)
    
    
    # Assigning a Subscript to a Name (line 254):
    
    # Assigning a Subscript to a Name (line 254):
    
    # Obtaining the type of the subscript
    int_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 26), 'int')
    # Getting the type of 'best' (line 254)
    best_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 21), 'best')
    # Obtaining the member '__getitem__' of a type (line 254)
    getitem___420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 21), best_419, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 254)
    subscript_call_result_421 = invoke(stypy.reporting.localization.Localization(__file__, 254, 21), getitem___420, int_418)
    
    # Assigning a type to the variable 'input_sets' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'input_sets', subscript_call_result_421)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'path' (line 256)
    path_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 11), 'path')
    # Assigning a type to the variable 'stypy_return_type' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type', path_422)
    
    # ################# End of '_greedy_path(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_greedy_path' in the type store
    # Getting the type of 'stypy_return_type' (line 180)
    stypy_return_type_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_423)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_greedy_path'
    return stypy_return_type_423

# Assigning a type to the variable '_greedy_path' (line 180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), '_greedy_path', _greedy_path)

@norecursion
def _parse_einsum_input(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_parse_einsum_input'
    module_type_store = module_type_store.open_function_context('_parse_einsum_input', 259, 0, False)
    
    # Passed parameters checking function
    _parse_einsum_input.stypy_localization = localization
    _parse_einsum_input.stypy_type_of_self = None
    _parse_einsum_input.stypy_type_store = module_type_store
    _parse_einsum_input.stypy_function_name = '_parse_einsum_input'
    _parse_einsum_input.stypy_param_names_list = ['operands']
    _parse_einsum_input.stypy_varargs_param_name = None
    _parse_einsum_input.stypy_kwargs_param_name = None
    _parse_einsum_input.stypy_call_defaults = defaults
    _parse_einsum_input.stypy_call_varargs = varargs
    _parse_einsum_input.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_parse_einsum_input', ['operands'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_parse_einsum_input', localization, ['operands'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_parse_einsum_input(...)' code ##################

    str_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, (-1)), 'str', "\n    A reproduction of einsum c side einsum parsing in python.\n\n    Returns\n    -------\n    input_strings : str\n        Parsed input strings\n    output_string : str\n        Parsed output string\n    operands : list of array_like\n        The operands to use in the numpy contraction\n\n    Examples\n    --------\n    The operand list is simplified to reduce printing:\n\n    >>> a = np.random.rand(4, 4)\n    >>> b = np.random.rand(4, 4, 4)\n    >>> __parse_einsum_input(('...a,...a->...', a, b))\n    ('za,xza', 'xz', [a, b])\n\n    >>> __parse_einsum_input((a, [Ellipsis, 0], b, [Ellipsis, 0]))\n    ('za,xza', 'xz', [a, b])\n    ")
    
    
    
    # Call to len(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'operands' (line 285)
    operands_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), 'operands', False)
    # Processing the call keyword arguments (line 285)
    kwargs_427 = {}
    # Getting the type of 'len' (line 285)
    len_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 7), 'len', False)
    # Calling len(args, kwargs) (line 285)
    len_call_result_428 = invoke(stypy.reporting.localization.Localization(__file__, 285, 7), len_425, *[operands_426], **kwargs_427)
    
    int_429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 24), 'int')
    # Applying the binary operator '==' (line 285)
    result_eq_430 = python_operator(stypy.reporting.localization.Localization(__file__, 285, 7), '==', len_call_result_428, int_429)
    
    # Testing the type of an if condition (line 285)
    if_condition_431 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 285, 4), result_eq_430)
    # Assigning a type to the variable 'if_condition_431' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'if_condition_431', if_condition_431)
    # SSA begins for if statement (line 285)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 286)
    # Processing the call arguments (line 286)
    str_433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 25), 'str', 'No input operands')
    # Processing the call keyword arguments (line 286)
    kwargs_434 = {}
    # Getting the type of 'ValueError' (line 286)
    ValueError_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 286)
    ValueError_call_result_435 = invoke(stypy.reporting.localization.Localization(__file__, 286, 14), ValueError_432, *[str_433], **kwargs_434)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 286, 8), ValueError_call_result_435, 'raise parameter', BaseException)
    # SSA join for if statement (line 285)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 288)
    # Getting the type of 'str' (line 288)
    str_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 31), 'str')
    
    # Obtaining the type of the subscript
    int_437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 27), 'int')
    # Getting the type of 'operands' (line 288)
    operands_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 18), 'operands')
    # Obtaining the member '__getitem__' of a type (line 288)
    getitem___439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 18), operands_438, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 288)
    subscript_call_result_440 = invoke(stypy.reporting.localization.Localization(__file__, 288, 18), getitem___439, int_437)
    
    
    (may_be_441, more_types_in_union_442) = may_be_subtype(str_436, subscript_call_result_440)

    if may_be_441:

        if more_types_in_union_442:
            # Runtime conditional SSA (line 288)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 289):
        
        # Assigning a Call to a Name (line 289):
        
        # Call to replace(...): (line 289)
        # Processing the call arguments (line 289)
        str_448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 41), 'str', ' ')
        str_449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 46), 'str', '')
        # Processing the call keyword arguments (line 289)
        kwargs_450 = {}
        
        # Obtaining the type of the subscript
        int_443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 30), 'int')
        # Getting the type of 'operands' (line 289)
        operands_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 21), 'operands', False)
        # Obtaining the member '__getitem__' of a type (line 289)
        getitem___445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 21), operands_444, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 289)
        subscript_call_result_446 = invoke(stypy.reporting.localization.Localization(__file__, 289, 21), getitem___445, int_443)
        
        # Obtaining the member 'replace' of a type (line 289)
        replace_447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 21), subscript_call_result_446, 'replace')
        # Calling replace(args, kwargs) (line 289)
        replace_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 289, 21), replace_447, *[str_448, str_449], **kwargs_450)
        
        # Assigning a type to the variable 'subscripts' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'subscripts', replace_call_result_451)
        
        # Assigning a ListComp to a Name (line 290):
        
        # Assigning a ListComp to a Name (line 290):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining the type of the subscript
        int_456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 52), 'int')
        slice_457 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 290, 43), int_456, None, None)
        # Getting the type of 'operands' (line 290)
        operands_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 43), 'operands')
        # Obtaining the member '__getitem__' of a type (line 290)
        getitem___459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 43), operands_458, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 290)
        subscript_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 290, 43), getitem___459, slice_457)
        
        comprehension_461 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 20), subscript_call_result_460)
        # Assigning a type to the variable 'v' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 20), 'v', comprehension_461)
        
        # Call to asanyarray(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'v' (line 290)
        v_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 31), 'v', False)
        # Processing the call keyword arguments (line 290)
        kwargs_454 = {}
        # Getting the type of 'asanyarray' (line 290)
        asanyarray_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 20), 'asanyarray', False)
        # Calling asanyarray(args, kwargs) (line 290)
        asanyarray_call_result_455 = invoke(stypy.reporting.localization.Localization(__file__, 290, 20), asanyarray_452, *[v_453], **kwargs_454)
        
        list_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 290, 20), list_462, asanyarray_call_result_455)
        # Assigning a type to the variable 'operands' (line 290)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'operands', list_462)
        
        # Getting the type of 'subscripts' (line 293)
        subscripts_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 17), 'subscripts')
        # Testing the type of a for loop iterable (line 293)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 293, 8), subscripts_463)
        # Getting the type of the for loop variable (line 293)
        for_loop_var_464 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 293, 8), subscripts_463)
        # Assigning a type to the variable 's' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 's', for_loop_var_464)
        # SSA begins for a for statement (line 293)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 's' (line 294)
        s_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 15), 's')
        str_466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 20), 'str', '.,->')
        # Applying the binary operator 'in' (line 294)
        result_contains_467 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 15), 'in', s_465, str_466)
        
        # Testing the type of an if condition (line 294)
        if_condition_468 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 12), result_contains_467)
        # Assigning a type to the variable 'if_condition_468' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'if_condition_468', if_condition_468)
        # SSA begins for if statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 294)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 's' (line 296)
        s_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 15), 's')
        # Getting the type of 'einsum_symbols' (line 296)
        einsum_symbols_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 24), 'einsum_symbols')
        # Applying the binary operator 'notin' (line 296)
        result_contains_471 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 15), 'notin', s_469, einsum_symbols_470)
        
        # Testing the type of an if condition (line 296)
        if_condition_472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 296, 12), result_contains_471)
        # Assigning a type to the variable 'if_condition_472' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'if_condition_472', if_condition_472)
        # SSA begins for if statement (line 296)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 297)
        # Processing the call arguments (line 297)
        str_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 33), 'str', 'Character %s is not a valid symbol.')
        # Getting the type of 's' (line 297)
        s_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 73), 's', False)
        # Applying the binary operator '%' (line 297)
        result_mod_476 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 33), '%', str_474, s_475)
        
        # Processing the call keyword arguments (line 297)
        kwargs_477 = {}
        # Getting the type of 'ValueError' (line 297)
        ValueError_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 297)
        ValueError_call_result_478 = invoke(stypy.reporting.localization.Localization(__file__, 297, 22), ValueError_473, *[result_mod_476], **kwargs_477)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 297, 16), ValueError_call_result_478, 'raise parameter', BaseException)
        # SSA join for if statement (line 296)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_442:
            # Runtime conditional SSA for else branch (line 288)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_441) or more_types_in_union_442):
        
        # Assigning a Call to a Name (line 300):
        
        # Assigning a Call to a Name (line 300):
        
        # Call to list(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 'operands' (line 300)
        operands_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 28), 'operands', False)
        # Processing the call keyword arguments (line 300)
        kwargs_481 = {}
        # Getting the type of 'list' (line 300)
        list_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 23), 'list', False)
        # Calling list(args, kwargs) (line 300)
        list_call_result_482 = invoke(stypy.reporting.localization.Localization(__file__, 300, 23), list_479, *[operands_480], **kwargs_481)
        
        # Assigning a type to the variable 'tmp_operands' (line 300)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'tmp_operands', list_call_result_482)
        
        # Assigning a List to a Name (line 301):
        
        # Assigning a List to a Name (line 301):
        
        # Obtaining an instance of the builtin type 'list' (line 301)
        list_483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 301)
        
        # Assigning a type to the variable 'operand_list' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'operand_list', list_483)
        
        # Assigning a List to a Name (line 302):
        
        # Assigning a List to a Name (line 302):
        
        # Obtaining an instance of the builtin type 'list' (line 302)
        list_484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 302)
        
        # Assigning a type to the variable 'subscript_list' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'subscript_list', list_484)
        
        
        # Call to range(...): (line 303)
        # Processing the call arguments (line 303)
        
        # Call to len(...): (line 303)
        # Processing the call arguments (line 303)
        # Getting the type of 'operands' (line 303)
        operands_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 27), 'operands', False)
        # Processing the call keyword arguments (line 303)
        kwargs_488 = {}
        # Getting the type of 'len' (line 303)
        len_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'len', False)
        # Calling len(args, kwargs) (line 303)
        len_call_result_489 = invoke(stypy.reporting.localization.Localization(__file__, 303, 23), len_486, *[operands_487], **kwargs_488)
        
        int_490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 40), 'int')
        # Applying the binary operator '//' (line 303)
        result_floordiv_491 = python_operator(stypy.reporting.localization.Localization(__file__, 303, 23), '//', len_call_result_489, int_490)
        
        # Processing the call keyword arguments (line 303)
        kwargs_492 = {}
        # Getting the type of 'range' (line 303)
        range_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 17), 'range', False)
        # Calling range(args, kwargs) (line 303)
        range_call_result_493 = invoke(stypy.reporting.localization.Localization(__file__, 303, 17), range_485, *[result_floordiv_491], **kwargs_492)
        
        # Testing the type of a for loop iterable (line 303)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 303, 8), range_call_result_493)
        # Getting the type of the for loop variable (line 303)
        for_loop_var_494 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 303, 8), range_call_result_493)
        # Assigning a type to the variable 'p' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'p', for_loop_var_494)
        # SSA begins for a for statement (line 303)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 304)
        # Processing the call arguments (line 304)
        
        # Call to pop(...): (line 304)
        # Processing the call arguments (line 304)
        int_499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 49), 'int')
        # Processing the call keyword arguments (line 304)
        kwargs_500 = {}
        # Getting the type of 'tmp_operands' (line 304)
        tmp_operands_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 32), 'tmp_operands', False)
        # Obtaining the member 'pop' of a type (line 304)
        pop_498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 32), tmp_operands_497, 'pop')
        # Calling pop(args, kwargs) (line 304)
        pop_call_result_501 = invoke(stypy.reporting.localization.Localization(__file__, 304, 32), pop_498, *[int_499], **kwargs_500)
        
        # Processing the call keyword arguments (line 304)
        kwargs_502 = {}
        # Getting the type of 'operand_list' (line 304)
        operand_list_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 12), 'operand_list', False)
        # Obtaining the member 'append' of a type (line 304)
        append_496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 12), operand_list_495, 'append')
        # Calling append(args, kwargs) (line 304)
        append_call_result_503 = invoke(stypy.reporting.localization.Localization(__file__, 304, 12), append_496, *[pop_call_result_501], **kwargs_502)
        
        
        # Call to append(...): (line 305)
        # Processing the call arguments (line 305)
        
        # Call to pop(...): (line 305)
        # Processing the call arguments (line 305)
        int_508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 51), 'int')
        # Processing the call keyword arguments (line 305)
        kwargs_509 = {}
        # Getting the type of 'tmp_operands' (line 305)
        tmp_operands_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 34), 'tmp_operands', False)
        # Obtaining the member 'pop' of a type (line 305)
        pop_507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 34), tmp_operands_506, 'pop')
        # Calling pop(args, kwargs) (line 305)
        pop_call_result_510 = invoke(stypy.reporting.localization.Localization(__file__, 305, 34), pop_507, *[int_508], **kwargs_509)
        
        # Processing the call keyword arguments (line 305)
        kwargs_511 = {}
        # Getting the type of 'subscript_list' (line 305)
        subscript_list_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 12), 'subscript_list', False)
        # Obtaining the member 'append' of a type (line 305)
        append_505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 12), subscript_list_504, 'append')
        # Calling append(args, kwargs) (line 305)
        append_call_result_512 = invoke(stypy.reporting.localization.Localization(__file__, 305, 12), append_505, *[pop_call_result_510], **kwargs_511)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a IfExp to a Name (line 307):
        
        # Assigning a IfExp to a Name (line 307):
        
        
        # Call to len(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 'tmp_operands' (line 307)
        tmp_operands_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 46), 'tmp_operands', False)
        # Processing the call keyword arguments (line 307)
        kwargs_515 = {}
        # Getting the type of 'len' (line 307)
        len_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 42), 'len', False)
        # Calling len(args, kwargs) (line 307)
        len_call_result_516 = invoke(stypy.reporting.localization.Localization(__file__, 307, 42), len_513, *[tmp_operands_514], **kwargs_515)
        
        # Testing the type of an if expression (line 307)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 307, 22), len_call_result_516)
        # SSA begins for if expression (line 307)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Obtaining the type of the subscript
        int_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 35), 'int')
        # Getting the type of 'tmp_operands' (line 307)
        tmp_operands_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 22), 'tmp_operands')
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 22), tmp_operands_518, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_520 = invoke(stypy.reporting.localization.Localization(__file__, 307, 22), getitem___519, int_517)
        
        # SSA branch for the else part of an if expression (line 307)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'None' (line 307)
        None_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 65), 'None')
        # SSA join for if expression (line 307)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_522 = union_type.UnionType.add(subscript_call_result_520, None_521)
        
        # Assigning a type to the variable 'output_list' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'output_list', if_exp_522)
        
        # Assigning a ListComp to a Name (line 308):
        
        # Assigning a ListComp to a Name (line 308):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'operand_list' (line 308)
        operand_list_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 43), 'operand_list')
        comprehension_528 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 20), operand_list_527)
        # Assigning a type to the variable 'v' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 20), 'v', comprehension_528)
        
        # Call to asanyarray(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 'v' (line 308)
        v_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 31), 'v', False)
        # Processing the call keyword arguments (line 308)
        kwargs_525 = {}
        # Getting the type of 'asanyarray' (line 308)
        asanyarray_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 20), 'asanyarray', False)
        # Calling asanyarray(args, kwargs) (line 308)
        asanyarray_call_result_526 = invoke(stypy.reporting.localization.Localization(__file__, 308, 20), asanyarray_523, *[v_524], **kwargs_525)
        
        list_529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 20), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 308, 20), list_529, asanyarray_call_result_526)
        # Assigning a type to the variable 'operands' (line 308)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'operands', list_529)
        
        # Assigning a Str to a Name (line 309):
        
        # Assigning a Str to a Name (line 309):
        str_530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 21), 'str', '')
        # Assigning a type to the variable 'subscripts' (line 309)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'subscripts', str_530)
        
        # Assigning a BinOp to a Name (line 310):
        
        # Assigning a BinOp to a Name (line 310):
        
        # Call to len(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 'subscript_list' (line 310)
        subscript_list_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 'subscript_list', False)
        # Processing the call keyword arguments (line 310)
        kwargs_533 = {}
        # Getting the type of 'len' (line 310)
        len_531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 15), 'len', False)
        # Calling len(args, kwargs) (line 310)
        len_call_result_534 = invoke(stypy.reporting.localization.Localization(__file__, 310, 15), len_531, *[subscript_list_532], **kwargs_533)
        
        int_535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 310, 37), 'int')
        # Applying the binary operator '-' (line 310)
        result_sub_536 = python_operator(stypy.reporting.localization.Localization(__file__, 310, 15), '-', len_call_result_534, int_535)
        
        # Assigning a type to the variable 'last' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'last', result_sub_536)
        
        
        # Call to enumerate(...): (line 311)
        # Processing the call arguments (line 311)
        # Getting the type of 'subscript_list' (line 311)
        subscript_list_538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 34), 'subscript_list', False)
        # Processing the call keyword arguments (line 311)
        kwargs_539 = {}
        # Getting the type of 'enumerate' (line 311)
        enumerate_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 24), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 311)
        enumerate_call_result_540 = invoke(stypy.reporting.localization.Localization(__file__, 311, 24), enumerate_537, *[subscript_list_538], **kwargs_539)
        
        # Testing the type of a for loop iterable (line 311)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 311, 8), enumerate_call_result_540)
        # Getting the type of the for loop variable (line 311)
        for_loop_var_541 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 311, 8), enumerate_call_result_540)
        # Assigning a type to the variable 'num' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'num', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 8), for_loop_var_541))
        # Assigning a type to the variable 'sub' (line 311)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'sub', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 311, 8), for_loop_var_541))
        # SSA begins for a for statement (line 311)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'sub' (line 312)
        sub_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 21), 'sub')
        # Testing the type of a for loop iterable (line 312)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 312, 12), sub_542)
        # Getting the type of the for loop variable (line 312)
        for_loop_var_543 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 312, 12), sub_542)
        # Assigning a type to the variable 's' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 12), 's', for_loop_var_543)
        # SSA begins for a for statement (line 312)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 's' (line 313)
        s_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 's')
        # Getting the type of 'Ellipsis' (line 313)
        Ellipsis_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 24), 'Ellipsis')
        # Applying the binary operator 'is' (line 313)
        result_is__546 = python_operator(stypy.reporting.localization.Localization(__file__, 313, 19), 'is', s_544, Ellipsis_545)
        
        # Testing the type of an if condition (line 313)
        if_condition_547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 313, 16), result_is__546)
        # Assigning a type to the variable 'if_condition_547' (line 313)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'if_condition_547', if_condition_547)
        # SSA begins for if statement (line 313)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'subscripts' (line 314)
        subscripts_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 20), 'subscripts')
        str_549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 34), 'str', '...')
        # Applying the binary operator '+=' (line 314)
        result_iadd_550 = python_operator(stypy.reporting.localization.Localization(__file__, 314, 20), '+=', subscripts_548, str_549)
        # Assigning a type to the variable 'subscripts' (line 314)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 20), 'subscripts', result_iadd_550)
        
        # SSA branch for the else part of an if statement (line 313)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 315)
        # Getting the type of 'int' (line 315)
        int_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 35), 'int')
        # Getting the type of 's' (line 315)
        s_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 32), 's')
        
        (may_be_553, more_types_in_union_554) = may_be_subtype(int_551, s_552)

        if may_be_553:

            if more_types_in_union_554:
                # Runtime conditional SSA (line 315)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 's' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 21), 's', remove_not_subtype_from_union(s_552, int))
            
            # Getting the type of 'subscripts' (line 316)
            subscripts_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 20), 'subscripts')
            
            # Obtaining the type of the subscript
            # Getting the type of 's' (line 316)
            s_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 49), 's')
            # Getting the type of 'einsum_symbols' (line 316)
            einsum_symbols_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 34), 'einsum_symbols')
            # Obtaining the member '__getitem__' of a type (line 316)
            getitem___558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 34), einsum_symbols_557, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 316)
            subscript_call_result_559 = invoke(stypy.reporting.localization.Localization(__file__, 316, 34), getitem___558, s_556)
            
            # Applying the binary operator '+=' (line 316)
            result_iadd_560 = python_operator(stypy.reporting.localization.Localization(__file__, 316, 20), '+=', subscripts_555, subscript_call_result_559)
            # Assigning a type to the variable 'subscripts' (line 316)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 20), 'subscripts', result_iadd_560)
            

            if more_types_in_union_554:
                # Runtime conditional SSA for else branch (line 315)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_553) or more_types_in_union_554):
            # Assigning a type to the variable 's' (line 315)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 21), 's', remove_subtype_from_union(s_552, int))
            
            # Call to TypeError(...): (line 318)
            # Processing the call arguments (line 318)
            str_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 36), 'str', 'For this input type lists must contain either int or Ellipsis')
            # Processing the call keyword arguments (line 318)
            kwargs_563 = {}
            # Getting the type of 'TypeError' (line 318)
            TypeError_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 26), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 318)
            TypeError_call_result_564 = invoke(stypy.reporting.localization.Localization(__file__, 318, 26), TypeError_561, *[str_562], **kwargs_563)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 318, 20), TypeError_call_result_564, 'raise parameter', BaseException)

            if (may_be_553 and more_types_in_union_554):
                # SSA join for if statement (line 315)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 313)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'num' (line 320)
        num_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 15), 'num')
        # Getting the type of 'last' (line 320)
        last_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 22), 'last')
        # Applying the binary operator '!=' (line 320)
        result_ne_567 = python_operator(stypy.reporting.localization.Localization(__file__, 320, 15), '!=', num_565, last_566)
        
        # Testing the type of an if condition (line 320)
        if_condition_568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 320, 12), result_ne_567)
        # Assigning a type to the variable 'if_condition_568' (line 320)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 12), 'if_condition_568', if_condition_568)
        # SSA begins for if statement (line 320)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'subscripts' (line 321)
        subscripts_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'subscripts')
        str_570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 321, 30), 'str', ',')
        # Applying the binary operator '+=' (line 321)
        result_iadd_571 = python_operator(stypy.reporting.localization.Localization(__file__, 321, 16), '+=', subscripts_569, str_570)
        # Assigning a type to the variable 'subscripts' (line 321)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 16), 'subscripts', result_iadd_571)
        
        # SSA join for if statement (line 320)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 323)
        # Getting the type of 'output_list' (line 323)
        output_list_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'output_list')
        # Getting the type of 'None' (line 323)
        None_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 30), 'None')
        
        (may_be_574, more_types_in_union_575) = may_not_be_none(output_list_572, None_573)

        if may_be_574:

            if more_types_in_union_575:
                # Runtime conditional SSA (line 323)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'subscripts' (line 324)
            subscripts_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'subscripts')
            str_577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 324, 26), 'str', '->')
            # Applying the binary operator '+=' (line 324)
            result_iadd_578 = python_operator(stypy.reporting.localization.Localization(__file__, 324, 12), '+=', subscripts_576, str_577)
            # Assigning a type to the variable 'subscripts' (line 324)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'subscripts', result_iadd_578)
            
            
            # Getting the type of 'output_list' (line 325)
            output_list_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 21), 'output_list')
            # Testing the type of a for loop iterable (line 325)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 325, 12), output_list_579)
            # Getting the type of the for loop variable (line 325)
            for_loop_var_580 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 325, 12), output_list_579)
            # Assigning a type to the variable 's' (line 325)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 's', for_loop_var_580)
            # SSA begins for a for statement (line 325)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Getting the type of 's' (line 326)
            s_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 19), 's')
            # Getting the type of 'Ellipsis' (line 326)
            Ellipsis_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 24), 'Ellipsis')
            # Applying the binary operator 'is' (line 326)
            result_is__583 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 19), 'is', s_581, Ellipsis_582)
            
            # Testing the type of an if condition (line 326)
            if_condition_584 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 326, 16), result_is__583)
            # Assigning a type to the variable 'if_condition_584' (line 326)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 16), 'if_condition_584', if_condition_584)
            # SSA begins for if statement (line 326)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'subscripts' (line 327)
            subscripts_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 20), 'subscripts')
            str_586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 34), 'str', '...')
            # Applying the binary operator '+=' (line 327)
            result_iadd_587 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 20), '+=', subscripts_585, str_586)
            # Assigning a type to the variable 'subscripts' (line 327)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 20), 'subscripts', result_iadd_587)
            
            # SSA branch for the else part of an if statement (line 326)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 328)
            # Getting the type of 'int' (line 328)
            int_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 35), 'int')
            # Getting the type of 's' (line 328)
            s_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 32), 's')
            
            (may_be_590, more_types_in_union_591) = may_be_subtype(int_588, s_589)

            if may_be_590:

                if more_types_in_union_591:
                    # Runtime conditional SSA (line 328)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 's' (line 328)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 21), 's', remove_not_subtype_from_union(s_589, int))
                
                # Getting the type of 'subscripts' (line 329)
                subscripts_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'subscripts')
                
                # Obtaining the type of the subscript
                # Getting the type of 's' (line 329)
                s_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 49), 's')
                # Getting the type of 'einsum_symbols' (line 329)
                einsum_symbols_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 34), 'einsum_symbols')
                # Obtaining the member '__getitem__' of a type (line 329)
                getitem___595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 34), einsum_symbols_594, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 329)
                subscript_call_result_596 = invoke(stypy.reporting.localization.Localization(__file__, 329, 34), getitem___595, s_593)
                
                # Applying the binary operator '+=' (line 329)
                result_iadd_597 = python_operator(stypy.reporting.localization.Localization(__file__, 329, 20), '+=', subscripts_592, subscript_call_result_596)
                # Assigning a type to the variable 'subscripts' (line 329)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'subscripts', result_iadd_597)
                

                if more_types_in_union_591:
                    # Runtime conditional SSA for else branch (line 328)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_590) or more_types_in_union_591):
                # Assigning a type to the variable 's' (line 328)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 21), 's', remove_subtype_from_union(s_589, int))
                
                # Call to TypeError(...): (line 331)
                # Processing the call arguments (line 331)
                str_599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 331, 36), 'str', 'For this input type lists must contain either int or Ellipsis')
                # Processing the call keyword arguments (line 331)
                kwargs_600 = {}
                # Getting the type of 'TypeError' (line 331)
                TypeError_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 26), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 331)
                TypeError_call_result_601 = invoke(stypy.reporting.localization.Localization(__file__, 331, 26), TypeError_598, *[str_599], **kwargs_600)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 331, 20), TypeError_call_result_601, 'raise parameter', BaseException)

                if (may_be_590 and more_types_in_union_591):
                    # SSA join for if statement (line 328)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 326)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_575:
                # SSA join for if statement (line 323)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_441 and more_types_in_union_442):
            # SSA join for if statement (line 288)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    str_602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 8), 'str', '-')
    # Getting the type of 'subscripts' (line 334)
    subscripts_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 15), 'subscripts')
    # Applying the binary operator 'in' (line 334)
    result_contains_604 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 8), 'in', str_602, subscripts_603)
    
    
    str_605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 334, 31), 'str', '>')
    # Getting the type of 'subscripts' (line 334)
    subscripts_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 38), 'subscripts')
    # Applying the binary operator 'in' (line 334)
    result_contains_607 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 31), 'in', str_605, subscripts_606)
    
    # Applying the binary operator 'or' (line 334)
    result_or_keyword_608 = python_operator(stypy.reporting.localization.Localization(__file__, 334, 7), 'or', result_contains_604, result_contains_607)
    
    # Testing the type of an if condition (line 334)
    if_condition_609 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 4), result_or_keyword_608)
    # Assigning a type to the variable 'if_condition_609' (line 334)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'if_condition_609', if_condition_609)
    # SSA begins for if statement (line 334)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BoolOp to a Name (line 335):
    
    # Assigning a BoolOp to a Name (line 335):
    
    # Evaluating a boolean operation
    
    
    # Call to count(...): (line 335)
    # Processing the call arguments (line 335)
    str_612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 36), 'str', '-')
    # Processing the call keyword arguments (line 335)
    kwargs_613 = {}
    # Getting the type of 'subscripts' (line 335)
    subscripts_610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 19), 'subscripts', False)
    # Obtaining the member 'count' of a type (line 335)
    count_611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 19), subscripts_610, 'count')
    # Calling count(args, kwargs) (line 335)
    count_call_result_614 = invoke(stypy.reporting.localization.Localization(__file__, 335, 19), count_611, *[str_612], **kwargs_613)
    
    int_615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 43), 'int')
    # Applying the binary operator '>' (line 335)
    result_gt_616 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 19), '>', count_call_result_614, int_615)
    
    
    
    # Call to count(...): (line 335)
    # Processing the call arguments (line 335)
    str_619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 67), 'str', '>')
    # Processing the call keyword arguments (line 335)
    kwargs_620 = {}
    # Getting the type of 'subscripts' (line 335)
    subscripts_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 50), 'subscripts', False)
    # Obtaining the member 'count' of a type (line 335)
    count_618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 50), subscripts_617, 'count')
    # Calling count(args, kwargs) (line 335)
    count_call_result_621 = invoke(stypy.reporting.localization.Localization(__file__, 335, 50), count_618, *[str_619], **kwargs_620)
    
    int_622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 74), 'int')
    # Applying the binary operator '>' (line 335)
    result_gt_623 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 50), '>', count_call_result_621, int_622)
    
    # Applying the binary operator 'or' (line 335)
    result_or_keyword_624 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 18), 'or', result_gt_616, result_gt_623)
    
    # Assigning a type to the variable 'invalid' (line 335)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'invalid', result_or_keyword_624)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'invalid' (line 336)
    invalid_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 'invalid')
    
    
    # Call to count(...): (line 336)
    # Processing the call arguments (line 336)
    str_628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 40), 'str', '->')
    # Processing the call keyword arguments (line 336)
    kwargs_629 = {}
    # Getting the type of 'subscripts' (line 336)
    subscripts_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 23), 'subscripts', False)
    # Obtaining the member 'count' of a type (line 336)
    count_627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 23), subscripts_626, 'count')
    # Calling count(args, kwargs) (line 336)
    count_call_result_630 = invoke(stypy.reporting.localization.Localization(__file__, 336, 23), count_627, *[str_628], **kwargs_629)
    
    int_631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 49), 'int')
    # Applying the binary operator '!=' (line 336)
    result_ne_632 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 23), '!=', count_call_result_630, int_631)
    
    # Applying the binary operator 'or' (line 336)
    result_or_keyword_633 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 11), 'or', invalid_625, result_ne_632)
    
    # Testing the type of an if condition (line 336)
    if_condition_634 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 8), result_or_keyword_633)
    # Assigning a type to the variable 'if_condition_634' (line 336)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'if_condition_634', if_condition_634)
    # SSA begins for if statement (line 336)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 337)
    # Processing the call arguments (line 337)
    str_636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 29), 'str', "Subscripts can only contain one '->'.")
    # Processing the call keyword arguments (line 337)
    kwargs_637 = {}
    # Getting the type of 'ValueError' (line 337)
    ValueError_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 337)
    ValueError_call_result_638 = invoke(stypy.reporting.localization.Localization(__file__, 337, 18), ValueError_635, *[str_636], **kwargs_637)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 337, 12), ValueError_call_result_638, 'raise parameter', BaseException)
    # SSA join for if statement (line 336)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 334)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 7), 'str', '.')
    # Getting the type of 'subscripts' (line 340)
    subscripts_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 14), 'subscripts')
    # Applying the binary operator 'in' (line 340)
    result_contains_641 = python_operator(stypy.reporting.localization.Localization(__file__, 340, 7), 'in', str_639, subscripts_640)
    
    # Testing the type of an if condition (line 340)
    if_condition_642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 340, 4), result_contains_641)
    # Assigning a type to the variable 'if_condition_642' (line 340)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'if_condition_642', if_condition_642)
    # SSA begins for if statement (line 340)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 341):
    
    # Assigning a Call to a Name (line 341):
    
    # Call to replace(...): (line 341)
    # Processing the call arguments (line 341)
    str_655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 68), 'str', '->')
    str_656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 74), 'str', '')
    # Processing the call keyword arguments (line 341)
    kwargs_657 = {}
    
    # Call to replace(...): (line 341)
    # Processing the call arguments (line 341)
    str_650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 51), 'str', ',')
    str_651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 56), 'str', '')
    # Processing the call keyword arguments (line 341)
    kwargs_652 = {}
    
    # Call to replace(...): (line 341)
    # Processing the call arguments (line 341)
    str_645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 34), 'str', '.')
    str_646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 39), 'str', '')
    # Processing the call keyword arguments (line 341)
    kwargs_647 = {}
    # Getting the type of 'subscripts' (line 341)
    subscripts_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 15), 'subscripts', False)
    # Obtaining the member 'replace' of a type (line 341)
    replace_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), subscripts_643, 'replace')
    # Calling replace(args, kwargs) (line 341)
    replace_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 341, 15), replace_644, *[str_645, str_646], **kwargs_647)
    
    # Obtaining the member 'replace' of a type (line 341)
    replace_649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), replace_call_result_648, 'replace')
    # Calling replace(args, kwargs) (line 341)
    replace_call_result_653 = invoke(stypy.reporting.localization.Localization(__file__, 341, 15), replace_649, *[str_650, str_651], **kwargs_652)
    
    # Obtaining the member 'replace' of a type (line 341)
    replace_654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 15), replace_call_result_653, 'replace')
    # Calling replace(args, kwargs) (line 341)
    replace_call_result_658 = invoke(stypy.reporting.localization.Localization(__file__, 341, 15), replace_654, *[str_655, str_656], **kwargs_657)
    
    # Assigning a type to the variable 'used' (line 341)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'used', replace_call_result_658)
    
    # Assigning a Call to a Name (line 342):
    
    # Assigning a Call to a Name (line 342):
    
    # Call to list(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'einsum_symbols_set' (line 342)
    einsum_symbols_set_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 22), 'einsum_symbols_set', False)
    
    # Call to set(...): (line 342)
    # Processing the call arguments (line 342)
    # Getting the type of 'used' (line 342)
    used_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 47), 'used', False)
    # Processing the call keyword arguments (line 342)
    kwargs_663 = {}
    # Getting the type of 'set' (line 342)
    set_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 43), 'set', False)
    # Calling set(args, kwargs) (line 342)
    set_call_result_664 = invoke(stypy.reporting.localization.Localization(__file__, 342, 43), set_661, *[used_662], **kwargs_663)
    
    # Applying the binary operator '-' (line 342)
    result_sub_665 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 22), '-', einsum_symbols_set_660, set_call_result_664)
    
    # Processing the call keyword arguments (line 342)
    kwargs_666 = {}
    # Getting the type of 'list' (line 342)
    list_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 17), 'list', False)
    # Calling list(args, kwargs) (line 342)
    list_call_result_667 = invoke(stypy.reporting.localization.Localization(__file__, 342, 17), list_659, *[result_sub_665], **kwargs_666)
    
    # Assigning a type to the variable 'unused' (line 342)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'unused', list_call_result_667)
    
    # Assigning a Call to a Name (line 343):
    
    # Assigning a Call to a Name (line 343):
    
    # Call to join(...): (line 343)
    # Processing the call arguments (line 343)
    # Getting the type of 'unused' (line 343)
    unused_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 31), 'unused', False)
    # Processing the call keyword arguments (line 343)
    kwargs_671 = {}
    str_668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 23), 'str', '')
    # Obtaining the member 'join' of a type (line 343)
    join_669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 23), str_668, 'join')
    # Calling join(args, kwargs) (line 343)
    join_call_result_672 = invoke(stypy.reporting.localization.Localization(__file__, 343, 23), join_669, *[unused_670], **kwargs_671)
    
    # Assigning a type to the variable 'ellipse_inds' (line 343)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'ellipse_inds', join_call_result_672)
    
    # Assigning a Num to a Name (line 344):
    
    # Assigning a Num to a Name (line 344):
    int_673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 18), 'int')
    # Assigning a type to the variable 'longest' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'longest', int_673)
    
    
    str_674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 11), 'str', '->')
    # Getting the type of 'subscripts' (line 346)
    subscripts_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 19), 'subscripts')
    # Applying the binary operator 'in' (line 346)
    result_contains_676 = python_operator(stypy.reporting.localization.Localization(__file__, 346, 11), 'in', str_674, subscripts_675)
    
    # Testing the type of an if condition (line 346)
    if_condition_677 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 346, 8), result_contains_676)
    # Assigning a type to the variable 'if_condition_677' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'if_condition_677', if_condition_677)
    # SSA begins for if statement (line 346)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 347):
    
    # Assigning a Subscript to a Name (line 347):
    
    # Obtaining the type of the subscript
    int_678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 12), 'int')
    
    # Call to split(...): (line 347)
    # Processing the call arguments (line 347)
    str_681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 53), 'str', '->')
    # Processing the call keyword arguments (line 347)
    kwargs_682 = {}
    # Getting the type of 'subscripts' (line 347)
    subscripts_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 36), 'subscripts', False)
    # Obtaining the member 'split' of a type (line 347)
    split_680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 36), subscripts_679, 'split')
    # Calling split(args, kwargs) (line 347)
    split_call_result_683 = invoke(stypy.reporting.localization.Localization(__file__, 347, 36), split_680, *[str_681], **kwargs_682)
    
    # Obtaining the member '__getitem__' of a type (line 347)
    getitem___684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 12), split_call_result_683, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 347)
    subscript_call_result_685 = invoke(stypy.reporting.localization.Localization(__file__, 347, 12), getitem___684, int_678)
    
    # Assigning a type to the variable 'tuple_var_assignment_12' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'tuple_var_assignment_12', subscript_call_result_685)
    
    # Assigning a Subscript to a Name (line 347):
    
    # Obtaining the type of the subscript
    int_686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 12), 'int')
    
    # Call to split(...): (line 347)
    # Processing the call arguments (line 347)
    str_689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 347, 53), 'str', '->')
    # Processing the call keyword arguments (line 347)
    kwargs_690 = {}
    # Getting the type of 'subscripts' (line 347)
    subscripts_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 36), 'subscripts', False)
    # Obtaining the member 'split' of a type (line 347)
    split_688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 36), subscripts_687, 'split')
    # Calling split(args, kwargs) (line 347)
    split_call_result_691 = invoke(stypy.reporting.localization.Localization(__file__, 347, 36), split_688, *[str_689], **kwargs_690)
    
    # Obtaining the member '__getitem__' of a type (line 347)
    getitem___692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 12), split_call_result_691, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 347)
    subscript_call_result_693 = invoke(stypy.reporting.localization.Localization(__file__, 347, 12), getitem___692, int_686)
    
    # Assigning a type to the variable 'tuple_var_assignment_13' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'tuple_var_assignment_13', subscript_call_result_693)
    
    # Assigning a Name to a Name (line 347):
    # Getting the type of 'tuple_var_assignment_12' (line 347)
    tuple_var_assignment_12_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'tuple_var_assignment_12')
    # Assigning a type to the variable 'input_tmp' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'input_tmp', tuple_var_assignment_12_694)
    
    # Assigning a Name to a Name (line 347):
    # Getting the type of 'tuple_var_assignment_13' (line 347)
    tuple_var_assignment_13_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 12), 'tuple_var_assignment_13')
    # Assigning a type to the variable 'output_sub' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 23), 'output_sub', tuple_var_assignment_13_695)
    
    # Assigning a Call to a Name (line 348):
    
    # Assigning a Call to a Name (line 348):
    
    # Call to split(...): (line 348)
    # Processing the call arguments (line 348)
    str_698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 47), 'str', ',')
    # Processing the call keyword arguments (line 348)
    kwargs_699 = {}
    # Getting the type of 'input_tmp' (line 348)
    input_tmp_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 31), 'input_tmp', False)
    # Obtaining the member 'split' of a type (line 348)
    split_697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 31), input_tmp_696, 'split')
    # Calling split(args, kwargs) (line 348)
    split_call_result_700 = invoke(stypy.reporting.localization.Localization(__file__, 348, 31), split_697, *[str_698], **kwargs_699)
    
    # Assigning a type to the variable 'split_subscripts' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 12), 'split_subscripts', split_call_result_700)
    
    # Assigning a Name to a Name (line 349):
    
    # Assigning a Name to a Name (line 349):
    # Getting the type of 'True' (line 349)
    True_701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 22), 'True')
    # Assigning a type to the variable 'out_sub' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'out_sub', True_701)
    # SSA branch for the else part of an if statement (line 346)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 351):
    
    # Assigning a Call to a Name (line 351):
    
    # Call to split(...): (line 351)
    # Processing the call arguments (line 351)
    str_704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 48), 'str', ',')
    # Processing the call keyword arguments (line 351)
    kwargs_705 = {}
    # Getting the type of 'subscripts' (line 351)
    subscripts_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 31), 'subscripts', False)
    # Obtaining the member 'split' of a type (line 351)
    split_703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 31), subscripts_702, 'split')
    # Calling split(args, kwargs) (line 351)
    split_call_result_706 = invoke(stypy.reporting.localization.Localization(__file__, 351, 31), split_703, *[str_704], **kwargs_705)
    
    # Assigning a type to the variable 'split_subscripts' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'split_subscripts', split_call_result_706)
    
    # Assigning a Name to a Name (line 352):
    
    # Assigning a Name to a Name (line 352):
    # Getting the type of 'False' (line 352)
    False_707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 22), 'False')
    # Assigning a type to the variable 'out_sub' (line 352)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'out_sub', False_707)
    # SSA join for if statement (line 346)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to enumerate(...): (line 354)
    # Processing the call arguments (line 354)
    # Getting the type of 'split_subscripts' (line 354)
    split_subscripts_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 34), 'split_subscripts', False)
    # Processing the call keyword arguments (line 354)
    kwargs_710 = {}
    # Getting the type of 'enumerate' (line 354)
    enumerate_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 24), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 354)
    enumerate_call_result_711 = invoke(stypy.reporting.localization.Localization(__file__, 354, 24), enumerate_708, *[split_subscripts_709], **kwargs_710)
    
    # Testing the type of a for loop iterable (line 354)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 354, 8), enumerate_call_result_711)
    # Getting the type of the for loop variable (line 354)
    for_loop_var_712 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 354, 8), enumerate_call_result_711)
    # Assigning a type to the variable 'num' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'num', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 8), for_loop_var_712))
    # Assigning a type to the variable 'sub' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'sub', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 354, 8), for_loop_var_712))
    # SSA begins for a for statement (line 354)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    str_713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 15), 'str', '.')
    # Getting the type of 'sub' (line 355)
    sub_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 22), 'sub')
    # Applying the binary operator 'in' (line 355)
    result_contains_715 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 15), 'in', str_713, sub_714)
    
    # Testing the type of an if condition (line 355)
    if_condition_716 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 12), result_contains_715)
    # Assigning a type to the variable 'if_condition_716' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'if_condition_716', if_condition_716)
    # SSA begins for if statement (line 355)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    
    # Call to count(...): (line 356)
    # Processing the call arguments (line 356)
    str_719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 30), 'str', '.')
    # Processing the call keyword arguments (line 356)
    kwargs_720 = {}
    # Getting the type of 'sub' (line 356)
    sub_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 20), 'sub', False)
    # Obtaining the member 'count' of a type (line 356)
    count_718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 20), sub_717, 'count')
    # Calling count(args, kwargs) (line 356)
    count_call_result_721 = invoke(stypy.reporting.localization.Localization(__file__, 356, 20), count_718, *[str_719], **kwargs_720)
    
    int_722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 38), 'int')
    # Applying the binary operator '!=' (line 356)
    result_ne_723 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 20), '!=', count_call_result_721, int_722)
    
    
    
    # Call to count(...): (line 356)
    # Processing the call arguments (line 356)
    str_726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 55), 'str', '...')
    # Processing the call keyword arguments (line 356)
    kwargs_727 = {}
    # Getting the type of 'sub' (line 356)
    sub_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 45), 'sub', False)
    # Obtaining the member 'count' of a type (line 356)
    count_725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 45), sub_724, 'count')
    # Calling count(args, kwargs) (line 356)
    count_call_result_728 = invoke(stypy.reporting.localization.Localization(__file__, 356, 45), count_725, *[str_726], **kwargs_727)
    
    int_729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 65), 'int')
    # Applying the binary operator '!=' (line 356)
    result_ne_730 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 45), '!=', count_call_result_728, int_729)
    
    # Applying the binary operator 'or' (line 356)
    result_or_keyword_731 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 19), 'or', result_ne_723, result_ne_730)
    
    # Testing the type of an if condition (line 356)
    if_condition_732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 356, 16), result_or_keyword_731)
    # Assigning a type to the variable 'if_condition_732' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 16), 'if_condition_732', if_condition_732)
    # SSA begins for if statement (line 356)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 357)
    # Processing the call arguments (line 357)
    str_734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 37), 'str', 'Invalid Ellipses.')
    # Processing the call keyword arguments (line 357)
    kwargs_735 = {}
    # Getting the type of 'ValueError' (line 357)
    ValueError_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 26), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 357)
    ValueError_call_result_736 = invoke(stypy.reporting.localization.Localization(__file__, 357, 26), ValueError_733, *[str_734], **kwargs_735)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 357, 20), ValueError_call_result_736, 'raise parameter', BaseException)
    # SSA join for if statement (line 356)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'num' (line 360)
    num_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 28), 'num')
    # Getting the type of 'operands' (line 360)
    operands_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 19), 'operands')
    # Obtaining the member '__getitem__' of a type (line 360)
    getitem___739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 19), operands_738, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 360)
    subscript_call_result_740 = invoke(stypy.reporting.localization.Localization(__file__, 360, 19), getitem___739, num_737)
    
    # Obtaining the member 'shape' of a type (line 360)
    shape_741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 19), subscript_call_result_740, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 360)
    tuple_742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 42), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 360)
    
    # Applying the binary operator '==' (line 360)
    result_eq_743 = python_operator(stypy.reporting.localization.Localization(__file__, 360, 19), '==', shape_741, tuple_742)
    
    # Testing the type of an if condition (line 360)
    if_condition_744 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 16), result_eq_743)
    # Assigning a type to the variable 'if_condition_744' (line 360)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 16), 'if_condition_744', if_condition_744)
    # SSA begins for if statement (line 360)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 361):
    
    # Assigning a Num to a Name (line 361):
    int_745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 36), 'int')
    # Assigning a type to the variable 'ellipse_count' (line 361)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'ellipse_count', int_745)
    # SSA branch for the else part of an if statement (line 360)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 363):
    
    # Assigning a Call to a Name (line 363):
    
    # Call to max(...): (line 363)
    # Processing the call arguments (line 363)
    
    # Obtaining the type of the subscript
    # Getting the type of 'num' (line 363)
    num_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 49), 'num', False)
    # Getting the type of 'operands' (line 363)
    operands_748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 40), 'operands', False)
    # Obtaining the member '__getitem__' of a type (line 363)
    getitem___749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 40), operands_748, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 363)
    subscript_call_result_750 = invoke(stypy.reporting.localization.Localization(__file__, 363, 40), getitem___749, num_747)
    
    # Obtaining the member 'ndim' of a type (line 363)
    ndim_751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 40), subscript_call_result_750, 'ndim')
    int_752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 60), 'int')
    # Processing the call keyword arguments (line 363)
    kwargs_753 = {}
    # Getting the type of 'max' (line 363)
    max_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 36), 'max', False)
    # Calling max(args, kwargs) (line 363)
    max_call_result_754 = invoke(stypy.reporting.localization.Localization(__file__, 363, 36), max_746, *[ndim_751, int_752], **kwargs_753)
    
    # Assigning a type to the variable 'ellipse_count' (line 363)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 20), 'ellipse_count', max_call_result_754)
    
    # Getting the type of 'ellipse_count' (line 364)
    ellipse_count_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 20), 'ellipse_count')
    
    # Call to len(...): (line 364)
    # Processing the call arguments (line 364)
    # Getting the type of 'sub' (line 364)
    sub_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 42), 'sub', False)
    # Processing the call keyword arguments (line 364)
    kwargs_758 = {}
    # Getting the type of 'len' (line 364)
    len_756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 38), 'len', False)
    # Calling len(args, kwargs) (line 364)
    len_call_result_759 = invoke(stypy.reporting.localization.Localization(__file__, 364, 38), len_756, *[sub_757], **kwargs_758)
    
    int_760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 49), 'int')
    # Applying the binary operator '-' (line 364)
    result_sub_761 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 38), '-', len_call_result_759, int_760)
    
    # Applying the binary operator '-=' (line 364)
    result_isub_762 = python_operator(stypy.reporting.localization.Localization(__file__, 364, 20), '-=', ellipse_count_755, result_sub_761)
    # Assigning a type to the variable 'ellipse_count' (line 364)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 364, 20), 'ellipse_count', result_isub_762)
    
    # SSA join for if statement (line 360)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ellipse_count' (line 366)
    ellipse_count_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 19), 'ellipse_count')
    # Getting the type of 'longest' (line 366)
    longest_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 35), 'longest')
    # Applying the binary operator '>' (line 366)
    result_gt_765 = python_operator(stypy.reporting.localization.Localization(__file__, 366, 19), '>', ellipse_count_763, longest_764)
    
    # Testing the type of an if condition (line 366)
    if_condition_766 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 16), result_gt_765)
    # Assigning a type to the variable 'if_condition_766' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 16), 'if_condition_766', if_condition_766)
    # SSA begins for if statement (line 366)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 367):
    
    # Assigning a Name to a Name (line 367):
    # Getting the type of 'ellipse_count' (line 367)
    ellipse_count_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 30), 'ellipse_count')
    # Assigning a type to the variable 'longest' (line 367)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 367, 20), 'longest', ellipse_count_767)
    # SSA join for if statement (line 366)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'ellipse_count' (line 369)
    ellipse_count_768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 'ellipse_count')
    int_769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 35), 'int')
    # Applying the binary operator '<' (line 369)
    result_lt_770 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 19), '<', ellipse_count_768, int_769)
    
    # Testing the type of an if condition (line 369)
    if_condition_771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 16), result_lt_770)
    # Assigning a type to the variable 'if_condition_771' (line 369)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 16), 'if_condition_771', if_condition_771)
    # SSA begins for if statement (line 369)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 370)
    # Processing the call arguments (line 370)
    str_773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 37), 'str', 'Ellipses lengths do not match.')
    # Processing the call keyword arguments (line 370)
    kwargs_774 = {}
    # Getting the type of 'ValueError' (line 370)
    ValueError_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 26), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 370)
    ValueError_call_result_775 = invoke(stypy.reporting.localization.Localization(__file__, 370, 26), ValueError_772, *[str_773], **kwargs_774)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 370, 20), ValueError_call_result_775, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 369)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'ellipse_count' (line 371)
    ellipse_count_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 21), 'ellipse_count')
    int_777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 38), 'int')
    # Applying the binary operator '==' (line 371)
    result_eq_778 = python_operator(stypy.reporting.localization.Localization(__file__, 371, 21), '==', ellipse_count_776, int_777)
    
    # Testing the type of an if condition (line 371)
    if_condition_779 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 371, 21), result_eq_778)
    # Assigning a type to the variable 'if_condition_779' (line 371)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 21), 'if_condition_779', if_condition_779)
    # SSA begins for if statement (line 371)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 372):
    
    # Assigning a Call to a Subscript (line 372):
    
    # Call to replace(...): (line 372)
    # Processing the call arguments (line 372)
    str_782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 56), 'str', '...')
    str_783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 63), 'str', '')
    # Processing the call keyword arguments (line 372)
    kwargs_784 = {}
    # Getting the type of 'sub' (line 372)
    sub_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 44), 'sub', False)
    # Obtaining the member 'replace' of a type (line 372)
    replace_781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 44), sub_780, 'replace')
    # Calling replace(args, kwargs) (line 372)
    replace_call_result_785 = invoke(stypy.reporting.localization.Localization(__file__, 372, 44), replace_781, *[str_782, str_783], **kwargs_784)
    
    # Getting the type of 'split_subscripts' (line 372)
    split_subscripts_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 20), 'split_subscripts')
    # Getting the type of 'num' (line 372)
    num_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 37), 'num')
    # Storing an element on a container (line 372)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 372, 20), split_subscripts_786, (num_787, replace_call_result_785))
    # SSA branch for the else part of an if statement (line 371)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 374):
    
    # Assigning a Subscript to a Name (line 374):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'ellipse_count' (line 374)
    ellipse_count_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 45), 'ellipse_count')
    # Applying the 'usub' unary operator (line 374)
    result___neg___789 = python_operator(stypy.reporting.localization.Localization(__file__, 374, 44), 'usub', ellipse_count_788)
    
    slice_790 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 374, 31), result___neg___789, None, None)
    # Getting the type of 'ellipse_inds' (line 374)
    ellipse_inds_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 31), 'ellipse_inds')
    # Obtaining the member '__getitem__' of a type (line 374)
    getitem___792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 31), ellipse_inds_791, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 374)
    subscript_call_result_793 = invoke(stypy.reporting.localization.Localization(__file__, 374, 31), getitem___792, slice_790)
    
    # Assigning a type to the variable 'rep_inds' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 20), 'rep_inds', subscript_call_result_793)
    
    # Assigning a Call to a Subscript (line 375):
    
    # Assigning a Call to a Subscript (line 375):
    
    # Call to replace(...): (line 375)
    # Processing the call arguments (line 375)
    str_796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 56), 'str', '...')
    # Getting the type of 'rep_inds' (line 375)
    rep_inds_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 63), 'rep_inds', False)
    # Processing the call keyword arguments (line 375)
    kwargs_798 = {}
    # Getting the type of 'sub' (line 375)
    sub_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 44), 'sub', False)
    # Obtaining the member 'replace' of a type (line 375)
    replace_795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 44), sub_794, 'replace')
    # Calling replace(args, kwargs) (line 375)
    replace_call_result_799 = invoke(stypy.reporting.localization.Localization(__file__, 375, 44), replace_795, *[str_796, rep_inds_797], **kwargs_798)
    
    # Getting the type of 'split_subscripts' (line 375)
    split_subscripts_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 20), 'split_subscripts')
    # Getting the type of 'num' (line 375)
    num_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 37), 'num')
    # Storing an element on a container (line 375)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 375, 20), split_subscripts_800, (num_801, replace_call_result_799))
    # SSA join for if statement (line 371)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 369)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 355)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 377):
    
    # Assigning a Call to a Name (line 377):
    
    # Call to join(...): (line 377)
    # Processing the call arguments (line 377)
    # Getting the type of 'split_subscripts' (line 377)
    split_subscripts_804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 30), 'split_subscripts', False)
    # Processing the call keyword arguments (line 377)
    kwargs_805 = {}
    str_802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 377, 21), 'str', ',')
    # Obtaining the member 'join' of a type (line 377)
    join_803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 21), str_802, 'join')
    # Calling join(args, kwargs) (line 377)
    join_call_result_806 = invoke(stypy.reporting.localization.Localization(__file__, 377, 21), join_803, *[split_subscripts_804], **kwargs_805)
    
    # Assigning a type to the variable 'subscripts' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 8), 'subscripts', join_call_result_806)
    
    
    # Getting the type of 'longest' (line 378)
    longest_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 11), 'longest')
    int_808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 22), 'int')
    # Applying the binary operator '==' (line 378)
    result_eq_809 = python_operator(stypy.reporting.localization.Localization(__file__, 378, 11), '==', longest_807, int_808)
    
    # Testing the type of an if condition (line 378)
    if_condition_810 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 378, 8), result_eq_809)
    # Assigning a type to the variable 'if_condition_810' (line 378)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'if_condition_810', if_condition_810)
    # SSA begins for if statement (line 378)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 379):
    
    # Assigning a Str to a Name (line 379):
    str_811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 26), 'str', '')
    # Assigning a type to the variable 'out_ellipse' (line 379)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 12), 'out_ellipse', str_811)
    # SSA branch for the else part of an if statement (line 378)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 381):
    
    # Assigning a Subscript to a Name (line 381):
    
    # Obtaining the type of the subscript
    
    # Getting the type of 'longest' (line 381)
    longest_812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 40), 'longest')
    # Applying the 'usub' unary operator (line 381)
    result___neg___813 = python_operator(stypy.reporting.localization.Localization(__file__, 381, 39), 'usub', longest_812)
    
    slice_814 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 381, 26), result___neg___813, None, None)
    # Getting the type of 'ellipse_inds' (line 381)
    ellipse_inds_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 26), 'ellipse_inds')
    # Obtaining the member '__getitem__' of a type (line 381)
    getitem___816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 381, 26), ellipse_inds_815, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 381)
    subscript_call_result_817 = invoke(stypy.reporting.localization.Localization(__file__, 381, 26), getitem___816, slice_814)
    
    # Assigning a type to the variable 'out_ellipse' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 12), 'out_ellipse', subscript_call_result_817)
    # SSA join for if statement (line 378)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'out_sub' (line 383)
    out_sub_818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 11), 'out_sub')
    # Testing the type of an if condition (line 383)
    if_condition_819 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 383, 8), out_sub_818)
    # Assigning a type to the variable 'if_condition_819' (line 383)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'if_condition_819', if_condition_819)
    # SSA begins for if statement (line 383)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'subscripts' (line 384)
    subscripts_820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'subscripts')
    str_821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 26), 'str', '->')
    
    # Call to replace(...): (line 384)
    # Processing the call arguments (line 384)
    str_824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 52), 'str', '...')
    # Getting the type of 'out_ellipse' (line 384)
    out_ellipse_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 59), 'out_ellipse', False)
    # Processing the call keyword arguments (line 384)
    kwargs_826 = {}
    # Getting the type of 'output_sub' (line 384)
    output_sub_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 33), 'output_sub', False)
    # Obtaining the member 'replace' of a type (line 384)
    replace_823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 33), output_sub_822, 'replace')
    # Calling replace(args, kwargs) (line 384)
    replace_call_result_827 = invoke(stypy.reporting.localization.Localization(__file__, 384, 33), replace_823, *[str_824, out_ellipse_825], **kwargs_826)
    
    # Applying the binary operator '+' (line 384)
    result_add_828 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 26), '+', str_821, replace_call_result_827)
    
    # Applying the binary operator '+=' (line 384)
    result_iadd_829 = python_operator(stypy.reporting.localization.Localization(__file__, 384, 12), '+=', subscripts_820, result_add_828)
    # Assigning a type to the variable 'subscripts' (line 384)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 12), 'subscripts', result_iadd_829)
    
    # SSA branch for the else part of an if statement (line 383)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 387):
    
    # Assigning a Str to a Name (line 387):
    str_830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 387, 31), 'str', '')
    # Assigning a type to the variable 'output_subscript' (line 387)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 12), 'output_subscript', str_830)
    
    # Assigning a Call to a Name (line 388):
    
    # Assigning a Call to a Name (line 388):
    
    # Call to replace(...): (line 388)
    # Processing the call arguments (line 388)
    str_833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 48), 'str', ',')
    str_834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 53), 'str', '')
    # Processing the call keyword arguments (line 388)
    kwargs_835 = {}
    # Getting the type of 'subscripts' (line 388)
    subscripts_831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 29), 'subscripts', False)
    # Obtaining the member 'replace' of a type (line 388)
    replace_832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 29), subscripts_831, 'replace')
    # Calling replace(args, kwargs) (line 388)
    replace_call_result_836 = invoke(stypy.reporting.localization.Localization(__file__, 388, 29), replace_832, *[str_833, str_834], **kwargs_835)
    
    # Assigning a type to the variable 'tmp_subscripts' (line 388)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 12), 'tmp_subscripts', replace_call_result_836)
    
    
    # Call to sorted(...): (line 389)
    # Processing the call arguments (line 389)
    
    # Call to set(...): (line 389)
    # Processing the call arguments (line 389)
    # Getting the type of 'tmp_subscripts' (line 389)
    tmp_subscripts_839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 32), 'tmp_subscripts', False)
    # Processing the call keyword arguments (line 389)
    kwargs_840 = {}
    # Getting the type of 'set' (line 389)
    set_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 28), 'set', False)
    # Calling set(args, kwargs) (line 389)
    set_call_result_841 = invoke(stypy.reporting.localization.Localization(__file__, 389, 28), set_838, *[tmp_subscripts_839], **kwargs_840)
    
    # Processing the call keyword arguments (line 389)
    kwargs_842 = {}
    # Getting the type of 'sorted' (line 389)
    sorted_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 21), 'sorted', False)
    # Calling sorted(args, kwargs) (line 389)
    sorted_call_result_843 = invoke(stypy.reporting.localization.Localization(__file__, 389, 21), sorted_837, *[set_call_result_841], **kwargs_842)
    
    # Testing the type of a for loop iterable (line 389)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 389, 12), sorted_call_result_843)
    # Getting the type of the for loop variable (line 389)
    for_loop_var_844 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 389, 12), sorted_call_result_843)
    # Assigning a type to the variable 's' (line 389)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 12), 's', for_loop_var_844)
    # SSA begins for a for statement (line 389)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 's' (line 390)
    s_845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 's')
    # Getting the type of 'einsum_symbols' (line 390)
    einsum_symbols_846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 29), 'einsum_symbols')
    # Applying the binary operator 'notin' (line 390)
    result_contains_847 = python_operator(stypy.reporting.localization.Localization(__file__, 390, 19), 'notin', s_845, einsum_symbols_846)
    
    # Testing the type of an if condition (line 390)
    if_condition_848 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 390, 16), result_contains_847)
    # Assigning a type to the variable 'if_condition_848' (line 390)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 16), 'if_condition_848', if_condition_848)
    # SSA begins for if statement (line 390)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 391)
    # Processing the call arguments (line 391)
    str_850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 391, 37), 'str', 'Character %s is not a valid symbol.')
    # Getting the type of 's' (line 391)
    s_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 77), 's', False)
    # Applying the binary operator '%' (line 391)
    result_mod_852 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 37), '%', str_850, s_851)
    
    # Processing the call keyword arguments (line 391)
    kwargs_853 = {}
    # Getting the type of 'ValueError' (line 391)
    ValueError_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 26), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 391)
    ValueError_call_result_854 = invoke(stypy.reporting.localization.Localization(__file__, 391, 26), ValueError_849, *[result_mod_852], **kwargs_853)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 391, 20), ValueError_call_result_854, 'raise parameter', BaseException)
    # SSA join for if statement (line 390)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to count(...): (line 392)
    # Processing the call arguments (line 392)
    # Getting the type of 's' (line 392)
    s_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 40), 's', False)
    # Processing the call keyword arguments (line 392)
    kwargs_858 = {}
    # Getting the type of 'tmp_subscripts' (line 392)
    tmp_subscripts_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'tmp_subscripts', False)
    # Obtaining the member 'count' of a type (line 392)
    count_856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 19), tmp_subscripts_855, 'count')
    # Calling count(args, kwargs) (line 392)
    count_call_result_859 = invoke(stypy.reporting.localization.Localization(__file__, 392, 19), count_856, *[s_857], **kwargs_858)
    
    int_860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 46), 'int')
    # Applying the binary operator '==' (line 392)
    result_eq_861 = python_operator(stypy.reporting.localization.Localization(__file__, 392, 19), '==', count_call_result_859, int_860)
    
    # Testing the type of an if condition (line 392)
    if_condition_862 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 392, 16), result_eq_861)
    # Assigning a type to the variable 'if_condition_862' (line 392)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 16), 'if_condition_862', if_condition_862)
    # SSA begins for if statement (line 392)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'output_subscript' (line 393)
    output_subscript_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 20), 'output_subscript')
    # Getting the type of 's' (line 393)
    s_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 40), 's')
    # Applying the binary operator '+=' (line 393)
    result_iadd_865 = python_operator(stypy.reporting.localization.Localization(__file__, 393, 20), '+=', output_subscript_863, s_864)
    # Assigning a type to the variable 'output_subscript' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 20), 'output_subscript', result_iadd_865)
    
    # SSA join for if statement (line 392)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 394):
    
    # Assigning a Call to a Name (line 394):
    
    # Call to join(...): (line 394)
    # Processing the call arguments (line 394)
    
    # Call to sorted(...): (line 394)
    # Processing the call arguments (line 394)
    
    # Call to set(...): (line 394)
    # Processing the call arguments (line 394)
    # Getting the type of 'output_subscript' (line 394)
    output_subscript_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 45), 'output_subscript', False)
    # Processing the call keyword arguments (line 394)
    kwargs_871 = {}
    # Getting the type of 'set' (line 394)
    set_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 41), 'set', False)
    # Calling set(args, kwargs) (line 394)
    set_call_result_872 = invoke(stypy.reporting.localization.Localization(__file__, 394, 41), set_869, *[output_subscript_870], **kwargs_871)
    
    
    # Call to set(...): (line 395)
    # Processing the call arguments (line 395)
    # Getting the type of 'out_ellipse' (line 395)
    out_ellipse_874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 45), 'out_ellipse', False)
    # Processing the call keyword arguments (line 395)
    kwargs_875 = {}
    # Getting the type of 'set' (line 395)
    set_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 41), 'set', False)
    # Calling set(args, kwargs) (line 395)
    set_call_result_876 = invoke(stypy.reporting.localization.Localization(__file__, 395, 41), set_873, *[out_ellipse_874], **kwargs_875)
    
    # Applying the binary operator '-' (line 394)
    result_sub_877 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 41), '-', set_call_result_872, set_call_result_876)
    
    # Processing the call keyword arguments (line 394)
    kwargs_878 = {}
    # Getting the type of 'sorted' (line 394)
    sorted_868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 34), 'sorted', False)
    # Calling sorted(args, kwargs) (line 394)
    sorted_call_result_879 = invoke(stypy.reporting.localization.Localization(__file__, 394, 34), sorted_868, *[result_sub_877], **kwargs_878)
    
    # Processing the call keyword arguments (line 394)
    kwargs_880 = {}
    str_866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 26), 'str', '')
    # Obtaining the member 'join' of a type (line 394)
    join_867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 26), str_866, 'join')
    # Calling join(args, kwargs) (line 394)
    join_call_result_881 = invoke(stypy.reporting.localization.Localization(__file__, 394, 26), join_867, *[sorted_call_result_879], **kwargs_880)
    
    # Assigning a type to the variable 'normal_inds' (line 394)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'normal_inds', join_call_result_881)
    
    # Getting the type of 'subscripts' (line 397)
    subscripts_882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'subscripts')
    str_883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 26), 'str', '->')
    # Getting the type of 'out_ellipse' (line 397)
    out_ellipse_884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 33), 'out_ellipse')
    # Applying the binary operator '+' (line 397)
    result_add_885 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 26), '+', str_883, out_ellipse_884)
    
    # Getting the type of 'normal_inds' (line 397)
    normal_inds_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 47), 'normal_inds')
    # Applying the binary operator '+' (line 397)
    result_add_887 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 45), '+', result_add_885, normal_inds_886)
    
    # Applying the binary operator '+=' (line 397)
    result_iadd_888 = python_operator(stypy.reporting.localization.Localization(__file__, 397, 12), '+=', subscripts_882, result_add_887)
    # Assigning a type to the variable 'subscripts' (line 397)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 12), 'subscripts', result_iadd_888)
    
    # SSA join for if statement (line 383)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 340)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 7), 'str', '->')
    # Getting the type of 'subscripts' (line 400)
    subscripts_890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 15), 'subscripts')
    # Applying the binary operator 'in' (line 400)
    result_contains_891 = python_operator(stypy.reporting.localization.Localization(__file__, 400, 7), 'in', str_889, subscripts_890)
    
    # Testing the type of an if condition (line 400)
    if_condition_892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 400, 4), result_contains_891)
    # Assigning a type to the variable 'if_condition_892' (line 400)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 4), 'if_condition_892', if_condition_892)
    # SSA begins for if statement (line 400)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Tuple (line 401):
    
    # Assigning a Subscript to a Name (line 401):
    
    # Obtaining the type of the subscript
    int_893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 8), 'int')
    
    # Call to split(...): (line 401)
    # Processing the call arguments (line 401)
    str_896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 62), 'str', '->')
    # Processing the call keyword arguments (line 401)
    kwargs_897 = {}
    # Getting the type of 'subscripts' (line 401)
    subscripts_894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 45), 'subscripts', False)
    # Obtaining the member 'split' of a type (line 401)
    split_895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 45), subscripts_894, 'split')
    # Calling split(args, kwargs) (line 401)
    split_call_result_898 = invoke(stypy.reporting.localization.Localization(__file__, 401, 45), split_895, *[str_896], **kwargs_897)
    
    # Obtaining the member '__getitem__' of a type (line 401)
    getitem___899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), split_call_result_898, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 401)
    subscript_call_result_900 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), getitem___899, int_893)
    
    # Assigning a type to the variable 'tuple_var_assignment_14' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_14', subscript_call_result_900)
    
    # Assigning a Subscript to a Name (line 401):
    
    # Obtaining the type of the subscript
    int_901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 8), 'int')
    
    # Call to split(...): (line 401)
    # Processing the call arguments (line 401)
    str_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 401, 62), 'str', '->')
    # Processing the call keyword arguments (line 401)
    kwargs_905 = {}
    # Getting the type of 'subscripts' (line 401)
    subscripts_902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 45), 'subscripts', False)
    # Obtaining the member 'split' of a type (line 401)
    split_903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 45), subscripts_902, 'split')
    # Calling split(args, kwargs) (line 401)
    split_call_result_906 = invoke(stypy.reporting.localization.Localization(__file__, 401, 45), split_903, *[str_904], **kwargs_905)
    
    # Obtaining the member '__getitem__' of a type (line 401)
    getitem___907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 8), split_call_result_906, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 401)
    subscript_call_result_908 = invoke(stypy.reporting.localization.Localization(__file__, 401, 8), getitem___907, int_901)
    
    # Assigning a type to the variable 'tuple_var_assignment_15' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_15', subscript_call_result_908)
    
    # Assigning a Name to a Name (line 401):
    # Getting the type of 'tuple_var_assignment_14' (line 401)
    tuple_var_assignment_14_909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_14')
    # Assigning a type to the variable 'input_subscripts' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'input_subscripts', tuple_var_assignment_14_909)
    
    # Assigning a Name to a Name (line 401):
    # Getting the type of 'tuple_var_assignment_15' (line 401)
    tuple_var_assignment_15_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 8), 'tuple_var_assignment_15')
    # Assigning a type to the variable 'output_subscript' (line 401)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 401, 26), 'output_subscript', tuple_var_assignment_15_910)
    # SSA branch for the else part of an if statement (line 400)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 403):
    
    # Assigning a Name to a Name (line 403):
    # Getting the type of 'subscripts' (line 403)
    subscripts_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 27), 'subscripts')
    # Assigning a type to the variable 'input_subscripts' (line 403)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'input_subscripts', subscripts_911)
    
    # Assigning a Call to a Name (line 405):
    
    # Assigning a Call to a Name (line 405):
    
    # Call to replace(...): (line 405)
    # Processing the call arguments (line 405)
    str_914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 44), 'str', ',')
    str_915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 405, 49), 'str', '')
    # Processing the call keyword arguments (line 405)
    kwargs_916 = {}
    # Getting the type of 'subscripts' (line 405)
    subscripts_912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 25), 'subscripts', False)
    # Obtaining the member 'replace' of a type (line 405)
    replace_913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 25), subscripts_912, 'replace')
    # Calling replace(args, kwargs) (line 405)
    replace_call_result_917 = invoke(stypy.reporting.localization.Localization(__file__, 405, 25), replace_913, *[str_914, str_915], **kwargs_916)
    
    # Assigning a type to the variable 'tmp_subscripts' (line 405)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 405, 8), 'tmp_subscripts', replace_call_result_917)
    
    # Assigning a Str to a Name (line 406):
    
    # Assigning a Str to a Name (line 406):
    str_918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 27), 'str', '')
    # Assigning a type to the variable 'output_subscript' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'output_subscript', str_918)
    
    
    # Call to sorted(...): (line 407)
    # Processing the call arguments (line 407)
    
    # Call to set(...): (line 407)
    # Processing the call arguments (line 407)
    # Getting the type of 'tmp_subscripts' (line 407)
    tmp_subscripts_921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 28), 'tmp_subscripts', False)
    # Processing the call keyword arguments (line 407)
    kwargs_922 = {}
    # Getting the type of 'set' (line 407)
    set_920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 24), 'set', False)
    # Calling set(args, kwargs) (line 407)
    set_call_result_923 = invoke(stypy.reporting.localization.Localization(__file__, 407, 24), set_920, *[tmp_subscripts_921], **kwargs_922)
    
    # Processing the call keyword arguments (line 407)
    kwargs_924 = {}
    # Getting the type of 'sorted' (line 407)
    sorted_919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 17), 'sorted', False)
    # Calling sorted(args, kwargs) (line 407)
    sorted_call_result_925 = invoke(stypy.reporting.localization.Localization(__file__, 407, 17), sorted_919, *[set_call_result_923], **kwargs_924)
    
    # Testing the type of a for loop iterable (line 407)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 407, 8), sorted_call_result_925)
    # Getting the type of the for loop variable (line 407)
    for_loop_var_926 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 407, 8), sorted_call_result_925)
    # Assigning a type to the variable 's' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 8), 's', for_loop_var_926)
    # SSA begins for a for statement (line 407)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 's' (line 408)
    s_927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 15), 's')
    # Getting the type of 'einsum_symbols' (line 408)
    einsum_symbols_928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 24), 'einsum_symbols')
    # Applying the binary operator 'notin' (line 408)
    result_contains_929 = python_operator(stypy.reporting.localization.Localization(__file__, 408, 15), 'notin', s_927, einsum_symbols_928)
    
    # Testing the type of an if condition (line 408)
    if_condition_930 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 408, 12), result_contains_929)
    # Assigning a type to the variable 'if_condition_930' (line 408)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 408, 12), 'if_condition_930', if_condition_930)
    # SSA begins for if statement (line 408)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 409)
    # Processing the call arguments (line 409)
    str_932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 33), 'str', 'Character %s is not a valid symbol.')
    # Getting the type of 's' (line 409)
    s_933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 73), 's', False)
    # Applying the binary operator '%' (line 409)
    result_mod_934 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 33), '%', str_932, s_933)
    
    # Processing the call keyword arguments (line 409)
    kwargs_935 = {}
    # Getting the type of 'ValueError' (line 409)
    ValueError_931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 22), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 409)
    ValueError_call_result_936 = invoke(stypy.reporting.localization.Localization(__file__, 409, 22), ValueError_931, *[result_mod_934], **kwargs_935)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 409, 16), ValueError_call_result_936, 'raise parameter', BaseException)
    # SSA join for if statement (line 408)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to count(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 's' (line 410)
    s_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 36), 's', False)
    # Processing the call keyword arguments (line 410)
    kwargs_940 = {}
    # Getting the type of 'tmp_subscripts' (line 410)
    tmp_subscripts_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 15), 'tmp_subscripts', False)
    # Obtaining the member 'count' of a type (line 410)
    count_938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 15), tmp_subscripts_937, 'count')
    # Calling count(args, kwargs) (line 410)
    count_call_result_941 = invoke(stypy.reporting.localization.Localization(__file__, 410, 15), count_938, *[s_939], **kwargs_940)
    
    int_942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 42), 'int')
    # Applying the binary operator '==' (line 410)
    result_eq_943 = python_operator(stypy.reporting.localization.Localization(__file__, 410, 15), '==', count_call_result_941, int_942)
    
    # Testing the type of an if condition (line 410)
    if_condition_944 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 410, 12), result_eq_943)
    # Assigning a type to the variable 'if_condition_944' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'if_condition_944', if_condition_944)
    # SSA begins for if statement (line 410)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'output_subscript' (line 411)
    output_subscript_945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'output_subscript')
    # Getting the type of 's' (line 411)
    s_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 36), 's')
    # Applying the binary operator '+=' (line 411)
    result_iadd_947 = python_operator(stypy.reporting.localization.Localization(__file__, 411, 16), '+=', output_subscript_945, s_946)
    # Assigning a type to the variable 'output_subscript' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 16), 'output_subscript', result_iadd_947)
    
    # SSA join for if statement (line 410)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 400)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'output_subscript' (line 414)
    output_subscript_948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 16), 'output_subscript')
    # Testing the type of a for loop iterable (line 414)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 414, 4), output_subscript_948)
    # Getting the type of the for loop variable (line 414)
    for_loop_var_949 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 414, 4), output_subscript_948)
    # Assigning a type to the variable 'char' (line 414)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 4), 'char', for_loop_var_949)
    # SSA begins for a for statement (line 414)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'char' (line 415)
    char_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 11), 'char')
    # Getting the type of 'input_subscripts' (line 415)
    input_subscripts_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 23), 'input_subscripts')
    # Applying the binary operator 'notin' (line 415)
    result_contains_952 = python_operator(stypy.reporting.localization.Localization(__file__, 415, 11), 'notin', char_950, input_subscripts_951)
    
    # Testing the type of an if condition (line 415)
    if_condition_953 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 415, 8), result_contains_952)
    # Assigning a type to the variable 'if_condition_953' (line 415)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'if_condition_953', if_condition_953)
    # SSA begins for if statement (line 415)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 416)
    # Processing the call arguments (line 416)
    str_955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 29), 'str', 'Output character %s did not appear in the input')
    # Getting the type of 'char' (line 417)
    char_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 31), 'char', False)
    # Applying the binary operator '%' (line 416)
    result_mod_957 = python_operator(stypy.reporting.localization.Localization(__file__, 416, 29), '%', str_955, char_956)
    
    # Processing the call keyword arguments (line 416)
    kwargs_958 = {}
    # Getting the type of 'ValueError' (line 416)
    ValueError_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 416)
    ValueError_call_result_959 = invoke(stypy.reporting.localization.Localization(__file__, 416, 18), ValueError_954, *[result_mod_957], **kwargs_958)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 416, 12), ValueError_call_result_959, 'raise parameter', BaseException)
    # SSA join for if statement (line 415)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to len(...): (line 420)
    # Processing the call arguments (line 420)
    
    # Call to split(...): (line 420)
    # Processing the call arguments (line 420)
    str_963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 34), 'str', ',')
    # Processing the call keyword arguments (line 420)
    kwargs_964 = {}
    # Getting the type of 'input_subscripts' (line 420)
    input_subscripts_961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 11), 'input_subscripts', False)
    # Obtaining the member 'split' of a type (line 420)
    split_962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 11), input_subscripts_961, 'split')
    # Calling split(args, kwargs) (line 420)
    split_call_result_965 = invoke(stypy.reporting.localization.Localization(__file__, 420, 11), split_962, *[str_963], **kwargs_964)
    
    # Processing the call keyword arguments (line 420)
    kwargs_966 = {}
    # Getting the type of 'len' (line 420)
    len_960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 7), 'len', False)
    # Calling len(args, kwargs) (line 420)
    len_call_result_967 = invoke(stypy.reporting.localization.Localization(__file__, 420, 7), len_960, *[split_call_result_965], **kwargs_966)
    
    
    # Call to len(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'operands' (line 420)
    operands_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 47), 'operands', False)
    # Processing the call keyword arguments (line 420)
    kwargs_970 = {}
    # Getting the type of 'len' (line 420)
    len_968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 43), 'len', False)
    # Calling len(args, kwargs) (line 420)
    len_call_result_971 = invoke(stypy.reporting.localization.Localization(__file__, 420, 43), len_968, *[operands_969], **kwargs_970)
    
    # Applying the binary operator '!=' (line 420)
    result_ne_972 = python_operator(stypy.reporting.localization.Localization(__file__, 420, 7), '!=', len_call_result_967, len_call_result_971)
    
    # Testing the type of an if condition (line 420)
    if_condition_973 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 420, 4), result_ne_972)
    # Assigning a type to the variable 'if_condition_973' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'if_condition_973', if_condition_973)
    # SSA begins for if statement (line 420)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 421)
    # Processing the call arguments (line 421)
    str_975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 421, 25), 'str', 'Number of einsum subscripts must be equal to the number of operands.')
    # Processing the call keyword arguments (line 421)
    kwargs_976 = {}
    # Getting the type of 'ValueError' (line 421)
    ValueError_974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 421)
    ValueError_call_result_977 = invoke(stypy.reporting.localization.Localization(__file__, 421, 14), ValueError_974, *[str_975], **kwargs_976)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 421, 8), ValueError_call_result_977, 'raise parameter', BaseException)
    # SSA join for if statement (line 420)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 424)
    tuple_978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 424, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 424)
    # Adding element type (line 424)
    # Getting the type of 'input_subscripts' (line 424)
    input_subscripts_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'input_subscripts')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 12), tuple_978, input_subscripts_979)
    # Adding element type (line 424)
    # Getting the type of 'output_subscript' (line 424)
    output_subscript_980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 30), 'output_subscript')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 12), tuple_978, output_subscript_980)
    # Adding element type (line 424)
    # Getting the type of 'operands' (line 424)
    operands_981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 48), 'operands')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 424, 12), tuple_978, operands_981)
    
    # Assigning a type to the variable 'stypy_return_type' (line 424)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 424, 4), 'stypy_return_type', tuple_978)
    
    # ################# End of '_parse_einsum_input(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_parse_einsum_input' in the type store
    # Getting the type of 'stypy_return_type' (line 259)
    stypy_return_type_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_982)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_parse_einsum_input'
    return stypy_return_type_982

# Assigning a type to the variable '_parse_einsum_input' (line 259)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 0), '_parse_einsum_input', _parse_einsum_input)

@norecursion
def einsum_path(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'einsum_path'
    module_type_store = module_type_store.open_function_context('einsum_path', 427, 0, False)
    
    # Passed parameters checking function
    einsum_path.stypy_localization = localization
    einsum_path.stypy_type_of_self = None
    einsum_path.stypy_type_store = module_type_store
    einsum_path.stypy_function_name = 'einsum_path'
    einsum_path.stypy_param_names_list = []
    einsum_path.stypy_varargs_param_name = 'operands'
    einsum_path.stypy_kwargs_param_name = 'kwargs'
    einsum_path.stypy_call_defaults = defaults
    einsum_path.stypy_call_varargs = varargs
    einsum_path.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'einsum_path', [], 'operands', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'einsum_path', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'einsum_path(...)' code ##################

    str_983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, (-1)), 'str', "\n    einsum_path(subscripts, *operands, optimize='greedy')\n\n    Evaluates the lowest cost contraction order for an einsum expression by\n    considering the creation of intermediate arrays.\n\n    Parameters\n    ----------\n    subscripts : str\n        Specifies the subscripts for summation.\n    *operands : list of array_like\n        These are the arrays for the operation.\n    optimize : {bool, list, tuple, 'greedy', 'optimal'}\n        Choose the type of path. If a tuple is provided, the second argument is\n        assumed to be the maximum intermediate size created. If only a single\n        argument is provided the largest input or output array size is used\n        as a maximum intermediate size.\n\n        * if a list is given that starts with ``einsum_path``, uses this as the\n          contraction path\n        * if False no optimization is taken\n        * if True defaults to the 'greedy' algorithm\n        * 'optimal' An algorithm that combinatorially explores all possible\n          ways of contracting the listed tensors and choosest the least costly\n          path. Scales exponentially with the number of terms in the\n          contraction.\n        * 'greedy' An algorithm that chooses the best pair contraction\n          at each step. Effectively, this algorithm searches the largest inner,\n          Hadamard, and then outer products at each step. Scales cubically with\n          the number of terms in the contraction. Equivalent to the 'optimal'\n          path for most contractions.\n\n        Default is 'greedy'.\n\n    Returns\n    -------\n    path : list of tuples\n        A list representation of the einsum path.\n    string_repr : str\n        A printable representation of the einsum path.\n\n    Notes\n    -----\n    The resulting path indicates which terms of the input contraction should be\n    contracted first, the result of this contraction is then appended to the\n    end of the contraction list. This list can then be iterated over until all\n    intermediate contractions are complete.\n\n    See Also\n    --------\n    einsum, linalg.multi_dot\n\n    Examples\n    --------\n\n    We can begin with a chain dot example. In this case, it is optimal to\n    contract the ``b`` and ``c`` tensors first as reprsented by the first\n    element of the path ``(1, 2)``. The resulting tensor is added to the end\n    of the contraction and the remaining contraction ``(0, 1)`` is then\n    completed.\n\n    >>> a = np.random.rand(2, 2)\n    >>> b = np.random.rand(2, 5)\n    >>> c = np.random.rand(5, 2)\n    >>> path_info = np.einsum_path('ij,jk,kl->il', a, b, c, optimize='greedy')\n    >>> print(path_info[0])\n    ['einsum_path', (1, 2), (0, 1)]\n    >>> print(path_info[1])\n      Complete contraction:  ij,jk,kl->il\n             Naive scaling:  4\n         Optimized scaling:  3\n          Naive FLOP count:  1.600e+02\n      Optimized FLOP count:  5.600e+01\n       Theoretical speedup:  2.857\n      Largest intermediate:  4.000e+00 elements\n    -------------------------------------------------------------------------\n    scaling                  current                                remaining\n    -------------------------------------------------------------------------\n       3                   kl,jk->jl                                ij,jl->il\n       3                   jl,ij->il                                   il->il\n\n\n    A more complex index transformation example.\n\n    >>> I = np.random.rand(10, 10, 10, 10)\n    >>> C = np.random.rand(10, 10)\n    >>> path_info = np.einsum_path('ea,fb,abcd,gc,hd->efgh', C, C, I, C, C,\n                                   optimize='greedy')\n\n    >>> print(path_info[0])\n    ['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)]\n    >>> print(path_info[1])\n      Complete contraction:  ea,fb,abcd,gc,hd->efgh\n             Naive scaling:  8\n         Optimized scaling:  5\n          Naive FLOP count:  8.000e+08\n      Optimized FLOP count:  8.000e+05\n       Theoretical speedup:  1000.000\n      Largest intermediate:  1.000e+04 elements\n    --------------------------------------------------------------------------\n    scaling                  current                                remaining\n    --------------------------------------------------------------------------\n       5               abcd,ea->bcde                      fb,gc,hd,bcde->efgh\n       5               bcde,fb->cdef                         gc,hd,cdef->efgh\n       5               cdef,gc->defg                            hd,defg->efgh\n       5               defg,hd->efgh                               efgh->efgh\n    ")
    
    # Assigning a List to a Name (line 537):
    
    # Assigning a List to a Name (line 537):
    
    # Obtaining an instance of the builtin type 'list' (line 537)
    list_984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 537)
    # Adding element type (line 537)
    str_985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 29), 'str', 'optimize')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 28), list_984, str_985)
    # Adding element type (line 537)
    str_986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, 41), 'str', 'einsum_call')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 537, 28), list_984, str_986)
    
    # Assigning a type to the variable 'valid_contract_kwargs' (line 537)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'valid_contract_kwargs', list_984)
    
    # Assigning a ListComp to a Name (line 538):
    
    # Assigning a ListComp to a Name (line 538):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to items(...): (line 538)
    # Processing the call keyword arguments (line 538)
    kwargs_993 = {}
    # Getting the type of 'kwargs' (line 538)
    kwargs_991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 38), 'kwargs', False)
    # Obtaining the member 'items' of a type (line 538)
    items_992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 38), kwargs_991, 'items')
    # Calling items(args, kwargs) (line 538)
    items_call_result_994 = invoke(stypy.reporting.localization.Localization(__file__, 538, 38), items_992, *[], **kwargs_993)
    
    comprehension_995 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 22), items_call_result_994)
    # Assigning a type to the variable 'k' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 22), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 22), comprehension_995))
    # Assigning a type to the variable 'v' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 22), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 22), comprehension_995))
    
    # Getting the type of 'k' (line 538)
    k_988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 56), 'k')
    # Getting the type of 'valid_contract_kwargs' (line 539)
    valid_contract_kwargs_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 29), 'valid_contract_kwargs')
    # Applying the binary operator 'notin' (line 538)
    result_contains_990 = python_operator(stypy.reporting.localization.Localization(__file__, 538, 56), 'notin', k_988, valid_contract_kwargs_989)
    
    # Getting the type of 'k' (line 538)
    k_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 22), 'k')
    list_996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 22), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 538, 22), list_996, k_987)
    # Assigning a type to the variable 'unknown_kwargs' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'unknown_kwargs', list_996)
    
    
    # Call to len(...): (line 540)
    # Processing the call arguments (line 540)
    # Getting the type of 'unknown_kwargs' (line 540)
    unknown_kwargs_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 11), 'unknown_kwargs', False)
    # Processing the call keyword arguments (line 540)
    kwargs_999 = {}
    # Getting the type of 'len' (line 540)
    len_997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 7), 'len', False)
    # Calling len(args, kwargs) (line 540)
    len_call_result_1000 = invoke(stypy.reporting.localization.Localization(__file__, 540, 7), len_997, *[unknown_kwargs_998], **kwargs_999)
    
    # Testing the type of an if condition (line 540)
    if_condition_1001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 540, 4), len_call_result_1000)
    # Assigning a type to the variable 'if_condition_1001' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'if_condition_1001', if_condition_1001)
    # SSA begins for if statement (line 540)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 541)
    # Processing the call arguments (line 541)
    str_1003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 541, 24), 'str', 'Did not understand the following kwargs: %s')
    # Getting the type of 'unknown_kwargs' (line 542)
    unknown_kwargs_1004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 32), 'unknown_kwargs', False)
    # Applying the binary operator '%' (line 541)
    result_mod_1005 = python_operator(stypy.reporting.localization.Localization(__file__, 541, 24), '%', str_1003, unknown_kwargs_1004)
    
    # Processing the call keyword arguments (line 541)
    kwargs_1006 = {}
    # Getting the type of 'TypeError' (line 541)
    TypeError_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 541)
    TypeError_call_result_1007 = invoke(stypy.reporting.localization.Localization(__file__, 541, 14), TypeError_1002, *[result_mod_1005], **kwargs_1006)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 541, 8), TypeError_call_result_1007, 'raise parameter', BaseException)
    # SSA join for if statement (line 540)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 545):
    
    # Assigning a Call to a Name (line 545):
    
    # Call to pop(...): (line 545)
    # Processing the call arguments (line 545)
    str_1010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 545, 27), 'str', 'optimize')
    # Getting the type of 'False' (line 545)
    False_1011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 39), 'False', False)
    # Processing the call keyword arguments (line 545)
    kwargs_1012 = {}
    # Getting the type of 'kwargs' (line 545)
    kwargs_1008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 16), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 545)
    pop_1009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 16), kwargs_1008, 'pop')
    # Calling pop(args, kwargs) (line 545)
    pop_call_result_1013 = invoke(stypy.reporting.localization.Localization(__file__, 545, 16), pop_1009, *[str_1010, False_1011], **kwargs_1012)
    
    # Assigning a type to the variable 'path_type' (line 545)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 4), 'path_type', pop_call_result_1013)
    
    
    # Getting the type of 'path_type' (line 546)
    path_type_1014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 7), 'path_type')
    # Getting the type of 'True' (line 546)
    True_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 20), 'True')
    # Applying the binary operator 'is' (line 546)
    result_is__1016 = python_operator(stypy.reporting.localization.Localization(__file__, 546, 7), 'is', path_type_1014, True_1015)
    
    # Testing the type of an if condition (line 546)
    if_condition_1017 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 546, 4), result_is__1016)
    # Assigning a type to the variable 'if_condition_1017' (line 546)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 4), 'if_condition_1017', if_condition_1017)
    # SSA begins for if statement (line 546)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 547):
    
    # Assigning a Str to a Name (line 547):
    str_1018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 20), 'str', 'greedy')
    # Assigning a type to the variable 'path_type' (line 547)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 8), 'path_type', str_1018)
    # SSA join for if statement (line 546)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 548)
    # Getting the type of 'path_type' (line 548)
    path_type_1019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 7), 'path_type')
    # Getting the type of 'None' (line 548)
    None_1020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 20), 'None')
    
    (may_be_1021, more_types_in_union_1022) = may_be_none(path_type_1019, None_1020)

    if may_be_1021:

        if more_types_in_union_1022:
            # Runtime conditional SSA (line 548)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 549):
        
        # Assigning a Name to a Name (line 549):
        # Getting the type of 'False' (line 549)
        False_1023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 20), 'False')
        # Assigning a type to the variable 'path_type' (line 549)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 8), 'path_type', False_1023)

        if more_types_in_union_1022:
            # SSA join for if statement (line 548)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Name to a Name (line 551):
    
    # Assigning a Name to a Name (line 551):
    # Getting the type of 'None' (line 551)
    None_1024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 19), 'None')
    # Assigning a type to the variable 'memory_limit' (line 551)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 551, 4), 'memory_limit', None_1024)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'path_type' (line 554)
    path_type_1025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 8), 'path_type')
    # Getting the type of 'False' (line 554)
    False_1026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 21), 'False')
    # Applying the binary operator 'is' (line 554)
    result_is__1027 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 8), 'is', path_type_1025, False_1026)
    
    
    # Call to isinstance(...): (line 554)
    # Processing the call arguments (line 554)
    # Getting the type of 'path_type' (line 554)
    path_type_1029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 42), 'path_type', False)
    # Getting the type of 'str' (line 554)
    str_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 53), 'str', False)
    # Processing the call keyword arguments (line 554)
    kwargs_1031 = {}
    # Getting the type of 'isinstance' (line 554)
    isinstance_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 31), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 554)
    isinstance_call_result_1032 = invoke(stypy.reporting.localization.Localization(__file__, 554, 31), isinstance_1028, *[path_type_1029, str_1030], **kwargs_1031)
    
    # Applying the binary operator 'or' (line 554)
    result_or_keyword_1033 = python_operator(stypy.reporting.localization.Localization(__file__, 554, 7), 'or', result_is__1027, isinstance_call_result_1032)
    
    # Testing the type of an if condition (line 554)
    if_condition_1034 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 554, 4), result_or_keyword_1033)
    # Assigning a type to the variable 'if_condition_1034' (line 554)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'if_condition_1034', if_condition_1034)
    # SSA begins for if statement (line 554)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 554)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Call to len(...): (line 558)
    # Processing the call arguments (line 558)
    # Getting the type of 'path_type' (line 558)
    path_type_1036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 13), 'path_type', False)
    # Processing the call keyword arguments (line 558)
    kwargs_1037 = {}
    # Getting the type of 'len' (line 558)
    len_1035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 9), 'len', False)
    # Calling len(args, kwargs) (line 558)
    len_call_result_1038 = invoke(stypy.reporting.localization.Localization(__file__, 558, 9), len_1035, *[path_type_1036], **kwargs_1037)
    
    
    
    # Obtaining the type of the subscript
    int_1039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 39), 'int')
    # Getting the type of 'path_type' (line 558)
    path_type_1040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 29), 'path_type')
    # Obtaining the member '__getitem__' of a type (line 558)
    getitem___1041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 29), path_type_1040, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 558)
    subscript_call_result_1042 = invoke(stypy.reporting.localization.Localization(__file__, 558, 29), getitem___1041, int_1039)
    
    str_1043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 45), 'str', 'einsum_path')
    # Applying the binary operator '==' (line 558)
    result_eq_1044 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 29), '==', subscript_call_result_1042, str_1043)
    
    # Applying the binary operator 'and' (line 558)
    result_and_keyword_1045 = python_operator(stypy.reporting.localization.Localization(__file__, 558, 9), 'and', len_call_result_1038, result_eq_1044)
    
    # Testing the type of an if condition (line 558)
    if_condition_1046 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 558, 9), result_and_keyword_1045)
    # Assigning a type to the variable 'if_condition_1046' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 9), 'if_condition_1046', if_condition_1046)
    # SSA begins for if statement (line 558)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 558)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 562)
    # Processing the call arguments (line 562)
    # Getting the type of 'path_type' (line 562)
    path_type_1048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 15), 'path_type', False)
    # Processing the call keyword arguments (line 562)
    kwargs_1049 = {}
    # Getting the type of 'len' (line 562)
    len_1047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 11), 'len', False)
    # Calling len(args, kwargs) (line 562)
    len_call_result_1050 = invoke(stypy.reporting.localization.Localization(__file__, 562, 11), len_1047, *[path_type_1048], **kwargs_1049)
    
    int_1051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 29), 'int')
    # Applying the binary operator '==' (line 562)
    result_eq_1052 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 11), '==', len_call_result_1050, int_1051)
    
    
    # Call to isinstance(...): (line 562)
    # Processing the call arguments (line 562)
    
    # Obtaining the type of the subscript
    int_1054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 57), 'int')
    # Getting the type of 'path_type' (line 562)
    path_type_1055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 47), 'path_type', False)
    # Obtaining the member '__getitem__' of a type (line 562)
    getitem___1056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 47), path_type_1055, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 562)
    subscript_call_result_1057 = invoke(stypy.reporting.localization.Localization(__file__, 562, 47), getitem___1056, int_1054)
    
    # Getting the type of 'str' (line 562)
    str_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 61), 'str', False)
    # Processing the call keyword arguments (line 562)
    kwargs_1059 = {}
    # Getting the type of 'isinstance' (line 562)
    isinstance_1053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 36), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 562)
    isinstance_call_result_1060 = invoke(stypy.reporting.localization.Localization(__file__, 562, 36), isinstance_1053, *[subscript_call_result_1057, str_1058], **kwargs_1059)
    
    # Applying the binary operator 'and' (line 562)
    result_and_keyword_1061 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 10), 'and', result_eq_1052, isinstance_call_result_1060)
    
    # Call to isinstance(...): (line 563)
    # Processing the call arguments (line 563)
    
    # Obtaining the type of the subscript
    int_1063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 33), 'int')
    # Getting the type of 'path_type' (line 563)
    path_type_1064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 23), 'path_type', False)
    # Obtaining the member '__getitem__' of a type (line 563)
    getitem___1065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 563, 23), path_type_1064, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 563)
    subscript_call_result_1066 = invoke(stypy.reporting.localization.Localization(__file__, 563, 23), getitem___1065, int_1063)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 563)
    tuple_1067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 38), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 563)
    # Adding element type (line 563)
    # Getting the type of 'int' (line 563)
    int_1068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 38), 'int', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 38), tuple_1067, int_1068)
    # Adding element type (line 563)
    # Getting the type of 'float' (line 563)
    float_1069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 43), 'float', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 38), tuple_1067, float_1069)
    
    # Processing the call keyword arguments (line 563)
    kwargs_1070 = {}
    # Getting the type of 'isinstance' (line 563)
    isinstance_1062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 12), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 563)
    isinstance_call_result_1071 = invoke(stypy.reporting.localization.Localization(__file__, 563, 12), isinstance_1062, *[subscript_call_result_1066, tuple_1067], **kwargs_1070)
    
    # Applying the binary operator 'and' (line 562)
    result_and_keyword_1072 = python_operator(stypy.reporting.localization.Localization(__file__, 562, 10), 'and', result_and_keyword_1061, isinstance_call_result_1071)
    
    # Testing the type of an if condition (line 562)
    if_condition_1073 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 562, 9), result_and_keyword_1072)
    # Assigning a type to the variable 'if_condition_1073' (line 562)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 9), 'if_condition_1073', if_condition_1073)
    # SSA begins for if statement (line 562)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 564):
    
    # Assigning a Call to a Name (line 564):
    
    # Call to int(...): (line 564)
    # Processing the call arguments (line 564)
    
    # Obtaining the type of the subscript
    int_1075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 37), 'int')
    # Getting the type of 'path_type' (line 564)
    path_type_1076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 27), 'path_type', False)
    # Obtaining the member '__getitem__' of a type (line 564)
    getitem___1077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 27), path_type_1076, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 564)
    subscript_call_result_1078 = invoke(stypy.reporting.localization.Localization(__file__, 564, 27), getitem___1077, int_1075)
    
    # Processing the call keyword arguments (line 564)
    kwargs_1079 = {}
    # Getting the type of 'int' (line 564)
    int_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 23), 'int', False)
    # Calling int(args, kwargs) (line 564)
    int_call_result_1080 = invoke(stypy.reporting.localization.Localization(__file__, 564, 23), int_1074, *[subscript_call_result_1078], **kwargs_1079)
    
    # Assigning a type to the variable 'memory_limit' (line 564)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 564, 8), 'memory_limit', int_call_result_1080)
    
    # Assigning a Subscript to a Name (line 565):
    
    # Assigning a Subscript to a Name (line 565):
    
    # Obtaining the type of the subscript
    int_1081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 565, 30), 'int')
    # Getting the type of 'path_type' (line 565)
    path_type_1082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 20), 'path_type')
    # Obtaining the member '__getitem__' of a type (line 565)
    getitem___1083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 20), path_type_1082, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 565)
    subscript_call_result_1084 = invoke(stypy.reporting.localization.Localization(__file__, 565, 20), getitem___1083, int_1081)
    
    # Assigning a type to the variable 'path_type' (line 565)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'path_type', subscript_call_result_1084)
    # SSA branch for the else part of an if statement (line 562)
    module_type_store.open_ssa_branch('else')
    
    # Call to TypeError(...): (line 568)
    # Processing the call arguments (line 568)
    str_1086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 568, 24), 'str', 'Did not understand the path: %s')
    
    # Call to str(...): (line 568)
    # Processing the call arguments (line 568)
    # Getting the type of 'path_type' (line 568)
    path_type_1088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 64), 'path_type', False)
    # Processing the call keyword arguments (line 568)
    kwargs_1089 = {}
    # Getting the type of 'str' (line 568)
    str_1087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 60), 'str', False)
    # Calling str(args, kwargs) (line 568)
    str_call_result_1090 = invoke(stypy.reporting.localization.Localization(__file__, 568, 60), str_1087, *[path_type_1088], **kwargs_1089)
    
    # Applying the binary operator '%' (line 568)
    result_mod_1091 = python_operator(stypy.reporting.localization.Localization(__file__, 568, 24), '%', str_1086, str_call_result_1090)
    
    # Processing the call keyword arguments (line 568)
    kwargs_1092 = {}
    # Getting the type of 'TypeError' (line 568)
    TypeError_1085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 568)
    TypeError_call_result_1093 = invoke(stypy.reporting.localization.Localization(__file__, 568, 14), TypeError_1085, *[result_mod_1091], **kwargs_1092)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 568, 8), TypeError_call_result_1093, 'raise parameter', BaseException)
    # SSA join for if statement (line 562)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 558)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 554)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 571):
    
    # Assigning a Call to a Name (line 571):
    
    # Call to pop(...): (line 571)
    # Processing the call arguments (line 571)
    str_1096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 571, 33), 'str', 'einsum_call')
    # Getting the type of 'False' (line 571)
    False_1097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 48), 'False', False)
    # Processing the call keyword arguments (line 571)
    kwargs_1098 = {}
    # Getting the type of 'kwargs' (line 571)
    kwargs_1094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 22), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 571)
    pop_1095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 22), kwargs_1094, 'pop')
    # Calling pop(args, kwargs) (line 571)
    pop_call_result_1099 = invoke(stypy.reporting.localization.Localization(__file__, 571, 22), pop_1095, *[str_1096, False_1097], **kwargs_1098)
    
    # Assigning a type to the variable 'einsum_call_arg' (line 571)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 4), 'einsum_call_arg', pop_call_result_1099)
    
    # Assigning a Call to a Tuple (line 574):
    
    # Assigning a Subscript to a Name (line 574):
    
    # Obtaining the type of the subscript
    int_1100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 4), 'int')
    
    # Call to _parse_einsum_input(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'operands' (line 574)
    operands_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 71), 'operands', False)
    # Processing the call keyword arguments (line 574)
    kwargs_1103 = {}
    # Getting the type of '_parse_einsum_input' (line 574)
    _parse_einsum_input_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 51), '_parse_einsum_input', False)
    # Calling _parse_einsum_input(args, kwargs) (line 574)
    _parse_einsum_input_call_result_1104 = invoke(stypy.reporting.localization.Localization(__file__, 574, 51), _parse_einsum_input_1101, *[operands_1102], **kwargs_1103)
    
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___1105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 4), _parse_einsum_input_call_result_1104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_1106 = invoke(stypy.reporting.localization.Localization(__file__, 574, 4), getitem___1105, int_1100)
    
    # Assigning a type to the variable 'tuple_var_assignment_16' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_16', subscript_call_result_1106)
    
    # Assigning a Subscript to a Name (line 574):
    
    # Obtaining the type of the subscript
    int_1107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 4), 'int')
    
    # Call to _parse_einsum_input(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'operands' (line 574)
    operands_1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 71), 'operands', False)
    # Processing the call keyword arguments (line 574)
    kwargs_1110 = {}
    # Getting the type of '_parse_einsum_input' (line 574)
    _parse_einsum_input_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 51), '_parse_einsum_input', False)
    # Calling _parse_einsum_input(args, kwargs) (line 574)
    _parse_einsum_input_call_result_1111 = invoke(stypy.reporting.localization.Localization(__file__, 574, 51), _parse_einsum_input_1108, *[operands_1109], **kwargs_1110)
    
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___1112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 4), _parse_einsum_input_call_result_1111, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_1113 = invoke(stypy.reporting.localization.Localization(__file__, 574, 4), getitem___1112, int_1107)
    
    # Assigning a type to the variable 'tuple_var_assignment_17' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_17', subscript_call_result_1113)
    
    # Assigning a Subscript to a Name (line 574):
    
    # Obtaining the type of the subscript
    int_1114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 4), 'int')
    
    # Call to _parse_einsum_input(...): (line 574)
    # Processing the call arguments (line 574)
    # Getting the type of 'operands' (line 574)
    operands_1116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 71), 'operands', False)
    # Processing the call keyword arguments (line 574)
    kwargs_1117 = {}
    # Getting the type of '_parse_einsum_input' (line 574)
    _parse_einsum_input_1115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 51), '_parse_einsum_input', False)
    # Calling _parse_einsum_input(args, kwargs) (line 574)
    _parse_einsum_input_call_result_1118 = invoke(stypy.reporting.localization.Localization(__file__, 574, 51), _parse_einsum_input_1115, *[operands_1116], **kwargs_1117)
    
    # Obtaining the member '__getitem__' of a type (line 574)
    getitem___1119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 4), _parse_einsum_input_call_result_1118, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 574)
    subscript_call_result_1120 = invoke(stypy.reporting.localization.Localization(__file__, 574, 4), getitem___1119, int_1114)
    
    # Assigning a type to the variable 'tuple_var_assignment_18' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_18', subscript_call_result_1120)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'tuple_var_assignment_16' (line 574)
    tuple_var_assignment_16_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_16')
    # Assigning a type to the variable 'input_subscripts' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'input_subscripts', tuple_var_assignment_16_1121)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'tuple_var_assignment_17' (line 574)
    tuple_var_assignment_17_1122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_17')
    # Assigning a type to the variable 'output_subscript' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 22), 'output_subscript', tuple_var_assignment_17_1122)
    
    # Assigning a Name to a Name (line 574):
    # Getting the type of 'tuple_var_assignment_18' (line 574)
    tuple_var_assignment_18_1123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 4), 'tuple_var_assignment_18')
    # Assigning a type to the variable 'operands' (line 574)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 574, 40), 'operands', tuple_var_assignment_18_1123)
    
    # Assigning a BinOp to a Name (line 575):
    
    # Assigning a BinOp to a Name (line 575):
    # Getting the type of 'input_subscripts' (line 575)
    input_subscripts_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 17), 'input_subscripts')
    str_1125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 575, 36), 'str', '->')
    # Applying the binary operator '+' (line 575)
    result_add_1126 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 17), '+', input_subscripts_1124, str_1125)
    
    # Getting the type of 'output_subscript' (line 575)
    output_subscript_1127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 43), 'output_subscript')
    # Applying the binary operator '+' (line 575)
    result_add_1128 = python_operator(stypy.reporting.localization.Localization(__file__, 575, 41), '+', result_add_1126, output_subscript_1127)
    
    # Assigning a type to the variable 'subscripts' (line 575)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'subscripts', result_add_1128)
    
    # Assigning a Call to a Name (line 578):
    
    # Assigning a Call to a Name (line 578):
    
    # Call to split(...): (line 578)
    # Processing the call arguments (line 578)
    str_1131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 40), 'str', ',')
    # Processing the call keyword arguments (line 578)
    kwargs_1132 = {}
    # Getting the type of 'input_subscripts' (line 578)
    input_subscripts_1129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 17), 'input_subscripts', False)
    # Obtaining the member 'split' of a type (line 578)
    split_1130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 17), input_subscripts_1129, 'split')
    # Calling split(args, kwargs) (line 578)
    split_call_result_1133 = invoke(stypy.reporting.localization.Localization(__file__, 578, 17), split_1130, *[str_1131], **kwargs_1132)
    
    # Assigning a type to the variable 'input_list' (line 578)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'input_list', split_call_result_1133)
    
    # Assigning a ListComp to a Name (line 579):
    
    # Assigning a ListComp to a Name (line 579):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'input_list' (line 579)
    input_list_1138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 34), 'input_list')
    comprehension_1139 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 18), input_list_1138)
    # Assigning a type to the variable 'x' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 18), 'x', comprehension_1139)
    
    # Call to set(...): (line 579)
    # Processing the call arguments (line 579)
    # Getting the type of 'x' (line 579)
    x_1135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 22), 'x', False)
    # Processing the call keyword arguments (line 579)
    kwargs_1136 = {}
    # Getting the type of 'set' (line 579)
    set_1134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 18), 'set', False)
    # Calling set(args, kwargs) (line 579)
    set_call_result_1137 = invoke(stypy.reporting.localization.Localization(__file__, 579, 18), set_1134, *[x_1135], **kwargs_1136)
    
    list_1140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 579, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 579, 18), list_1140, set_call_result_1137)
    # Assigning a type to the variable 'input_sets' (line 579)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'input_sets', list_1140)
    
    # Assigning a Call to a Name (line 580):
    
    # Assigning a Call to a Name (line 580):
    
    # Call to set(...): (line 580)
    # Processing the call arguments (line 580)
    # Getting the type of 'output_subscript' (line 580)
    output_subscript_1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 21), 'output_subscript', False)
    # Processing the call keyword arguments (line 580)
    kwargs_1143 = {}
    # Getting the type of 'set' (line 580)
    set_1141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 580, 17), 'set', False)
    # Calling set(args, kwargs) (line 580)
    set_call_result_1144 = invoke(stypy.reporting.localization.Localization(__file__, 580, 17), set_1141, *[output_subscript_1142], **kwargs_1143)
    
    # Assigning a type to the variable 'output_set' (line 580)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 580, 4), 'output_set', set_call_result_1144)
    
    # Assigning a Call to a Name (line 581):
    
    # Assigning a Call to a Name (line 581):
    
    # Call to set(...): (line 581)
    # Processing the call arguments (line 581)
    
    # Call to replace(...): (line 581)
    # Processing the call arguments (line 581)
    str_1148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 43), 'str', ',')
    str_1149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 581, 48), 'str', '')
    # Processing the call keyword arguments (line 581)
    kwargs_1150 = {}
    # Getting the type of 'input_subscripts' (line 581)
    input_subscripts_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 18), 'input_subscripts', False)
    # Obtaining the member 'replace' of a type (line 581)
    replace_1147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 581, 18), input_subscripts_1146, 'replace')
    # Calling replace(args, kwargs) (line 581)
    replace_call_result_1151 = invoke(stypy.reporting.localization.Localization(__file__, 581, 18), replace_1147, *[str_1148, str_1149], **kwargs_1150)
    
    # Processing the call keyword arguments (line 581)
    kwargs_1152 = {}
    # Getting the type of 'set' (line 581)
    set_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 14), 'set', False)
    # Calling set(args, kwargs) (line 581)
    set_call_result_1153 = invoke(stypy.reporting.localization.Localization(__file__, 581, 14), set_1145, *[replace_call_result_1151], **kwargs_1152)
    
    # Assigning a type to the variable 'indices' (line 581)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'indices', set_call_result_1153)
    
    # Assigning a Dict to a Name (line 584):
    
    # Assigning a Dict to a Name (line 584):
    
    # Obtaining an instance of the builtin type 'dict' (line 584)
    dict_1154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 21), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 584)
    
    # Assigning a type to the variable 'dimension_dict' (line 584)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 4), 'dimension_dict', dict_1154)
    
    
    # Call to enumerate(...): (line 585)
    # Processing the call arguments (line 585)
    # Getting the type of 'input_list' (line 585)
    input_list_1156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 32), 'input_list', False)
    # Processing the call keyword arguments (line 585)
    kwargs_1157 = {}
    # Getting the type of 'enumerate' (line 585)
    enumerate_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 22), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 585)
    enumerate_call_result_1158 = invoke(stypy.reporting.localization.Localization(__file__, 585, 22), enumerate_1155, *[input_list_1156], **kwargs_1157)
    
    # Testing the type of a for loop iterable (line 585)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 585, 4), enumerate_call_result_1158)
    # Getting the type of the for loop variable (line 585)
    for_loop_var_1159 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 585, 4), enumerate_call_result_1158)
    # Assigning a type to the variable 'tnum' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'tnum', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 4), for_loop_var_1159))
    # Assigning a type to the variable 'term' (line 585)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 4), 'term', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 4), for_loop_var_1159))
    # SSA begins for a for statement (line 585)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Attribute to a Name (line 586):
    
    # Assigning a Attribute to a Name (line 586):
    
    # Obtaining the type of the subscript
    # Getting the type of 'tnum' (line 586)
    tnum_1160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 22), 'tnum')
    # Getting the type of 'operands' (line 586)
    operands_1161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 13), 'operands')
    # Obtaining the member '__getitem__' of a type (line 586)
    getitem___1162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 13), operands_1161, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 586)
    subscript_call_result_1163 = invoke(stypy.reporting.localization.Localization(__file__, 586, 13), getitem___1162, tnum_1160)
    
    # Obtaining the member 'shape' of a type (line 586)
    shape_1164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 13), subscript_call_result_1163, 'shape')
    # Assigning a type to the variable 'sh' (line 586)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 8), 'sh', shape_1164)
    
    
    
    # Call to len(...): (line 587)
    # Processing the call arguments (line 587)
    # Getting the type of 'sh' (line 587)
    sh_1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 15), 'sh', False)
    # Processing the call keyword arguments (line 587)
    kwargs_1167 = {}
    # Getting the type of 'len' (line 587)
    len_1165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 11), 'len', False)
    # Calling len(args, kwargs) (line 587)
    len_call_result_1168 = invoke(stypy.reporting.localization.Localization(__file__, 587, 11), len_1165, *[sh_1166], **kwargs_1167)
    
    
    # Call to len(...): (line 587)
    # Processing the call arguments (line 587)
    # Getting the type of 'term' (line 587)
    term_1170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 26), 'term', False)
    # Processing the call keyword arguments (line 587)
    kwargs_1171 = {}
    # Getting the type of 'len' (line 587)
    len_1169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 22), 'len', False)
    # Calling len(args, kwargs) (line 587)
    len_call_result_1172 = invoke(stypy.reporting.localization.Localization(__file__, 587, 22), len_1169, *[term_1170], **kwargs_1171)
    
    # Applying the binary operator '!=' (line 587)
    result_ne_1173 = python_operator(stypy.reporting.localization.Localization(__file__, 587, 11), '!=', len_call_result_1168, len_call_result_1172)
    
    # Testing the type of an if condition (line 587)
    if_condition_1174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 587, 8), result_ne_1173)
    # Assigning a type to the variable 'if_condition_1174' (line 587)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'if_condition_1174', if_condition_1174)
    # SSA begins for if statement (line 587)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 588)
    # Processing the call arguments (line 588)
    str_1176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 29), 'str', 'Einstein sum subscript %s does not contain the correct number of indices for operand %d.')
    
    # Obtaining the type of the subscript
    # Getting the type of 'tnum' (line 590)
    tnum_1177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 46), 'tnum', False)
    # Getting the type of 'input_subscripts' (line 590)
    input_subscripts_1178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 29), 'input_subscripts', False)
    # Obtaining the member '__getitem__' of a type (line 590)
    getitem___1179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 29), input_subscripts_1178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 590)
    subscript_call_result_1180 = invoke(stypy.reporting.localization.Localization(__file__, 590, 29), getitem___1179, tnum_1177)
    
    # Getting the type of 'tnum' (line 590)
    tnum_1181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 53), 'tnum', False)
    # Processing the call keyword arguments (line 588)
    kwargs_1182 = {}
    # Getting the type of 'ValueError' (line 588)
    ValueError_1175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 18), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 588)
    ValueError_call_result_1183 = invoke(stypy.reporting.localization.Localization(__file__, 588, 18), ValueError_1175, *[str_1176, subscript_call_result_1180, tnum_1181], **kwargs_1182)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 588, 12), ValueError_call_result_1183, 'raise parameter', BaseException)
    # SSA join for if statement (line 587)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to enumerate(...): (line 591)
    # Processing the call arguments (line 591)
    # Getting the type of 'term' (line 591)
    term_1185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 36), 'term', False)
    # Processing the call keyword arguments (line 591)
    kwargs_1186 = {}
    # Getting the type of 'enumerate' (line 591)
    enumerate_1184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 26), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 591)
    enumerate_call_result_1187 = invoke(stypy.reporting.localization.Localization(__file__, 591, 26), enumerate_1184, *[term_1185], **kwargs_1186)
    
    # Testing the type of a for loop iterable (line 591)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 591, 8), enumerate_call_result_1187)
    # Getting the type of the for loop variable (line 591)
    for_loop_var_1188 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 591, 8), enumerate_call_result_1187)
    # Assigning a type to the variable 'cnum' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'cnum', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 8), for_loop_var_1188))
    # Assigning a type to the variable 'char' (line 591)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 8), 'char', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 591, 8), for_loop_var_1188))
    # SSA begins for a for statement (line 591)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Subscript to a Name (line 592):
    
    # Assigning a Subscript to a Name (line 592):
    
    # Obtaining the type of the subscript
    # Getting the type of 'cnum' (line 592)
    cnum_1189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 21), 'cnum')
    # Getting the type of 'sh' (line 592)
    sh_1190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 18), 'sh')
    # Obtaining the member '__getitem__' of a type (line 592)
    getitem___1191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 18), sh_1190, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 592)
    subscript_call_result_1192 = invoke(stypy.reporting.localization.Localization(__file__, 592, 18), getitem___1191, cnum_1189)
    
    # Assigning a type to the variable 'dim' (line 592)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 12), 'dim', subscript_call_result_1192)
    
    
    # Getting the type of 'char' (line 593)
    char_1193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 15), 'char')
    
    # Call to keys(...): (line 593)
    # Processing the call keyword arguments (line 593)
    kwargs_1196 = {}
    # Getting the type of 'dimension_dict' (line 593)
    dimension_dict_1194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 23), 'dimension_dict', False)
    # Obtaining the member 'keys' of a type (line 593)
    keys_1195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 23), dimension_dict_1194, 'keys')
    # Calling keys(args, kwargs) (line 593)
    keys_call_result_1197 = invoke(stypy.reporting.localization.Localization(__file__, 593, 23), keys_1195, *[], **kwargs_1196)
    
    # Applying the binary operator 'in' (line 593)
    result_contains_1198 = python_operator(stypy.reporting.localization.Localization(__file__, 593, 15), 'in', char_1193, keys_call_result_1197)
    
    # Testing the type of an if condition (line 593)
    if_condition_1199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 593, 12), result_contains_1198)
    # Assigning a type to the variable 'if_condition_1199' (line 593)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 593, 12), 'if_condition_1199', if_condition_1199)
    # SSA begins for if statement (line 593)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'char' (line 594)
    char_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 34), 'char')
    # Getting the type of 'dimension_dict' (line 594)
    dimension_dict_1201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 19), 'dimension_dict')
    # Obtaining the member '__getitem__' of a type (line 594)
    getitem___1202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 594, 19), dimension_dict_1201, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 594)
    subscript_call_result_1203 = invoke(stypy.reporting.localization.Localization(__file__, 594, 19), getitem___1202, char_1200)
    
    # Getting the type of 'dim' (line 594)
    dim_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 594, 43), 'dim')
    # Applying the binary operator '!=' (line 594)
    result_ne_1205 = python_operator(stypy.reporting.localization.Localization(__file__, 594, 19), '!=', subscript_call_result_1203, dim_1204)
    
    # Testing the type of an if condition (line 594)
    if_condition_1206 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 594, 16), result_ne_1205)
    # Assigning a type to the variable 'if_condition_1206' (line 594)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 594, 16), 'if_condition_1206', if_condition_1206)
    # SSA begins for if statement (line 594)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 595)
    # Processing the call arguments (line 595)
    str_1208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 595, 37), 'str', "Size of label '%s' for operand %d does not match previous terms.")
    # Getting the type of 'char' (line 596)
    char_1209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 66), 'char', False)
    # Getting the type of 'tnum' (line 596)
    tnum_1210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 72), 'tnum', False)
    # Processing the call keyword arguments (line 595)
    kwargs_1211 = {}
    # Getting the type of 'ValueError' (line 595)
    ValueError_1207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 26), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 595)
    ValueError_call_result_1212 = invoke(stypy.reporting.localization.Localization(__file__, 595, 26), ValueError_1207, *[str_1208, char_1209, tnum_1210], **kwargs_1211)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 595, 20), ValueError_call_result_1212, 'raise parameter', BaseException)
    # SSA join for if statement (line 594)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 593)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Subscript (line 598):
    
    # Assigning a Name to a Subscript (line 598):
    # Getting the type of 'dim' (line 598)
    dim_1213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 39), 'dim')
    # Getting the type of 'dimension_dict' (line 598)
    dimension_dict_1214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), 'dimension_dict')
    # Getting the type of 'char' (line 598)
    char_1215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 31), 'char')
    # Storing an element on a container (line 598)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 16), dimension_dict_1214, (char_1215, dim_1213))
    # SSA join for if statement (line 593)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 601):
    
    # Assigning a List to a Name (line 601):
    
    # Obtaining an instance of the builtin type 'list' (line 601)
    list_1216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 601)
    
    # Assigning a type to the variable 'size_list' (line 601)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 4), 'size_list', list_1216)
    
    # Getting the type of 'input_list' (line 602)
    input_list_1217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 16), 'input_list')
    
    # Obtaining an instance of the builtin type 'list' (line 602)
    list_1218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 602, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 602)
    # Adding element type (line 602)
    # Getting the type of 'output_subscript' (line 602)
    output_subscript_1219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 30), 'output_subscript')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 602, 29), list_1218, output_subscript_1219)
    
    # Applying the binary operator '+' (line 602)
    result_add_1220 = python_operator(stypy.reporting.localization.Localization(__file__, 602, 16), '+', input_list_1217, list_1218)
    
    # Testing the type of a for loop iterable (line 602)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 602, 4), result_add_1220)
    # Getting the type of the for loop variable (line 602)
    for_loop_var_1221 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 602, 4), result_add_1220)
    # Assigning a type to the variable 'term' (line 602)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 602, 4), 'term', for_loop_var_1221)
    # SSA begins for a for statement (line 602)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 603)
    # Processing the call arguments (line 603)
    
    # Call to _compute_size_by_dict(...): (line 603)
    # Processing the call arguments (line 603)
    # Getting the type of 'term' (line 603)
    term_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 47), 'term', False)
    # Getting the type of 'dimension_dict' (line 603)
    dimension_dict_1226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 53), 'dimension_dict', False)
    # Processing the call keyword arguments (line 603)
    kwargs_1227 = {}
    # Getting the type of '_compute_size_by_dict' (line 603)
    _compute_size_by_dict_1224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 25), '_compute_size_by_dict', False)
    # Calling _compute_size_by_dict(args, kwargs) (line 603)
    _compute_size_by_dict_call_result_1228 = invoke(stypy.reporting.localization.Localization(__file__, 603, 25), _compute_size_by_dict_1224, *[term_1225, dimension_dict_1226], **kwargs_1227)
    
    # Processing the call keyword arguments (line 603)
    kwargs_1229 = {}
    # Getting the type of 'size_list' (line 603)
    size_list_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 8), 'size_list', False)
    # Obtaining the member 'append' of a type (line 603)
    append_1223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 8), size_list_1222, 'append')
    # Calling append(args, kwargs) (line 603)
    append_call_result_1230 = invoke(stypy.reporting.localization.Localization(__file__, 603, 8), append_1223, *[_compute_size_by_dict_call_result_1228], **kwargs_1229)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 604):
    
    # Assigning a Call to a Name (line 604):
    
    # Call to max(...): (line 604)
    # Processing the call arguments (line 604)
    # Getting the type of 'size_list' (line 604)
    size_list_1232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 19), 'size_list', False)
    # Processing the call keyword arguments (line 604)
    kwargs_1233 = {}
    # Getting the type of 'max' (line 604)
    max_1231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 15), 'max', False)
    # Calling max(args, kwargs) (line 604)
    max_call_result_1234 = invoke(stypy.reporting.localization.Localization(__file__, 604, 15), max_1231, *[size_list_1232], **kwargs_1233)
    
    # Assigning a type to the variable 'max_size' (line 604)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 604, 4), 'max_size', max_call_result_1234)
    
    # Type idiom detected: calculating its left and rigth part (line 606)
    # Getting the type of 'memory_limit' (line 606)
    memory_limit_1235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 7), 'memory_limit')
    # Getting the type of 'None' (line 606)
    None_1236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 23), 'None')
    
    (may_be_1237, more_types_in_union_1238) = may_be_none(memory_limit_1235, None_1236)

    if may_be_1237:

        if more_types_in_union_1238:
            # Runtime conditional SSA (line 606)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 607):
        
        # Assigning a Name to a Name (line 607):
        # Getting the type of 'max_size' (line 607)
        max_size_1239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 21), 'max_size')
        # Assigning a type to the variable 'memory_arg' (line 607)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 8), 'memory_arg', max_size_1239)

        if more_types_in_union_1238:
            # Runtime conditional SSA for else branch (line 606)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_1237) or more_types_in_union_1238):
        
        # Assigning a Name to a Name (line 609):
        
        # Assigning a Name to a Name (line 609):
        # Getting the type of 'memory_limit' (line 609)
        memory_limit_1240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 21), 'memory_limit')
        # Assigning a type to the variable 'memory_arg' (line 609)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 609, 8), 'memory_arg', memory_limit_1240)

        if (may_be_1237 and more_types_in_union_1238):
            # SSA join for if statement (line 606)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 613):
    
    # Assigning a Call to a Name (line 613):
    
    # Call to _compute_size_by_dict(...): (line 613)
    # Processing the call arguments (line 613)
    # Getting the type of 'indices' (line 613)
    indices_1242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 39), 'indices', False)
    # Getting the type of 'dimension_dict' (line 613)
    dimension_dict_1243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 48), 'dimension_dict', False)
    # Processing the call keyword arguments (line 613)
    kwargs_1244 = {}
    # Getting the type of '_compute_size_by_dict' (line 613)
    _compute_size_by_dict_1241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 17), '_compute_size_by_dict', False)
    # Calling _compute_size_by_dict(args, kwargs) (line 613)
    _compute_size_by_dict_call_result_1245 = invoke(stypy.reporting.localization.Localization(__file__, 613, 17), _compute_size_by_dict_1241, *[indices_1242, dimension_dict_1243], **kwargs_1244)
    
    # Assigning a type to the variable 'naive_cost' (line 613)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'naive_cost', _compute_size_by_dict_call_result_1245)
    
    # Assigning a Call to a Name (line 614):
    
    # Assigning a Call to a Name (line 614):
    
    # Call to replace(...): (line 614)
    # Processing the call arguments (line 614)
    str_1248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 48), 'str', ',')
    str_1249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 53), 'str', '')
    # Processing the call keyword arguments (line 614)
    kwargs_1250 = {}
    # Getting the type of 'input_subscripts' (line 614)
    input_subscripts_1246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 23), 'input_subscripts', False)
    # Obtaining the member 'replace' of a type (line 614)
    replace_1247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 23), input_subscripts_1246, 'replace')
    # Calling replace(args, kwargs) (line 614)
    replace_call_result_1251 = invoke(stypy.reporting.localization.Localization(__file__, 614, 23), replace_1247, *[str_1248, str_1249], **kwargs_1250)
    
    # Assigning a type to the variable 'indices_in_input' (line 614)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 614, 4), 'indices_in_input', replace_call_result_1251)
    
    # Assigning a Call to a Name (line 615):
    
    # Assigning a Call to a Name (line 615):
    
    # Call to max(...): (line 615)
    # Processing the call arguments (line 615)
    
    # Call to len(...): (line 615)
    # Processing the call arguments (line 615)
    # Getting the type of 'input_list' (line 615)
    input_list_1254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 19), 'input_list', False)
    # Processing the call keyword arguments (line 615)
    kwargs_1255 = {}
    # Getting the type of 'len' (line 615)
    len_1253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 15), 'len', False)
    # Calling len(args, kwargs) (line 615)
    len_call_result_1256 = invoke(stypy.reporting.localization.Localization(__file__, 615, 15), len_1253, *[input_list_1254], **kwargs_1255)
    
    int_1257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 33), 'int')
    # Applying the binary operator '-' (line 615)
    result_sub_1258 = python_operator(stypy.reporting.localization.Localization(__file__, 615, 15), '-', len_call_result_1256, int_1257)
    
    int_1259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 36), 'int')
    # Processing the call keyword arguments (line 615)
    kwargs_1260 = {}
    # Getting the type of 'max' (line 615)
    max_1252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 11), 'max', False)
    # Calling max(args, kwargs) (line 615)
    max_call_result_1261 = invoke(stypy.reporting.localization.Localization(__file__, 615, 11), max_1252, *[result_sub_1258, int_1259], **kwargs_1260)
    
    # Assigning a type to the variable 'mult' (line 615)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 615, 4), 'mult', max_call_result_1261)
    
    
    # Call to len(...): (line 616)
    # Processing the call arguments (line 616)
    # Getting the type of 'indices_in_input' (line 616)
    indices_in_input_1263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'indices_in_input', False)
    # Processing the call keyword arguments (line 616)
    kwargs_1264 = {}
    # Getting the type of 'len' (line 616)
    len_1262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 8), 'len', False)
    # Calling len(args, kwargs) (line 616)
    len_call_result_1265 = invoke(stypy.reporting.localization.Localization(__file__, 616, 8), len_1262, *[indices_in_input_1263], **kwargs_1264)
    
    
    # Call to len(...): (line 616)
    # Processing the call arguments (line 616)
    
    # Call to set(...): (line 616)
    # Processing the call arguments (line 616)
    # Getting the type of 'indices_in_input' (line 616)
    indices_in_input_1268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 40), 'indices_in_input', False)
    # Processing the call keyword arguments (line 616)
    kwargs_1269 = {}
    # Getting the type of 'set' (line 616)
    set_1267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 36), 'set', False)
    # Calling set(args, kwargs) (line 616)
    set_call_result_1270 = invoke(stypy.reporting.localization.Localization(__file__, 616, 36), set_1267, *[indices_in_input_1268], **kwargs_1269)
    
    # Processing the call keyword arguments (line 616)
    kwargs_1271 = {}
    # Getting the type of 'len' (line 616)
    len_1266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 32), 'len', False)
    # Calling len(args, kwargs) (line 616)
    len_call_result_1272 = invoke(stypy.reporting.localization.Localization(__file__, 616, 32), len_1266, *[set_call_result_1270], **kwargs_1271)
    
    # Applying the binary operator '-' (line 616)
    result_sub_1273 = python_operator(stypy.reporting.localization.Localization(__file__, 616, 8), '-', len_call_result_1265, len_call_result_1272)
    
    # Testing the type of an if condition (line 616)
    if_condition_1274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 616, 4), result_sub_1273)
    # Assigning a type to the variable 'if_condition_1274' (line 616)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 616, 4), 'if_condition_1274', if_condition_1274)
    # SSA begins for if statement (line 616)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'mult' (line 617)
    mult_1275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'mult')
    int_1276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 617, 16), 'int')
    # Applying the binary operator '*=' (line 617)
    result_imul_1277 = python_operator(stypy.reporting.localization.Localization(__file__, 617, 8), '*=', mult_1275, int_1276)
    # Assigning a type to the variable 'mult' (line 617)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 617, 8), 'mult', result_imul_1277)
    
    # SSA join for if statement (line 616)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'naive_cost' (line 618)
    naive_cost_1278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'naive_cost')
    # Getting the type of 'mult' (line 618)
    mult_1279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 18), 'mult')
    # Applying the binary operator '*=' (line 618)
    result_imul_1280 = python_operator(stypy.reporting.localization.Localization(__file__, 618, 4), '*=', naive_cost_1278, mult_1279)
    # Assigning a type to the variable 'naive_cost' (line 618)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'naive_cost', result_imul_1280)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'path_type' (line 621)
    path_type_1281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'path_type')
    # Getting the type of 'False' (line 621)
    False_1282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 21), 'False')
    # Applying the binary operator 'is' (line 621)
    result_is__1283 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 8), 'is', path_type_1281, False_1282)
    
    
    
    # Call to len(...): (line 621)
    # Processing the call arguments (line 621)
    # Getting the type of 'input_list' (line 621)
    input_list_1285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 36), 'input_list', False)
    # Processing the call keyword arguments (line 621)
    kwargs_1286 = {}
    # Getting the type of 'len' (line 621)
    len_1284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 32), 'len', False)
    # Calling len(args, kwargs) (line 621)
    len_call_result_1287 = invoke(stypy.reporting.localization.Localization(__file__, 621, 32), len_1284, *[input_list_1285], **kwargs_1286)
    
    
    # Obtaining an instance of the builtin type 'list' (line 621)
    list_1288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 621)
    # Adding element type (line 621)
    int_1289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 51), list_1288, int_1289)
    # Adding element type (line 621)
    int_1290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 55), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 621, 51), list_1288, int_1290)
    
    # Applying the binary operator 'in' (line 621)
    result_contains_1291 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 32), 'in', len_call_result_1287, list_1288)
    
    # Applying the binary operator 'or' (line 621)
    result_or_keyword_1292 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 7), 'or', result_is__1283, result_contains_1291)
    
    # Getting the type of 'indices' (line 621)
    indices_1293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 63), 'indices')
    # Getting the type of 'output_set' (line 621)
    output_set_1294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 74), 'output_set')
    # Applying the binary operator '==' (line 621)
    result_eq_1295 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 63), '==', indices_1293, output_set_1294)
    
    # Applying the binary operator 'or' (line 621)
    result_or_keyword_1296 = python_operator(stypy.reporting.localization.Localization(__file__, 621, 7), 'or', result_or_keyword_1292, result_eq_1295)
    
    # Testing the type of an if condition (line 621)
    if_condition_1297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 621, 4), result_or_keyword_1296)
    # Assigning a type to the variable 'if_condition_1297' (line 621)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 4), 'if_condition_1297', if_condition_1297)
    # SSA begins for if statement (line 621)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 623):
    
    # Assigning a List to a Name (line 623):
    
    # Obtaining an instance of the builtin type 'list' (line 623)
    list_1298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 623)
    # Adding element type (line 623)
    
    # Call to tuple(...): (line 623)
    # Processing the call arguments (line 623)
    
    # Call to range(...): (line 623)
    # Processing the call arguments (line 623)
    
    # Call to len(...): (line 623)
    # Processing the call arguments (line 623)
    # Getting the type of 'input_list' (line 623)
    input_list_1302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 32), 'input_list', False)
    # Processing the call keyword arguments (line 623)
    kwargs_1303 = {}
    # Getting the type of 'len' (line 623)
    len_1301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 28), 'len', False)
    # Calling len(args, kwargs) (line 623)
    len_call_result_1304 = invoke(stypy.reporting.localization.Localization(__file__, 623, 28), len_1301, *[input_list_1302], **kwargs_1303)
    
    # Processing the call keyword arguments (line 623)
    kwargs_1305 = {}
    # Getting the type of 'range' (line 623)
    range_1300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 22), 'range', False)
    # Calling range(args, kwargs) (line 623)
    range_call_result_1306 = invoke(stypy.reporting.localization.Localization(__file__, 623, 22), range_1300, *[len_call_result_1304], **kwargs_1305)
    
    # Processing the call keyword arguments (line 623)
    kwargs_1307 = {}
    # Getting the type of 'tuple' (line 623)
    tuple_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'tuple', False)
    # Calling tuple(args, kwargs) (line 623)
    tuple_call_result_1308 = invoke(stypy.reporting.localization.Localization(__file__, 623, 16), tuple_1299, *[range_call_result_1306], **kwargs_1307)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 623, 15), list_1298, tuple_call_result_1308)
    
    # Assigning a type to the variable 'path' (line 623)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'path', list_1298)
    # SSA branch for the else part of an if statement (line 621)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'path_type' (line 624)
    path_type_1309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 9), 'path_type')
    str_1310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 22), 'str', 'greedy')
    # Applying the binary operator '==' (line 624)
    result_eq_1311 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 9), '==', path_type_1309, str_1310)
    
    # Testing the type of an if condition (line 624)
    if_condition_1312 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 624, 9), result_eq_1311)
    # Assigning a type to the variable 'if_condition_1312' (line 624)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 624, 9), 'if_condition_1312', if_condition_1312)
    # SSA begins for if statement (line 624)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 626):
    
    # Assigning a Call to a Name (line 626):
    
    # Call to min(...): (line 626)
    # Processing the call arguments (line 626)
    # Getting the type of 'memory_arg' (line 626)
    memory_arg_1314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 25), 'memory_arg', False)
    # Getting the type of 'max_size' (line 626)
    max_size_1315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 37), 'max_size', False)
    # Processing the call keyword arguments (line 626)
    kwargs_1316 = {}
    # Getting the type of 'min' (line 626)
    min_1313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 21), 'min', False)
    # Calling min(args, kwargs) (line 626)
    min_call_result_1317 = invoke(stypy.reporting.localization.Localization(__file__, 626, 21), min_1313, *[memory_arg_1314, max_size_1315], **kwargs_1316)
    
    # Assigning a type to the variable 'memory_arg' (line 626)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'memory_arg', min_call_result_1317)
    
    # Assigning a Call to a Name (line 627):
    
    # Assigning a Call to a Name (line 627):
    
    # Call to _greedy_path(...): (line 627)
    # Processing the call arguments (line 627)
    # Getting the type of 'input_sets' (line 627)
    input_sets_1319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 28), 'input_sets', False)
    # Getting the type of 'output_set' (line 627)
    output_set_1320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 40), 'output_set', False)
    # Getting the type of 'dimension_dict' (line 627)
    dimension_dict_1321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 52), 'dimension_dict', False)
    # Getting the type of 'memory_arg' (line 627)
    memory_arg_1322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 68), 'memory_arg', False)
    # Processing the call keyword arguments (line 627)
    kwargs_1323 = {}
    # Getting the type of '_greedy_path' (line 627)
    _greedy_path_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 15), '_greedy_path', False)
    # Calling _greedy_path(args, kwargs) (line 627)
    _greedy_path_call_result_1324 = invoke(stypy.reporting.localization.Localization(__file__, 627, 15), _greedy_path_1318, *[input_sets_1319, output_set_1320, dimension_dict_1321, memory_arg_1322], **kwargs_1323)
    
    # Assigning a type to the variable 'path' (line 627)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'path', _greedy_path_call_result_1324)
    # SSA branch for the else part of an if statement (line 624)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'path_type' (line 628)
    path_type_1325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 9), 'path_type')
    str_1326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 22), 'str', 'optimal')
    # Applying the binary operator '==' (line 628)
    result_eq_1327 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 9), '==', path_type_1325, str_1326)
    
    # Testing the type of an if condition (line 628)
    if_condition_1328 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 628, 9), result_eq_1327)
    # Assigning a type to the variable 'if_condition_1328' (line 628)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 628, 9), 'if_condition_1328', if_condition_1328)
    # SSA begins for if statement (line 628)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 629):
    
    # Assigning a Call to a Name (line 629):
    
    # Call to _optimal_path(...): (line 629)
    # Processing the call arguments (line 629)
    # Getting the type of 'input_sets' (line 629)
    input_sets_1330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 29), 'input_sets', False)
    # Getting the type of 'output_set' (line 629)
    output_set_1331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 41), 'output_set', False)
    # Getting the type of 'dimension_dict' (line 629)
    dimension_dict_1332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 53), 'dimension_dict', False)
    # Getting the type of 'memory_arg' (line 629)
    memory_arg_1333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 69), 'memory_arg', False)
    # Processing the call keyword arguments (line 629)
    kwargs_1334 = {}
    # Getting the type of '_optimal_path' (line 629)
    _optimal_path_1329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 15), '_optimal_path', False)
    # Calling _optimal_path(args, kwargs) (line 629)
    _optimal_path_call_result_1335 = invoke(stypy.reporting.localization.Localization(__file__, 629, 15), _optimal_path_1329, *[input_sets_1330, output_set_1331, dimension_dict_1332, memory_arg_1333], **kwargs_1334)
    
    # Assigning a type to the variable 'path' (line 629)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'path', _optimal_path_call_result_1335)
    # SSA branch for the else part of an if statement (line 628)
    module_type_store.open_ssa_branch('else')
    
    
    
    # Obtaining the type of the subscript
    int_1336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 19), 'int')
    # Getting the type of 'path_type' (line 630)
    path_type_1337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 630, 9), 'path_type')
    # Obtaining the member '__getitem__' of a type (line 630)
    getitem___1338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 630, 9), path_type_1337, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 630)
    subscript_call_result_1339 = invoke(stypy.reporting.localization.Localization(__file__, 630, 9), getitem___1338, int_1336)
    
    str_1340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 630, 25), 'str', 'einsum_path')
    # Applying the binary operator '==' (line 630)
    result_eq_1341 = python_operator(stypy.reporting.localization.Localization(__file__, 630, 9), '==', subscript_call_result_1339, str_1340)
    
    # Testing the type of an if condition (line 630)
    if_condition_1342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 630, 9), result_eq_1341)
    # Assigning a type to the variable 'if_condition_1342' (line 630)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 630, 9), 'if_condition_1342', if_condition_1342)
    # SSA begins for if statement (line 630)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 631):
    
    # Assigning a Subscript to a Name (line 631):
    
    # Obtaining the type of the subscript
    int_1343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 631, 25), 'int')
    slice_1344 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 631, 15), int_1343, None, None)
    # Getting the type of 'path_type' (line 631)
    path_type_1345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 15), 'path_type')
    # Obtaining the member '__getitem__' of a type (line 631)
    getitem___1346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 631, 15), path_type_1345, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 631)
    subscript_call_result_1347 = invoke(stypy.reporting.localization.Localization(__file__, 631, 15), getitem___1346, slice_1344)
    
    # Assigning a type to the variable 'path' (line 631)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 631, 8), 'path', subscript_call_result_1347)
    # SSA branch for the else part of an if statement (line 630)
    module_type_store.open_ssa_branch('else')
    
    # Call to KeyError(...): (line 633)
    # Processing the call arguments (line 633)
    str_1349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 23), 'str', 'Path name %s not found')
    # Getting the type of 'path_type' (line 633)
    path_type_1350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 49), 'path_type', False)
    # Processing the call keyword arguments (line 633)
    kwargs_1351 = {}
    # Getting the type of 'KeyError' (line 633)
    KeyError_1348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 14), 'KeyError', False)
    # Calling KeyError(args, kwargs) (line 633)
    KeyError_call_result_1352 = invoke(stypy.reporting.localization.Localization(__file__, 633, 14), KeyError_1348, *[str_1349, path_type_1350], **kwargs_1351)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 633, 8), KeyError_call_result_1352, 'raise parameter', BaseException)
    # SSA join for if statement (line 630)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 628)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 624)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 621)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Tuple to a Tuple (line 635):
    
    # Assigning a List to a Name (line 635):
    
    # Obtaining an instance of the builtin type 'list' (line 635)
    list_1353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 635)
    
    # Assigning a type to the variable 'tuple_assignment_19' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_assignment_19', list_1353)
    
    # Assigning a List to a Name (line 635):
    
    # Obtaining an instance of the builtin type 'list' (line 635)
    list_1354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 61), 'list')
    # Adding type elements to the builtin type 'list' instance (line 635)
    
    # Assigning a type to the variable 'tuple_assignment_20' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_assignment_20', list_1354)
    
    # Assigning a List to a Name (line 635):
    
    # Obtaining an instance of the builtin type 'list' (line 635)
    list_1355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 65), 'list')
    # Adding type elements to the builtin type 'list' instance (line 635)
    
    # Assigning a type to the variable 'tuple_assignment_21' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_assignment_21', list_1355)
    
    # Assigning a List to a Name (line 635):
    
    # Obtaining an instance of the builtin type 'list' (line 635)
    list_1356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 69), 'list')
    # Adding type elements to the builtin type 'list' instance (line 635)
    
    # Assigning a type to the variable 'tuple_assignment_22' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_assignment_22', list_1356)
    
    # Assigning a Name to a Name (line 635):
    # Getting the type of 'tuple_assignment_19' (line 635)
    tuple_assignment_19_1357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_assignment_19')
    # Assigning a type to the variable 'cost_list' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'cost_list', tuple_assignment_19_1357)
    
    # Assigning a Name to a Name (line 635):
    # Getting the type of 'tuple_assignment_20' (line 635)
    tuple_assignment_20_1358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_assignment_20')
    # Assigning a type to the variable 'scale_list' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 15), 'scale_list', tuple_assignment_20_1358)
    
    # Assigning a Name to a Name (line 635):
    # Getting the type of 'tuple_assignment_21' (line 635)
    tuple_assignment_21_1359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_assignment_21')
    # Assigning a type to the variable 'size_list' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 27), 'size_list', tuple_assignment_21_1359)
    
    # Assigning a Name to a Name (line 635):
    # Getting the type of 'tuple_assignment_22' (line 635)
    tuple_assignment_22_1360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 4), 'tuple_assignment_22')
    # Assigning a type to the variable 'contraction_list' (line 635)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 635, 38), 'contraction_list', tuple_assignment_22_1360)
    
    
    # Call to enumerate(...): (line 638)
    # Processing the call arguments (line 638)
    # Getting the type of 'path' (line 638)
    path_1362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 41), 'path', False)
    # Processing the call keyword arguments (line 638)
    kwargs_1363 = {}
    # Getting the type of 'enumerate' (line 638)
    enumerate_1361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 638, 31), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 638)
    enumerate_call_result_1364 = invoke(stypy.reporting.localization.Localization(__file__, 638, 31), enumerate_1361, *[path_1362], **kwargs_1363)
    
    # Testing the type of a for loop iterable (line 638)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 638, 4), enumerate_call_result_1364)
    # Getting the type of the for loop variable (line 638)
    for_loop_var_1365 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 638, 4), enumerate_call_result_1364)
    # Assigning a type to the variable 'cnum' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'cnum', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 4), for_loop_var_1365))
    # Assigning a type to the variable 'contract_inds' (line 638)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 638, 4), 'contract_inds', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 638, 4), for_loop_var_1365))
    # SSA begins for a for statement (line 638)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 640):
    
    # Assigning a Call to a Name (line 640):
    
    # Call to tuple(...): (line 640)
    # Processing the call arguments (line 640)
    
    # Call to sorted(...): (line 640)
    # Processing the call arguments (line 640)
    
    # Call to list(...): (line 640)
    # Processing the call arguments (line 640)
    # Getting the type of 'contract_inds' (line 640)
    contract_inds_1369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 42), 'contract_inds', False)
    # Processing the call keyword arguments (line 640)
    kwargs_1370 = {}
    # Getting the type of 'list' (line 640)
    list_1368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 37), 'list', False)
    # Calling list(args, kwargs) (line 640)
    list_call_result_1371 = invoke(stypy.reporting.localization.Localization(__file__, 640, 37), list_1368, *[contract_inds_1369], **kwargs_1370)
    
    # Processing the call keyword arguments (line 640)
    # Getting the type of 'True' (line 640)
    True_1372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 66), 'True', False)
    keyword_1373 = True_1372
    kwargs_1374 = {'reverse': keyword_1373}
    # Getting the type of 'sorted' (line 640)
    sorted_1367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 30), 'sorted', False)
    # Calling sorted(args, kwargs) (line 640)
    sorted_call_result_1375 = invoke(stypy.reporting.localization.Localization(__file__, 640, 30), sorted_1367, *[list_call_result_1371], **kwargs_1374)
    
    # Processing the call keyword arguments (line 640)
    kwargs_1376 = {}
    # Getting the type of 'tuple' (line 640)
    tuple_1366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 24), 'tuple', False)
    # Calling tuple(args, kwargs) (line 640)
    tuple_call_result_1377 = invoke(stypy.reporting.localization.Localization(__file__, 640, 24), tuple_1366, *[sorted_call_result_1375], **kwargs_1376)
    
    # Assigning a type to the variable 'contract_inds' (line 640)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'contract_inds', tuple_call_result_1377)
    
    # Assigning a Call to a Name (line 642):
    
    # Assigning a Call to a Name (line 642):
    
    # Call to _find_contraction(...): (line 642)
    # Processing the call arguments (line 642)
    # Getting the type of 'contract_inds' (line 642)
    contract_inds_1379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 37), 'contract_inds', False)
    # Getting the type of 'input_sets' (line 642)
    input_sets_1380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 52), 'input_sets', False)
    # Getting the type of 'output_set' (line 642)
    output_set_1381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 64), 'output_set', False)
    # Processing the call keyword arguments (line 642)
    kwargs_1382 = {}
    # Getting the type of '_find_contraction' (line 642)
    _find_contraction_1378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 19), '_find_contraction', False)
    # Calling _find_contraction(args, kwargs) (line 642)
    _find_contraction_call_result_1383 = invoke(stypy.reporting.localization.Localization(__file__, 642, 19), _find_contraction_1378, *[contract_inds_1379, input_sets_1380, output_set_1381], **kwargs_1382)
    
    # Assigning a type to the variable 'contract' (line 642)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 642, 8), 'contract', _find_contraction_call_result_1383)
    
    # Assigning a Name to a Tuple (line 643):
    
    # Assigning a Subscript to a Name (line 643):
    
    # Obtaining the type of the subscript
    int_1384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 8), 'int')
    # Getting the type of 'contract' (line 643)
    contract_1385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 58), 'contract')
    # Obtaining the member '__getitem__' of a type (line 643)
    getitem___1386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 8), contract_1385, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 643)
    subscript_call_result_1387 = invoke(stypy.reporting.localization.Localization(__file__, 643, 8), getitem___1386, int_1384)
    
    # Assigning a type to the variable 'tuple_var_assignment_23' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'tuple_var_assignment_23', subscript_call_result_1387)
    
    # Assigning a Subscript to a Name (line 643):
    
    # Obtaining the type of the subscript
    int_1388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 8), 'int')
    # Getting the type of 'contract' (line 643)
    contract_1389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 58), 'contract')
    # Obtaining the member '__getitem__' of a type (line 643)
    getitem___1390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 8), contract_1389, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 643)
    subscript_call_result_1391 = invoke(stypy.reporting.localization.Localization(__file__, 643, 8), getitem___1390, int_1388)
    
    # Assigning a type to the variable 'tuple_var_assignment_24' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'tuple_var_assignment_24', subscript_call_result_1391)
    
    # Assigning a Subscript to a Name (line 643):
    
    # Obtaining the type of the subscript
    int_1392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 8), 'int')
    # Getting the type of 'contract' (line 643)
    contract_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 58), 'contract')
    # Obtaining the member '__getitem__' of a type (line 643)
    getitem___1394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 8), contract_1393, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 643)
    subscript_call_result_1395 = invoke(stypy.reporting.localization.Localization(__file__, 643, 8), getitem___1394, int_1392)
    
    # Assigning a type to the variable 'tuple_var_assignment_25' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'tuple_var_assignment_25', subscript_call_result_1395)
    
    # Assigning a Subscript to a Name (line 643):
    
    # Obtaining the type of the subscript
    int_1396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 643, 8), 'int')
    # Getting the type of 'contract' (line 643)
    contract_1397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 58), 'contract')
    # Obtaining the member '__getitem__' of a type (line 643)
    getitem___1398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 643, 8), contract_1397, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 643)
    subscript_call_result_1399 = invoke(stypy.reporting.localization.Localization(__file__, 643, 8), getitem___1398, int_1396)
    
    # Assigning a type to the variable 'tuple_var_assignment_26' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'tuple_var_assignment_26', subscript_call_result_1399)
    
    # Assigning a Name to a Name (line 643):
    # Getting the type of 'tuple_var_assignment_23' (line 643)
    tuple_var_assignment_23_1400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'tuple_var_assignment_23')
    # Assigning a type to the variable 'out_inds' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'out_inds', tuple_var_assignment_23_1400)
    
    # Assigning a Name to a Name (line 643):
    # Getting the type of 'tuple_var_assignment_24' (line 643)
    tuple_var_assignment_24_1401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'tuple_var_assignment_24')
    # Assigning a type to the variable 'input_sets' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 18), 'input_sets', tuple_var_assignment_24_1401)
    
    # Assigning a Name to a Name (line 643):
    # Getting the type of 'tuple_var_assignment_25' (line 643)
    tuple_var_assignment_25_1402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'tuple_var_assignment_25')
    # Assigning a type to the variable 'idx_removed' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 30), 'idx_removed', tuple_var_assignment_25_1402)
    
    # Assigning a Name to a Name (line 643):
    # Getting the type of 'tuple_var_assignment_26' (line 643)
    tuple_var_assignment_26_1403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 643, 8), 'tuple_var_assignment_26')
    # Assigning a type to the variable 'idx_contract' (line 643)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 643, 43), 'idx_contract', tuple_var_assignment_26_1403)
    
    # Assigning a Call to a Name (line 645):
    
    # Assigning a Call to a Name (line 645):
    
    # Call to _compute_size_by_dict(...): (line 645)
    # Processing the call arguments (line 645)
    # Getting the type of 'idx_contract' (line 645)
    idx_contract_1405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 37), 'idx_contract', False)
    # Getting the type of 'dimension_dict' (line 645)
    dimension_dict_1406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 51), 'dimension_dict', False)
    # Processing the call keyword arguments (line 645)
    kwargs_1407 = {}
    # Getting the type of '_compute_size_by_dict' (line 645)
    _compute_size_by_dict_1404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 645, 15), '_compute_size_by_dict', False)
    # Calling _compute_size_by_dict(args, kwargs) (line 645)
    _compute_size_by_dict_call_result_1408 = invoke(stypy.reporting.localization.Localization(__file__, 645, 15), _compute_size_by_dict_1404, *[idx_contract_1405, dimension_dict_1406], **kwargs_1407)
    
    # Assigning a type to the variable 'cost' (line 645)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 645, 8), 'cost', _compute_size_by_dict_call_result_1408)
    
    # Getting the type of 'idx_removed' (line 646)
    idx_removed_1409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 646, 11), 'idx_removed')
    # Testing the type of an if condition (line 646)
    if_condition_1410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 646, 8), idx_removed_1409)
    # Assigning a type to the variable 'if_condition_1410' (line 646)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 646, 8), 'if_condition_1410', if_condition_1410)
    # SSA begins for if statement (line 646)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'cost' (line 647)
    cost_1411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 647, 12), 'cost')
    int_1412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 647, 20), 'int')
    # Applying the binary operator '*=' (line 647)
    result_imul_1413 = python_operator(stypy.reporting.localization.Localization(__file__, 647, 12), '*=', cost_1411, int_1412)
    # Assigning a type to the variable 'cost' (line 647)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 647, 12), 'cost', result_imul_1413)
    
    # SSA join for if statement (line 646)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 648)
    # Processing the call arguments (line 648)
    # Getting the type of 'cost' (line 648)
    cost_1416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 25), 'cost', False)
    # Processing the call keyword arguments (line 648)
    kwargs_1417 = {}
    # Getting the type of 'cost_list' (line 648)
    cost_list_1414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 648, 8), 'cost_list', False)
    # Obtaining the member 'append' of a type (line 648)
    append_1415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 648, 8), cost_list_1414, 'append')
    # Calling append(args, kwargs) (line 648)
    append_call_result_1418 = invoke(stypy.reporting.localization.Localization(__file__, 648, 8), append_1415, *[cost_1416], **kwargs_1417)
    
    
    # Call to append(...): (line 649)
    # Processing the call arguments (line 649)
    
    # Call to len(...): (line 649)
    # Processing the call arguments (line 649)
    # Getting the type of 'idx_contract' (line 649)
    idx_contract_1422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 30), 'idx_contract', False)
    # Processing the call keyword arguments (line 649)
    kwargs_1423 = {}
    # Getting the type of 'len' (line 649)
    len_1421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 26), 'len', False)
    # Calling len(args, kwargs) (line 649)
    len_call_result_1424 = invoke(stypy.reporting.localization.Localization(__file__, 649, 26), len_1421, *[idx_contract_1422], **kwargs_1423)
    
    # Processing the call keyword arguments (line 649)
    kwargs_1425 = {}
    # Getting the type of 'scale_list' (line 649)
    scale_list_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 649, 8), 'scale_list', False)
    # Obtaining the member 'append' of a type (line 649)
    append_1420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 649, 8), scale_list_1419, 'append')
    # Calling append(args, kwargs) (line 649)
    append_call_result_1426 = invoke(stypy.reporting.localization.Localization(__file__, 649, 8), append_1420, *[len_call_result_1424], **kwargs_1425)
    
    
    # Call to append(...): (line 650)
    # Processing the call arguments (line 650)
    
    # Call to _compute_size_by_dict(...): (line 650)
    # Processing the call arguments (line 650)
    # Getting the type of 'out_inds' (line 650)
    out_inds_1430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 47), 'out_inds', False)
    # Getting the type of 'dimension_dict' (line 650)
    dimension_dict_1431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 57), 'dimension_dict', False)
    # Processing the call keyword arguments (line 650)
    kwargs_1432 = {}
    # Getting the type of '_compute_size_by_dict' (line 650)
    _compute_size_by_dict_1429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 25), '_compute_size_by_dict', False)
    # Calling _compute_size_by_dict(args, kwargs) (line 650)
    _compute_size_by_dict_call_result_1433 = invoke(stypy.reporting.localization.Localization(__file__, 650, 25), _compute_size_by_dict_1429, *[out_inds_1430, dimension_dict_1431], **kwargs_1432)
    
    # Processing the call keyword arguments (line 650)
    kwargs_1434 = {}
    # Getting the type of 'size_list' (line 650)
    size_list_1427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 650, 8), 'size_list', False)
    # Obtaining the member 'append' of a type (line 650)
    append_1428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 650, 8), size_list_1427, 'append')
    # Calling append(args, kwargs) (line 650)
    append_call_result_1435 = invoke(stypy.reporting.localization.Localization(__file__, 650, 8), append_1428, *[_compute_size_by_dict_call_result_1433], **kwargs_1434)
    
    
    # Assigning a List to a Name (line 652):
    
    # Assigning a List to a Name (line 652):
    
    # Obtaining an instance of the builtin type 'list' (line 652)
    list_1436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 652, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 652)
    
    # Assigning a type to the variable 'tmp_inputs' (line 652)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 652, 8), 'tmp_inputs', list_1436)
    
    # Getting the type of 'contract_inds' (line 653)
    contract_inds_1437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 653, 17), 'contract_inds')
    # Testing the type of a for loop iterable (line 653)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 653, 8), contract_inds_1437)
    # Getting the type of the for loop variable (line 653)
    for_loop_var_1438 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 653, 8), contract_inds_1437)
    # Assigning a type to the variable 'x' (line 653)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 653, 8), 'x', for_loop_var_1438)
    # SSA begins for a for statement (line 653)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 654)
    # Processing the call arguments (line 654)
    
    # Call to pop(...): (line 654)
    # Processing the call arguments (line 654)
    # Getting the type of 'x' (line 654)
    x_1443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 45), 'x', False)
    # Processing the call keyword arguments (line 654)
    kwargs_1444 = {}
    # Getting the type of 'input_list' (line 654)
    input_list_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 30), 'input_list', False)
    # Obtaining the member 'pop' of a type (line 654)
    pop_1442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 30), input_list_1441, 'pop')
    # Calling pop(args, kwargs) (line 654)
    pop_call_result_1445 = invoke(stypy.reporting.localization.Localization(__file__, 654, 30), pop_1442, *[x_1443], **kwargs_1444)
    
    # Processing the call keyword arguments (line 654)
    kwargs_1446 = {}
    # Getting the type of 'tmp_inputs' (line 654)
    tmp_inputs_1439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 654, 12), 'tmp_inputs', False)
    # Obtaining the member 'append' of a type (line 654)
    append_1440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 654, 12), tmp_inputs_1439, 'append')
    # Calling append(args, kwargs) (line 654)
    append_call_result_1447 = invoke(stypy.reporting.localization.Localization(__file__, 654, 12), append_1440, *[pop_call_result_1445], **kwargs_1446)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnum' (line 657)
    cnum_1448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 12), 'cnum')
    
    # Call to len(...): (line 657)
    # Processing the call arguments (line 657)
    # Getting the type of 'path' (line 657)
    path_1450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 23), 'path', False)
    # Processing the call keyword arguments (line 657)
    kwargs_1451 = {}
    # Getting the type of 'len' (line 657)
    len_1449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 657, 19), 'len', False)
    # Calling len(args, kwargs) (line 657)
    len_call_result_1452 = invoke(stypy.reporting.localization.Localization(__file__, 657, 19), len_1449, *[path_1450], **kwargs_1451)
    
    # Applying the binary operator '-' (line 657)
    result_sub_1453 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 12), '-', cnum_1448, len_call_result_1452)
    
    int_1454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 657, 33), 'int')
    # Applying the binary operator '==' (line 657)
    result_eq_1455 = python_operator(stypy.reporting.localization.Localization(__file__, 657, 11), '==', result_sub_1453, int_1454)
    
    # Testing the type of an if condition (line 657)
    if_condition_1456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 657, 8), result_eq_1455)
    # Assigning a type to the variable 'if_condition_1456' (line 657)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 657, 8), 'if_condition_1456', if_condition_1456)
    # SSA begins for if statement (line 657)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 658):
    
    # Assigning a Name to a Name (line 658):
    # Getting the type of 'output_subscript' (line 658)
    output_subscript_1457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 658, 25), 'output_subscript')
    # Assigning a type to the variable 'idx_result' (line 658)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 658, 12), 'idx_result', output_subscript_1457)
    # SSA branch for the else part of an if statement (line 657)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a ListComp to a Name (line 660):
    
    # Assigning a ListComp to a Name (line 660):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'out_inds' (line 660)
    out_inds_1464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 65), 'out_inds')
    comprehension_1465 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 660, 27), out_inds_1464)
    # Assigning a type to the variable 'ind' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 27), 'ind', comprehension_1465)
    
    # Obtaining an instance of the builtin type 'tuple' (line 660)
    tuple_1458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 660)
    # Adding element type (line 660)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ind' (line 660)
    ind_1459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 43), 'ind')
    # Getting the type of 'dimension_dict' (line 660)
    dimension_dict_1460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 28), 'dimension_dict')
    # Obtaining the member '__getitem__' of a type (line 660)
    getitem___1461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 660, 28), dimension_dict_1460, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 660)
    subscript_call_result_1462 = invoke(stypy.reporting.localization.Localization(__file__, 660, 28), getitem___1461, ind_1459)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 660, 28), tuple_1458, subscript_call_result_1462)
    # Adding element type (line 660)
    # Getting the type of 'ind' (line 660)
    ind_1463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 660, 49), 'ind')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 660, 28), tuple_1458, ind_1463)
    
    list_1466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 660, 27), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 660, 27), list_1466, tuple_1458)
    # Assigning a type to the variable 'sort_result' (line 660)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 660, 12), 'sort_result', list_1466)
    
    # Assigning a Call to a Name (line 661):
    
    # Assigning a Call to a Name (line 661):
    
    # Call to join(...): (line 661)
    # Processing the call arguments (line 661)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to sorted(...): (line 661)
    # Processing the call arguments (line 661)
    # Getting the type of 'sort_result' (line 661)
    sort_result_1474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 55), 'sort_result', False)
    # Processing the call keyword arguments (line 661)
    kwargs_1475 = {}
    # Getting the type of 'sorted' (line 661)
    sorted_1473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 48), 'sorted', False)
    # Calling sorted(args, kwargs) (line 661)
    sorted_call_result_1476 = invoke(stypy.reporting.localization.Localization(__file__, 661, 48), sorted_1473, *[sort_result_1474], **kwargs_1475)
    
    comprehension_1477 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 661, 34), sorted_call_result_1476)
    # Assigning a type to the variable 'x' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 34), 'x', comprehension_1477)
    
    # Obtaining the type of the subscript
    int_1469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 36), 'int')
    # Getting the type of 'x' (line 661)
    x_1470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 661, 34), 'x', False)
    # Obtaining the member '__getitem__' of a type (line 661)
    getitem___1471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 34), x_1470, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 661)
    subscript_call_result_1472 = invoke(stypy.reporting.localization.Localization(__file__, 661, 34), getitem___1471, int_1469)
    
    list_1478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 34), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 661, 34), list_1478, subscript_call_result_1472)
    # Processing the call keyword arguments (line 661)
    kwargs_1479 = {}
    str_1467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 661, 25), 'str', '')
    # Obtaining the member 'join' of a type (line 661)
    join_1468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 661, 25), str_1467, 'join')
    # Calling join(args, kwargs) (line 661)
    join_call_result_1480 = invoke(stypy.reporting.localization.Localization(__file__, 661, 25), join_1468, *[list_1478], **kwargs_1479)
    
    # Assigning a type to the variable 'idx_result' (line 661)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 661, 12), 'idx_result', join_call_result_1480)
    # SSA join for if statement (line 657)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 663)
    # Processing the call arguments (line 663)
    # Getting the type of 'idx_result' (line 663)
    idx_result_1483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 26), 'idx_result', False)
    # Processing the call keyword arguments (line 663)
    kwargs_1484 = {}
    # Getting the type of 'input_list' (line 663)
    input_list_1481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 663, 8), 'input_list', False)
    # Obtaining the member 'append' of a type (line 663)
    append_1482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 663, 8), input_list_1481, 'append')
    # Calling append(args, kwargs) (line 663)
    append_call_result_1485 = invoke(stypy.reporting.localization.Localization(__file__, 663, 8), append_1482, *[idx_result_1483], **kwargs_1484)
    
    
    # Assigning a BinOp to a Name (line 664):
    
    # Assigning a BinOp to a Name (line 664):
    
    # Call to join(...): (line 664)
    # Processing the call arguments (line 664)
    # Getting the type of 'tmp_inputs' (line 664)
    tmp_inputs_1488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 30), 'tmp_inputs', False)
    # Processing the call keyword arguments (line 664)
    kwargs_1489 = {}
    str_1486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 21), 'str', ',')
    # Obtaining the member 'join' of a type (line 664)
    join_1487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 664, 21), str_1486, 'join')
    # Calling join(args, kwargs) (line 664)
    join_call_result_1490 = invoke(stypy.reporting.localization.Localization(__file__, 664, 21), join_1487, *[tmp_inputs_1488], **kwargs_1489)
    
    str_1491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 664, 44), 'str', '->')
    # Applying the binary operator '+' (line 664)
    result_add_1492 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 21), '+', join_call_result_1490, str_1491)
    
    # Getting the type of 'idx_result' (line 664)
    idx_result_1493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 664, 51), 'idx_result')
    # Applying the binary operator '+' (line 664)
    result_add_1494 = python_operator(stypy.reporting.localization.Localization(__file__, 664, 49), '+', result_add_1492, idx_result_1493)
    
    # Assigning a type to the variable 'einsum_str' (line 664)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 664, 8), 'einsum_str', result_add_1494)
    
    # Assigning a Tuple to a Name (line 666):
    
    # Assigning a Tuple to a Name (line 666):
    
    # Obtaining an instance of the builtin type 'tuple' (line 666)
    tuple_1495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 666, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 666)
    # Adding element type (line 666)
    # Getting the type of 'contract_inds' (line 666)
    contract_inds_1496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 23), 'contract_inds')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 23), tuple_1495, contract_inds_1496)
    # Adding element type (line 666)
    # Getting the type of 'idx_removed' (line 666)
    idx_removed_1497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 38), 'idx_removed')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 23), tuple_1495, idx_removed_1497)
    # Adding element type (line 666)
    # Getting the type of 'einsum_str' (line 666)
    einsum_str_1498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 51), 'einsum_str')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 23), tuple_1495, einsum_str_1498)
    # Adding element type (line 666)
    
    # Obtaining the type of the subscript
    slice_1499 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 666, 63), None, None, None)
    # Getting the type of 'input_list' (line 666)
    input_list_1500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 666, 63), 'input_list')
    # Obtaining the member '__getitem__' of a type (line 666)
    getitem___1501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 666, 63), input_list_1500, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 666)
    subscript_call_result_1502 = invoke(stypy.reporting.localization.Localization(__file__, 666, 63), getitem___1501, slice_1499)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 666, 23), tuple_1495, subscript_call_result_1502)
    
    # Assigning a type to the variable 'contraction' (line 666)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 666, 8), 'contraction', tuple_1495)
    
    # Call to append(...): (line 667)
    # Processing the call arguments (line 667)
    # Getting the type of 'contraction' (line 667)
    contraction_1505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 32), 'contraction', False)
    # Processing the call keyword arguments (line 667)
    kwargs_1506 = {}
    # Getting the type of 'contraction_list' (line 667)
    contraction_list_1503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 667, 8), 'contraction_list', False)
    # Obtaining the member 'append' of a type (line 667)
    append_1504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 667, 8), contraction_list_1503, 'append')
    # Calling append(args, kwargs) (line 667)
    append_call_result_1507 = invoke(stypy.reporting.localization.Localization(__file__, 667, 8), append_1504, *[contraction_1505], **kwargs_1506)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 669):
    
    # Assigning a BinOp to a Name (line 669):
    
    # Call to sum(...): (line 669)
    # Processing the call arguments (line 669)
    # Getting the type of 'cost_list' (line 669)
    cost_list_1509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 19), 'cost_list', False)
    # Processing the call keyword arguments (line 669)
    kwargs_1510 = {}
    # Getting the type of 'sum' (line 669)
    sum_1508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 669, 15), 'sum', False)
    # Calling sum(args, kwargs) (line 669)
    sum_call_result_1511 = invoke(stypy.reporting.localization.Localization(__file__, 669, 15), sum_1508, *[cost_list_1509], **kwargs_1510)
    
    int_1512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 669, 32), 'int')
    # Applying the binary operator '+' (line 669)
    result_add_1513 = python_operator(stypy.reporting.localization.Localization(__file__, 669, 15), '+', sum_call_result_1511, int_1512)
    
    # Assigning a type to the variable 'opt_cost' (line 669)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 669, 4), 'opt_cost', result_add_1513)
    
    # Getting the type of 'einsum_call_arg' (line 671)
    einsum_call_arg_1514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 671, 7), 'einsum_call_arg')
    # Testing the type of an if condition (line 671)
    if_condition_1515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 671, 4), einsum_call_arg_1514)
    # Assigning a type to the variable 'if_condition_1515' (line 671)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 671, 4), 'if_condition_1515', if_condition_1515)
    # SSA begins for if statement (line 671)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 672)
    tuple_1516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 672, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 672)
    # Adding element type (line 672)
    # Getting the type of 'operands' (line 672)
    operands_1517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 16), 'operands')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 16), tuple_1516, operands_1517)
    # Adding element type (line 672)
    # Getting the type of 'contraction_list' (line 672)
    contraction_list_1518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 672, 26), 'contraction_list')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 672, 16), tuple_1516, contraction_list_1518)
    
    # Assigning a type to the variable 'stypy_return_type' (line 672)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 672, 8), 'stypy_return_type', tuple_1516)
    # SSA join for if statement (line 671)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 675):
    
    # Assigning a BinOp to a Name (line 675):
    # Getting the type of 'input_subscripts' (line 675)
    input_subscripts_1519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 26), 'input_subscripts')
    str_1520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 675, 45), 'str', '->')
    # Applying the binary operator '+' (line 675)
    result_add_1521 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 26), '+', input_subscripts_1519, str_1520)
    
    # Getting the type of 'output_subscript' (line 675)
    output_subscript_1522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 675, 52), 'output_subscript')
    # Applying the binary operator '+' (line 675)
    result_add_1523 = python_operator(stypy.reporting.localization.Localization(__file__, 675, 50), '+', result_add_1521, output_subscript_1522)
    
    # Assigning a type to the variable 'overall_contraction' (line 675)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 675, 4), 'overall_contraction', result_add_1523)
    
    # Assigning a Tuple to a Name (line 676):
    
    # Assigning a Tuple to a Name (line 676):
    
    # Obtaining an instance of the builtin type 'tuple' (line 676)
    tuple_1524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 676)
    # Adding element type (line 676)
    str_1525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 14), 'str', 'scaling')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 14), tuple_1524, str_1525)
    # Adding element type (line 676)
    str_1526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 25), 'str', 'current')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 14), tuple_1524, str_1526)
    # Adding element type (line 676)
    str_1527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 676, 36), 'str', 'remaining')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 676, 14), tuple_1524, str_1527)
    
    # Assigning a type to the variable 'header' (line 676)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 676, 4), 'header', tuple_1524)
    
    # Assigning a BinOp to a Name (line 678):
    
    # Assigning a BinOp to a Name (line 678):
    # Getting the type of 'naive_cost' (line 678)
    naive_cost_1528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 14), 'naive_cost')
    # Getting the type of 'opt_cost' (line 678)
    opt_cost_1529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 678, 27), 'opt_cost')
    # Applying the binary operator 'div' (line 678)
    result_div_1530 = python_operator(stypy.reporting.localization.Localization(__file__, 678, 14), 'div', naive_cost_1528, opt_cost_1529)
    
    # Assigning a type to the variable 'speedup' (line 678)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 678, 4), 'speedup', result_div_1530)
    
    # Assigning a Call to a Name (line 679):
    
    # Assigning a Call to a Name (line 679):
    
    # Call to max(...): (line 679)
    # Processing the call arguments (line 679)
    # Getting the type of 'size_list' (line 679)
    size_list_1532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 16), 'size_list', False)
    # Processing the call keyword arguments (line 679)
    kwargs_1533 = {}
    # Getting the type of 'max' (line 679)
    max_1531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 679, 12), 'max', False)
    # Calling max(args, kwargs) (line 679)
    max_call_result_1534 = invoke(stypy.reporting.localization.Localization(__file__, 679, 12), max_1531, *[size_list_1532], **kwargs_1533)
    
    # Assigning a type to the variable 'max_i' (line 679)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 679, 4), 'max_i', max_call_result_1534)
    
    # Assigning a BinOp to a Name (line 681):
    
    # Assigning a BinOp to a Name (line 681):
    str_1535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 681, 18), 'str', '  Complete contraction:  %s\n')
    # Getting the type of 'overall_contraction' (line 681)
    overall_contraction_1536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 681, 52), 'overall_contraction')
    # Applying the binary operator '%' (line 681)
    result_mod_1537 = python_operator(stypy.reporting.localization.Localization(__file__, 681, 18), '%', str_1535, overall_contraction_1536)
    
    # Assigning a type to the variable 'path_print' (line 681)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 681, 4), 'path_print', result_mod_1537)
    
    # Getting the type of 'path_print' (line 682)
    path_print_1538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 4), 'path_print')
    str_1539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 682, 18), 'str', '         Naive scaling:  %d\n')
    
    # Call to len(...): (line 682)
    # Processing the call arguments (line 682)
    # Getting the type of 'indices' (line 682)
    indices_1541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 56), 'indices', False)
    # Processing the call keyword arguments (line 682)
    kwargs_1542 = {}
    # Getting the type of 'len' (line 682)
    len_1540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 682, 52), 'len', False)
    # Calling len(args, kwargs) (line 682)
    len_call_result_1543 = invoke(stypy.reporting.localization.Localization(__file__, 682, 52), len_1540, *[indices_1541], **kwargs_1542)
    
    # Applying the binary operator '%' (line 682)
    result_mod_1544 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 18), '%', str_1539, len_call_result_1543)
    
    # Applying the binary operator '+=' (line 682)
    result_iadd_1545 = python_operator(stypy.reporting.localization.Localization(__file__, 682, 4), '+=', path_print_1538, result_mod_1544)
    # Assigning a type to the variable 'path_print' (line 682)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 682, 4), 'path_print', result_iadd_1545)
    
    
    # Getting the type of 'path_print' (line 683)
    path_print_1546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'path_print')
    str_1547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 683, 18), 'str', '     Optimized scaling:  %d\n')
    
    # Call to max(...): (line 683)
    # Processing the call arguments (line 683)
    # Getting the type of 'scale_list' (line 683)
    scale_list_1549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 56), 'scale_list', False)
    # Processing the call keyword arguments (line 683)
    kwargs_1550 = {}
    # Getting the type of 'max' (line 683)
    max_1548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 683, 52), 'max', False)
    # Calling max(args, kwargs) (line 683)
    max_call_result_1551 = invoke(stypy.reporting.localization.Localization(__file__, 683, 52), max_1548, *[scale_list_1549], **kwargs_1550)
    
    # Applying the binary operator '%' (line 683)
    result_mod_1552 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 18), '%', str_1547, max_call_result_1551)
    
    # Applying the binary operator '+=' (line 683)
    result_iadd_1553 = python_operator(stypy.reporting.localization.Localization(__file__, 683, 4), '+=', path_print_1546, result_mod_1552)
    # Assigning a type to the variable 'path_print' (line 683)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 683, 4), 'path_print', result_iadd_1553)
    
    
    # Getting the type of 'path_print' (line 684)
    path_print_1554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'path_print')
    str_1555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 684, 18), 'str', '      Naive FLOP count:  %.3e\n')
    # Getting the type of 'naive_cost' (line 684)
    naive_cost_1556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 684, 54), 'naive_cost')
    # Applying the binary operator '%' (line 684)
    result_mod_1557 = python_operator(stypy.reporting.localization.Localization(__file__, 684, 18), '%', str_1555, naive_cost_1556)
    
    # Applying the binary operator '+=' (line 684)
    result_iadd_1558 = python_operator(stypy.reporting.localization.Localization(__file__, 684, 4), '+=', path_print_1554, result_mod_1557)
    # Assigning a type to the variable 'path_print' (line 684)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 684, 4), 'path_print', result_iadd_1558)
    
    
    # Getting the type of 'path_print' (line 685)
    path_print_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'path_print')
    str_1560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 685, 18), 'str', '  Optimized FLOP count:  %.3e\n')
    # Getting the type of 'opt_cost' (line 685)
    opt_cost_1561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 685, 54), 'opt_cost')
    # Applying the binary operator '%' (line 685)
    result_mod_1562 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 18), '%', str_1560, opt_cost_1561)
    
    # Applying the binary operator '+=' (line 685)
    result_iadd_1563 = python_operator(stypy.reporting.localization.Localization(__file__, 685, 4), '+=', path_print_1559, result_mod_1562)
    # Assigning a type to the variable 'path_print' (line 685)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 685, 4), 'path_print', result_iadd_1563)
    
    
    # Getting the type of 'path_print' (line 686)
    path_print_1564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'path_print')
    str_1565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 686, 18), 'str', '   Theoretical speedup:  %3.3f\n')
    # Getting the type of 'speedup' (line 686)
    speedup_1566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 686, 55), 'speedup')
    # Applying the binary operator '%' (line 686)
    result_mod_1567 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 18), '%', str_1565, speedup_1566)
    
    # Applying the binary operator '+=' (line 686)
    result_iadd_1568 = python_operator(stypy.reporting.localization.Localization(__file__, 686, 4), '+=', path_print_1564, result_mod_1567)
    # Assigning a type to the variable 'path_print' (line 686)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 686, 4), 'path_print', result_iadd_1568)
    
    
    # Getting the type of 'path_print' (line 687)
    path_print_1569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'path_print')
    str_1570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 687, 18), 'str', '  Largest intermediate:  %.3e elements\n')
    # Getting the type of 'max_i' (line 687)
    max_i_1571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 687, 63), 'max_i')
    # Applying the binary operator '%' (line 687)
    result_mod_1572 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 18), '%', str_1570, max_i_1571)
    
    # Applying the binary operator '+=' (line 687)
    result_iadd_1573 = python_operator(stypy.reporting.localization.Localization(__file__, 687, 4), '+=', path_print_1569, result_mod_1572)
    # Assigning a type to the variable 'path_print' (line 687)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 687, 4), 'path_print', result_iadd_1573)
    
    
    # Getting the type of 'path_print' (line 688)
    path_print_1574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 688, 4), 'path_print')
    str_1575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 18), 'str', '-')
    int_1576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 24), 'int')
    # Applying the binary operator '*' (line 688)
    result_mul_1577 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 18), '*', str_1575, int_1576)
    
    str_1578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 688, 29), 'str', '\n')
    # Applying the binary operator '+' (line 688)
    result_add_1579 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 18), '+', result_mul_1577, str_1578)
    
    # Applying the binary operator '+=' (line 688)
    result_iadd_1580 = python_operator(stypy.reporting.localization.Localization(__file__, 688, 4), '+=', path_print_1574, result_add_1579)
    # Assigning a type to the variable 'path_print' (line 688)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 688, 4), 'path_print', result_iadd_1580)
    
    
    # Getting the type of 'path_print' (line 689)
    path_print_1581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'path_print')
    str_1582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 689, 18), 'str', '%6s %24s %40s\n')
    # Getting the type of 'header' (line 689)
    header_1583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 689, 38), 'header')
    # Applying the binary operator '%' (line 689)
    result_mod_1584 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 18), '%', str_1582, header_1583)
    
    # Applying the binary operator '+=' (line 689)
    result_iadd_1585 = python_operator(stypy.reporting.localization.Localization(__file__, 689, 4), '+=', path_print_1581, result_mod_1584)
    # Assigning a type to the variable 'path_print' (line 689)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 689, 4), 'path_print', result_iadd_1585)
    
    
    # Getting the type of 'path_print' (line 690)
    path_print_1586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'path_print')
    str_1587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 18), 'str', '-')
    int_1588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 690, 24), 'int')
    # Applying the binary operator '*' (line 690)
    result_mul_1589 = python_operator(stypy.reporting.localization.Localization(__file__, 690, 18), '*', str_1587, int_1588)
    
    # Applying the binary operator '+=' (line 690)
    result_iadd_1590 = python_operator(stypy.reporting.localization.Localization(__file__, 690, 4), '+=', path_print_1586, result_mul_1589)
    # Assigning a type to the variable 'path_print' (line 690)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 690, 4), 'path_print', result_iadd_1590)
    
    
    
    # Call to enumerate(...): (line 692)
    # Processing the call arguments (line 692)
    # Getting the type of 'contraction_list' (line 692)
    contraction_list_1592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 36), 'contraction_list', False)
    # Processing the call keyword arguments (line 692)
    kwargs_1593 = {}
    # Getting the type of 'enumerate' (line 692)
    enumerate_1591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 692, 26), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 692)
    enumerate_call_result_1594 = invoke(stypy.reporting.localization.Localization(__file__, 692, 26), enumerate_1591, *[contraction_list_1592], **kwargs_1593)
    
    # Testing the type of a for loop iterable (line 692)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 692, 4), enumerate_call_result_1594)
    # Getting the type of the for loop variable (line 692)
    for_loop_var_1595 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 692, 4), enumerate_call_result_1594)
    # Assigning a type to the variable 'n' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 4), for_loop_var_1595))
    # Assigning a type to the variable 'contraction' (line 692)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 692, 4), 'contraction', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 692, 4), for_loop_var_1595))
    # SSA begins for a for statement (line 692)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Tuple (line 693):
    
    # Assigning a Subscript to a Name (line 693):
    
    # Obtaining the type of the subscript
    int_1596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 8), 'int')
    # Getting the type of 'contraction' (line 693)
    contraction_1597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 46), 'contraction')
    # Obtaining the member '__getitem__' of a type (line 693)
    getitem___1598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 8), contraction_1597, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 693)
    subscript_call_result_1599 = invoke(stypy.reporting.localization.Localization(__file__, 693, 8), getitem___1598, int_1596)
    
    # Assigning a type to the variable 'tuple_var_assignment_27' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_var_assignment_27', subscript_call_result_1599)
    
    # Assigning a Subscript to a Name (line 693):
    
    # Obtaining the type of the subscript
    int_1600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 8), 'int')
    # Getting the type of 'contraction' (line 693)
    contraction_1601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 46), 'contraction')
    # Obtaining the member '__getitem__' of a type (line 693)
    getitem___1602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 8), contraction_1601, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 693)
    subscript_call_result_1603 = invoke(stypy.reporting.localization.Localization(__file__, 693, 8), getitem___1602, int_1600)
    
    # Assigning a type to the variable 'tuple_var_assignment_28' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_var_assignment_28', subscript_call_result_1603)
    
    # Assigning a Subscript to a Name (line 693):
    
    # Obtaining the type of the subscript
    int_1604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 8), 'int')
    # Getting the type of 'contraction' (line 693)
    contraction_1605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 46), 'contraction')
    # Obtaining the member '__getitem__' of a type (line 693)
    getitem___1606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 8), contraction_1605, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 693)
    subscript_call_result_1607 = invoke(stypy.reporting.localization.Localization(__file__, 693, 8), getitem___1606, int_1604)
    
    # Assigning a type to the variable 'tuple_var_assignment_29' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_var_assignment_29', subscript_call_result_1607)
    
    # Assigning a Subscript to a Name (line 693):
    
    # Obtaining the type of the subscript
    int_1608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 693, 8), 'int')
    # Getting the type of 'contraction' (line 693)
    contraction_1609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 46), 'contraction')
    # Obtaining the member '__getitem__' of a type (line 693)
    getitem___1610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 693, 8), contraction_1609, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 693)
    subscript_call_result_1611 = invoke(stypy.reporting.localization.Localization(__file__, 693, 8), getitem___1610, int_1608)
    
    # Assigning a type to the variable 'tuple_var_assignment_30' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_var_assignment_30', subscript_call_result_1611)
    
    # Assigning a Name to a Name (line 693):
    # Getting the type of 'tuple_var_assignment_27' (line 693)
    tuple_var_assignment_27_1612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_var_assignment_27')
    # Assigning a type to the variable 'inds' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'inds', tuple_var_assignment_27_1612)
    
    # Assigning a Name to a Name (line 693):
    # Getting the type of 'tuple_var_assignment_28' (line 693)
    tuple_var_assignment_28_1613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_var_assignment_28')
    # Assigning a type to the variable 'idx_rm' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 14), 'idx_rm', tuple_var_assignment_28_1613)
    
    # Assigning a Name to a Name (line 693):
    # Getting the type of 'tuple_var_assignment_29' (line 693)
    tuple_var_assignment_29_1614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_var_assignment_29')
    # Assigning a type to the variable 'einsum_str' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 22), 'einsum_str', tuple_var_assignment_29_1614)
    
    # Assigning a Name to a Name (line 693):
    # Getting the type of 'tuple_var_assignment_30' (line 693)
    tuple_var_assignment_30_1615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 693, 8), 'tuple_var_assignment_30')
    # Assigning a type to the variable 'remaining' (line 693)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 693, 34), 'remaining', tuple_var_assignment_30_1615)
    
    # Assigning a BinOp to a Name (line 694):
    
    # Assigning a BinOp to a Name (line 694):
    
    # Call to join(...): (line 694)
    # Processing the call arguments (line 694)
    # Getting the type of 'remaining' (line 694)
    remaining_1618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 33), 'remaining', False)
    # Processing the call keyword arguments (line 694)
    kwargs_1619 = {}
    str_1616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 24), 'str', ',')
    # Obtaining the member 'join' of a type (line 694)
    join_1617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 694, 24), str_1616, 'join')
    # Calling join(args, kwargs) (line 694)
    join_call_result_1620 = invoke(stypy.reporting.localization.Localization(__file__, 694, 24), join_1617, *[remaining_1618], **kwargs_1619)
    
    str_1621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 694, 46), 'str', '->')
    # Applying the binary operator '+' (line 694)
    result_add_1622 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 24), '+', join_call_result_1620, str_1621)
    
    # Getting the type of 'output_subscript' (line 694)
    output_subscript_1623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 694, 53), 'output_subscript')
    # Applying the binary operator '+' (line 694)
    result_add_1624 = python_operator(stypy.reporting.localization.Localization(__file__, 694, 51), '+', result_add_1622, output_subscript_1623)
    
    # Assigning a type to the variable 'remaining_str' (line 694)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 694, 8), 'remaining_str', result_add_1624)
    
    # Assigning a Tuple to a Name (line 695):
    
    # Assigning a Tuple to a Name (line 695):
    
    # Obtaining an instance of the builtin type 'tuple' (line 695)
    tuple_1625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 695, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 695)
    # Adding element type (line 695)
    
    # Obtaining the type of the subscript
    # Getting the type of 'n' (line 695)
    n_1626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 31), 'n')
    # Getting the type of 'scale_list' (line 695)
    scale_list_1627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 20), 'scale_list')
    # Obtaining the member '__getitem__' of a type (line 695)
    getitem___1628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 695, 20), scale_list_1627, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 695)
    subscript_call_result_1629 = invoke(stypy.reporting.localization.Localization(__file__, 695, 20), getitem___1628, n_1626)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 20), tuple_1625, subscript_call_result_1629)
    # Adding element type (line 695)
    # Getting the type of 'einsum_str' (line 695)
    einsum_str_1630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 35), 'einsum_str')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 20), tuple_1625, einsum_str_1630)
    # Adding element type (line 695)
    # Getting the type of 'remaining_str' (line 695)
    remaining_str_1631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 695, 47), 'remaining_str')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 695, 20), tuple_1625, remaining_str_1631)
    
    # Assigning a type to the variable 'path_run' (line 695)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 695, 8), 'path_run', tuple_1625)
    
    # Getting the type of 'path_print' (line 696)
    path_print_1632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'path_print')
    str_1633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 696, 22), 'str', '\n%4d    %24s %40s')
    # Getting the type of 'path_run' (line 696)
    path_run_1634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 696, 45), 'path_run')
    # Applying the binary operator '%' (line 696)
    result_mod_1635 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 22), '%', str_1633, path_run_1634)
    
    # Applying the binary operator '+=' (line 696)
    result_iadd_1636 = python_operator(stypy.reporting.localization.Localization(__file__, 696, 8), '+=', path_print_1632, result_mod_1635)
    # Assigning a type to the variable 'path_print' (line 696)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 696, 8), 'path_print', result_iadd_1636)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 698):
    
    # Assigning a BinOp to a Name (line 698):
    
    # Obtaining an instance of the builtin type 'list' (line 698)
    list_1637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 698)
    # Adding element type (line 698)
    str_1638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 698, 12), 'str', 'einsum_path')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 698, 11), list_1637, str_1638)
    
    # Getting the type of 'path' (line 698)
    path_1639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 698, 29), 'path')
    # Applying the binary operator '+' (line 698)
    result_add_1640 = python_operator(stypy.reporting.localization.Localization(__file__, 698, 11), '+', list_1637, path_1639)
    
    # Assigning a type to the variable 'path' (line 698)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 698, 4), 'path', result_add_1640)
    
    # Obtaining an instance of the builtin type 'tuple' (line 699)
    tuple_1641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 699, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 699)
    # Adding element type (line 699)
    # Getting the type of 'path' (line 699)
    path_1642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 12), 'path')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 699, 12), tuple_1641, path_1642)
    # Adding element type (line 699)
    # Getting the type of 'path_print' (line 699)
    path_print_1643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 699, 18), 'path_print')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 699, 12), tuple_1641, path_print_1643)
    
    # Assigning a type to the variable 'stypy_return_type' (line 699)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 699, 4), 'stypy_return_type', tuple_1641)
    
    # ################# End of 'einsum_path(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'einsum_path' in the type store
    # Getting the type of 'stypy_return_type' (line 427)
    stypy_return_type_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1644)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'einsum_path'
    return stypy_return_type_1644

# Assigning a type to the variable 'einsum_path' (line 427)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 0), 'einsum_path', einsum_path)

@norecursion
def einsum(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'einsum'
    module_type_store = module_type_store.open_function_context('einsum', 703, 0, False)
    
    # Passed parameters checking function
    einsum.stypy_localization = localization
    einsum.stypy_type_of_self = None
    einsum.stypy_type_store = module_type_store
    einsum.stypy_function_name = 'einsum'
    einsum.stypy_param_names_list = []
    einsum.stypy_varargs_param_name = 'operands'
    einsum.stypy_kwargs_param_name = 'kwargs'
    einsum.stypy_call_defaults = defaults
    einsum.stypy_call_varargs = varargs
    einsum.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'einsum', [], 'operands', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'einsum', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'einsum(...)' code ##################

    str_1645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 941, (-1)), 'str', "\n    einsum(subscripts, *operands, out=None, dtype=None, order='K',\n           casting='safe', optimize=False)\n\n    Evaluates the Einstein summation convention on the operands.\n\n    Using the Einstein summation convention, many common multi-dimensional\n    array operations can be represented in a simple fashion.  This function\n    provides a way to compute such summations. The best way to understand this\n    function is to try the examples below, which show how many common NumPy\n    functions can be implemented as calls to `einsum`.\n\n    Parameters\n    ----------\n    subscripts : str\n        Specifies the subscripts for summation.\n    operands : list of array_like\n        These are the arrays for the operation.\n    out : {ndarray, None}, optional\n        If provided, the calculation is done into this array.\n    dtype : {data-type, None}, optional\n        If provided, forces the calculation to use the data type specified.\n        Note that you may have to also give a more liberal `casting`\n        parameter to allow the conversions. Default is None.\n    order : {'C', 'F', 'A', 'K'}, optional\n        Controls the memory layout of the output. 'C' means it should\n        be C contiguous. 'F' means it should be Fortran contiguous,\n        'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.\n        'K' means it should be as close to the layout as the inputs as\n        is possible, including arbitrarily permuted axes.\n        Default is 'K'.\n    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional\n        Controls what kind of data casting may occur.  Setting this to\n        'unsafe' is not recommended, as it can adversely affect accumulations.\n\n          * 'no' means the data types should not be cast at all.\n          * 'equiv' means only byte-order changes are allowed.\n          * 'safe' means only casts which can preserve values are allowed.\n          * 'same_kind' means only safe casts or casts within a kind,\n            like float64 to float32, are allowed.\n          * 'unsafe' means any data conversions may be done.\n\n        Default is 'safe'.\n    optimize : {False, True, 'greedy', 'optimal'}, optional\n        Controls if intermediate optimization should occur. No optimization\n        will occur if False and True will default to the 'greedy' algorithm.\n        Also accepts an explicit contraction list from the ``np.einsum_path``\n        function. See ``np.einsum_path`` for more details. Default is False.\n\n    Returns\n    -------\n    output : ndarray\n        The calculation based on the Einstein summation convention.\n\n    See Also\n    --------\n    einsum_path, dot, inner, outer, tensordot, linalg.multi_dot\n\n    Notes\n    -----\n    .. versionadded:: 1.6.0\n\n    The subscripts string is a comma-separated list of subscript labels,\n    where each label refers to a dimension of the corresponding operand.\n    Repeated subscripts labels in one operand take the diagonal.  For example,\n    ``np.einsum('ii', a)`` is equivalent to ``np.trace(a)``.\n\n    Whenever a label is repeated, it is summed, so ``np.einsum('i,i', a, b)``\n    is equivalent to ``np.inner(a,b)``.  If a label appears only once,\n    it is not summed, so ``np.einsum('i', a)`` produces a view of ``a``\n    with no changes.\n\n    The order of labels in the output is by default alphabetical.  This\n    means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while\n    ``np.einsum('ji', a)`` takes its transpose.\n\n    The output can be controlled by specifying output subscript labels\n    as well.  This specifies the label order, and allows summing to\n    be disallowed or forced when desired.  The call ``np.einsum('i->', a)``\n    is like ``np.sum(a, axis=-1)``, and ``np.einsum('ii->i', a)``\n    is like ``np.diag(a)``.  The difference is that `einsum` does not\n    allow broadcasting by default.\n\n    To enable and control broadcasting, use an ellipsis.  Default\n    NumPy-style broadcasting is done by adding an ellipsis\n    to the left of each term, like ``np.einsum('...ii->...i', a)``.\n    To take the trace along the first and last axes,\n    you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix\n    product with the left-most indices instead of rightmost, you can do\n    ``np.einsum('ij...,jk...->ik...', a, b)``.\n\n    When there is only one operand, no axes are summed, and no output\n    parameter is provided, a view into the operand is returned instead\n    of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``\n    produces a view.\n\n    An alternative way to provide the subscripts and operands is as\n    ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``. The examples\n    below have corresponding `einsum` calls with the two parameter methods.\n\n    .. versionadded:: 1.10.0\n\n    Views returned from einsum are now writeable whenever the input array\n    is writeable. For example, ``np.einsum('ijk...->kji...', a)`` will now\n    have the same effect as ``np.swapaxes(a, 0, 2)`` and\n    ``np.einsum('ii->i', a)`` will return a writeable view of the diagonal\n    of a 2D array.\n\n    .. versionadded:: 1.12.0\n\n    Added the ``optimize`` argument which will optimize the contraction order\n    of an einsum expression. For a contraction with three or more operands this\n    can greatly increase the computational efficiency at the cost of a larger\n    memory footprint during computation.\n\n    See ``np.einsum_path`` for more details.\n\n    Examples\n    --------\n    >>> a = np.arange(25).reshape(5,5)\n    >>> b = np.arange(5)\n    >>> c = np.arange(6).reshape(2,3)\n\n    >>> np.einsum('ii', a)\n    60\n    >>> np.einsum(a, [0,0])\n    60\n    >>> np.trace(a)\n    60\n\n    >>> np.einsum('ii->i', a)\n    array([ 0,  6, 12, 18, 24])\n    >>> np.einsum(a, [0,0], [0])\n    array([ 0,  6, 12, 18, 24])\n    >>> np.diag(a)\n    array([ 0,  6, 12, 18, 24])\n\n    >>> np.einsum('ij,j', a, b)\n    array([ 30,  80, 130, 180, 230])\n    >>> np.einsum(a, [0,1], b, [1])\n    array([ 30,  80, 130, 180, 230])\n    >>> np.dot(a, b)\n    array([ 30,  80, 130, 180, 230])\n    >>> np.einsum('...j,j', a, b)\n    array([ 30,  80, 130, 180, 230])\n\n    >>> np.einsum('ji', c)\n    array([[0, 3],\n           [1, 4],\n           [2, 5]])\n    >>> np.einsum(c, [1,0])\n    array([[0, 3],\n           [1, 4],\n           [2, 5]])\n    >>> c.T\n    array([[0, 3],\n           [1, 4],\n           [2, 5]])\n\n    >>> np.einsum('..., ...', 3, c)\n    array([[ 0,  3,  6],\n           [ 9, 12, 15]])\n    >>> np.einsum(',ij', 3, C)\n    array([[ 0,  3,  6],\n           [ 9, 12, 15]])\n    >>> np.einsum(3, [Ellipsis], c, [Ellipsis])\n    array([[ 0,  3,  6],\n           [ 9, 12, 15]])\n    >>> np.multiply(3, c)\n    array([[ 0,  3,  6],\n           [ 9, 12, 15]])\n\n    >>> np.einsum('i,i', b, b)\n    30\n    >>> np.einsum(b, [0], b, [0])\n    30\n    >>> np.inner(b,b)\n    30\n\n    >>> np.einsum('i,j', np.arange(2)+1, b)\n    array([[0, 1, 2, 3, 4],\n           [0, 2, 4, 6, 8]])\n    >>> np.einsum(np.arange(2)+1, [0], b, [1])\n    array([[0, 1, 2, 3, 4],\n           [0, 2, 4, 6, 8]])\n    >>> np.outer(np.arange(2)+1, b)\n    array([[0, 1, 2, 3, 4],\n           [0, 2, 4, 6, 8]])\n\n    >>> np.einsum('i...->...', a)\n    array([50, 55, 60, 65, 70])\n    >>> np.einsum(a, [0,Ellipsis], [Ellipsis])\n    array([50, 55, 60, 65, 70])\n    >>> np.sum(a, axis=0)\n    array([50, 55, 60, 65, 70])\n\n    >>> a = np.arange(60.).reshape(3,4,5)\n    >>> b = np.arange(24.).reshape(4,3,2)\n    >>> np.einsum('ijk,jil->kl', a, b)\n    array([[ 4400.,  4730.],\n           [ 4532.,  4874.],\n           [ 4664.,  5018.],\n           [ 4796.,  5162.],\n           [ 4928.,  5306.]])\n    >>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])\n    array([[ 4400.,  4730.],\n           [ 4532.,  4874.],\n           [ 4664.,  5018.],\n           [ 4796.,  5162.],\n           [ 4928.,  5306.]])\n    >>> np.tensordot(a,b, axes=([1,0],[0,1]))\n    array([[ 4400.,  4730.],\n           [ 4532.,  4874.],\n           [ 4664.,  5018.],\n           [ 4796.,  5162.],\n           [ 4928.,  5306.]])\n\n    >>> a = np.arange(6).reshape((3,2))\n    >>> b = np.arange(12).reshape((4,3))\n    >>> np.einsum('ki,jk->ij', a, b)\n    array([[10, 28, 46, 64],\n           [13, 40, 67, 94]])\n    >>> np.einsum('ki,...k->i...', a, b)\n    array([[10, 28, 46, 64],\n           [13, 40, 67, 94]])\n    >>> np.einsum('k...,jk', a, b)\n    array([[10, 28, 46, 64],\n           [13, 40, 67, 94]])\n\n    >>> # since version 1.10.0\n    >>> a = np.zeros((3, 3))\n    >>> np.einsum('ii->i', a)[:] = 1\n    >>> a\n    array([[ 1.,  0.,  0.],\n           [ 0.,  1.,  0.],\n           [ 0.,  0.,  1.]])\n\n    ")
    
    # Assigning a Call to a Name (line 944):
    
    # Assigning a Call to a Name (line 944):
    
    # Call to pop(...): (line 944)
    # Processing the call arguments (line 944)
    str_1648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 944, 30), 'str', 'optimize')
    # Getting the type of 'False' (line 944)
    False_1649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 42), 'False', False)
    # Processing the call keyword arguments (line 944)
    kwargs_1650 = {}
    # Getting the type of 'kwargs' (line 944)
    kwargs_1646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 944, 19), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 944)
    pop_1647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 944, 19), kwargs_1646, 'pop')
    # Calling pop(args, kwargs) (line 944)
    pop_call_result_1651 = invoke(stypy.reporting.localization.Localization(__file__, 944, 19), pop_1647, *[str_1648, False_1649], **kwargs_1650)
    
    # Assigning a type to the variable 'optimize_arg' (line 944)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 944, 4), 'optimize_arg', pop_call_result_1651)
    
    
    # Getting the type of 'optimize_arg' (line 947)
    optimize_arg_1652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 7), 'optimize_arg')
    # Getting the type of 'False' (line 947)
    False_1653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 947, 23), 'False')
    # Applying the binary operator 'is' (line 947)
    result_is__1654 = python_operator(stypy.reporting.localization.Localization(__file__, 947, 7), 'is', optimize_arg_1652, False_1653)
    
    # Testing the type of an if condition (line 947)
    if_condition_1655 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 947, 4), result_is__1654)
    # Assigning a type to the variable 'if_condition_1655' (line 947)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 947, 4), 'if_condition_1655', if_condition_1655)
    # SSA begins for if statement (line 947)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to c_einsum(...): (line 948)
    # Getting the type of 'operands' (line 948)
    operands_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 25), 'operands', False)
    # Processing the call keyword arguments (line 948)
    # Getting the type of 'kwargs' (line 948)
    kwargs_1658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 37), 'kwargs', False)
    kwargs_1659 = {'kwargs_1658': kwargs_1658}
    # Getting the type of 'c_einsum' (line 948)
    c_einsum_1656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 948, 15), 'c_einsum', False)
    # Calling c_einsum(args, kwargs) (line 948)
    c_einsum_call_result_1660 = invoke(stypy.reporting.localization.Localization(__file__, 948, 15), c_einsum_1656, *[operands_1657], **kwargs_1659)
    
    # Assigning a type to the variable 'stypy_return_type' (line 948)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 948, 8), 'stypy_return_type', c_einsum_call_result_1660)
    # SSA join for if statement (line 947)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 950):
    
    # Assigning a List to a Name (line 950):
    
    # Obtaining an instance of the builtin type 'list' (line 950)
    list_1661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 950)
    # Adding element type (line 950)
    str_1662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 27), 'str', 'out')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 950, 26), list_1661, str_1662)
    # Adding element type (line 950)
    str_1663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 34), 'str', 'dtype')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 950, 26), list_1661, str_1663)
    # Adding element type (line 950)
    str_1664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 43), 'str', 'order')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 950, 26), list_1661, str_1664)
    # Adding element type (line 950)
    str_1665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 950, 52), 'str', 'casting')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 950, 26), list_1661, str_1665)
    
    # Assigning a type to the variable 'valid_einsum_kwargs' (line 950)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 950, 4), 'valid_einsum_kwargs', list_1661)
    
    # Assigning a DictComp to a Name (line 951):
    
    # Assigning a DictComp to a Name (line 951):
    # Calculating dict comprehension
    module_type_store = module_type_store.open_function_context('dict comprehension expression', 951, 21, True)
    # Calculating comprehension expression
    
    # Call to items(...): (line 951)
    # Processing the call keyword arguments (line 951)
    kwargs_1673 = {}
    # Getting the type of 'kwargs' (line 951)
    kwargs_1671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 40), 'kwargs', False)
    # Obtaining the member 'items' of a type (line 951)
    items_1672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 951, 40), kwargs_1671, 'items')
    # Calling items(args, kwargs) (line 951)
    items_call_result_1674 = invoke(stypy.reporting.localization.Localization(__file__, 951, 40), items_1672, *[], **kwargs_1673)
    
    comprehension_1675 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 951, 21), items_call_result_1674)
    # Assigning a type to the variable 'k' (line 951)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 21), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 951, 21), comprehension_1675))
    # Assigning a type to the variable 'v' (line 951)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 21), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 951, 21), comprehension_1675))
    
    # Getting the type of 'k' (line 952)
    k_1668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 21), 'k')
    # Getting the type of 'valid_einsum_kwargs' (line 952)
    valid_einsum_kwargs_1669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 952, 26), 'valid_einsum_kwargs')
    # Applying the binary operator 'in' (line 952)
    result_contains_1670 = python_operator(stypy.reporting.localization.Localization(__file__, 952, 21), 'in', k_1668, valid_einsum_kwargs_1669)
    
    # Getting the type of 'k' (line 951)
    k_1666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 21), 'k')
    # Getting the type of 'v' (line 951)
    v_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 951, 24), 'v')
    dict_1676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 951, 21), 'dict')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 951, 21), dict_1676, (k_1666, v_1667))
    # Assigning a type to the variable 'einsum_kwargs' (line 951)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 951, 4), 'einsum_kwargs', dict_1676)
    
    # Assigning a BinOp to a Name (line 955):
    
    # Assigning a BinOp to a Name (line 955):
    
    # Obtaining an instance of the builtin type 'list' (line 955)
    list_1677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 955)
    # Adding element type (line 955)
    str_1678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 955, 29), 'str', 'optimize')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 955, 28), list_1677, str_1678)
    
    # Getting the type of 'valid_einsum_kwargs' (line 955)
    valid_einsum_kwargs_1679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 955, 43), 'valid_einsum_kwargs')
    # Applying the binary operator '+' (line 955)
    result_add_1680 = python_operator(stypy.reporting.localization.Localization(__file__, 955, 28), '+', list_1677, valid_einsum_kwargs_1679)
    
    # Assigning a type to the variable 'valid_contract_kwargs' (line 955)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 955, 4), 'valid_contract_kwargs', result_add_1680)
    
    # Assigning a ListComp to a Name (line 956):
    
    # Assigning a ListComp to a Name (line 956):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to items(...): (line 956)
    # Processing the call keyword arguments (line 956)
    kwargs_1687 = {}
    # Getting the type of 'kwargs' (line 956)
    kwargs_1685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 38), 'kwargs', False)
    # Obtaining the member 'items' of a type (line 956)
    items_1686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 956, 38), kwargs_1685, 'items')
    # Calling items(args, kwargs) (line 956)
    items_call_result_1688 = invoke(stypy.reporting.localization.Localization(__file__, 956, 38), items_1686, *[], **kwargs_1687)
    
    comprehension_1689 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 956, 22), items_call_result_1688)
    # Assigning a type to the variable 'k' (line 956)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 956, 22), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 956, 22), comprehension_1689))
    # Assigning a type to the variable 'v' (line 956)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 956, 22), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 956, 22), comprehension_1689))
    
    # Getting the type of 'k' (line 957)
    k_1682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 22), 'k')
    # Getting the type of 'valid_contract_kwargs' (line 957)
    valid_contract_kwargs_1683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 957, 31), 'valid_contract_kwargs')
    # Applying the binary operator 'notin' (line 957)
    result_contains_1684 = python_operator(stypy.reporting.localization.Localization(__file__, 957, 22), 'notin', k_1682, valid_contract_kwargs_1683)
    
    # Getting the type of 'k' (line 956)
    k_1681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 956, 22), 'k')
    list_1690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 956, 22), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 956, 22), list_1690, k_1681)
    # Assigning a type to the variable 'unknown_kwargs' (line 956)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 956, 4), 'unknown_kwargs', list_1690)
    
    
    # Call to len(...): (line 959)
    # Processing the call arguments (line 959)
    # Getting the type of 'unknown_kwargs' (line 959)
    unknown_kwargs_1692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 11), 'unknown_kwargs', False)
    # Processing the call keyword arguments (line 959)
    kwargs_1693 = {}
    # Getting the type of 'len' (line 959)
    len_1691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 959, 7), 'len', False)
    # Calling len(args, kwargs) (line 959)
    len_call_result_1694 = invoke(stypy.reporting.localization.Localization(__file__, 959, 7), len_1691, *[unknown_kwargs_1692], **kwargs_1693)
    
    # Testing the type of an if condition (line 959)
    if_condition_1695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 959, 4), len_call_result_1694)
    # Assigning a type to the variable 'if_condition_1695' (line 959)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 959, 4), 'if_condition_1695', if_condition_1695)
    # SSA begins for if statement (line 959)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 960)
    # Processing the call arguments (line 960)
    str_1697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 960, 24), 'str', 'Did not understand the following kwargs: %s')
    # Getting the type of 'unknown_kwargs' (line 961)
    unknown_kwargs_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 961, 26), 'unknown_kwargs', False)
    # Applying the binary operator '%' (line 960)
    result_mod_1699 = python_operator(stypy.reporting.localization.Localization(__file__, 960, 24), '%', str_1697, unknown_kwargs_1698)
    
    # Processing the call keyword arguments (line 960)
    kwargs_1700 = {}
    # Getting the type of 'TypeError' (line 960)
    TypeError_1696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 960, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 960)
    TypeError_call_result_1701 = invoke(stypy.reporting.localization.Localization(__file__, 960, 14), TypeError_1696, *[result_mod_1699], **kwargs_1700)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 960, 8), TypeError_call_result_1701, 'raise parameter', BaseException)
    # SSA join for if statement (line 959)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 964):
    
    # Assigning a Name to a Name (line 964):
    # Getting the type of 'False' (line 964)
    False_1702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 964, 20), 'False')
    # Assigning a type to the variable 'specified_out' (line 964)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 964, 4), 'specified_out', False_1702)
    
    # Assigning a Call to a Name (line 965):
    
    # Assigning a Call to a Name (line 965):
    
    # Call to pop(...): (line 965)
    # Processing the call arguments (line 965)
    str_1705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 965, 34), 'str', 'out')
    # Getting the type of 'None' (line 965)
    None_1706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 41), 'None', False)
    # Processing the call keyword arguments (line 965)
    kwargs_1707 = {}
    # Getting the type of 'einsum_kwargs' (line 965)
    einsum_kwargs_1703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 965, 16), 'einsum_kwargs', False)
    # Obtaining the member 'pop' of a type (line 965)
    pop_1704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 965, 16), einsum_kwargs_1703, 'pop')
    # Calling pop(args, kwargs) (line 965)
    pop_call_result_1708 = invoke(stypy.reporting.localization.Localization(__file__, 965, 16), pop_1704, *[str_1705, None_1706], **kwargs_1707)
    
    # Assigning a type to the variable 'out_array' (line 965)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 965, 4), 'out_array', pop_call_result_1708)
    
    # Type idiom detected: calculating its left and rigth part (line 966)
    # Getting the type of 'out_array' (line 966)
    out_array_1709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 4), 'out_array')
    # Getting the type of 'None' (line 966)
    None_1710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 966, 24), 'None')
    
    (may_be_1711, more_types_in_union_1712) = may_not_be_none(out_array_1709, None_1710)

    if may_be_1711:

        if more_types_in_union_1712:
            # Runtime conditional SSA (line 966)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 967):
        
        # Assigning a Name to a Name (line 967):
        # Getting the type of 'True' (line 967)
        True_1713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 967, 24), 'True')
        # Assigning a type to the variable 'specified_out' (line 967)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 967, 8), 'specified_out', True_1713)

        if more_types_in_union_1712:
            # SSA join for if statement (line 966)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Tuple (line 970):
    
    # Assigning a Subscript to a Name (line 970):
    
    # Obtaining the type of the subscript
    int_1714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 4), 'int')
    
    # Call to einsum_path(...): (line 970)
    # Getting the type of 'operands' (line 970)
    operands_1716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 46), 'operands', False)
    # Processing the call keyword arguments (line 970)
    # Getting the type of 'optimize_arg' (line 970)
    optimize_arg_1717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 65), 'optimize_arg', False)
    keyword_1718 = optimize_arg_1717
    # Getting the type of 'True' (line 971)
    True_1719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 57), 'True', False)
    keyword_1720 = True_1719
    kwargs_1721 = {'einsum_call': keyword_1720, 'optimize': keyword_1718}
    # Getting the type of 'einsum_path' (line 970)
    einsum_path_1715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 33), 'einsum_path', False)
    # Calling einsum_path(args, kwargs) (line 970)
    einsum_path_call_result_1722 = invoke(stypy.reporting.localization.Localization(__file__, 970, 33), einsum_path_1715, *[operands_1716], **kwargs_1721)
    
    # Obtaining the member '__getitem__' of a type (line 970)
    getitem___1723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 4), einsum_path_call_result_1722, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 970)
    subscript_call_result_1724 = invoke(stypy.reporting.localization.Localization(__file__, 970, 4), getitem___1723, int_1714)
    
    # Assigning a type to the variable 'tuple_var_assignment_31' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'tuple_var_assignment_31', subscript_call_result_1724)
    
    # Assigning a Subscript to a Name (line 970):
    
    # Obtaining the type of the subscript
    int_1725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 970, 4), 'int')
    
    # Call to einsum_path(...): (line 970)
    # Getting the type of 'operands' (line 970)
    operands_1727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 46), 'operands', False)
    # Processing the call keyword arguments (line 970)
    # Getting the type of 'optimize_arg' (line 970)
    optimize_arg_1728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 65), 'optimize_arg', False)
    keyword_1729 = optimize_arg_1728
    # Getting the type of 'True' (line 971)
    True_1730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 971, 57), 'True', False)
    keyword_1731 = True_1730
    kwargs_1732 = {'einsum_call': keyword_1731, 'optimize': keyword_1729}
    # Getting the type of 'einsum_path' (line 970)
    einsum_path_1726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 33), 'einsum_path', False)
    # Calling einsum_path(args, kwargs) (line 970)
    einsum_path_call_result_1733 = invoke(stypy.reporting.localization.Localization(__file__, 970, 33), einsum_path_1726, *[operands_1727], **kwargs_1732)
    
    # Obtaining the member '__getitem__' of a type (line 970)
    getitem___1734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 970, 4), einsum_path_call_result_1733, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 970)
    subscript_call_result_1735 = invoke(stypy.reporting.localization.Localization(__file__, 970, 4), getitem___1734, int_1725)
    
    # Assigning a type to the variable 'tuple_var_assignment_32' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'tuple_var_assignment_32', subscript_call_result_1735)
    
    # Assigning a Name to a Name (line 970):
    # Getting the type of 'tuple_var_assignment_31' (line 970)
    tuple_var_assignment_31_1736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'tuple_var_assignment_31')
    # Assigning a type to the variable 'operands' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'operands', tuple_var_assignment_31_1736)
    
    # Assigning a Name to a Name (line 970):
    # Getting the type of 'tuple_var_assignment_32' (line 970)
    tuple_var_assignment_32_1737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 970, 4), 'tuple_var_assignment_32')
    # Assigning a type to the variable 'contraction_list' (line 970)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 970, 14), 'contraction_list', tuple_var_assignment_32_1737)
    
    
    # Call to enumerate(...): (line 973)
    # Processing the call arguments (line 973)
    # Getting the type of 'contraction_list' (line 973)
    contraction_list_1739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 38), 'contraction_list', False)
    # Processing the call keyword arguments (line 973)
    kwargs_1740 = {}
    # Getting the type of 'enumerate' (line 973)
    enumerate_1738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 973, 28), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 973)
    enumerate_call_result_1741 = invoke(stypy.reporting.localization.Localization(__file__, 973, 28), enumerate_1738, *[contraction_list_1739], **kwargs_1740)
    
    # Testing the type of a for loop iterable (line 973)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 973, 4), enumerate_call_result_1741)
    # Getting the type of the for loop variable (line 973)
    for_loop_var_1742 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 973, 4), enumerate_call_result_1741)
    # Assigning a type to the variable 'num' (line 973)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 973, 4), 'num', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 973, 4), for_loop_var_1742))
    # Assigning a type to the variable 'contraction' (line 973)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 973, 4), 'contraction', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 973, 4), for_loop_var_1742))
    # SSA begins for a for statement (line 973)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Tuple (line 974):
    
    # Assigning a Subscript to a Name (line 974):
    
    # Obtaining the type of the subscript
    int_1743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 8), 'int')
    # Getting the type of 'contraction' (line 974)
    contraction_1744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 46), 'contraction')
    # Obtaining the member '__getitem__' of a type (line 974)
    getitem___1745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 8), contraction_1744, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 974)
    subscript_call_result_1746 = invoke(stypy.reporting.localization.Localization(__file__, 974, 8), getitem___1745, int_1743)
    
    # Assigning a type to the variable 'tuple_var_assignment_33' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'tuple_var_assignment_33', subscript_call_result_1746)
    
    # Assigning a Subscript to a Name (line 974):
    
    # Obtaining the type of the subscript
    int_1747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 8), 'int')
    # Getting the type of 'contraction' (line 974)
    contraction_1748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 46), 'contraction')
    # Obtaining the member '__getitem__' of a type (line 974)
    getitem___1749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 8), contraction_1748, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 974)
    subscript_call_result_1750 = invoke(stypy.reporting.localization.Localization(__file__, 974, 8), getitem___1749, int_1747)
    
    # Assigning a type to the variable 'tuple_var_assignment_34' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'tuple_var_assignment_34', subscript_call_result_1750)
    
    # Assigning a Subscript to a Name (line 974):
    
    # Obtaining the type of the subscript
    int_1751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 8), 'int')
    # Getting the type of 'contraction' (line 974)
    contraction_1752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 46), 'contraction')
    # Obtaining the member '__getitem__' of a type (line 974)
    getitem___1753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 8), contraction_1752, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 974)
    subscript_call_result_1754 = invoke(stypy.reporting.localization.Localization(__file__, 974, 8), getitem___1753, int_1751)
    
    # Assigning a type to the variable 'tuple_var_assignment_35' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'tuple_var_assignment_35', subscript_call_result_1754)
    
    # Assigning a Subscript to a Name (line 974):
    
    # Obtaining the type of the subscript
    int_1755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 974, 8), 'int')
    # Getting the type of 'contraction' (line 974)
    contraction_1756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 46), 'contraction')
    # Obtaining the member '__getitem__' of a type (line 974)
    getitem___1757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 974, 8), contraction_1756, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 974)
    subscript_call_result_1758 = invoke(stypy.reporting.localization.Localization(__file__, 974, 8), getitem___1757, int_1755)
    
    # Assigning a type to the variable 'tuple_var_assignment_36' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'tuple_var_assignment_36', subscript_call_result_1758)
    
    # Assigning a Name to a Name (line 974):
    # Getting the type of 'tuple_var_assignment_33' (line 974)
    tuple_var_assignment_33_1759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'tuple_var_assignment_33')
    # Assigning a type to the variable 'inds' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'inds', tuple_var_assignment_33_1759)
    
    # Assigning a Name to a Name (line 974):
    # Getting the type of 'tuple_var_assignment_34' (line 974)
    tuple_var_assignment_34_1760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'tuple_var_assignment_34')
    # Assigning a type to the variable 'idx_rm' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 14), 'idx_rm', tuple_var_assignment_34_1760)
    
    # Assigning a Name to a Name (line 974):
    # Getting the type of 'tuple_var_assignment_35' (line 974)
    tuple_var_assignment_35_1761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'tuple_var_assignment_35')
    # Assigning a type to the variable 'einsum_str' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 22), 'einsum_str', tuple_var_assignment_35_1761)
    
    # Assigning a Name to a Name (line 974):
    # Getting the type of 'tuple_var_assignment_36' (line 974)
    tuple_var_assignment_36_1762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 974, 8), 'tuple_var_assignment_36')
    # Assigning a type to the variable 'remaining' (line 974)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 974, 34), 'remaining', tuple_var_assignment_36_1762)
    
    # Assigning a List to a Name (line 975):
    
    # Assigning a List to a Name (line 975):
    
    # Obtaining an instance of the builtin type 'list' (line 975)
    list_1763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 975, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 975)
    
    # Assigning a type to the variable 'tmp_operands' (line 975)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 975, 8), 'tmp_operands', list_1763)
    
    # Getting the type of 'inds' (line 976)
    inds_1764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 976, 17), 'inds')
    # Testing the type of a for loop iterable (line 976)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 976, 8), inds_1764)
    # Getting the type of the for loop variable (line 976)
    for_loop_var_1765 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 976, 8), inds_1764)
    # Assigning a type to the variable 'x' (line 976)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 976, 8), 'x', for_loop_var_1765)
    # SSA begins for a for statement (line 976)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 977)
    # Processing the call arguments (line 977)
    
    # Call to pop(...): (line 977)
    # Processing the call arguments (line 977)
    # Getting the type of 'x' (line 977)
    x_1770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 45), 'x', False)
    # Processing the call keyword arguments (line 977)
    kwargs_1771 = {}
    # Getting the type of 'operands' (line 977)
    operands_1768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 32), 'operands', False)
    # Obtaining the member 'pop' of a type (line 977)
    pop_1769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 977, 32), operands_1768, 'pop')
    # Calling pop(args, kwargs) (line 977)
    pop_call_result_1772 = invoke(stypy.reporting.localization.Localization(__file__, 977, 32), pop_1769, *[x_1770], **kwargs_1771)
    
    # Processing the call keyword arguments (line 977)
    kwargs_1773 = {}
    # Getting the type of 'tmp_operands' (line 977)
    tmp_operands_1766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 977, 12), 'tmp_operands', False)
    # Obtaining the member 'append' of a type (line 977)
    append_1767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 977, 12), tmp_operands_1766, 'append')
    # Calling append(args, kwargs) (line 977)
    append_call_result_1774 = invoke(stypy.reporting.localization.Localization(__file__, 977, 12), append_1767, *[pop_call_result_1772], **kwargs_1773)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'specified_out' (line 980)
    specified_out_1775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 11), 'specified_out')
    
    # Getting the type of 'num' (line 980)
    num_1776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 31), 'num')
    int_1777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 980, 37), 'int')
    # Applying the binary operator '+' (line 980)
    result_add_1778 = python_operator(stypy.reporting.localization.Localization(__file__, 980, 31), '+', num_1776, int_1777)
    
    
    # Call to len(...): (line 980)
    # Processing the call arguments (line 980)
    # Getting the type of 'contraction_list' (line 980)
    contraction_list_1780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 47), 'contraction_list', False)
    # Processing the call keyword arguments (line 980)
    kwargs_1781 = {}
    # Getting the type of 'len' (line 980)
    len_1779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 980, 43), 'len', False)
    # Calling len(args, kwargs) (line 980)
    len_call_result_1782 = invoke(stypy.reporting.localization.Localization(__file__, 980, 43), len_1779, *[contraction_list_1780], **kwargs_1781)
    
    # Applying the binary operator '==' (line 980)
    result_eq_1783 = python_operator(stypy.reporting.localization.Localization(__file__, 980, 30), '==', result_add_1778, len_call_result_1782)
    
    # Applying the binary operator 'and' (line 980)
    result_and_keyword_1784 = python_operator(stypy.reporting.localization.Localization(__file__, 980, 11), 'and', specified_out_1775, result_eq_1783)
    
    # Testing the type of an if condition (line 980)
    if_condition_1785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 980, 8), result_and_keyword_1784)
    # Assigning a type to the variable 'if_condition_1785' (line 980)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 980, 8), 'if_condition_1785', if_condition_1785)
    # SSA begins for if statement (line 980)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 981):
    
    # Assigning a Name to a Subscript (line 981):
    # Getting the type of 'out_array' (line 981)
    out_array_1786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 35), 'out_array')
    # Getting the type of 'einsum_kwargs' (line 981)
    einsum_kwargs_1787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 981, 12), 'einsum_kwargs')
    str_1788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 981, 26), 'str', 'out')
    # Storing an element on a container (line 981)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 981, 12), einsum_kwargs_1787, (str_1788, out_array_1786))
    # SSA join for if statement (line 980)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 984):
    
    # Assigning a Call to a Name (line 984):
    
    # Call to c_einsum(...): (line 984)
    # Processing the call arguments (line 984)
    # Getting the type of 'einsum_str' (line 984)
    einsum_str_1790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 28), 'einsum_str', False)
    # Getting the type of 'tmp_operands' (line 984)
    tmp_operands_1791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 41), 'tmp_operands', False)
    # Processing the call keyword arguments (line 984)
    # Getting the type of 'einsum_kwargs' (line 984)
    einsum_kwargs_1792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 57), 'einsum_kwargs', False)
    kwargs_1793 = {'einsum_kwargs_1792': einsum_kwargs_1792}
    # Getting the type of 'c_einsum' (line 984)
    c_einsum_1789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 984, 19), 'c_einsum', False)
    # Calling c_einsum(args, kwargs) (line 984)
    c_einsum_call_result_1794 = invoke(stypy.reporting.localization.Localization(__file__, 984, 19), c_einsum_1789, *[einsum_str_1790, tmp_operands_1791], **kwargs_1793)
    
    # Assigning a type to the variable 'new_view' (line 984)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 984, 8), 'new_view', c_einsum_call_result_1794)
    
    # Call to append(...): (line 987)
    # Processing the call arguments (line 987)
    # Getting the type of 'new_view' (line 987)
    new_view_1797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 24), 'new_view', False)
    # Processing the call keyword arguments (line 987)
    kwargs_1798 = {}
    # Getting the type of 'operands' (line 987)
    operands_1795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 987, 8), 'operands', False)
    # Obtaining the member 'append' of a type (line 987)
    append_1796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 987, 8), operands_1795, 'append')
    # Calling append(args, kwargs) (line 987)
    append_call_result_1799 = invoke(stypy.reporting.localization.Localization(__file__, 987, 8), append_1796, *[new_view_1797], **kwargs_1798)
    
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 988, 8), module_type_store, 'tmp_operands')
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 988, 8), module_type_store, 'new_view')
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'specified_out' (line 990)
    specified_out_1800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 990, 7), 'specified_out')
    # Testing the type of an if condition (line 990)
    if_condition_1801 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 990, 4), specified_out_1800)
    # Assigning a type to the variable 'if_condition_1801' (line 990)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 990, 4), 'if_condition_1801', if_condition_1801)
    # SSA begins for if statement (line 990)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'out_array' (line 991)
    out_array_1802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 991, 15), 'out_array')
    # Assigning a type to the variable 'stypy_return_type' (line 991)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 991, 8), 'stypy_return_type', out_array_1802)
    # SSA branch for the else part of an if statement (line 990)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining the type of the subscript
    int_1803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 993, 24), 'int')
    # Getting the type of 'operands' (line 993)
    operands_1804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 993, 15), 'operands')
    # Obtaining the member '__getitem__' of a type (line 993)
    getitem___1805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 993, 15), operands_1804, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 993)
    subscript_call_result_1806 = invoke(stypy.reporting.localization.Localization(__file__, 993, 15), getitem___1805, int_1803)
    
    # Assigning a type to the variable 'stypy_return_type' (line 993)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 993, 8), 'stypy_return_type', subscript_call_result_1806)
    # SSA join for if statement (line 990)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'einsum(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'einsum' in the type store
    # Getting the type of 'stypy_return_type' (line 703)
    stypy_return_type_1807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 703, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1807)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'einsum'
    return stypy_return_type_1807

# Assigning a type to the variable 'einsum' (line 703)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 703, 0), 'einsum', einsum)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
