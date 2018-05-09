
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: This is a Python source code generator visitor, that transform an AST into valid source code. It is used to create
3: type annotated programs and type inference programs when their AST is finally created.
4: 
5: Adapted from: http://svn.python.org/view/python/trunk/Demo/parser/unparse.py
6: '''
7: 
8: import sys
9: import ast
10: from cStringIO import StringIO
11: 
12: # Large float and imaginary literals get turned into infinities in the AST.
13: # We unparse those infinities to INFSTR.
14: INFSTR = "1e" + repr(sys.float_info.max_10_exp + 1)
15: 
16: 
17: def interleave(inter, f, seq):
18:     '''
19:     Call f on each item in seq, calling inter() in between.
20:     '''
21:     seq = iter(seq)
22:     try:
23:         f(next(seq))
24:     except StopIteration:
25:         pass
26:     else:
27:         for x in seq:
28:             inter()
29:             f(x)
30: 
31: 
32: class PythonSrcGeneratorVisitor(ast.NodeVisitor):
33:     '''
34:     Methods in this class recursively traverse an AST and
35:     output source code for the abstract syntax; original formatting
36:     is disregarded.
37:     '''
38: 
39:     def __init__(self, tree, verbose=False):
40:         self.output = StringIO()
41:         self.future_imports = []
42:         self._indent = 0
43:         self._indent_str = "    "
44:         self.output.write("")
45:         self.output.flush()
46:         self.tree = tree
47:         self.verbose = verbose
48: 
49:     def generate_code(self):
50:         self.visit(self.tree)
51:         return self.output.getvalue()
52: 
53:     def fill(self, text=""):
54:         '''
55:         Indent a piece of text, according to the current indentation level
56:         '''
57:         if self.verbose:
58:             sys.stdout.write("\n" + self._indent_str * self._indent + text)
59:         self.output.write("\n" + self._indent_str * self._indent + text)
60: 
61:     def write(self, text):
62:         '''
63:         Append a piece of text to the current line.
64:         '''
65:         if self.verbose:
66:             sys.stdout.write(text)
67:         self.output.write(text)
68: 
69:     def enter(self):
70:         '''
71:         Print ':', and increase the indentation.
72:         '''
73:         self.write(":")
74:         self._indent += 1
75: 
76:     def leave(self):
77:         '''
78:         Decrease the indentation level.
79:         '''
80:         self._indent -= 1
81: 
82:     def visit(self, tree):
83:         '''
84:         General visit method, calling the appropriate visit method for type T
85:         '''
86:         if isinstance(tree, list):
87:             for t in tree:
88:                 self.visit(t)
89:             return
90: 
91:         if type(tree) is tuple:
92:             print (tree)
93: 
94:         meth = getattr(self, "visit_" + tree.__class__.__name__)
95:         meth(tree)
96: 
97:     # ############## Unparsing methods ######################
98:     # There should be one method per concrete grammar type #
99:     # Constructors should be grouped by sum type. Ideally, #
100:     # this would follow the order in the grammar, but      #
101:     # currently doesn't.                                   #
102:     # #######################################################
103: 
104:     def visit_Module(self, tree):
105:         for stmt in tree.body:
106:             self.visit(stmt)
107: 
108:     # stmt
109:     def visit_Expr(self, tree):
110:         self.fill()
111:         self.visit(tree.value)
112: 
113:     def visit_Import(self, t):
114:         self.fill("import ")
115:         interleave(lambda: self.write(", "), self.visit, t.names)
116:         self.write("\n")
117: 
118:     def visit_ImportFrom(self, t):
119:         # A from __future__ import may affect unparsing, so record it.
120:         if t.module and t.module == '__future__':
121:             self.future_imports.extend(n.name for n in t.names)
122: 
123:         self.fill("from ")
124:         self.write("." * t.level)
125:         if t.module:
126:             self.write(t.module)
127:         self.write(" import ")
128:         interleave(lambda: self.write(", "), self.visit, t.names)
129:         self.write("\n")
130: 
131:     def visit_Assign(self, t):
132:         self.fill()
133:         for target in t.targets:
134:             self.visit(target)
135:             self.write(" = ")
136:         self.visit(t.value)
137: 
138:     def visit_AugAssign(self, t):
139:         self.fill()
140:         self.visit(t.target)
141:         self.write(" " + self.binop[t.op.__class__.__name__] + "= ")
142:         self.visit(t.value)
143: 
144:     def visit_Return(self, t):
145:         self.fill("return")
146:         if t.value:
147:             self.write(" ")
148:             self.visit(t.value)
149:             # self.write("\n")
150: 
151:     def visit_Pass(self, t):
152:         self.fill("pass")
153: 
154:     def visit_Break(self, t):
155:         self.fill("break")
156: 
157:     def visit_Continue(self, t):
158:         self.fill("continue")
159: 
160:     def visit_Delete(self, t):
161:         self.fill("del ")
162:         interleave(lambda: self.write(", "), self.visit, t.targets)
163: 
164:     def visit_Assert(self, t):
165:         self.fill("assert ")
166:         self.visit(t.test)
167:         if t.msg:
168:             self.write(", ")
169:             self.visit(t.msg)
170: 
171:     def visit_Exec(self, t):
172:         self.fill("exec ")
173:         self.visit(t.body)
174:         if t.globals:
175:             self.write(" in ")
176:             self.visit(t.globals)
177:         if t.locals:
178:             self.write(", ")
179:             self.visit(t.locals)
180: 
181:     def visit_Print(self, t):
182:         self.fill("print ")
183:         do_comma = False
184:         if t.dest:
185:             self.write(">>")
186:             self.visit(t.dest)
187:             do_comma = True
188:         for e in t.values:
189:             if do_comma:
190:                 self.write(", ")
191:             else:
192:                 do_comma = True
193:             self.visit(e)
194:         if not t.nl:
195:             self.write(",")
196: 
197:     def visit_Global(self, t):
198:         self.fill("global ")
199:         interleave(lambda: self.write(", "), self.write, t.names)
200: 
201:     def visit_Yield(self, t):
202:         self.write("(")
203:         self.write("yield")
204:         if t.value:
205:             self.write(" ")
206:             self.visit(t.value)
207:         self.write(")")
208: 
209:     def visit_Raise(self, t):
210:         self.fill('raise ')
211:         if t.type:
212:             self.visit(t.type)
213:         if t.inst:
214:             self.write(", ")
215:             self.visit(t.inst)
216:         if t.tback:
217:             self.write(", ")
218:             self.visit(t.tback)
219: 
220:     def visit_TryExcept(self, t):
221:         self.fill("try")
222:         self.enter()
223:         self.visit(t.body)
224:         self.leave()
225: 
226:         for ex in t.handlers:
227:             self.visit(ex)
228:         if t.orelse:
229:             self.fill("else")
230:             self.enter()
231:             self.visit(t.orelse)
232:             self.leave()
233: 
234:     def visit_TryFinally(self, t):
235:         if len(t.body) == 1 and isinstance(t.body[0], ast.TryExcept):
236:             # try-except-finally
237:             self.visit(t.body)
238:         else:
239:             self.fill("try")
240:             self.enter()
241:             self.visit(t.body)
242:             self.leave()
243: 
244:         self.fill("finally")
245:         self.enter()
246:         self.visit(t.finalbody)
247:         self.leave()
248: 
249:     def visit_ExceptHandler(self, t):
250:         self.fill("except")
251:         if t.type:
252:             self.write(" ")
253:             self.visit(t.type)
254:         if t.name:
255:             self.write(" as ")
256:             self.visit(t.name)
257:         self.enter()
258:         self.visit(t.body)
259:         self.leave()
260: 
261:     def visit_ClassDef(self, t):
262:         self.write("\n")
263:         for deco in t.decorator_list:
264:             self.fill("@")
265:             self.visit(deco)
266:         self.fill("class " + t.name)
267:         if t.bases:
268:             self.write("(")
269:             for a in t.bases:
270:                 self.visit(a)
271:                 self.write(", ")
272:             self.write(")")
273:         self.enter()
274:         self.visit(t.body)
275:         self.leave()
276: 
277:     def visit_FunctionDef(self, t):
278:         self.write("\n")
279:         for deco in t.decorator_list:
280:             self.fill("@")
281:             self.visit(deco)
282:         self.fill("def " + t.name + "(")
283:         self.visit(t.args)
284:         self.write(")")
285:         self.enter()
286:         self.visit(t.body)
287:         self.write("\n")
288:         self.leave()
289: 
290:     def visit_For(self, t):
291:         self.fill("for ")
292:         self.visit(t.target)
293:         self.write(" in ")
294:         self.visit(t.iter)
295:         self.enter()
296:         self.visit(t.body)
297:         self.leave()
298:         if t.orelse:
299:             self.fill("else")
300:             self.enter()
301:             self.visit(t.orelse)
302:             self.leave()
303: 
304:     def visit_If(self, t):
305:         self.write("\n")
306:         self.fill("if ")
307:         self.visit(t.test)
308:         self.enter()
309:         self.visit(t.body)
310:         self.leave()
311:         # collapse nested ifs into equivalent elifs.
312:         while (t.orelse and len(t.orelse) == 1 and
313:                    isinstance(t.orelse[0], ast.If)):
314:             t = t.orelse[0]
315:             self.fill("elif ")
316:             self.visit(t.test)
317:             self.enter()
318:             self.visit(t.body)
319:             self.leave()
320:         # final else
321:         if t.orelse:
322:             self.fill("else")
323:             self.enter()
324:             self.visit(t.orelse)
325:             self.leave()
326:         self.write("\n")
327: 
328:     def visit_While(self, t):
329:         self.fill("while ")
330:         self.visit(t.test)
331:         self.enter()
332:         self.visit(t.body)
333:         self.leave()
334:         if t.orelse:
335:             self.fill("else")
336:             self.enter()
337:             self.visit(t.orelse)
338:             self.leave()
339: 
340:     def visit_With(self, t):
341:         self.fill("with ")
342:         self.visit(t.context_expr)
343:         if t.optional_vars:
344:             self.write(" as ")
345:             self.visit(t.optional_vars)
346:         self.enter()
347:         self.visit(t.body)
348:         self.leave()
349: 
350:     # expr
351:     def visit_Str(self, tree):
352:         # if from __future__ import unicode_literals is in effect,
353:         # then we want to output string literals using a 'b' prefix
354:         # and unicode literals with no prefix.
355:         if "unicode_literals" not in self.future_imports:
356:             self.write(repr(tree.s))
357:         elif isinstance(tree.s, str):
358:             self.write("b" + repr(tree.s))
359:         elif isinstance(tree.s, unicode):
360:             self.write(repr(tree.s).lstrip("u"))
361:         else:
362:             assert False, "shouldn't get here"
363: 
364:     def visit_Name(self, t):
365:         self.write(t.id)
366: 
367:     def visit_Repr(self, t):
368:         self.write("`")
369:         self.visit(t.value)
370:         self.write("`")
371: 
372:     def visit_Num(self, t):
373:         repr_n = repr(t.n)
374:         # Parenthesize negative numbers, to avoid turning (-1)**2 into -1**2.
375:         if repr_n.startswith("-"):
376:             self.write("(")
377:         # Substitute overflowing decimal literal for AST infinities.
378:         self.write(repr_n.replace("inf", INFSTR))
379:         if repr_n.startswith("-"):
380:             self.write(")")
381: 
382:     def visit_List(self, t):
383:         self.write("[")
384:         interleave(lambda: self.write(", "), self.visit, t.elts)
385:         self.write("]")
386: 
387:     def visit_ListComp(self, t):
388:         self.write("[")
389:         self.visit(t.elt)
390:         for gen in t.generators:
391:             self.visit(gen)
392:         self.write("]")
393: 
394:     def visit_GeneratorExp(self, t):
395:         self.write("(")
396:         self.visit(t.elt)
397:         for gen in t.generators:
398:             self.visit(gen)
399:         self.write(")")
400: 
401:     def visit_SetComp(self, t):
402:         self.write("{")
403:         self.visit(t.elt)
404:         for gen in t.generators:
405:             self.visit(gen)
406:         self.write("}")
407: 
408:     def visit_DictComp(self, t):
409:         self.write("{")
410:         self.visit(t.key)
411:         self.write(": ")
412:         self.visit(t.value)
413:         for gen in t.generators:
414:             self.visit(gen)
415:         self.write("}")
416: 
417:     def visit_comprehension(self, t):
418:         self.write(" for ")
419:         self.visit(t.target)
420:         self.write(" in ")
421:         self.visit(t.iter)
422:         for if_clause in t.ifs:
423:             self.write(" if ")
424:             self.visit(if_clause)
425: 
426:     def visit_IfExp(self, t):
427:         self.write("(")
428:         self.visit(t.body)
429:         self.write(" if ")
430:         self.visit(t.test)
431:         self.write(" else ")
432:         self.visit(t.orelse)
433:         self.write(")")
434: 
435:     def visit_Set(self, t):
436:         assert (t.elts)  # should be at least one element
437:         self.write("{")
438:         interleave(lambda: self.write(", "), self.visit, t.elts)
439:         self.write("}")
440: 
441:     def visit_Dict(self, t):
442:         self.write("{")
443: 
444:         def write_pair(pair):
445:             (k, v) = pair
446:             self.visit(k)
447:             self.write(": ")
448:             self.visit(v)
449: 
450:         interleave(lambda: self.write(", "), write_pair, zip(t.keys, t.values))
451:         self.write("}")
452: 
453:     def visit_Tuple(self, t):
454:         self.write("(")
455:         if len(t.elts) == 1:
456:             (elt,) = t.elts
457:             self.visit(elt)
458:             self.write(",")
459:         else:
460:             interleave(lambda: self.write(", "), self.visit, t.elts)
461:         self.write(")")
462: 
463:     unop = {"Invert": "~", "Not": "not", "UAdd": "+", "USub": "-"}
464: 
465:     def visit_UnaryOp(self, t):
466:         self.write("(")
467:         self.write(self.unop[t.op.__class__.__name__])
468:         self.write(" ")
469:         # If we're applying unary minus to a number, parenthesize the number.
470:         # This is necessary: -2147483648 is different from -(2147483648) on
471:         # a 32-bit machine (the first is an int, the second a long), and
472:         # -7j is different from -(7j).  (The first has real part 0.0, the second
473:         # has real part -0.0.)
474:         if isinstance(t.op, ast.USub) and isinstance(t.operand, ast.Num):
475:             self.write("(")
476:             self.visit(t.operand)
477:             self.write(")")
478:         else:
479:             self.visit(t.operand)
480:         self.write(")")
481: 
482:     binop = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "Mod": "%",
483:              "LShift": "<<", "RShift": ">>", "BitOr": "|", "BitXor": "^", "BitAnd": "&",
484:              "FloorDiv": "//", "Pow": "**"}
485: 
486:     def visit_BinOp(self, t):
487:         self.write("(")
488:         self.visit(t.left)
489:         self.write(" " + self.binop[t.op.__class__.__name__] + " ")
490:         self.visit(t.right)
491:         self.write(")")
492: 
493:     cmpops = {"Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
494:               "Is": "is", "IsNot": "is not", "In": "in", "NotIn": "not in"}
495: 
496:     def visit_Compare(self, t):
497:         self.write("(")
498:         self.visit(t.left)
499:         for o, e in zip(t.ops, t.comparators):
500:             self.write(" " + self.cmpops[o.__class__.__name__] + " ")
501:             self.visit(e)
502:         self.write(")")
503: 
504:     boolops = {ast.And: 'and', ast.Or: 'or'}
505: 
506:     def visit_BoolOp(self, t):
507:         self.write("(")
508:         s = " %s " % self.boolops[t.op.__class__]
509:         interleave(lambda: self.write(s), self.visit, t.values)
510:         self.write(")")
511: 
512:     def visit_Attribute(self, t):
513:         self.visit(t.value)
514:         # Special case: 3.__abs__() is a syntax error, so if t.value
515:         # is an integer literal then we need to either parenthesize
516:         # it or add an extra space to get 3 .__abs__().
517:         if isinstance(t.value, ast.Num) and isinstance(t.value.n, int):
518:             self.write(" ")
519:         self.write(".")
520:         self.write(t.attr)
521: 
522:     def visit_Call(self, t):
523:         self.visit(t.func)
524:         self.write("(")
525:         comma = False
526:         for e in t.args:
527:             if comma:
528:                 self.write(", ")
529:             else:
530:                 comma = True
531:             self.visit(e)
532:         for e in t.keywords:
533:             if comma:
534:                 self.write(", ")
535:             else:
536:                 comma = True
537:             self.visit(e)
538:         if t.starargs:
539:             if comma:
540:                 self.write(", ")
541:             else:
542:                 comma = True
543:             self.write("*")
544:             self.visit(t.starargs)
545:         if t.kwargs:
546:             if comma:
547:                 self.write(", ")
548:             else:
549:                 comma = True
550:             self.write("**")
551:             self.visit(t.kwargs)
552:         self.write(")")
553: 
554:     def visit_Subscript(self, t):
555:         self.visit(t.value)
556:         self.write("[")
557:         self.visit(t.slice)
558:         self.write("]")
559: 
560:     # slice
561:     def visit_Ellipsis(self, t):
562:         self.write("...")
563: 
564:     def visit_Index(self, t):
565:         self.visit(t.value)
566: 
567:     def visit_Slice(self, t):
568:         if t.lower:
569:             self.visit(t.lower)
570:         self.write(":")
571:         if t.upper:
572:             self.visit(t.upper)
573:         if t.step:
574:             self.write(":")
575:             self.visit(t.step)
576: 
577:     def visit_ExtSlice(self, t):
578:         interleave(lambda: self.write(', '), self.visit, t.dims)
579: 
580:     # others
581:     def visit_arguments(self, t):
582:         first = True
583:         # normal arguments
584:         defaults = [None] * (len(t.args) - len(t.defaults)) + t.defaults
585:         for a, d in zip(t.args, defaults):
586:             if first:
587:                 first = False
588:             else:
589:                 self.write(", ")
590:             self.visit(a),
591:             if d:
592:                 self.write("=")
593:                 self.visit(d)
594: 
595:         # varargs
596:         if t.vararg:
597:             if first:
598:                 first = False
599:             else:
600:                 self.write(", ")
601:             self.write("*")
602:             self.write(t.vararg)
603: 
604:         # kwargs
605:         if t.kwarg:
606:             if first:
607:                 first = False
608:             else:
609:                 self.write(", ")
610:             self.write("**" + t.kwarg)
611: 
612:     def visit_keyword(self, t):
613:         self.write(t.arg)
614:         self.write("=")
615:         self.visit(t.value)
616: 
617:     def visit_Lambda(self, t):
618:         self.write("(")
619:         self.write("lambda ")
620:         self.visit(t.args)
621:         self.write(": ")
622:         self.visit(t.body)
623:         self.write(")")
624: 
625:     def visit_alias(self, t):
626:         self.write(t.name)
627:         if t.asname:
628:             self.write(" as " + t.asname)
629: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_2299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', '\nThis is a Python source code generator visitor, that transform an AST into valid source code. It is used to create\ntype annotated programs and type inference programs when their AST is finally created.\n\nAdapted from: http://svn.python.org/view/python/trunk/Demo/parser/unparse.py\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import sys' statement (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import ast' statement (line 9)
import ast

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'ast', ast, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from cStringIO import StringIO' statement (line 10)
try:
    from cStringIO import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'cStringIO', None, module_type_store, ['StringIO'], [StringIO])


# Assigning a BinOp to a Name (line 14):

# Assigning a BinOp to a Name (line 14):
str_2300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 9), 'str', '1e')

# Call to repr(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'sys' (line 14)
sys_2302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 21), 'sys', False)
# Obtaining the member 'float_info' of a type (line 14)
float_info_2303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 21), sys_2302, 'float_info')
# Obtaining the member 'max_10_exp' of a type (line 14)
max_10_exp_2304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 21), float_info_2303, 'max_10_exp')
int_2305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 49), 'int')
# Applying the binary operator '+' (line 14)
result_add_2306 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 21), '+', max_10_exp_2304, int_2305)

# Processing the call keyword arguments (line 14)
kwargs_2307 = {}
# Getting the type of 'repr' (line 14)
repr_2301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'repr', False)
# Calling repr(args, kwargs) (line 14)
repr_call_result_2308 = invoke(stypy.reporting.localization.Localization(__file__, 14, 16), repr_2301, *[result_add_2306], **kwargs_2307)

# Applying the binary operator '+' (line 14)
result_add_2309 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '+', str_2300, repr_call_result_2308)

# Assigning a type to the variable 'INFSTR' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'INFSTR', result_add_2309)

@norecursion
def interleave(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'interleave'
    module_type_store = module_type_store.open_function_context('interleave', 17, 0, False)
    
    # Passed parameters checking function
    interleave.stypy_localization = localization
    interleave.stypy_type_of_self = None
    interleave.stypy_type_store = module_type_store
    interleave.stypy_function_name = 'interleave'
    interleave.stypy_param_names_list = ['inter', 'f', 'seq']
    interleave.stypy_varargs_param_name = None
    interleave.stypy_kwargs_param_name = None
    interleave.stypy_call_defaults = defaults
    interleave.stypy_call_varargs = varargs
    interleave.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'interleave', ['inter', 'f', 'seq'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'interleave', localization, ['inter', 'f', 'seq'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'interleave(...)' code ##################

    str_2310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'str', '\n    Call f on each item in seq, calling inter() in between.\n    ')
    
    # Assigning a Call to a Name (line 21):
    
    # Assigning a Call to a Name (line 21):
    
    # Call to iter(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'seq' (line 21)
    seq_2312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'seq', False)
    # Processing the call keyword arguments (line 21)
    kwargs_2313 = {}
    # Getting the type of 'iter' (line 21)
    iter_2311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'iter', False)
    # Calling iter(args, kwargs) (line 21)
    iter_call_result_2314 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), iter_2311, *[seq_2312], **kwargs_2313)
    
    # Assigning a type to the variable 'seq' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'seq', iter_call_result_2314)
    
    
    # SSA begins for try-except statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to f(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to next(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'seq' (line 23)
    seq_2317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'seq', False)
    # Processing the call keyword arguments (line 23)
    kwargs_2318 = {}
    # Getting the type of 'next' (line 23)
    next_2316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'next', False)
    # Calling next(args, kwargs) (line 23)
    next_call_result_2319 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), next_2316, *[seq_2317], **kwargs_2318)
    
    # Processing the call keyword arguments (line 23)
    kwargs_2320 = {}
    # Getting the type of 'f' (line 23)
    f_2315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'f', False)
    # Calling f(args, kwargs) (line 23)
    f_call_result_2321 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), f_2315, *[next_call_result_2319], **kwargs_2320)
    
    # SSA branch for the except part of a try statement (line 22)
    # SSA branch for the except 'StopIteration' branch of a try statement (line 22)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA branch for the else branch of a try statement (line 22)
    module_type_store.open_ssa_branch('except else')
    
    # Getting the type of 'seq' (line 27)
    seq_2322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'seq')
    # Assigning a type to the variable 'seq_2322' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'seq_2322', seq_2322)
    # Testing if the for loop is going to be iterated (line 27)
    # Testing the type of a for loop iterable (line 27)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 8), seq_2322)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 27, 8), seq_2322):
        # Getting the type of the for loop variable (line 27)
        for_loop_var_2323 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 8), seq_2322)
        # Assigning a type to the variable 'x' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'x', for_loop_var_2323)
        # SSA begins for a for statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to inter(...): (line 28)
        # Processing the call keyword arguments (line 28)
        kwargs_2325 = {}
        # Getting the type of 'inter' (line 28)
        inter_2324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'inter', False)
        # Calling inter(args, kwargs) (line 28)
        inter_call_result_2326 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), inter_2324, *[], **kwargs_2325)
        
        
        # Call to f(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'x' (line 29)
        x_2328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'x', False)
        # Processing the call keyword arguments (line 29)
        kwargs_2329 = {}
        # Getting the type of 'f' (line 29)
        f_2327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'f', False)
        # Calling f(args, kwargs) (line 29)
        f_call_result_2330 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), f_2327, *[x_2328], **kwargs_2329)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # SSA join for try-except statement (line 22)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'interleave(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'interleave' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_2331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2331)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'interleave'
    return stypy_return_type_2331

# Assigning a type to the variable 'interleave' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'interleave', interleave)
# Declaration of the 'PythonSrcGeneratorVisitor' class
# Getting the type of 'ast' (line 32)
ast_2332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 32), 'ast')
# Obtaining the member 'NodeVisitor' of a type (line 32)
NodeVisitor_2333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 32), ast_2332, 'NodeVisitor')

class PythonSrcGeneratorVisitor(NodeVisitor_2333, ):
    str_2334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, (-1)), 'str', '\n    Methods in this class recursively traverse an AST and\n    output source code for the abstract syntax; original formatting\n    is disregarded.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 39)
        False_2335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 37), 'False')
        defaults = [False_2335]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.__init__', ['tree', 'verbose'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['tree', 'verbose'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 40):
        
        # Assigning a Call to a Attribute (line 40):
        
        # Call to StringIO(...): (line 40)
        # Processing the call keyword arguments (line 40)
        kwargs_2337 = {}
        # Getting the type of 'StringIO' (line 40)
        StringIO_2336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 40)
        StringIO_call_result_2338 = invoke(stypy.reporting.localization.Localization(__file__, 40, 22), StringIO_2336, *[], **kwargs_2337)
        
        # Getting the type of 'self' (line 40)
        self_2339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'output' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_2339, 'output', StringIO_call_result_2338)
        
        # Assigning a List to a Attribute (line 41):
        
        # Assigning a List to a Attribute (line 41):
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_2340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        
        # Getting the type of 'self' (line 41)
        self_2341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'future_imports' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_2341, 'future_imports', list_2340)
        
        # Assigning a Num to a Attribute (line 42):
        
        # Assigning a Num to a Attribute (line 42):
        int_2342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 23), 'int')
        # Getting the type of 'self' (line 42)
        self_2343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member '_indent' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_2343, '_indent', int_2342)
        
        # Assigning a Str to a Attribute (line 43):
        
        # Assigning a Str to a Attribute (line 43):
        str_2344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 27), 'str', '    ')
        # Getting the type of 'self' (line 43)
        self_2345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member '_indent_str' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_2345, '_indent_str', str_2344)
        
        # Call to write(...): (line 44)
        # Processing the call arguments (line 44)
        str_2349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 26), 'str', '')
        # Processing the call keyword arguments (line 44)
        kwargs_2350 = {}
        # Getting the type of 'self' (line 44)
        self_2346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 44)
        output_2347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_2346, 'output')
        # Obtaining the member 'write' of a type (line 44)
        write_2348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), output_2347, 'write')
        # Calling write(args, kwargs) (line 44)
        write_call_result_2351 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), write_2348, *[str_2349], **kwargs_2350)
        
        
        # Call to flush(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_2355 = {}
        # Getting the type of 'self' (line 45)
        self_2352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 45)
        output_2353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_2352, 'output')
        # Obtaining the member 'flush' of a type (line 45)
        flush_2354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), output_2353, 'flush')
        # Calling flush(args, kwargs) (line 45)
        flush_call_result_2356 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), flush_2354, *[], **kwargs_2355)
        
        
        # Assigning a Name to a Attribute (line 46):
        
        # Assigning a Name to a Attribute (line 46):
        # Getting the type of 'tree' (line 46)
        tree_2357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'tree')
        # Getting the type of 'self' (line 46)
        self_2358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'tree' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_2358, 'tree', tree_2357)
        
        # Assigning a Name to a Attribute (line 47):
        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'verbose' (line 47)
        verbose_2359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'verbose')
        # Getting the type of 'self' (line 47)
        self_2360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member 'verbose' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_2360, 'verbose', verbose_2359)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def generate_code(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_code'
        module_type_store = module_type_store.open_function_context('generate_code', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.generate_code.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.generate_code.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.generate_code.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.generate_code.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.generate_code')
        PythonSrcGeneratorVisitor.generate_code.__dict__.__setitem__('stypy_param_names_list', [])
        PythonSrcGeneratorVisitor.generate_code.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.generate_code.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.generate_code.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.generate_code.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.generate_code.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.generate_code.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.generate_code', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_code', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_code(...)' code ##################

        
        # Call to visit(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'self' (line 50)
        self_2363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'self', False)
        # Obtaining the member 'tree' of a type (line 50)
        tree_2364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 19), self_2363, 'tree')
        # Processing the call keyword arguments (line 50)
        kwargs_2365 = {}
        # Getting the type of 'self' (line 50)
        self_2361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 50)
        visit_2362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_2361, 'visit')
        # Calling visit(args, kwargs) (line 50)
        visit_call_result_2366 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), visit_2362, *[tree_2364], **kwargs_2365)
        
        
        # Call to getvalue(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_2370 = {}
        # Getting the type of 'self' (line 51)
        self_2367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'self', False)
        # Obtaining the member 'output' of a type (line 51)
        output_2368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), self_2367, 'output')
        # Obtaining the member 'getvalue' of a type (line 51)
        getvalue_2369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), output_2368, 'getvalue')
        # Calling getvalue(args, kwargs) (line 51)
        getvalue_call_result_2371 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), getvalue_2369, *[], **kwargs_2370)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', getvalue_call_result_2371)
        
        # ################# End of 'generate_code(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_code' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_2372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2372)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_code'
        return stypy_return_type_2372


    @norecursion
    def fill(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_2373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'str', '')
        defaults = [str_2373]
        # Create a new context for function 'fill'
        module_type_store = module_type_store.open_function_context('fill', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.fill.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.fill.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.fill.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.fill.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.fill')
        PythonSrcGeneratorVisitor.fill.__dict__.__setitem__('stypy_param_names_list', ['text'])
        PythonSrcGeneratorVisitor.fill.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.fill.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.fill.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.fill.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.fill.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.fill.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.fill', ['text'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'fill', localization, ['text'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'fill(...)' code ##################

        str_2374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n        Indent a piece of text, according to the current indentation level\n        ')
        # Getting the type of 'self' (line 57)
        self_2375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'self')
        # Obtaining the member 'verbose' of a type (line 57)
        verbose_2376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), self_2375, 'verbose')
        # Testing if the type of an if condition is none (line 57)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 8), verbose_2376):
            pass
        else:
            
            # Testing the type of an if condition (line 57)
            if_condition_2377 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 8), verbose_2376)
            # Assigning a type to the variable 'if_condition_2377' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'if_condition_2377', if_condition_2377)
            # SSA begins for if statement (line 57)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 58)
            # Processing the call arguments (line 58)
            str_2381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 29), 'str', '\n')
            # Getting the type of 'self' (line 58)
            self_2382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 36), 'self', False)
            # Obtaining the member '_indent_str' of a type (line 58)
            _indent_str_2383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 36), self_2382, '_indent_str')
            # Getting the type of 'self' (line 58)
            self_2384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 55), 'self', False)
            # Obtaining the member '_indent' of a type (line 58)
            _indent_2385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 55), self_2384, '_indent')
            # Applying the binary operator '*' (line 58)
            result_mul_2386 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 36), '*', _indent_str_2383, _indent_2385)
            
            # Applying the binary operator '+' (line 58)
            result_add_2387 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 29), '+', str_2381, result_mul_2386)
            
            # Getting the type of 'text' (line 58)
            text_2388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 70), 'text', False)
            # Applying the binary operator '+' (line 58)
            result_add_2389 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 68), '+', result_add_2387, text_2388)
            
            # Processing the call keyword arguments (line 58)
            kwargs_2390 = {}
            # Getting the type of 'sys' (line 58)
            sys_2378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'sys', False)
            # Obtaining the member 'stdout' of a type (line 58)
            stdout_2379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), sys_2378, 'stdout')
            # Obtaining the member 'write' of a type (line 58)
            write_2380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), stdout_2379, 'write')
            # Calling write(args, kwargs) (line 58)
            write_call_result_2391 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), write_2380, *[result_add_2389], **kwargs_2390)
            
            # SSA join for if statement (line 57)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 59)
        # Processing the call arguments (line 59)
        str_2395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 26), 'str', '\n')
        # Getting the type of 'self' (line 59)
        self_2396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'self', False)
        # Obtaining the member '_indent_str' of a type (line 59)
        _indent_str_2397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 33), self_2396, '_indent_str')
        # Getting the type of 'self' (line 59)
        self_2398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 52), 'self', False)
        # Obtaining the member '_indent' of a type (line 59)
        _indent_2399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 52), self_2398, '_indent')
        # Applying the binary operator '*' (line 59)
        result_mul_2400 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 33), '*', _indent_str_2397, _indent_2399)
        
        # Applying the binary operator '+' (line 59)
        result_add_2401 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 26), '+', str_2395, result_mul_2400)
        
        # Getting the type of 'text' (line 59)
        text_2402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 67), 'text', False)
        # Applying the binary operator '+' (line 59)
        result_add_2403 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 65), '+', result_add_2401, text_2402)
        
        # Processing the call keyword arguments (line 59)
        kwargs_2404 = {}
        # Getting the type of 'self' (line 59)
        self_2392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 59)
        output_2393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_2392, 'output')
        # Obtaining the member 'write' of a type (line 59)
        write_2394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), output_2393, 'write')
        # Calling write(args, kwargs) (line 59)
        write_call_result_2405 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), write_2394, *[result_add_2403], **kwargs_2404)
        
        
        # ################# End of 'fill(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fill' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_2406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2406)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fill'
        return stypy_return_type_2406


    @norecursion
    def write(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write'
        module_type_store = module_type_store.open_function_context('write', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.write.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.write.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.write.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.write.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.write')
        PythonSrcGeneratorVisitor.write.__dict__.__setitem__('stypy_param_names_list', ['text'])
        PythonSrcGeneratorVisitor.write.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.write.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.write.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.write.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.write.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.write.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.write', ['text'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write', localization, ['text'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write(...)' code ##################

        str_2407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', '\n        Append a piece of text to the current line.\n        ')
        # Getting the type of 'self' (line 65)
        self_2408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'self')
        # Obtaining the member 'verbose' of a type (line 65)
        verbose_2409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), self_2408, 'verbose')
        # Testing if the type of an if condition is none (line 65)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 65, 8), verbose_2409):
            pass
        else:
            
            # Testing the type of an if condition (line 65)
            if_condition_2410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 8), verbose_2409)
            # Assigning a type to the variable 'if_condition_2410' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'if_condition_2410', if_condition_2410)
            # SSA begins for if statement (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'text' (line 66)
            text_2414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'text', False)
            # Processing the call keyword arguments (line 66)
            kwargs_2415 = {}
            # Getting the type of 'sys' (line 66)
            sys_2411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'sys', False)
            # Obtaining the member 'stdout' of a type (line 66)
            stdout_2412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), sys_2411, 'stdout')
            # Obtaining the member 'write' of a type (line 66)
            write_2413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), stdout_2412, 'write')
            # Calling write(args, kwargs) (line 66)
            write_call_result_2416 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), write_2413, *[text_2414], **kwargs_2415)
            
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'text' (line 67)
        text_2420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'text', False)
        # Processing the call keyword arguments (line 67)
        kwargs_2421 = {}
        # Getting the type of 'self' (line 67)
        self_2417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 67)
        output_2418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_2417, 'output')
        # Obtaining the member 'write' of a type (line 67)
        write_2419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), output_2418, 'write')
        # Calling write(args, kwargs) (line 67)
        write_call_result_2422 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), write_2419, *[text_2420], **kwargs_2421)
        
        
        # ################# End of 'write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_2423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2423)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write'
        return stypy_return_type_2423


    @norecursion
    def enter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'enter'
        module_type_store = module_type_store.open_function_context('enter', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.enter.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.enter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.enter.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.enter.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.enter')
        PythonSrcGeneratorVisitor.enter.__dict__.__setitem__('stypy_param_names_list', [])
        PythonSrcGeneratorVisitor.enter.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.enter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.enter.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.enter.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.enter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.enter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.enter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'enter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'enter(...)' code ##################

        str_2424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, (-1)), 'str', "\n        Print ':', and increase the indentation.\n        ")
        
        # Call to write(...): (line 73)
        # Processing the call arguments (line 73)
        str_2427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'str', ':')
        # Processing the call keyword arguments (line 73)
        kwargs_2428 = {}
        # Getting the type of 'self' (line 73)
        self_2425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 73)
        write_2426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_2425, 'write')
        # Calling write(args, kwargs) (line 73)
        write_call_result_2429 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), write_2426, *[str_2427], **kwargs_2428)
        
        
        # Getting the type of 'self' (line 74)
        self_2430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Obtaining the member '_indent' of a type (line 74)
        _indent_2431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_2430, '_indent')
        int_2432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'int')
        # Applying the binary operator '+=' (line 74)
        result_iadd_2433 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 8), '+=', _indent_2431, int_2432)
        # Getting the type of 'self' (line 74)
        self_2434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member '_indent' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_2434, '_indent', result_iadd_2433)
        
        
        # ################# End of 'enter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'enter' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_2435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2435)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'enter'
        return stypy_return_type_2435


    @norecursion
    def leave(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'leave'
        module_type_store = module_type_store.open_function_context('leave', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.leave.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.leave.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.leave.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.leave.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.leave')
        PythonSrcGeneratorVisitor.leave.__dict__.__setitem__('stypy_param_names_list', [])
        PythonSrcGeneratorVisitor.leave.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.leave.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.leave.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.leave.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.leave.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.leave.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.leave', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'leave', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'leave(...)' code ##################

        str_2436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, (-1)), 'str', '\n        Decrease the indentation level.\n        ')
        
        # Getting the type of 'self' (line 80)
        self_2437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self')
        # Obtaining the member '_indent' of a type (line 80)
        _indent_2438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_2437, '_indent')
        int_2439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 24), 'int')
        # Applying the binary operator '-=' (line 80)
        result_isub_2440 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 8), '-=', _indent_2438, int_2439)
        # Getting the type of 'self' (line 80)
        self_2441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self')
        # Setting the type of the member '_indent' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_2441, '_indent', result_isub_2440)
        
        
        # ################# End of 'leave(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'leave' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_2442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2442)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'leave'
        return stypy_return_type_2442


    @norecursion
    def visit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit'
        module_type_store = module_type_store.open_function_context('visit', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit')
        PythonSrcGeneratorVisitor.visit.__dict__.__setitem__('stypy_param_names_list', ['tree'])
        PythonSrcGeneratorVisitor.visit.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit', ['tree'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit', localization, ['tree'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit(...)' code ##################

        str_2443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'str', '\n        General visit method, calling the appropriate visit method for type T\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 86)
        # Getting the type of 'list' (line 86)
        list_2444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 28), 'list')
        # Getting the type of 'tree' (line 86)
        tree_2445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 22), 'tree')
        
        (may_be_2446, more_types_in_union_2447) = may_be_subtype(list_2444, tree_2445)

        if may_be_2446:

            if more_types_in_union_2447:
                # Runtime conditional SSA (line 86)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'tree' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tree', remove_not_subtype_from_union(tree_2445, list))
            
            # Getting the type of 'tree' (line 87)
            tree_2448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 21), 'tree')
            # Assigning a type to the variable 'tree_2448' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'tree_2448', tree_2448)
            # Testing if the for loop is going to be iterated (line 87)
            # Testing the type of a for loop iterable (line 87)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 12), tree_2448)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 87, 12), tree_2448):
                # Getting the type of the for loop variable (line 87)
                for_loop_var_2449 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 12), tree_2448)
                # Assigning a type to the variable 't' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 't', for_loop_var_2449)
                # SSA begins for a for statement (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to visit(...): (line 88)
                # Processing the call arguments (line 88)
                # Getting the type of 't' (line 88)
                t_2452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 't', False)
                # Processing the call keyword arguments (line 88)
                kwargs_2453 = {}
                # Getting the type of 'self' (line 88)
                self_2450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'self', False)
                # Obtaining the member 'visit' of a type (line 88)
                visit_2451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 16), self_2450, 'visit')
                # Calling visit(args, kwargs) (line 88)
                visit_call_result_2454 = invoke(stypy.reporting.localization.Localization(__file__, 88, 16), visit_2451, *[t_2452], **kwargs_2453)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Assigning a type to the variable 'stypy_return_type' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_2447:
                # SSA join for if statement (line 86)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 91)
        # Getting the type of 'tree' (line 91)
        tree_2455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'tree')
        # Getting the type of 'tuple' (line 91)
        tuple_2456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'tuple')
        
        (may_be_2457, more_types_in_union_2458) = may_be_type(tree_2455, tuple_2456)

        if may_be_2457:

            if more_types_in_union_2458:
                # Runtime conditional SSA (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'tree' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tree', tuple_2456())
            # Getting the type of 'tree' (line 92)
            tree_2459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'tree')

            if more_types_in_union_2458:
                # SSA join for if statement (line 91)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to getattr(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'self' (line 94)
        self_2461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'self', False)
        str_2462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 29), 'str', 'visit_')
        # Getting the type of 'tree' (line 94)
        tree_2463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 40), 'tree', False)
        # Obtaining the member '__class__' of a type (line 94)
        class___2464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 40), tree_2463, '__class__')
        # Obtaining the member '__name__' of a type (line 94)
        name___2465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 40), class___2464, '__name__')
        # Applying the binary operator '+' (line 94)
        result_add_2466 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 29), '+', str_2462, name___2465)
        
        # Processing the call keyword arguments (line 94)
        kwargs_2467 = {}
        # Getting the type of 'getattr' (line 94)
        getattr_2460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 94)
        getattr_call_result_2468 = invoke(stypy.reporting.localization.Localization(__file__, 94, 15), getattr_2460, *[self_2461, result_add_2466], **kwargs_2467)
        
        # Assigning a type to the variable 'meth' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'meth', getattr_call_result_2468)
        
        # Call to meth(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'tree' (line 95)
        tree_2470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'tree', False)
        # Processing the call keyword arguments (line 95)
        kwargs_2471 = {}
        # Getting the type of 'meth' (line 95)
        meth_2469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'meth', False)
        # Calling meth(args, kwargs) (line 95)
        meth_call_result_2472 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), meth_2469, *[tree_2470], **kwargs_2471)
        
        
        # ################# End of 'visit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_2473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2473)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit'
        return stypy_return_type_2473


    @norecursion
    def visit_Module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Module'
        module_type_store = module_type_store.open_function_context('visit_Module', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Module')
        PythonSrcGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_param_names_list', ['tree'])
        PythonSrcGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Module.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Module', ['tree'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Module', localization, ['tree'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Module(...)' code ##################

        
        # Getting the type of 'tree' (line 105)
        tree_2474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'tree')
        # Obtaining the member 'body' of a type (line 105)
        body_2475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 20), tree_2474, 'body')
        # Assigning a type to the variable 'body_2475' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'body_2475', body_2475)
        # Testing if the for loop is going to be iterated (line 105)
        # Testing the type of a for loop iterable (line 105)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 8), body_2475)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 105, 8), body_2475):
            # Getting the type of the for loop variable (line 105)
            for_loop_var_2476 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 8), body_2475)
            # Assigning a type to the variable 'stmt' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stmt', for_loop_var_2476)
            # SSA begins for a for statement (line 105)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 106)
            # Processing the call arguments (line 106)
            # Getting the type of 'stmt' (line 106)
            stmt_2479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'stmt', False)
            # Processing the call keyword arguments (line 106)
            kwargs_2480 = {}
            # Getting the type of 'self' (line 106)
            self_2477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 106)
            visit_2478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), self_2477, 'visit')
            # Calling visit(args, kwargs) (line 106)
            visit_call_result_2481 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), visit_2478, *[stmt_2479], **kwargs_2480)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'visit_Module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Module' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_2482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2482)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Module'
        return stypy_return_type_2482


    @norecursion
    def visit_Expr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Expr'
        module_type_store = module_type_store.open_function_context('visit_Expr', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Expr.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Expr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Expr.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Expr.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Expr')
        PythonSrcGeneratorVisitor.visit_Expr.__dict__.__setitem__('stypy_param_names_list', ['tree'])
        PythonSrcGeneratorVisitor.visit_Expr.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Expr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Expr.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Expr.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Expr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Expr.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Expr', ['tree'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Expr', localization, ['tree'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Expr(...)' code ##################

        
        # Call to fill(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_2485 = {}
        # Getting the type of 'self' (line 110)
        self_2483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 110)
        fill_2484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_2483, 'fill')
        # Calling fill(args, kwargs) (line 110)
        fill_call_result_2486 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), fill_2484, *[], **kwargs_2485)
        
        
        # Call to visit(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'tree' (line 111)
        tree_2489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'tree', False)
        # Obtaining the member 'value' of a type (line 111)
        value_2490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 19), tree_2489, 'value')
        # Processing the call keyword arguments (line 111)
        kwargs_2491 = {}
        # Getting the type of 'self' (line 111)
        self_2487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 111)
        visit_2488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_2487, 'visit')
        # Calling visit(args, kwargs) (line 111)
        visit_call_result_2492 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), visit_2488, *[value_2490], **kwargs_2491)
        
        
        # ################# End of 'visit_Expr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Expr' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_2493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2493)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Expr'
        return stypy_return_type_2493


    @norecursion
    def visit_Import(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Import'
        module_type_store = module_type_store.open_function_context('visit_Import', 113, 4, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Import.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Import.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Import.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Import.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Import')
        PythonSrcGeneratorVisitor.visit_Import.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Import.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Import.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Import.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Import.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Import.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Import.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Import', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Import', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Import(...)' code ##################

        
        # Call to fill(...): (line 114)
        # Processing the call arguments (line 114)
        str_2496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 18), 'str', 'import ')
        # Processing the call keyword arguments (line 114)
        kwargs_2497 = {}
        # Getting the type of 'self' (line 114)
        self_2494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 114)
        fill_2495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_2494, 'fill')
        # Calling fill(args, kwargs) (line 114)
        fill_call_result_2498 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), fill_2495, *[str_2496], **kwargs_2497)
        
        
        # Call to interleave(...): (line 115)
        # Processing the call arguments (line 115)

        @norecursion
        def _stypy_temp_lambda_5(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_5'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_5', 115, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_5.stypy_localization = localization
            _stypy_temp_lambda_5.stypy_type_of_self = None
            _stypy_temp_lambda_5.stypy_type_store = module_type_store
            _stypy_temp_lambda_5.stypy_function_name = '_stypy_temp_lambda_5'
            _stypy_temp_lambda_5.stypy_param_names_list = []
            _stypy_temp_lambda_5.stypy_varargs_param_name = None
            _stypy_temp_lambda_5.stypy_kwargs_param_name = None
            _stypy_temp_lambda_5.stypy_call_defaults = defaults
            _stypy_temp_lambda_5.stypy_call_varargs = varargs
            _stypy_temp_lambda_5.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_5', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_5', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 115)
            # Processing the call arguments (line 115)
            str_2502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 38), 'str', ', ')
            # Processing the call keyword arguments (line 115)
            kwargs_2503 = {}
            # Getting the type of 'self' (line 115)
            self_2500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 115)
            write_2501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 27), self_2500, 'write')
            # Calling write(args, kwargs) (line 115)
            write_call_result_2504 = invoke(stypy.reporting.localization.Localization(__file__, 115, 27), write_2501, *[str_2502], **kwargs_2503)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'stypy_return_type', write_call_result_2504)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_5' in the type store
            # Getting the type of 'stypy_return_type' (line 115)
            stypy_return_type_2505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_2505)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_5'
            return stypy_return_type_2505

        # Assigning a type to the variable '_stypy_temp_lambda_5' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), '_stypy_temp_lambda_5', _stypy_temp_lambda_5)
        # Getting the type of '_stypy_temp_lambda_5' (line 115)
        _stypy_temp_lambda_5_2506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), '_stypy_temp_lambda_5')
        # Getting the type of 'self' (line 115)
        self_2507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 115)
        visit_2508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 45), self_2507, 'visit')
        # Getting the type of 't' (line 115)
        t_2509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 57), 't', False)
        # Obtaining the member 'names' of a type (line 115)
        names_2510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 57), t_2509, 'names')
        # Processing the call keyword arguments (line 115)
        kwargs_2511 = {}
        # Getting the type of 'interleave' (line 115)
        interleave_2499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 115)
        interleave_call_result_2512 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), interleave_2499, *[_stypy_temp_lambda_5_2506, visit_2508, names_2510], **kwargs_2511)
        
        
        # Call to write(...): (line 116)
        # Processing the call arguments (line 116)
        str_2515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'str', '\n')
        # Processing the call keyword arguments (line 116)
        kwargs_2516 = {}
        # Getting the type of 'self' (line 116)
        self_2513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 116)
        write_2514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_2513, 'write')
        # Calling write(args, kwargs) (line 116)
        write_call_result_2517 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), write_2514, *[str_2515], **kwargs_2516)
        
        
        # ################# End of 'visit_Import(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Import' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_2518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2518)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Import'
        return stypy_return_type_2518


    @norecursion
    def visit_ImportFrom(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ImportFrom'
        module_type_store = module_type_store.open_function_context('visit_ImportFrom', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_ImportFrom.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_ImportFrom.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_ImportFrom.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_ImportFrom.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_ImportFrom')
        PythonSrcGeneratorVisitor.visit_ImportFrom.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_ImportFrom.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_ImportFrom.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_ImportFrom.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_ImportFrom.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_ImportFrom.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_ImportFrom.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_ImportFrom', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_ImportFrom', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_ImportFrom(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 't' (line 120)
        t_2519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 't')
        # Obtaining the member 'module' of a type (line 120)
        module_2520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 11), t_2519, 'module')
        
        # Getting the type of 't' (line 120)
        t_2521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 't')
        # Obtaining the member 'module' of a type (line 120)
        module_2522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 24), t_2521, 'module')
        str_2523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 36), 'str', '__future__')
        # Applying the binary operator '==' (line 120)
        result_eq_2524 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 24), '==', module_2522, str_2523)
        
        # Applying the binary operator 'and' (line 120)
        result_and_keyword_2525 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 11), 'and', module_2520, result_eq_2524)
        
        # Testing if the type of an if condition is none (line 120)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 120, 8), result_and_keyword_2525):
            pass
        else:
            
            # Testing the type of an if condition (line 120)
            if_condition_2526 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 8), result_and_keyword_2525)
            # Assigning a type to the variable 'if_condition_2526' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'if_condition_2526', if_condition_2526)
            # SSA begins for if statement (line 120)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to extend(...): (line 121)
            # Processing the call arguments (line 121)
            # Calculating generator expression
            module_type_store = module_type_store.open_function_context('list comprehension expression', 121, 39, True)
            # Calculating comprehension expression
            # Getting the type of 't' (line 121)
            t_2532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 55), 't', False)
            # Obtaining the member 'names' of a type (line 121)
            names_2533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 55), t_2532, 'names')
            comprehension_2534 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 39), names_2533)
            # Assigning a type to the variable 'n' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 39), 'n', comprehension_2534)
            # Getting the type of 'n' (line 121)
            n_2530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 39), 'n', False)
            # Obtaining the member 'name' of a type (line 121)
            name_2531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 39), n_2530, 'name')
            list_2535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'list')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 39), list_2535, name_2531)
            # Processing the call keyword arguments (line 121)
            kwargs_2536 = {}
            # Getting the type of 'self' (line 121)
            self_2527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'self', False)
            # Obtaining the member 'future_imports' of a type (line 121)
            future_imports_2528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), self_2527, 'future_imports')
            # Obtaining the member 'extend' of a type (line 121)
            extend_2529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), future_imports_2528, 'extend')
            # Calling extend(args, kwargs) (line 121)
            extend_call_result_2537 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), extend_2529, *[list_2535], **kwargs_2536)
            
            # SSA join for if statement (line 120)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to fill(...): (line 123)
        # Processing the call arguments (line 123)
        str_2540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 18), 'str', 'from ')
        # Processing the call keyword arguments (line 123)
        kwargs_2541 = {}
        # Getting the type of 'self' (line 123)
        self_2538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 123)
        fill_2539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_2538, 'fill')
        # Calling fill(args, kwargs) (line 123)
        fill_call_result_2542 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), fill_2539, *[str_2540], **kwargs_2541)
        
        
        # Call to write(...): (line 124)
        # Processing the call arguments (line 124)
        str_2545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 19), 'str', '.')
        # Getting the type of 't' (line 124)
        t_2546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 't', False)
        # Obtaining the member 'level' of a type (line 124)
        level_2547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 25), t_2546, 'level')
        # Applying the binary operator '*' (line 124)
        result_mul_2548 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 19), '*', str_2545, level_2547)
        
        # Processing the call keyword arguments (line 124)
        kwargs_2549 = {}
        # Getting the type of 'self' (line 124)
        self_2543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 124)
        write_2544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), self_2543, 'write')
        # Calling write(args, kwargs) (line 124)
        write_call_result_2550 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), write_2544, *[result_mul_2548], **kwargs_2549)
        
        # Getting the type of 't' (line 125)
        t_2551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 't')
        # Obtaining the member 'module' of a type (line 125)
        module_2552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 11), t_2551, 'module')
        # Testing if the type of an if condition is none (line 125)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 8), module_2552):
            pass
        else:
            
            # Testing the type of an if condition (line 125)
            if_condition_2553 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), module_2552)
            # Assigning a type to the variable 'if_condition_2553' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_2553', if_condition_2553)
            # SSA begins for if statement (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 126)
            # Processing the call arguments (line 126)
            # Getting the type of 't' (line 126)
            t_2556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 't', False)
            # Obtaining the member 'module' of a type (line 126)
            module_2557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), t_2556, 'module')
            # Processing the call keyword arguments (line 126)
            kwargs_2558 = {}
            # Getting the type of 'self' (line 126)
            self_2554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 126)
            write_2555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), self_2554, 'write')
            # Calling write(args, kwargs) (line 126)
            write_call_result_2559 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), write_2555, *[module_2557], **kwargs_2558)
            
            # SSA join for if statement (line 125)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 127)
        # Processing the call arguments (line 127)
        str_2562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 19), 'str', ' import ')
        # Processing the call keyword arguments (line 127)
        kwargs_2563 = {}
        # Getting the type of 'self' (line 127)
        self_2560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 127)
        write_2561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_2560, 'write')
        # Calling write(args, kwargs) (line 127)
        write_call_result_2564 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), write_2561, *[str_2562], **kwargs_2563)
        
        
        # Call to interleave(...): (line 128)
        # Processing the call arguments (line 128)

        @norecursion
        def _stypy_temp_lambda_6(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_6'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_6', 128, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_6.stypy_localization = localization
            _stypy_temp_lambda_6.stypy_type_of_self = None
            _stypy_temp_lambda_6.stypy_type_store = module_type_store
            _stypy_temp_lambda_6.stypy_function_name = '_stypy_temp_lambda_6'
            _stypy_temp_lambda_6.stypy_param_names_list = []
            _stypy_temp_lambda_6.stypy_varargs_param_name = None
            _stypy_temp_lambda_6.stypy_kwargs_param_name = None
            _stypy_temp_lambda_6.stypy_call_defaults = defaults
            _stypy_temp_lambda_6.stypy_call_varargs = varargs
            _stypy_temp_lambda_6.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_6', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_6', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 128)
            # Processing the call arguments (line 128)
            str_2568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 38), 'str', ', ')
            # Processing the call keyword arguments (line 128)
            kwargs_2569 = {}
            # Getting the type of 'self' (line 128)
            self_2566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 128)
            write_2567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 27), self_2566, 'write')
            # Calling write(args, kwargs) (line 128)
            write_call_result_2570 = invoke(stypy.reporting.localization.Localization(__file__, 128, 27), write_2567, *[str_2568], **kwargs_2569)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'stypy_return_type', write_call_result_2570)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_6' in the type store
            # Getting the type of 'stypy_return_type' (line 128)
            stypy_return_type_2571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_2571)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_6'
            return stypy_return_type_2571

        # Assigning a type to the variable '_stypy_temp_lambda_6' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), '_stypy_temp_lambda_6', _stypy_temp_lambda_6)
        # Getting the type of '_stypy_temp_lambda_6' (line 128)
        _stypy_temp_lambda_6_2572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), '_stypy_temp_lambda_6')
        # Getting the type of 'self' (line 128)
        self_2573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 128)
        visit_2574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 45), self_2573, 'visit')
        # Getting the type of 't' (line 128)
        t_2575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 57), 't', False)
        # Obtaining the member 'names' of a type (line 128)
        names_2576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 57), t_2575, 'names')
        # Processing the call keyword arguments (line 128)
        kwargs_2577 = {}
        # Getting the type of 'interleave' (line 128)
        interleave_2565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 128)
        interleave_call_result_2578 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), interleave_2565, *[_stypy_temp_lambda_6_2572, visit_2574, names_2576], **kwargs_2577)
        
        
        # Call to write(...): (line 129)
        # Processing the call arguments (line 129)
        str_2581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 19), 'str', '\n')
        # Processing the call keyword arguments (line 129)
        kwargs_2582 = {}
        # Getting the type of 'self' (line 129)
        self_2579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 129)
        write_2580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_2579, 'write')
        # Calling write(args, kwargs) (line 129)
        write_call_result_2583 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), write_2580, *[str_2581], **kwargs_2582)
        
        
        # ################# End of 'visit_ImportFrom(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ImportFrom' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_2584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2584)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ImportFrom'
        return stypy_return_type_2584


    @norecursion
    def visit_Assign(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Assign'
        module_type_store = module_type_store.open_function_context('visit_Assign', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Assign.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Assign.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Assign.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Assign.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Assign')
        PythonSrcGeneratorVisitor.visit_Assign.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Assign.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Assign.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Assign.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Assign.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Assign.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Assign.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Assign', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Assign', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Assign(...)' code ##################

        
        # Call to fill(...): (line 132)
        # Processing the call keyword arguments (line 132)
        kwargs_2587 = {}
        # Getting the type of 'self' (line 132)
        self_2585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 132)
        fill_2586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_2585, 'fill')
        # Calling fill(args, kwargs) (line 132)
        fill_call_result_2588 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), fill_2586, *[], **kwargs_2587)
        
        
        # Getting the type of 't' (line 133)
        t_2589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 't')
        # Obtaining the member 'targets' of a type (line 133)
        targets_2590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 22), t_2589, 'targets')
        # Assigning a type to the variable 'targets_2590' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'targets_2590', targets_2590)
        # Testing if the for loop is going to be iterated (line 133)
        # Testing the type of a for loop iterable (line 133)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 133, 8), targets_2590)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 133, 8), targets_2590):
            # Getting the type of the for loop variable (line 133)
            for_loop_var_2591 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 133, 8), targets_2590)
            # Assigning a type to the variable 'target' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'target', for_loop_var_2591)
            # SSA begins for a for statement (line 133)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 134)
            # Processing the call arguments (line 134)
            # Getting the type of 'target' (line 134)
            target_2594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'target', False)
            # Processing the call keyword arguments (line 134)
            kwargs_2595 = {}
            # Getting the type of 'self' (line 134)
            self_2592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 134)
            visit_2593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 12), self_2592, 'visit')
            # Calling visit(args, kwargs) (line 134)
            visit_call_result_2596 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), visit_2593, *[target_2594], **kwargs_2595)
            
            
            # Call to write(...): (line 135)
            # Processing the call arguments (line 135)
            str_2599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 23), 'str', ' = ')
            # Processing the call keyword arguments (line 135)
            kwargs_2600 = {}
            # Getting the type of 'self' (line 135)
            self_2597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 135)
            write_2598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), self_2597, 'write')
            # Calling write(args, kwargs) (line 135)
            write_call_result_2601 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), write_2598, *[str_2599], **kwargs_2600)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to visit(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 't' (line 136)
        t_2604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 't', False)
        # Obtaining the member 'value' of a type (line 136)
        value_2605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 19), t_2604, 'value')
        # Processing the call keyword arguments (line 136)
        kwargs_2606 = {}
        # Getting the type of 'self' (line 136)
        self_2602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 136)
        visit_2603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), self_2602, 'visit')
        # Calling visit(args, kwargs) (line 136)
        visit_call_result_2607 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), visit_2603, *[value_2605], **kwargs_2606)
        
        
        # ################# End of 'visit_Assign(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Assign' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_2608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2608)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Assign'
        return stypy_return_type_2608


    @norecursion
    def visit_AugAssign(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_AugAssign'
        module_type_store = module_type_store.open_function_context('visit_AugAssign', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_AugAssign.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_AugAssign.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_AugAssign.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_AugAssign.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_AugAssign')
        PythonSrcGeneratorVisitor.visit_AugAssign.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_AugAssign.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_AugAssign.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_AugAssign.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_AugAssign.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_AugAssign.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_AugAssign.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_AugAssign', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_AugAssign', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_AugAssign(...)' code ##################

        
        # Call to fill(...): (line 139)
        # Processing the call keyword arguments (line 139)
        kwargs_2611 = {}
        # Getting the type of 'self' (line 139)
        self_2609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 139)
        fill_2610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_2609, 'fill')
        # Calling fill(args, kwargs) (line 139)
        fill_call_result_2612 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), fill_2610, *[], **kwargs_2611)
        
        
        # Call to visit(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 't' (line 140)
        t_2615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 't', False)
        # Obtaining the member 'target' of a type (line 140)
        target_2616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 19), t_2615, 'target')
        # Processing the call keyword arguments (line 140)
        kwargs_2617 = {}
        # Getting the type of 'self' (line 140)
        self_2613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 140)
        visit_2614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_2613, 'visit')
        # Calling visit(args, kwargs) (line 140)
        visit_call_result_2618 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), visit_2614, *[target_2616], **kwargs_2617)
        
        
        # Call to write(...): (line 141)
        # Processing the call arguments (line 141)
        str_2621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 19), 'str', ' ')
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 141)
        t_2622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 36), 't', False)
        # Obtaining the member 'op' of a type (line 141)
        op_2623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 36), t_2622, 'op')
        # Obtaining the member '__class__' of a type (line 141)
        class___2624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 36), op_2623, '__class__')
        # Obtaining the member '__name__' of a type (line 141)
        name___2625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 36), class___2624, '__name__')
        # Getting the type of 'self' (line 141)
        self_2626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'self', False)
        # Obtaining the member 'binop' of a type (line 141)
        binop_2627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 25), self_2626, 'binop')
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___2628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 25), binop_2627, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_2629 = invoke(stypy.reporting.localization.Localization(__file__, 141, 25), getitem___2628, name___2625)
        
        # Applying the binary operator '+' (line 141)
        result_add_2630 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 19), '+', str_2621, subscript_call_result_2629)
        
        str_2631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 63), 'str', '= ')
        # Applying the binary operator '+' (line 141)
        result_add_2632 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 61), '+', result_add_2630, str_2631)
        
        # Processing the call keyword arguments (line 141)
        kwargs_2633 = {}
        # Getting the type of 'self' (line 141)
        self_2619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 141)
        write_2620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), self_2619, 'write')
        # Calling write(args, kwargs) (line 141)
        write_call_result_2634 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), write_2620, *[result_add_2632], **kwargs_2633)
        
        
        # Call to visit(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 't' (line 142)
        t_2637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 't', False)
        # Obtaining the member 'value' of a type (line 142)
        value_2638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 19), t_2637, 'value')
        # Processing the call keyword arguments (line 142)
        kwargs_2639 = {}
        # Getting the type of 'self' (line 142)
        self_2635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 142)
        visit_2636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_2635, 'visit')
        # Calling visit(args, kwargs) (line 142)
        visit_call_result_2640 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), visit_2636, *[value_2638], **kwargs_2639)
        
        
        # ################# End of 'visit_AugAssign(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_AugAssign' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_2641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2641)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_AugAssign'
        return stypy_return_type_2641


    @norecursion
    def visit_Return(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Return'
        module_type_store = module_type_store.open_function_context('visit_Return', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Return')
        PythonSrcGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Return.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Return', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Return', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Return(...)' code ##################

        
        # Call to fill(...): (line 145)
        # Processing the call arguments (line 145)
        str_2644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 18), 'str', 'return')
        # Processing the call keyword arguments (line 145)
        kwargs_2645 = {}
        # Getting the type of 'self' (line 145)
        self_2642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 145)
        fill_2643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_2642, 'fill')
        # Calling fill(args, kwargs) (line 145)
        fill_call_result_2646 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), fill_2643, *[str_2644], **kwargs_2645)
        
        # Getting the type of 't' (line 146)
        t_2647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 't')
        # Obtaining the member 'value' of a type (line 146)
        value_2648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 11), t_2647, 'value')
        # Testing if the type of an if condition is none (line 146)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 146, 8), value_2648):
            pass
        else:
            
            # Testing the type of an if condition (line 146)
            if_condition_2649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 8), value_2648)
            # Assigning a type to the variable 'if_condition_2649' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'if_condition_2649', if_condition_2649)
            # SSA begins for if statement (line 146)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 147)
            # Processing the call arguments (line 147)
            str_2652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 23), 'str', ' ')
            # Processing the call keyword arguments (line 147)
            kwargs_2653 = {}
            # Getting the type of 'self' (line 147)
            self_2650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 147)
            write_2651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), self_2650, 'write')
            # Calling write(args, kwargs) (line 147)
            write_call_result_2654 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), write_2651, *[str_2652], **kwargs_2653)
            
            
            # Call to visit(...): (line 148)
            # Processing the call arguments (line 148)
            # Getting the type of 't' (line 148)
            t_2657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 't', False)
            # Obtaining the member 'value' of a type (line 148)
            value_2658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 23), t_2657, 'value')
            # Processing the call keyword arguments (line 148)
            kwargs_2659 = {}
            # Getting the type of 'self' (line 148)
            self_2655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 148)
            visit_2656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), self_2655, 'visit')
            # Calling visit(args, kwargs) (line 148)
            visit_call_result_2660 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), visit_2656, *[value_2658], **kwargs_2659)
            
            # SSA join for if statement (line 146)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Return(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Return' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_2661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2661)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Return'
        return stypy_return_type_2661


    @norecursion
    def visit_Pass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Pass'
        module_type_store = module_type_store.open_function_context('visit_Pass', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Pass.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Pass.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Pass.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Pass.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Pass')
        PythonSrcGeneratorVisitor.visit_Pass.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Pass.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Pass.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Pass.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Pass.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Pass.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Pass.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Pass', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Pass', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Pass(...)' code ##################

        
        # Call to fill(...): (line 152)
        # Processing the call arguments (line 152)
        str_2664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 18), 'str', 'pass')
        # Processing the call keyword arguments (line 152)
        kwargs_2665 = {}
        # Getting the type of 'self' (line 152)
        self_2662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 152)
        fill_2663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), self_2662, 'fill')
        # Calling fill(args, kwargs) (line 152)
        fill_call_result_2666 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), fill_2663, *[str_2664], **kwargs_2665)
        
        
        # ################# End of 'visit_Pass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Pass' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_2667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2667)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Pass'
        return stypy_return_type_2667


    @norecursion
    def visit_Break(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Break'
        module_type_store = module_type_store.open_function_context('visit_Break', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Break.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Break.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Break.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Break.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Break')
        PythonSrcGeneratorVisitor.visit_Break.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Break.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Break.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Break.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Break.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Break.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Break.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Break', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Break', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Break(...)' code ##################

        
        # Call to fill(...): (line 155)
        # Processing the call arguments (line 155)
        str_2670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 18), 'str', 'break')
        # Processing the call keyword arguments (line 155)
        kwargs_2671 = {}
        # Getting the type of 'self' (line 155)
        self_2668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 155)
        fill_2669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_2668, 'fill')
        # Calling fill(args, kwargs) (line 155)
        fill_call_result_2672 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), fill_2669, *[str_2670], **kwargs_2671)
        
        
        # ################# End of 'visit_Break(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Break' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_2673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2673)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Break'
        return stypy_return_type_2673


    @norecursion
    def visit_Continue(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Continue'
        module_type_store = module_type_store.open_function_context('visit_Continue', 157, 4, False)
        # Assigning a type to the variable 'self' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Continue.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Continue.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Continue.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Continue.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Continue')
        PythonSrcGeneratorVisitor.visit_Continue.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Continue.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Continue.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Continue.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Continue.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Continue.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Continue.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Continue', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Continue', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Continue(...)' code ##################

        
        # Call to fill(...): (line 158)
        # Processing the call arguments (line 158)
        str_2676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 18), 'str', 'continue')
        # Processing the call keyword arguments (line 158)
        kwargs_2677 = {}
        # Getting the type of 'self' (line 158)
        self_2674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 158)
        fill_2675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), self_2674, 'fill')
        # Calling fill(args, kwargs) (line 158)
        fill_call_result_2678 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), fill_2675, *[str_2676], **kwargs_2677)
        
        
        # ################# End of 'visit_Continue(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Continue' in the type store
        # Getting the type of 'stypy_return_type' (line 157)
        stypy_return_type_2679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2679)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Continue'
        return stypy_return_type_2679


    @norecursion
    def visit_Delete(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Delete'
        module_type_store = module_type_store.open_function_context('visit_Delete', 160, 4, False)
        # Assigning a type to the variable 'self' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Delete.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Delete.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Delete.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Delete.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Delete')
        PythonSrcGeneratorVisitor.visit_Delete.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Delete.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Delete.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Delete.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Delete.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Delete.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Delete.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Delete', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Delete', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Delete(...)' code ##################

        
        # Call to fill(...): (line 161)
        # Processing the call arguments (line 161)
        str_2682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 18), 'str', 'del ')
        # Processing the call keyword arguments (line 161)
        kwargs_2683 = {}
        # Getting the type of 'self' (line 161)
        self_2680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 161)
        fill_2681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), self_2680, 'fill')
        # Calling fill(args, kwargs) (line 161)
        fill_call_result_2684 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), fill_2681, *[str_2682], **kwargs_2683)
        
        
        # Call to interleave(...): (line 162)
        # Processing the call arguments (line 162)

        @norecursion
        def _stypy_temp_lambda_7(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_7'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_7', 162, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_7.stypy_localization = localization
            _stypy_temp_lambda_7.stypy_type_of_self = None
            _stypy_temp_lambda_7.stypy_type_store = module_type_store
            _stypy_temp_lambda_7.stypy_function_name = '_stypy_temp_lambda_7'
            _stypy_temp_lambda_7.stypy_param_names_list = []
            _stypy_temp_lambda_7.stypy_varargs_param_name = None
            _stypy_temp_lambda_7.stypy_kwargs_param_name = None
            _stypy_temp_lambda_7.stypy_call_defaults = defaults
            _stypy_temp_lambda_7.stypy_call_varargs = varargs
            _stypy_temp_lambda_7.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_7', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_7', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 162)
            # Processing the call arguments (line 162)
            str_2688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 38), 'str', ', ')
            # Processing the call keyword arguments (line 162)
            kwargs_2689 = {}
            # Getting the type of 'self' (line 162)
            self_2686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 162)
            write_2687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 27), self_2686, 'write')
            # Calling write(args, kwargs) (line 162)
            write_call_result_2690 = invoke(stypy.reporting.localization.Localization(__file__, 162, 27), write_2687, *[str_2688], **kwargs_2689)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'stypy_return_type', write_call_result_2690)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_7' in the type store
            # Getting the type of 'stypy_return_type' (line 162)
            stypy_return_type_2691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_2691)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_7'
            return stypy_return_type_2691

        # Assigning a type to the variable '_stypy_temp_lambda_7' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), '_stypy_temp_lambda_7', _stypy_temp_lambda_7)
        # Getting the type of '_stypy_temp_lambda_7' (line 162)
        _stypy_temp_lambda_7_2692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), '_stypy_temp_lambda_7')
        # Getting the type of 'self' (line 162)
        self_2693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 162)
        visit_2694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 45), self_2693, 'visit')
        # Getting the type of 't' (line 162)
        t_2695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 57), 't', False)
        # Obtaining the member 'targets' of a type (line 162)
        targets_2696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 57), t_2695, 'targets')
        # Processing the call keyword arguments (line 162)
        kwargs_2697 = {}
        # Getting the type of 'interleave' (line 162)
        interleave_2685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 162)
        interleave_call_result_2698 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), interleave_2685, *[_stypy_temp_lambda_7_2692, visit_2694, targets_2696], **kwargs_2697)
        
        
        # ################# End of 'visit_Delete(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Delete' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_2699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2699)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Delete'
        return stypy_return_type_2699


    @norecursion
    def visit_Assert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Assert'
        module_type_store = module_type_store.open_function_context('visit_Assert', 164, 4, False)
        # Assigning a type to the variable 'self' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Assert.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Assert.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Assert.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Assert.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Assert')
        PythonSrcGeneratorVisitor.visit_Assert.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Assert.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Assert.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Assert.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Assert.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Assert.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Assert.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Assert', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Assert', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Assert(...)' code ##################

        
        # Call to fill(...): (line 165)
        # Processing the call arguments (line 165)
        str_2702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 18), 'str', 'assert ')
        # Processing the call keyword arguments (line 165)
        kwargs_2703 = {}
        # Getting the type of 'self' (line 165)
        self_2700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 165)
        fill_2701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), self_2700, 'fill')
        # Calling fill(args, kwargs) (line 165)
        fill_call_result_2704 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), fill_2701, *[str_2702], **kwargs_2703)
        
        
        # Call to visit(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 't' (line 166)
        t_2707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 19), 't', False)
        # Obtaining the member 'test' of a type (line 166)
        test_2708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 19), t_2707, 'test')
        # Processing the call keyword arguments (line 166)
        kwargs_2709 = {}
        # Getting the type of 'self' (line 166)
        self_2705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 166)
        visit_2706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_2705, 'visit')
        # Calling visit(args, kwargs) (line 166)
        visit_call_result_2710 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), visit_2706, *[test_2708], **kwargs_2709)
        
        # Getting the type of 't' (line 167)
        t_2711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 't')
        # Obtaining the member 'msg' of a type (line 167)
        msg_2712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 11), t_2711, 'msg')
        # Testing if the type of an if condition is none (line 167)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 167, 8), msg_2712):
            pass
        else:
            
            # Testing the type of an if condition (line 167)
            if_condition_2713 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 8), msg_2712)
            # Assigning a type to the variable 'if_condition_2713' (line 167)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'if_condition_2713', if_condition_2713)
            # SSA begins for if statement (line 167)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 168)
            # Processing the call arguments (line 168)
            str_2716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 23), 'str', ', ')
            # Processing the call keyword arguments (line 168)
            kwargs_2717 = {}
            # Getting the type of 'self' (line 168)
            self_2714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 168)
            write_2715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), self_2714, 'write')
            # Calling write(args, kwargs) (line 168)
            write_call_result_2718 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), write_2715, *[str_2716], **kwargs_2717)
            
            
            # Call to visit(...): (line 169)
            # Processing the call arguments (line 169)
            # Getting the type of 't' (line 169)
            t_2721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 't', False)
            # Obtaining the member 'msg' of a type (line 169)
            msg_2722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 23), t_2721, 'msg')
            # Processing the call keyword arguments (line 169)
            kwargs_2723 = {}
            # Getting the type of 'self' (line 169)
            self_2719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 169)
            visit_2720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), self_2719, 'visit')
            # Calling visit(args, kwargs) (line 169)
            visit_call_result_2724 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), visit_2720, *[msg_2722], **kwargs_2723)
            
            # SSA join for if statement (line 167)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Assert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Assert' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_2725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2725)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Assert'
        return stypy_return_type_2725


    @norecursion
    def visit_Exec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Exec'
        module_type_store = module_type_store.open_function_context('visit_Exec', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Exec.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Exec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Exec.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Exec.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Exec')
        PythonSrcGeneratorVisitor.visit_Exec.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Exec.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Exec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Exec.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Exec.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Exec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Exec.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Exec', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Exec', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Exec(...)' code ##################

        
        # Call to fill(...): (line 172)
        # Processing the call arguments (line 172)
        str_2728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 18), 'str', 'exec ')
        # Processing the call keyword arguments (line 172)
        kwargs_2729 = {}
        # Getting the type of 'self' (line 172)
        self_2726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 172)
        fill_2727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_2726, 'fill')
        # Calling fill(args, kwargs) (line 172)
        fill_call_result_2730 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), fill_2727, *[str_2728], **kwargs_2729)
        
        
        # Call to visit(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 't' (line 173)
        t_2733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 't', False)
        # Obtaining the member 'body' of a type (line 173)
        body_2734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 19), t_2733, 'body')
        # Processing the call keyword arguments (line 173)
        kwargs_2735 = {}
        # Getting the type of 'self' (line 173)
        self_2731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 173)
        visit_2732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_2731, 'visit')
        # Calling visit(args, kwargs) (line 173)
        visit_call_result_2736 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), visit_2732, *[body_2734], **kwargs_2735)
        
        # Getting the type of 't' (line 174)
        t_2737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 't')
        # Obtaining the member 'globals' of a type (line 174)
        globals_2738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 11), t_2737, 'globals')
        # Testing if the type of an if condition is none (line 174)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 174, 8), globals_2738):
            pass
        else:
            
            # Testing the type of an if condition (line 174)
            if_condition_2739 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 8), globals_2738)
            # Assigning a type to the variable 'if_condition_2739' (line 174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'if_condition_2739', if_condition_2739)
            # SSA begins for if statement (line 174)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 175)
            # Processing the call arguments (line 175)
            str_2742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 23), 'str', ' in ')
            # Processing the call keyword arguments (line 175)
            kwargs_2743 = {}
            # Getting the type of 'self' (line 175)
            self_2740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 175)
            write_2741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), self_2740, 'write')
            # Calling write(args, kwargs) (line 175)
            write_call_result_2744 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), write_2741, *[str_2742], **kwargs_2743)
            
            
            # Call to visit(...): (line 176)
            # Processing the call arguments (line 176)
            # Getting the type of 't' (line 176)
            t_2747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 't', False)
            # Obtaining the member 'globals' of a type (line 176)
            globals_2748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 23), t_2747, 'globals')
            # Processing the call keyword arguments (line 176)
            kwargs_2749 = {}
            # Getting the type of 'self' (line 176)
            self_2745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 176)
            visit_2746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), self_2745, 'visit')
            # Calling visit(args, kwargs) (line 176)
            visit_call_result_2750 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), visit_2746, *[globals_2748], **kwargs_2749)
            
            # SSA join for if statement (line 174)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 177)
        t_2751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 't')
        # Obtaining the member 'locals' of a type (line 177)
        locals_2752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 11), t_2751, 'locals')
        # Testing if the type of an if condition is none (line 177)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 177, 8), locals_2752):
            pass
        else:
            
            # Testing the type of an if condition (line 177)
            if_condition_2753 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 8), locals_2752)
            # Assigning a type to the variable 'if_condition_2753' (line 177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'if_condition_2753', if_condition_2753)
            # SSA begins for if statement (line 177)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 178)
            # Processing the call arguments (line 178)
            str_2756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 23), 'str', ', ')
            # Processing the call keyword arguments (line 178)
            kwargs_2757 = {}
            # Getting the type of 'self' (line 178)
            self_2754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 178)
            write_2755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), self_2754, 'write')
            # Calling write(args, kwargs) (line 178)
            write_call_result_2758 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), write_2755, *[str_2756], **kwargs_2757)
            
            
            # Call to visit(...): (line 179)
            # Processing the call arguments (line 179)
            # Getting the type of 't' (line 179)
            t_2761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 23), 't', False)
            # Obtaining the member 'locals' of a type (line 179)
            locals_2762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 23), t_2761, 'locals')
            # Processing the call keyword arguments (line 179)
            kwargs_2763 = {}
            # Getting the type of 'self' (line 179)
            self_2759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 179)
            visit_2760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), self_2759, 'visit')
            # Calling visit(args, kwargs) (line 179)
            visit_call_result_2764 = invoke(stypy.reporting.localization.Localization(__file__, 179, 12), visit_2760, *[locals_2762], **kwargs_2763)
            
            # SSA join for if statement (line 177)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Exec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Exec' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_2765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2765)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Exec'
        return stypy_return_type_2765


    @norecursion
    def visit_Print(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Print'
        module_type_store = module_type_store.open_function_context('visit_Print', 181, 4, False)
        # Assigning a type to the variable 'self' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Print.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Print.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Print.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Print.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Print')
        PythonSrcGeneratorVisitor.visit_Print.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Print.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Print.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Print.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Print.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Print.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Print.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Print', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Print', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Print(...)' code ##################

        
        # Call to fill(...): (line 182)
        # Processing the call arguments (line 182)
        str_2768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 18), 'str', 'print ')
        # Processing the call keyword arguments (line 182)
        kwargs_2769 = {}
        # Getting the type of 'self' (line 182)
        self_2766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 182)
        fill_2767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), self_2766, 'fill')
        # Calling fill(args, kwargs) (line 182)
        fill_call_result_2770 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), fill_2767, *[str_2768], **kwargs_2769)
        
        
        # Assigning a Name to a Name (line 183):
        
        # Assigning a Name to a Name (line 183):
        # Getting the type of 'False' (line 183)
        False_2771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'False')
        # Assigning a type to the variable 'do_comma' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'do_comma', False_2771)
        # Getting the type of 't' (line 184)
        t_2772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 't')
        # Obtaining the member 'dest' of a type (line 184)
        dest_2773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 11), t_2772, 'dest')
        # Testing if the type of an if condition is none (line 184)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 184, 8), dest_2773):
            pass
        else:
            
            # Testing the type of an if condition (line 184)
            if_condition_2774 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 8), dest_2773)
            # Assigning a type to the variable 'if_condition_2774' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'if_condition_2774', if_condition_2774)
            # SSA begins for if statement (line 184)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 185)
            # Processing the call arguments (line 185)
            str_2777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 23), 'str', '>>')
            # Processing the call keyword arguments (line 185)
            kwargs_2778 = {}
            # Getting the type of 'self' (line 185)
            self_2775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 185)
            write_2776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 12), self_2775, 'write')
            # Calling write(args, kwargs) (line 185)
            write_call_result_2779 = invoke(stypy.reporting.localization.Localization(__file__, 185, 12), write_2776, *[str_2777], **kwargs_2778)
            
            
            # Call to visit(...): (line 186)
            # Processing the call arguments (line 186)
            # Getting the type of 't' (line 186)
            t_2782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 23), 't', False)
            # Obtaining the member 'dest' of a type (line 186)
            dest_2783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 23), t_2782, 'dest')
            # Processing the call keyword arguments (line 186)
            kwargs_2784 = {}
            # Getting the type of 'self' (line 186)
            self_2780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 186)
            visit_2781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), self_2780, 'visit')
            # Calling visit(args, kwargs) (line 186)
            visit_call_result_2785 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), visit_2781, *[dest_2783], **kwargs_2784)
            
            
            # Assigning a Name to a Name (line 187):
            
            # Assigning a Name to a Name (line 187):
            # Getting the type of 'True' (line 187)
            True_2786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'True')
            # Assigning a type to the variable 'do_comma' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'do_comma', True_2786)
            # SSA join for if statement (line 184)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 't' (line 188)
        t_2787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 17), 't')
        # Obtaining the member 'values' of a type (line 188)
        values_2788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 17), t_2787, 'values')
        # Assigning a type to the variable 'values_2788' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'values_2788', values_2788)
        # Testing if the for loop is going to be iterated (line 188)
        # Testing the type of a for loop iterable (line 188)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 188, 8), values_2788)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 188, 8), values_2788):
            # Getting the type of the for loop variable (line 188)
            for_loop_var_2789 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 188, 8), values_2788)
            # Assigning a type to the variable 'e' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'e', for_loop_var_2789)
            # SSA begins for a for statement (line 188)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'do_comma' (line 189)
            do_comma_2790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'do_comma')
            # Testing if the type of an if condition is none (line 189)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 189, 12), do_comma_2790):
                
                # Assigning a Name to a Name (line 192):
                
                # Assigning a Name to a Name (line 192):
                # Getting the type of 'True' (line 192)
                True_2797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'True')
                # Assigning a type to the variable 'do_comma' (line 192)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'do_comma', True_2797)
            else:
                
                # Testing the type of an if condition (line 189)
                if_condition_2791 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 12), do_comma_2790)
                # Assigning a type to the variable 'if_condition_2791' (line 189)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'if_condition_2791', if_condition_2791)
                # SSA begins for if statement (line 189)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 190)
                # Processing the call arguments (line 190)
                str_2794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 27), 'str', ', ')
                # Processing the call keyword arguments (line 190)
                kwargs_2795 = {}
                # Getting the type of 'self' (line 190)
                self_2792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 190)
                write_2793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 16), self_2792, 'write')
                # Calling write(args, kwargs) (line 190)
                write_call_result_2796 = invoke(stypy.reporting.localization.Localization(__file__, 190, 16), write_2793, *[str_2794], **kwargs_2795)
                
                # SSA branch for the else part of an if statement (line 189)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 192):
                
                # Assigning a Name to a Name (line 192):
                # Getting the type of 'True' (line 192)
                True_2797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'True')
                # Assigning a type to the variable 'do_comma' (line 192)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'do_comma', True_2797)
                # SSA join for if statement (line 189)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to visit(...): (line 193)
            # Processing the call arguments (line 193)
            # Getting the type of 'e' (line 193)
            e_2800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'e', False)
            # Processing the call keyword arguments (line 193)
            kwargs_2801 = {}
            # Getting the type of 'self' (line 193)
            self_2798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 193)
            visit_2799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), self_2798, 'visit')
            # Calling visit(args, kwargs) (line 193)
            visit_call_result_2802 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), visit_2799, *[e_2800], **kwargs_2801)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 't' (line 194)
        t_2803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 't')
        # Obtaining the member 'nl' of a type (line 194)
        nl_2804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 15), t_2803, 'nl')
        # Applying the 'not' unary operator (line 194)
        result_not__2805 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 11), 'not', nl_2804)
        
        # Testing if the type of an if condition is none (line 194)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 194, 8), result_not__2805):
            pass
        else:
            
            # Testing the type of an if condition (line 194)
            if_condition_2806 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 8), result_not__2805)
            # Assigning a type to the variable 'if_condition_2806' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'if_condition_2806', if_condition_2806)
            # SSA begins for if statement (line 194)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 195)
            # Processing the call arguments (line 195)
            str_2809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 23), 'str', ',')
            # Processing the call keyword arguments (line 195)
            kwargs_2810 = {}
            # Getting the type of 'self' (line 195)
            self_2807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 195)
            write_2808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), self_2807, 'write')
            # Calling write(args, kwargs) (line 195)
            write_call_result_2811 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), write_2808, *[str_2809], **kwargs_2810)
            
            # SSA join for if statement (line 194)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Print(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Print' in the type store
        # Getting the type of 'stypy_return_type' (line 181)
        stypy_return_type_2812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2812)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Print'
        return stypy_return_type_2812


    @norecursion
    def visit_Global(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Global'
        module_type_store = module_type_store.open_function_context('visit_Global', 197, 4, False)
        # Assigning a type to the variable 'self' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Global.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Global.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Global.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Global.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Global')
        PythonSrcGeneratorVisitor.visit_Global.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Global.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Global.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Global.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Global.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Global.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Global.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Global', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Global', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Global(...)' code ##################

        
        # Call to fill(...): (line 198)
        # Processing the call arguments (line 198)
        str_2815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 18), 'str', 'global ')
        # Processing the call keyword arguments (line 198)
        kwargs_2816 = {}
        # Getting the type of 'self' (line 198)
        self_2813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 198)
        fill_2814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_2813, 'fill')
        # Calling fill(args, kwargs) (line 198)
        fill_call_result_2817 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), fill_2814, *[str_2815], **kwargs_2816)
        
        
        # Call to interleave(...): (line 199)
        # Processing the call arguments (line 199)

        @norecursion
        def _stypy_temp_lambda_8(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_8'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_8', 199, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_8.stypy_localization = localization
            _stypy_temp_lambda_8.stypy_type_of_self = None
            _stypy_temp_lambda_8.stypy_type_store = module_type_store
            _stypy_temp_lambda_8.stypy_function_name = '_stypy_temp_lambda_8'
            _stypy_temp_lambda_8.stypy_param_names_list = []
            _stypy_temp_lambda_8.stypy_varargs_param_name = None
            _stypy_temp_lambda_8.stypy_kwargs_param_name = None
            _stypy_temp_lambda_8.stypy_call_defaults = defaults
            _stypy_temp_lambda_8.stypy_call_varargs = varargs
            _stypy_temp_lambda_8.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_8', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_8', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 199)
            # Processing the call arguments (line 199)
            str_2821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 38), 'str', ', ')
            # Processing the call keyword arguments (line 199)
            kwargs_2822 = {}
            # Getting the type of 'self' (line 199)
            self_2819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 199)
            write_2820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 27), self_2819, 'write')
            # Calling write(args, kwargs) (line 199)
            write_call_result_2823 = invoke(stypy.reporting.localization.Localization(__file__, 199, 27), write_2820, *[str_2821], **kwargs_2822)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'stypy_return_type', write_call_result_2823)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_8' in the type store
            # Getting the type of 'stypy_return_type' (line 199)
            stypy_return_type_2824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_2824)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_8'
            return stypy_return_type_2824

        # Assigning a type to the variable '_stypy_temp_lambda_8' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), '_stypy_temp_lambda_8', _stypy_temp_lambda_8)
        # Getting the type of '_stypy_temp_lambda_8' (line 199)
        _stypy_temp_lambda_8_2825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), '_stypy_temp_lambda_8')
        # Getting the type of 'self' (line 199)
        self_2826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 45), 'self', False)
        # Obtaining the member 'write' of a type (line 199)
        write_2827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 45), self_2826, 'write')
        # Getting the type of 't' (line 199)
        t_2828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 57), 't', False)
        # Obtaining the member 'names' of a type (line 199)
        names_2829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 57), t_2828, 'names')
        # Processing the call keyword arguments (line 199)
        kwargs_2830 = {}
        # Getting the type of 'interleave' (line 199)
        interleave_2818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 199)
        interleave_call_result_2831 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), interleave_2818, *[_stypy_temp_lambda_8_2825, write_2827, names_2829], **kwargs_2830)
        
        
        # ################# End of 'visit_Global(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Global' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_2832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2832)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Global'
        return stypy_return_type_2832


    @norecursion
    def visit_Yield(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Yield'
        module_type_store = module_type_store.open_function_context('visit_Yield', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Yield.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Yield.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Yield.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Yield.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Yield')
        PythonSrcGeneratorVisitor.visit_Yield.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Yield.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Yield.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Yield.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Yield.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Yield.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Yield.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Yield', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Yield', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Yield(...)' code ##################

        
        # Call to write(...): (line 202)
        # Processing the call arguments (line 202)
        str_2835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 19), 'str', '(')
        # Processing the call keyword arguments (line 202)
        kwargs_2836 = {}
        # Getting the type of 'self' (line 202)
        self_2833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 202)
        write_2834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_2833, 'write')
        # Calling write(args, kwargs) (line 202)
        write_call_result_2837 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), write_2834, *[str_2835], **kwargs_2836)
        
        
        # Call to write(...): (line 203)
        # Processing the call arguments (line 203)
        str_2840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 19), 'str', 'yield')
        # Processing the call keyword arguments (line 203)
        kwargs_2841 = {}
        # Getting the type of 'self' (line 203)
        self_2838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 203)
        write_2839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_2838, 'write')
        # Calling write(args, kwargs) (line 203)
        write_call_result_2842 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), write_2839, *[str_2840], **kwargs_2841)
        
        # Getting the type of 't' (line 204)
        t_2843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 't')
        # Obtaining the member 'value' of a type (line 204)
        value_2844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 11), t_2843, 'value')
        # Testing if the type of an if condition is none (line 204)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 204, 8), value_2844):
            pass
        else:
            
            # Testing the type of an if condition (line 204)
            if_condition_2845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 8), value_2844)
            # Assigning a type to the variable 'if_condition_2845' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'if_condition_2845', if_condition_2845)
            # SSA begins for if statement (line 204)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 205)
            # Processing the call arguments (line 205)
            str_2848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 23), 'str', ' ')
            # Processing the call keyword arguments (line 205)
            kwargs_2849 = {}
            # Getting the type of 'self' (line 205)
            self_2846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 205)
            write_2847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), self_2846, 'write')
            # Calling write(args, kwargs) (line 205)
            write_call_result_2850 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), write_2847, *[str_2848], **kwargs_2849)
            
            
            # Call to visit(...): (line 206)
            # Processing the call arguments (line 206)
            # Getting the type of 't' (line 206)
            t_2853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 23), 't', False)
            # Obtaining the member 'value' of a type (line 206)
            value_2854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 23), t_2853, 'value')
            # Processing the call keyword arguments (line 206)
            kwargs_2855 = {}
            # Getting the type of 'self' (line 206)
            self_2851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 206)
            visit_2852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), self_2851, 'visit')
            # Calling visit(args, kwargs) (line 206)
            visit_call_result_2856 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), visit_2852, *[value_2854], **kwargs_2855)
            
            # SSA join for if statement (line 204)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 207)
        # Processing the call arguments (line 207)
        str_2859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 19), 'str', ')')
        # Processing the call keyword arguments (line 207)
        kwargs_2860 = {}
        # Getting the type of 'self' (line 207)
        self_2857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 207)
        write_2858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_2857, 'write')
        # Calling write(args, kwargs) (line 207)
        write_call_result_2861 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), write_2858, *[str_2859], **kwargs_2860)
        
        
        # ################# End of 'visit_Yield(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Yield' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_2862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2862)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Yield'
        return stypy_return_type_2862


    @norecursion
    def visit_Raise(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Raise'
        module_type_store = module_type_store.open_function_context('visit_Raise', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Raise.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Raise.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Raise.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Raise.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Raise')
        PythonSrcGeneratorVisitor.visit_Raise.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Raise.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Raise.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Raise.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Raise.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Raise.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Raise.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Raise', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Raise', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Raise(...)' code ##################

        
        # Call to fill(...): (line 210)
        # Processing the call arguments (line 210)
        str_2865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 18), 'str', 'raise ')
        # Processing the call keyword arguments (line 210)
        kwargs_2866 = {}
        # Getting the type of 'self' (line 210)
        self_2863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 210)
        fill_2864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), self_2863, 'fill')
        # Calling fill(args, kwargs) (line 210)
        fill_call_result_2867 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), fill_2864, *[str_2865], **kwargs_2866)
        
        # Getting the type of 't' (line 211)
        t_2868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 't')
        # Obtaining the member 'type' of a type (line 211)
        type_2869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), t_2868, 'type')
        # Testing if the type of an if condition is none (line 211)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 211, 8), type_2869):
            pass
        else:
            
            # Testing the type of an if condition (line 211)
            if_condition_2870 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), type_2869)
            # Assigning a type to the variable 'if_condition_2870' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_2870', if_condition_2870)
            # SSA begins for if statement (line 211)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to visit(...): (line 212)
            # Processing the call arguments (line 212)
            # Getting the type of 't' (line 212)
            t_2873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 23), 't', False)
            # Obtaining the member 'type' of a type (line 212)
            type_2874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 23), t_2873, 'type')
            # Processing the call keyword arguments (line 212)
            kwargs_2875 = {}
            # Getting the type of 'self' (line 212)
            self_2871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 212)
            visit_2872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), self_2871, 'visit')
            # Calling visit(args, kwargs) (line 212)
            visit_call_result_2876 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), visit_2872, *[type_2874], **kwargs_2875)
            
            # SSA join for if statement (line 211)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 213)
        t_2877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 't')
        # Obtaining the member 'inst' of a type (line 213)
        inst_2878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 11), t_2877, 'inst')
        # Testing if the type of an if condition is none (line 213)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 213, 8), inst_2878):
            pass
        else:
            
            # Testing the type of an if condition (line 213)
            if_condition_2879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 8), inst_2878)
            # Assigning a type to the variable 'if_condition_2879' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'if_condition_2879', if_condition_2879)
            # SSA begins for if statement (line 213)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 214)
            # Processing the call arguments (line 214)
            str_2882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 23), 'str', ', ')
            # Processing the call keyword arguments (line 214)
            kwargs_2883 = {}
            # Getting the type of 'self' (line 214)
            self_2880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 214)
            write_2881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 12), self_2880, 'write')
            # Calling write(args, kwargs) (line 214)
            write_call_result_2884 = invoke(stypy.reporting.localization.Localization(__file__, 214, 12), write_2881, *[str_2882], **kwargs_2883)
            
            
            # Call to visit(...): (line 215)
            # Processing the call arguments (line 215)
            # Getting the type of 't' (line 215)
            t_2887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 't', False)
            # Obtaining the member 'inst' of a type (line 215)
            inst_2888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 23), t_2887, 'inst')
            # Processing the call keyword arguments (line 215)
            kwargs_2889 = {}
            # Getting the type of 'self' (line 215)
            self_2885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 215)
            visit_2886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), self_2885, 'visit')
            # Calling visit(args, kwargs) (line 215)
            visit_call_result_2890 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), visit_2886, *[inst_2888], **kwargs_2889)
            
            # SSA join for if statement (line 213)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 216)
        t_2891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 11), 't')
        # Obtaining the member 'tback' of a type (line 216)
        tback_2892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 11), t_2891, 'tback')
        # Testing if the type of an if condition is none (line 216)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 216, 8), tback_2892):
            pass
        else:
            
            # Testing the type of an if condition (line 216)
            if_condition_2893 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 216, 8), tback_2892)
            # Assigning a type to the variable 'if_condition_2893' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'if_condition_2893', if_condition_2893)
            # SSA begins for if statement (line 216)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 217)
            # Processing the call arguments (line 217)
            str_2896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 23), 'str', ', ')
            # Processing the call keyword arguments (line 217)
            kwargs_2897 = {}
            # Getting the type of 'self' (line 217)
            self_2894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 217)
            write_2895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), self_2894, 'write')
            # Calling write(args, kwargs) (line 217)
            write_call_result_2898 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), write_2895, *[str_2896], **kwargs_2897)
            
            
            # Call to visit(...): (line 218)
            # Processing the call arguments (line 218)
            # Getting the type of 't' (line 218)
            t_2901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 't', False)
            # Obtaining the member 'tback' of a type (line 218)
            tback_2902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 23), t_2901, 'tback')
            # Processing the call keyword arguments (line 218)
            kwargs_2903 = {}
            # Getting the type of 'self' (line 218)
            self_2899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 218)
            visit_2900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), self_2899, 'visit')
            # Calling visit(args, kwargs) (line 218)
            visit_call_result_2904 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), visit_2900, *[tback_2902], **kwargs_2903)
            
            # SSA join for if statement (line 216)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Raise(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Raise' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_2905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2905)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Raise'
        return stypy_return_type_2905


    @norecursion
    def visit_TryExcept(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_TryExcept'
        module_type_store = module_type_store.open_function_context('visit_TryExcept', 220, 4, False)
        # Assigning a type to the variable 'self' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_TryExcept.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_TryExcept.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_TryExcept.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_TryExcept.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_TryExcept')
        PythonSrcGeneratorVisitor.visit_TryExcept.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_TryExcept.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_TryExcept.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_TryExcept.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_TryExcept.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_TryExcept.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_TryExcept.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_TryExcept', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_TryExcept', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_TryExcept(...)' code ##################

        
        # Call to fill(...): (line 221)
        # Processing the call arguments (line 221)
        str_2908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 18), 'str', 'try')
        # Processing the call keyword arguments (line 221)
        kwargs_2909 = {}
        # Getting the type of 'self' (line 221)
        self_2906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 221)
        fill_2907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_2906, 'fill')
        # Calling fill(args, kwargs) (line 221)
        fill_call_result_2910 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), fill_2907, *[str_2908], **kwargs_2909)
        
        
        # Call to enter(...): (line 222)
        # Processing the call keyword arguments (line 222)
        kwargs_2913 = {}
        # Getting the type of 'self' (line 222)
        self_2911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 222)
        enter_2912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_2911, 'enter')
        # Calling enter(args, kwargs) (line 222)
        enter_call_result_2914 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), enter_2912, *[], **kwargs_2913)
        
        
        # Call to visit(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 't' (line 223)
        t_2917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 19), 't', False)
        # Obtaining the member 'body' of a type (line 223)
        body_2918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 19), t_2917, 'body')
        # Processing the call keyword arguments (line 223)
        kwargs_2919 = {}
        # Getting the type of 'self' (line 223)
        self_2915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 223)
        visit_2916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), self_2915, 'visit')
        # Calling visit(args, kwargs) (line 223)
        visit_call_result_2920 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), visit_2916, *[body_2918], **kwargs_2919)
        
        
        # Call to leave(...): (line 224)
        # Processing the call keyword arguments (line 224)
        kwargs_2923 = {}
        # Getting the type of 'self' (line 224)
        self_2921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 224)
        leave_2922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), self_2921, 'leave')
        # Calling leave(args, kwargs) (line 224)
        leave_call_result_2924 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), leave_2922, *[], **kwargs_2923)
        
        
        # Getting the type of 't' (line 226)
        t_2925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 18), 't')
        # Obtaining the member 'handlers' of a type (line 226)
        handlers_2926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 18), t_2925, 'handlers')
        # Assigning a type to the variable 'handlers_2926' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'handlers_2926', handlers_2926)
        # Testing if the for loop is going to be iterated (line 226)
        # Testing the type of a for loop iterable (line 226)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 226, 8), handlers_2926)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 226, 8), handlers_2926):
            # Getting the type of the for loop variable (line 226)
            for_loop_var_2927 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 226, 8), handlers_2926)
            # Assigning a type to the variable 'ex' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'ex', for_loop_var_2927)
            # SSA begins for a for statement (line 226)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 227)
            # Processing the call arguments (line 227)
            # Getting the type of 'ex' (line 227)
            ex_2930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 23), 'ex', False)
            # Processing the call keyword arguments (line 227)
            kwargs_2931 = {}
            # Getting the type of 'self' (line 227)
            self_2928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 227)
            visit_2929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), self_2928, 'visit')
            # Calling visit(args, kwargs) (line 227)
            visit_call_result_2932 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), visit_2929, *[ex_2930], **kwargs_2931)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 't' (line 228)
        t_2933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 't')
        # Obtaining the member 'orelse' of a type (line 228)
        orelse_2934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 11), t_2933, 'orelse')
        # Testing if the type of an if condition is none (line 228)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 228, 8), orelse_2934):
            pass
        else:
            
            # Testing the type of an if condition (line 228)
            if_condition_2935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 8), orelse_2934)
            # Assigning a type to the variable 'if_condition_2935' (line 228)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'if_condition_2935', if_condition_2935)
            # SSA begins for if statement (line 228)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fill(...): (line 229)
            # Processing the call arguments (line 229)
            str_2938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 22), 'str', 'else')
            # Processing the call keyword arguments (line 229)
            kwargs_2939 = {}
            # Getting the type of 'self' (line 229)
            self_2936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 229)
            fill_2937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), self_2936, 'fill')
            # Calling fill(args, kwargs) (line 229)
            fill_call_result_2940 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), fill_2937, *[str_2938], **kwargs_2939)
            
            
            # Call to enter(...): (line 230)
            # Processing the call keyword arguments (line 230)
            kwargs_2943 = {}
            # Getting the type of 'self' (line 230)
            self_2941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 230)
            enter_2942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), self_2941, 'enter')
            # Calling enter(args, kwargs) (line 230)
            enter_call_result_2944 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), enter_2942, *[], **kwargs_2943)
            
            
            # Call to visit(...): (line 231)
            # Processing the call arguments (line 231)
            # Getting the type of 't' (line 231)
            t_2947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 't', False)
            # Obtaining the member 'orelse' of a type (line 231)
            orelse_2948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 23), t_2947, 'orelse')
            # Processing the call keyword arguments (line 231)
            kwargs_2949 = {}
            # Getting the type of 'self' (line 231)
            self_2945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 231)
            visit_2946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), self_2945, 'visit')
            # Calling visit(args, kwargs) (line 231)
            visit_call_result_2950 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), visit_2946, *[orelse_2948], **kwargs_2949)
            
            
            # Call to leave(...): (line 232)
            # Processing the call keyword arguments (line 232)
            kwargs_2953 = {}
            # Getting the type of 'self' (line 232)
            self_2951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 232)
            leave_2952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), self_2951, 'leave')
            # Calling leave(args, kwargs) (line 232)
            leave_call_result_2954 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), leave_2952, *[], **kwargs_2953)
            
            # SSA join for if statement (line 228)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_TryExcept(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_TryExcept' in the type store
        # Getting the type of 'stypy_return_type' (line 220)
        stypy_return_type_2955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2955)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_TryExcept'
        return stypy_return_type_2955


    @norecursion
    def visit_TryFinally(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_TryFinally'
        module_type_store = module_type_store.open_function_context('visit_TryFinally', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_TryFinally.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_TryFinally.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_TryFinally.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_TryFinally.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_TryFinally')
        PythonSrcGeneratorVisitor.visit_TryFinally.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_TryFinally.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_TryFinally.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_TryFinally.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_TryFinally.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_TryFinally.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_TryFinally.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_TryFinally', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_TryFinally', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_TryFinally(...)' code ##################

        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 235)
        # Processing the call arguments (line 235)
        # Getting the type of 't' (line 235)
        t_2957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 't', False)
        # Obtaining the member 'body' of a type (line 235)
        body_2958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 15), t_2957, 'body')
        # Processing the call keyword arguments (line 235)
        kwargs_2959 = {}
        # Getting the type of 'len' (line 235)
        len_2956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'len', False)
        # Calling len(args, kwargs) (line 235)
        len_call_result_2960 = invoke(stypy.reporting.localization.Localization(__file__, 235, 11), len_2956, *[body_2958], **kwargs_2959)
        
        int_2961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 26), 'int')
        # Applying the binary operator '==' (line 235)
        result_eq_2962 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 11), '==', len_call_result_2960, int_2961)
        
        
        # Call to isinstance(...): (line 235)
        # Processing the call arguments (line 235)
        
        # Obtaining the type of the subscript
        int_2964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 50), 'int')
        # Getting the type of 't' (line 235)
        t_2965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 43), 't', False)
        # Obtaining the member 'body' of a type (line 235)
        body_2966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 43), t_2965, 'body')
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___2967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 43), body_2966, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_2968 = invoke(stypy.reporting.localization.Localization(__file__, 235, 43), getitem___2967, int_2964)
        
        # Getting the type of 'ast' (line 235)
        ast_2969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 54), 'ast', False)
        # Obtaining the member 'TryExcept' of a type (line 235)
        TryExcept_2970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 54), ast_2969, 'TryExcept')
        # Processing the call keyword arguments (line 235)
        kwargs_2971 = {}
        # Getting the type of 'isinstance' (line 235)
        isinstance_2963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 32), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 235)
        isinstance_call_result_2972 = invoke(stypy.reporting.localization.Localization(__file__, 235, 32), isinstance_2963, *[subscript_call_result_2968, TryExcept_2970], **kwargs_2971)
        
        # Applying the binary operator 'and' (line 235)
        result_and_keyword_2973 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 11), 'and', result_eq_2962, isinstance_call_result_2972)
        
        # Testing if the type of an if condition is none (line 235)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 235, 8), result_and_keyword_2973):
            
            # Call to fill(...): (line 239)
            # Processing the call arguments (line 239)
            str_2983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 22), 'str', 'try')
            # Processing the call keyword arguments (line 239)
            kwargs_2984 = {}
            # Getting the type of 'self' (line 239)
            self_2981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 239)
            fill_2982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), self_2981, 'fill')
            # Calling fill(args, kwargs) (line 239)
            fill_call_result_2985 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), fill_2982, *[str_2983], **kwargs_2984)
            
            
            # Call to enter(...): (line 240)
            # Processing the call keyword arguments (line 240)
            kwargs_2988 = {}
            # Getting the type of 'self' (line 240)
            self_2986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 240)
            enter_2987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), self_2986, 'enter')
            # Calling enter(args, kwargs) (line 240)
            enter_call_result_2989 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), enter_2987, *[], **kwargs_2988)
            
            
            # Call to visit(...): (line 241)
            # Processing the call arguments (line 241)
            # Getting the type of 't' (line 241)
            t_2992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 't', False)
            # Obtaining the member 'body' of a type (line 241)
            body_2993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 23), t_2992, 'body')
            # Processing the call keyword arguments (line 241)
            kwargs_2994 = {}
            # Getting the type of 'self' (line 241)
            self_2990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 241)
            visit_2991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), self_2990, 'visit')
            # Calling visit(args, kwargs) (line 241)
            visit_call_result_2995 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), visit_2991, *[body_2993], **kwargs_2994)
            
            
            # Call to leave(...): (line 242)
            # Processing the call keyword arguments (line 242)
            kwargs_2998 = {}
            # Getting the type of 'self' (line 242)
            self_2996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 242)
            leave_2997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), self_2996, 'leave')
            # Calling leave(args, kwargs) (line 242)
            leave_call_result_2999 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), leave_2997, *[], **kwargs_2998)
            
        else:
            
            # Testing the type of an if condition (line 235)
            if_condition_2974 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 8), result_and_keyword_2973)
            # Assigning a type to the variable 'if_condition_2974' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'if_condition_2974', if_condition_2974)
            # SSA begins for if statement (line 235)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to visit(...): (line 237)
            # Processing the call arguments (line 237)
            # Getting the type of 't' (line 237)
            t_2977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 23), 't', False)
            # Obtaining the member 'body' of a type (line 237)
            body_2978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 23), t_2977, 'body')
            # Processing the call keyword arguments (line 237)
            kwargs_2979 = {}
            # Getting the type of 'self' (line 237)
            self_2975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 237)
            visit_2976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), self_2975, 'visit')
            # Calling visit(args, kwargs) (line 237)
            visit_call_result_2980 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), visit_2976, *[body_2978], **kwargs_2979)
            
            # SSA branch for the else part of an if statement (line 235)
            module_type_store.open_ssa_branch('else')
            
            # Call to fill(...): (line 239)
            # Processing the call arguments (line 239)
            str_2983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 22), 'str', 'try')
            # Processing the call keyword arguments (line 239)
            kwargs_2984 = {}
            # Getting the type of 'self' (line 239)
            self_2981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 239)
            fill_2982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), self_2981, 'fill')
            # Calling fill(args, kwargs) (line 239)
            fill_call_result_2985 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), fill_2982, *[str_2983], **kwargs_2984)
            
            
            # Call to enter(...): (line 240)
            # Processing the call keyword arguments (line 240)
            kwargs_2988 = {}
            # Getting the type of 'self' (line 240)
            self_2986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 240)
            enter_2987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), self_2986, 'enter')
            # Calling enter(args, kwargs) (line 240)
            enter_call_result_2989 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), enter_2987, *[], **kwargs_2988)
            
            
            # Call to visit(...): (line 241)
            # Processing the call arguments (line 241)
            # Getting the type of 't' (line 241)
            t_2992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 't', False)
            # Obtaining the member 'body' of a type (line 241)
            body_2993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 23), t_2992, 'body')
            # Processing the call keyword arguments (line 241)
            kwargs_2994 = {}
            # Getting the type of 'self' (line 241)
            self_2990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 241)
            visit_2991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), self_2990, 'visit')
            # Calling visit(args, kwargs) (line 241)
            visit_call_result_2995 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), visit_2991, *[body_2993], **kwargs_2994)
            
            
            # Call to leave(...): (line 242)
            # Processing the call keyword arguments (line 242)
            kwargs_2998 = {}
            # Getting the type of 'self' (line 242)
            self_2996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 242)
            leave_2997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), self_2996, 'leave')
            # Calling leave(args, kwargs) (line 242)
            leave_call_result_2999 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), leave_2997, *[], **kwargs_2998)
            
            # SSA join for if statement (line 235)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to fill(...): (line 244)
        # Processing the call arguments (line 244)
        str_3002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 18), 'str', 'finally')
        # Processing the call keyword arguments (line 244)
        kwargs_3003 = {}
        # Getting the type of 'self' (line 244)
        self_3000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 244)
        fill_3001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_3000, 'fill')
        # Calling fill(args, kwargs) (line 244)
        fill_call_result_3004 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), fill_3001, *[str_3002], **kwargs_3003)
        
        
        # Call to enter(...): (line 245)
        # Processing the call keyword arguments (line 245)
        kwargs_3007 = {}
        # Getting the type of 'self' (line 245)
        self_3005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 245)
        enter_3006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), self_3005, 'enter')
        # Calling enter(args, kwargs) (line 245)
        enter_call_result_3008 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), enter_3006, *[], **kwargs_3007)
        
        
        # Call to visit(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 't' (line 246)
        t_3011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 't', False)
        # Obtaining the member 'finalbody' of a type (line 246)
        finalbody_3012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 19), t_3011, 'finalbody')
        # Processing the call keyword arguments (line 246)
        kwargs_3013 = {}
        # Getting the type of 'self' (line 246)
        self_3009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 246)
        visit_3010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), self_3009, 'visit')
        # Calling visit(args, kwargs) (line 246)
        visit_call_result_3014 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), visit_3010, *[finalbody_3012], **kwargs_3013)
        
        
        # Call to leave(...): (line 247)
        # Processing the call keyword arguments (line 247)
        kwargs_3017 = {}
        # Getting the type of 'self' (line 247)
        self_3015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 247)
        leave_3016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), self_3015, 'leave')
        # Calling leave(args, kwargs) (line 247)
        leave_call_result_3018 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), leave_3016, *[], **kwargs_3017)
        
        
        # ################# End of 'visit_TryFinally(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_TryFinally' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_3019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3019)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_TryFinally'
        return stypy_return_type_3019


    @norecursion
    def visit_ExceptHandler(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ExceptHandler'
        module_type_store = module_type_store.open_function_context('visit_ExceptHandler', 249, 4, False)
        # Assigning a type to the variable 'self' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_ExceptHandler.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_ExceptHandler.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_ExceptHandler.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_ExceptHandler.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_ExceptHandler')
        PythonSrcGeneratorVisitor.visit_ExceptHandler.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_ExceptHandler.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_ExceptHandler.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_ExceptHandler.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_ExceptHandler.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_ExceptHandler.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_ExceptHandler.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_ExceptHandler', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_ExceptHandler', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_ExceptHandler(...)' code ##################

        
        # Call to fill(...): (line 250)
        # Processing the call arguments (line 250)
        str_3022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 18), 'str', 'except')
        # Processing the call keyword arguments (line 250)
        kwargs_3023 = {}
        # Getting the type of 'self' (line 250)
        self_3020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 250)
        fill_3021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_3020, 'fill')
        # Calling fill(args, kwargs) (line 250)
        fill_call_result_3024 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), fill_3021, *[str_3022], **kwargs_3023)
        
        # Getting the type of 't' (line 251)
        t_3025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 't')
        # Obtaining the member 'type' of a type (line 251)
        type_3026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 11), t_3025, 'type')
        # Testing if the type of an if condition is none (line 251)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 251, 8), type_3026):
            pass
        else:
            
            # Testing the type of an if condition (line 251)
            if_condition_3027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 8), type_3026)
            # Assigning a type to the variable 'if_condition_3027' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'if_condition_3027', if_condition_3027)
            # SSA begins for if statement (line 251)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 252)
            # Processing the call arguments (line 252)
            str_3030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 23), 'str', ' ')
            # Processing the call keyword arguments (line 252)
            kwargs_3031 = {}
            # Getting the type of 'self' (line 252)
            self_3028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 252)
            write_3029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 12), self_3028, 'write')
            # Calling write(args, kwargs) (line 252)
            write_call_result_3032 = invoke(stypy.reporting.localization.Localization(__file__, 252, 12), write_3029, *[str_3030], **kwargs_3031)
            
            
            # Call to visit(...): (line 253)
            # Processing the call arguments (line 253)
            # Getting the type of 't' (line 253)
            t_3035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 23), 't', False)
            # Obtaining the member 'type' of a type (line 253)
            type_3036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 23), t_3035, 'type')
            # Processing the call keyword arguments (line 253)
            kwargs_3037 = {}
            # Getting the type of 'self' (line 253)
            self_3033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 253)
            visit_3034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 12), self_3033, 'visit')
            # Calling visit(args, kwargs) (line 253)
            visit_call_result_3038 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), visit_3034, *[type_3036], **kwargs_3037)
            
            # SSA join for if statement (line 251)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 254)
        t_3039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 11), 't')
        # Obtaining the member 'name' of a type (line 254)
        name_3040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 11), t_3039, 'name')
        # Testing if the type of an if condition is none (line 254)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 254, 8), name_3040):
            pass
        else:
            
            # Testing the type of an if condition (line 254)
            if_condition_3041 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 8), name_3040)
            # Assigning a type to the variable 'if_condition_3041' (line 254)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'if_condition_3041', if_condition_3041)
            # SSA begins for if statement (line 254)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 255)
            # Processing the call arguments (line 255)
            str_3044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 23), 'str', ' as ')
            # Processing the call keyword arguments (line 255)
            kwargs_3045 = {}
            # Getting the type of 'self' (line 255)
            self_3042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 255)
            write_3043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), self_3042, 'write')
            # Calling write(args, kwargs) (line 255)
            write_call_result_3046 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), write_3043, *[str_3044], **kwargs_3045)
            
            
            # Call to visit(...): (line 256)
            # Processing the call arguments (line 256)
            # Getting the type of 't' (line 256)
            t_3049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 't', False)
            # Obtaining the member 'name' of a type (line 256)
            name_3050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 23), t_3049, 'name')
            # Processing the call keyword arguments (line 256)
            kwargs_3051 = {}
            # Getting the type of 'self' (line 256)
            self_3047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 256)
            visit_3048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), self_3047, 'visit')
            # Calling visit(args, kwargs) (line 256)
            visit_call_result_3052 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), visit_3048, *[name_3050], **kwargs_3051)
            
            # SSA join for if statement (line 254)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to enter(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_3055 = {}
        # Getting the type of 'self' (line 257)
        self_3053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 257)
        enter_3054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_3053, 'enter')
        # Calling enter(args, kwargs) (line 257)
        enter_call_result_3056 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), enter_3054, *[], **kwargs_3055)
        
        
        # Call to visit(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 't' (line 258)
        t_3059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 19), 't', False)
        # Obtaining the member 'body' of a type (line 258)
        body_3060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 19), t_3059, 'body')
        # Processing the call keyword arguments (line 258)
        kwargs_3061 = {}
        # Getting the type of 'self' (line 258)
        self_3057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 258)
        visit_3058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), self_3057, 'visit')
        # Calling visit(args, kwargs) (line 258)
        visit_call_result_3062 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), visit_3058, *[body_3060], **kwargs_3061)
        
        
        # Call to leave(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_3065 = {}
        # Getting the type of 'self' (line 259)
        self_3063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 259)
        leave_3064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_3063, 'leave')
        # Calling leave(args, kwargs) (line 259)
        leave_call_result_3066 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), leave_3064, *[], **kwargs_3065)
        
        
        # ################# End of 'visit_ExceptHandler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ExceptHandler' in the type store
        # Getting the type of 'stypy_return_type' (line 249)
        stypy_return_type_3067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3067)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ExceptHandler'
        return stypy_return_type_3067


    @norecursion
    def visit_ClassDef(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ClassDef'
        module_type_store = module_type_store.open_function_context('visit_ClassDef', 261, 4, False)
        # Assigning a type to the variable 'self' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_ClassDef.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_ClassDef.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_ClassDef.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_ClassDef.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_ClassDef')
        PythonSrcGeneratorVisitor.visit_ClassDef.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_ClassDef.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_ClassDef.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_ClassDef.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_ClassDef.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_ClassDef.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_ClassDef.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_ClassDef', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_ClassDef', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_ClassDef(...)' code ##################

        
        # Call to write(...): (line 262)
        # Processing the call arguments (line 262)
        str_3070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 19), 'str', '\n')
        # Processing the call keyword arguments (line 262)
        kwargs_3071 = {}
        # Getting the type of 'self' (line 262)
        self_3068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 262)
        write_3069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_3068, 'write')
        # Calling write(args, kwargs) (line 262)
        write_call_result_3072 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), write_3069, *[str_3070], **kwargs_3071)
        
        
        # Getting the type of 't' (line 263)
        t_3073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 20), 't')
        # Obtaining the member 'decorator_list' of a type (line 263)
        decorator_list_3074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 20), t_3073, 'decorator_list')
        # Assigning a type to the variable 'decorator_list_3074' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'decorator_list_3074', decorator_list_3074)
        # Testing if the for loop is going to be iterated (line 263)
        # Testing the type of a for loop iterable (line 263)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 263, 8), decorator_list_3074)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 263, 8), decorator_list_3074):
            # Getting the type of the for loop variable (line 263)
            for_loop_var_3075 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 263, 8), decorator_list_3074)
            # Assigning a type to the variable 'deco' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'deco', for_loop_var_3075)
            # SSA begins for a for statement (line 263)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to fill(...): (line 264)
            # Processing the call arguments (line 264)
            str_3078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 22), 'str', '@')
            # Processing the call keyword arguments (line 264)
            kwargs_3079 = {}
            # Getting the type of 'self' (line 264)
            self_3076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 264)
            fill_3077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), self_3076, 'fill')
            # Calling fill(args, kwargs) (line 264)
            fill_call_result_3080 = invoke(stypy.reporting.localization.Localization(__file__, 264, 12), fill_3077, *[str_3078], **kwargs_3079)
            
            
            # Call to visit(...): (line 265)
            # Processing the call arguments (line 265)
            # Getting the type of 'deco' (line 265)
            deco_3083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 23), 'deco', False)
            # Processing the call keyword arguments (line 265)
            kwargs_3084 = {}
            # Getting the type of 'self' (line 265)
            self_3081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 265)
            visit_3082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), self_3081, 'visit')
            # Calling visit(args, kwargs) (line 265)
            visit_call_result_3085 = invoke(stypy.reporting.localization.Localization(__file__, 265, 12), visit_3082, *[deco_3083], **kwargs_3084)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to fill(...): (line 266)
        # Processing the call arguments (line 266)
        str_3088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 18), 'str', 'class ')
        # Getting the type of 't' (line 266)
        t_3089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 29), 't', False)
        # Obtaining the member 'name' of a type (line 266)
        name_3090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 29), t_3089, 'name')
        # Applying the binary operator '+' (line 266)
        result_add_3091 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 18), '+', str_3088, name_3090)
        
        # Processing the call keyword arguments (line 266)
        kwargs_3092 = {}
        # Getting the type of 'self' (line 266)
        self_3086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 266)
        fill_3087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), self_3086, 'fill')
        # Calling fill(args, kwargs) (line 266)
        fill_call_result_3093 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), fill_3087, *[result_add_3091], **kwargs_3092)
        
        # Getting the type of 't' (line 267)
        t_3094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), 't')
        # Obtaining the member 'bases' of a type (line 267)
        bases_3095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 11), t_3094, 'bases')
        # Testing if the type of an if condition is none (line 267)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 267, 8), bases_3095):
            pass
        else:
            
            # Testing the type of an if condition (line 267)
            if_condition_3096 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 8), bases_3095)
            # Assigning a type to the variable 'if_condition_3096' (line 267)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'if_condition_3096', if_condition_3096)
            # SSA begins for if statement (line 267)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 268)
            # Processing the call arguments (line 268)
            str_3099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 23), 'str', '(')
            # Processing the call keyword arguments (line 268)
            kwargs_3100 = {}
            # Getting the type of 'self' (line 268)
            self_3097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 268)
            write_3098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 12), self_3097, 'write')
            # Calling write(args, kwargs) (line 268)
            write_call_result_3101 = invoke(stypy.reporting.localization.Localization(__file__, 268, 12), write_3098, *[str_3099], **kwargs_3100)
            
            
            # Getting the type of 't' (line 269)
            t_3102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 21), 't')
            # Obtaining the member 'bases' of a type (line 269)
            bases_3103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 21), t_3102, 'bases')
            # Assigning a type to the variable 'bases_3103' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'bases_3103', bases_3103)
            # Testing if the for loop is going to be iterated (line 269)
            # Testing the type of a for loop iterable (line 269)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 269, 12), bases_3103)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 269, 12), bases_3103):
                # Getting the type of the for loop variable (line 269)
                for_loop_var_3104 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 269, 12), bases_3103)
                # Assigning a type to the variable 'a' (line 269)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'a', for_loop_var_3104)
                # SSA begins for a for statement (line 269)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to visit(...): (line 270)
                # Processing the call arguments (line 270)
                # Getting the type of 'a' (line 270)
                a_3107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 27), 'a', False)
                # Processing the call keyword arguments (line 270)
                kwargs_3108 = {}
                # Getting the type of 'self' (line 270)
                self_3105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'self', False)
                # Obtaining the member 'visit' of a type (line 270)
                visit_3106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), self_3105, 'visit')
                # Calling visit(args, kwargs) (line 270)
                visit_call_result_3109 = invoke(stypy.reporting.localization.Localization(__file__, 270, 16), visit_3106, *[a_3107], **kwargs_3108)
                
                
                # Call to write(...): (line 271)
                # Processing the call arguments (line 271)
                str_3112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 27), 'str', ', ')
                # Processing the call keyword arguments (line 271)
                kwargs_3113 = {}
                # Getting the type of 'self' (line 271)
                self_3110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 271)
                write_3111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), self_3110, 'write')
                # Calling write(args, kwargs) (line 271)
                write_call_result_3114 = invoke(stypy.reporting.localization.Localization(__file__, 271, 16), write_3111, *[str_3112], **kwargs_3113)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to write(...): (line 272)
            # Processing the call arguments (line 272)
            str_3117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 23), 'str', ')')
            # Processing the call keyword arguments (line 272)
            kwargs_3118 = {}
            # Getting the type of 'self' (line 272)
            self_3115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 272)
            write_3116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 12), self_3115, 'write')
            # Calling write(args, kwargs) (line 272)
            write_call_result_3119 = invoke(stypy.reporting.localization.Localization(__file__, 272, 12), write_3116, *[str_3117], **kwargs_3118)
            
            # SSA join for if statement (line 267)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to enter(...): (line 273)
        # Processing the call keyword arguments (line 273)
        kwargs_3122 = {}
        # Getting the type of 'self' (line 273)
        self_3120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 273)
        enter_3121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), self_3120, 'enter')
        # Calling enter(args, kwargs) (line 273)
        enter_call_result_3123 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), enter_3121, *[], **kwargs_3122)
        
        
        # Call to visit(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 't' (line 274)
        t_3126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 't', False)
        # Obtaining the member 'body' of a type (line 274)
        body_3127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 19), t_3126, 'body')
        # Processing the call keyword arguments (line 274)
        kwargs_3128 = {}
        # Getting the type of 'self' (line 274)
        self_3124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 274)
        visit_3125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), self_3124, 'visit')
        # Calling visit(args, kwargs) (line 274)
        visit_call_result_3129 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), visit_3125, *[body_3127], **kwargs_3128)
        
        
        # Call to leave(...): (line 275)
        # Processing the call keyword arguments (line 275)
        kwargs_3132 = {}
        # Getting the type of 'self' (line 275)
        self_3130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 275)
        leave_3131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), self_3130, 'leave')
        # Calling leave(args, kwargs) (line 275)
        leave_call_result_3133 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), leave_3131, *[], **kwargs_3132)
        
        
        # ################# End of 'visit_ClassDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ClassDef' in the type store
        # Getting the type of 'stypy_return_type' (line 261)
        stypy_return_type_3134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3134)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ClassDef'
        return stypy_return_type_3134


    @norecursion
    def visit_FunctionDef(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_FunctionDef'
        module_type_store = module_type_store.open_function_context('visit_FunctionDef', 277, 4, False)
        # Assigning a type to the variable 'self' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_FunctionDef')
        PythonSrcGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_FunctionDef.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_FunctionDef', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_FunctionDef', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_FunctionDef(...)' code ##################

        
        # Call to write(...): (line 278)
        # Processing the call arguments (line 278)
        str_3137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 19), 'str', '\n')
        # Processing the call keyword arguments (line 278)
        kwargs_3138 = {}
        # Getting the type of 'self' (line 278)
        self_3135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 278)
        write_3136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), self_3135, 'write')
        # Calling write(args, kwargs) (line 278)
        write_call_result_3139 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), write_3136, *[str_3137], **kwargs_3138)
        
        
        # Getting the type of 't' (line 279)
        t_3140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 't')
        # Obtaining the member 'decorator_list' of a type (line 279)
        decorator_list_3141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 20), t_3140, 'decorator_list')
        # Assigning a type to the variable 'decorator_list_3141' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'decorator_list_3141', decorator_list_3141)
        # Testing if the for loop is going to be iterated (line 279)
        # Testing the type of a for loop iterable (line 279)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 279, 8), decorator_list_3141)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 279, 8), decorator_list_3141):
            # Getting the type of the for loop variable (line 279)
            for_loop_var_3142 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 279, 8), decorator_list_3141)
            # Assigning a type to the variable 'deco' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'deco', for_loop_var_3142)
            # SSA begins for a for statement (line 279)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to fill(...): (line 280)
            # Processing the call arguments (line 280)
            str_3145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 22), 'str', '@')
            # Processing the call keyword arguments (line 280)
            kwargs_3146 = {}
            # Getting the type of 'self' (line 280)
            self_3143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 280)
            fill_3144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), self_3143, 'fill')
            # Calling fill(args, kwargs) (line 280)
            fill_call_result_3147 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), fill_3144, *[str_3145], **kwargs_3146)
            
            
            # Call to visit(...): (line 281)
            # Processing the call arguments (line 281)
            # Getting the type of 'deco' (line 281)
            deco_3150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 23), 'deco', False)
            # Processing the call keyword arguments (line 281)
            kwargs_3151 = {}
            # Getting the type of 'self' (line 281)
            self_3148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 281)
            visit_3149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), self_3148, 'visit')
            # Calling visit(args, kwargs) (line 281)
            visit_call_result_3152 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), visit_3149, *[deco_3150], **kwargs_3151)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to fill(...): (line 282)
        # Processing the call arguments (line 282)
        str_3155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 18), 'str', 'def ')
        # Getting the type of 't' (line 282)
        t_3156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 27), 't', False)
        # Obtaining the member 'name' of a type (line 282)
        name_3157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 27), t_3156, 'name')
        # Applying the binary operator '+' (line 282)
        result_add_3158 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 18), '+', str_3155, name_3157)
        
        str_3159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 36), 'str', '(')
        # Applying the binary operator '+' (line 282)
        result_add_3160 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 34), '+', result_add_3158, str_3159)
        
        # Processing the call keyword arguments (line 282)
        kwargs_3161 = {}
        # Getting the type of 'self' (line 282)
        self_3153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 282)
        fill_3154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), self_3153, 'fill')
        # Calling fill(args, kwargs) (line 282)
        fill_call_result_3162 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), fill_3154, *[result_add_3160], **kwargs_3161)
        
        
        # Call to visit(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 't' (line 283)
        t_3165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 't', False)
        # Obtaining the member 'args' of a type (line 283)
        args_3166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 19), t_3165, 'args')
        # Processing the call keyword arguments (line 283)
        kwargs_3167 = {}
        # Getting the type of 'self' (line 283)
        self_3163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 283)
        visit_3164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), self_3163, 'visit')
        # Calling visit(args, kwargs) (line 283)
        visit_call_result_3168 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), visit_3164, *[args_3166], **kwargs_3167)
        
        
        # Call to write(...): (line 284)
        # Processing the call arguments (line 284)
        str_3171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 19), 'str', ')')
        # Processing the call keyword arguments (line 284)
        kwargs_3172 = {}
        # Getting the type of 'self' (line 284)
        self_3169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 284)
        write_3170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), self_3169, 'write')
        # Calling write(args, kwargs) (line 284)
        write_call_result_3173 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), write_3170, *[str_3171], **kwargs_3172)
        
        
        # Call to enter(...): (line 285)
        # Processing the call keyword arguments (line 285)
        kwargs_3176 = {}
        # Getting the type of 'self' (line 285)
        self_3174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 285)
        enter_3175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), self_3174, 'enter')
        # Calling enter(args, kwargs) (line 285)
        enter_call_result_3177 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), enter_3175, *[], **kwargs_3176)
        
        
        # Call to visit(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 't' (line 286)
        t_3180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 't', False)
        # Obtaining the member 'body' of a type (line 286)
        body_3181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 19), t_3180, 'body')
        # Processing the call keyword arguments (line 286)
        kwargs_3182 = {}
        # Getting the type of 'self' (line 286)
        self_3178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 286)
        visit_3179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), self_3178, 'visit')
        # Calling visit(args, kwargs) (line 286)
        visit_call_result_3183 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), visit_3179, *[body_3181], **kwargs_3182)
        
        
        # Call to write(...): (line 287)
        # Processing the call arguments (line 287)
        str_3186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 19), 'str', '\n')
        # Processing the call keyword arguments (line 287)
        kwargs_3187 = {}
        # Getting the type of 'self' (line 287)
        self_3184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 287)
        write_3185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_3184, 'write')
        # Calling write(args, kwargs) (line 287)
        write_call_result_3188 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), write_3185, *[str_3186], **kwargs_3187)
        
        
        # Call to leave(...): (line 288)
        # Processing the call keyword arguments (line 288)
        kwargs_3191 = {}
        # Getting the type of 'self' (line 288)
        self_3189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 288)
        leave_3190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), self_3189, 'leave')
        # Calling leave(args, kwargs) (line 288)
        leave_call_result_3192 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), leave_3190, *[], **kwargs_3191)
        
        
        # ################# End of 'visit_FunctionDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_FunctionDef' in the type store
        # Getting the type of 'stypy_return_type' (line 277)
        stypy_return_type_3193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3193)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_FunctionDef'
        return stypy_return_type_3193


    @norecursion
    def visit_For(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_For'
        module_type_store = module_type_store.open_function_context('visit_For', 290, 4, False)
        # Assigning a type to the variable 'self' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_For.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_For.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_For.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_For.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_For')
        PythonSrcGeneratorVisitor.visit_For.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_For.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_For.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_For.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_For.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_For.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_For.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_For', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_For', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_For(...)' code ##################

        
        # Call to fill(...): (line 291)
        # Processing the call arguments (line 291)
        str_3196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 18), 'str', 'for ')
        # Processing the call keyword arguments (line 291)
        kwargs_3197 = {}
        # Getting the type of 'self' (line 291)
        self_3194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 291)
        fill_3195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), self_3194, 'fill')
        # Calling fill(args, kwargs) (line 291)
        fill_call_result_3198 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), fill_3195, *[str_3196], **kwargs_3197)
        
        
        # Call to visit(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 't' (line 292)
        t_3201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 19), 't', False)
        # Obtaining the member 'target' of a type (line 292)
        target_3202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 19), t_3201, 'target')
        # Processing the call keyword arguments (line 292)
        kwargs_3203 = {}
        # Getting the type of 'self' (line 292)
        self_3199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 292)
        visit_3200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), self_3199, 'visit')
        # Calling visit(args, kwargs) (line 292)
        visit_call_result_3204 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), visit_3200, *[target_3202], **kwargs_3203)
        
        
        # Call to write(...): (line 293)
        # Processing the call arguments (line 293)
        str_3207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 19), 'str', ' in ')
        # Processing the call keyword arguments (line 293)
        kwargs_3208 = {}
        # Getting the type of 'self' (line 293)
        self_3205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 293)
        write_3206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), self_3205, 'write')
        # Calling write(args, kwargs) (line 293)
        write_call_result_3209 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), write_3206, *[str_3207], **kwargs_3208)
        
        
        # Call to visit(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 't' (line 294)
        t_3212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 't', False)
        # Obtaining the member 'iter' of a type (line 294)
        iter_3213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 19), t_3212, 'iter')
        # Processing the call keyword arguments (line 294)
        kwargs_3214 = {}
        # Getting the type of 'self' (line 294)
        self_3210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 294)
        visit_3211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), self_3210, 'visit')
        # Calling visit(args, kwargs) (line 294)
        visit_call_result_3215 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), visit_3211, *[iter_3213], **kwargs_3214)
        
        
        # Call to enter(...): (line 295)
        # Processing the call keyword arguments (line 295)
        kwargs_3218 = {}
        # Getting the type of 'self' (line 295)
        self_3216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 295)
        enter_3217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), self_3216, 'enter')
        # Calling enter(args, kwargs) (line 295)
        enter_call_result_3219 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), enter_3217, *[], **kwargs_3218)
        
        
        # Call to visit(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 't' (line 296)
        t_3222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 't', False)
        # Obtaining the member 'body' of a type (line 296)
        body_3223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 19), t_3222, 'body')
        # Processing the call keyword arguments (line 296)
        kwargs_3224 = {}
        # Getting the type of 'self' (line 296)
        self_3220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 296)
        visit_3221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), self_3220, 'visit')
        # Calling visit(args, kwargs) (line 296)
        visit_call_result_3225 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), visit_3221, *[body_3223], **kwargs_3224)
        
        
        # Call to leave(...): (line 297)
        # Processing the call keyword arguments (line 297)
        kwargs_3228 = {}
        # Getting the type of 'self' (line 297)
        self_3226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 297)
        leave_3227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), self_3226, 'leave')
        # Calling leave(args, kwargs) (line 297)
        leave_call_result_3229 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), leave_3227, *[], **kwargs_3228)
        
        # Getting the type of 't' (line 298)
        t_3230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 't')
        # Obtaining the member 'orelse' of a type (line 298)
        orelse_3231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 11), t_3230, 'orelse')
        # Testing if the type of an if condition is none (line 298)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 298, 8), orelse_3231):
            pass
        else:
            
            # Testing the type of an if condition (line 298)
            if_condition_3232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 8), orelse_3231)
            # Assigning a type to the variable 'if_condition_3232' (line 298)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'if_condition_3232', if_condition_3232)
            # SSA begins for if statement (line 298)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fill(...): (line 299)
            # Processing the call arguments (line 299)
            str_3235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 22), 'str', 'else')
            # Processing the call keyword arguments (line 299)
            kwargs_3236 = {}
            # Getting the type of 'self' (line 299)
            self_3233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 299)
            fill_3234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), self_3233, 'fill')
            # Calling fill(args, kwargs) (line 299)
            fill_call_result_3237 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), fill_3234, *[str_3235], **kwargs_3236)
            
            
            # Call to enter(...): (line 300)
            # Processing the call keyword arguments (line 300)
            kwargs_3240 = {}
            # Getting the type of 'self' (line 300)
            self_3238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 300)
            enter_3239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), self_3238, 'enter')
            # Calling enter(args, kwargs) (line 300)
            enter_call_result_3241 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), enter_3239, *[], **kwargs_3240)
            
            
            # Call to visit(...): (line 301)
            # Processing the call arguments (line 301)
            # Getting the type of 't' (line 301)
            t_3244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 23), 't', False)
            # Obtaining the member 'orelse' of a type (line 301)
            orelse_3245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 23), t_3244, 'orelse')
            # Processing the call keyword arguments (line 301)
            kwargs_3246 = {}
            # Getting the type of 'self' (line 301)
            self_3242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 301)
            visit_3243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), self_3242, 'visit')
            # Calling visit(args, kwargs) (line 301)
            visit_call_result_3247 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), visit_3243, *[orelse_3245], **kwargs_3246)
            
            
            # Call to leave(...): (line 302)
            # Processing the call keyword arguments (line 302)
            kwargs_3250 = {}
            # Getting the type of 'self' (line 302)
            self_3248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 302)
            leave_3249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 12), self_3248, 'leave')
            # Calling leave(args, kwargs) (line 302)
            leave_call_result_3251 = invoke(stypy.reporting.localization.Localization(__file__, 302, 12), leave_3249, *[], **kwargs_3250)
            
            # SSA join for if statement (line 298)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_For(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_For' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_3252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3252)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_For'
        return stypy_return_type_3252


    @norecursion
    def visit_If(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_If'
        module_type_store = module_type_store.open_function_context('visit_If', 304, 4, False)
        # Assigning a type to the variable 'self' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_If.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_If.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_If.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_If.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_If')
        PythonSrcGeneratorVisitor.visit_If.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_If.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_If.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_If.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_If.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_If.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_If.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_If', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_If', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_If(...)' code ##################

        
        # Call to write(...): (line 305)
        # Processing the call arguments (line 305)
        str_3255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 19), 'str', '\n')
        # Processing the call keyword arguments (line 305)
        kwargs_3256 = {}
        # Getting the type of 'self' (line 305)
        self_3253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 305)
        write_3254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 8), self_3253, 'write')
        # Calling write(args, kwargs) (line 305)
        write_call_result_3257 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), write_3254, *[str_3255], **kwargs_3256)
        
        
        # Call to fill(...): (line 306)
        # Processing the call arguments (line 306)
        str_3260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 18), 'str', 'if ')
        # Processing the call keyword arguments (line 306)
        kwargs_3261 = {}
        # Getting the type of 'self' (line 306)
        self_3258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 306)
        fill_3259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), self_3258, 'fill')
        # Calling fill(args, kwargs) (line 306)
        fill_call_result_3262 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), fill_3259, *[str_3260], **kwargs_3261)
        
        
        # Call to visit(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 't' (line 307)
        t_3265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 't', False)
        # Obtaining the member 'test' of a type (line 307)
        test_3266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 19), t_3265, 'test')
        # Processing the call keyword arguments (line 307)
        kwargs_3267 = {}
        # Getting the type of 'self' (line 307)
        self_3263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 307)
        visit_3264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), self_3263, 'visit')
        # Calling visit(args, kwargs) (line 307)
        visit_call_result_3268 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), visit_3264, *[test_3266], **kwargs_3267)
        
        
        # Call to enter(...): (line 308)
        # Processing the call keyword arguments (line 308)
        kwargs_3271 = {}
        # Getting the type of 'self' (line 308)
        self_3269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 308)
        enter_3270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), self_3269, 'enter')
        # Calling enter(args, kwargs) (line 308)
        enter_call_result_3272 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), enter_3270, *[], **kwargs_3271)
        
        
        # Call to visit(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 't' (line 309)
        t_3275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 't', False)
        # Obtaining the member 'body' of a type (line 309)
        body_3276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 19), t_3275, 'body')
        # Processing the call keyword arguments (line 309)
        kwargs_3277 = {}
        # Getting the type of 'self' (line 309)
        self_3273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 309)
        visit_3274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), self_3273, 'visit')
        # Calling visit(args, kwargs) (line 309)
        visit_call_result_3278 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), visit_3274, *[body_3276], **kwargs_3277)
        
        
        # Call to leave(...): (line 310)
        # Processing the call keyword arguments (line 310)
        kwargs_3281 = {}
        # Getting the type of 'self' (line 310)
        self_3279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 310)
        leave_3280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), self_3279, 'leave')
        # Calling leave(args, kwargs) (line 310)
        leave_call_result_3282 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), leave_3280, *[], **kwargs_3281)
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 't' (line 312)
        t_3283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 15), 't')
        # Obtaining the member 'orelse' of a type (line 312)
        orelse_3284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 15), t_3283, 'orelse')
        
        
        # Call to len(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 't' (line 312)
        t_3286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 32), 't', False)
        # Obtaining the member 'orelse' of a type (line 312)
        orelse_3287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 32), t_3286, 'orelse')
        # Processing the call keyword arguments (line 312)
        kwargs_3288 = {}
        # Getting the type of 'len' (line 312)
        len_3285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 28), 'len', False)
        # Calling len(args, kwargs) (line 312)
        len_call_result_3289 = invoke(stypy.reporting.localization.Localization(__file__, 312, 28), len_3285, *[orelse_3287], **kwargs_3288)
        
        int_3290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 45), 'int')
        # Applying the binary operator '==' (line 312)
        result_eq_3291 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 28), '==', len_call_result_3289, int_3290)
        
        # Applying the binary operator 'and' (line 312)
        result_and_keyword_3292 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 15), 'and', orelse_3284, result_eq_3291)
        
        # Call to isinstance(...): (line 313)
        # Processing the call arguments (line 313)
        
        # Obtaining the type of the subscript
        int_3294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 39), 'int')
        # Getting the type of 't' (line 313)
        t_3295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 30), 't', False)
        # Obtaining the member 'orelse' of a type (line 313)
        orelse_3296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 30), t_3295, 'orelse')
        # Obtaining the member '__getitem__' of a type (line 313)
        getitem___3297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 30), orelse_3296, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 313)
        subscript_call_result_3298 = invoke(stypy.reporting.localization.Localization(__file__, 313, 30), getitem___3297, int_3294)
        
        # Getting the type of 'ast' (line 313)
        ast_3299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 43), 'ast', False)
        # Obtaining the member 'If' of a type (line 313)
        If_3300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 43), ast_3299, 'If')
        # Processing the call keyword arguments (line 313)
        kwargs_3301 = {}
        # Getting the type of 'isinstance' (line 313)
        isinstance_3293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 313)
        isinstance_call_result_3302 = invoke(stypy.reporting.localization.Localization(__file__, 313, 19), isinstance_3293, *[subscript_call_result_3298, If_3300], **kwargs_3301)
        
        # Applying the binary operator 'and' (line 312)
        result_and_keyword_3303 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 15), 'and', result_and_keyword_3292, isinstance_call_result_3302)
        
        # Assigning a type to the variable 'result_and_keyword_3303' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'result_and_keyword_3303', result_and_keyword_3303)
        # Testing if the while is going to be iterated (line 312)
        # Testing the type of an if condition (line 312)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 8), result_and_keyword_3303)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 312, 8), result_and_keyword_3303):
            # SSA begins for while statement (line 312)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Subscript to a Name (line 314):
            
            # Assigning a Subscript to a Name (line 314):
            
            # Obtaining the type of the subscript
            int_3304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 25), 'int')
            # Getting the type of 't' (line 314)
            t_3305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 't')
            # Obtaining the member 'orelse' of a type (line 314)
            orelse_3306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 16), t_3305, 'orelse')
            # Obtaining the member '__getitem__' of a type (line 314)
            getitem___3307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 16), orelse_3306, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 314)
            subscript_call_result_3308 = invoke(stypy.reporting.localization.Localization(__file__, 314, 16), getitem___3307, int_3304)
            
            # Assigning a type to the variable 't' (line 314)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 't', subscript_call_result_3308)
            
            # Call to fill(...): (line 315)
            # Processing the call arguments (line 315)
            str_3311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 22), 'str', 'elif ')
            # Processing the call keyword arguments (line 315)
            kwargs_3312 = {}
            # Getting the type of 'self' (line 315)
            self_3309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 315)
            fill_3310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 12), self_3309, 'fill')
            # Calling fill(args, kwargs) (line 315)
            fill_call_result_3313 = invoke(stypy.reporting.localization.Localization(__file__, 315, 12), fill_3310, *[str_3311], **kwargs_3312)
            
            
            # Call to visit(...): (line 316)
            # Processing the call arguments (line 316)
            # Getting the type of 't' (line 316)
            t_3316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 't', False)
            # Obtaining the member 'test' of a type (line 316)
            test_3317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 23), t_3316, 'test')
            # Processing the call keyword arguments (line 316)
            kwargs_3318 = {}
            # Getting the type of 'self' (line 316)
            self_3314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 316)
            visit_3315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 12), self_3314, 'visit')
            # Calling visit(args, kwargs) (line 316)
            visit_call_result_3319 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), visit_3315, *[test_3317], **kwargs_3318)
            
            
            # Call to enter(...): (line 317)
            # Processing the call keyword arguments (line 317)
            kwargs_3322 = {}
            # Getting the type of 'self' (line 317)
            self_3320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 317)
            enter_3321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), self_3320, 'enter')
            # Calling enter(args, kwargs) (line 317)
            enter_call_result_3323 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), enter_3321, *[], **kwargs_3322)
            
            
            # Call to visit(...): (line 318)
            # Processing the call arguments (line 318)
            # Getting the type of 't' (line 318)
            t_3326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 23), 't', False)
            # Obtaining the member 'body' of a type (line 318)
            body_3327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 23), t_3326, 'body')
            # Processing the call keyword arguments (line 318)
            kwargs_3328 = {}
            # Getting the type of 'self' (line 318)
            self_3324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 318)
            visit_3325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 12), self_3324, 'visit')
            # Calling visit(args, kwargs) (line 318)
            visit_call_result_3329 = invoke(stypy.reporting.localization.Localization(__file__, 318, 12), visit_3325, *[body_3327], **kwargs_3328)
            
            
            # Call to leave(...): (line 319)
            # Processing the call keyword arguments (line 319)
            kwargs_3332 = {}
            # Getting the type of 'self' (line 319)
            self_3330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 319)
            leave_3331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 12), self_3330, 'leave')
            # Calling leave(args, kwargs) (line 319)
            leave_call_result_3333 = invoke(stypy.reporting.localization.Localization(__file__, 319, 12), leave_3331, *[], **kwargs_3332)
            
            # SSA join for while statement (line 312)
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 't' (line 321)
        t_3334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 11), 't')
        # Obtaining the member 'orelse' of a type (line 321)
        orelse_3335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 11), t_3334, 'orelse')
        # Testing if the type of an if condition is none (line 321)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 321, 8), orelse_3335):
            pass
        else:
            
            # Testing the type of an if condition (line 321)
            if_condition_3336 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 8), orelse_3335)
            # Assigning a type to the variable 'if_condition_3336' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'if_condition_3336', if_condition_3336)
            # SSA begins for if statement (line 321)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fill(...): (line 322)
            # Processing the call arguments (line 322)
            str_3339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 22), 'str', 'else')
            # Processing the call keyword arguments (line 322)
            kwargs_3340 = {}
            # Getting the type of 'self' (line 322)
            self_3337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 322)
            fill_3338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 12), self_3337, 'fill')
            # Calling fill(args, kwargs) (line 322)
            fill_call_result_3341 = invoke(stypy.reporting.localization.Localization(__file__, 322, 12), fill_3338, *[str_3339], **kwargs_3340)
            
            
            # Call to enter(...): (line 323)
            # Processing the call keyword arguments (line 323)
            kwargs_3344 = {}
            # Getting the type of 'self' (line 323)
            self_3342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 323)
            enter_3343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), self_3342, 'enter')
            # Calling enter(args, kwargs) (line 323)
            enter_call_result_3345 = invoke(stypy.reporting.localization.Localization(__file__, 323, 12), enter_3343, *[], **kwargs_3344)
            
            
            # Call to visit(...): (line 324)
            # Processing the call arguments (line 324)
            # Getting the type of 't' (line 324)
            t_3348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 23), 't', False)
            # Obtaining the member 'orelse' of a type (line 324)
            orelse_3349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 23), t_3348, 'orelse')
            # Processing the call keyword arguments (line 324)
            kwargs_3350 = {}
            # Getting the type of 'self' (line 324)
            self_3346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 324)
            visit_3347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), self_3346, 'visit')
            # Calling visit(args, kwargs) (line 324)
            visit_call_result_3351 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), visit_3347, *[orelse_3349], **kwargs_3350)
            
            
            # Call to leave(...): (line 325)
            # Processing the call keyword arguments (line 325)
            kwargs_3354 = {}
            # Getting the type of 'self' (line 325)
            self_3352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 325)
            leave_3353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), self_3352, 'leave')
            # Calling leave(args, kwargs) (line 325)
            leave_call_result_3355 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), leave_3353, *[], **kwargs_3354)
            
            # SSA join for if statement (line 321)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 326)
        # Processing the call arguments (line 326)
        str_3358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 19), 'str', '\n')
        # Processing the call keyword arguments (line 326)
        kwargs_3359 = {}
        # Getting the type of 'self' (line 326)
        self_3356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 326)
        write_3357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), self_3356, 'write')
        # Calling write(args, kwargs) (line 326)
        write_call_result_3360 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), write_3357, *[str_3358], **kwargs_3359)
        
        
        # ################# End of 'visit_If(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_If' in the type store
        # Getting the type of 'stypy_return_type' (line 304)
        stypy_return_type_3361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3361)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_If'
        return stypy_return_type_3361


    @norecursion
    def visit_While(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_While'
        module_type_store = module_type_store.open_function_context('visit_While', 328, 4, False)
        # Assigning a type to the variable 'self' (line 329)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_While.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_While.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_While.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_While.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_While')
        PythonSrcGeneratorVisitor.visit_While.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_While.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_While.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_While.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_While.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_While.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_While.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_While', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_While', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_While(...)' code ##################

        
        # Call to fill(...): (line 329)
        # Processing the call arguments (line 329)
        str_3364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 18), 'str', 'while ')
        # Processing the call keyword arguments (line 329)
        kwargs_3365 = {}
        # Getting the type of 'self' (line 329)
        self_3362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 329)
        fill_3363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), self_3362, 'fill')
        # Calling fill(args, kwargs) (line 329)
        fill_call_result_3366 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), fill_3363, *[str_3364], **kwargs_3365)
        
        
        # Call to visit(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 't' (line 330)
        t_3369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 't', False)
        # Obtaining the member 'test' of a type (line 330)
        test_3370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 19), t_3369, 'test')
        # Processing the call keyword arguments (line 330)
        kwargs_3371 = {}
        # Getting the type of 'self' (line 330)
        self_3367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 330)
        visit_3368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), self_3367, 'visit')
        # Calling visit(args, kwargs) (line 330)
        visit_call_result_3372 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), visit_3368, *[test_3370], **kwargs_3371)
        
        
        # Call to enter(...): (line 331)
        # Processing the call keyword arguments (line 331)
        kwargs_3375 = {}
        # Getting the type of 'self' (line 331)
        self_3373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 331)
        enter_3374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), self_3373, 'enter')
        # Calling enter(args, kwargs) (line 331)
        enter_call_result_3376 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), enter_3374, *[], **kwargs_3375)
        
        
        # Call to visit(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 't' (line 332)
        t_3379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 't', False)
        # Obtaining the member 'body' of a type (line 332)
        body_3380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), t_3379, 'body')
        # Processing the call keyword arguments (line 332)
        kwargs_3381 = {}
        # Getting the type of 'self' (line 332)
        self_3377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 332)
        visit_3378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), self_3377, 'visit')
        # Calling visit(args, kwargs) (line 332)
        visit_call_result_3382 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), visit_3378, *[body_3380], **kwargs_3381)
        
        
        # Call to leave(...): (line 333)
        # Processing the call keyword arguments (line 333)
        kwargs_3385 = {}
        # Getting the type of 'self' (line 333)
        self_3383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 333)
        leave_3384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_3383, 'leave')
        # Calling leave(args, kwargs) (line 333)
        leave_call_result_3386 = invoke(stypy.reporting.localization.Localization(__file__, 333, 8), leave_3384, *[], **kwargs_3385)
        
        # Getting the type of 't' (line 334)
        t_3387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 11), 't')
        # Obtaining the member 'orelse' of a type (line 334)
        orelse_3388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 11), t_3387, 'orelse')
        # Testing if the type of an if condition is none (line 334)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 334, 8), orelse_3388):
            pass
        else:
            
            # Testing the type of an if condition (line 334)
            if_condition_3389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 8), orelse_3388)
            # Assigning a type to the variable 'if_condition_3389' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'if_condition_3389', if_condition_3389)
            # SSA begins for if statement (line 334)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fill(...): (line 335)
            # Processing the call arguments (line 335)
            str_3392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 22), 'str', 'else')
            # Processing the call keyword arguments (line 335)
            kwargs_3393 = {}
            # Getting the type of 'self' (line 335)
            self_3390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 335)
            fill_3391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), self_3390, 'fill')
            # Calling fill(args, kwargs) (line 335)
            fill_call_result_3394 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), fill_3391, *[str_3392], **kwargs_3393)
            
            
            # Call to enter(...): (line 336)
            # Processing the call keyword arguments (line 336)
            kwargs_3397 = {}
            # Getting the type of 'self' (line 336)
            self_3395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 336)
            enter_3396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), self_3395, 'enter')
            # Calling enter(args, kwargs) (line 336)
            enter_call_result_3398 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), enter_3396, *[], **kwargs_3397)
            
            
            # Call to visit(...): (line 337)
            # Processing the call arguments (line 337)
            # Getting the type of 't' (line 337)
            t_3401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 23), 't', False)
            # Obtaining the member 'orelse' of a type (line 337)
            orelse_3402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 23), t_3401, 'orelse')
            # Processing the call keyword arguments (line 337)
            kwargs_3403 = {}
            # Getting the type of 'self' (line 337)
            self_3399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 337)
            visit_3400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), self_3399, 'visit')
            # Calling visit(args, kwargs) (line 337)
            visit_call_result_3404 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), visit_3400, *[orelse_3402], **kwargs_3403)
            
            
            # Call to leave(...): (line 338)
            # Processing the call keyword arguments (line 338)
            kwargs_3407 = {}
            # Getting the type of 'self' (line 338)
            self_3405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 338)
            leave_3406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), self_3405, 'leave')
            # Calling leave(args, kwargs) (line 338)
            leave_call_result_3408 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), leave_3406, *[], **kwargs_3407)
            
            # SSA join for if statement (line 334)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_While(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_While' in the type store
        # Getting the type of 'stypy_return_type' (line 328)
        stypy_return_type_3409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3409)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_While'
        return stypy_return_type_3409


    @norecursion
    def visit_With(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_With'
        module_type_store = module_type_store.open_function_context('visit_With', 340, 4, False)
        # Assigning a type to the variable 'self' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_With.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_With.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_With.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_With.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_With')
        PythonSrcGeneratorVisitor.visit_With.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_With.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_With.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_With.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_With.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_With.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_With.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_With', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_With', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_With(...)' code ##################

        
        # Call to fill(...): (line 341)
        # Processing the call arguments (line 341)
        str_3412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 18), 'str', 'with ')
        # Processing the call keyword arguments (line 341)
        kwargs_3413 = {}
        # Getting the type of 'self' (line 341)
        self_3410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 341)
        fill_3411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), self_3410, 'fill')
        # Calling fill(args, kwargs) (line 341)
        fill_call_result_3414 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), fill_3411, *[str_3412], **kwargs_3413)
        
        
        # Call to visit(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 't' (line 342)
        t_3417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 19), 't', False)
        # Obtaining the member 'context_expr' of a type (line 342)
        context_expr_3418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 19), t_3417, 'context_expr')
        # Processing the call keyword arguments (line 342)
        kwargs_3419 = {}
        # Getting the type of 'self' (line 342)
        self_3415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 342)
        visit_3416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), self_3415, 'visit')
        # Calling visit(args, kwargs) (line 342)
        visit_call_result_3420 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), visit_3416, *[context_expr_3418], **kwargs_3419)
        
        # Getting the type of 't' (line 343)
        t_3421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 11), 't')
        # Obtaining the member 'optional_vars' of a type (line 343)
        optional_vars_3422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 11), t_3421, 'optional_vars')
        # Testing if the type of an if condition is none (line 343)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 343, 8), optional_vars_3422):
            pass
        else:
            
            # Testing the type of an if condition (line 343)
            if_condition_3423 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 8), optional_vars_3422)
            # Assigning a type to the variable 'if_condition_3423' (line 343)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'if_condition_3423', if_condition_3423)
            # SSA begins for if statement (line 343)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 344)
            # Processing the call arguments (line 344)
            str_3426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 23), 'str', ' as ')
            # Processing the call keyword arguments (line 344)
            kwargs_3427 = {}
            # Getting the type of 'self' (line 344)
            self_3424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 344)
            write_3425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 12), self_3424, 'write')
            # Calling write(args, kwargs) (line 344)
            write_call_result_3428 = invoke(stypy.reporting.localization.Localization(__file__, 344, 12), write_3425, *[str_3426], **kwargs_3427)
            
            
            # Call to visit(...): (line 345)
            # Processing the call arguments (line 345)
            # Getting the type of 't' (line 345)
            t_3431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 't', False)
            # Obtaining the member 'optional_vars' of a type (line 345)
            optional_vars_3432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 23), t_3431, 'optional_vars')
            # Processing the call keyword arguments (line 345)
            kwargs_3433 = {}
            # Getting the type of 'self' (line 345)
            self_3429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 345)
            visit_3430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 12), self_3429, 'visit')
            # Calling visit(args, kwargs) (line 345)
            visit_call_result_3434 = invoke(stypy.reporting.localization.Localization(__file__, 345, 12), visit_3430, *[optional_vars_3432], **kwargs_3433)
            
            # SSA join for if statement (line 343)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to enter(...): (line 346)
        # Processing the call keyword arguments (line 346)
        kwargs_3437 = {}
        # Getting the type of 'self' (line 346)
        self_3435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 346)
        enter_3436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), self_3435, 'enter')
        # Calling enter(args, kwargs) (line 346)
        enter_call_result_3438 = invoke(stypy.reporting.localization.Localization(__file__, 346, 8), enter_3436, *[], **kwargs_3437)
        
        
        # Call to visit(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 't' (line 347)
        t_3441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 19), 't', False)
        # Obtaining the member 'body' of a type (line 347)
        body_3442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 19), t_3441, 'body')
        # Processing the call keyword arguments (line 347)
        kwargs_3443 = {}
        # Getting the type of 'self' (line 347)
        self_3439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 347)
        visit_3440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), self_3439, 'visit')
        # Calling visit(args, kwargs) (line 347)
        visit_call_result_3444 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), visit_3440, *[body_3442], **kwargs_3443)
        
        
        # Call to leave(...): (line 348)
        # Processing the call keyword arguments (line 348)
        kwargs_3447 = {}
        # Getting the type of 'self' (line 348)
        self_3445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 348)
        leave_3446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), self_3445, 'leave')
        # Calling leave(args, kwargs) (line 348)
        leave_call_result_3448 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), leave_3446, *[], **kwargs_3447)
        
        
        # ################# End of 'visit_With(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_With' in the type store
        # Getting the type of 'stypy_return_type' (line 340)
        stypy_return_type_3449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3449)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_With'
        return stypy_return_type_3449


    @norecursion
    def visit_Str(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Str'
        module_type_store = module_type_store.open_function_context('visit_Str', 351, 4, False)
        # Assigning a type to the variable 'self' (line 352)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Str.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Str.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Str.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Str.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Str')
        PythonSrcGeneratorVisitor.visit_Str.__dict__.__setitem__('stypy_param_names_list', ['tree'])
        PythonSrcGeneratorVisitor.visit_Str.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Str.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Str.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Str.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Str.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Str.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Str', ['tree'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Str', localization, ['tree'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Str(...)' code ##################

        
        str_3450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 11), 'str', 'unicode_literals')
        # Getting the type of 'self' (line 355)
        self_3451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 37), 'self')
        # Obtaining the member 'future_imports' of a type (line 355)
        future_imports_3452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 37), self_3451, 'future_imports')
        # Applying the binary operator 'notin' (line 355)
        result_contains_3453 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 11), 'notin', str_3450, future_imports_3452)
        
        # Testing if the type of an if condition is none (line 355)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 355, 8), result_contains_3453):
            
            # Type idiom detected: calculating its left and rigth part (line 357)
            # Getting the type of 'str' (line 357)
            str_3464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 32), 'str')
            # Getting the type of 'tree' (line 357)
            tree_3465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 24), 'tree')
            # Obtaining the member 's' of a type (line 357)
            s_3466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 24), tree_3465, 's')
            
            (may_be_3467, more_types_in_union_3468) = may_be_subtype(str_3464, s_3466)

            if may_be_3467:

                if more_types_in_union_3468:
                    # Runtime conditional SSA (line 357)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'tree' (line 357)
                tree_3469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'tree')
                # Obtaining the member 's' of a type (line 357)
                s_3470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_3469, 's')
                # Setting the type of the member 's' of a type (line 357)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_3469, 's', remove_not_subtype_from_union(s_3466, str))
                
                # Call to write(...): (line 358)
                # Processing the call arguments (line 358)
                str_3473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 23), 'str', 'b')
                
                # Call to repr(...): (line 358)
                # Processing the call arguments (line 358)
                # Getting the type of 'tree' (line 358)
                tree_3475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 34), 'tree', False)
                # Obtaining the member 's' of a type (line 358)
                s_3476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 34), tree_3475, 's')
                # Processing the call keyword arguments (line 358)
                kwargs_3477 = {}
                # Getting the type of 'repr' (line 358)
                repr_3474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 29), 'repr', False)
                # Calling repr(args, kwargs) (line 358)
                repr_call_result_3478 = invoke(stypy.reporting.localization.Localization(__file__, 358, 29), repr_3474, *[s_3476], **kwargs_3477)
                
                # Applying the binary operator '+' (line 358)
                result_add_3479 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 23), '+', str_3473, repr_call_result_3478)
                
                # Processing the call keyword arguments (line 358)
                kwargs_3480 = {}
                # Getting the type of 'self' (line 358)
                self_3471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'self', False)
                # Obtaining the member 'write' of a type (line 358)
                write_3472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), self_3471, 'write')
                # Calling write(args, kwargs) (line 358)
                write_call_result_3481 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), write_3472, *[result_add_3479], **kwargs_3480)
                

                if more_types_in_union_3468:
                    # Runtime conditional SSA for else branch (line 357)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_3467) or more_types_in_union_3468):
                # Getting the type of 'tree' (line 357)
                tree_3482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'tree')
                # Obtaining the member 's' of a type (line 357)
                s_3483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_3482, 's')
                # Setting the type of the member 's' of a type (line 357)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_3482, 's', remove_subtype_from_union(s_3466, str))
                
                # Type idiom detected: calculating its left and rigth part (line 359)
                # Getting the type of 'unicode' (line 359)
                unicode_3484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 32), 'unicode')
                # Getting the type of 'tree' (line 359)
                tree_3485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 24), 'tree')
                # Obtaining the member 's' of a type (line 359)
                s_3486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 24), tree_3485, 's')
                
                (may_be_3487, more_types_in_union_3488) = may_be_subtype(unicode_3484, s_3486)

                if may_be_3487:

                    if more_types_in_union_3488:
                        # Runtime conditional SSA (line 359)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Getting the type of 'tree' (line 359)
                    tree_3489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 13), 'tree')
                    # Obtaining the member 's' of a type (line 359)
                    s_3490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_3489, 's')
                    # Setting the type of the member 's' of a type (line 359)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_3489, 's', remove_not_subtype_from_union(s_3486, unicode))
                    
                    # Call to write(...): (line 360)
                    # Processing the call arguments (line 360)
                    
                    # Call to lstrip(...): (line 360)
                    # Processing the call arguments (line 360)
                    str_3499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 43), 'str', 'u')
                    # Processing the call keyword arguments (line 360)
                    kwargs_3500 = {}
                    
                    # Call to repr(...): (line 360)
                    # Processing the call arguments (line 360)
                    # Getting the type of 'tree' (line 360)
                    tree_3494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 28), 'tree', False)
                    # Obtaining the member 's' of a type (line 360)
                    s_3495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 28), tree_3494, 's')
                    # Processing the call keyword arguments (line 360)
                    kwargs_3496 = {}
                    # Getting the type of 'repr' (line 360)
                    repr_3493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'repr', False)
                    # Calling repr(args, kwargs) (line 360)
                    repr_call_result_3497 = invoke(stypy.reporting.localization.Localization(__file__, 360, 23), repr_3493, *[s_3495], **kwargs_3496)
                    
                    # Obtaining the member 'lstrip' of a type (line 360)
                    lstrip_3498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 23), repr_call_result_3497, 'lstrip')
                    # Calling lstrip(args, kwargs) (line 360)
                    lstrip_call_result_3501 = invoke(stypy.reporting.localization.Localization(__file__, 360, 23), lstrip_3498, *[str_3499], **kwargs_3500)
                    
                    # Processing the call keyword arguments (line 360)
                    kwargs_3502 = {}
                    # Getting the type of 'self' (line 360)
                    self_3491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'self', False)
                    # Obtaining the member 'write' of a type (line 360)
                    write_3492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), self_3491, 'write')
                    # Calling write(args, kwargs) (line 360)
                    write_call_result_3503 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), write_3492, *[lstrip_call_result_3501], **kwargs_3502)
                    

                    if more_types_in_union_3488:
                        # Runtime conditional SSA for else branch (line 359)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_3487) or more_types_in_union_3488):
                    # Getting the type of 'tree' (line 359)
                    tree_3504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 13), 'tree')
                    # Obtaining the member 's' of a type (line 359)
                    s_3505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_3504, 's')
                    # Setting the type of the member 's' of a type (line 359)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_3504, 's', remove_subtype_from_union(s_3486, unicode))
                    # Evaluating assert statement condition
                    # Getting the type of 'False' (line 362)
                    False_3506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 19), 'False')
                    assert_3507 = False_3506
                    # Assigning a type to the variable 'assert_3507' (line 362)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'assert_3507', False_3506)

                    if (may_be_3487 and more_types_in_union_3488):
                        # SSA join for if statement (line 359)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_3467 and more_types_in_union_3468):
                    # SSA join for if statement (line 357)
                    module_type_store = module_type_store.join_ssa_context()


            
        else:
            
            # Testing the type of an if condition (line 355)
            if_condition_3454 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 8), result_contains_3453)
            # Assigning a type to the variable 'if_condition_3454' (line 355)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'if_condition_3454', if_condition_3454)
            # SSA begins for if statement (line 355)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 356)
            # Processing the call arguments (line 356)
            
            # Call to repr(...): (line 356)
            # Processing the call arguments (line 356)
            # Getting the type of 'tree' (line 356)
            tree_3458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 28), 'tree', False)
            # Obtaining the member 's' of a type (line 356)
            s_3459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 28), tree_3458, 's')
            # Processing the call keyword arguments (line 356)
            kwargs_3460 = {}
            # Getting the type of 'repr' (line 356)
            repr_3457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 23), 'repr', False)
            # Calling repr(args, kwargs) (line 356)
            repr_call_result_3461 = invoke(stypy.reporting.localization.Localization(__file__, 356, 23), repr_3457, *[s_3459], **kwargs_3460)
            
            # Processing the call keyword arguments (line 356)
            kwargs_3462 = {}
            # Getting the type of 'self' (line 356)
            self_3455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 356)
            write_3456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 12), self_3455, 'write')
            # Calling write(args, kwargs) (line 356)
            write_call_result_3463 = invoke(stypy.reporting.localization.Localization(__file__, 356, 12), write_3456, *[repr_call_result_3461], **kwargs_3462)
            
            # SSA branch for the else part of an if statement (line 355)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 357)
            # Getting the type of 'str' (line 357)
            str_3464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 32), 'str')
            # Getting the type of 'tree' (line 357)
            tree_3465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 24), 'tree')
            # Obtaining the member 's' of a type (line 357)
            s_3466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 24), tree_3465, 's')
            
            (may_be_3467, more_types_in_union_3468) = may_be_subtype(str_3464, s_3466)

            if may_be_3467:

                if more_types_in_union_3468:
                    # Runtime conditional SSA (line 357)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'tree' (line 357)
                tree_3469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'tree')
                # Obtaining the member 's' of a type (line 357)
                s_3470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_3469, 's')
                # Setting the type of the member 's' of a type (line 357)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_3469, 's', remove_not_subtype_from_union(s_3466, str))
                
                # Call to write(...): (line 358)
                # Processing the call arguments (line 358)
                str_3473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 23), 'str', 'b')
                
                # Call to repr(...): (line 358)
                # Processing the call arguments (line 358)
                # Getting the type of 'tree' (line 358)
                tree_3475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 34), 'tree', False)
                # Obtaining the member 's' of a type (line 358)
                s_3476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 34), tree_3475, 's')
                # Processing the call keyword arguments (line 358)
                kwargs_3477 = {}
                # Getting the type of 'repr' (line 358)
                repr_3474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 29), 'repr', False)
                # Calling repr(args, kwargs) (line 358)
                repr_call_result_3478 = invoke(stypy.reporting.localization.Localization(__file__, 358, 29), repr_3474, *[s_3476], **kwargs_3477)
                
                # Applying the binary operator '+' (line 358)
                result_add_3479 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 23), '+', str_3473, repr_call_result_3478)
                
                # Processing the call keyword arguments (line 358)
                kwargs_3480 = {}
                # Getting the type of 'self' (line 358)
                self_3471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'self', False)
                # Obtaining the member 'write' of a type (line 358)
                write_3472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), self_3471, 'write')
                # Calling write(args, kwargs) (line 358)
                write_call_result_3481 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), write_3472, *[result_add_3479], **kwargs_3480)
                

                if more_types_in_union_3468:
                    # Runtime conditional SSA for else branch (line 357)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_3467) or more_types_in_union_3468):
                # Getting the type of 'tree' (line 357)
                tree_3482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'tree')
                # Obtaining the member 's' of a type (line 357)
                s_3483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_3482, 's')
                # Setting the type of the member 's' of a type (line 357)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_3482, 's', remove_subtype_from_union(s_3466, str))
                
                # Type idiom detected: calculating its left and rigth part (line 359)
                # Getting the type of 'unicode' (line 359)
                unicode_3484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 32), 'unicode')
                # Getting the type of 'tree' (line 359)
                tree_3485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 24), 'tree')
                # Obtaining the member 's' of a type (line 359)
                s_3486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 24), tree_3485, 's')
                
                (may_be_3487, more_types_in_union_3488) = may_be_subtype(unicode_3484, s_3486)

                if may_be_3487:

                    if more_types_in_union_3488:
                        # Runtime conditional SSA (line 359)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Getting the type of 'tree' (line 359)
                    tree_3489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 13), 'tree')
                    # Obtaining the member 's' of a type (line 359)
                    s_3490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_3489, 's')
                    # Setting the type of the member 's' of a type (line 359)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_3489, 's', remove_not_subtype_from_union(s_3486, unicode))
                    
                    # Call to write(...): (line 360)
                    # Processing the call arguments (line 360)
                    
                    # Call to lstrip(...): (line 360)
                    # Processing the call arguments (line 360)
                    str_3499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 43), 'str', 'u')
                    # Processing the call keyword arguments (line 360)
                    kwargs_3500 = {}
                    
                    # Call to repr(...): (line 360)
                    # Processing the call arguments (line 360)
                    # Getting the type of 'tree' (line 360)
                    tree_3494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 28), 'tree', False)
                    # Obtaining the member 's' of a type (line 360)
                    s_3495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 28), tree_3494, 's')
                    # Processing the call keyword arguments (line 360)
                    kwargs_3496 = {}
                    # Getting the type of 'repr' (line 360)
                    repr_3493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'repr', False)
                    # Calling repr(args, kwargs) (line 360)
                    repr_call_result_3497 = invoke(stypy.reporting.localization.Localization(__file__, 360, 23), repr_3493, *[s_3495], **kwargs_3496)
                    
                    # Obtaining the member 'lstrip' of a type (line 360)
                    lstrip_3498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 23), repr_call_result_3497, 'lstrip')
                    # Calling lstrip(args, kwargs) (line 360)
                    lstrip_call_result_3501 = invoke(stypy.reporting.localization.Localization(__file__, 360, 23), lstrip_3498, *[str_3499], **kwargs_3500)
                    
                    # Processing the call keyword arguments (line 360)
                    kwargs_3502 = {}
                    # Getting the type of 'self' (line 360)
                    self_3491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'self', False)
                    # Obtaining the member 'write' of a type (line 360)
                    write_3492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), self_3491, 'write')
                    # Calling write(args, kwargs) (line 360)
                    write_call_result_3503 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), write_3492, *[lstrip_call_result_3501], **kwargs_3502)
                    

                    if more_types_in_union_3488:
                        # Runtime conditional SSA for else branch (line 359)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_3487) or more_types_in_union_3488):
                    # Getting the type of 'tree' (line 359)
                    tree_3504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 13), 'tree')
                    # Obtaining the member 's' of a type (line 359)
                    s_3505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_3504, 's')
                    # Setting the type of the member 's' of a type (line 359)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_3504, 's', remove_subtype_from_union(s_3486, unicode))
                    # Evaluating assert statement condition
                    # Getting the type of 'False' (line 362)
                    False_3506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 19), 'False')
                    assert_3507 = False_3506
                    # Assigning a type to the variable 'assert_3507' (line 362)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'assert_3507', False_3506)

                    if (may_be_3487 and more_types_in_union_3488):
                        # SSA join for if statement (line 359)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_3467 and more_types_in_union_3468):
                    # SSA join for if statement (line 357)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 355)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Str(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Str' in the type store
        # Getting the type of 'stypy_return_type' (line 351)
        stypy_return_type_3508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3508)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Str'
        return stypy_return_type_3508


    @norecursion
    def visit_Name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Name'
        module_type_store = module_type_store.open_function_context('visit_Name', 364, 4, False)
        # Assigning a type to the variable 'self' (line 365)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Name.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Name.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Name.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Name.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Name')
        PythonSrcGeneratorVisitor.visit_Name.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Name.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Name.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Name.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Name.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Name', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Name', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Name(...)' code ##################

        
        # Call to write(...): (line 365)
        # Processing the call arguments (line 365)
        # Getting the type of 't' (line 365)
        t_3511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 19), 't', False)
        # Obtaining the member 'id' of a type (line 365)
        id_3512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 19), t_3511, 'id')
        # Processing the call keyword arguments (line 365)
        kwargs_3513 = {}
        # Getting the type of 'self' (line 365)
        self_3509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 365)
        write_3510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 8), self_3509, 'write')
        # Calling write(args, kwargs) (line 365)
        write_call_result_3514 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), write_3510, *[id_3512], **kwargs_3513)
        
        
        # ################# End of 'visit_Name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Name' in the type store
        # Getting the type of 'stypy_return_type' (line 364)
        stypy_return_type_3515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3515)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Name'
        return stypy_return_type_3515


    @norecursion
    def visit_Repr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Repr'
        module_type_store = module_type_store.open_function_context('visit_Repr', 367, 4, False)
        # Assigning a type to the variable 'self' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Repr.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Repr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Repr.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Repr.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Repr')
        PythonSrcGeneratorVisitor.visit_Repr.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Repr.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Repr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Repr.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Repr.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Repr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Repr.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Repr', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Repr', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Repr(...)' code ##################

        
        # Call to write(...): (line 368)
        # Processing the call arguments (line 368)
        str_3518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 19), 'str', '`')
        # Processing the call keyword arguments (line 368)
        kwargs_3519 = {}
        # Getting the type of 'self' (line 368)
        self_3516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 368)
        write_3517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), self_3516, 'write')
        # Calling write(args, kwargs) (line 368)
        write_call_result_3520 = invoke(stypy.reporting.localization.Localization(__file__, 368, 8), write_3517, *[str_3518], **kwargs_3519)
        
        
        # Call to visit(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 't' (line 369)
        t_3523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 't', False)
        # Obtaining the member 'value' of a type (line 369)
        value_3524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 19), t_3523, 'value')
        # Processing the call keyword arguments (line 369)
        kwargs_3525 = {}
        # Getting the type of 'self' (line 369)
        self_3521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 369)
        visit_3522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), self_3521, 'visit')
        # Calling visit(args, kwargs) (line 369)
        visit_call_result_3526 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), visit_3522, *[value_3524], **kwargs_3525)
        
        
        # Call to write(...): (line 370)
        # Processing the call arguments (line 370)
        str_3529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 19), 'str', '`')
        # Processing the call keyword arguments (line 370)
        kwargs_3530 = {}
        # Getting the type of 'self' (line 370)
        self_3527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 370)
        write_3528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), self_3527, 'write')
        # Calling write(args, kwargs) (line 370)
        write_call_result_3531 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), write_3528, *[str_3529], **kwargs_3530)
        
        
        # ################# End of 'visit_Repr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Repr' in the type store
        # Getting the type of 'stypy_return_type' (line 367)
        stypy_return_type_3532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3532)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Repr'
        return stypy_return_type_3532


    @norecursion
    def visit_Num(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Num'
        module_type_store = module_type_store.open_function_context('visit_Num', 372, 4, False)
        # Assigning a type to the variable 'self' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Num.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Num.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Num.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Num.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Num')
        PythonSrcGeneratorVisitor.visit_Num.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Num.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Num.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Num.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Num.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Num.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Num.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Num', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Num', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Num(...)' code ##################

        
        # Assigning a Call to a Name (line 373):
        
        # Assigning a Call to a Name (line 373):
        
        # Call to repr(...): (line 373)
        # Processing the call arguments (line 373)
        # Getting the type of 't' (line 373)
        t_3534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), 't', False)
        # Obtaining the member 'n' of a type (line 373)
        n_3535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 22), t_3534, 'n')
        # Processing the call keyword arguments (line 373)
        kwargs_3536 = {}
        # Getting the type of 'repr' (line 373)
        repr_3533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 17), 'repr', False)
        # Calling repr(args, kwargs) (line 373)
        repr_call_result_3537 = invoke(stypy.reporting.localization.Localization(__file__, 373, 17), repr_3533, *[n_3535], **kwargs_3536)
        
        # Assigning a type to the variable 'repr_n' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'repr_n', repr_call_result_3537)
        
        # Call to startswith(...): (line 375)
        # Processing the call arguments (line 375)
        str_3540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 29), 'str', '-')
        # Processing the call keyword arguments (line 375)
        kwargs_3541 = {}
        # Getting the type of 'repr_n' (line 375)
        repr_n_3538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'repr_n', False)
        # Obtaining the member 'startswith' of a type (line 375)
        startswith_3539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 11), repr_n_3538, 'startswith')
        # Calling startswith(args, kwargs) (line 375)
        startswith_call_result_3542 = invoke(stypy.reporting.localization.Localization(__file__, 375, 11), startswith_3539, *[str_3540], **kwargs_3541)
        
        # Testing if the type of an if condition is none (line 375)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 375, 8), startswith_call_result_3542):
            pass
        else:
            
            # Testing the type of an if condition (line 375)
            if_condition_3543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 8), startswith_call_result_3542)
            # Assigning a type to the variable 'if_condition_3543' (line 375)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'if_condition_3543', if_condition_3543)
            # SSA begins for if statement (line 375)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 376)
            # Processing the call arguments (line 376)
            str_3546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 23), 'str', '(')
            # Processing the call keyword arguments (line 376)
            kwargs_3547 = {}
            # Getting the type of 'self' (line 376)
            self_3544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 376)
            write_3545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 12), self_3544, 'write')
            # Calling write(args, kwargs) (line 376)
            write_call_result_3548 = invoke(stypy.reporting.localization.Localization(__file__, 376, 12), write_3545, *[str_3546], **kwargs_3547)
            
            # SSA join for if statement (line 375)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 378)
        # Processing the call arguments (line 378)
        
        # Call to replace(...): (line 378)
        # Processing the call arguments (line 378)
        str_3553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 34), 'str', 'inf')
        # Getting the type of 'INFSTR' (line 378)
        INFSTR_3554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 41), 'INFSTR', False)
        # Processing the call keyword arguments (line 378)
        kwargs_3555 = {}
        # Getting the type of 'repr_n' (line 378)
        repr_n_3551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 19), 'repr_n', False)
        # Obtaining the member 'replace' of a type (line 378)
        replace_3552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 19), repr_n_3551, 'replace')
        # Calling replace(args, kwargs) (line 378)
        replace_call_result_3556 = invoke(stypy.reporting.localization.Localization(__file__, 378, 19), replace_3552, *[str_3553, INFSTR_3554], **kwargs_3555)
        
        # Processing the call keyword arguments (line 378)
        kwargs_3557 = {}
        # Getting the type of 'self' (line 378)
        self_3549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 378)
        write_3550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_3549, 'write')
        # Calling write(args, kwargs) (line 378)
        write_call_result_3558 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), write_3550, *[replace_call_result_3556], **kwargs_3557)
        
        
        # Call to startswith(...): (line 379)
        # Processing the call arguments (line 379)
        str_3561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 29), 'str', '-')
        # Processing the call keyword arguments (line 379)
        kwargs_3562 = {}
        # Getting the type of 'repr_n' (line 379)
        repr_n_3559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 11), 'repr_n', False)
        # Obtaining the member 'startswith' of a type (line 379)
        startswith_3560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 11), repr_n_3559, 'startswith')
        # Calling startswith(args, kwargs) (line 379)
        startswith_call_result_3563 = invoke(stypy.reporting.localization.Localization(__file__, 379, 11), startswith_3560, *[str_3561], **kwargs_3562)
        
        # Testing if the type of an if condition is none (line 379)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 379, 8), startswith_call_result_3563):
            pass
        else:
            
            # Testing the type of an if condition (line 379)
            if_condition_3564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 8), startswith_call_result_3563)
            # Assigning a type to the variable 'if_condition_3564' (line 379)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'if_condition_3564', if_condition_3564)
            # SSA begins for if statement (line 379)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 380)
            # Processing the call arguments (line 380)
            str_3567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 23), 'str', ')')
            # Processing the call keyword arguments (line 380)
            kwargs_3568 = {}
            # Getting the type of 'self' (line 380)
            self_3565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 380)
            write_3566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 12), self_3565, 'write')
            # Calling write(args, kwargs) (line 380)
            write_call_result_3569 = invoke(stypy.reporting.localization.Localization(__file__, 380, 12), write_3566, *[str_3567], **kwargs_3568)
            
            # SSA join for if statement (line 379)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Num(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Num' in the type store
        # Getting the type of 'stypy_return_type' (line 372)
        stypy_return_type_3570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3570)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Num'
        return stypy_return_type_3570


    @norecursion
    def visit_List(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_List'
        module_type_store = module_type_store.open_function_context('visit_List', 382, 4, False)
        # Assigning a type to the variable 'self' (line 383)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 383, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_List.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_List.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_List.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_List.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_List')
        PythonSrcGeneratorVisitor.visit_List.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_List.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_List.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_List.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_List.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_List.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_List.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_List', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_List', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_List(...)' code ##################

        
        # Call to write(...): (line 383)
        # Processing the call arguments (line 383)
        str_3573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 19), 'str', '[')
        # Processing the call keyword arguments (line 383)
        kwargs_3574 = {}
        # Getting the type of 'self' (line 383)
        self_3571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 383)
        write_3572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_3571, 'write')
        # Calling write(args, kwargs) (line 383)
        write_call_result_3575 = invoke(stypy.reporting.localization.Localization(__file__, 383, 8), write_3572, *[str_3573], **kwargs_3574)
        
        
        # Call to interleave(...): (line 384)
        # Processing the call arguments (line 384)

        @norecursion
        def _stypy_temp_lambda_9(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_9'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_9', 384, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_9.stypy_localization = localization
            _stypy_temp_lambda_9.stypy_type_of_self = None
            _stypy_temp_lambda_9.stypy_type_store = module_type_store
            _stypy_temp_lambda_9.stypy_function_name = '_stypy_temp_lambda_9'
            _stypy_temp_lambda_9.stypy_param_names_list = []
            _stypy_temp_lambda_9.stypy_varargs_param_name = None
            _stypy_temp_lambda_9.stypy_kwargs_param_name = None
            _stypy_temp_lambda_9.stypy_call_defaults = defaults
            _stypy_temp_lambda_9.stypy_call_varargs = varargs
            _stypy_temp_lambda_9.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_9', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_9', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 384)
            # Processing the call arguments (line 384)
            str_3579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 38), 'str', ', ')
            # Processing the call keyword arguments (line 384)
            kwargs_3580 = {}
            # Getting the type of 'self' (line 384)
            self_3577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 384)
            write_3578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 27), self_3577, 'write')
            # Calling write(args, kwargs) (line 384)
            write_call_result_3581 = invoke(stypy.reporting.localization.Localization(__file__, 384, 27), write_3578, *[str_3579], **kwargs_3580)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 384)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), 'stypy_return_type', write_call_result_3581)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_9' in the type store
            # Getting the type of 'stypy_return_type' (line 384)
            stypy_return_type_3582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3582)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_9'
            return stypy_return_type_3582

        # Assigning a type to the variable '_stypy_temp_lambda_9' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), '_stypy_temp_lambda_9', _stypy_temp_lambda_9)
        # Getting the type of '_stypy_temp_lambda_9' (line 384)
        _stypy_temp_lambda_9_3583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), '_stypy_temp_lambda_9')
        # Getting the type of 'self' (line 384)
        self_3584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 384)
        visit_3585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 45), self_3584, 'visit')
        # Getting the type of 't' (line 384)
        t_3586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 57), 't', False)
        # Obtaining the member 'elts' of a type (line 384)
        elts_3587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 57), t_3586, 'elts')
        # Processing the call keyword arguments (line 384)
        kwargs_3588 = {}
        # Getting the type of 'interleave' (line 384)
        interleave_3576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 384)
        interleave_call_result_3589 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), interleave_3576, *[_stypy_temp_lambda_9_3583, visit_3585, elts_3587], **kwargs_3588)
        
        
        # Call to write(...): (line 385)
        # Processing the call arguments (line 385)
        str_3592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 19), 'str', ']')
        # Processing the call keyword arguments (line 385)
        kwargs_3593 = {}
        # Getting the type of 'self' (line 385)
        self_3590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 385)
        write_3591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 8), self_3590, 'write')
        # Calling write(args, kwargs) (line 385)
        write_call_result_3594 = invoke(stypy.reporting.localization.Localization(__file__, 385, 8), write_3591, *[str_3592], **kwargs_3593)
        
        
        # ################# End of 'visit_List(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_List' in the type store
        # Getting the type of 'stypy_return_type' (line 382)
        stypy_return_type_3595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3595)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_List'
        return stypy_return_type_3595


    @norecursion
    def visit_ListComp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ListComp'
        module_type_store = module_type_store.open_function_context('visit_ListComp', 387, 4, False)
        # Assigning a type to the variable 'self' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_ListComp.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_ListComp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_ListComp.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_ListComp.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_ListComp')
        PythonSrcGeneratorVisitor.visit_ListComp.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_ListComp.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_ListComp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_ListComp.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_ListComp.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_ListComp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_ListComp.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_ListComp', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_ListComp', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_ListComp(...)' code ##################

        
        # Call to write(...): (line 388)
        # Processing the call arguments (line 388)
        str_3598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 19), 'str', '[')
        # Processing the call keyword arguments (line 388)
        kwargs_3599 = {}
        # Getting the type of 'self' (line 388)
        self_3596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 388)
        write_3597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), self_3596, 'write')
        # Calling write(args, kwargs) (line 388)
        write_call_result_3600 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), write_3597, *[str_3598], **kwargs_3599)
        
        
        # Call to visit(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 't' (line 389)
        t_3603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 't', False)
        # Obtaining the member 'elt' of a type (line 389)
        elt_3604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 19), t_3603, 'elt')
        # Processing the call keyword arguments (line 389)
        kwargs_3605 = {}
        # Getting the type of 'self' (line 389)
        self_3601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 389)
        visit_3602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), self_3601, 'visit')
        # Calling visit(args, kwargs) (line 389)
        visit_call_result_3606 = invoke(stypy.reporting.localization.Localization(__file__, 389, 8), visit_3602, *[elt_3604], **kwargs_3605)
        
        
        # Getting the type of 't' (line 390)
        t_3607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 't')
        # Obtaining the member 'generators' of a type (line 390)
        generators_3608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), t_3607, 'generators')
        # Assigning a type to the variable 'generators_3608' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'generators_3608', generators_3608)
        # Testing if the for loop is going to be iterated (line 390)
        # Testing the type of a for loop iterable (line 390)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 390, 8), generators_3608)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 390, 8), generators_3608):
            # Getting the type of the for loop variable (line 390)
            for_loop_var_3609 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 390, 8), generators_3608)
            # Assigning a type to the variable 'gen' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'gen', for_loop_var_3609)
            # SSA begins for a for statement (line 390)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 391)
            # Processing the call arguments (line 391)
            # Getting the type of 'gen' (line 391)
            gen_3612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 23), 'gen', False)
            # Processing the call keyword arguments (line 391)
            kwargs_3613 = {}
            # Getting the type of 'self' (line 391)
            self_3610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 391)
            visit_3611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 12), self_3610, 'visit')
            # Calling visit(args, kwargs) (line 391)
            visit_call_result_3614 = invoke(stypy.reporting.localization.Localization(__file__, 391, 12), visit_3611, *[gen_3612], **kwargs_3613)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 392)
        # Processing the call arguments (line 392)
        str_3617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 19), 'str', ']')
        # Processing the call keyword arguments (line 392)
        kwargs_3618 = {}
        # Getting the type of 'self' (line 392)
        self_3615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 392)
        write_3616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), self_3615, 'write')
        # Calling write(args, kwargs) (line 392)
        write_call_result_3619 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), write_3616, *[str_3617], **kwargs_3618)
        
        
        # ################# End of 'visit_ListComp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ListComp' in the type store
        # Getting the type of 'stypy_return_type' (line 387)
        stypy_return_type_3620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3620)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ListComp'
        return stypy_return_type_3620


    @norecursion
    def visit_GeneratorExp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_GeneratorExp'
        module_type_store = module_type_store.open_function_context('visit_GeneratorExp', 394, 4, False)
        # Assigning a type to the variable 'self' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_GeneratorExp.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_GeneratorExp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_GeneratorExp.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_GeneratorExp.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_GeneratorExp')
        PythonSrcGeneratorVisitor.visit_GeneratorExp.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_GeneratorExp.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_GeneratorExp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_GeneratorExp.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_GeneratorExp.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_GeneratorExp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_GeneratorExp.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_GeneratorExp', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_GeneratorExp', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_GeneratorExp(...)' code ##################

        
        # Call to write(...): (line 395)
        # Processing the call arguments (line 395)
        str_3623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 19), 'str', '(')
        # Processing the call keyword arguments (line 395)
        kwargs_3624 = {}
        # Getting the type of 'self' (line 395)
        self_3621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 395)
        write_3622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), self_3621, 'write')
        # Calling write(args, kwargs) (line 395)
        write_call_result_3625 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), write_3622, *[str_3623], **kwargs_3624)
        
        
        # Call to visit(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 't' (line 396)
        t_3628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 't', False)
        # Obtaining the member 'elt' of a type (line 396)
        elt_3629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 19), t_3628, 'elt')
        # Processing the call keyword arguments (line 396)
        kwargs_3630 = {}
        # Getting the type of 'self' (line 396)
        self_3626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 396)
        visit_3627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), self_3626, 'visit')
        # Calling visit(args, kwargs) (line 396)
        visit_call_result_3631 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), visit_3627, *[elt_3629], **kwargs_3630)
        
        
        # Getting the type of 't' (line 397)
        t_3632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 't')
        # Obtaining the member 'generators' of a type (line 397)
        generators_3633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 19), t_3632, 'generators')
        # Assigning a type to the variable 'generators_3633' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'generators_3633', generators_3633)
        # Testing if the for loop is going to be iterated (line 397)
        # Testing the type of a for loop iterable (line 397)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 397, 8), generators_3633)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 397, 8), generators_3633):
            # Getting the type of the for loop variable (line 397)
            for_loop_var_3634 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 397, 8), generators_3633)
            # Assigning a type to the variable 'gen' (line 397)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'gen', for_loop_var_3634)
            # SSA begins for a for statement (line 397)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 398)
            # Processing the call arguments (line 398)
            # Getting the type of 'gen' (line 398)
            gen_3637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 23), 'gen', False)
            # Processing the call keyword arguments (line 398)
            kwargs_3638 = {}
            # Getting the type of 'self' (line 398)
            self_3635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 398)
            visit_3636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), self_3635, 'visit')
            # Calling visit(args, kwargs) (line 398)
            visit_call_result_3639 = invoke(stypy.reporting.localization.Localization(__file__, 398, 12), visit_3636, *[gen_3637], **kwargs_3638)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 399)
        # Processing the call arguments (line 399)
        str_3642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 19), 'str', ')')
        # Processing the call keyword arguments (line 399)
        kwargs_3643 = {}
        # Getting the type of 'self' (line 399)
        self_3640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 399)
        write_3641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), self_3640, 'write')
        # Calling write(args, kwargs) (line 399)
        write_call_result_3644 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), write_3641, *[str_3642], **kwargs_3643)
        
        
        # ################# End of 'visit_GeneratorExp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_GeneratorExp' in the type store
        # Getting the type of 'stypy_return_type' (line 394)
        stypy_return_type_3645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3645)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_GeneratorExp'
        return stypy_return_type_3645


    @norecursion
    def visit_SetComp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_SetComp'
        module_type_store = module_type_store.open_function_context('visit_SetComp', 401, 4, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_SetComp.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_SetComp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_SetComp.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_SetComp.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_SetComp')
        PythonSrcGeneratorVisitor.visit_SetComp.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_SetComp.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_SetComp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_SetComp.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_SetComp.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_SetComp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_SetComp.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_SetComp', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_SetComp', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_SetComp(...)' code ##################

        
        # Call to write(...): (line 402)
        # Processing the call arguments (line 402)
        str_3648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 19), 'str', '{')
        # Processing the call keyword arguments (line 402)
        kwargs_3649 = {}
        # Getting the type of 'self' (line 402)
        self_3646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 402)
        write_3647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), self_3646, 'write')
        # Calling write(args, kwargs) (line 402)
        write_call_result_3650 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), write_3647, *[str_3648], **kwargs_3649)
        
        
        # Call to visit(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 't' (line 403)
        t_3653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 't', False)
        # Obtaining the member 'elt' of a type (line 403)
        elt_3654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 19), t_3653, 'elt')
        # Processing the call keyword arguments (line 403)
        kwargs_3655 = {}
        # Getting the type of 'self' (line 403)
        self_3651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 403)
        visit_3652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), self_3651, 'visit')
        # Calling visit(args, kwargs) (line 403)
        visit_call_result_3656 = invoke(stypy.reporting.localization.Localization(__file__, 403, 8), visit_3652, *[elt_3654], **kwargs_3655)
        
        
        # Getting the type of 't' (line 404)
        t_3657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 19), 't')
        # Obtaining the member 'generators' of a type (line 404)
        generators_3658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 19), t_3657, 'generators')
        # Assigning a type to the variable 'generators_3658' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'generators_3658', generators_3658)
        # Testing if the for loop is going to be iterated (line 404)
        # Testing the type of a for loop iterable (line 404)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 404, 8), generators_3658)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 404, 8), generators_3658):
            # Getting the type of the for loop variable (line 404)
            for_loop_var_3659 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 404, 8), generators_3658)
            # Assigning a type to the variable 'gen' (line 404)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'gen', for_loop_var_3659)
            # SSA begins for a for statement (line 404)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 405)
            # Processing the call arguments (line 405)
            # Getting the type of 'gen' (line 405)
            gen_3662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 23), 'gen', False)
            # Processing the call keyword arguments (line 405)
            kwargs_3663 = {}
            # Getting the type of 'self' (line 405)
            self_3660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 405)
            visit_3661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 12), self_3660, 'visit')
            # Calling visit(args, kwargs) (line 405)
            visit_call_result_3664 = invoke(stypy.reporting.localization.Localization(__file__, 405, 12), visit_3661, *[gen_3662], **kwargs_3663)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 406)
        # Processing the call arguments (line 406)
        str_3667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 19), 'str', '}')
        # Processing the call keyword arguments (line 406)
        kwargs_3668 = {}
        # Getting the type of 'self' (line 406)
        self_3665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 406)
        write_3666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), self_3665, 'write')
        # Calling write(args, kwargs) (line 406)
        write_call_result_3669 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), write_3666, *[str_3667], **kwargs_3668)
        
        
        # ################# End of 'visit_SetComp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_SetComp' in the type store
        # Getting the type of 'stypy_return_type' (line 401)
        stypy_return_type_3670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3670)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_SetComp'
        return stypy_return_type_3670


    @norecursion
    def visit_DictComp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_DictComp'
        module_type_store = module_type_store.open_function_context('visit_DictComp', 408, 4, False)
        # Assigning a type to the variable 'self' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_DictComp.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_DictComp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_DictComp.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_DictComp.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_DictComp')
        PythonSrcGeneratorVisitor.visit_DictComp.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_DictComp.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_DictComp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_DictComp.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_DictComp.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_DictComp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_DictComp.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_DictComp', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_DictComp', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_DictComp(...)' code ##################

        
        # Call to write(...): (line 409)
        # Processing the call arguments (line 409)
        str_3673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 19), 'str', '{')
        # Processing the call keyword arguments (line 409)
        kwargs_3674 = {}
        # Getting the type of 'self' (line 409)
        self_3671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 409)
        write_3672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), self_3671, 'write')
        # Calling write(args, kwargs) (line 409)
        write_call_result_3675 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), write_3672, *[str_3673], **kwargs_3674)
        
        
        # Call to visit(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 't' (line 410)
        t_3678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 19), 't', False)
        # Obtaining the member 'key' of a type (line 410)
        key_3679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 19), t_3678, 'key')
        # Processing the call keyword arguments (line 410)
        kwargs_3680 = {}
        # Getting the type of 'self' (line 410)
        self_3676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 410)
        visit_3677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 8), self_3676, 'visit')
        # Calling visit(args, kwargs) (line 410)
        visit_call_result_3681 = invoke(stypy.reporting.localization.Localization(__file__, 410, 8), visit_3677, *[key_3679], **kwargs_3680)
        
        
        # Call to write(...): (line 411)
        # Processing the call arguments (line 411)
        str_3684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 19), 'str', ': ')
        # Processing the call keyword arguments (line 411)
        kwargs_3685 = {}
        # Getting the type of 'self' (line 411)
        self_3682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 411)
        write_3683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), self_3682, 'write')
        # Calling write(args, kwargs) (line 411)
        write_call_result_3686 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), write_3683, *[str_3684], **kwargs_3685)
        
        
        # Call to visit(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 't' (line 412)
        t_3689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 19), 't', False)
        # Obtaining the member 'value' of a type (line 412)
        value_3690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 19), t_3689, 'value')
        # Processing the call keyword arguments (line 412)
        kwargs_3691 = {}
        # Getting the type of 'self' (line 412)
        self_3687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 412)
        visit_3688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 8), self_3687, 'visit')
        # Calling visit(args, kwargs) (line 412)
        visit_call_result_3692 = invoke(stypy.reporting.localization.Localization(__file__, 412, 8), visit_3688, *[value_3690], **kwargs_3691)
        
        
        # Getting the type of 't' (line 413)
        t_3693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 19), 't')
        # Obtaining the member 'generators' of a type (line 413)
        generators_3694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 19), t_3693, 'generators')
        # Assigning a type to the variable 'generators_3694' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'generators_3694', generators_3694)
        # Testing if the for loop is going to be iterated (line 413)
        # Testing the type of a for loop iterable (line 413)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 413, 8), generators_3694)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 413, 8), generators_3694):
            # Getting the type of the for loop variable (line 413)
            for_loop_var_3695 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 413, 8), generators_3694)
            # Assigning a type to the variable 'gen' (line 413)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'gen', for_loop_var_3695)
            # SSA begins for a for statement (line 413)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 414)
            # Processing the call arguments (line 414)
            # Getting the type of 'gen' (line 414)
            gen_3698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 23), 'gen', False)
            # Processing the call keyword arguments (line 414)
            kwargs_3699 = {}
            # Getting the type of 'self' (line 414)
            self_3696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 414)
            visit_3697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 12), self_3696, 'visit')
            # Calling visit(args, kwargs) (line 414)
            visit_call_result_3700 = invoke(stypy.reporting.localization.Localization(__file__, 414, 12), visit_3697, *[gen_3698], **kwargs_3699)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 415)
        # Processing the call arguments (line 415)
        str_3703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 19), 'str', '}')
        # Processing the call keyword arguments (line 415)
        kwargs_3704 = {}
        # Getting the type of 'self' (line 415)
        self_3701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 415)
        write_3702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), self_3701, 'write')
        # Calling write(args, kwargs) (line 415)
        write_call_result_3705 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), write_3702, *[str_3703], **kwargs_3704)
        
        
        # ################# End of 'visit_DictComp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_DictComp' in the type store
        # Getting the type of 'stypy_return_type' (line 408)
        stypy_return_type_3706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3706)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_DictComp'
        return stypy_return_type_3706


    @norecursion
    def visit_comprehension(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_comprehension'
        module_type_store = module_type_store.open_function_context('visit_comprehension', 417, 4, False)
        # Assigning a type to the variable 'self' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_comprehension.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_comprehension.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_comprehension.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_comprehension.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_comprehension')
        PythonSrcGeneratorVisitor.visit_comprehension.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_comprehension.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_comprehension.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_comprehension.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_comprehension.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_comprehension.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_comprehension.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_comprehension', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_comprehension', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_comprehension(...)' code ##################

        
        # Call to write(...): (line 418)
        # Processing the call arguments (line 418)
        str_3709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 19), 'str', ' for ')
        # Processing the call keyword arguments (line 418)
        kwargs_3710 = {}
        # Getting the type of 'self' (line 418)
        self_3707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 418)
        write_3708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), self_3707, 'write')
        # Calling write(args, kwargs) (line 418)
        write_call_result_3711 = invoke(stypy.reporting.localization.Localization(__file__, 418, 8), write_3708, *[str_3709], **kwargs_3710)
        
        
        # Call to visit(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 't' (line 419)
        t_3714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 19), 't', False)
        # Obtaining the member 'target' of a type (line 419)
        target_3715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 19), t_3714, 'target')
        # Processing the call keyword arguments (line 419)
        kwargs_3716 = {}
        # Getting the type of 'self' (line 419)
        self_3712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 419)
        visit_3713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 8), self_3712, 'visit')
        # Calling visit(args, kwargs) (line 419)
        visit_call_result_3717 = invoke(stypy.reporting.localization.Localization(__file__, 419, 8), visit_3713, *[target_3715], **kwargs_3716)
        
        
        # Call to write(...): (line 420)
        # Processing the call arguments (line 420)
        str_3720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 19), 'str', ' in ')
        # Processing the call keyword arguments (line 420)
        kwargs_3721 = {}
        # Getting the type of 'self' (line 420)
        self_3718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 420)
        write_3719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 8), self_3718, 'write')
        # Calling write(args, kwargs) (line 420)
        write_call_result_3722 = invoke(stypy.reporting.localization.Localization(__file__, 420, 8), write_3719, *[str_3720], **kwargs_3721)
        
        
        # Call to visit(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 't' (line 421)
        t_3725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 19), 't', False)
        # Obtaining the member 'iter' of a type (line 421)
        iter_3726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 19), t_3725, 'iter')
        # Processing the call keyword arguments (line 421)
        kwargs_3727 = {}
        # Getting the type of 'self' (line 421)
        self_3723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 421)
        visit_3724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 8), self_3723, 'visit')
        # Calling visit(args, kwargs) (line 421)
        visit_call_result_3728 = invoke(stypy.reporting.localization.Localization(__file__, 421, 8), visit_3724, *[iter_3726], **kwargs_3727)
        
        
        # Getting the type of 't' (line 422)
        t_3729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 25), 't')
        # Obtaining the member 'ifs' of a type (line 422)
        ifs_3730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 25), t_3729, 'ifs')
        # Assigning a type to the variable 'ifs_3730' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'ifs_3730', ifs_3730)
        # Testing if the for loop is going to be iterated (line 422)
        # Testing the type of a for loop iterable (line 422)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 422, 8), ifs_3730)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 422, 8), ifs_3730):
            # Getting the type of the for loop variable (line 422)
            for_loop_var_3731 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 422, 8), ifs_3730)
            # Assigning a type to the variable 'if_clause' (line 422)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'if_clause', for_loop_var_3731)
            # SSA begins for a for statement (line 422)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to write(...): (line 423)
            # Processing the call arguments (line 423)
            str_3734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 23), 'str', ' if ')
            # Processing the call keyword arguments (line 423)
            kwargs_3735 = {}
            # Getting the type of 'self' (line 423)
            self_3732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 423)
            write_3733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), self_3732, 'write')
            # Calling write(args, kwargs) (line 423)
            write_call_result_3736 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), write_3733, *[str_3734], **kwargs_3735)
            
            
            # Call to visit(...): (line 424)
            # Processing the call arguments (line 424)
            # Getting the type of 'if_clause' (line 424)
            if_clause_3739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 23), 'if_clause', False)
            # Processing the call keyword arguments (line 424)
            kwargs_3740 = {}
            # Getting the type of 'self' (line 424)
            self_3737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 424)
            visit_3738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), self_3737, 'visit')
            # Calling visit(args, kwargs) (line 424)
            visit_call_result_3741 = invoke(stypy.reporting.localization.Localization(__file__, 424, 12), visit_3738, *[if_clause_3739], **kwargs_3740)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'visit_comprehension(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_comprehension' in the type store
        # Getting the type of 'stypy_return_type' (line 417)
        stypy_return_type_3742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3742)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_comprehension'
        return stypy_return_type_3742


    @norecursion
    def visit_IfExp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_IfExp'
        module_type_store = module_type_store.open_function_context('visit_IfExp', 426, 4, False)
        # Assigning a type to the variable 'self' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_IfExp.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_IfExp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_IfExp.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_IfExp.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_IfExp')
        PythonSrcGeneratorVisitor.visit_IfExp.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_IfExp.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_IfExp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_IfExp.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_IfExp.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_IfExp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_IfExp.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_IfExp', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_IfExp', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_IfExp(...)' code ##################

        
        # Call to write(...): (line 427)
        # Processing the call arguments (line 427)
        str_3745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 19), 'str', '(')
        # Processing the call keyword arguments (line 427)
        kwargs_3746 = {}
        # Getting the type of 'self' (line 427)
        self_3743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 427)
        write_3744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), self_3743, 'write')
        # Calling write(args, kwargs) (line 427)
        write_call_result_3747 = invoke(stypy.reporting.localization.Localization(__file__, 427, 8), write_3744, *[str_3745], **kwargs_3746)
        
        
        # Call to visit(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 't' (line 428)
        t_3750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 19), 't', False)
        # Obtaining the member 'body' of a type (line 428)
        body_3751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 19), t_3750, 'body')
        # Processing the call keyword arguments (line 428)
        kwargs_3752 = {}
        # Getting the type of 'self' (line 428)
        self_3748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 428)
        visit_3749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 8), self_3748, 'visit')
        # Calling visit(args, kwargs) (line 428)
        visit_call_result_3753 = invoke(stypy.reporting.localization.Localization(__file__, 428, 8), visit_3749, *[body_3751], **kwargs_3752)
        
        
        # Call to write(...): (line 429)
        # Processing the call arguments (line 429)
        str_3756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 19), 'str', ' if ')
        # Processing the call keyword arguments (line 429)
        kwargs_3757 = {}
        # Getting the type of 'self' (line 429)
        self_3754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 429)
        write_3755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 8), self_3754, 'write')
        # Calling write(args, kwargs) (line 429)
        write_call_result_3758 = invoke(stypy.reporting.localization.Localization(__file__, 429, 8), write_3755, *[str_3756], **kwargs_3757)
        
        
        # Call to visit(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 't' (line 430)
        t_3761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 19), 't', False)
        # Obtaining the member 'test' of a type (line 430)
        test_3762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 19), t_3761, 'test')
        # Processing the call keyword arguments (line 430)
        kwargs_3763 = {}
        # Getting the type of 'self' (line 430)
        self_3759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 430)
        visit_3760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), self_3759, 'visit')
        # Calling visit(args, kwargs) (line 430)
        visit_call_result_3764 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), visit_3760, *[test_3762], **kwargs_3763)
        
        
        # Call to write(...): (line 431)
        # Processing the call arguments (line 431)
        str_3767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 19), 'str', ' else ')
        # Processing the call keyword arguments (line 431)
        kwargs_3768 = {}
        # Getting the type of 'self' (line 431)
        self_3765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 431)
        write_3766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 8), self_3765, 'write')
        # Calling write(args, kwargs) (line 431)
        write_call_result_3769 = invoke(stypy.reporting.localization.Localization(__file__, 431, 8), write_3766, *[str_3767], **kwargs_3768)
        
        
        # Call to visit(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 't' (line 432)
        t_3772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 19), 't', False)
        # Obtaining the member 'orelse' of a type (line 432)
        orelse_3773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 19), t_3772, 'orelse')
        # Processing the call keyword arguments (line 432)
        kwargs_3774 = {}
        # Getting the type of 'self' (line 432)
        self_3770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 432)
        visit_3771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), self_3770, 'visit')
        # Calling visit(args, kwargs) (line 432)
        visit_call_result_3775 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), visit_3771, *[orelse_3773], **kwargs_3774)
        
        
        # Call to write(...): (line 433)
        # Processing the call arguments (line 433)
        str_3778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 19), 'str', ')')
        # Processing the call keyword arguments (line 433)
        kwargs_3779 = {}
        # Getting the type of 'self' (line 433)
        self_3776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 433)
        write_3777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), self_3776, 'write')
        # Calling write(args, kwargs) (line 433)
        write_call_result_3780 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), write_3777, *[str_3778], **kwargs_3779)
        
        
        # ################# End of 'visit_IfExp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_IfExp' in the type store
        # Getting the type of 'stypy_return_type' (line 426)
        stypy_return_type_3781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3781)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_IfExp'
        return stypy_return_type_3781


    @norecursion
    def visit_Set(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Set'
        module_type_store = module_type_store.open_function_context('visit_Set', 435, 4, False)
        # Assigning a type to the variable 'self' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Set.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Set.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Set.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Set.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Set')
        PythonSrcGeneratorVisitor.visit_Set.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Set.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Set.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Set.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Set.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Set.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Set.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Set', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Set', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Set(...)' code ##################

        # Evaluating assert statement condition
        # Getting the type of 't' (line 436)
        t_3782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 't')
        # Obtaining the member 'elts' of a type (line 436)
        elts_3783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 16), t_3782, 'elts')
        assert_3784 = elts_3783
        # Assigning a type to the variable 'assert_3784' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'assert_3784', elts_3783)
        
        # Call to write(...): (line 437)
        # Processing the call arguments (line 437)
        str_3787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 19), 'str', '{')
        # Processing the call keyword arguments (line 437)
        kwargs_3788 = {}
        # Getting the type of 'self' (line 437)
        self_3785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 437)
        write_3786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), self_3785, 'write')
        # Calling write(args, kwargs) (line 437)
        write_call_result_3789 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), write_3786, *[str_3787], **kwargs_3788)
        
        
        # Call to interleave(...): (line 438)
        # Processing the call arguments (line 438)

        @norecursion
        def _stypy_temp_lambda_10(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_10'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_10', 438, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_10.stypy_localization = localization
            _stypy_temp_lambda_10.stypy_type_of_self = None
            _stypy_temp_lambda_10.stypy_type_store = module_type_store
            _stypy_temp_lambda_10.stypy_function_name = '_stypy_temp_lambda_10'
            _stypy_temp_lambda_10.stypy_param_names_list = []
            _stypy_temp_lambda_10.stypy_varargs_param_name = None
            _stypy_temp_lambda_10.stypy_kwargs_param_name = None
            _stypy_temp_lambda_10.stypy_call_defaults = defaults
            _stypy_temp_lambda_10.stypy_call_varargs = varargs
            _stypy_temp_lambda_10.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_10', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_10', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 438)
            # Processing the call arguments (line 438)
            str_3793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 38), 'str', ', ')
            # Processing the call keyword arguments (line 438)
            kwargs_3794 = {}
            # Getting the type of 'self' (line 438)
            self_3791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 438)
            write_3792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 27), self_3791, 'write')
            # Calling write(args, kwargs) (line 438)
            write_call_result_3795 = invoke(stypy.reporting.localization.Localization(__file__, 438, 27), write_3792, *[str_3793], **kwargs_3794)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 438)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'stypy_return_type', write_call_result_3795)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_10' in the type store
            # Getting the type of 'stypy_return_type' (line 438)
            stypy_return_type_3796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3796)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_10'
            return stypy_return_type_3796

        # Assigning a type to the variable '_stypy_temp_lambda_10' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), '_stypy_temp_lambda_10', _stypy_temp_lambda_10)
        # Getting the type of '_stypy_temp_lambda_10' (line 438)
        _stypy_temp_lambda_10_3797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), '_stypy_temp_lambda_10')
        # Getting the type of 'self' (line 438)
        self_3798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 438)
        visit_3799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 45), self_3798, 'visit')
        # Getting the type of 't' (line 438)
        t_3800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 57), 't', False)
        # Obtaining the member 'elts' of a type (line 438)
        elts_3801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 57), t_3800, 'elts')
        # Processing the call keyword arguments (line 438)
        kwargs_3802 = {}
        # Getting the type of 'interleave' (line 438)
        interleave_3790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 438)
        interleave_call_result_3803 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), interleave_3790, *[_stypy_temp_lambda_10_3797, visit_3799, elts_3801], **kwargs_3802)
        
        
        # Call to write(...): (line 439)
        # Processing the call arguments (line 439)
        str_3806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 19), 'str', '}')
        # Processing the call keyword arguments (line 439)
        kwargs_3807 = {}
        # Getting the type of 'self' (line 439)
        self_3804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 439)
        write_3805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 8), self_3804, 'write')
        # Calling write(args, kwargs) (line 439)
        write_call_result_3808 = invoke(stypy.reporting.localization.Localization(__file__, 439, 8), write_3805, *[str_3806], **kwargs_3807)
        
        
        # ################# End of 'visit_Set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Set' in the type store
        # Getting the type of 'stypy_return_type' (line 435)
        stypy_return_type_3809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3809)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Set'
        return stypy_return_type_3809


    @norecursion
    def visit_Dict(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Dict'
        module_type_store = module_type_store.open_function_context('visit_Dict', 441, 4, False)
        # Assigning a type to the variable 'self' (line 442)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Dict.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Dict.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Dict.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Dict.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Dict')
        PythonSrcGeneratorVisitor.visit_Dict.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Dict.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Dict.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Dict.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Dict.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Dict.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Dict.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Dict', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Dict', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Dict(...)' code ##################

        
        # Call to write(...): (line 442)
        # Processing the call arguments (line 442)
        str_3812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 19), 'str', '{')
        # Processing the call keyword arguments (line 442)
        kwargs_3813 = {}
        # Getting the type of 'self' (line 442)
        self_3810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 442)
        write_3811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 8), self_3810, 'write')
        # Calling write(args, kwargs) (line 442)
        write_call_result_3814 = invoke(stypy.reporting.localization.Localization(__file__, 442, 8), write_3811, *[str_3812], **kwargs_3813)
        

        @norecursion
        def write_pair(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'write_pair'
            module_type_store = module_type_store.open_function_context('write_pair', 444, 8, False)
            
            # Passed parameters checking function
            write_pair.stypy_localization = localization
            write_pair.stypy_type_of_self = None
            write_pair.stypy_type_store = module_type_store
            write_pair.stypy_function_name = 'write_pair'
            write_pair.stypy_param_names_list = ['pair']
            write_pair.stypy_varargs_param_name = None
            write_pair.stypy_kwargs_param_name = None
            write_pair.stypy_call_defaults = defaults
            write_pair.stypy_call_varargs = varargs
            write_pair.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'write_pair', ['pair'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'write_pair', localization, ['pair'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'write_pair(...)' code ##################

            
            # Assigning a Name to a Tuple (line 445):
            
            # Assigning a Subscript to a Name (line 445):
            
            # Obtaining the type of the subscript
            int_3815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 12), 'int')
            # Getting the type of 'pair' (line 445)
            pair_3816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'pair')
            # Obtaining the member '__getitem__' of a type (line 445)
            getitem___3817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 12), pair_3816, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 445)
            subscript_call_result_3818 = invoke(stypy.reporting.localization.Localization(__file__, 445, 12), getitem___3817, int_3815)
            
            # Assigning a type to the variable 'tuple_var_assignment_2296' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'tuple_var_assignment_2296', subscript_call_result_3818)
            
            # Assigning a Subscript to a Name (line 445):
            
            # Obtaining the type of the subscript
            int_3819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 12), 'int')
            # Getting the type of 'pair' (line 445)
            pair_3820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'pair')
            # Obtaining the member '__getitem__' of a type (line 445)
            getitem___3821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 12), pair_3820, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 445)
            subscript_call_result_3822 = invoke(stypy.reporting.localization.Localization(__file__, 445, 12), getitem___3821, int_3819)
            
            # Assigning a type to the variable 'tuple_var_assignment_2297' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'tuple_var_assignment_2297', subscript_call_result_3822)
            
            # Assigning a Name to a Name (line 445):
            # Getting the type of 'tuple_var_assignment_2296' (line 445)
            tuple_var_assignment_2296_3823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'tuple_var_assignment_2296')
            # Assigning a type to the variable 'k' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 13), 'k', tuple_var_assignment_2296_3823)
            
            # Assigning a Name to a Name (line 445):
            # Getting the type of 'tuple_var_assignment_2297' (line 445)
            tuple_var_assignment_2297_3824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'tuple_var_assignment_2297')
            # Assigning a type to the variable 'v' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 16), 'v', tuple_var_assignment_2297_3824)
            
            # Call to visit(...): (line 446)
            # Processing the call arguments (line 446)
            # Getting the type of 'k' (line 446)
            k_3827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 23), 'k', False)
            # Processing the call keyword arguments (line 446)
            kwargs_3828 = {}
            # Getting the type of 'self' (line 446)
            self_3825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 446)
            visit_3826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 12), self_3825, 'visit')
            # Calling visit(args, kwargs) (line 446)
            visit_call_result_3829 = invoke(stypy.reporting.localization.Localization(__file__, 446, 12), visit_3826, *[k_3827], **kwargs_3828)
            
            
            # Call to write(...): (line 447)
            # Processing the call arguments (line 447)
            str_3832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 23), 'str', ': ')
            # Processing the call keyword arguments (line 447)
            kwargs_3833 = {}
            # Getting the type of 'self' (line 447)
            self_3830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 447)
            write_3831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 12), self_3830, 'write')
            # Calling write(args, kwargs) (line 447)
            write_call_result_3834 = invoke(stypy.reporting.localization.Localization(__file__, 447, 12), write_3831, *[str_3832], **kwargs_3833)
            
            
            # Call to visit(...): (line 448)
            # Processing the call arguments (line 448)
            # Getting the type of 'v' (line 448)
            v_3837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 23), 'v', False)
            # Processing the call keyword arguments (line 448)
            kwargs_3838 = {}
            # Getting the type of 'self' (line 448)
            self_3835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 448)
            visit_3836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 12), self_3835, 'visit')
            # Calling visit(args, kwargs) (line 448)
            visit_call_result_3839 = invoke(stypy.reporting.localization.Localization(__file__, 448, 12), visit_3836, *[v_3837], **kwargs_3838)
            
            
            # ################# End of 'write_pair(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'write_pair' in the type store
            # Getting the type of 'stypy_return_type' (line 444)
            stypy_return_type_3840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3840)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'write_pair'
            return stypy_return_type_3840

        # Assigning a type to the variable 'write_pair' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'write_pair', write_pair)
        
        # Call to interleave(...): (line 450)
        # Processing the call arguments (line 450)

        @norecursion
        def _stypy_temp_lambda_11(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_11'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_11', 450, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_11.stypy_localization = localization
            _stypy_temp_lambda_11.stypy_type_of_self = None
            _stypy_temp_lambda_11.stypy_type_store = module_type_store
            _stypy_temp_lambda_11.stypy_function_name = '_stypy_temp_lambda_11'
            _stypy_temp_lambda_11.stypy_param_names_list = []
            _stypy_temp_lambda_11.stypy_varargs_param_name = None
            _stypy_temp_lambda_11.stypy_kwargs_param_name = None
            _stypy_temp_lambda_11.stypy_call_defaults = defaults
            _stypy_temp_lambda_11.stypy_call_varargs = varargs
            _stypy_temp_lambda_11.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_11', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_11', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 450)
            # Processing the call arguments (line 450)
            str_3844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 38), 'str', ', ')
            # Processing the call keyword arguments (line 450)
            kwargs_3845 = {}
            # Getting the type of 'self' (line 450)
            self_3842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 450)
            write_3843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 27), self_3842, 'write')
            # Calling write(args, kwargs) (line 450)
            write_call_result_3846 = invoke(stypy.reporting.localization.Localization(__file__, 450, 27), write_3843, *[str_3844], **kwargs_3845)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 450)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), 'stypy_return_type', write_call_result_3846)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_11' in the type store
            # Getting the type of 'stypy_return_type' (line 450)
            stypy_return_type_3847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_3847)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_11'
            return stypy_return_type_3847

        # Assigning a type to the variable '_stypy_temp_lambda_11' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), '_stypy_temp_lambda_11', _stypy_temp_lambda_11)
        # Getting the type of '_stypy_temp_lambda_11' (line 450)
        _stypy_temp_lambda_11_3848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), '_stypy_temp_lambda_11')
        # Getting the type of 'write_pair' (line 450)
        write_pair_3849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 45), 'write_pair', False)
        
        # Call to zip(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 't' (line 450)
        t_3851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 61), 't', False)
        # Obtaining the member 'keys' of a type (line 450)
        keys_3852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 61), t_3851, 'keys')
        # Getting the type of 't' (line 450)
        t_3853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 69), 't', False)
        # Obtaining the member 'values' of a type (line 450)
        values_3854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 69), t_3853, 'values')
        # Processing the call keyword arguments (line 450)
        kwargs_3855 = {}
        # Getting the type of 'zip' (line 450)
        zip_3850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 57), 'zip', False)
        # Calling zip(args, kwargs) (line 450)
        zip_call_result_3856 = invoke(stypy.reporting.localization.Localization(__file__, 450, 57), zip_3850, *[keys_3852, values_3854], **kwargs_3855)
        
        # Processing the call keyword arguments (line 450)
        kwargs_3857 = {}
        # Getting the type of 'interleave' (line 450)
        interleave_3841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 450)
        interleave_call_result_3858 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), interleave_3841, *[_stypy_temp_lambda_11_3848, write_pair_3849, zip_call_result_3856], **kwargs_3857)
        
        
        # Call to write(...): (line 451)
        # Processing the call arguments (line 451)
        str_3861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 19), 'str', '}')
        # Processing the call keyword arguments (line 451)
        kwargs_3862 = {}
        # Getting the type of 'self' (line 451)
        self_3859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 451)
        write_3860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), self_3859, 'write')
        # Calling write(args, kwargs) (line 451)
        write_call_result_3863 = invoke(stypy.reporting.localization.Localization(__file__, 451, 8), write_3860, *[str_3861], **kwargs_3862)
        
        
        # ################# End of 'visit_Dict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Dict' in the type store
        # Getting the type of 'stypy_return_type' (line 441)
        stypy_return_type_3864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3864)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Dict'
        return stypy_return_type_3864


    @norecursion
    def visit_Tuple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Tuple'
        module_type_store = module_type_store.open_function_context('visit_Tuple', 453, 4, False)
        # Assigning a type to the variable 'self' (line 454)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Tuple.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Tuple.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Tuple.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Tuple.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Tuple')
        PythonSrcGeneratorVisitor.visit_Tuple.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Tuple.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Tuple.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Tuple.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Tuple.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Tuple.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Tuple.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Tuple', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Tuple', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Tuple(...)' code ##################

        
        # Call to write(...): (line 454)
        # Processing the call arguments (line 454)
        str_3867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 19), 'str', '(')
        # Processing the call keyword arguments (line 454)
        kwargs_3868 = {}
        # Getting the type of 'self' (line 454)
        self_3865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 454)
        write_3866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), self_3865, 'write')
        # Calling write(args, kwargs) (line 454)
        write_call_result_3869 = invoke(stypy.reporting.localization.Localization(__file__, 454, 8), write_3866, *[str_3867], **kwargs_3868)
        
        
        
        # Call to len(...): (line 455)
        # Processing the call arguments (line 455)
        # Getting the type of 't' (line 455)
        t_3871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 15), 't', False)
        # Obtaining the member 'elts' of a type (line 455)
        elts_3872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 15), t_3871, 'elts')
        # Processing the call keyword arguments (line 455)
        kwargs_3873 = {}
        # Getting the type of 'len' (line 455)
        len_3870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 11), 'len', False)
        # Calling len(args, kwargs) (line 455)
        len_call_result_3874 = invoke(stypy.reporting.localization.Localization(__file__, 455, 11), len_3870, *[elts_3872], **kwargs_3873)
        
        int_3875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 26), 'int')
        # Applying the binary operator '==' (line 455)
        result_eq_3876 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 11), '==', len_call_result_3874, int_3875)
        
        # Testing if the type of an if condition is none (line 455)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 455, 8), result_eq_3876):
            
            # Call to interleave(...): (line 460)
            # Processing the call arguments (line 460)

            @norecursion
            def _stypy_temp_lambda_12(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_12'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_12', 460, 23, True)
                # Passed parameters checking function
                _stypy_temp_lambda_12.stypy_localization = localization
                _stypy_temp_lambda_12.stypy_type_of_self = None
                _stypy_temp_lambda_12.stypy_type_store = module_type_store
                _stypy_temp_lambda_12.stypy_function_name = '_stypy_temp_lambda_12'
                _stypy_temp_lambda_12.stypy_param_names_list = []
                _stypy_temp_lambda_12.stypy_varargs_param_name = None
                _stypy_temp_lambda_12.stypy_kwargs_param_name = None
                _stypy_temp_lambda_12.stypy_call_defaults = defaults
                _stypy_temp_lambda_12.stypy_call_varargs = varargs
                _stypy_temp_lambda_12.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_12', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_12', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to write(...): (line 460)
                # Processing the call arguments (line 460)
                str_3897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 42), 'str', ', ')
                # Processing the call keyword arguments (line 460)
                kwargs_3898 = {}
                # Getting the type of 'self' (line 460)
                self_3895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 31), 'self', False)
                # Obtaining the member 'write' of a type (line 460)
                write_3896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 31), self_3895, 'write')
                # Calling write(args, kwargs) (line 460)
                write_call_result_3899 = invoke(stypy.reporting.localization.Localization(__file__, 460, 31), write_3896, *[str_3897], **kwargs_3898)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 460)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'stypy_return_type', write_call_result_3899)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_12' in the type store
                # Getting the type of 'stypy_return_type' (line 460)
                stypy_return_type_3900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_3900)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_12'
                return stypy_return_type_3900

            # Assigning a type to the variable '_stypy_temp_lambda_12' (line 460)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), '_stypy_temp_lambda_12', _stypy_temp_lambda_12)
            # Getting the type of '_stypy_temp_lambda_12' (line 460)
            _stypy_temp_lambda_12_3901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), '_stypy_temp_lambda_12')
            # Getting the type of 'self' (line 460)
            self_3902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 49), 'self', False)
            # Obtaining the member 'visit' of a type (line 460)
            visit_3903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 49), self_3902, 'visit')
            # Getting the type of 't' (line 460)
            t_3904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 61), 't', False)
            # Obtaining the member 'elts' of a type (line 460)
            elts_3905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 61), t_3904, 'elts')
            # Processing the call keyword arguments (line 460)
            kwargs_3906 = {}
            # Getting the type of 'interleave' (line 460)
            interleave_3894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'interleave', False)
            # Calling interleave(args, kwargs) (line 460)
            interleave_call_result_3907 = invoke(stypy.reporting.localization.Localization(__file__, 460, 12), interleave_3894, *[_stypy_temp_lambda_12_3901, visit_3903, elts_3905], **kwargs_3906)
            
        else:
            
            # Testing the type of an if condition (line 455)
            if_condition_3877 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 8), result_eq_3876)
            # Assigning a type to the variable 'if_condition_3877' (line 455)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'if_condition_3877', if_condition_3877)
            # SSA begins for if statement (line 455)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Tuple (line 456):
            
            # Assigning a Subscript to a Name (line 456):
            
            # Obtaining the type of the subscript
            int_3878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 12), 'int')
            # Getting the type of 't' (line 456)
            t_3879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 21), 't')
            # Obtaining the member 'elts' of a type (line 456)
            elts_3880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 21), t_3879, 'elts')
            # Obtaining the member '__getitem__' of a type (line 456)
            getitem___3881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 12), elts_3880, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 456)
            subscript_call_result_3882 = invoke(stypy.reporting.localization.Localization(__file__, 456, 12), getitem___3881, int_3878)
            
            # Assigning a type to the variable 'tuple_var_assignment_2298' (line 456)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'tuple_var_assignment_2298', subscript_call_result_3882)
            
            # Assigning a Name to a Name (line 456):
            # Getting the type of 'tuple_var_assignment_2298' (line 456)
            tuple_var_assignment_2298_3883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'tuple_var_assignment_2298')
            # Assigning a type to the variable 'elt' (line 456)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 13), 'elt', tuple_var_assignment_2298_3883)
            
            # Call to visit(...): (line 457)
            # Processing the call arguments (line 457)
            # Getting the type of 'elt' (line 457)
            elt_3886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 23), 'elt', False)
            # Processing the call keyword arguments (line 457)
            kwargs_3887 = {}
            # Getting the type of 'self' (line 457)
            self_3884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 457)
            visit_3885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 12), self_3884, 'visit')
            # Calling visit(args, kwargs) (line 457)
            visit_call_result_3888 = invoke(stypy.reporting.localization.Localization(__file__, 457, 12), visit_3885, *[elt_3886], **kwargs_3887)
            
            
            # Call to write(...): (line 458)
            # Processing the call arguments (line 458)
            str_3891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 23), 'str', ',')
            # Processing the call keyword arguments (line 458)
            kwargs_3892 = {}
            # Getting the type of 'self' (line 458)
            self_3889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 458)
            write_3890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 12), self_3889, 'write')
            # Calling write(args, kwargs) (line 458)
            write_call_result_3893 = invoke(stypy.reporting.localization.Localization(__file__, 458, 12), write_3890, *[str_3891], **kwargs_3892)
            
            # SSA branch for the else part of an if statement (line 455)
            module_type_store.open_ssa_branch('else')
            
            # Call to interleave(...): (line 460)
            # Processing the call arguments (line 460)

            @norecursion
            def _stypy_temp_lambda_12(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_12'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_12', 460, 23, True)
                # Passed parameters checking function
                _stypy_temp_lambda_12.stypy_localization = localization
                _stypy_temp_lambda_12.stypy_type_of_self = None
                _stypy_temp_lambda_12.stypy_type_store = module_type_store
                _stypy_temp_lambda_12.stypy_function_name = '_stypy_temp_lambda_12'
                _stypy_temp_lambda_12.stypy_param_names_list = []
                _stypy_temp_lambda_12.stypy_varargs_param_name = None
                _stypy_temp_lambda_12.stypy_kwargs_param_name = None
                _stypy_temp_lambda_12.stypy_call_defaults = defaults
                _stypy_temp_lambda_12.stypy_call_varargs = varargs
                _stypy_temp_lambda_12.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_12', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_12', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to write(...): (line 460)
                # Processing the call arguments (line 460)
                str_3897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 42), 'str', ', ')
                # Processing the call keyword arguments (line 460)
                kwargs_3898 = {}
                # Getting the type of 'self' (line 460)
                self_3895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 31), 'self', False)
                # Obtaining the member 'write' of a type (line 460)
                write_3896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 31), self_3895, 'write')
                # Calling write(args, kwargs) (line 460)
                write_call_result_3899 = invoke(stypy.reporting.localization.Localization(__file__, 460, 31), write_3896, *[str_3897], **kwargs_3898)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 460)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'stypy_return_type', write_call_result_3899)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_12' in the type store
                # Getting the type of 'stypy_return_type' (line 460)
                stypy_return_type_3900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_3900)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_12'
                return stypy_return_type_3900

            # Assigning a type to the variable '_stypy_temp_lambda_12' (line 460)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), '_stypy_temp_lambda_12', _stypy_temp_lambda_12)
            # Getting the type of '_stypy_temp_lambda_12' (line 460)
            _stypy_temp_lambda_12_3901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), '_stypy_temp_lambda_12')
            # Getting the type of 'self' (line 460)
            self_3902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 49), 'self', False)
            # Obtaining the member 'visit' of a type (line 460)
            visit_3903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 49), self_3902, 'visit')
            # Getting the type of 't' (line 460)
            t_3904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 61), 't', False)
            # Obtaining the member 'elts' of a type (line 460)
            elts_3905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 61), t_3904, 'elts')
            # Processing the call keyword arguments (line 460)
            kwargs_3906 = {}
            # Getting the type of 'interleave' (line 460)
            interleave_3894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'interleave', False)
            # Calling interleave(args, kwargs) (line 460)
            interleave_call_result_3907 = invoke(stypy.reporting.localization.Localization(__file__, 460, 12), interleave_3894, *[_stypy_temp_lambda_12_3901, visit_3903, elts_3905], **kwargs_3906)
            
            # SSA join for if statement (line 455)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 461)
        # Processing the call arguments (line 461)
        str_3910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 19), 'str', ')')
        # Processing the call keyword arguments (line 461)
        kwargs_3911 = {}
        # Getting the type of 'self' (line 461)
        self_3908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 461)
        write_3909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 8), self_3908, 'write')
        # Calling write(args, kwargs) (line 461)
        write_call_result_3912 = invoke(stypy.reporting.localization.Localization(__file__, 461, 8), write_3909, *[str_3910], **kwargs_3911)
        
        
        # ################# End of 'visit_Tuple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Tuple' in the type store
        # Getting the type of 'stypy_return_type' (line 453)
        stypy_return_type_3913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3913)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Tuple'
        return stypy_return_type_3913

    
    # Assigning a Dict to a Name (line 463):

    @norecursion
    def visit_UnaryOp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_UnaryOp'
        module_type_store = module_type_store.open_function_context('visit_UnaryOp', 465, 4, False)
        # Assigning a type to the variable 'self' (line 466)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_UnaryOp.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_UnaryOp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_UnaryOp.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_UnaryOp.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_UnaryOp')
        PythonSrcGeneratorVisitor.visit_UnaryOp.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_UnaryOp.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_UnaryOp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_UnaryOp.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_UnaryOp.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_UnaryOp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_UnaryOp.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_UnaryOp', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_UnaryOp', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_UnaryOp(...)' code ##################

        
        # Call to write(...): (line 466)
        # Processing the call arguments (line 466)
        str_3916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 19), 'str', '(')
        # Processing the call keyword arguments (line 466)
        kwargs_3917 = {}
        # Getting the type of 'self' (line 466)
        self_3914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 466)
        write_3915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 8), self_3914, 'write')
        # Calling write(args, kwargs) (line 466)
        write_call_result_3918 = invoke(stypy.reporting.localization.Localization(__file__, 466, 8), write_3915, *[str_3916], **kwargs_3917)
        
        
        # Call to write(...): (line 467)
        # Processing the call arguments (line 467)
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 467)
        t_3921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 29), 't', False)
        # Obtaining the member 'op' of a type (line 467)
        op_3922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 29), t_3921, 'op')
        # Obtaining the member '__class__' of a type (line 467)
        class___3923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 29), op_3922, '__class__')
        # Obtaining the member '__name__' of a type (line 467)
        name___3924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 29), class___3923, '__name__')
        # Getting the type of 'self' (line 467)
        self_3925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 19), 'self', False)
        # Obtaining the member 'unop' of a type (line 467)
        unop_3926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 19), self_3925, 'unop')
        # Obtaining the member '__getitem__' of a type (line 467)
        getitem___3927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 19), unop_3926, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 467)
        subscript_call_result_3928 = invoke(stypy.reporting.localization.Localization(__file__, 467, 19), getitem___3927, name___3924)
        
        # Processing the call keyword arguments (line 467)
        kwargs_3929 = {}
        # Getting the type of 'self' (line 467)
        self_3919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 467)
        write_3920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 8), self_3919, 'write')
        # Calling write(args, kwargs) (line 467)
        write_call_result_3930 = invoke(stypy.reporting.localization.Localization(__file__, 467, 8), write_3920, *[subscript_call_result_3928], **kwargs_3929)
        
        
        # Call to write(...): (line 468)
        # Processing the call arguments (line 468)
        str_3933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 19), 'str', ' ')
        # Processing the call keyword arguments (line 468)
        kwargs_3934 = {}
        # Getting the type of 'self' (line 468)
        self_3931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 468)
        write_3932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), self_3931, 'write')
        # Calling write(args, kwargs) (line 468)
        write_call_result_3935 = invoke(stypy.reporting.localization.Localization(__file__, 468, 8), write_3932, *[str_3933], **kwargs_3934)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 't' (line 474)
        t_3937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 22), 't', False)
        # Obtaining the member 'op' of a type (line 474)
        op_3938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 22), t_3937, 'op')
        # Getting the type of 'ast' (line 474)
        ast_3939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 28), 'ast', False)
        # Obtaining the member 'USub' of a type (line 474)
        USub_3940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 28), ast_3939, 'USub')
        # Processing the call keyword arguments (line 474)
        kwargs_3941 = {}
        # Getting the type of 'isinstance' (line 474)
        isinstance_3936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 474)
        isinstance_call_result_3942 = invoke(stypy.reporting.localization.Localization(__file__, 474, 11), isinstance_3936, *[op_3938, USub_3940], **kwargs_3941)
        
        
        # Call to isinstance(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 't' (line 474)
        t_3944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 53), 't', False)
        # Obtaining the member 'operand' of a type (line 474)
        operand_3945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 53), t_3944, 'operand')
        # Getting the type of 'ast' (line 474)
        ast_3946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 64), 'ast', False)
        # Obtaining the member 'Num' of a type (line 474)
        Num_3947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 64), ast_3946, 'Num')
        # Processing the call keyword arguments (line 474)
        kwargs_3948 = {}
        # Getting the type of 'isinstance' (line 474)
        isinstance_3943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 42), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 474)
        isinstance_call_result_3949 = invoke(stypy.reporting.localization.Localization(__file__, 474, 42), isinstance_3943, *[operand_3945, Num_3947], **kwargs_3948)
        
        # Applying the binary operator 'and' (line 474)
        result_and_keyword_3950 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 11), 'and', isinstance_call_result_3942, isinstance_call_result_3949)
        
        # Testing if the type of an if condition is none (line 474)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 474, 8), result_and_keyword_3950):
            
            # Call to visit(...): (line 479)
            # Processing the call arguments (line 479)
            # Getting the type of 't' (line 479)
            t_3970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 23), 't', False)
            # Obtaining the member 'operand' of a type (line 479)
            operand_3971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 23), t_3970, 'operand')
            # Processing the call keyword arguments (line 479)
            kwargs_3972 = {}
            # Getting the type of 'self' (line 479)
            self_3968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 479)
            visit_3969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 12), self_3968, 'visit')
            # Calling visit(args, kwargs) (line 479)
            visit_call_result_3973 = invoke(stypy.reporting.localization.Localization(__file__, 479, 12), visit_3969, *[operand_3971], **kwargs_3972)
            
        else:
            
            # Testing the type of an if condition (line 474)
            if_condition_3951 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 8), result_and_keyword_3950)
            # Assigning a type to the variable 'if_condition_3951' (line 474)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'if_condition_3951', if_condition_3951)
            # SSA begins for if statement (line 474)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 475)
            # Processing the call arguments (line 475)
            str_3954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 23), 'str', '(')
            # Processing the call keyword arguments (line 475)
            kwargs_3955 = {}
            # Getting the type of 'self' (line 475)
            self_3952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 475)
            write_3953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), self_3952, 'write')
            # Calling write(args, kwargs) (line 475)
            write_call_result_3956 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), write_3953, *[str_3954], **kwargs_3955)
            
            
            # Call to visit(...): (line 476)
            # Processing the call arguments (line 476)
            # Getting the type of 't' (line 476)
            t_3959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 23), 't', False)
            # Obtaining the member 'operand' of a type (line 476)
            operand_3960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 23), t_3959, 'operand')
            # Processing the call keyword arguments (line 476)
            kwargs_3961 = {}
            # Getting the type of 'self' (line 476)
            self_3957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 476)
            visit_3958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 12), self_3957, 'visit')
            # Calling visit(args, kwargs) (line 476)
            visit_call_result_3962 = invoke(stypy.reporting.localization.Localization(__file__, 476, 12), visit_3958, *[operand_3960], **kwargs_3961)
            
            
            # Call to write(...): (line 477)
            # Processing the call arguments (line 477)
            str_3965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 23), 'str', ')')
            # Processing the call keyword arguments (line 477)
            kwargs_3966 = {}
            # Getting the type of 'self' (line 477)
            self_3963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 477)
            write_3964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 12), self_3963, 'write')
            # Calling write(args, kwargs) (line 477)
            write_call_result_3967 = invoke(stypy.reporting.localization.Localization(__file__, 477, 12), write_3964, *[str_3965], **kwargs_3966)
            
            # SSA branch for the else part of an if statement (line 474)
            module_type_store.open_ssa_branch('else')
            
            # Call to visit(...): (line 479)
            # Processing the call arguments (line 479)
            # Getting the type of 't' (line 479)
            t_3970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 23), 't', False)
            # Obtaining the member 'operand' of a type (line 479)
            operand_3971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 23), t_3970, 'operand')
            # Processing the call keyword arguments (line 479)
            kwargs_3972 = {}
            # Getting the type of 'self' (line 479)
            self_3968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 479)
            visit_3969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 12), self_3968, 'visit')
            # Calling visit(args, kwargs) (line 479)
            visit_call_result_3973 = invoke(stypy.reporting.localization.Localization(__file__, 479, 12), visit_3969, *[operand_3971], **kwargs_3972)
            
            # SSA join for if statement (line 474)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 480)
        # Processing the call arguments (line 480)
        str_3976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 19), 'str', ')')
        # Processing the call keyword arguments (line 480)
        kwargs_3977 = {}
        # Getting the type of 'self' (line 480)
        self_3974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 480)
        write_3975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), self_3974, 'write')
        # Calling write(args, kwargs) (line 480)
        write_call_result_3978 = invoke(stypy.reporting.localization.Localization(__file__, 480, 8), write_3975, *[str_3976], **kwargs_3977)
        
        
        # ################# End of 'visit_UnaryOp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_UnaryOp' in the type store
        # Getting the type of 'stypy_return_type' (line 465)
        stypy_return_type_3979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3979)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_UnaryOp'
        return stypy_return_type_3979

    
    # Assigning a Dict to a Name (line 482):

    @norecursion
    def visit_BinOp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_BinOp'
        module_type_store = module_type_store.open_function_context('visit_BinOp', 486, 4, False)
        # Assigning a type to the variable 'self' (line 487)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 487, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_BinOp.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_BinOp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_BinOp.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_BinOp.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_BinOp')
        PythonSrcGeneratorVisitor.visit_BinOp.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_BinOp.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_BinOp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_BinOp.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_BinOp.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_BinOp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_BinOp.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_BinOp', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_BinOp', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_BinOp(...)' code ##################

        
        # Call to write(...): (line 487)
        # Processing the call arguments (line 487)
        str_3982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 19), 'str', '(')
        # Processing the call keyword arguments (line 487)
        kwargs_3983 = {}
        # Getting the type of 'self' (line 487)
        self_3980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 487)
        write_3981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 8), self_3980, 'write')
        # Calling write(args, kwargs) (line 487)
        write_call_result_3984 = invoke(stypy.reporting.localization.Localization(__file__, 487, 8), write_3981, *[str_3982], **kwargs_3983)
        
        
        # Call to visit(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 't' (line 488)
        t_3987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 19), 't', False)
        # Obtaining the member 'left' of a type (line 488)
        left_3988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 19), t_3987, 'left')
        # Processing the call keyword arguments (line 488)
        kwargs_3989 = {}
        # Getting the type of 'self' (line 488)
        self_3985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 488)
        visit_3986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 8), self_3985, 'visit')
        # Calling visit(args, kwargs) (line 488)
        visit_call_result_3990 = invoke(stypy.reporting.localization.Localization(__file__, 488, 8), visit_3986, *[left_3988], **kwargs_3989)
        
        
        # Call to write(...): (line 489)
        # Processing the call arguments (line 489)
        str_3993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 19), 'str', ' ')
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 489)
        t_3994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 36), 't', False)
        # Obtaining the member 'op' of a type (line 489)
        op_3995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 36), t_3994, 'op')
        # Obtaining the member '__class__' of a type (line 489)
        class___3996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 36), op_3995, '__class__')
        # Obtaining the member '__name__' of a type (line 489)
        name___3997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 36), class___3996, '__name__')
        # Getting the type of 'self' (line 489)
        self_3998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 25), 'self', False)
        # Obtaining the member 'binop' of a type (line 489)
        binop_3999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 25), self_3998, 'binop')
        # Obtaining the member '__getitem__' of a type (line 489)
        getitem___4000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 25), binop_3999, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 489)
        subscript_call_result_4001 = invoke(stypy.reporting.localization.Localization(__file__, 489, 25), getitem___4000, name___3997)
        
        # Applying the binary operator '+' (line 489)
        result_add_4002 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 19), '+', str_3993, subscript_call_result_4001)
        
        str_4003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 63), 'str', ' ')
        # Applying the binary operator '+' (line 489)
        result_add_4004 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 61), '+', result_add_4002, str_4003)
        
        # Processing the call keyword arguments (line 489)
        kwargs_4005 = {}
        # Getting the type of 'self' (line 489)
        self_3991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 489)
        write_3992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 8), self_3991, 'write')
        # Calling write(args, kwargs) (line 489)
        write_call_result_4006 = invoke(stypy.reporting.localization.Localization(__file__, 489, 8), write_3992, *[result_add_4004], **kwargs_4005)
        
        
        # Call to visit(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 't' (line 490)
        t_4009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 19), 't', False)
        # Obtaining the member 'right' of a type (line 490)
        right_4010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 19), t_4009, 'right')
        # Processing the call keyword arguments (line 490)
        kwargs_4011 = {}
        # Getting the type of 'self' (line 490)
        self_4007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 490)
        visit_4008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), self_4007, 'visit')
        # Calling visit(args, kwargs) (line 490)
        visit_call_result_4012 = invoke(stypy.reporting.localization.Localization(__file__, 490, 8), visit_4008, *[right_4010], **kwargs_4011)
        
        
        # Call to write(...): (line 491)
        # Processing the call arguments (line 491)
        str_4015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 19), 'str', ')')
        # Processing the call keyword arguments (line 491)
        kwargs_4016 = {}
        # Getting the type of 'self' (line 491)
        self_4013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 491)
        write_4014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 8), self_4013, 'write')
        # Calling write(args, kwargs) (line 491)
        write_call_result_4017 = invoke(stypy.reporting.localization.Localization(__file__, 491, 8), write_4014, *[str_4015], **kwargs_4016)
        
        
        # ################# End of 'visit_BinOp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_BinOp' in the type store
        # Getting the type of 'stypy_return_type' (line 486)
        stypy_return_type_4018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4018)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_BinOp'
        return stypy_return_type_4018

    
    # Assigning a Dict to a Name (line 493):

    @norecursion
    def visit_Compare(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Compare'
        module_type_store = module_type_store.open_function_context('visit_Compare', 496, 4, False)
        # Assigning a type to the variable 'self' (line 497)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 497, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Compare.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Compare.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Compare.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Compare.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Compare')
        PythonSrcGeneratorVisitor.visit_Compare.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Compare.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Compare.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Compare.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Compare.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Compare.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Compare.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Compare', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Compare', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Compare(...)' code ##################

        
        # Call to write(...): (line 497)
        # Processing the call arguments (line 497)
        str_4021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 19), 'str', '(')
        # Processing the call keyword arguments (line 497)
        kwargs_4022 = {}
        # Getting the type of 'self' (line 497)
        self_4019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 497)
        write_4020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), self_4019, 'write')
        # Calling write(args, kwargs) (line 497)
        write_call_result_4023 = invoke(stypy.reporting.localization.Localization(__file__, 497, 8), write_4020, *[str_4021], **kwargs_4022)
        
        
        # Call to visit(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 't' (line 498)
        t_4026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 19), 't', False)
        # Obtaining the member 'left' of a type (line 498)
        left_4027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 19), t_4026, 'left')
        # Processing the call keyword arguments (line 498)
        kwargs_4028 = {}
        # Getting the type of 'self' (line 498)
        self_4024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 498)
        visit_4025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), self_4024, 'visit')
        # Calling visit(args, kwargs) (line 498)
        visit_call_result_4029 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), visit_4025, *[left_4027], **kwargs_4028)
        
        
        
        # Call to zip(...): (line 499)
        # Processing the call arguments (line 499)
        # Getting the type of 't' (line 499)
        t_4031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 24), 't', False)
        # Obtaining the member 'ops' of a type (line 499)
        ops_4032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 24), t_4031, 'ops')
        # Getting the type of 't' (line 499)
        t_4033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 31), 't', False)
        # Obtaining the member 'comparators' of a type (line 499)
        comparators_4034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 31), t_4033, 'comparators')
        # Processing the call keyword arguments (line 499)
        kwargs_4035 = {}
        # Getting the type of 'zip' (line 499)
        zip_4030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 499)
        zip_call_result_4036 = invoke(stypy.reporting.localization.Localization(__file__, 499, 20), zip_4030, *[ops_4032, comparators_4034], **kwargs_4035)
        
        # Assigning a type to the variable 'zip_call_result_4036' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'zip_call_result_4036', zip_call_result_4036)
        # Testing if the for loop is going to be iterated (line 499)
        # Testing the type of a for loop iterable (line 499)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 499, 8), zip_call_result_4036)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 499, 8), zip_call_result_4036):
            # Getting the type of the for loop variable (line 499)
            for_loop_var_4037 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 499, 8), zip_call_result_4036)
            # Assigning a type to the variable 'o' (line 499)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'o', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 8), for_loop_var_4037, 2, 0))
            # Assigning a type to the variable 'e' (line 499)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'e', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 8), for_loop_var_4037, 2, 1))
            # SSA begins for a for statement (line 499)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to write(...): (line 500)
            # Processing the call arguments (line 500)
            str_4040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 23), 'str', ' ')
            
            # Obtaining the type of the subscript
            # Getting the type of 'o' (line 500)
            o_4041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 41), 'o', False)
            # Obtaining the member '__class__' of a type (line 500)
            class___4042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 41), o_4041, '__class__')
            # Obtaining the member '__name__' of a type (line 500)
            name___4043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 41), class___4042, '__name__')
            # Getting the type of 'self' (line 500)
            self_4044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 29), 'self', False)
            # Obtaining the member 'cmpops' of a type (line 500)
            cmpops_4045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 29), self_4044, 'cmpops')
            # Obtaining the member '__getitem__' of a type (line 500)
            getitem___4046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 29), cmpops_4045, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 500)
            subscript_call_result_4047 = invoke(stypy.reporting.localization.Localization(__file__, 500, 29), getitem___4046, name___4043)
            
            # Applying the binary operator '+' (line 500)
            result_add_4048 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 23), '+', str_4040, subscript_call_result_4047)
            
            str_4049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 65), 'str', ' ')
            # Applying the binary operator '+' (line 500)
            result_add_4050 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 63), '+', result_add_4048, str_4049)
            
            # Processing the call keyword arguments (line 500)
            kwargs_4051 = {}
            # Getting the type of 'self' (line 500)
            self_4038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 500)
            write_4039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 12), self_4038, 'write')
            # Calling write(args, kwargs) (line 500)
            write_call_result_4052 = invoke(stypy.reporting.localization.Localization(__file__, 500, 12), write_4039, *[result_add_4050], **kwargs_4051)
            
            
            # Call to visit(...): (line 501)
            # Processing the call arguments (line 501)
            # Getting the type of 'e' (line 501)
            e_4055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 23), 'e', False)
            # Processing the call keyword arguments (line 501)
            kwargs_4056 = {}
            # Getting the type of 'self' (line 501)
            self_4053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 501)
            visit_4054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 12), self_4053, 'visit')
            # Calling visit(args, kwargs) (line 501)
            visit_call_result_4057 = invoke(stypy.reporting.localization.Localization(__file__, 501, 12), visit_4054, *[e_4055], **kwargs_4056)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 502)
        # Processing the call arguments (line 502)
        str_4060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 19), 'str', ')')
        # Processing the call keyword arguments (line 502)
        kwargs_4061 = {}
        # Getting the type of 'self' (line 502)
        self_4058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 502)
        write_4059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 8), self_4058, 'write')
        # Calling write(args, kwargs) (line 502)
        write_call_result_4062 = invoke(stypy.reporting.localization.Localization(__file__, 502, 8), write_4059, *[str_4060], **kwargs_4061)
        
        
        # ################# End of 'visit_Compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Compare' in the type store
        # Getting the type of 'stypy_return_type' (line 496)
        stypy_return_type_4063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4063)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Compare'
        return stypy_return_type_4063

    
    # Assigning a Dict to a Name (line 504):

    @norecursion
    def visit_BoolOp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_BoolOp'
        module_type_store = module_type_store.open_function_context('visit_BoolOp', 506, 4, False)
        # Assigning a type to the variable 'self' (line 507)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 507, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_BoolOp.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_BoolOp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_BoolOp.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_BoolOp.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_BoolOp')
        PythonSrcGeneratorVisitor.visit_BoolOp.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_BoolOp.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_BoolOp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_BoolOp.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_BoolOp.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_BoolOp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_BoolOp.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_BoolOp', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_BoolOp', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_BoolOp(...)' code ##################

        
        # Call to write(...): (line 507)
        # Processing the call arguments (line 507)
        str_4066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 19), 'str', '(')
        # Processing the call keyword arguments (line 507)
        kwargs_4067 = {}
        # Getting the type of 'self' (line 507)
        self_4064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 507)
        write_4065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 8), self_4064, 'write')
        # Calling write(args, kwargs) (line 507)
        write_call_result_4068 = invoke(stypy.reporting.localization.Localization(__file__, 507, 8), write_4065, *[str_4066], **kwargs_4067)
        
        
        # Assigning a BinOp to a Name (line 508):
        
        # Assigning a BinOp to a Name (line 508):
        str_4069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 12), 'str', ' %s ')
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 508)
        t_4070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 34), 't')
        # Obtaining the member 'op' of a type (line 508)
        op_4071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 34), t_4070, 'op')
        # Obtaining the member '__class__' of a type (line 508)
        class___4072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 34), op_4071, '__class__')
        # Getting the type of 'self' (line 508)
        self_4073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 21), 'self')
        # Obtaining the member 'boolops' of a type (line 508)
        boolops_4074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 21), self_4073, 'boolops')
        # Obtaining the member '__getitem__' of a type (line 508)
        getitem___4075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 21), boolops_4074, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 508)
        subscript_call_result_4076 = invoke(stypy.reporting.localization.Localization(__file__, 508, 21), getitem___4075, class___4072)
        
        # Applying the binary operator '%' (line 508)
        result_mod_4077 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 12), '%', str_4069, subscript_call_result_4076)
        
        # Assigning a type to the variable 's' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 's', result_mod_4077)
        
        # Call to interleave(...): (line 509)
        # Processing the call arguments (line 509)

        @norecursion
        def _stypy_temp_lambda_13(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_13'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_13', 509, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_13.stypy_localization = localization
            _stypy_temp_lambda_13.stypy_type_of_self = None
            _stypy_temp_lambda_13.stypy_type_store = module_type_store
            _stypy_temp_lambda_13.stypy_function_name = '_stypy_temp_lambda_13'
            _stypy_temp_lambda_13.stypy_param_names_list = []
            _stypy_temp_lambda_13.stypy_varargs_param_name = None
            _stypy_temp_lambda_13.stypy_kwargs_param_name = None
            _stypy_temp_lambda_13.stypy_call_defaults = defaults
            _stypy_temp_lambda_13.stypy_call_varargs = varargs
            _stypy_temp_lambda_13.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_13', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_13', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 509)
            # Processing the call arguments (line 509)
            # Getting the type of 's' (line 509)
            s_4081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 38), 's', False)
            # Processing the call keyword arguments (line 509)
            kwargs_4082 = {}
            # Getting the type of 'self' (line 509)
            self_4079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 509)
            write_4080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 27), self_4079, 'write')
            # Calling write(args, kwargs) (line 509)
            write_call_result_4083 = invoke(stypy.reporting.localization.Localization(__file__, 509, 27), write_4080, *[s_4081], **kwargs_4082)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 509)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 19), 'stypy_return_type', write_call_result_4083)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_13' in the type store
            # Getting the type of 'stypy_return_type' (line 509)
            stypy_return_type_4084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_4084)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_13'
            return stypy_return_type_4084

        # Assigning a type to the variable '_stypy_temp_lambda_13' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 19), '_stypy_temp_lambda_13', _stypy_temp_lambda_13)
        # Getting the type of '_stypy_temp_lambda_13' (line 509)
        _stypy_temp_lambda_13_4085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 19), '_stypy_temp_lambda_13')
        # Getting the type of 'self' (line 509)
        self_4086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 42), 'self', False)
        # Obtaining the member 'visit' of a type (line 509)
        visit_4087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 42), self_4086, 'visit')
        # Getting the type of 't' (line 509)
        t_4088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 54), 't', False)
        # Obtaining the member 'values' of a type (line 509)
        values_4089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 54), t_4088, 'values')
        # Processing the call keyword arguments (line 509)
        kwargs_4090 = {}
        # Getting the type of 'interleave' (line 509)
        interleave_4078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 509)
        interleave_call_result_4091 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), interleave_4078, *[_stypy_temp_lambda_13_4085, visit_4087, values_4089], **kwargs_4090)
        
        
        # Call to write(...): (line 510)
        # Processing the call arguments (line 510)
        str_4094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 19), 'str', ')')
        # Processing the call keyword arguments (line 510)
        kwargs_4095 = {}
        # Getting the type of 'self' (line 510)
        self_4092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 510)
        write_4093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 8), self_4092, 'write')
        # Calling write(args, kwargs) (line 510)
        write_call_result_4096 = invoke(stypy.reporting.localization.Localization(__file__, 510, 8), write_4093, *[str_4094], **kwargs_4095)
        
        
        # ################# End of 'visit_BoolOp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_BoolOp' in the type store
        # Getting the type of 'stypy_return_type' (line 506)
        stypy_return_type_4097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4097)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_BoolOp'
        return stypy_return_type_4097


    @norecursion
    def visit_Attribute(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Attribute'
        module_type_store = module_type_store.open_function_context('visit_Attribute', 512, 4, False)
        # Assigning a type to the variable 'self' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Attribute.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Attribute.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Attribute.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Attribute.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Attribute')
        PythonSrcGeneratorVisitor.visit_Attribute.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Attribute.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Attribute.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Attribute.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Attribute.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Attribute.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Attribute.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Attribute', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Attribute', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Attribute(...)' code ##################

        
        # Call to visit(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 't' (line 513)
        t_4100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 19), 't', False)
        # Obtaining the member 'value' of a type (line 513)
        value_4101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 19), t_4100, 'value')
        # Processing the call keyword arguments (line 513)
        kwargs_4102 = {}
        # Getting the type of 'self' (line 513)
        self_4098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 513)
        visit_4099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 8), self_4098, 'visit')
        # Calling visit(args, kwargs) (line 513)
        visit_call_result_4103 = invoke(stypy.reporting.localization.Localization(__file__, 513, 8), visit_4099, *[value_4101], **kwargs_4102)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 't' (line 517)
        t_4105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 22), 't', False)
        # Obtaining the member 'value' of a type (line 517)
        value_4106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 22), t_4105, 'value')
        # Getting the type of 'ast' (line 517)
        ast_4107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 31), 'ast', False)
        # Obtaining the member 'Num' of a type (line 517)
        Num_4108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 31), ast_4107, 'Num')
        # Processing the call keyword arguments (line 517)
        kwargs_4109 = {}
        # Getting the type of 'isinstance' (line 517)
        isinstance_4104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 517)
        isinstance_call_result_4110 = invoke(stypy.reporting.localization.Localization(__file__, 517, 11), isinstance_4104, *[value_4106, Num_4108], **kwargs_4109)
        
        
        # Call to isinstance(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 't' (line 517)
        t_4112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 55), 't', False)
        # Obtaining the member 'value' of a type (line 517)
        value_4113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 55), t_4112, 'value')
        # Obtaining the member 'n' of a type (line 517)
        n_4114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 55), value_4113, 'n')
        # Getting the type of 'int' (line 517)
        int_4115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 66), 'int', False)
        # Processing the call keyword arguments (line 517)
        kwargs_4116 = {}
        # Getting the type of 'isinstance' (line 517)
        isinstance_4111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 44), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 517)
        isinstance_call_result_4117 = invoke(stypy.reporting.localization.Localization(__file__, 517, 44), isinstance_4111, *[n_4114, int_4115], **kwargs_4116)
        
        # Applying the binary operator 'and' (line 517)
        result_and_keyword_4118 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 11), 'and', isinstance_call_result_4110, isinstance_call_result_4117)
        
        # Testing if the type of an if condition is none (line 517)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 517, 8), result_and_keyword_4118):
            pass
        else:
            
            # Testing the type of an if condition (line 517)
            if_condition_4119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 8), result_and_keyword_4118)
            # Assigning a type to the variable 'if_condition_4119' (line 517)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'if_condition_4119', if_condition_4119)
            # SSA begins for if statement (line 517)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 518)
            # Processing the call arguments (line 518)
            str_4122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 23), 'str', ' ')
            # Processing the call keyword arguments (line 518)
            kwargs_4123 = {}
            # Getting the type of 'self' (line 518)
            self_4120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 518)
            write_4121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), self_4120, 'write')
            # Calling write(args, kwargs) (line 518)
            write_call_result_4124 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), write_4121, *[str_4122], **kwargs_4123)
            
            # SSA join for if statement (line 517)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 519)
        # Processing the call arguments (line 519)
        str_4127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 19), 'str', '.')
        # Processing the call keyword arguments (line 519)
        kwargs_4128 = {}
        # Getting the type of 'self' (line 519)
        self_4125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 519)
        write_4126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 8), self_4125, 'write')
        # Calling write(args, kwargs) (line 519)
        write_call_result_4129 = invoke(stypy.reporting.localization.Localization(__file__, 519, 8), write_4126, *[str_4127], **kwargs_4128)
        
        
        # Call to write(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 't' (line 520)
        t_4132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 19), 't', False)
        # Obtaining the member 'attr' of a type (line 520)
        attr_4133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 19), t_4132, 'attr')
        # Processing the call keyword arguments (line 520)
        kwargs_4134 = {}
        # Getting the type of 'self' (line 520)
        self_4130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 520)
        write_4131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 8), self_4130, 'write')
        # Calling write(args, kwargs) (line 520)
        write_call_result_4135 = invoke(stypy.reporting.localization.Localization(__file__, 520, 8), write_4131, *[attr_4133], **kwargs_4134)
        
        
        # ################# End of 'visit_Attribute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Attribute' in the type store
        # Getting the type of 'stypy_return_type' (line 512)
        stypy_return_type_4136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4136)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Attribute'
        return stypy_return_type_4136


    @norecursion
    def visit_Call(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Call'
        module_type_store = module_type_store.open_function_context('visit_Call', 522, 4, False)
        # Assigning a type to the variable 'self' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Call.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Call.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Call.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Call.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Call')
        PythonSrcGeneratorVisitor.visit_Call.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Call.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Call.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Call.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Call.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Call.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Call.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Call', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Call', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Call(...)' code ##################

        
        # Call to visit(...): (line 523)
        # Processing the call arguments (line 523)
        # Getting the type of 't' (line 523)
        t_4139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 19), 't', False)
        # Obtaining the member 'func' of a type (line 523)
        func_4140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 19), t_4139, 'func')
        # Processing the call keyword arguments (line 523)
        kwargs_4141 = {}
        # Getting the type of 'self' (line 523)
        self_4137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 523)
        visit_4138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 8), self_4137, 'visit')
        # Calling visit(args, kwargs) (line 523)
        visit_call_result_4142 = invoke(stypy.reporting.localization.Localization(__file__, 523, 8), visit_4138, *[func_4140], **kwargs_4141)
        
        
        # Call to write(...): (line 524)
        # Processing the call arguments (line 524)
        str_4145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 19), 'str', '(')
        # Processing the call keyword arguments (line 524)
        kwargs_4146 = {}
        # Getting the type of 'self' (line 524)
        self_4143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 524)
        write_4144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 8), self_4143, 'write')
        # Calling write(args, kwargs) (line 524)
        write_call_result_4147 = invoke(stypy.reporting.localization.Localization(__file__, 524, 8), write_4144, *[str_4145], **kwargs_4146)
        
        
        # Assigning a Name to a Name (line 525):
        
        # Assigning a Name to a Name (line 525):
        # Getting the type of 'False' (line 525)
        False_4148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'False')
        # Assigning a type to the variable 'comma' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'comma', False_4148)
        
        # Getting the type of 't' (line 526)
        t_4149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 17), 't')
        # Obtaining the member 'args' of a type (line 526)
        args_4150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 17), t_4149, 'args')
        # Assigning a type to the variable 'args_4150' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'args_4150', args_4150)
        # Testing if the for loop is going to be iterated (line 526)
        # Testing the type of a for loop iterable (line 526)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 526, 8), args_4150)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 526, 8), args_4150):
            # Getting the type of the for loop variable (line 526)
            for_loop_var_4151 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 526, 8), args_4150)
            # Assigning a type to the variable 'e' (line 526)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'e', for_loop_var_4151)
            # SSA begins for a for statement (line 526)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'comma' (line 527)
            comma_4152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 15), 'comma')
            # Testing if the type of an if condition is none (line 527)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 527, 12), comma_4152):
                
                # Assigning a Name to a Name (line 530):
                
                # Assigning a Name to a Name (line 530):
                # Getting the type of 'True' (line 530)
                True_4159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 24), 'True')
                # Assigning a type to the variable 'comma' (line 530)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 16), 'comma', True_4159)
            else:
                
                # Testing the type of an if condition (line 527)
                if_condition_4153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 527, 12), comma_4152)
                # Assigning a type to the variable 'if_condition_4153' (line 527)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'if_condition_4153', if_condition_4153)
                # SSA begins for if statement (line 527)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 528)
                # Processing the call arguments (line 528)
                str_4156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 27), 'str', ', ')
                # Processing the call keyword arguments (line 528)
                kwargs_4157 = {}
                # Getting the type of 'self' (line 528)
                self_4154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 528)
                write_4155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 16), self_4154, 'write')
                # Calling write(args, kwargs) (line 528)
                write_call_result_4158 = invoke(stypy.reporting.localization.Localization(__file__, 528, 16), write_4155, *[str_4156], **kwargs_4157)
                
                # SSA branch for the else part of an if statement (line 527)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 530):
                
                # Assigning a Name to a Name (line 530):
                # Getting the type of 'True' (line 530)
                True_4159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 24), 'True')
                # Assigning a type to the variable 'comma' (line 530)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 16), 'comma', True_4159)
                # SSA join for if statement (line 527)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to visit(...): (line 531)
            # Processing the call arguments (line 531)
            # Getting the type of 'e' (line 531)
            e_4162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 23), 'e', False)
            # Processing the call keyword arguments (line 531)
            kwargs_4163 = {}
            # Getting the type of 'self' (line 531)
            self_4160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 531)
            visit_4161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 12), self_4160, 'visit')
            # Calling visit(args, kwargs) (line 531)
            visit_call_result_4164 = invoke(stypy.reporting.localization.Localization(__file__, 531, 12), visit_4161, *[e_4162], **kwargs_4163)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 't' (line 532)
        t_4165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 17), 't')
        # Obtaining the member 'keywords' of a type (line 532)
        keywords_4166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 17), t_4165, 'keywords')
        # Assigning a type to the variable 'keywords_4166' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'keywords_4166', keywords_4166)
        # Testing if the for loop is going to be iterated (line 532)
        # Testing the type of a for loop iterable (line 532)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 532, 8), keywords_4166)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 532, 8), keywords_4166):
            # Getting the type of the for loop variable (line 532)
            for_loop_var_4167 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 532, 8), keywords_4166)
            # Assigning a type to the variable 'e' (line 532)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'e', for_loop_var_4167)
            # SSA begins for a for statement (line 532)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'comma' (line 533)
            comma_4168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 15), 'comma')
            # Testing if the type of an if condition is none (line 533)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 533, 12), comma_4168):
                
                # Assigning a Name to a Name (line 536):
                
                # Assigning a Name to a Name (line 536):
                # Getting the type of 'True' (line 536)
                True_4175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 24), 'True')
                # Assigning a type to the variable 'comma' (line 536)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'comma', True_4175)
            else:
                
                # Testing the type of an if condition (line 533)
                if_condition_4169 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 533, 12), comma_4168)
                # Assigning a type to the variable 'if_condition_4169' (line 533)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'if_condition_4169', if_condition_4169)
                # SSA begins for if statement (line 533)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 534)
                # Processing the call arguments (line 534)
                str_4172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 27), 'str', ', ')
                # Processing the call keyword arguments (line 534)
                kwargs_4173 = {}
                # Getting the type of 'self' (line 534)
                self_4170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 534)
                write_4171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 16), self_4170, 'write')
                # Calling write(args, kwargs) (line 534)
                write_call_result_4174 = invoke(stypy.reporting.localization.Localization(__file__, 534, 16), write_4171, *[str_4172], **kwargs_4173)
                
                # SSA branch for the else part of an if statement (line 533)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 536):
                
                # Assigning a Name to a Name (line 536):
                # Getting the type of 'True' (line 536)
                True_4175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 24), 'True')
                # Assigning a type to the variable 'comma' (line 536)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'comma', True_4175)
                # SSA join for if statement (line 533)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to visit(...): (line 537)
            # Processing the call arguments (line 537)
            # Getting the type of 'e' (line 537)
            e_4178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 23), 'e', False)
            # Processing the call keyword arguments (line 537)
            kwargs_4179 = {}
            # Getting the type of 'self' (line 537)
            self_4176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 537)
            visit_4177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), self_4176, 'visit')
            # Calling visit(args, kwargs) (line 537)
            visit_call_result_4180 = invoke(stypy.reporting.localization.Localization(__file__, 537, 12), visit_4177, *[e_4178], **kwargs_4179)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 't' (line 538)
        t_4181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 11), 't')
        # Obtaining the member 'starargs' of a type (line 538)
        starargs_4182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 11), t_4181, 'starargs')
        # Testing if the type of an if condition is none (line 538)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 538, 8), starargs_4182):
            pass
        else:
            
            # Testing the type of an if condition (line 538)
            if_condition_4183 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 538, 8), starargs_4182)
            # Assigning a type to the variable 'if_condition_4183' (line 538)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'if_condition_4183', if_condition_4183)
            # SSA begins for if statement (line 538)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'comma' (line 539)
            comma_4184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 15), 'comma')
            # Testing if the type of an if condition is none (line 539)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 539, 12), comma_4184):
                
                # Assigning a Name to a Name (line 542):
                
                # Assigning a Name to a Name (line 542):
                # Getting the type of 'True' (line 542)
                True_4191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 24), 'True')
                # Assigning a type to the variable 'comma' (line 542)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'comma', True_4191)
            else:
                
                # Testing the type of an if condition (line 539)
                if_condition_4185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 539, 12), comma_4184)
                # Assigning a type to the variable 'if_condition_4185' (line 539)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'if_condition_4185', if_condition_4185)
                # SSA begins for if statement (line 539)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 540)
                # Processing the call arguments (line 540)
                str_4188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 27), 'str', ', ')
                # Processing the call keyword arguments (line 540)
                kwargs_4189 = {}
                # Getting the type of 'self' (line 540)
                self_4186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 540)
                write_4187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 16), self_4186, 'write')
                # Calling write(args, kwargs) (line 540)
                write_call_result_4190 = invoke(stypy.reporting.localization.Localization(__file__, 540, 16), write_4187, *[str_4188], **kwargs_4189)
                
                # SSA branch for the else part of an if statement (line 539)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 542):
                
                # Assigning a Name to a Name (line 542):
                # Getting the type of 'True' (line 542)
                True_4191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 24), 'True')
                # Assigning a type to the variable 'comma' (line 542)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'comma', True_4191)
                # SSA join for if statement (line 539)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 543)
            # Processing the call arguments (line 543)
            str_4194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 23), 'str', '*')
            # Processing the call keyword arguments (line 543)
            kwargs_4195 = {}
            # Getting the type of 'self' (line 543)
            self_4192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 543)
            write_4193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 12), self_4192, 'write')
            # Calling write(args, kwargs) (line 543)
            write_call_result_4196 = invoke(stypy.reporting.localization.Localization(__file__, 543, 12), write_4193, *[str_4194], **kwargs_4195)
            
            
            # Call to visit(...): (line 544)
            # Processing the call arguments (line 544)
            # Getting the type of 't' (line 544)
            t_4199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 23), 't', False)
            # Obtaining the member 'starargs' of a type (line 544)
            starargs_4200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 23), t_4199, 'starargs')
            # Processing the call keyword arguments (line 544)
            kwargs_4201 = {}
            # Getting the type of 'self' (line 544)
            self_4197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 544)
            visit_4198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 12), self_4197, 'visit')
            # Calling visit(args, kwargs) (line 544)
            visit_call_result_4202 = invoke(stypy.reporting.localization.Localization(__file__, 544, 12), visit_4198, *[starargs_4200], **kwargs_4201)
            
            # SSA join for if statement (line 538)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 545)
        t_4203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 11), 't')
        # Obtaining the member 'kwargs' of a type (line 545)
        kwargs_4204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 11), t_4203, 'kwargs')
        # Testing if the type of an if condition is none (line 545)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 545, 8), kwargs_4204):
            pass
        else:
            
            # Testing the type of an if condition (line 545)
            if_condition_4205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 545, 8), kwargs_4204)
            # Assigning a type to the variable 'if_condition_4205' (line 545)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'if_condition_4205', if_condition_4205)
            # SSA begins for if statement (line 545)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'comma' (line 546)
            comma_4206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 15), 'comma')
            # Testing if the type of an if condition is none (line 546)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 546, 12), comma_4206):
                
                # Assigning a Name to a Name (line 549):
                
                # Assigning a Name to a Name (line 549):
                # Getting the type of 'True' (line 549)
                True_4213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 24), 'True')
                # Assigning a type to the variable 'comma' (line 549)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'comma', True_4213)
            else:
                
                # Testing the type of an if condition (line 546)
                if_condition_4207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 546, 12), comma_4206)
                # Assigning a type to the variable 'if_condition_4207' (line 546)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'if_condition_4207', if_condition_4207)
                # SSA begins for if statement (line 546)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 547)
                # Processing the call arguments (line 547)
                str_4210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 27), 'str', ', ')
                # Processing the call keyword arguments (line 547)
                kwargs_4211 = {}
                # Getting the type of 'self' (line 547)
                self_4208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 547)
                write_4209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 16), self_4208, 'write')
                # Calling write(args, kwargs) (line 547)
                write_call_result_4212 = invoke(stypy.reporting.localization.Localization(__file__, 547, 16), write_4209, *[str_4210], **kwargs_4211)
                
                # SSA branch for the else part of an if statement (line 546)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 549):
                
                # Assigning a Name to a Name (line 549):
                # Getting the type of 'True' (line 549)
                True_4213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 24), 'True')
                # Assigning a type to the variable 'comma' (line 549)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'comma', True_4213)
                # SSA join for if statement (line 546)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 550)
            # Processing the call arguments (line 550)
            str_4216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 23), 'str', '**')
            # Processing the call keyword arguments (line 550)
            kwargs_4217 = {}
            # Getting the type of 'self' (line 550)
            self_4214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 550)
            write_4215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 12), self_4214, 'write')
            # Calling write(args, kwargs) (line 550)
            write_call_result_4218 = invoke(stypy.reporting.localization.Localization(__file__, 550, 12), write_4215, *[str_4216], **kwargs_4217)
            
            
            # Call to visit(...): (line 551)
            # Processing the call arguments (line 551)
            # Getting the type of 't' (line 551)
            t_4221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 23), 't', False)
            # Obtaining the member 'kwargs' of a type (line 551)
            kwargs_4222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 23), t_4221, 'kwargs')
            # Processing the call keyword arguments (line 551)
            kwargs_4223 = {}
            # Getting the type of 'self' (line 551)
            self_4219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 551)
            visit_4220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 12), self_4219, 'visit')
            # Calling visit(args, kwargs) (line 551)
            visit_call_result_4224 = invoke(stypy.reporting.localization.Localization(__file__, 551, 12), visit_4220, *[kwargs_4222], **kwargs_4223)
            
            # SSA join for if statement (line 545)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 552)
        # Processing the call arguments (line 552)
        str_4227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 19), 'str', ')')
        # Processing the call keyword arguments (line 552)
        kwargs_4228 = {}
        # Getting the type of 'self' (line 552)
        self_4225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 552)
        write_4226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 8), self_4225, 'write')
        # Calling write(args, kwargs) (line 552)
        write_call_result_4229 = invoke(stypy.reporting.localization.Localization(__file__, 552, 8), write_4226, *[str_4227], **kwargs_4228)
        
        
        # ################# End of 'visit_Call(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Call' in the type store
        # Getting the type of 'stypy_return_type' (line 522)
        stypy_return_type_4230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4230)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Call'
        return stypy_return_type_4230


    @norecursion
    def visit_Subscript(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Subscript'
        module_type_store = module_type_store.open_function_context('visit_Subscript', 554, 4, False)
        # Assigning a type to the variable 'self' (line 555)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 555, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Subscript.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Subscript.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Subscript.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Subscript.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Subscript')
        PythonSrcGeneratorVisitor.visit_Subscript.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Subscript.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Subscript.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Subscript.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Subscript.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Subscript.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Subscript.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Subscript', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Subscript', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Subscript(...)' code ##################

        
        # Call to visit(...): (line 555)
        # Processing the call arguments (line 555)
        # Getting the type of 't' (line 555)
        t_4233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 19), 't', False)
        # Obtaining the member 'value' of a type (line 555)
        value_4234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 19), t_4233, 'value')
        # Processing the call keyword arguments (line 555)
        kwargs_4235 = {}
        # Getting the type of 'self' (line 555)
        self_4231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 555)
        visit_4232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 8), self_4231, 'visit')
        # Calling visit(args, kwargs) (line 555)
        visit_call_result_4236 = invoke(stypy.reporting.localization.Localization(__file__, 555, 8), visit_4232, *[value_4234], **kwargs_4235)
        
        
        # Call to write(...): (line 556)
        # Processing the call arguments (line 556)
        str_4239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 19), 'str', '[')
        # Processing the call keyword arguments (line 556)
        kwargs_4240 = {}
        # Getting the type of 'self' (line 556)
        self_4237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 556)
        write_4238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 8), self_4237, 'write')
        # Calling write(args, kwargs) (line 556)
        write_call_result_4241 = invoke(stypy.reporting.localization.Localization(__file__, 556, 8), write_4238, *[str_4239], **kwargs_4240)
        
        
        # Call to visit(...): (line 557)
        # Processing the call arguments (line 557)
        # Getting the type of 't' (line 557)
        t_4244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 19), 't', False)
        # Obtaining the member 'slice' of a type (line 557)
        slice_4245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 19), t_4244, 'slice')
        # Processing the call keyword arguments (line 557)
        kwargs_4246 = {}
        # Getting the type of 'self' (line 557)
        self_4242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 557)
        visit_4243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 8), self_4242, 'visit')
        # Calling visit(args, kwargs) (line 557)
        visit_call_result_4247 = invoke(stypy.reporting.localization.Localization(__file__, 557, 8), visit_4243, *[slice_4245], **kwargs_4246)
        
        
        # Call to write(...): (line 558)
        # Processing the call arguments (line 558)
        str_4250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 19), 'str', ']')
        # Processing the call keyword arguments (line 558)
        kwargs_4251 = {}
        # Getting the type of 'self' (line 558)
        self_4248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 558)
        write_4249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 8), self_4248, 'write')
        # Calling write(args, kwargs) (line 558)
        write_call_result_4252 = invoke(stypy.reporting.localization.Localization(__file__, 558, 8), write_4249, *[str_4250], **kwargs_4251)
        
        
        # ################# End of 'visit_Subscript(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Subscript' in the type store
        # Getting the type of 'stypy_return_type' (line 554)
        stypy_return_type_4253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4253)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Subscript'
        return stypy_return_type_4253


    @norecursion
    def visit_Ellipsis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Ellipsis'
        module_type_store = module_type_store.open_function_context('visit_Ellipsis', 561, 4, False)
        # Assigning a type to the variable 'self' (line 562)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 562, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Ellipsis.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Ellipsis.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Ellipsis.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Ellipsis.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Ellipsis')
        PythonSrcGeneratorVisitor.visit_Ellipsis.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Ellipsis.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Ellipsis.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Ellipsis.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Ellipsis.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Ellipsis.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Ellipsis.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Ellipsis', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Ellipsis', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Ellipsis(...)' code ##################

        
        # Call to write(...): (line 562)
        # Processing the call arguments (line 562)
        str_4256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 19), 'str', '...')
        # Processing the call keyword arguments (line 562)
        kwargs_4257 = {}
        # Getting the type of 'self' (line 562)
        self_4254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 562)
        write_4255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 8), self_4254, 'write')
        # Calling write(args, kwargs) (line 562)
        write_call_result_4258 = invoke(stypy.reporting.localization.Localization(__file__, 562, 8), write_4255, *[str_4256], **kwargs_4257)
        
        
        # ################# End of 'visit_Ellipsis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Ellipsis' in the type store
        # Getting the type of 'stypy_return_type' (line 561)
        stypy_return_type_4259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4259)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Ellipsis'
        return stypy_return_type_4259


    @norecursion
    def visit_Index(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Index'
        module_type_store = module_type_store.open_function_context('visit_Index', 564, 4, False)
        # Assigning a type to the variable 'self' (line 565)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 565, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Index.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Index.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Index.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Index.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Index')
        PythonSrcGeneratorVisitor.visit_Index.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Index.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Index.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Index.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Index.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Index.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Index.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Index', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Index', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Index(...)' code ##################

        
        # Call to visit(...): (line 565)
        # Processing the call arguments (line 565)
        # Getting the type of 't' (line 565)
        t_4262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 19), 't', False)
        # Obtaining the member 'value' of a type (line 565)
        value_4263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 19), t_4262, 'value')
        # Processing the call keyword arguments (line 565)
        kwargs_4264 = {}
        # Getting the type of 'self' (line 565)
        self_4260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 565)
        visit_4261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 8), self_4260, 'visit')
        # Calling visit(args, kwargs) (line 565)
        visit_call_result_4265 = invoke(stypy.reporting.localization.Localization(__file__, 565, 8), visit_4261, *[value_4263], **kwargs_4264)
        
        
        # ################# End of 'visit_Index(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Index' in the type store
        # Getting the type of 'stypy_return_type' (line 564)
        stypy_return_type_4266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4266)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Index'
        return stypy_return_type_4266


    @norecursion
    def visit_Slice(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Slice'
        module_type_store = module_type_store.open_function_context('visit_Slice', 567, 4, False)
        # Assigning a type to the variable 'self' (line 568)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Slice.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Slice.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Slice.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Slice.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Slice')
        PythonSrcGeneratorVisitor.visit_Slice.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Slice.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Slice.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Slice.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Slice.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Slice.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Slice.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Slice', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Slice', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Slice(...)' code ##################

        # Getting the type of 't' (line 568)
        t_4267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 11), 't')
        # Obtaining the member 'lower' of a type (line 568)
        lower_4268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 11), t_4267, 'lower')
        # Testing if the type of an if condition is none (line 568)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 568, 8), lower_4268):
            pass
        else:
            
            # Testing the type of an if condition (line 568)
            if_condition_4269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 8), lower_4268)
            # Assigning a type to the variable 'if_condition_4269' (line 568)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'if_condition_4269', if_condition_4269)
            # SSA begins for if statement (line 568)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to visit(...): (line 569)
            # Processing the call arguments (line 569)
            # Getting the type of 't' (line 569)
            t_4272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 23), 't', False)
            # Obtaining the member 'lower' of a type (line 569)
            lower_4273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 23), t_4272, 'lower')
            # Processing the call keyword arguments (line 569)
            kwargs_4274 = {}
            # Getting the type of 'self' (line 569)
            self_4270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 569)
            visit_4271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 12), self_4270, 'visit')
            # Calling visit(args, kwargs) (line 569)
            visit_call_result_4275 = invoke(stypy.reporting.localization.Localization(__file__, 569, 12), visit_4271, *[lower_4273], **kwargs_4274)
            
            # SSA join for if statement (line 568)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 570)
        # Processing the call arguments (line 570)
        str_4278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 19), 'str', ':')
        # Processing the call keyword arguments (line 570)
        kwargs_4279 = {}
        # Getting the type of 'self' (line 570)
        self_4276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 570)
        write_4277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), self_4276, 'write')
        # Calling write(args, kwargs) (line 570)
        write_call_result_4280 = invoke(stypy.reporting.localization.Localization(__file__, 570, 8), write_4277, *[str_4278], **kwargs_4279)
        
        # Getting the type of 't' (line 571)
        t_4281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 11), 't')
        # Obtaining the member 'upper' of a type (line 571)
        upper_4282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 11), t_4281, 'upper')
        # Testing if the type of an if condition is none (line 571)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 571, 8), upper_4282):
            pass
        else:
            
            # Testing the type of an if condition (line 571)
            if_condition_4283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 571, 8), upper_4282)
            # Assigning a type to the variable 'if_condition_4283' (line 571)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'if_condition_4283', if_condition_4283)
            # SSA begins for if statement (line 571)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to visit(...): (line 572)
            # Processing the call arguments (line 572)
            # Getting the type of 't' (line 572)
            t_4286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 23), 't', False)
            # Obtaining the member 'upper' of a type (line 572)
            upper_4287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 23), t_4286, 'upper')
            # Processing the call keyword arguments (line 572)
            kwargs_4288 = {}
            # Getting the type of 'self' (line 572)
            self_4284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 572)
            visit_4285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 12), self_4284, 'visit')
            # Calling visit(args, kwargs) (line 572)
            visit_call_result_4289 = invoke(stypy.reporting.localization.Localization(__file__, 572, 12), visit_4285, *[upper_4287], **kwargs_4288)
            
            # SSA join for if statement (line 571)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 573)
        t_4290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 11), 't')
        # Obtaining the member 'step' of a type (line 573)
        step_4291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 11), t_4290, 'step')
        # Testing if the type of an if condition is none (line 573)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 573, 8), step_4291):
            pass
        else:
            
            # Testing the type of an if condition (line 573)
            if_condition_4292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 573, 8), step_4291)
            # Assigning a type to the variable 'if_condition_4292' (line 573)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'if_condition_4292', if_condition_4292)
            # SSA begins for if statement (line 573)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 574)
            # Processing the call arguments (line 574)
            str_4295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 23), 'str', ':')
            # Processing the call keyword arguments (line 574)
            kwargs_4296 = {}
            # Getting the type of 'self' (line 574)
            self_4293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 574)
            write_4294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 12), self_4293, 'write')
            # Calling write(args, kwargs) (line 574)
            write_call_result_4297 = invoke(stypy.reporting.localization.Localization(__file__, 574, 12), write_4294, *[str_4295], **kwargs_4296)
            
            
            # Call to visit(...): (line 575)
            # Processing the call arguments (line 575)
            # Getting the type of 't' (line 575)
            t_4300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 23), 't', False)
            # Obtaining the member 'step' of a type (line 575)
            step_4301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 23), t_4300, 'step')
            # Processing the call keyword arguments (line 575)
            kwargs_4302 = {}
            # Getting the type of 'self' (line 575)
            self_4298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 575)
            visit_4299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), self_4298, 'visit')
            # Calling visit(args, kwargs) (line 575)
            visit_call_result_4303 = invoke(stypy.reporting.localization.Localization(__file__, 575, 12), visit_4299, *[step_4301], **kwargs_4302)
            
            # SSA join for if statement (line 573)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Slice(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Slice' in the type store
        # Getting the type of 'stypy_return_type' (line 567)
        stypy_return_type_4304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4304)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Slice'
        return stypy_return_type_4304


    @norecursion
    def visit_ExtSlice(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ExtSlice'
        module_type_store = module_type_store.open_function_context('visit_ExtSlice', 577, 4, False)
        # Assigning a type to the variable 'self' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_ExtSlice.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_ExtSlice.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_ExtSlice.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_ExtSlice.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_ExtSlice')
        PythonSrcGeneratorVisitor.visit_ExtSlice.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_ExtSlice.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_ExtSlice.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_ExtSlice.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_ExtSlice.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_ExtSlice.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_ExtSlice.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_ExtSlice', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_ExtSlice', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_ExtSlice(...)' code ##################

        
        # Call to interleave(...): (line 578)
        # Processing the call arguments (line 578)

        @norecursion
        def _stypy_temp_lambda_14(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_14'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_14', 578, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_14.stypy_localization = localization
            _stypy_temp_lambda_14.stypy_type_of_self = None
            _stypy_temp_lambda_14.stypy_type_store = module_type_store
            _stypy_temp_lambda_14.stypy_function_name = '_stypy_temp_lambda_14'
            _stypy_temp_lambda_14.stypy_param_names_list = []
            _stypy_temp_lambda_14.stypy_varargs_param_name = None
            _stypy_temp_lambda_14.stypy_kwargs_param_name = None
            _stypy_temp_lambda_14.stypy_call_defaults = defaults
            _stypy_temp_lambda_14.stypy_call_varargs = varargs
            _stypy_temp_lambda_14.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_14', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_14', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 578)
            # Processing the call arguments (line 578)
            str_4308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 38), 'str', ', ')
            # Processing the call keyword arguments (line 578)
            kwargs_4309 = {}
            # Getting the type of 'self' (line 578)
            self_4306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 578)
            write_4307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 27), self_4306, 'write')
            # Calling write(args, kwargs) (line 578)
            write_call_result_4310 = invoke(stypy.reporting.localization.Localization(__file__, 578, 27), write_4307, *[str_4308], **kwargs_4309)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 578)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 19), 'stypy_return_type', write_call_result_4310)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_14' in the type store
            # Getting the type of 'stypy_return_type' (line 578)
            stypy_return_type_4311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_4311)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_14'
            return stypy_return_type_4311

        # Assigning a type to the variable '_stypy_temp_lambda_14' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 19), '_stypy_temp_lambda_14', _stypy_temp_lambda_14)
        # Getting the type of '_stypy_temp_lambda_14' (line 578)
        _stypy_temp_lambda_14_4312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 19), '_stypy_temp_lambda_14')
        # Getting the type of 'self' (line 578)
        self_4313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 578)
        visit_4314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 45), self_4313, 'visit')
        # Getting the type of 't' (line 578)
        t_4315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 57), 't', False)
        # Obtaining the member 'dims' of a type (line 578)
        dims_4316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 57), t_4315, 'dims')
        # Processing the call keyword arguments (line 578)
        kwargs_4317 = {}
        # Getting the type of 'interleave' (line 578)
        interleave_4305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 578)
        interleave_call_result_4318 = invoke(stypy.reporting.localization.Localization(__file__, 578, 8), interleave_4305, *[_stypy_temp_lambda_14_4312, visit_4314, dims_4316], **kwargs_4317)
        
        
        # ################# End of 'visit_ExtSlice(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ExtSlice' in the type store
        # Getting the type of 'stypy_return_type' (line 577)
        stypy_return_type_4319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4319)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ExtSlice'
        return stypy_return_type_4319


    @norecursion
    def visit_arguments(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_arguments'
        module_type_store = module_type_store.open_function_context('visit_arguments', 581, 4, False)
        # Assigning a type to the variable 'self' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_arguments.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_arguments.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_arguments.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_arguments.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_arguments')
        PythonSrcGeneratorVisitor.visit_arguments.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_arguments.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_arguments.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_arguments.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_arguments.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_arguments.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_arguments.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_arguments', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_arguments', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_arguments(...)' code ##################

        
        # Assigning a Name to a Name (line 582):
        
        # Assigning a Name to a Name (line 582):
        # Getting the type of 'True' (line 582)
        True_4320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'True')
        # Assigning a type to the variable 'first' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'first', True_4320)
        
        # Assigning a BinOp to a Name (line 584):
        
        # Assigning a BinOp to a Name (line 584):
        
        # Obtaining an instance of the builtin type 'list' (line 584)
        list_4321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 584)
        # Adding element type (line 584)
        # Getting the type of 'None' (line 584)
        None_4322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 19), list_4321, None_4322)
        
        
        # Call to len(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 't' (line 584)
        t_4324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 33), 't', False)
        # Obtaining the member 'args' of a type (line 584)
        args_4325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 33), t_4324, 'args')
        # Processing the call keyword arguments (line 584)
        kwargs_4326 = {}
        # Getting the type of 'len' (line 584)
        len_4323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 29), 'len', False)
        # Calling len(args, kwargs) (line 584)
        len_call_result_4327 = invoke(stypy.reporting.localization.Localization(__file__, 584, 29), len_4323, *[args_4325], **kwargs_4326)
        
        
        # Call to len(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 't' (line 584)
        t_4329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 47), 't', False)
        # Obtaining the member 'defaults' of a type (line 584)
        defaults_4330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 47), t_4329, 'defaults')
        # Processing the call keyword arguments (line 584)
        kwargs_4331 = {}
        # Getting the type of 'len' (line 584)
        len_4328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 43), 'len', False)
        # Calling len(args, kwargs) (line 584)
        len_call_result_4332 = invoke(stypy.reporting.localization.Localization(__file__, 584, 43), len_4328, *[defaults_4330], **kwargs_4331)
        
        # Applying the binary operator '-' (line 584)
        result_sub_4333 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 29), '-', len_call_result_4327, len_call_result_4332)
        
        # Applying the binary operator '*' (line 584)
        result_mul_4334 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 19), '*', list_4321, result_sub_4333)
        
        # Getting the type of 't' (line 584)
        t_4335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 62), 't')
        # Obtaining the member 'defaults' of a type (line 584)
        defaults_4336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 62), t_4335, 'defaults')
        # Applying the binary operator '+' (line 584)
        result_add_4337 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 19), '+', result_mul_4334, defaults_4336)
        
        # Assigning a type to the variable 'defaults' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'defaults', result_add_4337)
        
        
        # Call to zip(...): (line 585)
        # Processing the call arguments (line 585)
        # Getting the type of 't' (line 585)
        t_4339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 24), 't', False)
        # Obtaining the member 'args' of a type (line 585)
        args_4340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 24), t_4339, 'args')
        # Getting the type of 'defaults' (line 585)
        defaults_4341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 32), 'defaults', False)
        # Processing the call keyword arguments (line 585)
        kwargs_4342 = {}
        # Getting the type of 'zip' (line 585)
        zip_4338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 585)
        zip_call_result_4343 = invoke(stypy.reporting.localization.Localization(__file__, 585, 20), zip_4338, *[args_4340, defaults_4341], **kwargs_4342)
        
        # Assigning a type to the variable 'zip_call_result_4343' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'zip_call_result_4343', zip_call_result_4343)
        # Testing if the for loop is going to be iterated (line 585)
        # Testing the type of a for loop iterable (line 585)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 585, 8), zip_call_result_4343)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 585, 8), zip_call_result_4343):
            # Getting the type of the for loop variable (line 585)
            for_loop_var_4344 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 585, 8), zip_call_result_4343)
            # Assigning a type to the variable 'a' (line 585)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 8), for_loop_var_4344, 2, 0))
            # Assigning a type to the variable 'd' (line 585)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 8), for_loop_var_4344, 2, 1))
            # SSA begins for a for statement (line 585)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'first' (line 586)
            first_4345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 15), 'first')
            # Testing if the type of an if condition is none (line 586)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 586, 12), first_4345):
                
                # Call to write(...): (line 589)
                # Processing the call arguments (line 589)
                str_4350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 27), 'str', ', ')
                # Processing the call keyword arguments (line 589)
                kwargs_4351 = {}
                # Getting the type of 'self' (line 589)
                self_4348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 589)
                write_4349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 16), self_4348, 'write')
                # Calling write(args, kwargs) (line 589)
                write_call_result_4352 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), write_4349, *[str_4350], **kwargs_4351)
                
            else:
                
                # Testing the type of an if condition (line 586)
                if_condition_4346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 586, 12), first_4345)
                # Assigning a type to the variable 'if_condition_4346' (line 586)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'if_condition_4346', if_condition_4346)
                # SSA begins for if statement (line 586)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 587):
                
                # Assigning a Name to a Name (line 587):
                # Getting the type of 'False' (line 587)
                False_4347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 24), 'False')
                # Assigning a type to the variable 'first' (line 587)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'first', False_4347)
                # SSA branch for the else part of an if statement (line 586)
                module_type_store.open_ssa_branch('else')
                
                # Call to write(...): (line 589)
                # Processing the call arguments (line 589)
                str_4350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 27), 'str', ', ')
                # Processing the call keyword arguments (line 589)
                kwargs_4351 = {}
                # Getting the type of 'self' (line 589)
                self_4348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 589)
                write_4349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 16), self_4348, 'write')
                # Calling write(args, kwargs) (line 589)
                write_call_result_4352 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), write_4349, *[str_4350], **kwargs_4351)
                
                # SSA join for if statement (line 586)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Obtaining an instance of the builtin type 'tuple' (line 590)
            tuple_4353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 12), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 590)
            # Adding element type (line 590)
            
            # Call to visit(...): (line 590)
            # Processing the call arguments (line 590)
            # Getting the type of 'a' (line 590)
            a_4356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 23), 'a', False)
            # Processing the call keyword arguments (line 590)
            kwargs_4357 = {}
            # Getting the type of 'self' (line 590)
            self_4354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 590)
            visit_4355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 12), self_4354, 'visit')
            # Calling visit(args, kwargs) (line 590)
            visit_call_result_4358 = invoke(stypy.reporting.localization.Localization(__file__, 590, 12), visit_4355, *[a_4356], **kwargs_4357)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 12), tuple_4353, visit_call_result_4358)
            
            # Getting the type of 'd' (line 591)
            d_4359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 15), 'd')
            # Testing if the type of an if condition is none (line 591)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 591, 12), d_4359):
                pass
            else:
                
                # Testing the type of an if condition (line 591)
                if_condition_4360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 591, 12), d_4359)
                # Assigning a type to the variable 'if_condition_4360' (line 591)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 12), 'if_condition_4360', if_condition_4360)
                # SSA begins for if statement (line 591)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 592)
                # Processing the call arguments (line 592)
                str_4363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 27), 'str', '=')
                # Processing the call keyword arguments (line 592)
                kwargs_4364 = {}
                # Getting the type of 'self' (line 592)
                self_4361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 592)
                write_4362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 16), self_4361, 'write')
                # Calling write(args, kwargs) (line 592)
                write_call_result_4365 = invoke(stypy.reporting.localization.Localization(__file__, 592, 16), write_4362, *[str_4363], **kwargs_4364)
                
                
                # Call to visit(...): (line 593)
                # Processing the call arguments (line 593)
                # Getting the type of 'd' (line 593)
                d_4368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 27), 'd', False)
                # Processing the call keyword arguments (line 593)
                kwargs_4369 = {}
                # Getting the type of 'self' (line 593)
                self_4366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 16), 'self', False)
                # Obtaining the member 'visit' of a type (line 593)
                visit_4367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 16), self_4366, 'visit')
                # Calling visit(args, kwargs) (line 593)
                visit_call_result_4370 = invoke(stypy.reporting.localization.Localization(__file__, 593, 16), visit_4367, *[d_4368], **kwargs_4369)
                
                # SSA join for if statement (line 591)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 't' (line 596)
        t_4371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 11), 't')
        # Obtaining the member 'vararg' of a type (line 596)
        vararg_4372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 11), t_4371, 'vararg')
        # Testing if the type of an if condition is none (line 596)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 596, 8), vararg_4372):
            pass
        else:
            
            # Testing the type of an if condition (line 596)
            if_condition_4373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 596, 8), vararg_4372)
            # Assigning a type to the variable 'if_condition_4373' (line 596)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'if_condition_4373', if_condition_4373)
            # SSA begins for if statement (line 596)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'first' (line 597)
            first_4374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 15), 'first')
            # Testing if the type of an if condition is none (line 597)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 597, 12), first_4374):
                
                # Call to write(...): (line 600)
                # Processing the call arguments (line 600)
                str_4379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 27), 'str', ', ')
                # Processing the call keyword arguments (line 600)
                kwargs_4380 = {}
                # Getting the type of 'self' (line 600)
                self_4377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 600)
                write_4378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 16), self_4377, 'write')
                # Calling write(args, kwargs) (line 600)
                write_call_result_4381 = invoke(stypy.reporting.localization.Localization(__file__, 600, 16), write_4378, *[str_4379], **kwargs_4380)
                
            else:
                
                # Testing the type of an if condition (line 597)
                if_condition_4375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 12), first_4374)
                # Assigning a type to the variable 'if_condition_4375' (line 597)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'if_condition_4375', if_condition_4375)
                # SSA begins for if statement (line 597)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 598):
                
                # Assigning a Name to a Name (line 598):
                # Getting the type of 'False' (line 598)
                False_4376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 24), 'False')
                # Assigning a type to the variable 'first' (line 598)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), 'first', False_4376)
                # SSA branch for the else part of an if statement (line 597)
                module_type_store.open_ssa_branch('else')
                
                # Call to write(...): (line 600)
                # Processing the call arguments (line 600)
                str_4379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 27), 'str', ', ')
                # Processing the call keyword arguments (line 600)
                kwargs_4380 = {}
                # Getting the type of 'self' (line 600)
                self_4377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 600)
                write_4378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 16), self_4377, 'write')
                # Calling write(args, kwargs) (line 600)
                write_call_result_4381 = invoke(stypy.reporting.localization.Localization(__file__, 600, 16), write_4378, *[str_4379], **kwargs_4380)
                
                # SSA join for if statement (line 597)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 601)
            # Processing the call arguments (line 601)
            str_4384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 23), 'str', '*')
            # Processing the call keyword arguments (line 601)
            kwargs_4385 = {}
            # Getting the type of 'self' (line 601)
            self_4382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 601)
            write_4383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 12), self_4382, 'write')
            # Calling write(args, kwargs) (line 601)
            write_call_result_4386 = invoke(stypy.reporting.localization.Localization(__file__, 601, 12), write_4383, *[str_4384], **kwargs_4385)
            
            
            # Call to write(...): (line 602)
            # Processing the call arguments (line 602)
            # Getting the type of 't' (line 602)
            t_4389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 23), 't', False)
            # Obtaining the member 'vararg' of a type (line 602)
            vararg_4390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 23), t_4389, 'vararg')
            # Processing the call keyword arguments (line 602)
            kwargs_4391 = {}
            # Getting the type of 'self' (line 602)
            self_4387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 602)
            write_4388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 12), self_4387, 'write')
            # Calling write(args, kwargs) (line 602)
            write_call_result_4392 = invoke(stypy.reporting.localization.Localization(__file__, 602, 12), write_4388, *[vararg_4390], **kwargs_4391)
            
            # SSA join for if statement (line 596)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 605)
        t_4393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 11), 't')
        # Obtaining the member 'kwarg' of a type (line 605)
        kwarg_4394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 11), t_4393, 'kwarg')
        # Testing if the type of an if condition is none (line 605)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 605, 8), kwarg_4394):
            pass
        else:
            
            # Testing the type of an if condition (line 605)
            if_condition_4395 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 605, 8), kwarg_4394)
            # Assigning a type to the variable 'if_condition_4395' (line 605)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'if_condition_4395', if_condition_4395)
            # SSA begins for if statement (line 605)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'first' (line 606)
            first_4396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 15), 'first')
            # Testing if the type of an if condition is none (line 606)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 606, 12), first_4396):
                
                # Call to write(...): (line 609)
                # Processing the call arguments (line 609)
                str_4401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 27), 'str', ', ')
                # Processing the call keyword arguments (line 609)
                kwargs_4402 = {}
                # Getting the type of 'self' (line 609)
                self_4399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 609)
                write_4400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 16), self_4399, 'write')
                # Calling write(args, kwargs) (line 609)
                write_call_result_4403 = invoke(stypy.reporting.localization.Localization(__file__, 609, 16), write_4400, *[str_4401], **kwargs_4402)
                
            else:
                
                # Testing the type of an if condition (line 606)
                if_condition_4397 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 606, 12), first_4396)
                # Assigning a type to the variable 'if_condition_4397' (line 606)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 12), 'if_condition_4397', if_condition_4397)
                # SSA begins for if statement (line 606)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 607):
                
                # Assigning a Name to a Name (line 607):
                # Getting the type of 'False' (line 607)
                False_4398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 24), 'False')
                # Assigning a type to the variable 'first' (line 607)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 16), 'first', False_4398)
                # SSA branch for the else part of an if statement (line 606)
                module_type_store.open_ssa_branch('else')
                
                # Call to write(...): (line 609)
                # Processing the call arguments (line 609)
                str_4401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 27), 'str', ', ')
                # Processing the call keyword arguments (line 609)
                kwargs_4402 = {}
                # Getting the type of 'self' (line 609)
                self_4399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 609)
                write_4400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 16), self_4399, 'write')
                # Calling write(args, kwargs) (line 609)
                write_call_result_4403 = invoke(stypy.reporting.localization.Localization(__file__, 609, 16), write_4400, *[str_4401], **kwargs_4402)
                
                # SSA join for if statement (line 606)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 610)
            # Processing the call arguments (line 610)
            str_4406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 23), 'str', '**')
            # Getting the type of 't' (line 610)
            t_4407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 30), 't', False)
            # Obtaining the member 'kwarg' of a type (line 610)
            kwarg_4408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 30), t_4407, 'kwarg')
            # Applying the binary operator '+' (line 610)
            result_add_4409 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 23), '+', str_4406, kwarg_4408)
            
            # Processing the call keyword arguments (line 610)
            kwargs_4410 = {}
            # Getting the type of 'self' (line 610)
            self_4404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 610)
            write_4405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 12), self_4404, 'write')
            # Calling write(args, kwargs) (line 610)
            write_call_result_4411 = invoke(stypy.reporting.localization.Localization(__file__, 610, 12), write_4405, *[result_add_4409], **kwargs_4410)
            
            # SSA join for if statement (line 605)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_arguments(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_arguments' in the type store
        # Getting the type of 'stypy_return_type' (line 581)
        stypy_return_type_4412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4412)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_arguments'
        return stypy_return_type_4412


    @norecursion
    def visit_keyword(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_keyword'
        module_type_store = module_type_store.open_function_context('visit_keyword', 612, 4, False)
        # Assigning a type to the variable 'self' (line 613)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 613, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_keyword.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_keyword.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_keyword.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_keyword.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_keyword')
        PythonSrcGeneratorVisitor.visit_keyword.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_keyword.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_keyword.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_keyword.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_keyword.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_keyword.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_keyword.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_keyword', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_keyword', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_keyword(...)' code ##################

        
        # Call to write(...): (line 613)
        # Processing the call arguments (line 613)
        # Getting the type of 't' (line 613)
        t_4415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 19), 't', False)
        # Obtaining the member 'arg' of a type (line 613)
        arg_4416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 19), t_4415, 'arg')
        # Processing the call keyword arguments (line 613)
        kwargs_4417 = {}
        # Getting the type of 'self' (line 613)
        self_4413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 613)
        write_4414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 8), self_4413, 'write')
        # Calling write(args, kwargs) (line 613)
        write_call_result_4418 = invoke(stypy.reporting.localization.Localization(__file__, 613, 8), write_4414, *[arg_4416], **kwargs_4417)
        
        
        # Call to write(...): (line 614)
        # Processing the call arguments (line 614)
        str_4421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 19), 'str', '=')
        # Processing the call keyword arguments (line 614)
        kwargs_4422 = {}
        # Getting the type of 'self' (line 614)
        self_4419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 614)
        write_4420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 8), self_4419, 'write')
        # Calling write(args, kwargs) (line 614)
        write_call_result_4423 = invoke(stypy.reporting.localization.Localization(__file__, 614, 8), write_4420, *[str_4421], **kwargs_4422)
        
        
        # Call to visit(...): (line 615)
        # Processing the call arguments (line 615)
        # Getting the type of 't' (line 615)
        t_4426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 19), 't', False)
        # Obtaining the member 'value' of a type (line 615)
        value_4427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 19), t_4426, 'value')
        # Processing the call keyword arguments (line 615)
        kwargs_4428 = {}
        # Getting the type of 'self' (line 615)
        self_4424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 615)
        visit_4425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 8), self_4424, 'visit')
        # Calling visit(args, kwargs) (line 615)
        visit_call_result_4429 = invoke(stypy.reporting.localization.Localization(__file__, 615, 8), visit_4425, *[value_4427], **kwargs_4428)
        
        
        # ################# End of 'visit_keyword(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_keyword' in the type store
        # Getting the type of 'stypy_return_type' (line 612)
        stypy_return_type_4430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4430)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_keyword'
        return stypy_return_type_4430


    @norecursion
    def visit_Lambda(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Lambda'
        module_type_store = module_type_store.open_function_context('visit_Lambda', 617, 4, False)
        # Assigning a type to the variable 'self' (line 618)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 618, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_Lambda.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_Lambda.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_Lambda.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_Lambda.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_Lambda')
        PythonSrcGeneratorVisitor.visit_Lambda.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_Lambda.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Lambda.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_Lambda.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_Lambda.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_Lambda.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_Lambda.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_Lambda', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_Lambda', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_Lambda(...)' code ##################

        
        # Call to write(...): (line 618)
        # Processing the call arguments (line 618)
        str_4433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 19), 'str', '(')
        # Processing the call keyword arguments (line 618)
        kwargs_4434 = {}
        # Getting the type of 'self' (line 618)
        self_4431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 618)
        write_4432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 8), self_4431, 'write')
        # Calling write(args, kwargs) (line 618)
        write_call_result_4435 = invoke(stypy.reporting.localization.Localization(__file__, 618, 8), write_4432, *[str_4433], **kwargs_4434)
        
        
        # Call to write(...): (line 619)
        # Processing the call arguments (line 619)
        str_4438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 19), 'str', 'lambda ')
        # Processing the call keyword arguments (line 619)
        kwargs_4439 = {}
        # Getting the type of 'self' (line 619)
        self_4436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 619)
        write_4437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 8), self_4436, 'write')
        # Calling write(args, kwargs) (line 619)
        write_call_result_4440 = invoke(stypy.reporting.localization.Localization(__file__, 619, 8), write_4437, *[str_4438], **kwargs_4439)
        
        
        # Call to visit(...): (line 620)
        # Processing the call arguments (line 620)
        # Getting the type of 't' (line 620)
        t_4443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 19), 't', False)
        # Obtaining the member 'args' of a type (line 620)
        args_4444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 19), t_4443, 'args')
        # Processing the call keyword arguments (line 620)
        kwargs_4445 = {}
        # Getting the type of 'self' (line 620)
        self_4441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 620)
        visit_4442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 8), self_4441, 'visit')
        # Calling visit(args, kwargs) (line 620)
        visit_call_result_4446 = invoke(stypy.reporting.localization.Localization(__file__, 620, 8), visit_4442, *[args_4444], **kwargs_4445)
        
        
        # Call to write(...): (line 621)
        # Processing the call arguments (line 621)
        str_4449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 19), 'str', ': ')
        # Processing the call keyword arguments (line 621)
        kwargs_4450 = {}
        # Getting the type of 'self' (line 621)
        self_4447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 621)
        write_4448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 8), self_4447, 'write')
        # Calling write(args, kwargs) (line 621)
        write_call_result_4451 = invoke(stypy.reporting.localization.Localization(__file__, 621, 8), write_4448, *[str_4449], **kwargs_4450)
        
        
        # Call to visit(...): (line 622)
        # Processing the call arguments (line 622)
        # Getting the type of 't' (line 622)
        t_4454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 19), 't', False)
        # Obtaining the member 'body' of a type (line 622)
        body_4455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 19), t_4454, 'body')
        # Processing the call keyword arguments (line 622)
        kwargs_4456 = {}
        # Getting the type of 'self' (line 622)
        self_4452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 622)
        visit_4453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 8), self_4452, 'visit')
        # Calling visit(args, kwargs) (line 622)
        visit_call_result_4457 = invoke(stypy.reporting.localization.Localization(__file__, 622, 8), visit_4453, *[body_4455], **kwargs_4456)
        
        
        # Call to write(...): (line 623)
        # Processing the call arguments (line 623)
        str_4460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 19), 'str', ')')
        # Processing the call keyword arguments (line 623)
        kwargs_4461 = {}
        # Getting the type of 'self' (line 623)
        self_4458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 623)
        write_4459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 8), self_4458, 'write')
        # Calling write(args, kwargs) (line 623)
        write_call_result_4462 = invoke(stypy.reporting.localization.Localization(__file__, 623, 8), write_4459, *[str_4460], **kwargs_4461)
        
        
        # ################# End of 'visit_Lambda(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Lambda' in the type store
        # Getting the type of 'stypy_return_type' (line 617)
        stypy_return_type_4463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4463)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Lambda'
        return stypy_return_type_4463


    @norecursion
    def visit_alias(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_alias'
        module_type_store = module_type_store.open_function_context('visit_alias', 625, 4, False)
        # Assigning a type to the variable 'self' (line 626)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.visit_alias.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.visit_alias.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.visit_alias.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.visit_alias.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.visit_alias')
        PythonSrcGeneratorVisitor.visit_alias.__dict__.__setitem__('stypy_param_names_list', ['t'])
        PythonSrcGeneratorVisitor.visit_alias.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_alias.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.visit_alias.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.visit_alias.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.visit_alias.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.visit_alias.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.visit_alias', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit_alias', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit_alias(...)' code ##################

        
        # Call to write(...): (line 626)
        # Processing the call arguments (line 626)
        # Getting the type of 't' (line 626)
        t_4466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 19), 't', False)
        # Obtaining the member 'name' of a type (line 626)
        name_4467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 19), t_4466, 'name')
        # Processing the call keyword arguments (line 626)
        kwargs_4468 = {}
        # Getting the type of 'self' (line 626)
        self_4464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 626)
        write_4465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 8), self_4464, 'write')
        # Calling write(args, kwargs) (line 626)
        write_call_result_4469 = invoke(stypy.reporting.localization.Localization(__file__, 626, 8), write_4465, *[name_4467], **kwargs_4468)
        
        # Getting the type of 't' (line 627)
        t_4470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 11), 't')
        # Obtaining the member 'asname' of a type (line 627)
        asname_4471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 11), t_4470, 'asname')
        # Testing if the type of an if condition is none (line 627)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 627, 8), asname_4471):
            pass
        else:
            
            # Testing the type of an if condition (line 627)
            if_condition_4472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 627, 8), asname_4471)
            # Assigning a type to the variable 'if_condition_4472' (line 627)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'if_condition_4472', if_condition_4472)
            # SSA begins for if statement (line 627)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 628)
            # Processing the call arguments (line 628)
            str_4475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 23), 'str', ' as ')
            # Getting the type of 't' (line 628)
            t_4476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 32), 't', False)
            # Obtaining the member 'asname' of a type (line 628)
            asname_4477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 32), t_4476, 'asname')
            # Applying the binary operator '+' (line 628)
            result_add_4478 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 23), '+', str_4475, asname_4477)
            
            # Processing the call keyword arguments (line 628)
            kwargs_4479 = {}
            # Getting the type of 'self' (line 628)
            self_4473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 628)
            write_4474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 12), self_4473, 'write')
            # Calling write(args, kwargs) (line 628)
            write_call_result_4480 = invoke(stypy.reporting.localization.Localization(__file__, 628, 12), write_4474, *[result_add_4478], **kwargs_4479)
            
            # SSA join for if statement (line 627)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_alias(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_alias' in the type store
        # Getting the type of 'stypy_return_type' (line 625)
        stypy_return_type_4481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4481)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_alias'
        return stypy_return_type_4481


# Assigning a type to the variable 'PythonSrcGeneratorVisitor' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'PythonSrcGeneratorVisitor', PythonSrcGeneratorVisitor)

# Assigning a Dict to a Name (line 463):

# Obtaining an instance of the builtin type 'dict' (line 463)
dict_4482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 463)
# Adding element type (key, value) (line 463)
str_4483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 12), 'str', 'Invert')
str_4484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 22), 'str', '~')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 11), dict_4482, (str_4483, str_4484))
# Adding element type (key, value) (line 463)
str_4485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 27), 'str', 'Not')
str_4486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 34), 'str', 'not')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 11), dict_4482, (str_4485, str_4486))
# Adding element type (key, value) (line 463)
str_4487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 41), 'str', 'UAdd')
str_4488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 49), 'str', '+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 11), dict_4482, (str_4487, str_4488))
# Adding element type (key, value) (line 463)
str_4489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 54), 'str', 'USub')
str_4490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 62), 'str', '-')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 11), dict_4482, (str_4489, str_4490))

# Getting the type of 'PythonSrcGeneratorVisitor'
PythonSrcGeneratorVisitor_4491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PythonSrcGeneratorVisitor')
# Setting the type of the member 'unop' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PythonSrcGeneratorVisitor_4491, 'unop', dict_4482)

# Assigning a Dict to a Name (line 482):

# Obtaining an instance of the builtin type 'dict' (line 482)
dict_4492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 482)
# Adding element type (key, value) (line 482)
str_4493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 13), 'str', 'Add')
str_4494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 20), 'str', '+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_4492, (str_4493, str_4494))
# Adding element type (key, value) (line 482)
str_4495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 25), 'str', 'Sub')
str_4496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 32), 'str', '-')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_4492, (str_4495, str_4496))
# Adding element type (key, value) (line 482)
str_4497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 37), 'str', 'Mult')
str_4498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 45), 'str', '*')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_4492, (str_4497, str_4498))
# Adding element type (key, value) (line 482)
str_4499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 50), 'str', 'Div')
str_4500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 57), 'str', '/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_4492, (str_4499, str_4500))
# Adding element type (key, value) (line 482)
str_4501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 62), 'str', 'Mod')
str_4502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 69), 'str', '%')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_4492, (str_4501, str_4502))
# Adding element type (key, value) (line 482)
str_4503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 13), 'str', 'LShift')
str_4504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 23), 'str', '<<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_4492, (str_4503, str_4504))
# Adding element type (key, value) (line 482)
str_4505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 29), 'str', 'RShift')
str_4506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 39), 'str', '>>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_4492, (str_4505, str_4506))
# Adding element type (key, value) (line 482)
str_4507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 45), 'str', 'BitOr')
str_4508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 54), 'str', '|')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_4492, (str_4507, str_4508))
# Adding element type (key, value) (line 482)
str_4509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 59), 'str', 'BitXor')
str_4510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 69), 'str', '^')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_4492, (str_4509, str_4510))
# Adding element type (key, value) (line 482)
str_4511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 74), 'str', 'BitAnd')
str_4512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 84), 'str', '&')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_4492, (str_4511, str_4512))
# Adding element type (key, value) (line 482)
str_4513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 13), 'str', 'FloorDiv')
str_4514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 25), 'str', '//')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_4492, (str_4513, str_4514))
# Adding element type (key, value) (line 482)
str_4515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 31), 'str', 'Pow')
str_4516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 38), 'str', '**')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_4492, (str_4515, str_4516))

# Getting the type of 'PythonSrcGeneratorVisitor'
PythonSrcGeneratorVisitor_4517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PythonSrcGeneratorVisitor')
# Setting the type of the member 'binop' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PythonSrcGeneratorVisitor_4517, 'binop', dict_4492)

# Assigning a Dict to a Name (line 493):

# Obtaining an instance of the builtin type 'dict' (line 493)
dict_4518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 493)
# Adding element type (key, value) (line 493)
str_4519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 14), 'str', 'Eq')
str_4520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 20), 'str', '==')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_4518, (str_4519, str_4520))
# Adding element type (key, value) (line 493)
str_4521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 26), 'str', 'NotEq')
str_4522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 35), 'str', '!=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_4518, (str_4521, str_4522))
# Adding element type (key, value) (line 493)
str_4523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 41), 'str', 'Lt')
str_4524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 47), 'str', '<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_4518, (str_4523, str_4524))
# Adding element type (key, value) (line 493)
str_4525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 52), 'str', 'LtE')
str_4526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 59), 'str', '<=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_4518, (str_4525, str_4526))
# Adding element type (key, value) (line 493)
str_4527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 65), 'str', 'Gt')
str_4528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 71), 'str', '>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_4518, (str_4527, str_4528))
# Adding element type (key, value) (line 493)
str_4529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 76), 'str', 'GtE')
str_4530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 83), 'str', '>=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_4518, (str_4529, str_4530))
# Adding element type (key, value) (line 493)
str_4531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 14), 'str', 'Is')
str_4532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 20), 'str', 'is')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_4518, (str_4531, str_4532))
# Adding element type (key, value) (line 493)
str_4533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 26), 'str', 'IsNot')
str_4534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 35), 'str', 'is not')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_4518, (str_4533, str_4534))
# Adding element type (key, value) (line 493)
str_4535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 45), 'str', 'In')
str_4536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 51), 'str', 'in')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_4518, (str_4535, str_4536))
# Adding element type (key, value) (line 493)
str_4537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 57), 'str', 'NotIn')
str_4538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 66), 'str', 'not in')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_4518, (str_4537, str_4538))

# Getting the type of 'PythonSrcGeneratorVisitor'
PythonSrcGeneratorVisitor_4539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PythonSrcGeneratorVisitor')
# Setting the type of the member 'cmpops' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PythonSrcGeneratorVisitor_4539, 'cmpops', dict_4518)

# Assigning a Dict to a Name (line 504):

# Obtaining an instance of the builtin type 'dict' (line 504)
dict_4540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 504)
# Adding element type (key, value) (line 504)
# Getting the type of 'ast' (line 504)
ast_4541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 15), 'ast')
# Obtaining the member 'And' of a type (line 504)
And_4542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 15), ast_4541, 'And')
str_4543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 24), 'str', 'and')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 14), dict_4540, (And_4542, str_4543))
# Adding element type (key, value) (line 504)
# Getting the type of 'ast' (line 504)
ast_4544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 31), 'ast')
# Obtaining the member 'Or' of a type (line 504)
Or_4545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 31), ast_4544, 'Or')
str_4546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 39), 'str', 'or')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 14), dict_4540, (Or_4545, str_4546))

# Getting the type of 'PythonSrcGeneratorVisitor'
PythonSrcGeneratorVisitor_4547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PythonSrcGeneratorVisitor')
# Setting the type of the member 'boolops' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PythonSrcGeneratorVisitor_4547, 'boolops', dict_4540)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
