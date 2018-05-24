
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
39:     def interleave(self, inter, f, seq):
40:         '''
41:         Call f on each item in seq, calling inter() in between.
42:         '''
43:         seq = iter(seq)
44:         try:
45:             f(next(seq))
46:         except StopIteration:
47:             pass
48:         else:
49:             for x in seq:
50:                 inter()
51:                 f(x)
52: 
53:     def __init__(self, tree, verbose=False):
54:         self.output = StringIO()
55:         self.future_imports = []
56:         self._indent = 0
57:         self._indent_str = "    "
58:         self.output.write("")
59:         self.output.flush()
60:         self.tree = tree
61:         self.verbose = verbose
62: 
63:     def generate_code(self):
64:         self.visit(self.tree)
65:         return self.output.getvalue()
66: 
67:     def fill(self, text=""):
68:         '''
69:         Indent a piece of text, according to the current indentation level
70:         '''
71:         if self.verbose:
72:             sys.stdout.write("\n" + self._indent_str * self._indent + text)
73:         self.output.write("\n" + self._indent_str * self._indent + text)
74: 
75:     def write(self, text):
76:         '''
77:         Append a piece of text to the current line.
78:         '''
79:         if self.verbose:
80:             sys.stdout.write(text)
81:         self.output.write(text)
82: 
83:     def enter(self):
84:         '''
85:         Print ':', and increase the indentation.
86:         '''
87:         self.write(":")
88:         self._indent += 1
89: 
90:     def leave(self):
91:         '''
92:         Decrease the indentation level.
93:         '''
94:         self._indent -= 1
95: 
96:     def visit(self, tree):
97:         '''
98:         General visit method, calling the appropriate visit method for type T
99:         '''
100:         if isinstance(tree, list):
101:             for t in tree:
102:                 self.visit(t)
103:             return
104: 
105:         if type(tree) is tuple:
106:             print (tree)
107: 
108:         meth = getattr(self, "visit_" + tree.__class__.__name__)
109:         meth(tree)
110: 
111:     # ############## Unparsing methods ######################
112:     # There should be one method per concrete grammar type #
113:     # Constructors should be grouped by sum type. Ideally, #
114:     # this would follow the order in the grammar, but      #
115:     # currently doesn't.                                   #
116:     # #######################################################
117: 
118:     def visit_Module(self, tree):
119:         for stmt in tree.body:
120:             self.visit(stmt)
121: 
122:     # stmt
123:     def visit_Expr(self, tree):
124:         self.fill()
125:         self.visit(tree.value)
126: 
127:     def visit_Import(self, t):
128:         self.fill("import ")
129:         interleave(lambda: self.write(", "), self.visit, t.names)
130:         self.write("\n")
131: 
132:     def visit_ImportFrom(self, t):
133:         # A from __future__ import may affect unparsing, so record it.
134:         if t.module and t.module == '__future__':
135:             self.future_imports.extend(n.name for n in t.names)
136: 
137:         self.fill("from ")
138:         self.write("." * t.level)
139:         if t.module:
140:             self.write(t.module)
141:         self.write(" import ")
142:         interleave(lambda: self.write(", "), self.visit, t.names)
143:         self.write("\n")
144: 
145:     def visit_Assign(self, t):
146:         self.fill()
147:         for target in t.targets:
148:             self.visit(target)
149:             self.write(" = ")
150:         self.visit(t.value)
151: 
152:     def visit_AugAssign(self, t):
153:         self.fill()
154:         self.visit(t.target)
155:         self.write(" " + self.binop[t.op.__class__.__name__] + "= ")
156:         self.visit(t.value)
157: 
158:     def visit_Return(self, t):
159:         self.fill("return")
160:         if t.value:
161:             self.write(" ")
162:             self.visit(t.value)
163:             # self.write("\n")
164: 
165:     def visit_Pass(self, t):
166:         self.fill("pass")
167: 
168:     def visit_Break(self, t):
169:         self.fill("break")
170: 
171:     def visit_Continue(self, t):
172:         self.fill("continue")
173: 
174:     def visit_Delete(self, t):
175:         self.fill("del ")
176:         interleave(lambda: self.write(", "), self.visit, t.targets)
177: 
178:     def visit_Assert(self, t):
179:         self.fill("assert ")
180:         self.visit(t.test)
181:         if t.msg:
182:             self.write(", ")
183:             self.visit(t.msg)
184: 
185:     def visit_Exec(self, t):
186:         self.fill("exec ")
187:         self.visit(t.body)
188:         if t.globals:
189:             self.write(" in ")
190:             self.visit(t.globals)
191:         if t.locals:
192:             self.write(", ")
193:             self.visit(t.locals)
194: 
195:     def visit_Print(self, t):
196:         self.fill("print ")
197:         do_comma = False
198:         if t.dest:
199:             self.write(">>")
200:             self.visit(t.dest)
201:             do_comma = True
202:         for e in t.values:
203:             if do_comma:
204:                 self.write(", ")
205:             else:
206:                 do_comma = True
207:             self.visit(e)
208:         if not t.nl:
209:             self.write(",")
210: 
211:     def visit_Global(self, t):
212:         self.fill("global ")
213:         interleave(lambda: self.write(", "), self.write, t.names)
214: 
215:     def visit_Yield(self, t):
216:         self.write("(")
217:         self.write("yield")
218:         if t.value:
219:             self.write(" ")
220:             self.visit(t.value)
221:         self.write(")")
222: 
223:     def visit_Raise(self, t):
224:         self.fill('raise ')
225:         if t.type:
226:             self.visit(t.type)
227:         if t.inst:
228:             self.write(", ")
229:             self.visit(t.inst)
230:         if t.tback:
231:             self.write(", ")
232:             self.visit(t.tback)
233: 
234:     def visit_TryExcept(self, t):
235:         self.fill("try")
236:         self.enter()
237:         self.visit(t.body)
238:         self.leave()
239: 
240:         for ex in t.handlers:
241:             self.visit(ex)
242:         if t.orelse:
243:             self.fill("else")
244:             self.enter()
245:             self.visit(t.orelse)
246:             self.leave()
247: 
248:     def visit_TryFinally(self, t):
249:         if len(t.body) == 1 and isinstance(t.body[0], ast.TryExcept):
250:             # try-except-finally
251:             self.visit(t.body)
252:         else:
253:             self.fill("try")
254:             self.enter()
255:             self.visit(t.body)
256:             self.leave()
257: 
258:         self.fill("finally")
259:         self.enter()
260:         self.visit(t.finalbody)
261:         self.leave()
262: 
263:     def visit_ExceptHandler(self, t):
264:         self.fill("except")
265:         if t.type:
266:             self.write(" ")
267:             self.visit(t.type)
268:         if t.name:
269:             self.write(" as ")
270:             self.visit(t.name)
271:         self.enter()
272:         self.visit(t.body)
273:         self.leave()
274: 
275:     def visit_ClassDef(self, t):
276:         self.write("\n")
277:         for deco in t.decorator_list:
278:             self.fill("@")
279:             self.visit(deco)
280:         self.fill("class " + t.name)
281:         if t.bases:
282:             self.write("(")
283:             for a in t.bases:
284:                 self.visit(a)
285:                 self.write(", ")
286:             self.write(")")
287:         self.enter()
288:         self.visit(t.body)
289:         self.leave()
290: 
291:     def visit_FunctionDef(self, t):
292:         self.write("\n")
293:         for deco in t.decorator_list:
294:             self.fill("@")
295:             self.visit(deco)
296:         self.fill("def " + t.name + "(")
297:         self.visit(t.args)
298:         self.write(")")
299:         self.enter()
300:         self.visit(t.body)
301:         self.write("\n")
302:         self.leave()
303: 
304:     def visit_For(self, t):
305:         self.fill("for ")
306:         self.visit(t.target)
307:         self.write(" in ")
308:         self.visit(t.iter)
309:         self.enter()
310:         self.visit(t.body)
311:         self.leave()
312:         if t.orelse:
313:             self.fill("else")
314:             self.enter()
315:             self.visit(t.orelse)
316:             self.leave()
317: 
318:     def visit_If(self, t):
319:         self.write("\n")
320:         self.fill("if ")
321:         self.visit(t.test)
322:         self.enter()
323:         self.visit(t.body)
324:         self.leave()
325:         # collapse nested ifs into equivalent elifs.
326:         while (t.orelse and len(t.orelse) == 1 and
327:                    isinstance(t.orelse[0], ast.If)):
328:             t = t.orelse[0]
329:             self.fill("elif ")
330:             self.visit(t.test)
331:             self.enter()
332:             self.visit(t.body)
333:             self.leave()
334:         # final else
335:         if t.orelse:
336:             self.fill("else")
337:             self.enter()
338:             self.visit(t.orelse)
339:             self.leave()
340:         self.write("\n")
341: 
342:     def visit_While(self, t):
343:         self.fill("while ")
344:         self.visit(t.test)
345:         self.enter()
346:         self.visit(t.body)
347:         self.leave()
348:         if t.orelse:
349:             self.fill("else")
350:             self.enter()
351:             self.visit(t.orelse)
352:             self.leave()
353: 
354:     def visit_With(self, t):
355:         self.fill("with ")
356:         self.visit(t.context_expr)
357:         if t.optional_vars:
358:             self.write(" as ")
359:             self.visit(t.optional_vars)
360:         self.enter()
361:         self.visit(t.body)
362:         self.leave()
363: 
364:     # expr
365:     def visit_Str(self, tree):
366:         # if from __future__ import unicode_literals is in effect,
367:         # then we want to output string literals using a 'b' prefix
368:         # and unicode literals with no prefix.
369:         if "unicode_literals" not in self.future_imports:
370:             self.write(repr(tree.s))
371:         elif isinstance(tree.s, str):
372:             self.write("b" + repr(tree.s))
373:         elif isinstance(tree.s, unicode):
374:             self.write(repr(tree.s).lstrip("u"))
375:         else:
376:             assert False, "shouldn't get here"
377: 
378:     def visit_Name(self, t):
379:         self.write(t.id)
380: 
381:     def visit_Repr(self, t):
382:         self.write("`")
383:         self.visit(t.value)
384:         self.write("`")
385: 
386:     def visit_Num(self, t):
387:         repr_n = repr(t.n)
388:         # Parenthesize negative numbers, to avoid turning (-1)**2 into -1**2.
389:         if repr_n.startswith("-"):
390:             self.write("(")
391:         # Substitute overflowing decimal literal for AST infinities.
392:         self.write(repr_n.replace("inf", INFSTR))
393:         if repr_n.startswith("-"):
394:             self.write(")")
395: 
396:     def visit_List(self, t):
397:         self.write("[")
398:         interleave(lambda: self.write(", "), self.visit, t.elts)
399:         self.write("]")
400: 
401:     def visit_ListComp(self, t):
402:         self.write("[")
403:         self.visit(t.elt)
404:         for gen in t.generators:
405:             self.visit(gen)
406:         self.write("]")
407: 
408:     def visit_GeneratorExp(self, t):
409:         self.write("(")
410:         self.visit(t.elt)
411:         for gen in t.generators:
412:             self.visit(gen)
413:         self.write(")")
414: 
415:     def visit_SetComp(self, t):
416:         self.write("{")
417:         self.visit(t.elt)
418:         for gen in t.generators:
419:             self.visit(gen)
420:         self.write("}")
421: 
422:     def visit_DictComp(self, t):
423:         self.write("{")
424:         self.visit(t.key)
425:         self.write(": ")
426:         self.visit(t.value)
427:         for gen in t.generators:
428:             self.visit(gen)
429:         self.write("}")
430: 
431:     def visit_comprehension(self, t):
432:         self.write(" for ")
433:         self.visit(t.target)
434:         self.write(" in ")
435:         self.visit(t.iter)
436:         for if_clause in t.ifs:
437:             self.write(" if ")
438:             self.visit(if_clause)
439: 
440:     def visit_IfExp(self, t):
441:         self.write("(")
442:         self.visit(t.body)
443:         self.write(" if ")
444:         self.visit(t.test)
445:         self.write(" else ")
446:         self.visit(t.orelse)
447:         self.write(")")
448: 
449:     def visit_Set(self, t):
450:         assert (t.elts)  # should be at least one element
451:         self.write("{")
452:         interleave(lambda: self.write(", "), self.visit, t.elts)
453:         self.write("}")
454: 
455:     def visit_Dict(self, t):
456:         self.write("{")
457: 
458:         def write_pair(pair):
459:             (k, v) = pair
460:             self.visit(k)
461:             self.write(": ")
462:             self.visit(v)
463: 
464:         self.interleave(lambda: self.write(", "), write_pair, zip(t.keys, t.values))
465:         self.write("}")
466: 
467:     def visit_Tuple(self, t):
468:         self.write("(")
469:         if len(t.elts) == 1:
470:             (elt,) = t.elts
471:             self.visit(elt)
472:             self.write(",")
473:         else:
474:             interleave(lambda: self.write(", "), self.visit, t.elts)
475:         self.write(")")
476: 
477:     unop = {"Invert": "~", "Not": "not", "UAdd": "+", "USub": "-"}
478: 
479:     def visit_UnaryOp(self, t):
480:         self.write("(")
481:         self.write(self.unop[t.op.__class__.__name__])
482:         self.write(" ")
483:         # If we're applying unary minus to a number, parenthesize the number.
484:         # This is necessary: -2147483648 is different from -(2147483648) on
485:         # a 32-bit machine (the first is an int, the second a long), and
486:         # -7j is different from -(7j).  (The first has real part 0.0, the second
487:         # has real part -0.0.)
488:         if isinstance(t.op, ast.USub) and isinstance(t.operand, ast.Num):
489:             self.write("(")
490:             self.visit(t.operand)
491:             self.write(")")
492:         else:
493:             self.visit(t.operand)
494:         self.write(")")
495: 
496:     binop = {"Add": "+", "Sub": "-", "Mult": "*", "Div": "/", "Mod": "%",
497:              "LShift": "<<", "RShift": ">>", "BitOr": "|", "BitXor": "^", "BitAnd": "&",
498:              "FloorDiv": "//", "Pow": "**"}
499: 
500:     def visit_BinOp(self, t):
501:         self.write("(")
502:         self.visit(t.left)
503:         self.write(" " + self.binop[t.op.__class__.__name__] + " ")
504:         self.visit(t.right)
505:         self.write(")")
506: 
507:     cmpops = {"Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=", "Gt": ">", "GtE": ">=",
508:               "Is": "is", "IsNot": "is not", "In": "in", "NotIn": "not in"}
509: 
510:     def visit_Compare(self, t):
511:         self.write("(")
512:         self.visit(t.left)
513:         for o, e in zip(t.ops, t.comparators):
514:             self.write(" " + self.cmpops[o.__class__.__name__] + " ")
515:             self.visit(e)
516:         self.write(")")
517: 
518:     boolops = {ast.And: 'and', ast.Or: 'or'}
519: 
520:     def visit_BoolOp(self, t):
521:         self.write("(")
522:         s = " %s " % self.boolops[t.op.__class__]
523:         interleave(lambda: self.write(s), self.visit, t.values)
524:         self.write(")")
525: 
526:     def visit_Attribute(self, t):
527:         self.visit(t.value)
528:         # Special case: 3.__abs__() is a syntax error, so if t.value
529:         # is an integer literal then we need to either parenthesize
530:         # it or add an extra space to get 3 .__abs__().
531:         if isinstance(t.value, ast.Num) and isinstance(t.value.n, int):
532:             self.write(" ")
533:         self.write(".")
534:         self.write(t.attr)
535: 
536:     def visit_Call(self, t):
537:         self.visit(t.func)
538:         self.write("(")
539:         comma = False
540:         for e in t.args:
541:             if comma:
542:                 self.write(", ")
543:             else:
544:                 comma = True
545:             self.visit(e)
546:         for e in t.keywords:
547:             if comma:
548:                 self.write(", ")
549:             else:
550:                 comma = True
551:             self.visit(e)
552:         if t.starargs:
553:             if comma:
554:                 self.write(", ")
555:             else:
556:                 comma = True
557:             self.write("*")
558:             self.visit(t.starargs)
559:         if t.kwargs:
560:             if comma:
561:                 self.write(", ")
562:             else:
563:                 comma = True
564:             self.write("**")
565:             self.visit(t.kwargs)
566:         self.write(")")
567: 
568:     def visit_Subscript(self, t):
569:         self.visit(t.value)
570:         self.write("[")
571:         self.visit(t.slice)
572:         self.write("]")
573: 
574:     # slice
575:     def visit_Ellipsis(self, t):
576:         self.write("...")
577: 
578:     def visit_Index(self, t):
579:         self.visit(t.value)
580: 
581:     def visit_Slice(self, t):
582:         if t.lower:
583:             self.visit(t.lower)
584:         self.write(":")
585:         if t.upper:
586:             self.visit(t.upper)
587:         if t.step:
588:             self.write(":")
589:             self.visit(t.step)
590: 
591:     def visit_ExtSlice(self, t):
592:         interleave(lambda: self.write(', '), self.visit, t.dims)
593: 
594:     # others
595:     def visit_arguments(self, t):
596:         first = True
597:         # normal arguments
598:         defaults = [None] * (len(t.args) - len(t.defaults)) + t.defaults
599:         for a, d in zip(t.args, defaults):
600:             if first:
601:                 first = False
602:             else:
603:                 self.write(", ")
604:             self.visit(a),
605:             if d:
606:                 self.write("=")
607:                 self.visit(d)
608: 
609:         # varargs
610:         if t.vararg:
611:             if first:
612:                 first = False
613:             else:
614:                 self.write(", ")
615:             self.write("*")
616:             self.write(t.vararg)
617: 
618:         # kwargs
619:         if t.kwarg:
620:             if first:
621:                 first = False
622:             else:
623:                 self.write(", ")
624:             self.write("**" + t.kwarg)
625: 
626:     def visit_keyword(self, t):
627:         self.write(t.arg)
628:         self.write("=")
629:         self.visit(t.value)
630: 
631:     def visit_Lambda(self, t):
632:         self.write("(")
633:         self.write("lambda ")
634:         self.visit(t.args)
635:         self.write(": ")
636:         self.visit(t.body)
637:         self.write(")")
638: 
639:     def visit_alias(self, t):
640:         self.write(t.name)
641:         if t.asname:
642:             self.write(" as " + t.asname)
643: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', '\nThis is a Python source code generator visitor, that transform an AST into valid source code. It is used to create\ntype annotated programs and type inference programs when their AST is finally created.\n\nAdapted from: http://svn.python.org/view/python/trunk/Demo/parser/unparse.py\n')
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
str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 9), 'str', '1e')

# Call to repr(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'sys' (line 14)
sys_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 21), 'sys', False)
# Obtaining the member 'float_info' of a type (line 14)
float_info_8 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 21), sys_7, 'float_info')
# Obtaining the member 'max_10_exp' of a type (line 14)
max_10_exp_9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 21), float_info_8, 'max_10_exp')
int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 49), 'int')
# Applying the binary operator '+' (line 14)
result_add_11 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 21), '+', max_10_exp_9, int_10)

# Processing the call keyword arguments (line 14)
kwargs_12 = {}
# Getting the type of 'repr' (line 14)
repr_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'repr', False)
# Calling repr(args, kwargs) (line 14)
repr_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 14, 16), repr_6, *[result_add_11], **kwargs_12)

# Applying the binary operator '+' (line 14)
result_add_14 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '+', str_5, repr_call_result_13)

# Assigning a type to the variable 'INFSTR' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'INFSTR', result_add_14)

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

    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'str', '\n    Call f on each item in seq, calling inter() in between.\n    ')
    
    # Assigning a Call to a Name (line 21):
    
    # Assigning a Call to a Name (line 21):
    
    # Call to iter(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'seq' (line 21)
    seq_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'seq', False)
    # Processing the call keyword arguments (line 21)
    kwargs_18 = {}
    # Getting the type of 'iter' (line 21)
    iter_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'iter', False)
    # Calling iter(args, kwargs) (line 21)
    iter_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), iter_16, *[seq_17], **kwargs_18)
    
    # Assigning a type to the variable 'seq' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'seq', iter_call_result_19)
    
    
    # SSA begins for try-except statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to f(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to next(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'seq' (line 23)
    seq_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'seq', False)
    # Processing the call keyword arguments (line 23)
    kwargs_23 = {}
    # Getting the type of 'next' (line 23)
    next_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'next', False)
    # Calling next(args, kwargs) (line 23)
    next_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), next_21, *[seq_22], **kwargs_23)
    
    # Processing the call keyword arguments (line 23)
    kwargs_25 = {}
    # Getting the type of 'f' (line 23)
    f_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'f', False)
    # Calling f(args, kwargs) (line 23)
    f_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), f_20, *[next_call_result_24], **kwargs_25)
    
    # SSA branch for the except part of a try statement (line 22)
    # SSA branch for the except 'StopIteration' branch of a try statement (line 22)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA branch for the else branch of a try statement (line 22)
    module_type_store.open_ssa_branch('except else')
    
    # Getting the type of 'seq' (line 27)
    seq_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'seq')
    # Assigning a type to the variable 'seq_27' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'seq_27', seq_27)
    # Testing if the for loop is going to be iterated (line 27)
    # Testing the type of a for loop iterable (line 27)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 8), seq_27)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 27, 8), seq_27):
        # Getting the type of the for loop variable (line 27)
        for_loop_var_28 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 8), seq_27)
        # Assigning a type to the variable 'x' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'x', for_loop_var_28)
        # SSA begins for a for statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to inter(...): (line 28)
        # Processing the call keyword arguments (line 28)
        kwargs_30 = {}
        # Getting the type of 'inter' (line 28)
        inter_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'inter', False)
        # Calling inter(args, kwargs) (line 28)
        inter_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), inter_29, *[], **kwargs_30)
        
        
        # Call to f(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'x' (line 29)
        x_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'x', False)
        # Processing the call keyword arguments (line 29)
        kwargs_34 = {}
        # Getting the type of 'f' (line 29)
        f_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'f', False)
        # Calling f(args, kwargs) (line 29)
        f_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), f_32, *[x_33], **kwargs_34)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # SSA join for try-except statement (line 22)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'interleave(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'interleave' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'interleave'
    return stypy_return_type_36

# Assigning a type to the variable 'interleave' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'interleave', interleave)
# Declaration of the 'PythonSrcGeneratorVisitor' class
# Getting the type of 'ast' (line 32)
ast_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 32), 'ast')
# Obtaining the member 'NodeVisitor' of a type (line 32)
NodeVisitor_38 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 32), ast_37, 'NodeVisitor')

class PythonSrcGeneratorVisitor(NodeVisitor_38, ):
    str_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, (-1)), 'str', '\n    Methods in this class recursively traverse an AST and\n    output source code for the abstract syntax; original formatting\n    is disregarded.\n    ')

    @norecursion
    def interleave(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'interleave'
        module_type_store = module_type_store.open_function_context('interleave', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PythonSrcGeneratorVisitor.interleave.__dict__.__setitem__('stypy_localization', localization)
        PythonSrcGeneratorVisitor.interleave.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PythonSrcGeneratorVisitor.interleave.__dict__.__setitem__('stypy_type_store', module_type_store)
        PythonSrcGeneratorVisitor.interleave.__dict__.__setitem__('stypy_function_name', 'PythonSrcGeneratorVisitor.interleave')
        PythonSrcGeneratorVisitor.interleave.__dict__.__setitem__('stypy_param_names_list', ['inter', 'f', 'seq'])
        PythonSrcGeneratorVisitor.interleave.__dict__.__setitem__('stypy_varargs_param_name', None)
        PythonSrcGeneratorVisitor.interleave.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PythonSrcGeneratorVisitor.interleave.__dict__.__setitem__('stypy_call_defaults', defaults)
        PythonSrcGeneratorVisitor.interleave.__dict__.__setitem__('stypy_call_varargs', varargs)
        PythonSrcGeneratorVisitor.interleave.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PythonSrcGeneratorVisitor.interleave.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PythonSrcGeneratorVisitor.interleave', ['inter', 'f', 'seq'], None, None, defaults, varargs, kwargs)

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

        str_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, (-1)), 'str', '\n        Call f on each item in seq, calling inter() in between.\n        ')
        
        # Assigning a Call to a Name (line 43):
        
        # Assigning a Call to a Name (line 43):
        
        # Call to iter(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'seq' (line 43)
        seq_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'seq', False)
        # Processing the call keyword arguments (line 43)
        kwargs_43 = {}
        # Getting the type of 'iter' (line 43)
        iter_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'iter', False)
        # Calling iter(args, kwargs) (line 43)
        iter_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 43, 14), iter_41, *[seq_42], **kwargs_43)
        
        # Assigning a type to the variable 'seq' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'seq', iter_call_result_44)
        
        
        # SSA begins for try-except statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to f(...): (line 45)
        # Processing the call arguments (line 45)
        
        # Call to next(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'seq' (line 45)
        seq_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'seq', False)
        # Processing the call keyword arguments (line 45)
        kwargs_48 = {}
        # Getting the type of 'next' (line 45)
        next_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 14), 'next', False)
        # Calling next(args, kwargs) (line 45)
        next_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 45, 14), next_46, *[seq_47], **kwargs_48)
        
        # Processing the call keyword arguments (line 45)
        kwargs_50 = {}
        # Getting the type of 'f' (line 45)
        f_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'f', False)
        # Calling f(args, kwargs) (line 45)
        f_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), f_45, *[next_call_result_49], **kwargs_50)
        
        # SSA branch for the except part of a try statement (line 44)
        # SSA branch for the except 'StopIteration' branch of a try statement (line 44)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA branch for the else branch of a try statement (line 44)
        module_type_store.open_ssa_branch('except else')
        
        # Getting the type of 'seq' (line 49)
        seq_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 21), 'seq')
        # Assigning a type to the variable 'seq_52' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'seq_52', seq_52)
        # Testing if the for loop is going to be iterated (line 49)
        # Testing the type of a for loop iterable (line 49)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 12), seq_52)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 49, 12), seq_52):
            # Getting the type of the for loop variable (line 49)
            for_loop_var_53 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 12), seq_52)
            # Assigning a type to the variable 'x' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'x', for_loop_var_53)
            # SSA begins for a for statement (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to inter(...): (line 50)
            # Processing the call keyword arguments (line 50)
            kwargs_55 = {}
            # Getting the type of 'inter' (line 50)
            inter_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'inter', False)
            # Calling inter(args, kwargs) (line 50)
            inter_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 50, 16), inter_54, *[], **kwargs_55)
            
            
            # Call to f(...): (line 51)
            # Processing the call arguments (line 51)
            # Getting the type of 'x' (line 51)
            x_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'x', False)
            # Processing the call keyword arguments (line 51)
            kwargs_59 = {}
            # Getting the type of 'f' (line 51)
            f_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'f', False)
            # Calling f(args, kwargs) (line 51)
            f_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), f_57, *[x_58], **kwargs_59)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for try-except statement (line 44)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'interleave(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'interleave' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'interleave'
        return stypy_return_type_61


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 53)
        False_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 37), 'False')
        defaults = [False_62]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
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

        
        # Assigning a Call to a Attribute (line 54):
        
        # Assigning a Call to a Attribute (line 54):
        
        # Call to StringIO(...): (line 54)
        # Processing the call keyword arguments (line 54)
        kwargs_64 = {}
        # Getting the type of 'StringIO' (line 54)
        StringIO_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 54)
        StringIO_call_result_65 = invoke(stypy.reporting.localization.Localization(__file__, 54, 22), StringIO_63, *[], **kwargs_64)
        
        # Getting the type of 'self' (line 54)
        self_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self')
        # Setting the type of the member 'output' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_66, 'output', StringIO_call_result_65)
        
        # Assigning a List to a Attribute (line 55):
        
        # Assigning a List to a Attribute (line 55):
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_67 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        
        # Getting the type of 'self' (line 55)
        self_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member 'future_imports' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_68, 'future_imports', list_67)
        
        # Assigning a Num to a Attribute (line 56):
        
        # Assigning a Num to a Attribute (line 56):
        int_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'int')
        # Getting the type of 'self' (line 56)
        self_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member '_indent' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_70, '_indent', int_69)
        
        # Assigning a Str to a Attribute (line 57):
        
        # Assigning a Str to a Attribute (line 57):
        str_71 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 27), 'str', '    ')
        # Getting the type of 'self' (line 57)
        self_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member '_indent_str' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_72, '_indent_str', str_71)
        
        # Call to write(...): (line 58)
        # Processing the call arguments (line 58)
        str_76 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 26), 'str', '')
        # Processing the call keyword arguments (line 58)
        kwargs_77 = {}
        # Getting the type of 'self' (line 58)
        self_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 58)
        output_74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_73, 'output')
        # Obtaining the member 'write' of a type (line 58)
        write_75 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), output_74, 'write')
        # Calling write(args, kwargs) (line 58)
        write_call_result_78 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), write_75, *[str_76], **kwargs_77)
        
        
        # Call to flush(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_82 = {}
        # Getting the type of 'self' (line 59)
        self_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 59)
        output_80 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_79, 'output')
        # Obtaining the member 'flush' of a type (line 59)
        flush_81 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), output_80, 'flush')
        # Calling flush(args, kwargs) (line 59)
        flush_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), flush_81, *[], **kwargs_82)
        
        
        # Assigning a Name to a Attribute (line 60):
        
        # Assigning a Name to a Attribute (line 60):
        # Getting the type of 'tree' (line 60)
        tree_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'tree')
        # Getting the type of 'self' (line 60)
        self_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'self')
        # Setting the type of the member 'tree' of a type (line 60)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), self_85, 'tree', tree_84)
        
        # Assigning a Name to a Attribute (line 61):
        
        # Assigning a Name to a Attribute (line 61):
        # Getting the type of 'verbose' (line 61)
        verbose_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'verbose')
        # Getting the type of 'self' (line 61)
        self_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member 'verbose' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_87, 'verbose', verbose_86)
        
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
        module_type_store = module_type_store.open_function_context('generate_code', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
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

        
        # Call to visit(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'self' (line 64)
        self_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'self', False)
        # Obtaining the member 'tree' of a type (line 64)
        tree_91 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 19), self_90, 'tree')
        # Processing the call keyword arguments (line 64)
        kwargs_92 = {}
        # Getting the type of 'self' (line 64)
        self_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 64)
        visit_89 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_88, 'visit')
        # Calling visit(args, kwargs) (line 64)
        visit_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 64, 8), visit_89, *[tree_91], **kwargs_92)
        
        
        # Call to getvalue(...): (line 65)
        # Processing the call keyword arguments (line 65)
        kwargs_97 = {}
        # Getting the type of 'self' (line 65)
        self_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'self', False)
        # Obtaining the member 'output' of a type (line 65)
        output_95 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), self_94, 'output')
        # Obtaining the member 'getvalue' of a type (line 65)
        getvalue_96 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), output_95, 'getvalue')
        # Calling getvalue(args, kwargs) (line 65)
        getvalue_call_result_98 = invoke(stypy.reporting.localization.Localization(__file__, 65, 15), getvalue_96, *[], **kwargs_97)
        
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', getvalue_call_result_98)
        
        # ################# End of 'generate_code(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_code' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_99)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_code'
        return stypy_return_type_99


    @norecursion
    def fill(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 24), 'str', '')
        defaults = [str_100]
        # Create a new context for function 'fill'
        module_type_store = module_type_store.open_function_context('fill', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
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

        str_101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, (-1)), 'str', '\n        Indent a piece of text, according to the current indentation level\n        ')
        # Getting the type of 'self' (line 71)
        self_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'self')
        # Obtaining the member 'verbose' of a type (line 71)
        verbose_103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 11), self_102, 'verbose')
        # Testing if the type of an if condition is none (line 71)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 71, 8), verbose_103):
            pass
        else:
            
            # Testing the type of an if condition (line 71)
            if_condition_104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 8), verbose_103)
            # Assigning a type to the variable 'if_condition_104' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'if_condition_104', if_condition_104)
            # SSA begins for if statement (line 71)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 72)
            # Processing the call arguments (line 72)
            str_108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 29), 'str', '\n')
            # Getting the type of 'self' (line 72)
            self_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 36), 'self', False)
            # Obtaining the member '_indent_str' of a type (line 72)
            _indent_str_110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 36), self_109, '_indent_str')
            # Getting the type of 'self' (line 72)
            self_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 55), 'self', False)
            # Obtaining the member '_indent' of a type (line 72)
            _indent_112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 55), self_111, '_indent')
            # Applying the binary operator '*' (line 72)
            result_mul_113 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 36), '*', _indent_str_110, _indent_112)
            
            # Applying the binary operator '+' (line 72)
            result_add_114 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 29), '+', str_108, result_mul_113)
            
            # Getting the type of 'text' (line 72)
            text_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 70), 'text', False)
            # Applying the binary operator '+' (line 72)
            result_add_116 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 68), '+', result_add_114, text_115)
            
            # Processing the call keyword arguments (line 72)
            kwargs_117 = {}
            # Getting the type of 'sys' (line 72)
            sys_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'sys', False)
            # Obtaining the member 'stdout' of a type (line 72)
            stdout_106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), sys_105, 'stdout')
            # Obtaining the member 'write' of a type (line 72)
            write_107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), stdout_106, 'write')
            # Calling write(args, kwargs) (line 72)
            write_call_result_118 = invoke(stypy.reporting.localization.Localization(__file__, 72, 12), write_107, *[result_add_116], **kwargs_117)
            
            # SSA join for if statement (line 71)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 73)
        # Processing the call arguments (line 73)
        str_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 26), 'str', '\n')
        # Getting the type of 'self' (line 73)
        self_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 33), 'self', False)
        # Obtaining the member '_indent_str' of a type (line 73)
        _indent_str_124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 33), self_123, '_indent_str')
        # Getting the type of 'self' (line 73)
        self_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 52), 'self', False)
        # Obtaining the member '_indent' of a type (line 73)
        _indent_126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 52), self_125, '_indent')
        # Applying the binary operator '*' (line 73)
        result_mul_127 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 33), '*', _indent_str_124, _indent_126)
        
        # Applying the binary operator '+' (line 73)
        result_add_128 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 26), '+', str_122, result_mul_127)
        
        # Getting the type of 'text' (line 73)
        text_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 67), 'text', False)
        # Applying the binary operator '+' (line 73)
        result_add_130 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 65), '+', result_add_128, text_129)
        
        # Processing the call keyword arguments (line 73)
        kwargs_131 = {}
        # Getting the type of 'self' (line 73)
        self_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 73)
        output_120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_119, 'output')
        # Obtaining the member 'write' of a type (line 73)
        write_121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), output_120, 'write')
        # Calling write(args, kwargs) (line 73)
        write_call_result_132 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), write_121, *[result_add_130], **kwargs_131)
        
        
        # ################# End of 'fill(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fill' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_133)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fill'
        return stypy_return_type_133


    @norecursion
    def write(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'write'
        module_type_store = module_type_store.open_function_context('write', 75, 4, False)
        # Assigning a type to the variable 'self' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'self', type_of_self)
        
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

        str_134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', '\n        Append a piece of text to the current line.\n        ')
        # Getting the type of 'self' (line 79)
        self_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'self')
        # Obtaining the member 'verbose' of a type (line 79)
        verbose_136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), self_135, 'verbose')
        # Testing if the type of an if condition is none (line 79)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 79, 8), verbose_136):
            pass
        else:
            
            # Testing the type of an if condition (line 79)
            if_condition_137 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), verbose_136)
            # Assigning a type to the variable 'if_condition_137' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_137', if_condition_137)
            # SSA begins for if statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 80)
            # Processing the call arguments (line 80)
            # Getting the type of 'text' (line 80)
            text_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'text', False)
            # Processing the call keyword arguments (line 80)
            kwargs_142 = {}
            # Getting the type of 'sys' (line 80)
            sys_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'sys', False)
            # Obtaining the member 'stdout' of a type (line 80)
            stdout_139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), sys_138, 'stdout')
            # Obtaining the member 'write' of a type (line 80)
            write_140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 12), stdout_139, 'write')
            # Calling write(args, kwargs) (line 80)
            write_call_result_143 = invoke(stypy.reporting.localization.Localization(__file__, 80, 12), write_140, *[text_141], **kwargs_142)
            
            # SSA join for if statement (line 79)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'text' (line 81)
        text_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 26), 'text', False)
        # Processing the call keyword arguments (line 81)
        kwargs_148 = {}
        # Getting the type of 'self' (line 81)
        self_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 81)
        output_145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_144, 'output')
        # Obtaining the member 'write' of a type (line 81)
        write_146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), output_145, 'write')
        # Calling write(args, kwargs) (line 81)
        write_call_result_149 = invoke(stypy.reporting.localization.Localization(__file__, 81, 8), write_146, *[text_147], **kwargs_148)
        
        
        # ################# End of 'write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_150)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write'
        return stypy_return_type_150


    @norecursion
    def enter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'enter'
        module_type_store = module_type_store.open_function_context('enter', 83, 4, False)
        # Assigning a type to the variable 'self' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'self', type_of_self)
        
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

        str_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, (-1)), 'str', "\n        Print ':', and increase the indentation.\n        ")
        
        # Call to write(...): (line 87)
        # Processing the call arguments (line 87)
        str_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 19), 'str', ':')
        # Processing the call keyword arguments (line 87)
        kwargs_155 = {}
        # Getting the type of 'self' (line 87)
        self_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 87)
        write_153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_152, 'write')
        # Calling write(args, kwargs) (line 87)
        write_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), write_153, *[str_154], **kwargs_155)
        
        
        # Getting the type of 'self' (line 88)
        self_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self')
        # Obtaining the member '_indent' of a type (line 88)
        _indent_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_157, '_indent')
        int_159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 24), 'int')
        # Applying the binary operator '+=' (line 88)
        result_iadd_160 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 8), '+=', _indent_158, int_159)
        # Getting the type of 'self' (line 88)
        self_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self')
        # Setting the type of the member '_indent' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_161, '_indent', result_iadd_160)
        
        
        # ################# End of 'enter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'enter' in the type store
        # Getting the type of 'stypy_return_type' (line 83)
        stypy_return_type_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_162)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'enter'
        return stypy_return_type_162


    @norecursion
    def leave(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'leave'
        module_type_store = module_type_store.open_function_context('leave', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
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

        str_163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, (-1)), 'str', '\n        Decrease the indentation level.\n        ')
        
        # Getting the type of 'self' (line 94)
        self_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Obtaining the member '_indent' of a type (line 94)
        _indent_165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_164, '_indent')
        int_166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 24), 'int')
        # Applying the binary operator '-=' (line 94)
        result_isub_167 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 8), '-=', _indent_165, int_166)
        # Getting the type of 'self' (line 94)
        self_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member '_indent' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_168, '_indent', result_isub_167)
        
        
        # ################# End of 'leave(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'leave' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'leave'
        return stypy_return_type_169


    @norecursion
    def visit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit'
        module_type_store = module_type_store.open_function_context('visit', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
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

        str_170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, (-1)), 'str', '\n        General visit method, calling the appropriate visit method for type T\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 100)
        # Getting the type of 'list' (line 100)
        list_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 28), 'list')
        # Getting the type of 'tree' (line 100)
        tree_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'tree')
        
        (may_be_173, more_types_in_union_174) = may_be_subtype(list_171, tree_172)

        if may_be_173:

            if more_types_in_union_174:
                # Runtime conditional SSA (line 100)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'tree' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'tree', remove_not_subtype_from_union(tree_172, list))
            
            # Getting the type of 'tree' (line 101)
            tree_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'tree')
            # Assigning a type to the variable 'tree_175' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'tree_175', tree_175)
            # Testing if the for loop is going to be iterated (line 101)
            # Testing the type of a for loop iterable (line 101)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 101, 12), tree_175)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 101, 12), tree_175):
                # Getting the type of the for loop variable (line 101)
                for_loop_var_176 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 101, 12), tree_175)
                # Assigning a type to the variable 't' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 't', for_loop_var_176)
                # SSA begins for a for statement (line 101)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to visit(...): (line 102)
                # Processing the call arguments (line 102)
                # Getting the type of 't' (line 102)
                t_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 't', False)
                # Processing the call keyword arguments (line 102)
                kwargs_180 = {}
                # Getting the type of 'self' (line 102)
                self_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'self', False)
                # Obtaining the member 'visit' of a type (line 102)
                visit_178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 16), self_177, 'visit')
                # Calling visit(args, kwargs) (line 102)
                visit_call_result_181 = invoke(stypy.reporting.localization.Localization(__file__, 102, 16), visit_178, *[t_179], **kwargs_180)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Assigning a type to the variable 'stypy_return_type' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_174:
                # SSA join for if statement (line 100)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 105)
        # Getting the type of 'tree' (line 105)
        tree_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'tree')
        # Getting the type of 'tuple' (line 105)
        tuple_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'tuple')
        
        (may_be_184, more_types_in_union_185) = may_be_type(tree_182, tuple_183)

        if may_be_184:

            if more_types_in_union_185:
                # Runtime conditional SSA (line 105)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'tree' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'tree', tuple_183())
            # Getting the type of 'tree' (line 106)
            tree_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 19), 'tree')

            if more_types_in_union_185:
                # SSA join for if statement (line 105)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 108):
        
        # Assigning a Call to a Name (line 108):
        
        # Call to getattr(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'self' (line 108)
        self_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'self', False)
        str_189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 29), 'str', 'visit_')
        # Getting the type of 'tree' (line 108)
        tree_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 40), 'tree', False)
        # Obtaining the member '__class__' of a type (line 108)
        class___191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 40), tree_190, '__class__')
        # Obtaining the member '__name__' of a type (line 108)
        name___192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 40), class___191, '__name__')
        # Applying the binary operator '+' (line 108)
        result_add_193 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 29), '+', str_189, name___192)
        
        # Processing the call keyword arguments (line 108)
        kwargs_194 = {}
        # Getting the type of 'getattr' (line 108)
        getattr_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 108)
        getattr_call_result_195 = invoke(stypy.reporting.localization.Localization(__file__, 108, 15), getattr_187, *[self_188, result_add_193], **kwargs_194)
        
        # Assigning a type to the variable 'meth' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'meth', getattr_call_result_195)
        
        # Call to meth(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'tree' (line 109)
        tree_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'tree', False)
        # Processing the call keyword arguments (line 109)
        kwargs_198 = {}
        # Getting the type of 'meth' (line 109)
        meth_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'meth', False)
        # Calling meth(args, kwargs) (line 109)
        meth_call_result_199 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), meth_196, *[tree_197], **kwargs_198)
        
        
        # ################# End of 'visit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_200)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit'
        return stypy_return_type_200


    @norecursion
    def visit_Module(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Module'
        module_type_store = module_type_store.open_function_context('visit_Module', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
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

        
        # Getting the type of 'tree' (line 119)
        tree_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'tree')
        # Obtaining the member 'body' of a type (line 119)
        body_202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 20), tree_201, 'body')
        # Assigning a type to the variable 'body_202' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'body_202', body_202)
        # Testing if the for loop is going to be iterated (line 119)
        # Testing the type of a for loop iterable (line 119)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 119, 8), body_202)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 119, 8), body_202):
            # Getting the type of the for loop variable (line 119)
            for_loop_var_203 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 119, 8), body_202)
            # Assigning a type to the variable 'stmt' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'stmt', for_loop_var_203)
            # SSA begins for a for statement (line 119)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 120)
            # Processing the call arguments (line 120)
            # Getting the type of 'stmt' (line 120)
            stmt_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 23), 'stmt', False)
            # Processing the call keyword arguments (line 120)
            kwargs_207 = {}
            # Getting the type of 'self' (line 120)
            self_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 120)
            visit_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), self_204, 'visit')
            # Calling visit(args, kwargs) (line 120)
            visit_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), visit_205, *[stmt_206], **kwargs_207)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'visit_Module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Module' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_209)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Module'
        return stypy_return_type_209


    @norecursion
    def visit_Expr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Expr'
        module_type_store = module_type_store.open_function_context('visit_Expr', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 124)
        # Processing the call keyword arguments (line 124)
        kwargs_212 = {}
        # Getting the type of 'self' (line 124)
        self_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 124)
        fill_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), self_210, 'fill')
        # Calling fill(args, kwargs) (line 124)
        fill_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), fill_211, *[], **kwargs_212)
        
        
        # Call to visit(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'tree' (line 125)
        tree_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'tree', False)
        # Obtaining the member 'value' of a type (line 125)
        value_217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 19), tree_216, 'value')
        # Processing the call keyword arguments (line 125)
        kwargs_218 = {}
        # Getting the type of 'self' (line 125)
        self_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 125)
        visit_215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_214, 'visit')
        # Calling visit(args, kwargs) (line 125)
        visit_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), visit_215, *[value_217], **kwargs_218)
        
        
        # ################# End of 'visit_Expr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Expr' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_220)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Expr'
        return stypy_return_type_220


    @norecursion
    def visit_Import(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Import'
        module_type_store = module_type_store.open_function_context('visit_Import', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 128)
        # Processing the call arguments (line 128)
        str_223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 18), 'str', 'import ')
        # Processing the call keyword arguments (line 128)
        kwargs_224 = {}
        # Getting the type of 'self' (line 128)
        self_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 128)
        fill_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_221, 'fill')
        # Calling fill(args, kwargs) (line 128)
        fill_call_result_225 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), fill_222, *[str_223], **kwargs_224)
        
        
        # Call to interleave(...): (line 129)
        # Processing the call arguments (line 129)

        @norecursion
        def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_1'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 129, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_1.stypy_localization = localization
            _stypy_temp_lambda_1.stypy_type_of_self = None
            _stypy_temp_lambda_1.stypy_type_store = module_type_store
            _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
            _stypy_temp_lambda_1.stypy_param_names_list = []
            _stypy_temp_lambda_1.stypy_varargs_param_name = None
            _stypy_temp_lambda_1.stypy_kwargs_param_name = None
            _stypy_temp_lambda_1.stypy_call_defaults = defaults
            _stypy_temp_lambda_1.stypy_call_varargs = varargs
            _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_1', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 129)
            # Processing the call arguments (line 129)
            str_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 38), 'str', ', ')
            # Processing the call keyword arguments (line 129)
            kwargs_230 = {}
            # Getting the type of 'self' (line 129)
            self_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 129)
            write_228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 27), self_227, 'write')
            # Calling write(args, kwargs) (line 129)
            write_call_result_231 = invoke(stypy.reporting.localization.Localization(__file__, 129, 27), write_228, *[str_229], **kwargs_230)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), 'stypy_return_type', write_call_result_231)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_1' in the type store
            # Getting the type of 'stypy_return_type' (line 129)
            stypy_return_type_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_232)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_1'
            return stypy_return_type_232

        # Assigning a type to the variable '_stypy_temp_lambda_1' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
        # Getting the type of '_stypy_temp_lambda_1' (line 129)
        _stypy_temp_lambda_1_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), '_stypy_temp_lambda_1')
        # Getting the type of 'self' (line 129)
        self_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 129)
        visit_235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 45), self_234, 'visit')
        # Getting the type of 't' (line 129)
        t_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 57), 't', False)
        # Obtaining the member 'names' of a type (line 129)
        names_237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 57), t_236, 'names')
        # Processing the call keyword arguments (line 129)
        kwargs_238 = {}
        # Getting the type of 'interleave' (line 129)
        interleave_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 129)
        interleave_call_result_239 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), interleave_226, *[_stypy_temp_lambda_1_233, visit_235, names_237], **kwargs_238)
        
        
        # Call to write(...): (line 130)
        # Processing the call arguments (line 130)
        str_242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 19), 'str', '\n')
        # Processing the call keyword arguments (line 130)
        kwargs_243 = {}
        # Getting the type of 'self' (line 130)
        self_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 130)
        write_241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), self_240, 'write')
        # Calling write(args, kwargs) (line 130)
        write_call_result_244 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), write_241, *[str_242], **kwargs_243)
        
        
        # ################# End of 'visit_Import(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Import' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_245)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Import'
        return stypy_return_type_245


    @norecursion
    def visit_ImportFrom(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ImportFrom'
        module_type_store = module_type_store.open_function_context('visit_ImportFrom', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
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
        # Getting the type of 't' (line 134)
        t_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 't')
        # Obtaining the member 'module' of a type (line 134)
        module_247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), t_246, 'module')
        
        # Getting the type of 't' (line 134)
        t_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 't')
        # Obtaining the member 'module' of a type (line 134)
        module_249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 24), t_248, 'module')
        str_250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 36), 'str', '__future__')
        # Applying the binary operator '==' (line 134)
        result_eq_251 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 24), '==', module_249, str_250)
        
        # Applying the binary operator 'and' (line 134)
        result_and_keyword_252 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 11), 'and', module_247, result_eq_251)
        
        # Testing if the type of an if condition is none (line 134)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 134, 8), result_and_keyword_252):
            pass
        else:
            
            # Testing the type of an if condition (line 134)
            if_condition_253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 8), result_and_keyword_252)
            # Assigning a type to the variable 'if_condition_253' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'if_condition_253', if_condition_253)
            # SSA begins for if statement (line 134)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to extend(...): (line 135)
            # Processing the call arguments (line 135)
            # Calculating generator expression
            module_type_store = module_type_store.open_function_context('list comprehension expression', 135, 39, True)
            # Calculating comprehension expression
            # Getting the type of 't' (line 135)
            t_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 55), 't', False)
            # Obtaining the member 'names' of a type (line 135)
            names_260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 55), t_259, 'names')
            comprehension_261 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 39), names_260)
            # Assigning a type to the variable 'n' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 39), 'n', comprehension_261)
            # Getting the type of 'n' (line 135)
            n_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 39), 'n', False)
            # Obtaining the member 'name' of a type (line 135)
            name_258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 39), n_257, 'name')
            list_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 39), 'list')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 39), list_262, name_258)
            # Processing the call keyword arguments (line 135)
            kwargs_263 = {}
            # Getting the type of 'self' (line 135)
            self_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'self', False)
            # Obtaining the member 'future_imports' of a type (line 135)
            future_imports_255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), self_254, 'future_imports')
            # Obtaining the member 'extend' of a type (line 135)
            extend_256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), future_imports_255, 'extend')
            # Calling extend(args, kwargs) (line 135)
            extend_call_result_264 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), extend_256, *[list_262], **kwargs_263)
            
            # SSA join for if statement (line 134)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to fill(...): (line 137)
        # Processing the call arguments (line 137)
        str_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 18), 'str', 'from ')
        # Processing the call keyword arguments (line 137)
        kwargs_268 = {}
        # Getting the type of 'self' (line 137)
        self_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 137)
        fill_266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), self_265, 'fill')
        # Calling fill(args, kwargs) (line 137)
        fill_call_result_269 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), fill_266, *[str_267], **kwargs_268)
        
        
        # Call to write(...): (line 138)
        # Processing the call arguments (line 138)
        str_272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 19), 'str', '.')
        # Getting the type of 't' (line 138)
        t_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 25), 't', False)
        # Obtaining the member 'level' of a type (line 138)
        level_274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 25), t_273, 'level')
        # Applying the binary operator '*' (line 138)
        result_mul_275 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 19), '*', str_272, level_274)
        
        # Processing the call keyword arguments (line 138)
        kwargs_276 = {}
        # Getting the type of 'self' (line 138)
        self_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 138)
        write_271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 8), self_270, 'write')
        # Calling write(args, kwargs) (line 138)
        write_call_result_277 = invoke(stypy.reporting.localization.Localization(__file__, 138, 8), write_271, *[result_mul_275], **kwargs_276)
        
        # Getting the type of 't' (line 139)
        t_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 11), 't')
        # Obtaining the member 'module' of a type (line 139)
        module_279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 11), t_278, 'module')
        # Testing if the type of an if condition is none (line 139)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 139, 8), module_279):
            pass
        else:
            
            # Testing the type of an if condition (line 139)
            if_condition_280 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 8), module_279)
            # Assigning a type to the variable 'if_condition_280' (line 139)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'if_condition_280', if_condition_280)
            # SSA begins for if statement (line 139)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 140)
            # Processing the call arguments (line 140)
            # Getting the type of 't' (line 140)
            t_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 't', False)
            # Obtaining the member 'module' of a type (line 140)
            module_284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 23), t_283, 'module')
            # Processing the call keyword arguments (line 140)
            kwargs_285 = {}
            # Getting the type of 'self' (line 140)
            self_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 140)
            write_282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), self_281, 'write')
            # Calling write(args, kwargs) (line 140)
            write_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), write_282, *[module_284], **kwargs_285)
            
            # SSA join for if statement (line 139)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 141)
        # Processing the call arguments (line 141)
        str_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 19), 'str', ' import ')
        # Processing the call keyword arguments (line 141)
        kwargs_290 = {}
        # Getting the type of 'self' (line 141)
        self_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 141)
        write_288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), self_287, 'write')
        # Calling write(args, kwargs) (line 141)
        write_call_result_291 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), write_288, *[str_289], **kwargs_290)
        
        
        # Call to interleave(...): (line 142)
        # Processing the call arguments (line 142)

        @norecursion
        def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_2'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 142, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_2.stypy_localization = localization
            _stypy_temp_lambda_2.stypy_type_of_self = None
            _stypy_temp_lambda_2.stypy_type_store = module_type_store
            _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
            _stypy_temp_lambda_2.stypy_param_names_list = []
            _stypy_temp_lambda_2.stypy_varargs_param_name = None
            _stypy_temp_lambda_2.stypy_kwargs_param_name = None
            _stypy_temp_lambda_2.stypy_call_defaults = defaults
            _stypy_temp_lambda_2.stypy_call_varargs = varargs
            _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_2', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 142)
            # Processing the call arguments (line 142)
            str_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 38), 'str', ', ')
            # Processing the call keyword arguments (line 142)
            kwargs_296 = {}
            # Getting the type of 'self' (line 142)
            self_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 142)
            write_294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 27), self_293, 'write')
            # Calling write(args, kwargs) (line 142)
            write_call_result_297 = invoke(stypy.reporting.localization.Localization(__file__, 142, 27), write_294, *[str_295], **kwargs_296)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'stypy_return_type', write_call_result_297)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_2' in the type store
            # Getting the type of 'stypy_return_type' (line 142)
            stypy_return_type_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_298)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_2'
            return stypy_return_type_298

        # Assigning a type to the variable '_stypy_temp_lambda_2' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
        # Getting the type of '_stypy_temp_lambda_2' (line 142)
        _stypy_temp_lambda_2_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), '_stypy_temp_lambda_2')
        # Getting the type of 'self' (line 142)
        self_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 142)
        visit_301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 45), self_300, 'visit')
        # Getting the type of 't' (line 142)
        t_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 57), 't', False)
        # Obtaining the member 'names' of a type (line 142)
        names_303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 57), t_302, 'names')
        # Processing the call keyword arguments (line 142)
        kwargs_304 = {}
        # Getting the type of 'interleave' (line 142)
        interleave_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 142)
        interleave_call_result_305 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), interleave_292, *[_stypy_temp_lambda_2_299, visit_301, names_303], **kwargs_304)
        
        
        # Call to write(...): (line 143)
        # Processing the call arguments (line 143)
        str_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 19), 'str', '\n')
        # Processing the call keyword arguments (line 143)
        kwargs_309 = {}
        # Getting the type of 'self' (line 143)
        self_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 143)
        write_307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_306, 'write')
        # Calling write(args, kwargs) (line 143)
        write_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), write_307, *[str_308], **kwargs_309)
        
        
        # ################# End of 'visit_ImportFrom(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ImportFrom' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_311)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ImportFrom'
        return stypy_return_type_311


    @norecursion
    def visit_Assign(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Assign'
        module_type_store = module_type_store.open_function_context('visit_Assign', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 146)
        # Processing the call keyword arguments (line 146)
        kwargs_314 = {}
        # Getting the type of 'self' (line 146)
        self_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 146)
        fill_313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_312, 'fill')
        # Calling fill(args, kwargs) (line 146)
        fill_call_result_315 = invoke(stypy.reporting.localization.Localization(__file__, 146, 8), fill_313, *[], **kwargs_314)
        
        
        # Getting the type of 't' (line 147)
        t_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 't')
        # Obtaining the member 'targets' of a type (line 147)
        targets_317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 22), t_316, 'targets')
        # Assigning a type to the variable 'targets_317' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'targets_317', targets_317)
        # Testing if the for loop is going to be iterated (line 147)
        # Testing the type of a for loop iterable (line 147)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 147, 8), targets_317)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 147, 8), targets_317):
            # Getting the type of the for loop variable (line 147)
            for_loop_var_318 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 147, 8), targets_317)
            # Assigning a type to the variable 'target' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'target', for_loop_var_318)
            # SSA begins for a for statement (line 147)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 148)
            # Processing the call arguments (line 148)
            # Getting the type of 'target' (line 148)
            target_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 'target', False)
            # Processing the call keyword arguments (line 148)
            kwargs_322 = {}
            # Getting the type of 'self' (line 148)
            self_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 148)
            visit_320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), self_319, 'visit')
            # Calling visit(args, kwargs) (line 148)
            visit_call_result_323 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), visit_320, *[target_321], **kwargs_322)
            
            
            # Call to write(...): (line 149)
            # Processing the call arguments (line 149)
            str_326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 23), 'str', ' = ')
            # Processing the call keyword arguments (line 149)
            kwargs_327 = {}
            # Getting the type of 'self' (line 149)
            self_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 149)
            write_325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 12), self_324, 'write')
            # Calling write(args, kwargs) (line 149)
            write_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 149, 12), write_325, *[str_326], **kwargs_327)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to visit(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 't' (line 150)
        t_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), 't', False)
        # Obtaining the member 'value' of a type (line 150)
        value_332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 19), t_331, 'value')
        # Processing the call keyword arguments (line 150)
        kwargs_333 = {}
        # Getting the type of 'self' (line 150)
        self_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 150)
        visit_330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_329, 'visit')
        # Calling visit(args, kwargs) (line 150)
        visit_call_result_334 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), visit_330, *[value_332], **kwargs_333)
        
        
        # ################# End of 'visit_Assign(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Assign' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_335)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Assign'
        return stypy_return_type_335


    @norecursion
    def visit_AugAssign(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_AugAssign'
        module_type_store = module_type_store.open_function_context('visit_AugAssign', 152, 4, False)
        # Assigning a type to the variable 'self' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 153)
        # Processing the call keyword arguments (line 153)
        kwargs_338 = {}
        # Getting the type of 'self' (line 153)
        self_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 153)
        fill_337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 8), self_336, 'fill')
        # Calling fill(args, kwargs) (line 153)
        fill_call_result_339 = invoke(stypy.reporting.localization.Localization(__file__, 153, 8), fill_337, *[], **kwargs_338)
        
        
        # Call to visit(...): (line 154)
        # Processing the call arguments (line 154)
        # Getting the type of 't' (line 154)
        t_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), 't', False)
        # Obtaining the member 'target' of a type (line 154)
        target_343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 19), t_342, 'target')
        # Processing the call keyword arguments (line 154)
        kwargs_344 = {}
        # Getting the type of 'self' (line 154)
        self_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 154)
        visit_341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 8), self_340, 'visit')
        # Calling visit(args, kwargs) (line 154)
        visit_call_result_345 = invoke(stypy.reporting.localization.Localization(__file__, 154, 8), visit_341, *[target_343], **kwargs_344)
        
        
        # Call to write(...): (line 155)
        # Processing the call arguments (line 155)
        str_348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 19), 'str', ' ')
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 155)
        t_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 36), 't', False)
        # Obtaining the member 'op' of a type (line 155)
        op_350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 36), t_349, 'op')
        # Obtaining the member '__class__' of a type (line 155)
        class___351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 36), op_350, '__class__')
        # Obtaining the member '__name__' of a type (line 155)
        name___352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 36), class___351, '__name__')
        # Getting the type of 'self' (line 155)
        self_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 25), 'self', False)
        # Obtaining the member 'binop' of a type (line 155)
        binop_354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 25), self_353, 'binop')
        # Obtaining the member '__getitem__' of a type (line 155)
        getitem___355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 25), binop_354, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 155)
        subscript_call_result_356 = invoke(stypy.reporting.localization.Localization(__file__, 155, 25), getitem___355, name___352)
        
        # Applying the binary operator '+' (line 155)
        result_add_357 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 19), '+', str_348, subscript_call_result_356)
        
        str_358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 63), 'str', '= ')
        # Applying the binary operator '+' (line 155)
        result_add_359 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 61), '+', result_add_357, str_358)
        
        # Processing the call keyword arguments (line 155)
        kwargs_360 = {}
        # Getting the type of 'self' (line 155)
        self_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 155)
        write_347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_346, 'write')
        # Calling write(args, kwargs) (line 155)
        write_call_result_361 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), write_347, *[result_add_359], **kwargs_360)
        
        
        # Call to visit(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 't' (line 156)
        t_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 't', False)
        # Obtaining the member 'value' of a type (line 156)
        value_365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 19), t_364, 'value')
        # Processing the call keyword arguments (line 156)
        kwargs_366 = {}
        # Getting the type of 'self' (line 156)
        self_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 156)
        visit_363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 8), self_362, 'visit')
        # Calling visit(args, kwargs) (line 156)
        visit_call_result_367 = invoke(stypy.reporting.localization.Localization(__file__, 156, 8), visit_363, *[value_365], **kwargs_366)
        
        
        # ################# End of 'visit_AugAssign(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_AugAssign' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_368)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_AugAssign'
        return stypy_return_type_368


    @norecursion
    def visit_Return(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Return'
        module_type_store = module_type_store.open_function_context('visit_Return', 158, 4, False)
        # Assigning a type to the variable 'self' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 159)
        # Processing the call arguments (line 159)
        str_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 18), 'str', 'return')
        # Processing the call keyword arguments (line 159)
        kwargs_372 = {}
        # Getting the type of 'self' (line 159)
        self_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 159)
        fill_370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 8), self_369, 'fill')
        # Calling fill(args, kwargs) (line 159)
        fill_call_result_373 = invoke(stypy.reporting.localization.Localization(__file__, 159, 8), fill_370, *[str_371], **kwargs_372)
        
        # Getting the type of 't' (line 160)
        t_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 11), 't')
        # Obtaining the member 'value' of a type (line 160)
        value_375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 11), t_374, 'value')
        # Testing if the type of an if condition is none (line 160)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 160, 8), value_375):
            pass
        else:
            
            # Testing the type of an if condition (line 160)
            if_condition_376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 8), value_375)
            # Assigning a type to the variable 'if_condition_376' (line 160)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'if_condition_376', if_condition_376)
            # SSA begins for if statement (line 160)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 161)
            # Processing the call arguments (line 161)
            str_379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 23), 'str', ' ')
            # Processing the call keyword arguments (line 161)
            kwargs_380 = {}
            # Getting the type of 'self' (line 161)
            self_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 161)
            write_378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), self_377, 'write')
            # Calling write(args, kwargs) (line 161)
            write_call_result_381 = invoke(stypy.reporting.localization.Localization(__file__, 161, 12), write_378, *[str_379], **kwargs_380)
            
            
            # Call to visit(...): (line 162)
            # Processing the call arguments (line 162)
            # Getting the type of 't' (line 162)
            t_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 23), 't', False)
            # Obtaining the member 'value' of a type (line 162)
            value_385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 23), t_384, 'value')
            # Processing the call keyword arguments (line 162)
            kwargs_386 = {}
            # Getting the type of 'self' (line 162)
            self_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 162)
            visit_383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 12), self_382, 'visit')
            # Calling visit(args, kwargs) (line 162)
            visit_call_result_387 = invoke(stypy.reporting.localization.Localization(__file__, 162, 12), visit_383, *[value_385], **kwargs_386)
            
            # SSA join for if statement (line 160)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Return(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Return' in the type store
        # Getting the type of 'stypy_return_type' (line 158)
        stypy_return_type_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_388)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Return'
        return stypy_return_type_388


    @norecursion
    def visit_Pass(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Pass'
        module_type_store = module_type_store.open_function_context('visit_Pass', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 166)
        # Processing the call arguments (line 166)
        str_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 18), 'str', 'pass')
        # Processing the call keyword arguments (line 166)
        kwargs_392 = {}
        # Getting the type of 'self' (line 166)
        self_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 166)
        fill_390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_389, 'fill')
        # Calling fill(args, kwargs) (line 166)
        fill_call_result_393 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), fill_390, *[str_391], **kwargs_392)
        
        
        # ################# End of 'visit_Pass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Pass' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_394)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Pass'
        return stypy_return_type_394


    @norecursion
    def visit_Break(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Break'
        module_type_store = module_type_store.open_function_context('visit_Break', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 169)
        # Processing the call arguments (line 169)
        str_397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 18), 'str', 'break')
        # Processing the call keyword arguments (line 169)
        kwargs_398 = {}
        # Getting the type of 'self' (line 169)
        self_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 169)
        fill_396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_395, 'fill')
        # Calling fill(args, kwargs) (line 169)
        fill_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), fill_396, *[str_397], **kwargs_398)
        
        
        # ################# End of 'visit_Break(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Break' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_400)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Break'
        return stypy_return_type_400


    @norecursion
    def visit_Continue(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Continue'
        module_type_store = module_type_store.open_function_context('visit_Continue', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 172)
        # Processing the call arguments (line 172)
        str_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 18), 'str', 'continue')
        # Processing the call keyword arguments (line 172)
        kwargs_404 = {}
        # Getting the type of 'self' (line 172)
        self_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 172)
        fill_402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_401, 'fill')
        # Calling fill(args, kwargs) (line 172)
        fill_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), fill_402, *[str_403], **kwargs_404)
        
        
        # ################# End of 'visit_Continue(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Continue' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_406)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Continue'
        return stypy_return_type_406


    @norecursion
    def visit_Delete(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Delete'
        module_type_store = module_type_store.open_function_context('visit_Delete', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 175)
        # Processing the call arguments (line 175)
        str_409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 18), 'str', 'del ')
        # Processing the call keyword arguments (line 175)
        kwargs_410 = {}
        # Getting the type of 'self' (line 175)
        self_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 175)
        fill_408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_407, 'fill')
        # Calling fill(args, kwargs) (line 175)
        fill_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), fill_408, *[str_409], **kwargs_410)
        
        
        # Call to interleave(...): (line 176)
        # Processing the call arguments (line 176)

        @norecursion
        def _stypy_temp_lambda_3(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_3'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_3', 176, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_3.stypy_localization = localization
            _stypy_temp_lambda_3.stypy_type_of_self = None
            _stypy_temp_lambda_3.stypy_type_store = module_type_store
            _stypy_temp_lambda_3.stypy_function_name = '_stypy_temp_lambda_3'
            _stypy_temp_lambda_3.stypy_param_names_list = []
            _stypy_temp_lambda_3.stypy_varargs_param_name = None
            _stypy_temp_lambda_3.stypy_kwargs_param_name = None
            _stypy_temp_lambda_3.stypy_call_defaults = defaults
            _stypy_temp_lambda_3.stypy_call_varargs = varargs
            _stypy_temp_lambda_3.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_3', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_3', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 176)
            # Processing the call arguments (line 176)
            str_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 38), 'str', ', ')
            # Processing the call keyword arguments (line 176)
            kwargs_416 = {}
            # Getting the type of 'self' (line 176)
            self_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 176)
            write_414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 27), self_413, 'write')
            # Calling write(args, kwargs) (line 176)
            write_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 176, 27), write_414, *[str_415], **kwargs_416)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 176)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'stypy_return_type', write_call_result_417)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_3' in the type store
            # Getting the type of 'stypy_return_type' (line 176)
            stypy_return_type_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_418)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_3'
            return stypy_return_type_418

        # Assigning a type to the variable '_stypy_temp_lambda_3' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), '_stypy_temp_lambda_3', _stypy_temp_lambda_3)
        # Getting the type of '_stypy_temp_lambda_3' (line 176)
        _stypy_temp_lambda_3_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 19), '_stypy_temp_lambda_3')
        # Getting the type of 'self' (line 176)
        self_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 176)
        visit_421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 45), self_420, 'visit')
        # Getting the type of 't' (line 176)
        t_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 57), 't', False)
        # Obtaining the member 'targets' of a type (line 176)
        targets_423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 57), t_422, 'targets')
        # Processing the call keyword arguments (line 176)
        kwargs_424 = {}
        # Getting the type of 'interleave' (line 176)
        interleave_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 176)
        interleave_call_result_425 = invoke(stypy.reporting.localization.Localization(__file__, 176, 8), interleave_412, *[_stypy_temp_lambda_3_419, visit_421, targets_423], **kwargs_424)
        
        
        # ################# End of 'visit_Delete(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Delete' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_426)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Delete'
        return stypy_return_type_426


    @norecursion
    def visit_Assert(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Assert'
        module_type_store = module_type_store.open_function_context('visit_Assert', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 179)
        # Processing the call arguments (line 179)
        str_429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 18), 'str', 'assert ')
        # Processing the call keyword arguments (line 179)
        kwargs_430 = {}
        # Getting the type of 'self' (line 179)
        self_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 179)
        fill_428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), self_427, 'fill')
        # Calling fill(args, kwargs) (line 179)
        fill_call_result_431 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), fill_428, *[str_429], **kwargs_430)
        
        
        # Call to visit(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 't' (line 180)
        t_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 't', False)
        # Obtaining the member 'test' of a type (line 180)
        test_435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 19), t_434, 'test')
        # Processing the call keyword arguments (line 180)
        kwargs_436 = {}
        # Getting the type of 'self' (line 180)
        self_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 180)
        visit_433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_432, 'visit')
        # Calling visit(args, kwargs) (line 180)
        visit_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), visit_433, *[test_435], **kwargs_436)
        
        # Getting the type of 't' (line 181)
        t_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 't')
        # Obtaining the member 'msg' of a type (line 181)
        msg_439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 11), t_438, 'msg')
        # Testing if the type of an if condition is none (line 181)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 181, 8), msg_439):
            pass
        else:
            
            # Testing the type of an if condition (line 181)
            if_condition_440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 8), msg_439)
            # Assigning a type to the variable 'if_condition_440' (line 181)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'if_condition_440', if_condition_440)
            # SSA begins for if statement (line 181)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 182)
            # Processing the call arguments (line 182)
            str_443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 23), 'str', ', ')
            # Processing the call keyword arguments (line 182)
            kwargs_444 = {}
            # Getting the type of 'self' (line 182)
            self_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 182)
            write_442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 12), self_441, 'write')
            # Calling write(args, kwargs) (line 182)
            write_call_result_445 = invoke(stypy.reporting.localization.Localization(__file__, 182, 12), write_442, *[str_443], **kwargs_444)
            
            
            # Call to visit(...): (line 183)
            # Processing the call arguments (line 183)
            # Getting the type of 't' (line 183)
            t_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 23), 't', False)
            # Obtaining the member 'msg' of a type (line 183)
            msg_449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 23), t_448, 'msg')
            # Processing the call keyword arguments (line 183)
            kwargs_450 = {}
            # Getting the type of 'self' (line 183)
            self_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 183)
            visit_447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 12), self_446, 'visit')
            # Calling visit(args, kwargs) (line 183)
            visit_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 183, 12), visit_447, *[msg_449], **kwargs_450)
            
            # SSA join for if statement (line 181)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Assert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Assert' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_452)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Assert'
        return stypy_return_type_452


    @norecursion
    def visit_Exec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Exec'
        module_type_store = module_type_store.open_function_context('visit_Exec', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 186)
        # Processing the call arguments (line 186)
        str_455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 18), 'str', 'exec ')
        # Processing the call keyword arguments (line 186)
        kwargs_456 = {}
        # Getting the type of 'self' (line 186)
        self_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 186)
        fill_454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_453, 'fill')
        # Calling fill(args, kwargs) (line 186)
        fill_call_result_457 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), fill_454, *[str_455], **kwargs_456)
        
        
        # Call to visit(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 't' (line 187)
        t_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 19), 't', False)
        # Obtaining the member 'body' of a type (line 187)
        body_461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 19), t_460, 'body')
        # Processing the call keyword arguments (line 187)
        kwargs_462 = {}
        # Getting the type of 'self' (line 187)
        self_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 187)
        visit_459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), self_458, 'visit')
        # Calling visit(args, kwargs) (line 187)
        visit_call_result_463 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), visit_459, *[body_461], **kwargs_462)
        
        # Getting the type of 't' (line 188)
        t_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 't')
        # Obtaining the member 'globals' of a type (line 188)
        globals_465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 11), t_464, 'globals')
        # Testing if the type of an if condition is none (line 188)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 188, 8), globals_465):
            pass
        else:
            
            # Testing the type of an if condition (line 188)
            if_condition_466 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 8), globals_465)
            # Assigning a type to the variable 'if_condition_466' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'if_condition_466', if_condition_466)
            # SSA begins for if statement (line 188)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 189)
            # Processing the call arguments (line 189)
            str_469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 23), 'str', ' in ')
            # Processing the call keyword arguments (line 189)
            kwargs_470 = {}
            # Getting the type of 'self' (line 189)
            self_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 189)
            write_468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), self_467, 'write')
            # Calling write(args, kwargs) (line 189)
            write_call_result_471 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), write_468, *[str_469], **kwargs_470)
            
            
            # Call to visit(...): (line 190)
            # Processing the call arguments (line 190)
            # Getting the type of 't' (line 190)
            t_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 23), 't', False)
            # Obtaining the member 'globals' of a type (line 190)
            globals_475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 23), t_474, 'globals')
            # Processing the call keyword arguments (line 190)
            kwargs_476 = {}
            # Getting the type of 'self' (line 190)
            self_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 190)
            visit_473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 12), self_472, 'visit')
            # Calling visit(args, kwargs) (line 190)
            visit_call_result_477 = invoke(stypy.reporting.localization.Localization(__file__, 190, 12), visit_473, *[globals_475], **kwargs_476)
            
            # SSA join for if statement (line 188)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 191)
        t_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 't')
        # Obtaining the member 'locals' of a type (line 191)
        locals_479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 11), t_478, 'locals')
        # Testing if the type of an if condition is none (line 191)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 191, 8), locals_479):
            pass
        else:
            
            # Testing the type of an if condition (line 191)
            if_condition_480 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 8), locals_479)
            # Assigning a type to the variable 'if_condition_480' (line 191)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'if_condition_480', if_condition_480)
            # SSA begins for if statement (line 191)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 192)
            # Processing the call arguments (line 192)
            str_483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 23), 'str', ', ')
            # Processing the call keyword arguments (line 192)
            kwargs_484 = {}
            # Getting the type of 'self' (line 192)
            self_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 192)
            write_482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 12), self_481, 'write')
            # Calling write(args, kwargs) (line 192)
            write_call_result_485 = invoke(stypy.reporting.localization.Localization(__file__, 192, 12), write_482, *[str_483], **kwargs_484)
            
            
            # Call to visit(...): (line 193)
            # Processing the call arguments (line 193)
            # Getting the type of 't' (line 193)
            t_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 't', False)
            # Obtaining the member 'locals' of a type (line 193)
            locals_489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 23), t_488, 'locals')
            # Processing the call keyword arguments (line 193)
            kwargs_490 = {}
            # Getting the type of 'self' (line 193)
            self_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 193)
            visit_487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), self_486, 'visit')
            # Calling visit(args, kwargs) (line 193)
            visit_call_result_491 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), visit_487, *[locals_489], **kwargs_490)
            
            # SSA join for if statement (line 191)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Exec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Exec' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_492)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Exec'
        return stypy_return_type_492


    @norecursion
    def visit_Print(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Print'
        module_type_store = module_type_store.open_function_context('visit_Print', 195, 4, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 196)
        # Processing the call arguments (line 196)
        str_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 18), 'str', 'print ')
        # Processing the call keyword arguments (line 196)
        kwargs_496 = {}
        # Getting the type of 'self' (line 196)
        self_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 196)
        fill_494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 8), self_493, 'fill')
        # Calling fill(args, kwargs) (line 196)
        fill_call_result_497 = invoke(stypy.reporting.localization.Localization(__file__, 196, 8), fill_494, *[str_495], **kwargs_496)
        
        
        # Assigning a Name to a Name (line 197):
        
        # Assigning a Name to a Name (line 197):
        # Getting the type of 'False' (line 197)
        False_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 19), 'False')
        # Assigning a type to the variable 'do_comma' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'do_comma', False_498)
        # Getting the type of 't' (line 198)
        t_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 11), 't')
        # Obtaining the member 'dest' of a type (line 198)
        dest_500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 11), t_499, 'dest')
        # Testing if the type of an if condition is none (line 198)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 198, 8), dest_500):
            pass
        else:
            
            # Testing the type of an if condition (line 198)
            if_condition_501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 8), dest_500)
            # Assigning a type to the variable 'if_condition_501' (line 198)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'if_condition_501', if_condition_501)
            # SSA begins for if statement (line 198)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 199)
            # Processing the call arguments (line 199)
            str_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 23), 'str', '>>')
            # Processing the call keyword arguments (line 199)
            kwargs_505 = {}
            # Getting the type of 'self' (line 199)
            self_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 199)
            write_503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 12), self_502, 'write')
            # Calling write(args, kwargs) (line 199)
            write_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), write_503, *[str_504], **kwargs_505)
            
            
            # Call to visit(...): (line 200)
            # Processing the call arguments (line 200)
            # Getting the type of 't' (line 200)
            t_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 23), 't', False)
            # Obtaining the member 'dest' of a type (line 200)
            dest_510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 23), t_509, 'dest')
            # Processing the call keyword arguments (line 200)
            kwargs_511 = {}
            # Getting the type of 'self' (line 200)
            self_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 200)
            visit_508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 12), self_507, 'visit')
            # Calling visit(args, kwargs) (line 200)
            visit_call_result_512 = invoke(stypy.reporting.localization.Localization(__file__, 200, 12), visit_508, *[dest_510], **kwargs_511)
            
            
            # Assigning a Name to a Name (line 201):
            
            # Assigning a Name to a Name (line 201):
            # Getting the type of 'True' (line 201)
            True_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'True')
            # Assigning a type to the variable 'do_comma' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'do_comma', True_513)
            # SSA join for if statement (line 198)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 't' (line 202)
        t_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 17), 't')
        # Obtaining the member 'values' of a type (line 202)
        values_515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 17), t_514, 'values')
        # Assigning a type to the variable 'values_515' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'values_515', values_515)
        # Testing if the for loop is going to be iterated (line 202)
        # Testing the type of a for loop iterable (line 202)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 202, 8), values_515)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 202, 8), values_515):
            # Getting the type of the for loop variable (line 202)
            for_loop_var_516 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 202, 8), values_515)
            # Assigning a type to the variable 'e' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'e', for_loop_var_516)
            # SSA begins for a for statement (line 202)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'do_comma' (line 203)
            do_comma_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 15), 'do_comma')
            # Testing if the type of an if condition is none (line 203)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 203, 12), do_comma_517):
                
                # Assigning a Name to a Name (line 206):
                
                # Assigning a Name to a Name (line 206):
                # Getting the type of 'True' (line 206)
                True_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 27), 'True')
                # Assigning a type to the variable 'do_comma' (line 206)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'do_comma', True_524)
            else:
                
                # Testing the type of an if condition (line 203)
                if_condition_518 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 12), do_comma_517)
                # Assigning a type to the variable 'if_condition_518' (line 203)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 12), 'if_condition_518', if_condition_518)
                # SSA begins for if statement (line 203)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 204)
                # Processing the call arguments (line 204)
                str_521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 27), 'str', ', ')
                # Processing the call keyword arguments (line 204)
                kwargs_522 = {}
                # Getting the type of 'self' (line 204)
                self_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 204)
                write_520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 16), self_519, 'write')
                # Calling write(args, kwargs) (line 204)
                write_call_result_523 = invoke(stypy.reporting.localization.Localization(__file__, 204, 16), write_520, *[str_521], **kwargs_522)
                
                # SSA branch for the else part of an if statement (line 203)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 206):
                
                # Assigning a Name to a Name (line 206):
                # Getting the type of 'True' (line 206)
                True_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 27), 'True')
                # Assigning a type to the variable 'do_comma' (line 206)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 16), 'do_comma', True_524)
                # SSA join for if statement (line 203)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to visit(...): (line 207)
            # Processing the call arguments (line 207)
            # Getting the type of 'e' (line 207)
            e_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 23), 'e', False)
            # Processing the call keyword arguments (line 207)
            kwargs_528 = {}
            # Getting the type of 'self' (line 207)
            self_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 207)
            visit_526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), self_525, 'visit')
            # Calling visit(args, kwargs) (line 207)
            visit_call_result_529 = invoke(stypy.reporting.localization.Localization(__file__, 207, 12), visit_526, *[e_527], **kwargs_528)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 't' (line 208)
        t_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 't')
        # Obtaining the member 'nl' of a type (line 208)
        nl_531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 15), t_530, 'nl')
        # Applying the 'not' unary operator (line 208)
        result_not__532 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 11), 'not', nl_531)
        
        # Testing if the type of an if condition is none (line 208)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 208, 8), result_not__532):
            pass
        else:
            
            # Testing the type of an if condition (line 208)
            if_condition_533 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 8), result_not__532)
            # Assigning a type to the variable 'if_condition_533' (line 208)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'if_condition_533', if_condition_533)
            # SSA begins for if statement (line 208)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 209)
            # Processing the call arguments (line 209)
            str_536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 23), 'str', ',')
            # Processing the call keyword arguments (line 209)
            kwargs_537 = {}
            # Getting the type of 'self' (line 209)
            self_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 209)
            write_535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 12), self_534, 'write')
            # Calling write(args, kwargs) (line 209)
            write_call_result_538 = invoke(stypy.reporting.localization.Localization(__file__, 209, 12), write_535, *[str_536], **kwargs_537)
            
            # SSA join for if statement (line 208)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Print(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Print' in the type store
        # Getting the type of 'stypy_return_type' (line 195)
        stypy_return_type_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_539)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Print'
        return stypy_return_type_539


    @norecursion
    def visit_Global(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Global'
        module_type_store = module_type_store.open_function_context('visit_Global', 211, 4, False)
        # Assigning a type to the variable 'self' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 212)
        # Processing the call arguments (line 212)
        str_542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 18), 'str', 'global ')
        # Processing the call keyword arguments (line 212)
        kwargs_543 = {}
        # Getting the type of 'self' (line 212)
        self_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 212)
        fill_541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 8), self_540, 'fill')
        # Calling fill(args, kwargs) (line 212)
        fill_call_result_544 = invoke(stypy.reporting.localization.Localization(__file__, 212, 8), fill_541, *[str_542], **kwargs_543)
        
        
        # Call to interleave(...): (line 213)
        # Processing the call arguments (line 213)

        @norecursion
        def _stypy_temp_lambda_4(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_4'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_4', 213, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_4.stypy_localization = localization
            _stypy_temp_lambda_4.stypy_type_of_self = None
            _stypy_temp_lambda_4.stypy_type_store = module_type_store
            _stypy_temp_lambda_4.stypy_function_name = '_stypy_temp_lambda_4'
            _stypy_temp_lambda_4.stypy_param_names_list = []
            _stypy_temp_lambda_4.stypy_varargs_param_name = None
            _stypy_temp_lambda_4.stypy_kwargs_param_name = None
            _stypy_temp_lambda_4.stypy_call_defaults = defaults
            _stypy_temp_lambda_4.stypy_call_varargs = varargs
            _stypy_temp_lambda_4.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_4', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_4', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 213)
            # Processing the call arguments (line 213)
            str_548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 38), 'str', ', ')
            # Processing the call keyword arguments (line 213)
            kwargs_549 = {}
            # Getting the type of 'self' (line 213)
            self_546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 213)
            write_547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 27), self_546, 'write')
            # Calling write(args, kwargs) (line 213)
            write_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 213, 27), write_547, *[str_548], **kwargs_549)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'stypy_return_type', write_call_result_550)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_4' in the type store
            # Getting the type of 'stypy_return_type' (line 213)
            stypy_return_type_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_551)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_4'
            return stypy_return_type_551

        # Assigning a type to the variable '_stypy_temp_lambda_4' (line 213)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), '_stypy_temp_lambda_4', _stypy_temp_lambda_4)
        # Getting the type of '_stypy_temp_lambda_4' (line 213)
        _stypy_temp_lambda_4_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), '_stypy_temp_lambda_4')
        # Getting the type of 'self' (line 213)
        self_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 45), 'self', False)
        # Obtaining the member 'write' of a type (line 213)
        write_554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 45), self_553, 'write')
        # Getting the type of 't' (line 213)
        t_555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 57), 't', False)
        # Obtaining the member 'names' of a type (line 213)
        names_556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 57), t_555, 'names')
        # Processing the call keyword arguments (line 213)
        kwargs_557 = {}
        # Getting the type of 'interleave' (line 213)
        interleave_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 213)
        interleave_call_result_558 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), interleave_545, *[_stypy_temp_lambda_4_552, write_554, names_556], **kwargs_557)
        
        
        # ################# End of 'visit_Global(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Global' in the type store
        # Getting the type of 'stypy_return_type' (line 211)
        stypy_return_type_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_559)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Global'
        return stypy_return_type_559


    @norecursion
    def visit_Yield(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Yield'
        module_type_store = module_type_store.open_function_context('visit_Yield', 215, 4, False)
        # Assigning a type to the variable 'self' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 216)
        # Processing the call arguments (line 216)
        str_562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 19), 'str', '(')
        # Processing the call keyword arguments (line 216)
        kwargs_563 = {}
        # Getting the type of 'self' (line 216)
        self_560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 216)
        write_561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 8), self_560, 'write')
        # Calling write(args, kwargs) (line 216)
        write_call_result_564 = invoke(stypy.reporting.localization.Localization(__file__, 216, 8), write_561, *[str_562], **kwargs_563)
        
        
        # Call to write(...): (line 217)
        # Processing the call arguments (line 217)
        str_567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 19), 'str', 'yield')
        # Processing the call keyword arguments (line 217)
        kwargs_568 = {}
        # Getting the type of 'self' (line 217)
        self_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 217)
        write_566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 8), self_565, 'write')
        # Calling write(args, kwargs) (line 217)
        write_call_result_569 = invoke(stypy.reporting.localization.Localization(__file__, 217, 8), write_566, *[str_567], **kwargs_568)
        
        # Getting the type of 't' (line 218)
        t_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 't')
        # Obtaining the member 'value' of a type (line 218)
        value_571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 11), t_570, 'value')
        # Testing if the type of an if condition is none (line 218)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 218, 8), value_571):
            pass
        else:
            
            # Testing the type of an if condition (line 218)
            if_condition_572 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 8), value_571)
            # Assigning a type to the variable 'if_condition_572' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'if_condition_572', if_condition_572)
            # SSA begins for if statement (line 218)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 219)
            # Processing the call arguments (line 219)
            str_575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 23), 'str', ' ')
            # Processing the call keyword arguments (line 219)
            kwargs_576 = {}
            # Getting the type of 'self' (line 219)
            self_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 219)
            write_574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 12), self_573, 'write')
            # Calling write(args, kwargs) (line 219)
            write_call_result_577 = invoke(stypy.reporting.localization.Localization(__file__, 219, 12), write_574, *[str_575], **kwargs_576)
            
            
            # Call to visit(...): (line 220)
            # Processing the call arguments (line 220)
            # Getting the type of 't' (line 220)
            t_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 23), 't', False)
            # Obtaining the member 'value' of a type (line 220)
            value_581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 23), t_580, 'value')
            # Processing the call keyword arguments (line 220)
            kwargs_582 = {}
            # Getting the type of 'self' (line 220)
            self_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 220)
            visit_579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 12), self_578, 'visit')
            # Calling visit(args, kwargs) (line 220)
            visit_call_result_583 = invoke(stypy.reporting.localization.Localization(__file__, 220, 12), visit_579, *[value_581], **kwargs_582)
            
            # SSA join for if statement (line 218)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 221)
        # Processing the call arguments (line 221)
        str_586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 19), 'str', ')')
        # Processing the call keyword arguments (line 221)
        kwargs_587 = {}
        # Getting the type of 'self' (line 221)
        self_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 221)
        write_585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_584, 'write')
        # Calling write(args, kwargs) (line 221)
        write_call_result_588 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), write_585, *[str_586], **kwargs_587)
        
        
        # ################# End of 'visit_Yield(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Yield' in the type store
        # Getting the type of 'stypy_return_type' (line 215)
        stypy_return_type_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_589)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Yield'
        return stypy_return_type_589


    @norecursion
    def visit_Raise(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Raise'
        module_type_store = module_type_store.open_function_context('visit_Raise', 223, 4, False)
        # Assigning a type to the variable 'self' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 224)
        # Processing the call arguments (line 224)
        str_592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 18), 'str', 'raise ')
        # Processing the call keyword arguments (line 224)
        kwargs_593 = {}
        # Getting the type of 'self' (line 224)
        self_590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 224)
        fill_591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), self_590, 'fill')
        # Calling fill(args, kwargs) (line 224)
        fill_call_result_594 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), fill_591, *[str_592], **kwargs_593)
        
        # Getting the type of 't' (line 225)
        t_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 't')
        # Obtaining the member 'type' of a type (line 225)
        type_596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 11), t_595, 'type')
        # Testing if the type of an if condition is none (line 225)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 225, 8), type_596):
            pass
        else:
            
            # Testing the type of an if condition (line 225)
            if_condition_597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 8), type_596)
            # Assigning a type to the variable 'if_condition_597' (line 225)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 8), 'if_condition_597', if_condition_597)
            # SSA begins for if statement (line 225)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to visit(...): (line 226)
            # Processing the call arguments (line 226)
            # Getting the type of 't' (line 226)
            t_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 23), 't', False)
            # Obtaining the member 'type' of a type (line 226)
            type_601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 23), t_600, 'type')
            # Processing the call keyword arguments (line 226)
            kwargs_602 = {}
            # Getting the type of 'self' (line 226)
            self_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 226)
            visit_599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 12), self_598, 'visit')
            # Calling visit(args, kwargs) (line 226)
            visit_call_result_603 = invoke(stypy.reporting.localization.Localization(__file__, 226, 12), visit_599, *[type_601], **kwargs_602)
            
            # SSA join for if statement (line 225)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 227)
        t_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 't')
        # Obtaining the member 'inst' of a type (line 227)
        inst_605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 11), t_604, 'inst')
        # Testing if the type of an if condition is none (line 227)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 227, 8), inst_605):
            pass
        else:
            
            # Testing the type of an if condition (line 227)
            if_condition_606 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 8), inst_605)
            # Assigning a type to the variable 'if_condition_606' (line 227)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 8), 'if_condition_606', if_condition_606)
            # SSA begins for if statement (line 227)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 228)
            # Processing the call arguments (line 228)
            str_609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 23), 'str', ', ')
            # Processing the call keyword arguments (line 228)
            kwargs_610 = {}
            # Getting the type of 'self' (line 228)
            self_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 228)
            write_608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 12), self_607, 'write')
            # Calling write(args, kwargs) (line 228)
            write_call_result_611 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), write_608, *[str_609], **kwargs_610)
            
            
            # Call to visit(...): (line 229)
            # Processing the call arguments (line 229)
            # Getting the type of 't' (line 229)
            t_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 23), 't', False)
            # Obtaining the member 'inst' of a type (line 229)
            inst_615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 23), t_614, 'inst')
            # Processing the call keyword arguments (line 229)
            kwargs_616 = {}
            # Getting the type of 'self' (line 229)
            self_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 229)
            visit_613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), self_612, 'visit')
            # Calling visit(args, kwargs) (line 229)
            visit_call_result_617 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), visit_613, *[inst_615], **kwargs_616)
            
            # SSA join for if statement (line 227)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 230)
        t_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 't')
        # Obtaining the member 'tback' of a type (line 230)
        tback_619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 11), t_618, 'tback')
        # Testing if the type of an if condition is none (line 230)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 230, 8), tback_619):
            pass
        else:
            
            # Testing the type of an if condition (line 230)
            if_condition_620 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), tback_619)
            # Assigning a type to the variable 'if_condition_620' (line 230)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_620', if_condition_620)
            # SSA begins for if statement (line 230)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 231)
            # Processing the call arguments (line 231)
            str_623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 23), 'str', ', ')
            # Processing the call keyword arguments (line 231)
            kwargs_624 = {}
            # Getting the type of 'self' (line 231)
            self_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 231)
            write_622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), self_621, 'write')
            # Calling write(args, kwargs) (line 231)
            write_call_result_625 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), write_622, *[str_623], **kwargs_624)
            
            
            # Call to visit(...): (line 232)
            # Processing the call arguments (line 232)
            # Getting the type of 't' (line 232)
            t_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 23), 't', False)
            # Obtaining the member 'tback' of a type (line 232)
            tback_629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 23), t_628, 'tback')
            # Processing the call keyword arguments (line 232)
            kwargs_630 = {}
            # Getting the type of 'self' (line 232)
            self_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 232)
            visit_627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), self_626, 'visit')
            # Calling visit(args, kwargs) (line 232)
            visit_call_result_631 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), visit_627, *[tback_629], **kwargs_630)
            
            # SSA join for if statement (line 230)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Raise(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Raise' in the type store
        # Getting the type of 'stypy_return_type' (line 223)
        stypy_return_type_632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_632)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Raise'
        return stypy_return_type_632


    @norecursion
    def visit_TryExcept(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_TryExcept'
        module_type_store = module_type_store.open_function_context('visit_TryExcept', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 235)
        # Processing the call arguments (line 235)
        str_635 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 18), 'str', 'try')
        # Processing the call keyword arguments (line 235)
        kwargs_636 = {}
        # Getting the type of 'self' (line 235)
        self_633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 235)
        fill_634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 8), self_633, 'fill')
        # Calling fill(args, kwargs) (line 235)
        fill_call_result_637 = invoke(stypy.reporting.localization.Localization(__file__, 235, 8), fill_634, *[str_635], **kwargs_636)
        
        
        # Call to enter(...): (line 236)
        # Processing the call keyword arguments (line 236)
        kwargs_640 = {}
        # Getting the type of 'self' (line 236)
        self_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 236)
        enter_639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 8), self_638, 'enter')
        # Calling enter(args, kwargs) (line 236)
        enter_call_result_641 = invoke(stypy.reporting.localization.Localization(__file__, 236, 8), enter_639, *[], **kwargs_640)
        
        
        # Call to visit(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 't' (line 237)
        t_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 't', False)
        # Obtaining the member 'body' of a type (line 237)
        body_645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 19), t_644, 'body')
        # Processing the call keyword arguments (line 237)
        kwargs_646 = {}
        # Getting the type of 'self' (line 237)
        self_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 237)
        visit_643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 8), self_642, 'visit')
        # Calling visit(args, kwargs) (line 237)
        visit_call_result_647 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), visit_643, *[body_645], **kwargs_646)
        
        
        # Call to leave(...): (line 238)
        # Processing the call keyword arguments (line 238)
        kwargs_650 = {}
        # Getting the type of 'self' (line 238)
        self_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 238)
        leave_649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), self_648, 'leave')
        # Calling leave(args, kwargs) (line 238)
        leave_call_result_651 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), leave_649, *[], **kwargs_650)
        
        
        # Getting the type of 't' (line 240)
        t_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 18), 't')
        # Obtaining the member 'handlers' of a type (line 240)
        handlers_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 18), t_652, 'handlers')
        # Assigning a type to the variable 'handlers_653' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'handlers_653', handlers_653)
        # Testing if the for loop is going to be iterated (line 240)
        # Testing the type of a for loop iterable (line 240)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 240, 8), handlers_653)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 240, 8), handlers_653):
            # Getting the type of the for loop variable (line 240)
            for_loop_var_654 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 240, 8), handlers_653)
            # Assigning a type to the variable 'ex' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'ex', for_loop_var_654)
            # SSA begins for a for statement (line 240)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 241)
            # Processing the call arguments (line 241)
            # Getting the type of 'ex' (line 241)
            ex_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 'ex', False)
            # Processing the call keyword arguments (line 241)
            kwargs_658 = {}
            # Getting the type of 'self' (line 241)
            self_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 241)
            visit_656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), self_655, 'visit')
            # Calling visit(args, kwargs) (line 241)
            visit_call_result_659 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), visit_656, *[ex_657], **kwargs_658)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 't' (line 242)
        t_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 't')
        # Obtaining the member 'orelse' of a type (line 242)
        orelse_661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 11), t_660, 'orelse')
        # Testing if the type of an if condition is none (line 242)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 242, 8), orelse_661):
            pass
        else:
            
            # Testing the type of an if condition (line 242)
            if_condition_662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 242, 8), orelse_661)
            # Assigning a type to the variable 'if_condition_662' (line 242)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'if_condition_662', if_condition_662)
            # SSA begins for if statement (line 242)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fill(...): (line 243)
            # Processing the call arguments (line 243)
            str_665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 22), 'str', 'else')
            # Processing the call keyword arguments (line 243)
            kwargs_666 = {}
            # Getting the type of 'self' (line 243)
            self_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 243)
            fill_664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 243, 12), self_663, 'fill')
            # Calling fill(args, kwargs) (line 243)
            fill_call_result_667 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), fill_664, *[str_665], **kwargs_666)
            
            
            # Call to enter(...): (line 244)
            # Processing the call keyword arguments (line 244)
            kwargs_670 = {}
            # Getting the type of 'self' (line 244)
            self_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 244)
            enter_669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 12), self_668, 'enter')
            # Calling enter(args, kwargs) (line 244)
            enter_call_result_671 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), enter_669, *[], **kwargs_670)
            
            
            # Call to visit(...): (line 245)
            # Processing the call arguments (line 245)
            # Getting the type of 't' (line 245)
            t_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 23), 't', False)
            # Obtaining the member 'orelse' of a type (line 245)
            orelse_675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 23), t_674, 'orelse')
            # Processing the call keyword arguments (line 245)
            kwargs_676 = {}
            # Getting the type of 'self' (line 245)
            self_672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 245)
            visit_673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 12), self_672, 'visit')
            # Calling visit(args, kwargs) (line 245)
            visit_call_result_677 = invoke(stypy.reporting.localization.Localization(__file__, 245, 12), visit_673, *[orelse_675], **kwargs_676)
            
            
            # Call to leave(...): (line 246)
            # Processing the call keyword arguments (line 246)
            kwargs_680 = {}
            # Getting the type of 'self' (line 246)
            self_678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 246)
            leave_679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 12), self_678, 'leave')
            # Calling leave(args, kwargs) (line 246)
            leave_call_result_681 = invoke(stypy.reporting.localization.Localization(__file__, 246, 12), leave_679, *[], **kwargs_680)
            
            # SSA join for if statement (line 242)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_TryExcept(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_TryExcept' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_682)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_TryExcept'
        return stypy_return_type_682


    @norecursion
    def visit_TryFinally(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_TryFinally'
        module_type_store = module_type_store.open_function_context('visit_TryFinally', 248, 4, False)
        # Assigning a type to the variable 'self' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'self', type_of_self)
        
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
        
        
        # Call to len(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 't' (line 249)
        t_684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 15), 't', False)
        # Obtaining the member 'body' of a type (line 249)
        body_685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 15), t_684, 'body')
        # Processing the call keyword arguments (line 249)
        kwargs_686 = {}
        # Getting the type of 'len' (line 249)
        len_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 11), 'len', False)
        # Calling len(args, kwargs) (line 249)
        len_call_result_687 = invoke(stypy.reporting.localization.Localization(__file__, 249, 11), len_683, *[body_685], **kwargs_686)
        
        int_688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 26), 'int')
        # Applying the binary operator '==' (line 249)
        result_eq_689 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 11), '==', len_call_result_687, int_688)
        
        
        # Call to isinstance(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Obtaining the type of the subscript
        int_691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 50), 'int')
        # Getting the type of 't' (line 249)
        t_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 43), 't', False)
        # Obtaining the member 'body' of a type (line 249)
        body_693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 43), t_692, 'body')
        # Obtaining the member '__getitem__' of a type (line 249)
        getitem___694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 43), body_693, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 249)
        subscript_call_result_695 = invoke(stypy.reporting.localization.Localization(__file__, 249, 43), getitem___694, int_691)
        
        # Getting the type of 'ast' (line 249)
        ast_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 54), 'ast', False)
        # Obtaining the member 'TryExcept' of a type (line 249)
        TryExcept_697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 54), ast_696, 'TryExcept')
        # Processing the call keyword arguments (line 249)
        kwargs_698 = {}
        # Getting the type of 'isinstance' (line 249)
        isinstance_690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 32), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 249)
        isinstance_call_result_699 = invoke(stypy.reporting.localization.Localization(__file__, 249, 32), isinstance_690, *[subscript_call_result_695, TryExcept_697], **kwargs_698)
        
        # Applying the binary operator 'and' (line 249)
        result_and_keyword_700 = python_operator(stypy.reporting.localization.Localization(__file__, 249, 11), 'and', result_eq_689, isinstance_call_result_699)
        
        # Testing if the type of an if condition is none (line 249)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 249, 8), result_and_keyword_700):
            
            # Call to fill(...): (line 253)
            # Processing the call arguments (line 253)
            str_710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 22), 'str', 'try')
            # Processing the call keyword arguments (line 253)
            kwargs_711 = {}
            # Getting the type of 'self' (line 253)
            self_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 253)
            fill_709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 12), self_708, 'fill')
            # Calling fill(args, kwargs) (line 253)
            fill_call_result_712 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), fill_709, *[str_710], **kwargs_711)
            
            
            # Call to enter(...): (line 254)
            # Processing the call keyword arguments (line 254)
            kwargs_715 = {}
            # Getting the type of 'self' (line 254)
            self_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 254)
            enter_714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), self_713, 'enter')
            # Calling enter(args, kwargs) (line 254)
            enter_call_result_716 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), enter_714, *[], **kwargs_715)
            
            
            # Call to visit(...): (line 255)
            # Processing the call arguments (line 255)
            # Getting the type of 't' (line 255)
            t_719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 23), 't', False)
            # Obtaining the member 'body' of a type (line 255)
            body_720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 23), t_719, 'body')
            # Processing the call keyword arguments (line 255)
            kwargs_721 = {}
            # Getting the type of 'self' (line 255)
            self_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 255)
            visit_718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), self_717, 'visit')
            # Calling visit(args, kwargs) (line 255)
            visit_call_result_722 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), visit_718, *[body_720], **kwargs_721)
            
            
            # Call to leave(...): (line 256)
            # Processing the call keyword arguments (line 256)
            kwargs_725 = {}
            # Getting the type of 'self' (line 256)
            self_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 256)
            leave_724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), self_723, 'leave')
            # Calling leave(args, kwargs) (line 256)
            leave_call_result_726 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), leave_724, *[], **kwargs_725)
            
        else:
            
            # Testing the type of an if condition (line 249)
            if_condition_701 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 8), result_and_keyword_700)
            # Assigning a type to the variable 'if_condition_701' (line 249)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'if_condition_701', if_condition_701)
            # SSA begins for if statement (line 249)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to visit(...): (line 251)
            # Processing the call arguments (line 251)
            # Getting the type of 't' (line 251)
            t_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 't', False)
            # Obtaining the member 'body' of a type (line 251)
            body_705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 23), t_704, 'body')
            # Processing the call keyword arguments (line 251)
            kwargs_706 = {}
            # Getting the type of 'self' (line 251)
            self_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 251)
            visit_703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 12), self_702, 'visit')
            # Calling visit(args, kwargs) (line 251)
            visit_call_result_707 = invoke(stypy.reporting.localization.Localization(__file__, 251, 12), visit_703, *[body_705], **kwargs_706)
            
            # SSA branch for the else part of an if statement (line 249)
            module_type_store.open_ssa_branch('else')
            
            # Call to fill(...): (line 253)
            # Processing the call arguments (line 253)
            str_710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 22), 'str', 'try')
            # Processing the call keyword arguments (line 253)
            kwargs_711 = {}
            # Getting the type of 'self' (line 253)
            self_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 253)
            fill_709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 12), self_708, 'fill')
            # Calling fill(args, kwargs) (line 253)
            fill_call_result_712 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), fill_709, *[str_710], **kwargs_711)
            
            
            # Call to enter(...): (line 254)
            # Processing the call keyword arguments (line 254)
            kwargs_715 = {}
            # Getting the type of 'self' (line 254)
            self_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 254)
            enter_714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 12), self_713, 'enter')
            # Calling enter(args, kwargs) (line 254)
            enter_call_result_716 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), enter_714, *[], **kwargs_715)
            
            
            # Call to visit(...): (line 255)
            # Processing the call arguments (line 255)
            # Getting the type of 't' (line 255)
            t_719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 23), 't', False)
            # Obtaining the member 'body' of a type (line 255)
            body_720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 23), t_719, 'body')
            # Processing the call keyword arguments (line 255)
            kwargs_721 = {}
            # Getting the type of 'self' (line 255)
            self_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 255)
            visit_718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), self_717, 'visit')
            # Calling visit(args, kwargs) (line 255)
            visit_call_result_722 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), visit_718, *[body_720], **kwargs_721)
            
            
            # Call to leave(...): (line 256)
            # Processing the call keyword arguments (line 256)
            kwargs_725 = {}
            # Getting the type of 'self' (line 256)
            self_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 256)
            leave_724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), self_723, 'leave')
            # Calling leave(args, kwargs) (line 256)
            leave_call_result_726 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), leave_724, *[], **kwargs_725)
            
            # SSA join for if statement (line 249)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to fill(...): (line 258)
        # Processing the call arguments (line 258)
        str_729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 18), 'str', 'finally')
        # Processing the call keyword arguments (line 258)
        kwargs_730 = {}
        # Getting the type of 'self' (line 258)
        self_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 258)
        fill_728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), self_727, 'fill')
        # Calling fill(args, kwargs) (line 258)
        fill_call_result_731 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), fill_728, *[str_729], **kwargs_730)
        
        
        # Call to enter(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_734 = {}
        # Getting the type of 'self' (line 259)
        self_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 259)
        enter_733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_732, 'enter')
        # Calling enter(args, kwargs) (line 259)
        enter_call_result_735 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), enter_733, *[], **kwargs_734)
        
        
        # Call to visit(...): (line 260)
        # Processing the call arguments (line 260)
        # Getting the type of 't' (line 260)
        t_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 19), 't', False)
        # Obtaining the member 'finalbody' of a type (line 260)
        finalbody_739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 19), t_738, 'finalbody')
        # Processing the call keyword arguments (line 260)
        kwargs_740 = {}
        # Getting the type of 'self' (line 260)
        self_736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 260)
        visit_737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 8), self_736, 'visit')
        # Calling visit(args, kwargs) (line 260)
        visit_call_result_741 = invoke(stypy.reporting.localization.Localization(__file__, 260, 8), visit_737, *[finalbody_739], **kwargs_740)
        
        
        # Call to leave(...): (line 261)
        # Processing the call keyword arguments (line 261)
        kwargs_744 = {}
        # Getting the type of 'self' (line 261)
        self_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 261)
        leave_743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), self_742, 'leave')
        # Calling leave(args, kwargs) (line 261)
        leave_call_result_745 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), leave_743, *[], **kwargs_744)
        
        
        # ################# End of 'visit_TryFinally(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_TryFinally' in the type store
        # Getting the type of 'stypy_return_type' (line 248)
        stypy_return_type_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_746)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_TryFinally'
        return stypy_return_type_746


    @norecursion
    def visit_ExceptHandler(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ExceptHandler'
        module_type_store = module_type_store.open_function_context('visit_ExceptHandler', 263, 4, False)
        # Assigning a type to the variable 'self' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 264)
        # Processing the call arguments (line 264)
        str_749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 18), 'str', 'except')
        # Processing the call keyword arguments (line 264)
        kwargs_750 = {}
        # Getting the type of 'self' (line 264)
        self_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 264)
        fill_748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 8), self_747, 'fill')
        # Calling fill(args, kwargs) (line 264)
        fill_call_result_751 = invoke(stypy.reporting.localization.Localization(__file__, 264, 8), fill_748, *[str_749], **kwargs_750)
        
        # Getting the type of 't' (line 265)
        t_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 't')
        # Obtaining the member 'type' of a type (line 265)
        type_753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 11), t_752, 'type')
        # Testing if the type of an if condition is none (line 265)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 265, 8), type_753):
            pass
        else:
            
            # Testing the type of an if condition (line 265)
            if_condition_754 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 8), type_753)
            # Assigning a type to the variable 'if_condition_754' (line 265)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'if_condition_754', if_condition_754)
            # SSA begins for if statement (line 265)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 266)
            # Processing the call arguments (line 266)
            str_757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 23), 'str', ' ')
            # Processing the call keyword arguments (line 266)
            kwargs_758 = {}
            # Getting the type of 'self' (line 266)
            self_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 266)
            write_756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), self_755, 'write')
            # Calling write(args, kwargs) (line 266)
            write_call_result_759 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), write_756, *[str_757], **kwargs_758)
            
            
            # Call to visit(...): (line 267)
            # Processing the call arguments (line 267)
            # Getting the type of 't' (line 267)
            t_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 23), 't', False)
            # Obtaining the member 'type' of a type (line 267)
            type_763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 23), t_762, 'type')
            # Processing the call keyword arguments (line 267)
            kwargs_764 = {}
            # Getting the type of 'self' (line 267)
            self_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 267)
            visit_761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 12), self_760, 'visit')
            # Calling visit(args, kwargs) (line 267)
            visit_call_result_765 = invoke(stypy.reporting.localization.Localization(__file__, 267, 12), visit_761, *[type_763], **kwargs_764)
            
            # SSA join for if statement (line 265)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 268)
        t_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 11), 't')
        # Obtaining the member 'name' of a type (line 268)
        name_767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 11), t_766, 'name')
        # Testing if the type of an if condition is none (line 268)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 268, 8), name_767):
            pass
        else:
            
            # Testing the type of an if condition (line 268)
            if_condition_768 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 8), name_767)
            # Assigning a type to the variable 'if_condition_768' (line 268)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'if_condition_768', if_condition_768)
            # SSA begins for if statement (line 268)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 269)
            # Processing the call arguments (line 269)
            str_771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 269, 23), 'str', ' as ')
            # Processing the call keyword arguments (line 269)
            kwargs_772 = {}
            # Getting the type of 'self' (line 269)
            self_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 269)
            write_770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 12), self_769, 'write')
            # Calling write(args, kwargs) (line 269)
            write_call_result_773 = invoke(stypy.reporting.localization.Localization(__file__, 269, 12), write_770, *[str_771], **kwargs_772)
            
            
            # Call to visit(...): (line 270)
            # Processing the call arguments (line 270)
            # Getting the type of 't' (line 270)
            t_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 23), 't', False)
            # Obtaining the member 'name' of a type (line 270)
            name_777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 23), t_776, 'name')
            # Processing the call keyword arguments (line 270)
            kwargs_778 = {}
            # Getting the type of 'self' (line 270)
            self_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 270)
            visit_775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 12), self_774, 'visit')
            # Calling visit(args, kwargs) (line 270)
            visit_call_result_779 = invoke(stypy.reporting.localization.Localization(__file__, 270, 12), visit_775, *[name_777], **kwargs_778)
            
            # SSA join for if statement (line 268)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to enter(...): (line 271)
        # Processing the call keyword arguments (line 271)
        kwargs_782 = {}
        # Getting the type of 'self' (line 271)
        self_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 271)
        enter_781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), self_780, 'enter')
        # Calling enter(args, kwargs) (line 271)
        enter_call_result_783 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), enter_781, *[], **kwargs_782)
        
        
        # Call to visit(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 't' (line 272)
        t_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 19), 't', False)
        # Obtaining the member 'body' of a type (line 272)
        body_787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 19), t_786, 'body')
        # Processing the call keyword arguments (line 272)
        kwargs_788 = {}
        # Getting the type of 'self' (line 272)
        self_784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 272)
        visit_785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_784, 'visit')
        # Calling visit(args, kwargs) (line 272)
        visit_call_result_789 = invoke(stypy.reporting.localization.Localization(__file__, 272, 8), visit_785, *[body_787], **kwargs_788)
        
        
        # Call to leave(...): (line 273)
        # Processing the call keyword arguments (line 273)
        kwargs_792 = {}
        # Getting the type of 'self' (line 273)
        self_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 273)
        leave_791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), self_790, 'leave')
        # Calling leave(args, kwargs) (line 273)
        leave_call_result_793 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), leave_791, *[], **kwargs_792)
        
        
        # ################# End of 'visit_ExceptHandler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ExceptHandler' in the type store
        # Getting the type of 'stypy_return_type' (line 263)
        stypy_return_type_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_794)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ExceptHandler'
        return stypy_return_type_794


    @norecursion
    def visit_ClassDef(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ClassDef'
        module_type_store = module_type_store.open_function_context('visit_ClassDef', 275, 4, False)
        # Assigning a type to the variable 'self' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 276)
        # Processing the call arguments (line 276)
        str_797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 19), 'str', '\n')
        # Processing the call keyword arguments (line 276)
        kwargs_798 = {}
        # Getting the type of 'self' (line 276)
        self_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 276)
        write_796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 8), self_795, 'write')
        # Calling write(args, kwargs) (line 276)
        write_call_result_799 = invoke(stypy.reporting.localization.Localization(__file__, 276, 8), write_796, *[str_797], **kwargs_798)
        
        
        # Getting the type of 't' (line 277)
        t_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 't')
        # Obtaining the member 'decorator_list' of a type (line 277)
        decorator_list_801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 277, 20), t_800, 'decorator_list')
        # Assigning a type to the variable 'decorator_list_801' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'decorator_list_801', decorator_list_801)
        # Testing if the for loop is going to be iterated (line 277)
        # Testing the type of a for loop iterable (line 277)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 277, 8), decorator_list_801)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 277, 8), decorator_list_801):
            # Getting the type of the for loop variable (line 277)
            for_loop_var_802 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 277, 8), decorator_list_801)
            # Assigning a type to the variable 'deco' (line 277)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 8), 'deco', for_loop_var_802)
            # SSA begins for a for statement (line 277)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to fill(...): (line 278)
            # Processing the call arguments (line 278)
            str_805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 22), 'str', '@')
            # Processing the call keyword arguments (line 278)
            kwargs_806 = {}
            # Getting the type of 'self' (line 278)
            self_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 278)
            fill_804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 12), self_803, 'fill')
            # Calling fill(args, kwargs) (line 278)
            fill_call_result_807 = invoke(stypy.reporting.localization.Localization(__file__, 278, 12), fill_804, *[str_805], **kwargs_806)
            
            
            # Call to visit(...): (line 279)
            # Processing the call arguments (line 279)
            # Getting the type of 'deco' (line 279)
            deco_810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 23), 'deco', False)
            # Processing the call keyword arguments (line 279)
            kwargs_811 = {}
            # Getting the type of 'self' (line 279)
            self_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 279)
            visit_809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), self_808, 'visit')
            # Calling visit(args, kwargs) (line 279)
            visit_call_result_812 = invoke(stypy.reporting.localization.Localization(__file__, 279, 12), visit_809, *[deco_810], **kwargs_811)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to fill(...): (line 280)
        # Processing the call arguments (line 280)
        str_815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 18), 'str', 'class ')
        # Getting the type of 't' (line 280)
        t_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 29), 't', False)
        # Obtaining the member 'name' of a type (line 280)
        name_817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 29), t_816, 'name')
        # Applying the binary operator '+' (line 280)
        result_add_818 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 18), '+', str_815, name_817)
        
        # Processing the call keyword arguments (line 280)
        kwargs_819 = {}
        # Getting the type of 'self' (line 280)
        self_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 280)
        fill_814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 8), self_813, 'fill')
        # Calling fill(args, kwargs) (line 280)
        fill_call_result_820 = invoke(stypy.reporting.localization.Localization(__file__, 280, 8), fill_814, *[result_add_818], **kwargs_819)
        
        # Getting the type of 't' (line 281)
        t_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 11), 't')
        # Obtaining the member 'bases' of a type (line 281)
        bases_822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 11), t_821, 'bases')
        # Testing if the type of an if condition is none (line 281)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 281, 8), bases_822):
            pass
        else:
            
            # Testing the type of an if condition (line 281)
            if_condition_823 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 8), bases_822)
            # Assigning a type to the variable 'if_condition_823' (line 281)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'if_condition_823', if_condition_823)
            # SSA begins for if statement (line 281)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 282)
            # Processing the call arguments (line 282)
            str_826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 23), 'str', '(')
            # Processing the call keyword arguments (line 282)
            kwargs_827 = {}
            # Getting the type of 'self' (line 282)
            self_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 282)
            write_825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 12), self_824, 'write')
            # Calling write(args, kwargs) (line 282)
            write_call_result_828 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), write_825, *[str_826], **kwargs_827)
            
            
            # Getting the type of 't' (line 283)
            t_829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 21), 't')
            # Obtaining the member 'bases' of a type (line 283)
            bases_830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 21), t_829, 'bases')
            # Assigning a type to the variable 'bases_830' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'bases_830', bases_830)
            # Testing if the for loop is going to be iterated (line 283)
            # Testing the type of a for loop iterable (line 283)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 283, 12), bases_830)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 283, 12), bases_830):
                # Getting the type of the for loop variable (line 283)
                for_loop_var_831 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 283, 12), bases_830)
                # Assigning a type to the variable 'a' (line 283)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 12), 'a', for_loop_var_831)
                # SSA begins for a for statement (line 283)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to visit(...): (line 284)
                # Processing the call arguments (line 284)
                # Getting the type of 'a' (line 284)
                a_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 27), 'a', False)
                # Processing the call keyword arguments (line 284)
                kwargs_835 = {}
                # Getting the type of 'self' (line 284)
                self_832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'self', False)
                # Obtaining the member 'visit' of a type (line 284)
                visit_833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 16), self_832, 'visit')
                # Calling visit(args, kwargs) (line 284)
                visit_call_result_836 = invoke(stypy.reporting.localization.Localization(__file__, 284, 16), visit_833, *[a_834], **kwargs_835)
                
                
                # Call to write(...): (line 285)
                # Processing the call arguments (line 285)
                str_839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 27), 'str', ', ')
                # Processing the call keyword arguments (line 285)
                kwargs_840 = {}
                # Getting the type of 'self' (line 285)
                self_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 285)
                write_838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 16), self_837, 'write')
                # Calling write(args, kwargs) (line 285)
                write_call_result_841 = invoke(stypy.reporting.localization.Localization(__file__, 285, 16), write_838, *[str_839], **kwargs_840)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to write(...): (line 286)
            # Processing the call arguments (line 286)
            str_844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 23), 'str', ')')
            # Processing the call keyword arguments (line 286)
            kwargs_845 = {}
            # Getting the type of 'self' (line 286)
            self_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 286)
            write_843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 12), self_842, 'write')
            # Calling write(args, kwargs) (line 286)
            write_call_result_846 = invoke(stypy.reporting.localization.Localization(__file__, 286, 12), write_843, *[str_844], **kwargs_845)
            
            # SSA join for if statement (line 281)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to enter(...): (line 287)
        # Processing the call keyword arguments (line 287)
        kwargs_849 = {}
        # Getting the type of 'self' (line 287)
        self_847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 287)
        enter_848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_847, 'enter')
        # Calling enter(args, kwargs) (line 287)
        enter_call_result_850 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), enter_848, *[], **kwargs_849)
        
        
        # Call to visit(...): (line 288)
        # Processing the call arguments (line 288)
        # Getting the type of 't' (line 288)
        t_853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 19), 't', False)
        # Obtaining the member 'body' of a type (line 288)
        body_854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 19), t_853, 'body')
        # Processing the call keyword arguments (line 288)
        kwargs_855 = {}
        # Getting the type of 'self' (line 288)
        self_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 288)
        visit_852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), self_851, 'visit')
        # Calling visit(args, kwargs) (line 288)
        visit_call_result_856 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), visit_852, *[body_854], **kwargs_855)
        
        
        # Call to leave(...): (line 289)
        # Processing the call keyword arguments (line 289)
        kwargs_859 = {}
        # Getting the type of 'self' (line 289)
        self_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 289)
        leave_858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 8), self_857, 'leave')
        # Calling leave(args, kwargs) (line 289)
        leave_call_result_860 = invoke(stypy.reporting.localization.Localization(__file__, 289, 8), leave_858, *[], **kwargs_859)
        
        
        # ################# End of 'visit_ClassDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ClassDef' in the type store
        # Getting the type of 'stypy_return_type' (line 275)
        stypy_return_type_861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_861)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ClassDef'
        return stypy_return_type_861


    @norecursion
    def visit_FunctionDef(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_FunctionDef'
        module_type_store = module_type_store.open_function_context('visit_FunctionDef', 291, 4, False)
        # Assigning a type to the variable 'self' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 292)
        # Processing the call arguments (line 292)
        str_864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 19), 'str', '\n')
        # Processing the call keyword arguments (line 292)
        kwargs_865 = {}
        # Getting the type of 'self' (line 292)
        self_862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 292)
        write_863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), self_862, 'write')
        # Calling write(args, kwargs) (line 292)
        write_call_result_866 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), write_863, *[str_864], **kwargs_865)
        
        
        # Getting the type of 't' (line 293)
        t_867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 20), 't')
        # Obtaining the member 'decorator_list' of a type (line 293)
        decorator_list_868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 20), t_867, 'decorator_list')
        # Assigning a type to the variable 'decorator_list_868' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'decorator_list_868', decorator_list_868)
        # Testing if the for loop is going to be iterated (line 293)
        # Testing the type of a for loop iterable (line 293)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 293, 8), decorator_list_868)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 293, 8), decorator_list_868):
            # Getting the type of the for loop variable (line 293)
            for_loop_var_869 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 293, 8), decorator_list_868)
            # Assigning a type to the variable 'deco' (line 293)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'deco', for_loop_var_869)
            # SSA begins for a for statement (line 293)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to fill(...): (line 294)
            # Processing the call arguments (line 294)
            str_872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 22), 'str', '@')
            # Processing the call keyword arguments (line 294)
            kwargs_873 = {}
            # Getting the type of 'self' (line 294)
            self_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 294)
            fill_871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 12), self_870, 'fill')
            # Calling fill(args, kwargs) (line 294)
            fill_call_result_874 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), fill_871, *[str_872], **kwargs_873)
            
            
            # Call to visit(...): (line 295)
            # Processing the call arguments (line 295)
            # Getting the type of 'deco' (line 295)
            deco_877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 23), 'deco', False)
            # Processing the call keyword arguments (line 295)
            kwargs_878 = {}
            # Getting the type of 'self' (line 295)
            self_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 295)
            visit_876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 12), self_875, 'visit')
            # Calling visit(args, kwargs) (line 295)
            visit_call_result_879 = invoke(stypy.reporting.localization.Localization(__file__, 295, 12), visit_876, *[deco_877], **kwargs_878)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to fill(...): (line 296)
        # Processing the call arguments (line 296)
        str_882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 18), 'str', 'def ')
        # Getting the type of 't' (line 296)
        t_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 27), 't', False)
        # Obtaining the member 'name' of a type (line 296)
        name_884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 27), t_883, 'name')
        # Applying the binary operator '+' (line 296)
        result_add_885 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 18), '+', str_882, name_884)
        
        str_886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 36), 'str', '(')
        # Applying the binary operator '+' (line 296)
        result_add_887 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 34), '+', result_add_885, str_886)
        
        # Processing the call keyword arguments (line 296)
        kwargs_888 = {}
        # Getting the type of 'self' (line 296)
        self_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 296)
        fill_881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), self_880, 'fill')
        # Calling fill(args, kwargs) (line 296)
        fill_call_result_889 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), fill_881, *[result_add_887], **kwargs_888)
        
        
        # Call to visit(...): (line 297)
        # Processing the call arguments (line 297)
        # Getting the type of 't' (line 297)
        t_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 19), 't', False)
        # Obtaining the member 'args' of a type (line 297)
        args_893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 19), t_892, 'args')
        # Processing the call keyword arguments (line 297)
        kwargs_894 = {}
        # Getting the type of 'self' (line 297)
        self_890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 297)
        visit_891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), self_890, 'visit')
        # Calling visit(args, kwargs) (line 297)
        visit_call_result_895 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), visit_891, *[args_893], **kwargs_894)
        
        
        # Call to write(...): (line 298)
        # Processing the call arguments (line 298)
        str_898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 19), 'str', ')')
        # Processing the call keyword arguments (line 298)
        kwargs_899 = {}
        # Getting the type of 'self' (line 298)
        self_896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 298)
        write_897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 8), self_896, 'write')
        # Calling write(args, kwargs) (line 298)
        write_call_result_900 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), write_897, *[str_898], **kwargs_899)
        
        
        # Call to enter(...): (line 299)
        # Processing the call keyword arguments (line 299)
        kwargs_903 = {}
        # Getting the type of 'self' (line 299)
        self_901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 299)
        enter_902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 8), self_901, 'enter')
        # Calling enter(args, kwargs) (line 299)
        enter_call_result_904 = invoke(stypy.reporting.localization.Localization(__file__, 299, 8), enter_902, *[], **kwargs_903)
        
        
        # Call to visit(...): (line 300)
        # Processing the call arguments (line 300)
        # Getting the type of 't' (line 300)
        t_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 19), 't', False)
        # Obtaining the member 'body' of a type (line 300)
        body_908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 19), t_907, 'body')
        # Processing the call keyword arguments (line 300)
        kwargs_909 = {}
        # Getting the type of 'self' (line 300)
        self_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 300)
        visit_906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 8), self_905, 'visit')
        # Calling visit(args, kwargs) (line 300)
        visit_call_result_910 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), visit_906, *[body_908], **kwargs_909)
        
        
        # Call to write(...): (line 301)
        # Processing the call arguments (line 301)
        str_913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 19), 'str', '\n')
        # Processing the call keyword arguments (line 301)
        kwargs_914 = {}
        # Getting the type of 'self' (line 301)
        self_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 301)
        write_912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 8), self_911, 'write')
        # Calling write(args, kwargs) (line 301)
        write_call_result_915 = invoke(stypy.reporting.localization.Localization(__file__, 301, 8), write_912, *[str_913], **kwargs_914)
        
        
        # Call to leave(...): (line 302)
        # Processing the call keyword arguments (line 302)
        kwargs_918 = {}
        # Getting the type of 'self' (line 302)
        self_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 302)
        leave_917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 8), self_916, 'leave')
        # Calling leave(args, kwargs) (line 302)
        leave_call_result_919 = invoke(stypy.reporting.localization.Localization(__file__, 302, 8), leave_917, *[], **kwargs_918)
        
        
        # ################# End of 'visit_FunctionDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_FunctionDef' in the type store
        # Getting the type of 'stypy_return_type' (line 291)
        stypy_return_type_920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_920)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_FunctionDef'
        return stypy_return_type_920


    @norecursion
    def visit_For(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_For'
        module_type_store = module_type_store.open_function_context('visit_For', 304, 4, False)
        # Assigning a type to the variable 'self' (line 305)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 305)
        # Processing the call arguments (line 305)
        str_923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 18), 'str', 'for ')
        # Processing the call keyword arguments (line 305)
        kwargs_924 = {}
        # Getting the type of 'self' (line 305)
        self_921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 305)
        fill_922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 8), self_921, 'fill')
        # Calling fill(args, kwargs) (line 305)
        fill_call_result_925 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), fill_922, *[str_923], **kwargs_924)
        
        
        # Call to visit(...): (line 306)
        # Processing the call arguments (line 306)
        # Getting the type of 't' (line 306)
        t_928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 't', False)
        # Obtaining the member 'target' of a type (line 306)
        target_929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 19), t_928, 'target')
        # Processing the call keyword arguments (line 306)
        kwargs_930 = {}
        # Getting the type of 'self' (line 306)
        self_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 306)
        visit_927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), self_926, 'visit')
        # Calling visit(args, kwargs) (line 306)
        visit_call_result_931 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), visit_927, *[target_929], **kwargs_930)
        
        
        # Call to write(...): (line 307)
        # Processing the call arguments (line 307)
        str_934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 19), 'str', ' in ')
        # Processing the call keyword arguments (line 307)
        kwargs_935 = {}
        # Getting the type of 'self' (line 307)
        self_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 307)
        write_933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), self_932, 'write')
        # Calling write(args, kwargs) (line 307)
        write_call_result_936 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), write_933, *[str_934], **kwargs_935)
        
        
        # Call to visit(...): (line 308)
        # Processing the call arguments (line 308)
        # Getting the type of 't' (line 308)
        t_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 19), 't', False)
        # Obtaining the member 'iter' of a type (line 308)
        iter_940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 19), t_939, 'iter')
        # Processing the call keyword arguments (line 308)
        kwargs_941 = {}
        # Getting the type of 'self' (line 308)
        self_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 308)
        visit_938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), self_937, 'visit')
        # Calling visit(args, kwargs) (line 308)
        visit_call_result_942 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), visit_938, *[iter_940], **kwargs_941)
        
        
        # Call to enter(...): (line 309)
        # Processing the call keyword arguments (line 309)
        kwargs_945 = {}
        # Getting the type of 'self' (line 309)
        self_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 309)
        enter_944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), self_943, 'enter')
        # Calling enter(args, kwargs) (line 309)
        enter_call_result_946 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), enter_944, *[], **kwargs_945)
        
        
        # Call to visit(...): (line 310)
        # Processing the call arguments (line 310)
        # Getting the type of 't' (line 310)
        t_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 19), 't', False)
        # Obtaining the member 'body' of a type (line 310)
        body_950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 19), t_949, 'body')
        # Processing the call keyword arguments (line 310)
        kwargs_951 = {}
        # Getting the type of 'self' (line 310)
        self_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 310)
        visit_948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), self_947, 'visit')
        # Calling visit(args, kwargs) (line 310)
        visit_call_result_952 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), visit_948, *[body_950], **kwargs_951)
        
        
        # Call to leave(...): (line 311)
        # Processing the call keyword arguments (line 311)
        kwargs_955 = {}
        # Getting the type of 'self' (line 311)
        self_953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 311)
        leave_954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), self_953, 'leave')
        # Calling leave(args, kwargs) (line 311)
        leave_call_result_956 = invoke(stypy.reporting.localization.Localization(__file__, 311, 8), leave_954, *[], **kwargs_955)
        
        # Getting the type of 't' (line 312)
        t_957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), 't')
        # Obtaining the member 'orelse' of a type (line 312)
        orelse_958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 11), t_957, 'orelse')
        # Testing if the type of an if condition is none (line 312)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 312, 8), orelse_958):
            pass
        else:
            
            # Testing the type of an if condition (line 312)
            if_condition_959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 8), orelse_958)
            # Assigning a type to the variable 'if_condition_959' (line 312)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'if_condition_959', if_condition_959)
            # SSA begins for if statement (line 312)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fill(...): (line 313)
            # Processing the call arguments (line 313)
            str_962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 22), 'str', 'else')
            # Processing the call keyword arguments (line 313)
            kwargs_963 = {}
            # Getting the type of 'self' (line 313)
            self_960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 313)
            fill_961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 12), self_960, 'fill')
            # Calling fill(args, kwargs) (line 313)
            fill_call_result_964 = invoke(stypy.reporting.localization.Localization(__file__, 313, 12), fill_961, *[str_962], **kwargs_963)
            
            
            # Call to enter(...): (line 314)
            # Processing the call keyword arguments (line 314)
            kwargs_967 = {}
            # Getting the type of 'self' (line 314)
            self_965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 314)
            enter_966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 12), self_965, 'enter')
            # Calling enter(args, kwargs) (line 314)
            enter_call_result_968 = invoke(stypy.reporting.localization.Localization(__file__, 314, 12), enter_966, *[], **kwargs_967)
            
            
            # Call to visit(...): (line 315)
            # Processing the call arguments (line 315)
            # Getting the type of 't' (line 315)
            t_971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 23), 't', False)
            # Obtaining the member 'orelse' of a type (line 315)
            orelse_972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 23), t_971, 'orelse')
            # Processing the call keyword arguments (line 315)
            kwargs_973 = {}
            # Getting the type of 'self' (line 315)
            self_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 315)
            visit_970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 12), self_969, 'visit')
            # Calling visit(args, kwargs) (line 315)
            visit_call_result_974 = invoke(stypy.reporting.localization.Localization(__file__, 315, 12), visit_970, *[orelse_972], **kwargs_973)
            
            
            # Call to leave(...): (line 316)
            # Processing the call keyword arguments (line 316)
            kwargs_977 = {}
            # Getting the type of 'self' (line 316)
            self_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 316)
            leave_976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 12), self_975, 'leave')
            # Calling leave(args, kwargs) (line 316)
            leave_call_result_978 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), leave_976, *[], **kwargs_977)
            
            # SSA join for if statement (line 312)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_For(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_For' in the type store
        # Getting the type of 'stypy_return_type' (line 304)
        stypy_return_type_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_979)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_For'
        return stypy_return_type_979


    @norecursion
    def visit_If(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_If'
        module_type_store = module_type_store.open_function_context('visit_If', 318, 4, False)
        # Assigning a type to the variable 'self' (line 319)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 319)
        # Processing the call arguments (line 319)
        str_982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 19), 'str', '\n')
        # Processing the call keyword arguments (line 319)
        kwargs_983 = {}
        # Getting the type of 'self' (line 319)
        self_980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 319)
        write_981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 8), self_980, 'write')
        # Calling write(args, kwargs) (line 319)
        write_call_result_984 = invoke(stypy.reporting.localization.Localization(__file__, 319, 8), write_981, *[str_982], **kwargs_983)
        
        
        # Call to fill(...): (line 320)
        # Processing the call arguments (line 320)
        str_987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 18), 'str', 'if ')
        # Processing the call keyword arguments (line 320)
        kwargs_988 = {}
        # Getting the type of 'self' (line 320)
        self_985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 320)
        fill_986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), self_985, 'fill')
        # Calling fill(args, kwargs) (line 320)
        fill_call_result_989 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), fill_986, *[str_987], **kwargs_988)
        
        
        # Call to visit(...): (line 321)
        # Processing the call arguments (line 321)
        # Getting the type of 't' (line 321)
        t_992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 19), 't', False)
        # Obtaining the member 'test' of a type (line 321)
        test_993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 19), t_992, 'test')
        # Processing the call keyword arguments (line 321)
        kwargs_994 = {}
        # Getting the type of 'self' (line 321)
        self_990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 321)
        visit_991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 8), self_990, 'visit')
        # Calling visit(args, kwargs) (line 321)
        visit_call_result_995 = invoke(stypy.reporting.localization.Localization(__file__, 321, 8), visit_991, *[test_993], **kwargs_994)
        
        
        # Call to enter(...): (line 322)
        # Processing the call keyword arguments (line 322)
        kwargs_998 = {}
        # Getting the type of 'self' (line 322)
        self_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 322)
        enter_997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 8), self_996, 'enter')
        # Calling enter(args, kwargs) (line 322)
        enter_call_result_999 = invoke(stypy.reporting.localization.Localization(__file__, 322, 8), enter_997, *[], **kwargs_998)
        
        
        # Call to visit(...): (line 323)
        # Processing the call arguments (line 323)
        # Getting the type of 't' (line 323)
        t_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 19), 't', False)
        # Obtaining the member 'body' of a type (line 323)
        body_1003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 19), t_1002, 'body')
        # Processing the call keyword arguments (line 323)
        kwargs_1004 = {}
        # Getting the type of 'self' (line 323)
        self_1000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 323)
        visit_1001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 8), self_1000, 'visit')
        # Calling visit(args, kwargs) (line 323)
        visit_call_result_1005 = invoke(stypy.reporting.localization.Localization(__file__, 323, 8), visit_1001, *[body_1003], **kwargs_1004)
        
        
        # Call to leave(...): (line 324)
        # Processing the call keyword arguments (line 324)
        kwargs_1008 = {}
        # Getting the type of 'self' (line 324)
        self_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 324)
        leave_1007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 8), self_1006, 'leave')
        # Calling leave(args, kwargs) (line 324)
        leave_call_result_1009 = invoke(stypy.reporting.localization.Localization(__file__, 324, 8), leave_1007, *[], **kwargs_1008)
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 't' (line 326)
        t_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 15), 't')
        # Obtaining the member 'orelse' of a type (line 326)
        orelse_1011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 15), t_1010, 'orelse')
        
        
        # Call to len(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 't' (line 326)
        t_1013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 32), 't', False)
        # Obtaining the member 'orelse' of a type (line 326)
        orelse_1014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 32), t_1013, 'orelse')
        # Processing the call keyword arguments (line 326)
        kwargs_1015 = {}
        # Getting the type of 'len' (line 326)
        len_1012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 28), 'len', False)
        # Calling len(args, kwargs) (line 326)
        len_call_result_1016 = invoke(stypy.reporting.localization.Localization(__file__, 326, 28), len_1012, *[orelse_1014], **kwargs_1015)
        
        int_1017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 45), 'int')
        # Applying the binary operator '==' (line 326)
        result_eq_1018 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 28), '==', len_call_result_1016, int_1017)
        
        # Applying the binary operator 'and' (line 326)
        result_and_keyword_1019 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 15), 'and', orelse_1011, result_eq_1018)
        
        # Call to isinstance(...): (line 327)
        # Processing the call arguments (line 327)
        
        # Obtaining the type of the subscript
        int_1021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 39), 'int')
        # Getting the type of 't' (line 327)
        t_1022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 30), 't', False)
        # Obtaining the member 'orelse' of a type (line 327)
        orelse_1023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 30), t_1022, 'orelse')
        # Obtaining the member '__getitem__' of a type (line 327)
        getitem___1024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 30), orelse_1023, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 327)
        subscript_call_result_1025 = invoke(stypy.reporting.localization.Localization(__file__, 327, 30), getitem___1024, int_1021)
        
        # Getting the type of 'ast' (line 327)
        ast_1026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 43), 'ast', False)
        # Obtaining the member 'If' of a type (line 327)
        If_1027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 43), ast_1026, 'If')
        # Processing the call keyword arguments (line 327)
        kwargs_1028 = {}
        # Getting the type of 'isinstance' (line 327)
        isinstance_1020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 327)
        isinstance_call_result_1029 = invoke(stypy.reporting.localization.Localization(__file__, 327, 19), isinstance_1020, *[subscript_call_result_1025, If_1027], **kwargs_1028)
        
        # Applying the binary operator 'and' (line 326)
        result_and_keyword_1030 = python_operator(stypy.reporting.localization.Localization(__file__, 326, 15), 'and', result_and_keyword_1019, isinstance_call_result_1029)
        
        # Assigning a type to the variable 'result_and_keyword_1030' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'result_and_keyword_1030', result_and_keyword_1030)
        # Testing if the while is going to be iterated (line 326)
        # Testing the type of an if condition (line 326)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 326, 8), result_and_keyword_1030)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 326, 8), result_and_keyword_1030):
            # SSA begins for while statement (line 326)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Subscript to a Name (line 328):
            
            # Assigning a Subscript to a Name (line 328):
            
            # Obtaining the type of the subscript
            int_1031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 25), 'int')
            # Getting the type of 't' (line 328)
            t_1032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 16), 't')
            # Obtaining the member 'orelse' of a type (line 328)
            orelse_1033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 16), t_1032, 'orelse')
            # Obtaining the member '__getitem__' of a type (line 328)
            getitem___1034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 16), orelse_1033, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 328)
            subscript_call_result_1035 = invoke(stypy.reporting.localization.Localization(__file__, 328, 16), getitem___1034, int_1031)
            
            # Assigning a type to the variable 't' (line 328)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 12), 't', subscript_call_result_1035)
            
            # Call to fill(...): (line 329)
            # Processing the call arguments (line 329)
            str_1038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 22), 'str', 'elif ')
            # Processing the call keyword arguments (line 329)
            kwargs_1039 = {}
            # Getting the type of 'self' (line 329)
            self_1036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 329)
            fill_1037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 12), self_1036, 'fill')
            # Calling fill(args, kwargs) (line 329)
            fill_call_result_1040 = invoke(stypy.reporting.localization.Localization(__file__, 329, 12), fill_1037, *[str_1038], **kwargs_1039)
            
            
            # Call to visit(...): (line 330)
            # Processing the call arguments (line 330)
            # Getting the type of 't' (line 330)
            t_1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 23), 't', False)
            # Obtaining the member 'test' of a type (line 330)
            test_1044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 23), t_1043, 'test')
            # Processing the call keyword arguments (line 330)
            kwargs_1045 = {}
            # Getting the type of 'self' (line 330)
            self_1041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 330)
            visit_1042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 12), self_1041, 'visit')
            # Calling visit(args, kwargs) (line 330)
            visit_call_result_1046 = invoke(stypy.reporting.localization.Localization(__file__, 330, 12), visit_1042, *[test_1044], **kwargs_1045)
            
            
            # Call to enter(...): (line 331)
            # Processing the call keyword arguments (line 331)
            kwargs_1049 = {}
            # Getting the type of 'self' (line 331)
            self_1047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 331)
            enter_1048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 12), self_1047, 'enter')
            # Calling enter(args, kwargs) (line 331)
            enter_call_result_1050 = invoke(stypy.reporting.localization.Localization(__file__, 331, 12), enter_1048, *[], **kwargs_1049)
            
            
            # Call to visit(...): (line 332)
            # Processing the call arguments (line 332)
            # Getting the type of 't' (line 332)
            t_1053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 23), 't', False)
            # Obtaining the member 'body' of a type (line 332)
            body_1054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 23), t_1053, 'body')
            # Processing the call keyword arguments (line 332)
            kwargs_1055 = {}
            # Getting the type of 'self' (line 332)
            self_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 332)
            visit_1052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 12), self_1051, 'visit')
            # Calling visit(args, kwargs) (line 332)
            visit_call_result_1056 = invoke(stypy.reporting.localization.Localization(__file__, 332, 12), visit_1052, *[body_1054], **kwargs_1055)
            
            
            # Call to leave(...): (line 333)
            # Processing the call keyword arguments (line 333)
            kwargs_1059 = {}
            # Getting the type of 'self' (line 333)
            self_1057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 333)
            leave_1058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 12), self_1057, 'leave')
            # Calling leave(args, kwargs) (line 333)
            leave_call_result_1060 = invoke(stypy.reporting.localization.Localization(__file__, 333, 12), leave_1058, *[], **kwargs_1059)
            
            # SSA join for while statement (line 326)
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 't' (line 335)
        t_1061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 11), 't')
        # Obtaining the member 'orelse' of a type (line 335)
        orelse_1062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 11), t_1061, 'orelse')
        # Testing if the type of an if condition is none (line 335)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 335, 8), orelse_1062):
            pass
        else:
            
            # Testing the type of an if condition (line 335)
            if_condition_1063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 335, 8), orelse_1062)
            # Assigning a type to the variable 'if_condition_1063' (line 335)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'if_condition_1063', if_condition_1063)
            # SSA begins for if statement (line 335)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fill(...): (line 336)
            # Processing the call arguments (line 336)
            str_1066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 22), 'str', 'else')
            # Processing the call keyword arguments (line 336)
            kwargs_1067 = {}
            # Getting the type of 'self' (line 336)
            self_1064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 336)
            fill_1065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), self_1064, 'fill')
            # Calling fill(args, kwargs) (line 336)
            fill_call_result_1068 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), fill_1065, *[str_1066], **kwargs_1067)
            
            
            # Call to enter(...): (line 337)
            # Processing the call keyword arguments (line 337)
            kwargs_1071 = {}
            # Getting the type of 'self' (line 337)
            self_1069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 337)
            enter_1070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), self_1069, 'enter')
            # Calling enter(args, kwargs) (line 337)
            enter_call_result_1072 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), enter_1070, *[], **kwargs_1071)
            
            
            # Call to visit(...): (line 338)
            # Processing the call arguments (line 338)
            # Getting the type of 't' (line 338)
            t_1075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 23), 't', False)
            # Obtaining the member 'orelse' of a type (line 338)
            orelse_1076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 23), t_1075, 'orelse')
            # Processing the call keyword arguments (line 338)
            kwargs_1077 = {}
            # Getting the type of 'self' (line 338)
            self_1073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 338)
            visit_1074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), self_1073, 'visit')
            # Calling visit(args, kwargs) (line 338)
            visit_call_result_1078 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), visit_1074, *[orelse_1076], **kwargs_1077)
            
            
            # Call to leave(...): (line 339)
            # Processing the call keyword arguments (line 339)
            kwargs_1081 = {}
            # Getting the type of 'self' (line 339)
            self_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 339)
            leave_1080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 12), self_1079, 'leave')
            # Calling leave(args, kwargs) (line 339)
            leave_call_result_1082 = invoke(stypy.reporting.localization.Localization(__file__, 339, 12), leave_1080, *[], **kwargs_1081)
            
            # SSA join for if statement (line 335)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 340)
        # Processing the call arguments (line 340)
        str_1085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, 19), 'str', '\n')
        # Processing the call keyword arguments (line 340)
        kwargs_1086 = {}
        # Getting the type of 'self' (line 340)
        self_1083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 340)
        write_1084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 340, 8), self_1083, 'write')
        # Calling write(args, kwargs) (line 340)
        write_call_result_1087 = invoke(stypy.reporting.localization.Localization(__file__, 340, 8), write_1084, *[str_1085], **kwargs_1086)
        
        
        # ################# End of 'visit_If(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_If' in the type store
        # Getting the type of 'stypy_return_type' (line 318)
        stypy_return_type_1088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1088)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_If'
        return stypy_return_type_1088


    @norecursion
    def visit_While(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_While'
        module_type_store = module_type_store.open_function_context('visit_While', 342, 4, False)
        # Assigning a type to the variable 'self' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 343)
        # Processing the call arguments (line 343)
        str_1091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 343, 18), 'str', 'while ')
        # Processing the call keyword arguments (line 343)
        kwargs_1092 = {}
        # Getting the type of 'self' (line 343)
        self_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 343)
        fill_1090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 8), self_1089, 'fill')
        # Calling fill(args, kwargs) (line 343)
        fill_call_result_1093 = invoke(stypy.reporting.localization.Localization(__file__, 343, 8), fill_1090, *[str_1091], **kwargs_1092)
        
        
        # Call to visit(...): (line 344)
        # Processing the call arguments (line 344)
        # Getting the type of 't' (line 344)
        t_1096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 19), 't', False)
        # Obtaining the member 'test' of a type (line 344)
        test_1097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 19), t_1096, 'test')
        # Processing the call keyword arguments (line 344)
        kwargs_1098 = {}
        # Getting the type of 'self' (line 344)
        self_1094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 344)
        visit_1095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 8), self_1094, 'visit')
        # Calling visit(args, kwargs) (line 344)
        visit_call_result_1099 = invoke(stypy.reporting.localization.Localization(__file__, 344, 8), visit_1095, *[test_1097], **kwargs_1098)
        
        
        # Call to enter(...): (line 345)
        # Processing the call keyword arguments (line 345)
        kwargs_1102 = {}
        # Getting the type of 'self' (line 345)
        self_1100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 345)
        enter_1101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 8), self_1100, 'enter')
        # Calling enter(args, kwargs) (line 345)
        enter_call_result_1103 = invoke(stypy.reporting.localization.Localization(__file__, 345, 8), enter_1101, *[], **kwargs_1102)
        
        
        # Call to visit(...): (line 346)
        # Processing the call arguments (line 346)
        # Getting the type of 't' (line 346)
        t_1106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 19), 't', False)
        # Obtaining the member 'body' of a type (line 346)
        body_1107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 19), t_1106, 'body')
        # Processing the call keyword arguments (line 346)
        kwargs_1108 = {}
        # Getting the type of 'self' (line 346)
        self_1104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 346)
        visit_1105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), self_1104, 'visit')
        # Calling visit(args, kwargs) (line 346)
        visit_call_result_1109 = invoke(stypy.reporting.localization.Localization(__file__, 346, 8), visit_1105, *[body_1107], **kwargs_1108)
        
        
        # Call to leave(...): (line 347)
        # Processing the call keyword arguments (line 347)
        kwargs_1112 = {}
        # Getting the type of 'self' (line 347)
        self_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 347)
        leave_1111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), self_1110, 'leave')
        # Calling leave(args, kwargs) (line 347)
        leave_call_result_1113 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), leave_1111, *[], **kwargs_1112)
        
        # Getting the type of 't' (line 348)
        t_1114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 11), 't')
        # Obtaining the member 'orelse' of a type (line 348)
        orelse_1115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 11), t_1114, 'orelse')
        # Testing if the type of an if condition is none (line 348)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 348, 8), orelse_1115):
            pass
        else:
            
            # Testing the type of an if condition (line 348)
            if_condition_1116 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 348, 8), orelse_1115)
            # Assigning a type to the variable 'if_condition_1116' (line 348)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'if_condition_1116', if_condition_1116)
            # SSA begins for if statement (line 348)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fill(...): (line 349)
            # Processing the call arguments (line 349)
            str_1119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 22), 'str', 'else')
            # Processing the call keyword arguments (line 349)
            kwargs_1120 = {}
            # Getting the type of 'self' (line 349)
            self_1117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 349)
            fill_1118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 12), self_1117, 'fill')
            # Calling fill(args, kwargs) (line 349)
            fill_call_result_1121 = invoke(stypy.reporting.localization.Localization(__file__, 349, 12), fill_1118, *[str_1119], **kwargs_1120)
            
            
            # Call to enter(...): (line 350)
            # Processing the call keyword arguments (line 350)
            kwargs_1124 = {}
            # Getting the type of 'self' (line 350)
            self_1122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 350)
            enter_1123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 350, 12), self_1122, 'enter')
            # Calling enter(args, kwargs) (line 350)
            enter_call_result_1125 = invoke(stypy.reporting.localization.Localization(__file__, 350, 12), enter_1123, *[], **kwargs_1124)
            
            
            # Call to visit(...): (line 351)
            # Processing the call arguments (line 351)
            # Getting the type of 't' (line 351)
            t_1128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 23), 't', False)
            # Obtaining the member 'orelse' of a type (line 351)
            orelse_1129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 23), t_1128, 'orelse')
            # Processing the call keyword arguments (line 351)
            kwargs_1130 = {}
            # Getting the type of 'self' (line 351)
            self_1126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 351)
            visit_1127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 12), self_1126, 'visit')
            # Calling visit(args, kwargs) (line 351)
            visit_call_result_1131 = invoke(stypy.reporting.localization.Localization(__file__, 351, 12), visit_1127, *[orelse_1129], **kwargs_1130)
            
            
            # Call to leave(...): (line 352)
            # Processing the call keyword arguments (line 352)
            kwargs_1134 = {}
            # Getting the type of 'self' (line 352)
            self_1132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 352)
            leave_1133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 352, 12), self_1132, 'leave')
            # Calling leave(args, kwargs) (line 352)
            leave_call_result_1135 = invoke(stypy.reporting.localization.Localization(__file__, 352, 12), leave_1133, *[], **kwargs_1134)
            
            # SSA join for if statement (line 348)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_While(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_While' in the type store
        # Getting the type of 'stypy_return_type' (line 342)
        stypy_return_type_1136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1136)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_While'
        return stypy_return_type_1136


    @norecursion
    def visit_With(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_With'
        module_type_store = module_type_store.open_function_context('visit_With', 354, 4, False)
        # Assigning a type to the variable 'self' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'self', type_of_self)
        
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

        
        # Call to fill(...): (line 355)
        # Processing the call arguments (line 355)
        str_1139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 18), 'str', 'with ')
        # Processing the call keyword arguments (line 355)
        kwargs_1140 = {}
        # Getting the type of 'self' (line 355)
        self_1137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 355)
        fill_1138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 8), self_1137, 'fill')
        # Calling fill(args, kwargs) (line 355)
        fill_call_result_1141 = invoke(stypy.reporting.localization.Localization(__file__, 355, 8), fill_1138, *[str_1139], **kwargs_1140)
        
        
        # Call to visit(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 't' (line 356)
        t_1144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 19), 't', False)
        # Obtaining the member 'context_expr' of a type (line 356)
        context_expr_1145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 19), t_1144, 'context_expr')
        # Processing the call keyword arguments (line 356)
        kwargs_1146 = {}
        # Getting the type of 'self' (line 356)
        self_1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 356)
        visit_1143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), self_1142, 'visit')
        # Calling visit(args, kwargs) (line 356)
        visit_call_result_1147 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), visit_1143, *[context_expr_1145], **kwargs_1146)
        
        # Getting the type of 't' (line 357)
        t_1148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 11), 't')
        # Obtaining the member 'optional_vars' of a type (line 357)
        optional_vars_1149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 11), t_1148, 'optional_vars')
        # Testing if the type of an if condition is none (line 357)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 357, 8), optional_vars_1149):
            pass
        else:
            
            # Testing the type of an if condition (line 357)
            if_condition_1150 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 357, 8), optional_vars_1149)
            # Assigning a type to the variable 'if_condition_1150' (line 357)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 357, 8), 'if_condition_1150', if_condition_1150)
            # SSA begins for if statement (line 357)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 358)
            # Processing the call arguments (line 358)
            str_1153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 23), 'str', ' as ')
            # Processing the call keyword arguments (line 358)
            kwargs_1154 = {}
            # Getting the type of 'self' (line 358)
            self_1151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 358)
            write_1152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), self_1151, 'write')
            # Calling write(args, kwargs) (line 358)
            write_call_result_1155 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), write_1152, *[str_1153], **kwargs_1154)
            
            
            # Call to visit(...): (line 359)
            # Processing the call arguments (line 359)
            # Getting the type of 't' (line 359)
            t_1158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 23), 't', False)
            # Obtaining the member 'optional_vars' of a type (line 359)
            optional_vars_1159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 23), t_1158, 'optional_vars')
            # Processing the call keyword arguments (line 359)
            kwargs_1160 = {}
            # Getting the type of 'self' (line 359)
            self_1156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 359)
            visit_1157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 12), self_1156, 'visit')
            # Calling visit(args, kwargs) (line 359)
            visit_call_result_1161 = invoke(stypy.reporting.localization.Localization(__file__, 359, 12), visit_1157, *[optional_vars_1159], **kwargs_1160)
            
            # SSA join for if statement (line 357)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to enter(...): (line 360)
        # Processing the call keyword arguments (line 360)
        kwargs_1164 = {}
        # Getting the type of 'self' (line 360)
        self_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 360)
        enter_1163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 8), self_1162, 'enter')
        # Calling enter(args, kwargs) (line 360)
        enter_call_result_1165 = invoke(stypy.reporting.localization.Localization(__file__, 360, 8), enter_1163, *[], **kwargs_1164)
        
        
        # Call to visit(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 't' (line 361)
        t_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 19), 't', False)
        # Obtaining the member 'body' of a type (line 361)
        body_1169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 19), t_1168, 'body')
        # Processing the call keyword arguments (line 361)
        kwargs_1170 = {}
        # Getting the type of 'self' (line 361)
        self_1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 361)
        visit_1167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 8), self_1166, 'visit')
        # Calling visit(args, kwargs) (line 361)
        visit_call_result_1171 = invoke(stypy.reporting.localization.Localization(__file__, 361, 8), visit_1167, *[body_1169], **kwargs_1170)
        
        
        # Call to leave(...): (line 362)
        # Processing the call keyword arguments (line 362)
        kwargs_1174 = {}
        # Getting the type of 'self' (line 362)
        self_1172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 362)
        leave_1173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 362, 8), self_1172, 'leave')
        # Calling leave(args, kwargs) (line 362)
        leave_call_result_1175 = invoke(stypy.reporting.localization.Localization(__file__, 362, 8), leave_1173, *[], **kwargs_1174)
        
        
        # ################# End of 'visit_With(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_With' in the type store
        # Getting the type of 'stypy_return_type' (line 354)
        stypy_return_type_1176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1176)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_With'
        return stypy_return_type_1176


    @norecursion
    def visit_Str(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Str'
        module_type_store = module_type_store.open_function_context('visit_Str', 365, 4, False)
        # Assigning a type to the variable 'self' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'self', type_of_self)
        
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

        
        str_1177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 369, 11), 'str', 'unicode_literals')
        # Getting the type of 'self' (line 369)
        self_1178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 37), 'self')
        # Obtaining the member 'future_imports' of a type (line 369)
        future_imports_1179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 37), self_1178, 'future_imports')
        # Applying the binary operator 'notin' (line 369)
        result_contains_1180 = python_operator(stypy.reporting.localization.Localization(__file__, 369, 11), 'notin', str_1177, future_imports_1179)
        
        # Testing if the type of an if condition is none (line 369)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 369, 8), result_contains_1180):
            
            # Type idiom detected: calculating its left and rigth part (line 371)
            # Getting the type of 'str' (line 371)
            str_1191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 32), 'str')
            # Getting the type of 'tree' (line 371)
            tree_1192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'tree')
            # Obtaining the member 's' of a type (line 371)
            s_1193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 24), tree_1192, 's')
            
            (may_be_1194, more_types_in_union_1195) = may_be_subtype(str_1191, s_1193)

            if may_be_1194:

                if more_types_in_union_1195:
                    # Runtime conditional SSA (line 371)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'tree' (line 371)
                tree_1196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 13), 'tree')
                # Obtaining the member 's' of a type (line 371)
                s_1197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 13), tree_1196, 's')
                # Setting the type of the member 's' of a type (line 371)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 13), tree_1196, 's', remove_not_subtype_from_union(s_1193, str))
                
                # Call to write(...): (line 372)
                # Processing the call arguments (line 372)
                str_1200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 23), 'str', 'b')
                
                # Call to repr(...): (line 372)
                # Processing the call arguments (line 372)
                # Getting the type of 'tree' (line 372)
                tree_1202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 34), 'tree', False)
                # Obtaining the member 's' of a type (line 372)
                s_1203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 34), tree_1202, 's')
                # Processing the call keyword arguments (line 372)
                kwargs_1204 = {}
                # Getting the type of 'repr' (line 372)
                repr_1201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 29), 'repr', False)
                # Calling repr(args, kwargs) (line 372)
                repr_call_result_1205 = invoke(stypy.reporting.localization.Localization(__file__, 372, 29), repr_1201, *[s_1203], **kwargs_1204)
                
                # Applying the binary operator '+' (line 372)
                result_add_1206 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 23), '+', str_1200, repr_call_result_1205)
                
                # Processing the call keyword arguments (line 372)
                kwargs_1207 = {}
                # Getting the type of 'self' (line 372)
                self_1198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'self', False)
                # Obtaining the member 'write' of a type (line 372)
                write_1199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 12), self_1198, 'write')
                # Calling write(args, kwargs) (line 372)
                write_call_result_1208 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), write_1199, *[result_add_1206], **kwargs_1207)
                

                if more_types_in_union_1195:
                    # Runtime conditional SSA for else branch (line 371)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_1194) or more_types_in_union_1195):
                # Getting the type of 'tree' (line 371)
                tree_1209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 13), 'tree')
                # Obtaining the member 's' of a type (line 371)
                s_1210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 13), tree_1209, 's')
                # Setting the type of the member 's' of a type (line 371)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 13), tree_1209, 's', remove_subtype_from_union(s_1193, str))
                
                # Type idiom detected: calculating its left and rigth part (line 373)
                # Getting the type of 'unicode' (line 373)
                unicode_1211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 32), 'unicode')
                # Getting the type of 'tree' (line 373)
                tree_1212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 24), 'tree')
                # Obtaining the member 's' of a type (line 373)
                s_1213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 24), tree_1212, 's')
                
                (may_be_1214, more_types_in_union_1215) = may_be_subtype(unicode_1211, s_1213)

                if may_be_1214:

                    if more_types_in_union_1215:
                        # Runtime conditional SSA (line 373)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Getting the type of 'tree' (line 373)
                    tree_1216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 13), 'tree')
                    # Obtaining the member 's' of a type (line 373)
                    s_1217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 13), tree_1216, 's')
                    # Setting the type of the member 's' of a type (line 373)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 13), tree_1216, 's', remove_not_subtype_from_union(s_1213, unicode))
                    
                    # Call to write(...): (line 374)
                    # Processing the call arguments (line 374)
                    
                    # Call to lstrip(...): (line 374)
                    # Processing the call arguments (line 374)
                    str_1226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 43), 'str', 'u')
                    # Processing the call keyword arguments (line 374)
                    kwargs_1227 = {}
                    
                    # Call to repr(...): (line 374)
                    # Processing the call arguments (line 374)
                    # Getting the type of 'tree' (line 374)
                    tree_1221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 28), 'tree', False)
                    # Obtaining the member 's' of a type (line 374)
                    s_1222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 28), tree_1221, 's')
                    # Processing the call keyword arguments (line 374)
                    kwargs_1223 = {}
                    # Getting the type of 'repr' (line 374)
                    repr_1220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'repr', False)
                    # Calling repr(args, kwargs) (line 374)
                    repr_call_result_1224 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), repr_1220, *[s_1222], **kwargs_1223)
                    
                    # Obtaining the member 'lstrip' of a type (line 374)
                    lstrip_1225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), repr_call_result_1224, 'lstrip')
                    # Calling lstrip(args, kwargs) (line 374)
                    lstrip_call_result_1228 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), lstrip_1225, *[str_1226], **kwargs_1227)
                    
                    # Processing the call keyword arguments (line 374)
                    kwargs_1229 = {}
                    # Getting the type of 'self' (line 374)
                    self_1218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'self', False)
                    # Obtaining the member 'write' of a type (line 374)
                    write_1219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), self_1218, 'write')
                    # Calling write(args, kwargs) (line 374)
                    write_call_result_1230 = invoke(stypy.reporting.localization.Localization(__file__, 374, 12), write_1219, *[lstrip_call_result_1228], **kwargs_1229)
                    

                    if more_types_in_union_1215:
                        # Runtime conditional SSA for else branch (line 373)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_1214) or more_types_in_union_1215):
                    # Getting the type of 'tree' (line 373)
                    tree_1231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 13), 'tree')
                    # Obtaining the member 's' of a type (line 373)
                    s_1232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 13), tree_1231, 's')
                    # Setting the type of the member 's' of a type (line 373)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 13), tree_1231, 's', remove_subtype_from_union(s_1213, unicode))
                    # Evaluating assert statement condition
                    # Getting the type of 'False' (line 376)
                    False_1233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'False')
                    assert_1234 = False_1233
                    # Assigning a type to the variable 'assert_1234' (line 376)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'assert_1234', False_1233)

                    if (may_be_1214 and more_types_in_union_1215):
                        # SSA join for if statement (line 373)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_1194 and more_types_in_union_1195):
                    # SSA join for if statement (line 371)
                    module_type_store = module_type_store.join_ssa_context()


            
        else:
            
            # Testing the type of an if condition (line 369)
            if_condition_1181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 369, 8), result_contains_1180)
            # Assigning a type to the variable 'if_condition_1181' (line 369)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'if_condition_1181', if_condition_1181)
            # SSA begins for if statement (line 369)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 370)
            # Processing the call arguments (line 370)
            
            # Call to repr(...): (line 370)
            # Processing the call arguments (line 370)
            # Getting the type of 'tree' (line 370)
            tree_1185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 28), 'tree', False)
            # Obtaining the member 's' of a type (line 370)
            s_1186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 28), tree_1185, 's')
            # Processing the call keyword arguments (line 370)
            kwargs_1187 = {}
            # Getting the type of 'repr' (line 370)
            repr_1184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 23), 'repr', False)
            # Calling repr(args, kwargs) (line 370)
            repr_call_result_1188 = invoke(stypy.reporting.localization.Localization(__file__, 370, 23), repr_1184, *[s_1186], **kwargs_1187)
            
            # Processing the call keyword arguments (line 370)
            kwargs_1189 = {}
            # Getting the type of 'self' (line 370)
            self_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 370)
            write_1183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 12), self_1182, 'write')
            # Calling write(args, kwargs) (line 370)
            write_call_result_1190 = invoke(stypy.reporting.localization.Localization(__file__, 370, 12), write_1183, *[repr_call_result_1188], **kwargs_1189)
            
            # SSA branch for the else part of an if statement (line 369)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 371)
            # Getting the type of 'str' (line 371)
            str_1191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 32), 'str')
            # Getting the type of 'tree' (line 371)
            tree_1192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 24), 'tree')
            # Obtaining the member 's' of a type (line 371)
            s_1193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 24), tree_1192, 's')
            
            (may_be_1194, more_types_in_union_1195) = may_be_subtype(str_1191, s_1193)

            if may_be_1194:

                if more_types_in_union_1195:
                    # Runtime conditional SSA (line 371)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'tree' (line 371)
                tree_1196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 13), 'tree')
                # Obtaining the member 's' of a type (line 371)
                s_1197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 13), tree_1196, 's')
                # Setting the type of the member 's' of a type (line 371)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 13), tree_1196, 's', remove_not_subtype_from_union(s_1193, str))
                
                # Call to write(...): (line 372)
                # Processing the call arguments (line 372)
                str_1200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 23), 'str', 'b')
                
                # Call to repr(...): (line 372)
                # Processing the call arguments (line 372)
                # Getting the type of 'tree' (line 372)
                tree_1202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 34), 'tree', False)
                # Obtaining the member 's' of a type (line 372)
                s_1203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 34), tree_1202, 's')
                # Processing the call keyword arguments (line 372)
                kwargs_1204 = {}
                # Getting the type of 'repr' (line 372)
                repr_1201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 29), 'repr', False)
                # Calling repr(args, kwargs) (line 372)
                repr_call_result_1205 = invoke(stypy.reporting.localization.Localization(__file__, 372, 29), repr_1201, *[s_1203], **kwargs_1204)
                
                # Applying the binary operator '+' (line 372)
                result_add_1206 = python_operator(stypy.reporting.localization.Localization(__file__, 372, 23), '+', str_1200, repr_call_result_1205)
                
                # Processing the call keyword arguments (line 372)
                kwargs_1207 = {}
                # Getting the type of 'self' (line 372)
                self_1198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 12), 'self', False)
                # Obtaining the member 'write' of a type (line 372)
                write_1199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 372, 12), self_1198, 'write')
                # Calling write(args, kwargs) (line 372)
                write_call_result_1208 = invoke(stypy.reporting.localization.Localization(__file__, 372, 12), write_1199, *[result_add_1206], **kwargs_1207)
                

                if more_types_in_union_1195:
                    # Runtime conditional SSA for else branch (line 371)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_1194) or more_types_in_union_1195):
                # Getting the type of 'tree' (line 371)
                tree_1209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 13), 'tree')
                # Obtaining the member 's' of a type (line 371)
                s_1210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 13), tree_1209, 's')
                # Setting the type of the member 's' of a type (line 371)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 371, 13), tree_1209, 's', remove_subtype_from_union(s_1193, str))
                
                # Type idiom detected: calculating its left and rigth part (line 373)
                # Getting the type of 'unicode' (line 373)
                unicode_1211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 32), 'unicode')
                # Getting the type of 'tree' (line 373)
                tree_1212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 24), 'tree')
                # Obtaining the member 's' of a type (line 373)
                s_1213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 24), tree_1212, 's')
                
                (may_be_1214, more_types_in_union_1215) = may_be_subtype(unicode_1211, s_1213)

                if may_be_1214:

                    if more_types_in_union_1215:
                        # Runtime conditional SSA (line 373)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Getting the type of 'tree' (line 373)
                    tree_1216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 13), 'tree')
                    # Obtaining the member 's' of a type (line 373)
                    s_1217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 13), tree_1216, 's')
                    # Setting the type of the member 's' of a type (line 373)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 13), tree_1216, 's', remove_not_subtype_from_union(s_1213, unicode))
                    
                    # Call to write(...): (line 374)
                    # Processing the call arguments (line 374)
                    
                    # Call to lstrip(...): (line 374)
                    # Processing the call arguments (line 374)
                    str_1226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 374, 43), 'str', 'u')
                    # Processing the call keyword arguments (line 374)
                    kwargs_1227 = {}
                    
                    # Call to repr(...): (line 374)
                    # Processing the call arguments (line 374)
                    # Getting the type of 'tree' (line 374)
                    tree_1221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 28), 'tree', False)
                    # Obtaining the member 's' of a type (line 374)
                    s_1222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 28), tree_1221, 's')
                    # Processing the call keyword arguments (line 374)
                    kwargs_1223 = {}
                    # Getting the type of 'repr' (line 374)
                    repr_1220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 23), 'repr', False)
                    # Calling repr(args, kwargs) (line 374)
                    repr_call_result_1224 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), repr_1220, *[s_1222], **kwargs_1223)
                    
                    # Obtaining the member 'lstrip' of a type (line 374)
                    lstrip_1225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 23), repr_call_result_1224, 'lstrip')
                    # Calling lstrip(args, kwargs) (line 374)
                    lstrip_call_result_1228 = invoke(stypy.reporting.localization.Localization(__file__, 374, 23), lstrip_1225, *[str_1226], **kwargs_1227)
                    
                    # Processing the call keyword arguments (line 374)
                    kwargs_1229 = {}
                    # Getting the type of 'self' (line 374)
                    self_1218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 12), 'self', False)
                    # Obtaining the member 'write' of a type (line 374)
                    write_1219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 374, 12), self_1218, 'write')
                    # Calling write(args, kwargs) (line 374)
                    write_call_result_1230 = invoke(stypy.reporting.localization.Localization(__file__, 374, 12), write_1219, *[lstrip_call_result_1228], **kwargs_1229)
                    

                    if more_types_in_union_1215:
                        # Runtime conditional SSA for else branch (line 373)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_1214) or more_types_in_union_1215):
                    # Getting the type of 'tree' (line 373)
                    tree_1231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 13), 'tree')
                    # Obtaining the member 's' of a type (line 373)
                    s_1232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 13), tree_1231, 's')
                    # Setting the type of the member 's' of a type (line 373)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 13), tree_1231, 's', remove_subtype_from_union(s_1213, unicode))
                    # Evaluating assert statement condition
                    # Getting the type of 'False' (line 376)
                    False_1233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 19), 'False')
                    assert_1234 = False_1233
                    # Assigning a type to the variable 'assert_1234' (line 376)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'assert_1234', False_1233)

                    if (may_be_1214 and more_types_in_union_1215):
                        # SSA join for if statement (line 373)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_1194 and more_types_in_union_1195):
                    # SSA join for if statement (line 371)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 369)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Str(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Str' in the type store
        # Getting the type of 'stypy_return_type' (line 365)
        stypy_return_type_1235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1235)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Str'
        return stypy_return_type_1235


    @norecursion
    def visit_Name(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Name'
        module_type_store = module_type_store.open_function_context('visit_Name', 378, 4, False)
        # Assigning a type to the variable 'self' (line 379)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 379)
        # Processing the call arguments (line 379)
        # Getting the type of 't' (line 379)
        t_1238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 19), 't', False)
        # Obtaining the member 'id' of a type (line 379)
        id_1239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 19), t_1238, 'id')
        # Processing the call keyword arguments (line 379)
        kwargs_1240 = {}
        # Getting the type of 'self' (line 379)
        self_1236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 379)
        write_1237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 8), self_1236, 'write')
        # Calling write(args, kwargs) (line 379)
        write_call_result_1241 = invoke(stypy.reporting.localization.Localization(__file__, 379, 8), write_1237, *[id_1239], **kwargs_1240)
        
        
        # ################# End of 'visit_Name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Name' in the type store
        # Getting the type of 'stypy_return_type' (line 378)
        stypy_return_type_1242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1242)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Name'
        return stypy_return_type_1242


    @norecursion
    def visit_Repr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Repr'
        module_type_store = module_type_store.open_function_context('visit_Repr', 381, 4, False)
        # Assigning a type to the variable 'self' (line 382)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 382)
        # Processing the call arguments (line 382)
        str_1245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 382, 19), 'str', '`')
        # Processing the call keyword arguments (line 382)
        kwargs_1246 = {}
        # Getting the type of 'self' (line 382)
        self_1243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 382)
        write_1244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 382, 8), self_1243, 'write')
        # Calling write(args, kwargs) (line 382)
        write_call_result_1247 = invoke(stypy.reporting.localization.Localization(__file__, 382, 8), write_1244, *[str_1245], **kwargs_1246)
        
        
        # Call to visit(...): (line 383)
        # Processing the call arguments (line 383)
        # Getting the type of 't' (line 383)
        t_1250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 19), 't', False)
        # Obtaining the member 'value' of a type (line 383)
        value_1251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 19), t_1250, 'value')
        # Processing the call keyword arguments (line 383)
        kwargs_1252 = {}
        # Getting the type of 'self' (line 383)
        self_1248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 383)
        visit_1249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_1248, 'visit')
        # Calling visit(args, kwargs) (line 383)
        visit_call_result_1253 = invoke(stypy.reporting.localization.Localization(__file__, 383, 8), visit_1249, *[value_1251], **kwargs_1252)
        
        
        # Call to write(...): (line 384)
        # Processing the call arguments (line 384)
        str_1256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 19), 'str', '`')
        # Processing the call keyword arguments (line 384)
        kwargs_1257 = {}
        # Getting the type of 'self' (line 384)
        self_1254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 384)
        write_1255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 8), self_1254, 'write')
        # Calling write(args, kwargs) (line 384)
        write_call_result_1258 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), write_1255, *[str_1256], **kwargs_1257)
        
        
        # ################# End of 'visit_Repr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Repr' in the type store
        # Getting the type of 'stypy_return_type' (line 381)
        stypy_return_type_1259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1259)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Repr'
        return stypy_return_type_1259


    @norecursion
    def visit_Num(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Num'
        module_type_store = module_type_store.open_function_context('visit_Num', 386, 4, False)
        # Assigning a type to the variable 'self' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'self', type_of_self)
        
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

        
        # Assigning a Call to a Name (line 387):
        
        # Assigning a Call to a Name (line 387):
        
        # Call to repr(...): (line 387)
        # Processing the call arguments (line 387)
        # Getting the type of 't' (line 387)
        t_1261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 22), 't', False)
        # Obtaining the member 'n' of a type (line 387)
        n_1262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 22), t_1261, 'n')
        # Processing the call keyword arguments (line 387)
        kwargs_1263 = {}
        # Getting the type of 'repr' (line 387)
        repr_1260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 17), 'repr', False)
        # Calling repr(args, kwargs) (line 387)
        repr_call_result_1264 = invoke(stypy.reporting.localization.Localization(__file__, 387, 17), repr_1260, *[n_1262], **kwargs_1263)
        
        # Assigning a type to the variable 'repr_n' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 8), 'repr_n', repr_call_result_1264)
        
        # Call to startswith(...): (line 389)
        # Processing the call arguments (line 389)
        str_1267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 389, 29), 'str', '-')
        # Processing the call keyword arguments (line 389)
        kwargs_1268 = {}
        # Getting the type of 'repr_n' (line 389)
        repr_n_1265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 11), 'repr_n', False)
        # Obtaining the member 'startswith' of a type (line 389)
        startswith_1266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 11), repr_n_1265, 'startswith')
        # Calling startswith(args, kwargs) (line 389)
        startswith_call_result_1269 = invoke(stypy.reporting.localization.Localization(__file__, 389, 11), startswith_1266, *[str_1267], **kwargs_1268)
        
        # Testing if the type of an if condition is none (line 389)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 389, 8), startswith_call_result_1269):
            pass
        else:
            
            # Testing the type of an if condition (line 389)
            if_condition_1270 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 389, 8), startswith_call_result_1269)
            # Assigning a type to the variable 'if_condition_1270' (line 389)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'if_condition_1270', if_condition_1270)
            # SSA begins for if statement (line 389)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 390)
            # Processing the call arguments (line 390)
            str_1273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 390, 23), 'str', '(')
            # Processing the call keyword arguments (line 390)
            kwargs_1274 = {}
            # Getting the type of 'self' (line 390)
            self_1271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 390)
            write_1272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 12), self_1271, 'write')
            # Calling write(args, kwargs) (line 390)
            write_call_result_1275 = invoke(stypy.reporting.localization.Localization(__file__, 390, 12), write_1272, *[str_1273], **kwargs_1274)
            
            # SSA join for if statement (line 389)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 392)
        # Processing the call arguments (line 392)
        
        # Call to replace(...): (line 392)
        # Processing the call arguments (line 392)
        str_1280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 34), 'str', 'inf')
        # Getting the type of 'INFSTR' (line 392)
        INFSTR_1281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 41), 'INFSTR', False)
        # Processing the call keyword arguments (line 392)
        kwargs_1282 = {}
        # Getting the type of 'repr_n' (line 392)
        repr_n_1278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 19), 'repr_n', False)
        # Obtaining the member 'replace' of a type (line 392)
        replace_1279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 19), repr_n_1278, 'replace')
        # Calling replace(args, kwargs) (line 392)
        replace_call_result_1283 = invoke(stypy.reporting.localization.Localization(__file__, 392, 19), replace_1279, *[str_1280, INFSTR_1281], **kwargs_1282)
        
        # Processing the call keyword arguments (line 392)
        kwargs_1284 = {}
        # Getting the type of 'self' (line 392)
        self_1276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 392)
        write_1277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), self_1276, 'write')
        # Calling write(args, kwargs) (line 392)
        write_call_result_1285 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), write_1277, *[replace_call_result_1283], **kwargs_1284)
        
        
        # Call to startswith(...): (line 393)
        # Processing the call arguments (line 393)
        str_1288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 29), 'str', '-')
        # Processing the call keyword arguments (line 393)
        kwargs_1289 = {}
        # Getting the type of 'repr_n' (line 393)
        repr_n_1286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 11), 'repr_n', False)
        # Obtaining the member 'startswith' of a type (line 393)
        startswith_1287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 11), repr_n_1286, 'startswith')
        # Calling startswith(args, kwargs) (line 393)
        startswith_call_result_1290 = invoke(stypy.reporting.localization.Localization(__file__, 393, 11), startswith_1287, *[str_1288], **kwargs_1289)
        
        # Testing if the type of an if condition is none (line 393)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 393, 8), startswith_call_result_1290):
            pass
        else:
            
            # Testing the type of an if condition (line 393)
            if_condition_1291 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 393, 8), startswith_call_result_1290)
            # Assigning a type to the variable 'if_condition_1291' (line 393)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 8), 'if_condition_1291', if_condition_1291)
            # SSA begins for if statement (line 393)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 394)
            # Processing the call arguments (line 394)
            str_1294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 394, 23), 'str', ')')
            # Processing the call keyword arguments (line 394)
            kwargs_1295 = {}
            # Getting the type of 'self' (line 394)
            self_1292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 394)
            write_1293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 394, 12), self_1292, 'write')
            # Calling write(args, kwargs) (line 394)
            write_call_result_1296 = invoke(stypy.reporting.localization.Localization(__file__, 394, 12), write_1293, *[str_1294], **kwargs_1295)
            
            # SSA join for if statement (line 393)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Num(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Num' in the type store
        # Getting the type of 'stypy_return_type' (line 386)
        stypy_return_type_1297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1297)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Num'
        return stypy_return_type_1297


    @norecursion
    def visit_List(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_List'
        module_type_store = module_type_store.open_function_context('visit_List', 396, 4, False)
        # Assigning a type to the variable 'self' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 397)
        # Processing the call arguments (line 397)
        str_1300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 397, 19), 'str', '[')
        # Processing the call keyword arguments (line 397)
        kwargs_1301 = {}
        # Getting the type of 'self' (line 397)
        self_1298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 397)
        write_1299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 8), self_1298, 'write')
        # Calling write(args, kwargs) (line 397)
        write_call_result_1302 = invoke(stypy.reporting.localization.Localization(__file__, 397, 8), write_1299, *[str_1300], **kwargs_1301)
        
        
        # Call to interleave(...): (line 398)
        # Processing the call arguments (line 398)

        @norecursion
        def _stypy_temp_lambda_5(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_5'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_5', 398, 19, True)
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

            
            # Call to write(...): (line 398)
            # Processing the call arguments (line 398)
            str_1306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 38), 'str', ', ')
            # Processing the call keyword arguments (line 398)
            kwargs_1307 = {}
            # Getting the type of 'self' (line 398)
            self_1304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 398)
            write_1305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 27), self_1304, 'write')
            # Calling write(args, kwargs) (line 398)
            write_call_result_1308 = invoke(stypy.reporting.localization.Localization(__file__, 398, 27), write_1305, *[str_1306], **kwargs_1307)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 398)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), 'stypy_return_type', write_call_result_1308)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_5' in the type store
            # Getting the type of 'stypy_return_type' (line 398)
            stypy_return_type_1309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1309)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_5'
            return stypy_return_type_1309

        # Assigning a type to the variable '_stypy_temp_lambda_5' (line 398)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), '_stypy_temp_lambda_5', _stypy_temp_lambda_5)
        # Getting the type of '_stypy_temp_lambda_5' (line 398)
        _stypy_temp_lambda_5_1310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 19), '_stypy_temp_lambda_5')
        # Getting the type of 'self' (line 398)
        self_1311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 398)
        visit_1312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 45), self_1311, 'visit')
        # Getting the type of 't' (line 398)
        t_1313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 57), 't', False)
        # Obtaining the member 'elts' of a type (line 398)
        elts_1314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 57), t_1313, 'elts')
        # Processing the call keyword arguments (line 398)
        kwargs_1315 = {}
        # Getting the type of 'interleave' (line 398)
        interleave_1303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 398)
        interleave_call_result_1316 = invoke(stypy.reporting.localization.Localization(__file__, 398, 8), interleave_1303, *[_stypy_temp_lambda_5_1310, visit_1312, elts_1314], **kwargs_1315)
        
        
        # Call to write(...): (line 399)
        # Processing the call arguments (line 399)
        str_1319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 19), 'str', ']')
        # Processing the call keyword arguments (line 399)
        kwargs_1320 = {}
        # Getting the type of 'self' (line 399)
        self_1317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 399)
        write_1318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), self_1317, 'write')
        # Calling write(args, kwargs) (line 399)
        write_call_result_1321 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), write_1318, *[str_1319], **kwargs_1320)
        
        
        # ################# End of 'visit_List(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_List' in the type store
        # Getting the type of 'stypy_return_type' (line 396)
        stypy_return_type_1322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1322)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_List'
        return stypy_return_type_1322


    @norecursion
    def visit_ListComp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ListComp'
        module_type_store = module_type_store.open_function_context('visit_ListComp', 401, 4, False)
        # Assigning a type to the variable 'self' (line 402)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 402, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 402)
        # Processing the call arguments (line 402)
        str_1325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 19), 'str', '[')
        # Processing the call keyword arguments (line 402)
        kwargs_1326 = {}
        # Getting the type of 'self' (line 402)
        self_1323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 402)
        write_1324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), self_1323, 'write')
        # Calling write(args, kwargs) (line 402)
        write_call_result_1327 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), write_1324, *[str_1325], **kwargs_1326)
        
        
        # Call to visit(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 't' (line 403)
        t_1330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 't', False)
        # Obtaining the member 'elt' of a type (line 403)
        elt_1331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 19), t_1330, 'elt')
        # Processing the call keyword arguments (line 403)
        kwargs_1332 = {}
        # Getting the type of 'self' (line 403)
        self_1328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 403)
        visit_1329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), self_1328, 'visit')
        # Calling visit(args, kwargs) (line 403)
        visit_call_result_1333 = invoke(stypy.reporting.localization.Localization(__file__, 403, 8), visit_1329, *[elt_1331], **kwargs_1332)
        
        
        # Getting the type of 't' (line 404)
        t_1334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 19), 't')
        # Obtaining the member 'generators' of a type (line 404)
        generators_1335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 19), t_1334, 'generators')
        # Assigning a type to the variable 'generators_1335' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'generators_1335', generators_1335)
        # Testing if the for loop is going to be iterated (line 404)
        # Testing the type of a for loop iterable (line 404)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 404, 8), generators_1335)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 404, 8), generators_1335):
            # Getting the type of the for loop variable (line 404)
            for_loop_var_1336 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 404, 8), generators_1335)
            # Assigning a type to the variable 'gen' (line 404)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'gen', for_loop_var_1336)
            # SSA begins for a for statement (line 404)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 405)
            # Processing the call arguments (line 405)
            # Getting the type of 'gen' (line 405)
            gen_1339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 23), 'gen', False)
            # Processing the call keyword arguments (line 405)
            kwargs_1340 = {}
            # Getting the type of 'self' (line 405)
            self_1337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 405)
            visit_1338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 12), self_1337, 'visit')
            # Calling visit(args, kwargs) (line 405)
            visit_call_result_1341 = invoke(stypy.reporting.localization.Localization(__file__, 405, 12), visit_1338, *[gen_1339], **kwargs_1340)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 406)
        # Processing the call arguments (line 406)
        str_1344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 19), 'str', ']')
        # Processing the call keyword arguments (line 406)
        kwargs_1345 = {}
        # Getting the type of 'self' (line 406)
        self_1342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 406)
        write_1343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), self_1342, 'write')
        # Calling write(args, kwargs) (line 406)
        write_call_result_1346 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), write_1343, *[str_1344], **kwargs_1345)
        
        
        # ################# End of 'visit_ListComp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ListComp' in the type store
        # Getting the type of 'stypy_return_type' (line 401)
        stypy_return_type_1347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1347)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ListComp'
        return stypy_return_type_1347


    @norecursion
    def visit_GeneratorExp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_GeneratorExp'
        module_type_store = module_type_store.open_function_context('visit_GeneratorExp', 408, 4, False)
        # Assigning a type to the variable 'self' (line 409)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 409)
        # Processing the call arguments (line 409)
        str_1350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 19), 'str', '(')
        # Processing the call keyword arguments (line 409)
        kwargs_1351 = {}
        # Getting the type of 'self' (line 409)
        self_1348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 409)
        write_1349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), self_1348, 'write')
        # Calling write(args, kwargs) (line 409)
        write_call_result_1352 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), write_1349, *[str_1350], **kwargs_1351)
        
        
        # Call to visit(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 't' (line 410)
        t_1355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 19), 't', False)
        # Obtaining the member 'elt' of a type (line 410)
        elt_1356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 19), t_1355, 'elt')
        # Processing the call keyword arguments (line 410)
        kwargs_1357 = {}
        # Getting the type of 'self' (line 410)
        self_1353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 410)
        visit_1354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 8), self_1353, 'visit')
        # Calling visit(args, kwargs) (line 410)
        visit_call_result_1358 = invoke(stypy.reporting.localization.Localization(__file__, 410, 8), visit_1354, *[elt_1356], **kwargs_1357)
        
        
        # Getting the type of 't' (line 411)
        t_1359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 19), 't')
        # Obtaining the member 'generators' of a type (line 411)
        generators_1360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 19), t_1359, 'generators')
        # Assigning a type to the variable 'generators_1360' (line 411)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'generators_1360', generators_1360)
        # Testing if the for loop is going to be iterated (line 411)
        # Testing the type of a for loop iterable (line 411)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 411, 8), generators_1360)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 411, 8), generators_1360):
            # Getting the type of the for loop variable (line 411)
            for_loop_var_1361 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 411, 8), generators_1360)
            # Assigning a type to the variable 'gen' (line 411)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'gen', for_loop_var_1361)
            # SSA begins for a for statement (line 411)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 412)
            # Processing the call arguments (line 412)
            # Getting the type of 'gen' (line 412)
            gen_1364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 23), 'gen', False)
            # Processing the call keyword arguments (line 412)
            kwargs_1365 = {}
            # Getting the type of 'self' (line 412)
            self_1362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 412)
            visit_1363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 12), self_1362, 'visit')
            # Calling visit(args, kwargs) (line 412)
            visit_call_result_1366 = invoke(stypy.reporting.localization.Localization(__file__, 412, 12), visit_1363, *[gen_1364], **kwargs_1365)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 413)
        # Processing the call arguments (line 413)
        str_1369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 413, 19), 'str', ')')
        # Processing the call keyword arguments (line 413)
        kwargs_1370 = {}
        # Getting the type of 'self' (line 413)
        self_1367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 413)
        write_1368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 8), self_1367, 'write')
        # Calling write(args, kwargs) (line 413)
        write_call_result_1371 = invoke(stypy.reporting.localization.Localization(__file__, 413, 8), write_1368, *[str_1369], **kwargs_1370)
        
        
        # ################# End of 'visit_GeneratorExp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_GeneratorExp' in the type store
        # Getting the type of 'stypy_return_type' (line 408)
        stypy_return_type_1372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1372)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_GeneratorExp'
        return stypy_return_type_1372


    @norecursion
    def visit_SetComp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_SetComp'
        module_type_store = module_type_store.open_function_context('visit_SetComp', 415, 4, False)
        # Assigning a type to the variable 'self' (line 416)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 416, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 416)
        # Processing the call arguments (line 416)
        str_1375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 416, 19), 'str', '{')
        # Processing the call keyword arguments (line 416)
        kwargs_1376 = {}
        # Getting the type of 'self' (line 416)
        self_1373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 416, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 416)
        write_1374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 416, 8), self_1373, 'write')
        # Calling write(args, kwargs) (line 416)
        write_call_result_1377 = invoke(stypy.reporting.localization.Localization(__file__, 416, 8), write_1374, *[str_1375], **kwargs_1376)
        
        
        # Call to visit(...): (line 417)
        # Processing the call arguments (line 417)
        # Getting the type of 't' (line 417)
        t_1380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 19), 't', False)
        # Obtaining the member 'elt' of a type (line 417)
        elt_1381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 19), t_1380, 'elt')
        # Processing the call keyword arguments (line 417)
        kwargs_1382 = {}
        # Getting the type of 'self' (line 417)
        self_1378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 417)
        visit_1379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 417, 8), self_1378, 'visit')
        # Calling visit(args, kwargs) (line 417)
        visit_call_result_1383 = invoke(stypy.reporting.localization.Localization(__file__, 417, 8), visit_1379, *[elt_1381], **kwargs_1382)
        
        
        # Getting the type of 't' (line 418)
        t_1384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 19), 't')
        # Obtaining the member 'generators' of a type (line 418)
        generators_1385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 19), t_1384, 'generators')
        # Assigning a type to the variable 'generators_1385' (line 418)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'generators_1385', generators_1385)
        # Testing if the for loop is going to be iterated (line 418)
        # Testing the type of a for loop iterable (line 418)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 418, 8), generators_1385)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 418, 8), generators_1385):
            # Getting the type of the for loop variable (line 418)
            for_loop_var_1386 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 418, 8), generators_1385)
            # Assigning a type to the variable 'gen' (line 418)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'gen', for_loop_var_1386)
            # SSA begins for a for statement (line 418)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 419)
            # Processing the call arguments (line 419)
            # Getting the type of 'gen' (line 419)
            gen_1389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 23), 'gen', False)
            # Processing the call keyword arguments (line 419)
            kwargs_1390 = {}
            # Getting the type of 'self' (line 419)
            self_1387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 419)
            visit_1388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 12), self_1387, 'visit')
            # Calling visit(args, kwargs) (line 419)
            visit_call_result_1391 = invoke(stypy.reporting.localization.Localization(__file__, 419, 12), visit_1388, *[gen_1389], **kwargs_1390)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 420)
        # Processing the call arguments (line 420)
        str_1394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 19), 'str', '}')
        # Processing the call keyword arguments (line 420)
        kwargs_1395 = {}
        # Getting the type of 'self' (line 420)
        self_1392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 420)
        write_1393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 8), self_1392, 'write')
        # Calling write(args, kwargs) (line 420)
        write_call_result_1396 = invoke(stypy.reporting.localization.Localization(__file__, 420, 8), write_1393, *[str_1394], **kwargs_1395)
        
        
        # ################# End of 'visit_SetComp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_SetComp' in the type store
        # Getting the type of 'stypy_return_type' (line 415)
        stypy_return_type_1397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1397)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_SetComp'
        return stypy_return_type_1397


    @norecursion
    def visit_DictComp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_DictComp'
        module_type_store = module_type_store.open_function_context('visit_DictComp', 422, 4, False)
        # Assigning a type to the variable 'self' (line 423)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 423)
        # Processing the call arguments (line 423)
        str_1400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 19), 'str', '{')
        # Processing the call keyword arguments (line 423)
        kwargs_1401 = {}
        # Getting the type of 'self' (line 423)
        self_1398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 423)
        write_1399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 8), self_1398, 'write')
        # Calling write(args, kwargs) (line 423)
        write_call_result_1402 = invoke(stypy.reporting.localization.Localization(__file__, 423, 8), write_1399, *[str_1400], **kwargs_1401)
        
        
        # Call to visit(...): (line 424)
        # Processing the call arguments (line 424)
        # Getting the type of 't' (line 424)
        t_1405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 19), 't', False)
        # Obtaining the member 'key' of a type (line 424)
        key_1406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 19), t_1405, 'key')
        # Processing the call keyword arguments (line 424)
        kwargs_1407 = {}
        # Getting the type of 'self' (line 424)
        self_1403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 424)
        visit_1404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 8), self_1403, 'visit')
        # Calling visit(args, kwargs) (line 424)
        visit_call_result_1408 = invoke(stypy.reporting.localization.Localization(__file__, 424, 8), visit_1404, *[key_1406], **kwargs_1407)
        
        
        # Call to write(...): (line 425)
        # Processing the call arguments (line 425)
        str_1411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 425, 19), 'str', ': ')
        # Processing the call keyword arguments (line 425)
        kwargs_1412 = {}
        # Getting the type of 'self' (line 425)
        self_1409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 425)
        write_1410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 425, 8), self_1409, 'write')
        # Calling write(args, kwargs) (line 425)
        write_call_result_1413 = invoke(stypy.reporting.localization.Localization(__file__, 425, 8), write_1410, *[str_1411], **kwargs_1412)
        
        
        # Call to visit(...): (line 426)
        # Processing the call arguments (line 426)
        # Getting the type of 't' (line 426)
        t_1416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 19), 't', False)
        # Obtaining the member 'value' of a type (line 426)
        value_1417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 19), t_1416, 'value')
        # Processing the call keyword arguments (line 426)
        kwargs_1418 = {}
        # Getting the type of 'self' (line 426)
        self_1414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 426)
        visit_1415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 426, 8), self_1414, 'visit')
        # Calling visit(args, kwargs) (line 426)
        visit_call_result_1419 = invoke(stypy.reporting.localization.Localization(__file__, 426, 8), visit_1415, *[value_1417], **kwargs_1418)
        
        
        # Getting the type of 't' (line 427)
        t_1420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 19), 't')
        # Obtaining the member 'generators' of a type (line 427)
        generators_1421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 19), t_1420, 'generators')
        # Assigning a type to the variable 'generators_1421' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'generators_1421', generators_1421)
        # Testing if the for loop is going to be iterated (line 427)
        # Testing the type of a for loop iterable (line 427)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 427, 8), generators_1421)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 427, 8), generators_1421):
            # Getting the type of the for loop variable (line 427)
            for_loop_var_1422 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 427, 8), generators_1421)
            # Assigning a type to the variable 'gen' (line 427)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'gen', for_loop_var_1422)
            # SSA begins for a for statement (line 427)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 428)
            # Processing the call arguments (line 428)
            # Getting the type of 'gen' (line 428)
            gen_1425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 23), 'gen', False)
            # Processing the call keyword arguments (line 428)
            kwargs_1426 = {}
            # Getting the type of 'self' (line 428)
            self_1423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 428)
            visit_1424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 12), self_1423, 'visit')
            # Calling visit(args, kwargs) (line 428)
            visit_call_result_1427 = invoke(stypy.reporting.localization.Localization(__file__, 428, 12), visit_1424, *[gen_1425], **kwargs_1426)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 429)
        # Processing the call arguments (line 429)
        str_1430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 19), 'str', '}')
        # Processing the call keyword arguments (line 429)
        kwargs_1431 = {}
        # Getting the type of 'self' (line 429)
        self_1428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 429)
        write_1429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 8), self_1428, 'write')
        # Calling write(args, kwargs) (line 429)
        write_call_result_1432 = invoke(stypy.reporting.localization.Localization(__file__, 429, 8), write_1429, *[str_1430], **kwargs_1431)
        
        
        # ################# End of 'visit_DictComp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_DictComp' in the type store
        # Getting the type of 'stypy_return_type' (line 422)
        stypy_return_type_1433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1433)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_DictComp'
        return stypy_return_type_1433


    @norecursion
    def visit_comprehension(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_comprehension'
        module_type_store = module_type_store.open_function_context('visit_comprehension', 431, 4, False)
        # Assigning a type to the variable 'self' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 432)
        # Processing the call arguments (line 432)
        str_1436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 19), 'str', ' for ')
        # Processing the call keyword arguments (line 432)
        kwargs_1437 = {}
        # Getting the type of 'self' (line 432)
        self_1434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 432)
        write_1435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), self_1434, 'write')
        # Calling write(args, kwargs) (line 432)
        write_call_result_1438 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), write_1435, *[str_1436], **kwargs_1437)
        
        
        # Call to visit(...): (line 433)
        # Processing the call arguments (line 433)
        # Getting the type of 't' (line 433)
        t_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 19), 't', False)
        # Obtaining the member 'target' of a type (line 433)
        target_1442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 19), t_1441, 'target')
        # Processing the call keyword arguments (line 433)
        kwargs_1443 = {}
        # Getting the type of 'self' (line 433)
        self_1439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 433)
        visit_1440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), self_1439, 'visit')
        # Calling visit(args, kwargs) (line 433)
        visit_call_result_1444 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), visit_1440, *[target_1442], **kwargs_1443)
        
        
        # Call to write(...): (line 434)
        # Processing the call arguments (line 434)
        str_1447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 434, 19), 'str', ' in ')
        # Processing the call keyword arguments (line 434)
        kwargs_1448 = {}
        # Getting the type of 'self' (line 434)
        self_1445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 434, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 434)
        write_1446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 434, 8), self_1445, 'write')
        # Calling write(args, kwargs) (line 434)
        write_call_result_1449 = invoke(stypy.reporting.localization.Localization(__file__, 434, 8), write_1446, *[str_1447], **kwargs_1448)
        
        
        # Call to visit(...): (line 435)
        # Processing the call arguments (line 435)
        # Getting the type of 't' (line 435)
        t_1452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 19), 't', False)
        # Obtaining the member 'iter' of a type (line 435)
        iter_1453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 19), t_1452, 'iter')
        # Processing the call keyword arguments (line 435)
        kwargs_1454 = {}
        # Getting the type of 'self' (line 435)
        self_1450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 435)
        visit_1451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 435, 8), self_1450, 'visit')
        # Calling visit(args, kwargs) (line 435)
        visit_call_result_1455 = invoke(stypy.reporting.localization.Localization(__file__, 435, 8), visit_1451, *[iter_1453], **kwargs_1454)
        
        
        # Getting the type of 't' (line 436)
        t_1456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 25), 't')
        # Obtaining the member 'ifs' of a type (line 436)
        ifs_1457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 25), t_1456, 'ifs')
        # Assigning a type to the variable 'ifs_1457' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'ifs_1457', ifs_1457)
        # Testing if the for loop is going to be iterated (line 436)
        # Testing the type of a for loop iterable (line 436)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 436, 8), ifs_1457)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 436, 8), ifs_1457):
            # Getting the type of the for loop variable (line 436)
            for_loop_var_1458 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 436, 8), ifs_1457)
            # Assigning a type to the variable 'if_clause' (line 436)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'if_clause', for_loop_var_1458)
            # SSA begins for a for statement (line 436)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to write(...): (line 437)
            # Processing the call arguments (line 437)
            str_1461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 23), 'str', ' if ')
            # Processing the call keyword arguments (line 437)
            kwargs_1462 = {}
            # Getting the type of 'self' (line 437)
            self_1459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 437)
            write_1460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 12), self_1459, 'write')
            # Calling write(args, kwargs) (line 437)
            write_call_result_1463 = invoke(stypy.reporting.localization.Localization(__file__, 437, 12), write_1460, *[str_1461], **kwargs_1462)
            
            
            # Call to visit(...): (line 438)
            # Processing the call arguments (line 438)
            # Getting the type of 'if_clause' (line 438)
            if_clause_1466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 23), 'if_clause', False)
            # Processing the call keyword arguments (line 438)
            kwargs_1467 = {}
            # Getting the type of 'self' (line 438)
            self_1464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 438)
            visit_1465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 12), self_1464, 'visit')
            # Calling visit(args, kwargs) (line 438)
            visit_call_result_1468 = invoke(stypy.reporting.localization.Localization(__file__, 438, 12), visit_1465, *[if_clause_1466], **kwargs_1467)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'visit_comprehension(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_comprehension' in the type store
        # Getting the type of 'stypy_return_type' (line 431)
        stypy_return_type_1469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1469)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_comprehension'
        return stypy_return_type_1469


    @norecursion
    def visit_IfExp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_IfExp'
        module_type_store = module_type_store.open_function_context('visit_IfExp', 440, 4, False)
        # Assigning a type to the variable 'self' (line 441)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 441)
        # Processing the call arguments (line 441)
        str_1472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, 19), 'str', '(')
        # Processing the call keyword arguments (line 441)
        kwargs_1473 = {}
        # Getting the type of 'self' (line 441)
        self_1470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 441)
        write_1471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 441, 8), self_1470, 'write')
        # Calling write(args, kwargs) (line 441)
        write_call_result_1474 = invoke(stypy.reporting.localization.Localization(__file__, 441, 8), write_1471, *[str_1472], **kwargs_1473)
        
        
        # Call to visit(...): (line 442)
        # Processing the call arguments (line 442)
        # Getting the type of 't' (line 442)
        t_1477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 19), 't', False)
        # Obtaining the member 'body' of a type (line 442)
        body_1478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 19), t_1477, 'body')
        # Processing the call keyword arguments (line 442)
        kwargs_1479 = {}
        # Getting the type of 'self' (line 442)
        self_1475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 442)
        visit_1476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 8), self_1475, 'visit')
        # Calling visit(args, kwargs) (line 442)
        visit_call_result_1480 = invoke(stypy.reporting.localization.Localization(__file__, 442, 8), visit_1476, *[body_1478], **kwargs_1479)
        
        
        # Call to write(...): (line 443)
        # Processing the call arguments (line 443)
        str_1483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, 19), 'str', ' if ')
        # Processing the call keyword arguments (line 443)
        kwargs_1484 = {}
        # Getting the type of 'self' (line 443)
        self_1481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 443)
        write_1482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 443, 8), self_1481, 'write')
        # Calling write(args, kwargs) (line 443)
        write_call_result_1485 = invoke(stypy.reporting.localization.Localization(__file__, 443, 8), write_1482, *[str_1483], **kwargs_1484)
        
        
        # Call to visit(...): (line 444)
        # Processing the call arguments (line 444)
        # Getting the type of 't' (line 444)
        t_1488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 19), 't', False)
        # Obtaining the member 'test' of a type (line 444)
        test_1489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 19), t_1488, 'test')
        # Processing the call keyword arguments (line 444)
        kwargs_1490 = {}
        # Getting the type of 'self' (line 444)
        self_1486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 444)
        visit_1487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 8), self_1486, 'visit')
        # Calling visit(args, kwargs) (line 444)
        visit_call_result_1491 = invoke(stypy.reporting.localization.Localization(__file__, 444, 8), visit_1487, *[test_1489], **kwargs_1490)
        
        
        # Call to write(...): (line 445)
        # Processing the call arguments (line 445)
        str_1494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 19), 'str', ' else ')
        # Processing the call keyword arguments (line 445)
        kwargs_1495 = {}
        # Getting the type of 'self' (line 445)
        self_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 445)
        write_1493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 8), self_1492, 'write')
        # Calling write(args, kwargs) (line 445)
        write_call_result_1496 = invoke(stypy.reporting.localization.Localization(__file__, 445, 8), write_1493, *[str_1494], **kwargs_1495)
        
        
        # Call to visit(...): (line 446)
        # Processing the call arguments (line 446)
        # Getting the type of 't' (line 446)
        t_1499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 19), 't', False)
        # Obtaining the member 'orelse' of a type (line 446)
        orelse_1500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 19), t_1499, 'orelse')
        # Processing the call keyword arguments (line 446)
        kwargs_1501 = {}
        # Getting the type of 'self' (line 446)
        self_1497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 446)
        visit_1498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 8), self_1497, 'visit')
        # Calling visit(args, kwargs) (line 446)
        visit_call_result_1502 = invoke(stypy.reporting.localization.Localization(__file__, 446, 8), visit_1498, *[orelse_1500], **kwargs_1501)
        
        
        # Call to write(...): (line 447)
        # Processing the call arguments (line 447)
        str_1505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 19), 'str', ')')
        # Processing the call keyword arguments (line 447)
        kwargs_1506 = {}
        # Getting the type of 'self' (line 447)
        self_1503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 447)
        write_1504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 8), self_1503, 'write')
        # Calling write(args, kwargs) (line 447)
        write_call_result_1507 = invoke(stypy.reporting.localization.Localization(__file__, 447, 8), write_1504, *[str_1505], **kwargs_1506)
        
        
        # ################# End of 'visit_IfExp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_IfExp' in the type store
        # Getting the type of 'stypy_return_type' (line 440)
        stypy_return_type_1508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 440, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1508)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_IfExp'
        return stypy_return_type_1508


    @norecursion
    def visit_Set(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Set'
        module_type_store = module_type_store.open_function_context('visit_Set', 449, 4, False)
        # Assigning a type to the variable 'self' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 4), 'self', type_of_self)
        
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
        # Getting the type of 't' (line 450)
        t_1509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 16), 't')
        # Obtaining the member 'elts' of a type (line 450)
        elts_1510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 16), t_1509, 'elts')
        assert_1511 = elts_1510
        # Assigning a type to the variable 'assert_1511' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'assert_1511', elts_1510)
        
        # Call to write(...): (line 451)
        # Processing the call arguments (line 451)
        str_1514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 19), 'str', '{')
        # Processing the call keyword arguments (line 451)
        kwargs_1515 = {}
        # Getting the type of 'self' (line 451)
        self_1512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 451)
        write_1513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), self_1512, 'write')
        # Calling write(args, kwargs) (line 451)
        write_call_result_1516 = invoke(stypy.reporting.localization.Localization(__file__, 451, 8), write_1513, *[str_1514], **kwargs_1515)
        
        
        # Call to interleave(...): (line 452)
        # Processing the call arguments (line 452)

        @norecursion
        def _stypy_temp_lambda_6(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_6'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_6', 452, 19, True)
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

            
            # Call to write(...): (line 452)
            # Processing the call arguments (line 452)
            str_1520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 38), 'str', ', ')
            # Processing the call keyword arguments (line 452)
            kwargs_1521 = {}
            # Getting the type of 'self' (line 452)
            self_1518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 452)
            write_1519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 27), self_1518, 'write')
            # Calling write(args, kwargs) (line 452)
            write_call_result_1522 = invoke(stypy.reporting.localization.Localization(__file__, 452, 27), write_1519, *[str_1520], **kwargs_1521)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 452)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 19), 'stypy_return_type', write_call_result_1522)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_6' in the type store
            # Getting the type of 'stypy_return_type' (line 452)
            stypy_return_type_1523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1523)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_6'
            return stypy_return_type_1523

        # Assigning a type to the variable '_stypy_temp_lambda_6' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 19), '_stypy_temp_lambda_6', _stypy_temp_lambda_6)
        # Getting the type of '_stypy_temp_lambda_6' (line 452)
        _stypy_temp_lambda_6_1524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 19), '_stypy_temp_lambda_6')
        # Getting the type of 'self' (line 452)
        self_1525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 452)
        visit_1526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 45), self_1525, 'visit')
        # Getting the type of 't' (line 452)
        t_1527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 57), 't', False)
        # Obtaining the member 'elts' of a type (line 452)
        elts_1528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 57), t_1527, 'elts')
        # Processing the call keyword arguments (line 452)
        kwargs_1529 = {}
        # Getting the type of 'interleave' (line 452)
        interleave_1517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 452)
        interleave_call_result_1530 = invoke(stypy.reporting.localization.Localization(__file__, 452, 8), interleave_1517, *[_stypy_temp_lambda_6_1524, visit_1526, elts_1528], **kwargs_1529)
        
        
        # Call to write(...): (line 453)
        # Processing the call arguments (line 453)
        str_1533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 453, 19), 'str', '}')
        # Processing the call keyword arguments (line 453)
        kwargs_1534 = {}
        # Getting the type of 'self' (line 453)
        self_1531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 453)
        write_1532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 453, 8), self_1531, 'write')
        # Calling write(args, kwargs) (line 453)
        write_call_result_1535 = invoke(stypy.reporting.localization.Localization(__file__, 453, 8), write_1532, *[str_1533], **kwargs_1534)
        
        
        # ################# End of 'visit_Set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Set' in the type store
        # Getting the type of 'stypy_return_type' (line 449)
        stypy_return_type_1536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1536)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Set'
        return stypy_return_type_1536


    @norecursion
    def visit_Dict(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Dict'
        module_type_store = module_type_store.open_function_context('visit_Dict', 455, 4, False)
        # Assigning a type to the variable 'self' (line 456)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 456)
        # Processing the call arguments (line 456)
        str_1539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 19), 'str', '{')
        # Processing the call keyword arguments (line 456)
        kwargs_1540 = {}
        # Getting the type of 'self' (line 456)
        self_1537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 456)
        write_1538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 8), self_1537, 'write')
        # Calling write(args, kwargs) (line 456)
        write_call_result_1541 = invoke(stypy.reporting.localization.Localization(__file__, 456, 8), write_1538, *[str_1539], **kwargs_1540)
        

        @norecursion
        def write_pair(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'write_pair'
            module_type_store = module_type_store.open_function_context('write_pair', 458, 8, True)
            
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

            
            # Assigning a Name to a Tuple (line 459):
            
            # Assigning a Subscript to a Name (line 459):
            
            # Obtaining the type of the subscript
            int_1542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 12), 'int')
            # Getting the type of 'pair' (line 459)
            pair_1543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 21), 'pair')
            # Obtaining the member '__getitem__' of a type (line 459)
            getitem___1544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 12), pair_1543, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 459)
            subscript_call_result_1545 = invoke(stypy.reporting.localization.Localization(__file__, 459, 12), getitem___1544, int_1542)
            
            # Assigning a type to the variable 'tuple_var_assignment_1' (line 459)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'tuple_var_assignment_1', subscript_call_result_1545)
            
            # Assigning a Subscript to a Name (line 459):
            
            # Obtaining the type of the subscript
            int_1546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 459, 12), 'int')
            # Getting the type of 'pair' (line 459)
            pair_1547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 21), 'pair')
            # Obtaining the member '__getitem__' of a type (line 459)
            getitem___1548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 459, 12), pair_1547, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 459)
            subscript_call_result_1549 = invoke(stypy.reporting.localization.Localization(__file__, 459, 12), getitem___1548, int_1546)
            
            # Assigning a type to the variable 'tuple_var_assignment_2' (line 459)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'tuple_var_assignment_2', subscript_call_result_1549)
            
            # Assigning a Name to a Name (line 459):
            # Getting the type of 'tuple_var_assignment_1' (line 459)
            tuple_var_assignment_1_1550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'tuple_var_assignment_1')
            # Assigning a type to the variable 'k' (line 459)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 13), 'k', tuple_var_assignment_1_1550)
            
            # Assigning a Name to a Name (line 459):
            # Getting the type of 'tuple_var_assignment_2' (line 459)
            tuple_var_assignment_2_1551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 12), 'tuple_var_assignment_2')
            # Assigning a type to the variable 'v' (line 459)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 16), 'v', tuple_var_assignment_2_1551)
            
            # Call to visit(...): (line 460)
            # Processing the call arguments (line 460)
            # Getting the type of 'k' (line 460)
            k_1554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'k', False)
            # Processing the call keyword arguments (line 460)
            kwargs_1555 = {}
            # Getting the type of 'self' (line 460)
            self_1552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 460)
            visit_1553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 12), self_1552, 'visit')
            # Calling visit(args, kwargs) (line 460)
            visit_call_result_1556 = invoke(stypy.reporting.localization.Localization(__file__, 460, 12), visit_1553, *[k_1554], **kwargs_1555)
            
            
            # Call to write(...): (line 461)
            # Processing the call arguments (line 461)
            str_1559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 23), 'str', ': ')
            # Processing the call keyword arguments (line 461)
            kwargs_1560 = {}
            # Getting the type of 'self' (line 461)
            self_1557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 461)
            write_1558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 12), self_1557, 'write')
            # Calling write(args, kwargs) (line 461)
            write_call_result_1561 = invoke(stypy.reporting.localization.Localization(__file__, 461, 12), write_1558, *[str_1559], **kwargs_1560)
            
            
            # Call to visit(...): (line 462)
            # Processing the call arguments (line 462)
            # Getting the type of 'v' (line 462)
            v_1564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 23), 'v', False)
            # Processing the call keyword arguments (line 462)
            kwargs_1565 = {}
            # Getting the type of 'self' (line 462)
            self_1562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 462)
            visit_1563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 12), self_1562, 'visit')
            # Calling visit(args, kwargs) (line 462)
            visit_call_result_1566 = invoke(stypy.reporting.localization.Localization(__file__, 462, 12), visit_1563, *[v_1564], **kwargs_1565)
            
            
            # ################# End of 'write_pair(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'write_pair' in the type store
            # Getting the type of 'stypy_return_type' (line 458)
            stypy_return_type_1567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1567)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'write_pair'
            return stypy_return_type_1567

        # Assigning a type to the variable 'write_pair' (line 458)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 458, 8), 'write_pair', write_pair)
        
        # Call to interleave(...): (line 464)
        # Processing the call arguments (line 464)

        @norecursion
        def _stypy_temp_lambda_7(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_7'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_7', 464, 24, True)
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

            
            # Call to write(...): (line 464)
            # Processing the call arguments (line 464)
            str_1572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 43), 'str', ', ')
            # Processing the call keyword arguments (line 464)
            kwargs_1573 = {}
            # Getting the type of 'self' (line 464)
            self_1570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 32), 'self', False)
            # Obtaining the member 'write' of a type (line 464)
            write_1571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 32), self_1570, 'write')
            # Calling write(args, kwargs) (line 464)
            write_call_result_1574 = invoke(stypy.reporting.localization.Localization(__file__, 464, 32), write_1571, *[str_1572], **kwargs_1573)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 464)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), 'stypy_return_type', write_call_result_1574)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_7' in the type store
            # Getting the type of 'stypy_return_type' (line 464)
            stypy_return_type_1575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1575)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_7'
            return stypy_return_type_1575

        # Assigning a type to the variable '_stypy_temp_lambda_7' (line 464)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), '_stypy_temp_lambda_7', _stypy_temp_lambda_7)
        # Getting the type of '_stypy_temp_lambda_7' (line 464)
        _stypy_temp_lambda_7_1576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 24), '_stypy_temp_lambda_7')
        # Getting the type of 'write_pair' (line 464)
        write_pair_1577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 50), 'write_pair', False)
        
        # Call to zip(...): (line 464)
        # Processing the call arguments (line 464)
        # Getting the type of 't' (line 464)
        t_1579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 66), 't', False)
        # Obtaining the member 'keys' of a type (line 464)
        keys_1580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 66), t_1579, 'keys')
        # Getting the type of 't' (line 464)
        t_1581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 74), 't', False)
        # Obtaining the member 'values' of a type (line 464)
        values_1582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 74), t_1581, 'values')
        # Processing the call keyword arguments (line 464)
        kwargs_1583 = {}
        # Getting the type of 'zip' (line 464)
        zip_1578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 62), 'zip', False)
        # Calling zip(args, kwargs) (line 464)
        zip_call_result_1584 = invoke(stypy.reporting.localization.Localization(__file__, 464, 62), zip_1578, *[keys_1580, values_1582], **kwargs_1583)
        
        # Processing the call keyword arguments (line 464)
        kwargs_1585 = {}
        # Getting the type of 'self' (line 464)
        self_1568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 8), 'self', False)
        # Obtaining the member 'interleave' of a type (line 464)
        interleave_1569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 464, 8), self_1568, 'interleave')
        # Calling interleave(args, kwargs) (line 464)
        interleave_call_result_1586 = invoke(stypy.reporting.localization.Localization(__file__, 464, 8), interleave_1569, *[_stypy_temp_lambda_7_1576, write_pair_1577, zip_call_result_1584], **kwargs_1585)
        
        
        # Call to write(...): (line 465)
        # Processing the call arguments (line 465)
        str_1589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 465, 19), 'str', '}')
        # Processing the call keyword arguments (line 465)
        kwargs_1590 = {}
        # Getting the type of 'self' (line 465)
        self_1587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 465)
        write_1588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 465, 8), self_1587, 'write')
        # Calling write(args, kwargs) (line 465)
        write_call_result_1591 = invoke(stypy.reporting.localization.Localization(__file__, 465, 8), write_1588, *[str_1589], **kwargs_1590)
        
        
        # ################# End of 'visit_Dict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Dict' in the type store
        # Getting the type of 'stypy_return_type' (line 455)
        stypy_return_type_1592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1592)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Dict'
        return stypy_return_type_1592


    @norecursion
    def visit_Tuple(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Tuple'
        module_type_store = module_type_store.open_function_context('visit_Tuple', 467, 4, False)
        # Assigning a type to the variable 'self' (line 468)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 468)
        # Processing the call arguments (line 468)
        str_1595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 19), 'str', '(')
        # Processing the call keyword arguments (line 468)
        kwargs_1596 = {}
        # Getting the type of 'self' (line 468)
        self_1593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 468)
        write_1594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), self_1593, 'write')
        # Calling write(args, kwargs) (line 468)
        write_call_result_1597 = invoke(stypy.reporting.localization.Localization(__file__, 468, 8), write_1594, *[str_1595], **kwargs_1596)
        
        
        
        # Call to len(...): (line 469)
        # Processing the call arguments (line 469)
        # Getting the type of 't' (line 469)
        t_1599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 15), 't', False)
        # Obtaining the member 'elts' of a type (line 469)
        elts_1600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 15), t_1599, 'elts')
        # Processing the call keyword arguments (line 469)
        kwargs_1601 = {}
        # Getting the type of 'len' (line 469)
        len_1598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 11), 'len', False)
        # Calling len(args, kwargs) (line 469)
        len_call_result_1602 = invoke(stypy.reporting.localization.Localization(__file__, 469, 11), len_1598, *[elts_1600], **kwargs_1601)
        
        int_1603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 469, 26), 'int')
        # Applying the binary operator '==' (line 469)
        result_eq_1604 = python_operator(stypy.reporting.localization.Localization(__file__, 469, 11), '==', len_call_result_1602, int_1603)
        
        # Testing if the type of an if condition is none (line 469)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 469, 8), result_eq_1604):
            
            # Call to interleave(...): (line 474)
            # Processing the call arguments (line 474)

            @norecursion
            def _stypy_temp_lambda_8(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_8'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_8', 474, 23, True)
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

                
                # Call to write(...): (line 474)
                # Processing the call arguments (line 474)
                str_1625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 42), 'str', ', ')
                # Processing the call keyword arguments (line 474)
                kwargs_1626 = {}
                # Getting the type of 'self' (line 474)
                self_1623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 31), 'self', False)
                # Obtaining the member 'write' of a type (line 474)
                write_1624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 31), self_1623, 'write')
                # Calling write(args, kwargs) (line 474)
                write_call_result_1627 = invoke(stypy.reporting.localization.Localization(__file__, 474, 31), write_1624, *[str_1625], **kwargs_1626)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 474)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 23), 'stypy_return_type', write_call_result_1627)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_8' in the type store
                # Getting the type of 'stypy_return_type' (line 474)
                stypy_return_type_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 23), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_1628)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_8'
                return stypy_return_type_1628

            # Assigning a type to the variable '_stypy_temp_lambda_8' (line 474)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 23), '_stypy_temp_lambda_8', _stypy_temp_lambda_8)
            # Getting the type of '_stypy_temp_lambda_8' (line 474)
            _stypy_temp_lambda_8_1629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 23), '_stypy_temp_lambda_8')
            # Getting the type of 'self' (line 474)
            self_1630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 49), 'self', False)
            # Obtaining the member 'visit' of a type (line 474)
            visit_1631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 49), self_1630, 'visit')
            # Getting the type of 't' (line 474)
            t_1632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 61), 't', False)
            # Obtaining the member 'elts' of a type (line 474)
            elts_1633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 61), t_1632, 'elts')
            # Processing the call keyword arguments (line 474)
            kwargs_1634 = {}
            # Getting the type of 'interleave' (line 474)
            interleave_1622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 12), 'interleave', False)
            # Calling interleave(args, kwargs) (line 474)
            interleave_call_result_1635 = invoke(stypy.reporting.localization.Localization(__file__, 474, 12), interleave_1622, *[_stypy_temp_lambda_8_1629, visit_1631, elts_1633], **kwargs_1634)
            
        else:
            
            # Testing the type of an if condition (line 469)
            if_condition_1605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 469, 8), result_eq_1604)
            # Assigning a type to the variable 'if_condition_1605' (line 469)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 8), 'if_condition_1605', if_condition_1605)
            # SSA begins for if statement (line 469)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Tuple (line 470):
            
            # Assigning a Subscript to a Name (line 470):
            
            # Obtaining the type of the subscript
            int_1606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 470, 12), 'int')
            # Getting the type of 't' (line 470)
            t_1607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 21), 't')
            # Obtaining the member 'elts' of a type (line 470)
            elts_1608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 21), t_1607, 'elts')
            # Obtaining the member '__getitem__' of a type (line 470)
            getitem___1609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 12), elts_1608, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 470)
            subscript_call_result_1610 = invoke(stypy.reporting.localization.Localization(__file__, 470, 12), getitem___1609, int_1606)
            
            # Assigning a type to the variable 'tuple_var_assignment_3' (line 470)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'tuple_var_assignment_3', subscript_call_result_1610)
            
            # Assigning a Name to a Name (line 470):
            # Getting the type of 'tuple_var_assignment_3' (line 470)
            tuple_var_assignment_3_1611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 12), 'tuple_var_assignment_3')
            # Assigning a type to the variable 'elt' (line 470)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 13), 'elt', tuple_var_assignment_3_1611)
            
            # Call to visit(...): (line 471)
            # Processing the call arguments (line 471)
            # Getting the type of 'elt' (line 471)
            elt_1614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 23), 'elt', False)
            # Processing the call keyword arguments (line 471)
            kwargs_1615 = {}
            # Getting the type of 'self' (line 471)
            self_1612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 471, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 471)
            visit_1613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 471, 12), self_1612, 'visit')
            # Calling visit(args, kwargs) (line 471)
            visit_call_result_1616 = invoke(stypy.reporting.localization.Localization(__file__, 471, 12), visit_1613, *[elt_1614], **kwargs_1615)
            
            
            # Call to write(...): (line 472)
            # Processing the call arguments (line 472)
            str_1619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 472, 23), 'str', ',')
            # Processing the call keyword arguments (line 472)
            kwargs_1620 = {}
            # Getting the type of 'self' (line 472)
            self_1617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 472)
            write_1618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 472, 12), self_1617, 'write')
            # Calling write(args, kwargs) (line 472)
            write_call_result_1621 = invoke(stypy.reporting.localization.Localization(__file__, 472, 12), write_1618, *[str_1619], **kwargs_1620)
            
            # SSA branch for the else part of an if statement (line 469)
            module_type_store.open_ssa_branch('else')
            
            # Call to interleave(...): (line 474)
            # Processing the call arguments (line 474)

            @norecursion
            def _stypy_temp_lambda_8(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_8'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_8', 474, 23, True)
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

                
                # Call to write(...): (line 474)
                # Processing the call arguments (line 474)
                str_1625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 474, 42), 'str', ', ')
                # Processing the call keyword arguments (line 474)
                kwargs_1626 = {}
                # Getting the type of 'self' (line 474)
                self_1623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 31), 'self', False)
                # Obtaining the member 'write' of a type (line 474)
                write_1624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 31), self_1623, 'write')
                # Calling write(args, kwargs) (line 474)
                write_call_result_1627 = invoke(stypy.reporting.localization.Localization(__file__, 474, 31), write_1624, *[str_1625], **kwargs_1626)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 474)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 23), 'stypy_return_type', write_call_result_1627)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_8' in the type store
                # Getting the type of 'stypy_return_type' (line 474)
                stypy_return_type_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 23), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_1628)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_8'
                return stypy_return_type_1628

            # Assigning a type to the variable '_stypy_temp_lambda_8' (line 474)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 23), '_stypy_temp_lambda_8', _stypy_temp_lambda_8)
            # Getting the type of '_stypy_temp_lambda_8' (line 474)
            _stypy_temp_lambda_8_1629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 23), '_stypy_temp_lambda_8')
            # Getting the type of 'self' (line 474)
            self_1630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 49), 'self', False)
            # Obtaining the member 'visit' of a type (line 474)
            visit_1631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 49), self_1630, 'visit')
            # Getting the type of 't' (line 474)
            t_1632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 61), 't', False)
            # Obtaining the member 'elts' of a type (line 474)
            elts_1633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 61), t_1632, 'elts')
            # Processing the call keyword arguments (line 474)
            kwargs_1634 = {}
            # Getting the type of 'interleave' (line 474)
            interleave_1622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 12), 'interleave', False)
            # Calling interleave(args, kwargs) (line 474)
            interleave_call_result_1635 = invoke(stypy.reporting.localization.Localization(__file__, 474, 12), interleave_1622, *[_stypy_temp_lambda_8_1629, visit_1631, elts_1633], **kwargs_1634)
            
            # SSA join for if statement (line 469)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 475)
        # Processing the call arguments (line 475)
        str_1638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 19), 'str', ')')
        # Processing the call keyword arguments (line 475)
        kwargs_1639 = {}
        # Getting the type of 'self' (line 475)
        self_1636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 475)
        write_1637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 8), self_1636, 'write')
        # Calling write(args, kwargs) (line 475)
        write_call_result_1640 = invoke(stypy.reporting.localization.Localization(__file__, 475, 8), write_1637, *[str_1638], **kwargs_1639)
        
        
        # ################# End of 'visit_Tuple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Tuple' in the type store
        # Getting the type of 'stypy_return_type' (line 467)
        stypy_return_type_1641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1641)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Tuple'
        return stypy_return_type_1641

    
    # Assigning a Dict to a Name (line 477):

    @norecursion
    def visit_UnaryOp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_UnaryOp'
        module_type_store = module_type_store.open_function_context('visit_UnaryOp', 479, 4, False)
        # Assigning a type to the variable 'self' (line 480)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 480)
        # Processing the call arguments (line 480)
        str_1644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 19), 'str', '(')
        # Processing the call keyword arguments (line 480)
        kwargs_1645 = {}
        # Getting the type of 'self' (line 480)
        self_1642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 480)
        write_1643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), self_1642, 'write')
        # Calling write(args, kwargs) (line 480)
        write_call_result_1646 = invoke(stypy.reporting.localization.Localization(__file__, 480, 8), write_1643, *[str_1644], **kwargs_1645)
        
        
        # Call to write(...): (line 481)
        # Processing the call arguments (line 481)
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 481)
        t_1649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 29), 't', False)
        # Obtaining the member 'op' of a type (line 481)
        op_1650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 29), t_1649, 'op')
        # Obtaining the member '__class__' of a type (line 481)
        class___1651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 29), op_1650, '__class__')
        # Obtaining the member '__name__' of a type (line 481)
        name___1652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 29), class___1651, '__name__')
        # Getting the type of 'self' (line 481)
        self_1653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 19), 'self', False)
        # Obtaining the member 'unop' of a type (line 481)
        unop_1654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 19), self_1653, 'unop')
        # Obtaining the member '__getitem__' of a type (line 481)
        getitem___1655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 19), unop_1654, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 481)
        subscript_call_result_1656 = invoke(stypy.reporting.localization.Localization(__file__, 481, 19), getitem___1655, name___1652)
        
        # Processing the call keyword arguments (line 481)
        kwargs_1657 = {}
        # Getting the type of 'self' (line 481)
        self_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 481)
        write_1648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 8), self_1647, 'write')
        # Calling write(args, kwargs) (line 481)
        write_call_result_1658 = invoke(stypy.reporting.localization.Localization(__file__, 481, 8), write_1648, *[subscript_call_result_1656], **kwargs_1657)
        
        
        # Call to write(...): (line 482)
        # Processing the call arguments (line 482)
        str_1661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 19), 'str', ' ')
        # Processing the call keyword arguments (line 482)
        kwargs_1662 = {}
        # Getting the type of 'self' (line 482)
        self_1659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 482)
        write_1660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 8), self_1659, 'write')
        # Calling write(args, kwargs) (line 482)
        write_call_result_1663 = invoke(stypy.reporting.localization.Localization(__file__, 482, 8), write_1660, *[str_1661], **kwargs_1662)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 't' (line 488)
        t_1665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 22), 't', False)
        # Obtaining the member 'op' of a type (line 488)
        op_1666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 22), t_1665, 'op')
        # Getting the type of 'ast' (line 488)
        ast_1667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 28), 'ast', False)
        # Obtaining the member 'USub' of a type (line 488)
        USub_1668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 28), ast_1667, 'USub')
        # Processing the call keyword arguments (line 488)
        kwargs_1669 = {}
        # Getting the type of 'isinstance' (line 488)
        isinstance_1664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 488)
        isinstance_call_result_1670 = invoke(stypy.reporting.localization.Localization(__file__, 488, 11), isinstance_1664, *[op_1666, USub_1668], **kwargs_1669)
        
        
        # Call to isinstance(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 't' (line 488)
        t_1672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 53), 't', False)
        # Obtaining the member 'operand' of a type (line 488)
        operand_1673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 53), t_1672, 'operand')
        # Getting the type of 'ast' (line 488)
        ast_1674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 64), 'ast', False)
        # Obtaining the member 'Num' of a type (line 488)
        Num_1675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 64), ast_1674, 'Num')
        # Processing the call keyword arguments (line 488)
        kwargs_1676 = {}
        # Getting the type of 'isinstance' (line 488)
        isinstance_1671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 42), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 488)
        isinstance_call_result_1677 = invoke(stypy.reporting.localization.Localization(__file__, 488, 42), isinstance_1671, *[operand_1673, Num_1675], **kwargs_1676)
        
        # Applying the binary operator 'and' (line 488)
        result_and_keyword_1678 = python_operator(stypy.reporting.localization.Localization(__file__, 488, 11), 'and', isinstance_call_result_1670, isinstance_call_result_1677)
        
        # Testing if the type of an if condition is none (line 488)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 488, 8), result_and_keyword_1678):
            
            # Call to visit(...): (line 493)
            # Processing the call arguments (line 493)
            # Getting the type of 't' (line 493)
            t_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 23), 't', False)
            # Obtaining the member 'operand' of a type (line 493)
            operand_1699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 23), t_1698, 'operand')
            # Processing the call keyword arguments (line 493)
            kwargs_1700 = {}
            # Getting the type of 'self' (line 493)
            self_1696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 493)
            visit_1697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 12), self_1696, 'visit')
            # Calling visit(args, kwargs) (line 493)
            visit_call_result_1701 = invoke(stypy.reporting.localization.Localization(__file__, 493, 12), visit_1697, *[operand_1699], **kwargs_1700)
            
        else:
            
            # Testing the type of an if condition (line 488)
            if_condition_1679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 488, 8), result_and_keyword_1678)
            # Assigning a type to the variable 'if_condition_1679' (line 488)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'if_condition_1679', if_condition_1679)
            # SSA begins for if statement (line 488)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 489)
            # Processing the call arguments (line 489)
            str_1682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 23), 'str', '(')
            # Processing the call keyword arguments (line 489)
            kwargs_1683 = {}
            # Getting the type of 'self' (line 489)
            self_1680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 489)
            write_1681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 12), self_1680, 'write')
            # Calling write(args, kwargs) (line 489)
            write_call_result_1684 = invoke(stypy.reporting.localization.Localization(__file__, 489, 12), write_1681, *[str_1682], **kwargs_1683)
            
            
            # Call to visit(...): (line 490)
            # Processing the call arguments (line 490)
            # Getting the type of 't' (line 490)
            t_1687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 23), 't', False)
            # Obtaining the member 'operand' of a type (line 490)
            operand_1688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 23), t_1687, 'operand')
            # Processing the call keyword arguments (line 490)
            kwargs_1689 = {}
            # Getting the type of 'self' (line 490)
            self_1685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 490)
            visit_1686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 12), self_1685, 'visit')
            # Calling visit(args, kwargs) (line 490)
            visit_call_result_1690 = invoke(stypy.reporting.localization.Localization(__file__, 490, 12), visit_1686, *[operand_1688], **kwargs_1689)
            
            
            # Call to write(...): (line 491)
            # Processing the call arguments (line 491)
            str_1693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 23), 'str', ')')
            # Processing the call keyword arguments (line 491)
            kwargs_1694 = {}
            # Getting the type of 'self' (line 491)
            self_1691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 491)
            write_1692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 12), self_1691, 'write')
            # Calling write(args, kwargs) (line 491)
            write_call_result_1695 = invoke(stypy.reporting.localization.Localization(__file__, 491, 12), write_1692, *[str_1693], **kwargs_1694)
            
            # SSA branch for the else part of an if statement (line 488)
            module_type_store.open_ssa_branch('else')
            
            # Call to visit(...): (line 493)
            # Processing the call arguments (line 493)
            # Getting the type of 't' (line 493)
            t_1698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 23), 't', False)
            # Obtaining the member 'operand' of a type (line 493)
            operand_1699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 23), t_1698, 'operand')
            # Processing the call keyword arguments (line 493)
            kwargs_1700 = {}
            # Getting the type of 'self' (line 493)
            self_1696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 493)
            visit_1697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 493, 12), self_1696, 'visit')
            # Calling visit(args, kwargs) (line 493)
            visit_call_result_1701 = invoke(stypy.reporting.localization.Localization(__file__, 493, 12), visit_1697, *[operand_1699], **kwargs_1700)
            
            # SSA join for if statement (line 488)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 494)
        # Processing the call arguments (line 494)
        str_1704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 19), 'str', ')')
        # Processing the call keyword arguments (line 494)
        kwargs_1705 = {}
        # Getting the type of 'self' (line 494)
        self_1702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 494, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 494)
        write_1703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 494, 8), self_1702, 'write')
        # Calling write(args, kwargs) (line 494)
        write_call_result_1706 = invoke(stypy.reporting.localization.Localization(__file__, 494, 8), write_1703, *[str_1704], **kwargs_1705)
        
        
        # ################# End of 'visit_UnaryOp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_UnaryOp' in the type store
        # Getting the type of 'stypy_return_type' (line 479)
        stypy_return_type_1707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1707)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_UnaryOp'
        return stypy_return_type_1707

    
    # Assigning a Dict to a Name (line 496):

    @norecursion
    def visit_BinOp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_BinOp'
        module_type_store = module_type_store.open_function_context('visit_BinOp', 500, 4, False)
        # Assigning a type to the variable 'self' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 501)
        # Processing the call arguments (line 501)
        str_1710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 501, 19), 'str', '(')
        # Processing the call keyword arguments (line 501)
        kwargs_1711 = {}
        # Getting the type of 'self' (line 501)
        self_1708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 501)
        write_1709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 8), self_1708, 'write')
        # Calling write(args, kwargs) (line 501)
        write_call_result_1712 = invoke(stypy.reporting.localization.Localization(__file__, 501, 8), write_1709, *[str_1710], **kwargs_1711)
        
        
        # Call to visit(...): (line 502)
        # Processing the call arguments (line 502)
        # Getting the type of 't' (line 502)
        t_1715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 19), 't', False)
        # Obtaining the member 'left' of a type (line 502)
        left_1716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 19), t_1715, 'left')
        # Processing the call keyword arguments (line 502)
        kwargs_1717 = {}
        # Getting the type of 'self' (line 502)
        self_1713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 502)
        visit_1714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 8), self_1713, 'visit')
        # Calling visit(args, kwargs) (line 502)
        visit_call_result_1718 = invoke(stypy.reporting.localization.Localization(__file__, 502, 8), visit_1714, *[left_1716], **kwargs_1717)
        
        
        # Call to write(...): (line 503)
        # Processing the call arguments (line 503)
        str_1721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 19), 'str', ' ')
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 503)
        t_1722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 36), 't', False)
        # Obtaining the member 'op' of a type (line 503)
        op_1723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 36), t_1722, 'op')
        # Obtaining the member '__class__' of a type (line 503)
        class___1724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 36), op_1723, '__class__')
        # Obtaining the member '__name__' of a type (line 503)
        name___1725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 36), class___1724, '__name__')
        # Getting the type of 'self' (line 503)
        self_1726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 25), 'self', False)
        # Obtaining the member 'binop' of a type (line 503)
        binop_1727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 25), self_1726, 'binop')
        # Obtaining the member '__getitem__' of a type (line 503)
        getitem___1728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 25), binop_1727, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 503)
        subscript_call_result_1729 = invoke(stypy.reporting.localization.Localization(__file__, 503, 25), getitem___1728, name___1725)
        
        # Applying the binary operator '+' (line 503)
        result_add_1730 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 19), '+', str_1721, subscript_call_result_1729)
        
        str_1731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 503, 63), 'str', ' ')
        # Applying the binary operator '+' (line 503)
        result_add_1732 = python_operator(stypy.reporting.localization.Localization(__file__, 503, 61), '+', result_add_1730, str_1731)
        
        # Processing the call keyword arguments (line 503)
        kwargs_1733 = {}
        # Getting the type of 'self' (line 503)
        self_1719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 503, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 503)
        write_1720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 503, 8), self_1719, 'write')
        # Calling write(args, kwargs) (line 503)
        write_call_result_1734 = invoke(stypy.reporting.localization.Localization(__file__, 503, 8), write_1720, *[result_add_1732], **kwargs_1733)
        
        
        # Call to visit(...): (line 504)
        # Processing the call arguments (line 504)
        # Getting the type of 't' (line 504)
        t_1737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 19), 't', False)
        # Obtaining the member 'right' of a type (line 504)
        right_1738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 19), t_1737, 'right')
        # Processing the call keyword arguments (line 504)
        kwargs_1739 = {}
        # Getting the type of 'self' (line 504)
        self_1735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 504)
        visit_1736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 8), self_1735, 'visit')
        # Calling visit(args, kwargs) (line 504)
        visit_call_result_1740 = invoke(stypy.reporting.localization.Localization(__file__, 504, 8), visit_1736, *[right_1738], **kwargs_1739)
        
        
        # Call to write(...): (line 505)
        # Processing the call arguments (line 505)
        str_1743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 505, 19), 'str', ')')
        # Processing the call keyword arguments (line 505)
        kwargs_1744 = {}
        # Getting the type of 'self' (line 505)
        self_1741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 505, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 505)
        write_1742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 505, 8), self_1741, 'write')
        # Calling write(args, kwargs) (line 505)
        write_call_result_1745 = invoke(stypy.reporting.localization.Localization(__file__, 505, 8), write_1742, *[str_1743], **kwargs_1744)
        
        
        # ################# End of 'visit_BinOp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_BinOp' in the type store
        # Getting the type of 'stypy_return_type' (line 500)
        stypy_return_type_1746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1746)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_BinOp'
        return stypy_return_type_1746

    
    # Assigning a Dict to a Name (line 507):

    @norecursion
    def visit_Compare(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Compare'
        module_type_store = module_type_store.open_function_context('visit_Compare', 510, 4, False)
        # Assigning a type to the variable 'self' (line 511)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 511)
        # Processing the call arguments (line 511)
        str_1749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 511, 19), 'str', '(')
        # Processing the call keyword arguments (line 511)
        kwargs_1750 = {}
        # Getting the type of 'self' (line 511)
        self_1747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 511)
        write_1748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 511, 8), self_1747, 'write')
        # Calling write(args, kwargs) (line 511)
        write_call_result_1751 = invoke(stypy.reporting.localization.Localization(__file__, 511, 8), write_1748, *[str_1749], **kwargs_1750)
        
        
        # Call to visit(...): (line 512)
        # Processing the call arguments (line 512)
        # Getting the type of 't' (line 512)
        t_1754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 19), 't', False)
        # Obtaining the member 'left' of a type (line 512)
        left_1755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 19), t_1754, 'left')
        # Processing the call keyword arguments (line 512)
        kwargs_1756 = {}
        # Getting the type of 'self' (line 512)
        self_1752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 512)
        visit_1753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 512, 8), self_1752, 'visit')
        # Calling visit(args, kwargs) (line 512)
        visit_call_result_1757 = invoke(stypy.reporting.localization.Localization(__file__, 512, 8), visit_1753, *[left_1755], **kwargs_1756)
        
        
        
        # Call to zip(...): (line 513)
        # Processing the call arguments (line 513)
        # Getting the type of 't' (line 513)
        t_1759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 24), 't', False)
        # Obtaining the member 'ops' of a type (line 513)
        ops_1760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 24), t_1759, 'ops')
        # Getting the type of 't' (line 513)
        t_1761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 31), 't', False)
        # Obtaining the member 'comparators' of a type (line 513)
        comparators_1762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 31), t_1761, 'comparators')
        # Processing the call keyword arguments (line 513)
        kwargs_1763 = {}
        # Getting the type of 'zip' (line 513)
        zip_1758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 513)
        zip_call_result_1764 = invoke(stypy.reporting.localization.Localization(__file__, 513, 20), zip_1758, *[ops_1760, comparators_1762], **kwargs_1763)
        
        # Assigning a type to the variable 'zip_call_result_1764' (line 513)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'zip_call_result_1764', zip_call_result_1764)
        # Testing if the for loop is going to be iterated (line 513)
        # Testing the type of a for loop iterable (line 513)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 513, 8), zip_call_result_1764)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 513, 8), zip_call_result_1764):
            # Getting the type of the for loop variable (line 513)
            for_loop_var_1765 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 513, 8), zip_call_result_1764)
            # Assigning a type to the variable 'o' (line 513)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'o', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 8), for_loop_var_1765, 2, 0))
            # Assigning a type to the variable 'e' (line 513)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'e', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 513, 8), for_loop_var_1765, 2, 1))
            # SSA begins for a for statement (line 513)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to write(...): (line 514)
            # Processing the call arguments (line 514)
            str_1768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 23), 'str', ' ')
            
            # Obtaining the type of the subscript
            # Getting the type of 'o' (line 514)
            o_1769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 41), 'o', False)
            # Obtaining the member '__class__' of a type (line 514)
            class___1770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 41), o_1769, '__class__')
            # Obtaining the member '__name__' of a type (line 514)
            name___1771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 41), class___1770, '__name__')
            # Getting the type of 'self' (line 514)
            self_1772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 29), 'self', False)
            # Obtaining the member 'cmpops' of a type (line 514)
            cmpops_1773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 29), self_1772, 'cmpops')
            # Obtaining the member '__getitem__' of a type (line 514)
            getitem___1774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 29), cmpops_1773, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 514)
            subscript_call_result_1775 = invoke(stypy.reporting.localization.Localization(__file__, 514, 29), getitem___1774, name___1771)
            
            # Applying the binary operator '+' (line 514)
            result_add_1776 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 23), '+', str_1768, subscript_call_result_1775)
            
            str_1777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 514, 65), 'str', ' ')
            # Applying the binary operator '+' (line 514)
            result_add_1778 = python_operator(stypy.reporting.localization.Localization(__file__, 514, 63), '+', result_add_1776, str_1777)
            
            # Processing the call keyword arguments (line 514)
            kwargs_1779 = {}
            # Getting the type of 'self' (line 514)
            self_1766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 514)
            write_1767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 514, 12), self_1766, 'write')
            # Calling write(args, kwargs) (line 514)
            write_call_result_1780 = invoke(stypy.reporting.localization.Localization(__file__, 514, 12), write_1767, *[result_add_1778], **kwargs_1779)
            
            
            # Call to visit(...): (line 515)
            # Processing the call arguments (line 515)
            # Getting the type of 'e' (line 515)
            e_1783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 23), 'e', False)
            # Processing the call keyword arguments (line 515)
            kwargs_1784 = {}
            # Getting the type of 'self' (line 515)
            self_1781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 515)
            visit_1782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 12), self_1781, 'visit')
            # Calling visit(args, kwargs) (line 515)
            visit_call_result_1785 = invoke(stypy.reporting.localization.Localization(__file__, 515, 12), visit_1782, *[e_1783], **kwargs_1784)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 516)
        # Processing the call arguments (line 516)
        str_1788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 516, 19), 'str', ')')
        # Processing the call keyword arguments (line 516)
        kwargs_1789 = {}
        # Getting the type of 'self' (line 516)
        self_1786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 516)
        write_1787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 8), self_1786, 'write')
        # Calling write(args, kwargs) (line 516)
        write_call_result_1790 = invoke(stypy.reporting.localization.Localization(__file__, 516, 8), write_1787, *[str_1788], **kwargs_1789)
        
        
        # ################# End of 'visit_Compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Compare' in the type store
        # Getting the type of 'stypy_return_type' (line 510)
        stypy_return_type_1791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1791)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Compare'
        return stypy_return_type_1791

    
    # Assigning a Dict to a Name (line 518):

    @norecursion
    def visit_BoolOp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_BoolOp'
        module_type_store = module_type_store.open_function_context('visit_BoolOp', 520, 4, False)
        # Assigning a type to the variable 'self' (line 521)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 521)
        # Processing the call arguments (line 521)
        str_1794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 19), 'str', '(')
        # Processing the call keyword arguments (line 521)
        kwargs_1795 = {}
        # Getting the type of 'self' (line 521)
        self_1792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 521)
        write_1793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 521, 8), self_1792, 'write')
        # Calling write(args, kwargs) (line 521)
        write_call_result_1796 = invoke(stypy.reporting.localization.Localization(__file__, 521, 8), write_1793, *[str_1794], **kwargs_1795)
        
        
        # Assigning a BinOp to a Name (line 522):
        
        # Assigning a BinOp to a Name (line 522):
        str_1797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 522, 12), 'str', ' %s ')
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 522)
        t_1798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 34), 't')
        # Obtaining the member 'op' of a type (line 522)
        op_1799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 34), t_1798, 'op')
        # Obtaining the member '__class__' of a type (line 522)
        class___1800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 34), op_1799, '__class__')
        # Getting the type of 'self' (line 522)
        self_1801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 21), 'self')
        # Obtaining the member 'boolops' of a type (line 522)
        boolops_1802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 21), self_1801, 'boolops')
        # Obtaining the member '__getitem__' of a type (line 522)
        getitem___1803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 522, 21), boolops_1802, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 522)
        subscript_call_result_1804 = invoke(stypy.reporting.localization.Localization(__file__, 522, 21), getitem___1803, class___1800)
        
        # Applying the binary operator '%' (line 522)
        result_mod_1805 = python_operator(stypy.reporting.localization.Localization(__file__, 522, 12), '%', str_1797, subscript_call_result_1804)
        
        # Assigning a type to the variable 's' (line 522)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 522, 8), 's', result_mod_1805)
        
        # Call to interleave(...): (line 523)
        # Processing the call arguments (line 523)

        @norecursion
        def _stypy_temp_lambda_9(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_9'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_9', 523, 19, True)
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

            
            # Call to write(...): (line 523)
            # Processing the call arguments (line 523)
            # Getting the type of 's' (line 523)
            s_1809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 38), 's', False)
            # Processing the call keyword arguments (line 523)
            kwargs_1810 = {}
            # Getting the type of 'self' (line 523)
            self_1807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 523)
            write_1808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 27), self_1807, 'write')
            # Calling write(args, kwargs) (line 523)
            write_call_result_1811 = invoke(stypy.reporting.localization.Localization(__file__, 523, 27), write_1808, *[s_1809], **kwargs_1810)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 523)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 19), 'stypy_return_type', write_call_result_1811)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_9' in the type store
            # Getting the type of 'stypy_return_type' (line 523)
            stypy_return_type_1812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1812)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_9'
            return stypy_return_type_1812

        # Assigning a type to the variable '_stypy_temp_lambda_9' (line 523)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 523, 19), '_stypy_temp_lambda_9', _stypy_temp_lambda_9)
        # Getting the type of '_stypy_temp_lambda_9' (line 523)
        _stypy_temp_lambda_9_1813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 19), '_stypy_temp_lambda_9')
        # Getting the type of 'self' (line 523)
        self_1814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 42), 'self', False)
        # Obtaining the member 'visit' of a type (line 523)
        visit_1815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 42), self_1814, 'visit')
        # Getting the type of 't' (line 523)
        t_1816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 54), 't', False)
        # Obtaining the member 'values' of a type (line 523)
        values_1817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 54), t_1816, 'values')
        # Processing the call keyword arguments (line 523)
        kwargs_1818 = {}
        # Getting the type of 'interleave' (line 523)
        interleave_1806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 523)
        interleave_call_result_1819 = invoke(stypy.reporting.localization.Localization(__file__, 523, 8), interleave_1806, *[_stypy_temp_lambda_9_1813, visit_1815, values_1817], **kwargs_1818)
        
        
        # Call to write(...): (line 524)
        # Processing the call arguments (line 524)
        str_1822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 19), 'str', ')')
        # Processing the call keyword arguments (line 524)
        kwargs_1823 = {}
        # Getting the type of 'self' (line 524)
        self_1820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 524)
        write_1821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 8), self_1820, 'write')
        # Calling write(args, kwargs) (line 524)
        write_call_result_1824 = invoke(stypy.reporting.localization.Localization(__file__, 524, 8), write_1821, *[str_1822], **kwargs_1823)
        
        
        # ################# End of 'visit_BoolOp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_BoolOp' in the type store
        # Getting the type of 'stypy_return_type' (line 520)
        stypy_return_type_1825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1825)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_BoolOp'
        return stypy_return_type_1825


    @norecursion
    def visit_Attribute(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Attribute'
        module_type_store = module_type_store.open_function_context('visit_Attribute', 526, 4, False)
        # Assigning a type to the variable 'self' (line 527)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 4), 'self', type_of_self)
        
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

        
        # Call to visit(...): (line 527)
        # Processing the call arguments (line 527)
        # Getting the type of 't' (line 527)
        t_1828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 19), 't', False)
        # Obtaining the member 'value' of a type (line 527)
        value_1829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 19), t_1828, 'value')
        # Processing the call keyword arguments (line 527)
        kwargs_1830 = {}
        # Getting the type of 'self' (line 527)
        self_1826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 527)
        visit_1827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 527, 8), self_1826, 'visit')
        # Calling visit(args, kwargs) (line 527)
        visit_call_result_1831 = invoke(stypy.reporting.localization.Localization(__file__, 527, 8), visit_1827, *[value_1829], **kwargs_1830)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 531)
        # Processing the call arguments (line 531)
        # Getting the type of 't' (line 531)
        t_1833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 22), 't', False)
        # Obtaining the member 'value' of a type (line 531)
        value_1834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 22), t_1833, 'value')
        # Getting the type of 'ast' (line 531)
        ast_1835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 31), 'ast', False)
        # Obtaining the member 'Num' of a type (line 531)
        Num_1836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 31), ast_1835, 'Num')
        # Processing the call keyword arguments (line 531)
        kwargs_1837 = {}
        # Getting the type of 'isinstance' (line 531)
        isinstance_1832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 531)
        isinstance_call_result_1838 = invoke(stypy.reporting.localization.Localization(__file__, 531, 11), isinstance_1832, *[value_1834, Num_1836], **kwargs_1837)
        
        
        # Call to isinstance(...): (line 531)
        # Processing the call arguments (line 531)
        # Getting the type of 't' (line 531)
        t_1840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 55), 't', False)
        # Obtaining the member 'value' of a type (line 531)
        value_1841 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 55), t_1840, 'value')
        # Obtaining the member 'n' of a type (line 531)
        n_1842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 55), value_1841, 'n')
        # Getting the type of 'int' (line 531)
        int_1843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 66), 'int', False)
        # Processing the call keyword arguments (line 531)
        kwargs_1844 = {}
        # Getting the type of 'isinstance' (line 531)
        isinstance_1839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 44), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 531)
        isinstance_call_result_1845 = invoke(stypy.reporting.localization.Localization(__file__, 531, 44), isinstance_1839, *[n_1842, int_1843], **kwargs_1844)
        
        # Applying the binary operator 'and' (line 531)
        result_and_keyword_1846 = python_operator(stypy.reporting.localization.Localization(__file__, 531, 11), 'and', isinstance_call_result_1838, isinstance_call_result_1845)
        
        # Testing if the type of an if condition is none (line 531)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 531, 8), result_and_keyword_1846):
            pass
        else:
            
            # Testing the type of an if condition (line 531)
            if_condition_1847 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 531, 8), result_and_keyword_1846)
            # Assigning a type to the variable 'if_condition_1847' (line 531)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 531, 8), 'if_condition_1847', if_condition_1847)
            # SSA begins for if statement (line 531)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 532)
            # Processing the call arguments (line 532)
            str_1850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 532, 23), 'str', ' ')
            # Processing the call keyword arguments (line 532)
            kwargs_1851 = {}
            # Getting the type of 'self' (line 532)
            self_1848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 532)
            write_1849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 12), self_1848, 'write')
            # Calling write(args, kwargs) (line 532)
            write_call_result_1852 = invoke(stypy.reporting.localization.Localization(__file__, 532, 12), write_1849, *[str_1850], **kwargs_1851)
            
            # SSA join for if statement (line 531)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 533)
        # Processing the call arguments (line 533)
        str_1855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 533, 19), 'str', '.')
        # Processing the call keyword arguments (line 533)
        kwargs_1856 = {}
        # Getting the type of 'self' (line 533)
        self_1853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 533)
        write_1854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 533, 8), self_1853, 'write')
        # Calling write(args, kwargs) (line 533)
        write_call_result_1857 = invoke(stypy.reporting.localization.Localization(__file__, 533, 8), write_1854, *[str_1855], **kwargs_1856)
        
        
        # Call to write(...): (line 534)
        # Processing the call arguments (line 534)
        # Getting the type of 't' (line 534)
        t_1860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 19), 't', False)
        # Obtaining the member 'attr' of a type (line 534)
        attr_1861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 19), t_1860, 'attr')
        # Processing the call keyword arguments (line 534)
        kwargs_1862 = {}
        # Getting the type of 'self' (line 534)
        self_1858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 534)
        write_1859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 8), self_1858, 'write')
        # Calling write(args, kwargs) (line 534)
        write_call_result_1863 = invoke(stypy.reporting.localization.Localization(__file__, 534, 8), write_1859, *[attr_1861], **kwargs_1862)
        
        
        # ################# End of 'visit_Attribute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Attribute' in the type store
        # Getting the type of 'stypy_return_type' (line 526)
        stypy_return_type_1864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1864)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Attribute'
        return stypy_return_type_1864


    @norecursion
    def visit_Call(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Call'
        module_type_store = module_type_store.open_function_context('visit_Call', 536, 4, False)
        # Assigning a type to the variable 'self' (line 537)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 537, 4), 'self', type_of_self)
        
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

        
        # Call to visit(...): (line 537)
        # Processing the call arguments (line 537)
        # Getting the type of 't' (line 537)
        t_1867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 19), 't', False)
        # Obtaining the member 'func' of a type (line 537)
        func_1868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 19), t_1867, 'func')
        # Processing the call keyword arguments (line 537)
        kwargs_1869 = {}
        # Getting the type of 'self' (line 537)
        self_1865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 537)
        visit_1866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 8), self_1865, 'visit')
        # Calling visit(args, kwargs) (line 537)
        visit_call_result_1870 = invoke(stypy.reporting.localization.Localization(__file__, 537, 8), visit_1866, *[func_1868], **kwargs_1869)
        
        
        # Call to write(...): (line 538)
        # Processing the call arguments (line 538)
        str_1873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 538, 19), 'str', '(')
        # Processing the call keyword arguments (line 538)
        kwargs_1874 = {}
        # Getting the type of 'self' (line 538)
        self_1871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 538)
        write_1872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 8), self_1871, 'write')
        # Calling write(args, kwargs) (line 538)
        write_call_result_1875 = invoke(stypy.reporting.localization.Localization(__file__, 538, 8), write_1872, *[str_1873], **kwargs_1874)
        
        
        # Assigning a Name to a Name (line 539):
        
        # Assigning a Name to a Name (line 539):
        # Getting the type of 'False' (line 539)
        False_1876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 16), 'False')
        # Assigning a type to the variable 'comma' (line 539)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 8), 'comma', False_1876)
        
        # Getting the type of 't' (line 540)
        t_1877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 17), 't')
        # Obtaining the member 'args' of a type (line 540)
        args_1878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 17), t_1877, 'args')
        # Assigning a type to the variable 'args_1878' (line 540)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'args_1878', args_1878)
        # Testing if the for loop is going to be iterated (line 540)
        # Testing the type of a for loop iterable (line 540)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 540, 8), args_1878)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 540, 8), args_1878):
            # Getting the type of the for loop variable (line 540)
            for_loop_var_1879 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 540, 8), args_1878)
            # Assigning a type to the variable 'e' (line 540)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 8), 'e', for_loop_var_1879)
            # SSA begins for a for statement (line 540)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'comma' (line 541)
            comma_1880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 15), 'comma')
            # Testing if the type of an if condition is none (line 541)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 541, 12), comma_1880):
                
                # Assigning a Name to a Name (line 544):
                
                # Assigning a Name to a Name (line 544):
                # Getting the type of 'True' (line 544)
                True_1887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 24), 'True')
                # Assigning a type to the variable 'comma' (line 544)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'comma', True_1887)
            else:
                
                # Testing the type of an if condition (line 541)
                if_condition_1881 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 541, 12), comma_1880)
                # Assigning a type to the variable 'if_condition_1881' (line 541)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 12), 'if_condition_1881', if_condition_1881)
                # SSA begins for if statement (line 541)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 542)
                # Processing the call arguments (line 542)
                str_1884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 542, 27), 'str', ', ')
                # Processing the call keyword arguments (line 542)
                kwargs_1885 = {}
                # Getting the type of 'self' (line 542)
                self_1882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 542)
                write_1883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 16), self_1882, 'write')
                # Calling write(args, kwargs) (line 542)
                write_call_result_1886 = invoke(stypy.reporting.localization.Localization(__file__, 542, 16), write_1883, *[str_1884], **kwargs_1885)
                
                # SSA branch for the else part of an if statement (line 541)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 544):
                
                # Assigning a Name to a Name (line 544):
                # Getting the type of 'True' (line 544)
                True_1887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 24), 'True')
                # Assigning a type to the variable 'comma' (line 544)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 16), 'comma', True_1887)
                # SSA join for if statement (line 541)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to visit(...): (line 545)
            # Processing the call arguments (line 545)
            # Getting the type of 'e' (line 545)
            e_1890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 23), 'e', False)
            # Processing the call keyword arguments (line 545)
            kwargs_1891 = {}
            # Getting the type of 'self' (line 545)
            self_1888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 545)
            visit_1889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 12), self_1888, 'visit')
            # Calling visit(args, kwargs) (line 545)
            visit_call_result_1892 = invoke(stypy.reporting.localization.Localization(__file__, 545, 12), visit_1889, *[e_1890], **kwargs_1891)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 't' (line 546)
        t_1893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 17), 't')
        # Obtaining the member 'keywords' of a type (line 546)
        keywords_1894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 546, 17), t_1893, 'keywords')
        # Assigning a type to the variable 'keywords_1894' (line 546)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'keywords_1894', keywords_1894)
        # Testing if the for loop is going to be iterated (line 546)
        # Testing the type of a for loop iterable (line 546)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 546, 8), keywords_1894)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 546, 8), keywords_1894):
            # Getting the type of the for loop variable (line 546)
            for_loop_var_1895 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 546, 8), keywords_1894)
            # Assigning a type to the variable 'e' (line 546)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 8), 'e', for_loop_var_1895)
            # SSA begins for a for statement (line 546)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'comma' (line 547)
            comma_1896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 15), 'comma')
            # Testing if the type of an if condition is none (line 547)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 547, 12), comma_1896):
                
                # Assigning a Name to a Name (line 550):
                
                # Assigning a Name to a Name (line 550):
                # Getting the type of 'True' (line 550)
                True_1903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 24), 'True')
                # Assigning a type to the variable 'comma' (line 550)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 16), 'comma', True_1903)
            else:
                
                # Testing the type of an if condition (line 547)
                if_condition_1897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 547, 12), comma_1896)
                # Assigning a type to the variable 'if_condition_1897' (line 547)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 12), 'if_condition_1897', if_condition_1897)
                # SSA begins for if statement (line 547)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 548)
                # Processing the call arguments (line 548)
                str_1900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 548, 27), 'str', ', ')
                # Processing the call keyword arguments (line 548)
                kwargs_1901 = {}
                # Getting the type of 'self' (line 548)
                self_1898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 548, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 548)
                write_1899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 548, 16), self_1898, 'write')
                # Calling write(args, kwargs) (line 548)
                write_call_result_1902 = invoke(stypy.reporting.localization.Localization(__file__, 548, 16), write_1899, *[str_1900], **kwargs_1901)
                
                # SSA branch for the else part of an if statement (line 547)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 550):
                
                # Assigning a Name to a Name (line 550):
                # Getting the type of 'True' (line 550)
                True_1903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 24), 'True')
                # Assigning a type to the variable 'comma' (line 550)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 550, 16), 'comma', True_1903)
                # SSA join for if statement (line 547)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to visit(...): (line 551)
            # Processing the call arguments (line 551)
            # Getting the type of 'e' (line 551)
            e_1906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 23), 'e', False)
            # Processing the call keyword arguments (line 551)
            kwargs_1907 = {}
            # Getting the type of 'self' (line 551)
            self_1904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 551)
            visit_1905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 12), self_1904, 'visit')
            # Calling visit(args, kwargs) (line 551)
            visit_call_result_1908 = invoke(stypy.reporting.localization.Localization(__file__, 551, 12), visit_1905, *[e_1906], **kwargs_1907)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 't' (line 552)
        t_1909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 11), 't')
        # Obtaining the member 'starargs' of a type (line 552)
        starargs_1910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 11), t_1909, 'starargs')
        # Testing if the type of an if condition is none (line 552)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 552, 8), starargs_1910):
            pass
        else:
            
            # Testing the type of an if condition (line 552)
            if_condition_1911 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 552, 8), starargs_1910)
            # Assigning a type to the variable 'if_condition_1911' (line 552)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'if_condition_1911', if_condition_1911)
            # SSA begins for if statement (line 552)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'comma' (line 553)
            comma_1912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 553, 15), 'comma')
            # Testing if the type of an if condition is none (line 553)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 553, 12), comma_1912):
                
                # Assigning a Name to a Name (line 556):
                
                # Assigning a Name to a Name (line 556):
                # Getting the type of 'True' (line 556)
                True_1919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 24), 'True')
                # Assigning a type to the variable 'comma' (line 556)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'comma', True_1919)
            else:
                
                # Testing the type of an if condition (line 553)
                if_condition_1913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 553, 12), comma_1912)
                # Assigning a type to the variable 'if_condition_1913' (line 553)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 553, 12), 'if_condition_1913', if_condition_1913)
                # SSA begins for if statement (line 553)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 554)
                # Processing the call arguments (line 554)
                str_1916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 554, 27), 'str', ', ')
                # Processing the call keyword arguments (line 554)
                kwargs_1917 = {}
                # Getting the type of 'self' (line 554)
                self_1914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 554)
                write_1915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 554, 16), self_1914, 'write')
                # Calling write(args, kwargs) (line 554)
                write_call_result_1918 = invoke(stypy.reporting.localization.Localization(__file__, 554, 16), write_1915, *[str_1916], **kwargs_1917)
                
                # SSA branch for the else part of an if statement (line 553)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 556):
                
                # Assigning a Name to a Name (line 556):
                # Getting the type of 'True' (line 556)
                True_1919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 24), 'True')
                # Assigning a type to the variable 'comma' (line 556)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 556, 16), 'comma', True_1919)
                # SSA join for if statement (line 553)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 557)
            # Processing the call arguments (line 557)
            str_1922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 557, 23), 'str', '*')
            # Processing the call keyword arguments (line 557)
            kwargs_1923 = {}
            # Getting the type of 'self' (line 557)
            self_1920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 557)
            write_1921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 12), self_1920, 'write')
            # Calling write(args, kwargs) (line 557)
            write_call_result_1924 = invoke(stypy.reporting.localization.Localization(__file__, 557, 12), write_1921, *[str_1922], **kwargs_1923)
            
            
            # Call to visit(...): (line 558)
            # Processing the call arguments (line 558)
            # Getting the type of 't' (line 558)
            t_1927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 23), 't', False)
            # Obtaining the member 'starargs' of a type (line 558)
            starargs_1928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 23), t_1927, 'starargs')
            # Processing the call keyword arguments (line 558)
            kwargs_1929 = {}
            # Getting the type of 'self' (line 558)
            self_1925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 558)
            visit_1926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 12), self_1925, 'visit')
            # Calling visit(args, kwargs) (line 558)
            visit_call_result_1930 = invoke(stypy.reporting.localization.Localization(__file__, 558, 12), visit_1926, *[starargs_1928], **kwargs_1929)
            
            # SSA join for if statement (line 552)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 559)
        t_1931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 11), 't')
        # Obtaining the member 'kwargs' of a type (line 559)
        kwargs_1932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 11), t_1931, 'kwargs')
        # Testing if the type of an if condition is none (line 559)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 559, 8), kwargs_1932):
            pass
        else:
            
            # Testing the type of an if condition (line 559)
            if_condition_1933 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 559, 8), kwargs_1932)
            # Assigning a type to the variable 'if_condition_1933' (line 559)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 8), 'if_condition_1933', if_condition_1933)
            # SSA begins for if statement (line 559)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'comma' (line 560)
            comma_1934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 15), 'comma')
            # Testing if the type of an if condition is none (line 560)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 560, 12), comma_1934):
                
                # Assigning a Name to a Name (line 563):
                
                # Assigning a Name to a Name (line 563):
                # Getting the type of 'True' (line 563)
                True_1941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 24), 'True')
                # Assigning a type to the variable 'comma' (line 563)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 16), 'comma', True_1941)
            else:
                
                # Testing the type of an if condition (line 560)
                if_condition_1935 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 560, 12), comma_1934)
                # Assigning a type to the variable 'if_condition_1935' (line 560)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 12), 'if_condition_1935', if_condition_1935)
                # SSA begins for if statement (line 560)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 561)
                # Processing the call arguments (line 561)
                str_1938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 561, 27), 'str', ', ')
                # Processing the call keyword arguments (line 561)
                kwargs_1939 = {}
                # Getting the type of 'self' (line 561)
                self_1936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 561)
                write_1937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 16), self_1936, 'write')
                # Calling write(args, kwargs) (line 561)
                write_call_result_1940 = invoke(stypy.reporting.localization.Localization(__file__, 561, 16), write_1937, *[str_1938], **kwargs_1939)
                
                # SSA branch for the else part of an if statement (line 560)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 563):
                
                # Assigning a Name to a Name (line 563):
                # Getting the type of 'True' (line 563)
                True_1941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 24), 'True')
                # Assigning a type to the variable 'comma' (line 563)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 16), 'comma', True_1941)
                # SSA join for if statement (line 560)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 564)
            # Processing the call arguments (line 564)
            str_1944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 564, 23), 'str', '**')
            # Processing the call keyword arguments (line 564)
            kwargs_1945 = {}
            # Getting the type of 'self' (line 564)
            self_1942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 564)
            write_1943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 564, 12), self_1942, 'write')
            # Calling write(args, kwargs) (line 564)
            write_call_result_1946 = invoke(stypy.reporting.localization.Localization(__file__, 564, 12), write_1943, *[str_1944], **kwargs_1945)
            
            
            # Call to visit(...): (line 565)
            # Processing the call arguments (line 565)
            # Getting the type of 't' (line 565)
            t_1949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 23), 't', False)
            # Obtaining the member 'kwargs' of a type (line 565)
            kwargs_1950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 23), t_1949, 'kwargs')
            # Processing the call keyword arguments (line 565)
            kwargs_1951 = {}
            # Getting the type of 'self' (line 565)
            self_1947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 565)
            visit_1948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 12), self_1947, 'visit')
            # Calling visit(args, kwargs) (line 565)
            visit_call_result_1952 = invoke(stypy.reporting.localization.Localization(__file__, 565, 12), visit_1948, *[kwargs_1950], **kwargs_1951)
            
            # SSA join for if statement (line 559)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 566)
        # Processing the call arguments (line 566)
        str_1955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 566, 19), 'str', ')')
        # Processing the call keyword arguments (line 566)
        kwargs_1956 = {}
        # Getting the type of 'self' (line 566)
        self_1953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 566, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 566)
        write_1954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 566, 8), self_1953, 'write')
        # Calling write(args, kwargs) (line 566)
        write_call_result_1957 = invoke(stypy.reporting.localization.Localization(__file__, 566, 8), write_1954, *[str_1955], **kwargs_1956)
        
        
        # ################# End of 'visit_Call(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Call' in the type store
        # Getting the type of 'stypy_return_type' (line 536)
        stypy_return_type_1958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1958)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Call'
        return stypy_return_type_1958


    @norecursion
    def visit_Subscript(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Subscript'
        module_type_store = module_type_store.open_function_context('visit_Subscript', 568, 4, False)
        # Assigning a type to the variable 'self' (line 569)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 569, 4), 'self', type_of_self)
        
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

        
        # Call to visit(...): (line 569)
        # Processing the call arguments (line 569)
        # Getting the type of 't' (line 569)
        t_1961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 19), 't', False)
        # Obtaining the member 'value' of a type (line 569)
        value_1962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 19), t_1961, 'value')
        # Processing the call keyword arguments (line 569)
        kwargs_1963 = {}
        # Getting the type of 'self' (line 569)
        self_1959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 569)
        visit_1960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 8), self_1959, 'visit')
        # Calling visit(args, kwargs) (line 569)
        visit_call_result_1964 = invoke(stypy.reporting.localization.Localization(__file__, 569, 8), visit_1960, *[value_1962], **kwargs_1963)
        
        
        # Call to write(...): (line 570)
        # Processing the call arguments (line 570)
        str_1967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 19), 'str', '[')
        # Processing the call keyword arguments (line 570)
        kwargs_1968 = {}
        # Getting the type of 'self' (line 570)
        self_1965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 570)
        write_1966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), self_1965, 'write')
        # Calling write(args, kwargs) (line 570)
        write_call_result_1969 = invoke(stypy.reporting.localization.Localization(__file__, 570, 8), write_1966, *[str_1967], **kwargs_1968)
        
        
        # Call to visit(...): (line 571)
        # Processing the call arguments (line 571)
        # Getting the type of 't' (line 571)
        t_1972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 19), 't', False)
        # Obtaining the member 'slice' of a type (line 571)
        slice_1973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 19), t_1972, 'slice')
        # Processing the call keyword arguments (line 571)
        kwargs_1974 = {}
        # Getting the type of 'self' (line 571)
        self_1970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 571)
        visit_1971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 8), self_1970, 'visit')
        # Calling visit(args, kwargs) (line 571)
        visit_call_result_1975 = invoke(stypy.reporting.localization.Localization(__file__, 571, 8), visit_1971, *[slice_1973], **kwargs_1974)
        
        
        # Call to write(...): (line 572)
        # Processing the call arguments (line 572)
        str_1978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 572, 19), 'str', ']')
        # Processing the call keyword arguments (line 572)
        kwargs_1979 = {}
        # Getting the type of 'self' (line 572)
        self_1976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 572)
        write_1977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 8), self_1976, 'write')
        # Calling write(args, kwargs) (line 572)
        write_call_result_1980 = invoke(stypy.reporting.localization.Localization(__file__, 572, 8), write_1977, *[str_1978], **kwargs_1979)
        
        
        # ################# End of 'visit_Subscript(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Subscript' in the type store
        # Getting the type of 'stypy_return_type' (line 568)
        stypy_return_type_1981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1981)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Subscript'
        return stypy_return_type_1981


    @norecursion
    def visit_Ellipsis(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Ellipsis'
        module_type_store = module_type_store.open_function_context('visit_Ellipsis', 575, 4, False)
        # Assigning a type to the variable 'self' (line 576)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 576, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 576)
        # Processing the call arguments (line 576)
        str_1984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 576, 19), 'str', '...')
        # Processing the call keyword arguments (line 576)
        kwargs_1985 = {}
        # Getting the type of 'self' (line 576)
        self_1982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 576, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 576)
        write_1983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 576, 8), self_1982, 'write')
        # Calling write(args, kwargs) (line 576)
        write_call_result_1986 = invoke(stypy.reporting.localization.Localization(__file__, 576, 8), write_1983, *[str_1984], **kwargs_1985)
        
        
        # ################# End of 'visit_Ellipsis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Ellipsis' in the type store
        # Getting the type of 'stypy_return_type' (line 575)
        stypy_return_type_1987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1987)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Ellipsis'
        return stypy_return_type_1987


    @norecursion
    def visit_Index(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Index'
        module_type_store = module_type_store.open_function_context('visit_Index', 578, 4, False)
        # Assigning a type to the variable 'self' (line 579)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 579, 4), 'self', type_of_self)
        
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

        
        # Call to visit(...): (line 579)
        # Processing the call arguments (line 579)
        # Getting the type of 't' (line 579)
        t_1990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 19), 't', False)
        # Obtaining the member 'value' of a type (line 579)
        value_1991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 19), t_1990, 'value')
        # Processing the call keyword arguments (line 579)
        kwargs_1992 = {}
        # Getting the type of 'self' (line 579)
        self_1988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 579, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 579)
        visit_1989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 579, 8), self_1988, 'visit')
        # Calling visit(args, kwargs) (line 579)
        visit_call_result_1993 = invoke(stypy.reporting.localization.Localization(__file__, 579, 8), visit_1989, *[value_1991], **kwargs_1992)
        
        
        # ################# End of 'visit_Index(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Index' in the type store
        # Getting the type of 'stypy_return_type' (line 578)
        stypy_return_type_1994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1994)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Index'
        return stypy_return_type_1994


    @norecursion
    def visit_Slice(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Slice'
        module_type_store = module_type_store.open_function_context('visit_Slice', 581, 4, False)
        # Assigning a type to the variable 'self' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 4), 'self', type_of_self)
        
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

        # Getting the type of 't' (line 582)
        t_1995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 11), 't')
        # Obtaining the member 'lower' of a type (line 582)
        lower_1996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 582, 11), t_1995, 'lower')
        # Testing if the type of an if condition is none (line 582)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 582, 8), lower_1996):
            pass
        else:
            
            # Testing the type of an if condition (line 582)
            if_condition_1997 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 582, 8), lower_1996)
            # Assigning a type to the variable 'if_condition_1997' (line 582)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'if_condition_1997', if_condition_1997)
            # SSA begins for if statement (line 582)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to visit(...): (line 583)
            # Processing the call arguments (line 583)
            # Getting the type of 't' (line 583)
            t_2000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 23), 't', False)
            # Obtaining the member 'lower' of a type (line 583)
            lower_2001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 23), t_2000, 'lower')
            # Processing the call keyword arguments (line 583)
            kwargs_2002 = {}
            # Getting the type of 'self' (line 583)
            self_1998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 583, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 583)
            visit_1999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 583, 12), self_1998, 'visit')
            # Calling visit(args, kwargs) (line 583)
            visit_call_result_2003 = invoke(stypy.reporting.localization.Localization(__file__, 583, 12), visit_1999, *[lower_2001], **kwargs_2002)
            
            # SSA join for if statement (line 582)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 584)
        # Processing the call arguments (line 584)
        str_2006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 19), 'str', ':')
        # Processing the call keyword arguments (line 584)
        kwargs_2007 = {}
        # Getting the type of 'self' (line 584)
        self_2004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 584)
        write_2005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 8), self_2004, 'write')
        # Calling write(args, kwargs) (line 584)
        write_call_result_2008 = invoke(stypy.reporting.localization.Localization(__file__, 584, 8), write_2005, *[str_2006], **kwargs_2007)
        
        # Getting the type of 't' (line 585)
        t_2009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 11), 't')
        # Obtaining the member 'upper' of a type (line 585)
        upper_2010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 11), t_2009, 'upper')
        # Testing if the type of an if condition is none (line 585)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 585, 8), upper_2010):
            pass
        else:
            
            # Testing the type of an if condition (line 585)
            if_condition_2011 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 585, 8), upper_2010)
            # Assigning a type to the variable 'if_condition_2011' (line 585)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'if_condition_2011', if_condition_2011)
            # SSA begins for if statement (line 585)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to visit(...): (line 586)
            # Processing the call arguments (line 586)
            # Getting the type of 't' (line 586)
            t_2014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 23), 't', False)
            # Obtaining the member 'upper' of a type (line 586)
            upper_2015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 23), t_2014, 'upper')
            # Processing the call keyword arguments (line 586)
            kwargs_2016 = {}
            # Getting the type of 'self' (line 586)
            self_2012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 586)
            visit_2013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 586, 12), self_2012, 'visit')
            # Calling visit(args, kwargs) (line 586)
            visit_call_result_2017 = invoke(stypy.reporting.localization.Localization(__file__, 586, 12), visit_2013, *[upper_2015], **kwargs_2016)
            
            # SSA join for if statement (line 585)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 587)
        t_2018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 11), 't')
        # Obtaining the member 'step' of a type (line 587)
        step_2019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 587, 11), t_2018, 'step')
        # Testing if the type of an if condition is none (line 587)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 587, 8), step_2019):
            pass
        else:
            
            # Testing the type of an if condition (line 587)
            if_condition_2020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 587, 8), step_2019)
            # Assigning a type to the variable 'if_condition_2020' (line 587)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 8), 'if_condition_2020', if_condition_2020)
            # SSA begins for if statement (line 587)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 588)
            # Processing the call arguments (line 588)
            str_2023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 588, 23), 'str', ':')
            # Processing the call keyword arguments (line 588)
            kwargs_2024 = {}
            # Getting the type of 'self' (line 588)
            self_2021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 588, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 588)
            write_2022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 588, 12), self_2021, 'write')
            # Calling write(args, kwargs) (line 588)
            write_call_result_2025 = invoke(stypy.reporting.localization.Localization(__file__, 588, 12), write_2022, *[str_2023], **kwargs_2024)
            
            
            # Call to visit(...): (line 589)
            # Processing the call arguments (line 589)
            # Getting the type of 't' (line 589)
            t_2028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 23), 't', False)
            # Obtaining the member 'step' of a type (line 589)
            step_2029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 23), t_2028, 'step')
            # Processing the call keyword arguments (line 589)
            kwargs_2030 = {}
            # Getting the type of 'self' (line 589)
            self_2026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 589)
            visit_2027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 12), self_2026, 'visit')
            # Calling visit(args, kwargs) (line 589)
            visit_call_result_2031 = invoke(stypy.reporting.localization.Localization(__file__, 589, 12), visit_2027, *[step_2029], **kwargs_2030)
            
            # SSA join for if statement (line 587)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Slice(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Slice' in the type store
        # Getting the type of 'stypy_return_type' (line 581)
        stypy_return_type_2032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2032)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Slice'
        return stypy_return_type_2032


    @norecursion
    def visit_ExtSlice(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_ExtSlice'
        module_type_store = module_type_store.open_function_context('visit_ExtSlice', 591, 4, False)
        # Assigning a type to the variable 'self' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 4), 'self', type_of_self)
        
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

        
        # Call to interleave(...): (line 592)
        # Processing the call arguments (line 592)

        @norecursion
        def _stypy_temp_lambda_10(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_10'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_10', 592, 19, True)
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

            
            # Call to write(...): (line 592)
            # Processing the call arguments (line 592)
            str_2036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 38), 'str', ', ')
            # Processing the call keyword arguments (line 592)
            kwargs_2037 = {}
            # Getting the type of 'self' (line 592)
            self_2034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 592)
            write_2035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 27), self_2034, 'write')
            # Calling write(args, kwargs) (line 592)
            write_call_result_2038 = invoke(stypy.reporting.localization.Localization(__file__, 592, 27), write_2035, *[str_2036], **kwargs_2037)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 592)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 19), 'stypy_return_type', write_call_result_2038)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_10' in the type store
            # Getting the type of 'stypy_return_type' (line 592)
            stypy_return_type_2039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_2039)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_10'
            return stypy_return_type_2039

        # Assigning a type to the variable '_stypy_temp_lambda_10' (line 592)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 592, 19), '_stypy_temp_lambda_10', _stypy_temp_lambda_10)
        # Getting the type of '_stypy_temp_lambda_10' (line 592)
        _stypy_temp_lambda_10_2040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 19), '_stypy_temp_lambda_10')
        # Getting the type of 'self' (line 592)
        self_2041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 592)
        visit_2042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 45), self_2041, 'visit')
        # Getting the type of 't' (line 592)
        t_2043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 57), 't', False)
        # Obtaining the member 'dims' of a type (line 592)
        dims_2044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 57), t_2043, 'dims')
        # Processing the call keyword arguments (line 592)
        kwargs_2045 = {}
        # Getting the type of 'interleave' (line 592)
        interleave_2033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 592)
        interleave_call_result_2046 = invoke(stypy.reporting.localization.Localization(__file__, 592, 8), interleave_2033, *[_stypy_temp_lambda_10_2040, visit_2042, dims_2044], **kwargs_2045)
        
        
        # ################# End of 'visit_ExtSlice(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ExtSlice' in the type store
        # Getting the type of 'stypy_return_type' (line 591)
        stypy_return_type_2047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2047)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ExtSlice'
        return stypy_return_type_2047


    @norecursion
    def visit_arguments(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_arguments'
        module_type_store = module_type_store.open_function_context('visit_arguments', 595, 4, False)
        # Assigning a type to the variable 'self' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 4), 'self', type_of_self)
        
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

        
        # Assigning a Name to a Name (line 596):
        
        # Assigning a Name to a Name (line 596):
        # Getting the type of 'True' (line 596)
        True_2048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 16), 'True')
        # Assigning a type to the variable 'first' (line 596)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'first', True_2048)
        
        # Assigning a BinOp to a Name (line 598):
        
        # Assigning a BinOp to a Name (line 598):
        
        # Obtaining an instance of the builtin type 'list' (line 598)
        list_2049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 598, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 598)
        # Adding element type (line 598)
        # Getting the type of 'None' (line 598)
        None_2050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 20), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 598, 19), list_2049, None_2050)
        
        
        # Call to len(...): (line 598)
        # Processing the call arguments (line 598)
        # Getting the type of 't' (line 598)
        t_2052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 33), 't', False)
        # Obtaining the member 'args' of a type (line 598)
        args_2053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 33), t_2052, 'args')
        # Processing the call keyword arguments (line 598)
        kwargs_2054 = {}
        # Getting the type of 'len' (line 598)
        len_2051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 29), 'len', False)
        # Calling len(args, kwargs) (line 598)
        len_call_result_2055 = invoke(stypy.reporting.localization.Localization(__file__, 598, 29), len_2051, *[args_2053], **kwargs_2054)
        
        
        # Call to len(...): (line 598)
        # Processing the call arguments (line 598)
        # Getting the type of 't' (line 598)
        t_2057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 47), 't', False)
        # Obtaining the member 'defaults' of a type (line 598)
        defaults_2058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 47), t_2057, 'defaults')
        # Processing the call keyword arguments (line 598)
        kwargs_2059 = {}
        # Getting the type of 'len' (line 598)
        len_2056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 43), 'len', False)
        # Calling len(args, kwargs) (line 598)
        len_call_result_2060 = invoke(stypy.reporting.localization.Localization(__file__, 598, 43), len_2056, *[defaults_2058], **kwargs_2059)
        
        # Applying the binary operator '-' (line 598)
        result_sub_2061 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 29), '-', len_call_result_2055, len_call_result_2060)
        
        # Applying the binary operator '*' (line 598)
        result_mul_2062 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 19), '*', list_2049, result_sub_2061)
        
        # Getting the type of 't' (line 598)
        t_2063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 62), 't')
        # Obtaining the member 'defaults' of a type (line 598)
        defaults_2064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 598, 62), t_2063, 'defaults')
        # Applying the binary operator '+' (line 598)
        result_add_2065 = python_operator(stypy.reporting.localization.Localization(__file__, 598, 19), '+', result_mul_2062, defaults_2064)
        
        # Assigning a type to the variable 'defaults' (line 598)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 8), 'defaults', result_add_2065)
        
        
        # Call to zip(...): (line 599)
        # Processing the call arguments (line 599)
        # Getting the type of 't' (line 599)
        t_2067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 24), 't', False)
        # Obtaining the member 'args' of a type (line 599)
        args_2068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 599, 24), t_2067, 'args')
        # Getting the type of 'defaults' (line 599)
        defaults_2069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 32), 'defaults', False)
        # Processing the call keyword arguments (line 599)
        kwargs_2070 = {}
        # Getting the type of 'zip' (line 599)
        zip_2066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 599, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 599)
        zip_call_result_2071 = invoke(stypy.reporting.localization.Localization(__file__, 599, 20), zip_2066, *[args_2068, defaults_2069], **kwargs_2070)
        
        # Assigning a type to the variable 'zip_call_result_2071' (line 599)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'zip_call_result_2071', zip_call_result_2071)
        # Testing if the for loop is going to be iterated (line 599)
        # Testing the type of a for loop iterable (line 599)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 599, 8), zip_call_result_2071)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 599, 8), zip_call_result_2071):
            # Getting the type of the for loop variable (line 599)
            for_loop_var_2072 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 599, 8), zip_call_result_2071)
            # Assigning a type to the variable 'a' (line 599)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 8), for_loop_var_2072, 2, 0))
            # Assigning a type to the variable 'd' (line 599)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 599, 8), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 599, 8), for_loop_var_2072, 2, 1))
            # SSA begins for a for statement (line 599)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'first' (line 600)
            first_2073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 15), 'first')
            # Testing if the type of an if condition is none (line 600)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 600, 12), first_2073):
                
                # Call to write(...): (line 603)
                # Processing the call arguments (line 603)
                str_2078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 27), 'str', ', ')
                # Processing the call keyword arguments (line 603)
                kwargs_2079 = {}
                # Getting the type of 'self' (line 603)
                self_2076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 603)
                write_2077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 16), self_2076, 'write')
                # Calling write(args, kwargs) (line 603)
                write_call_result_2080 = invoke(stypy.reporting.localization.Localization(__file__, 603, 16), write_2077, *[str_2078], **kwargs_2079)
                
            else:
                
                # Testing the type of an if condition (line 600)
                if_condition_2074 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 600, 12), first_2073)
                # Assigning a type to the variable 'if_condition_2074' (line 600)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 600, 12), 'if_condition_2074', if_condition_2074)
                # SSA begins for if statement (line 600)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 601):
                
                # Assigning a Name to a Name (line 601):
                # Getting the type of 'False' (line 601)
                False_2075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 24), 'False')
                # Assigning a type to the variable 'first' (line 601)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 601, 16), 'first', False_2075)
                # SSA branch for the else part of an if statement (line 600)
                module_type_store.open_ssa_branch('else')
                
                # Call to write(...): (line 603)
                # Processing the call arguments (line 603)
                str_2078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 603, 27), 'str', ', ')
                # Processing the call keyword arguments (line 603)
                kwargs_2079 = {}
                # Getting the type of 'self' (line 603)
                self_2076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 603, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 603)
                write_2077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 603, 16), self_2076, 'write')
                # Calling write(args, kwargs) (line 603)
                write_call_result_2080 = invoke(stypy.reporting.localization.Localization(__file__, 603, 16), write_2077, *[str_2078], **kwargs_2079)
                
                # SSA join for if statement (line 600)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Obtaining an instance of the builtin type 'tuple' (line 604)
            tuple_2081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 604, 12), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 604)
            # Adding element type (line 604)
            
            # Call to visit(...): (line 604)
            # Processing the call arguments (line 604)
            # Getting the type of 'a' (line 604)
            a_2084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 23), 'a', False)
            # Processing the call keyword arguments (line 604)
            kwargs_2085 = {}
            # Getting the type of 'self' (line 604)
            self_2082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 604, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 604)
            visit_2083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 604, 12), self_2082, 'visit')
            # Calling visit(args, kwargs) (line 604)
            visit_call_result_2086 = invoke(stypy.reporting.localization.Localization(__file__, 604, 12), visit_2083, *[a_2084], **kwargs_2085)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 604, 12), tuple_2081, visit_call_result_2086)
            
            # Getting the type of 'd' (line 605)
            d_2087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 15), 'd')
            # Testing if the type of an if condition is none (line 605)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 605, 12), d_2087):
                pass
            else:
                
                # Testing the type of an if condition (line 605)
                if_condition_2088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 605, 12), d_2087)
                # Assigning a type to the variable 'if_condition_2088' (line 605)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 12), 'if_condition_2088', if_condition_2088)
                # SSA begins for if statement (line 605)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 606)
                # Processing the call arguments (line 606)
                str_2091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 606, 27), 'str', '=')
                # Processing the call keyword arguments (line 606)
                kwargs_2092 = {}
                # Getting the type of 'self' (line 606)
                self_2089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 606)
                write_2090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 606, 16), self_2089, 'write')
                # Calling write(args, kwargs) (line 606)
                write_call_result_2093 = invoke(stypy.reporting.localization.Localization(__file__, 606, 16), write_2090, *[str_2091], **kwargs_2092)
                
                
                # Call to visit(...): (line 607)
                # Processing the call arguments (line 607)
                # Getting the type of 'd' (line 607)
                d_2096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 27), 'd', False)
                # Processing the call keyword arguments (line 607)
                kwargs_2097 = {}
                # Getting the type of 'self' (line 607)
                self_2094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 16), 'self', False)
                # Obtaining the member 'visit' of a type (line 607)
                visit_2095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 607, 16), self_2094, 'visit')
                # Calling visit(args, kwargs) (line 607)
                visit_call_result_2098 = invoke(stypy.reporting.localization.Localization(__file__, 607, 16), visit_2095, *[d_2096], **kwargs_2097)
                
                # SSA join for if statement (line 605)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 't' (line 610)
        t_2099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 11), 't')
        # Obtaining the member 'vararg' of a type (line 610)
        vararg_2100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 11), t_2099, 'vararg')
        # Testing if the type of an if condition is none (line 610)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 610, 8), vararg_2100):
            pass
        else:
            
            # Testing the type of an if condition (line 610)
            if_condition_2101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 610, 8), vararg_2100)
            # Assigning a type to the variable 'if_condition_2101' (line 610)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 610, 8), 'if_condition_2101', if_condition_2101)
            # SSA begins for if statement (line 610)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'first' (line 611)
            first_2102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 611, 15), 'first')
            # Testing if the type of an if condition is none (line 611)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 611, 12), first_2102):
                
                # Call to write(...): (line 614)
                # Processing the call arguments (line 614)
                str_2107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 27), 'str', ', ')
                # Processing the call keyword arguments (line 614)
                kwargs_2108 = {}
                # Getting the type of 'self' (line 614)
                self_2105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 614)
                write_2106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 16), self_2105, 'write')
                # Calling write(args, kwargs) (line 614)
                write_call_result_2109 = invoke(stypy.reporting.localization.Localization(__file__, 614, 16), write_2106, *[str_2107], **kwargs_2108)
                
            else:
                
                # Testing the type of an if condition (line 611)
                if_condition_2103 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 611, 12), first_2102)
                # Assigning a type to the variable 'if_condition_2103' (line 611)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 611, 12), 'if_condition_2103', if_condition_2103)
                # SSA begins for if statement (line 611)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 612):
                
                # Assigning a Name to a Name (line 612):
                # Getting the type of 'False' (line 612)
                False_2104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 24), 'False')
                # Assigning a type to the variable 'first' (line 612)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 612, 16), 'first', False_2104)
                # SSA branch for the else part of an if statement (line 611)
                module_type_store.open_ssa_branch('else')
                
                # Call to write(...): (line 614)
                # Processing the call arguments (line 614)
                str_2107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 27), 'str', ', ')
                # Processing the call keyword arguments (line 614)
                kwargs_2108 = {}
                # Getting the type of 'self' (line 614)
                self_2105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 614)
                write_2106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 16), self_2105, 'write')
                # Calling write(args, kwargs) (line 614)
                write_call_result_2109 = invoke(stypy.reporting.localization.Localization(__file__, 614, 16), write_2106, *[str_2107], **kwargs_2108)
                
                # SSA join for if statement (line 611)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 615)
            # Processing the call arguments (line 615)
            str_2112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 615, 23), 'str', '*')
            # Processing the call keyword arguments (line 615)
            kwargs_2113 = {}
            # Getting the type of 'self' (line 615)
            self_2110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 615)
            write_2111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 12), self_2110, 'write')
            # Calling write(args, kwargs) (line 615)
            write_call_result_2114 = invoke(stypy.reporting.localization.Localization(__file__, 615, 12), write_2111, *[str_2112], **kwargs_2113)
            
            
            # Call to write(...): (line 616)
            # Processing the call arguments (line 616)
            # Getting the type of 't' (line 616)
            t_2117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 23), 't', False)
            # Obtaining the member 'vararg' of a type (line 616)
            vararg_2118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 23), t_2117, 'vararg')
            # Processing the call keyword arguments (line 616)
            kwargs_2119 = {}
            # Getting the type of 'self' (line 616)
            self_2115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 616, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 616)
            write_2116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 616, 12), self_2115, 'write')
            # Calling write(args, kwargs) (line 616)
            write_call_result_2120 = invoke(stypy.reporting.localization.Localization(__file__, 616, 12), write_2116, *[vararg_2118], **kwargs_2119)
            
            # SSA join for if statement (line 610)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 619)
        t_2121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 11), 't')
        # Obtaining the member 'kwarg' of a type (line 619)
        kwarg_2122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 11), t_2121, 'kwarg')
        # Testing if the type of an if condition is none (line 619)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 619, 8), kwarg_2122):
            pass
        else:
            
            # Testing the type of an if condition (line 619)
            if_condition_2123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 619, 8), kwarg_2122)
            # Assigning a type to the variable 'if_condition_2123' (line 619)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'if_condition_2123', if_condition_2123)
            # SSA begins for if statement (line 619)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'first' (line 620)
            first_2124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 15), 'first')
            # Testing if the type of an if condition is none (line 620)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 620, 12), first_2124):
                
                # Call to write(...): (line 623)
                # Processing the call arguments (line 623)
                str_2129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 27), 'str', ', ')
                # Processing the call keyword arguments (line 623)
                kwargs_2130 = {}
                # Getting the type of 'self' (line 623)
                self_2127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 623)
                write_2128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 16), self_2127, 'write')
                # Calling write(args, kwargs) (line 623)
                write_call_result_2131 = invoke(stypy.reporting.localization.Localization(__file__, 623, 16), write_2128, *[str_2129], **kwargs_2130)
                
            else:
                
                # Testing the type of an if condition (line 620)
                if_condition_2125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 620, 12), first_2124)
                # Assigning a type to the variable 'if_condition_2125' (line 620)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 620, 12), 'if_condition_2125', if_condition_2125)
                # SSA begins for if statement (line 620)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 621):
                
                # Assigning a Name to a Name (line 621):
                # Getting the type of 'False' (line 621)
                False_2126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 24), 'False')
                # Assigning a type to the variable 'first' (line 621)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 621, 16), 'first', False_2126)
                # SSA branch for the else part of an if statement (line 620)
                module_type_store.open_ssa_branch('else')
                
                # Call to write(...): (line 623)
                # Processing the call arguments (line 623)
                str_2129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 27), 'str', ', ')
                # Processing the call keyword arguments (line 623)
                kwargs_2130 = {}
                # Getting the type of 'self' (line 623)
                self_2127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 623)
                write_2128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 16), self_2127, 'write')
                # Calling write(args, kwargs) (line 623)
                write_call_result_2131 = invoke(stypy.reporting.localization.Localization(__file__, 623, 16), write_2128, *[str_2129], **kwargs_2130)
                
                # SSA join for if statement (line 620)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 624)
            # Processing the call arguments (line 624)
            str_2134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 624, 23), 'str', '**')
            # Getting the type of 't' (line 624)
            t_2135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 30), 't', False)
            # Obtaining the member 'kwarg' of a type (line 624)
            kwarg_2136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 30), t_2135, 'kwarg')
            # Applying the binary operator '+' (line 624)
            result_add_2137 = python_operator(stypy.reporting.localization.Localization(__file__, 624, 23), '+', str_2134, kwarg_2136)
            
            # Processing the call keyword arguments (line 624)
            kwargs_2138 = {}
            # Getting the type of 'self' (line 624)
            self_2132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 624, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 624)
            write_2133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 624, 12), self_2132, 'write')
            # Calling write(args, kwargs) (line 624)
            write_call_result_2139 = invoke(stypy.reporting.localization.Localization(__file__, 624, 12), write_2133, *[result_add_2137], **kwargs_2138)
            
            # SSA join for if statement (line 619)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_arguments(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_arguments' in the type store
        # Getting the type of 'stypy_return_type' (line 595)
        stypy_return_type_2140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 595, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2140)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_arguments'
        return stypy_return_type_2140


    @norecursion
    def visit_keyword(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_keyword'
        module_type_store = module_type_store.open_function_context('visit_keyword', 626, 4, False)
        # Assigning a type to the variable 'self' (line 627)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 627)
        # Processing the call arguments (line 627)
        # Getting the type of 't' (line 627)
        t_2143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 19), 't', False)
        # Obtaining the member 'arg' of a type (line 627)
        arg_2144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 19), t_2143, 'arg')
        # Processing the call keyword arguments (line 627)
        kwargs_2145 = {}
        # Getting the type of 'self' (line 627)
        self_2141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 627)
        write_2142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 8), self_2141, 'write')
        # Calling write(args, kwargs) (line 627)
        write_call_result_2146 = invoke(stypy.reporting.localization.Localization(__file__, 627, 8), write_2142, *[arg_2144], **kwargs_2145)
        
        
        # Call to write(...): (line 628)
        # Processing the call arguments (line 628)
        str_2149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 19), 'str', '=')
        # Processing the call keyword arguments (line 628)
        kwargs_2150 = {}
        # Getting the type of 'self' (line 628)
        self_2147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 628)
        write_2148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 8), self_2147, 'write')
        # Calling write(args, kwargs) (line 628)
        write_call_result_2151 = invoke(stypy.reporting.localization.Localization(__file__, 628, 8), write_2148, *[str_2149], **kwargs_2150)
        
        
        # Call to visit(...): (line 629)
        # Processing the call arguments (line 629)
        # Getting the type of 't' (line 629)
        t_2154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 19), 't', False)
        # Obtaining the member 'value' of a type (line 629)
        value_2155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 19), t_2154, 'value')
        # Processing the call keyword arguments (line 629)
        kwargs_2156 = {}
        # Getting the type of 'self' (line 629)
        self_2152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 629, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 629)
        visit_2153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 629, 8), self_2152, 'visit')
        # Calling visit(args, kwargs) (line 629)
        visit_call_result_2157 = invoke(stypy.reporting.localization.Localization(__file__, 629, 8), visit_2153, *[value_2155], **kwargs_2156)
        
        
        # ################# End of 'visit_keyword(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_keyword' in the type store
        # Getting the type of 'stypy_return_type' (line 626)
        stypy_return_type_2158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2158)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_keyword'
        return stypy_return_type_2158


    @norecursion
    def visit_Lambda(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_Lambda'
        module_type_store = module_type_store.open_function_context('visit_Lambda', 631, 4, False)
        # Assigning a type to the variable 'self' (line 632)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 632, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 632)
        # Processing the call arguments (line 632)
        str_2161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 632, 19), 'str', '(')
        # Processing the call keyword arguments (line 632)
        kwargs_2162 = {}
        # Getting the type of 'self' (line 632)
        self_2159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 632, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 632)
        write_2160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 632, 8), self_2159, 'write')
        # Calling write(args, kwargs) (line 632)
        write_call_result_2163 = invoke(stypy.reporting.localization.Localization(__file__, 632, 8), write_2160, *[str_2161], **kwargs_2162)
        
        
        # Call to write(...): (line 633)
        # Processing the call arguments (line 633)
        str_2166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 633, 19), 'str', 'lambda ')
        # Processing the call keyword arguments (line 633)
        kwargs_2167 = {}
        # Getting the type of 'self' (line 633)
        self_2164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 633, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 633)
        write_2165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 633, 8), self_2164, 'write')
        # Calling write(args, kwargs) (line 633)
        write_call_result_2168 = invoke(stypy.reporting.localization.Localization(__file__, 633, 8), write_2165, *[str_2166], **kwargs_2167)
        
        
        # Call to visit(...): (line 634)
        # Processing the call arguments (line 634)
        # Getting the type of 't' (line 634)
        t_2171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 19), 't', False)
        # Obtaining the member 'args' of a type (line 634)
        args_2172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 19), t_2171, 'args')
        # Processing the call keyword arguments (line 634)
        kwargs_2173 = {}
        # Getting the type of 'self' (line 634)
        self_2169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 634, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 634)
        visit_2170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 634, 8), self_2169, 'visit')
        # Calling visit(args, kwargs) (line 634)
        visit_call_result_2174 = invoke(stypy.reporting.localization.Localization(__file__, 634, 8), visit_2170, *[args_2172], **kwargs_2173)
        
        
        # Call to write(...): (line 635)
        # Processing the call arguments (line 635)
        str_2177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 635, 19), 'str', ': ')
        # Processing the call keyword arguments (line 635)
        kwargs_2178 = {}
        # Getting the type of 'self' (line 635)
        self_2175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 635, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 635)
        write_2176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 635, 8), self_2175, 'write')
        # Calling write(args, kwargs) (line 635)
        write_call_result_2179 = invoke(stypy.reporting.localization.Localization(__file__, 635, 8), write_2176, *[str_2177], **kwargs_2178)
        
        
        # Call to visit(...): (line 636)
        # Processing the call arguments (line 636)
        # Getting the type of 't' (line 636)
        t_2182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 19), 't', False)
        # Obtaining the member 'body' of a type (line 636)
        body_2183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 19), t_2182, 'body')
        # Processing the call keyword arguments (line 636)
        kwargs_2184 = {}
        # Getting the type of 'self' (line 636)
        self_2180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 636, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 636)
        visit_2181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 636, 8), self_2180, 'visit')
        # Calling visit(args, kwargs) (line 636)
        visit_call_result_2185 = invoke(stypy.reporting.localization.Localization(__file__, 636, 8), visit_2181, *[body_2183], **kwargs_2184)
        
        
        # Call to write(...): (line 637)
        # Processing the call arguments (line 637)
        str_2188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 637, 19), 'str', ')')
        # Processing the call keyword arguments (line 637)
        kwargs_2189 = {}
        # Getting the type of 'self' (line 637)
        self_2186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 637, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 637)
        write_2187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 637, 8), self_2186, 'write')
        # Calling write(args, kwargs) (line 637)
        write_call_result_2190 = invoke(stypy.reporting.localization.Localization(__file__, 637, 8), write_2187, *[str_2188], **kwargs_2189)
        
        
        # ################# End of 'visit_Lambda(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Lambda' in the type store
        # Getting the type of 'stypy_return_type' (line 631)
        stypy_return_type_2191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 631, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2191)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Lambda'
        return stypy_return_type_2191


    @norecursion
    def visit_alias(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit_alias'
        module_type_store = module_type_store.open_function_context('visit_alias', 639, 4, False)
        # Assigning a type to the variable 'self' (line 640)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 640, 4), 'self', type_of_self)
        
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

        
        # Call to write(...): (line 640)
        # Processing the call arguments (line 640)
        # Getting the type of 't' (line 640)
        t_2194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 19), 't', False)
        # Obtaining the member 'name' of a type (line 640)
        name_2195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 19), t_2194, 'name')
        # Processing the call keyword arguments (line 640)
        kwargs_2196 = {}
        # Getting the type of 'self' (line 640)
        self_2192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 640, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 640)
        write_2193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 640, 8), self_2192, 'write')
        # Calling write(args, kwargs) (line 640)
        write_call_result_2197 = invoke(stypy.reporting.localization.Localization(__file__, 640, 8), write_2193, *[name_2195], **kwargs_2196)
        
        # Getting the type of 't' (line 641)
        t_2198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 641, 11), 't')
        # Obtaining the member 'asname' of a type (line 641)
        asname_2199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 641, 11), t_2198, 'asname')
        # Testing if the type of an if condition is none (line 641)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 641, 8), asname_2199):
            pass
        else:
            
            # Testing the type of an if condition (line 641)
            if_condition_2200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 641, 8), asname_2199)
            # Assigning a type to the variable 'if_condition_2200' (line 641)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 641, 8), 'if_condition_2200', if_condition_2200)
            # SSA begins for if statement (line 641)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 642)
            # Processing the call arguments (line 642)
            str_2203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 642, 23), 'str', ' as ')
            # Getting the type of 't' (line 642)
            t_2204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 32), 't', False)
            # Obtaining the member 'asname' of a type (line 642)
            asname_2205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 32), t_2204, 'asname')
            # Applying the binary operator '+' (line 642)
            result_add_2206 = python_operator(stypy.reporting.localization.Localization(__file__, 642, 23), '+', str_2203, asname_2205)
            
            # Processing the call keyword arguments (line 642)
            kwargs_2207 = {}
            # Getting the type of 'self' (line 642)
            self_2201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 642, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 642)
            write_2202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 642, 12), self_2201, 'write')
            # Calling write(args, kwargs) (line 642)
            write_call_result_2208 = invoke(stypy.reporting.localization.Localization(__file__, 642, 12), write_2202, *[result_add_2206], **kwargs_2207)
            
            # SSA join for if statement (line 641)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_alias(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_alias' in the type store
        # Getting the type of 'stypy_return_type' (line 639)
        stypy_return_type_2209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 639, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2209)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_alias'
        return stypy_return_type_2209


# Assigning a type to the variable 'PythonSrcGeneratorVisitor' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'PythonSrcGeneratorVisitor', PythonSrcGeneratorVisitor)

# Assigning a Dict to a Name (line 477):

# Obtaining an instance of the builtin type 'dict' (line 477)
dict_2210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 477)
# Adding element type (key, value) (line 477)
str_2211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 12), 'str', 'Invert')
str_2212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 22), 'str', '~')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 11), dict_2210, (str_2211, str_2212))
# Adding element type (key, value) (line 477)
str_2213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 27), 'str', 'Not')
str_2214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 34), 'str', 'not')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 11), dict_2210, (str_2213, str_2214))
# Adding element type (key, value) (line 477)
str_2215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 41), 'str', 'UAdd')
str_2216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 49), 'str', '+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 11), dict_2210, (str_2215, str_2216))
# Adding element type (key, value) (line 477)
str_2217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 54), 'str', 'USub')
str_2218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 62), 'str', '-')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 11), dict_2210, (str_2217, str_2218))

# Getting the type of 'PythonSrcGeneratorVisitor'
PythonSrcGeneratorVisitor_2219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PythonSrcGeneratorVisitor')
# Setting the type of the member 'unop' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PythonSrcGeneratorVisitor_2219, 'unop', dict_2210)

# Assigning a Dict to a Name (line 496):

# Obtaining an instance of the builtin type 'dict' (line 496)
dict_2220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 496)
# Adding element type (key, value) (line 496)
str_2221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 13), 'str', 'Add')
str_2222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 20), 'str', '+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), dict_2220, (str_2221, str_2222))
# Adding element type (key, value) (line 496)
str_2223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 25), 'str', 'Sub')
str_2224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 32), 'str', '-')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), dict_2220, (str_2223, str_2224))
# Adding element type (key, value) (line 496)
str_2225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 37), 'str', 'Mult')
str_2226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 45), 'str', '*')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), dict_2220, (str_2225, str_2226))
# Adding element type (key, value) (line 496)
str_2227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 50), 'str', 'Div')
str_2228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 57), 'str', '/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), dict_2220, (str_2227, str_2228))
# Adding element type (key, value) (line 496)
str_2229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 62), 'str', 'Mod')
str_2230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 496, 69), 'str', '%')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), dict_2220, (str_2229, str_2230))
# Adding element type (key, value) (line 496)
str_2231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 13), 'str', 'LShift')
str_2232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 23), 'str', '<<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), dict_2220, (str_2231, str_2232))
# Adding element type (key, value) (line 496)
str_2233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 29), 'str', 'RShift')
str_2234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 39), 'str', '>>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), dict_2220, (str_2233, str_2234))
# Adding element type (key, value) (line 496)
str_2235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 45), 'str', 'BitOr')
str_2236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 54), 'str', '|')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), dict_2220, (str_2235, str_2236))
# Adding element type (key, value) (line 496)
str_2237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 59), 'str', 'BitXor')
str_2238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 69), 'str', '^')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), dict_2220, (str_2237, str_2238))
# Adding element type (key, value) (line 496)
str_2239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 74), 'str', 'BitAnd')
str_2240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 84), 'str', '&')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), dict_2220, (str_2239, str_2240))
# Adding element type (key, value) (line 496)
str_2241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 13), 'str', 'FloorDiv')
str_2242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 25), 'str', '//')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), dict_2220, (str_2241, str_2242))
# Adding element type (key, value) (line 496)
str_2243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 31), 'str', 'Pow')
str_2244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 498, 38), 'str', '**')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 496, 12), dict_2220, (str_2243, str_2244))

# Getting the type of 'PythonSrcGeneratorVisitor'
PythonSrcGeneratorVisitor_2245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PythonSrcGeneratorVisitor')
# Setting the type of the member 'binop' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PythonSrcGeneratorVisitor_2245, 'binop', dict_2220)

# Assigning a Dict to a Name (line 507):

# Obtaining an instance of the builtin type 'dict' (line 507)
dict_2246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 507)
# Adding element type (key, value) (line 507)
str_2247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 14), 'str', 'Eq')
str_2248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 20), 'str', '==')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), dict_2246, (str_2247, str_2248))
# Adding element type (key, value) (line 507)
str_2249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 26), 'str', 'NotEq')
str_2250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 35), 'str', '!=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), dict_2246, (str_2249, str_2250))
# Adding element type (key, value) (line 507)
str_2251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 41), 'str', 'Lt')
str_2252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 47), 'str', '<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), dict_2246, (str_2251, str_2252))
# Adding element type (key, value) (line 507)
str_2253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 52), 'str', 'LtE')
str_2254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 59), 'str', '<=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), dict_2246, (str_2253, str_2254))
# Adding element type (key, value) (line 507)
str_2255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 65), 'str', 'Gt')
str_2256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 71), 'str', '>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), dict_2246, (str_2255, str_2256))
# Adding element type (key, value) (line 507)
str_2257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 76), 'str', 'GtE')
str_2258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 83), 'str', '>=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), dict_2246, (str_2257, str_2258))
# Adding element type (key, value) (line 507)
str_2259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 14), 'str', 'Is')
str_2260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 20), 'str', 'is')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), dict_2246, (str_2259, str_2260))
# Adding element type (key, value) (line 507)
str_2261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 26), 'str', 'IsNot')
str_2262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 35), 'str', 'is not')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), dict_2246, (str_2261, str_2262))
# Adding element type (key, value) (line 507)
str_2263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 45), 'str', 'In')
str_2264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 51), 'str', 'in')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), dict_2246, (str_2263, str_2264))
# Adding element type (key, value) (line 507)
str_2265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 57), 'str', 'NotIn')
str_2266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 66), 'str', 'not in')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 507, 13), dict_2246, (str_2265, str_2266))

# Getting the type of 'PythonSrcGeneratorVisitor'
PythonSrcGeneratorVisitor_2267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PythonSrcGeneratorVisitor')
# Setting the type of the member 'cmpops' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PythonSrcGeneratorVisitor_2267, 'cmpops', dict_2246)

# Assigning a Dict to a Name (line 518):

# Obtaining an instance of the builtin type 'dict' (line 518)
dict_2268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 518)
# Adding element type (key, value) (line 518)
# Getting the type of 'ast' (line 518)
ast_2269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 15), 'ast')
# Obtaining the member 'And' of a type (line 518)
And_2270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 15), ast_2269, 'And')
str_2271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 24), 'str', 'and')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 14), dict_2268, (And_2270, str_2271))
# Adding element type (key, value) (line 518)
# Getting the type of 'ast' (line 518)
ast_2272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 31), 'ast')
# Obtaining the member 'Or' of a type (line 518)
Or_2273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 31), ast_2272, 'Or')
str_2274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 39), 'str', 'or')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 518, 14), dict_2268, (Or_2273, str_2274))

# Getting the type of 'PythonSrcGeneratorVisitor'
PythonSrcGeneratorVisitor_2275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PythonSrcGeneratorVisitor')
# Setting the type of the member 'boolops' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PythonSrcGeneratorVisitor_2275, 'boolops', dict_2268)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
