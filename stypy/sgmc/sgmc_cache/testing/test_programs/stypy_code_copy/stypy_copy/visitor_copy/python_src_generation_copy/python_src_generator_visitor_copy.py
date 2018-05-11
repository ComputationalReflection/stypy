
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

str_18880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', '\nThis is a Python source code generator visitor, that transform an AST into valid source code. It is used to create\ntype annotated programs and type inference programs when their AST is finally created.\n\nAdapted from: http://svn.python.org/view/python/trunk/Demo/parser/unparse.py\n')
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
str_18881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 9), 'str', '1e')

# Call to repr(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'sys' (line 14)
sys_18883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 21), 'sys', False)
# Obtaining the member 'float_info' of a type (line 14)
float_info_18884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 21), sys_18883, 'float_info')
# Obtaining the member 'max_10_exp' of a type (line 14)
max_10_exp_18885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 21), float_info_18884, 'max_10_exp')
int_18886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 49), 'int')
# Applying the binary operator '+' (line 14)
result_add_18887 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 21), '+', max_10_exp_18885, int_18886)

# Processing the call keyword arguments (line 14)
kwargs_18888 = {}
# Getting the type of 'repr' (line 14)
repr_18882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'repr', False)
# Calling repr(args, kwargs) (line 14)
repr_call_result_18889 = invoke(stypy.reporting.localization.Localization(__file__, 14, 16), repr_18882, *[result_add_18887], **kwargs_18888)

# Applying the binary operator '+' (line 14)
result_add_18890 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 9), '+', str_18881, repr_call_result_18889)

# Assigning a type to the variable 'INFSTR' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'INFSTR', result_add_18890)

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

    str_18891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'str', '\n    Call f on each item in seq, calling inter() in between.\n    ')
    
    # Assigning a Call to a Name (line 21):
    
    # Assigning a Call to a Name (line 21):
    
    # Call to iter(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'seq' (line 21)
    seq_18893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'seq', False)
    # Processing the call keyword arguments (line 21)
    kwargs_18894 = {}
    # Getting the type of 'iter' (line 21)
    iter_18892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'iter', False)
    # Calling iter(args, kwargs) (line 21)
    iter_call_result_18895 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), iter_18892, *[seq_18893], **kwargs_18894)
    
    # Assigning a type to the variable 'seq' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'seq', iter_call_result_18895)
    
    
    # SSA begins for try-except statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to f(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to next(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'seq' (line 23)
    seq_18898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'seq', False)
    # Processing the call keyword arguments (line 23)
    kwargs_18899 = {}
    # Getting the type of 'next' (line 23)
    next_18897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'next', False)
    # Calling next(args, kwargs) (line 23)
    next_call_result_18900 = invoke(stypy.reporting.localization.Localization(__file__, 23, 10), next_18897, *[seq_18898], **kwargs_18899)
    
    # Processing the call keyword arguments (line 23)
    kwargs_18901 = {}
    # Getting the type of 'f' (line 23)
    f_18896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'f', False)
    # Calling f(args, kwargs) (line 23)
    f_call_result_18902 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), f_18896, *[next_call_result_18900], **kwargs_18901)
    
    # SSA branch for the except part of a try statement (line 22)
    # SSA branch for the except 'StopIteration' branch of a try statement (line 22)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA branch for the else branch of a try statement (line 22)
    module_type_store.open_ssa_branch('except else')
    
    # Getting the type of 'seq' (line 27)
    seq_18903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 17), 'seq')
    # Assigning a type to the variable 'seq_18903' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'seq_18903', seq_18903)
    # Testing if the for loop is going to be iterated (line 27)
    # Testing the type of a for loop iterable (line 27)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 8), seq_18903)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 27, 8), seq_18903):
        # Getting the type of the for loop variable (line 27)
        for_loop_var_18904 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 8), seq_18903)
        # Assigning a type to the variable 'x' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'x', for_loop_var_18904)
        # SSA begins for a for statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to inter(...): (line 28)
        # Processing the call keyword arguments (line 28)
        kwargs_18906 = {}
        # Getting the type of 'inter' (line 28)
        inter_18905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'inter', False)
        # Calling inter(args, kwargs) (line 28)
        inter_call_result_18907 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), inter_18905, *[], **kwargs_18906)
        
        
        # Call to f(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'x' (line 29)
        x_18909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'x', False)
        # Processing the call keyword arguments (line 29)
        kwargs_18910 = {}
        # Getting the type of 'f' (line 29)
        f_18908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'f', False)
        # Calling f(args, kwargs) (line 29)
        f_call_result_18911 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), f_18908, *[x_18909], **kwargs_18910)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # SSA join for try-except statement (line 22)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'interleave(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'interleave' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_18912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_18912)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'interleave'
    return stypy_return_type_18912

# Assigning a type to the variable 'interleave' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'interleave', interleave)
# Declaration of the 'PythonSrcGeneratorVisitor' class
# Getting the type of 'ast' (line 32)
ast_18913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 32), 'ast')
# Obtaining the member 'NodeVisitor' of a type (line 32)
NodeVisitor_18914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 32), ast_18913, 'NodeVisitor')

class PythonSrcGeneratorVisitor(NodeVisitor_18914, ):
    str_18915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, (-1)), 'str', '\n    Methods in this class recursively traverse an AST and\n    output source code for the abstract syntax; original formatting\n    is disregarded.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 39)
        False_18916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 37), 'False')
        defaults = [False_18916]
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
        kwargs_18918 = {}
        # Getting the type of 'StringIO' (line 40)
        StringIO_18917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 40)
        StringIO_call_result_18919 = invoke(stypy.reporting.localization.Localization(__file__, 40, 22), StringIO_18917, *[], **kwargs_18918)
        
        # Getting the type of 'self' (line 40)
        self_18920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'output' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_18920, 'output', StringIO_call_result_18919)
        
        # Assigning a List to a Attribute (line 41):
        
        # Assigning a List to a Attribute (line 41):
        
        # Obtaining an instance of the builtin type 'list' (line 41)
        list_18921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 41)
        
        # Getting the type of 'self' (line 41)
        self_18922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'future_imports' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_18922, 'future_imports', list_18921)
        
        # Assigning a Num to a Attribute (line 42):
        
        # Assigning a Num to a Attribute (line 42):
        int_18923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 23), 'int')
        # Getting the type of 'self' (line 42)
        self_18924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member '_indent' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_18924, '_indent', int_18923)
        
        # Assigning a Str to a Attribute (line 43):
        
        # Assigning a Str to a Attribute (line 43):
        str_18925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 27), 'str', '    ')
        # Getting the type of 'self' (line 43)
        self_18926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member '_indent_str' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_18926, '_indent_str', str_18925)
        
        # Call to write(...): (line 44)
        # Processing the call arguments (line 44)
        str_18930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 26), 'str', '')
        # Processing the call keyword arguments (line 44)
        kwargs_18931 = {}
        # Getting the type of 'self' (line 44)
        self_18927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 44)
        output_18928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_18927, 'output')
        # Obtaining the member 'write' of a type (line 44)
        write_18929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), output_18928, 'write')
        # Calling write(args, kwargs) (line 44)
        write_call_result_18932 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), write_18929, *[str_18930], **kwargs_18931)
        
        
        # Call to flush(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_18936 = {}
        # Getting the type of 'self' (line 45)
        self_18933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 45)
        output_18934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_18933, 'output')
        # Obtaining the member 'flush' of a type (line 45)
        flush_18935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), output_18934, 'flush')
        # Calling flush(args, kwargs) (line 45)
        flush_call_result_18937 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), flush_18935, *[], **kwargs_18936)
        
        
        # Assigning a Name to a Attribute (line 46):
        
        # Assigning a Name to a Attribute (line 46):
        # Getting the type of 'tree' (line 46)
        tree_18938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'tree')
        # Getting the type of 'self' (line 46)
        self_18939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'self')
        # Setting the type of the member 'tree' of a type (line 46)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), self_18939, 'tree', tree_18938)
        
        # Assigning a Name to a Attribute (line 47):
        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'verbose' (line 47)
        verbose_18940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'verbose')
        # Getting the type of 'self' (line 47)
        self_18941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member 'verbose' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_18941, 'verbose', verbose_18940)
        
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
        self_18944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'self', False)
        # Obtaining the member 'tree' of a type (line 50)
        tree_18945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 19), self_18944, 'tree')
        # Processing the call keyword arguments (line 50)
        kwargs_18946 = {}
        # Getting the type of 'self' (line 50)
        self_18942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 50)
        visit_18943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_18942, 'visit')
        # Calling visit(args, kwargs) (line 50)
        visit_call_result_18947 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), visit_18943, *[tree_18945], **kwargs_18946)
        
        
        # Call to getvalue(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_18951 = {}
        # Getting the type of 'self' (line 51)
        self_18948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'self', False)
        # Obtaining the member 'output' of a type (line 51)
        output_18949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), self_18948, 'output')
        # Obtaining the member 'getvalue' of a type (line 51)
        getvalue_18950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), output_18949, 'getvalue')
        # Calling getvalue(args, kwargs) (line 51)
        getvalue_call_result_18952 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), getvalue_18950, *[], **kwargs_18951)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', getvalue_call_result_18952)
        
        # ################# End of 'generate_code(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_code' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_18953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18953)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_code'
        return stypy_return_type_18953


    @norecursion
    def fill(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_18954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 24), 'str', '')
        defaults = [str_18954]
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

        str_18955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n        Indent a piece of text, according to the current indentation level\n        ')
        # Getting the type of 'self' (line 57)
        self_18956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'self')
        # Obtaining the member 'verbose' of a type (line 57)
        verbose_18957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), self_18956, 'verbose')
        # Testing if the type of an if condition is none (line 57)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 8), verbose_18957):
            pass
        else:
            
            # Testing the type of an if condition (line 57)
            if_condition_18958 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 8), verbose_18957)
            # Assigning a type to the variable 'if_condition_18958' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'if_condition_18958', if_condition_18958)
            # SSA begins for if statement (line 57)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 58)
            # Processing the call arguments (line 58)
            str_18962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 29), 'str', '\n')
            # Getting the type of 'self' (line 58)
            self_18963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 36), 'self', False)
            # Obtaining the member '_indent_str' of a type (line 58)
            _indent_str_18964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 36), self_18963, '_indent_str')
            # Getting the type of 'self' (line 58)
            self_18965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 55), 'self', False)
            # Obtaining the member '_indent' of a type (line 58)
            _indent_18966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 55), self_18965, '_indent')
            # Applying the binary operator '*' (line 58)
            result_mul_18967 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 36), '*', _indent_str_18964, _indent_18966)
            
            # Applying the binary operator '+' (line 58)
            result_add_18968 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 29), '+', str_18962, result_mul_18967)
            
            # Getting the type of 'text' (line 58)
            text_18969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 70), 'text', False)
            # Applying the binary operator '+' (line 58)
            result_add_18970 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 68), '+', result_add_18968, text_18969)
            
            # Processing the call keyword arguments (line 58)
            kwargs_18971 = {}
            # Getting the type of 'sys' (line 58)
            sys_18959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'sys', False)
            # Obtaining the member 'stdout' of a type (line 58)
            stdout_18960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), sys_18959, 'stdout')
            # Obtaining the member 'write' of a type (line 58)
            write_18961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), stdout_18960, 'write')
            # Calling write(args, kwargs) (line 58)
            write_call_result_18972 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), write_18961, *[result_add_18970], **kwargs_18971)
            
            # SSA join for if statement (line 57)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 59)
        # Processing the call arguments (line 59)
        str_18976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 26), 'str', '\n')
        # Getting the type of 'self' (line 59)
        self_18977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'self', False)
        # Obtaining the member '_indent_str' of a type (line 59)
        _indent_str_18978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 33), self_18977, '_indent_str')
        # Getting the type of 'self' (line 59)
        self_18979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 52), 'self', False)
        # Obtaining the member '_indent' of a type (line 59)
        _indent_18980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 52), self_18979, '_indent')
        # Applying the binary operator '*' (line 59)
        result_mul_18981 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 33), '*', _indent_str_18978, _indent_18980)
        
        # Applying the binary operator '+' (line 59)
        result_add_18982 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 26), '+', str_18976, result_mul_18981)
        
        # Getting the type of 'text' (line 59)
        text_18983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 67), 'text', False)
        # Applying the binary operator '+' (line 59)
        result_add_18984 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 65), '+', result_add_18982, text_18983)
        
        # Processing the call keyword arguments (line 59)
        kwargs_18985 = {}
        # Getting the type of 'self' (line 59)
        self_18973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 59)
        output_18974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_18973, 'output')
        # Obtaining the member 'write' of a type (line 59)
        write_18975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), output_18974, 'write')
        # Calling write(args, kwargs) (line 59)
        write_call_result_18986 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), write_18975, *[result_add_18984], **kwargs_18985)
        
        
        # ################# End of 'fill(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'fill' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_18987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_18987)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'fill'
        return stypy_return_type_18987


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

        str_18988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', '\n        Append a piece of text to the current line.\n        ')
        # Getting the type of 'self' (line 65)
        self_18989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'self')
        # Obtaining the member 'verbose' of a type (line 65)
        verbose_18990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 11), self_18989, 'verbose')
        # Testing if the type of an if condition is none (line 65)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 65, 8), verbose_18990):
            pass
        else:
            
            # Testing the type of an if condition (line 65)
            if_condition_18991 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 8), verbose_18990)
            # Assigning a type to the variable 'if_condition_18991' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'if_condition_18991', if_condition_18991)
            # SSA begins for if statement (line 65)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'text' (line 66)
            text_18995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'text', False)
            # Processing the call keyword arguments (line 66)
            kwargs_18996 = {}
            # Getting the type of 'sys' (line 66)
            sys_18992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'sys', False)
            # Obtaining the member 'stdout' of a type (line 66)
            stdout_18993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), sys_18992, 'stdout')
            # Obtaining the member 'write' of a type (line 66)
            write_18994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), stdout_18993, 'write')
            # Calling write(args, kwargs) (line 66)
            write_call_result_18997 = invoke(stypy.reporting.localization.Localization(__file__, 66, 12), write_18994, *[text_18995], **kwargs_18996)
            
            # SSA join for if statement (line 65)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'text' (line 67)
        text_19001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'text', False)
        # Processing the call keyword arguments (line 67)
        kwargs_19002 = {}
        # Getting the type of 'self' (line 67)
        self_18998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 67)
        output_18999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_18998, 'output')
        # Obtaining the member 'write' of a type (line 67)
        write_19000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), output_18999, 'write')
        # Calling write(args, kwargs) (line 67)
        write_call_result_19003 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), write_19000, *[text_19001], **kwargs_19002)
        
        
        # ################# End of 'write(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_19004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19004)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write'
        return stypy_return_type_19004


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

        str_19005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, (-1)), 'str', "\n        Print ':', and increase the indentation.\n        ")
        
        # Call to write(...): (line 73)
        # Processing the call arguments (line 73)
        str_19008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'str', ':')
        # Processing the call keyword arguments (line 73)
        kwargs_19009 = {}
        # Getting the type of 'self' (line 73)
        self_19006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 73)
        write_19007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), self_19006, 'write')
        # Calling write(args, kwargs) (line 73)
        write_call_result_19010 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), write_19007, *[str_19008], **kwargs_19009)
        
        
        # Getting the type of 'self' (line 74)
        self_19011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Obtaining the member '_indent' of a type (line 74)
        _indent_19012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_19011, '_indent')
        int_19013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'int')
        # Applying the binary operator '+=' (line 74)
        result_iadd_19014 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 8), '+=', _indent_19012, int_19013)
        # Getting the type of 'self' (line 74)
        self_19015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'self')
        # Setting the type of the member '_indent' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), self_19015, '_indent', result_iadd_19014)
        
        
        # ################# End of 'enter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'enter' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_19016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19016)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'enter'
        return stypy_return_type_19016


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

        str_19017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, (-1)), 'str', '\n        Decrease the indentation level.\n        ')
        
        # Getting the type of 'self' (line 80)
        self_19018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self')
        # Obtaining the member '_indent' of a type (line 80)
        _indent_19019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_19018, '_indent')
        int_19020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 24), 'int')
        # Applying the binary operator '-=' (line 80)
        result_isub_19021 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 8), '-=', _indent_19019, int_19020)
        # Getting the type of 'self' (line 80)
        self_19022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self')
        # Setting the type of the member '_indent' of a type (line 80)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 8), self_19022, '_indent', result_isub_19021)
        
        
        # ################# End of 'leave(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'leave' in the type store
        # Getting the type of 'stypy_return_type' (line 76)
        stypy_return_type_19023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19023)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'leave'
        return stypy_return_type_19023


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

        str_19024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'str', '\n        General visit method, calling the appropriate visit method for type T\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 86)
        # Getting the type of 'list' (line 86)
        list_19025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 28), 'list')
        # Getting the type of 'tree' (line 86)
        tree_19026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 22), 'tree')
        
        (may_be_19027, more_types_in_union_19028) = may_be_subtype(list_19025, tree_19026)

        if may_be_19027:

            if more_types_in_union_19028:
                # Runtime conditional SSA (line 86)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'tree' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tree', remove_not_subtype_from_union(tree_19026, list))
            
            # Getting the type of 'tree' (line 87)
            tree_19029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 21), 'tree')
            # Assigning a type to the variable 'tree_19029' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'tree_19029', tree_19029)
            # Testing if the for loop is going to be iterated (line 87)
            # Testing the type of a for loop iterable (line 87)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 12), tree_19029)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 87, 12), tree_19029):
                # Getting the type of the for loop variable (line 87)
                for_loop_var_19030 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 12), tree_19029)
                # Assigning a type to the variable 't' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 't', for_loop_var_19030)
                # SSA begins for a for statement (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to visit(...): (line 88)
                # Processing the call arguments (line 88)
                # Getting the type of 't' (line 88)
                t_19033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 't', False)
                # Processing the call keyword arguments (line 88)
                kwargs_19034 = {}
                # Getting the type of 'self' (line 88)
                self_19031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'self', False)
                # Obtaining the member 'visit' of a type (line 88)
                visit_19032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 16), self_19031, 'visit')
                # Calling visit(args, kwargs) (line 88)
                visit_call_result_19035 = invoke(stypy.reporting.localization.Localization(__file__, 88, 16), visit_19032, *[t_19033], **kwargs_19034)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # Assigning a type to the variable 'stypy_return_type' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'stypy_return_type', types.NoneType)

            if more_types_in_union_19028:
                # SSA join for if statement (line 86)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 91)
        # Getting the type of 'tree' (line 91)
        tree_19036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'tree')
        # Getting the type of 'tuple' (line 91)
        tuple_19037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'tuple')
        
        (may_be_19038, more_types_in_union_19039) = may_be_type(tree_19036, tuple_19037)

        if may_be_19038:

            if more_types_in_union_19039:
                # Runtime conditional SSA (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'tree' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'tree', tuple_19037())
            # Getting the type of 'tree' (line 92)
            tree_19040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'tree')

            if more_types_in_union_19039:
                # SSA join for if statement (line 91)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to getattr(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'self' (line 94)
        self_19042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'self', False)
        str_19043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 29), 'str', 'visit_')
        # Getting the type of 'tree' (line 94)
        tree_19044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 40), 'tree', False)
        # Obtaining the member '__class__' of a type (line 94)
        class___19045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 40), tree_19044, '__class__')
        # Obtaining the member '__name__' of a type (line 94)
        name___19046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 40), class___19045, '__name__')
        # Applying the binary operator '+' (line 94)
        result_add_19047 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 29), '+', str_19043, name___19046)
        
        # Processing the call keyword arguments (line 94)
        kwargs_19048 = {}
        # Getting the type of 'getattr' (line 94)
        getattr_19041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 94)
        getattr_call_result_19049 = invoke(stypy.reporting.localization.Localization(__file__, 94, 15), getattr_19041, *[self_19042, result_add_19047], **kwargs_19048)
        
        # Assigning a type to the variable 'meth' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'meth', getattr_call_result_19049)
        
        # Call to meth(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'tree' (line 95)
        tree_19051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'tree', False)
        # Processing the call keyword arguments (line 95)
        kwargs_19052 = {}
        # Getting the type of 'meth' (line 95)
        meth_19050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'meth', False)
        # Calling meth(args, kwargs) (line 95)
        meth_call_result_19053 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), meth_19050, *[tree_19051], **kwargs_19052)
        
        
        # ################# End of 'visit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_19054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19054)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit'
        return stypy_return_type_19054


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
        tree_19055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'tree')
        # Obtaining the member 'body' of a type (line 105)
        body_19056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 20), tree_19055, 'body')
        # Assigning a type to the variable 'body_19056' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'body_19056', body_19056)
        # Testing if the for loop is going to be iterated (line 105)
        # Testing the type of a for loop iterable (line 105)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 8), body_19056)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 105, 8), body_19056):
            # Getting the type of the for loop variable (line 105)
            for_loop_var_19057 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 8), body_19056)
            # Assigning a type to the variable 'stmt' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stmt', for_loop_var_19057)
            # SSA begins for a for statement (line 105)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 106)
            # Processing the call arguments (line 106)
            # Getting the type of 'stmt' (line 106)
            stmt_19060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 23), 'stmt', False)
            # Processing the call keyword arguments (line 106)
            kwargs_19061 = {}
            # Getting the type of 'self' (line 106)
            self_19058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 106)
            visit_19059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), self_19058, 'visit')
            # Calling visit(args, kwargs) (line 106)
            visit_call_result_19062 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), visit_19059, *[stmt_19060], **kwargs_19061)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'visit_Module(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Module' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_19063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19063)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Module'
        return stypy_return_type_19063


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
        kwargs_19066 = {}
        # Getting the type of 'self' (line 110)
        self_19064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 110)
        fill_19065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_19064, 'fill')
        # Calling fill(args, kwargs) (line 110)
        fill_call_result_19067 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), fill_19065, *[], **kwargs_19066)
        
        
        # Call to visit(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'tree' (line 111)
        tree_19070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'tree', False)
        # Obtaining the member 'value' of a type (line 111)
        value_19071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 19), tree_19070, 'value')
        # Processing the call keyword arguments (line 111)
        kwargs_19072 = {}
        # Getting the type of 'self' (line 111)
        self_19068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 111)
        visit_19069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_19068, 'visit')
        # Calling visit(args, kwargs) (line 111)
        visit_call_result_19073 = invoke(stypy.reporting.localization.Localization(__file__, 111, 8), visit_19069, *[value_19071], **kwargs_19072)
        
        
        # ################# End of 'visit_Expr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Expr' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_19074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19074)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Expr'
        return stypy_return_type_19074


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
        str_19077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 18), 'str', 'import ')
        # Processing the call keyword arguments (line 114)
        kwargs_19078 = {}
        # Getting the type of 'self' (line 114)
        self_19075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 114)
        fill_19076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_19075, 'fill')
        # Calling fill(args, kwargs) (line 114)
        fill_call_result_19079 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), fill_19076, *[str_19077], **kwargs_19078)
        
        
        # Call to interleave(...): (line 115)
        # Processing the call arguments (line 115)

        @norecursion
        def _stypy_temp_lambda_29(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_29'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_29', 115, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_29.stypy_localization = localization
            _stypy_temp_lambda_29.stypy_type_of_self = None
            _stypy_temp_lambda_29.stypy_type_store = module_type_store
            _stypy_temp_lambda_29.stypy_function_name = '_stypy_temp_lambda_29'
            _stypy_temp_lambda_29.stypy_param_names_list = []
            _stypy_temp_lambda_29.stypy_varargs_param_name = None
            _stypy_temp_lambda_29.stypy_kwargs_param_name = None
            _stypy_temp_lambda_29.stypy_call_defaults = defaults
            _stypy_temp_lambda_29.stypy_call_varargs = varargs
            _stypy_temp_lambda_29.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_29', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_29', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 115)
            # Processing the call arguments (line 115)
            str_19083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 38), 'str', ', ')
            # Processing the call keyword arguments (line 115)
            kwargs_19084 = {}
            # Getting the type of 'self' (line 115)
            self_19081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 115)
            write_19082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 27), self_19081, 'write')
            # Calling write(args, kwargs) (line 115)
            write_call_result_19085 = invoke(stypy.reporting.localization.Localization(__file__, 115, 27), write_19082, *[str_19083], **kwargs_19084)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'stypy_return_type', write_call_result_19085)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_29' in the type store
            # Getting the type of 'stypy_return_type' (line 115)
            stypy_return_type_19086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19086)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_29'
            return stypy_return_type_19086

        # Assigning a type to the variable '_stypy_temp_lambda_29' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), '_stypy_temp_lambda_29', _stypy_temp_lambda_29)
        # Getting the type of '_stypy_temp_lambda_29' (line 115)
        _stypy_temp_lambda_29_19087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), '_stypy_temp_lambda_29')
        # Getting the type of 'self' (line 115)
        self_19088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 115)
        visit_19089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 45), self_19088, 'visit')
        # Getting the type of 't' (line 115)
        t_19090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 57), 't', False)
        # Obtaining the member 'names' of a type (line 115)
        names_19091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 57), t_19090, 'names')
        # Processing the call keyword arguments (line 115)
        kwargs_19092 = {}
        # Getting the type of 'interleave' (line 115)
        interleave_19080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 115)
        interleave_call_result_19093 = invoke(stypy.reporting.localization.Localization(__file__, 115, 8), interleave_19080, *[_stypy_temp_lambda_29_19087, visit_19089, names_19091], **kwargs_19092)
        
        
        # Call to write(...): (line 116)
        # Processing the call arguments (line 116)
        str_19096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'str', '\n')
        # Processing the call keyword arguments (line 116)
        kwargs_19097 = {}
        # Getting the type of 'self' (line 116)
        self_19094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 116)
        write_19095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_19094, 'write')
        # Calling write(args, kwargs) (line 116)
        write_call_result_19098 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), write_19095, *[str_19096], **kwargs_19097)
        
        
        # ################# End of 'visit_Import(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Import' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_19099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19099)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Import'
        return stypy_return_type_19099


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
        t_19100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 11), 't')
        # Obtaining the member 'module' of a type (line 120)
        module_19101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 11), t_19100, 'module')
        
        # Getting the type of 't' (line 120)
        t_19102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 24), 't')
        # Obtaining the member 'module' of a type (line 120)
        module_19103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 24), t_19102, 'module')
        str_19104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 36), 'str', '__future__')
        # Applying the binary operator '==' (line 120)
        result_eq_19105 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 24), '==', module_19103, str_19104)
        
        # Applying the binary operator 'and' (line 120)
        result_and_keyword_19106 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 11), 'and', module_19101, result_eq_19105)
        
        # Testing if the type of an if condition is none (line 120)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 120, 8), result_and_keyword_19106):
            pass
        else:
            
            # Testing the type of an if condition (line 120)
            if_condition_19107 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 8), result_and_keyword_19106)
            # Assigning a type to the variable 'if_condition_19107' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'if_condition_19107', if_condition_19107)
            # SSA begins for if statement (line 120)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to extend(...): (line 121)
            # Processing the call arguments (line 121)
            # Calculating generator expression
            module_type_store = module_type_store.open_function_context('list comprehension expression', 121, 39, True)
            # Calculating comprehension expression
            # Getting the type of 't' (line 121)
            t_19113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 55), 't', False)
            # Obtaining the member 'names' of a type (line 121)
            names_19114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 55), t_19113, 'names')
            comprehension_19115 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 39), names_19114)
            # Assigning a type to the variable 'n' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 39), 'n', comprehension_19115)
            # Getting the type of 'n' (line 121)
            n_19111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 39), 'n', False)
            # Obtaining the member 'name' of a type (line 121)
            name_19112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 39), n_19111, 'name')
            list_19116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 39), 'list')
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 39), list_19116, name_19112)
            # Processing the call keyword arguments (line 121)
            kwargs_19117 = {}
            # Getting the type of 'self' (line 121)
            self_19108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'self', False)
            # Obtaining the member 'future_imports' of a type (line 121)
            future_imports_19109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), self_19108, 'future_imports')
            # Obtaining the member 'extend' of a type (line 121)
            extend_19110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), future_imports_19109, 'extend')
            # Calling extend(args, kwargs) (line 121)
            extend_call_result_19118 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), extend_19110, *[list_19116], **kwargs_19117)
            
            # SSA join for if statement (line 120)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to fill(...): (line 123)
        # Processing the call arguments (line 123)
        str_19121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 18), 'str', 'from ')
        # Processing the call keyword arguments (line 123)
        kwargs_19122 = {}
        # Getting the type of 'self' (line 123)
        self_19119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 123)
        fill_19120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_19119, 'fill')
        # Calling fill(args, kwargs) (line 123)
        fill_call_result_19123 = invoke(stypy.reporting.localization.Localization(__file__, 123, 8), fill_19120, *[str_19121], **kwargs_19122)
        
        
        # Call to write(...): (line 124)
        # Processing the call arguments (line 124)
        str_19126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 19), 'str', '.')
        # Getting the type of 't' (line 124)
        t_19127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 25), 't', False)
        # Obtaining the member 'level' of a type (line 124)
        level_19128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 25), t_19127, 'level')
        # Applying the binary operator '*' (line 124)
        result_mul_19129 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 19), '*', str_19126, level_19128)
        
        # Processing the call keyword arguments (line 124)
        kwargs_19130 = {}
        # Getting the type of 'self' (line 124)
        self_19124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 124)
        write_19125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), self_19124, 'write')
        # Calling write(args, kwargs) (line 124)
        write_call_result_19131 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), write_19125, *[result_mul_19129], **kwargs_19130)
        
        # Getting the type of 't' (line 125)
        t_19132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 't')
        # Obtaining the member 'module' of a type (line 125)
        module_19133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 11), t_19132, 'module')
        # Testing if the type of an if condition is none (line 125)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 8), module_19133):
            pass
        else:
            
            # Testing the type of an if condition (line 125)
            if_condition_19134 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), module_19133)
            # Assigning a type to the variable 'if_condition_19134' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_19134', if_condition_19134)
            # SSA begins for if statement (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 126)
            # Processing the call arguments (line 126)
            # Getting the type of 't' (line 126)
            t_19137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 't', False)
            # Obtaining the member 'module' of a type (line 126)
            module_19138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), t_19137, 'module')
            # Processing the call keyword arguments (line 126)
            kwargs_19139 = {}
            # Getting the type of 'self' (line 126)
            self_19135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 126)
            write_19136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), self_19135, 'write')
            # Calling write(args, kwargs) (line 126)
            write_call_result_19140 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), write_19136, *[module_19138], **kwargs_19139)
            
            # SSA join for if statement (line 125)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 127)
        # Processing the call arguments (line 127)
        str_19143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 19), 'str', ' import ')
        # Processing the call keyword arguments (line 127)
        kwargs_19144 = {}
        # Getting the type of 'self' (line 127)
        self_19141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 127)
        write_19142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_19141, 'write')
        # Calling write(args, kwargs) (line 127)
        write_call_result_19145 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), write_19142, *[str_19143], **kwargs_19144)
        
        
        # Call to interleave(...): (line 128)
        # Processing the call arguments (line 128)

        @norecursion
        def _stypy_temp_lambda_30(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_30'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_30', 128, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_30.stypy_localization = localization
            _stypy_temp_lambda_30.stypy_type_of_self = None
            _stypy_temp_lambda_30.stypy_type_store = module_type_store
            _stypy_temp_lambda_30.stypy_function_name = '_stypy_temp_lambda_30'
            _stypy_temp_lambda_30.stypy_param_names_list = []
            _stypy_temp_lambda_30.stypy_varargs_param_name = None
            _stypy_temp_lambda_30.stypy_kwargs_param_name = None
            _stypy_temp_lambda_30.stypy_call_defaults = defaults
            _stypy_temp_lambda_30.stypy_call_varargs = varargs
            _stypy_temp_lambda_30.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_30', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_30', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 128)
            # Processing the call arguments (line 128)
            str_19149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 38), 'str', ', ')
            # Processing the call keyword arguments (line 128)
            kwargs_19150 = {}
            # Getting the type of 'self' (line 128)
            self_19147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 128)
            write_19148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 27), self_19147, 'write')
            # Calling write(args, kwargs) (line 128)
            write_call_result_19151 = invoke(stypy.reporting.localization.Localization(__file__, 128, 27), write_19148, *[str_19149], **kwargs_19150)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'stypy_return_type', write_call_result_19151)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_30' in the type store
            # Getting the type of 'stypy_return_type' (line 128)
            stypy_return_type_19152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19152)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_30'
            return stypy_return_type_19152

        # Assigning a type to the variable '_stypy_temp_lambda_30' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), '_stypy_temp_lambda_30', _stypy_temp_lambda_30)
        # Getting the type of '_stypy_temp_lambda_30' (line 128)
        _stypy_temp_lambda_30_19153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), '_stypy_temp_lambda_30')
        # Getting the type of 'self' (line 128)
        self_19154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 128)
        visit_19155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 45), self_19154, 'visit')
        # Getting the type of 't' (line 128)
        t_19156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 57), 't', False)
        # Obtaining the member 'names' of a type (line 128)
        names_19157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 57), t_19156, 'names')
        # Processing the call keyword arguments (line 128)
        kwargs_19158 = {}
        # Getting the type of 'interleave' (line 128)
        interleave_19146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 128)
        interleave_call_result_19159 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), interleave_19146, *[_stypy_temp_lambda_30_19153, visit_19155, names_19157], **kwargs_19158)
        
        
        # Call to write(...): (line 129)
        # Processing the call arguments (line 129)
        str_19162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 19), 'str', '\n')
        # Processing the call keyword arguments (line 129)
        kwargs_19163 = {}
        # Getting the type of 'self' (line 129)
        self_19160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 129)
        write_19161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), self_19160, 'write')
        # Calling write(args, kwargs) (line 129)
        write_call_result_19164 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), write_19161, *[str_19162], **kwargs_19163)
        
        
        # ################# End of 'visit_ImportFrom(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ImportFrom' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_19165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19165)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ImportFrom'
        return stypy_return_type_19165


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
        kwargs_19168 = {}
        # Getting the type of 'self' (line 132)
        self_19166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 132)
        fill_19167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), self_19166, 'fill')
        # Calling fill(args, kwargs) (line 132)
        fill_call_result_19169 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), fill_19167, *[], **kwargs_19168)
        
        
        # Getting the type of 't' (line 133)
        t_19170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 't')
        # Obtaining the member 'targets' of a type (line 133)
        targets_19171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 22), t_19170, 'targets')
        # Assigning a type to the variable 'targets_19171' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'targets_19171', targets_19171)
        # Testing if the for loop is going to be iterated (line 133)
        # Testing the type of a for loop iterable (line 133)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 133, 8), targets_19171)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 133, 8), targets_19171):
            # Getting the type of the for loop variable (line 133)
            for_loop_var_19172 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 133, 8), targets_19171)
            # Assigning a type to the variable 'target' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'target', for_loop_var_19172)
            # SSA begins for a for statement (line 133)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 134)
            # Processing the call arguments (line 134)
            # Getting the type of 'target' (line 134)
            target_19175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'target', False)
            # Processing the call keyword arguments (line 134)
            kwargs_19176 = {}
            # Getting the type of 'self' (line 134)
            self_19173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 134)
            visit_19174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 12), self_19173, 'visit')
            # Calling visit(args, kwargs) (line 134)
            visit_call_result_19177 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), visit_19174, *[target_19175], **kwargs_19176)
            
            
            # Call to write(...): (line 135)
            # Processing the call arguments (line 135)
            str_19180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 23), 'str', ' = ')
            # Processing the call keyword arguments (line 135)
            kwargs_19181 = {}
            # Getting the type of 'self' (line 135)
            self_19178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 135)
            write_19179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 12), self_19178, 'write')
            # Calling write(args, kwargs) (line 135)
            write_call_result_19182 = invoke(stypy.reporting.localization.Localization(__file__, 135, 12), write_19179, *[str_19180], **kwargs_19181)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to visit(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 't' (line 136)
        t_19185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 't', False)
        # Obtaining the member 'value' of a type (line 136)
        value_19186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 19), t_19185, 'value')
        # Processing the call keyword arguments (line 136)
        kwargs_19187 = {}
        # Getting the type of 'self' (line 136)
        self_19183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 136)
        visit_19184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 8), self_19183, 'visit')
        # Calling visit(args, kwargs) (line 136)
        visit_call_result_19188 = invoke(stypy.reporting.localization.Localization(__file__, 136, 8), visit_19184, *[value_19186], **kwargs_19187)
        
        
        # ################# End of 'visit_Assign(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Assign' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_19189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19189)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Assign'
        return stypy_return_type_19189


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
        kwargs_19192 = {}
        # Getting the type of 'self' (line 139)
        self_19190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 139)
        fill_19191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 8), self_19190, 'fill')
        # Calling fill(args, kwargs) (line 139)
        fill_call_result_19193 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), fill_19191, *[], **kwargs_19192)
        
        
        # Call to visit(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 't' (line 140)
        t_19196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 19), 't', False)
        # Obtaining the member 'target' of a type (line 140)
        target_19197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 19), t_19196, 'target')
        # Processing the call keyword arguments (line 140)
        kwargs_19198 = {}
        # Getting the type of 'self' (line 140)
        self_19194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 140)
        visit_19195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_19194, 'visit')
        # Calling visit(args, kwargs) (line 140)
        visit_call_result_19199 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), visit_19195, *[target_19197], **kwargs_19198)
        
        
        # Call to write(...): (line 141)
        # Processing the call arguments (line 141)
        str_19202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 19), 'str', ' ')
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 141)
        t_19203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 36), 't', False)
        # Obtaining the member 'op' of a type (line 141)
        op_19204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 36), t_19203, 'op')
        # Obtaining the member '__class__' of a type (line 141)
        class___19205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 36), op_19204, '__class__')
        # Obtaining the member '__name__' of a type (line 141)
        name___19206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 36), class___19205, '__name__')
        # Getting the type of 'self' (line 141)
        self_19207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 'self', False)
        # Obtaining the member 'binop' of a type (line 141)
        binop_19208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 25), self_19207, 'binop')
        # Obtaining the member '__getitem__' of a type (line 141)
        getitem___19209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 25), binop_19208, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 141)
        subscript_call_result_19210 = invoke(stypy.reporting.localization.Localization(__file__, 141, 25), getitem___19209, name___19206)
        
        # Applying the binary operator '+' (line 141)
        result_add_19211 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 19), '+', str_19202, subscript_call_result_19210)
        
        str_19212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 63), 'str', '= ')
        # Applying the binary operator '+' (line 141)
        result_add_19213 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 61), '+', result_add_19211, str_19212)
        
        # Processing the call keyword arguments (line 141)
        kwargs_19214 = {}
        # Getting the type of 'self' (line 141)
        self_19200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 141)
        write_19201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), self_19200, 'write')
        # Calling write(args, kwargs) (line 141)
        write_call_result_19215 = invoke(stypy.reporting.localization.Localization(__file__, 141, 8), write_19201, *[result_add_19213], **kwargs_19214)
        
        
        # Call to visit(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 't' (line 142)
        t_19218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 't', False)
        # Obtaining the member 'value' of a type (line 142)
        value_19219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 19), t_19218, 'value')
        # Processing the call keyword arguments (line 142)
        kwargs_19220 = {}
        # Getting the type of 'self' (line 142)
        self_19216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 142)
        visit_19217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), self_19216, 'visit')
        # Calling visit(args, kwargs) (line 142)
        visit_call_result_19221 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), visit_19217, *[value_19219], **kwargs_19220)
        
        
        # ################# End of 'visit_AugAssign(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_AugAssign' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_19222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19222)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_AugAssign'
        return stypy_return_type_19222


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
        str_19225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 18), 'str', 'return')
        # Processing the call keyword arguments (line 145)
        kwargs_19226 = {}
        # Getting the type of 'self' (line 145)
        self_19223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 145)
        fill_19224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_19223, 'fill')
        # Calling fill(args, kwargs) (line 145)
        fill_call_result_19227 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), fill_19224, *[str_19225], **kwargs_19226)
        
        # Getting the type of 't' (line 146)
        t_19228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 't')
        # Obtaining the member 'value' of a type (line 146)
        value_19229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 11), t_19228, 'value')
        # Testing if the type of an if condition is none (line 146)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 146, 8), value_19229):
            pass
        else:
            
            # Testing the type of an if condition (line 146)
            if_condition_19230 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 8), value_19229)
            # Assigning a type to the variable 'if_condition_19230' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'if_condition_19230', if_condition_19230)
            # SSA begins for if statement (line 146)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 147)
            # Processing the call arguments (line 147)
            str_19233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 23), 'str', ' ')
            # Processing the call keyword arguments (line 147)
            kwargs_19234 = {}
            # Getting the type of 'self' (line 147)
            self_19231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 147)
            write_19232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), self_19231, 'write')
            # Calling write(args, kwargs) (line 147)
            write_call_result_19235 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), write_19232, *[str_19233], **kwargs_19234)
            
            
            # Call to visit(...): (line 148)
            # Processing the call arguments (line 148)
            # Getting the type of 't' (line 148)
            t_19238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 23), 't', False)
            # Obtaining the member 'value' of a type (line 148)
            value_19239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 23), t_19238, 'value')
            # Processing the call keyword arguments (line 148)
            kwargs_19240 = {}
            # Getting the type of 'self' (line 148)
            self_19236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 148)
            visit_19237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 12), self_19236, 'visit')
            # Calling visit(args, kwargs) (line 148)
            visit_call_result_19241 = invoke(stypy.reporting.localization.Localization(__file__, 148, 12), visit_19237, *[value_19239], **kwargs_19240)
            
            # SSA join for if statement (line 146)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Return(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Return' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_19242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19242)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Return'
        return stypy_return_type_19242


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
        str_19245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 18), 'str', 'pass')
        # Processing the call keyword arguments (line 152)
        kwargs_19246 = {}
        # Getting the type of 'self' (line 152)
        self_19243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 152)
        fill_19244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 8), self_19243, 'fill')
        # Calling fill(args, kwargs) (line 152)
        fill_call_result_19247 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), fill_19244, *[str_19245], **kwargs_19246)
        
        
        # ################# End of 'visit_Pass(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Pass' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_19248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19248)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Pass'
        return stypy_return_type_19248


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
        str_19251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 18), 'str', 'break')
        # Processing the call keyword arguments (line 155)
        kwargs_19252 = {}
        # Getting the type of 'self' (line 155)
        self_19249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 155)
        fill_19250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 8), self_19249, 'fill')
        # Calling fill(args, kwargs) (line 155)
        fill_call_result_19253 = invoke(stypy.reporting.localization.Localization(__file__, 155, 8), fill_19250, *[str_19251], **kwargs_19252)
        
        
        # ################# End of 'visit_Break(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Break' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_19254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19254)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Break'
        return stypy_return_type_19254


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
        str_19257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 18), 'str', 'continue')
        # Processing the call keyword arguments (line 158)
        kwargs_19258 = {}
        # Getting the type of 'self' (line 158)
        self_19255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 158)
        fill_19256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), self_19255, 'fill')
        # Calling fill(args, kwargs) (line 158)
        fill_call_result_19259 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), fill_19256, *[str_19257], **kwargs_19258)
        
        
        # ################# End of 'visit_Continue(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Continue' in the type store
        # Getting the type of 'stypy_return_type' (line 157)
        stypy_return_type_19260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19260)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Continue'
        return stypy_return_type_19260


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
        str_19263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 18), 'str', 'del ')
        # Processing the call keyword arguments (line 161)
        kwargs_19264 = {}
        # Getting the type of 'self' (line 161)
        self_19261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 161)
        fill_19262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), self_19261, 'fill')
        # Calling fill(args, kwargs) (line 161)
        fill_call_result_19265 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), fill_19262, *[str_19263], **kwargs_19264)
        
        
        # Call to interleave(...): (line 162)
        # Processing the call arguments (line 162)

        @norecursion
        def _stypy_temp_lambda_31(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_31'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_31', 162, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_31.stypy_localization = localization
            _stypy_temp_lambda_31.stypy_type_of_self = None
            _stypy_temp_lambda_31.stypy_type_store = module_type_store
            _stypy_temp_lambda_31.stypy_function_name = '_stypy_temp_lambda_31'
            _stypy_temp_lambda_31.stypy_param_names_list = []
            _stypy_temp_lambda_31.stypy_varargs_param_name = None
            _stypy_temp_lambda_31.stypy_kwargs_param_name = None
            _stypy_temp_lambda_31.stypy_call_defaults = defaults
            _stypy_temp_lambda_31.stypy_call_varargs = varargs
            _stypy_temp_lambda_31.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_31', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_31', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 162)
            # Processing the call arguments (line 162)
            str_19269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 38), 'str', ', ')
            # Processing the call keyword arguments (line 162)
            kwargs_19270 = {}
            # Getting the type of 'self' (line 162)
            self_19267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 162)
            write_19268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 27), self_19267, 'write')
            # Calling write(args, kwargs) (line 162)
            write_call_result_19271 = invoke(stypy.reporting.localization.Localization(__file__, 162, 27), write_19268, *[str_19269], **kwargs_19270)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'stypy_return_type', write_call_result_19271)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_31' in the type store
            # Getting the type of 'stypy_return_type' (line 162)
            stypy_return_type_19272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19272)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_31'
            return stypy_return_type_19272

        # Assigning a type to the variable '_stypy_temp_lambda_31' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), '_stypy_temp_lambda_31', _stypy_temp_lambda_31)
        # Getting the type of '_stypy_temp_lambda_31' (line 162)
        _stypy_temp_lambda_31_19273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), '_stypy_temp_lambda_31')
        # Getting the type of 'self' (line 162)
        self_19274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 162)
        visit_19275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 45), self_19274, 'visit')
        # Getting the type of 't' (line 162)
        t_19276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 57), 't', False)
        # Obtaining the member 'targets' of a type (line 162)
        targets_19277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 57), t_19276, 'targets')
        # Processing the call keyword arguments (line 162)
        kwargs_19278 = {}
        # Getting the type of 'interleave' (line 162)
        interleave_19266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 162)
        interleave_call_result_19279 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), interleave_19266, *[_stypy_temp_lambda_31_19273, visit_19275, targets_19277], **kwargs_19278)
        
        
        # ################# End of 'visit_Delete(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Delete' in the type store
        # Getting the type of 'stypy_return_type' (line 160)
        stypy_return_type_19280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19280)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Delete'
        return stypy_return_type_19280


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
        str_19283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 18), 'str', 'assert ')
        # Processing the call keyword arguments (line 165)
        kwargs_19284 = {}
        # Getting the type of 'self' (line 165)
        self_19281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 165)
        fill_19282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), self_19281, 'fill')
        # Calling fill(args, kwargs) (line 165)
        fill_call_result_19285 = invoke(stypy.reporting.localization.Localization(__file__, 165, 8), fill_19282, *[str_19283], **kwargs_19284)
        
        
        # Call to visit(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 't' (line 166)
        t_19288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 19), 't', False)
        # Obtaining the member 'test' of a type (line 166)
        test_19289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 19), t_19288, 'test')
        # Processing the call keyword arguments (line 166)
        kwargs_19290 = {}
        # Getting the type of 'self' (line 166)
        self_19286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 166)
        visit_19287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_19286, 'visit')
        # Calling visit(args, kwargs) (line 166)
        visit_call_result_19291 = invoke(stypy.reporting.localization.Localization(__file__, 166, 8), visit_19287, *[test_19289], **kwargs_19290)
        
        # Getting the type of 't' (line 167)
        t_19292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 't')
        # Obtaining the member 'msg' of a type (line 167)
        msg_19293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 11), t_19292, 'msg')
        # Testing if the type of an if condition is none (line 167)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 167, 8), msg_19293):
            pass
        else:
            
            # Testing the type of an if condition (line 167)
            if_condition_19294 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 8), msg_19293)
            # Assigning a type to the variable 'if_condition_19294' (line 167)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'if_condition_19294', if_condition_19294)
            # SSA begins for if statement (line 167)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 168)
            # Processing the call arguments (line 168)
            str_19297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 23), 'str', ', ')
            # Processing the call keyword arguments (line 168)
            kwargs_19298 = {}
            # Getting the type of 'self' (line 168)
            self_19295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 168)
            write_19296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), self_19295, 'write')
            # Calling write(args, kwargs) (line 168)
            write_call_result_19299 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), write_19296, *[str_19297], **kwargs_19298)
            
            
            # Call to visit(...): (line 169)
            # Processing the call arguments (line 169)
            # Getting the type of 't' (line 169)
            t_19302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), 't', False)
            # Obtaining the member 'msg' of a type (line 169)
            msg_19303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 23), t_19302, 'msg')
            # Processing the call keyword arguments (line 169)
            kwargs_19304 = {}
            # Getting the type of 'self' (line 169)
            self_19300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 169)
            visit_19301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 12), self_19300, 'visit')
            # Calling visit(args, kwargs) (line 169)
            visit_call_result_19305 = invoke(stypy.reporting.localization.Localization(__file__, 169, 12), visit_19301, *[msg_19303], **kwargs_19304)
            
            # SSA join for if statement (line 167)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Assert(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Assert' in the type store
        # Getting the type of 'stypy_return_type' (line 164)
        stypy_return_type_19306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19306)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Assert'
        return stypy_return_type_19306


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
        str_19309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 18), 'str', 'exec ')
        # Processing the call keyword arguments (line 172)
        kwargs_19310 = {}
        # Getting the type of 'self' (line 172)
        self_19307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 172)
        fill_19308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), self_19307, 'fill')
        # Calling fill(args, kwargs) (line 172)
        fill_call_result_19311 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), fill_19308, *[str_19309], **kwargs_19310)
        
        
        # Call to visit(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 't' (line 173)
        t_19314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 't', False)
        # Obtaining the member 'body' of a type (line 173)
        body_19315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 19), t_19314, 'body')
        # Processing the call keyword arguments (line 173)
        kwargs_19316 = {}
        # Getting the type of 'self' (line 173)
        self_19312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 173)
        visit_19313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), self_19312, 'visit')
        # Calling visit(args, kwargs) (line 173)
        visit_call_result_19317 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), visit_19313, *[body_19315], **kwargs_19316)
        
        # Getting the type of 't' (line 174)
        t_19318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 11), 't')
        # Obtaining the member 'globals' of a type (line 174)
        globals_19319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 11), t_19318, 'globals')
        # Testing if the type of an if condition is none (line 174)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 174, 8), globals_19319):
            pass
        else:
            
            # Testing the type of an if condition (line 174)
            if_condition_19320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 8), globals_19319)
            # Assigning a type to the variable 'if_condition_19320' (line 174)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'if_condition_19320', if_condition_19320)
            # SSA begins for if statement (line 174)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 175)
            # Processing the call arguments (line 175)
            str_19323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 23), 'str', ' in ')
            # Processing the call keyword arguments (line 175)
            kwargs_19324 = {}
            # Getting the type of 'self' (line 175)
            self_19321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 175)
            write_19322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 12), self_19321, 'write')
            # Calling write(args, kwargs) (line 175)
            write_call_result_19325 = invoke(stypy.reporting.localization.Localization(__file__, 175, 12), write_19322, *[str_19323], **kwargs_19324)
            
            
            # Call to visit(...): (line 176)
            # Processing the call arguments (line 176)
            # Getting the type of 't' (line 176)
            t_19328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 't', False)
            # Obtaining the member 'globals' of a type (line 176)
            globals_19329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 23), t_19328, 'globals')
            # Processing the call keyword arguments (line 176)
            kwargs_19330 = {}
            # Getting the type of 'self' (line 176)
            self_19326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 176)
            visit_19327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 12), self_19326, 'visit')
            # Calling visit(args, kwargs) (line 176)
            visit_call_result_19331 = invoke(stypy.reporting.localization.Localization(__file__, 176, 12), visit_19327, *[globals_19329], **kwargs_19330)
            
            # SSA join for if statement (line 174)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 177)
        t_19332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 't')
        # Obtaining the member 'locals' of a type (line 177)
        locals_19333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 11), t_19332, 'locals')
        # Testing if the type of an if condition is none (line 177)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 177, 8), locals_19333):
            pass
        else:
            
            # Testing the type of an if condition (line 177)
            if_condition_19334 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 8), locals_19333)
            # Assigning a type to the variable 'if_condition_19334' (line 177)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'if_condition_19334', if_condition_19334)
            # SSA begins for if statement (line 177)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 178)
            # Processing the call arguments (line 178)
            str_19337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 23), 'str', ', ')
            # Processing the call keyword arguments (line 178)
            kwargs_19338 = {}
            # Getting the type of 'self' (line 178)
            self_19335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 178)
            write_19336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), self_19335, 'write')
            # Calling write(args, kwargs) (line 178)
            write_call_result_19339 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), write_19336, *[str_19337], **kwargs_19338)
            
            
            # Call to visit(...): (line 179)
            # Processing the call arguments (line 179)
            # Getting the type of 't' (line 179)
            t_19342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 23), 't', False)
            # Obtaining the member 'locals' of a type (line 179)
            locals_19343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 23), t_19342, 'locals')
            # Processing the call keyword arguments (line 179)
            kwargs_19344 = {}
            # Getting the type of 'self' (line 179)
            self_19340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 179)
            visit_19341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), self_19340, 'visit')
            # Calling visit(args, kwargs) (line 179)
            visit_call_result_19345 = invoke(stypy.reporting.localization.Localization(__file__, 179, 12), visit_19341, *[locals_19343], **kwargs_19344)
            
            # SSA join for if statement (line 177)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Exec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Exec' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_19346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19346)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Exec'
        return stypy_return_type_19346


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
        str_19349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 18), 'str', 'print ')
        # Processing the call keyword arguments (line 182)
        kwargs_19350 = {}
        # Getting the type of 'self' (line 182)
        self_19347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 182)
        fill_19348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), self_19347, 'fill')
        # Calling fill(args, kwargs) (line 182)
        fill_call_result_19351 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), fill_19348, *[str_19349], **kwargs_19350)
        
        
        # Assigning a Name to a Name (line 183):
        
        # Assigning a Name to a Name (line 183):
        # Getting the type of 'False' (line 183)
        False_19352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'False')
        # Assigning a type to the variable 'do_comma' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'do_comma', False_19352)
        # Getting the type of 't' (line 184)
        t_19353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 't')
        # Obtaining the member 'dest' of a type (line 184)
        dest_19354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 11), t_19353, 'dest')
        # Testing if the type of an if condition is none (line 184)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 184, 8), dest_19354):
            pass
        else:
            
            # Testing the type of an if condition (line 184)
            if_condition_19355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 8), dest_19354)
            # Assigning a type to the variable 'if_condition_19355' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'if_condition_19355', if_condition_19355)
            # SSA begins for if statement (line 184)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 185)
            # Processing the call arguments (line 185)
            str_19358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 23), 'str', '>>')
            # Processing the call keyword arguments (line 185)
            kwargs_19359 = {}
            # Getting the type of 'self' (line 185)
            self_19356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 185)
            write_19357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 12), self_19356, 'write')
            # Calling write(args, kwargs) (line 185)
            write_call_result_19360 = invoke(stypy.reporting.localization.Localization(__file__, 185, 12), write_19357, *[str_19358], **kwargs_19359)
            
            
            # Call to visit(...): (line 186)
            # Processing the call arguments (line 186)
            # Getting the type of 't' (line 186)
            t_19363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 23), 't', False)
            # Obtaining the member 'dest' of a type (line 186)
            dest_19364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 23), t_19363, 'dest')
            # Processing the call keyword arguments (line 186)
            kwargs_19365 = {}
            # Getting the type of 'self' (line 186)
            self_19361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 186)
            visit_19362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 12), self_19361, 'visit')
            # Calling visit(args, kwargs) (line 186)
            visit_call_result_19366 = invoke(stypy.reporting.localization.Localization(__file__, 186, 12), visit_19362, *[dest_19364], **kwargs_19365)
            
            
            # Assigning a Name to a Name (line 187):
            
            # Assigning a Name to a Name (line 187):
            # Getting the type of 'True' (line 187)
            True_19367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 23), 'True')
            # Assigning a type to the variable 'do_comma' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'do_comma', True_19367)
            # SSA join for if statement (line 184)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 't' (line 188)
        t_19368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 17), 't')
        # Obtaining the member 'values' of a type (line 188)
        values_19369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 17), t_19368, 'values')
        # Assigning a type to the variable 'values_19369' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'values_19369', values_19369)
        # Testing if the for loop is going to be iterated (line 188)
        # Testing the type of a for loop iterable (line 188)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 188, 8), values_19369)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 188, 8), values_19369):
            # Getting the type of the for loop variable (line 188)
            for_loop_var_19370 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 188, 8), values_19369)
            # Assigning a type to the variable 'e' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'e', for_loop_var_19370)
            # SSA begins for a for statement (line 188)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'do_comma' (line 189)
            do_comma_19371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'do_comma')
            # Testing if the type of an if condition is none (line 189)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 189, 12), do_comma_19371):
                
                # Assigning a Name to a Name (line 192):
                
                # Assigning a Name to a Name (line 192):
                # Getting the type of 'True' (line 192)
                True_19378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'True')
                # Assigning a type to the variable 'do_comma' (line 192)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'do_comma', True_19378)
            else:
                
                # Testing the type of an if condition (line 189)
                if_condition_19372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 12), do_comma_19371)
                # Assigning a type to the variable 'if_condition_19372' (line 189)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'if_condition_19372', if_condition_19372)
                # SSA begins for if statement (line 189)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 190)
                # Processing the call arguments (line 190)
                str_19375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 27), 'str', ', ')
                # Processing the call keyword arguments (line 190)
                kwargs_19376 = {}
                # Getting the type of 'self' (line 190)
                self_19373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 190)
                write_19374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 16), self_19373, 'write')
                # Calling write(args, kwargs) (line 190)
                write_call_result_19377 = invoke(stypy.reporting.localization.Localization(__file__, 190, 16), write_19374, *[str_19375], **kwargs_19376)
                
                # SSA branch for the else part of an if statement (line 189)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 192):
                
                # Assigning a Name to a Name (line 192):
                # Getting the type of 'True' (line 192)
                True_19378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'True')
                # Assigning a type to the variable 'do_comma' (line 192)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 16), 'do_comma', True_19378)
                # SSA join for if statement (line 189)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to visit(...): (line 193)
            # Processing the call arguments (line 193)
            # Getting the type of 'e' (line 193)
            e_19381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 23), 'e', False)
            # Processing the call keyword arguments (line 193)
            kwargs_19382 = {}
            # Getting the type of 'self' (line 193)
            self_19379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 193)
            visit_19380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), self_19379, 'visit')
            # Calling visit(args, kwargs) (line 193)
            visit_call_result_19383 = invoke(stypy.reporting.localization.Localization(__file__, 193, 12), visit_19380, *[e_19381], **kwargs_19382)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 't' (line 194)
        t_19384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 't')
        # Obtaining the member 'nl' of a type (line 194)
        nl_19385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 15), t_19384, 'nl')
        # Applying the 'not' unary operator (line 194)
        result_not__19386 = python_operator(stypy.reporting.localization.Localization(__file__, 194, 11), 'not', nl_19385)
        
        # Testing if the type of an if condition is none (line 194)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 194, 8), result_not__19386):
            pass
        else:
            
            # Testing the type of an if condition (line 194)
            if_condition_19387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 8), result_not__19386)
            # Assigning a type to the variable 'if_condition_19387' (line 194)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'if_condition_19387', if_condition_19387)
            # SSA begins for if statement (line 194)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 195)
            # Processing the call arguments (line 195)
            str_19390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 23), 'str', ',')
            # Processing the call keyword arguments (line 195)
            kwargs_19391 = {}
            # Getting the type of 'self' (line 195)
            self_19388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 195)
            write_19389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), self_19388, 'write')
            # Calling write(args, kwargs) (line 195)
            write_call_result_19392 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), write_19389, *[str_19390], **kwargs_19391)
            
            # SSA join for if statement (line 194)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Print(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Print' in the type store
        # Getting the type of 'stypy_return_type' (line 181)
        stypy_return_type_19393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19393)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Print'
        return stypy_return_type_19393


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
        str_19396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 18), 'str', 'global ')
        # Processing the call keyword arguments (line 198)
        kwargs_19397 = {}
        # Getting the type of 'self' (line 198)
        self_19394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 198)
        fill_19395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), self_19394, 'fill')
        # Calling fill(args, kwargs) (line 198)
        fill_call_result_19398 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), fill_19395, *[str_19396], **kwargs_19397)
        
        
        # Call to interleave(...): (line 199)
        # Processing the call arguments (line 199)

        @norecursion
        def _stypy_temp_lambda_32(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_32'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_32', 199, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_32.stypy_localization = localization
            _stypy_temp_lambda_32.stypy_type_of_self = None
            _stypy_temp_lambda_32.stypy_type_store = module_type_store
            _stypy_temp_lambda_32.stypy_function_name = '_stypy_temp_lambda_32'
            _stypy_temp_lambda_32.stypy_param_names_list = []
            _stypy_temp_lambda_32.stypy_varargs_param_name = None
            _stypy_temp_lambda_32.stypy_kwargs_param_name = None
            _stypy_temp_lambda_32.stypy_call_defaults = defaults
            _stypy_temp_lambda_32.stypy_call_varargs = varargs
            _stypy_temp_lambda_32.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_32', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_32', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 199)
            # Processing the call arguments (line 199)
            str_19402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 38), 'str', ', ')
            # Processing the call keyword arguments (line 199)
            kwargs_19403 = {}
            # Getting the type of 'self' (line 199)
            self_19400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 199)
            write_19401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 27), self_19400, 'write')
            # Calling write(args, kwargs) (line 199)
            write_call_result_19404 = invoke(stypy.reporting.localization.Localization(__file__, 199, 27), write_19401, *[str_19402], **kwargs_19403)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'stypy_return_type', write_call_result_19404)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_32' in the type store
            # Getting the type of 'stypy_return_type' (line 199)
            stypy_return_type_19405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_19405)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_32'
            return stypy_return_type_19405

        # Assigning a type to the variable '_stypy_temp_lambda_32' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), '_stypy_temp_lambda_32', _stypy_temp_lambda_32)
        # Getting the type of '_stypy_temp_lambda_32' (line 199)
        _stypy_temp_lambda_32_19406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 19), '_stypy_temp_lambda_32')
        # Getting the type of 'self' (line 199)
        self_19407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 45), 'self', False)
        # Obtaining the member 'write' of a type (line 199)
        write_19408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 45), self_19407, 'write')
        # Getting the type of 't' (line 199)
        t_19409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 57), 't', False)
        # Obtaining the member 'names' of a type (line 199)
        names_19410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 57), t_19409, 'names')
        # Processing the call keyword arguments (line 199)
        kwargs_19411 = {}
        # Getting the type of 'interleave' (line 199)
        interleave_19399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 199)
        interleave_call_result_19412 = invoke(stypy.reporting.localization.Localization(__file__, 199, 8), interleave_19399, *[_stypy_temp_lambda_32_19406, write_19408, names_19410], **kwargs_19411)
        
        
        # ################# End of 'visit_Global(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Global' in the type store
        # Getting the type of 'stypy_return_type' (line 197)
        stypy_return_type_19413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19413)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Global'
        return stypy_return_type_19413


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
        str_19416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 19), 'str', '(')
        # Processing the call keyword arguments (line 202)
        kwargs_19417 = {}
        # Getting the type of 'self' (line 202)
        self_19414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 202)
        write_19415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), self_19414, 'write')
        # Calling write(args, kwargs) (line 202)
        write_call_result_19418 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), write_19415, *[str_19416], **kwargs_19417)
        
        
        # Call to write(...): (line 203)
        # Processing the call arguments (line 203)
        str_19421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 19), 'str', 'yield')
        # Processing the call keyword arguments (line 203)
        kwargs_19422 = {}
        # Getting the type of 'self' (line 203)
        self_19419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 203)
        write_19420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), self_19419, 'write')
        # Calling write(args, kwargs) (line 203)
        write_call_result_19423 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), write_19420, *[str_19421], **kwargs_19422)
        
        # Getting the type of 't' (line 204)
        t_19424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 't')
        # Obtaining the member 'value' of a type (line 204)
        value_19425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 11), t_19424, 'value')
        # Testing if the type of an if condition is none (line 204)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 204, 8), value_19425):
            pass
        else:
            
            # Testing the type of an if condition (line 204)
            if_condition_19426 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 8), value_19425)
            # Assigning a type to the variable 'if_condition_19426' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'if_condition_19426', if_condition_19426)
            # SSA begins for if statement (line 204)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 205)
            # Processing the call arguments (line 205)
            str_19429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 23), 'str', ' ')
            # Processing the call keyword arguments (line 205)
            kwargs_19430 = {}
            # Getting the type of 'self' (line 205)
            self_19427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 205)
            write_19428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 12), self_19427, 'write')
            # Calling write(args, kwargs) (line 205)
            write_call_result_19431 = invoke(stypy.reporting.localization.Localization(__file__, 205, 12), write_19428, *[str_19429], **kwargs_19430)
            
            
            # Call to visit(...): (line 206)
            # Processing the call arguments (line 206)
            # Getting the type of 't' (line 206)
            t_19434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 23), 't', False)
            # Obtaining the member 'value' of a type (line 206)
            value_19435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 23), t_19434, 'value')
            # Processing the call keyword arguments (line 206)
            kwargs_19436 = {}
            # Getting the type of 'self' (line 206)
            self_19432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 206)
            visit_19433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), self_19432, 'visit')
            # Calling visit(args, kwargs) (line 206)
            visit_call_result_19437 = invoke(stypy.reporting.localization.Localization(__file__, 206, 12), visit_19433, *[value_19435], **kwargs_19436)
            
            # SSA join for if statement (line 204)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 207)
        # Processing the call arguments (line 207)
        str_19440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 19), 'str', ')')
        # Processing the call keyword arguments (line 207)
        kwargs_19441 = {}
        # Getting the type of 'self' (line 207)
        self_19438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 207)
        write_19439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_19438, 'write')
        # Calling write(args, kwargs) (line 207)
        write_call_result_19442 = invoke(stypy.reporting.localization.Localization(__file__, 207, 8), write_19439, *[str_19440], **kwargs_19441)
        
        
        # ################# End of 'visit_Yield(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Yield' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_19443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19443)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Yield'
        return stypy_return_type_19443


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
        str_19446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 18), 'str', 'raise ')
        # Processing the call keyword arguments (line 210)
        kwargs_19447 = {}
        # Getting the type of 'self' (line 210)
        self_19444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 210)
        fill_19445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 8), self_19444, 'fill')
        # Calling fill(args, kwargs) (line 210)
        fill_call_result_19448 = invoke(stypy.reporting.localization.Localization(__file__, 210, 8), fill_19445, *[str_19446], **kwargs_19447)
        
        # Getting the type of 't' (line 211)
        t_19449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 't')
        # Obtaining the member 'type' of a type (line 211)
        type_19450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 11), t_19449, 'type')
        # Testing if the type of an if condition is none (line 211)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 211, 8), type_19450):
            pass
        else:
            
            # Testing the type of an if condition (line 211)
            if_condition_19451 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 8), type_19450)
            # Assigning a type to the variable 'if_condition_19451' (line 211)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'if_condition_19451', if_condition_19451)
            # SSA begins for if statement (line 211)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to visit(...): (line 212)
            # Processing the call arguments (line 212)
            # Getting the type of 't' (line 212)
            t_19454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 23), 't', False)
            # Obtaining the member 'type' of a type (line 212)
            type_19455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 23), t_19454, 'type')
            # Processing the call keyword arguments (line 212)
            kwargs_19456 = {}
            # Getting the type of 'self' (line 212)
            self_19452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 212)
            visit_19453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), self_19452, 'visit')
            # Calling visit(args, kwargs) (line 212)
            visit_call_result_19457 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), visit_19453, *[type_19455], **kwargs_19456)
            
            # SSA join for if statement (line 211)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 213)
        t_19458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 't')
        # Obtaining the member 'inst' of a type (line 213)
        inst_19459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 11), t_19458, 'inst')
        # Testing if the type of an if condition is none (line 213)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 213, 8), inst_19459):
            pass
        else:
            
            # Testing the type of an if condition (line 213)
            if_condition_19460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 8), inst_19459)
            # Assigning a type to the variable 'if_condition_19460' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'if_condition_19460', if_condition_19460)
            # SSA begins for if statement (line 213)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 214)
            # Processing the call arguments (line 214)
            str_19463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 23), 'str', ', ')
            # Processing the call keyword arguments (line 214)
            kwargs_19464 = {}
            # Getting the type of 'self' (line 214)
            self_19461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 214)
            write_19462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 12), self_19461, 'write')
            # Calling write(args, kwargs) (line 214)
            write_call_result_19465 = invoke(stypy.reporting.localization.Localization(__file__, 214, 12), write_19462, *[str_19463], **kwargs_19464)
            
            
            # Call to visit(...): (line 215)
            # Processing the call arguments (line 215)
            # Getting the type of 't' (line 215)
            t_19468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 't', False)
            # Obtaining the member 'inst' of a type (line 215)
            inst_19469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 23), t_19468, 'inst')
            # Processing the call keyword arguments (line 215)
            kwargs_19470 = {}
            # Getting the type of 'self' (line 215)
            self_19466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 215)
            visit_19467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), self_19466, 'visit')
            # Calling visit(args, kwargs) (line 215)
            visit_call_result_19471 = invoke(stypy.reporting.localization.Localization(__file__, 215, 12), visit_19467, *[inst_19469], **kwargs_19470)
            
            # SSA join for if statement (line 213)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 216)
        t_19472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 11), 't')
        # Obtaining the member 'tback' of a type (line 216)
        tback_19473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 11), t_19472, 'tback')
        # Testing if the type of an if condition is none (line 216)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 216, 8), tback_19473):
            pass
        else:
            
            # Testing the type of an if condition (line 216)
            if_condition_19474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 216, 8), tback_19473)
            # Assigning a type to the variable 'if_condition_19474' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'if_condition_19474', if_condition_19474)
            # SSA begins for if statement (line 216)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 217)
            # Processing the call arguments (line 217)
            str_19477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 23), 'str', ', ')
            # Processing the call keyword arguments (line 217)
            kwargs_19478 = {}
            # Getting the type of 'self' (line 217)
            self_19475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 217)
            write_19476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), self_19475, 'write')
            # Calling write(args, kwargs) (line 217)
            write_call_result_19479 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), write_19476, *[str_19477], **kwargs_19478)
            
            
            # Call to visit(...): (line 218)
            # Processing the call arguments (line 218)
            # Getting the type of 't' (line 218)
            t_19482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 't', False)
            # Obtaining the member 'tback' of a type (line 218)
            tback_19483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 23), t_19482, 'tback')
            # Processing the call keyword arguments (line 218)
            kwargs_19484 = {}
            # Getting the type of 'self' (line 218)
            self_19480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 218)
            visit_19481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 12), self_19480, 'visit')
            # Calling visit(args, kwargs) (line 218)
            visit_call_result_19485 = invoke(stypy.reporting.localization.Localization(__file__, 218, 12), visit_19481, *[tback_19483], **kwargs_19484)
            
            # SSA join for if statement (line 216)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Raise(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Raise' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_19486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19486)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Raise'
        return stypy_return_type_19486


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
        str_19489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 18), 'str', 'try')
        # Processing the call keyword arguments (line 221)
        kwargs_19490 = {}
        # Getting the type of 'self' (line 221)
        self_19487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 221)
        fill_19488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_19487, 'fill')
        # Calling fill(args, kwargs) (line 221)
        fill_call_result_19491 = invoke(stypy.reporting.localization.Localization(__file__, 221, 8), fill_19488, *[str_19489], **kwargs_19490)
        
        
        # Call to enter(...): (line 222)
        # Processing the call keyword arguments (line 222)
        kwargs_19494 = {}
        # Getting the type of 'self' (line 222)
        self_19492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 222)
        enter_19493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 8), self_19492, 'enter')
        # Calling enter(args, kwargs) (line 222)
        enter_call_result_19495 = invoke(stypy.reporting.localization.Localization(__file__, 222, 8), enter_19493, *[], **kwargs_19494)
        
        
        # Call to visit(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 't' (line 223)
        t_19498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 19), 't', False)
        # Obtaining the member 'body' of a type (line 223)
        body_19499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 19), t_19498, 'body')
        # Processing the call keyword arguments (line 223)
        kwargs_19500 = {}
        # Getting the type of 'self' (line 223)
        self_19496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 223)
        visit_19497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 8), self_19496, 'visit')
        # Calling visit(args, kwargs) (line 223)
        visit_call_result_19501 = invoke(stypy.reporting.localization.Localization(__file__, 223, 8), visit_19497, *[body_19499], **kwargs_19500)
        
        
        # Call to leave(...): (line 224)
        # Processing the call keyword arguments (line 224)
        kwargs_19504 = {}
        # Getting the type of 'self' (line 224)
        self_19502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 224)
        leave_19503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 8), self_19502, 'leave')
        # Calling leave(args, kwargs) (line 224)
        leave_call_result_19505 = invoke(stypy.reporting.localization.Localization(__file__, 224, 8), leave_19503, *[], **kwargs_19504)
        
        
        # Getting the type of 't' (line 226)
        t_19506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 18), 't')
        # Obtaining the member 'handlers' of a type (line 226)
        handlers_19507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 18), t_19506, 'handlers')
        # Assigning a type to the variable 'handlers_19507' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'handlers_19507', handlers_19507)
        # Testing if the for loop is going to be iterated (line 226)
        # Testing the type of a for loop iterable (line 226)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 226, 8), handlers_19507)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 226, 8), handlers_19507):
            # Getting the type of the for loop variable (line 226)
            for_loop_var_19508 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 226, 8), handlers_19507)
            # Assigning a type to the variable 'ex' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'ex', for_loop_var_19508)
            # SSA begins for a for statement (line 226)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 227)
            # Processing the call arguments (line 227)
            # Getting the type of 'ex' (line 227)
            ex_19511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 23), 'ex', False)
            # Processing the call keyword arguments (line 227)
            kwargs_19512 = {}
            # Getting the type of 'self' (line 227)
            self_19509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 227)
            visit_19510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 12), self_19509, 'visit')
            # Calling visit(args, kwargs) (line 227)
            visit_call_result_19513 = invoke(stypy.reporting.localization.Localization(__file__, 227, 12), visit_19510, *[ex_19511], **kwargs_19512)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 't' (line 228)
        t_19514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 't')
        # Obtaining the member 'orelse' of a type (line 228)
        orelse_19515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 11), t_19514, 'orelse')
        # Testing if the type of an if condition is none (line 228)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 228, 8), orelse_19515):
            pass
        else:
            
            # Testing the type of an if condition (line 228)
            if_condition_19516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 228, 8), orelse_19515)
            # Assigning a type to the variable 'if_condition_19516' (line 228)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'if_condition_19516', if_condition_19516)
            # SSA begins for if statement (line 228)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fill(...): (line 229)
            # Processing the call arguments (line 229)
            str_19519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 22), 'str', 'else')
            # Processing the call keyword arguments (line 229)
            kwargs_19520 = {}
            # Getting the type of 'self' (line 229)
            self_19517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 229)
            fill_19518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 12), self_19517, 'fill')
            # Calling fill(args, kwargs) (line 229)
            fill_call_result_19521 = invoke(stypy.reporting.localization.Localization(__file__, 229, 12), fill_19518, *[str_19519], **kwargs_19520)
            
            
            # Call to enter(...): (line 230)
            # Processing the call keyword arguments (line 230)
            kwargs_19524 = {}
            # Getting the type of 'self' (line 230)
            self_19522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 230)
            enter_19523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), self_19522, 'enter')
            # Calling enter(args, kwargs) (line 230)
            enter_call_result_19525 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), enter_19523, *[], **kwargs_19524)
            
            
            # Call to visit(...): (line 231)
            # Processing the call arguments (line 231)
            # Getting the type of 't' (line 231)
            t_19528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 23), 't', False)
            # Obtaining the member 'orelse' of a type (line 231)
            orelse_19529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 23), t_19528, 'orelse')
            # Processing the call keyword arguments (line 231)
            kwargs_19530 = {}
            # Getting the type of 'self' (line 231)
            self_19526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 231)
            visit_19527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 12), self_19526, 'visit')
            # Calling visit(args, kwargs) (line 231)
            visit_call_result_19531 = invoke(stypy.reporting.localization.Localization(__file__, 231, 12), visit_19527, *[orelse_19529], **kwargs_19530)
            
            
            # Call to leave(...): (line 232)
            # Processing the call keyword arguments (line 232)
            kwargs_19534 = {}
            # Getting the type of 'self' (line 232)
            self_19532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 232)
            leave_19533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 12), self_19532, 'leave')
            # Calling leave(args, kwargs) (line 232)
            leave_call_result_19535 = invoke(stypy.reporting.localization.Localization(__file__, 232, 12), leave_19533, *[], **kwargs_19534)
            
            # SSA join for if statement (line 228)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_TryExcept(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_TryExcept' in the type store
        # Getting the type of 'stypy_return_type' (line 220)
        stypy_return_type_19536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19536)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_TryExcept'
        return stypy_return_type_19536


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
        t_19538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 15), 't', False)
        # Obtaining the member 'body' of a type (line 235)
        body_19539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 15), t_19538, 'body')
        # Processing the call keyword arguments (line 235)
        kwargs_19540 = {}
        # Getting the type of 'len' (line 235)
        len_19537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'len', False)
        # Calling len(args, kwargs) (line 235)
        len_call_result_19541 = invoke(stypy.reporting.localization.Localization(__file__, 235, 11), len_19537, *[body_19539], **kwargs_19540)
        
        int_19542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 26), 'int')
        # Applying the binary operator '==' (line 235)
        result_eq_19543 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 11), '==', len_call_result_19541, int_19542)
        
        
        # Call to isinstance(...): (line 235)
        # Processing the call arguments (line 235)
        
        # Obtaining the type of the subscript
        int_19545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 50), 'int')
        # Getting the type of 't' (line 235)
        t_19546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 43), 't', False)
        # Obtaining the member 'body' of a type (line 235)
        body_19547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 43), t_19546, 'body')
        # Obtaining the member '__getitem__' of a type (line 235)
        getitem___19548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 43), body_19547, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 235)
        subscript_call_result_19549 = invoke(stypy.reporting.localization.Localization(__file__, 235, 43), getitem___19548, int_19545)
        
        # Getting the type of 'ast' (line 235)
        ast_19550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 54), 'ast', False)
        # Obtaining the member 'TryExcept' of a type (line 235)
        TryExcept_19551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 54), ast_19550, 'TryExcept')
        # Processing the call keyword arguments (line 235)
        kwargs_19552 = {}
        # Getting the type of 'isinstance' (line 235)
        isinstance_19544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 32), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 235)
        isinstance_call_result_19553 = invoke(stypy.reporting.localization.Localization(__file__, 235, 32), isinstance_19544, *[subscript_call_result_19549, TryExcept_19551], **kwargs_19552)
        
        # Applying the binary operator 'and' (line 235)
        result_and_keyword_19554 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 11), 'and', result_eq_19543, isinstance_call_result_19553)
        
        # Testing if the type of an if condition is none (line 235)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 235, 8), result_and_keyword_19554):
            
            # Call to fill(...): (line 239)
            # Processing the call arguments (line 239)
            str_19564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 22), 'str', 'try')
            # Processing the call keyword arguments (line 239)
            kwargs_19565 = {}
            # Getting the type of 'self' (line 239)
            self_19562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 239)
            fill_19563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), self_19562, 'fill')
            # Calling fill(args, kwargs) (line 239)
            fill_call_result_19566 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), fill_19563, *[str_19564], **kwargs_19565)
            
            
            # Call to enter(...): (line 240)
            # Processing the call keyword arguments (line 240)
            kwargs_19569 = {}
            # Getting the type of 'self' (line 240)
            self_19567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 240)
            enter_19568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), self_19567, 'enter')
            # Calling enter(args, kwargs) (line 240)
            enter_call_result_19570 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), enter_19568, *[], **kwargs_19569)
            
            
            # Call to visit(...): (line 241)
            # Processing the call arguments (line 241)
            # Getting the type of 't' (line 241)
            t_19573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 't', False)
            # Obtaining the member 'body' of a type (line 241)
            body_19574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 23), t_19573, 'body')
            # Processing the call keyword arguments (line 241)
            kwargs_19575 = {}
            # Getting the type of 'self' (line 241)
            self_19571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 241)
            visit_19572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), self_19571, 'visit')
            # Calling visit(args, kwargs) (line 241)
            visit_call_result_19576 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), visit_19572, *[body_19574], **kwargs_19575)
            
            
            # Call to leave(...): (line 242)
            # Processing the call keyword arguments (line 242)
            kwargs_19579 = {}
            # Getting the type of 'self' (line 242)
            self_19577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 242)
            leave_19578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), self_19577, 'leave')
            # Calling leave(args, kwargs) (line 242)
            leave_call_result_19580 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), leave_19578, *[], **kwargs_19579)
            
        else:
            
            # Testing the type of an if condition (line 235)
            if_condition_19555 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 8), result_and_keyword_19554)
            # Assigning a type to the variable 'if_condition_19555' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 8), 'if_condition_19555', if_condition_19555)
            # SSA begins for if statement (line 235)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to visit(...): (line 237)
            # Processing the call arguments (line 237)
            # Getting the type of 't' (line 237)
            t_19558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 23), 't', False)
            # Obtaining the member 'body' of a type (line 237)
            body_19559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 23), t_19558, 'body')
            # Processing the call keyword arguments (line 237)
            kwargs_19560 = {}
            # Getting the type of 'self' (line 237)
            self_19556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 237)
            visit_19557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), self_19556, 'visit')
            # Calling visit(args, kwargs) (line 237)
            visit_call_result_19561 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), visit_19557, *[body_19559], **kwargs_19560)
            
            # SSA branch for the else part of an if statement (line 235)
            module_type_store.open_ssa_branch('else')
            
            # Call to fill(...): (line 239)
            # Processing the call arguments (line 239)
            str_19564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 22), 'str', 'try')
            # Processing the call keyword arguments (line 239)
            kwargs_19565 = {}
            # Getting the type of 'self' (line 239)
            self_19562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 239)
            fill_19563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 12), self_19562, 'fill')
            # Calling fill(args, kwargs) (line 239)
            fill_call_result_19566 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), fill_19563, *[str_19564], **kwargs_19565)
            
            
            # Call to enter(...): (line 240)
            # Processing the call keyword arguments (line 240)
            kwargs_19569 = {}
            # Getting the type of 'self' (line 240)
            self_19567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 240)
            enter_19568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), self_19567, 'enter')
            # Calling enter(args, kwargs) (line 240)
            enter_call_result_19570 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), enter_19568, *[], **kwargs_19569)
            
            
            # Call to visit(...): (line 241)
            # Processing the call arguments (line 241)
            # Getting the type of 't' (line 241)
            t_19573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 't', False)
            # Obtaining the member 'body' of a type (line 241)
            body_19574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 23), t_19573, 'body')
            # Processing the call keyword arguments (line 241)
            kwargs_19575 = {}
            # Getting the type of 'self' (line 241)
            self_19571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 241)
            visit_19572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), self_19571, 'visit')
            # Calling visit(args, kwargs) (line 241)
            visit_call_result_19576 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), visit_19572, *[body_19574], **kwargs_19575)
            
            
            # Call to leave(...): (line 242)
            # Processing the call keyword arguments (line 242)
            kwargs_19579 = {}
            # Getting the type of 'self' (line 242)
            self_19577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 242)
            leave_19578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 12), self_19577, 'leave')
            # Calling leave(args, kwargs) (line 242)
            leave_call_result_19580 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), leave_19578, *[], **kwargs_19579)
            
            # SSA join for if statement (line 235)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to fill(...): (line 244)
        # Processing the call arguments (line 244)
        str_19583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 18), 'str', 'finally')
        # Processing the call keyword arguments (line 244)
        kwargs_19584 = {}
        # Getting the type of 'self' (line 244)
        self_19581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 244)
        fill_19582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), self_19581, 'fill')
        # Calling fill(args, kwargs) (line 244)
        fill_call_result_19585 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), fill_19582, *[str_19583], **kwargs_19584)
        
        
        # Call to enter(...): (line 245)
        # Processing the call keyword arguments (line 245)
        kwargs_19588 = {}
        # Getting the type of 'self' (line 245)
        self_19586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 245)
        enter_19587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 8), self_19586, 'enter')
        # Calling enter(args, kwargs) (line 245)
        enter_call_result_19589 = invoke(stypy.reporting.localization.Localization(__file__, 245, 8), enter_19587, *[], **kwargs_19588)
        
        
        # Call to visit(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 't' (line 246)
        t_19592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 19), 't', False)
        # Obtaining the member 'finalbody' of a type (line 246)
        finalbody_19593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 19), t_19592, 'finalbody')
        # Processing the call keyword arguments (line 246)
        kwargs_19594 = {}
        # Getting the type of 'self' (line 246)
        self_19590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 246)
        visit_19591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 8), self_19590, 'visit')
        # Calling visit(args, kwargs) (line 246)
        visit_call_result_19595 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), visit_19591, *[finalbody_19593], **kwargs_19594)
        
        
        # Call to leave(...): (line 247)
        # Processing the call keyword arguments (line 247)
        kwargs_19598 = {}
        # Getting the type of 'self' (line 247)
        self_19596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 247)
        leave_19597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 8), self_19596, 'leave')
        # Calling leave(args, kwargs) (line 247)
        leave_call_result_19599 = invoke(stypy.reporting.localization.Localization(__file__, 247, 8), leave_19597, *[], **kwargs_19598)
        
        
        # ################# End of 'visit_TryFinally(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_TryFinally' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_19600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19600)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_TryFinally'
        return stypy_return_type_19600


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
        str_19603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 18), 'str', 'except')
        # Processing the call keyword arguments (line 250)
        kwargs_19604 = {}
        # Getting the type of 'self' (line 250)
        self_19601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 250)
        fill_19602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_19601, 'fill')
        # Calling fill(args, kwargs) (line 250)
        fill_call_result_19605 = invoke(stypy.reporting.localization.Localization(__file__, 250, 8), fill_19602, *[str_19603], **kwargs_19604)
        
        # Getting the type of 't' (line 251)
        t_19606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 't')
        # Obtaining the member 'type' of a type (line 251)
        type_19607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 11), t_19606, 'type')
        # Testing if the type of an if condition is none (line 251)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 251, 8), type_19607):
            pass
        else:
            
            # Testing the type of an if condition (line 251)
            if_condition_19608 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 8), type_19607)
            # Assigning a type to the variable 'if_condition_19608' (line 251)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'if_condition_19608', if_condition_19608)
            # SSA begins for if statement (line 251)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 252)
            # Processing the call arguments (line 252)
            str_19611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 23), 'str', ' ')
            # Processing the call keyword arguments (line 252)
            kwargs_19612 = {}
            # Getting the type of 'self' (line 252)
            self_19609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 252)
            write_19610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 12), self_19609, 'write')
            # Calling write(args, kwargs) (line 252)
            write_call_result_19613 = invoke(stypy.reporting.localization.Localization(__file__, 252, 12), write_19610, *[str_19611], **kwargs_19612)
            
            
            # Call to visit(...): (line 253)
            # Processing the call arguments (line 253)
            # Getting the type of 't' (line 253)
            t_19616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 23), 't', False)
            # Obtaining the member 'type' of a type (line 253)
            type_19617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 23), t_19616, 'type')
            # Processing the call keyword arguments (line 253)
            kwargs_19618 = {}
            # Getting the type of 'self' (line 253)
            self_19614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 253)
            visit_19615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 12), self_19614, 'visit')
            # Calling visit(args, kwargs) (line 253)
            visit_call_result_19619 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), visit_19615, *[type_19617], **kwargs_19618)
            
            # SSA join for if statement (line 251)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 254)
        t_19620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 11), 't')
        # Obtaining the member 'name' of a type (line 254)
        name_19621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 11), t_19620, 'name')
        # Testing if the type of an if condition is none (line 254)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 254, 8), name_19621):
            pass
        else:
            
            # Testing the type of an if condition (line 254)
            if_condition_19622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 254, 8), name_19621)
            # Assigning a type to the variable 'if_condition_19622' (line 254)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'if_condition_19622', if_condition_19622)
            # SSA begins for if statement (line 254)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 255)
            # Processing the call arguments (line 255)
            str_19625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 23), 'str', ' as ')
            # Processing the call keyword arguments (line 255)
            kwargs_19626 = {}
            # Getting the type of 'self' (line 255)
            self_19623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 255)
            write_19624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 12), self_19623, 'write')
            # Calling write(args, kwargs) (line 255)
            write_call_result_19627 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), write_19624, *[str_19625], **kwargs_19626)
            
            
            # Call to visit(...): (line 256)
            # Processing the call arguments (line 256)
            # Getting the type of 't' (line 256)
            t_19630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 23), 't', False)
            # Obtaining the member 'name' of a type (line 256)
            name_19631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 23), t_19630, 'name')
            # Processing the call keyword arguments (line 256)
            kwargs_19632 = {}
            # Getting the type of 'self' (line 256)
            self_19628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 256)
            visit_19629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), self_19628, 'visit')
            # Calling visit(args, kwargs) (line 256)
            visit_call_result_19633 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), visit_19629, *[name_19631], **kwargs_19632)
            
            # SSA join for if statement (line 254)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to enter(...): (line 257)
        # Processing the call keyword arguments (line 257)
        kwargs_19636 = {}
        # Getting the type of 'self' (line 257)
        self_19634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 257)
        enter_19635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_19634, 'enter')
        # Calling enter(args, kwargs) (line 257)
        enter_call_result_19637 = invoke(stypy.reporting.localization.Localization(__file__, 257, 8), enter_19635, *[], **kwargs_19636)
        
        
        # Call to visit(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 't' (line 258)
        t_19640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 19), 't', False)
        # Obtaining the member 'body' of a type (line 258)
        body_19641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 19), t_19640, 'body')
        # Processing the call keyword arguments (line 258)
        kwargs_19642 = {}
        # Getting the type of 'self' (line 258)
        self_19638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 258)
        visit_19639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), self_19638, 'visit')
        # Calling visit(args, kwargs) (line 258)
        visit_call_result_19643 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), visit_19639, *[body_19641], **kwargs_19642)
        
        
        # Call to leave(...): (line 259)
        # Processing the call keyword arguments (line 259)
        kwargs_19646 = {}
        # Getting the type of 'self' (line 259)
        self_19644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 259)
        leave_19645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 8), self_19644, 'leave')
        # Calling leave(args, kwargs) (line 259)
        leave_call_result_19647 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), leave_19645, *[], **kwargs_19646)
        
        
        # ################# End of 'visit_ExceptHandler(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ExceptHandler' in the type store
        # Getting the type of 'stypy_return_type' (line 249)
        stypy_return_type_19648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19648)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ExceptHandler'
        return stypy_return_type_19648


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
        str_19651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 19), 'str', '\n')
        # Processing the call keyword arguments (line 262)
        kwargs_19652 = {}
        # Getting the type of 'self' (line 262)
        self_19649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 262)
        write_19650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 8), self_19649, 'write')
        # Calling write(args, kwargs) (line 262)
        write_call_result_19653 = invoke(stypy.reporting.localization.Localization(__file__, 262, 8), write_19650, *[str_19651], **kwargs_19652)
        
        
        # Getting the type of 't' (line 263)
        t_19654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 20), 't')
        # Obtaining the member 'decorator_list' of a type (line 263)
        decorator_list_19655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 20), t_19654, 'decorator_list')
        # Assigning a type to the variable 'decorator_list_19655' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'decorator_list_19655', decorator_list_19655)
        # Testing if the for loop is going to be iterated (line 263)
        # Testing the type of a for loop iterable (line 263)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 263, 8), decorator_list_19655)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 263, 8), decorator_list_19655):
            # Getting the type of the for loop variable (line 263)
            for_loop_var_19656 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 263, 8), decorator_list_19655)
            # Assigning a type to the variable 'deco' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'deco', for_loop_var_19656)
            # SSA begins for a for statement (line 263)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to fill(...): (line 264)
            # Processing the call arguments (line 264)
            str_19659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 22), 'str', '@')
            # Processing the call keyword arguments (line 264)
            kwargs_19660 = {}
            # Getting the type of 'self' (line 264)
            self_19657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 264)
            fill_19658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), self_19657, 'fill')
            # Calling fill(args, kwargs) (line 264)
            fill_call_result_19661 = invoke(stypy.reporting.localization.Localization(__file__, 264, 12), fill_19658, *[str_19659], **kwargs_19660)
            
            
            # Call to visit(...): (line 265)
            # Processing the call arguments (line 265)
            # Getting the type of 'deco' (line 265)
            deco_19664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 23), 'deco', False)
            # Processing the call keyword arguments (line 265)
            kwargs_19665 = {}
            # Getting the type of 'self' (line 265)
            self_19662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 265)
            visit_19663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 12), self_19662, 'visit')
            # Calling visit(args, kwargs) (line 265)
            visit_call_result_19666 = invoke(stypy.reporting.localization.Localization(__file__, 265, 12), visit_19663, *[deco_19664], **kwargs_19665)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to fill(...): (line 266)
        # Processing the call arguments (line 266)
        str_19669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 18), 'str', 'class ')
        # Getting the type of 't' (line 266)
        t_19670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 29), 't', False)
        # Obtaining the member 'name' of a type (line 266)
        name_19671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 29), t_19670, 'name')
        # Applying the binary operator '+' (line 266)
        result_add_19672 = python_operator(stypy.reporting.localization.Localization(__file__, 266, 18), '+', str_19669, name_19671)
        
        # Processing the call keyword arguments (line 266)
        kwargs_19673 = {}
        # Getting the type of 'self' (line 266)
        self_19667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 266)
        fill_19668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), self_19667, 'fill')
        # Calling fill(args, kwargs) (line 266)
        fill_call_result_19674 = invoke(stypy.reporting.localization.Localization(__file__, 266, 8), fill_19668, *[result_add_19672], **kwargs_19673)
        
        # Getting the type of 't' (line 267)
        t_19675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 11), 't')
        # Obtaining the member 'bases' of a type (line 267)
        bases_19676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 11), t_19675, 'bases')
        # Testing if the type of an if condition is none (line 267)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 267, 8), bases_19676):
            pass
        else:
            
            # Testing the type of an if condition (line 267)
            if_condition_19677 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 267, 8), bases_19676)
            # Assigning a type to the variable 'if_condition_19677' (line 267)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'if_condition_19677', if_condition_19677)
            # SSA begins for if statement (line 267)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 268)
            # Processing the call arguments (line 268)
            str_19680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 23), 'str', '(')
            # Processing the call keyword arguments (line 268)
            kwargs_19681 = {}
            # Getting the type of 'self' (line 268)
            self_19678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 268)
            write_19679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 12), self_19678, 'write')
            # Calling write(args, kwargs) (line 268)
            write_call_result_19682 = invoke(stypy.reporting.localization.Localization(__file__, 268, 12), write_19679, *[str_19680], **kwargs_19681)
            
            
            # Getting the type of 't' (line 269)
            t_19683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 21), 't')
            # Obtaining the member 'bases' of a type (line 269)
            bases_19684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 21), t_19683, 'bases')
            # Assigning a type to the variable 'bases_19684' (line 269)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'bases_19684', bases_19684)
            # Testing if the for loop is going to be iterated (line 269)
            # Testing the type of a for loop iterable (line 269)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 269, 12), bases_19684)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 269, 12), bases_19684):
                # Getting the type of the for loop variable (line 269)
                for_loop_var_19685 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 269, 12), bases_19684)
                # Assigning a type to the variable 'a' (line 269)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'a', for_loop_var_19685)
                # SSA begins for a for statement (line 269)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to visit(...): (line 270)
                # Processing the call arguments (line 270)
                # Getting the type of 'a' (line 270)
                a_19688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 27), 'a', False)
                # Processing the call keyword arguments (line 270)
                kwargs_19689 = {}
                # Getting the type of 'self' (line 270)
                self_19686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 16), 'self', False)
                # Obtaining the member 'visit' of a type (line 270)
                visit_19687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 16), self_19686, 'visit')
                # Calling visit(args, kwargs) (line 270)
                visit_call_result_19690 = invoke(stypy.reporting.localization.Localization(__file__, 270, 16), visit_19687, *[a_19688], **kwargs_19689)
                
                
                # Call to write(...): (line 271)
                # Processing the call arguments (line 271)
                str_19693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 27), 'str', ', ')
                # Processing the call keyword arguments (line 271)
                kwargs_19694 = {}
                # Getting the type of 'self' (line 271)
                self_19691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 271)
                write_19692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 16), self_19691, 'write')
                # Calling write(args, kwargs) (line 271)
                write_call_result_19695 = invoke(stypy.reporting.localization.Localization(__file__, 271, 16), write_19692, *[str_19693], **kwargs_19694)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to write(...): (line 272)
            # Processing the call arguments (line 272)
            str_19698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 23), 'str', ')')
            # Processing the call keyword arguments (line 272)
            kwargs_19699 = {}
            # Getting the type of 'self' (line 272)
            self_19696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 272)
            write_19697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 12), self_19696, 'write')
            # Calling write(args, kwargs) (line 272)
            write_call_result_19700 = invoke(stypy.reporting.localization.Localization(__file__, 272, 12), write_19697, *[str_19698], **kwargs_19699)
            
            # SSA join for if statement (line 267)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to enter(...): (line 273)
        # Processing the call keyword arguments (line 273)
        kwargs_19703 = {}
        # Getting the type of 'self' (line 273)
        self_19701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 273)
        enter_19702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 8), self_19701, 'enter')
        # Calling enter(args, kwargs) (line 273)
        enter_call_result_19704 = invoke(stypy.reporting.localization.Localization(__file__, 273, 8), enter_19702, *[], **kwargs_19703)
        
        
        # Call to visit(...): (line 274)
        # Processing the call arguments (line 274)
        # Getting the type of 't' (line 274)
        t_19707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 19), 't', False)
        # Obtaining the member 'body' of a type (line 274)
        body_19708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 19), t_19707, 'body')
        # Processing the call keyword arguments (line 274)
        kwargs_19709 = {}
        # Getting the type of 'self' (line 274)
        self_19705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 274)
        visit_19706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 8), self_19705, 'visit')
        # Calling visit(args, kwargs) (line 274)
        visit_call_result_19710 = invoke(stypy.reporting.localization.Localization(__file__, 274, 8), visit_19706, *[body_19708], **kwargs_19709)
        
        
        # Call to leave(...): (line 275)
        # Processing the call keyword arguments (line 275)
        kwargs_19713 = {}
        # Getting the type of 'self' (line 275)
        self_19711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 275)
        leave_19712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 8), self_19711, 'leave')
        # Calling leave(args, kwargs) (line 275)
        leave_call_result_19714 = invoke(stypy.reporting.localization.Localization(__file__, 275, 8), leave_19712, *[], **kwargs_19713)
        
        
        # ################# End of 'visit_ClassDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ClassDef' in the type store
        # Getting the type of 'stypy_return_type' (line 261)
        stypy_return_type_19715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19715)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ClassDef'
        return stypy_return_type_19715


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
        str_19718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 19), 'str', '\n')
        # Processing the call keyword arguments (line 278)
        kwargs_19719 = {}
        # Getting the type of 'self' (line 278)
        self_19716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 278)
        write_19717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 8), self_19716, 'write')
        # Calling write(args, kwargs) (line 278)
        write_call_result_19720 = invoke(stypy.reporting.localization.Localization(__file__, 278, 8), write_19717, *[str_19718], **kwargs_19719)
        
        
        # Getting the type of 't' (line 279)
        t_19721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 't')
        # Obtaining the member 'decorator_list' of a type (line 279)
        decorator_list_19722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 20), t_19721, 'decorator_list')
        # Assigning a type to the variable 'decorator_list_19722' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'decorator_list_19722', decorator_list_19722)
        # Testing if the for loop is going to be iterated (line 279)
        # Testing the type of a for loop iterable (line 279)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 279, 8), decorator_list_19722)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 279, 8), decorator_list_19722):
            # Getting the type of the for loop variable (line 279)
            for_loop_var_19723 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 279, 8), decorator_list_19722)
            # Assigning a type to the variable 'deco' (line 279)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'deco', for_loop_var_19723)
            # SSA begins for a for statement (line 279)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to fill(...): (line 280)
            # Processing the call arguments (line 280)
            str_19726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 22), 'str', '@')
            # Processing the call keyword arguments (line 280)
            kwargs_19727 = {}
            # Getting the type of 'self' (line 280)
            self_19724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 280)
            fill_19725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 12), self_19724, 'fill')
            # Calling fill(args, kwargs) (line 280)
            fill_call_result_19728 = invoke(stypy.reporting.localization.Localization(__file__, 280, 12), fill_19725, *[str_19726], **kwargs_19727)
            
            
            # Call to visit(...): (line 281)
            # Processing the call arguments (line 281)
            # Getting the type of 'deco' (line 281)
            deco_19731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 23), 'deco', False)
            # Processing the call keyword arguments (line 281)
            kwargs_19732 = {}
            # Getting the type of 'self' (line 281)
            self_19729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 281)
            visit_19730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 12), self_19729, 'visit')
            # Calling visit(args, kwargs) (line 281)
            visit_call_result_19733 = invoke(stypy.reporting.localization.Localization(__file__, 281, 12), visit_19730, *[deco_19731], **kwargs_19732)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to fill(...): (line 282)
        # Processing the call arguments (line 282)
        str_19736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 18), 'str', 'def ')
        # Getting the type of 't' (line 282)
        t_19737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 27), 't', False)
        # Obtaining the member 'name' of a type (line 282)
        name_19738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 27), t_19737, 'name')
        # Applying the binary operator '+' (line 282)
        result_add_19739 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 18), '+', str_19736, name_19738)
        
        str_19740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 36), 'str', '(')
        # Applying the binary operator '+' (line 282)
        result_add_19741 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 34), '+', result_add_19739, str_19740)
        
        # Processing the call keyword arguments (line 282)
        kwargs_19742 = {}
        # Getting the type of 'self' (line 282)
        self_19734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 282)
        fill_19735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 8), self_19734, 'fill')
        # Calling fill(args, kwargs) (line 282)
        fill_call_result_19743 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), fill_19735, *[result_add_19741], **kwargs_19742)
        
        
        # Call to visit(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 't' (line 283)
        t_19746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 19), 't', False)
        # Obtaining the member 'args' of a type (line 283)
        args_19747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 19), t_19746, 'args')
        # Processing the call keyword arguments (line 283)
        kwargs_19748 = {}
        # Getting the type of 'self' (line 283)
        self_19744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 283)
        visit_19745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 8), self_19744, 'visit')
        # Calling visit(args, kwargs) (line 283)
        visit_call_result_19749 = invoke(stypy.reporting.localization.Localization(__file__, 283, 8), visit_19745, *[args_19747], **kwargs_19748)
        
        
        # Call to write(...): (line 284)
        # Processing the call arguments (line 284)
        str_19752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 19), 'str', ')')
        # Processing the call keyword arguments (line 284)
        kwargs_19753 = {}
        # Getting the type of 'self' (line 284)
        self_19750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 284)
        write_19751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 8), self_19750, 'write')
        # Calling write(args, kwargs) (line 284)
        write_call_result_19754 = invoke(stypy.reporting.localization.Localization(__file__, 284, 8), write_19751, *[str_19752], **kwargs_19753)
        
        
        # Call to enter(...): (line 285)
        # Processing the call keyword arguments (line 285)
        kwargs_19757 = {}
        # Getting the type of 'self' (line 285)
        self_19755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 285)
        enter_19756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 8), self_19755, 'enter')
        # Calling enter(args, kwargs) (line 285)
        enter_call_result_19758 = invoke(stypy.reporting.localization.Localization(__file__, 285, 8), enter_19756, *[], **kwargs_19757)
        
        
        # Call to visit(...): (line 286)
        # Processing the call arguments (line 286)
        # Getting the type of 't' (line 286)
        t_19761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 19), 't', False)
        # Obtaining the member 'body' of a type (line 286)
        body_19762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 19), t_19761, 'body')
        # Processing the call keyword arguments (line 286)
        kwargs_19763 = {}
        # Getting the type of 'self' (line 286)
        self_19759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 286)
        visit_19760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 8), self_19759, 'visit')
        # Calling visit(args, kwargs) (line 286)
        visit_call_result_19764 = invoke(stypy.reporting.localization.Localization(__file__, 286, 8), visit_19760, *[body_19762], **kwargs_19763)
        
        
        # Call to write(...): (line 287)
        # Processing the call arguments (line 287)
        str_19767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 19), 'str', '\n')
        # Processing the call keyword arguments (line 287)
        kwargs_19768 = {}
        # Getting the type of 'self' (line 287)
        self_19765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 287)
        write_19766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 8), self_19765, 'write')
        # Calling write(args, kwargs) (line 287)
        write_call_result_19769 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), write_19766, *[str_19767], **kwargs_19768)
        
        
        # Call to leave(...): (line 288)
        # Processing the call keyword arguments (line 288)
        kwargs_19772 = {}
        # Getting the type of 'self' (line 288)
        self_19770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 288)
        leave_19771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 8), self_19770, 'leave')
        # Calling leave(args, kwargs) (line 288)
        leave_call_result_19773 = invoke(stypy.reporting.localization.Localization(__file__, 288, 8), leave_19771, *[], **kwargs_19772)
        
        
        # ################# End of 'visit_FunctionDef(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_FunctionDef' in the type store
        # Getting the type of 'stypy_return_type' (line 277)
        stypy_return_type_19774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19774)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_FunctionDef'
        return stypy_return_type_19774


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
        str_19777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 18), 'str', 'for ')
        # Processing the call keyword arguments (line 291)
        kwargs_19778 = {}
        # Getting the type of 'self' (line 291)
        self_19775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 291)
        fill_19776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 8), self_19775, 'fill')
        # Calling fill(args, kwargs) (line 291)
        fill_call_result_19779 = invoke(stypy.reporting.localization.Localization(__file__, 291, 8), fill_19776, *[str_19777], **kwargs_19778)
        
        
        # Call to visit(...): (line 292)
        # Processing the call arguments (line 292)
        # Getting the type of 't' (line 292)
        t_19782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 19), 't', False)
        # Obtaining the member 'target' of a type (line 292)
        target_19783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 19), t_19782, 'target')
        # Processing the call keyword arguments (line 292)
        kwargs_19784 = {}
        # Getting the type of 'self' (line 292)
        self_19780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 292)
        visit_19781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 8), self_19780, 'visit')
        # Calling visit(args, kwargs) (line 292)
        visit_call_result_19785 = invoke(stypy.reporting.localization.Localization(__file__, 292, 8), visit_19781, *[target_19783], **kwargs_19784)
        
        
        # Call to write(...): (line 293)
        # Processing the call arguments (line 293)
        str_19788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 19), 'str', ' in ')
        # Processing the call keyword arguments (line 293)
        kwargs_19789 = {}
        # Getting the type of 'self' (line 293)
        self_19786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 293)
        write_19787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 8), self_19786, 'write')
        # Calling write(args, kwargs) (line 293)
        write_call_result_19790 = invoke(stypy.reporting.localization.Localization(__file__, 293, 8), write_19787, *[str_19788], **kwargs_19789)
        
        
        # Call to visit(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 't' (line 294)
        t_19793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 19), 't', False)
        # Obtaining the member 'iter' of a type (line 294)
        iter_19794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 19), t_19793, 'iter')
        # Processing the call keyword arguments (line 294)
        kwargs_19795 = {}
        # Getting the type of 'self' (line 294)
        self_19791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 294)
        visit_19792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), self_19791, 'visit')
        # Calling visit(args, kwargs) (line 294)
        visit_call_result_19796 = invoke(stypy.reporting.localization.Localization(__file__, 294, 8), visit_19792, *[iter_19794], **kwargs_19795)
        
        
        # Call to enter(...): (line 295)
        # Processing the call keyword arguments (line 295)
        kwargs_19799 = {}
        # Getting the type of 'self' (line 295)
        self_19797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 295)
        enter_19798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 8), self_19797, 'enter')
        # Calling enter(args, kwargs) (line 295)
        enter_call_result_19800 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), enter_19798, *[], **kwargs_19799)
        
        
        # Call to visit(...): (line 296)
        # Processing the call arguments (line 296)
        # Getting the type of 't' (line 296)
        t_19803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 19), 't', False)
        # Obtaining the member 'body' of a type (line 296)
        body_19804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 19), t_19803, 'body')
        # Processing the call keyword arguments (line 296)
        kwargs_19805 = {}
        # Getting the type of 'self' (line 296)
        self_19801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 296)
        visit_19802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 8), self_19801, 'visit')
        # Calling visit(args, kwargs) (line 296)
        visit_call_result_19806 = invoke(stypy.reporting.localization.Localization(__file__, 296, 8), visit_19802, *[body_19804], **kwargs_19805)
        
        
        # Call to leave(...): (line 297)
        # Processing the call keyword arguments (line 297)
        kwargs_19809 = {}
        # Getting the type of 'self' (line 297)
        self_19807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 297)
        leave_19808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 8), self_19807, 'leave')
        # Calling leave(args, kwargs) (line 297)
        leave_call_result_19810 = invoke(stypy.reporting.localization.Localization(__file__, 297, 8), leave_19808, *[], **kwargs_19809)
        
        # Getting the type of 't' (line 298)
        t_19811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 11), 't')
        # Obtaining the member 'orelse' of a type (line 298)
        orelse_19812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 11), t_19811, 'orelse')
        # Testing if the type of an if condition is none (line 298)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 298, 8), orelse_19812):
            pass
        else:
            
            # Testing the type of an if condition (line 298)
            if_condition_19813 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 298, 8), orelse_19812)
            # Assigning a type to the variable 'if_condition_19813' (line 298)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'if_condition_19813', if_condition_19813)
            # SSA begins for if statement (line 298)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fill(...): (line 299)
            # Processing the call arguments (line 299)
            str_19816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 299, 22), 'str', 'else')
            # Processing the call keyword arguments (line 299)
            kwargs_19817 = {}
            # Getting the type of 'self' (line 299)
            self_19814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 299)
            fill_19815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 12), self_19814, 'fill')
            # Calling fill(args, kwargs) (line 299)
            fill_call_result_19818 = invoke(stypy.reporting.localization.Localization(__file__, 299, 12), fill_19815, *[str_19816], **kwargs_19817)
            
            
            # Call to enter(...): (line 300)
            # Processing the call keyword arguments (line 300)
            kwargs_19821 = {}
            # Getting the type of 'self' (line 300)
            self_19819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 300)
            enter_19820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 12), self_19819, 'enter')
            # Calling enter(args, kwargs) (line 300)
            enter_call_result_19822 = invoke(stypy.reporting.localization.Localization(__file__, 300, 12), enter_19820, *[], **kwargs_19821)
            
            
            # Call to visit(...): (line 301)
            # Processing the call arguments (line 301)
            # Getting the type of 't' (line 301)
            t_19825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 23), 't', False)
            # Obtaining the member 'orelse' of a type (line 301)
            orelse_19826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 23), t_19825, 'orelse')
            # Processing the call keyword arguments (line 301)
            kwargs_19827 = {}
            # Getting the type of 'self' (line 301)
            self_19823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 301)
            visit_19824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 12), self_19823, 'visit')
            # Calling visit(args, kwargs) (line 301)
            visit_call_result_19828 = invoke(stypy.reporting.localization.Localization(__file__, 301, 12), visit_19824, *[orelse_19826], **kwargs_19827)
            
            
            # Call to leave(...): (line 302)
            # Processing the call keyword arguments (line 302)
            kwargs_19831 = {}
            # Getting the type of 'self' (line 302)
            self_19829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 302)
            leave_19830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 12), self_19829, 'leave')
            # Calling leave(args, kwargs) (line 302)
            leave_call_result_19832 = invoke(stypy.reporting.localization.Localization(__file__, 302, 12), leave_19830, *[], **kwargs_19831)
            
            # SSA join for if statement (line 298)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_For(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_For' in the type store
        # Getting the type of 'stypy_return_type' (line 290)
        stypy_return_type_19833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19833)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_For'
        return stypy_return_type_19833


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
        str_19836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 19), 'str', '\n')
        # Processing the call keyword arguments (line 305)
        kwargs_19837 = {}
        # Getting the type of 'self' (line 305)
        self_19834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 305)
        write_19835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 8), self_19834, 'write')
        # Calling write(args, kwargs) (line 305)
        write_call_result_19838 = invoke(stypy.reporting.localization.Localization(__file__, 305, 8), write_19835, *[str_19836], **kwargs_19837)
        
        
        # Call to fill(...): (line 306)
        # Processing the call arguments (line 306)
        str_19841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 18), 'str', 'if ')
        # Processing the call keyword arguments (line 306)
        kwargs_19842 = {}
        # Getting the type of 'self' (line 306)
        self_19839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 306)
        fill_19840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 8), self_19839, 'fill')
        # Calling fill(args, kwargs) (line 306)
        fill_call_result_19843 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), fill_19840, *[str_19841], **kwargs_19842)
        
        
        # Call to visit(...): (line 307)
        # Processing the call arguments (line 307)
        # Getting the type of 't' (line 307)
        t_19846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 't', False)
        # Obtaining the member 'test' of a type (line 307)
        test_19847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 19), t_19846, 'test')
        # Processing the call keyword arguments (line 307)
        kwargs_19848 = {}
        # Getting the type of 'self' (line 307)
        self_19844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 307)
        visit_19845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), self_19844, 'visit')
        # Calling visit(args, kwargs) (line 307)
        visit_call_result_19849 = invoke(stypy.reporting.localization.Localization(__file__, 307, 8), visit_19845, *[test_19847], **kwargs_19848)
        
        
        # Call to enter(...): (line 308)
        # Processing the call keyword arguments (line 308)
        kwargs_19852 = {}
        # Getting the type of 'self' (line 308)
        self_19850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 308)
        enter_19851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 8), self_19850, 'enter')
        # Calling enter(args, kwargs) (line 308)
        enter_call_result_19853 = invoke(stypy.reporting.localization.Localization(__file__, 308, 8), enter_19851, *[], **kwargs_19852)
        
        
        # Call to visit(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 't' (line 309)
        t_19856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 19), 't', False)
        # Obtaining the member 'body' of a type (line 309)
        body_19857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 19), t_19856, 'body')
        # Processing the call keyword arguments (line 309)
        kwargs_19858 = {}
        # Getting the type of 'self' (line 309)
        self_19854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 309)
        visit_19855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), self_19854, 'visit')
        # Calling visit(args, kwargs) (line 309)
        visit_call_result_19859 = invoke(stypy.reporting.localization.Localization(__file__, 309, 8), visit_19855, *[body_19857], **kwargs_19858)
        
        
        # Call to leave(...): (line 310)
        # Processing the call keyword arguments (line 310)
        kwargs_19862 = {}
        # Getting the type of 'self' (line 310)
        self_19860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 310)
        leave_19861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 8), self_19860, 'leave')
        # Calling leave(args, kwargs) (line 310)
        leave_call_result_19863 = invoke(stypy.reporting.localization.Localization(__file__, 310, 8), leave_19861, *[], **kwargs_19862)
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 't' (line 312)
        t_19864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 15), 't')
        # Obtaining the member 'orelse' of a type (line 312)
        orelse_19865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 15), t_19864, 'orelse')
        
        
        # Call to len(...): (line 312)
        # Processing the call arguments (line 312)
        # Getting the type of 't' (line 312)
        t_19867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 32), 't', False)
        # Obtaining the member 'orelse' of a type (line 312)
        orelse_19868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 32), t_19867, 'orelse')
        # Processing the call keyword arguments (line 312)
        kwargs_19869 = {}
        # Getting the type of 'len' (line 312)
        len_19866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 28), 'len', False)
        # Calling len(args, kwargs) (line 312)
        len_call_result_19870 = invoke(stypy.reporting.localization.Localization(__file__, 312, 28), len_19866, *[orelse_19868], **kwargs_19869)
        
        int_19871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 312, 45), 'int')
        # Applying the binary operator '==' (line 312)
        result_eq_19872 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 28), '==', len_call_result_19870, int_19871)
        
        # Applying the binary operator 'and' (line 312)
        result_and_keyword_19873 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 15), 'and', orelse_19865, result_eq_19872)
        
        # Call to isinstance(...): (line 313)
        # Processing the call arguments (line 313)
        
        # Obtaining the type of the subscript
        int_19875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 39), 'int')
        # Getting the type of 't' (line 313)
        t_19876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 30), 't', False)
        # Obtaining the member 'orelse' of a type (line 313)
        orelse_19877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 30), t_19876, 'orelse')
        # Obtaining the member '__getitem__' of a type (line 313)
        getitem___19878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 30), orelse_19877, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 313)
        subscript_call_result_19879 = invoke(stypy.reporting.localization.Localization(__file__, 313, 30), getitem___19878, int_19875)
        
        # Getting the type of 'ast' (line 313)
        ast_19880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 43), 'ast', False)
        # Obtaining the member 'If' of a type (line 313)
        If_19881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 43), ast_19880, 'If')
        # Processing the call keyword arguments (line 313)
        kwargs_19882 = {}
        # Getting the type of 'isinstance' (line 313)
        isinstance_19874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 313)
        isinstance_call_result_19883 = invoke(stypy.reporting.localization.Localization(__file__, 313, 19), isinstance_19874, *[subscript_call_result_19879, If_19881], **kwargs_19882)
        
        # Applying the binary operator 'and' (line 312)
        result_and_keyword_19884 = python_operator(stypy.reporting.localization.Localization(__file__, 312, 15), 'and', result_and_keyword_19873, isinstance_call_result_19883)
        
        # Assigning a type to the variable 'result_and_keyword_19884' (line 312)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'result_and_keyword_19884', result_and_keyword_19884)
        # Testing if the while is going to be iterated (line 312)
        # Testing the type of an if condition (line 312)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 312, 8), result_and_keyword_19884)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 312, 8), result_and_keyword_19884):
            # SSA begins for while statement (line 312)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Subscript to a Name (line 314):
            
            # Assigning a Subscript to a Name (line 314):
            
            # Obtaining the type of the subscript
            int_19885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 314, 25), 'int')
            # Getting the type of 't' (line 314)
            t_19886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 't')
            # Obtaining the member 'orelse' of a type (line 314)
            orelse_19887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 16), t_19886, 'orelse')
            # Obtaining the member '__getitem__' of a type (line 314)
            getitem___19888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 16), orelse_19887, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 314)
            subscript_call_result_19889 = invoke(stypy.reporting.localization.Localization(__file__, 314, 16), getitem___19888, int_19885)
            
            # Assigning a type to the variable 't' (line 314)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 't', subscript_call_result_19889)
            
            # Call to fill(...): (line 315)
            # Processing the call arguments (line 315)
            str_19892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 22), 'str', 'elif ')
            # Processing the call keyword arguments (line 315)
            kwargs_19893 = {}
            # Getting the type of 'self' (line 315)
            self_19890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 315)
            fill_19891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 12), self_19890, 'fill')
            # Calling fill(args, kwargs) (line 315)
            fill_call_result_19894 = invoke(stypy.reporting.localization.Localization(__file__, 315, 12), fill_19891, *[str_19892], **kwargs_19893)
            
            
            # Call to visit(...): (line 316)
            # Processing the call arguments (line 316)
            # Getting the type of 't' (line 316)
            t_19897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 23), 't', False)
            # Obtaining the member 'test' of a type (line 316)
            test_19898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 23), t_19897, 'test')
            # Processing the call keyword arguments (line 316)
            kwargs_19899 = {}
            # Getting the type of 'self' (line 316)
            self_19895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 316)
            visit_19896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 12), self_19895, 'visit')
            # Calling visit(args, kwargs) (line 316)
            visit_call_result_19900 = invoke(stypy.reporting.localization.Localization(__file__, 316, 12), visit_19896, *[test_19898], **kwargs_19899)
            
            
            # Call to enter(...): (line 317)
            # Processing the call keyword arguments (line 317)
            kwargs_19903 = {}
            # Getting the type of 'self' (line 317)
            self_19901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 317)
            enter_19902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 12), self_19901, 'enter')
            # Calling enter(args, kwargs) (line 317)
            enter_call_result_19904 = invoke(stypy.reporting.localization.Localization(__file__, 317, 12), enter_19902, *[], **kwargs_19903)
            
            
            # Call to visit(...): (line 318)
            # Processing the call arguments (line 318)
            # Getting the type of 't' (line 318)
            t_19907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 23), 't', False)
            # Obtaining the member 'body' of a type (line 318)
            body_19908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 23), t_19907, 'body')
            # Processing the call keyword arguments (line 318)
            kwargs_19909 = {}
            # Getting the type of 'self' (line 318)
            self_19905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 318)
            visit_19906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 318, 12), self_19905, 'visit')
            # Calling visit(args, kwargs) (line 318)
            visit_call_result_19910 = invoke(stypy.reporting.localization.Localization(__file__, 318, 12), visit_19906, *[body_19908], **kwargs_19909)
            
            
            # Call to leave(...): (line 319)
            # Processing the call keyword arguments (line 319)
            kwargs_19913 = {}
            # Getting the type of 'self' (line 319)
            self_19911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 319)
            leave_19912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 12), self_19911, 'leave')
            # Calling leave(args, kwargs) (line 319)
            leave_call_result_19914 = invoke(stypy.reporting.localization.Localization(__file__, 319, 12), leave_19912, *[], **kwargs_19913)
            
            # SSA join for while statement (line 312)
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 't' (line 321)
        t_19915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 11), 't')
        # Obtaining the member 'orelse' of a type (line 321)
        orelse_19916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 321, 11), t_19915, 'orelse')
        # Testing if the type of an if condition is none (line 321)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 321, 8), orelse_19916):
            pass
        else:
            
            # Testing the type of an if condition (line 321)
            if_condition_19917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 321, 8), orelse_19916)
            # Assigning a type to the variable 'if_condition_19917' (line 321)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 8), 'if_condition_19917', if_condition_19917)
            # SSA begins for if statement (line 321)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fill(...): (line 322)
            # Processing the call arguments (line 322)
            str_19920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 322, 22), 'str', 'else')
            # Processing the call keyword arguments (line 322)
            kwargs_19921 = {}
            # Getting the type of 'self' (line 322)
            self_19918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 322, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 322)
            fill_19919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 322, 12), self_19918, 'fill')
            # Calling fill(args, kwargs) (line 322)
            fill_call_result_19922 = invoke(stypy.reporting.localization.Localization(__file__, 322, 12), fill_19919, *[str_19920], **kwargs_19921)
            
            
            # Call to enter(...): (line 323)
            # Processing the call keyword arguments (line 323)
            kwargs_19925 = {}
            # Getting the type of 'self' (line 323)
            self_19923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 323)
            enter_19924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 323, 12), self_19923, 'enter')
            # Calling enter(args, kwargs) (line 323)
            enter_call_result_19926 = invoke(stypy.reporting.localization.Localization(__file__, 323, 12), enter_19924, *[], **kwargs_19925)
            
            
            # Call to visit(...): (line 324)
            # Processing the call arguments (line 324)
            # Getting the type of 't' (line 324)
            t_19929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 23), 't', False)
            # Obtaining the member 'orelse' of a type (line 324)
            orelse_19930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 23), t_19929, 'orelse')
            # Processing the call keyword arguments (line 324)
            kwargs_19931 = {}
            # Getting the type of 'self' (line 324)
            self_19927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 324)
            visit_19928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 324, 12), self_19927, 'visit')
            # Calling visit(args, kwargs) (line 324)
            visit_call_result_19932 = invoke(stypy.reporting.localization.Localization(__file__, 324, 12), visit_19928, *[orelse_19930], **kwargs_19931)
            
            
            # Call to leave(...): (line 325)
            # Processing the call keyword arguments (line 325)
            kwargs_19935 = {}
            # Getting the type of 'self' (line 325)
            self_19933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 325)
            leave_19934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 12), self_19933, 'leave')
            # Calling leave(args, kwargs) (line 325)
            leave_call_result_19936 = invoke(stypy.reporting.localization.Localization(__file__, 325, 12), leave_19934, *[], **kwargs_19935)
            
            # SSA join for if statement (line 321)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 326)
        # Processing the call arguments (line 326)
        str_19939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 19), 'str', '\n')
        # Processing the call keyword arguments (line 326)
        kwargs_19940 = {}
        # Getting the type of 'self' (line 326)
        self_19937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 326)
        write_19938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 8), self_19937, 'write')
        # Calling write(args, kwargs) (line 326)
        write_call_result_19941 = invoke(stypy.reporting.localization.Localization(__file__, 326, 8), write_19938, *[str_19939], **kwargs_19940)
        
        
        # ################# End of 'visit_If(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_If' in the type store
        # Getting the type of 'stypy_return_type' (line 304)
        stypy_return_type_19942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19942)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_If'
        return stypy_return_type_19942


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
        str_19945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 18), 'str', 'while ')
        # Processing the call keyword arguments (line 329)
        kwargs_19946 = {}
        # Getting the type of 'self' (line 329)
        self_19943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 329)
        fill_19944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 329, 8), self_19943, 'fill')
        # Calling fill(args, kwargs) (line 329)
        fill_call_result_19947 = invoke(stypy.reporting.localization.Localization(__file__, 329, 8), fill_19944, *[str_19945], **kwargs_19946)
        
        
        # Call to visit(...): (line 330)
        # Processing the call arguments (line 330)
        # Getting the type of 't' (line 330)
        t_19950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 19), 't', False)
        # Obtaining the member 'test' of a type (line 330)
        test_19951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 19), t_19950, 'test')
        # Processing the call keyword arguments (line 330)
        kwargs_19952 = {}
        # Getting the type of 'self' (line 330)
        self_19948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 330, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 330)
        visit_19949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 330, 8), self_19948, 'visit')
        # Calling visit(args, kwargs) (line 330)
        visit_call_result_19953 = invoke(stypy.reporting.localization.Localization(__file__, 330, 8), visit_19949, *[test_19951], **kwargs_19952)
        
        
        # Call to enter(...): (line 331)
        # Processing the call keyword arguments (line 331)
        kwargs_19956 = {}
        # Getting the type of 'self' (line 331)
        self_19954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 331)
        enter_19955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 331, 8), self_19954, 'enter')
        # Calling enter(args, kwargs) (line 331)
        enter_call_result_19957 = invoke(stypy.reporting.localization.Localization(__file__, 331, 8), enter_19955, *[], **kwargs_19956)
        
        
        # Call to visit(...): (line 332)
        # Processing the call arguments (line 332)
        # Getting the type of 't' (line 332)
        t_19960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 19), 't', False)
        # Obtaining the member 'body' of a type (line 332)
        body_19961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 19), t_19960, 'body')
        # Processing the call keyword arguments (line 332)
        kwargs_19962 = {}
        # Getting the type of 'self' (line 332)
        self_19958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 332)
        visit_19959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 332, 8), self_19958, 'visit')
        # Calling visit(args, kwargs) (line 332)
        visit_call_result_19963 = invoke(stypy.reporting.localization.Localization(__file__, 332, 8), visit_19959, *[body_19961], **kwargs_19962)
        
        
        # Call to leave(...): (line 333)
        # Processing the call keyword arguments (line 333)
        kwargs_19966 = {}
        # Getting the type of 'self' (line 333)
        self_19964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 333)
        leave_19965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 8), self_19964, 'leave')
        # Calling leave(args, kwargs) (line 333)
        leave_call_result_19967 = invoke(stypy.reporting.localization.Localization(__file__, 333, 8), leave_19965, *[], **kwargs_19966)
        
        # Getting the type of 't' (line 334)
        t_19968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 11), 't')
        # Obtaining the member 'orelse' of a type (line 334)
        orelse_19969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 11), t_19968, 'orelse')
        # Testing if the type of an if condition is none (line 334)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 334, 8), orelse_19969):
            pass
        else:
            
            # Testing the type of an if condition (line 334)
            if_condition_19970 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 334, 8), orelse_19969)
            # Assigning a type to the variable 'if_condition_19970' (line 334)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'if_condition_19970', if_condition_19970)
            # SSA begins for if statement (line 334)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to fill(...): (line 335)
            # Processing the call arguments (line 335)
            str_19973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 22), 'str', 'else')
            # Processing the call keyword arguments (line 335)
            kwargs_19974 = {}
            # Getting the type of 'self' (line 335)
            self_19971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 12), 'self', False)
            # Obtaining the member 'fill' of a type (line 335)
            fill_19972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 12), self_19971, 'fill')
            # Calling fill(args, kwargs) (line 335)
            fill_call_result_19975 = invoke(stypy.reporting.localization.Localization(__file__, 335, 12), fill_19972, *[str_19973], **kwargs_19974)
            
            
            # Call to enter(...): (line 336)
            # Processing the call keyword arguments (line 336)
            kwargs_19978 = {}
            # Getting the type of 'self' (line 336)
            self_19976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 12), 'self', False)
            # Obtaining the member 'enter' of a type (line 336)
            enter_19977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 336, 12), self_19976, 'enter')
            # Calling enter(args, kwargs) (line 336)
            enter_call_result_19979 = invoke(stypy.reporting.localization.Localization(__file__, 336, 12), enter_19977, *[], **kwargs_19978)
            
            
            # Call to visit(...): (line 337)
            # Processing the call arguments (line 337)
            # Getting the type of 't' (line 337)
            t_19982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 23), 't', False)
            # Obtaining the member 'orelse' of a type (line 337)
            orelse_19983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 23), t_19982, 'orelse')
            # Processing the call keyword arguments (line 337)
            kwargs_19984 = {}
            # Getting the type of 'self' (line 337)
            self_19980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 337)
            visit_19981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 337, 12), self_19980, 'visit')
            # Calling visit(args, kwargs) (line 337)
            visit_call_result_19985 = invoke(stypy.reporting.localization.Localization(__file__, 337, 12), visit_19981, *[orelse_19983], **kwargs_19984)
            
            
            # Call to leave(...): (line 338)
            # Processing the call keyword arguments (line 338)
            kwargs_19988 = {}
            # Getting the type of 'self' (line 338)
            self_19986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 12), 'self', False)
            # Obtaining the member 'leave' of a type (line 338)
            leave_19987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 12), self_19986, 'leave')
            # Calling leave(args, kwargs) (line 338)
            leave_call_result_19989 = invoke(stypy.reporting.localization.Localization(__file__, 338, 12), leave_19987, *[], **kwargs_19988)
            
            # SSA join for if statement (line 334)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_While(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_While' in the type store
        # Getting the type of 'stypy_return_type' (line 328)
        stypy_return_type_19990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19990)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_While'
        return stypy_return_type_19990


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
        str_19993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 18), 'str', 'with ')
        # Processing the call keyword arguments (line 341)
        kwargs_19994 = {}
        # Getting the type of 'self' (line 341)
        self_19991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'self', False)
        # Obtaining the member 'fill' of a type (line 341)
        fill_19992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 341, 8), self_19991, 'fill')
        # Calling fill(args, kwargs) (line 341)
        fill_call_result_19995 = invoke(stypy.reporting.localization.Localization(__file__, 341, 8), fill_19992, *[str_19993], **kwargs_19994)
        
        
        # Call to visit(...): (line 342)
        # Processing the call arguments (line 342)
        # Getting the type of 't' (line 342)
        t_19998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 19), 't', False)
        # Obtaining the member 'context_expr' of a type (line 342)
        context_expr_19999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 19), t_19998, 'context_expr')
        # Processing the call keyword arguments (line 342)
        kwargs_20000 = {}
        # Getting the type of 'self' (line 342)
        self_19996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 342)
        visit_19997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 8), self_19996, 'visit')
        # Calling visit(args, kwargs) (line 342)
        visit_call_result_20001 = invoke(stypy.reporting.localization.Localization(__file__, 342, 8), visit_19997, *[context_expr_19999], **kwargs_20000)
        
        # Getting the type of 't' (line 343)
        t_20002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 11), 't')
        # Obtaining the member 'optional_vars' of a type (line 343)
        optional_vars_20003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 11), t_20002, 'optional_vars')
        # Testing if the type of an if condition is none (line 343)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 343, 8), optional_vars_20003):
            pass
        else:
            
            # Testing the type of an if condition (line 343)
            if_condition_20004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 8), optional_vars_20003)
            # Assigning a type to the variable 'if_condition_20004' (line 343)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'if_condition_20004', if_condition_20004)
            # SSA begins for if statement (line 343)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 344)
            # Processing the call arguments (line 344)
            str_20007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 23), 'str', ' as ')
            # Processing the call keyword arguments (line 344)
            kwargs_20008 = {}
            # Getting the type of 'self' (line 344)
            self_20005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 344)
            write_20006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 12), self_20005, 'write')
            # Calling write(args, kwargs) (line 344)
            write_call_result_20009 = invoke(stypy.reporting.localization.Localization(__file__, 344, 12), write_20006, *[str_20007], **kwargs_20008)
            
            
            # Call to visit(...): (line 345)
            # Processing the call arguments (line 345)
            # Getting the type of 't' (line 345)
            t_20012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 23), 't', False)
            # Obtaining the member 'optional_vars' of a type (line 345)
            optional_vars_20013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 23), t_20012, 'optional_vars')
            # Processing the call keyword arguments (line 345)
            kwargs_20014 = {}
            # Getting the type of 'self' (line 345)
            self_20010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 345)
            visit_20011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 12), self_20010, 'visit')
            # Calling visit(args, kwargs) (line 345)
            visit_call_result_20015 = invoke(stypy.reporting.localization.Localization(__file__, 345, 12), visit_20011, *[optional_vars_20013], **kwargs_20014)
            
            # SSA join for if statement (line 343)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to enter(...): (line 346)
        # Processing the call keyword arguments (line 346)
        kwargs_20018 = {}
        # Getting the type of 'self' (line 346)
        self_20016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'self', False)
        # Obtaining the member 'enter' of a type (line 346)
        enter_20017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 346, 8), self_20016, 'enter')
        # Calling enter(args, kwargs) (line 346)
        enter_call_result_20019 = invoke(stypy.reporting.localization.Localization(__file__, 346, 8), enter_20017, *[], **kwargs_20018)
        
        
        # Call to visit(...): (line 347)
        # Processing the call arguments (line 347)
        # Getting the type of 't' (line 347)
        t_20022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 19), 't', False)
        # Obtaining the member 'body' of a type (line 347)
        body_20023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 19), t_20022, 'body')
        # Processing the call keyword arguments (line 347)
        kwargs_20024 = {}
        # Getting the type of 'self' (line 347)
        self_20020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 347)
        visit_20021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 347, 8), self_20020, 'visit')
        # Calling visit(args, kwargs) (line 347)
        visit_call_result_20025 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), visit_20021, *[body_20023], **kwargs_20024)
        
        
        # Call to leave(...): (line 348)
        # Processing the call keyword arguments (line 348)
        kwargs_20028 = {}
        # Getting the type of 'self' (line 348)
        self_20026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 8), 'self', False)
        # Obtaining the member 'leave' of a type (line 348)
        leave_20027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 348, 8), self_20026, 'leave')
        # Calling leave(args, kwargs) (line 348)
        leave_call_result_20029 = invoke(stypy.reporting.localization.Localization(__file__, 348, 8), leave_20027, *[], **kwargs_20028)
        
        
        # ################# End of 'visit_With(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_With' in the type store
        # Getting the type of 'stypy_return_type' (line 340)
        stypy_return_type_20030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 340, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20030)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_With'
        return stypy_return_type_20030


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

        
        str_20031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 11), 'str', 'unicode_literals')
        # Getting the type of 'self' (line 355)
        self_20032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 37), 'self')
        # Obtaining the member 'future_imports' of a type (line 355)
        future_imports_20033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 37), self_20032, 'future_imports')
        # Applying the binary operator 'notin' (line 355)
        result_contains_20034 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 11), 'notin', str_20031, future_imports_20033)
        
        # Testing if the type of an if condition is none (line 355)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 355, 8), result_contains_20034):
            
            # Type idiom detected: calculating its left and rigth part (line 357)
            # Getting the type of 'str' (line 357)
            str_20045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 32), 'str')
            # Getting the type of 'tree' (line 357)
            tree_20046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 24), 'tree')
            # Obtaining the member 's' of a type (line 357)
            s_20047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 24), tree_20046, 's')
            
            (may_be_20048, more_types_in_union_20049) = may_be_subtype(str_20045, s_20047)

            if may_be_20048:

                if more_types_in_union_20049:
                    # Runtime conditional SSA (line 357)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'tree' (line 357)
                tree_20050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'tree')
                # Obtaining the member 's' of a type (line 357)
                s_20051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_20050, 's')
                # Setting the type of the member 's' of a type (line 357)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_20050, 's', remove_not_subtype_from_union(s_20047, str))
                
                # Call to write(...): (line 358)
                # Processing the call arguments (line 358)
                str_20054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 23), 'str', 'b')
                
                # Call to repr(...): (line 358)
                # Processing the call arguments (line 358)
                # Getting the type of 'tree' (line 358)
                tree_20056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 34), 'tree', False)
                # Obtaining the member 's' of a type (line 358)
                s_20057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 34), tree_20056, 's')
                # Processing the call keyword arguments (line 358)
                kwargs_20058 = {}
                # Getting the type of 'repr' (line 358)
                repr_20055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 29), 'repr', False)
                # Calling repr(args, kwargs) (line 358)
                repr_call_result_20059 = invoke(stypy.reporting.localization.Localization(__file__, 358, 29), repr_20055, *[s_20057], **kwargs_20058)
                
                # Applying the binary operator '+' (line 358)
                result_add_20060 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 23), '+', str_20054, repr_call_result_20059)
                
                # Processing the call keyword arguments (line 358)
                kwargs_20061 = {}
                # Getting the type of 'self' (line 358)
                self_20052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'self', False)
                # Obtaining the member 'write' of a type (line 358)
                write_20053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), self_20052, 'write')
                # Calling write(args, kwargs) (line 358)
                write_call_result_20062 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), write_20053, *[result_add_20060], **kwargs_20061)
                

                if more_types_in_union_20049:
                    # Runtime conditional SSA for else branch (line 357)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_20048) or more_types_in_union_20049):
                # Getting the type of 'tree' (line 357)
                tree_20063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'tree')
                # Obtaining the member 's' of a type (line 357)
                s_20064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_20063, 's')
                # Setting the type of the member 's' of a type (line 357)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_20063, 's', remove_subtype_from_union(s_20047, str))
                
                # Type idiom detected: calculating its left and rigth part (line 359)
                # Getting the type of 'unicode' (line 359)
                unicode_20065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 32), 'unicode')
                # Getting the type of 'tree' (line 359)
                tree_20066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 24), 'tree')
                # Obtaining the member 's' of a type (line 359)
                s_20067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 24), tree_20066, 's')
                
                (may_be_20068, more_types_in_union_20069) = may_be_subtype(unicode_20065, s_20067)

                if may_be_20068:

                    if more_types_in_union_20069:
                        # Runtime conditional SSA (line 359)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Getting the type of 'tree' (line 359)
                    tree_20070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 13), 'tree')
                    # Obtaining the member 's' of a type (line 359)
                    s_20071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_20070, 's')
                    # Setting the type of the member 's' of a type (line 359)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_20070, 's', remove_not_subtype_from_union(s_20067, unicode))
                    
                    # Call to write(...): (line 360)
                    # Processing the call arguments (line 360)
                    
                    # Call to lstrip(...): (line 360)
                    # Processing the call arguments (line 360)
                    str_20080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 43), 'str', 'u')
                    # Processing the call keyword arguments (line 360)
                    kwargs_20081 = {}
                    
                    # Call to repr(...): (line 360)
                    # Processing the call arguments (line 360)
                    # Getting the type of 'tree' (line 360)
                    tree_20075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 28), 'tree', False)
                    # Obtaining the member 's' of a type (line 360)
                    s_20076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 28), tree_20075, 's')
                    # Processing the call keyword arguments (line 360)
                    kwargs_20077 = {}
                    # Getting the type of 'repr' (line 360)
                    repr_20074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'repr', False)
                    # Calling repr(args, kwargs) (line 360)
                    repr_call_result_20078 = invoke(stypy.reporting.localization.Localization(__file__, 360, 23), repr_20074, *[s_20076], **kwargs_20077)
                    
                    # Obtaining the member 'lstrip' of a type (line 360)
                    lstrip_20079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 23), repr_call_result_20078, 'lstrip')
                    # Calling lstrip(args, kwargs) (line 360)
                    lstrip_call_result_20082 = invoke(stypy.reporting.localization.Localization(__file__, 360, 23), lstrip_20079, *[str_20080], **kwargs_20081)
                    
                    # Processing the call keyword arguments (line 360)
                    kwargs_20083 = {}
                    # Getting the type of 'self' (line 360)
                    self_20072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'self', False)
                    # Obtaining the member 'write' of a type (line 360)
                    write_20073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), self_20072, 'write')
                    # Calling write(args, kwargs) (line 360)
                    write_call_result_20084 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), write_20073, *[lstrip_call_result_20082], **kwargs_20083)
                    

                    if more_types_in_union_20069:
                        # Runtime conditional SSA for else branch (line 359)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_20068) or more_types_in_union_20069):
                    # Getting the type of 'tree' (line 359)
                    tree_20085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 13), 'tree')
                    # Obtaining the member 's' of a type (line 359)
                    s_20086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_20085, 's')
                    # Setting the type of the member 's' of a type (line 359)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_20085, 's', remove_subtype_from_union(s_20067, unicode))
                    # Evaluating assert statement condition
                    # Getting the type of 'False' (line 362)
                    False_20087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 19), 'False')
                    assert_20088 = False_20087
                    # Assigning a type to the variable 'assert_20088' (line 362)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'assert_20088', False_20087)

                    if (may_be_20068 and more_types_in_union_20069):
                        # SSA join for if statement (line 359)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_20048 and more_types_in_union_20049):
                    # SSA join for if statement (line 357)
                    module_type_store = module_type_store.join_ssa_context()


            
        else:
            
            # Testing the type of an if condition (line 355)
            if_condition_20035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 355, 8), result_contains_20034)
            # Assigning a type to the variable 'if_condition_20035' (line 355)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'if_condition_20035', if_condition_20035)
            # SSA begins for if statement (line 355)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 356)
            # Processing the call arguments (line 356)
            
            # Call to repr(...): (line 356)
            # Processing the call arguments (line 356)
            # Getting the type of 'tree' (line 356)
            tree_20039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 28), 'tree', False)
            # Obtaining the member 's' of a type (line 356)
            s_20040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 28), tree_20039, 's')
            # Processing the call keyword arguments (line 356)
            kwargs_20041 = {}
            # Getting the type of 'repr' (line 356)
            repr_20038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 23), 'repr', False)
            # Calling repr(args, kwargs) (line 356)
            repr_call_result_20042 = invoke(stypy.reporting.localization.Localization(__file__, 356, 23), repr_20038, *[s_20040], **kwargs_20041)
            
            # Processing the call keyword arguments (line 356)
            kwargs_20043 = {}
            # Getting the type of 'self' (line 356)
            self_20036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 356)
            write_20037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 12), self_20036, 'write')
            # Calling write(args, kwargs) (line 356)
            write_call_result_20044 = invoke(stypy.reporting.localization.Localization(__file__, 356, 12), write_20037, *[repr_call_result_20042], **kwargs_20043)
            
            # SSA branch for the else part of an if statement (line 355)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 357)
            # Getting the type of 'str' (line 357)
            str_20045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 32), 'str')
            # Getting the type of 'tree' (line 357)
            tree_20046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 24), 'tree')
            # Obtaining the member 's' of a type (line 357)
            s_20047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 24), tree_20046, 's')
            
            (may_be_20048, more_types_in_union_20049) = may_be_subtype(str_20045, s_20047)

            if may_be_20048:

                if more_types_in_union_20049:
                    # Runtime conditional SSA (line 357)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'tree' (line 357)
                tree_20050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'tree')
                # Obtaining the member 's' of a type (line 357)
                s_20051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_20050, 's')
                # Setting the type of the member 's' of a type (line 357)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_20050, 's', remove_not_subtype_from_union(s_20047, str))
                
                # Call to write(...): (line 358)
                # Processing the call arguments (line 358)
                str_20054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 23), 'str', 'b')
                
                # Call to repr(...): (line 358)
                # Processing the call arguments (line 358)
                # Getting the type of 'tree' (line 358)
                tree_20056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 34), 'tree', False)
                # Obtaining the member 's' of a type (line 358)
                s_20057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 34), tree_20056, 's')
                # Processing the call keyword arguments (line 358)
                kwargs_20058 = {}
                # Getting the type of 'repr' (line 358)
                repr_20055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 29), 'repr', False)
                # Calling repr(args, kwargs) (line 358)
                repr_call_result_20059 = invoke(stypy.reporting.localization.Localization(__file__, 358, 29), repr_20055, *[s_20057], **kwargs_20058)
                
                # Applying the binary operator '+' (line 358)
                result_add_20060 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 23), '+', str_20054, repr_call_result_20059)
                
                # Processing the call keyword arguments (line 358)
                kwargs_20061 = {}
                # Getting the type of 'self' (line 358)
                self_20052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 12), 'self', False)
                # Obtaining the member 'write' of a type (line 358)
                write_20053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 12), self_20052, 'write')
                # Calling write(args, kwargs) (line 358)
                write_call_result_20062 = invoke(stypy.reporting.localization.Localization(__file__, 358, 12), write_20053, *[result_add_20060], **kwargs_20061)
                

                if more_types_in_union_20049:
                    # Runtime conditional SSA for else branch (line 357)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_20048) or more_types_in_union_20049):
                # Getting the type of 'tree' (line 357)
                tree_20063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 13), 'tree')
                # Obtaining the member 's' of a type (line 357)
                s_20064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_20063, 's')
                # Setting the type of the member 's' of a type (line 357)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 357, 13), tree_20063, 's', remove_subtype_from_union(s_20047, str))
                
                # Type idiom detected: calculating its left and rigth part (line 359)
                # Getting the type of 'unicode' (line 359)
                unicode_20065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 32), 'unicode')
                # Getting the type of 'tree' (line 359)
                tree_20066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 24), 'tree')
                # Obtaining the member 's' of a type (line 359)
                s_20067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 24), tree_20066, 's')
                
                (may_be_20068, more_types_in_union_20069) = may_be_subtype(unicode_20065, s_20067)

                if may_be_20068:

                    if more_types_in_union_20069:
                        # Runtime conditional SSA (line 359)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Getting the type of 'tree' (line 359)
                    tree_20070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 13), 'tree')
                    # Obtaining the member 's' of a type (line 359)
                    s_20071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_20070, 's')
                    # Setting the type of the member 's' of a type (line 359)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_20070, 's', remove_not_subtype_from_union(s_20067, unicode))
                    
                    # Call to write(...): (line 360)
                    # Processing the call arguments (line 360)
                    
                    # Call to lstrip(...): (line 360)
                    # Processing the call arguments (line 360)
                    str_20080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 43), 'str', 'u')
                    # Processing the call keyword arguments (line 360)
                    kwargs_20081 = {}
                    
                    # Call to repr(...): (line 360)
                    # Processing the call arguments (line 360)
                    # Getting the type of 'tree' (line 360)
                    tree_20075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 28), 'tree', False)
                    # Obtaining the member 's' of a type (line 360)
                    s_20076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 28), tree_20075, 's')
                    # Processing the call keyword arguments (line 360)
                    kwargs_20077 = {}
                    # Getting the type of 'repr' (line 360)
                    repr_20074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 23), 'repr', False)
                    # Calling repr(args, kwargs) (line 360)
                    repr_call_result_20078 = invoke(stypy.reporting.localization.Localization(__file__, 360, 23), repr_20074, *[s_20076], **kwargs_20077)
                    
                    # Obtaining the member 'lstrip' of a type (line 360)
                    lstrip_20079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 23), repr_call_result_20078, 'lstrip')
                    # Calling lstrip(args, kwargs) (line 360)
                    lstrip_call_result_20082 = invoke(stypy.reporting.localization.Localization(__file__, 360, 23), lstrip_20079, *[str_20080], **kwargs_20081)
                    
                    # Processing the call keyword arguments (line 360)
                    kwargs_20083 = {}
                    # Getting the type of 'self' (line 360)
                    self_20072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 12), 'self', False)
                    # Obtaining the member 'write' of a type (line 360)
                    write_20073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 12), self_20072, 'write')
                    # Calling write(args, kwargs) (line 360)
                    write_call_result_20084 = invoke(stypy.reporting.localization.Localization(__file__, 360, 12), write_20073, *[lstrip_call_result_20082], **kwargs_20083)
                    

                    if more_types_in_union_20069:
                        # Runtime conditional SSA for else branch (line 359)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_20068) or more_types_in_union_20069):
                    # Getting the type of 'tree' (line 359)
                    tree_20085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 13), 'tree')
                    # Obtaining the member 's' of a type (line 359)
                    s_20086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_20085, 's')
                    # Setting the type of the member 's' of a type (line 359)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 13), tree_20085, 's', remove_subtype_from_union(s_20067, unicode))
                    # Evaluating assert statement condition
                    # Getting the type of 'False' (line 362)
                    False_20087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 362, 19), 'False')
                    assert_20088 = False_20087
                    # Assigning a type to the variable 'assert_20088' (line 362)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 362, 12), 'assert_20088', False_20087)

                    if (may_be_20068 and more_types_in_union_20069):
                        # SSA join for if statement (line 359)
                        module_type_store = module_type_store.join_ssa_context()


                

                if (may_be_20048 and more_types_in_union_20049):
                    # SSA join for if statement (line 357)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 355)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Str(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Str' in the type store
        # Getting the type of 'stypy_return_type' (line 351)
        stypy_return_type_20089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20089)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Str'
        return stypy_return_type_20089


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
        t_20092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 19), 't', False)
        # Obtaining the member 'id' of a type (line 365)
        id_20093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 19), t_20092, 'id')
        # Processing the call keyword arguments (line 365)
        kwargs_20094 = {}
        # Getting the type of 'self' (line 365)
        self_20090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 365, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 365)
        write_20091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 365, 8), self_20090, 'write')
        # Calling write(args, kwargs) (line 365)
        write_call_result_20095 = invoke(stypy.reporting.localization.Localization(__file__, 365, 8), write_20091, *[id_20093], **kwargs_20094)
        
        
        # ################# End of 'visit_Name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Name' in the type store
        # Getting the type of 'stypy_return_type' (line 364)
        stypy_return_type_20096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20096)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Name'
        return stypy_return_type_20096


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
        str_20099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 19), 'str', '`')
        # Processing the call keyword arguments (line 368)
        kwargs_20100 = {}
        # Getting the type of 'self' (line 368)
        self_20097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 368)
        write_20098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 368, 8), self_20097, 'write')
        # Calling write(args, kwargs) (line 368)
        write_call_result_20101 = invoke(stypy.reporting.localization.Localization(__file__, 368, 8), write_20098, *[str_20099], **kwargs_20100)
        
        
        # Call to visit(...): (line 369)
        # Processing the call arguments (line 369)
        # Getting the type of 't' (line 369)
        t_20104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 19), 't', False)
        # Obtaining the member 'value' of a type (line 369)
        value_20105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 19), t_20104, 'value')
        # Processing the call keyword arguments (line 369)
        kwargs_20106 = {}
        # Getting the type of 'self' (line 369)
        self_20102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 369, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 369)
        visit_20103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 369, 8), self_20102, 'visit')
        # Calling visit(args, kwargs) (line 369)
        visit_call_result_20107 = invoke(stypy.reporting.localization.Localization(__file__, 369, 8), visit_20103, *[value_20105], **kwargs_20106)
        
        
        # Call to write(...): (line 370)
        # Processing the call arguments (line 370)
        str_20110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 370, 19), 'str', '`')
        # Processing the call keyword arguments (line 370)
        kwargs_20111 = {}
        # Getting the type of 'self' (line 370)
        self_20108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 370)
        write_20109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 8), self_20108, 'write')
        # Calling write(args, kwargs) (line 370)
        write_call_result_20112 = invoke(stypy.reporting.localization.Localization(__file__, 370, 8), write_20109, *[str_20110], **kwargs_20111)
        
        
        # ################# End of 'visit_Repr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Repr' in the type store
        # Getting the type of 'stypy_return_type' (line 367)
        stypy_return_type_20113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20113)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Repr'
        return stypy_return_type_20113


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
        t_20115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 22), 't', False)
        # Obtaining the member 'n' of a type (line 373)
        n_20116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 22), t_20115, 'n')
        # Processing the call keyword arguments (line 373)
        kwargs_20117 = {}
        # Getting the type of 'repr' (line 373)
        repr_20114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 17), 'repr', False)
        # Calling repr(args, kwargs) (line 373)
        repr_call_result_20118 = invoke(stypy.reporting.localization.Localization(__file__, 373, 17), repr_20114, *[n_20116], **kwargs_20117)
        
        # Assigning a type to the variable 'repr_n' (line 373)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 8), 'repr_n', repr_call_result_20118)
        
        # Call to startswith(...): (line 375)
        # Processing the call arguments (line 375)
        str_20121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 375, 29), 'str', '-')
        # Processing the call keyword arguments (line 375)
        kwargs_20122 = {}
        # Getting the type of 'repr_n' (line 375)
        repr_n_20119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 11), 'repr_n', False)
        # Obtaining the member 'startswith' of a type (line 375)
        startswith_20120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 375, 11), repr_n_20119, 'startswith')
        # Calling startswith(args, kwargs) (line 375)
        startswith_call_result_20123 = invoke(stypy.reporting.localization.Localization(__file__, 375, 11), startswith_20120, *[str_20121], **kwargs_20122)
        
        # Testing if the type of an if condition is none (line 375)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 375, 8), startswith_call_result_20123):
            pass
        else:
            
            # Testing the type of an if condition (line 375)
            if_condition_20124 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 375, 8), startswith_call_result_20123)
            # Assigning a type to the variable 'if_condition_20124' (line 375)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 8), 'if_condition_20124', if_condition_20124)
            # SSA begins for if statement (line 375)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 376)
            # Processing the call arguments (line 376)
            str_20127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 376, 23), 'str', '(')
            # Processing the call keyword arguments (line 376)
            kwargs_20128 = {}
            # Getting the type of 'self' (line 376)
            self_20125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 376, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 376)
            write_20126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 376, 12), self_20125, 'write')
            # Calling write(args, kwargs) (line 376)
            write_call_result_20129 = invoke(stypy.reporting.localization.Localization(__file__, 376, 12), write_20126, *[str_20127], **kwargs_20128)
            
            # SSA join for if statement (line 375)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 378)
        # Processing the call arguments (line 378)
        
        # Call to replace(...): (line 378)
        # Processing the call arguments (line 378)
        str_20134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 34), 'str', 'inf')
        # Getting the type of 'INFSTR' (line 378)
        INFSTR_20135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 41), 'INFSTR', False)
        # Processing the call keyword arguments (line 378)
        kwargs_20136 = {}
        # Getting the type of 'repr_n' (line 378)
        repr_n_20132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 19), 'repr_n', False)
        # Obtaining the member 'replace' of a type (line 378)
        replace_20133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 19), repr_n_20132, 'replace')
        # Calling replace(args, kwargs) (line 378)
        replace_call_result_20137 = invoke(stypy.reporting.localization.Localization(__file__, 378, 19), replace_20133, *[str_20134, INFSTR_20135], **kwargs_20136)
        
        # Processing the call keyword arguments (line 378)
        kwargs_20138 = {}
        # Getting the type of 'self' (line 378)
        self_20130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 378)
        write_20131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 8), self_20130, 'write')
        # Calling write(args, kwargs) (line 378)
        write_call_result_20139 = invoke(stypy.reporting.localization.Localization(__file__, 378, 8), write_20131, *[replace_call_result_20137], **kwargs_20138)
        
        
        # Call to startswith(...): (line 379)
        # Processing the call arguments (line 379)
        str_20142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 379, 29), 'str', '-')
        # Processing the call keyword arguments (line 379)
        kwargs_20143 = {}
        # Getting the type of 'repr_n' (line 379)
        repr_n_20140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 11), 'repr_n', False)
        # Obtaining the member 'startswith' of a type (line 379)
        startswith_20141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 379, 11), repr_n_20140, 'startswith')
        # Calling startswith(args, kwargs) (line 379)
        startswith_call_result_20144 = invoke(stypy.reporting.localization.Localization(__file__, 379, 11), startswith_20141, *[str_20142], **kwargs_20143)
        
        # Testing if the type of an if condition is none (line 379)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 379, 8), startswith_call_result_20144):
            pass
        else:
            
            # Testing the type of an if condition (line 379)
            if_condition_20145 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 379, 8), startswith_call_result_20144)
            # Assigning a type to the variable 'if_condition_20145' (line 379)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 379, 8), 'if_condition_20145', if_condition_20145)
            # SSA begins for if statement (line 379)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 380)
            # Processing the call arguments (line 380)
            str_20148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 380, 23), 'str', ')')
            # Processing the call keyword arguments (line 380)
            kwargs_20149 = {}
            # Getting the type of 'self' (line 380)
            self_20146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 380, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 380)
            write_20147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 380, 12), self_20146, 'write')
            # Calling write(args, kwargs) (line 380)
            write_call_result_20150 = invoke(stypy.reporting.localization.Localization(__file__, 380, 12), write_20147, *[str_20148], **kwargs_20149)
            
            # SSA join for if statement (line 379)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Num(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Num' in the type store
        # Getting the type of 'stypy_return_type' (line 372)
        stypy_return_type_20151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20151)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Num'
        return stypy_return_type_20151


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
        str_20154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 383, 19), 'str', '[')
        # Processing the call keyword arguments (line 383)
        kwargs_20155 = {}
        # Getting the type of 'self' (line 383)
        self_20152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 383, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 383)
        write_20153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 383, 8), self_20152, 'write')
        # Calling write(args, kwargs) (line 383)
        write_call_result_20156 = invoke(stypy.reporting.localization.Localization(__file__, 383, 8), write_20153, *[str_20154], **kwargs_20155)
        
        
        # Call to interleave(...): (line 384)
        # Processing the call arguments (line 384)

        @norecursion
        def _stypy_temp_lambda_33(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_33'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_33', 384, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_33.stypy_localization = localization
            _stypy_temp_lambda_33.stypy_type_of_self = None
            _stypy_temp_lambda_33.stypy_type_store = module_type_store
            _stypy_temp_lambda_33.stypy_function_name = '_stypy_temp_lambda_33'
            _stypy_temp_lambda_33.stypy_param_names_list = []
            _stypy_temp_lambda_33.stypy_varargs_param_name = None
            _stypy_temp_lambda_33.stypy_kwargs_param_name = None
            _stypy_temp_lambda_33.stypy_call_defaults = defaults
            _stypy_temp_lambda_33.stypy_call_varargs = varargs
            _stypy_temp_lambda_33.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_33', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_33', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 384)
            # Processing the call arguments (line 384)
            str_20160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, 38), 'str', ', ')
            # Processing the call keyword arguments (line 384)
            kwargs_20161 = {}
            # Getting the type of 'self' (line 384)
            self_20158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 384)
            write_20159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 27), self_20158, 'write')
            # Calling write(args, kwargs) (line 384)
            write_call_result_20162 = invoke(stypy.reporting.localization.Localization(__file__, 384, 27), write_20159, *[str_20160], **kwargs_20161)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 384)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), 'stypy_return_type', write_call_result_20162)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_33' in the type store
            # Getting the type of 'stypy_return_type' (line 384)
            stypy_return_type_20163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_20163)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_33'
            return stypy_return_type_20163

        # Assigning a type to the variable '_stypy_temp_lambda_33' (line 384)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), '_stypy_temp_lambda_33', _stypy_temp_lambda_33)
        # Getting the type of '_stypy_temp_lambda_33' (line 384)
        _stypy_temp_lambda_33_20164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 19), '_stypy_temp_lambda_33')
        # Getting the type of 'self' (line 384)
        self_20165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 384)
        visit_20166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 45), self_20165, 'visit')
        # Getting the type of 't' (line 384)
        t_20167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 57), 't', False)
        # Obtaining the member 'elts' of a type (line 384)
        elts_20168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 384, 57), t_20167, 'elts')
        # Processing the call keyword arguments (line 384)
        kwargs_20169 = {}
        # Getting the type of 'interleave' (line 384)
        interleave_20157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 384)
        interleave_call_result_20170 = invoke(stypy.reporting.localization.Localization(__file__, 384, 8), interleave_20157, *[_stypy_temp_lambda_33_20164, visit_20166, elts_20168], **kwargs_20169)
        
        
        # Call to write(...): (line 385)
        # Processing the call arguments (line 385)
        str_20173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 19), 'str', ']')
        # Processing the call keyword arguments (line 385)
        kwargs_20174 = {}
        # Getting the type of 'self' (line 385)
        self_20171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 385)
        write_20172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 8), self_20171, 'write')
        # Calling write(args, kwargs) (line 385)
        write_call_result_20175 = invoke(stypy.reporting.localization.Localization(__file__, 385, 8), write_20172, *[str_20173], **kwargs_20174)
        
        
        # ################# End of 'visit_List(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_List' in the type store
        # Getting the type of 'stypy_return_type' (line 382)
        stypy_return_type_20176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 382, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20176)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_List'
        return stypy_return_type_20176


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
        str_20179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 388, 19), 'str', '[')
        # Processing the call keyword arguments (line 388)
        kwargs_20180 = {}
        # Getting the type of 'self' (line 388)
        self_20177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 388)
        write_20178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 8), self_20177, 'write')
        # Calling write(args, kwargs) (line 388)
        write_call_result_20181 = invoke(stypy.reporting.localization.Localization(__file__, 388, 8), write_20178, *[str_20179], **kwargs_20180)
        
        
        # Call to visit(...): (line 389)
        # Processing the call arguments (line 389)
        # Getting the type of 't' (line 389)
        t_20184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 19), 't', False)
        # Obtaining the member 'elt' of a type (line 389)
        elt_20185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 19), t_20184, 'elt')
        # Processing the call keyword arguments (line 389)
        kwargs_20186 = {}
        # Getting the type of 'self' (line 389)
        self_20182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 389, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 389)
        visit_20183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 389, 8), self_20182, 'visit')
        # Calling visit(args, kwargs) (line 389)
        visit_call_result_20187 = invoke(stypy.reporting.localization.Localization(__file__, 389, 8), visit_20183, *[elt_20185], **kwargs_20186)
        
        
        # Getting the type of 't' (line 390)
        t_20188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 390, 19), 't')
        # Obtaining the member 'generators' of a type (line 390)
        generators_20189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 390, 19), t_20188, 'generators')
        # Assigning a type to the variable 'generators_20189' (line 390)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'generators_20189', generators_20189)
        # Testing if the for loop is going to be iterated (line 390)
        # Testing the type of a for loop iterable (line 390)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 390, 8), generators_20189)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 390, 8), generators_20189):
            # Getting the type of the for loop variable (line 390)
            for_loop_var_20190 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 390, 8), generators_20189)
            # Assigning a type to the variable 'gen' (line 390)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 390, 8), 'gen', for_loop_var_20190)
            # SSA begins for a for statement (line 390)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 391)
            # Processing the call arguments (line 391)
            # Getting the type of 'gen' (line 391)
            gen_20193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 23), 'gen', False)
            # Processing the call keyword arguments (line 391)
            kwargs_20194 = {}
            # Getting the type of 'self' (line 391)
            self_20191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 391)
            visit_20192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 12), self_20191, 'visit')
            # Calling visit(args, kwargs) (line 391)
            visit_call_result_20195 = invoke(stypy.reporting.localization.Localization(__file__, 391, 12), visit_20192, *[gen_20193], **kwargs_20194)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 392)
        # Processing the call arguments (line 392)
        str_20198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 392, 19), 'str', ']')
        # Processing the call keyword arguments (line 392)
        kwargs_20199 = {}
        # Getting the type of 'self' (line 392)
        self_20196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 392)
        write_20197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 8), self_20196, 'write')
        # Calling write(args, kwargs) (line 392)
        write_call_result_20200 = invoke(stypy.reporting.localization.Localization(__file__, 392, 8), write_20197, *[str_20198], **kwargs_20199)
        
        
        # ################# End of 'visit_ListComp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ListComp' in the type store
        # Getting the type of 'stypy_return_type' (line 387)
        stypy_return_type_20201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20201)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ListComp'
        return stypy_return_type_20201


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
        str_20204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 19), 'str', '(')
        # Processing the call keyword arguments (line 395)
        kwargs_20205 = {}
        # Getting the type of 'self' (line 395)
        self_20202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 395)
        write_20203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 8), self_20202, 'write')
        # Calling write(args, kwargs) (line 395)
        write_call_result_20206 = invoke(stypy.reporting.localization.Localization(__file__, 395, 8), write_20203, *[str_20204], **kwargs_20205)
        
        
        # Call to visit(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 't' (line 396)
        t_20209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 19), 't', False)
        # Obtaining the member 'elt' of a type (line 396)
        elt_20210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 19), t_20209, 'elt')
        # Processing the call keyword arguments (line 396)
        kwargs_20211 = {}
        # Getting the type of 'self' (line 396)
        self_20207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 396)
        visit_20208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 8), self_20207, 'visit')
        # Calling visit(args, kwargs) (line 396)
        visit_call_result_20212 = invoke(stypy.reporting.localization.Localization(__file__, 396, 8), visit_20208, *[elt_20210], **kwargs_20211)
        
        
        # Getting the type of 't' (line 397)
        t_20213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 19), 't')
        # Obtaining the member 'generators' of a type (line 397)
        generators_20214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 19), t_20213, 'generators')
        # Assigning a type to the variable 'generators_20214' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'generators_20214', generators_20214)
        # Testing if the for loop is going to be iterated (line 397)
        # Testing the type of a for loop iterable (line 397)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 397, 8), generators_20214)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 397, 8), generators_20214):
            # Getting the type of the for loop variable (line 397)
            for_loop_var_20215 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 397, 8), generators_20214)
            # Assigning a type to the variable 'gen' (line 397)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'gen', for_loop_var_20215)
            # SSA begins for a for statement (line 397)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 398)
            # Processing the call arguments (line 398)
            # Getting the type of 'gen' (line 398)
            gen_20218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 23), 'gen', False)
            # Processing the call keyword arguments (line 398)
            kwargs_20219 = {}
            # Getting the type of 'self' (line 398)
            self_20216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 398)
            visit_20217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 12), self_20216, 'visit')
            # Calling visit(args, kwargs) (line 398)
            visit_call_result_20220 = invoke(stypy.reporting.localization.Localization(__file__, 398, 12), visit_20217, *[gen_20218], **kwargs_20219)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 399)
        # Processing the call arguments (line 399)
        str_20223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 19), 'str', ')')
        # Processing the call keyword arguments (line 399)
        kwargs_20224 = {}
        # Getting the type of 'self' (line 399)
        self_20221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 399)
        write_20222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 399, 8), self_20221, 'write')
        # Calling write(args, kwargs) (line 399)
        write_call_result_20225 = invoke(stypy.reporting.localization.Localization(__file__, 399, 8), write_20222, *[str_20223], **kwargs_20224)
        
        
        # ################# End of 'visit_GeneratorExp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_GeneratorExp' in the type store
        # Getting the type of 'stypy_return_type' (line 394)
        stypy_return_type_20226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20226)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_GeneratorExp'
        return stypy_return_type_20226


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
        str_20229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 402, 19), 'str', '{')
        # Processing the call keyword arguments (line 402)
        kwargs_20230 = {}
        # Getting the type of 'self' (line 402)
        self_20227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 402)
        write_20228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 402, 8), self_20227, 'write')
        # Calling write(args, kwargs) (line 402)
        write_call_result_20231 = invoke(stypy.reporting.localization.Localization(__file__, 402, 8), write_20228, *[str_20229], **kwargs_20230)
        
        
        # Call to visit(...): (line 403)
        # Processing the call arguments (line 403)
        # Getting the type of 't' (line 403)
        t_20234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 19), 't', False)
        # Obtaining the member 'elt' of a type (line 403)
        elt_20235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 19), t_20234, 'elt')
        # Processing the call keyword arguments (line 403)
        kwargs_20236 = {}
        # Getting the type of 'self' (line 403)
        self_20232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 403, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 403)
        visit_20233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 403, 8), self_20232, 'visit')
        # Calling visit(args, kwargs) (line 403)
        visit_call_result_20237 = invoke(stypy.reporting.localization.Localization(__file__, 403, 8), visit_20233, *[elt_20235], **kwargs_20236)
        
        
        # Getting the type of 't' (line 404)
        t_20238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 19), 't')
        # Obtaining the member 'generators' of a type (line 404)
        generators_20239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 404, 19), t_20238, 'generators')
        # Assigning a type to the variable 'generators_20239' (line 404)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'generators_20239', generators_20239)
        # Testing if the for loop is going to be iterated (line 404)
        # Testing the type of a for loop iterable (line 404)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 404, 8), generators_20239)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 404, 8), generators_20239):
            # Getting the type of the for loop variable (line 404)
            for_loop_var_20240 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 404, 8), generators_20239)
            # Assigning a type to the variable 'gen' (line 404)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 8), 'gen', for_loop_var_20240)
            # SSA begins for a for statement (line 404)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 405)
            # Processing the call arguments (line 405)
            # Getting the type of 'gen' (line 405)
            gen_20243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 23), 'gen', False)
            # Processing the call keyword arguments (line 405)
            kwargs_20244 = {}
            # Getting the type of 'self' (line 405)
            self_20241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 405, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 405)
            visit_20242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 405, 12), self_20241, 'visit')
            # Calling visit(args, kwargs) (line 405)
            visit_call_result_20245 = invoke(stypy.reporting.localization.Localization(__file__, 405, 12), visit_20242, *[gen_20243], **kwargs_20244)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 406)
        # Processing the call arguments (line 406)
        str_20248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 19), 'str', '}')
        # Processing the call keyword arguments (line 406)
        kwargs_20249 = {}
        # Getting the type of 'self' (line 406)
        self_20246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 406)
        write_20247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 406, 8), self_20246, 'write')
        # Calling write(args, kwargs) (line 406)
        write_call_result_20250 = invoke(stypy.reporting.localization.Localization(__file__, 406, 8), write_20247, *[str_20248], **kwargs_20249)
        
        
        # ################# End of 'visit_SetComp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_SetComp' in the type store
        # Getting the type of 'stypy_return_type' (line 401)
        stypy_return_type_20251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20251)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_SetComp'
        return stypy_return_type_20251


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
        str_20254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 409, 19), 'str', '{')
        # Processing the call keyword arguments (line 409)
        kwargs_20255 = {}
        # Getting the type of 'self' (line 409)
        self_20252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 409)
        write_20253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 409, 8), self_20252, 'write')
        # Calling write(args, kwargs) (line 409)
        write_call_result_20256 = invoke(stypy.reporting.localization.Localization(__file__, 409, 8), write_20253, *[str_20254], **kwargs_20255)
        
        
        # Call to visit(...): (line 410)
        # Processing the call arguments (line 410)
        # Getting the type of 't' (line 410)
        t_20259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 19), 't', False)
        # Obtaining the member 'key' of a type (line 410)
        key_20260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 19), t_20259, 'key')
        # Processing the call keyword arguments (line 410)
        kwargs_20261 = {}
        # Getting the type of 'self' (line 410)
        self_20257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 410)
        visit_20258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 8), self_20257, 'visit')
        # Calling visit(args, kwargs) (line 410)
        visit_call_result_20262 = invoke(stypy.reporting.localization.Localization(__file__, 410, 8), visit_20258, *[key_20260], **kwargs_20261)
        
        
        # Call to write(...): (line 411)
        # Processing the call arguments (line 411)
        str_20265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 411, 19), 'str', ': ')
        # Processing the call keyword arguments (line 411)
        kwargs_20266 = {}
        # Getting the type of 'self' (line 411)
        self_20263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 411)
        write_20264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 411, 8), self_20263, 'write')
        # Calling write(args, kwargs) (line 411)
        write_call_result_20267 = invoke(stypy.reporting.localization.Localization(__file__, 411, 8), write_20264, *[str_20265], **kwargs_20266)
        
        
        # Call to visit(...): (line 412)
        # Processing the call arguments (line 412)
        # Getting the type of 't' (line 412)
        t_20270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 19), 't', False)
        # Obtaining the member 'value' of a type (line 412)
        value_20271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 19), t_20270, 'value')
        # Processing the call keyword arguments (line 412)
        kwargs_20272 = {}
        # Getting the type of 'self' (line 412)
        self_20268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 412)
        visit_20269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 412, 8), self_20268, 'visit')
        # Calling visit(args, kwargs) (line 412)
        visit_call_result_20273 = invoke(stypy.reporting.localization.Localization(__file__, 412, 8), visit_20269, *[value_20271], **kwargs_20272)
        
        
        # Getting the type of 't' (line 413)
        t_20274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 413, 19), 't')
        # Obtaining the member 'generators' of a type (line 413)
        generators_20275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 413, 19), t_20274, 'generators')
        # Assigning a type to the variable 'generators_20275' (line 413)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'generators_20275', generators_20275)
        # Testing if the for loop is going to be iterated (line 413)
        # Testing the type of a for loop iterable (line 413)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 413, 8), generators_20275)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 413, 8), generators_20275):
            # Getting the type of the for loop variable (line 413)
            for_loop_var_20276 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 413, 8), generators_20275)
            # Assigning a type to the variable 'gen' (line 413)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 413, 8), 'gen', for_loop_var_20276)
            # SSA begins for a for statement (line 413)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to visit(...): (line 414)
            # Processing the call arguments (line 414)
            # Getting the type of 'gen' (line 414)
            gen_20279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 23), 'gen', False)
            # Processing the call keyword arguments (line 414)
            kwargs_20280 = {}
            # Getting the type of 'self' (line 414)
            self_20277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 414)
            visit_20278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 414, 12), self_20277, 'visit')
            # Calling visit(args, kwargs) (line 414)
            visit_call_result_20281 = invoke(stypy.reporting.localization.Localization(__file__, 414, 12), visit_20278, *[gen_20279], **kwargs_20280)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 415)
        # Processing the call arguments (line 415)
        str_20284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 415, 19), 'str', '}')
        # Processing the call keyword arguments (line 415)
        kwargs_20285 = {}
        # Getting the type of 'self' (line 415)
        self_20282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 415, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 415)
        write_20283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 415, 8), self_20282, 'write')
        # Calling write(args, kwargs) (line 415)
        write_call_result_20286 = invoke(stypy.reporting.localization.Localization(__file__, 415, 8), write_20283, *[str_20284], **kwargs_20285)
        
        
        # ################# End of 'visit_DictComp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_DictComp' in the type store
        # Getting the type of 'stypy_return_type' (line 408)
        stypy_return_type_20287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 408, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20287)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_DictComp'
        return stypy_return_type_20287


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
        str_20290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 19), 'str', ' for ')
        # Processing the call keyword arguments (line 418)
        kwargs_20291 = {}
        # Getting the type of 'self' (line 418)
        self_20288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 418)
        write_20289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 8), self_20288, 'write')
        # Calling write(args, kwargs) (line 418)
        write_call_result_20292 = invoke(stypy.reporting.localization.Localization(__file__, 418, 8), write_20289, *[str_20290], **kwargs_20291)
        
        
        # Call to visit(...): (line 419)
        # Processing the call arguments (line 419)
        # Getting the type of 't' (line 419)
        t_20295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 19), 't', False)
        # Obtaining the member 'target' of a type (line 419)
        target_20296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 19), t_20295, 'target')
        # Processing the call keyword arguments (line 419)
        kwargs_20297 = {}
        # Getting the type of 'self' (line 419)
        self_20293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 419)
        visit_20294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 8), self_20293, 'visit')
        # Calling visit(args, kwargs) (line 419)
        visit_call_result_20298 = invoke(stypy.reporting.localization.Localization(__file__, 419, 8), visit_20294, *[target_20296], **kwargs_20297)
        
        
        # Call to write(...): (line 420)
        # Processing the call arguments (line 420)
        str_20301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 420, 19), 'str', ' in ')
        # Processing the call keyword arguments (line 420)
        kwargs_20302 = {}
        # Getting the type of 'self' (line 420)
        self_20299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 420)
        write_20300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 8), self_20299, 'write')
        # Calling write(args, kwargs) (line 420)
        write_call_result_20303 = invoke(stypy.reporting.localization.Localization(__file__, 420, 8), write_20300, *[str_20301], **kwargs_20302)
        
        
        # Call to visit(...): (line 421)
        # Processing the call arguments (line 421)
        # Getting the type of 't' (line 421)
        t_20306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 19), 't', False)
        # Obtaining the member 'iter' of a type (line 421)
        iter_20307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 19), t_20306, 'iter')
        # Processing the call keyword arguments (line 421)
        kwargs_20308 = {}
        # Getting the type of 'self' (line 421)
        self_20304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 421)
        visit_20305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 421, 8), self_20304, 'visit')
        # Calling visit(args, kwargs) (line 421)
        visit_call_result_20309 = invoke(stypy.reporting.localization.Localization(__file__, 421, 8), visit_20305, *[iter_20307], **kwargs_20308)
        
        
        # Getting the type of 't' (line 422)
        t_20310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 25), 't')
        # Obtaining the member 'ifs' of a type (line 422)
        ifs_20311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 25), t_20310, 'ifs')
        # Assigning a type to the variable 'ifs_20311' (line 422)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'ifs_20311', ifs_20311)
        # Testing if the for loop is going to be iterated (line 422)
        # Testing the type of a for loop iterable (line 422)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 422, 8), ifs_20311)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 422, 8), ifs_20311):
            # Getting the type of the for loop variable (line 422)
            for_loop_var_20312 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 422, 8), ifs_20311)
            # Assigning a type to the variable 'if_clause' (line 422)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 422, 8), 'if_clause', for_loop_var_20312)
            # SSA begins for a for statement (line 422)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to write(...): (line 423)
            # Processing the call arguments (line 423)
            str_20315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 423, 23), 'str', ' if ')
            # Processing the call keyword arguments (line 423)
            kwargs_20316 = {}
            # Getting the type of 'self' (line 423)
            self_20313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 423)
            write_20314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 423, 12), self_20313, 'write')
            # Calling write(args, kwargs) (line 423)
            write_call_result_20317 = invoke(stypy.reporting.localization.Localization(__file__, 423, 12), write_20314, *[str_20315], **kwargs_20316)
            
            
            # Call to visit(...): (line 424)
            # Processing the call arguments (line 424)
            # Getting the type of 'if_clause' (line 424)
            if_clause_20320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 23), 'if_clause', False)
            # Processing the call keyword arguments (line 424)
            kwargs_20321 = {}
            # Getting the type of 'self' (line 424)
            self_20318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 424, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 424)
            visit_20319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 424, 12), self_20318, 'visit')
            # Calling visit(args, kwargs) (line 424)
            visit_call_result_20322 = invoke(stypy.reporting.localization.Localization(__file__, 424, 12), visit_20319, *[if_clause_20320], **kwargs_20321)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'visit_comprehension(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_comprehension' in the type store
        # Getting the type of 'stypy_return_type' (line 417)
        stypy_return_type_20323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 417, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20323)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_comprehension'
        return stypy_return_type_20323


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
        str_20326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 19), 'str', '(')
        # Processing the call keyword arguments (line 427)
        kwargs_20327 = {}
        # Getting the type of 'self' (line 427)
        self_20324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 427)
        write_20325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 8), self_20324, 'write')
        # Calling write(args, kwargs) (line 427)
        write_call_result_20328 = invoke(stypy.reporting.localization.Localization(__file__, 427, 8), write_20325, *[str_20326], **kwargs_20327)
        
        
        # Call to visit(...): (line 428)
        # Processing the call arguments (line 428)
        # Getting the type of 't' (line 428)
        t_20331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 19), 't', False)
        # Obtaining the member 'body' of a type (line 428)
        body_20332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 19), t_20331, 'body')
        # Processing the call keyword arguments (line 428)
        kwargs_20333 = {}
        # Getting the type of 'self' (line 428)
        self_20329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 428)
        visit_20330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 428, 8), self_20329, 'visit')
        # Calling visit(args, kwargs) (line 428)
        visit_call_result_20334 = invoke(stypy.reporting.localization.Localization(__file__, 428, 8), visit_20330, *[body_20332], **kwargs_20333)
        
        
        # Call to write(...): (line 429)
        # Processing the call arguments (line 429)
        str_20337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 429, 19), 'str', ' if ')
        # Processing the call keyword arguments (line 429)
        kwargs_20338 = {}
        # Getting the type of 'self' (line 429)
        self_20335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 429)
        write_20336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 8), self_20335, 'write')
        # Calling write(args, kwargs) (line 429)
        write_call_result_20339 = invoke(stypy.reporting.localization.Localization(__file__, 429, 8), write_20336, *[str_20337], **kwargs_20338)
        
        
        # Call to visit(...): (line 430)
        # Processing the call arguments (line 430)
        # Getting the type of 't' (line 430)
        t_20342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 19), 't', False)
        # Obtaining the member 'test' of a type (line 430)
        test_20343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 19), t_20342, 'test')
        # Processing the call keyword arguments (line 430)
        kwargs_20344 = {}
        # Getting the type of 'self' (line 430)
        self_20340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 430)
        visit_20341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 8), self_20340, 'visit')
        # Calling visit(args, kwargs) (line 430)
        visit_call_result_20345 = invoke(stypy.reporting.localization.Localization(__file__, 430, 8), visit_20341, *[test_20343], **kwargs_20344)
        
        
        # Call to write(...): (line 431)
        # Processing the call arguments (line 431)
        str_20348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 19), 'str', ' else ')
        # Processing the call keyword arguments (line 431)
        kwargs_20349 = {}
        # Getting the type of 'self' (line 431)
        self_20346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 431)
        write_20347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 8), self_20346, 'write')
        # Calling write(args, kwargs) (line 431)
        write_call_result_20350 = invoke(stypy.reporting.localization.Localization(__file__, 431, 8), write_20347, *[str_20348], **kwargs_20349)
        
        
        # Call to visit(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 't' (line 432)
        t_20353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 19), 't', False)
        # Obtaining the member 'orelse' of a type (line 432)
        orelse_20354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 19), t_20353, 'orelse')
        # Processing the call keyword arguments (line 432)
        kwargs_20355 = {}
        # Getting the type of 'self' (line 432)
        self_20351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 432)
        visit_20352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 8), self_20351, 'visit')
        # Calling visit(args, kwargs) (line 432)
        visit_call_result_20356 = invoke(stypy.reporting.localization.Localization(__file__, 432, 8), visit_20352, *[orelse_20354], **kwargs_20355)
        
        
        # Call to write(...): (line 433)
        # Processing the call arguments (line 433)
        str_20359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 19), 'str', ')')
        # Processing the call keyword arguments (line 433)
        kwargs_20360 = {}
        # Getting the type of 'self' (line 433)
        self_20357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 433)
        write_20358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 8), self_20357, 'write')
        # Calling write(args, kwargs) (line 433)
        write_call_result_20361 = invoke(stypy.reporting.localization.Localization(__file__, 433, 8), write_20358, *[str_20359], **kwargs_20360)
        
        
        # ################# End of 'visit_IfExp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_IfExp' in the type store
        # Getting the type of 'stypy_return_type' (line 426)
        stypy_return_type_20362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20362)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_IfExp'
        return stypy_return_type_20362


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
        t_20363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 16), 't')
        # Obtaining the member 'elts' of a type (line 436)
        elts_20364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 436, 16), t_20363, 'elts')
        assert_20365 = elts_20364
        # Assigning a type to the variable 'assert_20365' (line 436)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 8), 'assert_20365', elts_20364)
        
        # Call to write(...): (line 437)
        # Processing the call arguments (line 437)
        str_20368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 437, 19), 'str', '{')
        # Processing the call keyword arguments (line 437)
        kwargs_20369 = {}
        # Getting the type of 'self' (line 437)
        self_20366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 437, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 437)
        write_20367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 437, 8), self_20366, 'write')
        # Calling write(args, kwargs) (line 437)
        write_call_result_20370 = invoke(stypy.reporting.localization.Localization(__file__, 437, 8), write_20367, *[str_20368], **kwargs_20369)
        
        
        # Call to interleave(...): (line 438)
        # Processing the call arguments (line 438)

        @norecursion
        def _stypy_temp_lambda_34(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_34'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_34', 438, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_34.stypy_localization = localization
            _stypy_temp_lambda_34.stypy_type_of_self = None
            _stypy_temp_lambda_34.stypy_type_store = module_type_store
            _stypy_temp_lambda_34.stypy_function_name = '_stypy_temp_lambda_34'
            _stypy_temp_lambda_34.stypy_param_names_list = []
            _stypy_temp_lambda_34.stypy_varargs_param_name = None
            _stypy_temp_lambda_34.stypy_kwargs_param_name = None
            _stypy_temp_lambda_34.stypy_call_defaults = defaults
            _stypy_temp_lambda_34.stypy_call_varargs = varargs
            _stypy_temp_lambda_34.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_34', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_34', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 438)
            # Processing the call arguments (line 438)
            str_20374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 438, 38), 'str', ', ')
            # Processing the call keyword arguments (line 438)
            kwargs_20375 = {}
            # Getting the type of 'self' (line 438)
            self_20372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 438)
            write_20373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 27), self_20372, 'write')
            # Calling write(args, kwargs) (line 438)
            write_call_result_20376 = invoke(stypy.reporting.localization.Localization(__file__, 438, 27), write_20373, *[str_20374], **kwargs_20375)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 438)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'stypy_return_type', write_call_result_20376)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_34' in the type store
            # Getting the type of 'stypy_return_type' (line 438)
            stypy_return_type_20377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_20377)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_34'
            return stypy_return_type_20377

        # Assigning a type to the variable '_stypy_temp_lambda_34' (line 438)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), '_stypy_temp_lambda_34', _stypy_temp_lambda_34)
        # Getting the type of '_stypy_temp_lambda_34' (line 438)
        _stypy_temp_lambda_34_20378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 19), '_stypy_temp_lambda_34')
        # Getting the type of 'self' (line 438)
        self_20379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 438)
        visit_20380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 45), self_20379, 'visit')
        # Getting the type of 't' (line 438)
        t_20381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 57), 't', False)
        # Obtaining the member 'elts' of a type (line 438)
        elts_20382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 438, 57), t_20381, 'elts')
        # Processing the call keyword arguments (line 438)
        kwargs_20383 = {}
        # Getting the type of 'interleave' (line 438)
        interleave_20371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 438, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 438)
        interleave_call_result_20384 = invoke(stypy.reporting.localization.Localization(__file__, 438, 8), interleave_20371, *[_stypy_temp_lambda_34_20378, visit_20380, elts_20382], **kwargs_20383)
        
        
        # Call to write(...): (line 439)
        # Processing the call arguments (line 439)
        str_20387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 439, 19), 'str', '}')
        # Processing the call keyword arguments (line 439)
        kwargs_20388 = {}
        # Getting the type of 'self' (line 439)
        self_20385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 439, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 439)
        write_20386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 439, 8), self_20385, 'write')
        # Calling write(args, kwargs) (line 439)
        write_call_result_20389 = invoke(stypy.reporting.localization.Localization(__file__, 439, 8), write_20386, *[str_20387], **kwargs_20388)
        
        
        # ################# End of 'visit_Set(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Set' in the type store
        # Getting the type of 'stypy_return_type' (line 435)
        stypy_return_type_20390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20390)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Set'
        return stypy_return_type_20390


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
        str_20393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 442, 19), 'str', '{')
        # Processing the call keyword arguments (line 442)
        kwargs_20394 = {}
        # Getting the type of 'self' (line 442)
        self_20391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 442)
        write_20392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 8), self_20391, 'write')
        # Calling write(args, kwargs) (line 442)
        write_call_result_20395 = invoke(stypy.reporting.localization.Localization(__file__, 442, 8), write_20392, *[str_20393], **kwargs_20394)
        

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
            int_20396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 12), 'int')
            # Getting the type of 'pair' (line 445)
            pair_20397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'pair')
            # Obtaining the member '__getitem__' of a type (line 445)
            getitem___20398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 12), pair_20397, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 445)
            subscript_call_result_20399 = invoke(stypy.reporting.localization.Localization(__file__, 445, 12), getitem___20398, int_20396)
            
            # Assigning a type to the variable 'tuple_var_assignment_18877' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'tuple_var_assignment_18877', subscript_call_result_20399)
            
            # Assigning a Subscript to a Name (line 445):
            
            # Obtaining the type of the subscript
            int_20400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 445, 12), 'int')
            # Getting the type of 'pair' (line 445)
            pair_20401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 21), 'pair')
            # Obtaining the member '__getitem__' of a type (line 445)
            getitem___20402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 445, 12), pair_20401, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 445)
            subscript_call_result_20403 = invoke(stypy.reporting.localization.Localization(__file__, 445, 12), getitem___20402, int_20400)
            
            # Assigning a type to the variable 'tuple_var_assignment_18878' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'tuple_var_assignment_18878', subscript_call_result_20403)
            
            # Assigning a Name to a Name (line 445):
            # Getting the type of 'tuple_var_assignment_18877' (line 445)
            tuple_var_assignment_18877_20404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'tuple_var_assignment_18877')
            # Assigning a type to the variable 'k' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 13), 'k', tuple_var_assignment_18877_20404)
            
            # Assigning a Name to a Name (line 445):
            # Getting the type of 'tuple_var_assignment_18878' (line 445)
            tuple_var_assignment_18878_20405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 445, 12), 'tuple_var_assignment_18878')
            # Assigning a type to the variable 'v' (line 445)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 445, 16), 'v', tuple_var_assignment_18878_20405)
            
            # Call to visit(...): (line 446)
            # Processing the call arguments (line 446)
            # Getting the type of 'k' (line 446)
            k_20408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 23), 'k', False)
            # Processing the call keyword arguments (line 446)
            kwargs_20409 = {}
            # Getting the type of 'self' (line 446)
            self_20406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 446)
            visit_20407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 12), self_20406, 'visit')
            # Calling visit(args, kwargs) (line 446)
            visit_call_result_20410 = invoke(stypy.reporting.localization.Localization(__file__, 446, 12), visit_20407, *[k_20408], **kwargs_20409)
            
            
            # Call to write(...): (line 447)
            # Processing the call arguments (line 447)
            str_20413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 447, 23), 'str', ': ')
            # Processing the call keyword arguments (line 447)
            kwargs_20414 = {}
            # Getting the type of 'self' (line 447)
            self_20411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 447, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 447)
            write_20412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 447, 12), self_20411, 'write')
            # Calling write(args, kwargs) (line 447)
            write_call_result_20415 = invoke(stypy.reporting.localization.Localization(__file__, 447, 12), write_20412, *[str_20413], **kwargs_20414)
            
            
            # Call to visit(...): (line 448)
            # Processing the call arguments (line 448)
            # Getting the type of 'v' (line 448)
            v_20418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 23), 'v', False)
            # Processing the call keyword arguments (line 448)
            kwargs_20419 = {}
            # Getting the type of 'self' (line 448)
            self_20416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 448, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 448)
            visit_20417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 448, 12), self_20416, 'visit')
            # Calling visit(args, kwargs) (line 448)
            visit_call_result_20420 = invoke(stypy.reporting.localization.Localization(__file__, 448, 12), visit_20417, *[v_20418], **kwargs_20419)
            
            
            # ################# End of 'write_pair(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'write_pair' in the type store
            # Getting the type of 'stypy_return_type' (line 444)
            stypy_return_type_20421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_20421)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'write_pair'
            return stypy_return_type_20421

        # Assigning a type to the variable 'write_pair' (line 444)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 444, 8), 'write_pair', write_pair)
        
        # Call to interleave(...): (line 450)
        # Processing the call arguments (line 450)

        @norecursion
        def _stypy_temp_lambda_35(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_35'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_35', 450, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_35.stypy_localization = localization
            _stypy_temp_lambda_35.stypy_type_of_self = None
            _stypy_temp_lambda_35.stypy_type_store = module_type_store
            _stypy_temp_lambda_35.stypy_function_name = '_stypy_temp_lambda_35'
            _stypy_temp_lambda_35.stypy_param_names_list = []
            _stypy_temp_lambda_35.stypy_varargs_param_name = None
            _stypy_temp_lambda_35.stypy_kwargs_param_name = None
            _stypy_temp_lambda_35.stypy_call_defaults = defaults
            _stypy_temp_lambda_35.stypy_call_varargs = varargs
            _stypy_temp_lambda_35.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_35', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_35', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 450)
            # Processing the call arguments (line 450)
            str_20425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 450, 38), 'str', ', ')
            # Processing the call keyword arguments (line 450)
            kwargs_20426 = {}
            # Getting the type of 'self' (line 450)
            self_20423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 450)
            write_20424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 27), self_20423, 'write')
            # Calling write(args, kwargs) (line 450)
            write_call_result_20427 = invoke(stypy.reporting.localization.Localization(__file__, 450, 27), write_20424, *[str_20425], **kwargs_20426)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 450)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), 'stypy_return_type', write_call_result_20427)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_35' in the type store
            # Getting the type of 'stypy_return_type' (line 450)
            stypy_return_type_20428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_20428)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_35'
            return stypy_return_type_20428

        # Assigning a type to the variable '_stypy_temp_lambda_35' (line 450)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), '_stypy_temp_lambda_35', _stypy_temp_lambda_35)
        # Getting the type of '_stypy_temp_lambda_35' (line 450)
        _stypy_temp_lambda_35_20429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 19), '_stypy_temp_lambda_35')
        # Getting the type of 'write_pair' (line 450)
        write_pair_20430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 45), 'write_pair', False)
        
        # Call to zip(...): (line 450)
        # Processing the call arguments (line 450)
        # Getting the type of 't' (line 450)
        t_20432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 61), 't', False)
        # Obtaining the member 'keys' of a type (line 450)
        keys_20433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 61), t_20432, 'keys')
        # Getting the type of 't' (line 450)
        t_20434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 69), 't', False)
        # Obtaining the member 'values' of a type (line 450)
        values_20435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 450, 69), t_20434, 'values')
        # Processing the call keyword arguments (line 450)
        kwargs_20436 = {}
        # Getting the type of 'zip' (line 450)
        zip_20431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 57), 'zip', False)
        # Calling zip(args, kwargs) (line 450)
        zip_call_result_20437 = invoke(stypy.reporting.localization.Localization(__file__, 450, 57), zip_20431, *[keys_20433, values_20435], **kwargs_20436)
        
        # Processing the call keyword arguments (line 450)
        kwargs_20438 = {}
        # Getting the type of 'interleave' (line 450)
        interleave_20422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 450, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 450)
        interleave_call_result_20439 = invoke(stypy.reporting.localization.Localization(__file__, 450, 8), interleave_20422, *[_stypy_temp_lambda_35_20429, write_pair_20430, zip_call_result_20437], **kwargs_20438)
        
        
        # Call to write(...): (line 451)
        # Processing the call arguments (line 451)
        str_20442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 19), 'str', '}')
        # Processing the call keyword arguments (line 451)
        kwargs_20443 = {}
        # Getting the type of 'self' (line 451)
        self_20440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 451)
        write_20441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 8), self_20440, 'write')
        # Calling write(args, kwargs) (line 451)
        write_call_result_20444 = invoke(stypy.reporting.localization.Localization(__file__, 451, 8), write_20441, *[str_20442], **kwargs_20443)
        
        
        # ################# End of 'visit_Dict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Dict' in the type store
        # Getting the type of 'stypy_return_type' (line 441)
        stypy_return_type_20445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 441, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20445)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Dict'
        return stypy_return_type_20445


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
        str_20448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 454, 19), 'str', '(')
        # Processing the call keyword arguments (line 454)
        kwargs_20449 = {}
        # Getting the type of 'self' (line 454)
        self_20446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 454)
        write_20447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 454, 8), self_20446, 'write')
        # Calling write(args, kwargs) (line 454)
        write_call_result_20450 = invoke(stypy.reporting.localization.Localization(__file__, 454, 8), write_20447, *[str_20448], **kwargs_20449)
        
        
        
        # Call to len(...): (line 455)
        # Processing the call arguments (line 455)
        # Getting the type of 't' (line 455)
        t_20452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 15), 't', False)
        # Obtaining the member 'elts' of a type (line 455)
        elts_20453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 15), t_20452, 'elts')
        # Processing the call keyword arguments (line 455)
        kwargs_20454 = {}
        # Getting the type of 'len' (line 455)
        len_20451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 11), 'len', False)
        # Calling len(args, kwargs) (line 455)
        len_call_result_20455 = invoke(stypy.reporting.localization.Localization(__file__, 455, 11), len_20451, *[elts_20453], **kwargs_20454)
        
        int_20456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 26), 'int')
        # Applying the binary operator '==' (line 455)
        result_eq_20457 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 11), '==', len_call_result_20455, int_20456)
        
        # Testing if the type of an if condition is none (line 455)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 455, 8), result_eq_20457):
            
            # Call to interleave(...): (line 460)
            # Processing the call arguments (line 460)

            @norecursion
            def _stypy_temp_lambda_36(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_36'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_36', 460, 23, True)
                # Passed parameters checking function
                _stypy_temp_lambda_36.stypy_localization = localization
                _stypy_temp_lambda_36.stypy_type_of_self = None
                _stypy_temp_lambda_36.stypy_type_store = module_type_store
                _stypy_temp_lambda_36.stypy_function_name = '_stypy_temp_lambda_36'
                _stypy_temp_lambda_36.stypy_param_names_list = []
                _stypy_temp_lambda_36.stypy_varargs_param_name = None
                _stypy_temp_lambda_36.stypy_kwargs_param_name = None
                _stypy_temp_lambda_36.stypy_call_defaults = defaults
                _stypy_temp_lambda_36.stypy_call_varargs = varargs
                _stypy_temp_lambda_36.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_36', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_36', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to write(...): (line 460)
                # Processing the call arguments (line 460)
                str_20478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 42), 'str', ', ')
                # Processing the call keyword arguments (line 460)
                kwargs_20479 = {}
                # Getting the type of 'self' (line 460)
                self_20476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 31), 'self', False)
                # Obtaining the member 'write' of a type (line 460)
                write_20477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 31), self_20476, 'write')
                # Calling write(args, kwargs) (line 460)
                write_call_result_20480 = invoke(stypy.reporting.localization.Localization(__file__, 460, 31), write_20477, *[str_20478], **kwargs_20479)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 460)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'stypy_return_type', write_call_result_20480)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_36' in the type store
                # Getting the type of 'stypy_return_type' (line 460)
                stypy_return_type_20481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_20481)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_36'
                return stypy_return_type_20481

            # Assigning a type to the variable '_stypy_temp_lambda_36' (line 460)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), '_stypy_temp_lambda_36', _stypy_temp_lambda_36)
            # Getting the type of '_stypy_temp_lambda_36' (line 460)
            _stypy_temp_lambda_36_20482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), '_stypy_temp_lambda_36')
            # Getting the type of 'self' (line 460)
            self_20483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 49), 'self', False)
            # Obtaining the member 'visit' of a type (line 460)
            visit_20484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 49), self_20483, 'visit')
            # Getting the type of 't' (line 460)
            t_20485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 61), 't', False)
            # Obtaining the member 'elts' of a type (line 460)
            elts_20486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 61), t_20485, 'elts')
            # Processing the call keyword arguments (line 460)
            kwargs_20487 = {}
            # Getting the type of 'interleave' (line 460)
            interleave_20475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'interleave', False)
            # Calling interleave(args, kwargs) (line 460)
            interleave_call_result_20488 = invoke(stypy.reporting.localization.Localization(__file__, 460, 12), interleave_20475, *[_stypy_temp_lambda_36_20482, visit_20484, elts_20486], **kwargs_20487)
            
        else:
            
            # Testing the type of an if condition (line 455)
            if_condition_20458 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 455, 8), result_eq_20457)
            # Assigning a type to the variable 'if_condition_20458' (line 455)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 8), 'if_condition_20458', if_condition_20458)
            # SSA begins for if statement (line 455)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Tuple (line 456):
            
            # Assigning a Subscript to a Name (line 456):
            
            # Obtaining the type of the subscript
            int_20459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 12), 'int')
            # Getting the type of 't' (line 456)
            t_20460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 21), 't')
            # Obtaining the member 'elts' of a type (line 456)
            elts_20461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 21), t_20460, 'elts')
            # Obtaining the member '__getitem__' of a type (line 456)
            getitem___20462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 456, 12), elts_20461, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 456)
            subscript_call_result_20463 = invoke(stypy.reporting.localization.Localization(__file__, 456, 12), getitem___20462, int_20459)
            
            # Assigning a type to the variable 'tuple_var_assignment_18879' (line 456)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'tuple_var_assignment_18879', subscript_call_result_20463)
            
            # Assigning a Name to a Name (line 456):
            # Getting the type of 'tuple_var_assignment_18879' (line 456)
            tuple_var_assignment_18879_20464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 12), 'tuple_var_assignment_18879')
            # Assigning a type to the variable 'elt' (line 456)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 13), 'elt', tuple_var_assignment_18879_20464)
            
            # Call to visit(...): (line 457)
            # Processing the call arguments (line 457)
            # Getting the type of 'elt' (line 457)
            elt_20467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 23), 'elt', False)
            # Processing the call keyword arguments (line 457)
            kwargs_20468 = {}
            # Getting the type of 'self' (line 457)
            self_20465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 457)
            visit_20466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 457, 12), self_20465, 'visit')
            # Calling visit(args, kwargs) (line 457)
            visit_call_result_20469 = invoke(stypy.reporting.localization.Localization(__file__, 457, 12), visit_20466, *[elt_20467], **kwargs_20468)
            
            
            # Call to write(...): (line 458)
            # Processing the call arguments (line 458)
            str_20472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 458, 23), 'str', ',')
            # Processing the call keyword arguments (line 458)
            kwargs_20473 = {}
            # Getting the type of 'self' (line 458)
            self_20470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 458, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 458)
            write_20471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 458, 12), self_20470, 'write')
            # Calling write(args, kwargs) (line 458)
            write_call_result_20474 = invoke(stypy.reporting.localization.Localization(__file__, 458, 12), write_20471, *[str_20472], **kwargs_20473)
            
            # SSA branch for the else part of an if statement (line 455)
            module_type_store.open_ssa_branch('else')
            
            # Call to interleave(...): (line 460)
            # Processing the call arguments (line 460)

            @norecursion
            def _stypy_temp_lambda_36(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function '_stypy_temp_lambda_36'
                module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_36', 460, 23, True)
                # Passed parameters checking function
                _stypy_temp_lambda_36.stypy_localization = localization
                _stypy_temp_lambda_36.stypy_type_of_self = None
                _stypy_temp_lambda_36.stypy_type_store = module_type_store
                _stypy_temp_lambda_36.stypy_function_name = '_stypy_temp_lambda_36'
                _stypy_temp_lambda_36.stypy_param_names_list = []
                _stypy_temp_lambda_36.stypy_varargs_param_name = None
                _stypy_temp_lambda_36.stypy_kwargs_param_name = None
                _stypy_temp_lambda_36.stypy_call_defaults = defaults
                _stypy_temp_lambda_36.stypy_call_varargs = varargs
                _stypy_temp_lambda_36.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_36', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Stacktrace push for error reporting
                localization.set_stack_trace('_stypy_temp_lambda_36', [], arguments)
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of the lambda function code ##################

                
                # Call to write(...): (line 460)
                # Processing the call arguments (line 460)
                str_20478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 460, 42), 'str', ', ')
                # Processing the call keyword arguments (line 460)
                kwargs_20479 = {}
                # Getting the type of 'self' (line 460)
                self_20476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 31), 'self', False)
                # Obtaining the member 'write' of a type (line 460)
                write_20477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 31), self_20476, 'write')
                # Calling write(args, kwargs) (line 460)
                write_call_result_20480 = invoke(stypy.reporting.localization.Localization(__file__, 460, 31), write_20477, *[str_20478], **kwargs_20479)
                
                # Assigning the return type of the lambda function
                # Assigning a type to the variable 'stypy_return_type' (line 460)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'stypy_return_type', write_call_result_20480)
                
                # ################# End of the lambda function code ##################

                # Stacktrace pop (error reporting)
                localization.unset_stack_trace()
                
                # Storing the return type of function '_stypy_temp_lambda_36' in the type store
                # Getting the type of 'stypy_return_type' (line 460)
                stypy_return_type_20481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_20481)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function '_stypy_temp_lambda_36'
                return stypy_return_type_20481

            # Assigning a type to the variable '_stypy_temp_lambda_36' (line 460)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), '_stypy_temp_lambda_36', _stypy_temp_lambda_36)
            # Getting the type of '_stypy_temp_lambda_36' (line 460)
            _stypy_temp_lambda_36_20482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 23), '_stypy_temp_lambda_36')
            # Getting the type of 'self' (line 460)
            self_20483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 49), 'self', False)
            # Obtaining the member 'visit' of a type (line 460)
            visit_20484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 49), self_20483, 'visit')
            # Getting the type of 't' (line 460)
            t_20485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 61), 't', False)
            # Obtaining the member 'elts' of a type (line 460)
            elts_20486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 460, 61), t_20485, 'elts')
            # Processing the call keyword arguments (line 460)
            kwargs_20487 = {}
            # Getting the type of 'interleave' (line 460)
            interleave_20475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 12), 'interleave', False)
            # Calling interleave(args, kwargs) (line 460)
            interleave_call_result_20488 = invoke(stypy.reporting.localization.Localization(__file__, 460, 12), interleave_20475, *[_stypy_temp_lambda_36_20482, visit_20484, elts_20486], **kwargs_20487)
            
            # SSA join for if statement (line 455)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 461)
        # Processing the call arguments (line 461)
        str_20491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 461, 19), 'str', ')')
        # Processing the call keyword arguments (line 461)
        kwargs_20492 = {}
        # Getting the type of 'self' (line 461)
        self_20489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 461)
        write_20490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 8), self_20489, 'write')
        # Calling write(args, kwargs) (line 461)
        write_call_result_20493 = invoke(stypy.reporting.localization.Localization(__file__, 461, 8), write_20490, *[str_20491], **kwargs_20492)
        
        
        # ################# End of 'visit_Tuple(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Tuple' in the type store
        # Getting the type of 'stypy_return_type' (line 453)
        stypy_return_type_20494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 453, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20494)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Tuple'
        return stypy_return_type_20494

    
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
        str_20497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 19), 'str', '(')
        # Processing the call keyword arguments (line 466)
        kwargs_20498 = {}
        # Getting the type of 'self' (line 466)
        self_20495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 466)
        write_20496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 8), self_20495, 'write')
        # Calling write(args, kwargs) (line 466)
        write_call_result_20499 = invoke(stypy.reporting.localization.Localization(__file__, 466, 8), write_20496, *[str_20497], **kwargs_20498)
        
        
        # Call to write(...): (line 467)
        # Processing the call arguments (line 467)
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 467)
        t_20502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 29), 't', False)
        # Obtaining the member 'op' of a type (line 467)
        op_20503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 29), t_20502, 'op')
        # Obtaining the member '__class__' of a type (line 467)
        class___20504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 29), op_20503, '__class__')
        # Obtaining the member '__name__' of a type (line 467)
        name___20505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 29), class___20504, '__name__')
        # Getting the type of 'self' (line 467)
        self_20506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 19), 'self', False)
        # Obtaining the member 'unop' of a type (line 467)
        unop_20507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 19), self_20506, 'unop')
        # Obtaining the member '__getitem__' of a type (line 467)
        getitem___20508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 19), unop_20507, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 467)
        subscript_call_result_20509 = invoke(stypy.reporting.localization.Localization(__file__, 467, 19), getitem___20508, name___20505)
        
        # Processing the call keyword arguments (line 467)
        kwargs_20510 = {}
        # Getting the type of 'self' (line 467)
        self_20500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 467)
        write_20501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 8), self_20500, 'write')
        # Calling write(args, kwargs) (line 467)
        write_call_result_20511 = invoke(stypy.reporting.localization.Localization(__file__, 467, 8), write_20501, *[subscript_call_result_20509], **kwargs_20510)
        
        
        # Call to write(...): (line 468)
        # Processing the call arguments (line 468)
        str_20514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 468, 19), 'str', ' ')
        # Processing the call keyword arguments (line 468)
        kwargs_20515 = {}
        # Getting the type of 'self' (line 468)
        self_20512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 468)
        write_20513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 8), self_20512, 'write')
        # Calling write(args, kwargs) (line 468)
        write_call_result_20516 = invoke(stypy.reporting.localization.Localization(__file__, 468, 8), write_20513, *[str_20514], **kwargs_20515)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 't' (line 474)
        t_20518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 22), 't', False)
        # Obtaining the member 'op' of a type (line 474)
        op_20519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 22), t_20518, 'op')
        # Getting the type of 'ast' (line 474)
        ast_20520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 28), 'ast', False)
        # Obtaining the member 'USub' of a type (line 474)
        USub_20521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 28), ast_20520, 'USub')
        # Processing the call keyword arguments (line 474)
        kwargs_20522 = {}
        # Getting the type of 'isinstance' (line 474)
        isinstance_20517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 474)
        isinstance_call_result_20523 = invoke(stypy.reporting.localization.Localization(__file__, 474, 11), isinstance_20517, *[op_20519, USub_20521], **kwargs_20522)
        
        
        # Call to isinstance(...): (line 474)
        # Processing the call arguments (line 474)
        # Getting the type of 't' (line 474)
        t_20525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 53), 't', False)
        # Obtaining the member 'operand' of a type (line 474)
        operand_20526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 53), t_20525, 'operand')
        # Getting the type of 'ast' (line 474)
        ast_20527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 64), 'ast', False)
        # Obtaining the member 'Num' of a type (line 474)
        Num_20528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 64), ast_20527, 'Num')
        # Processing the call keyword arguments (line 474)
        kwargs_20529 = {}
        # Getting the type of 'isinstance' (line 474)
        isinstance_20524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 42), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 474)
        isinstance_call_result_20530 = invoke(stypy.reporting.localization.Localization(__file__, 474, 42), isinstance_20524, *[operand_20526, Num_20528], **kwargs_20529)
        
        # Applying the binary operator 'and' (line 474)
        result_and_keyword_20531 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 11), 'and', isinstance_call_result_20523, isinstance_call_result_20530)
        
        # Testing if the type of an if condition is none (line 474)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 474, 8), result_and_keyword_20531):
            
            # Call to visit(...): (line 479)
            # Processing the call arguments (line 479)
            # Getting the type of 't' (line 479)
            t_20551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 23), 't', False)
            # Obtaining the member 'operand' of a type (line 479)
            operand_20552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 23), t_20551, 'operand')
            # Processing the call keyword arguments (line 479)
            kwargs_20553 = {}
            # Getting the type of 'self' (line 479)
            self_20549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 479)
            visit_20550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 12), self_20549, 'visit')
            # Calling visit(args, kwargs) (line 479)
            visit_call_result_20554 = invoke(stypy.reporting.localization.Localization(__file__, 479, 12), visit_20550, *[operand_20552], **kwargs_20553)
            
        else:
            
            # Testing the type of an if condition (line 474)
            if_condition_20532 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 8), result_and_keyword_20531)
            # Assigning a type to the variable 'if_condition_20532' (line 474)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 8), 'if_condition_20532', if_condition_20532)
            # SSA begins for if statement (line 474)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 475)
            # Processing the call arguments (line 475)
            str_20535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 475, 23), 'str', '(')
            # Processing the call keyword arguments (line 475)
            kwargs_20536 = {}
            # Getting the type of 'self' (line 475)
            self_20533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 475)
            write_20534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 475, 12), self_20533, 'write')
            # Calling write(args, kwargs) (line 475)
            write_call_result_20537 = invoke(stypy.reporting.localization.Localization(__file__, 475, 12), write_20534, *[str_20535], **kwargs_20536)
            
            
            # Call to visit(...): (line 476)
            # Processing the call arguments (line 476)
            # Getting the type of 't' (line 476)
            t_20540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 23), 't', False)
            # Obtaining the member 'operand' of a type (line 476)
            operand_20541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 23), t_20540, 'operand')
            # Processing the call keyword arguments (line 476)
            kwargs_20542 = {}
            # Getting the type of 'self' (line 476)
            self_20538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 476)
            visit_20539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 476, 12), self_20538, 'visit')
            # Calling visit(args, kwargs) (line 476)
            visit_call_result_20543 = invoke(stypy.reporting.localization.Localization(__file__, 476, 12), visit_20539, *[operand_20541], **kwargs_20542)
            
            
            # Call to write(...): (line 477)
            # Processing the call arguments (line 477)
            str_20546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 23), 'str', ')')
            # Processing the call keyword arguments (line 477)
            kwargs_20547 = {}
            # Getting the type of 'self' (line 477)
            self_20544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 477)
            write_20545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 12), self_20544, 'write')
            # Calling write(args, kwargs) (line 477)
            write_call_result_20548 = invoke(stypy.reporting.localization.Localization(__file__, 477, 12), write_20545, *[str_20546], **kwargs_20547)
            
            # SSA branch for the else part of an if statement (line 474)
            module_type_store.open_ssa_branch('else')
            
            # Call to visit(...): (line 479)
            # Processing the call arguments (line 479)
            # Getting the type of 't' (line 479)
            t_20551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 23), 't', False)
            # Obtaining the member 'operand' of a type (line 479)
            operand_20552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 23), t_20551, 'operand')
            # Processing the call keyword arguments (line 479)
            kwargs_20553 = {}
            # Getting the type of 'self' (line 479)
            self_20549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 479)
            visit_20550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 12), self_20549, 'visit')
            # Calling visit(args, kwargs) (line 479)
            visit_call_result_20554 = invoke(stypy.reporting.localization.Localization(__file__, 479, 12), visit_20550, *[operand_20552], **kwargs_20553)
            
            # SSA join for if statement (line 474)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 480)
        # Processing the call arguments (line 480)
        str_20557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 480, 19), 'str', ')')
        # Processing the call keyword arguments (line 480)
        kwargs_20558 = {}
        # Getting the type of 'self' (line 480)
        self_20555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 480)
        write_20556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 480, 8), self_20555, 'write')
        # Calling write(args, kwargs) (line 480)
        write_call_result_20559 = invoke(stypy.reporting.localization.Localization(__file__, 480, 8), write_20556, *[str_20557], **kwargs_20558)
        
        
        # ################# End of 'visit_UnaryOp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_UnaryOp' in the type store
        # Getting the type of 'stypy_return_type' (line 465)
        stypy_return_type_20560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20560)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_UnaryOp'
        return stypy_return_type_20560

    
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
        str_20563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 487, 19), 'str', '(')
        # Processing the call keyword arguments (line 487)
        kwargs_20564 = {}
        # Getting the type of 'self' (line 487)
        self_20561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 487, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 487)
        write_20562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 487, 8), self_20561, 'write')
        # Calling write(args, kwargs) (line 487)
        write_call_result_20565 = invoke(stypy.reporting.localization.Localization(__file__, 487, 8), write_20562, *[str_20563], **kwargs_20564)
        
        
        # Call to visit(...): (line 488)
        # Processing the call arguments (line 488)
        # Getting the type of 't' (line 488)
        t_20568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 19), 't', False)
        # Obtaining the member 'left' of a type (line 488)
        left_20569 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 19), t_20568, 'left')
        # Processing the call keyword arguments (line 488)
        kwargs_20570 = {}
        # Getting the type of 'self' (line 488)
        self_20566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 488)
        visit_20567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 488, 8), self_20566, 'visit')
        # Calling visit(args, kwargs) (line 488)
        visit_call_result_20571 = invoke(stypy.reporting.localization.Localization(__file__, 488, 8), visit_20567, *[left_20569], **kwargs_20570)
        
        
        # Call to write(...): (line 489)
        # Processing the call arguments (line 489)
        str_20574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 19), 'str', ' ')
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 489)
        t_20575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 36), 't', False)
        # Obtaining the member 'op' of a type (line 489)
        op_20576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 36), t_20575, 'op')
        # Obtaining the member '__class__' of a type (line 489)
        class___20577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 36), op_20576, '__class__')
        # Obtaining the member '__name__' of a type (line 489)
        name___20578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 36), class___20577, '__name__')
        # Getting the type of 'self' (line 489)
        self_20579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 25), 'self', False)
        # Obtaining the member 'binop' of a type (line 489)
        binop_20580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 25), self_20579, 'binop')
        # Obtaining the member '__getitem__' of a type (line 489)
        getitem___20581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 25), binop_20580, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 489)
        subscript_call_result_20582 = invoke(stypy.reporting.localization.Localization(__file__, 489, 25), getitem___20581, name___20578)
        
        # Applying the binary operator '+' (line 489)
        result_add_20583 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 19), '+', str_20574, subscript_call_result_20582)
        
        str_20584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 489, 63), 'str', ' ')
        # Applying the binary operator '+' (line 489)
        result_add_20585 = python_operator(stypy.reporting.localization.Localization(__file__, 489, 61), '+', result_add_20583, str_20584)
        
        # Processing the call keyword arguments (line 489)
        kwargs_20586 = {}
        # Getting the type of 'self' (line 489)
        self_20572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 489)
        write_20573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 489, 8), self_20572, 'write')
        # Calling write(args, kwargs) (line 489)
        write_call_result_20587 = invoke(stypy.reporting.localization.Localization(__file__, 489, 8), write_20573, *[result_add_20585], **kwargs_20586)
        
        
        # Call to visit(...): (line 490)
        # Processing the call arguments (line 490)
        # Getting the type of 't' (line 490)
        t_20590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 19), 't', False)
        # Obtaining the member 'right' of a type (line 490)
        right_20591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 19), t_20590, 'right')
        # Processing the call keyword arguments (line 490)
        kwargs_20592 = {}
        # Getting the type of 'self' (line 490)
        self_20588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 490)
        visit_20589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 8), self_20588, 'visit')
        # Calling visit(args, kwargs) (line 490)
        visit_call_result_20593 = invoke(stypy.reporting.localization.Localization(__file__, 490, 8), visit_20589, *[right_20591], **kwargs_20592)
        
        
        # Call to write(...): (line 491)
        # Processing the call arguments (line 491)
        str_20596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 19), 'str', ')')
        # Processing the call keyword arguments (line 491)
        kwargs_20597 = {}
        # Getting the type of 'self' (line 491)
        self_20594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 491)
        write_20595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 8), self_20594, 'write')
        # Calling write(args, kwargs) (line 491)
        write_call_result_20598 = invoke(stypy.reporting.localization.Localization(__file__, 491, 8), write_20595, *[str_20596], **kwargs_20597)
        
        
        # ################# End of 'visit_BinOp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_BinOp' in the type store
        # Getting the type of 'stypy_return_type' (line 486)
        stypy_return_type_20599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 486, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20599)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_BinOp'
        return stypy_return_type_20599

    
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
        str_20602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 497, 19), 'str', '(')
        # Processing the call keyword arguments (line 497)
        kwargs_20603 = {}
        # Getting the type of 'self' (line 497)
        self_20600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 497, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 497)
        write_20601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 497, 8), self_20600, 'write')
        # Calling write(args, kwargs) (line 497)
        write_call_result_20604 = invoke(stypy.reporting.localization.Localization(__file__, 497, 8), write_20601, *[str_20602], **kwargs_20603)
        
        
        # Call to visit(...): (line 498)
        # Processing the call arguments (line 498)
        # Getting the type of 't' (line 498)
        t_20607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 19), 't', False)
        # Obtaining the member 'left' of a type (line 498)
        left_20608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 19), t_20607, 'left')
        # Processing the call keyword arguments (line 498)
        kwargs_20609 = {}
        # Getting the type of 'self' (line 498)
        self_20605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 498)
        visit_20606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 498, 8), self_20605, 'visit')
        # Calling visit(args, kwargs) (line 498)
        visit_call_result_20610 = invoke(stypy.reporting.localization.Localization(__file__, 498, 8), visit_20606, *[left_20608], **kwargs_20609)
        
        
        
        # Call to zip(...): (line 499)
        # Processing the call arguments (line 499)
        # Getting the type of 't' (line 499)
        t_20612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 24), 't', False)
        # Obtaining the member 'ops' of a type (line 499)
        ops_20613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 24), t_20612, 'ops')
        # Getting the type of 't' (line 499)
        t_20614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 31), 't', False)
        # Obtaining the member 'comparators' of a type (line 499)
        comparators_20615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 31), t_20614, 'comparators')
        # Processing the call keyword arguments (line 499)
        kwargs_20616 = {}
        # Getting the type of 'zip' (line 499)
        zip_20611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 499)
        zip_call_result_20617 = invoke(stypy.reporting.localization.Localization(__file__, 499, 20), zip_20611, *[ops_20613, comparators_20615], **kwargs_20616)
        
        # Assigning a type to the variable 'zip_call_result_20617' (line 499)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'zip_call_result_20617', zip_call_result_20617)
        # Testing if the for loop is going to be iterated (line 499)
        # Testing the type of a for loop iterable (line 499)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 499, 8), zip_call_result_20617)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 499, 8), zip_call_result_20617):
            # Getting the type of the for loop variable (line 499)
            for_loop_var_20618 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 499, 8), zip_call_result_20617)
            # Assigning a type to the variable 'o' (line 499)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'o', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 8), for_loop_var_20618, 2, 0))
            # Assigning a type to the variable 'e' (line 499)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 8), 'e', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 499, 8), for_loop_var_20618, 2, 1))
            # SSA begins for a for statement (line 499)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to write(...): (line 500)
            # Processing the call arguments (line 500)
            str_20621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 23), 'str', ' ')
            
            # Obtaining the type of the subscript
            # Getting the type of 'o' (line 500)
            o_20622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 41), 'o', False)
            # Obtaining the member '__class__' of a type (line 500)
            class___20623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 41), o_20622, '__class__')
            # Obtaining the member '__name__' of a type (line 500)
            name___20624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 41), class___20623, '__name__')
            # Getting the type of 'self' (line 500)
            self_20625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 29), 'self', False)
            # Obtaining the member 'cmpops' of a type (line 500)
            cmpops_20626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 29), self_20625, 'cmpops')
            # Obtaining the member '__getitem__' of a type (line 500)
            getitem___20627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 29), cmpops_20626, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 500)
            subscript_call_result_20628 = invoke(stypy.reporting.localization.Localization(__file__, 500, 29), getitem___20627, name___20624)
            
            # Applying the binary operator '+' (line 500)
            result_add_20629 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 23), '+', str_20621, subscript_call_result_20628)
            
            str_20630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 500, 65), 'str', ' ')
            # Applying the binary operator '+' (line 500)
            result_add_20631 = python_operator(stypy.reporting.localization.Localization(__file__, 500, 63), '+', result_add_20629, str_20630)
            
            # Processing the call keyword arguments (line 500)
            kwargs_20632 = {}
            # Getting the type of 'self' (line 500)
            self_20619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 500, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 500)
            write_20620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 500, 12), self_20619, 'write')
            # Calling write(args, kwargs) (line 500)
            write_call_result_20633 = invoke(stypy.reporting.localization.Localization(__file__, 500, 12), write_20620, *[result_add_20631], **kwargs_20632)
            
            
            # Call to visit(...): (line 501)
            # Processing the call arguments (line 501)
            # Getting the type of 'e' (line 501)
            e_20636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 23), 'e', False)
            # Processing the call keyword arguments (line 501)
            kwargs_20637 = {}
            # Getting the type of 'self' (line 501)
            self_20634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 501)
            visit_20635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 501, 12), self_20634, 'visit')
            # Calling visit(args, kwargs) (line 501)
            visit_call_result_20638 = invoke(stypy.reporting.localization.Localization(__file__, 501, 12), visit_20635, *[e_20636], **kwargs_20637)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to write(...): (line 502)
        # Processing the call arguments (line 502)
        str_20641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 19), 'str', ')')
        # Processing the call keyword arguments (line 502)
        kwargs_20642 = {}
        # Getting the type of 'self' (line 502)
        self_20639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 502)
        write_20640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 8), self_20639, 'write')
        # Calling write(args, kwargs) (line 502)
        write_call_result_20643 = invoke(stypy.reporting.localization.Localization(__file__, 502, 8), write_20640, *[str_20641], **kwargs_20642)
        
        
        # ################# End of 'visit_Compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Compare' in the type store
        # Getting the type of 'stypy_return_type' (line 496)
        stypy_return_type_20644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 496, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20644)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Compare'
        return stypy_return_type_20644

    
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
        str_20647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 507, 19), 'str', '(')
        # Processing the call keyword arguments (line 507)
        kwargs_20648 = {}
        # Getting the type of 'self' (line 507)
        self_20645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 507, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 507)
        write_20646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 507, 8), self_20645, 'write')
        # Calling write(args, kwargs) (line 507)
        write_call_result_20649 = invoke(stypy.reporting.localization.Localization(__file__, 507, 8), write_20646, *[str_20647], **kwargs_20648)
        
        
        # Assigning a BinOp to a Name (line 508):
        
        # Assigning a BinOp to a Name (line 508):
        str_20650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 508, 12), 'str', ' %s ')
        
        # Obtaining the type of the subscript
        # Getting the type of 't' (line 508)
        t_20651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 34), 't')
        # Obtaining the member 'op' of a type (line 508)
        op_20652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 34), t_20651, 'op')
        # Obtaining the member '__class__' of a type (line 508)
        class___20653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 34), op_20652, '__class__')
        # Getting the type of 'self' (line 508)
        self_20654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 21), 'self')
        # Obtaining the member 'boolops' of a type (line 508)
        boolops_20655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 21), self_20654, 'boolops')
        # Obtaining the member '__getitem__' of a type (line 508)
        getitem___20656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 21), boolops_20655, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 508)
        subscript_call_result_20657 = invoke(stypy.reporting.localization.Localization(__file__, 508, 21), getitem___20656, class___20653)
        
        # Applying the binary operator '%' (line 508)
        result_mod_20658 = python_operator(stypy.reporting.localization.Localization(__file__, 508, 12), '%', str_20650, subscript_call_result_20657)
        
        # Assigning a type to the variable 's' (line 508)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 8), 's', result_mod_20658)
        
        # Call to interleave(...): (line 509)
        # Processing the call arguments (line 509)

        @norecursion
        def _stypy_temp_lambda_37(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_37'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_37', 509, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_37.stypy_localization = localization
            _stypy_temp_lambda_37.stypy_type_of_self = None
            _stypy_temp_lambda_37.stypy_type_store = module_type_store
            _stypy_temp_lambda_37.stypy_function_name = '_stypy_temp_lambda_37'
            _stypy_temp_lambda_37.stypy_param_names_list = []
            _stypy_temp_lambda_37.stypy_varargs_param_name = None
            _stypy_temp_lambda_37.stypy_kwargs_param_name = None
            _stypy_temp_lambda_37.stypy_call_defaults = defaults
            _stypy_temp_lambda_37.stypy_call_varargs = varargs
            _stypy_temp_lambda_37.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_37', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_37', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 509)
            # Processing the call arguments (line 509)
            # Getting the type of 's' (line 509)
            s_20662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 38), 's', False)
            # Processing the call keyword arguments (line 509)
            kwargs_20663 = {}
            # Getting the type of 'self' (line 509)
            self_20660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 509)
            write_20661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 27), self_20660, 'write')
            # Calling write(args, kwargs) (line 509)
            write_call_result_20664 = invoke(stypy.reporting.localization.Localization(__file__, 509, 27), write_20661, *[s_20662], **kwargs_20663)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 509)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 19), 'stypy_return_type', write_call_result_20664)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_37' in the type store
            # Getting the type of 'stypy_return_type' (line 509)
            stypy_return_type_20665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_20665)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_37'
            return stypy_return_type_20665

        # Assigning a type to the variable '_stypy_temp_lambda_37' (line 509)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 509, 19), '_stypy_temp_lambda_37', _stypy_temp_lambda_37)
        # Getting the type of '_stypy_temp_lambda_37' (line 509)
        _stypy_temp_lambda_37_20666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 19), '_stypy_temp_lambda_37')
        # Getting the type of 'self' (line 509)
        self_20667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 42), 'self', False)
        # Obtaining the member 'visit' of a type (line 509)
        visit_20668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 42), self_20667, 'visit')
        # Getting the type of 't' (line 509)
        t_20669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 54), 't', False)
        # Obtaining the member 'values' of a type (line 509)
        values_20670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 54), t_20669, 'values')
        # Processing the call keyword arguments (line 509)
        kwargs_20671 = {}
        # Getting the type of 'interleave' (line 509)
        interleave_20659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 509)
        interleave_call_result_20672 = invoke(stypy.reporting.localization.Localization(__file__, 509, 8), interleave_20659, *[_stypy_temp_lambda_37_20666, visit_20668, values_20670], **kwargs_20671)
        
        
        # Call to write(...): (line 510)
        # Processing the call arguments (line 510)
        str_20675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 510, 19), 'str', ')')
        # Processing the call keyword arguments (line 510)
        kwargs_20676 = {}
        # Getting the type of 'self' (line 510)
        self_20673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 510, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 510)
        write_20674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 510, 8), self_20673, 'write')
        # Calling write(args, kwargs) (line 510)
        write_call_result_20677 = invoke(stypy.reporting.localization.Localization(__file__, 510, 8), write_20674, *[str_20675], **kwargs_20676)
        
        
        # ################# End of 'visit_BoolOp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_BoolOp' in the type store
        # Getting the type of 'stypy_return_type' (line 506)
        stypy_return_type_20678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20678)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_BoolOp'
        return stypy_return_type_20678


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
        t_20681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 19), 't', False)
        # Obtaining the member 'value' of a type (line 513)
        value_20682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 19), t_20681, 'value')
        # Processing the call keyword arguments (line 513)
        kwargs_20683 = {}
        # Getting the type of 'self' (line 513)
        self_20679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 513, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 513)
        visit_20680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 513, 8), self_20679, 'visit')
        # Calling visit(args, kwargs) (line 513)
        visit_call_result_20684 = invoke(stypy.reporting.localization.Localization(__file__, 513, 8), visit_20680, *[value_20682], **kwargs_20683)
        
        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 't' (line 517)
        t_20686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 22), 't', False)
        # Obtaining the member 'value' of a type (line 517)
        value_20687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 22), t_20686, 'value')
        # Getting the type of 'ast' (line 517)
        ast_20688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 31), 'ast', False)
        # Obtaining the member 'Num' of a type (line 517)
        Num_20689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 31), ast_20688, 'Num')
        # Processing the call keyword arguments (line 517)
        kwargs_20690 = {}
        # Getting the type of 'isinstance' (line 517)
        isinstance_20685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 517)
        isinstance_call_result_20691 = invoke(stypy.reporting.localization.Localization(__file__, 517, 11), isinstance_20685, *[value_20687, Num_20689], **kwargs_20690)
        
        
        # Call to isinstance(...): (line 517)
        # Processing the call arguments (line 517)
        # Getting the type of 't' (line 517)
        t_20693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 55), 't', False)
        # Obtaining the member 'value' of a type (line 517)
        value_20694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 55), t_20693, 'value')
        # Obtaining the member 'n' of a type (line 517)
        n_20695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 517, 55), value_20694, 'n')
        # Getting the type of 'int' (line 517)
        int_20696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 66), 'int', False)
        # Processing the call keyword arguments (line 517)
        kwargs_20697 = {}
        # Getting the type of 'isinstance' (line 517)
        isinstance_20692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 517, 44), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 517)
        isinstance_call_result_20698 = invoke(stypy.reporting.localization.Localization(__file__, 517, 44), isinstance_20692, *[n_20695, int_20696], **kwargs_20697)
        
        # Applying the binary operator 'and' (line 517)
        result_and_keyword_20699 = python_operator(stypy.reporting.localization.Localization(__file__, 517, 11), 'and', isinstance_call_result_20691, isinstance_call_result_20698)
        
        # Testing if the type of an if condition is none (line 517)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 517, 8), result_and_keyword_20699):
            pass
        else:
            
            # Testing the type of an if condition (line 517)
            if_condition_20700 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 517, 8), result_and_keyword_20699)
            # Assigning a type to the variable 'if_condition_20700' (line 517)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 517, 8), 'if_condition_20700', if_condition_20700)
            # SSA begins for if statement (line 517)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 518)
            # Processing the call arguments (line 518)
            str_20703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 518, 23), 'str', ' ')
            # Processing the call keyword arguments (line 518)
            kwargs_20704 = {}
            # Getting the type of 'self' (line 518)
            self_20701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 518)
            write_20702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 518, 12), self_20701, 'write')
            # Calling write(args, kwargs) (line 518)
            write_call_result_20705 = invoke(stypy.reporting.localization.Localization(__file__, 518, 12), write_20702, *[str_20703], **kwargs_20704)
            
            # SSA join for if statement (line 517)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 519)
        # Processing the call arguments (line 519)
        str_20708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 19), 'str', '.')
        # Processing the call keyword arguments (line 519)
        kwargs_20709 = {}
        # Getting the type of 'self' (line 519)
        self_20706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 519)
        write_20707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 8), self_20706, 'write')
        # Calling write(args, kwargs) (line 519)
        write_call_result_20710 = invoke(stypy.reporting.localization.Localization(__file__, 519, 8), write_20707, *[str_20708], **kwargs_20709)
        
        
        # Call to write(...): (line 520)
        # Processing the call arguments (line 520)
        # Getting the type of 't' (line 520)
        t_20713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 19), 't', False)
        # Obtaining the member 'attr' of a type (line 520)
        attr_20714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 19), t_20713, 'attr')
        # Processing the call keyword arguments (line 520)
        kwargs_20715 = {}
        # Getting the type of 'self' (line 520)
        self_20711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 520, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 520)
        write_20712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 520, 8), self_20711, 'write')
        # Calling write(args, kwargs) (line 520)
        write_call_result_20716 = invoke(stypy.reporting.localization.Localization(__file__, 520, 8), write_20712, *[attr_20714], **kwargs_20715)
        
        
        # ################# End of 'visit_Attribute(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Attribute' in the type store
        # Getting the type of 'stypy_return_type' (line 512)
        stypy_return_type_20717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 512, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20717)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Attribute'
        return stypy_return_type_20717


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
        t_20720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 19), 't', False)
        # Obtaining the member 'func' of a type (line 523)
        func_20721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 19), t_20720, 'func')
        # Processing the call keyword arguments (line 523)
        kwargs_20722 = {}
        # Getting the type of 'self' (line 523)
        self_20718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 523, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 523)
        visit_20719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 523, 8), self_20718, 'visit')
        # Calling visit(args, kwargs) (line 523)
        visit_call_result_20723 = invoke(stypy.reporting.localization.Localization(__file__, 523, 8), visit_20719, *[func_20721], **kwargs_20722)
        
        
        # Call to write(...): (line 524)
        # Processing the call arguments (line 524)
        str_20726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 524, 19), 'str', '(')
        # Processing the call keyword arguments (line 524)
        kwargs_20727 = {}
        # Getting the type of 'self' (line 524)
        self_20724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 524, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 524)
        write_20725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 524, 8), self_20724, 'write')
        # Calling write(args, kwargs) (line 524)
        write_call_result_20728 = invoke(stypy.reporting.localization.Localization(__file__, 524, 8), write_20725, *[str_20726], **kwargs_20727)
        
        
        # Assigning a Name to a Name (line 525):
        
        # Assigning a Name to a Name (line 525):
        # Getting the type of 'False' (line 525)
        False_20729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 525, 16), 'False')
        # Assigning a type to the variable 'comma' (line 525)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 525, 8), 'comma', False_20729)
        
        # Getting the type of 't' (line 526)
        t_20730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 526, 17), 't')
        # Obtaining the member 'args' of a type (line 526)
        args_20731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 526, 17), t_20730, 'args')
        # Assigning a type to the variable 'args_20731' (line 526)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'args_20731', args_20731)
        # Testing if the for loop is going to be iterated (line 526)
        # Testing the type of a for loop iterable (line 526)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 526, 8), args_20731)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 526, 8), args_20731):
            # Getting the type of the for loop variable (line 526)
            for_loop_var_20732 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 526, 8), args_20731)
            # Assigning a type to the variable 'e' (line 526)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 526, 8), 'e', for_loop_var_20732)
            # SSA begins for a for statement (line 526)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'comma' (line 527)
            comma_20733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 15), 'comma')
            # Testing if the type of an if condition is none (line 527)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 527, 12), comma_20733):
                
                # Assigning a Name to a Name (line 530):
                
                # Assigning a Name to a Name (line 530):
                # Getting the type of 'True' (line 530)
                True_20740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 24), 'True')
                # Assigning a type to the variable 'comma' (line 530)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 16), 'comma', True_20740)
            else:
                
                # Testing the type of an if condition (line 527)
                if_condition_20734 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 527, 12), comma_20733)
                # Assigning a type to the variable 'if_condition_20734' (line 527)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 12), 'if_condition_20734', if_condition_20734)
                # SSA begins for if statement (line 527)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 528)
                # Processing the call arguments (line 528)
                str_20737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 528, 27), 'str', ', ')
                # Processing the call keyword arguments (line 528)
                kwargs_20738 = {}
                # Getting the type of 'self' (line 528)
                self_20735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 528, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 528)
                write_20736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 528, 16), self_20735, 'write')
                # Calling write(args, kwargs) (line 528)
                write_call_result_20739 = invoke(stypy.reporting.localization.Localization(__file__, 528, 16), write_20736, *[str_20737], **kwargs_20738)
                
                # SSA branch for the else part of an if statement (line 527)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 530):
                
                # Assigning a Name to a Name (line 530):
                # Getting the type of 'True' (line 530)
                True_20740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 530, 24), 'True')
                # Assigning a type to the variable 'comma' (line 530)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 530, 16), 'comma', True_20740)
                # SSA join for if statement (line 527)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to visit(...): (line 531)
            # Processing the call arguments (line 531)
            # Getting the type of 'e' (line 531)
            e_20743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 23), 'e', False)
            # Processing the call keyword arguments (line 531)
            kwargs_20744 = {}
            # Getting the type of 'self' (line 531)
            self_20741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 531, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 531)
            visit_20742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 531, 12), self_20741, 'visit')
            # Calling visit(args, kwargs) (line 531)
            visit_call_result_20745 = invoke(stypy.reporting.localization.Localization(__file__, 531, 12), visit_20742, *[e_20743], **kwargs_20744)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 't' (line 532)
        t_20746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 532, 17), 't')
        # Obtaining the member 'keywords' of a type (line 532)
        keywords_20747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 532, 17), t_20746, 'keywords')
        # Assigning a type to the variable 'keywords_20747' (line 532)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'keywords_20747', keywords_20747)
        # Testing if the for loop is going to be iterated (line 532)
        # Testing the type of a for loop iterable (line 532)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 532, 8), keywords_20747)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 532, 8), keywords_20747):
            # Getting the type of the for loop variable (line 532)
            for_loop_var_20748 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 532, 8), keywords_20747)
            # Assigning a type to the variable 'e' (line 532)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 532, 8), 'e', for_loop_var_20748)
            # SSA begins for a for statement (line 532)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'comma' (line 533)
            comma_20749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 533, 15), 'comma')
            # Testing if the type of an if condition is none (line 533)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 533, 12), comma_20749):
                
                # Assigning a Name to a Name (line 536):
                
                # Assigning a Name to a Name (line 536):
                # Getting the type of 'True' (line 536)
                True_20756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 24), 'True')
                # Assigning a type to the variable 'comma' (line 536)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'comma', True_20756)
            else:
                
                # Testing the type of an if condition (line 533)
                if_condition_20750 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 533, 12), comma_20749)
                # Assigning a type to the variable 'if_condition_20750' (line 533)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 533, 12), 'if_condition_20750', if_condition_20750)
                # SSA begins for if statement (line 533)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 534)
                # Processing the call arguments (line 534)
                str_20753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 534, 27), 'str', ', ')
                # Processing the call keyword arguments (line 534)
                kwargs_20754 = {}
                # Getting the type of 'self' (line 534)
                self_20751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 534, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 534)
                write_20752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 534, 16), self_20751, 'write')
                # Calling write(args, kwargs) (line 534)
                write_call_result_20755 = invoke(stypy.reporting.localization.Localization(__file__, 534, 16), write_20752, *[str_20753], **kwargs_20754)
                
                # SSA branch for the else part of an if statement (line 533)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 536):
                
                # Assigning a Name to a Name (line 536):
                # Getting the type of 'True' (line 536)
                True_20756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 536, 24), 'True')
                # Assigning a type to the variable 'comma' (line 536)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 536, 16), 'comma', True_20756)
                # SSA join for if statement (line 533)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to visit(...): (line 537)
            # Processing the call arguments (line 537)
            # Getting the type of 'e' (line 537)
            e_20759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 23), 'e', False)
            # Processing the call keyword arguments (line 537)
            kwargs_20760 = {}
            # Getting the type of 'self' (line 537)
            self_20757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 537, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 537)
            visit_20758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 537, 12), self_20757, 'visit')
            # Calling visit(args, kwargs) (line 537)
            visit_call_result_20761 = invoke(stypy.reporting.localization.Localization(__file__, 537, 12), visit_20758, *[e_20759], **kwargs_20760)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 't' (line 538)
        t_20762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 11), 't')
        # Obtaining the member 'starargs' of a type (line 538)
        starargs_20763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 538, 11), t_20762, 'starargs')
        # Testing if the type of an if condition is none (line 538)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 538, 8), starargs_20763):
            pass
        else:
            
            # Testing the type of an if condition (line 538)
            if_condition_20764 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 538, 8), starargs_20763)
            # Assigning a type to the variable 'if_condition_20764' (line 538)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 8), 'if_condition_20764', if_condition_20764)
            # SSA begins for if statement (line 538)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'comma' (line 539)
            comma_20765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 15), 'comma')
            # Testing if the type of an if condition is none (line 539)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 539, 12), comma_20765):
                
                # Assigning a Name to a Name (line 542):
                
                # Assigning a Name to a Name (line 542):
                # Getting the type of 'True' (line 542)
                True_20772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 24), 'True')
                # Assigning a type to the variable 'comma' (line 542)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'comma', True_20772)
            else:
                
                # Testing the type of an if condition (line 539)
                if_condition_20766 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 539, 12), comma_20765)
                # Assigning a type to the variable 'if_condition_20766' (line 539)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 12), 'if_condition_20766', if_condition_20766)
                # SSA begins for if statement (line 539)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 540)
                # Processing the call arguments (line 540)
                str_20769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 27), 'str', ', ')
                # Processing the call keyword arguments (line 540)
                kwargs_20770 = {}
                # Getting the type of 'self' (line 540)
                self_20767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 540)
                write_20768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 16), self_20767, 'write')
                # Calling write(args, kwargs) (line 540)
                write_call_result_20771 = invoke(stypy.reporting.localization.Localization(__file__, 540, 16), write_20768, *[str_20769], **kwargs_20770)
                
                # SSA branch for the else part of an if statement (line 539)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 542):
                
                # Assigning a Name to a Name (line 542):
                # Getting the type of 'True' (line 542)
                True_20772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 24), 'True')
                # Assigning a type to the variable 'comma' (line 542)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 542, 16), 'comma', True_20772)
                # SSA join for if statement (line 539)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 543)
            # Processing the call arguments (line 543)
            str_20775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 543, 23), 'str', '*')
            # Processing the call keyword arguments (line 543)
            kwargs_20776 = {}
            # Getting the type of 'self' (line 543)
            self_20773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 543, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 543)
            write_20774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 543, 12), self_20773, 'write')
            # Calling write(args, kwargs) (line 543)
            write_call_result_20777 = invoke(stypy.reporting.localization.Localization(__file__, 543, 12), write_20774, *[str_20775], **kwargs_20776)
            
            
            # Call to visit(...): (line 544)
            # Processing the call arguments (line 544)
            # Getting the type of 't' (line 544)
            t_20780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 23), 't', False)
            # Obtaining the member 'starargs' of a type (line 544)
            starargs_20781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 23), t_20780, 'starargs')
            # Processing the call keyword arguments (line 544)
            kwargs_20782 = {}
            # Getting the type of 'self' (line 544)
            self_20778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 544)
            visit_20779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 544, 12), self_20778, 'visit')
            # Calling visit(args, kwargs) (line 544)
            visit_call_result_20783 = invoke(stypy.reporting.localization.Localization(__file__, 544, 12), visit_20779, *[starargs_20781], **kwargs_20782)
            
            # SSA join for if statement (line 538)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 545)
        t_20784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 545, 11), 't')
        # Obtaining the member 'kwargs' of a type (line 545)
        kwargs_20785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 545, 11), t_20784, 'kwargs')
        # Testing if the type of an if condition is none (line 545)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 545, 8), kwargs_20785):
            pass
        else:
            
            # Testing the type of an if condition (line 545)
            if_condition_20786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 545, 8), kwargs_20785)
            # Assigning a type to the variable 'if_condition_20786' (line 545)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 545, 8), 'if_condition_20786', if_condition_20786)
            # SSA begins for if statement (line 545)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'comma' (line 546)
            comma_20787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 546, 15), 'comma')
            # Testing if the type of an if condition is none (line 546)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 546, 12), comma_20787):
                
                # Assigning a Name to a Name (line 549):
                
                # Assigning a Name to a Name (line 549):
                # Getting the type of 'True' (line 549)
                True_20794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 24), 'True')
                # Assigning a type to the variable 'comma' (line 549)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'comma', True_20794)
            else:
                
                # Testing the type of an if condition (line 546)
                if_condition_20788 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 546, 12), comma_20787)
                # Assigning a type to the variable 'if_condition_20788' (line 546)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 546, 12), 'if_condition_20788', if_condition_20788)
                # SSA begins for if statement (line 546)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 547)
                # Processing the call arguments (line 547)
                str_20791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 547, 27), 'str', ', ')
                # Processing the call keyword arguments (line 547)
                kwargs_20792 = {}
                # Getting the type of 'self' (line 547)
                self_20789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 547)
                write_20790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 547, 16), self_20789, 'write')
                # Calling write(args, kwargs) (line 547)
                write_call_result_20793 = invoke(stypy.reporting.localization.Localization(__file__, 547, 16), write_20790, *[str_20791], **kwargs_20792)
                
                # SSA branch for the else part of an if statement (line 546)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 549):
                
                # Assigning a Name to a Name (line 549):
                # Getting the type of 'True' (line 549)
                True_20794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 549, 24), 'True')
                # Assigning a type to the variable 'comma' (line 549)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 549, 16), 'comma', True_20794)
                # SSA join for if statement (line 546)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 550)
            # Processing the call arguments (line 550)
            str_20797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 550, 23), 'str', '**')
            # Processing the call keyword arguments (line 550)
            kwargs_20798 = {}
            # Getting the type of 'self' (line 550)
            self_20795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 550, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 550)
            write_20796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 550, 12), self_20795, 'write')
            # Calling write(args, kwargs) (line 550)
            write_call_result_20799 = invoke(stypy.reporting.localization.Localization(__file__, 550, 12), write_20796, *[str_20797], **kwargs_20798)
            
            
            # Call to visit(...): (line 551)
            # Processing the call arguments (line 551)
            # Getting the type of 't' (line 551)
            t_20802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 23), 't', False)
            # Obtaining the member 'kwargs' of a type (line 551)
            kwargs_20803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 23), t_20802, 'kwargs')
            # Processing the call keyword arguments (line 551)
            kwargs_20804 = {}
            # Getting the type of 'self' (line 551)
            self_20800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 551, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 551)
            visit_20801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 551, 12), self_20800, 'visit')
            # Calling visit(args, kwargs) (line 551)
            visit_call_result_20805 = invoke(stypy.reporting.localization.Localization(__file__, 551, 12), visit_20801, *[kwargs_20803], **kwargs_20804)
            
            # SSA join for if statement (line 545)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 552)
        # Processing the call arguments (line 552)
        str_20808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 552, 19), 'str', ')')
        # Processing the call keyword arguments (line 552)
        kwargs_20809 = {}
        # Getting the type of 'self' (line 552)
        self_20806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 552, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 552)
        write_20807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 552, 8), self_20806, 'write')
        # Calling write(args, kwargs) (line 552)
        write_call_result_20810 = invoke(stypy.reporting.localization.Localization(__file__, 552, 8), write_20807, *[str_20808], **kwargs_20809)
        
        
        # ################# End of 'visit_Call(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Call' in the type store
        # Getting the type of 'stypy_return_type' (line 522)
        stypy_return_type_20811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 522, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20811)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Call'
        return stypy_return_type_20811


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
        t_20814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 19), 't', False)
        # Obtaining the member 'value' of a type (line 555)
        value_20815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 19), t_20814, 'value')
        # Processing the call keyword arguments (line 555)
        kwargs_20816 = {}
        # Getting the type of 'self' (line 555)
        self_20812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 555, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 555)
        visit_20813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 555, 8), self_20812, 'visit')
        # Calling visit(args, kwargs) (line 555)
        visit_call_result_20817 = invoke(stypy.reporting.localization.Localization(__file__, 555, 8), visit_20813, *[value_20815], **kwargs_20816)
        
        
        # Call to write(...): (line 556)
        # Processing the call arguments (line 556)
        str_20820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, 19), 'str', '[')
        # Processing the call keyword arguments (line 556)
        kwargs_20821 = {}
        # Getting the type of 'self' (line 556)
        self_20818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 556, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 556)
        write_20819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 556, 8), self_20818, 'write')
        # Calling write(args, kwargs) (line 556)
        write_call_result_20822 = invoke(stypy.reporting.localization.Localization(__file__, 556, 8), write_20819, *[str_20820], **kwargs_20821)
        
        
        # Call to visit(...): (line 557)
        # Processing the call arguments (line 557)
        # Getting the type of 't' (line 557)
        t_20825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 19), 't', False)
        # Obtaining the member 'slice' of a type (line 557)
        slice_20826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 19), t_20825, 'slice')
        # Processing the call keyword arguments (line 557)
        kwargs_20827 = {}
        # Getting the type of 'self' (line 557)
        self_20823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 557)
        visit_20824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 557, 8), self_20823, 'visit')
        # Calling visit(args, kwargs) (line 557)
        visit_call_result_20828 = invoke(stypy.reporting.localization.Localization(__file__, 557, 8), visit_20824, *[slice_20826], **kwargs_20827)
        
        
        # Call to write(...): (line 558)
        # Processing the call arguments (line 558)
        str_20831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 558, 19), 'str', ']')
        # Processing the call keyword arguments (line 558)
        kwargs_20832 = {}
        # Getting the type of 'self' (line 558)
        self_20829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 558)
        write_20830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 8), self_20829, 'write')
        # Calling write(args, kwargs) (line 558)
        write_call_result_20833 = invoke(stypy.reporting.localization.Localization(__file__, 558, 8), write_20830, *[str_20831], **kwargs_20832)
        
        
        # ################# End of 'visit_Subscript(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Subscript' in the type store
        # Getting the type of 'stypy_return_type' (line 554)
        stypy_return_type_20834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 554, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20834)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Subscript'
        return stypy_return_type_20834


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
        str_20837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 562, 19), 'str', '...')
        # Processing the call keyword arguments (line 562)
        kwargs_20838 = {}
        # Getting the type of 'self' (line 562)
        self_20835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 562, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 562)
        write_20836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 562, 8), self_20835, 'write')
        # Calling write(args, kwargs) (line 562)
        write_call_result_20839 = invoke(stypy.reporting.localization.Localization(__file__, 562, 8), write_20836, *[str_20837], **kwargs_20838)
        
        
        # ################# End of 'visit_Ellipsis(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Ellipsis' in the type store
        # Getting the type of 'stypy_return_type' (line 561)
        stypy_return_type_20840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20840)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Ellipsis'
        return stypy_return_type_20840


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
        t_20843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 19), 't', False)
        # Obtaining the member 'value' of a type (line 565)
        value_20844 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 19), t_20843, 'value')
        # Processing the call keyword arguments (line 565)
        kwargs_20845 = {}
        # Getting the type of 'self' (line 565)
        self_20841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 565, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 565)
        visit_20842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 565, 8), self_20841, 'visit')
        # Calling visit(args, kwargs) (line 565)
        visit_call_result_20846 = invoke(stypy.reporting.localization.Localization(__file__, 565, 8), visit_20842, *[value_20844], **kwargs_20845)
        
        
        # ################# End of 'visit_Index(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Index' in the type store
        # Getting the type of 'stypy_return_type' (line 564)
        stypy_return_type_20847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 564, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20847)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Index'
        return stypy_return_type_20847


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
        t_20848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 568, 11), 't')
        # Obtaining the member 'lower' of a type (line 568)
        lower_20849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 568, 11), t_20848, 'lower')
        # Testing if the type of an if condition is none (line 568)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 568, 8), lower_20849):
            pass
        else:
            
            # Testing the type of an if condition (line 568)
            if_condition_20850 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 568, 8), lower_20849)
            # Assigning a type to the variable 'if_condition_20850' (line 568)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 568, 8), 'if_condition_20850', if_condition_20850)
            # SSA begins for if statement (line 568)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to visit(...): (line 569)
            # Processing the call arguments (line 569)
            # Getting the type of 't' (line 569)
            t_20853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 23), 't', False)
            # Obtaining the member 'lower' of a type (line 569)
            lower_20854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 23), t_20853, 'lower')
            # Processing the call keyword arguments (line 569)
            kwargs_20855 = {}
            # Getting the type of 'self' (line 569)
            self_20851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 569, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 569)
            visit_20852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 569, 12), self_20851, 'visit')
            # Calling visit(args, kwargs) (line 569)
            visit_call_result_20856 = invoke(stypy.reporting.localization.Localization(__file__, 569, 12), visit_20852, *[lower_20854], **kwargs_20855)
            
            # SSA join for if statement (line 568)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to write(...): (line 570)
        # Processing the call arguments (line 570)
        str_20859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 570, 19), 'str', ':')
        # Processing the call keyword arguments (line 570)
        kwargs_20860 = {}
        # Getting the type of 'self' (line 570)
        self_20857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 570, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 570)
        write_20858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 570, 8), self_20857, 'write')
        # Calling write(args, kwargs) (line 570)
        write_call_result_20861 = invoke(stypy.reporting.localization.Localization(__file__, 570, 8), write_20858, *[str_20859], **kwargs_20860)
        
        # Getting the type of 't' (line 571)
        t_20862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 571, 11), 't')
        # Obtaining the member 'upper' of a type (line 571)
        upper_20863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 571, 11), t_20862, 'upper')
        # Testing if the type of an if condition is none (line 571)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 571, 8), upper_20863):
            pass
        else:
            
            # Testing the type of an if condition (line 571)
            if_condition_20864 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 571, 8), upper_20863)
            # Assigning a type to the variable 'if_condition_20864' (line 571)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 571, 8), 'if_condition_20864', if_condition_20864)
            # SSA begins for if statement (line 571)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to visit(...): (line 572)
            # Processing the call arguments (line 572)
            # Getting the type of 't' (line 572)
            t_20867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 23), 't', False)
            # Obtaining the member 'upper' of a type (line 572)
            upper_20868 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 23), t_20867, 'upper')
            # Processing the call keyword arguments (line 572)
            kwargs_20869 = {}
            # Getting the type of 'self' (line 572)
            self_20865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 572, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 572)
            visit_20866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 572, 12), self_20865, 'visit')
            # Calling visit(args, kwargs) (line 572)
            visit_call_result_20870 = invoke(stypy.reporting.localization.Localization(__file__, 572, 12), visit_20866, *[upper_20868], **kwargs_20869)
            
            # SSA join for if statement (line 571)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 573)
        t_20871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 573, 11), 't')
        # Obtaining the member 'step' of a type (line 573)
        step_20872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 573, 11), t_20871, 'step')
        # Testing if the type of an if condition is none (line 573)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 573, 8), step_20872):
            pass
        else:
            
            # Testing the type of an if condition (line 573)
            if_condition_20873 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 573, 8), step_20872)
            # Assigning a type to the variable 'if_condition_20873' (line 573)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 573, 8), 'if_condition_20873', if_condition_20873)
            # SSA begins for if statement (line 573)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 574)
            # Processing the call arguments (line 574)
            str_20876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 574, 23), 'str', ':')
            # Processing the call keyword arguments (line 574)
            kwargs_20877 = {}
            # Getting the type of 'self' (line 574)
            self_20874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 574, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 574)
            write_20875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 574, 12), self_20874, 'write')
            # Calling write(args, kwargs) (line 574)
            write_call_result_20878 = invoke(stypy.reporting.localization.Localization(__file__, 574, 12), write_20875, *[str_20876], **kwargs_20877)
            
            
            # Call to visit(...): (line 575)
            # Processing the call arguments (line 575)
            # Getting the type of 't' (line 575)
            t_20881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 23), 't', False)
            # Obtaining the member 'step' of a type (line 575)
            step_20882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 23), t_20881, 'step')
            # Processing the call keyword arguments (line 575)
            kwargs_20883 = {}
            # Getting the type of 'self' (line 575)
            self_20879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 575, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 575)
            visit_20880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 575, 12), self_20879, 'visit')
            # Calling visit(args, kwargs) (line 575)
            visit_call_result_20884 = invoke(stypy.reporting.localization.Localization(__file__, 575, 12), visit_20880, *[step_20882], **kwargs_20883)
            
            # SSA join for if statement (line 573)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_Slice(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Slice' in the type store
        # Getting the type of 'stypy_return_type' (line 567)
        stypy_return_type_20885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 567, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20885)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Slice'
        return stypy_return_type_20885


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
        def _stypy_temp_lambda_38(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_38'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_38', 578, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_38.stypy_localization = localization
            _stypy_temp_lambda_38.stypy_type_of_self = None
            _stypy_temp_lambda_38.stypy_type_store = module_type_store
            _stypy_temp_lambda_38.stypy_function_name = '_stypy_temp_lambda_38'
            _stypy_temp_lambda_38.stypy_param_names_list = []
            _stypy_temp_lambda_38.stypy_varargs_param_name = None
            _stypy_temp_lambda_38.stypy_kwargs_param_name = None
            _stypy_temp_lambda_38.stypy_call_defaults = defaults
            _stypy_temp_lambda_38.stypy_call_varargs = varargs
            _stypy_temp_lambda_38.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_38', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_38', [], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to write(...): (line 578)
            # Processing the call arguments (line 578)
            str_20889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 578, 38), 'str', ', ')
            # Processing the call keyword arguments (line 578)
            kwargs_20890 = {}
            # Getting the type of 'self' (line 578)
            self_20887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 27), 'self', False)
            # Obtaining the member 'write' of a type (line 578)
            write_20888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 27), self_20887, 'write')
            # Calling write(args, kwargs) (line 578)
            write_call_result_20891 = invoke(stypy.reporting.localization.Localization(__file__, 578, 27), write_20888, *[str_20889], **kwargs_20890)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 578)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 19), 'stypy_return_type', write_call_result_20891)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_38' in the type store
            # Getting the type of 'stypy_return_type' (line 578)
            stypy_return_type_20892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_20892)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_38'
            return stypy_return_type_20892

        # Assigning a type to the variable '_stypy_temp_lambda_38' (line 578)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 578, 19), '_stypy_temp_lambda_38', _stypy_temp_lambda_38)
        # Getting the type of '_stypy_temp_lambda_38' (line 578)
        _stypy_temp_lambda_38_20893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 19), '_stypy_temp_lambda_38')
        # Getting the type of 'self' (line 578)
        self_20894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 45), 'self', False)
        # Obtaining the member 'visit' of a type (line 578)
        visit_20895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 45), self_20894, 'visit')
        # Getting the type of 't' (line 578)
        t_20896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 57), 't', False)
        # Obtaining the member 'dims' of a type (line 578)
        dims_20897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 578, 57), t_20896, 'dims')
        # Processing the call keyword arguments (line 578)
        kwargs_20898 = {}
        # Getting the type of 'interleave' (line 578)
        interleave_20886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 578, 8), 'interleave', False)
        # Calling interleave(args, kwargs) (line 578)
        interleave_call_result_20899 = invoke(stypy.reporting.localization.Localization(__file__, 578, 8), interleave_20886, *[_stypy_temp_lambda_38_20893, visit_20895, dims_20897], **kwargs_20898)
        
        
        # ################# End of 'visit_ExtSlice(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_ExtSlice' in the type store
        # Getting the type of 'stypy_return_type' (line 577)
        stypy_return_type_20900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 577, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20900)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_ExtSlice'
        return stypy_return_type_20900


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
        True_20901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 582, 16), 'True')
        # Assigning a type to the variable 'first' (line 582)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 582, 8), 'first', True_20901)
        
        # Assigning a BinOp to a Name (line 584):
        
        # Assigning a BinOp to a Name (line 584):
        
        # Obtaining an instance of the builtin type 'list' (line 584)
        list_20902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 584, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 584)
        # Adding element type (line 584)
        # Getting the type of 'None' (line 584)
        None_20903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 20), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 584, 19), list_20902, None_20903)
        
        
        # Call to len(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 't' (line 584)
        t_20905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 33), 't', False)
        # Obtaining the member 'args' of a type (line 584)
        args_20906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 33), t_20905, 'args')
        # Processing the call keyword arguments (line 584)
        kwargs_20907 = {}
        # Getting the type of 'len' (line 584)
        len_20904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 29), 'len', False)
        # Calling len(args, kwargs) (line 584)
        len_call_result_20908 = invoke(stypy.reporting.localization.Localization(__file__, 584, 29), len_20904, *[args_20906], **kwargs_20907)
        
        
        # Call to len(...): (line 584)
        # Processing the call arguments (line 584)
        # Getting the type of 't' (line 584)
        t_20910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 47), 't', False)
        # Obtaining the member 'defaults' of a type (line 584)
        defaults_20911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 47), t_20910, 'defaults')
        # Processing the call keyword arguments (line 584)
        kwargs_20912 = {}
        # Getting the type of 'len' (line 584)
        len_20909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 43), 'len', False)
        # Calling len(args, kwargs) (line 584)
        len_call_result_20913 = invoke(stypy.reporting.localization.Localization(__file__, 584, 43), len_20909, *[defaults_20911], **kwargs_20912)
        
        # Applying the binary operator '-' (line 584)
        result_sub_20914 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 29), '-', len_call_result_20908, len_call_result_20913)
        
        # Applying the binary operator '*' (line 584)
        result_mul_20915 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 19), '*', list_20902, result_sub_20914)
        
        # Getting the type of 't' (line 584)
        t_20916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 584, 62), 't')
        # Obtaining the member 'defaults' of a type (line 584)
        defaults_20917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 584, 62), t_20916, 'defaults')
        # Applying the binary operator '+' (line 584)
        result_add_20918 = python_operator(stypy.reporting.localization.Localization(__file__, 584, 19), '+', result_mul_20915, defaults_20917)
        
        # Assigning a type to the variable 'defaults' (line 584)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 584, 8), 'defaults', result_add_20918)
        
        
        # Call to zip(...): (line 585)
        # Processing the call arguments (line 585)
        # Getting the type of 't' (line 585)
        t_20920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 24), 't', False)
        # Obtaining the member 'args' of a type (line 585)
        args_20921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 585, 24), t_20920, 'args')
        # Getting the type of 'defaults' (line 585)
        defaults_20922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 32), 'defaults', False)
        # Processing the call keyword arguments (line 585)
        kwargs_20923 = {}
        # Getting the type of 'zip' (line 585)
        zip_20919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 585, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 585)
        zip_call_result_20924 = invoke(stypy.reporting.localization.Localization(__file__, 585, 20), zip_20919, *[args_20921, defaults_20922], **kwargs_20923)
        
        # Assigning a type to the variable 'zip_call_result_20924' (line 585)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'zip_call_result_20924', zip_call_result_20924)
        # Testing if the for loop is going to be iterated (line 585)
        # Testing the type of a for loop iterable (line 585)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 585, 8), zip_call_result_20924)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 585, 8), zip_call_result_20924):
            # Getting the type of the for loop variable (line 585)
            for_loop_var_20925 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 585, 8), zip_call_result_20924)
            # Assigning a type to the variable 'a' (line 585)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'a', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 8), for_loop_var_20925, 2, 0))
            # Assigning a type to the variable 'd' (line 585)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 585, 8), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 585, 8), for_loop_var_20925, 2, 1))
            # SSA begins for a for statement (line 585)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'first' (line 586)
            first_20926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 586, 15), 'first')
            # Testing if the type of an if condition is none (line 586)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 586, 12), first_20926):
                
                # Call to write(...): (line 589)
                # Processing the call arguments (line 589)
                str_20931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 27), 'str', ', ')
                # Processing the call keyword arguments (line 589)
                kwargs_20932 = {}
                # Getting the type of 'self' (line 589)
                self_20929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 589)
                write_20930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 16), self_20929, 'write')
                # Calling write(args, kwargs) (line 589)
                write_call_result_20933 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), write_20930, *[str_20931], **kwargs_20932)
                
            else:
                
                # Testing the type of an if condition (line 586)
                if_condition_20927 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 586, 12), first_20926)
                # Assigning a type to the variable 'if_condition_20927' (line 586)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 586, 12), 'if_condition_20927', if_condition_20927)
                # SSA begins for if statement (line 586)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 587):
                
                # Assigning a Name to a Name (line 587):
                # Getting the type of 'False' (line 587)
                False_20928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 587, 24), 'False')
                # Assigning a type to the variable 'first' (line 587)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 587, 16), 'first', False_20928)
                # SSA branch for the else part of an if statement (line 586)
                module_type_store.open_ssa_branch('else')
                
                # Call to write(...): (line 589)
                # Processing the call arguments (line 589)
                str_20931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 589, 27), 'str', ', ')
                # Processing the call keyword arguments (line 589)
                kwargs_20932 = {}
                # Getting the type of 'self' (line 589)
                self_20929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 589, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 589)
                write_20930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 589, 16), self_20929, 'write')
                # Calling write(args, kwargs) (line 589)
                write_call_result_20933 = invoke(stypy.reporting.localization.Localization(__file__, 589, 16), write_20930, *[str_20931], **kwargs_20932)
                
                # SSA join for if statement (line 586)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Obtaining an instance of the builtin type 'tuple' (line 590)
            tuple_20934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 590, 12), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 590)
            # Adding element type (line 590)
            
            # Call to visit(...): (line 590)
            # Processing the call arguments (line 590)
            # Getting the type of 'a' (line 590)
            a_20937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 23), 'a', False)
            # Processing the call keyword arguments (line 590)
            kwargs_20938 = {}
            # Getting the type of 'self' (line 590)
            self_20935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 590, 12), 'self', False)
            # Obtaining the member 'visit' of a type (line 590)
            visit_20936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 590, 12), self_20935, 'visit')
            # Calling visit(args, kwargs) (line 590)
            visit_call_result_20939 = invoke(stypy.reporting.localization.Localization(__file__, 590, 12), visit_20936, *[a_20937], **kwargs_20938)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 590, 12), tuple_20934, visit_call_result_20939)
            
            # Getting the type of 'd' (line 591)
            d_20940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 591, 15), 'd')
            # Testing if the type of an if condition is none (line 591)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 591, 12), d_20940):
                pass
            else:
                
                # Testing the type of an if condition (line 591)
                if_condition_20941 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 591, 12), d_20940)
                # Assigning a type to the variable 'if_condition_20941' (line 591)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 591, 12), 'if_condition_20941', if_condition_20941)
                # SSA begins for if statement (line 591)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to write(...): (line 592)
                # Processing the call arguments (line 592)
                str_20944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 592, 27), 'str', '=')
                # Processing the call keyword arguments (line 592)
                kwargs_20945 = {}
                # Getting the type of 'self' (line 592)
                self_20942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 592, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 592)
                write_20943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 592, 16), self_20942, 'write')
                # Calling write(args, kwargs) (line 592)
                write_call_result_20946 = invoke(stypy.reporting.localization.Localization(__file__, 592, 16), write_20943, *[str_20944], **kwargs_20945)
                
                
                # Call to visit(...): (line 593)
                # Processing the call arguments (line 593)
                # Getting the type of 'd' (line 593)
                d_20949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 27), 'd', False)
                # Processing the call keyword arguments (line 593)
                kwargs_20950 = {}
                # Getting the type of 'self' (line 593)
                self_20947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 593, 16), 'self', False)
                # Obtaining the member 'visit' of a type (line 593)
                visit_20948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 593, 16), self_20947, 'visit')
                # Calling visit(args, kwargs) (line 593)
                visit_call_result_20951 = invoke(stypy.reporting.localization.Localization(__file__, 593, 16), visit_20948, *[d_20949], **kwargs_20950)
                
                # SSA join for if statement (line 591)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 't' (line 596)
        t_20952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 596, 11), 't')
        # Obtaining the member 'vararg' of a type (line 596)
        vararg_20953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 596, 11), t_20952, 'vararg')
        # Testing if the type of an if condition is none (line 596)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 596, 8), vararg_20953):
            pass
        else:
            
            # Testing the type of an if condition (line 596)
            if_condition_20954 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 596, 8), vararg_20953)
            # Assigning a type to the variable 'if_condition_20954' (line 596)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 596, 8), 'if_condition_20954', if_condition_20954)
            # SSA begins for if statement (line 596)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'first' (line 597)
            first_20955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 597, 15), 'first')
            # Testing if the type of an if condition is none (line 597)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 597, 12), first_20955):
                
                # Call to write(...): (line 600)
                # Processing the call arguments (line 600)
                str_20960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 27), 'str', ', ')
                # Processing the call keyword arguments (line 600)
                kwargs_20961 = {}
                # Getting the type of 'self' (line 600)
                self_20958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 600)
                write_20959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 16), self_20958, 'write')
                # Calling write(args, kwargs) (line 600)
                write_call_result_20962 = invoke(stypy.reporting.localization.Localization(__file__, 600, 16), write_20959, *[str_20960], **kwargs_20961)
                
            else:
                
                # Testing the type of an if condition (line 597)
                if_condition_20956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 597, 12), first_20955)
                # Assigning a type to the variable 'if_condition_20956' (line 597)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 597, 12), 'if_condition_20956', if_condition_20956)
                # SSA begins for if statement (line 597)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 598):
                
                # Assigning a Name to a Name (line 598):
                # Getting the type of 'False' (line 598)
                False_20957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 598, 24), 'False')
                # Assigning a type to the variable 'first' (line 598)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 598, 16), 'first', False_20957)
                # SSA branch for the else part of an if statement (line 597)
                module_type_store.open_ssa_branch('else')
                
                # Call to write(...): (line 600)
                # Processing the call arguments (line 600)
                str_20960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 600, 27), 'str', ', ')
                # Processing the call keyword arguments (line 600)
                kwargs_20961 = {}
                # Getting the type of 'self' (line 600)
                self_20958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 600, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 600)
                write_20959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 600, 16), self_20958, 'write')
                # Calling write(args, kwargs) (line 600)
                write_call_result_20962 = invoke(stypy.reporting.localization.Localization(__file__, 600, 16), write_20959, *[str_20960], **kwargs_20961)
                
                # SSA join for if statement (line 597)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 601)
            # Processing the call arguments (line 601)
            str_20965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 601, 23), 'str', '*')
            # Processing the call keyword arguments (line 601)
            kwargs_20966 = {}
            # Getting the type of 'self' (line 601)
            self_20963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 601, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 601)
            write_20964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 601, 12), self_20963, 'write')
            # Calling write(args, kwargs) (line 601)
            write_call_result_20967 = invoke(stypy.reporting.localization.Localization(__file__, 601, 12), write_20964, *[str_20965], **kwargs_20966)
            
            
            # Call to write(...): (line 602)
            # Processing the call arguments (line 602)
            # Getting the type of 't' (line 602)
            t_20970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 23), 't', False)
            # Obtaining the member 'vararg' of a type (line 602)
            vararg_20971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 23), t_20970, 'vararg')
            # Processing the call keyword arguments (line 602)
            kwargs_20972 = {}
            # Getting the type of 'self' (line 602)
            self_20968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 602, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 602)
            write_20969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 602, 12), self_20968, 'write')
            # Calling write(args, kwargs) (line 602)
            write_call_result_20973 = invoke(stypy.reporting.localization.Localization(__file__, 602, 12), write_20969, *[vararg_20971], **kwargs_20972)
            
            # SSA join for if statement (line 596)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 605)
        t_20974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 605, 11), 't')
        # Obtaining the member 'kwarg' of a type (line 605)
        kwarg_20975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 605, 11), t_20974, 'kwarg')
        # Testing if the type of an if condition is none (line 605)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 605, 8), kwarg_20975):
            pass
        else:
            
            # Testing the type of an if condition (line 605)
            if_condition_20976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 605, 8), kwarg_20975)
            # Assigning a type to the variable 'if_condition_20976' (line 605)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 605, 8), 'if_condition_20976', if_condition_20976)
            # SSA begins for if statement (line 605)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'first' (line 606)
            first_20977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 606, 15), 'first')
            # Testing if the type of an if condition is none (line 606)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 606, 12), first_20977):
                
                # Call to write(...): (line 609)
                # Processing the call arguments (line 609)
                str_20982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 27), 'str', ', ')
                # Processing the call keyword arguments (line 609)
                kwargs_20983 = {}
                # Getting the type of 'self' (line 609)
                self_20980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 609)
                write_20981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 16), self_20980, 'write')
                # Calling write(args, kwargs) (line 609)
                write_call_result_20984 = invoke(stypy.reporting.localization.Localization(__file__, 609, 16), write_20981, *[str_20982], **kwargs_20983)
                
            else:
                
                # Testing the type of an if condition (line 606)
                if_condition_20978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 606, 12), first_20977)
                # Assigning a type to the variable 'if_condition_20978' (line 606)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 606, 12), 'if_condition_20978', if_condition_20978)
                # SSA begins for if statement (line 606)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 607):
                
                # Assigning a Name to a Name (line 607):
                # Getting the type of 'False' (line 607)
                False_20979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 607, 24), 'False')
                # Assigning a type to the variable 'first' (line 607)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 607, 16), 'first', False_20979)
                # SSA branch for the else part of an if statement (line 606)
                module_type_store.open_ssa_branch('else')
                
                # Call to write(...): (line 609)
                # Processing the call arguments (line 609)
                str_20982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 609, 27), 'str', ', ')
                # Processing the call keyword arguments (line 609)
                kwargs_20983 = {}
                # Getting the type of 'self' (line 609)
                self_20980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 609, 16), 'self', False)
                # Obtaining the member 'write' of a type (line 609)
                write_20981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 609, 16), self_20980, 'write')
                # Calling write(args, kwargs) (line 609)
                write_call_result_20984 = invoke(stypy.reporting.localization.Localization(__file__, 609, 16), write_20981, *[str_20982], **kwargs_20983)
                
                # SSA join for if statement (line 606)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to write(...): (line 610)
            # Processing the call arguments (line 610)
            str_20987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 610, 23), 'str', '**')
            # Getting the type of 't' (line 610)
            t_20988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 30), 't', False)
            # Obtaining the member 'kwarg' of a type (line 610)
            kwarg_20989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 30), t_20988, 'kwarg')
            # Applying the binary operator '+' (line 610)
            result_add_20990 = python_operator(stypy.reporting.localization.Localization(__file__, 610, 23), '+', str_20987, kwarg_20989)
            
            # Processing the call keyword arguments (line 610)
            kwargs_20991 = {}
            # Getting the type of 'self' (line 610)
            self_20985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 610, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 610)
            write_20986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 610, 12), self_20985, 'write')
            # Calling write(args, kwargs) (line 610)
            write_call_result_20992 = invoke(stypy.reporting.localization.Localization(__file__, 610, 12), write_20986, *[result_add_20990], **kwargs_20991)
            
            # SSA join for if statement (line 605)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_arguments(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_arguments' in the type store
        # Getting the type of 'stypy_return_type' (line 581)
        stypy_return_type_20993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 581, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20993)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_arguments'
        return stypy_return_type_20993


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
        t_20996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 19), 't', False)
        # Obtaining the member 'arg' of a type (line 613)
        arg_20997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 19), t_20996, 'arg')
        # Processing the call keyword arguments (line 613)
        kwargs_20998 = {}
        # Getting the type of 'self' (line 613)
        self_20994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 613, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 613)
        write_20995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 613, 8), self_20994, 'write')
        # Calling write(args, kwargs) (line 613)
        write_call_result_20999 = invoke(stypy.reporting.localization.Localization(__file__, 613, 8), write_20995, *[arg_20997], **kwargs_20998)
        
        
        # Call to write(...): (line 614)
        # Processing the call arguments (line 614)
        str_21002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 614, 19), 'str', '=')
        # Processing the call keyword arguments (line 614)
        kwargs_21003 = {}
        # Getting the type of 'self' (line 614)
        self_21000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 614, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 614)
        write_21001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 614, 8), self_21000, 'write')
        # Calling write(args, kwargs) (line 614)
        write_call_result_21004 = invoke(stypy.reporting.localization.Localization(__file__, 614, 8), write_21001, *[str_21002], **kwargs_21003)
        
        
        # Call to visit(...): (line 615)
        # Processing the call arguments (line 615)
        # Getting the type of 't' (line 615)
        t_21007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 19), 't', False)
        # Obtaining the member 'value' of a type (line 615)
        value_21008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 19), t_21007, 'value')
        # Processing the call keyword arguments (line 615)
        kwargs_21009 = {}
        # Getting the type of 'self' (line 615)
        self_21005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 615, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 615)
        visit_21006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 615, 8), self_21005, 'visit')
        # Calling visit(args, kwargs) (line 615)
        visit_call_result_21010 = invoke(stypy.reporting.localization.Localization(__file__, 615, 8), visit_21006, *[value_21008], **kwargs_21009)
        
        
        # ################# End of 'visit_keyword(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_keyword' in the type store
        # Getting the type of 'stypy_return_type' (line 612)
        stypy_return_type_21011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 612, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21011)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_keyword'
        return stypy_return_type_21011


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
        str_21014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 618, 19), 'str', '(')
        # Processing the call keyword arguments (line 618)
        kwargs_21015 = {}
        # Getting the type of 'self' (line 618)
        self_21012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 618, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 618)
        write_21013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 618, 8), self_21012, 'write')
        # Calling write(args, kwargs) (line 618)
        write_call_result_21016 = invoke(stypy.reporting.localization.Localization(__file__, 618, 8), write_21013, *[str_21014], **kwargs_21015)
        
        
        # Call to write(...): (line 619)
        # Processing the call arguments (line 619)
        str_21019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 619, 19), 'str', 'lambda ')
        # Processing the call keyword arguments (line 619)
        kwargs_21020 = {}
        # Getting the type of 'self' (line 619)
        self_21017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 619, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 619)
        write_21018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 619, 8), self_21017, 'write')
        # Calling write(args, kwargs) (line 619)
        write_call_result_21021 = invoke(stypy.reporting.localization.Localization(__file__, 619, 8), write_21018, *[str_21019], **kwargs_21020)
        
        
        # Call to visit(...): (line 620)
        # Processing the call arguments (line 620)
        # Getting the type of 't' (line 620)
        t_21024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 19), 't', False)
        # Obtaining the member 'args' of a type (line 620)
        args_21025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 19), t_21024, 'args')
        # Processing the call keyword arguments (line 620)
        kwargs_21026 = {}
        # Getting the type of 'self' (line 620)
        self_21022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 620, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 620)
        visit_21023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 620, 8), self_21022, 'visit')
        # Calling visit(args, kwargs) (line 620)
        visit_call_result_21027 = invoke(stypy.reporting.localization.Localization(__file__, 620, 8), visit_21023, *[args_21025], **kwargs_21026)
        
        
        # Call to write(...): (line 621)
        # Processing the call arguments (line 621)
        str_21030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 621, 19), 'str', ': ')
        # Processing the call keyword arguments (line 621)
        kwargs_21031 = {}
        # Getting the type of 'self' (line 621)
        self_21028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 621, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 621)
        write_21029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 621, 8), self_21028, 'write')
        # Calling write(args, kwargs) (line 621)
        write_call_result_21032 = invoke(stypy.reporting.localization.Localization(__file__, 621, 8), write_21029, *[str_21030], **kwargs_21031)
        
        
        # Call to visit(...): (line 622)
        # Processing the call arguments (line 622)
        # Getting the type of 't' (line 622)
        t_21035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 19), 't', False)
        # Obtaining the member 'body' of a type (line 622)
        body_21036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 19), t_21035, 'body')
        # Processing the call keyword arguments (line 622)
        kwargs_21037 = {}
        # Getting the type of 'self' (line 622)
        self_21033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 622, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 622)
        visit_21034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 622, 8), self_21033, 'visit')
        # Calling visit(args, kwargs) (line 622)
        visit_call_result_21038 = invoke(stypy.reporting.localization.Localization(__file__, 622, 8), visit_21034, *[body_21036], **kwargs_21037)
        
        
        # Call to write(...): (line 623)
        # Processing the call arguments (line 623)
        str_21041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 623, 19), 'str', ')')
        # Processing the call keyword arguments (line 623)
        kwargs_21042 = {}
        # Getting the type of 'self' (line 623)
        self_21039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 623, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 623)
        write_21040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 623, 8), self_21039, 'write')
        # Calling write(args, kwargs) (line 623)
        write_call_result_21043 = invoke(stypy.reporting.localization.Localization(__file__, 623, 8), write_21040, *[str_21041], **kwargs_21042)
        
        
        # ################# End of 'visit_Lambda(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_Lambda' in the type store
        # Getting the type of 'stypy_return_type' (line 617)
        stypy_return_type_21044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 617, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21044)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_Lambda'
        return stypy_return_type_21044


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
        t_21047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 19), 't', False)
        # Obtaining the member 'name' of a type (line 626)
        name_21048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 19), t_21047, 'name')
        # Processing the call keyword arguments (line 626)
        kwargs_21049 = {}
        # Getting the type of 'self' (line 626)
        self_21045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 626, 8), 'self', False)
        # Obtaining the member 'write' of a type (line 626)
        write_21046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 626, 8), self_21045, 'write')
        # Calling write(args, kwargs) (line 626)
        write_call_result_21050 = invoke(stypy.reporting.localization.Localization(__file__, 626, 8), write_21046, *[name_21048], **kwargs_21049)
        
        # Getting the type of 't' (line 627)
        t_21051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 627, 11), 't')
        # Obtaining the member 'asname' of a type (line 627)
        asname_21052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 627, 11), t_21051, 'asname')
        # Testing if the type of an if condition is none (line 627)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 627, 8), asname_21052):
            pass
        else:
            
            # Testing the type of an if condition (line 627)
            if_condition_21053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 627, 8), asname_21052)
            # Assigning a type to the variable 'if_condition_21053' (line 627)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 627, 8), 'if_condition_21053', if_condition_21053)
            # SSA begins for if statement (line 627)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 628)
            # Processing the call arguments (line 628)
            str_21056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 628, 23), 'str', ' as ')
            # Getting the type of 't' (line 628)
            t_21057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 32), 't', False)
            # Obtaining the member 'asname' of a type (line 628)
            asname_21058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 32), t_21057, 'asname')
            # Applying the binary operator '+' (line 628)
            result_add_21059 = python_operator(stypy.reporting.localization.Localization(__file__, 628, 23), '+', str_21056, asname_21058)
            
            # Processing the call keyword arguments (line 628)
            kwargs_21060 = {}
            # Getting the type of 'self' (line 628)
            self_21054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 628, 12), 'self', False)
            # Obtaining the member 'write' of a type (line 628)
            write_21055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 628, 12), self_21054, 'write')
            # Calling write(args, kwargs) (line 628)
            write_call_result_21061 = invoke(stypy.reporting.localization.Localization(__file__, 628, 12), write_21055, *[result_add_21059], **kwargs_21060)
            
            # SSA join for if statement (line 627)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'visit_alias(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit_alias' in the type store
        # Getting the type of 'stypy_return_type' (line 625)
        stypy_return_type_21062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 625, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21062)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit_alias'
        return stypy_return_type_21062


# Assigning a type to the variable 'PythonSrcGeneratorVisitor' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'PythonSrcGeneratorVisitor', PythonSrcGeneratorVisitor)

# Assigning a Dict to a Name (line 463):

# Obtaining an instance of the builtin type 'dict' (line 463)
dict_21063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 463)
# Adding element type (key, value) (line 463)
str_21064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 12), 'str', 'Invert')
str_21065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 22), 'str', '~')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 11), dict_21063, (str_21064, str_21065))
# Adding element type (key, value) (line 463)
str_21066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 27), 'str', 'Not')
str_21067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 34), 'str', 'not')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 11), dict_21063, (str_21066, str_21067))
# Adding element type (key, value) (line 463)
str_21068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 41), 'str', 'UAdd')
str_21069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 49), 'str', '+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 11), dict_21063, (str_21068, str_21069))
# Adding element type (key, value) (line 463)
str_21070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 54), 'str', 'USub')
str_21071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 463, 62), 'str', '-')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 463, 11), dict_21063, (str_21070, str_21071))

# Getting the type of 'PythonSrcGeneratorVisitor'
PythonSrcGeneratorVisitor_21072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PythonSrcGeneratorVisitor')
# Setting the type of the member 'unop' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PythonSrcGeneratorVisitor_21072, 'unop', dict_21063)

# Assigning a Dict to a Name (line 482):

# Obtaining an instance of the builtin type 'dict' (line 482)
dict_21073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 12), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 482)
# Adding element type (key, value) (line 482)
str_21074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 13), 'str', 'Add')
str_21075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 20), 'str', '+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_21073, (str_21074, str_21075))
# Adding element type (key, value) (line 482)
str_21076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 25), 'str', 'Sub')
str_21077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 32), 'str', '-')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_21073, (str_21076, str_21077))
# Adding element type (key, value) (line 482)
str_21078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 37), 'str', 'Mult')
str_21079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 45), 'str', '*')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_21073, (str_21078, str_21079))
# Adding element type (key, value) (line 482)
str_21080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 50), 'str', 'Div')
str_21081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 57), 'str', '/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_21073, (str_21080, str_21081))
# Adding element type (key, value) (line 482)
str_21082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 62), 'str', 'Mod')
str_21083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 69), 'str', '%')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_21073, (str_21082, str_21083))
# Adding element type (key, value) (line 482)
str_21084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 13), 'str', 'LShift')
str_21085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 23), 'str', '<<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_21073, (str_21084, str_21085))
# Adding element type (key, value) (line 482)
str_21086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 29), 'str', 'RShift')
str_21087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 39), 'str', '>>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_21073, (str_21086, str_21087))
# Adding element type (key, value) (line 482)
str_21088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 45), 'str', 'BitOr')
str_21089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 54), 'str', '|')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_21073, (str_21088, str_21089))
# Adding element type (key, value) (line 482)
str_21090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 59), 'str', 'BitXor')
str_21091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 69), 'str', '^')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_21073, (str_21090, str_21091))
# Adding element type (key, value) (line 482)
str_21092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 74), 'str', 'BitAnd')
str_21093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 84), 'str', '&')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_21073, (str_21092, str_21093))
# Adding element type (key, value) (line 482)
str_21094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 13), 'str', 'FloorDiv')
str_21095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 25), 'str', '//')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_21073, (str_21094, str_21095))
# Adding element type (key, value) (line 482)
str_21096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 31), 'str', 'Pow')
str_21097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 484, 38), 'str', '**')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 482, 12), dict_21073, (str_21096, str_21097))

# Getting the type of 'PythonSrcGeneratorVisitor'
PythonSrcGeneratorVisitor_21098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PythonSrcGeneratorVisitor')
# Setting the type of the member 'binop' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PythonSrcGeneratorVisitor_21098, 'binop', dict_21073)

# Assigning a Dict to a Name (line 493):

# Obtaining an instance of the builtin type 'dict' (line 493)
dict_21099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 13), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 493)
# Adding element type (key, value) (line 493)
str_21100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 14), 'str', 'Eq')
str_21101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 20), 'str', '==')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_21099, (str_21100, str_21101))
# Adding element type (key, value) (line 493)
str_21102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 26), 'str', 'NotEq')
str_21103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 35), 'str', '!=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_21099, (str_21102, str_21103))
# Adding element type (key, value) (line 493)
str_21104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 41), 'str', 'Lt')
str_21105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 47), 'str', '<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_21099, (str_21104, str_21105))
# Adding element type (key, value) (line 493)
str_21106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 52), 'str', 'LtE')
str_21107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 59), 'str', '<=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_21099, (str_21106, str_21107))
# Adding element type (key, value) (line 493)
str_21108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 65), 'str', 'Gt')
str_21109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 71), 'str', '>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_21099, (str_21108, str_21109))
# Adding element type (key, value) (line 493)
str_21110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 76), 'str', 'GtE')
str_21111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 493, 83), 'str', '>=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_21099, (str_21110, str_21111))
# Adding element type (key, value) (line 493)
str_21112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 14), 'str', 'Is')
str_21113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 20), 'str', 'is')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_21099, (str_21112, str_21113))
# Adding element type (key, value) (line 493)
str_21114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 26), 'str', 'IsNot')
str_21115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 35), 'str', 'is not')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_21099, (str_21114, str_21115))
# Adding element type (key, value) (line 493)
str_21116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 45), 'str', 'In')
str_21117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 51), 'str', 'in')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_21099, (str_21116, str_21117))
# Adding element type (key, value) (line 493)
str_21118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 57), 'str', 'NotIn')
str_21119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 494, 66), 'str', 'not in')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 493, 13), dict_21099, (str_21118, str_21119))

# Getting the type of 'PythonSrcGeneratorVisitor'
PythonSrcGeneratorVisitor_21120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PythonSrcGeneratorVisitor')
# Setting the type of the member 'cmpops' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PythonSrcGeneratorVisitor_21120, 'cmpops', dict_21099)

# Assigning a Dict to a Name (line 504):

# Obtaining an instance of the builtin type 'dict' (line 504)
dict_21121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 504)
# Adding element type (key, value) (line 504)
# Getting the type of 'ast' (line 504)
ast_21122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 15), 'ast')
# Obtaining the member 'And' of a type (line 504)
And_21123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 15), ast_21122, 'And')
str_21124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 24), 'str', 'and')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 14), dict_21121, (And_21123, str_21124))
# Adding element type (key, value) (line 504)
# Getting the type of 'ast' (line 504)
ast_21125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 31), 'ast')
# Obtaining the member 'Or' of a type (line 504)
Or_21126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 31), ast_21125, 'Or')
str_21127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 39), 'str', 'or')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 504, 14), dict_21121, (Or_21126, str_21127))

# Getting the type of 'PythonSrcGeneratorVisitor'
PythonSrcGeneratorVisitor_21128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PythonSrcGeneratorVisitor')
# Setting the type of the member 'boolops' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PythonSrcGeneratorVisitor_21128, 'boolops', dict_21121)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
