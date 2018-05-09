
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import ast
2: 
3: import core_language_copy
4: import functions_copy
5: import operators_copy
6: from stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy import operator_name_to_symbol
7: from stypy_copy.stypy_parameters_copy import ENABLE_CODING_ADVICES
8: from stypy_copy.reporting_copy.module_line_numbering_copy import ModuleLineNumbering
9: 
10: '''
11: This file contains helper functions_copy to generate type inference code. These functions_copy refer to common language elements
12: such as assignments, numbers, strings and so on.
13: '''
14: 
15: default_function_ret_var_name = "__stypy_ret_value"
16: default_module_type_store_var_name = "type_store"
17: default_type_error_var_name = "module_errors"
18: default_type_warning_var_name = "module_warnings"
19: default_lambda_var_name = "__temp_lambda_"
20: 
21: # ############################################# TEMP VARIABLE CREATION ##############################################
22: 
23: '''
24: Keeps the global count of temp_<x> variables created during type inference code creation.
25: '''
26: __temp_variable_counter = 0
27: 
28: 
29: def __new_temp():
30:     global __temp_variable_counter
31:     __temp_variable_counter += 1
32:     return __temp_variable_counter
33: 
34: 
35: def __new_temp_str(descriptive_var_name):
36:     return "__temp_" + descriptive_var_name + str(__new_temp())
37: 
38: 
39: def new_temp_Name(right_hand_side=True, descriptive_var_name="", lineno=0, col_offset=0):
40:     '''
41:     Creates an AST Name node with a suitable name for a new temp variable. If descriptive_var_name has a value, then
42:     this value is added to the variable predefined name
43:     '''
44:     return core_language_copy.create_Name(__new_temp_str(descriptive_var_name), right_hand_side, lineno, col_offset)
45: 
46: 
47: def create_temp_Assign(right_hand_side, line, column, descriptive_var_name=""):
48:     '''
49:     Creates an assignmen to a newly created temp variable
50:     '''
51:     left_hand_side = new_temp_Name(right_hand_side=False, descriptive_var_name=descriptive_var_name, lineno=line,
52:                                    col_offset=column)
53:     right_hand_side.ctx = ast.Load()
54:     left_hand_side.ctx = ast.Store()
55:     assign_statement = ast.Assign([left_hand_side], right_hand_side)
56:     return assign_statement, left_hand_side
57: 
58: 
59: # ################################# TEMP LAMBDA FUNCTION NAME CREATION ##############################################
60: 
61: '''
62: Keeps the global count of temp_<x> variables created during type inference code creation.
63: '''
64: __temp_lambda_counter = 0
65: 
66: 
67: def __new_temp_lambda():
68:     global __temp_lambda_counter
69:     __temp_lambda_counter += 1
70:     return __temp_lambda_counter
71: 
72: 
73: def new_temp_lambda_str(descriptive_var_name=""):
74:     '''
75:     Creates a new name for a lambda function. If descriptive_var_name has a value, then
76:     this value is added to the variable predefined name
77:     '''
78:     return default_lambda_var_name + descriptive_var_name + str(__new_temp_lambda())
79: 
80: 
81: # ################################################### COMMENTS ####################################################
82: 
83: def __create_src_comment(comment_txt):
84:     comment_node = core_language_copy.create_Name(comment_txt)
85:     comment_expr = ast.Expr()
86:     comment_expr.value = comment_node
87: 
88:     return comment_expr
89: 
90: 
91: def is_blank_line(node):
92:     '''
93:     Determines if a node represent a blank source code line
94:     '''
95:     if isinstance(node, ast.Expr):
96:         if isinstance(node.value, ast.Name):
97:             if node.value.id == "":
98:                 return True
99: 
100:     return False
101: 
102: 
103: def create_blank_line():
104:     '''
105:     Creates a blank line in the source code
106:     '''
107:     return __create_src_comment("")
108: 
109: 
110: def is_src_comment(node):
111:     '''
112:     Determines if a node represent a Python comment
113:     '''
114:     if isinstance(node, ast.Expr):
115:         if isinstance(node.value, ast.Name):
116:             if node.value.id.startswith("#"):
117:                 return True
118: 
119:     return False
120: 
121: 
122: def create_src_comment(comment_txt, lineno=0):
123:     '''
124:     Creates a Python comment with comment_txt
125:     '''
126:     if lineno != 0:
127:         line_str = " (line {0})".format(lineno)
128:     else:
129:         line_str = ""
130: 
131:     return __create_src_comment("# " + comment_txt + line_str)
132: 
133: 
134: def create_program_section_src_comment(comment_txt):
135:     '''
136:     Creates a Python comment with comment_txt and additional characters to mark code blocks
137:     '''
138:     return __create_src_comment("\n################## " + comment_txt + " ##################\n")
139: 
140: 
141: def create_begin_block_src_comment(comment_txt):
142:     '''
143:     Creates a Python comment with comment_txt to init a block of code
144:     '''
145:     return __create_src_comment("\n# " + comment_txt)
146: 
147: 
148: def create_end_block_src_comment(comment_txt):
149:     '''
150:     Creates a Python comment with comment_txt to finish a block of code
151:     '''
152:     return __create_src_comment("# " + comment_txt + "\n")
153: 
154: 
155: def create_original_code_comment(file_name, original_code):
156:     '''
157:     Creates a Python block comment with the original source file code
158:     '''
159:     # Remove block comments, as this code will be placed in a block comment
160:     original_code = original_code.replace("\"\"\"", "'''")
161: 
162:     numbered_original_code = ModuleLineNumbering.put_line_numbers_to_module_code(file_name, original_code)
163: 
164:     comment_txt = core_language_copy.create_Name(
165:         "\"\"\"\nORIGINAL PROGRAM SOURCE CODE:\n" + numbered_original_code + "\n\"\"\"\n")
166:     initial_comment = ast.Expr()
167:     initial_comment.value = comment_txt
168: 
169:     return initial_comment
170: 
171: 
172: # ####################################### MISCELLANEOUS STYPY UTILITY FUNCTIONS ########################################
173: 
174: def flatten_lists(*args):
175:     '''
176:     Recursive function to convert a list of lists into a single "flattened" list, mostly used to streamline lists
177:     of instructions that can contain other instruction lists
178:     '''
179:     if len(args) == 0:
180:         return []
181:     if isinstance(args[0], list):
182:         arguments = args[0] + list(args[1:])
183:         return flatten_lists(*arguments)
184:     return [args[0]] + flatten_lists(*args[1:])
185: 
186: 
187: def create_print_var(variable):
188:     '''
189:     Creates a node to print a variable
190:     '''
191:     node = ast.Print()
192:     node.nl = True
193:     node.dest = None
194:     node.values = [core_language_copy.create_Name(variable)]
195: 
196:     return node
197: 
198: 
199: def assign_line_and_column(dest_node, src_node):
200:     '''
201:     Assign to dest_node the same source line and column of src_node
202:     '''
203:     dest_node.lineno = src_node.lineno
204:     dest_node.col_offset = src_node.col_offset
205: 
206: 
207: def create_localization(line, col):
208:     '''
209:     Creates AST Nodes that creates a new Localization instance
210:     '''
211:     linen = core_language_copy.create_num(line)
212:     coln = core_language_copy.create_num(col)
213:     file_namen = core_language_copy.create_Name('__file__')
214:     loc_namen = core_language_copy.create_Name('stypy.python_lib.python_types.type_inference.localization.Localization')
215:     loc_call = functions_copy.create_call(loc_namen, [file_namen, linen, coln])
216: 
217:     return loc_call
218: 
219: 
220: def create_import_stypy():
221:     '''
222:     Creates AST Nodes that encode "from stypy import *"
223:     '''
224:     alias = core_language_copy.create_alias('*')
225:     importfrom = core_language_copy.create_importfrom("stypy", alias)
226: 
227:     return importfrom
228: 
229: 
230: def create_print_errors():
231:     '''
232:     Creates AST Nodes that encode "ErrorType.print_error_msgs()"
233:     '''
234:     attribute = core_language_copy.create_attribute("ErrorType", "print_error_msgs")
235:     expr = ast.Expr()
236:     expr.value = functions_copy.create_call(attribute, [])
237: 
238:     return expr
239: 
240: 
241: def create_default_return_variable():
242:     '''
243:     Creates AST Nodes that adds the default return variable to a function. Functions of generated type inference
244:      programs only has a return clause
245:     '''
246:     assign_target = core_language_copy.create_Name(default_function_ret_var_name, False)
247:     assign = core_language_copy.create_Assign(assign_target, core_language_copy.create_Name("None"))
248: 
249:     return assign
250: 
251: 
252: def create_store_return_from_function(lineno, col_offset):
253:     set_type_of_comment = create_src_comment("Storing return type", lineno)
254:     set_type_of_method = core_language_copy.create_attribute(default_module_type_store_var_name,
255:                                                         "store_return_type_of_current_context")
256: 
257:     return_var_name = core_language_copy.create_Name(default_function_ret_var_name)
258:     set_type_of_call = functions_copy.create_call_expression(set_type_of_method,
259:                                                         [return_var_name])
260: 
261:     return flatten_lists(set_type_of_comment, set_type_of_call)
262: 
263: 
264: def create_return_from_function(lineno, col_offset):
265:     '''
266:     Creates an AST node to return from a function
267:     '''
268:     return_ = ast.Return()
269:     return_var_name = core_language_copy.create_Name(default_function_ret_var_name)
270:     return_.value = return_var_name
271: 
272:     return flatten_lists(return_)
273: 
274: 
275: def get_descritive_element_name(node):
276:     '''
277:     Gets the name of an AST Name node or an AST Attribute node
278:     '''
279:     if isinstance(node, ast.Name):
280:         return node.id
281:     if isinstance(node, ast.Attribute):
282:         return node.attr
283: 
284:     return ""
285: 
286: 
287: def create_pass_node():
288:     '''
289:     Creates an AST Pass node
290:     '''
291:     return ast.Pass()
292: 
293: 
294: def assign_as_return_type(value):
295:     '''
296:     Creates AST nodes to store in default_function_ret_var_name a possible return type
297:     '''
298:     default_function_ret_var = core_language_copy.create_Name(default_function_ret_var_name)
299:     return core_language_copy.create_Assign(default_function_ret_var, value)
300: 
301: 
302: def create_unsupported_feature_call(localization, feature_name, feature_desc, lineno, col_offset):
303:     '''
304:     Creates AST nodes to call to the unsupported_python_feature function
305:     '''
306:     unsupported_feature_func = core_language_copy.create_Name('unsupported_python_feature',
307:                                                          line=lineno,
308:                                                          column=col_offset)
309:     unsupported_feature = core_language_copy.create_str(feature_name)
310:     unsupported_description = core_language_copy.create_str(
311:         feature_desc)
312:     return functions_copy.create_call_expression(unsupported_feature_func,
313:                                             [localization, unsupported_feature,
314:                                              unsupported_description])
315: 
316: 
317: # TODO: Remove?
318: # def needs_self_object_information(context, node):
319: #     if type(context[-1]) is ast.Call:
320: #         call = context[-1]
321: #         if type(node) is ast.Attribute:
322: #             if type(node.value) is ast.Name:
323: #                 if node.value.id == "type_store":
324: #                     return False
325: #         if type(node) is ast.Name:
326: #             if node.id == "type_store":
327: #                 return False
328: #             introspection_funcs = dir(runtime_type_inspection)
329: #             if node.id in introspection_funcs:
330: #                 return False
331: #     else:
332: #         return False
333: #
334: #     return True
335: 
336: # ################################## GET/SET TYPE AND MEMBERS FUNCTIONS ############################################
337: 
338: '''
339: Functions to get / set the type of variables
340: '''
341: 
342: 
343: def create_add_alias(alias_name, var_name, lineno, col_offset):
344:     get_type_of_comment = create_src_comment("Adding an alias")
345:     get_type_of_method = core_language_copy.create_attribute(default_module_type_store_var_name,
346:                                                         "add_alias", line=lineno,
347:                                                         column=col_offset)
348: 
349:     get_type_of_call = functions_copy.create_call_expression(get_type_of_method, [alias_name, var_name])
350: 
351:     return flatten_lists(get_type_of_comment, get_type_of_call)
352: 
353: 
354: def create_get_type_of(var_name, lineno, col_offset, test_unreferenced=True):
355:     get_type_of_comment = create_src_comment("Getting the type of '{0}'".format(var_name), lineno)
356:     get_type_of_method = core_language_copy.create_attribute(default_module_type_store_var_name,
357:                                                         "get_type_of", line=lineno,
358:                                                         column=col_offset)
359:     localization = create_localization(lineno, col_offset)
360:     if test_unreferenced:
361:         get_type_of_call = functions_copy.create_call(get_type_of_method, [localization, core_language_copy.create_str(var_name)])
362:     else:
363:         get_type_of_call = functions_copy.create_call(get_type_of_method, [localization, core_language_copy.create_str(var_name),
364:                                                                       core_language_copy.create_Name('False')])
365: 
366:     assign_stmts, temp_assign = create_temp_Assign(get_type_of_call, lineno, col_offset)
367: 
368:     return flatten_lists(get_type_of_comment, assign_stmts), temp_assign
369: 
370: 
371: def create_set_type_of(var_name, new_value, lineno, col_offset):
372:     set_type_of_comment = create_src_comment("Type assignment", lineno)
373:     set_type_of_method = core_language_copy.create_attribute(default_module_type_store_var_name, "set_type_of")
374: 
375:     localization = create_localization(lineno, col_offset)
376: 
377:     set_type_of_call = functions_copy.create_call_expression(set_type_of_method,
378:                                                         [localization, core_language_copy.create_str(var_name, lineno,
379:                                                                                                 col_offset), new_value])
380: 
381:     return flatten_lists(set_type_of_comment, set_type_of_call)
382: 
383: 
384: def create_get_type_of_member(owner_var, member_name, lineno, col_offset, test_unreferenced=True):
385:     comment = create_src_comment("Obtaining the member '{0}' of a type".format(member_name), lineno)
386:     localization = create_localization(lineno, col_offset)
387:     # TODO: Remove?
388:     # get_type_of_member_func = core_language_copy.create_Name('get_type_of_member')
389:     # get_type_of_member_call = functions_copy.create_call(get_type_of_member_func, [localization, owner_var,
390:     #                                                                           core_language_copy.create_str(
391:     #                                                                               member_name)])
392: 
393:     get_type_of_member_func = core_language_copy.create_attribute(owner_var, 'get_type_of_member')
394:     if not test_unreferenced:
395:         get_type_of_member_call = functions_copy.create_call(get_type_of_member_func, [localization,
396:                                                                                   core_language_copy.create_str(
397:                                                                                       member_name),
398:                                                                                   core_language_copy.create_Name('False')])
399:     else:
400:         get_type_of_member_call = functions_copy.create_call(get_type_of_member_func, [localization,
401:                                                                                   core_language_copy.create_str(
402:                                                                                       member_name)])
403: 
404:     member_stmts, member_var = create_temp_Assign(get_type_of_member_call, lineno, col_offset)
405: 
406:     return flatten_lists(comment, member_stmts), member_var
407: 
408: 
409: def create_set_type_of_member(owner_var, member_name, value, lineno, col_offset):
410:     comment = create_src_comment("Setting the type of the member '{0}' of a type".format(member_name), lineno)
411:     localization = create_localization(lineno, col_offset)
412:     # TODO: Remove?
413:     # set_type_of_member_func = core_language_copy.create_Name('set_type_of_member')
414:     # set_type_of_member_call = functions_copy.create_call_expression(set_type_of_member_func, [localization, onwer_var,
415:     #                                                                           core_language_copy.create_str(
416:     #                                                                               member_name), value])
417: 
418:     set_type_of_member_func = core_language_copy.create_attribute(owner_var, 'set_type_of_member')
419:     set_type_of_member_call = functions_copy.create_call_expression(set_type_of_member_func, [localization,
420:                                                                                          core_language_copy.create_str(
421:                                                                                              member_name), value])
422: 
423:     return flatten_lists(comment, set_type_of_member_call)
424: 
425: 
426: def create_add_stored_type(owner_var, index, value, lineno, col_offset):
427:     comment = create_src_comment("Storing an element on a container", lineno)
428:     localization = create_localization(lineno, col_offset)
429: 
430:     add_type_func = core_language_copy.create_attribute(owner_var, 'add_key_and_value_type')
431:     param_tuple = ast.Tuple()
432:     param_tuple.elts = [index, value]
433:     set_type_of_member_call = functions_copy.create_call_expression(add_type_func, [localization, param_tuple])
434: 
435:     return flatten_lists(comment, set_type_of_member_call)
436: 
437: 
438: # ############################################# TYPE STORE FUNCTIONS ##############################################
439: 
440: '''
441: Code to deal with type store related functions_copy, assignments, cloning and other operations needed for the SSA algorithm
442: implementation
443: '''
444: 
445: # Keeps the global count of type_store_<x> variables created during type inference code creation.
446: __temp_type_store_counter = 0
447: 
448: 
449: def __new_temp_type_store():
450:     global __temp_type_store_counter
451:     __temp_type_store_counter += 1
452:     return __temp_type_store_counter
453: 
454: 
455: def __new_type_store_name_str():
456:     return "__temp_type_store" + str(__new_temp_type_store())
457: 
458: 
459: def __new_temp_type_store_Name(right_hand_side=True):
460:     if right_hand_side:
461:         return ast.Name(id=__new_type_store_name_str(), ctx=ast.Load())
462:     return ast.Name(id=__new_type_store_name_str(), ctx=ast.Store())
463: 
464: 
465: def create_type_store(type_store_name=default_module_type_store_var_name):
466:     call_arg = core_language_copy.create_Name("__file__")
467:     call_func = core_language_copy.create_Name("TypeStore")
468:     call = functions_copy.create_call(call_func, call_arg)
469:     assign_target = core_language_copy.create_Name(type_store_name, False)
470:     assign = core_language_copy.create_Assign(assign_target, call)
471: 
472:     return assign
473: 
474: 
475: def create_temp_type_store_Assign(right_hand_side):
476:     left_hand_side = __new_temp_type_store_Name(right_hand_side=False)
477:     assign_statement = ast.Assign([left_hand_side], right_hand_side)
478:     return assign_statement, left_hand_side
479: 
480: 
481: def create_clone_type_store():
482:     attribute = core_language_copy.create_attribute("type_store", "clone_type_store")
483:     clone_call = functions_copy.create_call(attribute, [])
484: 
485:     return create_temp_type_store_Assign(clone_call)
486: 
487: 
488: def create_set_unreferenced_var_check(state):
489:     if ENABLE_CODING_ADVICES:
490:         attribute = core_language_copy.create_attribute("type_store", "set_check_unreferenced_vars")
491:         call_ = functions_copy.create_call_expression(attribute, [core_language_copy.create_Name(str(state))])
492: 
493:         return call_
494:     else:
495:         return []
496: 
497: 
498: def create_set_type_store(type_store_param, clone=True):
499:     attribute = core_language_copy.create_attribute("type_store", "set_type_store")
500: 
501:     if clone:
502:         clone_param = core_language_copy.create_Name("True")
503:     else:
504:         clone_param = core_language_copy.create_Name("False")
505: 
506:     set_call = functions_copy.create_call(attribute, [type_store_param, clone_param])
507: 
508:     set_expr = ast.Expr()
509:     set_expr.value = set_call
510: 
511:     return set_expr
512: 
513: 
514: def create_join_type_store(join_func_name, type_stores_to_join):
515:     join_func = core_language_copy.create_Name(join_func_name)
516:     join_call = functions_copy.create_call(join_func, type_stores_to_join)
517: 
518:     left_hand_side = __new_temp_type_store_Name(right_hand_side=False)
519:     join_statement = ast.Assign([left_hand_side], join_call)
520: 
521:     return join_statement, left_hand_side
522: 
523: 
524: # ############################################# OPERATOR FUNCTIONS ##############################################
525: 
526: 
527: def create_binary_operator(operator_name, left_op, rigth_op, lineno, col_offset):
528:     '''
529:     Creates AST Nodes to model a binary operator
530: 
531:     :param operator_name: Name of the operator
532:     :param left_op: Left operand (AST Node)
533:     :param rigth_op: Right operand (AST Node)
534:     :param lineno: Line
535:     :param col_offset: Column
536:     :return: List of instructions
537:     '''
538:     operator_symbol = operator_name_to_symbol(operator_name)
539:     op_name = core_language_copy.create_str(operator_symbol)
540:     operation_comment = create_src_comment("Applying the '{0}' binary operator".format(operator_symbol), lineno)
541:     operator_call, result_var = create_temp_Assign(
542:         operators_copy.create_binary_operator(op_name, left_op, rigth_op, lineno, col_offset), lineno, col_offset)
543: 
544:     return flatten_lists(operation_comment, operator_call), result_var
545: 
546: 
547: def create_unary_operator(operator_name, left_op, lineno, col_offset):
548:     '''
549:     Creates AST Nodes to model an unary operator
550: 
551:     :param operator_name: Name of the operator
552:     :param left_op: operand (AST Node)
553:     :param lineno: Line
554:     :param col_offset: Column
555:     :return: List of instructions
556:     '''
557:     operator_symbol = operator_name_to_symbol(operator_name)
558:     op_name = core_language_copy.create_str(operator_symbol)
559:     operation_comment = create_src_comment("Applying the '{0}' unary operator".format(operator_symbol), lineno)
560:     operator_call, result_var = create_temp_Assign(
561:         operators_copy.create_unary_operator(op_name, left_op, lineno, col_offset), lineno, col_offset)
562: 
563:     return flatten_lists(operation_comment, operator_call), result_var
564: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import ast' statement (line 1)
import ast

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'ast', ast, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import core_language_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_16818 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'core_language_copy')

if (type(import_16818) is not StypyTypeError):

    if (import_16818 != 'pyd_module'):
        __import__(import_16818)
        sys_modules_16819 = sys.modules[import_16818]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'core_language_copy', sys_modules_16819.module_type_store, module_type_store)
    else:
        import core_language_copy

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'core_language_copy', core_language_copy, module_type_store)

else:
    # Assigning a type to the variable 'core_language_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'core_language_copy', import_16818)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import functions_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_16820 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'functions_copy')

if (type(import_16820) is not StypyTypeError):

    if (import_16820 != 'pyd_module'):
        __import__(import_16820)
        sys_modules_16821 = sys.modules[import_16820]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'functions_copy', sys_modules_16821.module_type_store, module_type_store)
    else:
        import functions_copy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'functions_copy', functions_copy, module_type_store)

else:
    # Assigning a type to the variable 'functions_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'functions_copy', import_16820)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import operators_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_16822 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'operators_copy')

if (type(import_16822) is not StypyTypeError):

    if (import_16822 != 'pyd_module'):
        __import__(import_16822)
        sys_modules_16823 = sys.modules[import_16822]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'operators_copy', sys_modules_16823.module_type_store, module_type_store)
    else:
        import operators_copy

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'operators_copy', operators_copy, module_type_store)

else:
    # Assigning a type to the variable 'operators_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'operators_copy', import_16822)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy import operator_name_to_symbol' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_16824 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy')

if (type(import_16824) is not StypyTypeError):

    if (import_16824 != 'pyd_module'):
        __import__(import_16824)
        sys_modules_16825 = sys.modules[import_16824]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy', sys_modules_16825.module_type_store, module_type_store, ['operator_name_to_symbol'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_16825, sys_modules_16825.module_type_store, module_type_store)
    else:
        from stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy import operator_name_to_symbol

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy', None, module_type_store, ['operator_name_to_symbol'], [operator_name_to_symbol])

else:
    # Assigning a type to the variable 'stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy', import_16824)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from stypy_copy.stypy_parameters_copy import ENABLE_CODING_ADVICES' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_16826 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.stypy_parameters_copy')

if (type(import_16826) is not StypyTypeError):

    if (import_16826 != 'pyd_module'):
        __import__(import_16826)
        sys_modules_16827 = sys.modules[import_16826]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.stypy_parameters_copy', sys_modules_16827.module_type_store, module_type_store, ['ENABLE_CODING_ADVICES'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_16827, sys_modules_16827.module_type_store, module_type_store)
    else:
        from stypy_copy.stypy_parameters_copy import ENABLE_CODING_ADVICES

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.stypy_parameters_copy', None, module_type_store, ['ENABLE_CODING_ADVICES'], [ENABLE_CODING_ADVICES])

else:
    # Assigning a type to the variable 'stypy_copy.stypy_parameters_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.stypy_parameters_copy', import_16826)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from stypy_copy.reporting_copy.module_line_numbering_copy import ModuleLineNumbering' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_16828 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.reporting_copy.module_line_numbering_copy')

if (type(import_16828) is not StypyTypeError):

    if (import_16828 != 'pyd_module'):
        __import__(import_16828)
        sys_modules_16829 = sys.modules[import_16828]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.reporting_copy.module_line_numbering_copy', sys_modules_16829.module_type_store, module_type_store, ['ModuleLineNumbering'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_16829, sys_modules_16829.module_type_store, module_type_store)
    else:
        from stypy_copy.reporting_copy.module_line_numbering_copy import ModuleLineNumbering

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.reporting_copy.module_line_numbering_copy', None, module_type_store, ['ModuleLineNumbering'], [ModuleLineNumbering])

else:
    # Assigning a type to the variable 'stypy_copy.reporting_copy.module_line_numbering_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.reporting_copy.module_line_numbering_copy', import_16828)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

str_16830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '\nThis file contains helper functions_copy to generate type inference code. These functions_copy refer to common language elements\nsuch as assignments, numbers, strings and so on.\n')

# Assigning a Str to a Name (line 15):

# Assigning a Str to a Name (line 15):
str_16831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'str', '__stypy_ret_value')
# Assigning a type to the variable 'default_function_ret_var_name' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'default_function_ret_var_name', str_16831)

# Assigning a Str to a Name (line 16):

# Assigning a Str to a Name (line 16):
str_16832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 37), 'str', 'type_store')
# Assigning a type to the variable 'default_module_type_store_var_name' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'default_module_type_store_var_name', str_16832)

# Assigning a Str to a Name (line 17):

# Assigning a Str to a Name (line 17):
str_16833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 30), 'str', 'module_errors')
# Assigning a type to the variable 'default_type_error_var_name' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'default_type_error_var_name', str_16833)

# Assigning a Str to a Name (line 18):

# Assigning a Str to a Name (line 18):
str_16834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 32), 'str', 'module_warnings')
# Assigning a type to the variable 'default_type_warning_var_name' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'default_type_warning_var_name', str_16834)

# Assigning a Str to a Name (line 19):

# Assigning a Str to a Name (line 19):
str_16835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', '__temp_lambda_')
# Assigning a type to the variable 'default_lambda_var_name' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'default_lambda_var_name', str_16835)
str_16836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', '\nKeeps the global count of temp_<x> variables created during type inference code creation.\n')

# Assigning a Num to a Name (line 26):

# Assigning a Num to a Name (line 26):
int_16837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'int')
# Assigning a type to the variable '__temp_variable_counter' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '__temp_variable_counter', int_16837)

@norecursion
def __new_temp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__new_temp'
    module_type_store = module_type_store.open_function_context('__new_temp', 29, 0, False)
    
    # Passed parameters checking function
    __new_temp.stypy_localization = localization
    __new_temp.stypy_type_of_self = None
    __new_temp.stypy_type_store = module_type_store
    __new_temp.stypy_function_name = '__new_temp'
    __new_temp.stypy_param_names_list = []
    __new_temp.stypy_varargs_param_name = None
    __new_temp.stypy_kwargs_param_name = None
    __new_temp.stypy_call_defaults = defaults
    __new_temp.stypy_call_varargs = varargs
    __new_temp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__new_temp', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__new_temp', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__new_temp(...)' code ##################

    # Marking variables as global (line 30)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 30, 4), '__temp_variable_counter')
    
    # Getting the type of '__temp_variable_counter' (line 31)
    temp_variable_counter_16838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), '__temp_variable_counter')
    int_16839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'int')
    # Applying the binary operator '+=' (line 31)
    result_iadd_16840 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 4), '+=', temp_variable_counter_16838, int_16839)
    # Assigning a type to the variable '__temp_variable_counter' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), '__temp_variable_counter', result_iadd_16840)
    
    # Getting the type of '__temp_variable_counter' (line 32)
    temp_variable_counter_16841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), '__temp_variable_counter')
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type', temp_variable_counter_16841)
    
    # ################# End of '__new_temp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__new_temp' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_16842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16842)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__new_temp'
    return stypy_return_type_16842

# Assigning a type to the variable '__new_temp' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), '__new_temp', __new_temp)

@norecursion
def __new_temp_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__new_temp_str'
    module_type_store = module_type_store.open_function_context('__new_temp_str', 35, 0, False)
    
    # Passed parameters checking function
    __new_temp_str.stypy_localization = localization
    __new_temp_str.stypy_type_of_self = None
    __new_temp_str.stypy_type_store = module_type_store
    __new_temp_str.stypy_function_name = '__new_temp_str'
    __new_temp_str.stypy_param_names_list = ['descriptive_var_name']
    __new_temp_str.stypy_varargs_param_name = None
    __new_temp_str.stypy_kwargs_param_name = None
    __new_temp_str.stypy_call_defaults = defaults
    __new_temp_str.stypy_call_varargs = varargs
    __new_temp_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__new_temp_str', ['descriptive_var_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__new_temp_str', localization, ['descriptive_var_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__new_temp_str(...)' code ##################

    str_16843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 11), 'str', '__temp_')
    # Getting the type of 'descriptive_var_name' (line 36)
    descriptive_var_name_16844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'descriptive_var_name')
    # Applying the binary operator '+' (line 36)
    result_add_16845 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 11), '+', str_16843, descriptive_var_name_16844)
    
    
    # Call to str(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Call to __new_temp(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_16848 = {}
    # Getting the type of '__new_temp' (line 36)
    new_temp_16847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 50), '__new_temp', False)
    # Calling __new_temp(args, kwargs) (line 36)
    new_temp_call_result_16849 = invoke(stypy.reporting.localization.Localization(__file__, 36, 50), new_temp_16847, *[], **kwargs_16848)
    
    # Processing the call keyword arguments (line 36)
    kwargs_16850 = {}
    # Getting the type of 'str' (line 36)
    str_16846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 46), 'str', False)
    # Calling str(args, kwargs) (line 36)
    str_call_result_16851 = invoke(stypy.reporting.localization.Localization(__file__, 36, 46), str_16846, *[new_temp_call_result_16849], **kwargs_16850)
    
    # Applying the binary operator '+' (line 36)
    result_add_16852 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 44), '+', result_add_16845, str_call_result_16851)
    
    # Assigning a type to the variable 'stypy_return_type' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type', result_add_16852)
    
    # ################# End of '__new_temp_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__new_temp_str' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_16853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16853)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__new_temp_str'
    return stypy_return_type_16853

# Assigning a type to the variable '__new_temp_str' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), '__new_temp_str', __new_temp_str)

@norecursion
def new_temp_Name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 39)
    True_16854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 34), 'True')
    str_16855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 61), 'str', '')
    int_16856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 72), 'int')
    int_16857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 86), 'int')
    defaults = [True_16854, str_16855, int_16856, int_16857]
    # Create a new context for function 'new_temp_Name'
    module_type_store = module_type_store.open_function_context('new_temp_Name', 39, 0, False)
    
    # Passed parameters checking function
    new_temp_Name.stypy_localization = localization
    new_temp_Name.stypy_type_of_self = None
    new_temp_Name.stypy_type_store = module_type_store
    new_temp_Name.stypy_function_name = 'new_temp_Name'
    new_temp_Name.stypy_param_names_list = ['right_hand_side', 'descriptive_var_name', 'lineno', 'col_offset']
    new_temp_Name.stypy_varargs_param_name = None
    new_temp_Name.stypy_kwargs_param_name = None
    new_temp_Name.stypy_call_defaults = defaults
    new_temp_Name.stypy_call_varargs = varargs
    new_temp_Name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'new_temp_Name', ['right_hand_side', 'descriptive_var_name', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'new_temp_Name', localization, ['right_hand_side', 'descriptive_var_name', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'new_temp_Name(...)' code ##################

    str_16858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', '\n    Creates an AST Name node with a suitable name for a new temp variable. If descriptive_var_name has a value, then\n    this value is added to the variable predefined name\n    ')
    
    # Call to create_Name(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Call to __new_temp_str(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'descriptive_var_name' (line 44)
    descriptive_var_name_16862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 57), 'descriptive_var_name', False)
    # Processing the call keyword arguments (line 44)
    kwargs_16863 = {}
    # Getting the type of '__new_temp_str' (line 44)
    new_temp_str_16861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 42), '__new_temp_str', False)
    # Calling __new_temp_str(args, kwargs) (line 44)
    new_temp_str_call_result_16864 = invoke(stypy.reporting.localization.Localization(__file__, 44, 42), new_temp_str_16861, *[descriptive_var_name_16862], **kwargs_16863)
    
    # Getting the type of 'right_hand_side' (line 44)
    right_hand_side_16865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 80), 'right_hand_side', False)
    # Getting the type of 'lineno' (line 44)
    lineno_16866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 97), 'lineno', False)
    # Getting the type of 'col_offset' (line 44)
    col_offset_16867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 105), 'col_offset', False)
    # Processing the call keyword arguments (line 44)
    kwargs_16868 = {}
    # Getting the type of 'core_language_copy' (line 44)
    core_language_copy_16859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 44)
    create_Name_16860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 11), core_language_copy_16859, 'create_Name')
    # Calling create_Name(args, kwargs) (line 44)
    create_Name_call_result_16869 = invoke(stypy.reporting.localization.Localization(__file__, 44, 11), create_Name_16860, *[new_temp_str_call_result_16864, right_hand_side_16865, lineno_16866, col_offset_16867], **kwargs_16868)
    
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type', create_Name_call_result_16869)
    
    # ################# End of 'new_temp_Name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'new_temp_Name' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_16870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16870)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'new_temp_Name'
    return stypy_return_type_16870

# Assigning a type to the variable 'new_temp_Name' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'new_temp_Name', new_temp_Name)

@norecursion
def create_temp_Assign(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_16871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 75), 'str', '')
    defaults = [str_16871]
    # Create a new context for function 'create_temp_Assign'
    module_type_store = module_type_store.open_function_context('create_temp_Assign', 47, 0, False)
    
    # Passed parameters checking function
    create_temp_Assign.stypy_localization = localization
    create_temp_Assign.stypy_type_of_self = None
    create_temp_Assign.stypy_type_store = module_type_store
    create_temp_Assign.stypy_function_name = 'create_temp_Assign'
    create_temp_Assign.stypy_param_names_list = ['right_hand_side', 'line', 'column', 'descriptive_var_name']
    create_temp_Assign.stypy_varargs_param_name = None
    create_temp_Assign.stypy_kwargs_param_name = None
    create_temp_Assign.stypy_call_defaults = defaults
    create_temp_Assign.stypy_call_varargs = varargs
    create_temp_Assign.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_temp_Assign', ['right_hand_side', 'line', 'column', 'descriptive_var_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_temp_Assign', localization, ['right_hand_side', 'line', 'column', 'descriptive_var_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_temp_Assign(...)' code ##################

    str_16872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', '\n    Creates an assignmen to a newly created temp variable\n    ')
    
    # Assigning a Call to a Name (line 51):
    
    # Assigning a Call to a Name (line 51):
    
    # Call to new_temp_Name(...): (line 51)
    # Processing the call keyword arguments (line 51)
    # Getting the type of 'False' (line 51)
    False_16874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 51), 'False', False)
    keyword_16875 = False_16874
    # Getting the type of 'descriptive_var_name' (line 51)
    descriptive_var_name_16876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 79), 'descriptive_var_name', False)
    keyword_16877 = descriptive_var_name_16876
    # Getting the type of 'line' (line 51)
    line_16878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 108), 'line', False)
    keyword_16879 = line_16878
    # Getting the type of 'column' (line 52)
    column_16880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 46), 'column', False)
    keyword_16881 = column_16880
    kwargs_16882 = {'descriptive_var_name': keyword_16877, 'lineno': keyword_16879, 'col_offset': keyword_16881, 'right_hand_side': keyword_16875}
    # Getting the type of 'new_temp_Name' (line 51)
    new_temp_Name_16873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 21), 'new_temp_Name', False)
    # Calling new_temp_Name(args, kwargs) (line 51)
    new_temp_Name_call_result_16883 = invoke(stypy.reporting.localization.Localization(__file__, 51, 21), new_temp_Name_16873, *[], **kwargs_16882)
    
    # Assigning a type to the variable 'left_hand_side' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'left_hand_side', new_temp_Name_call_result_16883)
    
    # Assigning a Call to a Attribute (line 53):
    
    # Assigning a Call to a Attribute (line 53):
    
    # Call to Load(...): (line 53)
    # Processing the call keyword arguments (line 53)
    kwargs_16886 = {}
    # Getting the type of 'ast' (line 53)
    ast_16884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'ast', False)
    # Obtaining the member 'Load' of a type (line 53)
    Load_16885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 26), ast_16884, 'Load')
    # Calling Load(args, kwargs) (line 53)
    Load_call_result_16887 = invoke(stypy.reporting.localization.Localization(__file__, 53, 26), Load_16885, *[], **kwargs_16886)
    
    # Getting the type of 'right_hand_side' (line 53)
    right_hand_side_16888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'right_hand_side')
    # Setting the type of the member 'ctx' of a type (line 53)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 4), right_hand_side_16888, 'ctx', Load_call_result_16887)
    
    # Assigning a Call to a Attribute (line 54):
    
    # Assigning a Call to a Attribute (line 54):
    
    # Call to Store(...): (line 54)
    # Processing the call keyword arguments (line 54)
    kwargs_16891 = {}
    # Getting the type of 'ast' (line 54)
    ast_16889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'ast', False)
    # Obtaining the member 'Store' of a type (line 54)
    Store_16890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 25), ast_16889, 'Store')
    # Calling Store(args, kwargs) (line 54)
    Store_call_result_16892 = invoke(stypy.reporting.localization.Localization(__file__, 54, 25), Store_16890, *[], **kwargs_16891)
    
    # Getting the type of 'left_hand_side' (line 54)
    left_hand_side_16893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'left_hand_side')
    # Setting the type of the member 'ctx' of a type (line 54)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 4), left_hand_side_16893, 'ctx', Store_call_result_16892)
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to Assign(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_16896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    # Getting the type of 'left_hand_side' (line 55)
    left_hand_side_16897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'left_hand_side', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 34), list_16896, left_hand_side_16897)
    
    # Getting the type of 'right_hand_side' (line 55)
    right_hand_side_16898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 52), 'right_hand_side', False)
    # Processing the call keyword arguments (line 55)
    kwargs_16899 = {}
    # Getting the type of 'ast' (line 55)
    ast_16894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'ast', False)
    # Obtaining the member 'Assign' of a type (line 55)
    Assign_16895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 23), ast_16894, 'Assign')
    # Calling Assign(args, kwargs) (line 55)
    Assign_call_result_16900 = invoke(stypy.reporting.localization.Localization(__file__, 55, 23), Assign_16895, *[list_16896, right_hand_side_16898], **kwargs_16899)
    
    # Assigning a type to the variable 'assign_statement' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'assign_statement', Assign_call_result_16900)
    
    # Obtaining an instance of the builtin type 'tuple' (line 56)
    tuple_16901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 56)
    # Adding element type (line 56)
    # Getting the type of 'assign_statement' (line 56)
    assign_statement_16902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'assign_statement')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 11), tuple_16901, assign_statement_16902)
    # Adding element type (line 56)
    # Getting the type of 'left_hand_side' (line 56)
    left_hand_side_16903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'left_hand_side')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 11), tuple_16901, left_hand_side_16903)
    
    # Assigning a type to the variable 'stypy_return_type' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type', tuple_16901)
    
    # ################# End of 'create_temp_Assign(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_temp_Assign' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_16904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16904)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_temp_Assign'
    return stypy_return_type_16904

# Assigning a type to the variable 'create_temp_Assign' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'create_temp_Assign', create_temp_Assign)
str_16905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, (-1)), 'str', '\nKeeps the global count of temp_<x> variables created during type inference code creation.\n')

# Assigning a Num to a Name (line 64):

# Assigning a Num to a Name (line 64):
int_16906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 24), 'int')
# Assigning a type to the variable '__temp_lambda_counter' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), '__temp_lambda_counter', int_16906)

@norecursion
def __new_temp_lambda(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__new_temp_lambda'
    module_type_store = module_type_store.open_function_context('__new_temp_lambda', 67, 0, False)
    
    # Passed parameters checking function
    __new_temp_lambda.stypy_localization = localization
    __new_temp_lambda.stypy_type_of_self = None
    __new_temp_lambda.stypy_type_store = module_type_store
    __new_temp_lambda.stypy_function_name = '__new_temp_lambda'
    __new_temp_lambda.stypy_param_names_list = []
    __new_temp_lambda.stypy_varargs_param_name = None
    __new_temp_lambda.stypy_kwargs_param_name = None
    __new_temp_lambda.stypy_call_defaults = defaults
    __new_temp_lambda.stypy_call_varargs = varargs
    __new_temp_lambda.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__new_temp_lambda', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__new_temp_lambda', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__new_temp_lambda(...)' code ##################

    # Marking variables as global (line 68)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 68, 4), '__temp_lambda_counter')
    
    # Getting the type of '__temp_lambda_counter' (line 69)
    temp_lambda_counter_16907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), '__temp_lambda_counter')
    int_16908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 29), 'int')
    # Applying the binary operator '+=' (line 69)
    result_iadd_16909 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 4), '+=', temp_lambda_counter_16907, int_16908)
    # Assigning a type to the variable '__temp_lambda_counter' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), '__temp_lambda_counter', result_iadd_16909)
    
    # Getting the type of '__temp_lambda_counter' (line 70)
    temp_lambda_counter_16910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), '__temp_lambda_counter')
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', temp_lambda_counter_16910)
    
    # ################# End of '__new_temp_lambda(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__new_temp_lambda' in the type store
    # Getting the type of 'stypy_return_type' (line 67)
    stypy_return_type_16911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16911)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__new_temp_lambda'
    return stypy_return_type_16911

# Assigning a type to the variable '__new_temp_lambda' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), '__new_temp_lambda', __new_temp_lambda)

@norecursion
def new_temp_lambda_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_16912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 45), 'str', '')
    defaults = [str_16912]
    # Create a new context for function 'new_temp_lambda_str'
    module_type_store = module_type_store.open_function_context('new_temp_lambda_str', 73, 0, False)
    
    # Passed parameters checking function
    new_temp_lambda_str.stypy_localization = localization
    new_temp_lambda_str.stypy_type_of_self = None
    new_temp_lambda_str.stypy_type_store = module_type_store
    new_temp_lambda_str.stypy_function_name = 'new_temp_lambda_str'
    new_temp_lambda_str.stypy_param_names_list = ['descriptive_var_name']
    new_temp_lambda_str.stypy_varargs_param_name = None
    new_temp_lambda_str.stypy_kwargs_param_name = None
    new_temp_lambda_str.stypy_call_defaults = defaults
    new_temp_lambda_str.stypy_call_varargs = varargs
    new_temp_lambda_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'new_temp_lambda_str', ['descriptive_var_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'new_temp_lambda_str', localization, ['descriptive_var_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'new_temp_lambda_str(...)' code ##################

    str_16913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, (-1)), 'str', '\n    Creates a new name for a lambda function. If descriptive_var_name has a value, then\n    this value is added to the variable predefined name\n    ')
    # Getting the type of 'default_lambda_var_name' (line 78)
    default_lambda_var_name_16914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'default_lambda_var_name')
    # Getting the type of 'descriptive_var_name' (line 78)
    descriptive_var_name_16915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 37), 'descriptive_var_name')
    # Applying the binary operator '+' (line 78)
    result_add_16916 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), '+', default_lambda_var_name_16914, descriptive_var_name_16915)
    
    
    # Call to str(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Call to __new_temp_lambda(...): (line 78)
    # Processing the call keyword arguments (line 78)
    kwargs_16919 = {}
    # Getting the type of '__new_temp_lambda' (line 78)
    new_temp_lambda_16918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 64), '__new_temp_lambda', False)
    # Calling __new_temp_lambda(args, kwargs) (line 78)
    new_temp_lambda_call_result_16920 = invoke(stypy.reporting.localization.Localization(__file__, 78, 64), new_temp_lambda_16918, *[], **kwargs_16919)
    
    # Processing the call keyword arguments (line 78)
    kwargs_16921 = {}
    # Getting the type of 'str' (line 78)
    str_16917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 60), 'str', False)
    # Calling str(args, kwargs) (line 78)
    str_call_result_16922 = invoke(stypy.reporting.localization.Localization(__file__, 78, 60), str_16917, *[new_temp_lambda_call_result_16920], **kwargs_16921)
    
    # Applying the binary operator '+' (line 78)
    result_add_16923 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 58), '+', result_add_16916, str_call_result_16922)
    
    # Assigning a type to the variable 'stypy_return_type' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type', result_add_16923)
    
    # ################# End of 'new_temp_lambda_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'new_temp_lambda_str' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_16924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16924)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'new_temp_lambda_str'
    return stypy_return_type_16924

# Assigning a type to the variable 'new_temp_lambda_str' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'new_temp_lambda_str', new_temp_lambda_str)

@norecursion
def __create_src_comment(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__create_src_comment'
    module_type_store = module_type_store.open_function_context('__create_src_comment', 83, 0, False)
    
    # Passed parameters checking function
    __create_src_comment.stypy_localization = localization
    __create_src_comment.stypy_type_of_self = None
    __create_src_comment.stypy_type_store = module_type_store
    __create_src_comment.stypy_function_name = '__create_src_comment'
    __create_src_comment.stypy_param_names_list = ['comment_txt']
    __create_src_comment.stypy_varargs_param_name = None
    __create_src_comment.stypy_kwargs_param_name = None
    __create_src_comment.stypy_call_defaults = defaults
    __create_src_comment.stypy_call_varargs = varargs
    __create_src_comment.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__create_src_comment', ['comment_txt'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__create_src_comment', localization, ['comment_txt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__create_src_comment(...)' code ##################

    
    # Assigning a Call to a Name (line 84):
    
    # Assigning a Call to a Name (line 84):
    
    # Call to create_Name(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'comment_txt' (line 84)
    comment_txt_16927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 50), 'comment_txt', False)
    # Processing the call keyword arguments (line 84)
    kwargs_16928 = {}
    # Getting the type of 'core_language_copy' (line 84)
    core_language_copy_16925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 84)
    create_Name_16926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 19), core_language_copy_16925, 'create_Name')
    # Calling create_Name(args, kwargs) (line 84)
    create_Name_call_result_16929 = invoke(stypy.reporting.localization.Localization(__file__, 84, 19), create_Name_16926, *[comment_txt_16927], **kwargs_16928)
    
    # Assigning a type to the variable 'comment_node' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'comment_node', create_Name_call_result_16929)
    
    # Assigning a Call to a Name (line 85):
    
    # Assigning a Call to a Name (line 85):
    
    # Call to Expr(...): (line 85)
    # Processing the call keyword arguments (line 85)
    kwargs_16932 = {}
    # Getting the type of 'ast' (line 85)
    ast_16930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 85)
    Expr_16931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 19), ast_16930, 'Expr')
    # Calling Expr(args, kwargs) (line 85)
    Expr_call_result_16933 = invoke(stypy.reporting.localization.Localization(__file__, 85, 19), Expr_16931, *[], **kwargs_16932)
    
    # Assigning a type to the variable 'comment_expr' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'comment_expr', Expr_call_result_16933)
    
    # Assigning a Name to a Attribute (line 86):
    
    # Assigning a Name to a Attribute (line 86):
    # Getting the type of 'comment_node' (line 86)
    comment_node_16934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'comment_node')
    # Getting the type of 'comment_expr' (line 86)
    comment_expr_16935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'comment_expr')
    # Setting the type of the member 'value' of a type (line 86)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 4), comment_expr_16935, 'value', comment_node_16934)
    # Getting the type of 'comment_expr' (line 88)
    comment_expr_16936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'comment_expr')
    # Assigning a type to the variable 'stypy_return_type' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type', comment_expr_16936)
    
    # ################# End of '__create_src_comment(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__create_src_comment' in the type store
    # Getting the type of 'stypy_return_type' (line 83)
    stypy_return_type_16937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16937)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__create_src_comment'
    return stypy_return_type_16937

# Assigning a type to the variable '__create_src_comment' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), '__create_src_comment', __create_src_comment)

@norecursion
def is_blank_line(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_blank_line'
    module_type_store = module_type_store.open_function_context('is_blank_line', 91, 0, False)
    
    # Passed parameters checking function
    is_blank_line.stypy_localization = localization
    is_blank_line.stypy_type_of_self = None
    is_blank_line.stypy_type_store = module_type_store
    is_blank_line.stypy_function_name = 'is_blank_line'
    is_blank_line.stypy_param_names_list = ['node']
    is_blank_line.stypy_varargs_param_name = None
    is_blank_line.stypy_kwargs_param_name = None
    is_blank_line.stypy_call_defaults = defaults
    is_blank_line.stypy_call_varargs = varargs
    is_blank_line.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_blank_line', ['node'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_blank_line', localization, ['node'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_blank_line(...)' code ##################

    str_16938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, (-1)), 'str', '\n    Determines if a node represent a blank source code line\n    ')
    
    # Call to isinstance(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'node' (line 95)
    node_16940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 18), 'node', False)
    # Getting the type of 'ast' (line 95)
    ast_16941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 95)
    Expr_16942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 24), ast_16941, 'Expr')
    # Processing the call keyword arguments (line 95)
    kwargs_16943 = {}
    # Getting the type of 'isinstance' (line 95)
    isinstance_16939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 95)
    isinstance_call_result_16944 = invoke(stypy.reporting.localization.Localization(__file__, 95, 7), isinstance_16939, *[node_16940, Expr_16942], **kwargs_16943)
    
    # Testing if the type of an if condition is none (line 95)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 95, 4), isinstance_call_result_16944):
        pass
    else:
        
        # Testing the type of an if condition (line 95)
        if_condition_16945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 4), isinstance_call_result_16944)
        # Assigning a type to the variable 'if_condition_16945' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'if_condition_16945', if_condition_16945)
        # SSA begins for if statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to isinstance(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'node' (line 96)
        node_16947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'node', False)
        # Obtaining the member 'value' of a type (line 96)
        value_16948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 22), node_16947, 'value')
        # Getting the type of 'ast' (line 96)
        ast_16949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 34), 'ast', False)
        # Obtaining the member 'Name' of a type (line 96)
        Name_16950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 34), ast_16949, 'Name')
        # Processing the call keyword arguments (line 96)
        kwargs_16951 = {}
        # Getting the type of 'isinstance' (line 96)
        isinstance_16946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 96)
        isinstance_call_result_16952 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), isinstance_16946, *[value_16948, Name_16950], **kwargs_16951)
        
        # Testing if the type of an if condition is none (line 96)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 96, 8), isinstance_call_result_16952):
            pass
        else:
            
            # Testing the type of an if condition (line 96)
            if_condition_16953 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 8), isinstance_call_result_16952)
            # Assigning a type to the variable 'if_condition_16953' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'if_condition_16953', if_condition_16953)
            # SSA begins for if statement (line 96)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'node' (line 97)
            node_16954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'node')
            # Obtaining the member 'value' of a type (line 97)
            value_16955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), node_16954, 'value')
            # Obtaining the member 'id' of a type (line 97)
            id_16956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), value_16955, 'id')
            str_16957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 32), 'str', '')
            # Applying the binary operator '==' (line 97)
            result_eq_16958 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 15), '==', id_16956, str_16957)
            
            # Testing if the type of an if condition is none (line 97)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 97, 12), result_eq_16958):
                pass
            else:
                
                # Testing the type of an if condition (line 97)
                if_condition_16959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 12), result_eq_16958)
                # Assigning a type to the variable 'if_condition_16959' (line 97)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'if_condition_16959', if_condition_16959)
                # SSA begins for if statement (line 97)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 98)
                True_16960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 98)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'stypy_return_type', True_16960)
                # SSA join for if statement (line 97)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 96)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'False' (line 100)
    False_16961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type', False_16961)
    
    # ################# End of 'is_blank_line(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_blank_line' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_16962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16962)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_blank_line'
    return stypy_return_type_16962

# Assigning a type to the variable 'is_blank_line' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'is_blank_line', is_blank_line)

@norecursion
def create_blank_line(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_blank_line'
    module_type_store = module_type_store.open_function_context('create_blank_line', 103, 0, False)
    
    # Passed parameters checking function
    create_blank_line.stypy_localization = localization
    create_blank_line.stypy_type_of_self = None
    create_blank_line.stypy_type_store = module_type_store
    create_blank_line.stypy_function_name = 'create_blank_line'
    create_blank_line.stypy_param_names_list = []
    create_blank_line.stypy_varargs_param_name = None
    create_blank_line.stypy_kwargs_param_name = None
    create_blank_line.stypy_call_defaults = defaults
    create_blank_line.stypy_call_varargs = varargs
    create_blank_line.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_blank_line', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_blank_line', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_blank_line(...)' code ##################

    str_16963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, (-1)), 'str', '\n    Creates a blank line in the source code\n    ')
    
    # Call to __create_src_comment(...): (line 107)
    # Processing the call arguments (line 107)
    str_16965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 32), 'str', '')
    # Processing the call keyword arguments (line 107)
    kwargs_16966 = {}
    # Getting the type of '__create_src_comment' (line 107)
    create_src_comment_16964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), '__create_src_comment', False)
    # Calling __create_src_comment(args, kwargs) (line 107)
    create_src_comment_call_result_16967 = invoke(stypy.reporting.localization.Localization(__file__, 107, 11), create_src_comment_16964, *[str_16965], **kwargs_16966)
    
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type', create_src_comment_call_result_16967)
    
    # ################# End of 'create_blank_line(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_blank_line' in the type store
    # Getting the type of 'stypy_return_type' (line 103)
    stypy_return_type_16968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16968)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_blank_line'
    return stypy_return_type_16968

# Assigning a type to the variable 'create_blank_line' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'create_blank_line', create_blank_line)

@norecursion
def is_src_comment(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_src_comment'
    module_type_store = module_type_store.open_function_context('is_src_comment', 110, 0, False)
    
    # Passed parameters checking function
    is_src_comment.stypy_localization = localization
    is_src_comment.stypy_type_of_self = None
    is_src_comment.stypy_type_store = module_type_store
    is_src_comment.stypy_function_name = 'is_src_comment'
    is_src_comment.stypy_param_names_list = ['node']
    is_src_comment.stypy_varargs_param_name = None
    is_src_comment.stypy_kwargs_param_name = None
    is_src_comment.stypy_call_defaults = defaults
    is_src_comment.stypy_call_varargs = varargs
    is_src_comment.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_src_comment', ['node'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_src_comment', localization, ['node'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_src_comment(...)' code ##################

    str_16969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, (-1)), 'str', '\n    Determines if a node represent a Python comment\n    ')
    
    # Call to isinstance(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'node' (line 114)
    node_16971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'node', False)
    # Getting the type of 'ast' (line 114)
    ast_16972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 114)
    Expr_16973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 24), ast_16972, 'Expr')
    # Processing the call keyword arguments (line 114)
    kwargs_16974 = {}
    # Getting the type of 'isinstance' (line 114)
    isinstance_16970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 114)
    isinstance_call_result_16975 = invoke(stypy.reporting.localization.Localization(__file__, 114, 7), isinstance_16970, *[node_16971, Expr_16973], **kwargs_16974)
    
    # Testing if the type of an if condition is none (line 114)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 114, 4), isinstance_call_result_16975):
        pass
    else:
        
        # Testing the type of an if condition (line 114)
        if_condition_16976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 4), isinstance_call_result_16975)
        # Assigning a type to the variable 'if_condition_16976' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'if_condition_16976', if_condition_16976)
        # SSA begins for if statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to isinstance(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'node' (line 115)
        node_16978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 22), 'node', False)
        # Obtaining the member 'value' of a type (line 115)
        value_16979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 22), node_16978, 'value')
        # Getting the type of 'ast' (line 115)
        ast_16980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 34), 'ast', False)
        # Obtaining the member 'Name' of a type (line 115)
        Name_16981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 34), ast_16980, 'Name')
        # Processing the call keyword arguments (line 115)
        kwargs_16982 = {}
        # Getting the type of 'isinstance' (line 115)
        isinstance_16977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 115)
        isinstance_call_result_16983 = invoke(stypy.reporting.localization.Localization(__file__, 115, 11), isinstance_16977, *[value_16979, Name_16981], **kwargs_16982)
        
        # Testing if the type of an if condition is none (line 115)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 115, 8), isinstance_call_result_16983):
            pass
        else:
            
            # Testing the type of an if condition (line 115)
            if_condition_16984 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 8), isinstance_call_result_16983)
            # Assigning a type to the variable 'if_condition_16984' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'if_condition_16984', if_condition_16984)
            # SSA begins for if statement (line 115)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to startswith(...): (line 116)
            # Processing the call arguments (line 116)
            str_16989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 40), 'str', '#')
            # Processing the call keyword arguments (line 116)
            kwargs_16990 = {}
            # Getting the type of 'node' (line 116)
            node_16985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'node', False)
            # Obtaining the member 'value' of a type (line 116)
            value_16986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 15), node_16985, 'value')
            # Obtaining the member 'id' of a type (line 116)
            id_16987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 15), value_16986, 'id')
            # Obtaining the member 'startswith' of a type (line 116)
            startswith_16988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 15), id_16987, 'startswith')
            # Calling startswith(args, kwargs) (line 116)
            startswith_call_result_16991 = invoke(stypy.reporting.localization.Localization(__file__, 116, 15), startswith_16988, *[str_16989], **kwargs_16990)
            
            # Testing if the type of an if condition is none (line 116)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 116, 12), startswith_call_result_16991):
                pass
            else:
                
                # Testing the type of an if condition (line 116)
                if_condition_16992 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 12), startswith_call_result_16991)
                # Assigning a type to the variable 'if_condition_16992' (line 116)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'if_condition_16992', if_condition_16992)
                # SSA begins for if statement (line 116)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 117)
                True_16993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 117)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'stypy_return_type', True_16993)
                # SSA join for if statement (line 116)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 115)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 114)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'False' (line 119)
    False_16994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type', False_16994)
    
    # ################# End of 'is_src_comment(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_src_comment' in the type store
    # Getting the type of 'stypy_return_type' (line 110)
    stypy_return_type_16995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16995)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_src_comment'
    return stypy_return_type_16995

# Assigning a type to the variable 'is_src_comment' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'is_src_comment', is_src_comment)

@norecursion
def create_src_comment(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_16996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 43), 'int')
    defaults = [int_16996]
    # Create a new context for function 'create_src_comment'
    module_type_store = module_type_store.open_function_context('create_src_comment', 122, 0, False)
    
    # Passed parameters checking function
    create_src_comment.stypy_localization = localization
    create_src_comment.stypy_type_of_self = None
    create_src_comment.stypy_type_store = module_type_store
    create_src_comment.stypy_function_name = 'create_src_comment'
    create_src_comment.stypy_param_names_list = ['comment_txt', 'lineno']
    create_src_comment.stypy_varargs_param_name = None
    create_src_comment.stypy_kwargs_param_name = None
    create_src_comment.stypy_call_defaults = defaults
    create_src_comment.stypy_call_varargs = varargs
    create_src_comment.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_src_comment', ['comment_txt', 'lineno'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_src_comment', localization, ['comment_txt', 'lineno'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_src_comment(...)' code ##################

    str_16997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, (-1)), 'str', '\n    Creates a Python comment with comment_txt\n    ')
    
    # Getting the type of 'lineno' (line 126)
    lineno_16998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 7), 'lineno')
    int_16999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 17), 'int')
    # Applying the binary operator '!=' (line 126)
    result_ne_17000 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 7), '!=', lineno_16998, int_16999)
    
    # Testing if the type of an if condition is none (line 126)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 126, 4), result_ne_17000):
        
        # Assigning a Str to a Name (line 129):
        
        # Assigning a Str to a Name (line 129):
        str_17007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 19), 'str', '')
        # Assigning a type to the variable 'line_str' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'line_str', str_17007)
    else:
        
        # Testing the type of an if condition (line 126)
        if_condition_17001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 4), result_ne_17000)
        # Assigning a type to the variable 'if_condition_17001' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'if_condition_17001', if_condition_17001)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 127):
        
        # Assigning a Call to a Name (line 127):
        
        # Call to format(...): (line 127)
        # Processing the call arguments (line 127)
        # Getting the type of 'lineno' (line 127)
        lineno_17004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 40), 'lineno', False)
        # Processing the call keyword arguments (line 127)
        kwargs_17005 = {}
        str_17002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 19), 'str', ' (line {0})')
        # Obtaining the member 'format' of a type (line 127)
        format_17003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), str_17002, 'format')
        # Calling format(args, kwargs) (line 127)
        format_call_result_17006 = invoke(stypy.reporting.localization.Localization(__file__, 127, 19), format_17003, *[lineno_17004], **kwargs_17005)
        
        # Assigning a type to the variable 'line_str' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'line_str', format_call_result_17006)
        # SSA branch for the else part of an if statement (line 126)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 129):
        
        # Assigning a Str to a Name (line 129):
        str_17007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 19), 'str', '')
        # Assigning a type to the variable 'line_str' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'line_str', str_17007)
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to __create_src_comment(...): (line 131)
    # Processing the call arguments (line 131)
    str_17009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 32), 'str', '# ')
    # Getting the type of 'comment_txt' (line 131)
    comment_txt_17010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 39), 'comment_txt', False)
    # Applying the binary operator '+' (line 131)
    result_add_17011 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 32), '+', str_17009, comment_txt_17010)
    
    # Getting the type of 'line_str' (line 131)
    line_str_17012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 53), 'line_str', False)
    # Applying the binary operator '+' (line 131)
    result_add_17013 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 51), '+', result_add_17011, line_str_17012)
    
    # Processing the call keyword arguments (line 131)
    kwargs_17014 = {}
    # Getting the type of '__create_src_comment' (line 131)
    create_src_comment_17008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), '__create_src_comment', False)
    # Calling __create_src_comment(args, kwargs) (line 131)
    create_src_comment_call_result_17015 = invoke(stypy.reporting.localization.Localization(__file__, 131, 11), create_src_comment_17008, *[result_add_17013], **kwargs_17014)
    
    # Assigning a type to the variable 'stypy_return_type' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type', create_src_comment_call_result_17015)
    
    # ################# End of 'create_src_comment(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_src_comment' in the type store
    # Getting the type of 'stypy_return_type' (line 122)
    stypy_return_type_17016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17016)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_src_comment'
    return stypy_return_type_17016

# Assigning a type to the variable 'create_src_comment' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'create_src_comment', create_src_comment)

@norecursion
def create_program_section_src_comment(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_program_section_src_comment'
    module_type_store = module_type_store.open_function_context('create_program_section_src_comment', 134, 0, False)
    
    # Passed parameters checking function
    create_program_section_src_comment.stypy_localization = localization
    create_program_section_src_comment.stypy_type_of_self = None
    create_program_section_src_comment.stypy_type_store = module_type_store
    create_program_section_src_comment.stypy_function_name = 'create_program_section_src_comment'
    create_program_section_src_comment.stypy_param_names_list = ['comment_txt']
    create_program_section_src_comment.stypy_varargs_param_name = None
    create_program_section_src_comment.stypy_kwargs_param_name = None
    create_program_section_src_comment.stypy_call_defaults = defaults
    create_program_section_src_comment.stypy_call_varargs = varargs
    create_program_section_src_comment.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_program_section_src_comment', ['comment_txt'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_program_section_src_comment', localization, ['comment_txt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_program_section_src_comment(...)' code ##################

    str_17017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, (-1)), 'str', '\n    Creates a Python comment with comment_txt and additional characters to mark code blocks\n    ')
    
    # Call to __create_src_comment(...): (line 138)
    # Processing the call arguments (line 138)
    str_17019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 32), 'str', '\n################## ')
    # Getting the type of 'comment_txt' (line 138)
    comment_txt_17020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 58), 'comment_txt', False)
    # Applying the binary operator '+' (line 138)
    result_add_17021 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 32), '+', str_17019, comment_txt_17020)
    
    str_17022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 72), 'str', ' ##################\n')
    # Applying the binary operator '+' (line 138)
    result_add_17023 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 70), '+', result_add_17021, str_17022)
    
    # Processing the call keyword arguments (line 138)
    kwargs_17024 = {}
    # Getting the type of '__create_src_comment' (line 138)
    create_src_comment_17018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), '__create_src_comment', False)
    # Calling __create_src_comment(args, kwargs) (line 138)
    create_src_comment_call_result_17025 = invoke(stypy.reporting.localization.Localization(__file__, 138, 11), create_src_comment_17018, *[result_add_17023], **kwargs_17024)
    
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type', create_src_comment_call_result_17025)
    
    # ################# End of 'create_program_section_src_comment(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_program_section_src_comment' in the type store
    # Getting the type of 'stypy_return_type' (line 134)
    stypy_return_type_17026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17026)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_program_section_src_comment'
    return stypy_return_type_17026

# Assigning a type to the variable 'create_program_section_src_comment' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'create_program_section_src_comment', create_program_section_src_comment)

@norecursion
def create_begin_block_src_comment(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_begin_block_src_comment'
    module_type_store = module_type_store.open_function_context('create_begin_block_src_comment', 141, 0, False)
    
    # Passed parameters checking function
    create_begin_block_src_comment.stypy_localization = localization
    create_begin_block_src_comment.stypy_type_of_self = None
    create_begin_block_src_comment.stypy_type_store = module_type_store
    create_begin_block_src_comment.stypy_function_name = 'create_begin_block_src_comment'
    create_begin_block_src_comment.stypy_param_names_list = ['comment_txt']
    create_begin_block_src_comment.stypy_varargs_param_name = None
    create_begin_block_src_comment.stypy_kwargs_param_name = None
    create_begin_block_src_comment.stypy_call_defaults = defaults
    create_begin_block_src_comment.stypy_call_varargs = varargs
    create_begin_block_src_comment.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_begin_block_src_comment', ['comment_txt'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_begin_block_src_comment', localization, ['comment_txt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_begin_block_src_comment(...)' code ##################

    str_17027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, (-1)), 'str', '\n    Creates a Python comment with comment_txt to init a block of code\n    ')
    
    # Call to __create_src_comment(...): (line 145)
    # Processing the call arguments (line 145)
    str_17029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 32), 'str', '\n# ')
    # Getting the type of 'comment_txt' (line 145)
    comment_txt_17030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 41), 'comment_txt', False)
    # Applying the binary operator '+' (line 145)
    result_add_17031 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 32), '+', str_17029, comment_txt_17030)
    
    # Processing the call keyword arguments (line 145)
    kwargs_17032 = {}
    # Getting the type of '__create_src_comment' (line 145)
    create_src_comment_17028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), '__create_src_comment', False)
    # Calling __create_src_comment(args, kwargs) (line 145)
    create_src_comment_call_result_17033 = invoke(stypy.reporting.localization.Localization(__file__, 145, 11), create_src_comment_17028, *[result_add_17031], **kwargs_17032)
    
    # Assigning a type to the variable 'stypy_return_type' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type', create_src_comment_call_result_17033)
    
    # ################# End of 'create_begin_block_src_comment(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_begin_block_src_comment' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_17034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17034)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_begin_block_src_comment'
    return stypy_return_type_17034

# Assigning a type to the variable 'create_begin_block_src_comment' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'create_begin_block_src_comment', create_begin_block_src_comment)

@norecursion
def create_end_block_src_comment(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_end_block_src_comment'
    module_type_store = module_type_store.open_function_context('create_end_block_src_comment', 148, 0, False)
    
    # Passed parameters checking function
    create_end_block_src_comment.stypy_localization = localization
    create_end_block_src_comment.stypy_type_of_self = None
    create_end_block_src_comment.stypy_type_store = module_type_store
    create_end_block_src_comment.stypy_function_name = 'create_end_block_src_comment'
    create_end_block_src_comment.stypy_param_names_list = ['comment_txt']
    create_end_block_src_comment.stypy_varargs_param_name = None
    create_end_block_src_comment.stypy_kwargs_param_name = None
    create_end_block_src_comment.stypy_call_defaults = defaults
    create_end_block_src_comment.stypy_call_varargs = varargs
    create_end_block_src_comment.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_end_block_src_comment', ['comment_txt'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_end_block_src_comment', localization, ['comment_txt'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_end_block_src_comment(...)' code ##################

    str_17035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, (-1)), 'str', '\n    Creates a Python comment with comment_txt to finish a block of code\n    ')
    
    # Call to __create_src_comment(...): (line 152)
    # Processing the call arguments (line 152)
    str_17037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 32), 'str', '# ')
    # Getting the type of 'comment_txt' (line 152)
    comment_txt_17038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 39), 'comment_txt', False)
    # Applying the binary operator '+' (line 152)
    result_add_17039 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 32), '+', str_17037, comment_txt_17038)
    
    str_17040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 53), 'str', '\n')
    # Applying the binary operator '+' (line 152)
    result_add_17041 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 51), '+', result_add_17039, str_17040)
    
    # Processing the call keyword arguments (line 152)
    kwargs_17042 = {}
    # Getting the type of '__create_src_comment' (line 152)
    create_src_comment_17036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 11), '__create_src_comment', False)
    # Calling __create_src_comment(args, kwargs) (line 152)
    create_src_comment_call_result_17043 = invoke(stypy.reporting.localization.Localization(__file__, 152, 11), create_src_comment_17036, *[result_add_17041], **kwargs_17042)
    
    # Assigning a type to the variable 'stypy_return_type' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type', create_src_comment_call_result_17043)
    
    # ################# End of 'create_end_block_src_comment(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_end_block_src_comment' in the type store
    # Getting the type of 'stypy_return_type' (line 148)
    stypy_return_type_17044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17044)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_end_block_src_comment'
    return stypy_return_type_17044

# Assigning a type to the variable 'create_end_block_src_comment' (line 148)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'create_end_block_src_comment', create_end_block_src_comment)

@norecursion
def create_original_code_comment(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_original_code_comment'
    module_type_store = module_type_store.open_function_context('create_original_code_comment', 155, 0, False)
    
    # Passed parameters checking function
    create_original_code_comment.stypy_localization = localization
    create_original_code_comment.stypy_type_of_self = None
    create_original_code_comment.stypy_type_store = module_type_store
    create_original_code_comment.stypy_function_name = 'create_original_code_comment'
    create_original_code_comment.stypy_param_names_list = ['file_name', 'original_code']
    create_original_code_comment.stypy_varargs_param_name = None
    create_original_code_comment.stypy_kwargs_param_name = None
    create_original_code_comment.stypy_call_defaults = defaults
    create_original_code_comment.stypy_call_varargs = varargs
    create_original_code_comment.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_original_code_comment', ['file_name', 'original_code'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_original_code_comment', localization, ['file_name', 'original_code'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_original_code_comment(...)' code ##################

    str_17045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, (-1)), 'str', '\n    Creates a Python block comment with the original source file code\n    ')
    
    # Assigning a Call to a Name (line 160):
    
    # Assigning a Call to a Name (line 160):
    
    # Call to replace(...): (line 160)
    # Processing the call arguments (line 160)
    str_17048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 42), 'str', '"""')
    str_17049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 52), 'str', "'''")
    # Processing the call keyword arguments (line 160)
    kwargs_17050 = {}
    # Getting the type of 'original_code' (line 160)
    original_code_17046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 20), 'original_code', False)
    # Obtaining the member 'replace' of a type (line 160)
    replace_17047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 20), original_code_17046, 'replace')
    # Calling replace(args, kwargs) (line 160)
    replace_call_result_17051 = invoke(stypy.reporting.localization.Localization(__file__, 160, 20), replace_17047, *[str_17048, str_17049], **kwargs_17050)
    
    # Assigning a type to the variable 'original_code' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'original_code', replace_call_result_17051)
    
    # Assigning a Call to a Name (line 162):
    
    # Assigning a Call to a Name (line 162):
    
    # Call to put_line_numbers_to_module_code(...): (line 162)
    # Processing the call arguments (line 162)
    # Getting the type of 'file_name' (line 162)
    file_name_17054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 81), 'file_name', False)
    # Getting the type of 'original_code' (line 162)
    original_code_17055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 92), 'original_code', False)
    # Processing the call keyword arguments (line 162)
    kwargs_17056 = {}
    # Getting the type of 'ModuleLineNumbering' (line 162)
    ModuleLineNumbering_17052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'ModuleLineNumbering', False)
    # Obtaining the member 'put_line_numbers_to_module_code' of a type (line 162)
    put_line_numbers_to_module_code_17053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 29), ModuleLineNumbering_17052, 'put_line_numbers_to_module_code')
    # Calling put_line_numbers_to_module_code(args, kwargs) (line 162)
    put_line_numbers_to_module_code_call_result_17057 = invoke(stypy.reporting.localization.Localization(__file__, 162, 29), put_line_numbers_to_module_code_17053, *[file_name_17054, original_code_17055], **kwargs_17056)
    
    # Assigning a type to the variable 'numbered_original_code' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'numbered_original_code', put_line_numbers_to_module_code_call_result_17057)
    
    # Assigning a Call to a Name (line 164):
    
    # Assigning a Call to a Name (line 164):
    
    # Call to create_Name(...): (line 164)
    # Processing the call arguments (line 164)
    str_17060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 8), 'str', '"""\nORIGINAL PROGRAM SOURCE CODE:\n')
    # Getting the type of 'numbered_original_code' (line 165)
    numbered_original_code_17061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 52), 'numbered_original_code', False)
    # Applying the binary operator '+' (line 165)
    result_add_17062 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 8), '+', str_17060, numbered_original_code_17061)
    
    str_17063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 77), 'str', '\n"""\n')
    # Applying the binary operator '+' (line 165)
    result_add_17064 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 75), '+', result_add_17062, str_17063)
    
    # Processing the call keyword arguments (line 164)
    kwargs_17065 = {}
    # Getting the type of 'core_language_copy' (line 164)
    core_language_copy_17058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 18), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 164)
    create_Name_17059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 18), core_language_copy_17058, 'create_Name')
    # Calling create_Name(args, kwargs) (line 164)
    create_Name_call_result_17066 = invoke(stypy.reporting.localization.Localization(__file__, 164, 18), create_Name_17059, *[result_add_17064], **kwargs_17065)
    
    # Assigning a type to the variable 'comment_txt' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'comment_txt', create_Name_call_result_17066)
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to Expr(...): (line 166)
    # Processing the call keyword arguments (line 166)
    kwargs_17069 = {}
    # Getting the type of 'ast' (line 166)
    ast_17067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 166)
    Expr_17068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 22), ast_17067, 'Expr')
    # Calling Expr(args, kwargs) (line 166)
    Expr_call_result_17070 = invoke(stypy.reporting.localization.Localization(__file__, 166, 22), Expr_17068, *[], **kwargs_17069)
    
    # Assigning a type to the variable 'initial_comment' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'initial_comment', Expr_call_result_17070)
    
    # Assigning a Name to a Attribute (line 167):
    
    # Assigning a Name to a Attribute (line 167):
    # Getting the type of 'comment_txt' (line 167)
    comment_txt_17071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 28), 'comment_txt')
    # Getting the type of 'initial_comment' (line 167)
    initial_comment_17072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'initial_comment')
    # Setting the type of the member 'value' of a type (line 167)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 4), initial_comment_17072, 'value', comment_txt_17071)
    # Getting the type of 'initial_comment' (line 169)
    initial_comment_17073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 11), 'initial_comment')
    # Assigning a type to the variable 'stypy_return_type' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'stypy_return_type', initial_comment_17073)
    
    # ################# End of 'create_original_code_comment(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_original_code_comment' in the type store
    # Getting the type of 'stypy_return_type' (line 155)
    stypy_return_type_17074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17074)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_original_code_comment'
    return stypy_return_type_17074

# Assigning a type to the variable 'create_original_code_comment' (line 155)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 0), 'create_original_code_comment', create_original_code_comment)

@norecursion
def flatten_lists(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'flatten_lists'
    module_type_store = module_type_store.open_function_context('flatten_lists', 174, 0, False)
    
    # Passed parameters checking function
    flatten_lists.stypy_localization = localization
    flatten_lists.stypy_type_of_self = None
    flatten_lists.stypy_type_store = module_type_store
    flatten_lists.stypy_function_name = 'flatten_lists'
    flatten_lists.stypy_param_names_list = []
    flatten_lists.stypy_varargs_param_name = 'args'
    flatten_lists.stypy_kwargs_param_name = None
    flatten_lists.stypy_call_defaults = defaults
    flatten_lists.stypy_call_varargs = varargs
    flatten_lists.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'flatten_lists', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'flatten_lists', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'flatten_lists(...)' code ##################

    str_17075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, (-1)), 'str', '\n    Recursive function to convert a list of lists into a single "flattened" list, mostly used to streamline lists\n    of instructions that can contain other instruction lists\n    ')
    
    
    # Call to len(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'args' (line 179)
    args_17077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 11), 'args', False)
    # Processing the call keyword arguments (line 179)
    kwargs_17078 = {}
    # Getting the type of 'len' (line 179)
    len_17076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 7), 'len', False)
    # Calling len(args, kwargs) (line 179)
    len_call_result_17079 = invoke(stypy.reporting.localization.Localization(__file__, 179, 7), len_17076, *[args_17077], **kwargs_17078)
    
    int_17080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 20), 'int')
    # Applying the binary operator '==' (line 179)
    result_eq_17081 = python_operator(stypy.reporting.localization.Localization(__file__, 179, 7), '==', len_call_result_17079, int_17080)
    
    # Testing if the type of an if condition is none (line 179)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 179, 4), result_eq_17081):
        pass
    else:
        
        # Testing the type of an if condition (line 179)
        if_condition_17082 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 179, 4), result_eq_17081)
        # Assigning a type to the variable 'if_condition_17082' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'if_condition_17082', if_condition_17082)
        # SSA begins for if statement (line 179)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 180)
        list_17083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 180)
        
        # Assigning a type to the variable 'stypy_return_type' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'stypy_return_type', list_17083)
        # SSA join for if statement (line 179)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Type idiom detected: calculating its left and rigth part (line 181)
    # Getting the type of 'list' (line 181)
    list_17084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 27), 'list')
    
    # Obtaining the type of the subscript
    int_17085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 23), 'int')
    # Getting the type of 'args' (line 181)
    args_17086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 18), 'args')
    # Obtaining the member '__getitem__' of a type (line 181)
    getitem___17087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 18), args_17086, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 181)
    subscript_call_result_17088 = invoke(stypy.reporting.localization.Localization(__file__, 181, 18), getitem___17087, int_17085)
    
    
    (may_be_17089, more_types_in_union_17090) = may_be_subtype(list_17084, subscript_call_result_17088)

    if may_be_17089:

        if more_types_in_union_17090:
            # Runtime conditional SSA (line 181)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 182):
        
        # Assigning a BinOp to a Name (line 182):
        
        # Obtaining the type of the subscript
        int_17091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 25), 'int')
        # Getting the type of 'args' (line 182)
        args_17092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'args')
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___17093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 20), args_17092, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_17094 = invoke(stypy.reporting.localization.Localization(__file__, 182, 20), getitem___17093, int_17091)
        
        
        # Call to list(...): (line 182)
        # Processing the call arguments (line 182)
        
        # Obtaining the type of the subscript
        int_17096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 40), 'int')
        slice_17097 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 182, 35), int_17096, None, None)
        # Getting the type of 'args' (line 182)
        args_17098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 35), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___17099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 35), args_17098, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_17100 = invoke(stypy.reporting.localization.Localization(__file__, 182, 35), getitem___17099, slice_17097)
        
        # Processing the call keyword arguments (line 182)
        kwargs_17101 = {}
        # Getting the type of 'list' (line 182)
        list_17095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'list', False)
        # Calling list(args, kwargs) (line 182)
        list_call_result_17102 = invoke(stypy.reporting.localization.Localization(__file__, 182, 30), list_17095, *[subscript_call_result_17100], **kwargs_17101)
        
        # Applying the binary operator '+' (line 182)
        result_add_17103 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 20), '+', subscript_call_result_17094, list_call_result_17102)
        
        # Assigning a type to the variable 'arguments' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'arguments', result_add_17103)
        
        # Call to flatten_lists(...): (line 183)
        # Getting the type of 'arguments' (line 183)
        arguments_17105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 30), 'arguments', False)
        # Processing the call keyword arguments (line 183)
        kwargs_17106 = {}
        # Getting the type of 'flatten_lists' (line 183)
        flatten_lists_17104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'flatten_lists', False)
        # Calling flatten_lists(args, kwargs) (line 183)
        flatten_lists_call_result_17107 = invoke(stypy.reporting.localization.Localization(__file__, 183, 15), flatten_lists_17104, *[arguments_17105], **kwargs_17106)
        
        # Assigning a type to the variable 'stypy_return_type' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stypy_return_type', flatten_lists_call_result_17107)

        if more_types_in_union_17090:
            # SSA join for if statement (line 181)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'list' (line 184)
    list_17108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 184)
    # Adding element type (line 184)
    
    # Obtaining the type of the subscript
    int_17109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 17), 'int')
    # Getting the type of 'args' (line 184)
    args_17110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'args')
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___17111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 12), args_17110, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_17112 = invoke(stypy.reporting.localization.Localization(__file__, 184, 12), getitem___17111, int_17109)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 11), list_17108, subscript_call_result_17112)
    
    
    # Call to flatten_lists(...): (line 184)
    
    # Obtaining the type of the subscript
    int_17114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 43), 'int')
    slice_17115 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 184, 38), int_17114, None, None)
    # Getting the type of 'args' (line 184)
    args_17116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 38), 'args', False)
    # Obtaining the member '__getitem__' of a type (line 184)
    getitem___17117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 38), args_17116, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 184)
    subscript_call_result_17118 = invoke(stypy.reporting.localization.Localization(__file__, 184, 38), getitem___17117, slice_17115)
    
    # Processing the call keyword arguments (line 184)
    kwargs_17119 = {}
    # Getting the type of 'flatten_lists' (line 184)
    flatten_lists_17113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 23), 'flatten_lists', False)
    # Calling flatten_lists(args, kwargs) (line 184)
    flatten_lists_call_result_17120 = invoke(stypy.reporting.localization.Localization(__file__, 184, 23), flatten_lists_17113, *[subscript_call_result_17118], **kwargs_17119)
    
    # Applying the binary operator '+' (line 184)
    result_add_17121 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 11), '+', list_17108, flatten_lists_call_result_17120)
    
    # Assigning a type to the variable 'stypy_return_type' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type', result_add_17121)
    
    # ################# End of 'flatten_lists(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'flatten_lists' in the type store
    # Getting the type of 'stypy_return_type' (line 174)
    stypy_return_type_17122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17122)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'flatten_lists'
    return stypy_return_type_17122

# Assigning a type to the variable 'flatten_lists' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'flatten_lists', flatten_lists)

@norecursion
def create_print_var(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_print_var'
    module_type_store = module_type_store.open_function_context('create_print_var', 187, 0, False)
    
    # Passed parameters checking function
    create_print_var.stypy_localization = localization
    create_print_var.stypy_type_of_self = None
    create_print_var.stypy_type_store = module_type_store
    create_print_var.stypy_function_name = 'create_print_var'
    create_print_var.stypy_param_names_list = ['variable']
    create_print_var.stypy_varargs_param_name = None
    create_print_var.stypy_kwargs_param_name = None
    create_print_var.stypy_call_defaults = defaults
    create_print_var.stypy_call_varargs = varargs
    create_print_var.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_print_var', ['variable'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_print_var', localization, ['variable'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_print_var(...)' code ##################

    str_17123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, (-1)), 'str', '\n    Creates a node to print a variable\n    ')
    
    # Assigning a Call to a Name (line 191):
    
    # Assigning a Call to a Name (line 191):
    
    # Call to Print(...): (line 191)
    # Processing the call keyword arguments (line 191)
    kwargs_17126 = {}
    # Getting the type of 'ast' (line 191)
    ast_17124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'ast', False)
    # Obtaining the member 'Print' of a type (line 191)
    Print_17125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 11), ast_17124, 'Print')
    # Calling Print(args, kwargs) (line 191)
    Print_call_result_17127 = invoke(stypy.reporting.localization.Localization(__file__, 191, 11), Print_17125, *[], **kwargs_17126)
    
    # Assigning a type to the variable 'node' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'node', Print_call_result_17127)
    
    # Assigning a Name to a Attribute (line 192):
    
    # Assigning a Name to a Attribute (line 192):
    # Getting the type of 'True' (line 192)
    True_17128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 14), 'True')
    # Getting the type of 'node' (line 192)
    node_17129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'node')
    # Setting the type of the member 'nl' of a type (line 192)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 4), node_17129, 'nl', True_17128)
    
    # Assigning a Name to a Attribute (line 193):
    
    # Assigning a Name to a Attribute (line 193):
    # Getting the type of 'None' (line 193)
    None_17130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 16), 'None')
    # Getting the type of 'node' (line 193)
    node_17131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'node')
    # Setting the type of the member 'dest' of a type (line 193)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 4), node_17131, 'dest', None_17130)
    
    # Assigning a List to a Attribute (line 194):
    
    # Assigning a List to a Attribute (line 194):
    
    # Obtaining an instance of the builtin type 'list' (line 194)
    list_17132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 194)
    # Adding element type (line 194)
    
    # Call to create_Name(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'variable' (line 194)
    variable_17135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 50), 'variable', False)
    # Processing the call keyword arguments (line 194)
    kwargs_17136 = {}
    # Getting the type of 'core_language_copy' (line 194)
    core_language_copy_17133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 19), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 194)
    create_Name_17134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 19), core_language_copy_17133, 'create_Name')
    # Calling create_Name(args, kwargs) (line 194)
    create_Name_call_result_17137 = invoke(stypy.reporting.localization.Localization(__file__, 194, 19), create_Name_17134, *[variable_17135], **kwargs_17136)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 194, 18), list_17132, create_Name_call_result_17137)
    
    # Getting the type of 'node' (line 194)
    node_17138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'node')
    # Setting the type of the member 'values' of a type (line 194)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 4), node_17138, 'values', list_17132)
    # Getting the type of 'node' (line 196)
    node_17139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 11), 'node')
    # Assigning a type to the variable 'stypy_return_type' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type', node_17139)
    
    # ################# End of 'create_print_var(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_print_var' in the type store
    # Getting the type of 'stypy_return_type' (line 187)
    stypy_return_type_17140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17140)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_print_var'
    return stypy_return_type_17140

# Assigning a type to the variable 'create_print_var' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'create_print_var', create_print_var)

@norecursion
def assign_line_and_column(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assign_line_and_column'
    module_type_store = module_type_store.open_function_context('assign_line_and_column', 199, 0, False)
    
    # Passed parameters checking function
    assign_line_and_column.stypy_localization = localization
    assign_line_and_column.stypy_type_of_self = None
    assign_line_and_column.stypy_type_store = module_type_store
    assign_line_and_column.stypy_function_name = 'assign_line_and_column'
    assign_line_and_column.stypy_param_names_list = ['dest_node', 'src_node']
    assign_line_and_column.stypy_varargs_param_name = None
    assign_line_and_column.stypy_kwargs_param_name = None
    assign_line_and_column.stypy_call_defaults = defaults
    assign_line_and_column.stypy_call_varargs = varargs
    assign_line_and_column.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assign_line_and_column', ['dest_node', 'src_node'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assign_line_and_column', localization, ['dest_node', 'src_node'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assign_line_and_column(...)' code ##################

    str_17141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, (-1)), 'str', '\n    Assign to dest_node the same source line and column of src_node\n    ')
    
    # Assigning a Attribute to a Attribute (line 203):
    
    # Assigning a Attribute to a Attribute (line 203):
    # Getting the type of 'src_node' (line 203)
    src_node_17142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 23), 'src_node')
    # Obtaining the member 'lineno' of a type (line 203)
    lineno_17143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 23), src_node_17142, 'lineno')
    # Getting the type of 'dest_node' (line 203)
    dest_node_17144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'dest_node')
    # Setting the type of the member 'lineno' of a type (line 203)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 4), dest_node_17144, 'lineno', lineno_17143)
    
    # Assigning a Attribute to a Attribute (line 204):
    
    # Assigning a Attribute to a Attribute (line 204):
    # Getting the type of 'src_node' (line 204)
    src_node_17145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 27), 'src_node')
    # Obtaining the member 'col_offset' of a type (line 204)
    col_offset_17146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 27), src_node_17145, 'col_offset')
    # Getting the type of 'dest_node' (line 204)
    dest_node_17147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'dest_node')
    # Setting the type of the member 'col_offset' of a type (line 204)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 4), dest_node_17147, 'col_offset', col_offset_17146)
    
    # ################# End of 'assign_line_and_column(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assign_line_and_column' in the type store
    # Getting the type of 'stypy_return_type' (line 199)
    stypy_return_type_17148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17148)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assign_line_and_column'
    return stypy_return_type_17148

# Assigning a type to the variable 'assign_line_and_column' (line 199)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 0), 'assign_line_and_column', assign_line_and_column)

@norecursion
def create_localization(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_localization'
    module_type_store = module_type_store.open_function_context('create_localization', 207, 0, False)
    
    # Passed parameters checking function
    create_localization.stypy_localization = localization
    create_localization.stypy_type_of_self = None
    create_localization.stypy_type_store = module_type_store
    create_localization.stypy_function_name = 'create_localization'
    create_localization.stypy_param_names_list = ['line', 'col']
    create_localization.stypy_varargs_param_name = None
    create_localization.stypy_kwargs_param_name = None
    create_localization.stypy_call_defaults = defaults
    create_localization.stypy_call_varargs = varargs
    create_localization.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_localization', ['line', 'col'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_localization', localization, ['line', 'col'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_localization(...)' code ##################

    str_17149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, (-1)), 'str', '\n    Creates AST Nodes that creates a new Localization instance\n    ')
    
    # Assigning a Call to a Name (line 211):
    
    # Assigning a Call to a Name (line 211):
    
    # Call to create_num(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'line' (line 211)
    line_17152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 42), 'line', False)
    # Processing the call keyword arguments (line 211)
    kwargs_17153 = {}
    # Getting the type of 'core_language_copy' (line 211)
    core_language_copy_17150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'core_language_copy', False)
    # Obtaining the member 'create_num' of a type (line 211)
    create_num_17151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), core_language_copy_17150, 'create_num')
    # Calling create_num(args, kwargs) (line 211)
    create_num_call_result_17154 = invoke(stypy.reporting.localization.Localization(__file__, 211, 12), create_num_17151, *[line_17152], **kwargs_17153)
    
    # Assigning a type to the variable 'linen' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'linen', create_num_call_result_17154)
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to create_num(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'col' (line 212)
    col_17157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 41), 'col', False)
    # Processing the call keyword arguments (line 212)
    kwargs_17158 = {}
    # Getting the type of 'core_language_copy' (line 212)
    core_language_copy_17155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 11), 'core_language_copy', False)
    # Obtaining the member 'create_num' of a type (line 212)
    create_num_17156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 11), core_language_copy_17155, 'create_num')
    # Calling create_num(args, kwargs) (line 212)
    create_num_call_result_17159 = invoke(stypy.reporting.localization.Localization(__file__, 212, 11), create_num_17156, *[col_17157], **kwargs_17158)
    
    # Assigning a type to the variable 'coln' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'coln', create_num_call_result_17159)
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to create_Name(...): (line 213)
    # Processing the call arguments (line 213)
    str_17162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 48), 'str', '__file__')
    # Processing the call keyword arguments (line 213)
    kwargs_17163 = {}
    # Getting the type of 'core_language_copy' (line 213)
    core_language_copy_17160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 17), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 213)
    create_Name_17161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 17), core_language_copy_17160, 'create_Name')
    # Calling create_Name(args, kwargs) (line 213)
    create_Name_call_result_17164 = invoke(stypy.reporting.localization.Localization(__file__, 213, 17), create_Name_17161, *[str_17162], **kwargs_17163)
    
    # Assigning a type to the variable 'file_namen' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'file_namen', create_Name_call_result_17164)
    
    # Assigning a Call to a Name (line 214):
    
    # Assigning a Call to a Name (line 214):
    
    # Call to create_Name(...): (line 214)
    # Processing the call arguments (line 214)
    str_17167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 47), 'str', 'stypy.python_lib.python_types.type_inference.localization.Localization')
    # Processing the call keyword arguments (line 214)
    kwargs_17168 = {}
    # Getting the type of 'core_language_copy' (line 214)
    core_language_copy_17165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 214)
    create_Name_17166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 16), core_language_copy_17165, 'create_Name')
    # Calling create_Name(args, kwargs) (line 214)
    create_Name_call_result_17169 = invoke(stypy.reporting.localization.Localization(__file__, 214, 16), create_Name_17166, *[str_17167], **kwargs_17168)
    
    # Assigning a type to the variable 'loc_namen' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'loc_namen', create_Name_call_result_17169)
    
    # Assigning a Call to a Name (line 215):
    
    # Assigning a Call to a Name (line 215):
    
    # Call to create_call(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'loc_namen' (line 215)
    loc_namen_17172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 42), 'loc_namen', False)
    
    # Obtaining an instance of the builtin type 'list' (line 215)
    list_17173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 215)
    # Adding element type (line 215)
    # Getting the type of 'file_namen' (line 215)
    file_namen_17174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 54), 'file_namen', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 53), list_17173, file_namen_17174)
    # Adding element type (line 215)
    # Getting the type of 'linen' (line 215)
    linen_17175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 66), 'linen', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 53), list_17173, linen_17175)
    # Adding element type (line 215)
    # Getting the type of 'coln' (line 215)
    coln_17176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 73), 'coln', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 53), list_17173, coln_17176)
    
    # Processing the call keyword arguments (line 215)
    kwargs_17177 = {}
    # Getting the type of 'functions_copy' (line 215)
    functions_copy_17170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'functions_copy', False)
    # Obtaining the member 'create_call' of a type (line 215)
    create_call_17171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 15), functions_copy_17170, 'create_call')
    # Calling create_call(args, kwargs) (line 215)
    create_call_call_result_17178 = invoke(stypy.reporting.localization.Localization(__file__, 215, 15), create_call_17171, *[loc_namen_17172, list_17173], **kwargs_17177)
    
    # Assigning a type to the variable 'loc_call' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'loc_call', create_call_call_result_17178)
    # Getting the type of 'loc_call' (line 217)
    loc_call_17179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 11), 'loc_call')
    # Assigning a type to the variable 'stypy_return_type' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type', loc_call_17179)
    
    # ################# End of 'create_localization(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_localization' in the type store
    # Getting the type of 'stypy_return_type' (line 207)
    stypy_return_type_17180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17180)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_localization'
    return stypy_return_type_17180

# Assigning a type to the variable 'create_localization' (line 207)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 0), 'create_localization', create_localization)

@norecursion
def create_import_stypy(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_import_stypy'
    module_type_store = module_type_store.open_function_context('create_import_stypy', 220, 0, False)
    
    # Passed parameters checking function
    create_import_stypy.stypy_localization = localization
    create_import_stypy.stypy_type_of_self = None
    create_import_stypy.stypy_type_store = module_type_store
    create_import_stypy.stypy_function_name = 'create_import_stypy'
    create_import_stypy.stypy_param_names_list = []
    create_import_stypy.stypy_varargs_param_name = None
    create_import_stypy.stypy_kwargs_param_name = None
    create_import_stypy.stypy_call_defaults = defaults
    create_import_stypy.stypy_call_varargs = varargs
    create_import_stypy.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_import_stypy', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_import_stypy', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_import_stypy(...)' code ##################

    str_17181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, (-1)), 'str', '\n    Creates AST Nodes that encode "from stypy import *"\n    ')
    
    # Assigning a Call to a Name (line 224):
    
    # Assigning a Call to a Name (line 224):
    
    # Call to create_alias(...): (line 224)
    # Processing the call arguments (line 224)
    str_17184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 44), 'str', '*')
    # Processing the call keyword arguments (line 224)
    kwargs_17185 = {}
    # Getting the type of 'core_language_copy' (line 224)
    core_language_copy_17182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'core_language_copy', False)
    # Obtaining the member 'create_alias' of a type (line 224)
    create_alias_17183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 12), core_language_copy_17182, 'create_alias')
    # Calling create_alias(args, kwargs) (line 224)
    create_alias_call_result_17186 = invoke(stypy.reporting.localization.Localization(__file__, 224, 12), create_alias_17183, *[str_17184], **kwargs_17185)
    
    # Assigning a type to the variable 'alias' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'alias', create_alias_call_result_17186)
    
    # Assigning a Call to a Name (line 225):
    
    # Assigning a Call to a Name (line 225):
    
    # Call to create_importfrom(...): (line 225)
    # Processing the call arguments (line 225)
    str_17189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 54), 'str', 'stypy')
    # Getting the type of 'alias' (line 225)
    alias_17190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 63), 'alias', False)
    # Processing the call keyword arguments (line 225)
    kwargs_17191 = {}
    # Getting the type of 'core_language_copy' (line 225)
    core_language_copy_17187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 17), 'core_language_copy', False)
    # Obtaining the member 'create_importfrom' of a type (line 225)
    create_importfrom_17188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 17), core_language_copy_17187, 'create_importfrom')
    # Calling create_importfrom(args, kwargs) (line 225)
    create_importfrom_call_result_17192 = invoke(stypy.reporting.localization.Localization(__file__, 225, 17), create_importfrom_17188, *[str_17189, alias_17190], **kwargs_17191)
    
    # Assigning a type to the variable 'importfrom' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'importfrom', create_importfrom_call_result_17192)
    # Getting the type of 'importfrom' (line 227)
    importfrom_17193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'importfrom')
    # Assigning a type to the variable 'stypy_return_type' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type', importfrom_17193)
    
    # ################# End of 'create_import_stypy(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_import_stypy' in the type store
    # Getting the type of 'stypy_return_type' (line 220)
    stypy_return_type_17194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17194)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_import_stypy'
    return stypy_return_type_17194

# Assigning a type to the variable 'create_import_stypy' (line 220)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 0), 'create_import_stypy', create_import_stypy)

@norecursion
def create_print_errors(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_print_errors'
    module_type_store = module_type_store.open_function_context('create_print_errors', 230, 0, False)
    
    # Passed parameters checking function
    create_print_errors.stypy_localization = localization
    create_print_errors.stypy_type_of_self = None
    create_print_errors.stypy_type_store = module_type_store
    create_print_errors.stypy_function_name = 'create_print_errors'
    create_print_errors.stypy_param_names_list = []
    create_print_errors.stypy_varargs_param_name = None
    create_print_errors.stypy_kwargs_param_name = None
    create_print_errors.stypy_call_defaults = defaults
    create_print_errors.stypy_call_varargs = varargs
    create_print_errors.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_print_errors', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_print_errors', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_print_errors(...)' code ##################

    str_17195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, (-1)), 'str', '\n    Creates AST Nodes that encode "ErrorType.print_error_msgs()"\n    ')
    
    # Assigning a Call to a Name (line 234):
    
    # Assigning a Call to a Name (line 234):
    
    # Call to create_attribute(...): (line 234)
    # Processing the call arguments (line 234)
    str_17198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 52), 'str', 'ErrorType')
    str_17199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 65), 'str', 'print_error_msgs')
    # Processing the call keyword arguments (line 234)
    kwargs_17200 = {}
    # Getting the type of 'core_language_copy' (line 234)
    core_language_copy_17196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 16), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 234)
    create_attribute_17197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 16), core_language_copy_17196, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 234)
    create_attribute_call_result_17201 = invoke(stypy.reporting.localization.Localization(__file__, 234, 16), create_attribute_17197, *[str_17198, str_17199], **kwargs_17200)
    
    # Assigning a type to the variable 'attribute' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'attribute', create_attribute_call_result_17201)
    
    # Assigning a Call to a Name (line 235):
    
    # Assigning a Call to a Name (line 235):
    
    # Call to Expr(...): (line 235)
    # Processing the call keyword arguments (line 235)
    kwargs_17204 = {}
    # Getting the type of 'ast' (line 235)
    ast_17202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 11), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 235)
    Expr_17203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 11), ast_17202, 'Expr')
    # Calling Expr(args, kwargs) (line 235)
    Expr_call_result_17205 = invoke(stypy.reporting.localization.Localization(__file__, 235, 11), Expr_17203, *[], **kwargs_17204)
    
    # Assigning a type to the variable 'expr' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'expr', Expr_call_result_17205)
    
    # Assigning a Call to a Attribute (line 236):
    
    # Assigning a Call to a Attribute (line 236):
    
    # Call to create_call(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'attribute' (line 236)
    attribute_17208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 44), 'attribute', False)
    
    # Obtaining an instance of the builtin type 'list' (line 236)
    list_17209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 55), 'list')
    # Adding type elements to the builtin type 'list' instance (line 236)
    
    # Processing the call keyword arguments (line 236)
    kwargs_17210 = {}
    # Getting the type of 'functions_copy' (line 236)
    functions_copy_17206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 17), 'functions_copy', False)
    # Obtaining the member 'create_call' of a type (line 236)
    create_call_17207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 17), functions_copy_17206, 'create_call')
    # Calling create_call(args, kwargs) (line 236)
    create_call_call_result_17211 = invoke(stypy.reporting.localization.Localization(__file__, 236, 17), create_call_17207, *[attribute_17208, list_17209], **kwargs_17210)
    
    # Getting the type of 'expr' (line 236)
    expr_17212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'expr')
    # Setting the type of the member 'value' of a type (line 236)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 4), expr_17212, 'value', create_call_call_result_17211)
    # Getting the type of 'expr' (line 238)
    expr_17213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 11), 'expr')
    # Assigning a type to the variable 'stypy_return_type' (line 238)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'stypy_return_type', expr_17213)
    
    # ################# End of 'create_print_errors(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_print_errors' in the type store
    # Getting the type of 'stypy_return_type' (line 230)
    stypy_return_type_17214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17214)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_print_errors'
    return stypy_return_type_17214

# Assigning a type to the variable 'create_print_errors' (line 230)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'create_print_errors', create_print_errors)

@norecursion
def create_default_return_variable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_default_return_variable'
    module_type_store = module_type_store.open_function_context('create_default_return_variable', 241, 0, False)
    
    # Passed parameters checking function
    create_default_return_variable.stypy_localization = localization
    create_default_return_variable.stypy_type_of_self = None
    create_default_return_variable.stypy_type_store = module_type_store
    create_default_return_variable.stypy_function_name = 'create_default_return_variable'
    create_default_return_variable.stypy_param_names_list = []
    create_default_return_variable.stypy_varargs_param_name = None
    create_default_return_variable.stypy_kwargs_param_name = None
    create_default_return_variable.stypy_call_defaults = defaults
    create_default_return_variable.stypy_call_varargs = varargs
    create_default_return_variable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_default_return_variable', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_default_return_variable', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_default_return_variable(...)' code ##################

    str_17215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, (-1)), 'str', '\n    Creates AST Nodes that adds the default return variable to a function. Functions of generated type inference\n     programs only has a return clause\n    ')
    
    # Assigning a Call to a Name (line 246):
    
    # Assigning a Call to a Name (line 246):
    
    # Call to create_Name(...): (line 246)
    # Processing the call arguments (line 246)
    # Getting the type of 'default_function_ret_var_name' (line 246)
    default_function_ret_var_name_17218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 51), 'default_function_ret_var_name', False)
    # Getting the type of 'False' (line 246)
    False_17219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 82), 'False', False)
    # Processing the call keyword arguments (line 246)
    kwargs_17220 = {}
    # Getting the type of 'core_language_copy' (line 246)
    core_language_copy_17216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 246)
    create_Name_17217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 20), core_language_copy_17216, 'create_Name')
    # Calling create_Name(args, kwargs) (line 246)
    create_Name_call_result_17221 = invoke(stypy.reporting.localization.Localization(__file__, 246, 20), create_Name_17217, *[default_function_ret_var_name_17218, False_17219], **kwargs_17220)
    
    # Assigning a type to the variable 'assign_target' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'assign_target', create_Name_call_result_17221)
    
    # Assigning a Call to a Name (line 247):
    
    # Assigning a Call to a Name (line 247):
    
    # Call to create_Assign(...): (line 247)
    # Processing the call arguments (line 247)
    # Getting the type of 'assign_target' (line 247)
    assign_target_17224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 46), 'assign_target', False)
    
    # Call to create_Name(...): (line 247)
    # Processing the call arguments (line 247)
    str_17227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 92), 'str', 'None')
    # Processing the call keyword arguments (line 247)
    kwargs_17228 = {}
    # Getting the type of 'core_language_copy' (line 247)
    core_language_copy_17225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 61), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 247)
    create_Name_17226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 61), core_language_copy_17225, 'create_Name')
    # Calling create_Name(args, kwargs) (line 247)
    create_Name_call_result_17229 = invoke(stypy.reporting.localization.Localization(__file__, 247, 61), create_Name_17226, *[str_17227], **kwargs_17228)
    
    # Processing the call keyword arguments (line 247)
    kwargs_17230 = {}
    # Getting the type of 'core_language_copy' (line 247)
    core_language_copy_17222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 13), 'core_language_copy', False)
    # Obtaining the member 'create_Assign' of a type (line 247)
    create_Assign_17223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 13), core_language_copy_17222, 'create_Assign')
    # Calling create_Assign(args, kwargs) (line 247)
    create_Assign_call_result_17231 = invoke(stypy.reporting.localization.Localization(__file__, 247, 13), create_Assign_17223, *[assign_target_17224, create_Name_call_result_17229], **kwargs_17230)
    
    # Assigning a type to the variable 'assign' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'assign', create_Assign_call_result_17231)
    # Getting the type of 'assign' (line 249)
    assign_17232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 11), 'assign')
    # Assigning a type to the variable 'stypy_return_type' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'stypy_return_type', assign_17232)
    
    # ################# End of 'create_default_return_variable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_default_return_variable' in the type store
    # Getting the type of 'stypy_return_type' (line 241)
    stypy_return_type_17233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17233)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_default_return_variable'
    return stypy_return_type_17233

# Assigning a type to the variable 'create_default_return_variable' (line 241)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 0), 'create_default_return_variable', create_default_return_variable)

@norecursion
def create_store_return_from_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_store_return_from_function'
    module_type_store = module_type_store.open_function_context('create_store_return_from_function', 252, 0, False)
    
    # Passed parameters checking function
    create_store_return_from_function.stypy_localization = localization
    create_store_return_from_function.stypy_type_of_self = None
    create_store_return_from_function.stypy_type_store = module_type_store
    create_store_return_from_function.stypy_function_name = 'create_store_return_from_function'
    create_store_return_from_function.stypy_param_names_list = ['lineno', 'col_offset']
    create_store_return_from_function.stypy_varargs_param_name = None
    create_store_return_from_function.stypy_kwargs_param_name = None
    create_store_return_from_function.stypy_call_defaults = defaults
    create_store_return_from_function.stypy_call_varargs = varargs
    create_store_return_from_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_store_return_from_function', ['lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_store_return_from_function', localization, ['lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_store_return_from_function(...)' code ##################

    
    # Assigning a Call to a Name (line 253):
    
    # Assigning a Call to a Name (line 253):
    
    # Call to create_src_comment(...): (line 253)
    # Processing the call arguments (line 253)
    str_17235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 45), 'str', 'Storing return type')
    # Getting the type of 'lineno' (line 253)
    lineno_17236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 68), 'lineno', False)
    # Processing the call keyword arguments (line 253)
    kwargs_17237 = {}
    # Getting the type of 'create_src_comment' (line 253)
    create_src_comment_17234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), 'create_src_comment', False)
    # Calling create_src_comment(args, kwargs) (line 253)
    create_src_comment_call_result_17238 = invoke(stypy.reporting.localization.Localization(__file__, 253, 26), create_src_comment_17234, *[str_17235, lineno_17236], **kwargs_17237)
    
    # Assigning a type to the variable 'set_type_of_comment' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'set_type_of_comment', create_src_comment_call_result_17238)
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to create_attribute(...): (line 254)
    # Processing the call arguments (line 254)
    # Getting the type of 'default_module_type_store_var_name' (line 254)
    default_module_type_store_var_name_17241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 61), 'default_module_type_store_var_name', False)
    str_17242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 56), 'str', 'store_return_type_of_current_context')
    # Processing the call keyword arguments (line 254)
    kwargs_17243 = {}
    # Getting the type of 'core_language_copy' (line 254)
    core_language_copy_17239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 25), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 254)
    create_attribute_17240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 25), core_language_copy_17239, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 254)
    create_attribute_call_result_17244 = invoke(stypy.reporting.localization.Localization(__file__, 254, 25), create_attribute_17240, *[default_module_type_store_var_name_17241, str_17242], **kwargs_17243)
    
    # Assigning a type to the variable 'set_type_of_method' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'set_type_of_method', create_attribute_call_result_17244)
    
    # Assigning a Call to a Name (line 257):
    
    # Assigning a Call to a Name (line 257):
    
    # Call to create_Name(...): (line 257)
    # Processing the call arguments (line 257)
    # Getting the type of 'default_function_ret_var_name' (line 257)
    default_function_ret_var_name_17247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 53), 'default_function_ret_var_name', False)
    # Processing the call keyword arguments (line 257)
    kwargs_17248 = {}
    # Getting the type of 'core_language_copy' (line 257)
    core_language_copy_17245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 22), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 257)
    create_Name_17246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 22), core_language_copy_17245, 'create_Name')
    # Calling create_Name(args, kwargs) (line 257)
    create_Name_call_result_17249 = invoke(stypy.reporting.localization.Localization(__file__, 257, 22), create_Name_17246, *[default_function_ret_var_name_17247], **kwargs_17248)
    
    # Assigning a type to the variable 'return_var_name' (line 257)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'return_var_name', create_Name_call_result_17249)
    
    # Assigning a Call to a Name (line 258):
    
    # Assigning a Call to a Name (line 258):
    
    # Call to create_call_expression(...): (line 258)
    # Processing the call arguments (line 258)
    # Getting the type of 'set_type_of_method' (line 258)
    set_type_of_method_17252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 61), 'set_type_of_method', False)
    
    # Obtaining an instance of the builtin type 'list' (line 259)
    list_17253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 56), 'list')
    # Adding type elements to the builtin type 'list' instance (line 259)
    # Adding element type (line 259)
    # Getting the type of 'return_var_name' (line 259)
    return_var_name_17254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 57), 'return_var_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 259, 56), list_17253, return_var_name_17254)
    
    # Processing the call keyword arguments (line 258)
    kwargs_17255 = {}
    # Getting the type of 'functions_copy' (line 258)
    functions_copy_17250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 23), 'functions_copy', False)
    # Obtaining the member 'create_call_expression' of a type (line 258)
    create_call_expression_17251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 23), functions_copy_17250, 'create_call_expression')
    # Calling create_call_expression(args, kwargs) (line 258)
    create_call_expression_call_result_17256 = invoke(stypy.reporting.localization.Localization(__file__, 258, 23), create_call_expression_17251, *[set_type_of_method_17252, list_17253], **kwargs_17255)
    
    # Assigning a type to the variable 'set_type_of_call' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'set_type_of_call', create_call_expression_call_result_17256)
    
    # Call to flatten_lists(...): (line 261)
    # Processing the call arguments (line 261)
    # Getting the type of 'set_type_of_comment' (line 261)
    set_type_of_comment_17258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 25), 'set_type_of_comment', False)
    # Getting the type of 'set_type_of_call' (line 261)
    set_type_of_call_17259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 46), 'set_type_of_call', False)
    # Processing the call keyword arguments (line 261)
    kwargs_17260 = {}
    # Getting the type of 'flatten_lists' (line 261)
    flatten_lists_17257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'flatten_lists', False)
    # Calling flatten_lists(args, kwargs) (line 261)
    flatten_lists_call_result_17261 = invoke(stypy.reporting.localization.Localization(__file__, 261, 11), flatten_lists_17257, *[set_type_of_comment_17258, set_type_of_call_17259], **kwargs_17260)
    
    # Assigning a type to the variable 'stypy_return_type' (line 261)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'stypy_return_type', flatten_lists_call_result_17261)
    
    # ################# End of 'create_store_return_from_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_store_return_from_function' in the type store
    # Getting the type of 'stypy_return_type' (line 252)
    stypy_return_type_17262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17262)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_store_return_from_function'
    return stypy_return_type_17262

# Assigning a type to the variable 'create_store_return_from_function' (line 252)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 0), 'create_store_return_from_function', create_store_return_from_function)

@norecursion
def create_return_from_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_return_from_function'
    module_type_store = module_type_store.open_function_context('create_return_from_function', 264, 0, False)
    
    # Passed parameters checking function
    create_return_from_function.stypy_localization = localization
    create_return_from_function.stypy_type_of_self = None
    create_return_from_function.stypy_type_store = module_type_store
    create_return_from_function.stypy_function_name = 'create_return_from_function'
    create_return_from_function.stypy_param_names_list = ['lineno', 'col_offset']
    create_return_from_function.stypy_varargs_param_name = None
    create_return_from_function.stypy_kwargs_param_name = None
    create_return_from_function.stypy_call_defaults = defaults
    create_return_from_function.stypy_call_varargs = varargs
    create_return_from_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_return_from_function', ['lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_return_from_function', localization, ['lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_return_from_function(...)' code ##################

    str_17263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'str', '\n    Creates an AST node to return from a function\n    ')
    
    # Assigning a Call to a Name (line 268):
    
    # Assigning a Call to a Name (line 268):
    
    # Call to Return(...): (line 268)
    # Processing the call keyword arguments (line 268)
    kwargs_17266 = {}
    # Getting the type of 'ast' (line 268)
    ast_17264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 14), 'ast', False)
    # Obtaining the member 'Return' of a type (line 268)
    Return_17265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 14), ast_17264, 'Return')
    # Calling Return(args, kwargs) (line 268)
    Return_call_result_17267 = invoke(stypy.reporting.localization.Localization(__file__, 268, 14), Return_17265, *[], **kwargs_17266)
    
    # Assigning a type to the variable 'return_' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'return_', Return_call_result_17267)
    
    # Assigning a Call to a Name (line 269):
    
    # Assigning a Call to a Name (line 269):
    
    # Call to create_Name(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'default_function_ret_var_name' (line 269)
    default_function_ret_var_name_17270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 53), 'default_function_ret_var_name', False)
    # Processing the call keyword arguments (line 269)
    kwargs_17271 = {}
    # Getting the type of 'core_language_copy' (line 269)
    core_language_copy_17268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 269)
    create_Name_17269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 22), core_language_copy_17268, 'create_Name')
    # Calling create_Name(args, kwargs) (line 269)
    create_Name_call_result_17272 = invoke(stypy.reporting.localization.Localization(__file__, 269, 22), create_Name_17269, *[default_function_ret_var_name_17270], **kwargs_17271)
    
    # Assigning a type to the variable 'return_var_name' (line 269)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'return_var_name', create_Name_call_result_17272)
    
    # Assigning a Name to a Attribute (line 270):
    
    # Assigning a Name to a Attribute (line 270):
    # Getting the type of 'return_var_name' (line 270)
    return_var_name_17273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 20), 'return_var_name')
    # Getting the type of 'return_' (line 270)
    return__17274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'return_')
    # Setting the type of the member 'value' of a type (line 270)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 4), return__17274, 'value', return_var_name_17273)
    
    # Call to flatten_lists(...): (line 272)
    # Processing the call arguments (line 272)
    # Getting the type of 'return_' (line 272)
    return__17276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 25), 'return_', False)
    # Processing the call keyword arguments (line 272)
    kwargs_17277 = {}
    # Getting the type of 'flatten_lists' (line 272)
    flatten_lists_17275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 11), 'flatten_lists', False)
    # Calling flatten_lists(args, kwargs) (line 272)
    flatten_lists_call_result_17278 = invoke(stypy.reporting.localization.Localization(__file__, 272, 11), flatten_lists_17275, *[return__17276], **kwargs_17277)
    
    # Assigning a type to the variable 'stypy_return_type' (line 272)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 4), 'stypy_return_type', flatten_lists_call_result_17278)
    
    # ################# End of 'create_return_from_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_return_from_function' in the type store
    # Getting the type of 'stypy_return_type' (line 264)
    stypy_return_type_17279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17279)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_return_from_function'
    return stypy_return_type_17279

# Assigning a type to the variable 'create_return_from_function' (line 264)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'create_return_from_function', create_return_from_function)

@norecursion
def get_descritive_element_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_descritive_element_name'
    module_type_store = module_type_store.open_function_context('get_descritive_element_name', 275, 0, False)
    
    # Passed parameters checking function
    get_descritive_element_name.stypy_localization = localization
    get_descritive_element_name.stypy_type_of_self = None
    get_descritive_element_name.stypy_type_store = module_type_store
    get_descritive_element_name.stypy_function_name = 'get_descritive_element_name'
    get_descritive_element_name.stypy_param_names_list = ['node']
    get_descritive_element_name.stypy_varargs_param_name = None
    get_descritive_element_name.stypy_kwargs_param_name = None
    get_descritive_element_name.stypy_call_defaults = defaults
    get_descritive_element_name.stypy_call_varargs = varargs
    get_descritive_element_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_descritive_element_name', ['node'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_descritive_element_name', localization, ['node'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_descritive_element_name(...)' code ##################

    str_17280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, (-1)), 'str', '\n    Gets the name of an AST Name node or an AST Attribute node\n    ')
    
    # Call to isinstance(...): (line 279)
    # Processing the call arguments (line 279)
    # Getting the type of 'node' (line 279)
    node_17282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 18), 'node', False)
    # Getting the type of 'ast' (line 279)
    ast_17283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 24), 'ast', False)
    # Obtaining the member 'Name' of a type (line 279)
    Name_17284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 24), ast_17283, 'Name')
    # Processing the call keyword arguments (line 279)
    kwargs_17285 = {}
    # Getting the type of 'isinstance' (line 279)
    isinstance_17281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 279)
    isinstance_call_result_17286 = invoke(stypy.reporting.localization.Localization(__file__, 279, 7), isinstance_17281, *[node_17282, Name_17284], **kwargs_17285)
    
    # Testing if the type of an if condition is none (line 279)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 279, 4), isinstance_call_result_17286):
        pass
    else:
        
        # Testing the type of an if condition (line 279)
        if_condition_17287 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 279, 4), isinstance_call_result_17286)
        # Assigning a type to the variable 'if_condition_17287' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 4), 'if_condition_17287', if_condition_17287)
        # SSA begins for if statement (line 279)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'node' (line 280)
        node_17288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'node')
        # Obtaining the member 'id' of a type (line 280)
        id_17289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 15), node_17288, 'id')
        # Assigning a type to the variable 'stypy_return_type' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'stypy_return_type', id_17289)
        # SSA join for if statement (line 279)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to isinstance(...): (line 281)
    # Processing the call arguments (line 281)
    # Getting the type of 'node' (line 281)
    node_17291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 18), 'node', False)
    # Getting the type of 'ast' (line 281)
    ast_17292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 24), 'ast', False)
    # Obtaining the member 'Attribute' of a type (line 281)
    Attribute_17293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 281, 24), ast_17292, 'Attribute')
    # Processing the call keyword arguments (line 281)
    kwargs_17294 = {}
    # Getting the type of 'isinstance' (line 281)
    isinstance_17290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 281)
    isinstance_call_result_17295 = invoke(stypy.reporting.localization.Localization(__file__, 281, 7), isinstance_17290, *[node_17291, Attribute_17293], **kwargs_17294)
    
    # Testing if the type of an if condition is none (line 281)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 281, 4), isinstance_call_result_17295):
        pass
    else:
        
        # Testing the type of an if condition (line 281)
        if_condition_17296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 4), isinstance_call_result_17295)
        # Assigning a type to the variable 'if_condition_17296' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'if_condition_17296', if_condition_17296)
        # SSA begins for if statement (line 281)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'node' (line 282)
        node_17297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'node')
        # Obtaining the member 'attr' of a type (line 282)
        attr_17298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 15), node_17297, 'attr')
        # Assigning a type to the variable 'stypy_return_type' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'stypy_return_type', attr_17298)
        # SSA join for if statement (line 281)
        module_type_store = module_type_store.join_ssa_context()
        

    str_17299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 11), 'str', '')
    # Assigning a type to the variable 'stypy_return_type' (line 284)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type', str_17299)
    
    # ################# End of 'get_descritive_element_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_descritive_element_name' in the type store
    # Getting the type of 'stypy_return_type' (line 275)
    stypy_return_type_17300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17300)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_descritive_element_name'
    return stypy_return_type_17300

# Assigning a type to the variable 'get_descritive_element_name' (line 275)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 0), 'get_descritive_element_name', get_descritive_element_name)

@norecursion
def create_pass_node(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_pass_node'
    module_type_store = module_type_store.open_function_context('create_pass_node', 287, 0, False)
    
    # Passed parameters checking function
    create_pass_node.stypy_localization = localization
    create_pass_node.stypy_type_of_self = None
    create_pass_node.stypy_type_store = module_type_store
    create_pass_node.stypy_function_name = 'create_pass_node'
    create_pass_node.stypy_param_names_list = []
    create_pass_node.stypy_varargs_param_name = None
    create_pass_node.stypy_kwargs_param_name = None
    create_pass_node.stypy_call_defaults = defaults
    create_pass_node.stypy_call_varargs = varargs
    create_pass_node.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_pass_node', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_pass_node', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_pass_node(...)' code ##################

    str_17301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, (-1)), 'str', '\n    Creates an AST Pass node\n    ')
    
    # Call to Pass(...): (line 291)
    # Processing the call keyword arguments (line 291)
    kwargs_17304 = {}
    # Getting the type of 'ast' (line 291)
    ast_17302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 11), 'ast', False)
    # Obtaining the member 'Pass' of a type (line 291)
    Pass_17303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 11), ast_17302, 'Pass')
    # Calling Pass(args, kwargs) (line 291)
    Pass_call_result_17305 = invoke(stypy.reporting.localization.Localization(__file__, 291, 11), Pass_17303, *[], **kwargs_17304)
    
    # Assigning a type to the variable 'stypy_return_type' (line 291)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'stypy_return_type', Pass_call_result_17305)
    
    # ################# End of 'create_pass_node(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_pass_node' in the type store
    # Getting the type of 'stypy_return_type' (line 287)
    stypy_return_type_17306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17306)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_pass_node'
    return stypy_return_type_17306

# Assigning a type to the variable 'create_pass_node' (line 287)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 0), 'create_pass_node', create_pass_node)

@norecursion
def assign_as_return_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assign_as_return_type'
    module_type_store = module_type_store.open_function_context('assign_as_return_type', 294, 0, False)
    
    # Passed parameters checking function
    assign_as_return_type.stypy_localization = localization
    assign_as_return_type.stypy_type_of_self = None
    assign_as_return_type.stypy_type_store = module_type_store
    assign_as_return_type.stypy_function_name = 'assign_as_return_type'
    assign_as_return_type.stypy_param_names_list = ['value']
    assign_as_return_type.stypy_varargs_param_name = None
    assign_as_return_type.stypy_kwargs_param_name = None
    assign_as_return_type.stypy_call_defaults = defaults
    assign_as_return_type.stypy_call_varargs = varargs
    assign_as_return_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assign_as_return_type', ['value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assign_as_return_type', localization, ['value'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assign_as_return_type(...)' code ##################

    str_17307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, (-1)), 'str', '\n    Creates AST nodes to store in default_function_ret_var_name a possible return type\n    ')
    
    # Assigning a Call to a Name (line 298):
    
    # Assigning a Call to a Name (line 298):
    
    # Call to create_Name(...): (line 298)
    # Processing the call arguments (line 298)
    # Getting the type of 'default_function_ret_var_name' (line 298)
    default_function_ret_var_name_17310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 62), 'default_function_ret_var_name', False)
    # Processing the call keyword arguments (line 298)
    kwargs_17311 = {}
    # Getting the type of 'core_language_copy' (line 298)
    core_language_copy_17308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 31), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 298)
    create_Name_17309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 31), core_language_copy_17308, 'create_Name')
    # Calling create_Name(args, kwargs) (line 298)
    create_Name_call_result_17312 = invoke(stypy.reporting.localization.Localization(__file__, 298, 31), create_Name_17309, *[default_function_ret_var_name_17310], **kwargs_17311)
    
    # Assigning a type to the variable 'default_function_ret_var' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'default_function_ret_var', create_Name_call_result_17312)
    
    # Call to create_Assign(...): (line 299)
    # Processing the call arguments (line 299)
    # Getting the type of 'default_function_ret_var' (line 299)
    default_function_ret_var_17315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 44), 'default_function_ret_var', False)
    # Getting the type of 'value' (line 299)
    value_17316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 70), 'value', False)
    # Processing the call keyword arguments (line 299)
    kwargs_17317 = {}
    # Getting the type of 'core_language_copy' (line 299)
    core_language_copy_17313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 11), 'core_language_copy', False)
    # Obtaining the member 'create_Assign' of a type (line 299)
    create_Assign_17314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 11), core_language_copy_17313, 'create_Assign')
    # Calling create_Assign(args, kwargs) (line 299)
    create_Assign_call_result_17318 = invoke(stypy.reporting.localization.Localization(__file__, 299, 11), create_Assign_17314, *[default_function_ret_var_17315, value_17316], **kwargs_17317)
    
    # Assigning a type to the variable 'stypy_return_type' (line 299)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 4), 'stypy_return_type', create_Assign_call_result_17318)
    
    # ################# End of 'assign_as_return_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assign_as_return_type' in the type store
    # Getting the type of 'stypy_return_type' (line 294)
    stypy_return_type_17319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17319)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assign_as_return_type'
    return stypy_return_type_17319

# Assigning a type to the variable 'assign_as_return_type' (line 294)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 0), 'assign_as_return_type', assign_as_return_type)

@norecursion
def create_unsupported_feature_call(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_unsupported_feature_call'
    module_type_store = module_type_store.open_function_context('create_unsupported_feature_call', 302, 0, False)
    
    # Passed parameters checking function
    create_unsupported_feature_call.stypy_localization = localization
    create_unsupported_feature_call.stypy_type_of_self = None
    create_unsupported_feature_call.stypy_type_store = module_type_store
    create_unsupported_feature_call.stypy_function_name = 'create_unsupported_feature_call'
    create_unsupported_feature_call.stypy_param_names_list = ['localization', 'feature_name', 'feature_desc', 'lineno', 'col_offset']
    create_unsupported_feature_call.stypy_varargs_param_name = None
    create_unsupported_feature_call.stypy_kwargs_param_name = None
    create_unsupported_feature_call.stypy_call_defaults = defaults
    create_unsupported_feature_call.stypy_call_varargs = varargs
    create_unsupported_feature_call.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_unsupported_feature_call', ['localization', 'feature_name', 'feature_desc', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_unsupported_feature_call', localization, ['localization', 'feature_name', 'feature_desc', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_unsupported_feature_call(...)' code ##################

    str_17320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, (-1)), 'str', '\n    Creates AST nodes to call to the unsupported_python_feature function\n    ')
    
    # Assigning a Call to a Name (line 306):
    
    # Assigning a Call to a Name (line 306):
    
    # Call to create_Name(...): (line 306)
    # Processing the call arguments (line 306)
    str_17323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 306, 62), 'str', 'unsupported_python_feature')
    # Processing the call keyword arguments (line 306)
    # Getting the type of 'lineno' (line 307)
    lineno_17324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 62), 'lineno', False)
    keyword_17325 = lineno_17324
    # Getting the type of 'col_offset' (line 308)
    col_offset_17326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 64), 'col_offset', False)
    keyword_17327 = col_offset_17326
    kwargs_17328 = {'column': keyword_17327, 'line': keyword_17325}
    # Getting the type of 'core_language_copy' (line 306)
    core_language_copy_17321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 31), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 306)
    create_Name_17322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 31), core_language_copy_17321, 'create_Name')
    # Calling create_Name(args, kwargs) (line 306)
    create_Name_call_result_17329 = invoke(stypy.reporting.localization.Localization(__file__, 306, 31), create_Name_17322, *[str_17323], **kwargs_17328)
    
    # Assigning a type to the variable 'unsupported_feature_func' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'unsupported_feature_func', create_Name_call_result_17329)
    
    # Assigning a Call to a Name (line 309):
    
    # Assigning a Call to a Name (line 309):
    
    # Call to create_str(...): (line 309)
    # Processing the call arguments (line 309)
    # Getting the type of 'feature_name' (line 309)
    feature_name_17332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 56), 'feature_name', False)
    # Processing the call keyword arguments (line 309)
    kwargs_17333 = {}
    # Getting the type of 'core_language_copy' (line 309)
    core_language_copy_17330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 26), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 309)
    create_str_17331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 26), core_language_copy_17330, 'create_str')
    # Calling create_str(args, kwargs) (line 309)
    create_str_call_result_17334 = invoke(stypy.reporting.localization.Localization(__file__, 309, 26), create_str_17331, *[feature_name_17332], **kwargs_17333)
    
    # Assigning a type to the variable 'unsupported_feature' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'unsupported_feature', create_str_call_result_17334)
    
    # Assigning a Call to a Name (line 310):
    
    # Assigning a Call to a Name (line 310):
    
    # Call to create_str(...): (line 310)
    # Processing the call arguments (line 310)
    # Getting the type of 'feature_desc' (line 311)
    feature_desc_17337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'feature_desc', False)
    # Processing the call keyword arguments (line 310)
    kwargs_17338 = {}
    # Getting the type of 'core_language_copy' (line 310)
    core_language_copy_17335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 30), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 310)
    create_str_17336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 30), core_language_copy_17335, 'create_str')
    # Calling create_str(args, kwargs) (line 310)
    create_str_call_result_17339 = invoke(stypy.reporting.localization.Localization(__file__, 310, 30), create_str_17336, *[feature_desc_17337], **kwargs_17338)
    
    # Assigning a type to the variable 'unsupported_description' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'unsupported_description', create_str_call_result_17339)
    
    # Call to create_call_expression(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'unsupported_feature_func' (line 312)
    unsupported_feature_func_17342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 49), 'unsupported_feature_func', False)
    
    # Obtaining an instance of the builtin type 'list' (line 313)
    list_17343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 313)
    # Adding element type (line 313)
    # Getting the type of 'localization' (line 313)
    localization_17344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 45), 'localization', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 44), list_17343, localization_17344)
    # Adding element type (line 313)
    # Getting the type of 'unsupported_feature' (line 313)
    unsupported_feature_17345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 59), 'unsupported_feature', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 44), list_17343, unsupported_feature_17345)
    # Adding element type (line 313)
    # Getting the type of 'unsupported_description' (line 314)
    unsupported_description_17346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 45), 'unsupported_description', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 44), list_17343, unsupported_description_17346)
    
    # Processing the call keyword arguments (line 312)
    kwargs_17347 = {}
    # Getting the type of 'functions_copy' (line 312)
    functions_copy_17340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), 'functions_copy', False)
    # Obtaining the member 'create_call_expression' of a type (line 312)
    create_call_expression_17341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 11), functions_copy_17340, 'create_call_expression')
    # Calling create_call_expression(args, kwargs) (line 312)
    create_call_expression_call_result_17348 = invoke(stypy.reporting.localization.Localization(__file__, 312, 11), create_call_expression_17341, *[unsupported_feature_func_17342, list_17343], **kwargs_17347)
    
    # Assigning a type to the variable 'stypy_return_type' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'stypy_return_type', create_call_expression_call_result_17348)
    
    # ################# End of 'create_unsupported_feature_call(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_unsupported_feature_call' in the type store
    # Getting the type of 'stypy_return_type' (line 302)
    stypy_return_type_17349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17349)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_unsupported_feature_call'
    return stypy_return_type_17349

# Assigning a type to the variable 'create_unsupported_feature_call' (line 302)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 0), 'create_unsupported_feature_call', create_unsupported_feature_call)
str_17350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 340, (-1)), 'str', '\nFunctions to get / set the type of variables\n')

@norecursion
def create_add_alias(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_add_alias'
    module_type_store = module_type_store.open_function_context('create_add_alias', 343, 0, False)
    
    # Passed parameters checking function
    create_add_alias.stypy_localization = localization
    create_add_alias.stypy_type_of_self = None
    create_add_alias.stypy_type_store = module_type_store
    create_add_alias.stypy_function_name = 'create_add_alias'
    create_add_alias.stypy_param_names_list = ['alias_name', 'var_name', 'lineno', 'col_offset']
    create_add_alias.stypy_varargs_param_name = None
    create_add_alias.stypy_kwargs_param_name = None
    create_add_alias.stypy_call_defaults = defaults
    create_add_alias.stypy_call_varargs = varargs
    create_add_alias.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_add_alias', ['alias_name', 'var_name', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_add_alias', localization, ['alias_name', 'var_name', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_add_alias(...)' code ##################

    
    # Assigning a Call to a Name (line 344):
    
    # Assigning a Call to a Name (line 344):
    
    # Call to create_src_comment(...): (line 344)
    # Processing the call arguments (line 344)
    str_17352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 45), 'str', 'Adding an alias')
    # Processing the call keyword arguments (line 344)
    kwargs_17353 = {}
    # Getting the type of 'create_src_comment' (line 344)
    create_src_comment_17351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 26), 'create_src_comment', False)
    # Calling create_src_comment(args, kwargs) (line 344)
    create_src_comment_call_result_17354 = invoke(stypy.reporting.localization.Localization(__file__, 344, 26), create_src_comment_17351, *[str_17352], **kwargs_17353)
    
    # Assigning a type to the variable 'get_type_of_comment' (line 344)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 4), 'get_type_of_comment', create_src_comment_call_result_17354)
    
    # Assigning a Call to a Name (line 345):
    
    # Assigning a Call to a Name (line 345):
    
    # Call to create_attribute(...): (line 345)
    # Processing the call arguments (line 345)
    # Getting the type of 'default_module_type_store_var_name' (line 345)
    default_module_type_store_var_name_17357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 61), 'default_module_type_store_var_name', False)
    str_17358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 56), 'str', 'add_alias')
    # Processing the call keyword arguments (line 345)
    # Getting the type of 'lineno' (line 346)
    lineno_17359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 74), 'lineno', False)
    keyword_17360 = lineno_17359
    # Getting the type of 'col_offset' (line 347)
    col_offset_17361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 63), 'col_offset', False)
    keyword_17362 = col_offset_17361
    kwargs_17363 = {'column': keyword_17362, 'line': keyword_17360}
    # Getting the type of 'core_language_copy' (line 345)
    core_language_copy_17355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 25), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 345)
    create_attribute_17356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 345, 25), core_language_copy_17355, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 345)
    create_attribute_call_result_17364 = invoke(stypy.reporting.localization.Localization(__file__, 345, 25), create_attribute_17356, *[default_module_type_store_var_name_17357, str_17358], **kwargs_17363)
    
    # Assigning a type to the variable 'get_type_of_method' (line 345)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 4), 'get_type_of_method', create_attribute_call_result_17364)
    
    # Assigning a Call to a Name (line 349):
    
    # Assigning a Call to a Name (line 349):
    
    # Call to create_call_expression(...): (line 349)
    # Processing the call arguments (line 349)
    # Getting the type of 'get_type_of_method' (line 349)
    get_type_of_method_17367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 61), 'get_type_of_method', False)
    
    # Obtaining an instance of the builtin type 'list' (line 349)
    list_17368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 81), 'list')
    # Adding type elements to the builtin type 'list' instance (line 349)
    # Adding element type (line 349)
    # Getting the type of 'alias_name' (line 349)
    alias_name_17369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 82), 'alias_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 81), list_17368, alias_name_17369)
    # Adding element type (line 349)
    # Getting the type of 'var_name' (line 349)
    var_name_17370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 94), 'var_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 349, 81), list_17368, var_name_17370)
    
    # Processing the call keyword arguments (line 349)
    kwargs_17371 = {}
    # Getting the type of 'functions_copy' (line 349)
    functions_copy_17365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 23), 'functions_copy', False)
    # Obtaining the member 'create_call_expression' of a type (line 349)
    create_call_expression_17366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 23), functions_copy_17365, 'create_call_expression')
    # Calling create_call_expression(args, kwargs) (line 349)
    create_call_expression_call_result_17372 = invoke(stypy.reporting.localization.Localization(__file__, 349, 23), create_call_expression_17366, *[get_type_of_method_17367, list_17368], **kwargs_17371)
    
    # Assigning a type to the variable 'get_type_of_call' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'get_type_of_call', create_call_expression_call_result_17372)
    
    # Call to flatten_lists(...): (line 351)
    # Processing the call arguments (line 351)
    # Getting the type of 'get_type_of_comment' (line 351)
    get_type_of_comment_17374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 25), 'get_type_of_comment', False)
    # Getting the type of 'get_type_of_call' (line 351)
    get_type_of_call_17375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 46), 'get_type_of_call', False)
    # Processing the call keyword arguments (line 351)
    kwargs_17376 = {}
    # Getting the type of 'flatten_lists' (line 351)
    flatten_lists_17373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 11), 'flatten_lists', False)
    # Calling flatten_lists(args, kwargs) (line 351)
    flatten_lists_call_result_17377 = invoke(stypy.reporting.localization.Localization(__file__, 351, 11), flatten_lists_17373, *[get_type_of_comment_17374, get_type_of_call_17375], **kwargs_17376)
    
    # Assigning a type to the variable 'stypy_return_type' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'stypy_return_type', flatten_lists_call_result_17377)
    
    # ################# End of 'create_add_alias(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_add_alias' in the type store
    # Getting the type of 'stypy_return_type' (line 343)
    stypy_return_type_17378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17378)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_add_alias'
    return stypy_return_type_17378

# Assigning a type to the variable 'create_add_alias' (line 343)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 0), 'create_add_alias', create_add_alias)

@norecursion
def create_get_type_of(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 354)
    True_17379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 71), 'True')
    defaults = [True_17379]
    # Create a new context for function 'create_get_type_of'
    module_type_store = module_type_store.open_function_context('create_get_type_of', 354, 0, False)
    
    # Passed parameters checking function
    create_get_type_of.stypy_localization = localization
    create_get_type_of.stypy_type_of_self = None
    create_get_type_of.stypy_type_store = module_type_store
    create_get_type_of.stypy_function_name = 'create_get_type_of'
    create_get_type_of.stypy_param_names_list = ['var_name', 'lineno', 'col_offset', 'test_unreferenced']
    create_get_type_of.stypy_varargs_param_name = None
    create_get_type_of.stypy_kwargs_param_name = None
    create_get_type_of.stypy_call_defaults = defaults
    create_get_type_of.stypy_call_varargs = varargs
    create_get_type_of.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_get_type_of', ['var_name', 'lineno', 'col_offset', 'test_unreferenced'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_get_type_of', localization, ['var_name', 'lineno', 'col_offset', 'test_unreferenced'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_get_type_of(...)' code ##################

    
    # Assigning a Call to a Name (line 355):
    
    # Assigning a Call to a Name (line 355):
    
    # Call to create_src_comment(...): (line 355)
    # Processing the call arguments (line 355)
    
    # Call to format(...): (line 355)
    # Processing the call arguments (line 355)
    # Getting the type of 'var_name' (line 355)
    var_name_17383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 80), 'var_name', False)
    # Processing the call keyword arguments (line 355)
    kwargs_17384 = {}
    str_17381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 45), 'str', "Getting the type of '{0}'")
    # Obtaining the member 'format' of a type (line 355)
    format_17382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 355, 45), str_17381, 'format')
    # Calling format(args, kwargs) (line 355)
    format_call_result_17385 = invoke(stypy.reporting.localization.Localization(__file__, 355, 45), format_17382, *[var_name_17383], **kwargs_17384)
    
    # Getting the type of 'lineno' (line 355)
    lineno_17386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 91), 'lineno', False)
    # Processing the call keyword arguments (line 355)
    kwargs_17387 = {}
    # Getting the type of 'create_src_comment' (line 355)
    create_src_comment_17380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 26), 'create_src_comment', False)
    # Calling create_src_comment(args, kwargs) (line 355)
    create_src_comment_call_result_17388 = invoke(stypy.reporting.localization.Localization(__file__, 355, 26), create_src_comment_17380, *[format_call_result_17385, lineno_17386], **kwargs_17387)
    
    # Assigning a type to the variable 'get_type_of_comment' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'get_type_of_comment', create_src_comment_call_result_17388)
    
    # Assigning a Call to a Name (line 356):
    
    # Assigning a Call to a Name (line 356):
    
    # Call to create_attribute(...): (line 356)
    # Processing the call arguments (line 356)
    # Getting the type of 'default_module_type_store_var_name' (line 356)
    default_module_type_store_var_name_17391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 61), 'default_module_type_store_var_name', False)
    str_17392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 357, 56), 'str', 'get_type_of')
    # Processing the call keyword arguments (line 356)
    # Getting the type of 'lineno' (line 357)
    lineno_17393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 357, 76), 'lineno', False)
    keyword_17394 = lineno_17393
    # Getting the type of 'col_offset' (line 358)
    col_offset_17395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 63), 'col_offset', False)
    keyword_17396 = col_offset_17395
    kwargs_17397 = {'column': keyword_17396, 'line': keyword_17394}
    # Getting the type of 'core_language_copy' (line 356)
    core_language_copy_17389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 25), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 356)
    create_attribute_17390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 25), core_language_copy_17389, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 356)
    create_attribute_call_result_17398 = invoke(stypy.reporting.localization.Localization(__file__, 356, 25), create_attribute_17390, *[default_module_type_store_var_name_17391, str_17392], **kwargs_17397)
    
    # Assigning a type to the variable 'get_type_of_method' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'get_type_of_method', create_attribute_call_result_17398)
    
    # Assigning a Call to a Name (line 359):
    
    # Assigning a Call to a Name (line 359):
    
    # Call to create_localization(...): (line 359)
    # Processing the call arguments (line 359)
    # Getting the type of 'lineno' (line 359)
    lineno_17400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 39), 'lineno', False)
    # Getting the type of 'col_offset' (line 359)
    col_offset_17401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 47), 'col_offset', False)
    # Processing the call keyword arguments (line 359)
    kwargs_17402 = {}
    # Getting the type of 'create_localization' (line 359)
    create_localization_17399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 19), 'create_localization', False)
    # Calling create_localization(args, kwargs) (line 359)
    create_localization_call_result_17403 = invoke(stypy.reporting.localization.Localization(__file__, 359, 19), create_localization_17399, *[lineno_17400, col_offset_17401], **kwargs_17402)
    
    # Assigning a type to the variable 'localization' (line 359)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'localization', create_localization_call_result_17403)
    # Getting the type of 'test_unreferenced' (line 360)
    test_unreferenced_17404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 7), 'test_unreferenced')
    # Testing if the type of an if condition is none (line 360)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 360, 4), test_unreferenced_17404):
        
        # Assigning a Call to a Name (line 363):
        
        # Assigning a Call to a Name (line 363):
        
        # Call to create_call(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'get_type_of_method' (line 363)
        get_type_of_method_17420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 54), 'get_type_of_method', False)
        
        # Obtaining an instance of the builtin type 'list' (line 363)
        list_17421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 74), 'list')
        # Adding type elements to the builtin type 'list' instance (line 363)
        # Adding element type (line 363)
        # Getting the type of 'localization' (line 363)
        localization_17422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 75), 'localization', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 74), list_17421, localization_17422)
        # Adding element type (line 363)
        
        # Call to create_str(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'var_name' (line 363)
        var_name_17425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 119), 'var_name', False)
        # Processing the call keyword arguments (line 363)
        kwargs_17426 = {}
        # Getting the type of 'core_language_copy' (line 363)
        core_language_copy_17423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 89), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 363)
        create_str_17424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 89), core_language_copy_17423, 'create_str')
        # Calling create_str(args, kwargs) (line 363)
        create_str_call_result_17427 = invoke(stypy.reporting.localization.Localization(__file__, 363, 89), create_str_17424, *[var_name_17425], **kwargs_17426)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 74), list_17421, create_str_call_result_17427)
        # Adding element type (line 363)
        
        # Call to create_Name(...): (line 364)
        # Processing the call arguments (line 364)
        str_17430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 101), 'str', 'False')
        # Processing the call keyword arguments (line 364)
        kwargs_17431 = {}
        # Getting the type of 'core_language_copy' (line 364)
        core_language_copy_17428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 70), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 364)
        create_Name_17429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 70), core_language_copy_17428, 'create_Name')
        # Calling create_Name(args, kwargs) (line 364)
        create_Name_call_result_17432 = invoke(stypy.reporting.localization.Localization(__file__, 364, 70), create_Name_17429, *[str_17430], **kwargs_17431)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 74), list_17421, create_Name_call_result_17432)
        
        # Processing the call keyword arguments (line 363)
        kwargs_17433 = {}
        # Getting the type of 'functions_copy' (line 363)
        functions_copy_17418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 27), 'functions_copy', False)
        # Obtaining the member 'create_call' of a type (line 363)
        create_call_17419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 27), functions_copy_17418, 'create_call')
        # Calling create_call(args, kwargs) (line 363)
        create_call_call_result_17434 = invoke(stypy.reporting.localization.Localization(__file__, 363, 27), create_call_17419, *[get_type_of_method_17420, list_17421], **kwargs_17433)
        
        # Assigning a type to the variable 'get_type_of_call' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'get_type_of_call', create_call_call_result_17434)
    else:
        
        # Testing the type of an if condition (line 360)
        if_condition_17405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 360, 4), test_unreferenced_17404)
        # Assigning a type to the variable 'if_condition_17405' (line 360)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'if_condition_17405', if_condition_17405)
        # SSA begins for if statement (line 360)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 361):
        
        # Assigning a Call to a Name (line 361):
        
        # Call to create_call(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'get_type_of_method' (line 361)
        get_type_of_method_17408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 54), 'get_type_of_method', False)
        
        # Obtaining an instance of the builtin type 'list' (line 361)
        list_17409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 74), 'list')
        # Adding type elements to the builtin type 'list' instance (line 361)
        # Adding element type (line 361)
        # Getting the type of 'localization' (line 361)
        localization_17410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 75), 'localization', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 74), list_17409, localization_17410)
        # Adding element type (line 361)
        
        # Call to create_str(...): (line 361)
        # Processing the call arguments (line 361)
        # Getting the type of 'var_name' (line 361)
        var_name_17413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 119), 'var_name', False)
        # Processing the call keyword arguments (line 361)
        kwargs_17414 = {}
        # Getting the type of 'core_language_copy' (line 361)
        core_language_copy_17411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 89), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 361)
        create_str_17412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 89), core_language_copy_17411, 'create_str')
        # Calling create_str(args, kwargs) (line 361)
        create_str_call_result_17415 = invoke(stypy.reporting.localization.Localization(__file__, 361, 89), create_str_17412, *[var_name_17413], **kwargs_17414)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 361, 74), list_17409, create_str_call_result_17415)
        
        # Processing the call keyword arguments (line 361)
        kwargs_17416 = {}
        # Getting the type of 'functions_copy' (line 361)
        functions_copy_17406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 27), 'functions_copy', False)
        # Obtaining the member 'create_call' of a type (line 361)
        create_call_17407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 27), functions_copy_17406, 'create_call')
        # Calling create_call(args, kwargs) (line 361)
        create_call_call_result_17417 = invoke(stypy.reporting.localization.Localization(__file__, 361, 27), create_call_17407, *[get_type_of_method_17408, list_17409], **kwargs_17416)
        
        # Assigning a type to the variable 'get_type_of_call' (line 361)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 361, 8), 'get_type_of_call', create_call_call_result_17417)
        # SSA branch for the else part of an if statement (line 360)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 363):
        
        # Assigning a Call to a Name (line 363):
        
        # Call to create_call(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'get_type_of_method' (line 363)
        get_type_of_method_17420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 54), 'get_type_of_method', False)
        
        # Obtaining an instance of the builtin type 'list' (line 363)
        list_17421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 74), 'list')
        # Adding type elements to the builtin type 'list' instance (line 363)
        # Adding element type (line 363)
        # Getting the type of 'localization' (line 363)
        localization_17422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 75), 'localization', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 74), list_17421, localization_17422)
        # Adding element type (line 363)
        
        # Call to create_str(...): (line 363)
        # Processing the call arguments (line 363)
        # Getting the type of 'var_name' (line 363)
        var_name_17425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 119), 'var_name', False)
        # Processing the call keyword arguments (line 363)
        kwargs_17426 = {}
        # Getting the type of 'core_language_copy' (line 363)
        core_language_copy_17423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 89), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 363)
        create_str_17424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 89), core_language_copy_17423, 'create_str')
        # Calling create_str(args, kwargs) (line 363)
        create_str_call_result_17427 = invoke(stypy.reporting.localization.Localization(__file__, 363, 89), create_str_17424, *[var_name_17425], **kwargs_17426)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 74), list_17421, create_str_call_result_17427)
        # Adding element type (line 363)
        
        # Call to create_Name(...): (line 364)
        # Processing the call arguments (line 364)
        str_17430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 364, 101), 'str', 'False')
        # Processing the call keyword arguments (line 364)
        kwargs_17431 = {}
        # Getting the type of 'core_language_copy' (line 364)
        core_language_copy_17428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 364, 70), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 364)
        create_Name_17429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 364, 70), core_language_copy_17428, 'create_Name')
        # Calling create_Name(args, kwargs) (line 364)
        create_Name_call_result_17432 = invoke(stypy.reporting.localization.Localization(__file__, 364, 70), create_Name_17429, *[str_17430], **kwargs_17431)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 363, 74), list_17421, create_Name_call_result_17432)
        
        # Processing the call keyword arguments (line 363)
        kwargs_17433 = {}
        # Getting the type of 'functions_copy' (line 363)
        functions_copy_17418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 27), 'functions_copy', False)
        # Obtaining the member 'create_call' of a type (line 363)
        create_call_17419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 27), functions_copy_17418, 'create_call')
        # Calling create_call(args, kwargs) (line 363)
        create_call_call_result_17434 = invoke(stypy.reporting.localization.Localization(__file__, 363, 27), create_call_17419, *[get_type_of_method_17420, list_17421], **kwargs_17433)
        
        # Assigning a type to the variable 'get_type_of_call' (line 363)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 363, 8), 'get_type_of_call', create_call_call_result_17434)
        # SSA join for if statement (line 360)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Tuple (line 366):
    
    # Assigning a Call to a Name:
    
    # Call to create_temp_Assign(...): (line 366)
    # Processing the call arguments (line 366)
    # Getting the type of 'get_type_of_call' (line 366)
    get_type_of_call_17436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 51), 'get_type_of_call', False)
    # Getting the type of 'lineno' (line 366)
    lineno_17437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 69), 'lineno', False)
    # Getting the type of 'col_offset' (line 366)
    col_offset_17438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 77), 'col_offset', False)
    # Processing the call keyword arguments (line 366)
    kwargs_17439 = {}
    # Getting the type of 'create_temp_Assign' (line 366)
    create_temp_Assign_17435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 32), 'create_temp_Assign', False)
    # Calling create_temp_Assign(args, kwargs) (line 366)
    create_temp_Assign_call_result_17440 = invoke(stypy.reporting.localization.Localization(__file__, 366, 32), create_temp_Assign_17435, *[get_type_of_call_17436, lineno_17437, col_offset_17438], **kwargs_17439)
    
    # Assigning a type to the variable 'call_assignment_16806' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'call_assignment_16806', create_temp_Assign_call_result_17440)
    
    # Assigning a Call to a Name (line 366):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_16806' (line 366)
    call_assignment_16806_17441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'call_assignment_16806', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_17442 = stypy_get_value_from_tuple(call_assignment_16806_17441, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_16807' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'call_assignment_16807', stypy_get_value_from_tuple_call_result_17442)
    
    # Assigning a Name to a Name (line 366):
    # Getting the type of 'call_assignment_16807' (line 366)
    call_assignment_16807_17443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'call_assignment_16807')
    # Assigning a type to the variable 'assign_stmts' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'assign_stmts', call_assignment_16807_17443)
    
    # Assigning a Call to a Name (line 366):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_16806' (line 366)
    call_assignment_16806_17444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'call_assignment_16806', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_17445 = stypy_get_value_from_tuple(call_assignment_16806_17444, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_16808' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'call_assignment_16808', stypy_get_value_from_tuple_call_result_17445)
    
    # Assigning a Name to a Name (line 366):
    # Getting the type of 'call_assignment_16808' (line 366)
    call_assignment_16808_17446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'call_assignment_16808')
    # Assigning a type to the variable 'temp_assign' (line 366)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 18), 'temp_assign', call_assignment_16808_17446)
    
    # Obtaining an instance of the builtin type 'tuple' (line 368)
    tuple_17447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 368, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 368)
    # Adding element type (line 368)
    
    # Call to flatten_lists(...): (line 368)
    # Processing the call arguments (line 368)
    # Getting the type of 'get_type_of_comment' (line 368)
    get_type_of_comment_17449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 25), 'get_type_of_comment', False)
    # Getting the type of 'assign_stmts' (line 368)
    assign_stmts_17450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 46), 'assign_stmts', False)
    # Processing the call keyword arguments (line 368)
    kwargs_17451 = {}
    # Getting the type of 'flatten_lists' (line 368)
    flatten_lists_17448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 11), 'flatten_lists', False)
    # Calling flatten_lists(args, kwargs) (line 368)
    flatten_lists_call_result_17452 = invoke(stypy.reporting.localization.Localization(__file__, 368, 11), flatten_lists_17448, *[get_type_of_comment_17449, assign_stmts_17450], **kwargs_17451)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 11), tuple_17447, flatten_lists_call_result_17452)
    # Adding element type (line 368)
    # Getting the type of 'temp_assign' (line 368)
    temp_assign_17453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 61), 'temp_assign')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 368, 11), tuple_17447, temp_assign_17453)
    
    # Assigning a type to the variable 'stypy_return_type' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type', tuple_17447)
    
    # ################# End of 'create_get_type_of(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_get_type_of' in the type store
    # Getting the type of 'stypy_return_type' (line 354)
    stypy_return_type_17454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17454)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_get_type_of'
    return stypy_return_type_17454

# Assigning a type to the variable 'create_get_type_of' (line 354)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 0), 'create_get_type_of', create_get_type_of)

@norecursion
def create_set_type_of(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_set_type_of'
    module_type_store = module_type_store.open_function_context('create_set_type_of', 371, 0, False)
    
    # Passed parameters checking function
    create_set_type_of.stypy_localization = localization
    create_set_type_of.stypy_type_of_self = None
    create_set_type_of.stypy_type_store = module_type_store
    create_set_type_of.stypy_function_name = 'create_set_type_of'
    create_set_type_of.stypy_param_names_list = ['var_name', 'new_value', 'lineno', 'col_offset']
    create_set_type_of.stypy_varargs_param_name = None
    create_set_type_of.stypy_kwargs_param_name = None
    create_set_type_of.stypy_call_defaults = defaults
    create_set_type_of.stypy_call_varargs = varargs
    create_set_type_of.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_set_type_of', ['var_name', 'new_value', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_set_type_of', localization, ['var_name', 'new_value', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_set_type_of(...)' code ##################

    
    # Assigning a Call to a Name (line 372):
    
    # Assigning a Call to a Name (line 372):
    
    # Call to create_src_comment(...): (line 372)
    # Processing the call arguments (line 372)
    str_17456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 372, 45), 'str', 'Type assignment')
    # Getting the type of 'lineno' (line 372)
    lineno_17457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 64), 'lineno', False)
    # Processing the call keyword arguments (line 372)
    kwargs_17458 = {}
    # Getting the type of 'create_src_comment' (line 372)
    create_src_comment_17455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 26), 'create_src_comment', False)
    # Calling create_src_comment(args, kwargs) (line 372)
    create_src_comment_call_result_17459 = invoke(stypy.reporting.localization.Localization(__file__, 372, 26), create_src_comment_17455, *[str_17456, lineno_17457], **kwargs_17458)
    
    # Assigning a type to the variable 'set_type_of_comment' (line 372)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 4), 'set_type_of_comment', create_src_comment_call_result_17459)
    
    # Assigning a Call to a Name (line 373):
    
    # Assigning a Call to a Name (line 373):
    
    # Call to create_attribute(...): (line 373)
    # Processing the call arguments (line 373)
    # Getting the type of 'default_module_type_store_var_name' (line 373)
    default_module_type_store_var_name_17462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 61), 'default_module_type_store_var_name', False)
    str_17463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 373, 97), 'str', 'set_type_of')
    # Processing the call keyword arguments (line 373)
    kwargs_17464 = {}
    # Getting the type of 'core_language_copy' (line 373)
    core_language_copy_17460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 373, 25), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 373)
    create_attribute_17461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 373, 25), core_language_copy_17460, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 373)
    create_attribute_call_result_17465 = invoke(stypy.reporting.localization.Localization(__file__, 373, 25), create_attribute_17461, *[default_module_type_store_var_name_17462, str_17463], **kwargs_17464)
    
    # Assigning a type to the variable 'set_type_of_method' (line 373)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 373, 4), 'set_type_of_method', create_attribute_call_result_17465)
    
    # Assigning a Call to a Name (line 375):
    
    # Assigning a Call to a Name (line 375):
    
    # Call to create_localization(...): (line 375)
    # Processing the call arguments (line 375)
    # Getting the type of 'lineno' (line 375)
    lineno_17467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 39), 'lineno', False)
    # Getting the type of 'col_offset' (line 375)
    col_offset_17468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 47), 'col_offset', False)
    # Processing the call keyword arguments (line 375)
    kwargs_17469 = {}
    # Getting the type of 'create_localization' (line 375)
    create_localization_17466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 375, 19), 'create_localization', False)
    # Calling create_localization(args, kwargs) (line 375)
    create_localization_call_result_17470 = invoke(stypy.reporting.localization.Localization(__file__, 375, 19), create_localization_17466, *[lineno_17467, col_offset_17468], **kwargs_17469)
    
    # Assigning a type to the variable 'localization' (line 375)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 375, 4), 'localization', create_localization_call_result_17470)
    
    # Assigning a Call to a Name (line 377):
    
    # Assigning a Call to a Name (line 377):
    
    # Call to create_call_expression(...): (line 377)
    # Processing the call arguments (line 377)
    # Getting the type of 'set_type_of_method' (line 377)
    set_type_of_method_17473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 61), 'set_type_of_method', False)
    
    # Obtaining an instance of the builtin type 'list' (line 378)
    list_17474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 378, 56), 'list')
    # Adding type elements to the builtin type 'list' instance (line 378)
    # Adding element type (line 378)
    # Getting the type of 'localization' (line 378)
    localization_17475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 57), 'localization', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 56), list_17474, localization_17475)
    # Adding element type (line 378)
    
    # Call to create_str(...): (line 378)
    # Processing the call arguments (line 378)
    # Getting the type of 'var_name' (line 378)
    var_name_17478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 101), 'var_name', False)
    # Getting the type of 'lineno' (line 378)
    lineno_17479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 111), 'lineno', False)
    # Getting the type of 'col_offset' (line 379)
    col_offset_17480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 96), 'col_offset', False)
    # Processing the call keyword arguments (line 378)
    kwargs_17481 = {}
    # Getting the type of 'core_language_copy' (line 378)
    core_language_copy_17476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 378, 71), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 378)
    create_str_17477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 378, 71), core_language_copy_17476, 'create_str')
    # Calling create_str(args, kwargs) (line 378)
    create_str_call_result_17482 = invoke(stypy.reporting.localization.Localization(__file__, 378, 71), create_str_17477, *[var_name_17478, lineno_17479, col_offset_17480], **kwargs_17481)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 56), list_17474, create_str_call_result_17482)
    # Adding element type (line 378)
    # Getting the type of 'new_value' (line 379)
    new_value_17483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 379, 109), 'new_value', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 378, 56), list_17474, new_value_17483)
    
    # Processing the call keyword arguments (line 377)
    kwargs_17484 = {}
    # Getting the type of 'functions_copy' (line 377)
    functions_copy_17471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 23), 'functions_copy', False)
    # Obtaining the member 'create_call_expression' of a type (line 377)
    create_call_expression_17472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 377, 23), functions_copy_17471, 'create_call_expression')
    # Calling create_call_expression(args, kwargs) (line 377)
    create_call_expression_call_result_17485 = invoke(stypy.reporting.localization.Localization(__file__, 377, 23), create_call_expression_17472, *[set_type_of_method_17473, list_17474], **kwargs_17484)
    
    # Assigning a type to the variable 'set_type_of_call' (line 377)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 4), 'set_type_of_call', create_call_expression_call_result_17485)
    
    # Call to flatten_lists(...): (line 381)
    # Processing the call arguments (line 381)
    # Getting the type of 'set_type_of_comment' (line 381)
    set_type_of_comment_17487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 25), 'set_type_of_comment', False)
    # Getting the type of 'set_type_of_call' (line 381)
    set_type_of_call_17488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 46), 'set_type_of_call', False)
    # Processing the call keyword arguments (line 381)
    kwargs_17489 = {}
    # Getting the type of 'flatten_lists' (line 381)
    flatten_lists_17486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 381, 11), 'flatten_lists', False)
    # Calling flatten_lists(args, kwargs) (line 381)
    flatten_lists_call_result_17490 = invoke(stypy.reporting.localization.Localization(__file__, 381, 11), flatten_lists_17486, *[set_type_of_comment_17487, set_type_of_call_17488], **kwargs_17489)
    
    # Assigning a type to the variable 'stypy_return_type' (line 381)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 381, 4), 'stypy_return_type', flatten_lists_call_result_17490)
    
    # ################# End of 'create_set_type_of(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_set_type_of' in the type store
    # Getting the type of 'stypy_return_type' (line 371)
    stypy_return_type_17491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17491)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_set_type_of'
    return stypy_return_type_17491

# Assigning a type to the variable 'create_set_type_of' (line 371)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'create_set_type_of', create_set_type_of)

@norecursion
def create_get_type_of_member(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 384)
    True_17492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 92), 'True')
    defaults = [True_17492]
    # Create a new context for function 'create_get_type_of_member'
    module_type_store = module_type_store.open_function_context('create_get_type_of_member', 384, 0, False)
    
    # Passed parameters checking function
    create_get_type_of_member.stypy_localization = localization
    create_get_type_of_member.stypy_type_of_self = None
    create_get_type_of_member.stypy_type_store = module_type_store
    create_get_type_of_member.stypy_function_name = 'create_get_type_of_member'
    create_get_type_of_member.stypy_param_names_list = ['owner_var', 'member_name', 'lineno', 'col_offset', 'test_unreferenced']
    create_get_type_of_member.stypy_varargs_param_name = None
    create_get_type_of_member.stypy_kwargs_param_name = None
    create_get_type_of_member.stypy_call_defaults = defaults
    create_get_type_of_member.stypy_call_varargs = varargs
    create_get_type_of_member.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_get_type_of_member', ['owner_var', 'member_name', 'lineno', 'col_offset', 'test_unreferenced'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_get_type_of_member', localization, ['owner_var', 'member_name', 'lineno', 'col_offset', 'test_unreferenced'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_get_type_of_member(...)' code ##################

    
    # Assigning a Call to a Name (line 385):
    
    # Assigning a Call to a Name (line 385):
    
    # Call to create_src_comment(...): (line 385)
    # Processing the call arguments (line 385)
    
    # Call to format(...): (line 385)
    # Processing the call arguments (line 385)
    # Getting the type of 'member_name' (line 385)
    member_name_17496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 79), 'member_name', False)
    # Processing the call keyword arguments (line 385)
    kwargs_17497 = {}
    str_17494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 385, 33), 'str', "Obtaining the member '{0}' of a type")
    # Obtaining the member 'format' of a type (line 385)
    format_17495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 385, 33), str_17494, 'format')
    # Calling format(args, kwargs) (line 385)
    format_call_result_17498 = invoke(stypy.reporting.localization.Localization(__file__, 385, 33), format_17495, *[member_name_17496], **kwargs_17497)
    
    # Getting the type of 'lineno' (line 385)
    lineno_17499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 93), 'lineno', False)
    # Processing the call keyword arguments (line 385)
    kwargs_17500 = {}
    # Getting the type of 'create_src_comment' (line 385)
    create_src_comment_17493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 385, 14), 'create_src_comment', False)
    # Calling create_src_comment(args, kwargs) (line 385)
    create_src_comment_call_result_17501 = invoke(stypy.reporting.localization.Localization(__file__, 385, 14), create_src_comment_17493, *[format_call_result_17498, lineno_17499], **kwargs_17500)
    
    # Assigning a type to the variable 'comment' (line 385)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 385, 4), 'comment', create_src_comment_call_result_17501)
    
    # Assigning a Call to a Name (line 386):
    
    # Assigning a Call to a Name (line 386):
    
    # Call to create_localization(...): (line 386)
    # Processing the call arguments (line 386)
    # Getting the type of 'lineno' (line 386)
    lineno_17503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 39), 'lineno', False)
    # Getting the type of 'col_offset' (line 386)
    col_offset_17504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 47), 'col_offset', False)
    # Processing the call keyword arguments (line 386)
    kwargs_17505 = {}
    # Getting the type of 'create_localization' (line 386)
    create_localization_17502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 386, 19), 'create_localization', False)
    # Calling create_localization(args, kwargs) (line 386)
    create_localization_call_result_17506 = invoke(stypy.reporting.localization.Localization(__file__, 386, 19), create_localization_17502, *[lineno_17503, col_offset_17504], **kwargs_17505)
    
    # Assigning a type to the variable 'localization' (line 386)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 386, 4), 'localization', create_localization_call_result_17506)
    
    # Assigning a Call to a Name (line 393):
    
    # Assigning a Call to a Name (line 393):
    
    # Call to create_attribute(...): (line 393)
    # Processing the call arguments (line 393)
    # Getting the type of 'owner_var' (line 393)
    owner_var_17509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 66), 'owner_var', False)
    str_17510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 393, 77), 'str', 'get_type_of_member')
    # Processing the call keyword arguments (line 393)
    kwargs_17511 = {}
    # Getting the type of 'core_language_copy' (line 393)
    core_language_copy_17507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 393, 30), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 393)
    create_attribute_17508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 393, 30), core_language_copy_17507, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 393)
    create_attribute_call_result_17512 = invoke(stypy.reporting.localization.Localization(__file__, 393, 30), create_attribute_17508, *[owner_var_17509, str_17510], **kwargs_17511)
    
    # Assigning a type to the variable 'get_type_of_member_func' (line 393)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 393, 4), 'get_type_of_member_func', create_attribute_call_result_17512)
    
    # Getting the type of 'test_unreferenced' (line 394)
    test_unreferenced_17513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 394, 11), 'test_unreferenced')
    # Applying the 'not' unary operator (line 394)
    result_not__17514 = python_operator(stypy.reporting.localization.Localization(__file__, 394, 7), 'not', test_unreferenced_17513)
    
    # Testing if the type of an if condition is none (line 394)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 394, 4), result_not__17514):
        
        # Assigning a Call to a Name (line 400):
        
        # Assigning a Call to a Name (line 400):
        
        # Call to create_call(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'get_type_of_member_func' (line 400)
        get_type_of_member_func_17535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 61), 'get_type_of_member_func', False)
        
        # Obtaining an instance of the builtin type 'list' (line 400)
        list_17536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 86), 'list')
        # Adding type elements to the builtin type 'list' instance (line 400)
        # Adding element type (line 400)
        # Getting the type of 'localization' (line 400)
        localization_17537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 87), 'localization', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 86), list_17536, localization_17537)
        # Adding element type (line 400)
        
        # Call to create_str(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'member_name' (line 402)
        member_name_17540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 86), 'member_name', False)
        # Processing the call keyword arguments (line 401)
        kwargs_17541 = {}
        # Getting the type of 'core_language_copy' (line 401)
        core_language_copy_17538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 82), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 401)
        create_str_17539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 82), core_language_copy_17538, 'create_str')
        # Calling create_str(args, kwargs) (line 401)
        create_str_call_result_17542 = invoke(stypy.reporting.localization.Localization(__file__, 401, 82), create_str_17539, *[member_name_17540], **kwargs_17541)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 86), list_17536, create_str_call_result_17542)
        
        # Processing the call keyword arguments (line 400)
        kwargs_17543 = {}
        # Getting the type of 'functions_copy' (line 400)
        functions_copy_17533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 34), 'functions_copy', False)
        # Obtaining the member 'create_call' of a type (line 400)
        create_call_17534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 34), functions_copy_17533, 'create_call')
        # Calling create_call(args, kwargs) (line 400)
        create_call_call_result_17544 = invoke(stypy.reporting.localization.Localization(__file__, 400, 34), create_call_17534, *[get_type_of_member_func_17535, list_17536], **kwargs_17543)
        
        # Assigning a type to the variable 'get_type_of_member_call' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'get_type_of_member_call', create_call_call_result_17544)
    else:
        
        # Testing the type of an if condition (line 394)
        if_condition_17515 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 394, 4), result_not__17514)
        # Assigning a type to the variable 'if_condition_17515' (line 394)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 394, 4), 'if_condition_17515', if_condition_17515)
        # SSA begins for if statement (line 394)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 395):
        
        # Assigning a Call to a Name (line 395):
        
        # Call to create_call(...): (line 395)
        # Processing the call arguments (line 395)
        # Getting the type of 'get_type_of_member_func' (line 395)
        get_type_of_member_func_17518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 61), 'get_type_of_member_func', False)
        
        # Obtaining an instance of the builtin type 'list' (line 395)
        list_17519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 395, 86), 'list')
        # Adding type elements to the builtin type 'list' instance (line 395)
        # Adding element type (line 395)
        # Getting the type of 'localization' (line 395)
        localization_17520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 87), 'localization', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 86), list_17519, localization_17520)
        # Adding element type (line 395)
        
        # Call to create_str(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'member_name' (line 397)
        member_name_17523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 86), 'member_name', False)
        # Processing the call keyword arguments (line 396)
        kwargs_17524 = {}
        # Getting the type of 'core_language_copy' (line 396)
        core_language_copy_17521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 82), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 396)
        create_str_17522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 82), core_language_copy_17521, 'create_str')
        # Calling create_str(args, kwargs) (line 396)
        create_str_call_result_17525 = invoke(stypy.reporting.localization.Localization(__file__, 396, 82), create_str_17522, *[member_name_17523], **kwargs_17524)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 86), list_17519, create_str_call_result_17525)
        # Adding element type (line 395)
        
        # Call to create_Name(...): (line 398)
        # Processing the call arguments (line 398)
        str_17528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 398, 113), 'str', 'False')
        # Processing the call keyword arguments (line 398)
        kwargs_17529 = {}
        # Getting the type of 'core_language_copy' (line 398)
        core_language_copy_17526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 398, 82), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 398)
        create_Name_17527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 398, 82), core_language_copy_17526, 'create_Name')
        # Calling create_Name(args, kwargs) (line 398)
        create_Name_call_result_17530 = invoke(stypy.reporting.localization.Localization(__file__, 398, 82), create_Name_17527, *[str_17528], **kwargs_17529)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 395, 86), list_17519, create_Name_call_result_17530)
        
        # Processing the call keyword arguments (line 395)
        kwargs_17531 = {}
        # Getting the type of 'functions_copy' (line 395)
        functions_copy_17516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 34), 'functions_copy', False)
        # Obtaining the member 'create_call' of a type (line 395)
        create_call_17517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 34), functions_copy_17516, 'create_call')
        # Calling create_call(args, kwargs) (line 395)
        create_call_call_result_17532 = invoke(stypy.reporting.localization.Localization(__file__, 395, 34), create_call_17517, *[get_type_of_member_func_17518, list_17519], **kwargs_17531)
        
        # Assigning a type to the variable 'get_type_of_member_call' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 8), 'get_type_of_member_call', create_call_call_result_17532)
        # SSA branch for the else part of an if statement (line 394)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 400):
        
        # Assigning a Call to a Name (line 400):
        
        # Call to create_call(...): (line 400)
        # Processing the call arguments (line 400)
        # Getting the type of 'get_type_of_member_func' (line 400)
        get_type_of_member_func_17535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 61), 'get_type_of_member_func', False)
        
        # Obtaining an instance of the builtin type 'list' (line 400)
        list_17536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 400, 86), 'list')
        # Adding type elements to the builtin type 'list' instance (line 400)
        # Adding element type (line 400)
        # Getting the type of 'localization' (line 400)
        localization_17537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 87), 'localization', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 86), list_17536, localization_17537)
        # Adding element type (line 400)
        
        # Call to create_str(...): (line 401)
        # Processing the call arguments (line 401)
        # Getting the type of 'member_name' (line 402)
        member_name_17540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 402, 86), 'member_name', False)
        # Processing the call keyword arguments (line 401)
        kwargs_17541 = {}
        # Getting the type of 'core_language_copy' (line 401)
        core_language_copy_17538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 401, 82), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 401)
        create_str_17539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 401, 82), core_language_copy_17538, 'create_str')
        # Calling create_str(args, kwargs) (line 401)
        create_str_call_result_17542 = invoke(stypy.reporting.localization.Localization(__file__, 401, 82), create_str_17539, *[member_name_17540], **kwargs_17541)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 400, 86), list_17536, create_str_call_result_17542)
        
        # Processing the call keyword arguments (line 400)
        kwargs_17543 = {}
        # Getting the type of 'functions_copy' (line 400)
        functions_copy_17533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 400, 34), 'functions_copy', False)
        # Obtaining the member 'create_call' of a type (line 400)
        create_call_17534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 400, 34), functions_copy_17533, 'create_call')
        # Calling create_call(args, kwargs) (line 400)
        create_call_call_result_17544 = invoke(stypy.reporting.localization.Localization(__file__, 400, 34), create_call_17534, *[get_type_of_member_func_17535, list_17536], **kwargs_17543)
        
        # Assigning a type to the variable 'get_type_of_member_call' (line 400)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 400, 8), 'get_type_of_member_call', create_call_call_result_17544)
        # SSA join for if statement (line 394)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Tuple (line 404):
    
    # Assigning a Call to a Name:
    
    # Call to create_temp_Assign(...): (line 404)
    # Processing the call arguments (line 404)
    # Getting the type of 'get_type_of_member_call' (line 404)
    get_type_of_member_call_17546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 50), 'get_type_of_member_call', False)
    # Getting the type of 'lineno' (line 404)
    lineno_17547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 75), 'lineno', False)
    # Getting the type of 'col_offset' (line 404)
    col_offset_17548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 83), 'col_offset', False)
    # Processing the call keyword arguments (line 404)
    kwargs_17549 = {}
    # Getting the type of 'create_temp_Assign' (line 404)
    create_temp_Assign_17545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 31), 'create_temp_Assign', False)
    # Calling create_temp_Assign(args, kwargs) (line 404)
    create_temp_Assign_call_result_17550 = invoke(stypy.reporting.localization.Localization(__file__, 404, 31), create_temp_Assign_17545, *[get_type_of_member_call_17546, lineno_17547, col_offset_17548], **kwargs_17549)
    
    # Assigning a type to the variable 'call_assignment_16809' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'call_assignment_16809', create_temp_Assign_call_result_17550)
    
    # Assigning a Call to a Name (line 404):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_16809' (line 404)
    call_assignment_16809_17551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'call_assignment_16809', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_17552 = stypy_get_value_from_tuple(call_assignment_16809_17551, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_16810' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'call_assignment_16810', stypy_get_value_from_tuple_call_result_17552)
    
    # Assigning a Name to a Name (line 404):
    # Getting the type of 'call_assignment_16810' (line 404)
    call_assignment_16810_17553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'call_assignment_16810')
    # Assigning a type to the variable 'member_stmts' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'member_stmts', call_assignment_16810_17553)
    
    # Assigning a Call to a Name (line 404):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_16809' (line 404)
    call_assignment_16809_17554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'call_assignment_16809', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_17555 = stypy_get_value_from_tuple(call_assignment_16809_17554, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_16811' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'call_assignment_16811', stypy_get_value_from_tuple_call_result_17555)
    
    # Assigning a Name to a Name (line 404):
    # Getting the type of 'call_assignment_16811' (line 404)
    call_assignment_16811_17556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 4), 'call_assignment_16811')
    # Assigning a type to the variable 'member_var' (line 404)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 18), 'member_var', call_assignment_16811_17556)
    
    # Obtaining an instance of the builtin type 'tuple' (line 406)
    tuple_17557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 406, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 406)
    # Adding element type (line 406)
    
    # Call to flatten_lists(...): (line 406)
    # Processing the call arguments (line 406)
    # Getting the type of 'comment' (line 406)
    comment_17559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 25), 'comment', False)
    # Getting the type of 'member_stmts' (line 406)
    member_stmts_17560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 34), 'member_stmts', False)
    # Processing the call keyword arguments (line 406)
    kwargs_17561 = {}
    # Getting the type of 'flatten_lists' (line 406)
    flatten_lists_17558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 11), 'flatten_lists', False)
    # Calling flatten_lists(args, kwargs) (line 406)
    flatten_lists_call_result_17562 = invoke(stypy.reporting.localization.Localization(__file__, 406, 11), flatten_lists_17558, *[comment_17559, member_stmts_17560], **kwargs_17561)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 11), tuple_17557, flatten_lists_call_result_17562)
    # Adding element type (line 406)
    # Getting the type of 'member_var' (line 406)
    member_var_17563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 406, 49), 'member_var')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 406, 11), tuple_17557, member_var_17563)
    
    # Assigning a type to the variable 'stypy_return_type' (line 406)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 406, 4), 'stypy_return_type', tuple_17557)
    
    # ################# End of 'create_get_type_of_member(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_get_type_of_member' in the type store
    # Getting the type of 'stypy_return_type' (line 384)
    stypy_return_type_17564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 384, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17564)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_get_type_of_member'
    return stypy_return_type_17564

# Assigning a type to the variable 'create_get_type_of_member' (line 384)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 384, 0), 'create_get_type_of_member', create_get_type_of_member)

@norecursion
def create_set_type_of_member(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_set_type_of_member'
    module_type_store = module_type_store.open_function_context('create_set_type_of_member', 409, 0, False)
    
    # Passed parameters checking function
    create_set_type_of_member.stypy_localization = localization
    create_set_type_of_member.stypy_type_of_self = None
    create_set_type_of_member.stypy_type_store = module_type_store
    create_set_type_of_member.stypy_function_name = 'create_set_type_of_member'
    create_set_type_of_member.stypy_param_names_list = ['owner_var', 'member_name', 'value', 'lineno', 'col_offset']
    create_set_type_of_member.stypy_varargs_param_name = None
    create_set_type_of_member.stypy_kwargs_param_name = None
    create_set_type_of_member.stypy_call_defaults = defaults
    create_set_type_of_member.stypy_call_varargs = varargs
    create_set_type_of_member.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_set_type_of_member', ['owner_var', 'member_name', 'value', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_set_type_of_member', localization, ['owner_var', 'member_name', 'value', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_set_type_of_member(...)' code ##################

    
    # Assigning a Call to a Name (line 410):
    
    # Assigning a Call to a Name (line 410):
    
    # Call to create_src_comment(...): (line 410)
    # Processing the call arguments (line 410)
    
    # Call to format(...): (line 410)
    # Processing the call arguments (line 410)
    # Getting the type of 'member_name' (line 410)
    member_name_17568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 89), 'member_name', False)
    # Processing the call keyword arguments (line 410)
    kwargs_17569 = {}
    str_17566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 410, 33), 'str', "Setting the type of the member '{0}' of a type")
    # Obtaining the member 'format' of a type (line 410)
    format_17567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 410, 33), str_17566, 'format')
    # Calling format(args, kwargs) (line 410)
    format_call_result_17570 = invoke(stypy.reporting.localization.Localization(__file__, 410, 33), format_17567, *[member_name_17568], **kwargs_17569)
    
    # Getting the type of 'lineno' (line 410)
    lineno_17571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 103), 'lineno', False)
    # Processing the call keyword arguments (line 410)
    kwargs_17572 = {}
    # Getting the type of 'create_src_comment' (line 410)
    create_src_comment_17565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 14), 'create_src_comment', False)
    # Calling create_src_comment(args, kwargs) (line 410)
    create_src_comment_call_result_17573 = invoke(stypy.reporting.localization.Localization(__file__, 410, 14), create_src_comment_17565, *[format_call_result_17570, lineno_17571], **kwargs_17572)
    
    # Assigning a type to the variable 'comment' (line 410)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 4), 'comment', create_src_comment_call_result_17573)
    
    # Assigning a Call to a Name (line 411):
    
    # Assigning a Call to a Name (line 411):
    
    # Call to create_localization(...): (line 411)
    # Processing the call arguments (line 411)
    # Getting the type of 'lineno' (line 411)
    lineno_17575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 39), 'lineno', False)
    # Getting the type of 'col_offset' (line 411)
    col_offset_17576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 47), 'col_offset', False)
    # Processing the call keyword arguments (line 411)
    kwargs_17577 = {}
    # Getting the type of 'create_localization' (line 411)
    create_localization_17574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 411, 19), 'create_localization', False)
    # Calling create_localization(args, kwargs) (line 411)
    create_localization_call_result_17578 = invoke(stypy.reporting.localization.Localization(__file__, 411, 19), create_localization_17574, *[lineno_17575, col_offset_17576], **kwargs_17577)
    
    # Assigning a type to the variable 'localization' (line 411)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 411, 4), 'localization', create_localization_call_result_17578)
    
    # Assigning a Call to a Name (line 418):
    
    # Assigning a Call to a Name (line 418):
    
    # Call to create_attribute(...): (line 418)
    # Processing the call arguments (line 418)
    # Getting the type of 'owner_var' (line 418)
    owner_var_17581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 66), 'owner_var', False)
    str_17582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 418, 77), 'str', 'set_type_of_member')
    # Processing the call keyword arguments (line 418)
    kwargs_17583 = {}
    # Getting the type of 'core_language_copy' (line 418)
    core_language_copy_17579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 418, 30), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 418)
    create_attribute_17580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 418, 30), core_language_copy_17579, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 418)
    create_attribute_call_result_17584 = invoke(stypy.reporting.localization.Localization(__file__, 418, 30), create_attribute_17580, *[owner_var_17581, str_17582], **kwargs_17583)
    
    # Assigning a type to the variable 'set_type_of_member_func' (line 418)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 418, 4), 'set_type_of_member_func', create_attribute_call_result_17584)
    
    # Assigning a Call to a Name (line 419):
    
    # Assigning a Call to a Name (line 419):
    
    # Call to create_call_expression(...): (line 419)
    # Processing the call arguments (line 419)
    # Getting the type of 'set_type_of_member_func' (line 419)
    set_type_of_member_func_17587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 68), 'set_type_of_member_func', False)
    
    # Obtaining an instance of the builtin type 'list' (line 419)
    list_17588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, 93), 'list')
    # Adding type elements to the builtin type 'list' instance (line 419)
    # Adding element type (line 419)
    # Getting the type of 'localization' (line 419)
    localization_17589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 94), 'localization', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 93), list_17588, localization_17589)
    # Adding element type (line 419)
    
    # Call to create_str(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'member_name' (line 421)
    member_name_17592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 93), 'member_name', False)
    # Processing the call keyword arguments (line 420)
    kwargs_17593 = {}
    # Getting the type of 'core_language_copy' (line 420)
    core_language_copy_17590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 89), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 420)
    create_str_17591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 89), core_language_copy_17590, 'create_str')
    # Calling create_str(args, kwargs) (line 420)
    create_str_call_result_17594 = invoke(stypy.reporting.localization.Localization(__file__, 420, 89), create_str_17591, *[member_name_17592], **kwargs_17593)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 93), list_17588, create_str_call_result_17594)
    # Adding element type (line 419)
    # Getting the type of 'value' (line 421)
    value_17595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 107), 'value', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 419, 93), list_17588, value_17595)
    
    # Processing the call keyword arguments (line 419)
    kwargs_17596 = {}
    # Getting the type of 'functions_copy' (line 419)
    functions_copy_17585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 419, 30), 'functions_copy', False)
    # Obtaining the member 'create_call_expression' of a type (line 419)
    create_call_expression_17586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 419, 30), functions_copy_17585, 'create_call_expression')
    # Calling create_call_expression(args, kwargs) (line 419)
    create_call_expression_call_result_17597 = invoke(stypy.reporting.localization.Localization(__file__, 419, 30), create_call_expression_17586, *[set_type_of_member_func_17587, list_17588], **kwargs_17596)
    
    # Assigning a type to the variable 'set_type_of_member_call' (line 419)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 419, 4), 'set_type_of_member_call', create_call_expression_call_result_17597)
    
    # Call to flatten_lists(...): (line 423)
    # Processing the call arguments (line 423)
    # Getting the type of 'comment' (line 423)
    comment_17599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 25), 'comment', False)
    # Getting the type of 'set_type_of_member_call' (line 423)
    set_type_of_member_call_17600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 34), 'set_type_of_member_call', False)
    # Processing the call keyword arguments (line 423)
    kwargs_17601 = {}
    # Getting the type of 'flatten_lists' (line 423)
    flatten_lists_17598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 423, 11), 'flatten_lists', False)
    # Calling flatten_lists(args, kwargs) (line 423)
    flatten_lists_call_result_17602 = invoke(stypy.reporting.localization.Localization(__file__, 423, 11), flatten_lists_17598, *[comment_17599, set_type_of_member_call_17600], **kwargs_17601)
    
    # Assigning a type to the variable 'stypy_return_type' (line 423)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 423, 4), 'stypy_return_type', flatten_lists_call_result_17602)
    
    # ################# End of 'create_set_type_of_member(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_set_type_of_member' in the type store
    # Getting the type of 'stypy_return_type' (line 409)
    stypy_return_type_17603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17603)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_set_type_of_member'
    return stypy_return_type_17603

# Assigning a type to the variable 'create_set_type_of_member' (line 409)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 0), 'create_set_type_of_member', create_set_type_of_member)

@norecursion
def create_add_stored_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_add_stored_type'
    module_type_store = module_type_store.open_function_context('create_add_stored_type', 426, 0, False)
    
    # Passed parameters checking function
    create_add_stored_type.stypy_localization = localization
    create_add_stored_type.stypy_type_of_self = None
    create_add_stored_type.stypy_type_store = module_type_store
    create_add_stored_type.stypy_function_name = 'create_add_stored_type'
    create_add_stored_type.stypy_param_names_list = ['owner_var', 'index', 'value', 'lineno', 'col_offset']
    create_add_stored_type.stypy_varargs_param_name = None
    create_add_stored_type.stypy_kwargs_param_name = None
    create_add_stored_type.stypy_call_defaults = defaults
    create_add_stored_type.stypy_call_varargs = varargs
    create_add_stored_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_add_stored_type', ['owner_var', 'index', 'value', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_add_stored_type', localization, ['owner_var', 'index', 'value', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_add_stored_type(...)' code ##################

    
    # Assigning a Call to a Name (line 427):
    
    # Assigning a Call to a Name (line 427):
    
    # Call to create_src_comment(...): (line 427)
    # Processing the call arguments (line 427)
    str_17605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 427, 33), 'str', 'Storing an element on a container')
    # Getting the type of 'lineno' (line 427)
    lineno_17606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 70), 'lineno', False)
    # Processing the call keyword arguments (line 427)
    kwargs_17607 = {}
    # Getting the type of 'create_src_comment' (line 427)
    create_src_comment_17604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 14), 'create_src_comment', False)
    # Calling create_src_comment(args, kwargs) (line 427)
    create_src_comment_call_result_17608 = invoke(stypy.reporting.localization.Localization(__file__, 427, 14), create_src_comment_17604, *[str_17605, lineno_17606], **kwargs_17607)
    
    # Assigning a type to the variable 'comment' (line 427)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'comment', create_src_comment_call_result_17608)
    
    # Assigning a Call to a Name (line 428):
    
    # Assigning a Call to a Name (line 428):
    
    # Call to create_localization(...): (line 428)
    # Processing the call arguments (line 428)
    # Getting the type of 'lineno' (line 428)
    lineno_17610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 39), 'lineno', False)
    # Getting the type of 'col_offset' (line 428)
    col_offset_17611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 47), 'col_offset', False)
    # Processing the call keyword arguments (line 428)
    kwargs_17612 = {}
    # Getting the type of 'create_localization' (line 428)
    create_localization_17609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 428, 19), 'create_localization', False)
    # Calling create_localization(args, kwargs) (line 428)
    create_localization_call_result_17613 = invoke(stypy.reporting.localization.Localization(__file__, 428, 19), create_localization_17609, *[lineno_17610, col_offset_17611], **kwargs_17612)
    
    # Assigning a type to the variable 'localization' (line 428)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 428, 4), 'localization', create_localization_call_result_17613)
    
    # Assigning a Call to a Name (line 430):
    
    # Assigning a Call to a Name (line 430):
    
    # Call to create_attribute(...): (line 430)
    # Processing the call arguments (line 430)
    # Getting the type of 'owner_var' (line 430)
    owner_var_17616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 56), 'owner_var', False)
    str_17617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 430, 67), 'str', 'add_key_and_value_type')
    # Processing the call keyword arguments (line 430)
    kwargs_17618 = {}
    # Getting the type of 'core_language_copy' (line 430)
    core_language_copy_17614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 20), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 430)
    create_attribute_17615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 430, 20), core_language_copy_17614, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 430)
    create_attribute_call_result_17619 = invoke(stypy.reporting.localization.Localization(__file__, 430, 20), create_attribute_17615, *[owner_var_17616, str_17617], **kwargs_17618)
    
    # Assigning a type to the variable 'add_type_func' (line 430)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 4), 'add_type_func', create_attribute_call_result_17619)
    
    # Assigning a Call to a Name (line 431):
    
    # Assigning a Call to a Name (line 431):
    
    # Call to Tuple(...): (line 431)
    # Processing the call keyword arguments (line 431)
    kwargs_17622 = {}
    # Getting the type of 'ast' (line 431)
    ast_17620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 18), 'ast', False)
    # Obtaining the member 'Tuple' of a type (line 431)
    Tuple_17621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 431, 18), ast_17620, 'Tuple')
    # Calling Tuple(args, kwargs) (line 431)
    Tuple_call_result_17623 = invoke(stypy.reporting.localization.Localization(__file__, 431, 18), Tuple_17621, *[], **kwargs_17622)
    
    # Assigning a type to the variable 'param_tuple' (line 431)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 4), 'param_tuple', Tuple_call_result_17623)
    
    # Assigning a List to a Attribute (line 432):
    
    # Assigning a List to a Attribute (line 432):
    
    # Obtaining an instance of the builtin type 'list' (line 432)
    list_17624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 432)
    # Adding element type (line 432)
    # Getting the type of 'index' (line 432)
    index_17625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 24), 'index')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 23), list_17624, index_17625)
    # Adding element type (line 432)
    # Getting the type of 'value' (line 432)
    value_17626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 31), 'value')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 23), list_17624, value_17626)
    
    # Getting the type of 'param_tuple' (line 432)
    param_tuple_17627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 4), 'param_tuple')
    # Setting the type of the member 'elts' of a type (line 432)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 4), param_tuple_17627, 'elts', list_17624)
    
    # Assigning a Call to a Name (line 433):
    
    # Assigning a Call to a Name (line 433):
    
    # Call to create_call_expression(...): (line 433)
    # Processing the call arguments (line 433)
    # Getting the type of 'add_type_func' (line 433)
    add_type_func_17630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 68), 'add_type_func', False)
    
    # Obtaining an instance of the builtin type 'list' (line 433)
    list_17631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 83), 'list')
    # Adding type elements to the builtin type 'list' instance (line 433)
    # Adding element type (line 433)
    # Getting the type of 'localization' (line 433)
    localization_17632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 84), 'localization', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 83), list_17631, localization_17632)
    # Adding element type (line 433)
    # Getting the type of 'param_tuple' (line 433)
    param_tuple_17633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 98), 'param_tuple', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 83), list_17631, param_tuple_17633)
    
    # Processing the call keyword arguments (line 433)
    kwargs_17634 = {}
    # Getting the type of 'functions_copy' (line 433)
    functions_copy_17628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 30), 'functions_copy', False)
    # Obtaining the member 'create_call_expression' of a type (line 433)
    create_call_expression_17629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 433, 30), functions_copy_17628, 'create_call_expression')
    # Calling create_call_expression(args, kwargs) (line 433)
    create_call_expression_call_result_17635 = invoke(stypy.reporting.localization.Localization(__file__, 433, 30), create_call_expression_17629, *[add_type_func_17630, list_17631], **kwargs_17634)
    
    # Assigning a type to the variable 'set_type_of_member_call' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'set_type_of_member_call', create_call_expression_call_result_17635)
    
    # Call to flatten_lists(...): (line 435)
    # Processing the call arguments (line 435)
    # Getting the type of 'comment' (line 435)
    comment_17637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 25), 'comment', False)
    # Getting the type of 'set_type_of_member_call' (line 435)
    set_type_of_member_call_17638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 34), 'set_type_of_member_call', False)
    # Processing the call keyword arguments (line 435)
    kwargs_17639 = {}
    # Getting the type of 'flatten_lists' (line 435)
    flatten_lists_17636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 435, 11), 'flatten_lists', False)
    # Calling flatten_lists(args, kwargs) (line 435)
    flatten_lists_call_result_17640 = invoke(stypy.reporting.localization.Localization(__file__, 435, 11), flatten_lists_17636, *[comment_17637, set_type_of_member_call_17638], **kwargs_17639)
    
    # Assigning a type to the variable 'stypy_return_type' (line 435)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 435, 4), 'stypy_return_type', flatten_lists_call_result_17640)
    
    # ################# End of 'create_add_stored_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_add_stored_type' in the type store
    # Getting the type of 'stypy_return_type' (line 426)
    stypy_return_type_17641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17641)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_add_stored_type'
    return stypy_return_type_17641

# Assigning a type to the variable 'create_add_stored_type' (line 426)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 0), 'create_add_stored_type', create_add_stored_type)
str_17642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 443, (-1)), 'str', '\nCode to deal with type store related functions_copy, assignments, cloning and other operations needed for the SSA algorithm\nimplementation\n')

# Assigning a Num to a Name (line 446):

# Assigning a Num to a Name (line 446):
int_17643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 446, 28), 'int')
# Assigning a type to the variable '__temp_type_store_counter' (line 446)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 0), '__temp_type_store_counter', int_17643)

@norecursion
def __new_temp_type_store(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__new_temp_type_store'
    module_type_store = module_type_store.open_function_context('__new_temp_type_store', 449, 0, False)
    
    # Passed parameters checking function
    __new_temp_type_store.stypy_localization = localization
    __new_temp_type_store.stypy_type_of_self = None
    __new_temp_type_store.stypy_type_store = module_type_store
    __new_temp_type_store.stypy_function_name = '__new_temp_type_store'
    __new_temp_type_store.stypy_param_names_list = []
    __new_temp_type_store.stypy_varargs_param_name = None
    __new_temp_type_store.stypy_kwargs_param_name = None
    __new_temp_type_store.stypy_call_defaults = defaults
    __new_temp_type_store.stypy_call_varargs = varargs
    __new_temp_type_store.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__new_temp_type_store', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__new_temp_type_store', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__new_temp_type_store(...)' code ##################

    # Marking variables as global (line 450)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 450, 4), '__temp_type_store_counter')
    
    # Getting the type of '__temp_type_store_counter' (line 451)
    temp_type_store_counter_17644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), '__temp_type_store_counter')
    int_17645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 451, 33), 'int')
    # Applying the binary operator '+=' (line 451)
    result_iadd_17646 = python_operator(stypy.reporting.localization.Localization(__file__, 451, 4), '+=', temp_type_store_counter_17644, int_17645)
    # Assigning a type to the variable '__temp_type_store_counter' (line 451)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 4), '__temp_type_store_counter', result_iadd_17646)
    
    # Getting the type of '__temp_type_store_counter' (line 452)
    temp_type_store_counter_17647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 11), '__temp_type_store_counter')
    # Assigning a type to the variable 'stypy_return_type' (line 452)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 4), 'stypy_return_type', temp_type_store_counter_17647)
    
    # ################# End of '__new_temp_type_store(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__new_temp_type_store' in the type store
    # Getting the type of 'stypy_return_type' (line 449)
    stypy_return_type_17648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 449, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17648)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__new_temp_type_store'
    return stypy_return_type_17648

# Assigning a type to the variable '__new_temp_type_store' (line 449)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 449, 0), '__new_temp_type_store', __new_temp_type_store)

@norecursion
def __new_type_store_name_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__new_type_store_name_str'
    module_type_store = module_type_store.open_function_context('__new_type_store_name_str', 455, 0, False)
    
    # Passed parameters checking function
    __new_type_store_name_str.stypy_localization = localization
    __new_type_store_name_str.stypy_type_of_self = None
    __new_type_store_name_str.stypy_type_store = module_type_store
    __new_type_store_name_str.stypy_function_name = '__new_type_store_name_str'
    __new_type_store_name_str.stypy_param_names_list = []
    __new_type_store_name_str.stypy_varargs_param_name = None
    __new_type_store_name_str.stypy_kwargs_param_name = None
    __new_type_store_name_str.stypy_call_defaults = defaults
    __new_type_store_name_str.stypy_call_varargs = varargs
    __new_type_store_name_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__new_type_store_name_str', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__new_type_store_name_str', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__new_type_store_name_str(...)' code ##################

    str_17649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 456, 11), 'str', '__temp_type_store')
    
    # Call to str(...): (line 456)
    # Processing the call arguments (line 456)
    
    # Call to __new_temp_type_store(...): (line 456)
    # Processing the call keyword arguments (line 456)
    kwargs_17652 = {}
    # Getting the type of '__new_temp_type_store' (line 456)
    new_temp_type_store_17651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 37), '__new_temp_type_store', False)
    # Calling __new_temp_type_store(args, kwargs) (line 456)
    new_temp_type_store_call_result_17653 = invoke(stypy.reporting.localization.Localization(__file__, 456, 37), new_temp_type_store_17651, *[], **kwargs_17652)
    
    # Processing the call keyword arguments (line 456)
    kwargs_17654 = {}
    # Getting the type of 'str' (line 456)
    str_17650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 456, 33), 'str', False)
    # Calling str(args, kwargs) (line 456)
    str_call_result_17655 = invoke(stypy.reporting.localization.Localization(__file__, 456, 33), str_17650, *[new_temp_type_store_call_result_17653], **kwargs_17654)
    
    # Applying the binary operator '+' (line 456)
    result_add_17656 = python_operator(stypy.reporting.localization.Localization(__file__, 456, 11), '+', str_17649, str_call_result_17655)
    
    # Assigning a type to the variable 'stypy_return_type' (line 456)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 456, 4), 'stypy_return_type', result_add_17656)
    
    # ################# End of '__new_type_store_name_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__new_type_store_name_str' in the type store
    # Getting the type of 'stypy_return_type' (line 455)
    stypy_return_type_17657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17657)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__new_type_store_name_str'
    return stypy_return_type_17657

# Assigning a type to the variable '__new_type_store_name_str' (line 455)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 0), '__new_type_store_name_str', __new_type_store_name_str)

@norecursion
def __new_temp_type_store_Name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 459)
    True_17658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 47), 'True')
    defaults = [True_17658]
    # Create a new context for function '__new_temp_type_store_Name'
    module_type_store = module_type_store.open_function_context('__new_temp_type_store_Name', 459, 0, False)
    
    # Passed parameters checking function
    __new_temp_type_store_Name.stypy_localization = localization
    __new_temp_type_store_Name.stypy_type_of_self = None
    __new_temp_type_store_Name.stypy_type_store = module_type_store
    __new_temp_type_store_Name.stypy_function_name = '__new_temp_type_store_Name'
    __new_temp_type_store_Name.stypy_param_names_list = ['right_hand_side']
    __new_temp_type_store_Name.stypy_varargs_param_name = None
    __new_temp_type_store_Name.stypy_kwargs_param_name = None
    __new_temp_type_store_Name.stypy_call_defaults = defaults
    __new_temp_type_store_Name.stypy_call_varargs = varargs
    __new_temp_type_store_Name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__new_temp_type_store_Name', ['right_hand_side'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__new_temp_type_store_Name', localization, ['right_hand_side'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__new_temp_type_store_Name(...)' code ##################

    # Getting the type of 'right_hand_side' (line 460)
    right_hand_side_17659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 460, 7), 'right_hand_side')
    # Testing if the type of an if condition is none (line 460)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 460, 4), right_hand_side_17659):
        pass
    else:
        
        # Testing the type of an if condition (line 460)
        if_condition_17660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 460, 4), right_hand_side_17659)
        # Assigning a type to the variable 'if_condition_17660' (line 460)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 460, 4), 'if_condition_17660', if_condition_17660)
        # SSA begins for if statement (line 460)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to Name(...): (line 461)
        # Processing the call keyword arguments (line 461)
        
        # Call to __new_type_store_name_str(...): (line 461)
        # Processing the call keyword arguments (line 461)
        kwargs_17664 = {}
        # Getting the type of '__new_type_store_name_str' (line 461)
        new_type_store_name_str_17663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 27), '__new_type_store_name_str', False)
        # Calling __new_type_store_name_str(args, kwargs) (line 461)
        new_type_store_name_str_call_result_17665 = invoke(stypy.reporting.localization.Localization(__file__, 461, 27), new_type_store_name_str_17663, *[], **kwargs_17664)
        
        keyword_17666 = new_type_store_name_str_call_result_17665
        
        # Call to Load(...): (line 461)
        # Processing the call keyword arguments (line 461)
        kwargs_17669 = {}
        # Getting the type of 'ast' (line 461)
        ast_17667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 60), 'ast', False)
        # Obtaining the member 'Load' of a type (line 461)
        Load_17668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 60), ast_17667, 'Load')
        # Calling Load(args, kwargs) (line 461)
        Load_call_result_17670 = invoke(stypy.reporting.localization.Localization(__file__, 461, 60), Load_17668, *[], **kwargs_17669)
        
        keyword_17671 = Load_call_result_17670
        kwargs_17672 = {'ctx': keyword_17671, 'id': keyword_17666}
        # Getting the type of 'ast' (line 461)
        ast_17661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 15), 'ast', False)
        # Obtaining the member 'Name' of a type (line 461)
        Name_17662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 461, 15), ast_17661, 'Name')
        # Calling Name(args, kwargs) (line 461)
        Name_call_result_17673 = invoke(stypy.reporting.localization.Localization(__file__, 461, 15), Name_17662, *[], **kwargs_17672)
        
        # Assigning a type to the variable 'stypy_return_type' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 8), 'stypy_return_type', Name_call_result_17673)
        # SSA join for if statement (line 460)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to Name(...): (line 462)
    # Processing the call keyword arguments (line 462)
    
    # Call to __new_type_store_name_str(...): (line 462)
    # Processing the call keyword arguments (line 462)
    kwargs_17677 = {}
    # Getting the type of '__new_type_store_name_str' (line 462)
    new_type_store_name_str_17676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 23), '__new_type_store_name_str', False)
    # Calling __new_type_store_name_str(args, kwargs) (line 462)
    new_type_store_name_str_call_result_17678 = invoke(stypy.reporting.localization.Localization(__file__, 462, 23), new_type_store_name_str_17676, *[], **kwargs_17677)
    
    keyword_17679 = new_type_store_name_str_call_result_17678
    
    # Call to Store(...): (line 462)
    # Processing the call keyword arguments (line 462)
    kwargs_17682 = {}
    # Getting the type of 'ast' (line 462)
    ast_17680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 56), 'ast', False)
    # Obtaining the member 'Store' of a type (line 462)
    Store_17681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 56), ast_17680, 'Store')
    # Calling Store(args, kwargs) (line 462)
    Store_call_result_17683 = invoke(stypy.reporting.localization.Localization(__file__, 462, 56), Store_17681, *[], **kwargs_17682)
    
    keyword_17684 = Store_call_result_17683
    kwargs_17685 = {'ctx': keyword_17684, 'id': keyword_17679}
    # Getting the type of 'ast' (line 462)
    ast_17674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 11), 'ast', False)
    # Obtaining the member 'Name' of a type (line 462)
    Name_17675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 462, 11), ast_17674, 'Name')
    # Calling Name(args, kwargs) (line 462)
    Name_call_result_17686 = invoke(stypy.reporting.localization.Localization(__file__, 462, 11), Name_17675, *[], **kwargs_17685)
    
    # Assigning a type to the variable 'stypy_return_type' (line 462)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 4), 'stypy_return_type', Name_call_result_17686)
    
    # ################# End of '__new_temp_type_store_Name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__new_temp_type_store_Name' in the type store
    # Getting the type of 'stypy_return_type' (line 459)
    stypy_return_type_17687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 459, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17687)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__new_temp_type_store_Name'
    return stypy_return_type_17687

# Assigning a type to the variable '__new_temp_type_store_Name' (line 459)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 459, 0), '__new_temp_type_store_Name', __new_temp_type_store_Name)

@norecursion
def create_type_store(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'default_module_type_store_var_name' (line 465)
    default_module_type_store_var_name_17688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 38), 'default_module_type_store_var_name')
    defaults = [default_module_type_store_var_name_17688]
    # Create a new context for function 'create_type_store'
    module_type_store = module_type_store.open_function_context('create_type_store', 465, 0, False)
    
    # Passed parameters checking function
    create_type_store.stypy_localization = localization
    create_type_store.stypy_type_of_self = None
    create_type_store.stypy_type_store = module_type_store
    create_type_store.stypy_function_name = 'create_type_store'
    create_type_store.stypy_param_names_list = ['type_store_name']
    create_type_store.stypy_varargs_param_name = None
    create_type_store.stypy_kwargs_param_name = None
    create_type_store.stypy_call_defaults = defaults
    create_type_store.stypy_call_varargs = varargs
    create_type_store.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_type_store', ['type_store_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_type_store', localization, ['type_store_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_type_store(...)' code ##################

    
    # Assigning a Call to a Name (line 466):
    
    # Assigning a Call to a Name (line 466):
    
    # Call to create_Name(...): (line 466)
    # Processing the call arguments (line 466)
    str_17691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 466, 46), 'str', '__file__')
    # Processing the call keyword arguments (line 466)
    kwargs_17692 = {}
    # Getting the type of 'core_language_copy' (line 466)
    core_language_copy_17689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 466, 15), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 466)
    create_Name_17690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 466, 15), core_language_copy_17689, 'create_Name')
    # Calling create_Name(args, kwargs) (line 466)
    create_Name_call_result_17693 = invoke(stypy.reporting.localization.Localization(__file__, 466, 15), create_Name_17690, *[str_17691], **kwargs_17692)
    
    # Assigning a type to the variable 'call_arg' (line 466)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 466, 4), 'call_arg', create_Name_call_result_17693)
    
    # Assigning a Call to a Name (line 467):
    
    # Assigning a Call to a Name (line 467):
    
    # Call to create_Name(...): (line 467)
    # Processing the call arguments (line 467)
    str_17696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 467, 47), 'str', 'TypeStore')
    # Processing the call keyword arguments (line 467)
    kwargs_17697 = {}
    # Getting the type of 'core_language_copy' (line 467)
    core_language_copy_17694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 16), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 467)
    create_Name_17695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 467, 16), core_language_copy_17694, 'create_Name')
    # Calling create_Name(args, kwargs) (line 467)
    create_Name_call_result_17698 = invoke(stypy.reporting.localization.Localization(__file__, 467, 16), create_Name_17695, *[str_17696], **kwargs_17697)
    
    # Assigning a type to the variable 'call_func' (line 467)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 4), 'call_func', create_Name_call_result_17698)
    
    # Assigning a Call to a Name (line 468):
    
    # Assigning a Call to a Name (line 468):
    
    # Call to create_call(...): (line 468)
    # Processing the call arguments (line 468)
    # Getting the type of 'call_func' (line 468)
    call_func_17701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 38), 'call_func', False)
    # Getting the type of 'call_arg' (line 468)
    call_arg_17702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 49), 'call_arg', False)
    # Processing the call keyword arguments (line 468)
    kwargs_17703 = {}
    # Getting the type of 'functions_copy' (line 468)
    functions_copy_17699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 468, 11), 'functions_copy', False)
    # Obtaining the member 'create_call' of a type (line 468)
    create_call_17700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 468, 11), functions_copy_17699, 'create_call')
    # Calling create_call(args, kwargs) (line 468)
    create_call_call_result_17704 = invoke(stypy.reporting.localization.Localization(__file__, 468, 11), create_call_17700, *[call_func_17701, call_arg_17702], **kwargs_17703)
    
    # Assigning a type to the variable 'call' (line 468)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 468, 4), 'call', create_call_call_result_17704)
    
    # Assigning a Call to a Name (line 469):
    
    # Assigning a Call to a Name (line 469):
    
    # Call to create_Name(...): (line 469)
    # Processing the call arguments (line 469)
    # Getting the type of 'type_store_name' (line 469)
    type_store_name_17707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 51), 'type_store_name', False)
    # Getting the type of 'False' (line 469)
    False_17708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 68), 'False', False)
    # Processing the call keyword arguments (line 469)
    kwargs_17709 = {}
    # Getting the type of 'core_language_copy' (line 469)
    core_language_copy_17705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 469, 20), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 469)
    create_Name_17706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 469, 20), core_language_copy_17705, 'create_Name')
    # Calling create_Name(args, kwargs) (line 469)
    create_Name_call_result_17710 = invoke(stypy.reporting.localization.Localization(__file__, 469, 20), create_Name_17706, *[type_store_name_17707, False_17708], **kwargs_17709)
    
    # Assigning a type to the variable 'assign_target' (line 469)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 469, 4), 'assign_target', create_Name_call_result_17710)
    
    # Assigning a Call to a Name (line 470):
    
    # Assigning a Call to a Name (line 470):
    
    # Call to create_Assign(...): (line 470)
    # Processing the call arguments (line 470)
    # Getting the type of 'assign_target' (line 470)
    assign_target_17713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 46), 'assign_target', False)
    # Getting the type of 'call' (line 470)
    call_17714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 61), 'call', False)
    # Processing the call keyword arguments (line 470)
    kwargs_17715 = {}
    # Getting the type of 'core_language_copy' (line 470)
    core_language_copy_17711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 470, 13), 'core_language_copy', False)
    # Obtaining the member 'create_Assign' of a type (line 470)
    create_Assign_17712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 470, 13), core_language_copy_17711, 'create_Assign')
    # Calling create_Assign(args, kwargs) (line 470)
    create_Assign_call_result_17716 = invoke(stypy.reporting.localization.Localization(__file__, 470, 13), create_Assign_17712, *[assign_target_17713, call_17714], **kwargs_17715)
    
    # Assigning a type to the variable 'assign' (line 470)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 470, 4), 'assign', create_Assign_call_result_17716)
    # Getting the type of 'assign' (line 472)
    assign_17717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 472, 11), 'assign')
    # Assigning a type to the variable 'stypy_return_type' (line 472)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 472, 4), 'stypy_return_type', assign_17717)
    
    # ################# End of 'create_type_store(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_type_store' in the type store
    # Getting the type of 'stypy_return_type' (line 465)
    stypy_return_type_17718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 465, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17718)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_type_store'
    return stypy_return_type_17718

# Assigning a type to the variable 'create_type_store' (line 465)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 465, 0), 'create_type_store', create_type_store)

@norecursion
def create_temp_type_store_Assign(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_temp_type_store_Assign'
    module_type_store = module_type_store.open_function_context('create_temp_type_store_Assign', 475, 0, False)
    
    # Passed parameters checking function
    create_temp_type_store_Assign.stypy_localization = localization
    create_temp_type_store_Assign.stypy_type_of_self = None
    create_temp_type_store_Assign.stypy_type_store = module_type_store
    create_temp_type_store_Assign.stypy_function_name = 'create_temp_type_store_Assign'
    create_temp_type_store_Assign.stypy_param_names_list = ['right_hand_side']
    create_temp_type_store_Assign.stypy_varargs_param_name = None
    create_temp_type_store_Assign.stypy_kwargs_param_name = None
    create_temp_type_store_Assign.stypy_call_defaults = defaults
    create_temp_type_store_Assign.stypy_call_varargs = varargs
    create_temp_type_store_Assign.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_temp_type_store_Assign', ['right_hand_side'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_temp_type_store_Assign', localization, ['right_hand_side'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_temp_type_store_Assign(...)' code ##################

    
    # Assigning a Call to a Name (line 476):
    
    # Assigning a Call to a Name (line 476):
    
    # Call to __new_temp_type_store_Name(...): (line 476)
    # Processing the call keyword arguments (line 476)
    # Getting the type of 'False' (line 476)
    False_17720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 64), 'False', False)
    keyword_17721 = False_17720
    kwargs_17722 = {'right_hand_side': keyword_17721}
    # Getting the type of '__new_temp_type_store_Name' (line 476)
    new_temp_type_store_Name_17719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 21), '__new_temp_type_store_Name', False)
    # Calling __new_temp_type_store_Name(args, kwargs) (line 476)
    new_temp_type_store_Name_call_result_17723 = invoke(stypy.reporting.localization.Localization(__file__, 476, 21), new_temp_type_store_Name_17719, *[], **kwargs_17722)
    
    # Assigning a type to the variable 'left_hand_side' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'left_hand_side', new_temp_type_store_Name_call_result_17723)
    
    # Assigning a Call to a Name (line 477):
    
    # Assigning a Call to a Name (line 477):
    
    # Call to Assign(...): (line 477)
    # Processing the call arguments (line 477)
    
    # Obtaining an instance of the builtin type 'list' (line 477)
    list_17726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 477, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 477)
    # Adding element type (line 477)
    # Getting the type of 'left_hand_side' (line 477)
    left_hand_side_17727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 35), 'left_hand_side', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 477, 34), list_17726, left_hand_side_17727)
    
    # Getting the type of 'right_hand_side' (line 477)
    right_hand_side_17728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 52), 'right_hand_side', False)
    # Processing the call keyword arguments (line 477)
    kwargs_17729 = {}
    # Getting the type of 'ast' (line 477)
    ast_17724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 23), 'ast', False)
    # Obtaining the member 'Assign' of a type (line 477)
    Assign_17725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 23), ast_17724, 'Assign')
    # Calling Assign(args, kwargs) (line 477)
    Assign_call_result_17730 = invoke(stypy.reporting.localization.Localization(__file__, 477, 23), Assign_17725, *[list_17726, right_hand_side_17728], **kwargs_17729)
    
    # Assigning a type to the variable 'assign_statement' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'assign_statement', Assign_call_result_17730)
    
    # Obtaining an instance of the builtin type 'tuple' (line 478)
    tuple_17731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 478, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 478)
    # Adding element type (line 478)
    # Getting the type of 'assign_statement' (line 478)
    assign_statement_17732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 11), 'assign_statement')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 11), tuple_17731, assign_statement_17732)
    # Adding element type (line 478)
    # Getting the type of 'left_hand_side' (line 478)
    left_hand_side_17733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 29), 'left_hand_side')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 478, 11), tuple_17731, left_hand_side_17733)
    
    # Assigning a type to the variable 'stypy_return_type' (line 478)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 478, 4), 'stypy_return_type', tuple_17731)
    
    # ################# End of 'create_temp_type_store_Assign(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_temp_type_store_Assign' in the type store
    # Getting the type of 'stypy_return_type' (line 475)
    stypy_return_type_17734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17734)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_temp_type_store_Assign'
    return stypy_return_type_17734

# Assigning a type to the variable 'create_temp_type_store_Assign' (line 475)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 0), 'create_temp_type_store_Assign', create_temp_type_store_Assign)

@norecursion
def create_clone_type_store(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_clone_type_store'
    module_type_store = module_type_store.open_function_context('create_clone_type_store', 481, 0, False)
    
    # Passed parameters checking function
    create_clone_type_store.stypy_localization = localization
    create_clone_type_store.stypy_type_of_self = None
    create_clone_type_store.stypy_type_store = module_type_store
    create_clone_type_store.stypy_function_name = 'create_clone_type_store'
    create_clone_type_store.stypy_param_names_list = []
    create_clone_type_store.stypy_varargs_param_name = None
    create_clone_type_store.stypy_kwargs_param_name = None
    create_clone_type_store.stypy_call_defaults = defaults
    create_clone_type_store.stypy_call_varargs = varargs
    create_clone_type_store.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_clone_type_store', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_clone_type_store', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_clone_type_store(...)' code ##################

    
    # Assigning a Call to a Name (line 482):
    
    # Assigning a Call to a Name (line 482):
    
    # Call to create_attribute(...): (line 482)
    # Processing the call arguments (line 482)
    str_17737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 52), 'str', 'type_store')
    str_17738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 482, 66), 'str', 'clone_type_store')
    # Processing the call keyword arguments (line 482)
    kwargs_17739 = {}
    # Getting the type of 'core_language_copy' (line 482)
    core_language_copy_17735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 16), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 482)
    create_attribute_17736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 482, 16), core_language_copy_17735, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 482)
    create_attribute_call_result_17740 = invoke(stypy.reporting.localization.Localization(__file__, 482, 16), create_attribute_17736, *[str_17737, str_17738], **kwargs_17739)
    
    # Assigning a type to the variable 'attribute' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'attribute', create_attribute_call_result_17740)
    
    # Assigning a Call to a Name (line 483):
    
    # Assigning a Call to a Name (line 483):
    
    # Call to create_call(...): (line 483)
    # Processing the call arguments (line 483)
    # Getting the type of 'attribute' (line 483)
    attribute_17743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 44), 'attribute', False)
    
    # Obtaining an instance of the builtin type 'list' (line 483)
    list_17744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 483, 55), 'list')
    # Adding type elements to the builtin type 'list' instance (line 483)
    
    # Processing the call keyword arguments (line 483)
    kwargs_17745 = {}
    # Getting the type of 'functions_copy' (line 483)
    functions_copy_17741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 483, 17), 'functions_copy', False)
    # Obtaining the member 'create_call' of a type (line 483)
    create_call_17742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 483, 17), functions_copy_17741, 'create_call')
    # Calling create_call(args, kwargs) (line 483)
    create_call_call_result_17746 = invoke(stypy.reporting.localization.Localization(__file__, 483, 17), create_call_17742, *[attribute_17743, list_17744], **kwargs_17745)
    
    # Assigning a type to the variable 'clone_call' (line 483)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 483, 4), 'clone_call', create_call_call_result_17746)
    
    # Call to create_temp_type_store_Assign(...): (line 485)
    # Processing the call arguments (line 485)
    # Getting the type of 'clone_call' (line 485)
    clone_call_17748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 41), 'clone_call', False)
    # Processing the call keyword arguments (line 485)
    kwargs_17749 = {}
    # Getting the type of 'create_temp_type_store_Assign' (line 485)
    create_temp_type_store_Assign_17747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 485, 11), 'create_temp_type_store_Assign', False)
    # Calling create_temp_type_store_Assign(args, kwargs) (line 485)
    create_temp_type_store_Assign_call_result_17750 = invoke(stypy.reporting.localization.Localization(__file__, 485, 11), create_temp_type_store_Assign_17747, *[clone_call_17748], **kwargs_17749)
    
    # Assigning a type to the variable 'stypy_return_type' (line 485)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 485, 4), 'stypy_return_type', create_temp_type_store_Assign_call_result_17750)
    
    # ################# End of 'create_clone_type_store(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_clone_type_store' in the type store
    # Getting the type of 'stypy_return_type' (line 481)
    stypy_return_type_17751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17751)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_clone_type_store'
    return stypy_return_type_17751

# Assigning a type to the variable 'create_clone_type_store' (line 481)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 0), 'create_clone_type_store', create_clone_type_store)

@norecursion
def create_set_unreferenced_var_check(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_set_unreferenced_var_check'
    module_type_store = module_type_store.open_function_context('create_set_unreferenced_var_check', 488, 0, False)
    
    # Passed parameters checking function
    create_set_unreferenced_var_check.stypy_localization = localization
    create_set_unreferenced_var_check.stypy_type_of_self = None
    create_set_unreferenced_var_check.stypy_type_store = module_type_store
    create_set_unreferenced_var_check.stypy_function_name = 'create_set_unreferenced_var_check'
    create_set_unreferenced_var_check.stypy_param_names_list = ['state']
    create_set_unreferenced_var_check.stypy_varargs_param_name = None
    create_set_unreferenced_var_check.stypy_kwargs_param_name = None
    create_set_unreferenced_var_check.stypy_call_defaults = defaults
    create_set_unreferenced_var_check.stypy_call_varargs = varargs
    create_set_unreferenced_var_check.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_set_unreferenced_var_check', ['state'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_set_unreferenced_var_check', localization, ['state'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_set_unreferenced_var_check(...)' code ##################

    # Getting the type of 'ENABLE_CODING_ADVICES' (line 489)
    ENABLE_CODING_ADVICES_17752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 489, 7), 'ENABLE_CODING_ADVICES')
    # Testing if the type of an if condition is none (line 489)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 489, 4), ENABLE_CODING_ADVICES_17752):
        
        # Obtaining an instance of the builtin type 'list' (line 495)
        list_17775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 495)
        
        # Assigning a type to the variable 'stypy_return_type' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'stypy_return_type', list_17775)
    else:
        
        # Testing the type of an if condition (line 489)
        if_condition_17753 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 489, 4), ENABLE_CODING_ADVICES_17752)
        # Assigning a type to the variable 'if_condition_17753' (line 489)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 489, 4), 'if_condition_17753', if_condition_17753)
        # SSA begins for if statement (line 489)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 490):
        
        # Assigning a Call to a Name (line 490):
        
        # Call to create_attribute(...): (line 490)
        # Processing the call arguments (line 490)
        str_17756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 56), 'str', 'type_store')
        str_17757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 490, 70), 'str', 'set_check_unreferenced_vars')
        # Processing the call keyword arguments (line 490)
        kwargs_17758 = {}
        # Getting the type of 'core_language_copy' (line 490)
        core_language_copy_17754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 490, 20), 'core_language_copy', False)
        # Obtaining the member 'create_attribute' of a type (line 490)
        create_attribute_17755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 490, 20), core_language_copy_17754, 'create_attribute')
        # Calling create_attribute(args, kwargs) (line 490)
        create_attribute_call_result_17759 = invoke(stypy.reporting.localization.Localization(__file__, 490, 20), create_attribute_17755, *[str_17756, str_17757], **kwargs_17758)
        
        # Assigning a type to the variable 'attribute' (line 490)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 490, 8), 'attribute', create_attribute_call_result_17759)
        
        # Assigning a Call to a Name (line 491):
        
        # Assigning a Call to a Name (line 491):
        
        # Call to create_call_expression(...): (line 491)
        # Processing the call arguments (line 491)
        # Getting the type of 'attribute' (line 491)
        attribute_17762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 54), 'attribute', False)
        
        # Obtaining an instance of the builtin type 'list' (line 491)
        list_17763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 491, 65), 'list')
        # Adding type elements to the builtin type 'list' instance (line 491)
        # Adding element type (line 491)
        
        # Call to create_Name(...): (line 491)
        # Processing the call arguments (line 491)
        
        # Call to str(...): (line 491)
        # Processing the call arguments (line 491)
        # Getting the type of 'state' (line 491)
        state_17767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 101), 'state', False)
        # Processing the call keyword arguments (line 491)
        kwargs_17768 = {}
        # Getting the type of 'str' (line 491)
        str_17766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 97), 'str', False)
        # Calling str(args, kwargs) (line 491)
        str_call_result_17769 = invoke(stypy.reporting.localization.Localization(__file__, 491, 97), str_17766, *[state_17767], **kwargs_17768)
        
        # Processing the call keyword arguments (line 491)
        kwargs_17770 = {}
        # Getting the type of 'core_language_copy' (line 491)
        core_language_copy_17764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 66), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 491)
        create_Name_17765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 66), core_language_copy_17764, 'create_Name')
        # Calling create_Name(args, kwargs) (line 491)
        create_Name_call_result_17771 = invoke(stypy.reporting.localization.Localization(__file__, 491, 66), create_Name_17765, *[str_call_result_17769], **kwargs_17770)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 491, 65), list_17763, create_Name_call_result_17771)
        
        # Processing the call keyword arguments (line 491)
        kwargs_17772 = {}
        # Getting the type of 'functions_copy' (line 491)
        functions_copy_17760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 491, 16), 'functions_copy', False)
        # Obtaining the member 'create_call_expression' of a type (line 491)
        create_call_expression_17761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 491, 16), functions_copy_17760, 'create_call_expression')
        # Calling create_call_expression(args, kwargs) (line 491)
        create_call_expression_call_result_17773 = invoke(stypy.reporting.localization.Localization(__file__, 491, 16), create_call_expression_17761, *[attribute_17762, list_17763], **kwargs_17772)
        
        # Assigning a type to the variable 'call_' (line 491)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 491, 8), 'call_', create_call_expression_call_result_17773)
        # Getting the type of 'call_' (line 493)
        call__17774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 493, 15), 'call_')
        # Assigning a type to the variable 'stypy_return_type' (line 493)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 493, 8), 'stypy_return_type', call__17774)
        # SSA branch for the else part of an if statement (line 489)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'list' (line 495)
        list_17775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 495, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 495)
        
        # Assigning a type to the variable 'stypy_return_type' (line 495)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 495, 8), 'stypy_return_type', list_17775)
        # SSA join for if statement (line 489)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'create_set_unreferenced_var_check(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_set_unreferenced_var_check' in the type store
    # Getting the type of 'stypy_return_type' (line 488)
    stypy_return_type_17776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 488, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17776)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_set_unreferenced_var_check'
    return stypy_return_type_17776

# Assigning a type to the variable 'create_set_unreferenced_var_check' (line 488)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 488, 0), 'create_set_unreferenced_var_check', create_set_unreferenced_var_check)

@norecursion
def create_set_type_store(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 498)
    True_17777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 50), 'True')
    defaults = [True_17777]
    # Create a new context for function 'create_set_type_store'
    module_type_store = module_type_store.open_function_context('create_set_type_store', 498, 0, False)
    
    # Passed parameters checking function
    create_set_type_store.stypy_localization = localization
    create_set_type_store.stypy_type_of_self = None
    create_set_type_store.stypy_type_store = module_type_store
    create_set_type_store.stypy_function_name = 'create_set_type_store'
    create_set_type_store.stypy_param_names_list = ['type_store_param', 'clone']
    create_set_type_store.stypy_varargs_param_name = None
    create_set_type_store.stypy_kwargs_param_name = None
    create_set_type_store.stypy_call_defaults = defaults
    create_set_type_store.stypy_call_varargs = varargs
    create_set_type_store.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_set_type_store', ['type_store_param', 'clone'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_set_type_store', localization, ['type_store_param', 'clone'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_set_type_store(...)' code ##################

    
    # Assigning a Call to a Name (line 499):
    
    # Assigning a Call to a Name (line 499):
    
    # Call to create_attribute(...): (line 499)
    # Processing the call arguments (line 499)
    str_17780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 52), 'str', 'type_store')
    str_17781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 499, 66), 'str', 'set_type_store')
    # Processing the call keyword arguments (line 499)
    kwargs_17782 = {}
    # Getting the type of 'core_language_copy' (line 499)
    core_language_copy_17778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 499, 16), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 499)
    create_attribute_17779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 499, 16), core_language_copy_17778, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 499)
    create_attribute_call_result_17783 = invoke(stypy.reporting.localization.Localization(__file__, 499, 16), create_attribute_17779, *[str_17780, str_17781], **kwargs_17782)
    
    # Assigning a type to the variable 'attribute' (line 499)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 499, 4), 'attribute', create_attribute_call_result_17783)
    # Getting the type of 'clone' (line 501)
    clone_17784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 501, 7), 'clone')
    # Testing if the type of an if condition is none (line 501)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 501, 4), clone_17784):
        
        # Assigning a Call to a Name (line 504):
        
        # Assigning a Call to a Name (line 504):
        
        # Call to create_Name(...): (line 504)
        # Processing the call arguments (line 504)
        str_17793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 53), 'str', 'False')
        # Processing the call keyword arguments (line 504)
        kwargs_17794 = {}
        # Getting the type of 'core_language_copy' (line 504)
        core_language_copy_17791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 22), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 504)
        create_Name_17792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 22), core_language_copy_17791, 'create_Name')
        # Calling create_Name(args, kwargs) (line 504)
        create_Name_call_result_17795 = invoke(stypy.reporting.localization.Localization(__file__, 504, 22), create_Name_17792, *[str_17793], **kwargs_17794)
        
        # Assigning a type to the variable 'clone_param' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'clone_param', create_Name_call_result_17795)
    else:
        
        # Testing the type of an if condition (line 501)
        if_condition_17785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 501, 4), clone_17784)
        # Assigning a type to the variable 'if_condition_17785' (line 501)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 501, 4), 'if_condition_17785', if_condition_17785)
        # SSA begins for if statement (line 501)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 502):
        
        # Assigning a Call to a Name (line 502):
        
        # Call to create_Name(...): (line 502)
        # Processing the call arguments (line 502)
        str_17788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 502, 53), 'str', 'True')
        # Processing the call keyword arguments (line 502)
        kwargs_17789 = {}
        # Getting the type of 'core_language_copy' (line 502)
        core_language_copy_17786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 502, 22), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 502)
        create_Name_17787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 502, 22), core_language_copy_17786, 'create_Name')
        # Calling create_Name(args, kwargs) (line 502)
        create_Name_call_result_17790 = invoke(stypy.reporting.localization.Localization(__file__, 502, 22), create_Name_17787, *[str_17788], **kwargs_17789)
        
        # Assigning a type to the variable 'clone_param' (line 502)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 502, 8), 'clone_param', create_Name_call_result_17790)
        # SSA branch for the else part of an if statement (line 501)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 504):
        
        # Assigning a Call to a Name (line 504):
        
        # Call to create_Name(...): (line 504)
        # Processing the call arguments (line 504)
        str_17793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 504, 53), 'str', 'False')
        # Processing the call keyword arguments (line 504)
        kwargs_17794 = {}
        # Getting the type of 'core_language_copy' (line 504)
        core_language_copy_17791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 504, 22), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 504)
        create_Name_17792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 504, 22), core_language_copy_17791, 'create_Name')
        # Calling create_Name(args, kwargs) (line 504)
        create_Name_call_result_17795 = invoke(stypy.reporting.localization.Localization(__file__, 504, 22), create_Name_17792, *[str_17793], **kwargs_17794)
        
        # Assigning a type to the variable 'clone_param' (line 504)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 504, 8), 'clone_param', create_Name_call_result_17795)
        # SSA join for if statement (line 501)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 506):
    
    # Assigning a Call to a Name (line 506):
    
    # Call to create_call(...): (line 506)
    # Processing the call arguments (line 506)
    # Getting the type of 'attribute' (line 506)
    attribute_17798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 42), 'attribute', False)
    
    # Obtaining an instance of the builtin type 'list' (line 506)
    list_17799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 506, 53), 'list')
    # Adding type elements to the builtin type 'list' instance (line 506)
    # Adding element type (line 506)
    # Getting the type of 'type_store_param' (line 506)
    type_store_param_17800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 54), 'type_store_param', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 53), list_17799, type_store_param_17800)
    # Adding element type (line 506)
    # Getting the type of 'clone_param' (line 506)
    clone_param_17801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 72), 'clone_param', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 506, 53), list_17799, clone_param_17801)
    
    # Processing the call keyword arguments (line 506)
    kwargs_17802 = {}
    # Getting the type of 'functions_copy' (line 506)
    functions_copy_17796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 506, 15), 'functions_copy', False)
    # Obtaining the member 'create_call' of a type (line 506)
    create_call_17797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 506, 15), functions_copy_17796, 'create_call')
    # Calling create_call(args, kwargs) (line 506)
    create_call_call_result_17803 = invoke(stypy.reporting.localization.Localization(__file__, 506, 15), create_call_17797, *[attribute_17798, list_17799], **kwargs_17802)
    
    # Assigning a type to the variable 'set_call' (line 506)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 506, 4), 'set_call', create_call_call_result_17803)
    
    # Assigning a Call to a Name (line 508):
    
    # Assigning a Call to a Name (line 508):
    
    # Call to Expr(...): (line 508)
    # Processing the call keyword arguments (line 508)
    kwargs_17806 = {}
    # Getting the type of 'ast' (line 508)
    ast_17804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 508, 15), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 508)
    Expr_17805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 508, 15), ast_17804, 'Expr')
    # Calling Expr(args, kwargs) (line 508)
    Expr_call_result_17807 = invoke(stypy.reporting.localization.Localization(__file__, 508, 15), Expr_17805, *[], **kwargs_17806)
    
    # Assigning a type to the variable 'set_expr' (line 508)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 508, 4), 'set_expr', Expr_call_result_17807)
    
    # Assigning a Name to a Attribute (line 509):
    
    # Assigning a Name to a Attribute (line 509):
    # Getting the type of 'set_call' (line 509)
    set_call_17808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 21), 'set_call')
    # Getting the type of 'set_expr' (line 509)
    set_expr_17809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 509, 4), 'set_expr')
    # Setting the type of the member 'value' of a type (line 509)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 509, 4), set_expr_17809, 'value', set_call_17808)
    # Getting the type of 'set_expr' (line 511)
    set_expr_17810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 511, 11), 'set_expr')
    # Assigning a type to the variable 'stypy_return_type' (line 511)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 511, 4), 'stypy_return_type', set_expr_17810)
    
    # ################# End of 'create_set_type_store(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_set_type_store' in the type store
    # Getting the type of 'stypy_return_type' (line 498)
    stypy_return_type_17811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 498, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17811)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_set_type_store'
    return stypy_return_type_17811

# Assigning a type to the variable 'create_set_type_store' (line 498)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 498, 0), 'create_set_type_store', create_set_type_store)

@norecursion
def create_join_type_store(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_join_type_store'
    module_type_store = module_type_store.open_function_context('create_join_type_store', 514, 0, False)
    
    # Passed parameters checking function
    create_join_type_store.stypy_localization = localization
    create_join_type_store.stypy_type_of_self = None
    create_join_type_store.stypy_type_store = module_type_store
    create_join_type_store.stypy_function_name = 'create_join_type_store'
    create_join_type_store.stypy_param_names_list = ['join_func_name', 'type_stores_to_join']
    create_join_type_store.stypy_varargs_param_name = None
    create_join_type_store.stypy_kwargs_param_name = None
    create_join_type_store.stypy_call_defaults = defaults
    create_join_type_store.stypy_call_varargs = varargs
    create_join_type_store.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_join_type_store', ['join_func_name', 'type_stores_to_join'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_join_type_store', localization, ['join_func_name', 'type_stores_to_join'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_join_type_store(...)' code ##################

    
    # Assigning a Call to a Name (line 515):
    
    # Assigning a Call to a Name (line 515):
    
    # Call to create_Name(...): (line 515)
    # Processing the call arguments (line 515)
    # Getting the type of 'join_func_name' (line 515)
    join_func_name_17814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 47), 'join_func_name', False)
    # Processing the call keyword arguments (line 515)
    kwargs_17815 = {}
    # Getting the type of 'core_language_copy' (line 515)
    core_language_copy_17812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 515, 16), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 515)
    create_Name_17813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 515, 16), core_language_copy_17812, 'create_Name')
    # Calling create_Name(args, kwargs) (line 515)
    create_Name_call_result_17816 = invoke(stypy.reporting.localization.Localization(__file__, 515, 16), create_Name_17813, *[join_func_name_17814], **kwargs_17815)
    
    # Assigning a type to the variable 'join_func' (line 515)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 515, 4), 'join_func', create_Name_call_result_17816)
    
    # Assigning a Call to a Name (line 516):
    
    # Assigning a Call to a Name (line 516):
    
    # Call to create_call(...): (line 516)
    # Processing the call arguments (line 516)
    # Getting the type of 'join_func' (line 516)
    join_func_17819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 43), 'join_func', False)
    # Getting the type of 'type_stores_to_join' (line 516)
    type_stores_to_join_17820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 54), 'type_stores_to_join', False)
    # Processing the call keyword arguments (line 516)
    kwargs_17821 = {}
    # Getting the type of 'functions_copy' (line 516)
    functions_copy_17817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 516, 16), 'functions_copy', False)
    # Obtaining the member 'create_call' of a type (line 516)
    create_call_17818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 516, 16), functions_copy_17817, 'create_call')
    # Calling create_call(args, kwargs) (line 516)
    create_call_call_result_17822 = invoke(stypy.reporting.localization.Localization(__file__, 516, 16), create_call_17818, *[join_func_17819, type_stores_to_join_17820], **kwargs_17821)
    
    # Assigning a type to the variable 'join_call' (line 516)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 516, 4), 'join_call', create_call_call_result_17822)
    
    # Assigning a Call to a Name (line 518):
    
    # Assigning a Call to a Name (line 518):
    
    # Call to __new_temp_type_store_Name(...): (line 518)
    # Processing the call keyword arguments (line 518)
    # Getting the type of 'False' (line 518)
    False_17824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 64), 'False', False)
    keyword_17825 = False_17824
    kwargs_17826 = {'right_hand_side': keyword_17825}
    # Getting the type of '__new_temp_type_store_Name' (line 518)
    new_temp_type_store_Name_17823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 518, 21), '__new_temp_type_store_Name', False)
    # Calling __new_temp_type_store_Name(args, kwargs) (line 518)
    new_temp_type_store_Name_call_result_17827 = invoke(stypy.reporting.localization.Localization(__file__, 518, 21), new_temp_type_store_Name_17823, *[], **kwargs_17826)
    
    # Assigning a type to the variable 'left_hand_side' (line 518)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 518, 4), 'left_hand_side', new_temp_type_store_Name_call_result_17827)
    
    # Assigning a Call to a Name (line 519):
    
    # Assigning a Call to a Name (line 519):
    
    # Call to Assign(...): (line 519)
    # Processing the call arguments (line 519)
    
    # Obtaining an instance of the builtin type 'list' (line 519)
    list_17830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 519, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 519)
    # Adding element type (line 519)
    # Getting the type of 'left_hand_side' (line 519)
    left_hand_side_17831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 33), 'left_hand_side', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 519, 32), list_17830, left_hand_side_17831)
    
    # Getting the type of 'join_call' (line 519)
    join_call_17832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 50), 'join_call', False)
    # Processing the call keyword arguments (line 519)
    kwargs_17833 = {}
    # Getting the type of 'ast' (line 519)
    ast_17828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 519, 21), 'ast', False)
    # Obtaining the member 'Assign' of a type (line 519)
    Assign_17829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 519, 21), ast_17828, 'Assign')
    # Calling Assign(args, kwargs) (line 519)
    Assign_call_result_17834 = invoke(stypy.reporting.localization.Localization(__file__, 519, 21), Assign_17829, *[list_17830, join_call_17832], **kwargs_17833)
    
    # Assigning a type to the variable 'join_statement' (line 519)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 519, 4), 'join_statement', Assign_call_result_17834)
    
    # Obtaining an instance of the builtin type 'tuple' (line 521)
    tuple_17835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 521, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 521)
    # Adding element type (line 521)
    # Getting the type of 'join_statement' (line 521)
    join_statement_17836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 11), 'join_statement')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 11), tuple_17835, join_statement_17836)
    # Adding element type (line 521)
    # Getting the type of 'left_hand_side' (line 521)
    left_hand_side_17837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 521, 27), 'left_hand_side')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 521, 11), tuple_17835, left_hand_side_17837)
    
    # Assigning a type to the variable 'stypy_return_type' (line 521)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 521, 4), 'stypy_return_type', tuple_17835)
    
    # ################# End of 'create_join_type_store(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_join_type_store' in the type store
    # Getting the type of 'stypy_return_type' (line 514)
    stypy_return_type_17838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 514, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17838)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_join_type_store'
    return stypy_return_type_17838

# Assigning a type to the variable 'create_join_type_store' (line 514)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 514, 0), 'create_join_type_store', create_join_type_store)

@norecursion
def create_binary_operator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_binary_operator'
    module_type_store = module_type_store.open_function_context('create_binary_operator', 527, 0, False)
    
    # Passed parameters checking function
    create_binary_operator.stypy_localization = localization
    create_binary_operator.stypy_type_of_self = None
    create_binary_operator.stypy_type_store = module_type_store
    create_binary_operator.stypy_function_name = 'create_binary_operator'
    create_binary_operator.stypy_param_names_list = ['operator_name', 'left_op', 'rigth_op', 'lineno', 'col_offset']
    create_binary_operator.stypy_varargs_param_name = None
    create_binary_operator.stypy_kwargs_param_name = None
    create_binary_operator.stypy_call_defaults = defaults
    create_binary_operator.stypy_call_varargs = varargs
    create_binary_operator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_binary_operator', ['operator_name', 'left_op', 'rigth_op', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_binary_operator', localization, ['operator_name', 'left_op', 'rigth_op', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_binary_operator(...)' code ##################

    str_17839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 537, (-1)), 'str', '\n    Creates AST Nodes to model a binary operator\n\n    :param operator_name: Name of the operator\n    :param left_op: Left operand (AST Node)\n    :param rigth_op: Right operand (AST Node)\n    :param lineno: Line\n    :param col_offset: Column\n    :return: List of instructions\n    ')
    
    # Assigning a Call to a Name (line 538):
    
    # Assigning a Call to a Name (line 538):
    
    # Call to operator_name_to_symbol(...): (line 538)
    # Processing the call arguments (line 538)
    # Getting the type of 'operator_name' (line 538)
    operator_name_17841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 46), 'operator_name', False)
    # Processing the call keyword arguments (line 538)
    kwargs_17842 = {}
    # Getting the type of 'operator_name_to_symbol' (line 538)
    operator_name_to_symbol_17840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 538, 22), 'operator_name_to_symbol', False)
    # Calling operator_name_to_symbol(args, kwargs) (line 538)
    operator_name_to_symbol_call_result_17843 = invoke(stypy.reporting.localization.Localization(__file__, 538, 22), operator_name_to_symbol_17840, *[operator_name_17841], **kwargs_17842)
    
    # Assigning a type to the variable 'operator_symbol' (line 538)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 538, 4), 'operator_symbol', operator_name_to_symbol_call_result_17843)
    
    # Assigning a Call to a Name (line 539):
    
    # Assigning a Call to a Name (line 539):
    
    # Call to create_str(...): (line 539)
    # Processing the call arguments (line 539)
    # Getting the type of 'operator_symbol' (line 539)
    operator_symbol_17846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 44), 'operator_symbol', False)
    # Processing the call keyword arguments (line 539)
    kwargs_17847 = {}
    # Getting the type of 'core_language_copy' (line 539)
    core_language_copy_17844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 539, 14), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 539)
    create_str_17845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 539, 14), core_language_copy_17844, 'create_str')
    # Calling create_str(args, kwargs) (line 539)
    create_str_call_result_17848 = invoke(stypy.reporting.localization.Localization(__file__, 539, 14), create_str_17845, *[operator_symbol_17846], **kwargs_17847)
    
    # Assigning a type to the variable 'op_name' (line 539)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 539, 4), 'op_name', create_str_call_result_17848)
    
    # Assigning a Call to a Name (line 540):
    
    # Assigning a Call to a Name (line 540):
    
    # Call to create_src_comment(...): (line 540)
    # Processing the call arguments (line 540)
    
    # Call to format(...): (line 540)
    # Processing the call arguments (line 540)
    # Getting the type of 'operator_symbol' (line 540)
    operator_symbol_17852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 87), 'operator_symbol', False)
    # Processing the call keyword arguments (line 540)
    kwargs_17853 = {}
    str_17850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 540, 43), 'str', "Applying the '{0}' binary operator")
    # Obtaining the member 'format' of a type (line 540)
    format_17851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 540, 43), str_17850, 'format')
    # Calling format(args, kwargs) (line 540)
    format_call_result_17854 = invoke(stypy.reporting.localization.Localization(__file__, 540, 43), format_17851, *[operator_symbol_17852], **kwargs_17853)
    
    # Getting the type of 'lineno' (line 540)
    lineno_17855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 105), 'lineno', False)
    # Processing the call keyword arguments (line 540)
    kwargs_17856 = {}
    # Getting the type of 'create_src_comment' (line 540)
    create_src_comment_17849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 540, 24), 'create_src_comment', False)
    # Calling create_src_comment(args, kwargs) (line 540)
    create_src_comment_call_result_17857 = invoke(stypy.reporting.localization.Localization(__file__, 540, 24), create_src_comment_17849, *[format_call_result_17854, lineno_17855], **kwargs_17856)
    
    # Assigning a type to the variable 'operation_comment' (line 540)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 540, 4), 'operation_comment', create_src_comment_call_result_17857)
    
    # Assigning a Call to a Tuple (line 541):
    
    # Assigning a Call to a Name:
    
    # Call to create_temp_Assign(...): (line 541)
    # Processing the call arguments (line 541)
    
    # Call to create_binary_operator(...): (line 542)
    # Processing the call arguments (line 542)
    # Getting the type of 'op_name' (line 542)
    op_name_17861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 46), 'op_name', False)
    # Getting the type of 'left_op' (line 542)
    left_op_17862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 55), 'left_op', False)
    # Getting the type of 'rigth_op' (line 542)
    rigth_op_17863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 64), 'rigth_op', False)
    # Getting the type of 'lineno' (line 542)
    lineno_17864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 74), 'lineno', False)
    # Getting the type of 'col_offset' (line 542)
    col_offset_17865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 82), 'col_offset', False)
    # Processing the call keyword arguments (line 542)
    kwargs_17866 = {}
    # Getting the type of 'operators_copy' (line 542)
    operators_copy_17859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 8), 'operators_copy', False)
    # Obtaining the member 'create_binary_operator' of a type (line 542)
    create_binary_operator_17860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 542, 8), operators_copy_17859, 'create_binary_operator')
    # Calling create_binary_operator(args, kwargs) (line 542)
    create_binary_operator_call_result_17867 = invoke(stypy.reporting.localization.Localization(__file__, 542, 8), create_binary_operator_17860, *[op_name_17861, left_op_17862, rigth_op_17863, lineno_17864, col_offset_17865], **kwargs_17866)
    
    # Getting the type of 'lineno' (line 542)
    lineno_17868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 95), 'lineno', False)
    # Getting the type of 'col_offset' (line 542)
    col_offset_17869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 542, 103), 'col_offset', False)
    # Processing the call keyword arguments (line 541)
    kwargs_17870 = {}
    # Getting the type of 'create_temp_Assign' (line 541)
    create_temp_Assign_17858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 32), 'create_temp_Assign', False)
    # Calling create_temp_Assign(args, kwargs) (line 541)
    create_temp_Assign_call_result_17871 = invoke(stypy.reporting.localization.Localization(__file__, 541, 32), create_temp_Assign_17858, *[create_binary_operator_call_result_17867, lineno_17868, col_offset_17869], **kwargs_17870)
    
    # Assigning a type to the variable 'call_assignment_16812' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'call_assignment_16812', create_temp_Assign_call_result_17871)
    
    # Assigning a Call to a Name (line 541):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_16812' (line 541)
    call_assignment_16812_17872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'call_assignment_16812', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_17873 = stypy_get_value_from_tuple(call_assignment_16812_17872, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_16813' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'call_assignment_16813', stypy_get_value_from_tuple_call_result_17873)
    
    # Assigning a Name to a Name (line 541):
    # Getting the type of 'call_assignment_16813' (line 541)
    call_assignment_16813_17874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'call_assignment_16813')
    # Assigning a type to the variable 'operator_call' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'operator_call', call_assignment_16813_17874)
    
    # Assigning a Call to a Name (line 541):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_16812' (line 541)
    call_assignment_16812_17875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'call_assignment_16812', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_17876 = stypy_get_value_from_tuple(call_assignment_16812_17875, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_16814' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'call_assignment_16814', stypy_get_value_from_tuple_call_result_17876)
    
    # Assigning a Name to a Name (line 541):
    # Getting the type of 'call_assignment_16814' (line 541)
    call_assignment_16814_17877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 541, 4), 'call_assignment_16814')
    # Assigning a type to the variable 'result_var' (line 541)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 541, 19), 'result_var', call_assignment_16814_17877)
    
    # Obtaining an instance of the builtin type 'tuple' (line 544)
    tuple_17878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 544, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 544)
    # Adding element type (line 544)
    
    # Call to flatten_lists(...): (line 544)
    # Processing the call arguments (line 544)
    # Getting the type of 'operation_comment' (line 544)
    operation_comment_17880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 25), 'operation_comment', False)
    # Getting the type of 'operator_call' (line 544)
    operator_call_17881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 44), 'operator_call', False)
    # Processing the call keyword arguments (line 544)
    kwargs_17882 = {}
    # Getting the type of 'flatten_lists' (line 544)
    flatten_lists_17879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 11), 'flatten_lists', False)
    # Calling flatten_lists(args, kwargs) (line 544)
    flatten_lists_call_result_17883 = invoke(stypy.reporting.localization.Localization(__file__, 544, 11), flatten_lists_17879, *[operation_comment_17880, operator_call_17881], **kwargs_17882)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 11), tuple_17878, flatten_lists_call_result_17883)
    # Adding element type (line 544)
    # Getting the type of 'result_var' (line 544)
    result_var_17884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 544, 60), 'result_var')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 544, 11), tuple_17878, result_var_17884)
    
    # Assigning a type to the variable 'stypy_return_type' (line 544)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 544, 4), 'stypy_return_type', tuple_17878)
    
    # ################# End of 'create_binary_operator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_binary_operator' in the type store
    # Getting the type of 'stypy_return_type' (line 527)
    stypy_return_type_17885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 527, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17885)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_binary_operator'
    return stypy_return_type_17885

# Assigning a type to the variable 'create_binary_operator' (line 527)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 527, 0), 'create_binary_operator', create_binary_operator)

@norecursion
def create_unary_operator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_unary_operator'
    module_type_store = module_type_store.open_function_context('create_unary_operator', 547, 0, False)
    
    # Passed parameters checking function
    create_unary_operator.stypy_localization = localization
    create_unary_operator.stypy_type_of_self = None
    create_unary_operator.stypy_type_store = module_type_store
    create_unary_operator.stypy_function_name = 'create_unary_operator'
    create_unary_operator.stypy_param_names_list = ['operator_name', 'left_op', 'lineno', 'col_offset']
    create_unary_operator.stypy_varargs_param_name = None
    create_unary_operator.stypy_kwargs_param_name = None
    create_unary_operator.stypy_call_defaults = defaults
    create_unary_operator.stypy_call_varargs = varargs
    create_unary_operator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_unary_operator', ['operator_name', 'left_op', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_unary_operator', localization, ['operator_name', 'left_op', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_unary_operator(...)' code ##################

    str_17886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 556, (-1)), 'str', '\n    Creates AST Nodes to model an unary operator\n\n    :param operator_name: Name of the operator\n    :param left_op: operand (AST Node)\n    :param lineno: Line\n    :param col_offset: Column\n    :return: List of instructions\n    ')
    
    # Assigning a Call to a Name (line 557):
    
    # Assigning a Call to a Name (line 557):
    
    # Call to operator_name_to_symbol(...): (line 557)
    # Processing the call arguments (line 557)
    # Getting the type of 'operator_name' (line 557)
    operator_name_17888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 46), 'operator_name', False)
    # Processing the call keyword arguments (line 557)
    kwargs_17889 = {}
    # Getting the type of 'operator_name_to_symbol' (line 557)
    operator_name_to_symbol_17887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 557, 22), 'operator_name_to_symbol', False)
    # Calling operator_name_to_symbol(args, kwargs) (line 557)
    operator_name_to_symbol_call_result_17890 = invoke(stypy.reporting.localization.Localization(__file__, 557, 22), operator_name_to_symbol_17887, *[operator_name_17888], **kwargs_17889)
    
    # Assigning a type to the variable 'operator_symbol' (line 557)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 557, 4), 'operator_symbol', operator_name_to_symbol_call_result_17890)
    
    # Assigning a Call to a Name (line 558):
    
    # Assigning a Call to a Name (line 558):
    
    # Call to create_str(...): (line 558)
    # Processing the call arguments (line 558)
    # Getting the type of 'operator_symbol' (line 558)
    operator_symbol_17893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 44), 'operator_symbol', False)
    # Processing the call keyword arguments (line 558)
    kwargs_17894 = {}
    # Getting the type of 'core_language_copy' (line 558)
    core_language_copy_17891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 558, 14), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 558)
    create_str_17892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 558, 14), core_language_copy_17891, 'create_str')
    # Calling create_str(args, kwargs) (line 558)
    create_str_call_result_17895 = invoke(stypy.reporting.localization.Localization(__file__, 558, 14), create_str_17892, *[operator_symbol_17893], **kwargs_17894)
    
    # Assigning a type to the variable 'op_name' (line 558)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 558, 4), 'op_name', create_str_call_result_17895)
    
    # Assigning a Call to a Name (line 559):
    
    # Assigning a Call to a Name (line 559):
    
    # Call to create_src_comment(...): (line 559)
    # Processing the call arguments (line 559)
    
    # Call to format(...): (line 559)
    # Processing the call arguments (line 559)
    # Getting the type of 'operator_symbol' (line 559)
    operator_symbol_17899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 86), 'operator_symbol', False)
    # Processing the call keyword arguments (line 559)
    kwargs_17900 = {}
    str_17897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 559, 43), 'str', "Applying the '{0}' unary operator")
    # Obtaining the member 'format' of a type (line 559)
    format_17898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 559, 43), str_17897, 'format')
    # Calling format(args, kwargs) (line 559)
    format_call_result_17901 = invoke(stypy.reporting.localization.Localization(__file__, 559, 43), format_17898, *[operator_symbol_17899], **kwargs_17900)
    
    # Getting the type of 'lineno' (line 559)
    lineno_17902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 104), 'lineno', False)
    # Processing the call keyword arguments (line 559)
    kwargs_17903 = {}
    # Getting the type of 'create_src_comment' (line 559)
    create_src_comment_17896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 559, 24), 'create_src_comment', False)
    # Calling create_src_comment(args, kwargs) (line 559)
    create_src_comment_call_result_17904 = invoke(stypy.reporting.localization.Localization(__file__, 559, 24), create_src_comment_17896, *[format_call_result_17901, lineno_17902], **kwargs_17903)
    
    # Assigning a type to the variable 'operation_comment' (line 559)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 559, 4), 'operation_comment', create_src_comment_call_result_17904)
    
    # Assigning a Call to a Tuple (line 560):
    
    # Assigning a Call to a Name:
    
    # Call to create_temp_Assign(...): (line 560)
    # Processing the call arguments (line 560)
    
    # Call to create_unary_operator(...): (line 561)
    # Processing the call arguments (line 561)
    # Getting the type of 'op_name' (line 561)
    op_name_17908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 45), 'op_name', False)
    # Getting the type of 'left_op' (line 561)
    left_op_17909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 54), 'left_op', False)
    # Getting the type of 'lineno' (line 561)
    lineno_17910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 63), 'lineno', False)
    # Getting the type of 'col_offset' (line 561)
    col_offset_17911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 71), 'col_offset', False)
    # Processing the call keyword arguments (line 561)
    kwargs_17912 = {}
    # Getting the type of 'operators_copy' (line 561)
    operators_copy_17906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 8), 'operators_copy', False)
    # Obtaining the member 'create_unary_operator' of a type (line 561)
    create_unary_operator_17907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 561, 8), operators_copy_17906, 'create_unary_operator')
    # Calling create_unary_operator(args, kwargs) (line 561)
    create_unary_operator_call_result_17913 = invoke(stypy.reporting.localization.Localization(__file__, 561, 8), create_unary_operator_17907, *[op_name_17908, left_op_17909, lineno_17910, col_offset_17911], **kwargs_17912)
    
    # Getting the type of 'lineno' (line 561)
    lineno_17914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 84), 'lineno', False)
    # Getting the type of 'col_offset' (line 561)
    col_offset_17915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 561, 92), 'col_offset', False)
    # Processing the call keyword arguments (line 560)
    kwargs_17916 = {}
    # Getting the type of 'create_temp_Assign' (line 560)
    create_temp_Assign_17905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 32), 'create_temp_Assign', False)
    # Calling create_temp_Assign(args, kwargs) (line 560)
    create_temp_Assign_call_result_17917 = invoke(stypy.reporting.localization.Localization(__file__, 560, 32), create_temp_Assign_17905, *[create_unary_operator_call_result_17913, lineno_17914, col_offset_17915], **kwargs_17916)
    
    # Assigning a type to the variable 'call_assignment_16815' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'call_assignment_16815', create_temp_Assign_call_result_17917)
    
    # Assigning a Call to a Name (line 560):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_16815' (line 560)
    call_assignment_16815_17918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'call_assignment_16815', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_17919 = stypy_get_value_from_tuple(call_assignment_16815_17918, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_16816' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'call_assignment_16816', stypy_get_value_from_tuple_call_result_17919)
    
    # Assigning a Name to a Name (line 560):
    # Getting the type of 'call_assignment_16816' (line 560)
    call_assignment_16816_17920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'call_assignment_16816')
    # Assigning a type to the variable 'operator_call' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'operator_call', call_assignment_16816_17920)
    
    # Assigning a Call to a Name (line 560):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_16815' (line 560)
    call_assignment_16815_17921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'call_assignment_16815', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_17922 = stypy_get_value_from_tuple(call_assignment_16815_17921, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_16817' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'call_assignment_16817', stypy_get_value_from_tuple_call_result_17922)
    
    # Assigning a Name to a Name (line 560):
    # Getting the type of 'call_assignment_16817' (line 560)
    call_assignment_16817_17923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 560, 4), 'call_assignment_16817')
    # Assigning a type to the variable 'result_var' (line 560)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 560, 19), 'result_var', call_assignment_16817_17923)
    
    # Obtaining an instance of the builtin type 'tuple' (line 563)
    tuple_17924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 563, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 563)
    # Adding element type (line 563)
    
    # Call to flatten_lists(...): (line 563)
    # Processing the call arguments (line 563)
    # Getting the type of 'operation_comment' (line 563)
    operation_comment_17926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 25), 'operation_comment', False)
    # Getting the type of 'operator_call' (line 563)
    operator_call_17927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 44), 'operator_call', False)
    # Processing the call keyword arguments (line 563)
    kwargs_17928 = {}
    # Getting the type of 'flatten_lists' (line 563)
    flatten_lists_17925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 11), 'flatten_lists', False)
    # Calling flatten_lists(args, kwargs) (line 563)
    flatten_lists_call_result_17929 = invoke(stypy.reporting.localization.Localization(__file__, 563, 11), flatten_lists_17925, *[operation_comment_17926, operator_call_17927], **kwargs_17928)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 11), tuple_17924, flatten_lists_call_result_17929)
    # Adding element type (line 563)
    # Getting the type of 'result_var' (line 563)
    result_var_17930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 563, 60), 'result_var')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 563, 11), tuple_17924, result_var_17930)
    
    # Assigning a type to the variable 'stypy_return_type' (line 563)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 563, 4), 'stypy_return_type', tuple_17924)
    
    # ################# End of 'create_unary_operator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_unary_operator' in the type store
    # Getting the type of 'stypy_return_type' (line 547)
    stypy_return_type_17931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 547, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17931)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_unary_operator'
    return stypy_return_type_17931

# Assigning a type to the variable 'create_unary_operator' (line 547)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 547, 0), 'create_unary_operator', create_unary_operator)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
