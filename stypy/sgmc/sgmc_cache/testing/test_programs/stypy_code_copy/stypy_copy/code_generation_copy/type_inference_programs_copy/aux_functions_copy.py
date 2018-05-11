
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import ast
2: 
3: from ...errors_copy.type_error_copy import TypeError
4: from ...errors_copy.type_warning_copy import TypeWarning
5: from ...visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, data_structures_copy
6: from .... import stypy_copy
7: from ...stypy_parameters_copy import ENABLE_CODING_ADVICES
8: from ...reporting_copy.print_utils_copy import format_function_name
9: from ...python_lib_copy.type_rules_copy.type_groups_copy.type_group_generator_copy import Str, IterableObject
10: 
11: '''
12: This file holds functions that are invoked by the generated source code of type inference programs. This code
13: usually need functions to perform common tasks that can be directly generated in Python source code, but this
14: will turn the source code of the type inference program into something much less manageable. Therefore, we
15: identified common tasks within this source code and encapsulated it into functions to be called, in order to
16: make the type inference programs smaller and more clear.
17: '''
18: 
19: 
20: # ########################### FUNCTIONS TO PROCESS FUNCTION CALLS ############################
21: 
22: 
23: def __assign_arguments(localization, type_store, declared_arguments_list, type_of_args):
24:     '''
25:     Auxiliar function to assign the declared argument names and types to a type store. The purpose of this function
26:     is to insert into the current type store and function context the name and value of the parameters so the code
27:     can use them.
28:     :param localization: Caller information
29:     :param type_store: Current type store (it is assumed that the generated source code has created a new function
30:     context for the called function
31:     :param declared_arguments_list: Argument names
32:     :param type_of_args: Type of arguments (order of argument names is used to establish a correspondence. If len of
33:     declared_argument_list and type_of_args is not the same, the lower length is used)
34:     :return:
35:     '''
36: 
37:     # Calculate which list is sorter previous to iteration
38:     if len(declared_arguments_list) < len(type_of_args):
39:         min_len = len(declared_arguments_list)
40:     else:
41:         min_len = len(type_of_args)
42: 
43:     # Assign arguments one by one to the type store
44:     for i in range(min_len):
45:         type_store.set_type_of(localization, declared_arguments_list[i], type_of_args[i])
46: 
47: 
48: def __process_call_type_or_args(function_name, localization, declared_argument_name_list, call_varargs,
49:                                 call_kwargs, defaults):
50:     '''
51:     As type inference programs convers any function call to the func(*args, **kwargs) form, this function checks both
52:      parameters to assign its elements to declared named parameters, see if the call have been done with enough ones,
53:      if the call has an excess or parameters and, in the end, calculate the real variable argument list and keywords
54:      dictionary
55: 
56:     :param function_name: Called function name
57:     :param localization: Caller information
58:     :param declared_argument_name_list: Declared argument list names (example: ['a', 'b'] in def f(a, b))
59:     :param call_varargs: First parameter of the type inference function call
60:     :param call_kwargs: Second parameter of the type inference function call
61:     :param defaults: Declared defaults (values for those parameters that have a default value (are optional) in the
62:     function declaration (example: [3,4] in def f(a, b=3, c=4))
63:     :return:
64:     '''
65: 
66:     # First: Calculate the type of the passed named parameters. They can be calculated by extracting them from
67:     # the parameter list or the keyword list. Values in both lists are not allowed for a parameter, and an error is
68:     # reported in that case
69:     call_type_of_args = []  # Per-call passed args
70:     cont = 0
71:     error_msg = ""
72:     found_error = False
73:     arg_count = 1
74: 
75:     #call_varargs = list(call_varargs)
76: 
77:     # Named parameters are extracted in declaration order
78:     for name in declared_argument_name_list:
79:         # Exist an argument on that position in the passed args?
80:         if len(call_varargs) > cont:
81:             call_type_of_args.append(call_varargs[cont])
82:             # If there is a keyword with the same name of the argument, a value is found for it in the arg list,
83:             # and it is not a default, report an error
84:             if name in call_kwargs and name not in defaults:
85:                 found_error = True
86:                 msg = "{0} got multiple values for keyword argument '{1}'; ".format(format_function_name(function_name),
87:                                                                                     name)
88:                 error_msg += msg
89:                 call_type_of_args.append(TypeError(localization, msg, prints_msg=False))
90: 
91:             # One the argument is processed, we delete it to not to consider it as an extra vararg
92:             del call_varargs[cont]
93:         else:
94:             # If no argument is passed in the vararg list, search for a compatible keyword argument
95:             if name in call_kwargs:
96:                 call_type_of_args.append(call_kwargs[name])
97:                 # Remove the argument from the kwargs dict as we don't want it to appear as an extra kwarg
98:                 del call_kwargs[name]
99:             else:
100:                 # No value for this named argument, report the error
101:                 found_error = True
102:                 msg = "Insufficient number of arguments for {0}: Cannot find a value for argument" \
103:                       " number {1} ('{2}'); ".format(format_function_name(function_name), arg_count, name)
104:                 error_msg += msg
105: 
106:                 call_type_of_args.append(TypeError(localization, msg, prints_msg=False))
107: 
108: 
109:         arg_count += 1
110: 
111:     # Errors found: Return the type of named args (with errors among them, the error messages, and an error found flag
112:     if found_error:
113:         return call_type_of_args, error_msg, True
114: 
115:     # No errors found: Return the type of named args, no error message and a no error found flag
116:     return call_type_of_args, "", False
117: 
118: 
119: def process_argument_values(localization, type_of_self, type_store, function_name,
120:                             declared_argument_name_list,
121:                             declared_varargs_var,
122:                             declared_kwargs_var,
123:                             declared_defaults,
124:                             call_varargs=list(),  # List of arguments to unpack (if present)
125:                             call_kwargs={},
126:                             allow_argument_keywords=True):  # Dictionary of keyword arguments to unpack (if present)
127:     '''
128:     This long function is the responsible of checking all the parameters passed to a function call and make sure that
129:     the call is valid and possible. Argument passing in Python is a complex task, because there are several argument
130:     types and combinations, and the mechanism is rather flexible, so care was taken to try to identify those
131:     combinations, identify misuses of the call mechanism and assign the correct values to arguments. The function is
132:     long and complex, so documentation was placed to try to clarify the behaviour of each part.
133: 
134:     :param localization: Caller information
135:     :param type_of_self: Type of the owner of the function/method. Currently unused, may be used for reporting errors.
136:     :param type_store: Current type store (a function context for the current function have to be already set up)
137:     :param function_name: Name of the function/method/lambda function that is being invoked
138:     :param declared_argument_name_list: List of named arguments declared in the source code of the function
139:     (example ['a', 'n'] in def f(a, n))
140:     :param declared_varargs_var: Name of the parameter that holds the variable argument list (if any)
141:     (example: "args" in def f(*args))
142:     :param declared_kwargs_var: Name of the parameter that holds the keyword argument dictionary (if any).
143:     (example: "kwargs" in def f(**kwargs))
144:     :param declared_defaults: Declared default values for arguments (if present).
145:     (example: [3, 4] in def f(a=3, n=4)
146:     :param call_varargs: Calls to functions/methods in type inference programs only have two parameters: args
147:     (variable argument list) and kwargs (keyword argument dictionary). This is done in order to simplify call
148:      handling, as any function call can be expressed this way.
149:     Example: f(*args, **kwargs)
150:     Values for the declared arguments are extracted from the varargs list in order (so the rest of the arguments are
151:     the real variable list of arguments of the original function).
152:     :param call_kwargs: This dictionary holds pairs of (name, type). If name is in the declared argument list, the
153:      corresponding type is assigned to this named parameter. If it is not, it is left inside the kwargs dictionary of
154:      the function. In the end, the declared_kwargs_var values are those that will not be assigned to named parameters.
155:     :param allow_argument_keywords: Python API functions do not allow the usage of named keywords when calling them.
156:     This disallow the usage of this kind of calls and report errors if used with this kind of functions.
157:     :return:
158:     '''
159: 
160:     # Error initialization
161:     found_error = False
162:     error_msg = ""
163: 
164:     # Is this function allowing argument keywords (f(a=3)? Then we must check if the call_kwargs parameter contains
165:     # a value. If it contains values, we must check that all of them belong to the declared defaults list. If one of
166:     # them it is not in this list, it is an error because the function do not admit initialized keyword parameters
167:     if not allow_argument_keywords and len(call_kwargs) > 0:
168:         for arg in call_kwargs:
169:             if arg not in declared_defaults:
170:                 found_error = True
171:                 error_msg += "{0} takes no keyword arguments; ".format(format_function_name(function_name))
172:                 break
173: 
174:     # Store in the current context the declared function variable name information
175:     context = type_store.get_context()
176:     context.declared_argument_name_list = declared_argument_name_list
177:     context.declared_varargs_var = declared_varargs_var
178:     context.declared_kwargs_var = declared_kwargs_var
179:     context.declared_defaults = declared_defaults
180: 
181:     # Defaults can be provided in the form of a list or tuple (values only, no name - value pairing) or in the form of
182:     # a dict (name-value pairing already done). In the first case we build a dictionary first to homogenize the
183:     # processing of these data
184:     if type(declared_defaults) is list or type(declared_defaults) is tuple:
185:         defaults_dict = {}
186:         # Defaults values are assigned beginning from the last parameter
187:         declared_argument_name_list.reverse()
188:         declared_defaults.reverse()
189:         cont = 0
190:         for value in declared_defaults:
191:             defaults_dict[declared_argument_name_list[cont]] = value
192:             cont += 1
193: 
194:         declared_argument_name_list.reverse()
195:         declared_defaults.reverse()
196:     else:
197:         defaults_dict = declared_defaults
198: 
199:     # Make varargs modifiable
200:     call_varargs = list(call_varargs)
201: 
202:     # Assign defaults to those values not present in the passed parameters
203:     for elem in defaults_dict:
204:         if elem not in call_kwargs:
205:             call_kwargs[elem] = defaults_dict[elem]
206: 
207:     # Check named parameters, variable argument list and argument keywords, assigning a value to the named arguments
208:     # (returned in call_type_of_args). call_vargargs and call_kwargs can lose elements if they are assigned as named
209:     # arguments types.
210:     call_type_of_args, error, found_error_on_call_args = __process_call_type_or_args(function_name,
211:                                                                                      localization,
212:                                                                                      declared_argument_name_list,
213:                                                                                      call_varargs,
214:                                                                                      call_kwargs,
215:                                                                                      defaults_dict)
216: 
217:     if found_error_on_call_args:  # Arg. arity error, return it
218:         error_msg += error
219: 
220:     found_error |= found_error_on_call_args
221: 
222:     # Assign arguments values
223:     __assign_arguments(localization, type_store, declared_argument_name_list, call_type_of_args)
224: 
225:     # Delete left defaults to not to consider them extra parameters (if there is any left it means that the
226:     # function has extra parameters, some of them have default values and a different value for them have not been
227:     # passed)
228:     left_kwargs = call_kwargs.keys()
229:     for name in left_kwargs:
230:         if name in defaults_dict:
231:             del call_kwargs[name]
232: 
233:     # var (star) args are composed by excess of args in the call once named arguments are processed
234:     if declared_varargs_var is not None:
235:         # Var args is a tuple of all the rest of the passed types. This tuple is created here and is assigned to
236:         # the current type store with the declared_varargs_var name
237:         excess_arguments = stypy_copy.python_interface_copy.get_builtin_type(localization, "tuple")
238:         excess_arguments.add_types_from_list(localization, call_varargs, record_annotation=False)
239:         type_store.set_type_of(localization, declared_varargs_var, excess_arguments)
240:     else:
241:         # Arguments left in call_varargs and no variable list of arguments is declared? Somebody call us with too
242:         # many arguments.
243:         if len(call_varargs) > 0:
244:             found_error = True
245:             error_msg += "{0} got {1} more arguments than expected; ".format(format_function_name(function_name),
246:                                                                              str(len(call_varargs)))
247: 
248:     # keyword args are composed by the contents of the keywords list left from the argument processing. We create
249:     # a dictionary with these values and its associated var name.
250:     if declared_kwargs_var is not None:
251:         kwargs_variable = stypy_copy.python_interface_copy.get_builtin_type(localization, "dict")
252: 
253:         # Create the kwargs dictionary
254:         for name in call_kwargs:
255:             str_ = stypy_copy.python_interface_copy.get_builtin_type(localization, "str", value=name)
256:             kwargs_variable.add_key_and_value_type(localization, (str_, call_kwargs[name]), record_annotation=False)
257: 
258:         type_store.set_type_of(localization, declared_kwargs_var, kwargs_variable)
259:     else:
260:         # Arguments left in call_kwargs and no keyword arguments variable declared? Somebody call us with keyword
261:         # parameters without accepting them or with wrong names.
262:         if len(call_kwargs) > 0:
263:             found_error = True
264:             error_msg += "{0} got unexpected keyword arguments: {1}; ".format(format_function_name(function_name),
265:                                                                               str(call_kwargs))
266: 
267:     # Create an error with the accumulated error messages of all the argument processing steps
268:     if found_error:
269:         return TypeError(localization, error_msg)
270: 
271:     return call_type_of_args, call_varargs, call_kwargs
272: 
273: 
274: def create_call_to_type_inference_code(func, localization, keywords=list(), kwargs=None, starargs=None, line=0,
275:                                        column=0):
276:     '''
277:     Create the necessary Python code to call a function that performs the type inference of an existing function.
278:      Basically it calls the invoke method of the TypeInferenceProxy that represent the callable code, creating
279:      the *args and **kwargs call parameters we mentioned before.
280:     :param func: Function name to call
281:     :param localization: Caller information
282:     :param keywords: Unused. May be removed TODO
283:     :param kwargs: keyword dictionary
284:     :param starargs: variable argument list
285:     :param line: Source line when this call is produced
286:     :param column: Source column when this call is produced
287:     :return:
288:     '''
289:     call = ast.Call()
290: 
291:     # TODO: Remove?
292:     if type(func) is tuple:
293:         tuple_node = ast.Tuple()
294:         tuple_node.elts = list(func)
295:         func = tuple_node
296: 
297:     # Initialize the arguments of the call. localization always goes first
298:     ti_args = [localization]
299: 
300:     # Call to type_inference_proxy_of_the_func.invoke
301:     call.func = core_language_copy.create_attribute(func, 'invoke')
302:     call.lineno = line
303:     call.col_offset = column
304: 
305:     # Create and assign starargs
306:     if starargs is None:
307:         call.starargs = data_structures_copy.create_list([])
308:     else:
309:         call.starargs = data_structures_copy.create_list(starargs)
310: 
311:     # Create and assign kwargs
312:     if kwargs is None:
313:         call.kwargs = data_structures_copy.create_keyword_dict(None)
314:     else:
315:         call.kwargs = kwargs
316: 
317:     call.keywords = []
318: 
319:     # Assign named args (only localization)
320:     call.args = ti_args
321: 
322:     # Return call AST node
323:     return call
324: 
325: 
326: # ########################### VARIABLES FOR CONDITIONS TYPE CHECKING ############################
327: 
328: 
329: def is_suitable_condition(localization, condition_type):
330:     '''
331:     Checks if the type of a condition is suitable. Only checks if the type of a condition is an error, except if
332:     coding advices is enabled. In that case a warning is issued if the condition is not bool.
333:     :param localization: Caller information
334:     :param condition_type: Type of the condition
335:     :return:
336:     '''
337:     if is_error_type(condition_type):
338:         TypeError(localization, "The type of this condition is erroneous")
339:         return False
340: 
341:     if ENABLE_CODING_ADVICES:
342:         if not condition_type.get_python_type() is bool:
343:             TypeWarning.instance(localization,
344:                                  "The type of this condition is not boolean ({0}). Is that what you really intend?".
345:                                  format(condition_type))
346: 
347:     return True
348: 
349: 
350: def is_error_type(type_):
351:     '''
352:     Tells if the passed type represent some kind of error
353:     :param type_: Passed type
354:     :return: bool value
355:     '''
356:     return isinstance(type_, TypeError)
357: 
358: 
359: def is_suitable_for_loop_condition(localization, condition_type):
360:     '''
361:     A loop must iterate an iterable object or data structure or an string. This function checks this fact
362:     :param localization: Caller information
363:     :param condition_type: Type of the condition
364:     :return:
365:     '''
366:     if is_error_type(condition_type):
367:         TypeError(localization, "The type of this for loop condition is erroneous")
368:         return False
369: 
370:     if not (condition_type.can_store_elements() or (Str == condition_type) or (IterableObject == condition_type)):
371:         TypeError(localization, "The type of this for loop condition is erroneous")
372:         return False
373: 
374:     return True
375: 
376: 
377: def get_type_of_for_loop_variable(localization, condition_type):
378:     '''
379:     A loop must iterate an iterable object or data structure or an string. This function returns the contents of
380:     whatever the loop is iterating
381:     :param localization: Caller information
382:     :param condition_type: Type of the condition
383:     :return:
384:     '''
385: 
386:     # If the type of the condition can store elements, return the type of stored elements
387:     if condition_type.can_store_elements() and condition_type.is_type_instance():
388:         return condition_type.get_elements_type()
389: 
390:     # If the type of the condition is some kind of string, return the type of string
391:     if Str == condition_type and condition_type.is_type_instance():
392:         return condition_type.get_python_type()
393: 
394:     # If the type of the condition is something iterable, return the result of calling its __iter__ method
395:     if IterableObject == condition_type and condition_type.is_type_instance():
396:         iter_method = condition_type.get_type_of_member(localization, "__iter__")
397:         return iter_method.invoke(localization)
398: 
399:     return TypeError(localization, "Invalid iterable type for a loop target")
400: 
401: 
402: # ################################## TYPE IDIOMS FUNCTIONS ###################################
403: 
404: def __type_is_in_union(type_list, expected_type):
405:     #type_to_search = expected_type.get_python_entity()
406: 
407:     for typ in type_list:
408:         #if typ.get_python_entity() == type_to_search:
409:         if typ == expected_type:
410:             return True
411: 
412:     return False
413: 
414: def may_be_type(actual_type, expected_type):
415:     '''
416:     Returns:
417:      1) if the actual type is the expected one, including the semantics of union types (int\/str may be int).
418:      2) It the number of types in the union type, if we suppress the actual type
419:      '''
420:     expected_type = stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_copy.TypeInferenceProxy.instance(
421:         expected_type)
422:     expected_type.set_type_instance(True)
423: 
424:     #if actual_type.get_python_type() is expected_type.get_python_type():
425:     if actual_type == expected_type:
426:         return True, 0
427:     if stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy.is_union_type(actual_type):
428:         #type_is_in_union = expected_type in actual_type.types
429:         type_is_in_union = __type_is_in_union(actual_type.types, expected_type)
430:         if not type_is_in_union:
431:             return False, 0
432:         return True, len(actual_type.types) - 1
433:     return False, 0
434: 
435: 
436: def may_not_be_type(actual_type, expected_type):
437:     '''
438:     Returns:
439:      1) if the actual type is not the expected one, including the semantics of union types (int\/str may not be float).
440:      2) It the number of types in the union type, if we suppress the actual type
441:      '''
442:     expected_type = stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_copy.TypeInferenceProxy.instance(
443:         expected_type)
444:     expected_type.set_type_instance(True)
445: 
446:     if stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy.is_union_type(actual_type):
447:         # Type is not found, so it may not be type. Also type is found, but there are more types,
448:         # so it may not also be the type
449:         # type_is_not_in_union = expected_type not in actual_type.types or \
450:         #                        expected_type in actual_type.types and len(actual_type.types) > 1
451:         found = __type_is_in_union(actual_type.types, expected_type)
452:         type_is_not_in_union = not found or (found and len(actual_type.types) > 1)
453: 
454:         if type_is_not_in_union:
455:             return True, len(actual_type.types) - 1
456:         # All types found it is impossible that it may not be type
457:         return False, 0
458: 
459:     # if actual_type is not expected_type:
460:     #     return True, 0
461:     if not actual_type == expected_type:
462:         return True, 0
463: 
464:     return False, 0
465: 
466: 
467: def remove_type_from_union(union_type, type_to_remove):
468:     '''
469:     Removes the specified type from the passed union type
470:     :param union_type: Union type to remove from
471:     :param type_to_remove: Type to remove
472:     :return:
473:     '''
474:     if not stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy.is_union_type(union_type):
475:         return union_type
476:     result = None
477:     type_to_remove = stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_copy.TypeInferenceProxy.instance(
478:         type_to_remove)
479:     for type_ in union_type.types:
480:         if not type_ == type_to_remove:
481:             result = stypy_copy.python_lib.python_types.type_inference.union_type.UnionType.add(result, type_)
482:     return result
483: 

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

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')
import_1877 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_1877) is not StypyTypeError):

    if (import_1877 != 'pyd_module'):
        __import__(import_1877)
        sys_modules_1878 = sys.modules[import_1877]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_1878.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_1878, sys_modules_1878.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_1877)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy import TypeWarning' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')
import_1879 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy')

if (type(import_1879) is not StypyTypeError):

    if (import_1879 != 'pyd_module'):
        __import__(import_1879)
        sys_modules_1880 = sys.modules[import_1879]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', sys_modules_1880.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_1880, sys_modules_1880.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_warning_copy', import_1879)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, data_structures_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')
import_1881 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy')

if (type(import_1881) is not StypyTypeError):

    if (import_1881 != 'pyd_module'):
        __import__(import_1881)
        sys_modules_1882 = sys.modules[import_1881]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', sys_modules_1882.module_type_store, module_type_store, ['core_language_copy', 'data_structures_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_1882, sys_modules_1882.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, data_structures_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', None, module_type_store, ['core_language_copy', 'data_structures_copy'], [core_language_copy, data_structures_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', import_1881)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy import stypy_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')
import_1883 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy')

if (type(import_1883) is not StypyTypeError):

    if (import_1883 != 'pyd_module'):
        __import__(import_1883)
        sys_modules_1884 = sys.modules[import_1883]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy', sys_modules_1884.module_type_store, module_type_store, ['stypy_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_1884, sys_modules_1884.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy import stypy_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy', None, module_type_store, ['stypy_copy'], [stypy_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy', import_1883)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy import ENABLE_CODING_ADVICES' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')
import_1885 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy')

if (type(import_1885) is not StypyTypeError):

    if (import_1885 != 'pyd_module'):
        __import__(import_1885)
        sys_modules_1886 = sys.modules[import_1885]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy', sys_modules_1886.module_type_store, module_type_store, ['ENABLE_CODING_ADVICES'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_1886, sys_modules_1886.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy import ENABLE_CODING_ADVICES

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy', None, module_type_store, ['ENABLE_CODING_ADVICES'], [ENABLE_CODING_ADVICES])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.stypy_parameters_copy', import_1885)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy import format_function_name' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')
import_1887 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy')

if (type(import_1887) is not StypyTypeError):

    if (import_1887 != 'pyd_module'):
        __import__(import_1887)
        sys_modules_1888 = sys.modules[import_1887]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy', sys_modules_1888.module_type_store, module_type_store, ['format_function_name'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_1888, sys_modules_1888.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy import format_function_name

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy', None, module_type_store, ['format_function_name'], [format_function_name])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.reporting_copy.print_utils_copy', import_1887)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_generator_copy import Str, IterableObject' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')
import_1889 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_generator_copy')

if (type(import_1889) is not StypyTypeError):

    if (import_1889 != 'pyd_module'):
        __import__(import_1889)
        sys_modules_1890 = sys.modules[import_1889]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_generator_copy', sys_modules_1890.module_type_store, module_type_store, ['Str', 'IterableObject'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_1890, sys_modules_1890.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_generator_copy import Str, IterableObject

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_generator_copy', None, module_type_store, ['Str', 'IterableObject'], [Str, IterableObject])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_generator_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy.type_group_generator_copy', import_1889)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/')

str_1891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\nThis file holds functions that are invoked by the generated source code of type inference programs. This code\nusually need functions to perform common tasks that can be directly generated in Python source code, but this\nwill turn the source code of the type inference program into something much less manageable. Therefore, we\nidentified common tasks within this source code and encapsulated it into functions to be called, in order to\nmake the type inference programs smaller and more clear.\n')

@norecursion
def __assign_arguments(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__assign_arguments'
    module_type_store = module_type_store.open_function_context('__assign_arguments', 23, 0, False)
    
    # Passed parameters checking function
    __assign_arguments.stypy_localization = localization
    __assign_arguments.stypy_type_of_self = None
    __assign_arguments.stypy_type_store = module_type_store
    __assign_arguments.stypy_function_name = '__assign_arguments'
    __assign_arguments.stypy_param_names_list = ['localization', 'type_store', 'declared_arguments_list', 'type_of_args']
    __assign_arguments.stypy_varargs_param_name = None
    __assign_arguments.stypy_kwargs_param_name = None
    __assign_arguments.stypy_call_defaults = defaults
    __assign_arguments.stypy_call_varargs = varargs
    __assign_arguments.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__assign_arguments', ['localization', 'type_store', 'declared_arguments_list', 'type_of_args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__assign_arguments', localization, ['localization', 'type_store', 'declared_arguments_list', 'type_of_args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__assign_arguments(...)' code ##################

    str_1892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', '\n    Auxiliar function to assign the declared argument names and types to a type store. The purpose of this function\n    is to insert into the current type store and function context the name and value of the parameters so the code\n    can use them.\n    :param localization: Caller information\n    :param type_store: Current type store (it is assumed that the generated source code has created a new function\n    context for the called function\n    :param declared_arguments_list: Argument names\n    :param type_of_args: Type of arguments (order of argument names is used to establish a correspondence. If len of\n    declared_argument_list and type_of_args is not the same, the lower length is used)\n    :return:\n    ')
    
    
    # Call to len(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'declared_arguments_list' (line 38)
    declared_arguments_list_1894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'declared_arguments_list', False)
    # Processing the call keyword arguments (line 38)
    kwargs_1895 = {}
    # Getting the type of 'len' (line 38)
    len_1893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 7), 'len', False)
    # Calling len(args, kwargs) (line 38)
    len_call_result_1896 = invoke(stypy.reporting.localization.Localization(__file__, 38, 7), len_1893, *[declared_arguments_list_1894], **kwargs_1895)
    
    
    # Call to len(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'type_of_args' (line 38)
    type_of_args_1898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 42), 'type_of_args', False)
    # Processing the call keyword arguments (line 38)
    kwargs_1899 = {}
    # Getting the type of 'len' (line 38)
    len_1897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 38), 'len', False)
    # Calling len(args, kwargs) (line 38)
    len_call_result_1900 = invoke(stypy.reporting.localization.Localization(__file__, 38, 38), len_1897, *[type_of_args_1898], **kwargs_1899)
    
    # Applying the binary operator '<' (line 38)
    result_lt_1901 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 7), '<', len_call_result_1896, len_call_result_1900)
    
    # Testing if the type of an if condition is none (line 38)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 4), result_lt_1901):
        
        # Assigning a Call to a Name (line 41):
        
        # Assigning a Call to a Name (line 41):
        
        # Call to len(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'type_of_args' (line 41)
        type_of_args_1908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'type_of_args', False)
        # Processing the call keyword arguments (line 41)
        kwargs_1909 = {}
        # Getting the type of 'len' (line 41)
        len_1907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 18), 'len', False)
        # Calling len(args, kwargs) (line 41)
        len_call_result_1910 = invoke(stypy.reporting.localization.Localization(__file__, 41, 18), len_1907, *[type_of_args_1908], **kwargs_1909)
        
        # Assigning a type to the variable 'min_len' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'min_len', len_call_result_1910)
    else:
        
        # Testing the type of an if condition (line 38)
        if_condition_1902 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 4), result_lt_1901)
        # Assigning a type to the variable 'if_condition_1902' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'if_condition_1902', if_condition_1902)
        # SSA begins for if statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 39):
        
        # Assigning a Call to a Name (line 39):
        
        # Call to len(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'declared_arguments_list' (line 39)
        declared_arguments_list_1904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 22), 'declared_arguments_list', False)
        # Processing the call keyword arguments (line 39)
        kwargs_1905 = {}
        # Getting the type of 'len' (line 39)
        len_1903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'len', False)
        # Calling len(args, kwargs) (line 39)
        len_call_result_1906 = invoke(stypy.reporting.localization.Localization(__file__, 39, 18), len_1903, *[declared_arguments_list_1904], **kwargs_1905)
        
        # Assigning a type to the variable 'min_len' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'min_len', len_call_result_1906)
        # SSA branch for the else part of an if statement (line 38)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 41):
        
        # Assigning a Call to a Name (line 41):
        
        # Call to len(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'type_of_args' (line 41)
        type_of_args_1908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'type_of_args', False)
        # Processing the call keyword arguments (line 41)
        kwargs_1909 = {}
        # Getting the type of 'len' (line 41)
        len_1907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 18), 'len', False)
        # Calling len(args, kwargs) (line 41)
        len_call_result_1910 = invoke(stypy.reporting.localization.Localization(__file__, 41, 18), len_1907, *[type_of_args_1908], **kwargs_1909)
        
        # Assigning a type to the variable 'min_len' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'min_len', len_call_result_1910)
        # SSA join for if statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to range(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'min_len' (line 44)
    min_len_1912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'min_len', False)
    # Processing the call keyword arguments (line 44)
    kwargs_1913 = {}
    # Getting the type of 'range' (line 44)
    range_1911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'range', False)
    # Calling range(args, kwargs) (line 44)
    range_call_result_1914 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), range_1911, *[min_len_1912], **kwargs_1913)
    
    # Assigning a type to the variable 'range_call_result_1914' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'range_call_result_1914', range_call_result_1914)
    # Testing if the for loop is going to be iterated (line 44)
    # Testing the type of a for loop iterable (line 44)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 44, 4), range_call_result_1914)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 44, 4), range_call_result_1914):
        # Getting the type of the for loop variable (line 44)
        for_loop_var_1915 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 44, 4), range_call_result_1914)
        # Assigning a type to the variable 'i' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'i', for_loop_var_1915)
        # SSA begins for a for statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to set_type_of(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'localization' (line 45)
        localization_1918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'localization', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 45)
        i_1919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 69), 'i', False)
        # Getting the type of 'declared_arguments_list' (line 45)
        declared_arguments_list_1920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 45), 'declared_arguments_list', False)
        # Obtaining the member '__getitem__' of a type (line 45)
        getitem___1921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 45), declared_arguments_list_1920, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 45)
        subscript_call_result_1922 = invoke(stypy.reporting.localization.Localization(__file__, 45, 45), getitem___1921, i_1919)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 45)
        i_1923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 86), 'i', False)
        # Getting the type of 'type_of_args' (line 45)
        type_of_args_1924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 73), 'type_of_args', False)
        # Obtaining the member '__getitem__' of a type (line 45)
        getitem___1925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 73), type_of_args_1924, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 45)
        subscript_call_result_1926 = invoke(stypy.reporting.localization.Localization(__file__, 45, 73), getitem___1925, i_1923)
        
        # Processing the call keyword arguments (line 45)
        kwargs_1927 = {}
        # Getting the type of 'type_store' (line 45)
        type_store_1916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'type_store', False)
        # Obtaining the member 'set_type_of' of a type (line 45)
        set_type_of_1917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), type_store_1916, 'set_type_of')
        # Calling set_type_of(args, kwargs) (line 45)
        set_type_of_call_result_1928 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), set_type_of_1917, *[localization_1918, subscript_call_result_1922, subscript_call_result_1926], **kwargs_1927)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of '__assign_arguments(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__assign_arguments' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_1929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1929)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__assign_arguments'
    return stypy_return_type_1929

# Assigning a type to the variable '__assign_arguments' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), '__assign_arguments', __assign_arguments)

@norecursion
def __process_call_type_or_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__process_call_type_or_args'
    module_type_store = module_type_store.open_function_context('__process_call_type_or_args', 48, 0, False)
    
    # Passed parameters checking function
    __process_call_type_or_args.stypy_localization = localization
    __process_call_type_or_args.stypy_type_of_self = None
    __process_call_type_or_args.stypy_type_store = module_type_store
    __process_call_type_or_args.stypy_function_name = '__process_call_type_or_args'
    __process_call_type_or_args.stypy_param_names_list = ['function_name', 'localization', 'declared_argument_name_list', 'call_varargs', 'call_kwargs', 'defaults']
    __process_call_type_or_args.stypy_varargs_param_name = None
    __process_call_type_or_args.stypy_kwargs_param_name = None
    __process_call_type_or_args.stypy_call_defaults = defaults
    __process_call_type_or_args.stypy_call_varargs = varargs
    __process_call_type_or_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__process_call_type_or_args', ['function_name', 'localization', 'declared_argument_name_list', 'call_varargs', 'call_kwargs', 'defaults'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__process_call_type_or_args', localization, ['function_name', 'localization', 'declared_argument_name_list', 'call_varargs', 'call_kwargs', 'defaults'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__process_call_type_or_args(...)' code ##################

    str_1930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', "\n    As type inference programs convers any function call to the func(*args, **kwargs) form, this function checks both\n     parameters to assign its elements to declared named parameters, see if the call have been done with enough ones,\n     if the call has an excess or parameters and, in the end, calculate the real variable argument list and keywords\n     dictionary\n\n    :param function_name: Called function name\n    :param localization: Caller information\n    :param declared_argument_name_list: Declared argument list names (example: ['a', 'b'] in def f(a, b))\n    :param call_varargs: First parameter of the type inference function call\n    :param call_kwargs: Second parameter of the type inference function call\n    :param defaults: Declared defaults (values for those parameters that have a default value (are optional) in the\n    function declaration (example: [3,4] in def f(a, b=3, c=4))\n    :return:\n    ")
    
    # Assigning a List to a Name (line 69):
    
    # Assigning a List to a Name (line 69):
    
    # Obtaining an instance of the builtin type 'list' (line 69)
    list_1931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 69)
    
    # Assigning a type to the variable 'call_type_of_args' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'call_type_of_args', list_1931)
    
    # Assigning a Num to a Name (line 70):
    
    # Assigning a Num to a Name (line 70):
    int_1932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 11), 'int')
    # Assigning a type to the variable 'cont' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'cont', int_1932)
    
    # Assigning a Str to a Name (line 71):
    
    # Assigning a Str to a Name (line 71):
    str_1933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 16), 'str', '')
    # Assigning a type to the variable 'error_msg' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'error_msg', str_1933)
    
    # Assigning a Name to a Name (line 72):
    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'False' (line 72)
    False_1934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'False')
    # Assigning a type to the variable 'found_error' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'found_error', False_1934)
    
    # Assigning a Num to a Name (line 73):
    
    # Assigning a Num to a Name (line 73):
    int_1935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 16), 'int')
    # Assigning a type to the variable 'arg_count' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'arg_count', int_1935)
    
    # Getting the type of 'declared_argument_name_list' (line 78)
    declared_argument_name_list_1936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'declared_argument_name_list')
    # Assigning a type to the variable 'declared_argument_name_list_1936' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'declared_argument_name_list_1936', declared_argument_name_list_1936)
    # Testing if the for loop is going to be iterated (line 78)
    # Testing the type of a for loop iterable (line 78)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 4), declared_argument_name_list_1936)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 78, 4), declared_argument_name_list_1936):
        # Getting the type of the for loop variable (line 78)
        for_loop_var_1937 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 4), declared_argument_name_list_1936)
        # Assigning a type to the variable 'name' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'name', for_loop_var_1937)
        # SSA begins for a for statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to len(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'call_varargs' (line 80)
        call_varargs_1939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'call_varargs', False)
        # Processing the call keyword arguments (line 80)
        kwargs_1940 = {}
        # Getting the type of 'len' (line 80)
        len_1938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'len', False)
        # Calling len(args, kwargs) (line 80)
        len_call_result_1941 = invoke(stypy.reporting.localization.Localization(__file__, 80, 11), len_1938, *[call_varargs_1939], **kwargs_1940)
        
        # Getting the type of 'cont' (line 80)
        cont_1942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'cont')
        # Applying the binary operator '>' (line 80)
        result_gt_1943 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 11), '>', len_call_result_1941, cont_1942)
        
        # Testing if the type of an if condition is none (line 80)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 8), result_gt_1943):
            
            # Getting the type of 'name' (line 95)
            name_1990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'name')
            # Getting the type of 'call_kwargs' (line 95)
            call_kwargs_1991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'call_kwargs')
            # Applying the binary operator 'in' (line 95)
            result_contains_1992 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 15), 'in', name_1990, call_kwargs_1991)
            
            # Testing if the type of an if condition is none (line 95)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 95, 12), result_contains_1992):
                
                # Assigning a Name to a Name (line 101):
                
                # Assigning a Name to a Name (line 101):
                # Getting the type of 'True' (line 101)
                True_2007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'True')
                # Assigning a type to the variable 'found_error' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'found_error', True_2007)
                
                # Assigning a Call to a Name (line 102):
                
                # Assigning a Call to a Name (line 102):
                
                # Call to format(...): (line 102)
                # Processing the call arguments (line 102)
                
                # Call to format_function_name(...): (line 103)
                # Processing the call arguments (line 103)
                # Getting the type of 'function_name' (line 103)
                function_name_2011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 74), 'function_name', False)
                # Processing the call keyword arguments (line 103)
                kwargs_2012 = {}
                # Getting the type of 'format_function_name' (line 103)
                format_function_name_2010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 53), 'format_function_name', False)
                # Calling format_function_name(args, kwargs) (line 103)
                format_function_name_call_result_2013 = invoke(stypy.reporting.localization.Localization(__file__, 103, 53), format_function_name_2010, *[function_name_2011], **kwargs_2012)
                
                # Getting the type of 'arg_count' (line 103)
                arg_count_2014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 90), 'arg_count', False)
                # Getting the type of 'name' (line 103)
                name_2015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 101), 'name', False)
                # Processing the call keyword arguments (line 102)
                kwargs_2016 = {}
                str_2008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 22), 'str', "Insufficient number of arguments for {0}: Cannot find a value for argument number {1} ('{2}'); ")
                # Obtaining the member 'format' of a type (line 102)
                format_2009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 22), str_2008, 'format')
                # Calling format(args, kwargs) (line 102)
                format_call_result_2017 = invoke(stypy.reporting.localization.Localization(__file__, 102, 22), format_2009, *[format_function_name_call_result_2013, arg_count_2014, name_2015], **kwargs_2016)
                
                # Assigning a type to the variable 'msg' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'msg', format_call_result_2017)
                
                # Getting the type of 'error_msg' (line 104)
                error_msg_2018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'error_msg')
                # Getting the type of 'msg' (line 104)
                msg_2019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'msg')
                # Applying the binary operator '+=' (line 104)
                result_iadd_2020 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 16), '+=', error_msg_2018, msg_2019)
                # Assigning a type to the variable 'error_msg' (line 104)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'error_msg', result_iadd_2020)
                
                
                # Call to append(...): (line 106)
                # Processing the call arguments (line 106)
                
                # Call to TypeError(...): (line 106)
                # Processing the call arguments (line 106)
                # Getting the type of 'localization' (line 106)
                localization_2024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 51), 'localization', False)
                # Getting the type of 'msg' (line 106)
                msg_2025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 65), 'msg', False)
                # Processing the call keyword arguments (line 106)
                # Getting the type of 'False' (line 106)
                False_2026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 81), 'False', False)
                keyword_2027 = False_2026
                kwargs_2028 = {'prints_msg': keyword_2027}
                # Getting the type of 'TypeError' (line 106)
                TypeError_2023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 41), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 106)
                TypeError_call_result_2029 = invoke(stypy.reporting.localization.Localization(__file__, 106, 41), TypeError_2023, *[localization_2024, msg_2025], **kwargs_2028)
                
                # Processing the call keyword arguments (line 106)
                kwargs_2030 = {}
                # Getting the type of 'call_type_of_args' (line 106)
                call_type_of_args_2021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'call_type_of_args', False)
                # Obtaining the member 'append' of a type (line 106)
                append_2022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), call_type_of_args_2021, 'append')
                # Calling append(args, kwargs) (line 106)
                append_call_result_2031 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), append_2022, *[TypeError_call_result_2029], **kwargs_2030)
                
            else:
                
                # Testing the type of an if condition (line 95)
                if_condition_1993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 12), result_contains_1992)
                # Assigning a type to the variable 'if_condition_1993' (line 95)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'if_condition_1993', if_condition_1993)
                # SSA begins for if statement (line 95)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 96)
                # Processing the call arguments (line 96)
                
                # Obtaining the type of the subscript
                # Getting the type of 'name' (line 96)
                name_1996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 53), 'name', False)
                # Getting the type of 'call_kwargs' (line 96)
                call_kwargs_1997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 41), 'call_kwargs', False)
                # Obtaining the member '__getitem__' of a type (line 96)
                getitem___1998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 41), call_kwargs_1997, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 96)
                subscript_call_result_1999 = invoke(stypy.reporting.localization.Localization(__file__, 96, 41), getitem___1998, name_1996)
                
                # Processing the call keyword arguments (line 96)
                kwargs_2000 = {}
                # Getting the type of 'call_type_of_args' (line 96)
                call_type_of_args_1994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'call_type_of_args', False)
                # Obtaining the member 'append' of a type (line 96)
                append_1995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 16), call_type_of_args_1994, 'append')
                # Calling append(args, kwargs) (line 96)
                append_call_result_2001 = invoke(stypy.reporting.localization.Localization(__file__, 96, 16), append_1995, *[subscript_call_result_1999], **kwargs_2000)
                
                # Deleting a member
                # Getting the type of 'call_kwargs' (line 98)
                call_kwargs_2002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_kwargs')
                
                # Obtaining the type of the subscript
                # Getting the type of 'name' (line 98)
                name_2003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 32), 'name')
                # Getting the type of 'call_kwargs' (line 98)
                call_kwargs_2004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_kwargs')
                # Obtaining the member '__getitem__' of a type (line 98)
                getitem___2005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 20), call_kwargs_2004, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 98)
                subscript_call_result_2006 = invoke(stypy.reporting.localization.Localization(__file__, 98, 20), getitem___2005, name_2003)
                
                del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 16), call_kwargs_2002, subscript_call_result_2006)
                # SSA branch for the else part of an if statement (line 95)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 101):
                
                # Assigning a Name to a Name (line 101):
                # Getting the type of 'True' (line 101)
                True_2007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'True')
                # Assigning a type to the variable 'found_error' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'found_error', True_2007)
                
                # Assigning a Call to a Name (line 102):
                
                # Assigning a Call to a Name (line 102):
                
                # Call to format(...): (line 102)
                # Processing the call arguments (line 102)
                
                # Call to format_function_name(...): (line 103)
                # Processing the call arguments (line 103)
                # Getting the type of 'function_name' (line 103)
                function_name_2011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 74), 'function_name', False)
                # Processing the call keyword arguments (line 103)
                kwargs_2012 = {}
                # Getting the type of 'format_function_name' (line 103)
                format_function_name_2010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 53), 'format_function_name', False)
                # Calling format_function_name(args, kwargs) (line 103)
                format_function_name_call_result_2013 = invoke(stypy.reporting.localization.Localization(__file__, 103, 53), format_function_name_2010, *[function_name_2011], **kwargs_2012)
                
                # Getting the type of 'arg_count' (line 103)
                arg_count_2014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 90), 'arg_count', False)
                # Getting the type of 'name' (line 103)
                name_2015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 101), 'name', False)
                # Processing the call keyword arguments (line 102)
                kwargs_2016 = {}
                str_2008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 22), 'str', "Insufficient number of arguments for {0}: Cannot find a value for argument number {1} ('{2}'); ")
                # Obtaining the member 'format' of a type (line 102)
                format_2009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 22), str_2008, 'format')
                # Calling format(args, kwargs) (line 102)
                format_call_result_2017 = invoke(stypy.reporting.localization.Localization(__file__, 102, 22), format_2009, *[format_function_name_call_result_2013, arg_count_2014, name_2015], **kwargs_2016)
                
                # Assigning a type to the variable 'msg' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'msg', format_call_result_2017)
                
                # Getting the type of 'error_msg' (line 104)
                error_msg_2018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'error_msg')
                # Getting the type of 'msg' (line 104)
                msg_2019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'msg')
                # Applying the binary operator '+=' (line 104)
                result_iadd_2020 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 16), '+=', error_msg_2018, msg_2019)
                # Assigning a type to the variable 'error_msg' (line 104)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'error_msg', result_iadd_2020)
                
                
                # Call to append(...): (line 106)
                # Processing the call arguments (line 106)
                
                # Call to TypeError(...): (line 106)
                # Processing the call arguments (line 106)
                # Getting the type of 'localization' (line 106)
                localization_2024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 51), 'localization', False)
                # Getting the type of 'msg' (line 106)
                msg_2025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 65), 'msg', False)
                # Processing the call keyword arguments (line 106)
                # Getting the type of 'False' (line 106)
                False_2026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 81), 'False', False)
                keyword_2027 = False_2026
                kwargs_2028 = {'prints_msg': keyword_2027}
                # Getting the type of 'TypeError' (line 106)
                TypeError_2023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 41), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 106)
                TypeError_call_result_2029 = invoke(stypy.reporting.localization.Localization(__file__, 106, 41), TypeError_2023, *[localization_2024, msg_2025], **kwargs_2028)
                
                # Processing the call keyword arguments (line 106)
                kwargs_2030 = {}
                # Getting the type of 'call_type_of_args' (line 106)
                call_type_of_args_2021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'call_type_of_args', False)
                # Obtaining the member 'append' of a type (line 106)
                append_2022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), call_type_of_args_2021, 'append')
                # Calling append(args, kwargs) (line 106)
                append_call_result_2031 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), append_2022, *[TypeError_call_result_2029], **kwargs_2030)
                
                # SSA join for if statement (line 95)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 80)
            if_condition_1944 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), result_gt_1943)
            # Assigning a type to the variable 'if_condition_1944' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_1944', if_condition_1944)
            # SSA begins for if statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 81)
            # Processing the call arguments (line 81)
            
            # Obtaining the type of the subscript
            # Getting the type of 'cont' (line 81)
            cont_1947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 50), 'cont', False)
            # Getting the type of 'call_varargs' (line 81)
            call_varargs_1948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 37), 'call_varargs', False)
            # Obtaining the member '__getitem__' of a type (line 81)
            getitem___1949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 37), call_varargs_1948, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 81)
            subscript_call_result_1950 = invoke(stypy.reporting.localization.Localization(__file__, 81, 37), getitem___1949, cont_1947)
            
            # Processing the call keyword arguments (line 81)
            kwargs_1951 = {}
            # Getting the type of 'call_type_of_args' (line 81)
            call_type_of_args_1945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'call_type_of_args', False)
            # Obtaining the member 'append' of a type (line 81)
            append_1946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), call_type_of_args_1945, 'append')
            # Calling append(args, kwargs) (line 81)
            append_call_result_1952 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), append_1946, *[subscript_call_result_1950], **kwargs_1951)
            
            
            # Evaluating a boolean operation
            
            # Getting the type of 'name' (line 84)
            name_1953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'name')
            # Getting the type of 'call_kwargs' (line 84)
            call_kwargs_1954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'call_kwargs')
            # Applying the binary operator 'in' (line 84)
            result_contains_1955 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 15), 'in', name_1953, call_kwargs_1954)
            
            
            # Getting the type of 'name' (line 84)
            name_1956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 39), 'name')
            # Getting the type of 'defaults' (line 84)
            defaults_1957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 51), 'defaults')
            # Applying the binary operator 'notin' (line 84)
            result_contains_1958 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 39), 'notin', name_1956, defaults_1957)
            
            # Applying the binary operator 'and' (line 84)
            result_and_keyword_1959 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 15), 'and', result_contains_1955, result_contains_1958)
            
            # Testing if the type of an if condition is none (line 84)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 84, 12), result_and_keyword_1959):
                pass
            else:
                
                # Testing the type of an if condition (line 84)
                if_condition_1960 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 12), result_and_keyword_1959)
                # Assigning a type to the variable 'if_condition_1960' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'if_condition_1960', if_condition_1960)
                # SSA begins for if statement (line 84)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 85):
                
                # Assigning a Name to a Name (line 85):
                # Getting the type of 'True' (line 85)
                True_1961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 30), 'True')
                # Assigning a type to the variable 'found_error' (line 85)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'found_error', True_1961)
                
                # Assigning a Call to a Name (line 86):
                
                # Assigning a Call to a Name (line 86):
                
                # Call to format(...): (line 86)
                # Processing the call arguments (line 86)
                
                # Call to format_function_name(...): (line 86)
                # Processing the call arguments (line 86)
                # Getting the type of 'function_name' (line 86)
                function_name_1965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 105), 'function_name', False)
                # Processing the call keyword arguments (line 86)
                kwargs_1966 = {}
                # Getting the type of 'format_function_name' (line 86)
                format_function_name_1964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 84), 'format_function_name', False)
                # Calling format_function_name(args, kwargs) (line 86)
                format_function_name_call_result_1967 = invoke(stypy.reporting.localization.Localization(__file__, 86, 84), format_function_name_1964, *[function_name_1965], **kwargs_1966)
                
                # Getting the type of 'name' (line 87)
                name_1968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 84), 'name', False)
                # Processing the call keyword arguments (line 86)
                kwargs_1969 = {}
                str_1962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 22), 'str', "{0} got multiple values for keyword argument '{1}'; ")
                # Obtaining the member 'format' of a type (line 86)
                format_1963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 22), str_1962, 'format')
                # Calling format(args, kwargs) (line 86)
                format_call_result_1970 = invoke(stypy.reporting.localization.Localization(__file__, 86, 22), format_1963, *[format_function_name_call_result_1967, name_1968], **kwargs_1969)
                
                # Assigning a type to the variable 'msg' (line 86)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'msg', format_call_result_1970)
                
                # Getting the type of 'error_msg' (line 88)
                error_msg_1971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'error_msg')
                # Getting the type of 'msg' (line 88)
                msg_1972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'msg')
                # Applying the binary operator '+=' (line 88)
                result_iadd_1973 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 16), '+=', error_msg_1971, msg_1972)
                # Assigning a type to the variable 'error_msg' (line 88)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'error_msg', result_iadd_1973)
                
                
                # Call to append(...): (line 89)
                # Processing the call arguments (line 89)
                
                # Call to TypeError(...): (line 89)
                # Processing the call arguments (line 89)
                # Getting the type of 'localization' (line 89)
                localization_1977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 51), 'localization', False)
                # Getting the type of 'msg' (line 89)
                msg_1978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 65), 'msg', False)
                # Processing the call keyword arguments (line 89)
                # Getting the type of 'False' (line 89)
                False_1979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 81), 'False', False)
                keyword_1980 = False_1979
                kwargs_1981 = {'prints_msg': keyword_1980}
                # Getting the type of 'TypeError' (line 89)
                TypeError_1976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 41), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 89)
                TypeError_call_result_1982 = invoke(stypy.reporting.localization.Localization(__file__, 89, 41), TypeError_1976, *[localization_1977, msg_1978], **kwargs_1981)
                
                # Processing the call keyword arguments (line 89)
                kwargs_1983 = {}
                # Getting the type of 'call_type_of_args' (line 89)
                call_type_of_args_1974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'call_type_of_args', False)
                # Obtaining the member 'append' of a type (line 89)
                append_1975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 16), call_type_of_args_1974, 'append')
                # Calling append(args, kwargs) (line 89)
                append_call_result_1984 = invoke(stypy.reporting.localization.Localization(__file__, 89, 16), append_1975, *[TypeError_call_result_1982], **kwargs_1983)
                
                # SSA join for if statement (line 84)
                module_type_store = module_type_store.join_ssa_context()
                

            # Deleting a member
            # Getting the type of 'call_varargs' (line 92)
            call_varargs_1985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'call_varargs')
            
            # Obtaining the type of the subscript
            # Getting the type of 'cont' (line 92)
            cont_1986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 29), 'cont')
            # Getting the type of 'call_varargs' (line 92)
            call_varargs_1987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'call_varargs')
            # Obtaining the member '__getitem__' of a type (line 92)
            getitem___1988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 16), call_varargs_1987, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 92)
            subscript_call_result_1989 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), getitem___1988, cont_1986)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 12), call_varargs_1985, subscript_call_result_1989)
            # SSA branch for the else part of an if statement (line 80)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'name' (line 95)
            name_1990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'name')
            # Getting the type of 'call_kwargs' (line 95)
            call_kwargs_1991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 23), 'call_kwargs')
            # Applying the binary operator 'in' (line 95)
            result_contains_1992 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 15), 'in', name_1990, call_kwargs_1991)
            
            # Testing if the type of an if condition is none (line 95)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 95, 12), result_contains_1992):
                
                # Assigning a Name to a Name (line 101):
                
                # Assigning a Name to a Name (line 101):
                # Getting the type of 'True' (line 101)
                True_2007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'True')
                # Assigning a type to the variable 'found_error' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'found_error', True_2007)
                
                # Assigning a Call to a Name (line 102):
                
                # Assigning a Call to a Name (line 102):
                
                # Call to format(...): (line 102)
                # Processing the call arguments (line 102)
                
                # Call to format_function_name(...): (line 103)
                # Processing the call arguments (line 103)
                # Getting the type of 'function_name' (line 103)
                function_name_2011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 74), 'function_name', False)
                # Processing the call keyword arguments (line 103)
                kwargs_2012 = {}
                # Getting the type of 'format_function_name' (line 103)
                format_function_name_2010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 53), 'format_function_name', False)
                # Calling format_function_name(args, kwargs) (line 103)
                format_function_name_call_result_2013 = invoke(stypy.reporting.localization.Localization(__file__, 103, 53), format_function_name_2010, *[function_name_2011], **kwargs_2012)
                
                # Getting the type of 'arg_count' (line 103)
                arg_count_2014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 90), 'arg_count', False)
                # Getting the type of 'name' (line 103)
                name_2015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 101), 'name', False)
                # Processing the call keyword arguments (line 102)
                kwargs_2016 = {}
                str_2008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 22), 'str', "Insufficient number of arguments for {0}: Cannot find a value for argument number {1} ('{2}'); ")
                # Obtaining the member 'format' of a type (line 102)
                format_2009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 22), str_2008, 'format')
                # Calling format(args, kwargs) (line 102)
                format_call_result_2017 = invoke(stypy.reporting.localization.Localization(__file__, 102, 22), format_2009, *[format_function_name_call_result_2013, arg_count_2014, name_2015], **kwargs_2016)
                
                # Assigning a type to the variable 'msg' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'msg', format_call_result_2017)
                
                # Getting the type of 'error_msg' (line 104)
                error_msg_2018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'error_msg')
                # Getting the type of 'msg' (line 104)
                msg_2019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'msg')
                # Applying the binary operator '+=' (line 104)
                result_iadd_2020 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 16), '+=', error_msg_2018, msg_2019)
                # Assigning a type to the variable 'error_msg' (line 104)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'error_msg', result_iadd_2020)
                
                
                # Call to append(...): (line 106)
                # Processing the call arguments (line 106)
                
                # Call to TypeError(...): (line 106)
                # Processing the call arguments (line 106)
                # Getting the type of 'localization' (line 106)
                localization_2024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 51), 'localization', False)
                # Getting the type of 'msg' (line 106)
                msg_2025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 65), 'msg', False)
                # Processing the call keyword arguments (line 106)
                # Getting the type of 'False' (line 106)
                False_2026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 81), 'False', False)
                keyword_2027 = False_2026
                kwargs_2028 = {'prints_msg': keyword_2027}
                # Getting the type of 'TypeError' (line 106)
                TypeError_2023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 41), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 106)
                TypeError_call_result_2029 = invoke(stypy.reporting.localization.Localization(__file__, 106, 41), TypeError_2023, *[localization_2024, msg_2025], **kwargs_2028)
                
                # Processing the call keyword arguments (line 106)
                kwargs_2030 = {}
                # Getting the type of 'call_type_of_args' (line 106)
                call_type_of_args_2021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'call_type_of_args', False)
                # Obtaining the member 'append' of a type (line 106)
                append_2022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), call_type_of_args_2021, 'append')
                # Calling append(args, kwargs) (line 106)
                append_call_result_2031 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), append_2022, *[TypeError_call_result_2029], **kwargs_2030)
                
            else:
                
                # Testing the type of an if condition (line 95)
                if_condition_1993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 12), result_contains_1992)
                # Assigning a type to the variable 'if_condition_1993' (line 95)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'if_condition_1993', if_condition_1993)
                # SSA begins for if statement (line 95)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to append(...): (line 96)
                # Processing the call arguments (line 96)
                
                # Obtaining the type of the subscript
                # Getting the type of 'name' (line 96)
                name_1996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 53), 'name', False)
                # Getting the type of 'call_kwargs' (line 96)
                call_kwargs_1997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 41), 'call_kwargs', False)
                # Obtaining the member '__getitem__' of a type (line 96)
                getitem___1998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 41), call_kwargs_1997, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 96)
                subscript_call_result_1999 = invoke(stypy.reporting.localization.Localization(__file__, 96, 41), getitem___1998, name_1996)
                
                # Processing the call keyword arguments (line 96)
                kwargs_2000 = {}
                # Getting the type of 'call_type_of_args' (line 96)
                call_type_of_args_1994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'call_type_of_args', False)
                # Obtaining the member 'append' of a type (line 96)
                append_1995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 16), call_type_of_args_1994, 'append')
                # Calling append(args, kwargs) (line 96)
                append_call_result_2001 = invoke(stypy.reporting.localization.Localization(__file__, 96, 16), append_1995, *[subscript_call_result_1999], **kwargs_2000)
                
                # Deleting a member
                # Getting the type of 'call_kwargs' (line 98)
                call_kwargs_2002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_kwargs')
                
                # Obtaining the type of the subscript
                # Getting the type of 'name' (line 98)
                name_2003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 32), 'name')
                # Getting the type of 'call_kwargs' (line 98)
                call_kwargs_2004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'call_kwargs')
                # Obtaining the member '__getitem__' of a type (line 98)
                getitem___2005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 20), call_kwargs_2004, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 98)
                subscript_call_result_2006 = invoke(stypy.reporting.localization.Localization(__file__, 98, 20), getitem___2005, name_2003)
                
                del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 16), call_kwargs_2002, subscript_call_result_2006)
                # SSA branch for the else part of an if statement (line 95)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Name to a Name (line 101):
                
                # Assigning a Name to a Name (line 101):
                # Getting the type of 'True' (line 101)
                True_2007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'True')
                # Assigning a type to the variable 'found_error' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'found_error', True_2007)
                
                # Assigning a Call to a Name (line 102):
                
                # Assigning a Call to a Name (line 102):
                
                # Call to format(...): (line 102)
                # Processing the call arguments (line 102)
                
                # Call to format_function_name(...): (line 103)
                # Processing the call arguments (line 103)
                # Getting the type of 'function_name' (line 103)
                function_name_2011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 74), 'function_name', False)
                # Processing the call keyword arguments (line 103)
                kwargs_2012 = {}
                # Getting the type of 'format_function_name' (line 103)
                format_function_name_2010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 53), 'format_function_name', False)
                # Calling format_function_name(args, kwargs) (line 103)
                format_function_name_call_result_2013 = invoke(stypy.reporting.localization.Localization(__file__, 103, 53), format_function_name_2010, *[function_name_2011], **kwargs_2012)
                
                # Getting the type of 'arg_count' (line 103)
                arg_count_2014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 90), 'arg_count', False)
                # Getting the type of 'name' (line 103)
                name_2015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 101), 'name', False)
                # Processing the call keyword arguments (line 102)
                kwargs_2016 = {}
                str_2008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 22), 'str', "Insufficient number of arguments for {0}: Cannot find a value for argument number {1} ('{2}'); ")
                # Obtaining the member 'format' of a type (line 102)
                format_2009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 22), str_2008, 'format')
                # Calling format(args, kwargs) (line 102)
                format_call_result_2017 = invoke(stypy.reporting.localization.Localization(__file__, 102, 22), format_2009, *[format_function_name_call_result_2013, arg_count_2014, name_2015], **kwargs_2016)
                
                # Assigning a type to the variable 'msg' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'msg', format_call_result_2017)
                
                # Getting the type of 'error_msg' (line 104)
                error_msg_2018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'error_msg')
                # Getting the type of 'msg' (line 104)
                msg_2019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 29), 'msg')
                # Applying the binary operator '+=' (line 104)
                result_iadd_2020 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 16), '+=', error_msg_2018, msg_2019)
                # Assigning a type to the variable 'error_msg' (line 104)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'error_msg', result_iadd_2020)
                
                
                # Call to append(...): (line 106)
                # Processing the call arguments (line 106)
                
                # Call to TypeError(...): (line 106)
                # Processing the call arguments (line 106)
                # Getting the type of 'localization' (line 106)
                localization_2024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 51), 'localization', False)
                # Getting the type of 'msg' (line 106)
                msg_2025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 65), 'msg', False)
                # Processing the call keyword arguments (line 106)
                # Getting the type of 'False' (line 106)
                False_2026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 81), 'False', False)
                keyword_2027 = False_2026
                kwargs_2028 = {'prints_msg': keyword_2027}
                # Getting the type of 'TypeError' (line 106)
                TypeError_2023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 41), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 106)
                TypeError_call_result_2029 = invoke(stypy.reporting.localization.Localization(__file__, 106, 41), TypeError_2023, *[localization_2024, msg_2025], **kwargs_2028)
                
                # Processing the call keyword arguments (line 106)
                kwargs_2030 = {}
                # Getting the type of 'call_type_of_args' (line 106)
                call_type_of_args_2021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'call_type_of_args', False)
                # Obtaining the member 'append' of a type (line 106)
                append_2022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 16), call_type_of_args_2021, 'append')
                # Calling append(args, kwargs) (line 106)
                append_call_result_2031 = invoke(stypy.reporting.localization.Localization(__file__, 106, 16), append_2022, *[TypeError_call_result_2029], **kwargs_2030)
                
                # SSA join for if statement (line 95)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 80)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'arg_count' (line 109)
        arg_count_2032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'arg_count')
        int_2033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'int')
        # Applying the binary operator '+=' (line 109)
        result_iadd_2034 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 8), '+=', arg_count_2032, int_2033)
        # Assigning a type to the variable 'arg_count' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'arg_count', result_iadd_2034)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'found_error' (line 112)
    found_error_2035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'found_error')
    # Testing if the type of an if condition is none (line 112)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 112, 4), found_error_2035):
        pass
    else:
        
        # Testing the type of an if condition (line 112)
        if_condition_2036 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 4), found_error_2035)
        # Assigning a type to the variable 'if_condition_2036' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'if_condition_2036', if_condition_2036)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 113)
        tuple_2037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 113)
        # Adding element type (line 113)
        # Getting the type of 'call_type_of_args' (line 113)
        call_type_of_args_2038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'call_type_of_args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), tuple_2037, call_type_of_args_2038)
        # Adding element type (line 113)
        # Getting the type of 'error_msg' (line 113)
        error_msg_2039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 34), 'error_msg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), tuple_2037, error_msg_2039)
        # Adding element type (line 113)
        # Getting the type of 'True' (line 113)
        True_2040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 45), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 15), tuple_2037, True_2040)
        
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'stypy_return_type', tuple_2037)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'tuple' (line 116)
    tuple_2041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 116)
    # Adding element type (line 116)
    # Getting the type of 'call_type_of_args' (line 116)
    call_type_of_args_2042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'call_type_of_args')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 11), tuple_2041, call_type_of_args_2042)
    # Adding element type (line 116)
    str_2043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 30), 'str', '')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 11), tuple_2041, str_2043)
    # Adding element type (line 116)
    # Getting the type of 'False' (line 116)
    False_2044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 34), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 11), tuple_2041, False_2044)
    
    # Assigning a type to the variable 'stypy_return_type' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type', tuple_2041)
    
    # ################# End of '__process_call_type_or_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__process_call_type_or_args' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_2045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2045)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__process_call_type_or_args'
    return stypy_return_type_2045

# Assigning a type to the variable '__process_call_type_or_args' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), '__process_call_type_or_args', __process_call_type_or_args)

@norecursion
def process_argument_values(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Call to list(...): (line 124)
    # Processing the call keyword arguments (line 124)
    kwargs_2047 = {}
    # Getting the type of 'list' (line 124)
    list_2046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 41), 'list', False)
    # Calling list(args, kwargs) (line 124)
    list_call_result_2048 = invoke(stypy.reporting.localization.Localization(__file__, 124, 41), list_2046, *[], **kwargs_2047)
    
    
    # Obtaining an instance of the builtin type 'dict' (line 125)
    dict_2049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 40), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 125)
    
    # Getting the type of 'True' (line 126)
    True_2050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 52), 'True')
    defaults = [list_call_result_2048, dict_2049, True_2050]
    # Create a new context for function 'process_argument_values'
    module_type_store = module_type_store.open_function_context('process_argument_values', 119, 0, False)
    
    # Passed parameters checking function
    process_argument_values.stypy_localization = localization
    process_argument_values.stypy_type_of_self = None
    process_argument_values.stypy_type_store = module_type_store
    process_argument_values.stypy_function_name = 'process_argument_values'
    process_argument_values.stypy_param_names_list = ['localization', 'type_of_self', 'type_store', 'function_name', 'declared_argument_name_list', 'declared_varargs_var', 'declared_kwargs_var', 'declared_defaults', 'call_varargs', 'call_kwargs', 'allow_argument_keywords']
    process_argument_values.stypy_varargs_param_name = None
    process_argument_values.stypy_kwargs_param_name = None
    process_argument_values.stypy_call_defaults = defaults
    process_argument_values.stypy_call_varargs = varargs
    process_argument_values.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'process_argument_values', ['localization', 'type_of_self', 'type_store', 'function_name', 'declared_argument_name_list', 'declared_varargs_var', 'declared_kwargs_var', 'declared_defaults', 'call_varargs', 'call_kwargs', 'allow_argument_keywords'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'process_argument_values', localization, ['localization', 'type_of_self', 'type_store', 'function_name', 'declared_argument_name_list', 'declared_varargs_var', 'declared_kwargs_var', 'declared_defaults', 'call_varargs', 'call_kwargs', 'allow_argument_keywords'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'process_argument_values(...)' code ##################

    str_2051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, (-1)), 'str', '\n    This long function is the responsible of checking all the parameters passed to a function call and make sure that\n    the call is valid and possible. Argument passing in Python is a complex task, because there are several argument\n    types and combinations, and the mechanism is rather flexible, so care was taken to try to identify those\n    combinations, identify misuses of the call mechanism and assign the correct values to arguments. The function is\n    long and complex, so documentation was placed to try to clarify the behaviour of each part.\n\n    :param localization: Caller information\n    :param type_of_self: Type of the owner of the function/method. Currently unused, may be used for reporting errors.\n    :param type_store: Current type store (a function context for the current function have to be already set up)\n    :param function_name: Name of the function/method/lambda function that is being invoked\n    :param declared_argument_name_list: List of named arguments declared in the source code of the function\n    (example [\'a\', \'n\'] in def f(a, n))\n    :param declared_varargs_var: Name of the parameter that holds the variable argument list (if any)\n    (example: "args" in def f(*args))\n    :param declared_kwargs_var: Name of the parameter that holds the keyword argument dictionary (if any).\n    (example: "kwargs" in def f(**kwargs))\n    :param declared_defaults: Declared default values for arguments (if present).\n    (example: [3, 4] in def f(a=3, n=4)\n    :param call_varargs: Calls to functions/methods in type inference programs only have two parameters: args\n    (variable argument list) and kwargs (keyword argument dictionary). This is done in order to simplify call\n     handling, as any function call can be expressed this way.\n    Example: f(*args, **kwargs)\n    Values for the declared arguments are extracted from the varargs list in order (so the rest of the arguments are\n    the real variable list of arguments of the original function).\n    :param call_kwargs: This dictionary holds pairs of (name, type). If name is in the declared argument list, the\n     corresponding type is assigned to this named parameter. If it is not, it is left inside the kwargs dictionary of\n     the function. In the end, the declared_kwargs_var values are those that will not be assigned to named parameters.\n    :param allow_argument_keywords: Python API functions do not allow the usage of named keywords when calling them.\n    This disallow the usage of this kind of calls and report errors if used with this kind of functions.\n    :return:\n    ')
    
    # Assigning a Name to a Name (line 161):
    
    # Assigning a Name to a Name (line 161):
    # Getting the type of 'False' (line 161)
    False_2052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 18), 'False')
    # Assigning a type to the variable 'found_error' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'found_error', False_2052)
    
    # Assigning a Str to a Name (line 162):
    
    # Assigning a Str to a Name (line 162):
    str_2053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 16), 'str', '')
    # Assigning a type to the variable 'error_msg' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'error_msg', str_2053)
    
    # Evaluating a boolean operation
    
    # Getting the type of 'allow_argument_keywords' (line 167)
    allow_argument_keywords_2054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'allow_argument_keywords')
    # Applying the 'not' unary operator (line 167)
    result_not__2055 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 7), 'not', allow_argument_keywords_2054)
    
    
    
    # Call to len(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'call_kwargs' (line 167)
    call_kwargs_2057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 43), 'call_kwargs', False)
    # Processing the call keyword arguments (line 167)
    kwargs_2058 = {}
    # Getting the type of 'len' (line 167)
    len_2056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 39), 'len', False)
    # Calling len(args, kwargs) (line 167)
    len_call_result_2059 = invoke(stypy.reporting.localization.Localization(__file__, 167, 39), len_2056, *[call_kwargs_2057], **kwargs_2058)
    
    int_2060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 58), 'int')
    # Applying the binary operator '>' (line 167)
    result_gt_2061 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 39), '>', len_call_result_2059, int_2060)
    
    # Applying the binary operator 'and' (line 167)
    result_and_keyword_2062 = python_operator(stypy.reporting.localization.Localization(__file__, 167, 7), 'and', result_not__2055, result_gt_2061)
    
    # Testing if the type of an if condition is none (line 167)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 167, 4), result_and_keyword_2062):
        pass
    else:
        
        # Testing the type of an if condition (line 167)
        if_condition_2063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 167, 4), result_and_keyword_2062)
        # Assigning a type to the variable 'if_condition_2063' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'if_condition_2063', if_condition_2063)
        # SSA begins for if statement (line 167)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'call_kwargs' (line 168)
        call_kwargs_2064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'call_kwargs')
        # Assigning a type to the variable 'call_kwargs_2064' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'call_kwargs_2064', call_kwargs_2064)
        # Testing if the for loop is going to be iterated (line 168)
        # Testing the type of a for loop iterable (line 168)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 168, 8), call_kwargs_2064)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 168, 8), call_kwargs_2064):
            # Getting the type of the for loop variable (line 168)
            for_loop_var_2065 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 168, 8), call_kwargs_2064)
            # Assigning a type to the variable 'arg' (line 168)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'arg', for_loop_var_2065)
            # SSA begins for a for statement (line 168)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'arg' (line 169)
            arg_2066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'arg')
            # Getting the type of 'declared_defaults' (line 169)
            declared_defaults_2067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'declared_defaults')
            # Applying the binary operator 'notin' (line 169)
            result_contains_2068 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 15), 'notin', arg_2066, declared_defaults_2067)
            
            # Testing if the type of an if condition is none (line 169)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 169, 12), result_contains_2068):
                pass
            else:
                
                # Testing the type of an if condition (line 169)
                if_condition_2069 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 12), result_contains_2068)
                # Assigning a type to the variable 'if_condition_2069' (line 169)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'if_condition_2069', if_condition_2069)
                # SSA begins for if statement (line 169)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 170):
                
                # Assigning a Name to a Name (line 170):
                # Getting the type of 'True' (line 170)
                True_2070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 30), 'True')
                # Assigning a type to the variable 'found_error' (line 170)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'found_error', True_2070)
                
                # Getting the type of 'error_msg' (line 171)
                error_msg_2071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'error_msg')
                
                # Call to format(...): (line 171)
                # Processing the call arguments (line 171)
                
                # Call to format_function_name(...): (line 171)
                # Processing the call arguments (line 171)
                # Getting the type of 'function_name' (line 171)
                function_name_2075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 92), 'function_name', False)
                # Processing the call keyword arguments (line 171)
                kwargs_2076 = {}
                # Getting the type of 'format_function_name' (line 171)
                format_function_name_2074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 71), 'format_function_name', False)
                # Calling format_function_name(args, kwargs) (line 171)
                format_function_name_call_result_2077 = invoke(stypy.reporting.localization.Localization(__file__, 171, 71), format_function_name_2074, *[function_name_2075], **kwargs_2076)
                
                # Processing the call keyword arguments (line 171)
                kwargs_2078 = {}
                str_2072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 29), 'str', '{0} takes no keyword arguments; ')
                # Obtaining the member 'format' of a type (line 171)
                format_2073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 29), str_2072, 'format')
                # Calling format(args, kwargs) (line 171)
                format_call_result_2079 = invoke(stypy.reporting.localization.Localization(__file__, 171, 29), format_2073, *[format_function_name_call_result_2077], **kwargs_2078)
                
                # Applying the binary operator '+=' (line 171)
                result_iadd_2080 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 16), '+=', error_msg_2071, format_call_result_2079)
                # Assigning a type to the variable 'error_msg' (line 171)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 16), 'error_msg', result_iadd_2080)
                
                # SSA join for if statement (line 169)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 167)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to get_context(...): (line 175)
    # Processing the call keyword arguments (line 175)
    kwargs_2083 = {}
    # Getting the type of 'type_store' (line 175)
    type_store_2081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 14), 'type_store', False)
    # Obtaining the member 'get_context' of a type (line 175)
    get_context_2082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 14), type_store_2081, 'get_context')
    # Calling get_context(args, kwargs) (line 175)
    get_context_call_result_2084 = invoke(stypy.reporting.localization.Localization(__file__, 175, 14), get_context_2082, *[], **kwargs_2083)
    
    # Assigning a type to the variable 'context' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'context', get_context_call_result_2084)
    
    # Assigning a Name to a Attribute (line 176):
    
    # Assigning a Name to a Attribute (line 176):
    # Getting the type of 'declared_argument_name_list' (line 176)
    declared_argument_name_list_2085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 42), 'declared_argument_name_list')
    # Getting the type of 'context' (line 176)
    context_2086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'context')
    # Setting the type of the member 'declared_argument_name_list' of a type (line 176)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 4), context_2086, 'declared_argument_name_list', declared_argument_name_list_2085)
    
    # Assigning a Name to a Attribute (line 177):
    
    # Assigning a Name to a Attribute (line 177):
    # Getting the type of 'declared_varargs_var' (line 177)
    declared_varargs_var_2087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 35), 'declared_varargs_var')
    # Getting the type of 'context' (line 177)
    context_2088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'context')
    # Setting the type of the member 'declared_varargs_var' of a type (line 177)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 4), context_2088, 'declared_varargs_var', declared_varargs_var_2087)
    
    # Assigning a Name to a Attribute (line 178):
    
    # Assigning a Name to a Attribute (line 178):
    # Getting the type of 'declared_kwargs_var' (line 178)
    declared_kwargs_var_2089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 34), 'declared_kwargs_var')
    # Getting the type of 'context' (line 178)
    context_2090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'context')
    # Setting the type of the member 'declared_kwargs_var' of a type (line 178)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 4), context_2090, 'declared_kwargs_var', declared_kwargs_var_2089)
    
    # Assigning a Name to a Attribute (line 179):
    
    # Assigning a Name to a Attribute (line 179):
    # Getting the type of 'declared_defaults' (line 179)
    declared_defaults_2091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 32), 'declared_defaults')
    # Getting the type of 'context' (line 179)
    context_2092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'context')
    # Setting the type of the member 'declared_defaults' of a type (line 179)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 4), context_2092, 'declared_defaults', declared_defaults_2091)
    
    # Evaluating a boolean operation
    
    
    # Call to type(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'declared_defaults' (line 184)
    declared_defaults_2094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'declared_defaults', False)
    # Processing the call keyword arguments (line 184)
    kwargs_2095 = {}
    # Getting the type of 'type' (line 184)
    type_2093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 7), 'type', False)
    # Calling type(args, kwargs) (line 184)
    type_call_result_2096 = invoke(stypy.reporting.localization.Localization(__file__, 184, 7), type_2093, *[declared_defaults_2094], **kwargs_2095)
    
    # Getting the type of 'list' (line 184)
    list_2097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'list')
    # Applying the binary operator 'is' (line 184)
    result_is__2098 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 7), 'is', type_call_result_2096, list_2097)
    
    
    
    # Call to type(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'declared_defaults' (line 184)
    declared_defaults_2100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 47), 'declared_defaults', False)
    # Processing the call keyword arguments (line 184)
    kwargs_2101 = {}
    # Getting the type of 'type' (line 184)
    type_2099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 42), 'type', False)
    # Calling type(args, kwargs) (line 184)
    type_call_result_2102 = invoke(stypy.reporting.localization.Localization(__file__, 184, 42), type_2099, *[declared_defaults_2100], **kwargs_2101)
    
    # Getting the type of 'tuple' (line 184)
    tuple_2103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 69), 'tuple')
    # Applying the binary operator 'is' (line 184)
    result_is__2104 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 42), 'is', type_call_result_2102, tuple_2103)
    
    # Applying the binary operator 'or' (line 184)
    result_or_keyword_2105 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 7), 'or', result_is__2098, result_is__2104)
    
    # Testing if the type of an if condition is none (line 184)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 184, 4), result_or_keyword_2105):
        
        # Assigning a Name to a Name (line 197):
        
        # Assigning a Name to a Name (line 197):
        # Getting the type of 'declared_defaults' (line 197)
        declared_defaults_2136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 24), 'declared_defaults')
        # Assigning a type to the variable 'defaults_dict' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'defaults_dict', declared_defaults_2136)
    else:
        
        # Testing the type of an if condition (line 184)
        if_condition_2106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 4), result_or_keyword_2105)
        # Assigning a type to the variable 'if_condition_2106' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'if_condition_2106', if_condition_2106)
        # SSA begins for if statement (line 184)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Dict to a Name (line 185):
        
        # Assigning a Dict to a Name (line 185):
        
        # Obtaining an instance of the builtin type 'dict' (line 185)
        dict_2107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 185)
        
        # Assigning a type to the variable 'defaults_dict' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'defaults_dict', dict_2107)
        
        # Call to reverse(...): (line 187)
        # Processing the call keyword arguments (line 187)
        kwargs_2110 = {}
        # Getting the type of 'declared_argument_name_list' (line 187)
        declared_argument_name_list_2108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'declared_argument_name_list', False)
        # Obtaining the member 'reverse' of a type (line 187)
        reverse_2109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 8), declared_argument_name_list_2108, 'reverse')
        # Calling reverse(args, kwargs) (line 187)
        reverse_call_result_2111 = invoke(stypy.reporting.localization.Localization(__file__, 187, 8), reverse_2109, *[], **kwargs_2110)
        
        
        # Call to reverse(...): (line 188)
        # Processing the call keyword arguments (line 188)
        kwargs_2114 = {}
        # Getting the type of 'declared_defaults' (line 188)
        declared_defaults_2112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'declared_defaults', False)
        # Obtaining the member 'reverse' of a type (line 188)
        reverse_2113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 8), declared_defaults_2112, 'reverse')
        # Calling reverse(args, kwargs) (line 188)
        reverse_call_result_2115 = invoke(stypy.reporting.localization.Localization(__file__, 188, 8), reverse_2113, *[], **kwargs_2114)
        
        
        # Assigning a Num to a Name (line 189):
        
        # Assigning a Num to a Name (line 189):
        int_2116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 15), 'int')
        # Assigning a type to the variable 'cont' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'cont', int_2116)
        
        # Getting the type of 'declared_defaults' (line 190)
        declared_defaults_2117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 21), 'declared_defaults')
        # Assigning a type to the variable 'declared_defaults_2117' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'declared_defaults_2117', declared_defaults_2117)
        # Testing if the for loop is going to be iterated (line 190)
        # Testing the type of a for loop iterable (line 190)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 190, 8), declared_defaults_2117)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 190, 8), declared_defaults_2117):
            # Getting the type of the for loop variable (line 190)
            for_loop_var_2118 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 190, 8), declared_defaults_2117)
            # Assigning a type to the variable 'value' (line 190)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'value', for_loop_var_2118)
            # SSA begins for a for statement (line 190)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Name to a Subscript (line 191):
            
            # Assigning a Name to a Subscript (line 191):
            # Getting the type of 'value' (line 191)
            value_2119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 63), 'value')
            # Getting the type of 'defaults_dict' (line 191)
            defaults_dict_2120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'defaults_dict')
            
            # Obtaining the type of the subscript
            # Getting the type of 'cont' (line 191)
            cont_2121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 54), 'cont')
            # Getting the type of 'declared_argument_name_list' (line 191)
            declared_argument_name_list_2122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 26), 'declared_argument_name_list')
            # Obtaining the member '__getitem__' of a type (line 191)
            getitem___2123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 26), declared_argument_name_list_2122, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 191)
            subscript_call_result_2124 = invoke(stypy.reporting.localization.Localization(__file__, 191, 26), getitem___2123, cont_2121)
            
            # Storing an element on a container (line 191)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 12), defaults_dict_2120, (subscript_call_result_2124, value_2119))
            
            # Getting the type of 'cont' (line 192)
            cont_2125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'cont')
            int_2126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 20), 'int')
            # Applying the binary operator '+=' (line 192)
            result_iadd_2127 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 12), '+=', cont_2125, int_2126)
            # Assigning a type to the variable 'cont' (line 192)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'cont', result_iadd_2127)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to reverse(...): (line 194)
        # Processing the call keyword arguments (line 194)
        kwargs_2130 = {}
        # Getting the type of 'declared_argument_name_list' (line 194)
        declared_argument_name_list_2128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'declared_argument_name_list', False)
        # Obtaining the member 'reverse' of a type (line 194)
        reverse_2129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 8), declared_argument_name_list_2128, 'reverse')
        # Calling reverse(args, kwargs) (line 194)
        reverse_call_result_2131 = invoke(stypy.reporting.localization.Localization(__file__, 194, 8), reverse_2129, *[], **kwargs_2130)
        
        
        # Call to reverse(...): (line 195)
        # Processing the call keyword arguments (line 195)
        kwargs_2134 = {}
        # Getting the type of 'declared_defaults' (line 195)
        declared_defaults_2132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'declared_defaults', False)
        # Obtaining the member 'reverse' of a type (line 195)
        reverse_2133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 8), declared_defaults_2132, 'reverse')
        # Calling reverse(args, kwargs) (line 195)
        reverse_call_result_2135 = invoke(stypy.reporting.localization.Localization(__file__, 195, 8), reverse_2133, *[], **kwargs_2134)
        
        # SSA branch for the else part of an if statement (line 184)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 197):
        
        # Assigning a Name to a Name (line 197):
        # Getting the type of 'declared_defaults' (line 197)
        declared_defaults_2136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 24), 'declared_defaults')
        # Assigning a type to the variable 'defaults_dict' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'defaults_dict', declared_defaults_2136)
        # SSA join for if statement (line 184)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 200):
    
    # Assigning a Call to a Name (line 200):
    
    # Call to list(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'call_varargs' (line 200)
    call_varargs_2138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 24), 'call_varargs', False)
    # Processing the call keyword arguments (line 200)
    kwargs_2139 = {}
    # Getting the type of 'list' (line 200)
    list_2137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'list', False)
    # Calling list(args, kwargs) (line 200)
    list_call_result_2140 = invoke(stypy.reporting.localization.Localization(__file__, 200, 19), list_2137, *[call_varargs_2138], **kwargs_2139)
    
    # Assigning a type to the variable 'call_varargs' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'call_varargs', list_call_result_2140)
    
    # Getting the type of 'defaults_dict' (line 203)
    defaults_dict_2141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'defaults_dict')
    # Assigning a type to the variable 'defaults_dict_2141' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'defaults_dict_2141', defaults_dict_2141)
    # Testing if the for loop is going to be iterated (line 203)
    # Testing the type of a for loop iterable (line 203)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 203, 4), defaults_dict_2141)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 203, 4), defaults_dict_2141):
        # Getting the type of the for loop variable (line 203)
        for_loop_var_2142 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 203, 4), defaults_dict_2141)
        # Assigning a type to the variable 'elem' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'elem', for_loop_var_2142)
        # SSA begins for a for statement (line 203)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'elem' (line 204)
        elem_2143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 11), 'elem')
        # Getting the type of 'call_kwargs' (line 204)
        call_kwargs_2144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 23), 'call_kwargs')
        # Applying the binary operator 'notin' (line 204)
        result_contains_2145 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 11), 'notin', elem_2143, call_kwargs_2144)
        
        # Testing if the type of an if condition is none (line 204)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 204, 8), result_contains_2145):
            pass
        else:
            
            # Testing the type of an if condition (line 204)
            if_condition_2146 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 204, 8), result_contains_2145)
            # Assigning a type to the variable 'if_condition_2146' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'if_condition_2146', if_condition_2146)
            # SSA begins for if statement (line 204)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Subscript (line 205):
            
            # Assigning a Subscript to a Subscript (line 205):
            
            # Obtaining the type of the subscript
            # Getting the type of 'elem' (line 205)
            elem_2147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 46), 'elem')
            # Getting the type of 'defaults_dict' (line 205)
            defaults_dict_2148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 32), 'defaults_dict')
            # Obtaining the member '__getitem__' of a type (line 205)
            getitem___2149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 32), defaults_dict_2148, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 205)
            subscript_call_result_2150 = invoke(stypy.reporting.localization.Localization(__file__, 205, 32), getitem___2149, elem_2147)
            
            # Getting the type of 'call_kwargs' (line 205)
            call_kwargs_2151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'call_kwargs')
            # Getting the type of 'elem' (line 205)
            elem_2152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'elem')
            # Storing an element on a container (line 205)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 12), call_kwargs_2151, (elem_2152, subscript_call_result_2150))
            # SSA join for if statement (line 204)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Call to a Tuple (line 210):
    
    # Assigning a Call to a Name:
    
    # Call to __process_call_type_or_args(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'function_name' (line 210)
    function_name_2154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 85), 'function_name', False)
    # Getting the type of 'localization' (line 211)
    localization_2155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 85), 'localization', False)
    # Getting the type of 'declared_argument_name_list' (line 212)
    declared_argument_name_list_2156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 85), 'declared_argument_name_list', False)
    # Getting the type of 'call_varargs' (line 213)
    call_varargs_2157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 85), 'call_varargs', False)
    # Getting the type of 'call_kwargs' (line 214)
    call_kwargs_2158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 85), 'call_kwargs', False)
    # Getting the type of 'defaults_dict' (line 215)
    defaults_dict_2159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 85), 'defaults_dict', False)
    # Processing the call keyword arguments (line 210)
    kwargs_2160 = {}
    # Getting the type of '__process_call_type_or_args' (line 210)
    process_call_type_or_args_2153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 57), '__process_call_type_or_args', False)
    # Calling __process_call_type_or_args(args, kwargs) (line 210)
    process_call_type_or_args_call_result_2161 = invoke(stypy.reporting.localization.Localization(__file__, 210, 57), process_call_type_or_args_2153, *[function_name_2154, localization_2155, declared_argument_name_list_2156, call_varargs_2157, call_kwargs_2158, defaults_dict_2159], **kwargs_2160)
    
    # Assigning a type to the variable 'call_assignment_1873' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_1873', process_call_type_or_args_call_result_2161)
    
    # Assigning a Call to a Name (line 210):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_1873' (line 210)
    call_assignment_1873_2162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_1873', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_2163 = stypy_get_value_from_tuple(call_assignment_1873_2162, 3, 0)
    
    # Assigning a type to the variable 'call_assignment_1874' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_1874', stypy_get_value_from_tuple_call_result_2163)
    
    # Assigning a Name to a Name (line 210):
    # Getting the type of 'call_assignment_1874' (line 210)
    call_assignment_1874_2164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_1874')
    # Assigning a type to the variable 'call_type_of_args' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_type_of_args', call_assignment_1874_2164)
    
    # Assigning a Call to a Name (line 210):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_1873' (line 210)
    call_assignment_1873_2165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_1873', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_2166 = stypy_get_value_from_tuple(call_assignment_1873_2165, 3, 1)
    
    # Assigning a type to the variable 'call_assignment_1875' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_1875', stypy_get_value_from_tuple_call_result_2166)
    
    # Assigning a Name to a Name (line 210):
    # Getting the type of 'call_assignment_1875' (line 210)
    call_assignment_1875_2167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_1875')
    # Assigning a type to the variable 'error' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 23), 'error', call_assignment_1875_2167)
    
    # Assigning a Call to a Name (line 210):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_1873' (line 210)
    call_assignment_1873_2168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_1873', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_2169 = stypy_get_value_from_tuple(call_assignment_1873_2168, 3, 2)
    
    # Assigning a type to the variable 'call_assignment_1876' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_1876', stypy_get_value_from_tuple_call_result_2169)
    
    # Assigning a Name to a Name (line 210):
    # Getting the type of 'call_assignment_1876' (line 210)
    call_assignment_1876_2170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'call_assignment_1876')
    # Assigning a type to the variable 'found_error_on_call_args' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'found_error_on_call_args', call_assignment_1876_2170)
    # Getting the type of 'found_error_on_call_args' (line 217)
    found_error_on_call_args_2171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 7), 'found_error_on_call_args')
    # Testing if the type of an if condition is none (line 217)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 217, 4), found_error_on_call_args_2171):
        pass
    else:
        
        # Testing the type of an if condition (line 217)
        if_condition_2172 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 4), found_error_on_call_args_2171)
        # Assigning a type to the variable 'if_condition_2172' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'if_condition_2172', if_condition_2172)
        # SSA begins for if statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'error_msg' (line 218)
        error_msg_2173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'error_msg')
        # Getting the type of 'error' (line 218)
        error_2174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 21), 'error')
        # Applying the binary operator '+=' (line 218)
        result_iadd_2175 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 8), '+=', error_msg_2173, error_2174)
        # Assigning a type to the variable 'error_msg' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'error_msg', result_iadd_2175)
        
        # SSA join for if statement (line 217)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'found_error' (line 220)
    found_error_2176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'found_error')
    # Getting the type of 'found_error_on_call_args' (line 220)
    found_error_on_call_args_2177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'found_error_on_call_args')
    # Applying the binary operator '|=' (line 220)
    result_ior_2178 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 4), '|=', found_error_2176, found_error_on_call_args_2177)
    # Assigning a type to the variable 'found_error' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'found_error', result_ior_2178)
    
    
    # Call to __assign_arguments(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'localization' (line 223)
    localization_2180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'localization', False)
    # Getting the type of 'type_store' (line 223)
    type_store_2181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 37), 'type_store', False)
    # Getting the type of 'declared_argument_name_list' (line 223)
    declared_argument_name_list_2182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 49), 'declared_argument_name_list', False)
    # Getting the type of 'call_type_of_args' (line 223)
    call_type_of_args_2183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 78), 'call_type_of_args', False)
    # Processing the call keyword arguments (line 223)
    kwargs_2184 = {}
    # Getting the type of '__assign_arguments' (line 223)
    assign_arguments_2179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), '__assign_arguments', False)
    # Calling __assign_arguments(args, kwargs) (line 223)
    assign_arguments_call_result_2185 = invoke(stypy.reporting.localization.Localization(__file__, 223, 4), assign_arguments_2179, *[localization_2180, type_store_2181, declared_argument_name_list_2182, call_type_of_args_2183], **kwargs_2184)
    
    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Call to keys(...): (line 228)
    # Processing the call keyword arguments (line 228)
    kwargs_2188 = {}
    # Getting the type of 'call_kwargs' (line 228)
    call_kwargs_2186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 18), 'call_kwargs', False)
    # Obtaining the member 'keys' of a type (line 228)
    keys_2187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 18), call_kwargs_2186, 'keys')
    # Calling keys(args, kwargs) (line 228)
    keys_call_result_2189 = invoke(stypy.reporting.localization.Localization(__file__, 228, 18), keys_2187, *[], **kwargs_2188)
    
    # Assigning a type to the variable 'left_kwargs' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'left_kwargs', keys_call_result_2189)
    
    # Getting the type of 'left_kwargs' (line 229)
    left_kwargs_2190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 16), 'left_kwargs')
    # Assigning a type to the variable 'left_kwargs_2190' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'left_kwargs_2190', left_kwargs_2190)
    # Testing if the for loop is going to be iterated (line 229)
    # Testing the type of a for loop iterable (line 229)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 229, 4), left_kwargs_2190)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 229, 4), left_kwargs_2190):
        # Getting the type of the for loop variable (line 229)
        for_loop_var_2191 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 229, 4), left_kwargs_2190)
        # Assigning a type to the variable 'name' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'name', for_loop_var_2191)
        # SSA begins for a for statement (line 229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'name' (line 230)
        name_2192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 11), 'name')
        # Getting the type of 'defaults_dict' (line 230)
        defaults_dict_2193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 19), 'defaults_dict')
        # Applying the binary operator 'in' (line 230)
        result_contains_2194 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 11), 'in', name_2192, defaults_dict_2193)
        
        # Testing if the type of an if condition is none (line 230)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 230, 8), result_contains_2194):
            pass
        else:
            
            # Testing the type of an if condition (line 230)
            if_condition_2195 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 8), result_contains_2194)
            # Assigning a type to the variable 'if_condition_2195' (line 230)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'if_condition_2195', if_condition_2195)
            # SSA begins for if statement (line 230)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Deleting a member
            # Getting the type of 'call_kwargs' (line 231)
            call_kwargs_2196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'call_kwargs')
            
            # Obtaining the type of the subscript
            # Getting the type of 'name' (line 231)
            name_2197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 28), 'name')
            # Getting the type of 'call_kwargs' (line 231)
            call_kwargs_2198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'call_kwargs')
            # Obtaining the member '__getitem__' of a type (line 231)
            getitem___2199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 16), call_kwargs_2198, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 231)
            subscript_call_result_2200 = invoke(stypy.reporting.localization.Localization(__file__, 231, 16), getitem___2199, name_2197)
            
            del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 12), call_kwargs_2196, subscript_call_result_2200)
            # SSA join for if statement (line 230)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Type idiom detected: calculating its left and rigth part (line 234)
    # Getting the type of 'declared_varargs_var' (line 234)
    declared_varargs_var_2201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'declared_varargs_var')
    # Getting the type of 'None' (line 234)
    None_2202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 35), 'None')
    
    (may_be_2203, more_types_in_union_2204) = may_not_be_none(declared_varargs_var_2201, None_2202)

    if may_be_2203:

        if more_types_in_union_2204:
            # Runtime conditional SSA (line 234)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 237):
        
        # Assigning a Call to a Name (line 237):
        
        # Call to get_builtin_type(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'localization' (line 237)
        localization_2208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 77), 'localization', False)
        str_2209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 91), 'str', 'tuple')
        # Processing the call keyword arguments (line 237)
        kwargs_2210 = {}
        # Getting the type of 'stypy_copy' (line 237)
        stypy_copy_2205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 27), 'stypy_copy', False)
        # Obtaining the member 'python_interface_copy' of a type (line 237)
        python_interface_copy_2206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 27), stypy_copy_2205, 'python_interface_copy')
        # Obtaining the member 'get_builtin_type' of a type (line 237)
        get_builtin_type_2207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 27), python_interface_copy_2206, 'get_builtin_type')
        # Calling get_builtin_type(args, kwargs) (line 237)
        get_builtin_type_call_result_2211 = invoke(stypy.reporting.localization.Localization(__file__, 237, 27), get_builtin_type_2207, *[localization_2208, str_2209], **kwargs_2210)
        
        # Assigning a type to the variable 'excess_arguments' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'excess_arguments', get_builtin_type_call_result_2211)
        
        # Call to add_types_from_list(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'localization' (line 238)
        localization_2214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 45), 'localization', False)
        # Getting the type of 'call_varargs' (line 238)
        call_varargs_2215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 59), 'call_varargs', False)
        # Processing the call keyword arguments (line 238)
        # Getting the type of 'False' (line 238)
        False_2216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 91), 'False', False)
        keyword_2217 = False_2216
        kwargs_2218 = {'record_annotation': keyword_2217}
        # Getting the type of 'excess_arguments' (line 238)
        excess_arguments_2212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'excess_arguments', False)
        # Obtaining the member 'add_types_from_list' of a type (line 238)
        add_types_from_list_2213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 8), excess_arguments_2212, 'add_types_from_list')
        # Calling add_types_from_list(args, kwargs) (line 238)
        add_types_from_list_call_result_2219 = invoke(stypy.reporting.localization.Localization(__file__, 238, 8), add_types_from_list_2213, *[localization_2214, call_varargs_2215], **kwargs_2218)
        
        
        # Call to set_type_of(...): (line 239)
        # Processing the call arguments (line 239)
        # Getting the type of 'localization' (line 239)
        localization_2222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 31), 'localization', False)
        # Getting the type of 'declared_varargs_var' (line 239)
        declared_varargs_var_2223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 45), 'declared_varargs_var', False)
        # Getting the type of 'excess_arguments' (line 239)
        excess_arguments_2224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 67), 'excess_arguments', False)
        # Processing the call keyword arguments (line 239)
        kwargs_2225 = {}
        # Getting the type of 'type_store' (line 239)
        type_store_2220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'type_store', False)
        # Obtaining the member 'set_type_of' of a type (line 239)
        set_type_of_2221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), type_store_2220, 'set_type_of')
        # Calling set_type_of(args, kwargs) (line 239)
        set_type_of_call_result_2226 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), set_type_of_2221, *[localization_2222, declared_varargs_var_2223, excess_arguments_2224], **kwargs_2225)
        

        if more_types_in_union_2204:
            # Runtime conditional SSA for else branch (line 234)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_2203) or more_types_in_union_2204):
        
        
        # Call to len(...): (line 243)
        # Processing the call arguments (line 243)
        # Getting the type of 'call_varargs' (line 243)
        call_varargs_2228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 15), 'call_varargs', False)
        # Processing the call keyword arguments (line 243)
        kwargs_2229 = {}
        # Getting the type of 'len' (line 243)
        len_2227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'len', False)
        # Calling len(args, kwargs) (line 243)
        len_call_result_2230 = invoke(stypy.reporting.localization.Localization(__file__, 243, 11), len_2227, *[call_varargs_2228], **kwargs_2229)
        
        int_2231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 31), 'int')
        # Applying the binary operator '>' (line 243)
        result_gt_2232 = python_operator(stypy.reporting.localization.Localization(__file__, 243, 11), '>', len_call_result_2230, int_2231)
        
        # Testing if the type of an if condition is none (line 243)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 243, 8), result_gt_2232):
            pass
        else:
            
            # Testing the type of an if condition (line 243)
            if_condition_2233 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 243, 8), result_gt_2232)
            # Assigning a type to the variable 'if_condition_2233' (line 243)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 8), 'if_condition_2233', if_condition_2233)
            # SSA begins for if statement (line 243)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 244):
            
            # Assigning a Name to a Name (line 244):
            # Getting the type of 'True' (line 244)
            True_2234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 26), 'True')
            # Assigning a type to the variable 'found_error' (line 244)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'found_error', True_2234)
            
            # Getting the type of 'error_msg' (line 245)
            error_msg_2235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'error_msg')
            
            # Call to format(...): (line 245)
            # Processing the call arguments (line 245)
            
            # Call to format_function_name(...): (line 245)
            # Processing the call arguments (line 245)
            # Getting the type of 'function_name' (line 245)
            function_name_2239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 98), 'function_name', False)
            # Processing the call keyword arguments (line 245)
            kwargs_2240 = {}
            # Getting the type of 'format_function_name' (line 245)
            format_function_name_2238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 77), 'format_function_name', False)
            # Calling format_function_name(args, kwargs) (line 245)
            format_function_name_call_result_2241 = invoke(stypy.reporting.localization.Localization(__file__, 245, 77), format_function_name_2238, *[function_name_2239], **kwargs_2240)
            
            
            # Call to str(...): (line 246)
            # Processing the call arguments (line 246)
            
            # Call to len(...): (line 246)
            # Processing the call arguments (line 246)
            # Getting the type of 'call_varargs' (line 246)
            call_varargs_2244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 85), 'call_varargs', False)
            # Processing the call keyword arguments (line 246)
            kwargs_2245 = {}
            # Getting the type of 'len' (line 246)
            len_2243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 81), 'len', False)
            # Calling len(args, kwargs) (line 246)
            len_call_result_2246 = invoke(stypy.reporting.localization.Localization(__file__, 246, 81), len_2243, *[call_varargs_2244], **kwargs_2245)
            
            # Processing the call keyword arguments (line 246)
            kwargs_2247 = {}
            # Getting the type of 'str' (line 246)
            str_2242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 77), 'str', False)
            # Calling str(args, kwargs) (line 246)
            str_call_result_2248 = invoke(stypy.reporting.localization.Localization(__file__, 246, 77), str_2242, *[len_call_result_2246], **kwargs_2247)
            
            # Processing the call keyword arguments (line 245)
            kwargs_2249 = {}
            str_2236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 25), 'str', '{0} got {1} more arguments than expected; ')
            # Obtaining the member 'format' of a type (line 245)
            format_2237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 25), str_2236, 'format')
            # Calling format(args, kwargs) (line 245)
            format_call_result_2250 = invoke(stypy.reporting.localization.Localization(__file__, 245, 25), format_2237, *[format_function_name_call_result_2241, str_call_result_2248], **kwargs_2249)
            
            # Applying the binary operator '+=' (line 245)
            result_iadd_2251 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 12), '+=', error_msg_2235, format_call_result_2250)
            # Assigning a type to the variable 'error_msg' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'error_msg', result_iadd_2251)
            
            # SSA join for if statement (line 243)
            module_type_store = module_type_store.join_ssa_context()
            


        if (may_be_2203 and more_types_in_union_2204):
            # SSA join for if statement (line 234)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 250)
    # Getting the type of 'declared_kwargs_var' (line 250)
    declared_kwargs_var_2252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'declared_kwargs_var')
    # Getting the type of 'None' (line 250)
    None_2253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 34), 'None')
    
    (may_be_2254, more_types_in_union_2255) = may_not_be_none(declared_kwargs_var_2252, None_2253)

    if may_be_2254:

        if more_types_in_union_2255:
            # Runtime conditional SSA (line 250)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 251):
        
        # Assigning a Call to a Name (line 251):
        
        # Call to get_builtin_type(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'localization' (line 251)
        localization_2259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 76), 'localization', False)
        str_2260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 90), 'str', 'dict')
        # Processing the call keyword arguments (line 251)
        kwargs_2261 = {}
        # Getting the type of 'stypy_copy' (line 251)
        stypy_copy_2256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 26), 'stypy_copy', False)
        # Obtaining the member 'python_interface_copy' of a type (line 251)
        python_interface_copy_2257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 26), stypy_copy_2256, 'python_interface_copy')
        # Obtaining the member 'get_builtin_type' of a type (line 251)
        get_builtin_type_2258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 26), python_interface_copy_2257, 'get_builtin_type')
        # Calling get_builtin_type(args, kwargs) (line 251)
        get_builtin_type_call_result_2262 = invoke(stypy.reporting.localization.Localization(__file__, 251, 26), get_builtin_type_2258, *[localization_2259, str_2260], **kwargs_2261)
        
        # Assigning a type to the variable 'kwargs_variable' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'kwargs_variable', get_builtin_type_call_result_2262)
        
        # Getting the type of 'call_kwargs' (line 254)
        call_kwargs_2263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 20), 'call_kwargs')
        # Assigning a type to the variable 'call_kwargs_2263' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'call_kwargs_2263', call_kwargs_2263)
        # Testing if the for loop is going to be iterated (line 254)
        # Testing the type of a for loop iterable (line 254)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 254, 8), call_kwargs_2263)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 254, 8), call_kwargs_2263):
            # Getting the type of the for loop variable (line 254)
            for_loop_var_2264 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 254, 8), call_kwargs_2263)
            # Assigning a type to the variable 'name' (line 254)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'name', for_loop_var_2264)
            # SSA begins for a for statement (line 254)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 255):
            
            # Assigning a Call to a Name (line 255):
            
            # Call to get_builtin_type(...): (line 255)
            # Processing the call arguments (line 255)
            # Getting the type of 'localization' (line 255)
            localization_2268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 69), 'localization', False)
            str_2269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 83), 'str', 'str')
            # Processing the call keyword arguments (line 255)
            # Getting the type of 'name' (line 255)
            name_2270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 96), 'name', False)
            keyword_2271 = name_2270
            kwargs_2272 = {'value': keyword_2271}
            # Getting the type of 'stypy_copy' (line 255)
            stypy_copy_2265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 19), 'stypy_copy', False)
            # Obtaining the member 'python_interface_copy' of a type (line 255)
            python_interface_copy_2266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 19), stypy_copy_2265, 'python_interface_copy')
            # Obtaining the member 'get_builtin_type' of a type (line 255)
            get_builtin_type_2267 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 255, 19), python_interface_copy_2266, 'get_builtin_type')
            # Calling get_builtin_type(args, kwargs) (line 255)
            get_builtin_type_call_result_2273 = invoke(stypy.reporting.localization.Localization(__file__, 255, 19), get_builtin_type_2267, *[localization_2268, str_2269], **kwargs_2272)
            
            # Assigning a type to the variable 'str_' (line 255)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'str_', get_builtin_type_call_result_2273)
            
            # Call to add_key_and_value_type(...): (line 256)
            # Processing the call arguments (line 256)
            # Getting the type of 'localization' (line 256)
            localization_2276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 51), 'localization', False)
            
            # Obtaining an instance of the builtin type 'tuple' (line 256)
            tuple_2277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 66), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 256)
            # Adding element type (line 256)
            # Getting the type of 'str_' (line 256)
            str__2278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 66), 'str_', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 66), tuple_2277, str__2278)
            # Adding element type (line 256)
            
            # Obtaining the type of the subscript
            # Getting the type of 'name' (line 256)
            name_2279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 84), 'name', False)
            # Getting the type of 'call_kwargs' (line 256)
            call_kwargs_2280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 72), 'call_kwargs', False)
            # Obtaining the member '__getitem__' of a type (line 256)
            getitem___2281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 72), call_kwargs_2280, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 256)
            subscript_call_result_2282 = invoke(stypy.reporting.localization.Localization(__file__, 256, 72), getitem___2281, name_2279)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 66), tuple_2277, subscript_call_result_2282)
            
            # Processing the call keyword arguments (line 256)
            # Getting the type of 'False' (line 256)
            False_2283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 110), 'False', False)
            keyword_2284 = False_2283
            kwargs_2285 = {'record_annotation': keyword_2284}
            # Getting the type of 'kwargs_variable' (line 256)
            kwargs_variable_2274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'kwargs_variable', False)
            # Obtaining the member 'add_key_and_value_type' of a type (line 256)
            add_key_and_value_type_2275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 12), kwargs_variable_2274, 'add_key_and_value_type')
            # Calling add_key_and_value_type(args, kwargs) (line 256)
            add_key_and_value_type_call_result_2286 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), add_key_and_value_type_2275, *[localization_2276, tuple_2277], **kwargs_2285)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to set_type_of(...): (line 258)
        # Processing the call arguments (line 258)
        # Getting the type of 'localization' (line 258)
        localization_2289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 31), 'localization', False)
        # Getting the type of 'declared_kwargs_var' (line 258)
        declared_kwargs_var_2290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 45), 'declared_kwargs_var', False)
        # Getting the type of 'kwargs_variable' (line 258)
        kwargs_variable_2291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 66), 'kwargs_variable', False)
        # Processing the call keyword arguments (line 258)
        kwargs_2292 = {}
        # Getting the type of 'type_store' (line 258)
        type_store_2287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'type_store', False)
        # Obtaining the member 'set_type_of' of a type (line 258)
        set_type_of_2288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 8), type_store_2287, 'set_type_of')
        # Calling set_type_of(args, kwargs) (line 258)
        set_type_of_call_result_2293 = invoke(stypy.reporting.localization.Localization(__file__, 258, 8), set_type_of_2288, *[localization_2289, declared_kwargs_var_2290, kwargs_variable_2291], **kwargs_2292)
        

        if more_types_in_union_2255:
            # Runtime conditional SSA for else branch (line 250)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_2254) or more_types_in_union_2255):
        
        
        # Call to len(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'call_kwargs' (line 262)
        call_kwargs_2295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 15), 'call_kwargs', False)
        # Processing the call keyword arguments (line 262)
        kwargs_2296 = {}
        # Getting the type of 'len' (line 262)
        len_2294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 11), 'len', False)
        # Calling len(args, kwargs) (line 262)
        len_call_result_2297 = invoke(stypy.reporting.localization.Localization(__file__, 262, 11), len_2294, *[call_kwargs_2295], **kwargs_2296)
        
        int_2298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 30), 'int')
        # Applying the binary operator '>' (line 262)
        result_gt_2299 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 11), '>', len_call_result_2297, int_2298)
        
        # Testing if the type of an if condition is none (line 262)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 262, 8), result_gt_2299):
            pass
        else:
            
            # Testing the type of an if condition (line 262)
            if_condition_2300 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 262, 8), result_gt_2299)
            # Assigning a type to the variable 'if_condition_2300' (line 262)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'if_condition_2300', if_condition_2300)
            # SSA begins for if statement (line 262)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 263):
            
            # Assigning a Name to a Name (line 263):
            # Getting the type of 'True' (line 263)
            True_2301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 26), 'True')
            # Assigning a type to the variable 'found_error' (line 263)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'found_error', True_2301)
            
            # Getting the type of 'error_msg' (line 264)
            error_msg_2302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'error_msg')
            
            # Call to format(...): (line 264)
            # Processing the call arguments (line 264)
            
            # Call to format_function_name(...): (line 264)
            # Processing the call arguments (line 264)
            # Getting the type of 'function_name' (line 264)
            function_name_2306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 99), 'function_name', False)
            # Processing the call keyword arguments (line 264)
            kwargs_2307 = {}
            # Getting the type of 'format_function_name' (line 264)
            format_function_name_2305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 78), 'format_function_name', False)
            # Calling format_function_name(args, kwargs) (line 264)
            format_function_name_call_result_2308 = invoke(stypy.reporting.localization.Localization(__file__, 264, 78), format_function_name_2305, *[function_name_2306], **kwargs_2307)
            
            
            # Call to str(...): (line 265)
            # Processing the call arguments (line 265)
            # Getting the type of 'call_kwargs' (line 265)
            call_kwargs_2310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 82), 'call_kwargs', False)
            # Processing the call keyword arguments (line 265)
            kwargs_2311 = {}
            # Getting the type of 'str' (line 265)
            str_2309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 78), 'str', False)
            # Calling str(args, kwargs) (line 265)
            str_call_result_2312 = invoke(stypy.reporting.localization.Localization(__file__, 265, 78), str_2309, *[call_kwargs_2310], **kwargs_2311)
            
            # Processing the call keyword arguments (line 264)
            kwargs_2313 = {}
            str_2303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 25), 'str', '{0} got unexpected keyword arguments: {1}; ')
            # Obtaining the member 'format' of a type (line 264)
            format_2304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 25), str_2303, 'format')
            # Calling format(args, kwargs) (line 264)
            format_call_result_2314 = invoke(stypy.reporting.localization.Localization(__file__, 264, 25), format_2304, *[format_function_name_call_result_2308, str_call_result_2312], **kwargs_2313)
            
            # Applying the binary operator '+=' (line 264)
            result_iadd_2315 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 12), '+=', error_msg_2302, format_call_result_2314)
            # Assigning a type to the variable 'error_msg' (line 264)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'error_msg', result_iadd_2315)
            
            # SSA join for if statement (line 262)
            module_type_store = module_type_store.join_ssa_context()
            


        if (may_be_2254 and more_types_in_union_2255):
            # SSA join for if statement (line 250)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'found_error' (line 268)
    found_error_2316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 7), 'found_error')
    # Testing if the type of an if condition is none (line 268)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 268, 4), found_error_2316):
        pass
    else:
        
        # Testing the type of an if condition (line 268)
        if_condition_2317 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 268, 4), found_error_2316)
        # Assigning a type to the variable 'if_condition_2317' (line 268)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'if_condition_2317', if_condition_2317)
        # SSA begins for if statement (line 268)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 269)
        # Processing the call arguments (line 269)
        # Getting the type of 'localization' (line 269)
        localization_2319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 25), 'localization', False)
        # Getting the type of 'error_msg' (line 269)
        error_msg_2320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 39), 'error_msg', False)
        # Processing the call keyword arguments (line 269)
        kwargs_2321 = {}
        # Getting the type of 'TypeError' (line 269)
        TypeError_2318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 269)
        TypeError_call_result_2322 = invoke(stypy.reporting.localization.Localization(__file__, 269, 15), TypeError_2318, *[localization_2319, error_msg_2320], **kwargs_2321)
        
        # Assigning a type to the variable 'stypy_return_type' (line 269)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'stypy_return_type', TypeError_call_result_2322)
        # SSA join for if statement (line 268)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'tuple' (line 271)
    tuple_2323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 271)
    # Adding element type (line 271)
    # Getting the type of 'call_type_of_args' (line 271)
    call_type_of_args_2324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 11), 'call_type_of_args')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 11), tuple_2323, call_type_of_args_2324)
    # Adding element type (line 271)
    # Getting the type of 'call_varargs' (line 271)
    call_varargs_2325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 30), 'call_varargs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 11), tuple_2323, call_varargs_2325)
    # Adding element type (line 271)
    # Getting the type of 'call_kwargs' (line 271)
    call_kwargs_2326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 44), 'call_kwargs')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 11), tuple_2323, call_kwargs_2326)
    
    # Assigning a type to the variable 'stypy_return_type' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'stypy_return_type', tuple_2323)
    
    # ################# End of 'process_argument_values(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'process_argument_values' in the type store
    # Getting the type of 'stypy_return_type' (line 119)
    stypy_return_type_2327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2327)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'process_argument_values'
    return stypy_return_type_2327

# Assigning a type to the variable 'process_argument_values' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'process_argument_values', process_argument_values)

@norecursion
def create_call_to_type_inference_code(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Call to list(...): (line 274)
    # Processing the call keyword arguments (line 274)
    kwargs_2329 = {}
    # Getting the type of 'list' (line 274)
    list_2328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 68), 'list', False)
    # Calling list(args, kwargs) (line 274)
    list_call_result_2330 = invoke(stypy.reporting.localization.Localization(__file__, 274, 68), list_2328, *[], **kwargs_2329)
    
    # Getting the type of 'None' (line 274)
    None_2331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 83), 'None')
    # Getting the type of 'None' (line 274)
    None_2332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 98), 'None')
    int_2333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 109), 'int')
    int_2334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 46), 'int')
    defaults = [list_call_result_2330, None_2331, None_2332, int_2333, int_2334]
    # Create a new context for function 'create_call_to_type_inference_code'
    module_type_store = module_type_store.open_function_context('create_call_to_type_inference_code', 274, 0, False)
    
    # Passed parameters checking function
    create_call_to_type_inference_code.stypy_localization = localization
    create_call_to_type_inference_code.stypy_type_of_self = None
    create_call_to_type_inference_code.stypy_type_store = module_type_store
    create_call_to_type_inference_code.stypy_function_name = 'create_call_to_type_inference_code'
    create_call_to_type_inference_code.stypy_param_names_list = ['func', 'localization', 'keywords', 'kwargs', 'starargs', 'line', 'column']
    create_call_to_type_inference_code.stypy_varargs_param_name = None
    create_call_to_type_inference_code.stypy_kwargs_param_name = None
    create_call_to_type_inference_code.stypy_call_defaults = defaults
    create_call_to_type_inference_code.stypy_call_varargs = varargs
    create_call_to_type_inference_code.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_call_to_type_inference_code', ['func', 'localization', 'keywords', 'kwargs', 'starargs', 'line', 'column'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_call_to_type_inference_code', localization, ['func', 'localization', 'keywords', 'kwargs', 'starargs', 'line', 'column'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_call_to_type_inference_code(...)' code ##################

    str_2335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, (-1)), 'str', '\n    Create the necessary Python code to call a function that performs the type inference of an existing function.\n     Basically it calls the invoke method of the TypeInferenceProxy that represent the callable code, creating\n     the *args and **kwargs call parameters we mentioned before.\n    :param func: Function name to call\n    :param localization: Caller information\n    :param keywords: Unused. May be removed TODO\n    :param kwargs: keyword dictionary\n    :param starargs: variable argument list\n    :param line: Source line when this call is produced\n    :param column: Source column when this call is produced\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 289):
    
    # Assigning a Call to a Name (line 289):
    
    # Call to Call(...): (line 289)
    # Processing the call keyword arguments (line 289)
    kwargs_2338 = {}
    # Getting the type of 'ast' (line 289)
    ast_2336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 11), 'ast', False)
    # Obtaining the member 'Call' of a type (line 289)
    Call_2337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 11), ast_2336, 'Call')
    # Calling Call(args, kwargs) (line 289)
    Call_call_result_2339 = invoke(stypy.reporting.localization.Localization(__file__, 289, 11), Call_2337, *[], **kwargs_2338)
    
    # Assigning a type to the variable 'call' (line 289)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'call', Call_call_result_2339)
    
    # Type idiom detected: calculating its left and rigth part (line 292)
    # Getting the type of 'func' (line 292)
    func_2340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'func')
    # Getting the type of 'tuple' (line 292)
    tuple_2341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 21), 'tuple')
    
    (may_be_2342, more_types_in_union_2343) = may_be_type(func_2340, tuple_2341)

    if may_be_2342:

        if more_types_in_union_2343:
            # Runtime conditional SSA (line 292)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'func' (line 292)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'func', tuple_2341())
        
        # Assigning a Call to a Name (line 293):
        
        # Assigning a Call to a Name (line 293):
        
        # Call to Tuple(...): (line 293)
        # Processing the call keyword arguments (line 293)
        kwargs_2346 = {}
        # Getting the type of 'ast' (line 293)
        ast_2344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 21), 'ast', False)
        # Obtaining the member 'Tuple' of a type (line 293)
        Tuple_2345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 21), ast_2344, 'Tuple')
        # Calling Tuple(args, kwargs) (line 293)
        Tuple_call_result_2347 = invoke(stypy.reporting.localization.Localization(__file__, 293, 21), Tuple_2345, *[], **kwargs_2346)
        
        # Assigning a type to the variable 'tuple_node' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'tuple_node', Tuple_call_result_2347)
        
        # Assigning a Call to a Attribute (line 294):
        
        # Assigning a Call to a Attribute (line 294):
        
        # Call to list(...): (line 294)
        # Processing the call arguments (line 294)
        # Getting the type of 'func' (line 294)
        func_2349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 31), 'func', False)
        # Processing the call keyword arguments (line 294)
        kwargs_2350 = {}
        # Getting the type of 'list' (line 294)
        list_2348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 26), 'list', False)
        # Calling list(args, kwargs) (line 294)
        list_call_result_2351 = invoke(stypy.reporting.localization.Localization(__file__, 294, 26), list_2348, *[func_2349], **kwargs_2350)
        
        # Getting the type of 'tuple_node' (line 294)
        tuple_node_2352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 8), 'tuple_node')
        # Setting the type of the member 'elts' of a type (line 294)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 8), tuple_node_2352, 'elts', list_call_result_2351)
        
        # Assigning a Name to a Name (line 295):
        
        # Assigning a Name to a Name (line 295):
        # Getting the type of 'tuple_node' (line 295)
        tuple_node_2353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 15), 'tuple_node')
        # Assigning a type to the variable 'func' (line 295)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'func', tuple_node_2353)

        if more_types_in_union_2343:
            # SSA join for if statement (line 292)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a List to a Name (line 298):
    
    # Assigning a List to a Name (line 298):
    
    # Obtaining an instance of the builtin type 'list' (line 298)
    list_2354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 298)
    # Adding element type (line 298)
    # Getting the type of 'localization' (line 298)
    localization_2355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 15), 'localization')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 298, 14), list_2354, localization_2355)
    
    # Assigning a type to the variable 'ti_args' (line 298)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 298, 4), 'ti_args', list_2354)
    
    # Assigning a Call to a Attribute (line 301):
    
    # Assigning a Call to a Attribute (line 301):
    
    # Call to create_attribute(...): (line 301)
    # Processing the call arguments (line 301)
    # Getting the type of 'func' (line 301)
    func_2358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 52), 'func', False)
    str_2359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 58), 'str', 'invoke')
    # Processing the call keyword arguments (line 301)
    kwargs_2360 = {}
    # Getting the type of 'core_language_copy' (line 301)
    core_language_copy_2356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 16), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 301)
    create_attribute_2357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 16), core_language_copy_2356, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 301)
    create_attribute_call_result_2361 = invoke(stypy.reporting.localization.Localization(__file__, 301, 16), create_attribute_2357, *[func_2358, str_2359], **kwargs_2360)
    
    # Getting the type of 'call' (line 301)
    call_2362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'call')
    # Setting the type of the member 'func' of a type (line 301)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 4), call_2362, 'func', create_attribute_call_result_2361)
    
    # Assigning a Name to a Attribute (line 302):
    
    # Assigning a Name to a Attribute (line 302):
    # Getting the type of 'line' (line 302)
    line_2363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 18), 'line')
    # Getting the type of 'call' (line 302)
    call_2364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'call')
    # Setting the type of the member 'lineno' of a type (line 302)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 4), call_2364, 'lineno', line_2363)
    
    # Assigning a Name to a Attribute (line 303):
    
    # Assigning a Name to a Attribute (line 303):
    # Getting the type of 'column' (line 303)
    column_2365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 22), 'column')
    # Getting the type of 'call' (line 303)
    call_2366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'call')
    # Setting the type of the member 'col_offset' of a type (line 303)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 4), call_2366, 'col_offset', column_2365)
    
    # Type idiom detected: calculating its left and rigth part (line 306)
    # Getting the type of 'starargs' (line 306)
    starargs_2367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 7), 'starargs')
    # Getting the type of 'None' (line 306)
    None_2368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'None')
    
    (may_be_2369, more_types_in_union_2370) = may_be_none(starargs_2367, None_2368)

    if may_be_2369:

        if more_types_in_union_2370:
            # Runtime conditional SSA (line 306)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Attribute (line 307):
        
        # Assigning a Call to a Attribute (line 307):
        
        # Call to create_list(...): (line 307)
        # Processing the call arguments (line 307)
        
        # Obtaining an instance of the builtin type 'list' (line 307)
        list_2373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 57), 'list')
        # Adding type elements to the builtin type 'list' instance (line 307)
        
        # Processing the call keyword arguments (line 307)
        kwargs_2374 = {}
        # Getting the type of 'data_structures_copy' (line 307)
        data_structures_copy_2371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 24), 'data_structures_copy', False)
        # Obtaining the member 'create_list' of a type (line 307)
        create_list_2372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 24), data_structures_copy_2371, 'create_list')
        # Calling create_list(args, kwargs) (line 307)
        create_list_call_result_2375 = invoke(stypy.reporting.localization.Localization(__file__, 307, 24), create_list_2372, *[list_2373], **kwargs_2374)
        
        # Getting the type of 'call' (line 307)
        call_2376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'call')
        # Setting the type of the member 'starargs' of a type (line 307)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 8), call_2376, 'starargs', create_list_call_result_2375)

        if more_types_in_union_2370:
            # Runtime conditional SSA for else branch (line 306)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_2369) or more_types_in_union_2370):
        
        # Assigning a Call to a Attribute (line 309):
        
        # Assigning a Call to a Attribute (line 309):
        
        # Call to create_list(...): (line 309)
        # Processing the call arguments (line 309)
        # Getting the type of 'starargs' (line 309)
        starargs_2379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 57), 'starargs', False)
        # Processing the call keyword arguments (line 309)
        kwargs_2380 = {}
        # Getting the type of 'data_structures_copy' (line 309)
        data_structures_copy_2377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 24), 'data_structures_copy', False)
        # Obtaining the member 'create_list' of a type (line 309)
        create_list_2378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 24), data_structures_copy_2377, 'create_list')
        # Calling create_list(args, kwargs) (line 309)
        create_list_call_result_2381 = invoke(stypy.reporting.localization.Localization(__file__, 309, 24), create_list_2378, *[starargs_2379], **kwargs_2380)
        
        # Getting the type of 'call' (line 309)
        call_2382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 8), 'call')
        # Setting the type of the member 'starargs' of a type (line 309)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 8), call_2382, 'starargs', create_list_call_result_2381)

        if (may_be_2369 and more_types_in_union_2370):
            # SSA join for if statement (line 306)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 312)
    # Getting the type of 'kwargs' (line 312)
    kwargs_2383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 7), 'kwargs')
    # Getting the type of 'None' (line 312)
    None_2384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 17), 'None')
    
    (may_be_2385, more_types_in_union_2386) = may_be_none(kwargs_2383, None_2384)

    if may_be_2385:

        if more_types_in_union_2386:
            # Runtime conditional SSA (line 312)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Attribute (line 313):
        
        # Assigning a Call to a Attribute (line 313):
        
        # Call to create_keyword_dict(...): (line 313)
        # Processing the call arguments (line 313)
        # Getting the type of 'None' (line 313)
        None_2389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 63), 'None', False)
        # Processing the call keyword arguments (line 313)
        kwargs_2390 = {}
        # Getting the type of 'data_structures_copy' (line 313)
        data_structures_copy_2387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 22), 'data_structures_copy', False)
        # Obtaining the member 'create_keyword_dict' of a type (line 313)
        create_keyword_dict_2388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 22), data_structures_copy_2387, 'create_keyword_dict')
        # Calling create_keyword_dict(args, kwargs) (line 313)
        create_keyword_dict_call_result_2391 = invoke(stypy.reporting.localization.Localization(__file__, 313, 22), create_keyword_dict_2388, *[None_2389], **kwargs_2390)
        
        # Getting the type of 'call' (line 313)
        call_2392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'call')
        # Setting the type of the member 'kwargs' of a type (line 313)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), call_2392, 'kwargs', create_keyword_dict_call_result_2391)

        if more_types_in_union_2386:
            # Runtime conditional SSA for else branch (line 312)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_2385) or more_types_in_union_2386):
        
        # Assigning a Name to a Attribute (line 315):
        
        # Assigning a Name to a Attribute (line 315):
        # Getting the type of 'kwargs' (line 315)
        kwargs_2393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 22), 'kwargs')
        # Getting the type of 'call' (line 315)
        call_2394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 8), 'call')
        # Setting the type of the member 'kwargs' of a type (line 315)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 315, 8), call_2394, 'kwargs', kwargs_2393)

        if (may_be_2385 and more_types_in_union_2386):
            # SSA join for if statement (line 312)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a List to a Attribute (line 317):
    
    # Assigning a List to a Attribute (line 317):
    
    # Obtaining an instance of the builtin type 'list' (line 317)
    list_2395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 317)
    
    # Getting the type of 'call' (line 317)
    call_2396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 4), 'call')
    # Setting the type of the member 'keywords' of a type (line 317)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 4), call_2396, 'keywords', list_2395)
    
    # Assigning a Name to a Attribute (line 320):
    
    # Assigning a Name to a Attribute (line 320):
    # Getting the type of 'ti_args' (line 320)
    ti_args_2397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 16), 'ti_args')
    # Getting the type of 'call' (line 320)
    call_2398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'call')
    # Setting the type of the member 'args' of a type (line 320)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 4), call_2398, 'args', ti_args_2397)
    # Getting the type of 'call' (line 323)
    call_2399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 11), 'call')
    # Assigning a type to the variable 'stypy_return_type' (line 323)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'stypy_return_type', call_2399)
    
    # ################# End of 'create_call_to_type_inference_code(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_call_to_type_inference_code' in the type store
    # Getting the type of 'stypy_return_type' (line 274)
    stypy_return_type_2400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2400)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_call_to_type_inference_code'
    return stypy_return_type_2400

# Assigning a type to the variable 'create_call_to_type_inference_code' (line 274)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 0), 'create_call_to_type_inference_code', create_call_to_type_inference_code)

@norecursion
def is_suitable_condition(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_suitable_condition'
    module_type_store = module_type_store.open_function_context('is_suitable_condition', 329, 0, False)
    
    # Passed parameters checking function
    is_suitable_condition.stypy_localization = localization
    is_suitable_condition.stypy_type_of_self = None
    is_suitable_condition.stypy_type_store = module_type_store
    is_suitable_condition.stypy_function_name = 'is_suitable_condition'
    is_suitable_condition.stypy_param_names_list = ['localization', 'condition_type']
    is_suitable_condition.stypy_varargs_param_name = None
    is_suitable_condition.stypy_kwargs_param_name = None
    is_suitable_condition.stypy_call_defaults = defaults
    is_suitable_condition.stypy_call_varargs = varargs
    is_suitable_condition.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_suitable_condition', ['localization', 'condition_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_suitable_condition', localization, ['localization', 'condition_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_suitable_condition(...)' code ##################

    str_2401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, (-1)), 'str', '\n    Checks if the type of a condition is suitable. Only checks if the type of a condition is an error, except if\n    coding advices is enabled. In that case a warning is issued if the condition is not bool.\n    :param localization: Caller information\n    :param condition_type: Type of the condition\n    :return:\n    ')
    
    # Call to is_error_type(...): (line 337)
    # Processing the call arguments (line 337)
    # Getting the type of 'condition_type' (line 337)
    condition_type_2403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 21), 'condition_type', False)
    # Processing the call keyword arguments (line 337)
    kwargs_2404 = {}
    # Getting the type of 'is_error_type' (line 337)
    is_error_type_2402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 7), 'is_error_type', False)
    # Calling is_error_type(args, kwargs) (line 337)
    is_error_type_call_result_2405 = invoke(stypy.reporting.localization.Localization(__file__, 337, 7), is_error_type_2402, *[condition_type_2403], **kwargs_2404)
    
    # Testing if the type of an if condition is none (line 337)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 337, 4), is_error_type_call_result_2405):
        pass
    else:
        
        # Testing the type of an if condition (line 337)
        if_condition_2406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 337, 4), is_error_type_call_result_2405)
        # Assigning a type to the variable 'if_condition_2406' (line 337)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 4), 'if_condition_2406', if_condition_2406)
        # SSA begins for if statement (line 337)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'localization' (line 338)
        localization_2408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 18), 'localization', False)
        str_2409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 32), 'str', 'The type of this condition is erroneous')
        # Processing the call keyword arguments (line 338)
        kwargs_2410 = {}
        # Getting the type of 'TypeError' (line 338)
        TypeError_2407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 338)
        TypeError_call_result_2411 = invoke(stypy.reporting.localization.Localization(__file__, 338, 8), TypeError_2407, *[localization_2408, str_2409], **kwargs_2410)
        
        # Getting the type of 'False' (line 339)
        False_2412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'stypy_return_type', False_2412)
        # SSA join for if statement (line 337)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'ENABLE_CODING_ADVICES' (line 341)
    ENABLE_CODING_ADVICES_2413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 341, 7), 'ENABLE_CODING_ADVICES')
    # Testing if the type of an if condition is none (line 341)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 341, 4), ENABLE_CODING_ADVICES_2413):
        pass
    else:
        
        # Testing the type of an if condition (line 341)
        if_condition_2414 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 341, 4), ENABLE_CODING_ADVICES_2413)
        # Assigning a type to the variable 'if_condition_2414' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 4), 'if_condition_2414', if_condition_2414)
        # SSA begins for if statement (line 341)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        
        # Call to get_python_type(...): (line 342)
        # Processing the call keyword arguments (line 342)
        kwargs_2417 = {}
        # Getting the type of 'condition_type' (line 342)
        condition_type_2415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 15), 'condition_type', False)
        # Obtaining the member 'get_python_type' of a type (line 342)
        get_python_type_2416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 15), condition_type_2415, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 342)
        get_python_type_call_result_2418 = invoke(stypy.reporting.localization.Localization(__file__, 342, 15), get_python_type_2416, *[], **kwargs_2417)
        
        # Getting the type of 'bool' (line 342)
        bool_2419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 51), 'bool')
        # Applying the binary operator 'is' (line 342)
        result_is__2420 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 15), 'is', get_python_type_call_result_2418, bool_2419)
        
        # Applying the 'not' unary operator (line 342)
        result_not__2421 = python_operator(stypy.reporting.localization.Localization(__file__, 342, 11), 'not', result_is__2420)
        
        # Testing if the type of an if condition is none (line 342)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 342, 8), result_not__2421):
            pass
        else:
            
            # Testing the type of an if condition (line 342)
            if_condition_2422 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 342, 8), result_not__2421)
            # Assigning a type to the variable 'if_condition_2422' (line 342)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'if_condition_2422', if_condition_2422)
            # SSA begins for if statement (line 342)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to instance(...): (line 343)
            # Processing the call arguments (line 343)
            # Getting the type of 'localization' (line 343)
            localization_2425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 33), 'localization', False)
            
            # Call to format(...): (line 344)
            # Processing the call arguments (line 344)
            # Getting the type of 'condition_type' (line 345)
            condition_type_2428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 40), 'condition_type', False)
            # Processing the call keyword arguments (line 344)
            kwargs_2429 = {}
            str_2426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 344, 33), 'str', 'The type of this condition is not boolean ({0}). Is that what you really intend?')
            # Obtaining the member 'format' of a type (line 344)
            format_2427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 33), str_2426, 'format')
            # Calling format(args, kwargs) (line 344)
            format_call_result_2430 = invoke(stypy.reporting.localization.Localization(__file__, 344, 33), format_2427, *[condition_type_2428], **kwargs_2429)
            
            # Processing the call keyword arguments (line 343)
            kwargs_2431 = {}
            # Getting the type of 'TypeWarning' (line 343)
            TypeWarning_2423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 12), 'TypeWarning', False)
            # Obtaining the member 'instance' of a type (line 343)
            instance_2424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 343, 12), TypeWarning_2423, 'instance')
            # Calling instance(args, kwargs) (line 343)
            instance_call_result_2432 = invoke(stypy.reporting.localization.Localization(__file__, 343, 12), instance_2424, *[localization_2425, format_call_result_2430], **kwargs_2431)
            
            # SSA join for if statement (line 342)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 341)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'True' (line 347)
    True_2433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'stypy_return_type', True_2433)
    
    # ################# End of 'is_suitable_condition(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_suitable_condition' in the type store
    # Getting the type of 'stypy_return_type' (line 329)
    stypy_return_type_2434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2434)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_suitable_condition'
    return stypy_return_type_2434

# Assigning a type to the variable 'is_suitable_condition' (line 329)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 0), 'is_suitable_condition', is_suitable_condition)

@norecursion
def is_error_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_error_type'
    module_type_store = module_type_store.open_function_context('is_error_type', 350, 0, False)
    
    # Passed parameters checking function
    is_error_type.stypy_localization = localization
    is_error_type.stypy_type_of_self = None
    is_error_type.stypy_type_store = module_type_store
    is_error_type.stypy_function_name = 'is_error_type'
    is_error_type.stypy_param_names_list = ['type_']
    is_error_type.stypy_varargs_param_name = None
    is_error_type.stypy_kwargs_param_name = None
    is_error_type.stypy_call_defaults = defaults
    is_error_type.stypy_call_varargs = varargs
    is_error_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_error_type', ['type_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_error_type', localization, ['type_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_error_type(...)' code ##################

    str_2435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, (-1)), 'str', '\n    Tells if the passed type represent some kind of error\n    :param type_: Passed type\n    :return: bool value\n    ')
    
    # Call to isinstance(...): (line 356)
    # Processing the call arguments (line 356)
    # Getting the type of 'type_' (line 356)
    type__2437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 22), 'type_', False)
    # Getting the type of 'TypeError' (line 356)
    TypeError_2438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 29), 'TypeError', False)
    # Processing the call keyword arguments (line 356)
    kwargs_2439 = {}
    # Getting the type of 'isinstance' (line 356)
    isinstance_2436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 356)
    isinstance_call_result_2440 = invoke(stypy.reporting.localization.Localization(__file__, 356, 11), isinstance_2436, *[type__2437, TypeError_2438], **kwargs_2439)
    
    # Assigning a type to the variable 'stypy_return_type' (line 356)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 356, 4), 'stypy_return_type', isinstance_call_result_2440)
    
    # ################# End of 'is_error_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_error_type' in the type store
    # Getting the type of 'stypy_return_type' (line 350)
    stypy_return_type_2441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 350, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2441)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_error_type'
    return stypy_return_type_2441

# Assigning a type to the variable 'is_error_type' (line 350)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 0), 'is_error_type', is_error_type)

@norecursion
def is_suitable_for_loop_condition(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_suitable_for_loop_condition'
    module_type_store = module_type_store.open_function_context('is_suitable_for_loop_condition', 359, 0, False)
    
    # Passed parameters checking function
    is_suitable_for_loop_condition.stypy_localization = localization
    is_suitable_for_loop_condition.stypy_type_of_self = None
    is_suitable_for_loop_condition.stypy_type_store = module_type_store
    is_suitable_for_loop_condition.stypy_function_name = 'is_suitable_for_loop_condition'
    is_suitable_for_loop_condition.stypy_param_names_list = ['localization', 'condition_type']
    is_suitable_for_loop_condition.stypy_varargs_param_name = None
    is_suitable_for_loop_condition.stypy_kwargs_param_name = None
    is_suitable_for_loop_condition.stypy_call_defaults = defaults
    is_suitable_for_loop_condition.stypy_call_varargs = varargs
    is_suitable_for_loop_condition.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_suitable_for_loop_condition', ['localization', 'condition_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_suitable_for_loop_condition', localization, ['localization', 'condition_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_suitable_for_loop_condition(...)' code ##################

    str_2442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 365, (-1)), 'str', '\n    A loop must iterate an iterable object or data structure or an string. This function checks this fact\n    :param localization: Caller information\n    :param condition_type: Type of the condition\n    :return:\n    ')
    
    # Call to is_error_type(...): (line 366)
    # Processing the call arguments (line 366)
    # Getting the type of 'condition_type' (line 366)
    condition_type_2444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 21), 'condition_type', False)
    # Processing the call keyword arguments (line 366)
    kwargs_2445 = {}
    # Getting the type of 'is_error_type' (line 366)
    is_error_type_2443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 7), 'is_error_type', False)
    # Calling is_error_type(args, kwargs) (line 366)
    is_error_type_call_result_2446 = invoke(stypy.reporting.localization.Localization(__file__, 366, 7), is_error_type_2443, *[condition_type_2444], **kwargs_2445)
    
    # Testing if the type of an if condition is none (line 366)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 366, 4), is_error_type_call_result_2446):
        pass
    else:
        
        # Testing the type of an if condition (line 366)
        if_condition_2447 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 366, 4), is_error_type_call_result_2446)
        # Assigning a type to the variable 'if_condition_2447' (line 366)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 4), 'if_condition_2447', if_condition_2447)
        # SSA begins for if statement (line 366)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 367)
        # Processing the call arguments (line 367)
        # Getting the type of 'localization' (line 367)
        localization_2449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 18), 'localization', False)
        str_2450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 367, 32), 'str', 'The type of this for loop condition is erroneous')
        # Processing the call keyword arguments (line 367)
        kwargs_2451 = {}
        # Getting the type of 'TypeError' (line 367)
        TypeError_2448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 367)
        TypeError_call_result_2452 = invoke(stypy.reporting.localization.Localization(__file__, 367, 8), TypeError_2448, *[localization_2449, str_2450], **kwargs_2451)
        
        # Getting the type of 'False' (line 368)
        False_2453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 368)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 8), 'stypy_return_type', False_2453)
        # SSA join for if statement (line 366)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Evaluating a boolean operation
    
    # Call to can_store_elements(...): (line 370)
    # Processing the call keyword arguments (line 370)
    kwargs_2456 = {}
    # Getting the type of 'condition_type' (line 370)
    condition_type_2454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 12), 'condition_type', False)
    # Obtaining the member 'can_store_elements' of a type (line 370)
    can_store_elements_2455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 370, 12), condition_type_2454, 'can_store_elements')
    # Calling can_store_elements(args, kwargs) (line 370)
    can_store_elements_call_result_2457 = invoke(stypy.reporting.localization.Localization(__file__, 370, 12), can_store_elements_2455, *[], **kwargs_2456)
    
    
    # Getting the type of 'Str' (line 370)
    Str_2458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 52), 'Str')
    # Getting the type of 'condition_type' (line 370)
    condition_type_2459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 59), 'condition_type')
    # Applying the binary operator '==' (line 370)
    result_eq_2460 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 52), '==', Str_2458, condition_type_2459)
    
    # Applying the binary operator 'or' (line 370)
    result_or_keyword_2461 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 12), 'or', can_store_elements_call_result_2457, result_eq_2460)
    
    # Getting the type of 'IterableObject' (line 370)
    IterableObject_2462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 79), 'IterableObject')
    # Getting the type of 'condition_type' (line 370)
    condition_type_2463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 370, 97), 'condition_type')
    # Applying the binary operator '==' (line 370)
    result_eq_2464 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 79), '==', IterableObject_2462, condition_type_2463)
    
    # Applying the binary operator 'or' (line 370)
    result_or_keyword_2465 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 12), 'or', result_or_keyword_2461, result_eq_2464)
    
    # Applying the 'not' unary operator (line 370)
    result_not__2466 = python_operator(stypy.reporting.localization.Localization(__file__, 370, 7), 'not', result_or_keyword_2465)
    
    # Testing if the type of an if condition is none (line 370)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 370, 4), result_not__2466):
        pass
    else:
        
        # Testing the type of an if condition (line 370)
        if_condition_2467 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 370, 4), result_not__2466)
        # Assigning a type to the variable 'if_condition_2467' (line 370)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 370, 4), 'if_condition_2467', if_condition_2467)
        # SSA begins for if statement (line 370)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 371)
        # Processing the call arguments (line 371)
        # Getting the type of 'localization' (line 371)
        localization_2469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 18), 'localization', False)
        str_2470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 371, 32), 'str', 'The type of this for loop condition is erroneous')
        # Processing the call keyword arguments (line 371)
        kwargs_2471 = {}
        # Getting the type of 'TypeError' (line 371)
        TypeError_2468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 371)
        TypeError_call_result_2472 = invoke(stypy.reporting.localization.Localization(__file__, 371, 8), TypeError_2468, *[localization_2469, str_2470], **kwargs_2471)
        
        # Getting the type of 'False' (line 372)
        False_2473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 372, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 372)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 372, 8), 'stypy_return_type', False_2473)
        # SSA join for if statement (line 370)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'True' (line 374)
    True_2474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 374, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 374)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 374, 4), 'stypy_return_type', True_2474)
    
    # ################# End of 'is_suitable_for_loop_condition(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_suitable_for_loop_condition' in the type store
    # Getting the type of 'stypy_return_type' (line 359)
    stypy_return_type_2475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2475)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_suitable_for_loop_condition'
    return stypy_return_type_2475

# Assigning a type to the variable 'is_suitable_for_loop_condition' (line 359)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 359, 0), 'is_suitable_for_loop_condition', is_suitable_for_loop_condition)

@norecursion
def get_type_of_for_loop_variable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_type_of_for_loop_variable'
    module_type_store = module_type_store.open_function_context('get_type_of_for_loop_variable', 377, 0, False)
    
    # Passed parameters checking function
    get_type_of_for_loop_variable.stypy_localization = localization
    get_type_of_for_loop_variable.stypy_type_of_self = None
    get_type_of_for_loop_variable.stypy_type_store = module_type_store
    get_type_of_for_loop_variable.stypy_function_name = 'get_type_of_for_loop_variable'
    get_type_of_for_loop_variable.stypy_param_names_list = ['localization', 'condition_type']
    get_type_of_for_loop_variable.stypy_varargs_param_name = None
    get_type_of_for_loop_variable.stypy_kwargs_param_name = None
    get_type_of_for_loop_variable.stypy_call_defaults = defaults
    get_type_of_for_loop_variable.stypy_call_varargs = varargs
    get_type_of_for_loop_variable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_type_of_for_loop_variable', ['localization', 'condition_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_type_of_for_loop_variable', localization, ['localization', 'condition_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_type_of_for_loop_variable(...)' code ##################

    str_2476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 384, (-1)), 'str', '\n    A loop must iterate an iterable object or data structure or an string. This function returns the contents of\n    whatever the loop is iterating\n    :param localization: Caller information\n    :param condition_type: Type of the condition\n    :return:\n    ')
    
    # Evaluating a boolean operation
    
    # Call to can_store_elements(...): (line 387)
    # Processing the call keyword arguments (line 387)
    kwargs_2479 = {}
    # Getting the type of 'condition_type' (line 387)
    condition_type_2477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 7), 'condition_type', False)
    # Obtaining the member 'can_store_elements' of a type (line 387)
    can_store_elements_2478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 7), condition_type_2477, 'can_store_elements')
    # Calling can_store_elements(args, kwargs) (line 387)
    can_store_elements_call_result_2480 = invoke(stypy.reporting.localization.Localization(__file__, 387, 7), can_store_elements_2478, *[], **kwargs_2479)
    
    
    # Call to is_type_instance(...): (line 387)
    # Processing the call keyword arguments (line 387)
    kwargs_2483 = {}
    # Getting the type of 'condition_type' (line 387)
    condition_type_2481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 387, 47), 'condition_type', False)
    # Obtaining the member 'is_type_instance' of a type (line 387)
    is_type_instance_2482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 387, 47), condition_type_2481, 'is_type_instance')
    # Calling is_type_instance(args, kwargs) (line 387)
    is_type_instance_call_result_2484 = invoke(stypy.reporting.localization.Localization(__file__, 387, 47), is_type_instance_2482, *[], **kwargs_2483)
    
    # Applying the binary operator 'and' (line 387)
    result_and_keyword_2485 = python_operator(stypy.reporting.localization.Localization(__file__, 387, 7), 'and', can_store_elements_call_result_2480, is_type_instance_call_result_2484)
    
    # Testing if the type of an if condition is none (line 387)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 387, 4), result_and_keyword_2485):
        pass
    else:
        
        # Testing the type of an if condition (line 387)
        if_condition_2486 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 387, 4), result_and_keyword_2485)
        # Assigning a type to the variable 'if_condition_2486' (line 387)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 387, 4), 'if_condition_2486', if_condition_2486)
        # SSA begins for if statement (line 387)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to get_elements_type(...): (line 388)
        # Processing the call keyword arguments (line 388)
        kwargs_2489 = {}
        # Getting the type of 'condition_type' (line 388)
        condition_type_2487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 388, 15), 'condition_type', False)
        # Obtaining the member 'get_elements_type' of a type (line 388)
        get_elements_type_2488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 388, 15), condition_type_2487, 'get_elements_type')
        # Calling get_elements_type(args, kwargs) (line 388)
        get_elements_type_call_result_2490 = invoke(stypy.reporting.localization.Localization(__file__, 388, 15), get_elements_type_2488, *[], **kwargs_2489)
        
        # Assigning a type to the variable 'stypy_return_type' (line 388)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 388, 8), 'stypy_return_type', get_elements_type_call_result_2490)
        # SSA join for if statement (line 387)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    
    # Getting the type of 'Str' (line 391)
    Str_2491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 7), 'Str')
    # Getting the type of 'condition_type' (line 391)
    condition_type_2492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 14), 'condition_type')
    # Applying the binary operator '==' (line 391)
    result_eq_2493 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 7), '==', Str_2491, condition_type_2492)
    
    
    # Call to is_type_instance(...): (line 391)
    # Processing the call keyword arguments (line 391)
    kwargs_2496 = {}
    # Getting the type of 'condition_type' (line 391)
    condition_type_2494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 391, 33), 'condition_type', False)
    # Obtaining the member 'is_type_instance' of a type (line 391)
    is_type_instance_2495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 391, 33), condition_type_2494, 'is_type_instance')
    # Calling is_type_instance(args, kwargs) (line 391)
    is_type_instance_call_result_2497 = invoke(stypy.reporting.localization.Localization(__file__, 391, 33), is_type_instance_2495, *[], **kwargs_2496)
    
    # Applying the binary operator 'and' (line 391)
    result_and_keyword_2498 = python_operator(stypy.reporting.localization.Localization(__file__, 391, 7), 'and', result_eq_2493, is_type_instance_call_result_2497)
    
    # Testing if the type of an if condition is none (line 391)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 391, 4), result_and_keyword_2498):
        pass
    else:
        
        # Testing the type of an if condition (line 391)
        if_condition_2499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 391, 4), result_and_keyword_2498)
        # Assigning a type to the variable 'if_condition_2499' (line 391)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 391, 4), 'if_condition_2499', if_condition_2499)
        # SSA begins for if statement (line 391)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to get_python_type(...): (line 392)
        # Processing the call keyword arguments (line 392)
        kwargs_2502 = {}
        # Getting the type of 'condition_type' (line 392)
        condition_type_2500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 392, 15), 'condition_type', False)
        # Obtaining the member 'get_python_type' of a type (line 392)
        get_python_type_2501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 392, 15), condition_type_2500, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 392)
        get_python_type_call_result_2503 = invoke(stypy.reporting.localization.Localization(__file__, 392, 15), get_python_type_2501, *[], **kwargs_2502)
        
        # Assigning a type to the variable 'stypy_return_type' (line 392)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 392, 8), 'stypy_return_type', get_python_type_call_result_2503)
        # SSA join for if statement (line 391)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    
    # Getting the type of 'IterableObject' (line 395)
    IterableObject_2504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 7), 'IterableObject')
    # Getting the type of 'condition_type' (line 395)
    condition_type_2505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 25), 'condition_type')
    # Applying the binary operator '==' (line 395)
    result_eq_2506 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 7), '==', IterableObject_2504, condition_type_2505)
    
    
    # Call to is_type_instance(...): (line 395)
    # Processing the call keyword arguments (line 395)
    kwargs_2509 = {}
    # Getting the type of 'condition_type' (line 395)
    condition_type_2507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 395, 44), 'condition_type', False)
    # Obtaining the member 'is_type_instance' of a type (line 395)
    is_type_instance_2508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 395, 44), condition_type_2507, 'is_type_instance')
    # Calling is_type_instance(args, kwargs) (line 395)
    is_type_instance_call_result_2510 = invoke(stypy.reporting.localization.Localization(__file__, 395, 44), is_type_instance_2508, *[], **kwargs_2509)
    
    # Applying the binary operator 'and' (line 395)
    result_and_keyword_2511 = python_operator(stypy.reporting.localization.Localization(__file__, 395, 7), 'and', result_eq_2506, is_type_instance_call_result_2510)
    
    # Testing if the type of an if condition is none (line 395)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 395, 4), result_and_keyword_2511):
        pass
    else:
        
        # Testing the type of an if condition (line 395)
        if_condition_2512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 395, 4), result_and_keyword_2511)
        # Assigning a type to the variable 'if_condition_2512' (line 395)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 395, 4), 'if_condition_2512', if_condition_2512)
        # SSA begins for if statement (line 395)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 396):
        
        # Assigning a Call to a Name (line 396):
        
        # Call to get_type_of_member(...): (line 396)
        # Processing the call arguments (line 396)
        # Getting the type of 'localization' (line 396)
        localization_2515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 56), 'localization', False)
        str_2516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 396, 70), 'str', '__iter__')
        # Processing the call keyword arguments (line 396)
        kwargs_2517 = {}
        # Getting the type of 'condition_type' (line 396)
        condition_type_2513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 396, 22), 'condition_type', False)
        # Obtaining the member 'get_type_of_member' of a type (line 396)
        get_type_of_member_2514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 396, 22), condition_type_2513, 'get_type_of_member')
        # Calling get_type_of_member(args, kwargs) (line 396)
        get_type_of_member_call_result_2518 = invoke(stypy.reporting.localization.Localization(__file__, 396, 22), get_type_of_member_2514, *[localization_2515, str_2516], **kwargs_2517)
        
        # Assigning a type to the variable 'iter_method' (line 396)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 396, 8), 'iter_method', get_type_of_member_call_result_2518)
        
        # Call to invoke(...): (line 397)
        # Processing the call arguments (line 397)
        # Getting the type of 'localization' (line 397)
        localization_2521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 34), 'localization', False)
        # Processing the call keyword arguments (line 397)
        kwargs_2522 = {}
        # Getting the type of 'iter_method' (line 397)
        iter_method_2519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 397, 15), 'iter_method', False)
        # Obtaining the member 'invoke' of a type (line 397)
        invoke_2520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 397, 15), iter_method_2519, 'invoke')
        # Calling invoke(args, kwargs) (line 397)
        invoke_call_result_2523 = invoke(stypy.reporting.localization.Localization(__file__, 397, 15), invoke_2520, *[localization_2521], **kwargs_2522)
        
        # Assigning a type to the variable 'stypy_return_type' (line 397)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 397, 8), 'stypy_return_type', invoke_call_result_2523)
        # SSA join for if statement (line 395)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to TypeError(...): (line 399)
    # Processing the call arguments (line 399)
    # Getting the type of 'localization' (line 399)
    localization_2525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 21), 'localization', False)
    str_2526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 399, 35), 'str', 'Invalid iterable type for a loop target')
    # Processing the call keyword arguments (line 399)
    kwargs_2527 = {}
    # Getting the type of 'TypeError' (line 399)
    TypeError_2524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 399, 11), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 399)
    TypeError_call_result_2528 = invoke(stypy.reporting.localization.Localization(__file__, 399, 11), TypeError_2524, *[localization_2525, str_2526], **kwargs_2527)
    
    # Assigning a type to the variable 'stypy_return_type' (line 399)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 399, 4), 'stypy_return_type', TypeError_call_result_2528)
    
    # ################# End of 'get_type_of_for_loop_variable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_type_of_for_loop_variable' in the type store
    # Getting the type of 'stypy_return_type' (line 377)
    stypy_return_type_2529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2529)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_type_of_for_loop_variable'
    return stypy_return_type_2529

# Assigning a type to the variable 'get_type_of_for_loop_variable' (line 377)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 377, 0), 'get_type_of_for_loop_variable', get_type_of_for_loop_variable)

@norecursion
def __type_is_in_union(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__type_is_in_union'
    module_type_store = module_type_store.open_function_context('__type_is_in_union', 404, 0, False)
    
    # Passed parameters checking function
    __type_is_in_union.stypy_localization = localization
    __type_is_in_union.stypy_type_of_self = None
    __type_is_in_union.stypy_type_store = module_type_store
    __type_is_in_union.stypy_function_name = '__type_is_in_union'
    __type_is_in_union.stypy_param_names_list = ['type_list', 'expected_type']
    __type_is_in_union.stypy_varargs_param_name = None
    __type_is_in_union.stypy_kwargs_param_name = None
    __type_is_in_union.stypy_call_defaults = defaults
    __type_is_in_union.stypy_call_varargs = varargs
    __type_is_in_union.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__type_is_in_union', ['type_list', 'expected_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__type_is_in_union', localization, ['type_list', 'expected_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__type_is_in_union(...)' code ##################

    
    # Getting the type of 'type_list' (line 407)
    type_list_2530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 407, 15), 'type_list')
    # Assigning a type to the variable 'type_list_2530' (line 407)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'type_list_2530', type_list_2530)
    # Testing if the for loop is going to be iterated (line 407)
    # Testing the type of a for loop iterable (line 407)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 407, 4), type_list_2530)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 407, 4), type_list_2530):
        # Getting the type of the for loop variable (line 407)
        for_loop_var_2531 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 407, 4), type_list_2530)
        # Assigning a type to the variable 'typ' (line 407)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 407, 4), 'typ', for_loop_var_2531)
        # SSA begins for a for statement (line 407)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'typ' (line 409)
        typ_2532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 11), 'typ')
        # Getting the type of 'expected_type' (line 409)
        expected_type_2533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 409, 18), 'expected_type')
        # Applying the binary operator '==' (line 409)
        result_eq_2534 = python_operator(stypy.reporting.localization.Localization(__file__, 409, 11), '==', typ_2532, expected_type_2533)
        
        # Testing if the type of an if condition is none (line 409)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 409, 8), result_eq_2534):
            pass
        else:
            
            # Testing the type of an if condition (line 409)
            if_condition_2535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 409, 8), result_eq_2534)
            # Assigning a type to the variable 'if_condition_2535' (line 409)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 409, 8), 'if_condition_2535', if_condition_2535)
            # SSA begins for if statement (line 409)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 410)
            True_2536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 410, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 410)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 410, 12), 'stypy_return_type', True_2536)
            # SSA join for if statement (line 409)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'False' (line 412)
    False_2537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 412, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 412)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 412, 4), 'stypy_return_type', False_2537)
    
    # ################# End of '__type_is_in_union(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__type_is_in_union' in the type store
    # Getting the type of 'stypy_return_type' (line 404)
    stypy_return_type_2538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2538)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__type_is_in_union'
    return stypy_return_type_2538

# Assigning a type to the variable '__type_is_in_union' (line 404)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 404, 0), '__type_is_in_union', __type_is_in_union)

@norecursion
def may_be_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'may_be_type'
    module_type_store = module_type_store.open_function_context('may_be_type', 414, 0, False)
    
    # Passed parameters checking function
    may_be_type.stypy_localization = localization
    may_be_type.stypy_type_of_self = None
    may_be_type.stypy_type_store = module_type_store
    may_be_type.stypy_function_name = 'may_be_type'
    may_be_type.stypy_param_names_list = ['actual_type', 'expected_type']
    may_be_type.stypy_varargs_param_name = None
    may_be_type.stypy_kwargs_param_name = None
    may_be_type.stypy_call_defaults = defaults
    may_be_type.stypy_call_varargs = varargs
    may_be_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'may_be_type', ['actual_type', 'expected_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'may_be_type', localization, ['actual_type', 'expected_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'may_be_type(...)' code ##################

    str_2539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 419, (-1)), 'str', '\n    Returns:\n     1) if the actual type is the expected one, including the semantics of union types (int\\/str may be int).\n     2) It the number of types in the union type, if we suppress the actual type\n     ')
    
    # Assigning a Call to a Name (line 420):
    
    # Assigning a Call to a Name (line 420):
    
    # Call to instance(...): (line 420)
    # Processing the call arguments (line 420)
    # Getting the type of 'expected_type' (line 421)
    expected_type_2547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 421, 8), 'expected_type', False)
    # Processing the call keyword arguments (line 420)
    kwargs_2548 = {}
    # Getting the type of 'stypy_copy' (line 420)
    stypy_copy_2540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 420, 20), 'stypy_copy', False)
    # Obtaining the member 'python_lib_copy' of a type (line 420)
    python_lib_copy_2541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 20), stypy_copy_2540, 'python_lib_copy')
    # Obtaining the member 'python_types_copy' of a type (line 420)
    python_types_copy_2542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 20), python_lib_copy_2541, 'python_types_copy')
    # Obtaining the member 'type_inference_copy' of a type (line 420)
    type_inference_copy_2543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 20), python_types_copy_2542, 'type_inference_copy')
    # Obtaining the member 'type_inference_proxy_copy' of a type (line 420)
    type_inference_proxy_copy_2544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 20), type_inference_copy_2543, 'type_inference_proxy_copy')
    # Obtaining the member 'TypeInferenceProxy' of a type (line 420)
    TypeInferenceProxy_2545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 20), type_inference_proxy_copy_2544, 'TypeInferenceProxy')
    # Obtaining the member 'instance' of a type (line 420)
    instance_2546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 420, 20), TypeInferenceProxy_2545, 'instance')
    # Calling instance(args, kwargs) (line 420)
    instance_call_result_2549 = invoke(stypy.reporting.localization.Localization(__file__, 420, 20), instance_2546, *[expected_type_2547], **kwargs_2548)
    
    # Assigning a type to the variable 'expected_type' (line 420)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 420, 4), 'expected_type', instance_call_result_2549)
    
    # Call to set_type_instance(...): (line 422)
    # Processing the call arguments (line 422)
    # Getting the type of 'True' (line 422)
    True_2552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 36), 'True', False)
    # Processing the call keyword arguments (line 422)
    kwargs_2553 = {}
    # Getting the type of 'expected_type' (line 422)
    expected_type_2550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 422, 4), 'expected_type', False)
    # Obtaining the member 'set_type_instance' of a type (line 422)
    set_type_instance_2551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 422, 4), expected_type_2550, 'set_type_instance')
    # Calling set_type_instance(args, kwargs) (line 422)
    set_type_instance_call_result_2554 = invoke(stypy.reporting.localization.Localization(__file__, 422, 4), set_type_instance_2551, *[True_2552], **kwargs_2553)
    
    
    # Getting the type of 'actual_type' (line 425)
    actual_type_2555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 7), 'actual_type')
    # Getting the type of 'expected_type' (line 425)
    expected_type_2556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 425, 22), 'expected_type')
    # Applying the binary operator '==' (line 425)
    result_eq_2557 = python_operator(stypy.reporting.localization.Localization(__file__, 425, 7), '==', actual_type_2555, expected_type_2556)
    
    # Testing if the type of an if condition is none (line 425)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 425, 4), result_eq_2557):
        pass
    else:
        
        # Testing the type of an if condition (line 425)
        if_condition_2558 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 425, 4), result_eq_2557)
        # Assigning a type to the variable 'if_condition_2558' (line 425)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 425, 4), 'if_condition_2558', if_condition_2558)
        # SSA begins for if statement (line 425)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 426)
        tuple_2559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 426)
        # Adding element type (line 426)
        # Getting the type of 'True' (line 426)
        True_2560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 426, 15), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 15), tuple_2559, True_2560)
        # Adding element type (line 426)
        int_2561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 426, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 426, 15), tuple_2559, int_2561)
        
        # Assigning a type to the variable 'stypy_return_type' (line 426)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 426, 8), 'stypy_return_type', tuple_2559)
        # SSA join for if statement (line 425)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to is_union_type(...): (line 427)
    # Processing the call arguments (line 427)
    # Getting the type of 'actual_type' (line 427)
    actual_type_2568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 119), 'actual_type', False)
    # Processing the call keyword arguments (line 427)
    kwargs_2569 = {}
    # Getting the type of 'stypy_copy' (line 427)
    stypy_copy_2562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 427, 7), 'stypy_copy', False)
    # Obtaining the member 'python_lib_copy' of a type (line 427)
    python_lib_copy_2563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 7), stypy_copy_2562, 'python_lib_copy')
    # Obtaining the member 'python_types_copy' of a type (line 427)
    python_types_copy_2564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 7), python_lib_copy_2563, 'python_types_copy')
    # Obtaining the member 'type_introspection_copy' of a type (line 427)
    type_introspection_copy_2565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 7), python_types_copy_2564, 'type_introspection_copy')
    # Obtaining the member 'runtime_type_inspection_copy' of a type (line 427)
    runtime_type_inspection_copy_2566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 7), type_introspection_copy_2565, 'runtime_type_inspection_copy')
    # Obtaining the member 'is_union_type' of a type (line 427)
    is_union_type_2567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 427, 7), runtime_type_inspection_copy_2566, 'is_union_type')
    # Calling is_union_type(args, kwargs) (line 427)
    is_union_type_call_result_2570 = invoke(stypy.reporting.localization.Localization(__file__, 427, 7), is_union_type_2567, *[actual_type_2568], **kwargs_2569)
    
    # Testing if the type of an if condition is none (line 427)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 427, 4), is_union_type_call_result_2570):
        pass
    else:
        
        # Testing the type of an if condition (line 427)
        if_condition_2571 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 427, 4), is_union_type_call_result_2570)
        # Assigning a type to the variable 'if_condition_2571' (line 427)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 427, 4), 'if_condition_2571', if_condition_2571)
        # SSA begins for if statement (line 427)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 429):
        
        # Assigning a Call to a Name (line 429):
        
        # Call to __type_is_in_union(...): (line 429)
        # Processing the call arguments (line 429)
        # Getting the type of 'actual_type' (line 429)
        actual_type_2573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 46), 'actual_type', False)
        # Obtaining the member 'types' of a type (line 429)
        types_2574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 429, 46), actual_type_2573, 'types')
        # Getting the type of 'expected_type' (line 429)
        expected_type_2575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 65), 'expected_type', False)
        # Processing the call keyword arguments (line 429)
        kwargs_2576 = {}
        # Getting the type of '__type_is_in_union' (line 429)
        type_is_in_union_2572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 429, 27), '__type_is_in_union', False)
        # Calling __type_is_in_union(args, kwargs) (line 429)
        type_is_in_union_call_result_2577 = invoke(stypy.reporting.localization.Localization(__file__, 429, 27), type_is_in_union_2572, *[types_2574, expected_type_2575], **kwargs_2576)
        
        # Assigning a type to the variable 'type_is_in_union' (line 429)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 429, 8), 'type_is_in_union', type_is_in_union_call_result_2577)
        
        # Getting the type of 'type_is_in_union' (line 430)
        type_is_in_union_2578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 430, 15), 'type_is_in_union')
        # Applying the 'not' unary operator (line 430)
        result_not__2579 = python_operator(stypy.reporting.localization.Localization(__file__, 430, 11), 'not', type_is_in_union_2578)
        
        # Testing if the type of an if condition is none (line 430)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 430, 8), result_not__2579):
            pass
        else:
            
            # Testing the type of an if condition (line 430)
            if_condition_2580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 430, 8), result_not__2579)
            # Assigning a type to the variable 'if_condition_2580' (line 430)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 430, 8), 'if_condition_2580', if_condition_2580)
            # SSA begins for if statement (line 430)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'tuple' (line 431)
            tuple_2581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 431)
            # Adding element type (line 431)
            # Getting the type of 'False' (line 431)
            False_2582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 431, 19), 'False')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 19), tuple_2581, False_2582)
            # Adding element type (line 431)
            int_2583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 431, 26), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 431, 19), tuple_2581, int_2583)
            
            # Assigning a type to the variable 'stypy_return_type' (line 431)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 431, 12), 'stypy_return_type', tuple_2581)
            # SSA join for if statement (line 430)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'tuple' (line 432)
        tuple_2584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 432)
        # Adding element type (line 432)
        # Getting the type of 'True' (line 432)
        True_2585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 15), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 15), tuple_2584, True_2585)
        # Adding element type (line 432)
        
        # Call to len(...): (line 432)
        # Processing the call arguments (line 432)
        # Getting the type of 'actual_type' (line 432)
        actual_type_2587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 25), 'actual_type', False)
        # Obtaining the member 'types' of a type (line 432)
        types_2588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 432, 25), actual_type_2587, 'types')
        # Processing the call keyword arguments (line 432)
        kwargs_2589 = {}
        # Getting the type of 'len' (line 432)
        len_2586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 432, 21), 'len', False)
        # Calling len(args, kwargs) (line 432)
        len_call_result_2590 = invoke(stypy.reporting.localization.Localization(__file__, 432, 21), len_2586, *[types_2588], **kwargs_2589)
        
        int_2591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 432, 46), 'int')
        # Applying the binary operator '-' (line 432)
        result_sub_2592 = python_operator(stypy.reporting.localization.Localization(__file__, 432, 21), '-', len_call_result_2590, int_2591)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 432, 15), tuple_2584, result_sub_2592)
        
        # Assigning a type to the variable 'stypy_return_type' (line 432)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 432, 8), 'stypy_return_type', tuple_2584)
        # SSA join for if statement (line 427)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'tuple' (line 433)
    tuple_2593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 433)
    # Adding element type (line 433)
    # Getting the type of 'False' (line 433)
    False_2594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 433, 11), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 11), tuple_2593, False_2594)
    # Adding element type (line 433)
    int_2595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 433, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 433, 11), tuple_2593, int_2595)
    
    # Assigning a type to the variable 'stypy_return_type' (line 433)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 433, 4), 'stypy_return_type', tuple_2593)
    
    # ################# End of 'may_be_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'may_be_type' in the type store
    # Getting the type of 'stypy_return_type' (line 414)
    stypy_return_type_2596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 414, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2596)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'may_be_type'
    return stypy_return_type_2596

# Assigning a type to the variable 'may_be_type' (line 414)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 414, 0), 'may_be_type', may_be_type)

@norecursion
def may_not_be_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'may_not_be_type'
    module_type_store = module_type_store.open_function_context('may_not_be_type', 436, 0, False)
    
    # Passed parameters checking function
    may_not_be_type.stypy_localization = localization
    may_not_be_type.stypy_type_of_self = None
    may_not_be_type.stypy_type_store = module_type_store
    may_not_be_type.stypy_function_name = 'may_not_be_type'
    may_not_be_type.stypy_param_names_list = ['actual_type', 'expected_type']
    may_not_be_type.stypy_varargs_param_name = None
    may_not_be_type.stypy_kwargs_param_name = None
    may_not_be_type.stypy_call_defaults = defaults
    may_not_be_type.stypy_call_varargs = varargs
    may_not_be_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'may_not_be_type', ['actual_type', 'expected_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'may_not_be_type', localization, ['actual_type', 'expected_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'may_not_be_type(...)' code ##################

    str_2597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 441, (-1)), 'str', '\n    Returns:\n     1) if the actual type is not the expected one, including the semantics of union types (int\\/str may not be float).\n     2) It the number of types in the union type, if we suppress the actual type\n     ')
    
    # Assigning a Call to a Name (line 442):
    
    # Assigning a Call to a Name (line 442):
    
    # Call to instance(...): (line 442)
    # Processing the call arguments (line 442)
    # Getting the type of 'expected_type' (line 443)
    expected_type_2605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 443, 8), 'expected_type', False)
    # Processing the call keyword arguments (line 442)
    kwargs_2606 = {}
    # Getting the type of 'stypy_copy' (line 442)
    stypy_copy_2598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 442, 20), 'stypy_copy', False)
    # Obtaining the member 'python_lib_copy' of a type (line 442)
    python_lib_copy_2599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 20), stypy_copy_2598, 'python_lib_copy')
    # Obtaining the member 'python_types_copy' of a type (line 442)
    python_types_copy_2600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 20), python_lib_copy_2599, 'python_types_copy')
    # Obtaining the member 'type_inference_copy' of a type (line 442)
    type_inference_copy_2601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 20), python_types_copy_2600, 'type_inference_copy')
    # Obtaining the member 'type_inference_proxy_copy' of a type (line 442)
    type_inference_proxy_copy_2602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 20), type_inference_copy_2601, 'type_inference_proxy_copy')
    # Obtaining the member 'TypeInferenceProxy' of a type (line 442)
    TypeInferenceProxy_2603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 20), type_inference_proxy_copy_2602, 'TypeInferenceProxy')
    # Obtaining the member 'instance' of a type (line 442)
    instance_2604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 442, 20), TypeInferenceProxy_2603, 'instance')
    # Calling instance(args, kwargs) (line 442)
    instance_call_result_2607 = invoke(stypy.reporting.localization.Localization(__file__, 442, 20), instance_2604, *[expected_type_2605], **kwargs_2606)
    
    # Assigning a type to the variable 'expected_type' (line 442)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 442, 4), 'expected_type', instance_call_result_2607)
    
    # Call to set_type_instance(...): (line 444)
    # Processing the call arguments (line 444)
    # Getting the type of 'True' (line 444)
    True_2610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 36), 'True', False)
    # Processing the call keyword arguments (line 444)
    kwargs_2611 = {}
    # Getting the type of 'expected_type' (line 444)
    expected_type_2608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 444, 4), 'expected_type', False)
    # Obtaining the member 'set_type_instance' of a type (line 444)
    set_type_instance_2609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 444, 4), expected_type_2608, 'set_type_instance')
    # Calling set_type_instance(args, kwargs) (line 444)
    set_type_instance_call_result_2612 = invoke(stypy.reporting.localization.Localization(__file__, 444, 4), set_type_instance_2609, *[True_2610], **kwargs_2611)
    
    
    # Call to is_union_type(...): (line 446)
    # Processing the call arguments (line 446)
    # Getting the type of 'actual_type' (line 446)
    actual_type_2619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 119), 'actual_type', False)
    # Processing the call keyword arguments (line 446)
    kwargs_2620 = {}
    # Getting the type of 'stypy_copy' (line 446)
    stypy_copy_2613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 446, 7), 'stypy_copy', False)
    # Obtaining the member 'python_lib_copy' of a type (line 446)
    python_lib_copy_2614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 7), stypy_copy_2613, 'python_lib_copy')
    # Obtaining the member 'python_types_copy' of a type (line 446)
    python_types_copy_2615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 7), python_lib_copy_2614, 'python_types_copy')
    # Obtaining the member 'type_introspection_copy' of a type (line 446)
    type_introspection_copy_2616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 7), python_types_copy_2615, 'type_introspection_copy')
    # Obtaining the member 'runtime_type_inspection_copy' of a type (line 446)
    runtime_type_inspection_copy_2617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 7), type_introspection_copy_2616, 'runtime_type_inspection_copy')
    # Obtaining the member 'is_union_type' of a type (line 446)
    is_union_type_2618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 446, 7), runtime_type_inspection_copy_2617, 'is_union_type')
    # Calling is_union_type(args, kwargs) (line 446)
    is_union_type_call_result_2621 = invoke(stypy.reporting.localization.Localization(__file__, 446, 7), is_union_type_2618, *[actual_type_2619], **kwargs_2620)
    
    # Testing if the type of an if condition is none (line 446)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 446, 4), is_union_type_call_result_2621):
        pass
    else:
        
        # Testing the type of an if condition (line 446)
        if_condition_2622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 446, 4), is_union_type_call_result_2621)
        # Assigning a type to the variable 'if_condition_2622' (line 446)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 446, 4), 'if_condition_2622', if_condition_2622)
        # SSA begins for if statement (line 446)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 451):
        
        # Assigning a Call to a Name (line 451):
        
        # Call to __type_is_in_union(...): (line 451)
        # Processing the call arguments (line 451)
        # Getting the type of 'actual_type' (line 451)
        actual_type_2624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 35), 'actual_type', False)
        # Obtaining the member 'types' of a type (line 451)
        types_2625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 451, 35), actual_type_2624, 'types')
        # Getting the type of 'expected_type' (line 451)
        expected_type_2626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 54), 'expected_type', False)
        # Processing the call keyword arguments (line 451)
        kwargs_2627 = {}
        # Getting the type of '__type_is_in_union' (line 451)
        type_is_in_union_2623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 451, 16), '__type_is_in_union', False)
        # Calling __type_is_in_union(args, kwargs) (line 451)
        type_is_in_union_call_result_2628 = invoke(stypy.reporting.localization.Localization(__file__, 451, 16), type_is_in_union_2623, *[types_2625, expected_type_2626], **kwargs_2627)
        
        # Assigning a type to the variable 'found' (line 451)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 451, 8), 'found', type_is_in_union_call_result_2628)
        
        # Assigning a BoolOp to a Name (line 452):
        
        # Assigning a BoolOp to a Name (line 452):
        
        # Evaluating a boolean operation
        
        # Getting the type of 'found' (line 452)
        found_2629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 35), 'found')
        # Applying the 'not' unary operator (line 452)
        result_not__2630 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 31), 'not', found_2629)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'found' (line 452)
        found_2631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 45), 'found')
        
        
        # Call to len(...): (line 452)
        # Processing the call arguments (line 452)
        # Getting the type of 'actual_type' (line 452)
        actual_type_2633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 59), 'actual_type', False)
        # Obtaining the member 'types' of a type (line 452)
        types_2634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 452, 59), actual_type_2633, 'types')
        # Processing the call keyword arguments (line 452)
        kwargs_2635 = {}
        # Getting the type of 'len' (line 452)
        len_2632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 452, 55), 'len', False)
        # Calling len(args, kwargs) (line 452)
        len_call_result_2636 = invoke(stypy.reporting.localization.Localization(__file__, 452, 55), len_2632, *[types_2634], **kwargs_2635)
        
        int_2637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 452, 80), 'int')
        # Applying the binary operator '>' (line 452)
        result_gt_2638 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 55), '>', len_call_result_2636, int_2637)
        
        # Applying the binary operator 'and' (line 452)
        result_and_keyword_2639 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 45), 'and', found_2631, result_gt_2638)
        
        # Applying the binary operator 'or' (line 452)
        result_or_keyword_2640 = python_operator(stypy.reporting.localization.Localization(__file__, 452, 31), 'or', result_not__2630, result_and_keyword_2639)
        
        # Assigning a type to the variable 'type_is_not_in_union' (line 452)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 452, 8), 'type_is_not_in_union', result_or_keyword_2640)
        # Getting the type of 'type_is_not_in_union' (line 454)
        type_is_not_in_union_2641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 454, 11), 'type_is_not_in_union')
        # Testing if the type of an if condition is none (line 454)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 454, 8), type_is_not_in_union_2641):
            pass
        else:
            
            # Testing the type of an if condition (line 454)
            if_condition_2642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 454, 8), type_is_not_in_union_2641)
            # Assigning a type to the variable 'if_condition_2642' (line 454)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 454, 8), 'if_condition_2642', if_condition_2642)
            # SSA begins for if statement (line 454)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'tuple' (line 455)
            tuple_2643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 455)
            # Adding element type (line 455)
            # Getting the type of 'True' (line 455)
            True_2644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 19), 'True')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 19), tuple_2643, True_2644)
            # Adding element type (line 455)
            
            # Call to len(...): (line 455)
            # Processing the call arguments (line 455)
            # Getting the type of 'actual_type' (line 455)
            actual_type_2646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 29), 'actual_type', False)
            # Obtaining the member 'types' of a type (line 455)
            types_2647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 455, 29), actual_type_2646, 'types')
            # Processing the call keyword arguments (line 455)
            kwargs_2648 = {}
            # Getting the type of 'len' (line 455)
            len_2645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 455, 25), 'len', False)
            # Calling len(args, kwargs) (line 455)
            len_call_result_2649 = invoke(stypy.reporting.localization.Localization(__file__, 455, 25), len_2645, *[types_2647], **kwargs_2648)
            
            int_2650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 455, 50), 'int')
            # Applying the binary operator '-' (line 455)
            result_sub_2651 = python_operator(stypy.reporting.localization.Localization(__file__, 455, 25), '-', len_call_result_2649, int_2650)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 455, 19), tuple_2643, result_sub_2651)
            
            # Assigning a type to the variable 'stypy_return_type' (line 455)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 455, 12), 'stypy_return_type', tuple_2643)
            # SSA join for if statement (line 454)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'tuple' (line 457)
        tuple_2652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 457)
        # Adding element type (line 457)
        # Getting the type of 'False' (line 457)
        False_2653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 457, 15), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 15), tuple_2652, False_2653)
        # Adding element type (line 457)
        int_2654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 457, 22), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 457, 15), tuple_2652, int_2654)
        
        # Assigning a type to the variable 'stypy_return_type' (line 457)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 457, 8), 'stypy_return_type', tuple_2652)
        # SSA join for if statement (line 446)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Getting the type of 'actual_type' (line 461)
    actual_type_2655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 11), 'actual_type')
    # Getting the type of 'expected_type' (line 461)
    expected_type_2656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 461, 26), 'expected_type')
    # Applying the binary operator '==' (line 461)
    result_eq_2657 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 11), '==', actual_type_2655, expected_type_2656)
    
    # Applying the 'not' unary operator (line 461)
    result_not__2658 = python_operator(stypy.reporting.localization.Localization(__file__, 461, 7), 'not', result_eq_2657)
    
    # Testing if the type of an if condition is none (line 461)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 461, 4), result_not__2658):
        pass
    else:
        
        # Testing the type of an if condition (line 461)
        if_condition_2659 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 461, 4), result_not__2658)
        # Assigning a type to the variable 'if_condition_2659' (line 461)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 461, 4), 'if_condition_2659', if_condition_2659)
        # SSA begins for if statement (line 461)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 462)
        tuple_2660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 462)
        # Adding element type (line 462)
        # Getting the type of 'True' (line 462)
        True_2661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 462, 15), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 15), tuple_2660, True_2661)
        # Adding element type (line 462)
        int_2662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 462, 21), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 462, 15), tuple_2660, int_2662)
        
        # Assigning a type to the variable 'stypy_return_type' (line 462)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 462, 8), 'stypy_return_type', tuple_2660)
        # SSA join for if statement (line 461)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Obtaining an instance of the builtin type 'tuple' (line 464)
    tuple_2663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 464)
    # Adding element type (line 464)
    # Getting the type of 'False' (line 464)
    False_2664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 464, 11), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 11), tuple_2663, False_2664)
    # Adding element type (line 464)
    int_2665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 464, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 464, 11), tuple_2663, int_2665)
    
    # Assigning a type to the variable 'stypy_return_type' (line 464)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 464, 4), 'stypy_return_type', tuple_2663)
    
    # ################# End of 'may_not_be_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'may_not_be_type' in the type store
    # Getting the type of 'stypy_return_type' (line 436)
    stypy_return_type_2666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 436, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2666)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'may_not_be_type'
    return stypy_return_type_2666

# Assigning a type to the variable 'may_not_be_type' (line 436)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 436, 0), 'may_not_be_type', may_not_be_type)

@norecursion
def remove_type_from_union(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'remove_type_from_union'
    module_type_store = module_type_store.open_function_context('remove_type_from_union', 467, 0, False)
    
    # Passed parameters checking function
    remove_type_from_union.stypy_localization = localization
    remove_type_from_union.stypy_type_of_self = None
    remove_type_from_union.stypy_type_store = module_type_store
    remove_type_from_union.stypy_function_name = 'remove_type_from_union'
    remove_type_from_union.stypy_param_names_list = ['union_type', 'type_to_remove']
    remove_type_from_union.stypy_varargs_param_name = None
    remove_type_from_union.stypy_kwargs_param_name = None
    remove_type_from_union.stypy_call_defaults = defaults
    remove_type_from_union.stypy_call_varargs = varargs
    remove_type_from_union.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'remove_type_from_union', ['union_type', 'type_to_remove'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'remove_type_from_union', localization, ['union_type', 'type_to_remove'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'remove_type_from_union(...)' code ##################

    str_2667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 473, (-1)), 'str', '\n    Removes the specified type from the passed union type\n    :param union_type: Union type to remove from\n    :param type_to_remove: Type to remove\n    :return:\n    ')
    
    
    # Call to is_union_type(...): (line 474)
    # Processing the call arguments (line 474)
    # Getting the type of 'union_type' (line 474)
    union_type_2674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 123), 'union_type', False)
    # Processing the call keyword arguments (line 474)
    kwargs_2675 = {}
    # Getting the type of 'stypy_copy' (line 474)
    stypy_copy_2668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 474, 11), 'stypy_copy', False)
    # Obtaining the member 'python_lib_copy' of a type (line 474)
    python_lib_copy_2669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 11), stypy_copy_2668, 'python_lib_copy')
    # Obtaining the member 'python_types_copy' of a type (line 474)
    python_types_copy_2670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 11), python_lib_copy_2669, 'python_types_copy')
    # Obtaining the member 'type_introspection_copy' of a type (line 474)
    type_introspection_copy_2671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 11), python_types_copy_2670, 'type_introspection_copy')
    # Obtaining the member 'runtime_type_inspection_copy' of a type (line 474)
    runtime_type_inspection_copy_2672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 11), type_introspection_copy_2671, 'runtime_type_inspection_copy')
    # Obtaining the member 'is_union_type' of a type (line 474)
    is_union_type_2673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 474, 11), runtime_type_inspection_copy_2672, 'is_union_type')
    # Calling is_union_type(args, kwargs) (line 474)
    is_union_type_call_result_2676 = invoke(stypy.reporting.localization.Localization(__file__, 474, 11), is_union_type_2673, *[union_type_2674], **kwargs_2675)
    
    # Applying the 'not' unary operator (line 474)
    result_not__2677 = python_operator(stypy.reporting.localization.Localization(__file__, 474, 7), 'not', is_union_type_call_result_2676)
    
    # Testing if the type of an if condition is none (line 474)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 474, 4), result_not__2677):
        pass
    else:
        
        # Testing the type of an if condition (line 474)
        if_condition_2678 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 474, 4), result_not__2677)
        # Assigning a type to the variable 'if_condition_2678' (line 474)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 474, 4), 'if_condition_2678', if_condition_2678)
        # SSA begins for if statement (line 474)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'union_type' (line 475)
        union_type_2679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 475, 15), 'union_type')
        # Assigning a type to the variable 'stypy_return_type' (line 475)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 475, 8), 'stypy_return_type', union_type_2679)
        # SSA join for if statement (line 474)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Name to a Name (line 476):
    
    # Assigning a Name to a Name (line 476):
    # Getting the type of 'None' (line 476)
    None_2680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 476, 13), 'None')
    # Assigning a type to the variable 'result' (line 476)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 476, 4), 'result', None_2680)
    
    # Assigning a Call to a Name (line 477):
    
    # Assigning a Call to a Name (line 477):
    
    # Call to instance(...): (line 477)
    # Processing the call arguments (line 477)
    # Getting the type of 'type_to_remove' (line 478)
    type_to_remove_2688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 478, 8), 'type_to_remove', False)
    # Processing the call keyword arguments (line 477)
    kwargs_2689 = {}
    # Getting the type of 'stypy_copy' (line 477)
    stypy_copy_2681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 477, 21), 'stypy_copy', False)
    # Obtaining the member 'python_lib_copy' of a type (line 477)
    python_lib_copy_2682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 21), stypy_copy_2681, 'python_lib_copy')
    # Obtaining the member 'python_types_copy' of a type (line 477)
    python_types_copy_2683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 21), python_lib_copy_2682, 'python_types_copy')
    # Obtaining the member 'type_inference_copy' of a type (line 477)
    type_inference_copy_2684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 21), python_types_copy_2683, 'type_inference_copy')
    # Obtaining the member 'type_inference_proxy_copy' of a type (line 477)
    type_inference_proxy_copy_2685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 21), type_inference_copy_2684, 'type_inference_proxy_copy')
    # Obtaining the member 'TypeInferenceProxy' of a type (line 477)
    TypeInferenceProxy_2686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 21), type_inference_proxy_copy_2685, 'TypeInferenceProxy')
    # Obtaining the member 'instance' of a type (line 477)
    instance_2687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 477, 21), TypeInferenceProxy_2686, 'instance')
    # Calling instance(args, kwargs) (line 477)
    instance_call_result_2690 = invoke(stypy.reporting.localization.Localization(__file__, 477, 21), instance_2687, *[type_to_remove_2688], **kwargs_2689)
    
    # Assigning a type to the variable 'type_to_remove' (line 477)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 477, 4), 'type_to_remove', instance_call_result_2690)
    
    # Getting the type of 'union_type' (line 479)
    union_type_2691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 479, 17), 'union_type')
    # Obtaining the member 'types' of a type (line 479)
    types_2692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 479, 17), union_type_2691, 'types')
    # Assigning a type to the variable 'types_2692' (line 479)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'types_2692', types_2692)
    # Testing if the for loop is going to be iterated (line 479)
    # Testing the type of a for loop iterable (line 479)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 479, 4), types_2692)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 479, 4), types_2692):
        # Getting the type of the for loop variable (line 479)
        for_loop_var_2693 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 479, 4), types_2692)
        # Assigning a type to the variable 'type_' (line 479)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 479, 4), 'type_', for_loop_var_2693)
        # SSA begins for a for statement (line 479)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'type_' (line 480)
        type__2694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 15), 'type_')
        # Getting the type of 'type_to_remove' (line 480)
        type_to_remove_2695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 480, 24), 'type_to_remove')
        # Applying the binary operator '==' (line 480)
        result_eq_2696 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 15), '==', type__2694, type_to_remove_2695)
        
        # Applying the 'not' unary operator (line 480)
        result_not__2697 = python_operator(stypy.reporting.localization.Localization(__file__, 480, 11), 'not', result_eq_2696)
        
        # Testing if the type of an if condition is none (line 480)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 480, 8), result_not__2697):
            pass
        else:
            
            # Testing the type of an if condition (line 480)
            if_condition_2698 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 480, 8), result_not__2697)
            # Assigning a type to the variable 'if_condition_2698' (line 480)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 480, 8), 'if_condition_2698', if_condition_2698)
            # SSA begins for if statement (line 480)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 481):
            
            # Assigning a Call to a Name (line 481):
            
            # Call to add(...): (line 481)
            # Processing the call arguments (line 481)
            # Getting the type of 'result' (line 481)
            result_2706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 96), 'result', False)
            # Getting the type of 'type_' (line 481)
            type__2707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 104), 'type_', False)
            # Processing the call keyword arguments (line 481)
            kwargs_2708 = {}
            # Getting the type of 'stypy_copy' (line 481)
            stypy_copy_2699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 481, 21), 'stypy_copy', False)
            # Obtaining the member 'python_lib' of a type (line 481)
            python_lib_2700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 21), stypy_copy_2699, 'python_lib')
            # Obtaining the member 'python_types' of a type (line 481)
            python_types_2701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 21), python_lib_2700, 'python_types')
            # Obtaining the member 'type_inference' of a type (line 481)
            type_inference_2702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 21), python_types_2701, 'type_inference')
            # Obtaining the member 'union_type' of a type (line 481)
            union_type_2703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 21), type_inference_2702, 'union_type')
            # Obtaining the member 'UnionType' of a type (line 481)
            UnionType_2704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 21), union_type_2703, 'UnionType')
            # Obtaining the member 'add' of a type (line 481)
            add_2705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 481, 21), UnionType_2704, 'add')
            # Calling add(args, kwargs) (line 481)
            add_call_result_2709 = invoke(stypy.reporting.localization.Localization(__file__, 481, 21), add_2705, *[result_2706, type__2707], **kwargs_2708)
            
            # Assigning a type to the variable 'result' (line 481)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 481, 12), 'result', add_call_result_2709)
            # SSA join for if statement (line 480)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'result' (line 482)
    result_2710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 482, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 482)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 482, 4), 'stypy_return_type', result_2710)
    
    # ################# End of 'remove_type_from_union(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'remove_type_from_union' in the type store
    # Getting the type of 'stypy_return_type' (line 467)
    stypy_return_type_2711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 467, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2711)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'remove_type_from_union'
    return stypy_return_type_2711

# Assigning a type to the variable 'remove_type_from_union' (line 467)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 467, 0), 'remove_type_from_union', remove_type_from_union)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
