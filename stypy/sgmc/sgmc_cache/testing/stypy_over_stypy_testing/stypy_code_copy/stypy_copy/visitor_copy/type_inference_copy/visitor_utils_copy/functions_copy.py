
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import ast
2: 
3: import core_language_copy
4: import data_structures_copy
5: import conditional_statements_copy
6: import stypy_functions_copy
7: 
8: '''
9: This file contains helper functions to generate type inference code.
10: These functions refer to function-related language elements such as declarations and invokations.
11: '''
12: 
13: 
14: def create_call(func, args, keywords=list(), kwargs=None, starargs=None, line=0, column=0):
15:     '''
16:     Creates an AST Call node
17: 
18:     :param func: Function name
19:     :param args: List of arguments
20:     :param keywords: List of default arguments
21:     :param kwargs: Dict of keyword arguments
22:     :param starargs: Variable list of arguments
23:     :param line: Line
24:     :param column: Column
25:     :return: AST Call node
26:     '''
27:     call = ast.Call()
28:     call.args = []
29: 
30:     if data_structures_copy.is_iterable(args):
31:         for arg in args:
32:             call.args.append(arg)
33:     else:
34:         call.args.append(args)
35: 
36:     call.func = func
37:     call.lineno = line
38:     call.col_offset = column
39:     call.keywords = keywords
40:     call.kwargs = kwargs
41:     call.starargs = starargs
42: 
43:     return call
44: 
45: 
46: def create_call_expression(func, args, keywords=list(), kwargs=None, starargs=None, line=0, column=0):
47:     '''
48:     Creates an AST Call node that will be enclosed in an expression node. This is used when the call are not a part
49:     of a longer expression, but the expression itself
50: 
51:     :param func: Function name
52:     :param args: List of arguments
53:     :param keywords: List of default arguments
54:     :param kwargs: Dict of keyword arguments
55:     :param starargs: Variable list of arguments
56:     :param line: Line
57:     :param column: Column
58:     :return: AST Expr node
59:     '''
60:     call = create_call(func, args, keywords, kwargs, starargs, line, column)
61:     call_expression = ast.Expr()
62:     call_expression.value = call
63:     call_expression.lineno = line
64:     call_expression.col_offset = column
65: 
66:     return call_expression
67: 
68: 
69: def is_method(context):
70:     '''
71:     Determines if an AST Function node represent a method (belongs to an AST ClassDef node)
72:     :param context:
73:     :return:
74:     '''
75:     ismethod = False
76: 
77:     if not len(context) == 0:
78:         ismethod = isinstance(context[-1], ast.ClassDef)
79: 
80:     return ismethod
81: 
82: 
83: def is_static_method(node):
84:     if not hasattr(node, "decorator_list"):
85:         return False
86:     if len(node.decorator_list) == 0:
87:         return False
88:     for dec_name in node.decorator_list:
89:         if hasattr(dec_name, "id"):
90:             if dec_name.id == "staticmethod":
91:                 return True
92:     return False
93: 
94: 
95: def is_constructor(node):
96:     '''
97:     Determines if an AST Function node represent a constructor (its name is __init__)
98:     :param node: AST Function node or str
99:     :return: bool
100:     '''
101:     if type(node) is str:
102:         return node == "__init__"
103: 
104:     return node.name == "__init__"
105: 
106: 
107: def create_function_def(name, localization, decorators, context, line=0, column=0):
108:     '''
109:     Creates a FunctionDef node, that represent a function declaration. This is used in type inference code, so every
110:     created function has the following parameters (type_of_self, localization, *varargs, **kwargs) for methods and
111:     (localization, *varargs, **kwargs) for functions.
112: 
113:     :param name: Name of the function
114:     :param localization: Localization parameter
115:     :param decorators: Decorators of the function, mainly the norecursion one
116:     :param context: Context passed to this method
117:     :param line: Line
118:     :param column: Column
119:     :return: An AST FunctionDef node
120:     '''
121:     function_def_arguments = ast.arguments()
122:     function_def_arguments.args = [localization]
123: 
124:     isconstructor = is_constructor(name)
125:     ismethod = is_method(context)
126: 
127:     function_def = ast.FunctionDef()
128:     function_def.lineno = line
129:     function_def.col_offset = column
130:     function_def.name = name
131: 
132:     function_def.args = function_def_arguments
133: 
134:     function_def_arguments.args = []
135: 
136:     if isconstructor:
137:         function_def_arguments.args.append(core_language_copy.create_Name('type_of_self'))
138: 
139:     if ismethod and not isconstructor:
140:         function_def_arguments.args.append(core_language_copy.create_Name('type_of_self'))
141: 
142:     function_def_arguments.args.append(localization)
143: 
144:     function_def_arguments.kwarg = "kwargs"
145:     function_def_arguments.vararg = "varargs"
146:     function_def_arguments.defaults = []
147: 
148:     if data_structures_copy.is_iterable(decorators):
149:         function_def.decorator_list = decorators
150:     else:
151:         function_def.decorator_list = [decorators]
152: 
153:     function_def.body = []
154: 
155:     return function_def
156: 
157: 
158: def create_return(value):
159:     '''
160:     Creates an AST Return node
161:     :param value: Value to return
162:     :return: An AST Return node
163:     '''
164:     node = ast.Return()
165:     node.value = value
166: 
167:     return node
168: 
169: 
170: def obtain_arg_list(args, ismethod=False, isstaticmethod=False):
171:     '''
172:     Creates an AST List node with the names of the arguments passed to a function
173:     :param args: Arguments
174:     :param ismethod: Whether to count the first argument (self) or not
175:     :return: An AST List
176:     '''
177:     arg_list = ast.List()
178: 
179:     arg_list.elts = []
180:     if ismethod and not isstaticmethod:
181:         arg_list_contents = args.args[1:]
182:     else:
183:         arg_list_contents = args.args
184: 
185:     for arg in arg_list_contents:
186:         arg_list.elts.append(core_language_copy.create_str(arg.id))
187: 
188:     return arg_list
189: 
190: 
191: def create_stacktrace_push(func_name, declared_arguments):
192:     '''
193:     Creates an AST Node that model the call to the localitazion.set_stack_trace method
194: 
195:     :param func_name: Name of the function that will do the push to the stack trace
196:     :param declared_arguments: Arguments of the call
197:     :return: An AST Expr node
198:     '''
199:     # Code to push a new stack trace to handle errors.
200:     attribute = core_language_copy.create_attribute("localization", "set_stack_trace")
201:     arguments_var = core_language_copy.create_Name("arguments")
202:     stack_push_call = create_call(attribute, [core_language_copy.create_str(func_name), declared_arguments, arguments_var])
203:     stack_push = ast.Expr()
204:     stack_push.value = stack_push_call
205: 
206:     return stack_push
207: 
208: 
209: def create_stacktrace_pop():
210:     '''
211:     Creates an AST Node that model the call to the localitazion.unset_stack_trace method
212: 
213:     :return: An AST Expr node
214:     '''
215:     # Code to pop a stack trace once the function finishes.
216:     attribute = core_language_copy.create_attribute("localization", "unset_stack_trace")
217:     stack_pop_call = create_call(attribute, [])
218:     stack_pop = ast.Expr()
219:     stack_pop.value = stack_pop_call
220: 
221:     return stack_pop
222: 
223: 
224: def create_context_set(func_name, lineno, col_offset):
225:     '''
226:     Creates an AST Node that model the call to the type_store.set_context method
227: 
228:     :param func_name: Name of the function that will do the push to the stack trace
229:     :param lineno: Line
230:     :param col_offset: Column
231:     :return: An AST Expr node
232:     '''
233:     attribute = core_language_copy.create_attribute("type_store", "set_context")
234:     context_set_call = create_call(attribute, [core_language_copy.create_str(func_name),
235:                                                core_language_copy.create_num(lineno),
236:                                                core_language_copy.create_num(col_offset)])
237:     context_set = ast.Expr()
238:     context_set.value = context_set_call
239: 
240:     return context_set
241: 
242: 
243: def create_context_unset():
244:     '''
245:     Creates an AST Node that model the call to the type_store.unset_context method
246: 
247:     :return: An AST Expr node
248:     '''
249:     # Code to pop a stack trace once the function finishes.
250:     attribute = core_language_copy.create_attribute("type_store", "unset_context")
251:     context_unset_call = create_call(attribute, [])
252:     context_unset = ast.Expr()
253:     context_unset.value = context_unset_call
254: 
255:     return context_unset
256: 
257: 
258: def create_arg_number_test(function_def_node, context=[]):
259:     '''
260:     Creates an AST Node that model the call to the process_argument_values method. This method is used to check
261:     the parameters passed to a function/method in a type inference program
262: 
263:     :param function_def_node: AST Node with the function definition
264:     :param context: Context passed to the call
265:     :return: List of AST nodes that perform the call to the mentioned function and make the necessary tests once it
266:     is called
267:     '''
268:     args_test_resul = core_language_copy.create_Name('arguments', False)
269: 
270:     # Call to arg test function
271:     func = core_language_copy.create_Name('process_argument_values')
272:     # Fixed parameters
273:     localization_arg = core_language_copy.create_Name('localization')
274:     type_store_arg = core_language_copy.create_Name('type_store')
275: 
276:     # Declaration data arguments
277:     # Func name
278:     if is_method(context):
279:         function_name_arg = core_language_copy.create_str(context[-1].name + "." + function_def_node.name)
280:         type_of_self_arg = core_language_copy.create_Name('type_of_self')
281:     else:
282:         function_name_arg = core_language_copy.create_str(function_def_node.name)
283:         type_of_self_arg = core_language_copy.create_Name('None')
284: 
285:     # Declared param names list
286:     param_names_list_arg = obtain_arg_list(function_def_node.args, is_method(context),
287:                                            is_static_method(function_def_node))
288: 
289:     # Declared var args parameter name
290:     if function_def_node.args.vararg is None:
291:         declared_varargs = None
292:     else:
293:         declared_varargs = function_def_node.args.vararg
294:     varargs_param_name = core_language_copy.create_str(declared_varargs)
295:     # Declared kwargs parameter name
296:     if function_def_node.args.kwarg is None:
297:         declared_kwargs = None
298:     else:
299:         declared_kwargs = function_def_node.args.kwarg
300:     kwargs_param_name = core_language_copy.create_str(declared_kwargs)
301: 
302:     # Call data arguments
303:     # Declared defaults list name
304:     call_defaults = core_language_copy.create_Name('defaults')  # function_def_node.args.defaults
305: 
306:     # Call varargs
307:     call_varargs = core_language_copy.create_Name('varargs')
308:     # Call kwargs
309:     call_kwargs = core_language_copy.create_Name('kwargs')
310: 
311:     # Parameter number check call
312:     call = create_call(func,
313:                        [localization_arg, type_of_self_arg, type_store_arg, function_name_arg, param_names_list_arg,
314:                         varargs_param_name, kwargs_param_name, call_defaults, call_varargs, call_kwargs])
315: 
316:     assign = core_language_copy.create_Assign(args_test_resul, call)
317: 
318:     # After parameter number check call
319:     argument_errors = core_language_copy.create_Name('arguments')
320:     is_error_type = core_language_copy.create_Name('is_error_type')
321:     if_test = create_call(is_error_type, argument_errors)
322: 
323:     if is_constructor(function_def_node):
324:         argument_errors = None  # core_language.create_Name('None')
325: 
326:     body = [create_context_unset(), create_return(argument_errors)]
327:     if_ = conditional_statements_copy.create_if(if_test, body)
328: 
329:     return [assign, if_]
330: 
331: 
332: def create_type_for_lambda_function(function_name, lambda_call, lineno, col_offset):
333:     '''
334:     Creates a variable to store a lambda function definition
335: 
336:     :param function_name: Name of the lambda function
337:     :param lambda_call: Lambda function
338:     :param lineno: Line
339:     :param col_offset: Column
340:     :return: Statements to create the lambda function type
341:     '''
342:     # TODO: Remove?
343:     # call_arg = core_language.create_Name(lambda_call)
344:     # call_func = core_language.create_Name("LambdaFunctionType")
345:     # call = create_call(call_func, call_arg)
346:     # assign_target = core_language.create_Name(lambda_call, False)
347:     # assign = core_language.create_Assign(assign_target, call)
348: 
349:     call_arg = core_language_copy.create_Name(lambda_call)
350: 
351:     set_type_stmts = stypy_functions_copy.create_set_type_of(function_name, call_arg, lineno, col_offset)
352: 
353:     # return stypy_functions.flatten_lists(assign, set_type_stmts)
354:     return stypy_functions_copy.flatten_lists(set_type_stmts)
355: 

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
import_15564 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'core_language_copy')

if (type(import_15564) is not StypyTypeError):

    if (import_15564 != 'pyd_module'):
        __import__(import_15564)
        sys_modules_15565 = sys.modules[import_15564]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'core_language_copy', sys_modules_15565.module_type_store, module_type_store)
    else:
        import core_language_copy

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'core_language_copy', core_language_copy, module_type_store)

else:
    # Assigning a type to the variable 'core_language_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'core_language_copy', import_15564)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import data_structures_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_15566 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'data_structures_copy')

if (type(import_15566) is not StypyTypeError):

    if (import_15566 != 'pyd_module'):
        __import__(import_15566)
        sys_modules_15567 = sys.modules[import_15566]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'data_structures_copy', sys_modules_15567.module_type_store, module_type_store)
    else:
        import data_structures_copy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'data_structures_copy', data_structures_copy, module_type_store)

else:
    # Assigning a type to the variable 'data_structures_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'data_structures_copy', import_15566)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import conditional_statements_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_15568 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'conditional_statements_copy')

if (type(import_15568) is not StypyTypeError):

    if (import_15568 != 'pyd_module'):
        __import__(import_15568)
        sys_modules_15569 = sys.modules[import_15568]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'conditional_statements_copy', sys_modules_15569.module_type_store, module_type_store)
    else:
        import conditional_statements_copy

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'conditional_statements_copy', conditional_statements_copy, module_type_store)

else:
    # Assigning a type to the variable 'conditional_statements_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'conditional_statements_copy', import_15568)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import stypy_functions_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_15570 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_functions_copy')

if (type(import_15570) is not StypyTypeError):

    if (import_15570 != 'pyd_module'):
        __import__(import_15570)
        sys_modules_15571 = sys.modules[import_15570]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_functions_copy', sys_modules_15571.module_type_store, module_type_store)
    else:
        import stypy_functions_copy

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_functions_copy', stypy_functions_copy, module_type_store)

else:
    # Assigning a type to the variable 'stypy_functions_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_functions_copy', import_15570)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

str_15572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\nThis file contains helper functions to generate type inference code.\nThese functions refer to function-related language elements such as declarations and invokations.\n')

@norecursion
def create_call(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Call to list(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_15574 = {}
    # Getting the type of 'list' (line 14)
    list_15573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 37), 'list', False)
    # Calling list(args, kwargs) (line 14)
    list_call_result_15575 = invoke(stypy.reporting.localization.Localization(__file__, 14, 37), list_15573, *[], **kwargs_15574)
    
    # Getting the type of 'None' (line 14)
    None_15576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 52), 'None')
    # Getting the type of 'None' (line 14)
    None_15577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 67), 'None')
    int_15578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 78), 'int')
    int_15579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 88), 'int')
    defaults = [list_call_result_15575, None_15576, None_15577, int_15578, int_15579]
    # Create a new context for function 'create_call'
    module_type_store = module_type_store.open_function_context('create_call', 14, 0, False)
    
    # Passed parameters checking function
    create_call.stypy_localization = localization
    create_call.stypy_type_of_self = None
    create_call.stypy_type_store = module_type_store
    create_call.stypy_function_name = 'create_call'
    create_call.stypy_param_names_list = ['func', 'args', 'keywords', 'kwargs', 'starargs', 'line', 'column']
    create_call.stypy_varargs_param_name = None
    create_call.stypy_kwargs_param_name = None
    create_call.stypy_call_defaults = defaults
    create_call.stypy_call_varargs = varargs
    create_call.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_call', ['func', 'args', 'keywords', 'kwargs', 'starargs', 'line', 'column'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_call', localization, ['func', 'args', 'keywords', 'kwargs', 'starargs', 'line', 'column'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_call(...)' code ##################

    str_15580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'str', '\n    Creates an AST Call node\n\n    :param func: Function name\n    :param args: List of arguments\n    :param keywords: List of default arguments\n    :param kwargs: Dict of keyword arguments\n    :param starargs: Variable list of arguments\n    :param line: Line\n    :param column: Column\n    :return: AST Call node\n    ')
    
    # Assigning a Call to a Name (line 27):
    
    # Call to Call(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_15583 = {}
    # Getting the type of 'ast' (line 27)
    ast_15581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'ast', False)
    # Obtaining the member 'Call' of a type (line 27)
    Call_15582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 11), ast_15581, 'Call')
    # Calling Call(args, kwargs) (line 27)
    Call_call_result_15584 = invoke(stypy.reporting.localization.Localization(__file__, 27, 11), Call_15582, *[], **kwargs_15583)
    
    # Assigning a type to the variable 'call' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'call', Call_call_result_15584)
    
    # Assigning a List to a Attribute (line 28):
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_15585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    
    # Getting the type of 'call' (line 28)
    call_15586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'call')
    # Setting the type of the member 'args' of a type (line 28)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), call_15586, 'args', list_15585)
    
    # Call to is_iterable(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'args' (line 30)
    args_15589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 40), 'args', False)
    # Processing the call keyword arguments (line 30)
    kwargs_15590 = {}
    # Getting the type of 'data_structures_copy' (line 30)
    data_structures_copy_15587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'data_structures_copy', False)
    # Obtaining the member 'is_iterable' of a type (line 30)
    is_iterable_15588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 7), data_structures_copy_15587, 'is_iterable')
    # Calling is_iterable(args, kwargs) (line 30)
    is_iterable_call_result_15591 = invoke(stypy.reporting.localization.Localization(__file__, 30, 7), is_iterable_15588, *[args_15589], **kwargs_15590)
    
    # Testing if the type of an if condition is none (line 30)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 30, 4), is_iterable_call_result_15591):
        
        # Call to append(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'args' (line 34)
        args_15604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'args', False)
        # Processing the call keyword arguments (line 34)
        kwargs_15605 = {}
        # Getting the type of 'call' (line 34)
        call_15601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'call', False)
        # Obtaining the member 'args' of a type (line 34)
        args_15602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), call_15601, 'args')
        # Obtaining the member 'append' of a type (line 34)
        append_15603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), args_15602, 'append')
        # Calling append(args, kwargs) (line 34)
        append_call_result_15606 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), append_15603, *[args_15604], **kwargs_15605)
        
    else:
        
        # Testing the type of an if condition (line 30)
        if_condition_15592 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 4), is_iterable_call_result_15591)
        # Assigning a type to the variable 'if_condition_15592' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'if_condition_15592', if_condition_15592)
        # SSA begins for if statement (line 30)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'args' (line 31)
        args_15593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'args')
        # Assigning a type to the variable 'args_15593' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'args_15593', args_15593)
        # Testing if the for loop is going to be iterated (line 31)
        # Testing the type of a for loop iterable (line 31)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 8), args_15593)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 31, 8), args_15593):
            # Getting the type of the for loop variable (line 31)
            for_loop_var_15594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 8), args_15593)
            # Assigning a type to the variable 'arg' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'arg', for_loop_var_15594)
            # SSA begins for a for statement (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 32)
            # Processing the call arguments (line 32)
            # Getting the type of 'arg' (line 32)
            arg_15598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 29), 'arg', False)
            # Processing the call keyword arguments (line 32)
            kwargs_15599 = {}
            # Getting the type of 'call' (line 32)
            call_15595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'call', False)
            # Obtaining the member 'args' of a type (line 32)
            args_15596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), call_15595, 'args')
            # Obtaining the member 'append' of a type (line 32)
            append_15597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), args_15596, 'append')
            # Calling append(args, kwargs) (line 32)
            append_call_result_15600 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), append_15597, *[arg_15598], **kwargs_15599)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA branch for the else part of an if statement (line 30)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'args' (line 34)
        args_15604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'args', False)
        # Processing the call keyword arguments (line 34)
        kwargs_15605 = {}
        # Getting the type of 'call' (line 34)
        call_15601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'call', False)
        # Obtaining the member 'args' of a type (line 34)
        args_15602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), call_15601, 'args')
        # Obtaining the member 'append' of a type (line 34)
        append_15603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), args_15602, 'append')
        # Calling append(args, kwargs) (line 34)
        append_call_result_15606 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), append_15603, *[args_15604], **kwargs_15605)
        
        # SSA join for if statement (line 30)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Name to a Attribute (line 36):
    # Getting the type of 'func' (line 36)
    func_15607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'func')
    # Getting the type of 'call' (line 36)
    call_15608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'call')
    # Setting the type of the member 'func' of a type (line 36)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), call_15608, 'func', func_15607)
    
    # Assigning a Name to a Attribute (line 37):
    # Getting the type of 'line' (line 37)
    line_15609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'line')
    # Getting the type of 'call' (line 37)
    call_15610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'call')
    # Setting the type of the member 'lineno' of a type (line 37)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), call_15610, 'lineno', line_15609)
    
    # Assigning a Name to a Attribute (line 38):
    # Getting the type of 'column' (line 38)
    column_15611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'column')
    # Getting the type of 'call' (line 38)
    call_15612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'call')
    # Setting the type of the member 'col_offset' of a type (line 38)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), call_15612, 'col_offset', column_15611)
    
    # Assigning a Name to a Attribute (line 39):
    # Getting the type of 'keywords' (line 39)
    keywords_15613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'keywords')
    # Getting the type of 'call' (line 39)
    call_15614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call')
    # Setting the type of the member 'keywords' of a type (line 39)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 4), call_15614, 'keywords', keywords_15613)
    
    # Assigning a Name to a Attribute (line 40):
    # Getting the type of 'kwargs' (line 40)
    kwargs_15615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'kwargs')
    # Getting the type of 'call' (line 40)
    call_15616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'call')
    # Setting the type of the member 'kwargs' of a type (line 40)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 4), call_15616, 'kwargs', kwargs_15615)
    
    # Assigning a Name to a Attribute (line 41):
    # Getting the type of 'starargs' (line 41)
    starargs_15617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'starargs')
    # Getting the type of 'call' (line 41)
    call_15618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'call')
    # Setting the type of the member 'starargs' of a type (line 41)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), call_15618, 'starargs', starargs_15617)
    # Getting the type of 'call' (line 43)
    call_15619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'call')
    # Assigning a type to the variable 'stypy_return_type' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type', call_15619)
    
    # ################# End of 'create_call(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_call' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_15620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15620)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_call'
    return stypy_return_type_15620

# Assigning a type to the variable 'create_call' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'create_call', create_call)

@norecursion
def create_call_expression(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Call to list(...): (line 46)
    # Processing the call keyword arguments (line 46)
    kwargs_15622 = {}
    # Getting the type of 'list' (line 46)
    list_15621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 48), 'list', False)
    # Calling list(args, kwargs) (line 46)
    list_call_result_15623 = invoke(stypy.reporting.localization.Localization(__file__, 46, 48), list_15621, *[], **kwargs_15622)
    
    # Getting the type of 'None' (line 46)
    None_15624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 63), 'None')
    # Getting the type of 'None' (line 46)
    None_15625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 78), 'None')
    int_15626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 89), 'int')
    int_15627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 99), 'int')
    defaults = [list_call_result_15623, None_15624, None_15625, int_15626, int_15627]
    # Create a new context for function 'create_call_expression'
    module_type_store = module_type_store.open_function_context('create_call_expression', 46, 0, False)
    
    # Passed parameters checking function
    create_call_expression.stypy_localization = localization
    create_call_expression.stypy_type_of_self = None
    create_call_expression.stypy_type_store = module_type_store
    create_call_expression.stypy_function_name = 'create_call_expression'
    create_call_expression.stypy_param_names_list = ['func', 'args', 'keywords', 'kwargs', 'starargs', 'line', 'column']
    create_call_expression.stypy_varargs_param_name = None
    create_call_expression.stypy_kwargs_param_name = None
    create_call_expression.stypy_call_defaults = defaults
    create_call_expression.stypy_call_varargs = varargs
    create_call_expression.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_call_expression', ['func', 'args', 'keywords', 'kwargs', 'starargs', 'line', 'column'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_call_expression', localization, ['func', 'args', 'keywords', 'kwargs', 'starargs', 'line', 'column'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_call_expression(...)' code ##################

    str_15628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '\n    Creates an AST Call node that will be enclosed in an expression node. This is used when the call are not a part\n    of a longer expression, but the expression itself\n\n    :param func: Function name\n    :param args: List of arguments\n    :param keywords: List of default arguments\n    :param kwargs: Dict of keyword arguments\n    :param starargs: Variable list of arguments\n    :param line: Line\n    :param column: Column\n    :return: AST Expr node\n    ')
    
    # Assigning a Call to a Name (line 60):
    
    # Call to create_call(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'func' (line 60)
    func_15630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'func', False)
    # Getting the type of 'args' (line 60)
    args_15631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'args', False)
    # Getting the type of 'keywords' (line 60)
    keywords_15632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 35), 'keywords', False)
    # Getting the type of 'kwargs' (line 60)
    kwargs_15633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 45), 'kwargs', False)
    # Getting the type of 'starargs' (line 60)
    starargs_15634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 53), 'starargs', False)
    # Getting the type of 'line' (line 60)
    line_15635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 63), 'line', False)
    # Getting the type of 'column' (line 60)
    column_15636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 69), 'column', False)
    # Processing the call keyword arguments (line 60)
    kwargs_15637 = {}
    # Getting the type of 'create_call' (line 60)
    create_call_15629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'create_call', False)
    # Calling create_call(args, kwargs) (line 60)
    create_call_call_result_15638 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), create_call_15629, *[func_15630, args_15631, keywords_15632, kwargs_15633, starargs_15634, line_15635, column_15636], **kwargs_15637)
    
    # Assigning a type to the variable 'call' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'call', create_call_call_result_15638)
    
    # Assigning a Call to a Name (line 61):
    
    # Call to Expr(...): (line 61)
    # Processing the call keyword arguments (line 61)
    kwargs_15641 = {}
    # Getting the type of 'ast' (line 61)
    ast_15639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 22), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 61)
    Expr_15640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 22), ast_15639, 'Expr')
    # Calling Expr(args, kwargs) (line 61)
    Expr_call_result_15642 = invoke(stypy.reporting.localization.Localization(__file__, 61, 22), Expr_15640, *[], **kwargs_15641)
    
    # Assigning a type to the variable 'call_expression' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'call_expression', Expr_call_result_15642)
    
    # Assigning a Name to a Attribute (line 62):
    # Getting the type of 'call' (line 62)
    call_15643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'call')
    # Getting the type of 'call_expression' (line 62)
    call_expression_15644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'call_expression')
    # Setting the type of the member 'value' of a type (line 62)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), call_expression_15644, 'value', call_15643)
    
    # Assigning a Name to a Attribute (line 63):
    # Getting the type of 'line' (line 63)
    line_15645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'line')
    # Getting the type of 'call_expression' (line 63)
    call_expression_15646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'call_expression')
    # Setting the type of the member 'lineno' of a type (line 63)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 4), call_expression_15646, 'lineno', line_15645)
    
    # Assigning a Name to a Attribute (line 64):
    # Getting the type of 'column' (line 64)
    column_15647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'column')
    # Getting the type of 'call_expression' (line 64)
    call_expression_15648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'call_expression')
    # Setting the type of the member 'col_offset' of a type (line 64)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), call_expression_15648, 'col_offset', column_15647)
    # Getting the type of 'call_expression' (line 66)
    call_expression_15649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'call_expression')
    # Assigning a type to the variable 'stypy_return_type' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type', call_expression_15649)
    
    # ################# End of 'create_call_expression(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_call_expression' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_15650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15650)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_call_expression'
    return stypy_return_type_15650

# Assigning a type to the variable 'create_call_expression' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'create_call_expression', create_call_expression)

@norecursion
def is_method(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_method'
    module_type_store = module_type_store.open_function_context('is_method', 69, 0, False)
    
    # Passed parameters checking function
    is_method.stypy_localization = localization
    is_method.stypy_type_of_self = None
    is_method.stypy_type_store = module_type_store
    is_method.stypy_function_name = 'is_method'
    is_method.stypy_param_names_list = ['context']
    is_method.stypy_varargs_param_name = None
    is_method.stypy_kwargs_param_name = None
    is_method.stypy_call_defaults = defaults
    is_method.stypy_call_varargs = varargs
    is_method.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_method', ['context'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_method', localization, ['context'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_method(...)' code ##################

    str_15651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, (-1)), 'str', '\n    Determines if an AST Function node represent a method (belongs to an AST ClassDef node)\n    :param context:\n    :return:\n    ')
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'False' (line 75)
    False_15652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'False')
    # Assigning a type to the variable 'ismethod' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'ismethod', False_15652)
    
    
    
    # Call to len(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'context' (line 77)
    context_15654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'context', False)
    # Processing the call keyword arguments (line 77)
    kwargs_15655 = {}
    # Getting the type of 'len' (line 77)
    len_15653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'len', False)
    # Calling len(args, kwargs) (line 77)
    len_call_result_15656 = invoke(stypy.reporting.localization.Localization(__file__, 77, 11), len_15653, *[context_15654], **kwargs_15655)
    
    int_15657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'int')
    # Applying the binary operator '==' (line 77)
    result_eq_15658 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 11), '==', len_call_result_15656, int_15657)
    
    # Applying the 'not' unary operator (line 77)
    result_not__15659 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 7), 'not', result_eq_15658)
    
    # Testing if the type of an if condition is none (line 77)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 4), result_not__15659):
        pass
    else:
        
        # Testing the type of an if condition (line 77)
        if_condition_15660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 4), result_not__15659)
        # Assigning a type to the variable 'if_condition_15660' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'if_condition_15660', if_condition_15660)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 78):
        
        # Call to isinstance(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Obtaining the type of the subscript
        int_15662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 38), 'int')
        # Getting the type of 'context' (line 78)
        context_15663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'context', False)
        # Obtaining the member '__getitem__' of a type (line 78)
        getitem___15664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 30), context_15663, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 78)
        subscript_call_result_15665 = invoke(stypy.reporting.localization.Localization(__file__, 78, 30), getitem___15664, int_15662)
        
        # Getting the type of 'ast' (line 78)
        ast_15666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 43), 'ast', False)
        # Obtaining the member 'ClassDef' of a type (line 78)
        ClassDef_15667 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 43), ast_15666, 'ClassDef')
        # Processing the call keyword arguments (line 78)
        kwargs_15668 = {}
        # Getting the type of 'isinstance' (line 78)
        isinstance_15661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 78)
        isinstance_call_result_15669 = invoke(stypy.reporting.localization.Localization(__file__, 78, 19), isinstance_15661, *[subscript_call_result_15665, ClassDef_15667], **kwargs_15668)
        
        # Assigning a type to the variable 'ismethod' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'ismethod', isinstance_call_result_15669)
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'ismethod' (line 80)
    ismethod_15670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'ismethod')
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type', ismethod_15670)
    
    # ################# End of 'is_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_method' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_15671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15671)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_method'
    return stypy_return_type_15671

# Assigning a type to the variable 'is_method' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'is_method', is_method)

@norecursion
def is_static_method(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_static_method'
    module_type_store = module_type_store.open_function_context('is_static_method', 83, 0, False)
    
    # Passed parameters checking function
    is_static_method.stypy_localization = localization
    is_static_method.stypy_type_of_self = None
    is_static_method.stypy_type_store = module_type_store
    is_static_method.stypy_function_name = 'is_static_method'
    is_static_method.stypy_param_names_list = ['node']
    is_static_method.stypy_varargs_param_name = None
    is_static_method.stypy_kwargs_param_name = None
    is_static_method.stypy_call_defaults = defaults
    is_static_method.stypy_call_varargs = varargs
    is_static_method.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_static_method', ['node'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_static_method', localization, ['node'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_static_method(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 84)
    str_15672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 25), 'str', 'decorator_list')
    # Getting the type of 'node' (line 84)
    node_15673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'node')
    
    (may_be_15674, more_types_in_union_15675) = may_not_provide_member(str_15672, node_15673)

    if may_be_15674:

        if more_types_in_union_15675:
            # Runtime conditional SSA (line 84)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'node' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'node', remove_member_provider_from_union(node_15673, 'decorator_list'))
        # Getting the type of 'False' (line 85)
        False_15676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', False_15676)

        if more_types_in_union_15675:
            # SSA join for if statement (line 84)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to len(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'node' (line 86)
    node_15678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'node', False)
    # Obtaining the member 'decorator_list' of a type (line 86)
    decorator_list_15679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 11), node_15678, 'decorator_list')
    # Processing the call keyword arguments (line 86)
    kwargs_15680 = {}
    # Getting the type of 'len' (line 86)
    len_15677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 7), 'len', False)
    # Calling len(args, kwargs) (line 86)
    len_call_result_15681 = invoke(stypy.reporting.localization.Localization(__file__, 86, 7), len_15677, *[decorator_list_15679], **kwargs_15680)
    
    int_15682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 35), 'int')
    # Applying the binary operator '==' (line 86)
    result_eq_15683 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 7), '==', len_call_result_15681, int_15682)
    
    # Testing if the type of an if condition is none (line 86)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 4), result_eq_15683):
        pass
    else:
        
        # Testing the type of an if condition (line 86)
        if_condition_15684 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 4), result_eq_15683)
        # Assigning a type to the variable 'if_condition_15684' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'if_condition_15684', if_condition_15684)
        # SSA begins for if statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 87)
        False_15685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', False_15685)
        # SSA join for if statement (line 86)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'node' (line 88)
    node_15686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'node')
    # Obtaining the member 'decorator_list' of a type (line 88)
    decorator_list_15687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 20), node_15686, 'decorator_list')
    # Assigning a type to the variable 'decorator_list_15687' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'decorator_list_15687', decorator_list_15687)
    # Testing if the for loop is going to be iterated (line 88)
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 4), decorator_list_15687)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 88, 4), decorator_list_15687):
        # Getting the type of the for loop variable (line 88)
        for_loop_var_15688 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 4), decorator_list_15687)
        # Assigning a type to the variable 'dec_name' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'dec_name', for_loop_var_15688)
        # SSA begins for a for statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 89)
        str_15689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'str', 'id')
        # Getting the type of 'dec_name' (line 89)
        dec_name_15690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'dec_name')
        
        (may_be_15691, more_types_in_union_15692) = may_provide_member(str_15689, dec_name_15690)

        if may_be_15691:

            if more_types_in_union_15692:
                # Runtime conditional SSA (line 89)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'dec_name' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'dec_name', remove_not_member_provider_from_union(dec_name_15690, 'id'))
            
            # Getting the type of 'dec_name' (line 90)
            dec_name_15693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'dec_name')
            # Obtaining the member 'id' of a type (line 90)
            id_15694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 15), dec_name_15693, 'id')
            str_15695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 30), 'str', 'staticmethod')
            # Applying the binary operator '==' (line 90)
            result_eq_15696 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 15), '==', id_15694, str_15695)
            
            # Testing if the type of an if condition is none (line 90)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 90, 12), result_eq_15696):
                pass
            else:
                
                # Testing the type of an if condition (line 90)
                if_condition_15697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 12), result_eq_15696)
                # Assigning a type to the variable 'if_condition_15697' (line 90)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'if_condition_15697', if_condition_15697)
                # SSA begins for if statement (line 90)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 91)
                True_15698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'stypy_return_type', True_15698)
                # SSA join for if statement (line 90)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_15692:
                # SSA join for if statement (line 89)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'False' (line 92)
    False_15699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', False_15699)
    
    # ################# End of 'is_static_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_static_method' in the type store
    # Getting the type of 'stypy_return_type' (line 83)
    stypy_return_type_15700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15700)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_static_method'
    return stypy_return_type_15700

# Assigning a type to the variable 'is_static_method' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'is_static_method', is_static_method)

@norecursion
def is_constructor(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_constructor'
    module_type_store = module_type_store.open_function_context('is_constructor', 95, 0, False)
    
    # Passed parameters checking function
    is_constructor.stypy_localization = localization
    is_constructor.stypy_type_of_self = None
    is_constructor.stypy_type_store = module_type_store
    is_constructor.stypy_function_name = 'is_constructor'
    is_constructor.stypy_param_names_list = ['node']
    is_constructor.stypy_varargs_param_name = None
    is_constructor.stypy_kwargs_param_name = None
    is_constructor.stypy_call_defaults = defaults
    is_constructor.stypy_call_varargs = varargs
    is_constructor.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_constructor', ['node'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_constructor', localization, ['node'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_constructor(...)' code ##################

    str_15701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, (-1)), 'str', '\n    Determines if an AST Function node represent a constructor (its name is __init__)\n    :param node: AST Function node or str\n    :return: bool\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 101)
    # Getting the type of 'node' (line 101)
    node_15702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'node')
    # Getting the type of 'str' (line 101)
    str_15703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'str')
    
    (may_be_15704, more_types_in_union_15705) = may_be_type(node_15702, str_15703)

    if may_be_15704:

        if more_types_in_union_15705:
            # Runtime conditional SSA (line 101)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'node' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'node', str_15703())
        
        # Getting the type of 'node' (line 102)
        node_15706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'node')
        str_15707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 23), 'str', '__init__')
        # Applying the binary operator '==' (line 102)
        result_eq_15708 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 15), '==', node_15706, str_15707)
        
        # Assigning a type to the variable 'stypy_return_type' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'stypy_return_type', result_eq_15708)

        if more_types_in_union_15705:
            # SSA join for if statement (line 101)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'node' (line 104)
    node_15709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'node')
    # Obtaining the member 'name' of a type (line 104)
    name_15710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), node_15709, 'name')
    str_15711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'str', '__init__')
    # Applying the binary operator '==' (line 104)
    result_eq_15712 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 11), '==', name_15710, str_15711)
    
    # Assigning a type to the variable 'stypy_return_type' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type', result_eq_15712)
    
    # ################# End of 'is_constructor(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_constructor' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_15713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15713)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_constructor'
    return stypy_return_type_15713

# Assigning a type to the variable 'is_constructor' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'is_constructor', is_constructor)

@norecursion
def create_function_def(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_15714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 70), 'int')
    int_15715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 80), 'int')
    defaults = [int_15714, int_15715]
    # Create a new context for function 'create_function_def'
    module_type_store = module_type_store.open_function_context('create_function_def', 107, 0, False)
    
    # Passed parameters checking function
    create_function_def.stypy_localization = localization
    create_function_def.stypy_type_of_self = None
    create_function_def.stypy_type_store = module_type_store
    create_function_def.stypy_function_name = 'create_function_def'
    create_function_def.stypy_param_names_list = ['name', 'localization', 'decorators', 'context', 'line', 'column']
    create_function_def.stypy_varargs_param_name = None
    create_function_def.stypy_kwargs_param_name = None
    create_function_def.stypy_call_defaults = defaults
    create_function_def.stypy_call_varargs = varargs
    create_function_def.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_function_def', ['name', 'localization', 'decorators', 'context', 'line', 'column'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_function_def', localization, ['name', 'localization', 'decorators', 'context', 'line', 'column'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_function_def(...)' code ##################

    str_15716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, (-1)), 'str', '\n    Creates a FunctionDef node, that represent a function declaration. This is used in type inference code, so every\n    created function has the following parameters (type_of_self, localization, *varargs, **kwargs) for methods and\n    (localization, *varargs, **kwargs) for functions.\n\n    :param name: Name of the function\n    :param localization: Localization parameter\n    :param decorators: Decorators of the function, mainly the norecursion one\n    :param context: Context passed to this method\n    :param line: Line\n    :param column: Column\n    :return: An AST FunctionDef node\n    ')
    
    # Assigning a Call to a Name (line 121):
    
    # Call to arguments(...): (line 121)
    # Processing the call keyword arguments (line 121)
    kwargs_15719 = {}
    # Getting the type of 'ast' (line 121)
    ast_15717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'ast', False)
    # Obtaining the member 'arguments' of a type (line 121)
    arguments_15718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 29), ast_15717, 'arguments')
    # Calling arguments(args, kwargs) (line 121)
    arguments_call_result_15720 = invoke(stypy.reporting.localization.Localization(__file__, 121, 29), arguments_15718, *[], **kwargs_15719)
    
    # Assigning a type to the variable 'function_def_arguments' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'function_def_arguments', arguments_call_result_15720)
    
    # Assigning a List to a Attribute (line 122):
    
    # Obtaining an instance of the builtin type 'list' (line 122)
    list_15721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 122)
    # Adding element type (line 122)
    # Getting the type of 'localization' (line 122)
    localization_15722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 35), 'localization')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 34), list_15721, localization_15722)
    
    # Getting the type of 'function_def_arguments' (line 122)
    function_def_arguments_15723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'function_def_arguments')
    # Setting the type of the member 'args' of a type (line 122)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 4), function_def_arguments_15723, 'args', list_15721)
    
    # Assigning a Call to a Name (line 124):
    
    # Call to is_constructor(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'name' (line 124)
    name_15725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 35), 'name', False)
    # Processing the call keyword arguments (line 124)
    kwargs_15726 = {}
    # Getting the type of 'is_constructor' (line 124)
    is_constructor_15724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'is_constructor', False)
    # Calling is_constructor(args, kwargs) (line 124)
    is_constructor_call_result_15727 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), is_constructor_15724, *[name_15725], **kwargs_15726)
    
    # Assigning a type to the variable 'isconstructor' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'isconstructor', is_constructor_call_result_15727)
    
    # Assigning a Call to a Name (line 125):
    
    # Call to is_method(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'context' (line 125)
    context_15729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'context', False)
    # Processing the call keyword arguments (line 125)
    kwargs_15730 = {}
    # Getting the type of 'is_method' (line 125)
    is_method_15728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'is_method', False)
    # Calling is_method(args, kwargs) (line 125)
    is_method_call_result_15731 = invoke(stypy.reporting.localization.Localization(__file__, 125, 15), is_method_15728, *[context_15729], **kwargs_15730)
    
    # Assigning a type to the variable 'ismethod' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'ismethod', is_method_call_result_15731)
    
    # Assigning a Call to a Name (line 127):
    
    # Call to FunctionDef(...): (line 127)
    # Processing the call keyword arguments (line 127)
    kwargs_15734 = {}
    # Getting the type of 'ast' (line 127)
    ast_15732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'ast', False)
    # Obtaining the member 'FunctionDef' of a type (line 127)
    FunctionDef_15733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), ast_15732, 'FunctionDef')
    # Calling FunctionDef(args, kwargs) (line 127)
    FunctionDef_call_result_15735 = invoke(stypy.reporting.localization.Localization(__file__, 127, 19), FunctionDef_15733, *[], **kwargs_15734)
    
    # Assigning a type to the variable 'function_def' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'function_def', FunctionDef_call_result_15735)
    
    # Assigning a Name to a Attribute (line 128):
    # Getting the type of 'line' (line 128)
    line_15736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 26), 'line')
    # Getting the type of 'function_def' (line 128)
    function_def_15737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'function_def')
    # Setting the type of the member 'lineno' of a type (line 128)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 4), function_def_15737, 'lineno', line_15736)
    
    # Assigning a Name to a Attribute (line 129):
    # Getting the type of 'column' (line 129)
    column_15738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'column')
    # Getting the type of 'function_def' (line 129)
    function_def_15739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'function_def')
    # Setting the type of the member 'col_offset' of a type (line 129)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 4), function_def_15739, 'col_offset', column_15738)
    
    # Assigning a Name to a Attribute (line 130):
    # Getting the type of 'name' (line 130)
    name_15740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'name')
    # Getting the type of 'function_def' (line 130)
    function_def_15741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'function_def')
    # Setting the type of the member 'name' of a type (line 130)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 4), function_def_15741, 'name', name_15740)
    
    # Assigning a Name to a Attribute (line 132):
    # Getting the type of 'function_def_arguments' (line 132)
    function_def_arguments_15742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'function_def_arguments')
    # Getting the type of 'function_def' (line 132)
    function_def_15743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'function_def')
    # Setting the type of the member 'args' of a type (line 132)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 4), function_def_15743, 'args', function_def_arguments_15742)
    
    # Assigning a List to a Attribute (line 134):
    
    # Obtaining an instance of the builtin type 'list' (line 134)
    list_15744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 134)
    
    # Getting the type of 'function_def_arguments' (line 134)
    function_def_arguments_15745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'function_def_arguments')
    # Setting the type of the member 'args' of a type (line 134)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 4), function_def_arguments_15745, 'args', list_15744)
    # Getting the type of 'isconstructor' (line 136)
    isconstructor_15746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 7), 'isconstructor')
    # Testing if the type of an if condition is none (line 136)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 136, 4), isconstructor_15746):
        pass
    else:
        
        # Testing the type of an if condition (line 136)
        if_condition_15747 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 4), isconstructor_15746)
        # Assigning a type to the variable 'if_condition_15747' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'if_condition_15747', if_condition_15747)
        # SSA begins for if statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Call to create_Name(...): (line 137)
        # Processing the call arguments (line 137)
        str_15753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 74), 'str', 'type_of_self')
        # Processing the call keyword arguments (line 137)
        kwargs_15754 = {}
        # Getting the type of 'core_language_copy' (line 137)
        core_language_copy_15751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 43), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 137)
        create_Name_15752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 43), core_language_copy_15751, 'create_Name')
        # Calling create_Name(args, kwargs) (line 137)
        create_Name_call_result_15755 = invoke(stypy.reporting.localization.Localization(__file__, 137, 43), create_Name_15752, *[str_15753], **kwargs_15754)
        
        # Processing the call keyword arguments (line 137)
        kwargs_15756 = {}
        # Getting the type of 'function_def_arguments' (line 137)
        function_def_arguments_15748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'function_def_arguments', False)
        # Obtaining the member 'args' of a type (line 137)
        args_15749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), function_def_arguments_15748, 'args')
        # Obtaining the member 'append' of a type (line 137)
        append_15750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), args_15749, 'append')
        # Calling append(args, kwargs) (line 137)
        append_call_result_15757 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), append_15750, *[create_Name_call_result_15755], **kwargs_15756)
        
        # SSA join for if statement (line 136)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    # Getting the type of 'ismethod' (line 139)
    ismethod_15758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 7), 'ismethod')
    
    # Getting the type of 'isconstructor' (line 139)
    isconstructor_15759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'isconstructor')
    # Applying the 'not' unary operator (line 139)
    result_not__15760 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 20), 'not', isconstructor_15759)
    
    # Applying the binary operator 'and' (line 139)
    result_and_keyword_15761 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 7), 'and', ismethod_15758, result_not__15760)
    
    # Testing if the type of an if condition is none (line 139)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 139, 4), result_and_keyword_15761):
        pass
    else:
        
        # Testing the type of an if condition (line 139)
        if_condition_15762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 4), result_and_keyword_15761)
        # Assigning a type to the variable 'if_condition_15762' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'if_condition_15762', if_condition_15762)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Call to create_Name(...): (line 140)
        # Processing the call arguments (line 140)
        str_15768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 74), 'str', 'type_of_self')
        # Processing the call keyword arguments (line 140)
        kwargs_15769 = {}
        # Getting the type of 'core_language_copy' (line 140)
        core_language_copy_15766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 43), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 140)
        create_Name_15767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 43), core_language_copy_15766, 'create_Name')
        # Calling create_Name(args, kwargs) (line 140)
        create_Name_call_result_15770 = invoke(stypy.reporting.localization.Localization(__file__, 140, 43), create_Name_15767, *[str_15768], **kwargs_15769)
        
        # Processing the call keyword arguments (line 140)
        kwargs_15771 = {}
        # Getting the type of 'function_def_arguments' (line 140)
        function_def_arguments_15763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'function_def_arguments', False)
        # Obtaining the member 'args' of a type (line 140)
        args_15764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), function_def_arguments_15763, 'args')
        # Obtaining the member 'append' of a type (line 140)
        append_15765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), args_15764, 'append')
        # Calling append(args, kwargs) (line 140)
        append_call_result_15772 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), append_15765, *[create_Name_call_result_15770], **kwargs_15771)
        
        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to append(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'localization' (line 142)
    localization_15776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 39), 'localization', False)
    # Processing the call keyword arguments (line 142)
    kwargs_15777 = {}
    # Getting the type of 'function_def_arguments' (line 142)
    function_def_arguments_15773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'function_def_arguments', False)
    # Obtaining the member 'args' of a type (line 142)
    args_15774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 4), function_def_arguments_15773, 'args')
    # Obtaining the member 'append' of a type (line 142)
    append_15775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 4), args_15774, 'append')
    # Calling append(args, kwargs) (line 142)
    append_call_result_15778 = invoke(stypy.reporting.localization.Localization(__file__, 142, 4), append_15775, *[localization_15776], **kwargs_15777)
    
    
    # Assigning a Str to a Attribute (line 144):
    str_15779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 35), 'str', 'kwargs')
    # Getting the type of 'function_def_arguments' (line 144)
    function_def_arguments_15780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'function_def_arguments')
    # Setting the type of the member 'kwarg' of a type (line 144)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 4), function_def_arguments_15780, 'kwarg', str_15779)
    
    # Assigning a Str to a Attribute (line 145):
    str_15781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 36), 'str', 'varargs')
    # Getting the type of 'function_def_arguments' (line 145)
    function_def_arguments_15782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'function_def_arguments')
    # Setting the type of the member 'vararg' of a type (line 145)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 4), function_def_arguments_15782, 'vararg', str_15781)
    
    # Assigning a List to a Attribute (line 146):
    
    # Obtaining an instance of the builtin type 'list' (line 146)
    list_15783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 146)
    
    # Getting the type of 'function_def_arguments' (line 146)
    function_def_arguments_15784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'function_def_arguments')
    # Setting the type of the member 'defaults' of a type (line 146)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 4), function_def_arguments_15784, 'defaults', list_15783)
    
    # Call to is_iterable(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'decorators' (line 148)
    decorators_15787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 40), 'decorators', False)
    # Processing the call keyword arguments (line 148)
    kwargs_15788 = {}
    # Getting the type of 'data_structures_copy' (line 148)
    data_structures_copy_15785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'data_structures_copy', False)
    # Obtaining the member 'is_iterable' of a type (line 148)
    is_iterable_15786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 7), data_structures_copy_15785, 'is_iterable')
    # Calling is_iterable(args, kwargs) (line 148)
    is_iterable_call_result_15789 = invoke(stypy.reporting.localization.Localization(__file__, 148, 7), is_iterable_15786, *[decorators_15787], **kwargs_15788)
    
    # Testing if the type of an if condition is none (line 148)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 148, 4), is_iterable_call_result_15789):
        
        # Assigning a List to a Attribute (line 151):
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_15793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        # Getting the type of 'decorators' (line 151)
        decorators_15794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 39), 'decorators')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 38), list_15793, decorators_15794)
        
        # Getting the type of 'function_def' (line 151)
        function_def_15795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'function_def')
        # Setting the type of the member 'decorator_list' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), function_def_15795, 'decorator_list', list_15793)
    else:
        
        # Testing the type of an if condition (line 148)
        if_condition_15790 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), is_iterable_call_result_15789)
        # Assigning a type to the variable 'if_condition_15790' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_15790', if_condition_15790)
        # SSA begins for if statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 149):
        # Getting the type of 'decorators' (line 149)
        decorators_15791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 38), 'decorators')
        # Getting the type of 'function_def' (line 149)
        function_def_15792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'function_def')
        # Setting the type of the member 'decorator_list' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), function_def_15792, 'decorator_list', decorators_15791)
        # SSA branch for the else part of an if statement (line 148)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 151):
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_15793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        # Getting the type of 'decorators' (line 151)
        decorators_15794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 39), 'decorators')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 38), list_15793, decorators_15794)
        
        # Getting the type of 'function_def' (line 151)
        function_def_15795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'function_def')
        # Setting the type of the member 'decorator_list' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), function_def_15795, 'decorator_list', list_15793)
        # SSA join for if statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Attribute (line 153):
    
    # Obtaining an instance of the builtin type 'list' (line 153)
    list_15796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 153)
    
    # Getting the type of 'function_def' (line 153)
    function_def_15797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'function_def')
    # Setting the type of the member 'body' of a type (line 153)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), function_def_15797, 'body', list_15796)
    # Getting the type of 'function_def' (line 155)
    function_def_15798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'function_def')
    # Assigning a type to the variable 'stypy_return_type' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type', function_def_15798)
    
    # ################# End of 'create_function_def(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_function_def' in the type store
    # Getting the type of 'stypy_return_type' (line 107)
    stypy_return_type_15799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15799)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_function_def'
    return stypy_return_type_15799

# Assigning a type to the variable 'create_function_def' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'create_function_def', create_function_def)

@norecursion
def create_return(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_return'
    module_type_store = module_type_store.open_function_context('create_return', 158, 0, False)
    
    # Passed parameters checking function
    create_return.stypy_localization = localization
    create_return.stypy_type_of_self = None
    create_return.stypy_type_store = module_type_store
    create_return.stypy_function_name = 'create_return'
    create_return.stypy_param_names_list = ['value']
    create_return.stypy_varargs_param_name = None
    create_return.stypy_kwargs_param_name = None
    create_return.stypy_call_defaults = defaults
    create_return.stypy_call_varargs = varargs
    create_return.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_return', ['value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_return', localization, ['value'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_return(...)' code ##################

    str_15800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, (-1)), 'str', '\n    Creates an AST Return node\n    :param value: Value to return\n    :return: An AST Return node\n    ')
    
    # Assigning a Call to a Name (line 164):
    
    # Call to Return(...): (line 164)
    # Processing the call keyword arguments (line 164)
    kwargs_15803 = {}
    # Getting the type of 'ast' (line 164)
    ast_15801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'ast', False)
    # Obtaining the member 'Return' of a type (line 164)
    Return_15802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 11), ast_15801, 'Return')
    # Calling Return(args, kwargs) (line 164)
    Return_call_result_15804 = invoke(stypy.reporting.localization.Localization(__file__, 164, 11), Return_15802, *[], **kwargs_15803)
    
    # Assigning a type to the variable 'node' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'node', Return_call_result_15804)
    
    # Assigning a Name to a Attribute (line 165):
    # Getting the type of 'value' (line 165)
    value_15805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'value')
    # Getting the type of 'node' (line 165)
    node_15806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'node')
    # Setting the type of the member 'value' of a type (line 165)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), node_15806, 'value', value_15805)
    # Getting the type of 'node' (line 167)
    node_15807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'node')
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type', node_15807)
    
    # ################# End of 'create_return(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_return' in the type store
    # Getting the type of 'stypy_return_type' (line 158)
    stypy_return_type_15808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15808)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_return'
    return stypy_return_type_15808

# Assigning a type to the variable 'create_return' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'create_return', create_return)

@norecursion
def obtain_arg_list(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 170)
    False_15809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 35), 'False')
    # Getting the type of 'False' (line 170)
    False_15810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 57), 'False')
    defaults = [False_15809, False_15810]
    # Create a new context for function 'obtain_arg_list'
    module_type_store = module_type_store.open_function_context('obtain_arg_list', 170, 0, False)
    
    # Passed parameters checking function
    obtain_arg_list.stypy_localization = localization
    obtain_arg_list.stypy_type_of_self = None
    obtain_arg_list.stypy_type_store = module_type_store
    obtain_arg_list.stypy_function_name = 'obtain_arg_list'
    obtain_arg_list.stypy_param_names_list = ['args', 'ismethod', 'isstaticmethod']
    obtain_arg_list.stypy_varargs_param_name = None
    obtain_arg_list.stypy_kwargs_param_name = None
    obtain_arg_list.stypy_call_defaults = defaults
    obtain_arg_list.stypy_call_varargs = varargs
    obtain_arg_list.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'obtain_arg_list', ['args', 'ismethod', 'isstaticmethod'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'obtain_arg_list', localization, ['args', 'ismethod', 'isstaticmethod'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'obtain_arg_list(...)' code ##################

    str_15811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, (-1)), 'str', '\n    Creates an AST List node with the names of the arguments passed to a function\n    :param args: Arguments\n    :param ismethod: Whether to count the first argument (self) or not\n    :return: An AST List\n    ')
    
    # Assigning a Call to a Name (line 177):
    
    # Call to List(...): (line 177)
    # Processing the call keyword arguments (line 177)
    kwargs_15814 = {}
    # Getting the type of 'ast' (line 177)
    ast_15812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'ast', False)
    # Obtaining the member 'List' of a type (line 177)
    List_15813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 15), ast_15812, 'List')
    # Calling List(args, kwargs) (line 177)
    List_call_result_15815 = invoke(stypy.reporting.localization.Localization(__file__, 177, 15), List_15813, *[], **kwargs_15814)
    
    # Assigning a type to the variable 'arg_list' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'arg_list', List_call_result_15815)
    
    # Assigning a List to a Attribute (line 179):
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_15816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    
    # Getting the type of 'arg_list' (line 179)
    arg_list_15817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'arg_list')
    # Setting the type of the member 'elts' of a type (line 179)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 4), arg_list_15817, 'elts', list_15816)
    
    # Evaluating a boolean operation
    # Getting the type of 'ismethod' (line 180)
    ismethod_15818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 7), 'ismethod')
    
    # Getting the type of 'isstaticmethod' (line 180)
    isstaticmethod_15819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 24), 'isstaticmethod')
    # Applying the 'not' unary operator (line 180)
    result_not__15820 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 20), 'not', isstaticmethod_15819)
    
    # Applying the binary operator 'and' (line 180)
    result_and_keyword_15821 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 7), 'and', ismethod_15818, result_not__15820)
    
    # Testing if the type of an if condition is none (line 180)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 180, 4), result_and_keyword_15821):
        
        # Assigning a Attribute to a Name (line 183):
        # Getting the type of 'args' (line 183)
        args_15829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'args')
        # Obtaining the member 'args' of a type (line 183)
        args_15830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 28), args_15829, 'args')
        # Assigning a type to the variable 'arg_list_contents' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'arg_list_contents', args_15830)
    else:
        
        # Testing the type of an if condition (line 180)
        if_condition_15822 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 4), result_and_keyword_15821)
        # Assigning a type to the variable 'if_condition_15822' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'if_condition_15822', if_condition_15822)
        # SSA begins for if statement (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 181):
        
        # Obtaining the type of the subscript
        int_15823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 38), 'int')
        slice_15824 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 181, 28), int_15823, None, None)
        # Getting the type of 'args' (line 181)
        args_15825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'args')
        # Obtaining the member 'args' of a type (line 181)
        args_15826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 28), args_15825, 'args')
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___15827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 28), args_15826, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_15828 = invoke(stypy.reporting.localization.Localization(__file__, 181, 28), getitem___15827, slice_15824)
        
        # Assigning a type to the variable 'arg_list_contents' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'arg_list_contents', subscript_call_result_15828)
        # SSA branch for the else part of an if statement (line 180)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 183):
        # Getting the type of 'args' (line 183)
        args_15829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'args')
        # Obtaining the member 'args' of a type (line 183)
        args_15830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 28), args_15829, 'args')
        # Assigning a type to the variable 'arg_list_contents' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'arg_list_contents', args_15830)
        # SSA join for if statement (line 180)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'arg_list_contents' (line 185)
    arg_list_contents_15831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'arg_list_contents')
    # Assigning a type to the variable 'arg_list_contents_15831' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'arg_list_contents_15831', arg_list_contents_15831)
    # Testing if the for loop is going to be iterated (line 185)
    # Testing the type of a for loop iterable (line 185)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 185, 4), arg_list_contents_15831)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 185, 4), arg_list_contents_15831):
        # Getting the type of the for loop variable (line 185)
        for_loop_var_15832 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 185, 4), arg_list_contents_15831)
        # Assigning a type to the variable 'arg' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'arg', for_loop_var_15832)
        # SSA begins for a for statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Call to create_str(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'arg' (line 186)
        arg_15838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 59), 'arg', False)
        # Obtaining the member 'id' of a type (line 186)
        id_15839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 59), arg_15838, 'id')
        # Processing the call keyword arguments (line 186)
        kwargs_15840 = {}
        # Getting the type of 'core_language_copy' (line 186)
        core_language_copy_15836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 29), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 186)
        create_str_15837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 29), core_language_copy_15836, 'create_str')
        # Calling create_str(args, kwargs) (line 186)
        create_str_call_result_15841 = invoke(stypy.reporting.localization.Localization(__file__, 186, 29), create_str_15837, *[id_15839], **kwargs_15840)
        
        # Processing the call keyword arguments (line 186)
        kwargs_15842 = {}
        # Getting the type of 'arg_list' (line 186)
        arg_list_15833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'arg_list', False)
        # Obtaining the member 'elts' of a type (line 186)
        elts_15834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), arg_list_15833, 'elts')
        # Obtaining the member 'append' of a type (line 186)
        append_15835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), elts_15834, 'append')
        # Calling append(args, kwargs) (line 186)
        append_call_result_15843 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), append_15835, *[create_str_call_result_15841], **kwargs_15842)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'arg_list' (line 188)
    arg_list_15844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'arg_list')
    # Assigning a type to the variable 'stypy_return_type' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'stypy_return_type', arg_list_15844)
    
    # ################# End of 'obtain_arg_list(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'obtain_arg_list' in the type store
    # Getting the type of 'stypy_return_type' (line 170)
    stypy_return_type_15845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15845)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'obtain_arg_list'
    return stypy_return_type_15845

# Assigning a type to the variable 'obtain_arg_list' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'obtain_arg_list', obtain_arg_list)

@norecursion
def create_stacktrace_push(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_stacktrace_push'
    module_type_store = module_type_store.open_function_context('create_stacktrace_push', 191, 0, False)
    
    # Passed parameters checking function
    create_stacktrace_push.stypy_localization = localization
    create_stacktrace_push.stypy_type_of_self = None
    create_stacktrace_push.stypy_type_store = module_type_store
    create_stacktrace_push.stypy_function_name = 'create_stacktrace_push'
    create_stacktrace_push.stypy_param_names_list = ['func_name', 'declared_arguments']
    create_stacktrace_push.stypy_varargs_param_name = None
    create_stacktrace_push.stypy_kwargs_param_name = None
    create_stacktrace_push.stypy_call_defaults = defaults
    create_stacktrace_push.stypy_call_varargs = varargs
    create_stacktrace_push.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_stacktrace_push', ['func_name', 'declared_arguments'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_stacktrace_push', localization, ['func_name', 'declared_arguments'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_stacktrace_push(...)' code ##################

    str_15846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, (-1)), 'str', '\n    Creates an AST Node that model the call to the localitazion.set_stack_trace method\n\n    :param func_name: Name of the function that will do the push to the stack trace\n    :param declared_arguments: Arguments of the call\n    :return: An AST Expr node\n    ')
    
    # Assigning a Call to a Name (line 200):
    
    # Call to create_attribute(...): (line 200)
    # Processing the call arguments (line 200)
    str_15849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 52), 'str', 'localization')
    str_15850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 68), 'str', 'set_stack_trace')
    # Processing the call keyword arguments (line 200)
    kwargs_15851 = {}
    # Getting the type of 'core_language_copy' (line 200)
    core_language_copy_15847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 200)
    create_attribute_15848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 16), core_language_copy_15847, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 200)
    create_attribute_call_result_15852 = invoke(stypy.reporting.localization.Localization(__file__, 200, 16), create_attribute_15848, *[str_15849, str_15850], **kwargs_15851)
    
    # Assigning a type to the variable 'attribute' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'attribute', create_attribute_call_result_15852)
    
    # Assigning a Call to a Name (line 201):
    
    # Call to create_Name(...): (line 201)
    # Processing the call arguments (line 201)
    str_15855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 51), 'str', 'arguments')
    # Processing the call keyword arguments (line 201)
    kwargs_15856 = {}
    # Getting the type of 'core_language_copy' (line 201)
    core_language_copy_15853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 20), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 201)
    create_Name_15854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 20), core_language_copy_15853, 'create_Name')
    # Calling create_Name(args, kwargs) (line 201)
    create_Name_call_result_15857 = invoke(stypy.reporting.localization.Localization(__file__, 201, 20), create_Name_15854, *[str_15855], **kwargs_15856)
    
    # Assigning a type to the variable 'arguments_var' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'arguments_var', create_Name_call_result_15857)
    
    # Assigning a Call to a Name (line 202):
    
    # Call to create_call(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'attribute' (line 202)
    attribute_15859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 34), 'attribute', False)
    
    # Obtaining an instance of the builtin type 'list' (line 202)
    list_15860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 45), 'list')
    # Adding type elements to the builtin type 'list' instance (line 202)
    # Adding element type (line 202)
    
    # Call to create_str(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'func_name' (line 202)
    func_name_15863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 76), 'func_name', False)
    # Processing the call keyword arguments (line 202)
    kwargs_15864 = {}
    # Getting the type of 'core_language_copy' (line 202)
    core_language_copy_15861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 46), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 202)
    create_str_15862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 46), core_language_copy_15861, 'create_str')
    # Calling create_str(args, kwargs) (line 202)
    create_str_call_result_15865 = invoke(stypy.reporting.localization.Localization(__file__, 202, 46), create_str_15862, *[func_name_15863], **kwargs_15864)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 45), list_15860, create_str_call_result_15865)
    # Adding element type (line 202)
    # Getting the type of 'declared_arguments' (line 202)
    declared_arguments_15866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 88), 'declared_arguments', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 45), list_15860, declared_arguments_15866)
    # Adding element type (line 202)
    # Getting the type of 'arguments_var' (line 202)
    arguments_var_15867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 108), 'arguments_var', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 45), list_15860, arguments_var_15867)
    
    # Processing the call keyword arguments (line 202)
    kwargs_15868 = {}
    # Getting the type of 'create_call' (line 202)
    create_call_15858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'create_call', False)
    # Calling create_call(args, kwargs) (line 202)
    create_call_call_result_15869 = invoke(stypy.reporting.localization.Localization(__file__, 202, 22), create_call_15858, *[attribute_15859, list_15860], **kwargs_15868)
    
    # Assigning a type to the variable 'stack_push_call' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stack_push_call', create_call_call_result_15869)
    
    # Assigning a Call to a Name (line 203):
    
    # Call to Expr(...): (line 203)
    # Processing the call keyword arguments (line 203)
    kwargs_15872 = {}
    # Getting the type of 'ast' (line 203)
    ast_15870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 17), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 203)
    Expr_15871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 17), ast_15870, 'Expr')
    # Calling Expr(args, kwargs) (line 203)
    Expr_call_result_15873 = invoke(stypy.reporting.localization.Localization(__file__, 203, 17), Expr_15871, *[], **kwargs_15872)
    
    # Assigning a type to the variable 'stack_push' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stack_push', Expr_call_result_15873)
    
    # Assigning a Name to a Attribute (line 204):
    # Getting the type of 'stack_push_call' (line 204)
    stack_push_call_15874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 23), 'stack_push_call')
    # Getting the type of 'stack_push' (line 204)
    stack_push_15875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stack_push')
    # Setting the type of the member 'value' of a type (line 204)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 4), stack_push_15875, 'value', stack_push_call_15874)
    # Getting the type of 'stack_push' (line 206)
    stack_push_15876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'stack_push')
    # Assigning a type to the variable 'stypy_return_type' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'stypy_return_type', stack_push_15876)
    
    # ################# End of 'create_stacktrace_push(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_stacktrace_push' in the type store
    # Getting the type of 'stypy_return_type' (line 191)
    stypy_return_type_15877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15877)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_stacktrace_push'
    return stypy_return_type_15877

# Assigning a type to the variable 'create_stacktrace_push' (line 191)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'create_stacktrace_push', create_stacktrace_push)

@norecursion
def create_stacktrace_pop(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_stacktrace_pop'
    module_type_store = module_type_store.open_function_context('create_stacktrace_pop', 209, 0, False)
    
    # Passed parameters checking function
    create_stacktrace_pop.stypy_localization = localization
    create_stacktrace_pop.stypy_type_of_self = None
    create_stacktrace_pop.stypy_type_store = module_type_store
    create_stacktrace_pop.stypy_function_name = 'create_stacktrace_pop'
    create_stacktrace_pop.stypy_param_names_list = []
    create_stacktrace_pop.stypy_varargs_param_name = None
    create_stacktrace_pop.stypy_kwargs_param_name = None
    create_stacktrace_pop.stypy_call_defaults = defaults
    create_stacktrace_pop.stypy_call_varargs = varargs
    create_stacktrace_pop.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_stacktrace_pop', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_stacktrace_pop', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_stacktrace_pop(...)' code ##################

    str_15878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, (-1)), 'str', '\n    Creates an AST Node that model the call to the localitazion.unset_stack_trace method\n\n    :return: An AST Expr node\n    ')
    
    # Assigning a Call to a Name (line 216):
    
    # Call to create_attribute(...): (line 216)
    # Processing the call arguments (line 216)
    str_15881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 52), 'str', 'localization')
    str_15882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 68), 'str', 'unset_stack_trace')
    # Processing the call keyword arguments (line 216)
    kwargs_15883 = {}
    # Getting the type of 'core_language_copy' (line 216)
    core_language_copy_15879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 216)
    create_attribute_15880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), core_language_copy_15879, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 216)
    create_attribute_call_result_15884 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), create_attribute_15880, *[str_15881, str_15882], **kwargs_15883)
    
    # Assigning a type to the variable 'attribute' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'attribute', create_attribute_call_result_15884)
    
    # Assigning a Call to a Name (line 217):
    
    # Call to create_call(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'attribute' (line 217)
    attribute_15886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 33), 'attribute', False)
    
    # Obtaining an instance of the builtin type 'list' (line 217)
    list_15887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 217)
    
    # Processing the call keyword arguments (line 217)
    kwargs_15888 = {}
    # Getting the type of 'create_call' (line 217)
    create_call_15885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'create_call', False)
    # Calling create_call(args, kwargs) (line 217)
    create_call_call_result_15889 = invoke(stypy.reporting.localization.Localization(__file__, 217, 21), create_call_15885, *[attribute_15886, list_15887], **kwargs_15888)
    
    # Assigning a type to the variable 'stack_pop_call' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stack_pop_call', create_call_call_result_15889)
    
    # Assigning a Call to a Name (line 218):
    
    # Call to Expr(...): (line 218)
    # Processing the call keyword arguments (line 218)
    kwargs_15892 = {}
    # Getting the type of 'ast' (line 218)
    ast_15890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 218)
    Expr_15891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 16), ast_15890, 'Expr')
    # Calling Expr(args, kwargs) (line 218)
    Expr_call_result_15893 = invoke(stypy.reporting.localization.Localization(__file__, 218, 16), Expr_15891, *[], **kwargs_15892)
    
    # Assigning a type to the variable 'stack_pop' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'stack_pop', Expr_call_result_15893)
    
    # Assigning a Name to a Attribute (line 219):
    # Getting the type of 'stack_pop_call' (line 219)
    stack_pop_call_15894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 22), 'stack_pop_call')
    # Getting the type of 'stack_pop' (line 219)
    stack_pop_15895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stack_pop')
    # Setting the type of the member 'value' of a type (line 219)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 4), stack_pop_15895, 'value', stack_pop_call_15894)
    # Getting the type of 'stack_pop' (line 221)
    stack_pop_15896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'stack_pop')
    # Assigning a type to the variable 'stypy_return_type' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type', stack_pop_15896)
    
    # ################# End of 'create_stacktrace_pop(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_stacktrace_pop' in the type store
    # Getting the type of 'stypy_return_type' (line 209)
    stypy_return_type_15897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15897)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_stacktrace_pop'
    return stypy_return_type_15897

# Assigning a type to the variable 'create_stacktrace_pop' (line 209)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'create_stacktrace_pop', create_stacktrace_pop)

@norecursion
def create_context_set(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_context_set'
    module_type_store = module_type_store.open_function_context('create_context_set', 224, 0, False)
    
    # Passed parameters checking function
    create_context_set.stypy_localization = localization
    create_context_set.stypy_type_of_self = None
    create_context_set.stypy_type_store = module_type_store
    create_context_set.stypy_function_name = 'create_context_set'
    create_context_set.stypy_param_names_list = ['func_name', 'lineno', 'col_offset']
    create_context_set.stypy_varargs_param_name = None
    create_context_set.stypy_kwargs_param_name = None
    create_context_set.stypy_call_defaults = defaults
    create_context_set.stypy_call_varargs = varargs
    create_context_set.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_context_set', ['func_name', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_context_set', localization, ['func_name', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_context_set(...)' code ##################

    str_15898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, (-1)), 'str', '\n    Creates an AST Node that model the call to the type_store.set_context method\n\n    :param func_name: Name of the function that will do the push to the stack trace\n    :param lineno: Line\n    :param col_offset: Column\n    :return: An AST Expr node\n    ')
    
    # Assigning a Call to a Name (line 233):
    
    # Call to create_attribute(...): (line 233)
    # Processing the call arguments (line 233)
    str_15901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 52), 'str', 'type_store')
    str_15902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 66), 'str', 'set_context')
    # Processing the call keyword arguments (line 233)
    kwargs_15903 = {}
    # Getting the type of 'core_language_copy' (line 233)
    core_language_copy_15899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 233)
    create_attribute_15900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 16), core_language_copy_15899, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 233)
    create_attribute_call_result_15904 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), create_attribute_15900, *[str_15901, str_15902], **kwargs_15903)
    
    # Assigning a type to the variable 'attribute' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'attribute', create_attribute_call_result_15904)
    
    # Assigning a Call to a Name (line 234):
    
    # Call to create_call(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'attribute' (line 234)
    attribute_15906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 35), 'attribute', False)
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_15907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    # Adding element type (line 234)
    
    # Call to create_str(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'func_name' (line 234)
    func_name_15910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 77), 'func_name', False)
    # Processing the call keyword arguments (line 234)
    kwargs_15911 = {}
    # Getting the type of 'core_language_copy' (line 234)
    core_language_copy_15908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 47), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 234)
    create_str_15909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 47), core_language_copy_15908, 'create_str')
    # Calling create_str(args, kwargs) (line 234)
    create_str_call_result_15912 = invoke(stypy.reporting.localization.Localization(__file__, 234, 47), create_str_15909, *[func_name_15910], **kwargs_15911)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 46), list_15907, create_str_call_result_15912)
    # Adding element type (line 234)
    
    # Call to create_num(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'lineno' (line 235)
    lineno_15915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 77), 'lineno', False)
    # Processing the call keyword arguments (line 235)
    kwargs_15916 = {}
    # Getting the type of 'core_language_copy' (line 235)
    core_language_copy_15913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 47), 'core_language_copy', False)
    # Obtaining the member 'create_num' of a type (line 235)
    create_num_15914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 47), core_language_copy_15913, 'create_num')
    # Calling create_num(args, kwargs) (line 235)
    create_num_call_result_15917 = invoke(stypy.reporting.localization.Localization(__file__, 235, 47), create_num_15914, *[lineno_15915], **kwargs_15916)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 46), list_15907, create_num_call_result_15917)
    # Adding element type (line 234)
    
    # Call to create_num(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'col_offset' (line 236)
    col_offset_15920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 77), 'col_offset', False)
    # Processing the call keyword arguments (line 236)
    kwargs_15921 = {}
    # Getting the type of 'core_language_copy' (line 236)
    core_language_copy_15918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 47), 'core_language_copy', False)
    # Obtaining the member 'create_num' of a type (line 236)
    create_num_15919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 47), core_language_copy_15918, 'create_num')
    # Calling create_num(args, kwargs) (line 236)
    create_num_call_result_15922 = invoke(stypy.reporting.localization.Localization(__file__, 236, 47), create_num_15919, *[col_offset_15920], **kwargs_15921)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 46), list_15907, create_num_call_result_15922)
    
    # Processing the call keyword arguments (line 234)
    kwargs_15923 = {}
    # Getting the type of 'create_call' (line 234)
    create_call_15905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 23), 'create_call', False)
    # Calling create_call(args, kwargs) (line 234)
    create_call_call_result_15924 = invoke(stypy.reporting.localization.Localization(__file__, 234, 23), create_call_15905, *[attribute_15906, list_15907], **kwargs_15923)
    
    # Assigning a type to the variable 'context_set_call' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'context_set_call', create_call_call_result_15924)
    
    # Assigning a Call to a Name (line 237):
    
    # Call to Expr(...): (line 237)
    # Processing the call keyword arguments (line 237)
    kwargs_15927 = {}
    # Getting the type of 'ast' (line 237)
    ast_15925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 18), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 237)
    Expr_15926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 18), ast_15925, 'Expr')
    # Calling Expr(args, kwargs) (line 237)
    Expr_call_result_15928 = invoke(stypy.reporting.localization.Localization(__file__, 237, 18), Expr_15926, *[], **kwargs_15927)
    
    # Assigning a type to the variable 'context_set' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'context_set', Expr_call_result_15928)
    
    # Assigning a Name to a Attribute (line 238):
    # Getting the type of 'context_set_call' (line 238)
    context_set_call_15929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'context_set_call')
    # Getting the type of 'context_set' (line 238)
    context_set_15930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'context_set')
    # Setting the type of the member 'value' of a type (line 238)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 4), context_set_15930, 'value', context_set_call_15929)
    # Getting the type of 'context_set' (line 240)
    context_set_15931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'context_set')
    # Assigning a type to the variable 'stypy_return_type' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type', context_set_15931)
    
    # ################# End of 'create_context_set(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_context_set' in the type store
    # Getting the type of 'stypy_return_type' (line 224)
    stypy_return_type_15932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15932)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_context_set'
    return stypy_return_type_15932

# Assigning a type to the variable 'create_context_set' (line 224)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'create_context_set', create_context_set)

@norecursion
def create_context_unset(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_context_unset'
    module_type_store = module_type_store.open_function_context('create_context_unset', 243, 0, False)
    
    # Passed parameters checking function
    create_context_unset.stypy_localization = localization
    create_context_unset.stypy_type_of_self = None
    create_context_unset.stypy_type_store = module_type_store
    create_context_unset.stypy_function_name = 'create_context_unset'
    create_context_unset.stypy_param_names_list = []
    create_context_unset.stypy_varargs_param_name = None
    create_context_unset.stypy_kwargs_param_name = None
    create_context_unset.stypy_call_defaults = defaults
    create_context_unset.stypy_call_varargs = varargs
    create_context_unset.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_context_unset', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_context_unset', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_context_unset(...)' code ##################

    str_15933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, (-1)), 'str', '\n    Creates an AST Node that model the call to the type_store.unset_context method\n\n    :return: An AST Expr node\n    ')
    
    # Assigning a Call to a Name (line 250):
    
    # Call to create_attribute(...): (line 250)
    # Processing the call arguments (line 250)
    str_15936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 52), 'str', 'type_store')
    str_15937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 66), 'str', 'unset_context')
    # Processing the call keyword arguments (line 250)
    kwargs_15938 = {}
    # Getting the type of 'core_language_copy' (line 250)
    core_language_copy_15934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 250)
    create_attribute_15935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), core_language_copy_15934, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 250)
    create_attribute_call_result_15939 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), create_attribute_15935, *[str_15936, str_15937], **kwargs_15938)
    
    # Assigning a type to the variable 'attribute' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'attribute', create_attribute_call_result_15939)
    
    # Assigning a Call to a Name (line 251):
    
    # Call to create_call(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'attribute' (line 251)
    attribute_15941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 37), 'attribute', False)
    
    # Obtaining an instance of the builtin type 'list' (line 251)
    list_15942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 48), 'list')
    # Adding type elements to the builtin type 'list' instance (line 251)
    
    # Processing the call keyword arguments (line 251)
    kwargs_15943 = {}
    # Getting the type of 'create_call' (line 251)
    create_call_15940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 25), 'create_call', False)
    # Calling create_call(args, kwargs) (line 251)
    create_call_call_result_15944 = invoke(stypy.reporting.localization.Localization(__file__, 251, 25), create_call_15940, *[attribute_15941, list_15942], **kwargs_15943)
    
    # Assigning a type to the variable 'context_unset_call' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'context_unset_call', create_call_call_result_15944)
    
    # Assigning a Call to a Name (line 252):
    
    # Call to Expr(...): (line 252)
    # Processing the call keyword arguments (line 252)
    kwargs_15947 = {}
    # Getting the type of 'ast' (line 252)
    ast_15945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 20), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 252)
    Expr_15946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 20), ast_15945, 'Expr')
    # Calling Expr(args, kwargs) (line 252)
    Expr_call_result_15948 = invoke(stypy.reporting.localization.Localization(__file__, 252, 20), Expr_15946, *[], **kwargs_15947)
    
    # Assigning a type to the variable 'context_unset' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'context_unset', Expr_call_result_15948)
    
    # Assigning a Name to a Attribute (line 253):
    # Getting the type of 'context_unset_call' (line 253)
    context_unset_call_15949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), 'context_unset_call')
    # Getting the type of 'context_unset' (line 253)
    context_unset_15950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'context_unset')
    # Setting the type of the member 'value' of a type (line 253)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 4), context_unset_15950, 'value', context_unset_call_15949)
    # Getting the type of 'context_unset' (line 255)
    context_unset_15951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 11), 'context_unset')
    # Assigning a type to the variable 'stypy_return_type' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'stypy_return_type', context_unset_15951)
    
    # ################# End of 'create_context_unset(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_context_unset' in the type store
    # Getting the type of 'stypy_return_type' (line 243)
    stypy_return_type_15952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15952)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_context_unset'
    return stypy_return_type_15952

# Assigning a type to the variable 'create_context_unset' (line 243)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'create_context_unset', create_context_unset)

@norecursion
def create_arg_number_test(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'list' (line 258)
    list_15953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 54), 'list')
    # Adding type elements to the builtin type 'list' instance (line 258)
    
    defaults = [list_15953]
    # Create a new context for function 'create_arg_number_test'
    module_type_store = module_type_store.open_function_context('create_arg_number_test', 258, 0, False)
    
    # Passed parameters checking function
    create_arg_number_test.stypy_localization = localization
    create_arg_number_test.stypy_type_of_self = None
    create_arg_number_test.stypy_type_store = module_type_store
    create_arg_number_test.stypy_function_name = 'create_arg_number_test'
    create_arg_number_test.stypy_param_names_list = ['function_def_node', 'context']
    create_arg_number_test.stypy_varargs_param_name = None
    create_arg_number_test.stypy_kwargs_param_name = None
    create_arg_number_test.stypy_call_defaults = defaults
    create_arg_number_test.stypy_call_varargs = varargs
    create_arg_number_test.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_arg_number_test', ['function_def_node', 'context'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_arg_number_test', localization, ['function_def_node', 'context'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_arg_number_test(...)' code ##################

    str_15954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'str', '\n    Creates an AST Node that model the call to the process_argument_values method. This method is used to check\n    the parameters passed to a function/method in a type inference program\n\n    :param function_def_node: AST Node with the function definition\n    :param context: Context passed to the call\n    :return: List of AST nodes that perform the call to the mentioned function and make the necessary tests once it\n    is called\n    ')
    
    # Assigning a Call to a Name (line 268):
    
    # Call to create_Name(...): (line 268)
    # Processing the call arguments (line 268)
    str_15957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 53), 'str', 'arguments')
    # Getting the type of 'False' (line 268)
    False_15958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 66), 'False', False)
    # Processing the call keyword arguments (line 268)
    kwargs_15959 = {}
    # Getting the type of 'core_language_copy' (line 268)
    core_language_copy_15955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 268)
    create_Name_15956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 22), core_language_copy_15955, 'create_Name')
    # Calling create_Name(args, kwargs) (line 268)
    create_Name_call_result_15960 = invoke(stypy.reporting.localization.Localization(__file__, 268, 22), create_Name_15956, *[str_15957, False_15958], **kwargs_15959)
    
    # Assigning a type to the variable 'args_test_resul' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'args_test_resul', create_Name_call_result_15960)
    
    # Assigning a Call to a Name (line 271):
    
    # Call to create_Name(...): (line 271)
    # Processing the call arguments (line 271)
    str_15963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 42), 'str', 'process_argument_values')
    # Processing the call keyword arguments (line 271)
    kwargs_15964 = {}
    # Getting the type of 'core_language_copy' (line 271)
    core_language_copy_15961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 11), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 271)
    create_Name_15962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 11), core_language_copy_15961, 'create_Name')
    # Calling create_Name(args, kwargs) (line 271)
    create_Name_call_result_15965 = invoke(stypy.reporting.localization.Localization(__file__, 271, 11), create_Name_15962, *[str_15963], **kwargs_15964)
    
    # Assigning a type to the variable 'func' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'func', create_Name_call_result_15965)
    
    # Assigning a Call to a Name (line 273):
    
    # Call to create_Name(...): (line 273)
    # Processing the call arguments (line 273)
    str_15968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 54), 'str', 'localization')
    # Processing the call keyword arguments (line 273)
    kwargs_15969 = {}
    # Getting the type of 'core_language_copy' (line 273)
    core_language_copy_15966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 23), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 273)
    create_Name_15967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 23), core_language_copy_15966, 'create_Name')
    # Calling create_Name(args, kwargs) (line 273)
    create_Name_call_result_15970 = invoke(stypy.reporting.localization.Localization(__file__, 273, 23), create_Name_15967, *[str_15968], **kwargs_15969)
    
    # Assigning a type to the variable 'localization_arg' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'localization_arg', create_Name_call_result_15970)
    
    # Assigning a Call to a Name (line 274):
    
    # Call to create_Name(...): (line 274)
    # Processing the call arguments (line 274)
    str_15973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 52), 'str', 'type_store')
    # Processing the call keyword arguments (line 274)
    kwargs_15974 = {}
    # Getting the type of 'core_language_copy' (line 274)
    core_language_copy_15971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 21), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 274)
    create_Name_15972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 21), core_language_copy_15971, 'create_Name')
    # Calling create_Name(args, kwargs) (line 274)
    create_Name_call_result_15975 = invoke(stypy.reporting.localization.Localization(__file__, 274, 21), create_Name_15972, *[str_15973], **kwargs_15974)
    
    # Assigning a type to the variable 'type_store_arg' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'type_store_arg', create_Name_call_result_15975)
    
    # Call to is_method(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'context' (line 278)
    context_15977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 17), 'context', False)
    # Processing the call keyword arguments (line 278)
    kwargs_15978 = {}
    # Getting the type of 'is_method' (line 278)
    is_method_15976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 7), 'is_method', False)
    # Calling is_method(args, kwargs) (line 278)
    is_method_call_result_15979 = invoke(stypy.reporting.localization.Localization(__file__, 278, 7), is_method_15976, *[context_15977], **kwargs_15978)
    
    # Testing if the type of an if condition is none (line 278)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 278, 4), is_method_call_result_15979):
        
        # Assigning a Call to a Name (line 282):
        
        # Call to create_str(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'function_def_node' (line 282)
        function_def_node_16002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 58), 'function_def_node', False)
        # Obtaining the member 'name' of a type (line 282)
        name_16003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 58), function_def_node_16002, 'name')
        # Processing the call keyword arguments (line 282)
        kwargs_16004 = {}
        # Getting the type of 'core_language_copy' (line 282)
        core_language_copy_16000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 28), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 282)
        create_str_16001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 28), core_language_copy_16000, 'create_str')
        # Calling create_str(args, kwargs) (line 282)
        create_str_call_result_16005 = invoke(stypy.reporting.localization.Localization(__file__, 282, 28), create_str_16001, *[name_16003], **kwargs_16004)
        
        # Assigning a type to the variable 'function_name_arg' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'function_name_arg', create_str_call_result_16005)
        
        # Assigning a Call to a Name (line 283):
        
        # Call to create_Name(...): (line 283)
        # Processing the call arguments (line 283)
        str_16008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 58), 'str', 'None')
        # Processing the call keyword arguments (line 283)
        kwargs_16009 = {}
        # Getting the type of 'core_language_copy' (line 283)
        core_language_copy_16006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 283)
        create_Name_16007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 27), core_language_copy_16006, 'create_Name')
        # Calling create_Name(args, kwargs) (line 283)
        create_Name_call_result_16010 = invoke(stypy.reporting.localization.Localization(__file__, 283, 27), create_Name_16007, *[str_16008], **kwargs_16009)
        
        # Assigning a type to the variable 'type_of_self_arg' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'type_of_self_arg', create_Name_call_result_16010)
    else:
        
        # Testing the type of an if condition (line 278)
        if_condition_15980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 4), is_method_call_result_15979)
        # Assigning a type to the variable 'if_condition_15980' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'if_condition_15980', if_condition_15980)
        # SSA begins for if statement (line 278)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 279):
        
        # Call to create_str(...): (line 279)
        # Processing the call arguments (line 279)
        
        # Obtaining the type of the subscript
        int_15983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 66), 'int')
        # Getting the type of 'context' (line 279)
        context_15984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 58), 'context', False)
        # Obtaining the member '__getitem__' of a type (line 279)
        getitem___15985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 58), context_15984, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 279)
        subscript_call_result_15986 = invoke(stypy.reporting.localization.Localization(__file__, 279, 58), getitem___15985, int_15983)
        
        # Obtaining the member 'name' of a type (line 279)
        name_15987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 58), subscript_call_result_15986, 'name')
        str_15988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 77), 'str', '.')
        # Applying the binary operator '+' (line 279)
        result_add_15989 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 58), '+', name_15987, str_15988)
        
        # Getting the type of 'function_def_node' (line 279)
        function_def_node_15990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 83), 'function_def_node', False)
        # Obtaining the member 'name' of a type (line 279)
        name_15991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 83), function_def_node_15990, 'name')
        # Applying the binary operator '+' (line 279)
        result_add_15992 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 81), '+', result_add_15989, name_15991)
        
        # Processing the call keyword arguments (line 279)
        kwargs_15993 = {}
        # Getting the type of 'core_language_copy' (line 279)
        core_language_copy_15981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 28), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 279)
        create_str_15982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 28), core_language_copy_15981, 'create_str')
        # Calling create_str(args, kwargs) (line 279)
        create_str_call_result_15994 = invoke(stypy.reporting.localization.Localization(__file__, 279, 28), create_str_15982, *[result_add_15992], **kwargs_15993)
        
        # Assigning a type to the variable 'function_name_arg' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'function_name_arg', create_str_call_result_15994)
        
        # Assigning a Call to a Name (line 280):
        
        # Call to create_Name(...): (line 280)
        # Processing the call arguments (line 280)
        str_15997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 58), 'str', 'type_of_self')
        # Processing the call keyword arguments (line 280)
        kwargs_15998 = {}
        # Getting the type of 'core_language_copy' (line 280)
        core_language_copy_15995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 27), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 280)
        create_Name_15996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 27), core_language_copy_15995, 'create_Name')
        # Calling create_Name(args, kwargs) (line 280)
        create_Name_call_result_15999 = invoke(stypy.reporting.localization.Localization(__file__, 280, 27), create_Name_15996, *[str_15997], **kwargs_15998)
        
        # Assigning a type to the variable 'type_of_self_arg' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'type_of_self_arg', create_Name_call_result_15999)
        # SSA branch for the else part of an if statement (line 278)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 282):
        
        # Call to create_str(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'function_def_node' (line 282)
        function_def_node_16002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 58), 'function_def_node', False)
        # Obtaining the member 'name' of a type (line 282)
        name_16003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 58), function_def_node_16002, 'name')
        # Processing the call keyword arguments (line 282)
        kwargs_16004 = {}
        # Getting the type of 'core_language_copy' (line 282)
        core_language_copy_16000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 28), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 282)
        create_str_16001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 28), core_language_copy_16000, 'create_str')
        # Calling create_str(args, kwargs) (line 282)
        create_str_call_result_16005 = invoke(stypy.reporting.localization.Localization(__file__, 282, 28), create_str_16001, *[name_16003], **kwargs_16004)
        
        # Assigning a type to the variable 'function_name_arg' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'function_name_arg', create_str_call_result_16005)
        
        # Assigning a Call to a Name (line 283):
        
        # Call to create_Name(...): (line 283)
        # Processing the call arguments (line 283)
        str_16008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 58), 'str', 'None')
        # Processing the call keyword arguments (line 283)
        kwargs_16009 = {}
        # Getting the type of 'core_language_copy' (line 283)
        core_language_copy_16006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 283)
        create_Name_16007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 27), core_language_copy_16006, 'create_Name')
        # Calling create_Name(args, kwargs) (line 283)
        create_Name_call_result_16010 = invoke(stypy.reporting.localization.Localization(__file__, 283, 27), create_Name_16007, *[str_16008], **kwargs_16009)
        
        # Assigning a type to the variable 'type_of_self_arg' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'type_of_self_arg', create_Name_call_result_16010)
        # SSA join for if statement (line 278)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 286):
    
    # Call to obtain_arg_list(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'function_def_node' (line 286)
    function_def_node_16012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 43), 'function_def_node', False)
    # Obtaining the member 'args' of a type (line 286)
    args_16013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 43), function_def_node_16012, 'args')
    
    # Call to is_method(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'context' (line 286)
    context_16015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 77), 'context', False)
    # Processing the call keyword arguments (line 286)
    kwargs_16016 = {}
    # Getting the type of 'is_method' (line 286)
    is_method_16014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 67), 'is_method', False)
    # Calling is_method(args, kwargs) (line 286)
    is_method_call_result_16017 = invoke(stypy.reporting.localization.Localization(__file__, 286, 67), is_method_16014, *[context_16015], **kwargs_16016)
    
    
    # Call to is_static_method(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'function_def_node' (line 287)
    function_def_node_16019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 60), 'function_def_node', False)
    # Processing the call keyword arguments (line 287)
    kwargs_16020 = {}
    # Getting the type of 'is_static_method' (line 287)
    is_static_method_16018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 43), 'is_static_method', False)
    # Calling is_static_method(args, kwargs) (line 287)
    is_static_method_call_result_16021 = invoke(stypy.reporting.localization.Localization(__file__, 287, 43), is_static_method_16018, *[function_def_node_16019], **kwargs_16020)
    
    # Processing the call keyword arguments (line 286)
    kwargs_16022 = {}
    # Getting the type of 'obtain_arg_list' (line 286)
    obtain_arg_list_16011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 27), 'obtain_arg_list', False)
    # Calling obtain_arg_list(args, kwargs) (line 286)
    obtain_arg_list_call_result_16023 = invoke(stypy.reporting.localization.Localization(__file__, 286, 27), obtain_arg_list_16011, *[args_16013, is_method_call_result_16017, is_static_method_call_result_16021], **kwargs_16022)
    
    # Assigning a type to the variable 'param_names_list_arg' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'param_names_list_arg', obtain_arg_list_call_result_16023)
    
    # Type idiom detected: calculating its left and rigth part (line 290)
    # Getting the type of 'function_def_node' (line 290)
    function_def_node_16024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 7), 'function_def_node')
    # Obtaining the member 'args' of a type (line 290)
    args_16025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 7), function_def_node_16024, 'args')
    # Obtaining the member 'vararg' of a type (line 290)
    vararg_16026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 7), args_16025, 'vararg')
    # Getting the type of 'None' (line 290)
    None_16027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 40), 'None')
    
    (may_be_16028, more_types_in_union_16029) = may_be_none(vararg_16026, None_16027)

    if may_be_16028:

        if more_types_in_union_16029:
            # Runtime conditional SSA (line 290)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 291):
        # Getting the type of 'None' (line 291)
        None_16030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 27), 'None')
        # Assigning a type to the variable 'declared_varargs' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'declared_varargs', None_16030)

        if more_types_in_union_16029:
            # Runtime conditional SSA for else branch (line 290)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_16028) or more_types_in_union_16029):
        
        # Assigning a Attribute to a Name (line 293):
        # Getting the type of 'function_def_node' (line 293)
        function_def_node_16031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 27), 'function_def_node')
        # Obtaining the member 'args' of a type (line 293)
        args_16032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 27), function_def_node_16031, 'args')
        # Obtaining the member 'vararg' of a type (line 293)
        vararg_16033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 27), args_16032, 'vararg')
        # Assigning a type to the variable 'declared_varargs' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'declared_varargs', vararg_16033)

        if (may_be_16028 and more_types_in_union_16029):
            # SSA join for if statement (line 290)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 294):
    
    # Call to create_str(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'declared_varargs' (line 294)
    declared_varargs_16036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 55), 'declared_varargs', False)
    # Processing the call keyword arguments (line 294)
    kwargs_16037 = {}
    # Getting the type of 'core_language_copy' (line 294)
    core_language_copy_16034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 25), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 294)
    create_str_16035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 25), core_language_copy_16034, 'create_str')
    # Calling create_str(args, kwargs) (line 294)
    create_str_call_result_16038 = invoke(stypy.reporting.localization.Localization(__file__, 294, 25), create_str_16035, *[declared_varargs_16036], **kwargs_16037)
    
    # Assigning a type to the variable 'varargs_param_name' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'varargs_param_name', create_str_call_result_16038)
    
    # Type idiom detected: calculating its left and rigth part (line 296)
    # Getting the type of 'function_def_node' (line 296)
    function_def_node_16039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 7), 'function_def_node')
    # Obtaining the member 'args' of a type (line 296)
    args_16040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 7), function_def_node_16039, 'args')
    # Obtaining the member 'kwarg' of a type (line 296)
    kwarg_16041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 7), args_16040, 'kwarg')
    # Getting the type of 'None' (line 296)
    None_16042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 39), 'None')
    
    (may_be_16043, more_types_in_union_16044) = may_be_none(kwarg_16041, None_16042)

    if may_be_16043:

        if more_types_in_union_16044:
            # Runtime conditional SSA (line 296)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 297):
        # Getting the type of 'None' (line 297)
        None_16045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 26), 'None')
        # Assigning a type to the variable 'declared_kwargs' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'declared_kwargs', None_16045)

        if more_types_in_union_16044:
            # Runtime conditional SSA for else branch (line 296)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_16043) or more_types_in_union_16044):
        
        # Assigning a Attribute to a Name (line 299):
        # Getting the type of 'function_def_node' (line 299)
        function_def_node_16046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 26), 'function_def_node')
        # Obtaining the member 'args' of a type (line 299)
        args_16047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 26), function_def_node_16046, 'args')
        # Obtaining the member 'kwarg' of a type (line 299)
        kwarg_16048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 26), args_16047, 'kwarg')
        # Assigning a type to the variable 'declared_kwargs' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'declared_kwargs', kwarg_16048)

        if (may_be_16043 and more_types_in_union_16044):
            # SSA join for if statement (line 296)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 300):
    
    # Call to create_str(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 'declared_kwargs' (line 300)
    declared_kwargs_16051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 54), 'declared_kwargs', False)
    # Processing the call keyword arguments (line 300)
    kwargs_16052 = {}
    # Getting the type of 'core_language_copy' (line 300)
    core_language_copy_16049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 24), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 300)
    create_str_16050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 24), core_language_copy_16049, 'create_str')
    # Calling create_str(args, kwargs) (line 300)
    create_str_call_result_16053 = invoke(stypy.reporting.localization.Localization(__file__, 300, 24), create_str_16050, *[declared_kwargs_16051], **kwargs_16052)
    
    # Assigning a type to the variable 'kwargs_param_name' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'kwargs_param_name', create_str_call_result_16053)
    
    # Assigning a Call to a Name (line 304):
    
    # Call to create_Name(...): (line 304)
    # Processing the call arguments (line 304)
    str_16056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 51), 'str', 'defaults')
    # Processing the call keyword arguments (line 304)
    kwargs_16057 = {}
    # Getting the type of 'core_language_copy' (line 304)
    core_language_copy_16054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 304)
    create_Name_16055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 20), core_language_copy_16054, 'create_Name')
    # Calling create_Name(args, kwargs) (line 304)
    create_Name_call_result_16058 = invoke(stypy.reporting.localization.Localization(__file__, 304, 20), create_Name_16055, *[str_16056], **kwargs_16057)
    
    # Assigning a type to the variable 'call_defaults' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'call_defaults', create_Name_call_result_16058)
    
    # Assigning a Call to a Name (line 307):
    
    # Call to create_Name(...): (line 307)
    # Processing the call arguments (line 307)
    str_16061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 50), 'str', 'varargs')
    # Processing the call keyword arguments (line 307)
    kwargs_16062 = {}
    # Getting the type of 'core_language_copy' (line 307)
    core_language_copy_16059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 307)
    create_Name_16060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 19), core_language_copy_16059, 'create_Name')
    # Calling create_Name(args, kwargs) (line 307)
    create_Name_call_result_16063 = invoke(stypy.reporting.localization.Localization(__file__, 307, 19), create_Name_16060, *[str_16061], **kwargs_16062)
    
    # Assigning a type to the variable 'call_varargs' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'call_varargs', create_Name_call_result_16063)
    
    # Assigning a Call to a Name (line 309):
    
    # Call to create_Name(...): (line 309)
    # Processing the call arguments (line 309)
    str_16066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 49), 'str', 'kwargs')
    # Processing the call keyword arguments (line 309)
    kwargs_16067 = {}
    # Getting the type of 'core_language_copy' (line 309)
    core_language_copy_16064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 18), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 309)
    create_Name_16065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 18), core_language_copy_16064, 'create_Name')
    # Calling create_Name(args, kwargs) (line 309)
    create_Name_call_result_16068 = invoke(stypy.reporting.localization.Localization(__file__, 309, 18), create_Name_16065, *[str_16066], **kwargs_16067)
    
    # Assigning a type to the variable 'call_kwargs' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'call_kwargs', create_Name_call_result_16068)
    
    # Assigning a Call to a Name (line 312):
    
    # Call to create_call(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'func' (line 312)
    func_16070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 23), 'func', False)
    
    # Obtaining an instance of the builtin type 'list' (line 313)
    list_16071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 313)
    # Adding element type (line 313)
    # Getting the type of 'localization_arg' (line 313)
    localization_arg_16072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 24), 'localization_arg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_16071, localization_arg_16072)
    # Adding element type (line 313)
    # Getting the type of 'type_of_self_arg' (line 313)
    type_of_self_arg_16073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 42), 'type_of_self_arg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_16071, type_of_self_arg_16073)
    # Adding element type (line 313)
    # Getting the type of 'type_store_arg' (line 313)
    type_store_arg_16074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 60), 'type_store_arg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_16071, type_store_arg_16074)
    # Adding element type (line 313)
    # Getting the type of 'function_name_arg' (line 313)
    function_name_arg_16075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 76), 'function_name_arg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_16071, function_name_arg_16075)
    # Adding element type (line 313)
    # Getting the type of 'param_names_list_arg' (line 313)
    param_names_list_arg_16076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 95), 'param_names_list_arg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_16071, param_names_list_arg_16076)
    # Adding element type (line 313)
    # Getting the type of 'varargs_param_name' (line 314)
    varargs_param_name_16077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 24), 'varargs_param_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_16071, varargs_param_name_16077)
    # Adding element type (line 313)
    # Getting the type of 'kwargs_param_name' (line 314)
    kwargs_param_name_16078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 44), 'kwargs_param_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_16071, kwargs_param_name_16078)
    # Adding element type (line 313)
    # Getting the type of 'call_defaults' (line 314)
    call_defaults_16079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 63), 'call_defaults', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_16071, call_defaults_16079)
    # Adding element type (line 313)
    # Getting the type of 'call_varargs' (line 314)
    call_varargs_16080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 78), 'call_varargs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_16071, call_varargs_16080)
    # Adding element type (line 313)
    # Getting the type of 'call_kwargs' (line 314)
    call_kwargs_16081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 92), 'call_kwargs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_16071, call_kwargs_16081)
    
    # Processing the call keyword arguments (line 312)
    kwargs_16082 = {}
    # Getting the type of 'create_call' (line 312)
    create_call_16069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), 'create_call', False)
    # Calling create_call(args, kwargs) (line 312)
    create_call_call_result_16083 = invoke(stypy.reporting.localization.Localization(__file__, 312, 11), create_call_16069, *[func_16070, list_16071], **kwargs_16082)
    
    # Assigning a type to the variable 'call' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'call', create_call_call_result_16083)
    
    # Assigning a Call to a Name (line 316):
    
    # Call to create_Assign(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'args_test_resul' (line 316)
    args_test_resul_16086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 46), 'args_test_resul', False)
    # Getting the type of 'call' (line 316)
    call_16087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 63), 'call', False)
    # Processing the call keyword arguments (line 316)
    kwargs_16088 = {}
    # Getting the type of 'core_language_copy' (line 316)
    core_language_copy_16084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 13), 'core_language_copy', False)
    # Obtaining the member 'create_Assign' of a type (line 316)
    create_Assign_16085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 13), core_language_copy_16084, 'create_Assign')
    # Calling create_Assign(args, kwargs) (line 316)
    create_Assign_call_result_16089 = invoke(stypy.reporting.localization.Localization(__file__, 316, 13), create_Assign_16085, *[args_test_resul_16086, call_16087], **kwargs_16088)
    
    # Assigning a type to the variable 'assign' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'assign', create_Assign_call_result_16089)
    
    # Assigning a Call to a Name (line 319):
    
    # Call to create_Name(...): (line 319)
    # Processing the call arguments (line 319)
    str_16092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 53), 'str', 'arguments')
    # Processing the call keyword arguments (line 319)
    kwargs_16093 = {}
    # Getting the type of 'core_language_copy' (line 319)
    core_language_copy_16090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 22), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 319)
    create_Name_16091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 22), core_language_copy_16090, 'create_Name')
    # Calling create_Name(args, kwargs) (line 319)
    create_Name_call_result_16094 = invoke(stypy.reporting.localization.Localization(__file__, 319, 22), create_Name_16091, *[str_16092], **kwargs_16093)
    
    # Assigning a type to the variable 'argument_errors' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'argument_errors', create_Name_call_result_16094)
    
    # Assigning a Call to a Name (line 320):
    
    # Call to create_Name(...): (line 320)
    # Processing the call arguments (line 320)
    str_16097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 51), 'str', 'is_error_type')
    # Processing the call keyword arguments (line 320)
    kwargs_16098 = {}
    # Getting the type of 'core_language_copy' (line 320)
    core_language_copy_16095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 20), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 320)
    create_Name_16096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 20), core_language_copy_16095, 'create_Name')
    # Calling create_Name(args, kwargs) (line 320)
    create_Name_call_result_16099 = invoke(stypy.reporting.localization.Localization(__file__, 320, 20), create_Name_16096, *[str_16097], **kwargs_16098)
    
    # Assigning a type to the variable 'is_error_type' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'is_error_type', create_Name_call_result_16099)
    
    # Assigning a Call to a Name (line 321):
    
    # Call to create_call(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'is_error_type' (line 321)
    is_error_type_16101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 26), 'is_error_type', False)
    # Getting the type of 'argument_errors' (line 321)
    argument_errors_16102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 41), 'argument_errors', False)
    # Processing the call keyword arguments (line 321)
    kwargs_16103 = {}
    # Getting the type of 'create_call' (line 321)
    create_call_16100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 14), 'create_call', False)
    # Calling create_call(args, kwargs) (line 321)
    create_call_call_result_16104 = invoke(stypy.reporting.localization.Localization(__file__, 321, 14), create_call_16100, *[is_error_type_16101, argument_errors_16102], **kwargs_16103)
    
    # Assigning a type to the variable 'if_test' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'if_test', create_call_call_result_16104)
    
    # Call to is_constructor(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'function_def_node' (line 323)
    function_def_node_16106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 22), 'function_def_node', False)
    # Processing the call keyword arguments (line 323)
    kwargs_16107 = {}
    # Getting the type of 'is_constructor' (line 323)
    is_constructor_16105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 7), 'is_constructor', False)
    # Calling is_constructor(args, kwargs) (line 323)
    is_constructor_call_result_16108 = invoke(stypy.reporting.localization.Localization(__file__, 323, 7), is_constructor_16105, *[function_def_node_16106], **kwargs_16107)
    
    # Testing if the type of an if condition is none (line 323)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 323, 4), is_constructor_call_result_16108):
        pass
    else:
        
        # Testing the type of an if condition (line 323)
        if_condition_16109 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 4), is_constructor_call_result_16108)
        # Assigning a type to the variable 'if_condition_16109' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'if_condition_16109', if_condition_16109)
        # SSA begins for if statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 324):
        # Getting the type of 'None' (line 324)
        None_16110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 26), 'None')
        # Assigning a type to the variable 'argument_errors' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'argument_errors', None_16110)
        # SSA join for if statement (line 323)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Name (line 326):
    
    # Obtaining an instance of the builtin type 'list' (line 326)
    list_16111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 326)
    # Adding element type (line 326)
    
    # Call to create_context_unset(...): (line 326)
    # Processing the call keyword arguments (line 326)
    kwargs_16113 = {}
    # Getting the type of 'create_context_unset' (line 326)
    create_context_unset_16112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'create_context_unset', False)
    # Calling create_context_unset(args, kwargs) (line 326)
    create_context_unset_call_result_16114 = invoke(stypy.reporting.localization.Localization(__file__, 326, 12), create_context_unset_16112, *[], **kwargs_16113)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 11), list_16111, create_context_unset_call_result_16114)
    # Adding element type (line 326)
    
    # Call to create_return(...): (line 326)
    # Processing the call arguments (line 326)
    # Getting the type of 'argument_errors' (line 326)
    argument_errors_16116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 50), 'argument_errors', False)
    # Processing the call keyword arguments (line 326)
    kwargs_16117 = {}
    # Getting the type of 'create_return' (line 326)
    create_return_16115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'create_return', False)
    # Calling create_return(args, kwargs) (line 326)
    create_return_call_result_16118 = invoke(stypy.reporting.localization.Localization(__file__, 326, 36), create_return_16115, *[argument_errors_16116], **kwargs_16117)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 11), list_16111, create_return_call_result_16118)
    
    # Assigning a type to the variable 'body' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'body', list_16111)
    
    # Assigning a Call to a Name (line 327):
    
    # Call to create_if(...): (line 327)
    # Processing the call arguments (line 327)
    # Getting the type of 'if_test' (line 327)
    if_test_16121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 48), 'if_test', False)
    # Getting the type of 'body' (line 327)
    body_16122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 57), 'body', False)
    # Processing the call keyword arguments (line 327)
    kwargs_16123 = {}
    # Getting the type of 'conditional_statements_copy' (line 327)
    conditional_statements_copy_16119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 10), 'conditional_statements_copy', False)
    # Obtaining the member 'create_if' of a type (line 327)
    create_if_16120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 10), conditional_statements_copy_16119, 'create_if')
    # Calling create_if(args, kwargs) (line 327)
    create_if_call_result_16124 = invoke(stypy.reporting.localization.Localization(__file__, 327, 10), create_if_16120, *[if_test_16121, body_16122], **kwargs_16123)
    
    # Assigning a type to the variable 'if_' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'if_', create_if_call_result_16124)
    
    # Obtaining an instance of the builtin type 'list' (line 329)
    list_16125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 329)
    # Adding element type (line 329)
    # Getting the type of 'assign' (line 329)
    assign_16126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'assign')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 11), list_16125, assign_16126)
    # Adding element type (line 329)
    # Getting the type of 'if_' (line 329)
    if__16127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'if_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 11), list_16125, if__16127)
    
    # Assigning a type to the variable 'stypy_return_type' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type', list_16125)
    
    # ################# End of 'create_arg_number_test(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_arg_number_test' in the type store
    # Getting the type of 'stypy_return_type' (line 258)
    stypy_return_type_16128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16128)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_arg_number_test'
    return stypy_return_type_16128

# Assigning a type to the variable 'create_arg_number_test' (line 258)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 0), 'create_arg_number_test', create_arg_number_test)

@norecursion
def create_type_for_lambda_function(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_type_for_lambda_function'
    module_type_store = module_type_store.open_function_context('create_type_for_lambda_function', 332, 0, False)
    
    # Passed parameters checking function
    create_type_for_lambda_function.stypy_localization = localization
    create_type_for_lambda_function.stypy_type_of_self = None
    create_type_for_lambda_function.stypy_type_store = module_type_store
    create_type_for_lambda_function.stypy_function_name = 'create_type_for_lambda_function'
    create_type_for_lambda_function.stypy_param_names_list = ['function_name', 'lambda_call', 'lineno', 'col_offset']
    create_type_for_lambda_function.stypy_varargs_param_name = None
    create_type_for_lambda_function.stypy_kwargs_param_name = None
    create_type_for_lambda_function.stypy_call_defaults = defaults
    create_type_for_lambda_function.stypy_call_varargs = varargs
    create_type_for_lambda_function.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_type_for_lambda_function', ['function_name', 'lambda_call', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_type_for_lambda_function', localization, ['function_name', 'lambda_call', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_type_for_lambda_function(...)' code ##################

    str_16129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, (-1)), 'str', '\n    Creates a variable to store a lambda function definition\n\n    :param function_name: Name of the lambda function\n    :param lambda_call: Lambda function\n    :param lineno: Line\n    :param col_offset: Column\n    :return: Statements to create the lambda function type\n    ')
    
    # Assigning a Call to a Name (line 349):
    
    # Call to create_Name(...): (line 349)
    # Processing the call arguments (line 349)
    # Getting the type of 'lambda_call' (line 349)
    lambda_call_16132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 46), 'lambda_call', False)
    # Processing the call keyword arguments (line 349)
    kwargs_16133 = {}
    # Getting the type of 'core_language_copy' (line 349)
    core_language_copy_16130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 349)
    create_Name_16131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 15), core_language_copy_16130, 'create_Name')
    # Calling create_Name(args, kwargs) (line 349)
    create_Name_call_result_16134 = invoke(stypy.reporting.localization.Localization(__file__, 349, 15), create_Name_16131, *[lambda_call_16132], **kwargs_16133)
    
    # Assigning a type to the variable 'call_arg' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'call_arg', create_Name_call_result_16134)
    
    # Assigning a Call to a Name (line 351):
    
    # Call to create_set_type_of(...): (line 351)
    # Processing the call arguments (line 351)
    # Getting the type of 'function_name' (line 351)
    function_name_16137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 61), 'function_name', False)
    # Getting the type of 'call_arg' (line 351)
    call_arg_16138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 76), 'call_arg', False)
    # Getting the type of 'lineno' (line 351)
    lineno_16139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 86), 'lineno', False)
    # Getting the type of 'col_offset' (line 351)
    col_offset_16140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 94), 'col_offset', False)
    # Processing the call keyword arguments (line 351)
    kwargs_16141 = {}
    # Getting the type of 'stypy_functions_copy' (line 351)
    stypy_functions_copy_16135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 21), 'stypy_functions_copy', False)
    # Obtaining the member 'create_set_type_of' of a type (line 351)
    create_set_type_of_16136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 21), stypy_functions_copy_16135, 'create_set_type_of')
    # Calling create_set_type_of(args, kwargs) (line 351)
    create_set_type_of_call_result_16142 = invoke(stypy.reporting.localization.Localization(__file__, 351, 21), create_set_type_of_16136, *[function_name_16137, call_arg_16138, lineno_16139, col_offset_16140], **kwargs_16141)
    
    # Assigning a type to the variable 'set_type_stmts' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'set_type_stmts', create_set_type_of_call_result_16142)
    
    # Call to flatten_lists(...): (line 354)
    # Processing the call arguments (line 354)
    # Getting the type of 'set_type_stmts' (line 354)
    set_type_stmts_16145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 46), 'set_type_stmts', False)
    # Processing the call keyword arguments (line 354)
    kwargs_16146 = {}
    # Getting the type of 'stypy_functions_copy' (line 354)
    stypy_functions_copy_16143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 11), 'stypy_functions_copy', False)
    # Obtaining the member 'flatten_lists' of a type (line 354)
    flatten_lists_16144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 11), stypy_functions_copy_16143, 'flatten_lists')
    # Calling flatten_lists(args, kwargs) (line 354)
    flatten_lists_call_result_16147 = invoke(stypy.reporting.localization.Localization(__file__, 354, 11), flatten_lists_16144, *[set_type_stmts_16145], **kwargs_16146)
    
    # Assigning a type to the variable 'stypy_return_type' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'stypy_return_type', flatten_lists_call_result_16147)
    
    # ################# End of 'create_type_for_lambda_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_type_for_lambda_function' in the type store
    # Getting the type of 'stypy_return_type' (line 332)
    stypy_return_type_16148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16148)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_type_for_lambda_function'
    return stypy_return_type_16148

# Assigning a type to the variable 'create_type_for_lambda_function' (line 332)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'create_type_for_lambda_function', create_type_for_lambda_function)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
