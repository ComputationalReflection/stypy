
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
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_31129 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'core_language_copy')

if (type(import_31129) is not StypyTypeError):

    if (import_31129 != 'pyd_module'):
        __import__(import_31129)
        sys_modules_31130 = sys.modules[import_31129]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'core_language_copy', sys_modules_31130.module_type_store, module_type_store)
    else:
        import core_language_copy

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'core_language_copy', core_language_copy, module_type_store)

else:
    # Assigning a type to the variable 'core_language_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'core_language_copy', import_31129)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import data_structures_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_31131 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'data_structures_copy')

if (type(import_31131) is not StypyTypeError):

    if (import_31131 != 'pyd_module'):
        __import__(import_31131)
        sys_modules_31132 = sys.modules[import_31131]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'data_structures_copy', sys_modules_31132.module_type_store, module_type_store)
    else:
        import data_structures_copy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'data_structures_copy', data_structures_copy, module_type_store)

else:
    # Assigning a type to the variable 'data_structures_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'data_structures_copy', import_31131)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import conditional_statements_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_31133 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'conditional_statements_copy')

if (type(import_31133) is not StypyTypeError):

    if (import_31133 != 'pyd_module'):
        __import__(import_31133)
        sys_modules_31134 = sys.modules[import_31133]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'conditional_statements_copy', sys_modules_31134.module_type_store, module_type_store)
    else:
        import conditional_statements_copy

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'conditional_statements_copy', conditional_statements_copy, module_type_store)

else:
    # Assigning a type to the variable 'conditional_statements_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'conditional_statements_copy', import_31133)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import stypy_functions_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_31135 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_functions_copy')

if (type(import_31135) is not StypyTypeError):

    if (import_31135 != 'pyd_module'):
        __import__(import_31135)
        sys_modules_31136 = sys.modules[import_31135]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_functions_copy', sys_modules_31136.module_type_store, module_type_store)
    else:
        import stypy_functions_copy

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_functions_copy', stypy_functions_copy, module_type_store)

else:
    # Assigning a type to the variable 'stypy_functions_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_functions_copy', import_31135)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

str_31137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\nThis file contains helper functions to generate type inference code.\nThese functions refer to function-related language elements such as declarations and invokations.\n')

@norecursion
def create_call(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Call to list(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_31139 = {}
    # Getting the type of 'list' (line 14)
    list_31138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 37), 'list', False)
    # Calling list(args, kwargs) (line 14)
    list_call_result_31140 = invoke(stypy.reporting.localization.Localization(__file__, 14, 37), list_31138, *[], **kwargs_31139)
    
    # Getting the type of 'None' (line 14)
    None_31141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 52), 'None')
    # Getting the type of 'None' (line 14)
    None_31142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 67), 'None')
    int_31143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 78), 'int')
    int_31144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 88), 'int')
    defaults = [list_call_result_31140, None_31141, None_31142, int_31143, int_31144]
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

    str_31145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'str', '\n    Creates an AST Call node\n\n    :param func: Function name\n    :param args: List of arguments\n    :param keywords: List of default arguments\n    :param kwargs: Dict of keyword arguments\n    :param starargs: Variable list of arguments\n    :param line: Line\n    :param column: Column\n    :return: AST Call node\n    ')
    
    # Assigning a Call to a Name (line 27):
    
    # Call to Call(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_31148 = {}
    # Getting the type of 'ast' (line 27)
    ast_31146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'ast', False)
    # Obtaining the member 'Call' of a type (line 27)
    Call_31147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 11), ast_31146, 'Call')
    # Calling Call(args, kwargs) (line 27)
    Call_call_result_31149 = invoke(stypy.reporting.localization.Localization(__file__, 27, 11), Call_31147, *[], **kwargs_31148)
    
    # Assigning a type to the variable 'call' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'call', Call_call_result_31149)
    
    # Assigning a List to a Attribute (line 28):
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_31150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    
    # Getting the type of 'call' (line 28)
    call_31151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'call')
    # Setting the type of the member 'args' of a type (line 28)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), call_31151, 'args', list_31150)
    
    # Call to is_iterable(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'args' (line 30)
    args_31154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 40), 'args', False)
    # Processing the call keyword arguments (line 30)
    kwargs_31155 = {}
    # Getting the type of 'data_structures_copy' (line 30)
    data_structures_copy_31152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'data_structures_copy', False)
    # Obtaining the member 'is_iterable' of a type (line 30)
    is_iterable_31153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 7), data_structures_copy_31152, 'is_iterable')
    # Calling is_iterable(args, kwargs) (line 30)
    is_iterable_call_result_31156 = invoke(stypy.reporting.localization.Localization(__file__, 30, 7), is_iterable_31153, *[args_31154], **kwargs_31155)
    
    # Testing if the type of an if condition is none (line 30)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 30, 4), is_iterable_call_result_31156):
        
        # Call to append(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'args' (line 34)
        args_31169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'args', False)
        # Processing the call keyword arguments (line 34)
        kwargs_31170 = {}
        # Getting the type of 'call' (line 34)
        call_31166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'call', False)
        # Obtaining the member 'args' of a type (line 34)
        args_31167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), call_31166, 'args')
        # Obtaining the member 'append' of a type (line 34)
        append_31168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), args_31167, 'append')
        # Calling append(args, kwargs) (line 34)
        append_call_result_31171 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), append_31168, *[args_31169], **kwargs_31170)
        
    else:
        
        # Testing the type of an if condition (line 30)
        if_condition_31157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 4), is_iterable_call_result_31156)
        # Assigning a type to the variable 'if_condition_31157' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'if_condition_31157', if_condition_31157)
        # SSA begins for if statement (line 30)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'args' (line 31)
        args_31158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'args')
        # Assigning a type to the variable 'args_31158' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'args_31158', args_31158)
        # Testing if the for loop is going to be iterated (line 31)
        # Testing the type of a for loop iterable (line 31)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 8), args_31158)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 31, 8), args_31158):
            # Getting the type of the for loop variable (line 31)
            for_loop_var_31159 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 8), args_31158)
            # Assigning a type to the variable 'arg' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'arg', for_loop_var_31159)
            # SSA begins for a for statement (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 32)
            # Processing the call arguments (line 32)
            # Getting the type of 'arg' (line 32)
            arg_31163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 29), 'arg', False)
            # Processing the call keyword arguments (line 32)
            kwargs_31164 = {}
            # Getting the type of 'call' (line 32)
            call_31160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'call', False)
            # Obtaining the member 'args' of a type (line 32)
            args_31161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), call_31160, 'args')
            # Obtaining the member 'append' of a type (line 32)
            append_31162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), args_31161, 'append')
            # Calling append(args, kwargs) (line 32)
            append_call_result_31165 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), append_31162, *[arg_31163], **kwargs_31164)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA branch for the else part of an if statement (line 30)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'args' (line 34)
        args_31169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'args', False)
        # Processing the call keyword arguments (line 34)
        kwargs_31170 = {}
        # Getting the type of 'call' (line 34)
        call_31166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'call', False)
        # Obtaining the member 'args' of a type (line 34)
        args_31167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), call_31166, 'args')
        # Obtaining the member 'append' of a type (line 34)
        append_31168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), args_31167, 'append')
        # Calling append(args, kwargs) (line 34)
        append_call_result_31171 = invoke(stypy.reporting.localization.Localization(__file__, 34, 8), append_31168, *[args_31169], **kwargs_31170)
        
        # SSA join for if statement (line 30)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Name to a Attribute (line 36):
    # Getting the type of 'func' (line 36)
    func_31172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'func')
    # Getting the type of 'call' (line 36)
    call_31173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'call')
    # Setting the type of the member 'func' of a type (line 36)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), call_31173, 'func', func_31172)
    
    # Assigning a Name to a Attribute (line 37):
    # Getting the type of 'line' (line 37)
    line_31174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'line')
    # Getting the type of 'call' (line 37)
    call_31175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'call')
    # Setting the type of the member 'lineno' of a type (line 37)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), call_31175, 'lineno', line_31174)
    
    # Assigning a Name to a Attribute (line 38):
    # Getting the type of 'column' (line 38)
    column_31176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'column')
    # Getting the type of 'call' (line 38)
    call_31177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'call')
    # Setting the type of the member 'col_offset' of a type (line 38)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), call_31177, 'col_offset', column_31176)
    
    # Assigning a Name to a Attribute (line 39):
    # Getting the type of 'keywords' (line 39)
    keywords_31178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'keywords')
    # Getting the type of 'call' (line 39)
    call_31179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'call')
    # Setting the type of the member 'keywords' of a type (line 39)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 4), call_31179, 'keywords', keywords_31178)
    
    # Assigning a Name to a Attribute (line 40):
    # Getting the type of 'kwargs' (line 40)
    kwargs_31180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'kwargs')
    # Getting the type of 'call' (line 40)
    call_31181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'call')
    # Setting the type of the member 'kwargs' of a type (line 40)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 4), call_31181, 'kwargs', kwargs_31180)
    
    # Assigning a Name to a Attribute (line 41):
    # Getting the type of 'starargs' (line 41)
    starargs_31182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'starargs')
    # Getting the type of 'call' (line 41)
    call_31183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'call')
    # Setting the type of the member 'starargs' of a type (line 41)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), call_31183, 'starargs', starargs_31182)
    # Getting the type of 'call' (line 43)
    call_31184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'call')
    # Assigning a type to the variable 'stypy_return_type' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type', call_31184)
    
    # ################# End of 'create_call(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_call' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_31185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31185)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_call'
    return stypy_return_type_31185

# Assigning a type to the variable 'create_call' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'create_call', create_call)

@norecursion
def create_call_expression(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Call to list(...): (line 46)
    # Processing the call keyword arguments (line 46)
    kwargs_31187 = {}
    # Getting the type of 'list' (line 46)
    list_31186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 48), 'list', False)
    # Calling list(args, kwargs) (line 46)
    list_call_result_31188 = invoke(stypy.reporting.localization.Localization(__file__, 46, 48), list_31186, *[], **kwargs_31187)
    
    # Getting the type of 'None' (line 46)
    None_31189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 63), 'None')
    # Getting the type of 'None' (line 46)
    None_31190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 78), 'None')
    int_31191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 89), 'int')
    int_31192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 99), 'int')
    defaults = [list_call_result_31188, None_31189, None_31190, int_31191, int_31192]
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

    str_31193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '\n    Creates an AST Call node that will be enclosed in an expression node. This is used when the call are not a part\n    of a longer expression, but the expression itself\n\n    :param func: Function name\n    :param args: List of arguments\n    :param keywords: List of default arguments\n    :param kwargs: Dict of keyword arguments\n    :param starargs: Variable list of arguments\n    :param line: Line\n    :param column: Column\n    :return: AST Expr node\n    ')
    
    # Assigning a Call to a Name (line 60):
    
    # Call to create_call(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'func' (line 60)
    func_31195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'func', False)
    # Getting the type of 'args' (line 60)
    args_31196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'args', False)
    # Getting the type of 'keywords' (line 60)
    keywords_31197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 35), 'keywords', False)
    # Getting the type of 'kwargs' (line 60)
    kwargs_31198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 45), 'kwargs', False)
    # Getting the type of 'starargs' (line 60)
    starargs_31199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 53), 'starargs', False)
    # Getting the type of 'line' (line 60)
    line_31200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 63), 'line', False)
    # Getting the type of 'column' (line 60)
    column_31201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 69), 'column', False)
    # Processing the call keyword arguments (line 60)
    kwargs_31202 = {}
    # Getting the type of 'create_call' (line 60)
    create_call_31194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'create_call', False)
    # Calling create_call(args, kwargs) (line 60)
    create_call_call_result_31203 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), create_call_31194, *[func_31195, args_31196, keywords_31197, kwargs_31198, starargs_31199, line_31200, column_31201], **kwargs_31202)
    
    # Assigning a type to the variable 'call' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'call', create_call_call_result_31203)
    
    # Assigning a Call to a Name (line 61):
    
    # Call to Expr(...): (line 61)
    # Processing the call keyword arguments (line 61)
    kwargs_31206 = {}
    # Getting the type of 'ast' (line 61)
    ast_31204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 22), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 61)
    Expr_31205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 22), ast_31204, 'Expr')
    # Calling Expr(args, kwargs) (line 61)
    Expr_call_result_31207 = invoke(stypy.reporting.localization.Localization(__file__, 61, 22), Expr_31205, *[], **kwargs_31206)
    
    # Assigning a type to the variable 'call_expression' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'call_expression', Expr_call_result_31207)
    
    # Assigning a Name to a Attribute (line 62):
    # Getting the type of 'call' (line 62)
    call_31208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'call')
    # Getting the type of 'call_expression' (line 62)
    call_expression_31209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'call_expression')
    # Setting the type of the member 'value' of a type (line 62)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 4), call_expression_31209, 'value', call_31208)
    
    # Assigning a Name to a Attribute (line 63):
    # Getting the type of 'line' (line 63)
    line_31210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'line')
    # Getting the type of 'call_expression' (line 63)
    call_expression_31211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'call_expression')
    # Setting the type of the member 'lineno' of a type (line 63)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 4), call_expression_31211, 'lineno', line_31210)
    
    # Assigning a Name to a Attribute (line 64):
    # Getting the type of 'column' (line 64)
    column_31212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'column')
    # Getting the type of 'call_expression' (line 64)
    call_expression_31213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'call_expression')
    # Setting the type of the member 'col_offset' of a type (line 64)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 4), call_expression_31213, 'col_offset', column_31212)
    # Getting the type of 'call_expression' (line 66)
    call_expression_31214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'call_expression')
    # Assigning a type to the variable 'stypy_return_type' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type', call_expression_31214)
    
    # ################# End of 'create_call_expression(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_call_expression' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_31215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31215)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_call_expression'
    return stypy_return_type_31215

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

    str_31216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, (-1)), 'str', '\n    Determines if an AST Function node represent a method (belongs to an AST ClassDef node)\n    :param context:\n    :return:\n    ')
    
    # Assigning a Name to a Name (line 75):
    # Getting the type of 'False' (line 75)
    False_31217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'False')
    # Assigning a type to the variable 'ismethod' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'ismethod', False_31217)
    
    
    
    # Call to len(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'context' (line 77)
    context_31219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'context', False)
    # Processing the call keyword arguments (line 77)
    kwargs_31220 = {}
    # Getting the type of 'len' (line 77)
    len_31218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'len', False)
    # Calling len(args, kwargs) (line 77)
    len_call_result_31221 = invoke(stypy.reporting.localization.Localization(__file__, 77, 11), len_31218, *[context_31219], **kwargs_31220)
    
    int_31222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 27), 'int')
    # Applying the binary operator '==' (line 77)
    result_eq_31223 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 11), '==', len_call_result_31221, int_31222)
    
    # Applying the 'not' unary operator (line 77)
    result_not__31224 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 7), 'not', result_eq_31223)
    
    # Testing if the type of an if condition is none (line 77)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 4), result_not__31224):
        pass
    else:
        
        # Testing the type of an if condition (line 77)
        if_condition_31225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 4), result_not__31224)
        # Assigning a type to the variable 'if_condition_31225' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'if_condition_31225', if_condition_31225)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 78):
        
        # Call to isinstance(...): (line 78)
        # Processing the call arguments (line 78)
        
        # Obtaining the type of the subscript
        int_31227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 38), 'int')
        # Getting the type of 'context' (line 78)
        context_31228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'context', False)
        # Obtaining the member '__getitem__' of a type (line 78)
        getitem___31229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 30), context_31228, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 78)
        subscript_call_result_31230 = invoke(stypy.reporting.localization.Localization(__file__, 78, 30), getitem___31229, int_31227)
        
        # Getting the type of 'ast' (line 78)
        ast_31231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 43), 'ast', False)
        # Obtaining the member 'ClassDef' of a type (line 78)
        ClassDef_31232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 43), ast_31231, 'ClassDef')
        # Processing the call keyword arguments (line 78)
        kwargs_31233 = {}
        # Getting the type of 'isinstance' (line 78)
        isinstance_31226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 78)
        isinstance_call_result_31234 = invoke(stypy.reporting.localization.Localization(__file__, 78, 19), isinstance_31226, *[subscript_call_result_31230, ClassDef_31232], **kwargs_31233)
        
        # Assigning a type to the variable 'ismethod' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'ismethod', isinstance_call_result_31234)
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'ismethod' (line 80)
    ismethod_31235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'ismethod')
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type', ismethod_31235)
    
    # ################# End of 'is_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_method' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_31236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31236)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_method'
    return stypy_return_type_31236

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
    str_31237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 25), 'str', 'decorator_list')
    # Getting the type of 'node' (line 84)
    node_31238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'node')
    
    (may_be_31239, more_types_in_union_31240) = may_not_provide_member(str_31237, node_31238)

    if may_be_31239:

        if more_types_in_union_31240:
            # Runtime conditional SSA (line 84)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'node' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'node', remove_member_provider_from_union(node_31238, 'decorator_list'))
        # Getting the type of 'False' (line 85)
        False_31241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', False_31241)

        if more_types_in_union_31240:
            # SSA join for if statement (line 84)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to len(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'node' (line 86)
    node_31243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'node', False)
    # Obtaining the member 'decorator_list' of a type (line 86)
    decorator_list_31244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 11), node_31243, 'decorator_list')
    # Processing the call keyword arguments (line 86)
    kwargs_31245 = {}
    # Getting the type of 'len' (line 86)
    len_31242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 7), 'len', False)
    # Calling len(args, kwargs) (line 86)
    len_call_result_31246 = invoke(stypy.reporting.localization.Localization(__file__, 86, 7), len_31242, *[decorator_list_31244], **kwargs_31245)
    
    int_31247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 35), 'int')
    # Applying the binary operator '==' (line 86)
    result_eq_31248 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 7), '==', len_call_result_31246, int_31247)
    
    # Testing if the type of an if condition is none (line 86)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 4), result_eq_31248):
        pass
    else:
        
        # Testing the type of an if condition (line 86)
        if_condition_31249 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 4), result_eq_31248)
        # Assigning a type to the variable 'if_condition_31249' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'if_condition_31249', if_condition_31249)
        # SSA begins for if statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 87)
        False_31250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', False_31250)
        # SSA join for if statement (line 86)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'node' (line 88)
    node_31251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'node')
    # Obtaining the member 'decorator_list' of a type (line 88)
    decorator_list_31252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 20), node_31251, 'decorator_list')
    # Assigning a type to the variable 'decorator_list_31252' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'decorator_list_31252', decorator_list_31252)
    # Testing if the for loop is going to be iterated (line 88)
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 4), decorator_list_31252)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 88, 4), decorator_list_31252):
        # Getting the type of the for loop variable (line 88)
        for_loop_var_31253 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 4), decorator_list_31252)
        # Assigning a type to the variable 'dec_name' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'dec_name', for_loop_var_31253)
        # SSA begins for a for statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 89)
        str_31254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'str', 'id')
        # Getting the type of 'dec_name' (line 89)
        dec_name_31255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'dec_name')
        
        (may_be_31256, more_types_in_union_31257) = may_provide_member(str_31254, dec_name_31255)

        if may_be_31256:

            if more_types_in_union_31257:
                # Runtime conditional SSA (line 89)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'dec_name' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'dec_name', remove_not_member_provider_from_union(dec_name_31255, 'id'))
            
            # Getting the type of 'dec_name' (line 90)
            dec_name_31258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'dec_name')
            # Obtaining the member 'id' of a type (line 90)
            id_31259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 15), dec_name_31258, 'id')
            str_31260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 30), 'str', 'staticmethod')
            # Applying the binary operator '==' (line 90)
            result_eq_31261 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 15), '==', id_31259, str_31260)
            
            # Testing if the type of an if condition is none (line 90)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 90, 12), result_eq_31261):
                pass
            else:
                
                # Testing the type of an if condition (line 90)
                if_condition_31262 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 12), result_eq_31261)
                # Assigning a type to the variable 'if_condition_31262' (line 90)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'if_condition_31262', if_condition_31262)
                # SSA begins for if statement (line 90)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 91)
                True_31263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'stypy_return_type', True_31263)
                # SSA join for if statement (line 90)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_31257:
                # SSA join for if statement (line 89)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'False' (line 92)
    False_31264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', False_31264)
    
    # ################# End of 'is_static_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_static_method' in the type store
    # Getting the type of 'stypy_return_type' (line 83)
    stypy_return_type_31265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31265)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_static_method'
    return stypy_return_type_31265

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

    str_31266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, (-1)), 'str', '\n    Determines if an AST Function node represent a constructor (its name is __init__)\n    :param node: AST Function node or str\n    :return: bool\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 101)
    # Getting the type of 'node' (line 101)
    node_31267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'node')
    # Getting the type of 'str' (line 101)
    str_31268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'str')
    
    (may_be_31269, more_types_in_union_31270) = may_be_type(node_31267, str_31268)

    if may_be_31269:

        if more_types_in_union_31270:
            # Runtime conditional SSA (line 101)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'node' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'node', str_31268())
        
        # Getting the type of 'node' (line 102)
        node_31271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'node')
        str_31272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 23), 'str', '__init__')
        # Applying the binary operator '==' (line 102)
        result_eq_31273 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 15), '==', node_31271, str_31272)
        
        # Assigning a type to the variable 'stypy_return_type' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'stypy_return_type', result_eq_31273)

        if more_types_in_union_31270:
            # SSA join for if statement (line 101)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'node' (line 104)
    node_31274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'node')
    # Obtaining the member 'name' of a type (line 104)
    name_31275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), node_31274, 'name')
    str_31276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 24), 'str', '__init__')
    # Applying the binary operator '==' (line 104)
    result_eq_31277 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 11), '==', name_31275, str_31276)
    
    # Assigning a type to the variable 'stypy_return_type' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type', result_eq_31277)
    
    # ################# End of 'is_constructor(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_constructor' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_31278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31278)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_constructor'
    return stypy_return_type_31278

# Assigning a type to the variable 'is_constructor' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'is_constructor', is_constructor)

@norecursion
def create_function_def(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_31279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 70), 'int')
    int_31280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 80), 'int')
    defaults = [int_31279, int_31280]
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

    str_31281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, (-1)), 'str', '\n    Creates a FunctionDef node, that represent a function declaration. This is used in type inference code, so every\n    created function has the following parameters (type_of_self, localization, *varargs, **kwargs) for methods and\n    (localization, *varargs, **kwargs) for functions.\n\n    :param name: Name of the function\n    :param localization: Localization parameter\n    :param decorators: Decorators of the function, mainly the norecursion one\n    :param context: Context passed to this method\n    :param line: Line\n    :param column: Column\n    :return: An AST FunctionDef node\n    ')
    
    # Assigning a Call to a Name (line 121):
    
    # Call to arguments(...): (line 121)
    # Processing the call keyword arguments (line 121)
    kwargs_31284 = {}
    # Getting the type of 'ast' (line 121)
    ast_31282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'ast', False)
    # Obtaining the member 'arguments' of a type (line 121)
    arguments_31283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 29), ast_31282, 'arguments')
    # Calling arguments(args, kwargs) (line 121)
    arguments_call_result_31285 = invoke(stypy.reporting.localization.Localization(__file__, 121, 29), arguments_31283, *[], **kwargs_31284)
    
    # Assigning a type to the variable 'function_def_arguments' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'function_def_arguments', arguments_call_result_31285)
    
    # Assigning a List to a Attribute (line 122):
    
    # Obtaining an instance of the builtin type 'list' (line 122)
    list_31286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 122)
    # Adding element type (line 122)
    # Getting the type of 'localization' (line 122)
    localization_31287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 35), 'localization')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 34), list_31286, localization_31287)
    
    # Getting the type of 'function_def_arguments' (line 122)
    function_def_arguments_31288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'function_def_arguments')
    # Setting the type of the member 'args' of a type (line 122)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 4), function_def_arguments_31288, 'args', list_31286)
    
    # Assigning a Call to a Name (line 124):
    
    # Call to is_constructor(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'name' (line 124)
    name_31290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 35), 'name', False)
    # Processing the call keyword arguments (line 124)
    kwargs_31291 = {}
    # Getting the type of 'is_constructor' (line 124)
    is_constructor_31289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'is_constructor', False)
    # Calling is_constructor(args, kwargs) (line 124)
    is_constructor_call_result_31292 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), is_constructor_31289, *[name_31290], **kwargs_31291)
    
    # Assigning a type to the variable 'isconstructor' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'isconstructor', is_constructor_call_result_31292)
    
    # Assigning a Call to a Name (line 125):
    
    # Call to is_method(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'context' (line 125)
    context_31294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 25), 'context', False)
    # Processing the call keyword arguments (line 125)
    kwargs_31295 = {}
    # Getting the type of 'is_method' (line 125)
    is_method_31293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'is_method', False)
    # Calling is_method(args, kwargs) (line 125)
    is_method_call_result_31296 = invoke(stypy.reporting.localization.Localization(__file__, 125, 15), is_method_31293, *[context_31294], **kwargs_31295)
    
    # Assigning a type to the variable 'ismethod' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'ismethod', is_method_call_result_31296)
    
    # Assigning a Call to a Name (line 127):
    
    # Call to FunctionDef(...): (line 127)
    # Processing the call keyword arguments (line 127)
    kwargs_31299 = {}
    # Getting the type of 'ast' (line 127)
    ast_31297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'ast', False)
    # Obtaining the member 'FunctionDef' of a type (line 127)
    FunctionDef_31298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), ast_31297, 'FunctionDef')
    # Calling FunctionDef(args, kwargs) (line 127)
    FunctionDef_call_result_31300 = invoke(stypy.reporting.localization.Localization(__file__, 127, 19), FunctionDef_31298, *[], **kwargs_31299)
    
    # Assigning a type to the variable 'function_def' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'function_def', FunctionDef_call_result_31300)
    
    # Assigning a Name to a Attribute (line 128):
    # Getting the type of 'line' (line 128)
    line_31301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 26), 'line')
    # Getting the type of 'function_def' (line 128)
    function_def_31302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'function_def')
    # Setting the type of the member 'lineno' of a type (line 128)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 4), function_def_31302, 'lineno', line_31301)
    
    # Assigning a Name to a Attribute (line 129):
    # Getting the type of 'column' (line 129)
    column_31303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'column')
    # Getting the type of 'function_def' (line 129)
    function_def_31304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'function_def')
    # Setting the type of the member 'col_offset' of a type (line 129)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 4), function_def_31304, 'col_offset', column_31303)
    
    # Assigning a Name to a Attribute (line 130):
    # Getting the type of 'name' (line 130)
    name_31305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'name')
    # Getting the type of 'function_def' (line 130)
    function_def_31306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'function_def')
    # Setting the type of the member 'name' of a type (line 130)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 4), function_def_31306, 'name', name_31305)
    
    # Assigning a Name to a Attribute (line 132):
    # Getting the type of 'function_def_arguments' (line 132)
    function_def_arguments_31307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'function_def_arguments')
    # Getting the type of 'function_def' (line 132)
    function_def_31308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'function_def')
    # Setting the type of the member 'args' of a type (line 132)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 4), function_def_31308, 'args', function_def_arguments_31307)
    
    # Assigning a List to a Attribute (line 134):
    
    # Obtaining an instance of the builtin type 'list' (line 134)
    list_31309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 134)
    
    # Getting the type of 'function_def_arguments' (line 134)
    function_def_arguments_31310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'function_def_arguments')
    # Setting the type of the member 'args' of a type (line 134)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 4), function_def_arguments_31310, 'args', list_31309)
    # Getting the type of 'isconstructor' (line 136)
    isconstructor_31311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 7), 'isconstructor')
    # Testing if the type of an if condition is none (line 136)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 136, 4), isconstructor_31311):
        pass
    else:
        
        # Testing the type of an if condition (line 136)
        if_condition_31312 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 4), isconstructor_31311)
        # Assigning a type to the variable 'if_condition_31312' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'if_condition_31312', if_condition_31312)
        # SSA begins for if statement (line 136)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 137)
        # Processing the call arguments (line 137)
        
        # Call to create_Name(...): (line 137)
        # Processing the call arguments (line 137)
        str_31318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 74), 'str', 'type_of_self')
        # Processing the call keyword arguments (line 137)
        kwargs_31319 = {}
        # Getting the type of 'core_language_copy' (line 137)
        core_language_copy_31316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 43), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 137)
        create_Name_31317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 43), core_language_copy_31316, 'create_Name')
        # Calling create_Name(args, kwargs) (line 137)
        create_Name_call_result_31320 = invoke(stypy.reporting.localization.Localization(__file__, 137, 43), create_Name_31317, *[str_31318], **kwargs_31319)
        
        # Processing the call keyword arguments (line 137)
        kwargs_31321 = {}
        # Getting the type of 'function_def_arguments' (line 137)
        function_def_arguments_31313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'function_def_arguments', False)
        # Obtaining the member 'args' of a type (line 137)
        args_31314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), function_def_arguments_31313, 'args')
        # Obtaining the member 'append' of a type (line 137)
        append_31315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 8), args_31314, 'append')
        # Calling append(args, kwargs) (line 137)
        append_call_result_31322 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), append_31315, *[create_Name_call_result_31320], **kwargs_31321)
        
        # SSA join for if statement (line 136)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    # Getting the type of 'ismethod' (line 139)
    ismethod_31323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 7), 'ismethod')
    
    # Getting the type of 'isconstructor' (line 139)
    isconstructor_31324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'isconstructor')
    # Applying the 'not' unary operator (line 139)
    result_not__31325 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 20), 'not', isconstructor_31324)
    
    # Applying the binary operator 'and' (line 139)
    result_and_keyword_31326 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 7), 'and', ismethod_31323, result_not__31325)
    
    # Testing if the type of an if condition is none (line 139)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 139, 4), result_and_keyword_31326):
        pass
    else:
        
        # Testing the type of an if condition (line 139)
        if_condition_31327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 4), result_and_keyword_31326)
        # Assigning a type to the variable 'if_condition_31327' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'if_condition_31327', if_condition_31327)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 140)
        # Processing the call arguments (line 140)
        
        # Call to create_Name(...): (line 140)
        # Processing the call arguments (line 140)
        str_31333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 74), 'str', 'type_of_self')
        # Processing the call keyword arguments (line 140)
        kwargs_31334 = {}
        # Getting the type of 'core_language_copy' (line 140)
        core_language_copy_31331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 43), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 140)
        create_Name_31332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 43), core_language_copy_31331, 'create_Name')
        # Calling create_Name(args, kwargs) (line 140)
        create_Name_call_result_31335 = invoke(stypy.reporting.localization.Localization(__file__, 140, 43), create_Name_31332, *[str_31333], **kwargs_31334)
        
        # Processing the call keyword arguments (line 140)
        kwargs_31336 = {}
        # Getting the type of 'function_def_arguments' (line 140)
        function_def_arguments_31328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'function_def_arguments', False)
        # Obtaining the member 'args' of a type (line 140)
        args_31329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), function_def_arguments_31328, 'args')
        # Obtaining the member 'append' of a type (line 140)
        append_31330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), args_31329, 'append')
        # Calling append(args, kwargs) (line 140)
        append_call_result_31337 = invoke(stypy.reporting.localization.Localization(__file__, 140, 8), append_31330, *[create_Name_call_result_31335], **kwargs_31336)
        
        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to append(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'localization' (line 142)
    localization_31341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 39), 'localization', False)
    # Processing the call keyword arguments (line 142)
    kwargs_31342 = {}
    # Getting the type of 'function_def_arguments' (line 142)
    function_def_arguments_31338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'function_def_arguments', False)
    # Obtaining the member 'args' of a type (line 142)
    args_31339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 4), function_def_arguments_31338, 'args')
    # Obtaining the member 'append' of a type (line 142)
    append_31340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 4), args_31339, 'append')
    # Calling append(args, kwargs) (line 142)
    append_call_result_31343 = invoke(stypy.reporting.localization.Localization(__file__, 142, 4), append_31340, *[localization_31341], **kwargs_31342)
    
    
    # Assigning a Str to a Attribute (line 144):
    str_31344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 35), 'str', 'kwargs')
    # Getting the type of 'function_def_arguments' (line 144)
    function_def_arguments_31345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'function_def_arguments')
    # Setting the type of the member 'kwarg' of a type (line 144)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 4), function_def_arguments_31345, 'kwarg', str_31344)
    
    # Assigning a Str to a Attribute (line 145):
    str_31346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 36), 'str', 'varargs')
    # Getting the type of 'function_def_arguments' (line 145)
    function_def_arguments_31347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'function_def_arguments')
    # Setting the type of the member 'vararg' of a type (line 145)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 4), function_def_arguments_31347, 'vararg', str_31346)
    
    # Assigning a List to a Attribute (line 146):
    
    # Obtaining an instance of the builtin type 'list' (line 146)
    list_31348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 146)
    
    # Getting the type of 'function_def_arguments' (line 146)
    function_def_arguments_31349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'function_def_arguments')
    # Setting the type of the member 'defaults' of a type (line 146)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 4), function_def_arguments_31349, 'defaults', list_31348)
    
    # Call to is_iterable(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'decorators' (line 148)
    decorators_31352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 40), 'decorators', False)
    # Processing the call keyword arguments (line 148)
    kwargs_31353 = {}
    # Getting the type of 'data_structures_copy' (line 148)
    data_structures_copy_31350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'data_structures_copy', False)
    # Obtaining the member 'is_iterable' of a type (line 148)
    is_iterable_31351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 7), data_structures_copy_31350, 'is_iterable')
    # Calling is_iterable(args, kwargs) (line 148)
    is_iterable_call_result_31354 = invoke(stypy.reporting.localization.Localization(__file__, 148, 7), is_iterable_31351, *[decorators_31352], **kwargs_31353)
    
    # Testing if the type of an if condition is none (line 148)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 148, 4), is_iterable_call_result_31354):
        
        # Assigning a List to a Attribute (line 151):
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_31358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        # Getting the type of 'decorators' (line 151)
        decorators_31359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 39), 'decorators')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 38), list_31358, decorators_31359)
        
        # Getting the type of 'function_def' (line 151)
        function_def_31360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'function_def')
        # Setting the type of the member 'decorator_list' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), function_def_31360, 'decorator_list', list_31358)
    else:
        
        # Testing the type of an if condition (line 148)
        if_condition_31355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), is_iterable_call_result_31354)
        # Assigning a type to the variable 'if_condition_31355' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_31355', if_condition_31355)
        # SSA begins for if statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 149):
        # Getting the type of 'decorators' (line 149)
        decorators_31356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 38), 'decorators')
        # Getting the type of 'function_def' (line 149)
        function_def_31357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'function_def')
        # Setting the type of the member 'decorator_list' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), function_def_31357, 'decorator_list', decorators_31356)
        # SSA branch for the else part of an if statement (line 148)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 151):
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_31358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        # Adding element type (line 151)
        # Getting the type of 'decorators' (line 151)
        decorators_31359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 39), 'decorators')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 38), list_31358, decorators_31359)
        
        # Getting the type of 'function_def' (line 151)
        function_def_31360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'function_def')
        # Setting the type of the member 'decorator_list' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), function_def_31360, 'decorator_list', list_31358)
        # SSA join for if statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Attribute (line 153):
    
    # Obtaining an instance of the builtin type 'list' (line 153)
    list_31361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 153)
    
    # Getting the type of 'function_def' (line 153)
    function_def_31362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'function_def')
    # Setting the type of the member 'body' of a type (line 153)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), function_def_31362, 'body', list_31361)
    # Getting the type of 'function_def' (line 155)
    function_def_31363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'function_def')
    # Assigning a type to the variable 'stypy_return_type' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type', function_def_31363)
    
    # ################# End of 'create_function_def(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_function_def' in the type store
    # Getting the type of 'stypy_return_type' (line 107)
    stypy_return_type_31364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31364)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_function_def'
    return stypy_return_type_31364

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

    str_31365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, (-1)), 'str', '\n    Creates an AST Return node\n    :param value: Value to return\n    :return: An AST Return node\n    ')
    
    # Assigning a Call to a Name (line 164):
    
    # Call to Return(...): (line 164)
    # Processing the call keyword arguments (line 164)
    kwargs_31368 = {}
    # Getting the type of 'ast' (line 164)
    ast_31366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'ast', False)
    # Obtaining the member 'Return' of a type (line 164)
    Return_31367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 11), ast_31366, 'Return')
    # Calling Return(args, kwargs) (line 164)
    Return_call_result_31369 = invoke(stypy.reporting.localization.Localization(__file__, 164, 11), Return_31367, *[], **kwargs_31368)
    
    # Assigning a type to the variable 'node' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'node', Return_call_result_31369)
    
    # Assigning a Name to a Attribute (line 165):
    # Getting the type of 'value' (line 165)
    value_31370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 17), 'value')
    # Getting the type of 'node' (line 165)
    node_31371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'node')
    # Setting the type of the member 'value' of a type (line 165)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 4), node_31371, 'value', value_31370)
    # Getting the type of 'node' (line 167)
    node_31372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'node')
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type', node_31372)
    
    # ################# End of 'create_return(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_return' in the type store
    # Getting the type of 'stypy_return_type' (line 158)
    stypy_return_type_31373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31373)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_return'
    return stypy_return_type_31373

# Assigning a type to the variable 'create_return' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'create_return', create_return)

@norecursion
def obtain_arg_list(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 170)
    False_31374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 35), 'False')
    # Getting the type of 'False' (line 170)
    False_31375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 57), 'False')
    defaults = [False_31374, False_31375]
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

    str_31376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, (-1)), 'str', '\n    Creates an AST List node with the names of the arguments passed to a function\n    :param args: Arguments\n    :param ismethod: Whether to count the first argument (self) or not\n    :return: An AST List\n    ')
    
    # Assigning a Call to a Name (line 177):
    
    # Call to List(...): (line 177)
    # Processing the call keyword arguments (line 177)
    kwargs_31379 = {}
    # Getting the type of 'ast' (line 177)
    ast_31377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'ast', False)
    # Obtaining the member 'List' of a type (line 177)
    List_31378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 15), ast_31377, 'List')
    # Calling List(args, kwargs) (line 177)
    List_call_result_31380 = invoke(stypy.reporting.localization.Localization(__file__, 177, 15), List_31378, *[], **kwargs_31379)
    
    # Assigning a type to the variable 'arg_list' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'arg_list', List_call_result_31380)
    
    # Assigning a List to a Attribute (line 179):
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_31381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    
    # Getting the type of 'arg_list' (line 179)
    arg_list_31382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'arg_list')
    # Setting the type of the member 'elts' of a type (line 179)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 4), arg_list_31382, 'elts', list_31381)
    
    # Evaluating a boolean operation
    # Getting the type of 'ismethod' (line 180)
    ismethod_31383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 7), 'ismethod')
    
    # Getting the type of 'isstaticmethod' (line 180)
    isstaticmethod_31384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 24), 'isstaticmethod')
    # Applying the 'not' unary operator (line 180)
    result_not__31385 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 20), 'not', isstaticmethod_31384)
    
    # Applying the binary operator 'and' (line 180)
    result_and_keyword_31386 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 7), 'and', ismethod_31383, result_not__31385)
    
    # Testing if the type of an if condition is none (line 180)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 180, 4), result_and_keyword_31386):
        
        # Assigning a Attribute to a Name (line 183):
        # Getting the type of 'args' (line 183)
        args_31394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'args')
        # Obtaining the member 'args' of a type (line 183)
        args_31395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 28), args_31394, 'args')
        # Assigning a type to the variable 'arg_list_contents' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'arg_list_contents', args_31395)
    else:
        
        # Testing the type of an if condition (line 180)
        if_condition_31387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 180, 4), result_and_keyword_31386)
        # Assigning a type to the variable 'if_condition_31387' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'if_condition_31387', if_condition_31387)
        # SSA begins for if statement (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 181):
        
        # Obtaining the type of the subscript
        int_31388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 38), 'int')
        slice_31389 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 181, 28), int_31388, None, None)
        # Getting the type of 'args' (line 181)
        args_31390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'args')
        # Obtaining the member 'args' of a type (line 181)
        args_31391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 28), args_31390, 'args')
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___31392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 28), args_31391, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_31393 = invoke(stypy.reporting.localization.Localization(__file__, 181, 28), getitem___31392, slice_31389)
        
        # Assigning a type to the variable 'arg_list_contents' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'arg_list_contents', subscript_call_result_31393)
        # SSA branch for the else part of an if statement (line 180)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 183):
        # Getting the type of 'args' (line 183)
        args_31394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 28), 'args')
        # Obtaining the member 'args' of a type (line 183)
        args_31395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 28), args_31394, 'args')
        # Assigning a type to the variable 'arg_list_contents' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'arg_list_contents', args_31395)
        # SSA join for if statement (line 180)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'arg_list_contents' (line 185)
    arg_list_contents_31396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'arg_list_contents')
    # Assigning a type to the variable 'arg_list_contents_31396' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'arg_list_contents_31396', arg_list_contents_31396)
    # Testing if the for loop is going to be iterated (line 185)
    # Testing the type of a for loop iterable (line 185)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 185, 4), arg_list_contents_31396)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 185, 4), arg_list_contents_31396):
        # Getting the type of the for loop variable (line 185)
        for_loop_var_31397 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 185, 4), arg_list_contents_31396)
        # Assigning a type to the variable 'arg' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'arg', for_loop_var_31397)
        # SSA begins for a for statement (line 185)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 186)
        # Processing the call arguments (line 186)
        
        # Call to create_str(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'arg' (line 186)
        arg_31403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 59), 'arg', False)
        # Obtaining the member 'id' of a type (line 186)
        id_31404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 59), arg_31403, 'id')
        # Processing the call keyword arguments (line 186)
        kwargs_31405 = {}
        # Getting the type of 'core_language_copy' (line 186)
        core_language_copy_31401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 29), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 186)
        create_str_31402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 29), core_language_copy_31401, 'create_str')
        # Calling create_str(args, kwargs) (line 186)
        create_str_call_result_31406 = invoke(stypy.reporting.localization.Localization(__file__, 186, 29), create_str_31402, *[id_31404], **kwargs_31405)
        
        # Processing the call keyword arguments (line 186)
        kwargs_31407 = {}
        # Getting the type of 'arg_list' (line 186)
        arg_list_31398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'arg_list', False)
        # Obtaining the member 'elts' of a type (line 186)
        elts_31399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), arg_list_31398, 'elts')
        # Obtaining the member 'append' of a type (line 186)
        append_31400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), elts_31399, 'append')
        # Calling append(args, kwargs) (line 186)
        append_call_result_31408 = invoke(stypy.reporting.localization.Localization(__file__, 186, 8), append_31400, *[create_str_call_result_31406], **kwargs_31407)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'arg_list' (line 188)
    arg_list_31409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 11), 'arg_list')
    # Assigning a type to the variable 'stypy_return_type' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'stypy_return_type', arg_list_31409)
    
    # ################# End of 'obtain_arg_list(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'obtain_arg_list' in the type store
    # Getting the type of 'stypy_return_type' (line 170)
    stypy_return_type_31410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31410)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'obtain_arg_list'
    return stypy_return_type_31410

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

    str_31411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, (-1)), 'str', '\n    Creates an AST Node that model the call to the localitazion.set_stack_trace method\n\n    :param func_name: Name of the function that will do the push to the stack trace\n    :param declared_arguments: Arguments of the call\n    :return: An AST Expr node\n    ')
    
    # Assigning a Call to a Name (line 200):
    
    # Call to create_attribute(...): (line 200)
    # Processing the call arguments (line 200)
    str_31414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 52), 'str', 'localization')
    str_31415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 68), 'str', 'set_stack_trace')
    # Processing the call keyword arguments (line 200)
    kwargs_31416 = {}
    # Getting the type of 'core_language_copy' (line 200)
    core_language_copy_31412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 16), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 200)
    create_attribute_31413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 16), core_language_copy_31412, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 200)
    create_attribute_call_result_31417 = invoke(stypy.reporting.localization.Localization(__file__, 200, 16), create_attribute_31413, *[str_31414, str_31415], **kwargs_31416)
    
    # Assigning a type to the variable 'attribute' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'attribute', create_attribute_call_result_31417)
    
    # Assigning a Call to a Name (line 201):
    
    # Call to create_Name(...): (line 201)
    # Processing the call arguments (line 201)
    str_31420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 51), 'str', 'arguments')
    # Processing the call keyword arguments (line 201)
    kwargs_31421 = {}
    # Getting the type of 'core_language_copy' (line 201)
    core_language_copy_31418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 20), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 201)
    create_Name_31419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 20), core_language_copy_31418, 'create_Name')
    # Calling create_Name(args, kwargs) (line 201)
    create_Name_call_result_31422 = invoke(stypy.reporting.localization.Localization(__file__, 201, 20), create_Name_31419, *[str_31420], **kwargs_31421)
    
    # Assigning a type to the variable 'arguments_var' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'arguments_var', create_Name_call_result_31422)
    
    # Assigning a Call to a Name (line 202):
    
    # Call to create_call(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'attribute' (line 202)
    attribute_31424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 34), 'attribute', False)
    
    # Obtaining an instance of the builtin type 'list' (line 202)
    list_31425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 45), 'list')
    # Adding type elements to the builtin type 'list' instance (line 202)
    # Adding element type (line 202)
    
    # Call to create_str(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'func_name' (line 202)
    func_name_31428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 76), 'func_name', False)
    # Processing the call keyword arguments (line 202)
    kwargs_31429 = {}
    # Getting the type of 'core_language_copy' (line 202)
    core_language_copy_31426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 46), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 202)
    create_str_31427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 46), core_language_copy_31426, 'create_str')
    # Calling create_str(args, kwargs) (line 202)
    create_str_call_result_31430 = invoke(stypy.reporting.localization.Localization(__file__, 202, 46), create_str_31427, *[func_name_31428], **kwargs_31429)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 45), list_31425, create_str_call_result_31430)
    # Adding element type (line 202)
    # Getting the type of 'declared_arguments' (line 202)
    declared_arguments_31431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 88), 'declared_arguments', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 45), list_31425, declared_arguments_31431)
    # Adding element type (line 202)
    # Getting the type of 'arguments_var' (line 202)
    arguments_var_31432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 108), 'arguments_var', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 45), list_31425, arguments_var_31432)
    
    # Processing the call keyword arguments (line 202)
    kwargs_31433 = {}
    # Getting the type of 'create_call' (line 202)
    create_call_31423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'create_call', False)
    # Calling create_call(args, kwargs) (line 202)
    create_call_call_result_31434 = invoke(stypy.reporting.localization.Localization(__file__, 202, 22), create_call_31423, *[attribute_31424, list_31425], **kwargs_31433)
    
    # Assigning a type to the variable 'stack_push_call' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stack_push_call', create_call_call_result_31434)
    
    # Assigning a Call to a Name (line 203):
    
    # Call to Expr(...): (line 203)
    # Processing the call keyword arguments (line 203)
    kwargs_31437 = {}
    # Getting the type of 'ast' (line 203)
    ast_31435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 17), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 203)
    Expr_31436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 17), ast_31435, 'Expr')
    # Calling Expr(args, kwargs) (line 203)
    Expr_call_result_31438 = invoke(stypy.reporting.localization.Localization(__file__, 203, 17), Expr_31436, *[], **kwargs_31437)
    
    # Assigning a type to the variable 'stack_push' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stack_push', Expr_call_result_31438)
    
    # Assigning a Name to a Attribute (line 204):
    # Getting the type of 'stack_push_call' (line 204)
    stack_push_call_31439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 23), 'stack_push_call')
    # Getting the type of 'stack_push' (line 204)
    stack_push_31440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stack_push')
    # Setting the type of the member 'value' of a type (line 204)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 4), stack_push_31440, 'value', stack_push_call_31439)
    # Getting the type of 'stack_push' (line 206)
    stack_push_31441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 11), 'stack_push')
    # Assigning a type to the variable 'stypy_return_type' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'stypy_return_type', stack_push_31441)
    
    # ################# End of 'create_stacktrace_push(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_stacktrace_push' in the type store
    # Getting the type of 'stypy_return_type' (line 191)
    stypy_return_type_31442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31442)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_stacktrace_push'
    return stypy_return_type_31442

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

    str_31443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, (-1)), 'str', '\n    Creates an AST Node that model the call to the localitazion.unset_stack_trace method\n\n    :return: An AST Expr node\n    ')
    
    # Assigning a Call to a Name (line 216):
    
    # Call to create_attribute(...): (line 216)
    # Processing the call arguments (line 216)
    str_31446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 52), 'str', 'localization')
    str_31447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 68), 'str', 'unset_stack_trace')
    # Processing the call keyword arguments (line 216)
    kwargs_31448 = {}
    # Getting the type of 'core_language_copy' (line 216)
    core_language_copy_31444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 216)
    create_attribute_31445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 16), core_language_copy_31444, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 216)
    create_attribute_call_result_31449 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), create_attribute_31445, *[str_31446, str_31447], **kwargs_31448)
    
    # Assigning a type to the variable 'attribute' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'attribute', create_attribute_call_result_31449)
    
    # Assigning a Call to a Name (line 217):
    
    # Call to create_call(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'attribute' (line 217)
    attribute_31451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 33), 'attribute', False)
    
    # Obtaining an instance of the builtin type 'list' (line 217)
    list_31452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 217)
    
    # Processing the call keyword arguments (line 217)
    kwargs_31453 = {}
    # Getting the type of 'create_call' (line 217)
    create_call_31450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'create_call', False)
    # Calling create_call(args, kwargs) (line 217)
    create_call_call_result_31454 = invoke(stypy.reporting.localization.Localization(__file__, 217, 21), create_call_31450, *[attribute_31451, list_31452], **kwargs_31453)
    
    # Assigning a type to the variable 'stack_pop_call' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stack_pop_call', create_call_call_result_31454)
    
    # Assigning a Call to a Name (line 218):
    
    # Call to Expr(...): (line 218)
    # Processing the call keyword arguments (line 218)
    kwargs_31457 = {}
    # Getting the type of 'ast' (line 218)
    ast_31455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 218)
    Expr_31456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 16), ast_31455, 'Expr')
    # Calling Expr(args, kwargs) (line 218)
    Expr_call_result_31458 = invoke(stypy.reporting.localization.Localization(__file__, 218, 16), Expr_31456, *[], **kwargs_31457)
    
    # Assigning a type to the variable 'stack_pop' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'stack_pop', Expr_call_result_31458)
    
    # Assigning a Name to a Attribute (line 219):
    # Getting the type of 'stack_pop_call' (line 219)
    stack_pop_call_31459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 22), 'stack_pop_call')
    # Getting the type of 'stack_pop' (line 219)
    stack_pop_31460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stack_pop')
    # Setting the type of the member 'value' of a type (line 219)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 4), stack_pop_31460, 'value', stack_pop_call_31459)
    # Getting the type of 'stack_pop' (line 221)
    stack_pop_31461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'stack_pop')
    # Assigning a type to the variable 'stypy_return_type' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type', stack_pop_31461)
    
    # ################# End of 'create_stacktrace_pop(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_stacktrace_pop' in the type store
    # Getting the type of 'stypy_return_type' (line 209)
    stypy_return_type_31462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31462)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_stacktrace_pop'
    return stypy_return_type_31462

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

    str_31463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, (-1)), 'str', '\n    Creates an AST Node that model the call to the type_store.set_context method\n\n    :param func_name: Name of the function that will do the push to the stack trace\n    :param lineno: Line\n    :param col_offset: Column\n    :return: An AST Expr node\n    ')
    
    # Assigning a Call to a Name (line 233):
    
    # Call to create_attribute(...): (line 233)
    # Processing the call arguments (line 233)
    str_31466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 52), 'str', 'type_store')
    str_31467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 66), 'str', 'set_context')
    # Processing the call keyword arguments (line 233)
    kwargs_31468 = {}
    # Getting the type of 'core_language_copy' (line 233)
    core_language_copy_31464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 16), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 233)
    create_attribute_31465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 16), core_language_copy_31464, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 233)
    create_attribute_call_result_31469 = invoke(stypy.reporting.localization.Localization(__file__, 233, 16), create_attribute_31465, *[str_31466, str_31467], **kwargs_31468)
    
    # Assigning a type to the variable 'attribute' (line 233)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'attribute', create_attribute_call_result_31469)
    
    # Assigning a Call to a Name (line 234):
    
    # Call to create_call(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'attribute' (line 234)
    attribute_31471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 35), 'attribute', False)
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_31472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    # Adding element type (line 234)
    
    # Call to create_str(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'func_name' (line 234)
    func_name_31475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 77), 'func_name', False)
    # Processing the call keyword arguments (line 234)
    kwargs_31476 = {}
    # Getting the type of 'core_language_copy' (line 234)
    core_language_copy_31473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 47), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 234)
    create_str_31474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 47), core_language_copy_31473, 'create_str')
    # Calling create_str(args, kwargs) (line 234)
    create_str_call_result_31477 = invoke(stypy.reporting.localization.Localization(__file__, 234, 47), create_str_31474, *[func_name_31475], **kwargs_31476)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 46), list_31472, create_str_call_result_31477)
    # Adding element type (line 234)
    
    # Call to create_num(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'lineno' (line 235)
    lineno_31480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 77), 'lineno', False)
    # Processing the call keyword arguments (line 235)
    kwargs_31481 = {}
    # Getting the type of 'core_language_copy' (line 235)
    core_language_copy_31478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 47), 'core_language_copy', False)
    # Obtaining the member 'create_num' of a type (line 235)
    create_num_31479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 47), core_language_copy_31478, 'create_num')
    # Calling create_num(args, kwargs) (line 235)
    create_num_call_result_31482 = invoke(stypy.reporting.localization.Localization(__file__, 235, 47), create_num_31479, *[lineno_31480], **kwargs_31481)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 46), list_31472, create_num_call_result_31482)
    # Adding element type (line 234)
    
    # Call to create_num(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'col_offset' (line 236)
    col_offset_31485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 77), 'col_offset', False)
    # Processing the call keyword arguments (line 236)
    kwargs_31486 = {}
    # Getting the type of 'core_language_copy' (line 236)
    core_language_copy_31483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 47), 'core_language_copy', False)
    # Obtaining the member 'create_num' of a type (line 236)
    create_num_31484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 47), core_language_copy_31483, 'create_num')
    # Calling create_num(args, kwargs) (line 236)
    create_num_call_result_31487 = invoke(stypy.reporting.localization.Localization(__file__, 236, 47), create_num_31484, *[col_offset_31485], **kwargs_31486)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 46), list_31472, create_num_call_result_31487)
    
    # Processing the call keyword arguments (line 234)
    kwargs_31488 = {}
    # Getting the type of 'create_call' (line 234)
    create_call_31470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 23), 'create_call', False)
    # Calling create_call(args, kwargs) (line 234)
    create_call_call_result_31489 = invoke(stypy.reporting.localization.Localization(__file__, 234, 23), create_call_31470, *[attribute_31471, list_31472], **kwargs_31488)
    
    # Assigning a type to the variable 'context_set_call' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'context_set_call', create_call_call_result_31489)
    
    # Assigning a Call to a Name (line 237):
    
    # Call to Expr(...): (line 237)
    # Processing the call keyword arguments (line 237)
    kwargs_31492 = {}
    # Getting the type of 'ast' (line 237)
    ast_31490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 18), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 237)
    Expr_31491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 18), ast_31490, 'Expr')
    # Calling Expr(args, kwargs) (line 237)
    Expr_call_result_31493 = invoke(stypy.reporting.localization.Localization(__file__, 237, 18), Expr_31491, *[], **kwargs_31492)
    
    # Assigning a type to the variable 'context_set' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'context_set', Expr_call_result_31493)
    
    # Assigning a Name to a Attribute (line 238):
    # Getting the type of 'context_set_call' (line 238)
    context_set_call_31494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'context_set_call')
    # Getting the type of 'context_set' (line 238)
    context_set_31495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'context_set')
    # Setting the type of the member 'value' of a type (line 238)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 4), context_set_31495, 'value', context_set_call_31494)
    # Getting the type of 'context_set' (line 240)
    context_set_31496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'context_set')
    # Assigning a type to the variable 'stypy_return_type' (line 240)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 4), 'stypy_return_type', context_set_31496)
    
    # ################# End of 'create_context_set(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_context_set' in the type store
    # Getting the type of 'stypy_return_type' (line 224)
    stypy_return_type_31497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31497)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_context_set'
    return stypy_return_type_31497

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

    str_31498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, (-1)), 'str', '\n    Creates an AST Node that model the call to the type_store.unset_context method\n\n    :return: An AST Expr node\n    ')
    
    # Assigning a Call to a Name (line 250):
    
    # Call to create_attribute(...): (line 250)
    # Processing the call arguments (line 250)
    str_31501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 52), 'str', 'type_store')
    str_31502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 66), 'str', 'unset_context')
    # Processing the call keyword arguments (line 250)
    kwargs_31503 = {}
    # Getting the type of 'core_language_copy' (line 250)
    core_language_copy_31499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 16), 'core_language_copy', False)
    # Obtaining the member 'create_attribute' of a type (line 250)
    create_attribute_31500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 16), core_language_copy_31499, 'create_attribute')
    # Calling create_attribute(args, kwargs) (line 250)
    create_attribute_call_result_31504 = invoke(stypy.reporting.localization.Localization(__file__, 250, 16), create_attribute_31500, *[str_31501, str_31502], **kwargs_31503)
    
    # Assigning a type to the variable 'attribute' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'attribute', create_attribute_call_result_31504)
    
    # Assigning a Call to a Name (line 251):
    
    # Call to create_call(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'attribute' (line 251)
    attribute_31506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 37), 'attribute', False)
    
    # Obtaining an instance of the builtin type 'list' (line 251)
    list_31507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 48), 'list')
    # Adding type elements to the builtin type 'list' instance (line 251)
    
    # Processing the call keyword arguments (line 251)
    kwargs_31508 = {}
    # Getting the type of 'create_call' (line 251)
    create_call_31505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 25), 'create_call', False)
    # Calling create_call(args, kwargs) (line 251)
    create_call_call_result_31509 = invoke(stypy.reporting.localization.Localization(__file__, 251, 25), create_call_31505, *[attribute_31506, list_31507], **kwargs_31508)
    
    # Assigning a type to the variable 'context_unset_call' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'context_unset_call', create_call_call_result_31509)
    
    # Assigning a Call to a Name (line 252):
    
    # Call to Expr(...): (line 252)
    # Processing the call keyword arguments (line 252)
    kwargs_31512 = {}
    # Getting the type of 'ast' (line 252)
    ast_31510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 20), 'ast', False)
    # Obtaining the member 'Expr' of a type (line 252)
    Expr_31511 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 20), ast_31510, 'Expr')
    # Calling Expr(args, kwargs) (line 252)
    Expr_call_result_31513 = invoke(stypy.reporting.localization.Localization(__file__, 252, 20), Expr_31511, *[], **kwargs_31512)
    
    # Assigning a type to the variable 'context_unset' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 4), 'context_unset', Expr_call_result_31513)
    
    # Assigning a Name to a Attribute (line 253):
    # Getting the type of 'context_unset_call' (line 253)
    context_unset_call_31514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 26), 'context_unset_call')
    # Getting the type of 'context_unset' (line 253)
    context_unset_31515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'context_unset')
    # Setting the type of the member 'value' of a type (line 253)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 4), context_unset_31515, 'value', context_unset_call_31514)
    # Getting the type of 'context_unset' (line 255)
    context_unset_31516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 11), 'context_unset')
    # Assigning a type to the variable 'stypy_return_type' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'stypy_return_type', context_unset_31516)
    
    # ################# End of 'create_context_unset(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_context_unset' in the type store
    # Getting the type of 'stypy_return_type' (line 243)
    stypy_return_type_31517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31517)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_context_unset'
    return stypy_return_type_31517

# Assigning a type to the variable 'create_context_unset' (line 243)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 0), 'create_context_unset', create_context_unset)

@norecursion
def create_arg_number_test(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'list' (line 258)
    list_31518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 54), 'list')
    # Adding type elements to the builtin type 'list' instance (line 258)
    
    defaults = [list_31518]
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

    str_31519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, (-1)), 'str', '\n    Creates an AST Node that model the call to the process_argument_values method. This method is used to check\n    the parameters passed to a function/method in a type inference program\n\n    :param function_def_node: AST Node with the function definition\n    :param context: Context passed to the call\n    :return: List of AST nodes that perform the call to the mentioned function and make the necessary tests once it\n    is called\n    ')
    
    # Assigning a Call to a Name (line 268):
    
    # Call to create_Name(...): (line 268)
    # Processing the call arguments (line 268)
    str_31522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 268, 53), 'str', 'arguments')
    # Getting the type of 'False' (line 268)
    False_31523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 66), 'False', False)
    # Processing the call keyword arguments (line 268)
    kwargs_31524 = {}
    # Getting the type of 'core_language_copy' (line 268)
    core_language_copy_31520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 22), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 268)
    create_Name_31521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 22), core_language_copy_31520, 'create_Name')
    # Calling create_Name(args, kwargs) (line 268)
    create_Name_call_result_31525 = invoke(stypy.reporting.localization.Localization(__file__, 268, 22), create_Name_31521, *[str_31522, False_31523], **kwargs_31524)
    
    # Assigning a type to the variable 'args_test_resul' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'args_test_resul', create_Name_call_result_31525)
    
    # Assigning a Call to a Name (line 271):
    
    # Call to create_Name(...): (line 271)
    # Processing the call arguments (line 271)
    str_31528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 42), 'str', 'process_argument_values')
    # Processing the call keyword arguments (line 271)
    kwargs_31529 = {}
    # Getting the type of 'core_language_copy' (line 271)
    core_language_copy_31526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 11), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 271)
    create_Name_31527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 11), core_language_copy_31526, 'create_Name')
    # Calling create_Name(args, kwargs) (line 271)
    create_Name_call_result_31530 = invoke(stypy.reporting.localization.Localization(__file__, 271, 11), create_Name_31527, *[str_31528], **kwargs_31529)
    
    # Assigning a type to the variable 'func' (line 271)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'func', create_Name_call_result_31530)
    
    # Assigning a Call to a Name (line 273):
    
    # Call to create_Name(...): (line 273)
    # Processing the call arguments (line 273)
    str_31533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 54), 'str', 'localization')
    # Processing the call keyword arguments (line 273)
    kwargs_31534 = {}
    # Getting the type of 'core_language_copy' (line 273)
    core_language_copy_31531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 23), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 273)
    create_Name_31532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 23), core_language_copy_31531, 'create_Name')
    # Calling create_Name(args, kwargs) (line 273)
    create_Name_call_result_31535 = invoke(stypy.reporting.localization.Localization(__file__, 273, 23), create_Name_31532, *[str_31533], **kwargs_31534)
    
    # Assigning a type to the variable 'localization_arg' (line 273)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 4), 'localization_arg', create_Name_call_result_31535)
    
    # Assigning a Call to a Name (line 274):
    
    # Call to create_Name(...): (line 274)
    # Processing the call arguments (line 274)
    str_31538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 52), 'str', 'type_store')
    # Processing the call keyword arguments (line 274)
    kwargs_31539 = {}
    # Getting the type of 'core_language_copy' (line 274)
    core_language_copy_31536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 21), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 274)
    create_Name_31537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 274, 21), core_language_copy_31536, 'create_Name')
    # Calling create_Name(args, kwargs) (line 274)
    create_Name_call_result_31540 = invoke(stypy.reporting.localization.Localization(__file__, 274, 21), create_Name_31537, *[str_31538], **kwargs_31539)
    
    # Assigning a type to the variable 'type_store_arg' (line 274)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'type_store_arg', create_Name_call_result_31540)
    
    # Call to is_method(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'context' (line 278)
    context_31542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 17), 'context', False)
    # Processing the call keyword arguments (line 278)
    kwargs_31543 = {}
    # Getting the type of 'is_method' (line 278)
    is_method_31541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 7), 'is_method', False)
    # Calling is_method(args, kwargs) (line 278)
    is_method_call_result_31544 = invoke(stypy.reporting.localization.Localization(__file__, 278, 7), is_method_31541, *[context_31542], **kwargs_31543)
    
    # Testing if the type of an if condition is none (line 278)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 278, 4), is_method_call_result_31544):
        
        # Assigning a Call to a Name (line 282):
        
        # Call to create_str(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'function_def_node' (line 282)
        function_def_node_31567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 58), 'function_def_node', False)
        # Obtaining the member 'name' of a type (line 282)
        name_31568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 58), function_def_node_31567, 'name')
        # Processing the call keyword arguments (line 282)
        kwargs_31569 = {}
        # Getting the type of 'core_language_copy' (line 282)
        core_language_copy_31565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 28), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 282)
        create_str_31566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 28), core_language_copy_31565, 'create_str')
        # Calling create_str(args, kwargs) (line 282)
        create_str_call_result_31570 = invoke(stypy.reporting.localization.Localization(__file__, 282, 28), create_str_31566, *[name_31568], **kwargs_31569)
        
        # Assigning a type to the variable 'function_name_arg' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'function_name_arg', create_str_call_result_31570)
        
        # Assigning a Call to a Name (line 283):
        
        # Call to create_Name(...): (line 283)
        # Processing the call arguments (line 283)
        str_31573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 58), 'str', 'None')
        # Processing the call keyword arguments (line 283)
        kwargs_31574 = {}
        # Getting the type of 'core_language_copy' (line 283)
        core_language_copy_31571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 283)
        create_Name_31572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 27), core_language_copy_31571, 'create_Name')
        # Calling create_Name(args, kwargs) (line 283)
        create_Name_call_result_31575 = invoke(stypy.reporting.localization.Localization(__file__, 283, 27), create_Name_31572, *[str_31573], **kwargs_31574)
        
        # Assigning a type to the variable 'type_of_self_arg' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'type_of_self_arg', create_Name_call_result_31575)
    else:
        
        # Testing the type of an if condition (line 278)
        if_condition_31545 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 4), is_method_call_result_31544)
        # Assigning a type to the variable 'if_condition_31545' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'if_condition_31545', if_condition_31545)
        # SSA begins for if statement (line 278)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 279):
        
        # Call to create_str(...): (line 279)
        # Processing the call arguments (line 279)
        
        # Obtaining the type of the subscript
        int_31548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 66), 'int')
        # Getting the type of 'context' (line 279)
        context_31549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 58), 'context', False)
        # Obtaining the member '__getitem__' of a type (line 279)
        getitem___31550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 58), context_31549, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 279)
        subscript_call_result_31551 = invoke(stypy.reporting.localization.Localization(__file__, 279, 58), getitem___31550, int_31548)
        
        # Obtaining the member 'name' of a type (line 279)
        name_31552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 58), subscript_call_result_31551, 'name')
        str_31553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 77), 'str', '.')
        # Applying the binary operator '+' (line 279)
        result_add_31554 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 58), '+', name_31552, str_31553)
        
        # Getting the type of 'function_def_node' (line 279)
        function_def_node_31555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 83), 'function_def_node', False)
        # Obtaining the member 'name' of a type (line 279)
        name_31556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 83), function_def_node_31555, 'name')
        # Applying the binary operator '+' (line 279)
        result_add_31557 = python_operator(stypy.reporting.localization.Localization(__file__, 279, 81), '+', result_add_31554, name_31556)
        
        # Processing the call keyword arguments (line 279)
        kwargs_31558 = {}
        # Getting the type of 'core_language_copy' (line 279)
        core_language_copy_31546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 28), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 279)
        create_str_31547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 28), core_language_copy_31546, 'create_str')
        # Calling create_str(args, kwargs) (line 279)
        create_str_call_result_31559 = invoke(stypy.reporting.localization.Localization(__file__, 279, 28), create_str_31547, *[result_add_31557], **kwargs_31558)
        
        # Assigning a type to the variable 'function_name_arg' (line 279)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'function_name_arg', create_str_call_result_31559)
        
        # Assigning a Call to a Name (line 280):
        
        # Call to create_Name(...): (line 280)
        # Processing the call arguments (line 280)
        str_31562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 280, 58), 'str', 'type_of_self')
        # Processing the call keyword arguments (line 280)
        kwargs_31563 = {}
        # Getting the type of 'core_language_copy' (line 280)
        core_language_copy_31560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 27), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 280)
        create_Name_31561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 27), core_language_copy_31560, 'create_Name')
        # Calling create_Name(args, kwargs) (line 280)
        create_Name_call_result_31564 = invoke(stypy.reporting.localization.Localization(__file__, 280, 27), create_Name_31561, *[str_31562], **kwargs_31563)
        
        # Assigning a type to the variable 'type_of_self_arg' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'type_of_self_arg', create_Name_call_result_31564)
        # SSA branch for the else part of an if statement (line 278)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 282):
        
        # Call to create_str(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'function_def_node' (line 282)
        function_def_node_31567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 58), 'function_def_node', False)
        # Obtaining the member 'name' of a type (line 282)
        name_31568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 58), function_def_node_31567, 'name')
        # Processing the call keyword arguments (line 282)
        kwargs_31569 = {}
        # Getting the type of 'core_language_copy' (line 282)
        core_language_copy_31565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 28), 'core_language_copy', False)
        # Obtaining the member 'create_str' of a type (line 282)
        create_str_31566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 28), core_language_copy_31565, 'create_str')
        # Calling create_str(args, kwargs) (line 282)
        create_str_call_result_31570 = invoke(stypy.reporting.localization.Localization(__file__, 282, 28), create_str_31566, *[name_31568], **kwargs_31569)
        
        # Assigning a type to the variable 'function_name_arg' (line 282)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'function_name_arg', create_str_call_result_31570)
        
        # Assigning a Call to a Name (line 283):
        
        # Call to create_Name(...): (line 283)
        # Processing the call arguments (line 283)
        str_31573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 283, 58), 'str', 'None')
        # Processing the call keyword arguments (line 283)
        kwargs_31574 = {}
        # Getting the type of 'core_language_copy' (line 283)
        core_language_copy_31571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 27), 'core_language_copy', False)
        # Obtaining the member 'create_Name' of a type (line 283)
        create_Name_31572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 27), core_language_copy_31571, 'create_Name')
        # Calling create_Name(args, kwargs) (line 283)
        create_Name_call_result_31575 = invoke(stypy.reporting.localization.Localization(__file__, 283, 27), create_Name_31572, *[str_31573], **kwargs_31574)
        
        # Assigning a type to the variable 'type_of_self_arg' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'type_of_self_arg', create_Name_call_result_31575)
        # SSA join for if statement (line 278)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 286):
    
    # Call to obtain_arg_list(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'function_def_node' (line 286)
    function_def_node_31577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 43), 'function_def_node', False)
    # Obtaining the member 'args' of a type (line 286)
    args_31578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 43), function_def_node_31577, 'args')
    
    # Call to is_method(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'context' (line 286)
    context_31580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 77), 'context', False)
    # Processing the call keyword arguments (line 286)
    kwargs_31581 = {}
    # Getting the type of 'is_method' (line 286)
    is_method_31579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 67), 'is_method', False)
    # Calling is_method(args, kwargs) (line 286)
    is_method_call_result_31582 = invoke(stypy.reporting.localization.Localization(__file__, 286, 67), is_method_31579, *[context_31580], **kwargs_31581)
    
    
    # Call to is_static_method(...): (line 287)
    # Processing the call arguments (line 287)
    # Getting the type of 'function_def_node' (line 287)
    function_def_node_31584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 60), 'function_def_node', False)
    # Processing the call keyword arguments (line 287)
    kwargs_31585 = {}
    # Getting the type of 'is_static_method' (line 287)
    is_static_method_31583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 43), 'is_static_method', False)
    # Calling is_static_method(args, kwargs) (line 287)
    is_static_method_call_result_31586 = invoke(stypy.reporting.localization.Localization(__file__, 287, 43), is_static_method_31583, *[function_def_node_31584], **kwargs_31585)
    
    # Processing the call keyword arguments (line 286)
    kwargs_31587 = {}
    # Getting the type of 'obtain_arg_list' (line 286)
    obtain_arg_list_31576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 27), 'obtain_arg_list', False)
    # Calling obtain_arg_list(args, kwargs) (line 286)
    obtain_arg_list_call_result_31588 = invoke(stypy.reporting.localization.Localization(__file__, 286, 27), obtain_arg_list_31576, *[args_31578, is_method_call_result_31582, is_static_method_call_result_31586], **kwargs_31587)
    
    # Assigning a type to the variable 'param_names_list_arg' (line 286)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'param_names_list_arg', obtain_arg_list_call_result_31588)
    
    # Type idiom detected: calculating its left and rigth part (line 290)
    # Getting the type of 'function_def_node' (line 290)
    function_def_node_31589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 7), 'function_def_node')
    # Obtaining the member 'args' of a type (line 290)
    args_31590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 7), function_def_node_31589, 'args')
    # Obtaining the member 'vararg' of a type (line 290)
    vararg_31591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 7), args_31590, 'vararg')
    # Getting the type of 'None' (line 290)
    None_31592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 40), 'None')
    
    (may_be_31593, more_types_in_union_31594) = may_be_none(vararg_31591, None_31592)

    if may_be_31593:

        if more_types_in_union_31594:
            # Runtime conditional SSA (line 290)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 291):
        # Getting the type of 'None' (line 291)
        None_31595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 27), 'None')
        # Assigning a type to the variable 'declared_varargs' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'declared_varargs', None_31595)

        if more_types_in_union_31594:
            # Runtime conditional SSA for else branch (line 290)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_31593) or more_types_in_union_31594):
        
        # Assigning a Attribute to a Name (line 293):
        # Getting the type of 'function_def_node' (line 293)
        function_def_node_31596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 27), 'function_def_node')
        # Obtaining the member 'args' of a type (line 293)
        args_31597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 27), function_def_node_31596, 'args')
        # Obtaining the member 'vararg' of a type (line 293)
        vararg_31598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 27), args_31597, 'vararg')
        # Assigning a type to the variable 'declared_varargs' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'declared_varargs', vararg_31598)

        if (may_be_31593 and more_types_in_union_31594):
            # SSA join for if statement (line 290)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 294):
    
    # Call to create_str(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'declared_varargs' (line 294)
    declared_varargs_31601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 55), 'declared_varargs', False)
    # Processing the call keyword arguments (line 294)
    kwargs_31602 = {}
    # Getting the type of 'core_language_copy' (line 294)
    core_language_copy_31599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 25), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 294)
    create_str_31600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 294, 25), core_language_copy_31599, 'create_str')
    # Calling create_str(args, kwargs) (line 294)
    create_str_call_result_31603 = invoke(stypy.reporting.localization.Localization(__file__, 294, 25), create_str_31600, *[declared_varargs_31601], **kwargs_31602)
    
    # Assigning a type to the variable 'varargs_param_name' (line 294)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'varargs_param_name', create_str_call_result_31603)
    
    # Type idiom detected: calculating its left and rigth part (line 296)
    # Getting the type of 'function_def_node' (line 296)
    function_def_node_31604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 7), 'function_def_node')
    # Obtaining the member 'args' of a type (line 296)
    args_31605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 7), function_def_node_31604, 'args')
    # Obtaining the member 'kwarg' of a type (line 296)
    kwarg_31606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 7), args_31605, 'kwarg')
    # Getting the type of 'None' (line 296)
    None_31607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 39), 'None')
    
    (may_be_31608, more_types_in_union_31609) = may_be_none(kwarg_31606, None_31607)

    if may_be_31608:

        if more_types_in_union_31609:
            # Runtime conditional SSA (line 296)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 297):
        # Getting the type of 'None' (line 297)
        None_31610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 26), 'None')
        # Assigning a type to the variable 'declared_kwargs' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 8), 'declared_kwargs', None_31610)

        if more_types_in_union_31609:
            # Runtime conditional SSA for else branch (line 296)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_31608) or more_types_in_union_31609):
        
        # Assigning a Attribute to a Name (line 299):
        # Getting the type of 'function_def_node' (line 299)
        function_def_node_31611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 26), 'function_def_node')
        # Obtaining the member 'args' of a type (line 299)
        args_31612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 26), function_def_node_31611, 'args')
        # Obtaining the member 'kwarg' of a type (line 299)
        kwarg_31613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 26), args_31612, 'kwarg')
        # Assigning a type to the variable 'declared_kwargs' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'declared_kwargs', kwarg_31613)

        if (may_be_31608 and more_types_in_union_31609):
            # SSA join for if statement (line 296)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 300):
    
    # Call to create_str(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 'declared_kwargs' (line 300)
    declared_kwargs_31616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 54), 'declared_kwargs', False)
    # Processing the call keyword arguments (line 300)
    kwargs_31617 = {}
    # Getting the type of 'core_language_copy' (line 300)
    core_language_copy_31614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 24), 'core_language_copy', False)
    # Obtaining the member 'create_str' of a type (line 300)
    create_str_31615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 24), core_language_copy_31614, 'create_str')
    # Calling create_str(args, kwargs) (line 300)
    create_str_call_result_31618 = invoke(stypy.reporting.localization.Localization(__file__, 300, 24), create_str_31615, *[declared_kwargs_31616], **kwargs_31617)
    
    # Assigning a type to the variable 'kwargs_param_name' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'kwargs_param_name', create_str_call_result_31618)
    
    # Assigning a Call to a Name (line 304):
    
    # Call to create_Name(...): (line 304)
    # Processing the call arguments (line 304)
    str_31621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 51), 'str', 'defaults')
    # Processing the call keyword arguments (line 304)
    kwargs_31622 = {}
    # Getting the type of 'core_language_copy' (line 304)
    core_language_copy_31619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 304)
    create_Name_31620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 20), core_language_copy_31619, 'create_Name')
    # Calling create_Name(args, kwargs) (line 304)
    create_Name_call_result_31623 = invoke(stypy.reporting.localization.Localization(__file__, 304, 20), create_Name_31620, *[str_31621], **kwargs_31622)
    
    # Assigning a type to the variable 'call_defaults' (line 304)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'call_defaults', create_Name_call_result_31623)
    
    # Assigning a Call to a Name (line 307):
    
    # Call to create_Name(...): (line 307)
    # Processing the call arguments (line 307)
    str_31626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 50), 'str', 'varargs')
    # Processing the call keyword arguments (line 307)
    kwargs_31627 = {}
    # Getting the type of 'core_language_copy' (line 307)
    core_language_copy_31624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 19), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 307)
    create_Name_31625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 19), core_language_copy_31624, 'create_Name')
    # Calling create_Name(args, kwargs) (line 307)
    create_Name_call_result_31628 = invoke(stypy.reporting.localization.Localization(__file__, 307, 19), create_Name_31625, *[str_31626], **kwargs_31627)
    
    # Assigning a type to the variable 'call_varargs' (line 307)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'call_varargs', create_Name_call_result_31628)
    
    # Assigning a Call to a Name (line 309):
    
    # Call to create_Name(...): (line 309)
    # Processing the call arguments (line 309)
    str_31631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 49), 'str', 'kwargs')
    # Processing the call keyword arguments (line 309)
    kwargs_31632 = {}
    # Getting the type of 'core_language_copy' (line 309)
    core_language_copy_31629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 18), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 309)
    create_Name_31630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 18), core_language_copy_31629, 'create_Name')
    # Calling create_Name(args, kwargs) (line 309)
    create_Name_call_result_31633 = invoke(stypy.reporting.localization.Localization(__file__, 309, 18), create_Name_31630, *[str_31631], **kwargs_31632)
    
    # Assigning a type to the variable 'call_kwargs' (line 309)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'call_kwargs', create_Name_call_result_31633)
    
    # Assigning a Call to a Name (line 312):
    
    # Call to create_call(...): (line 312)
    # Processing the call arguments (line 312)
    # Getting the type of 'func' (line 312)
    func_31635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 23), 'func', False)
    
    # Obtaining an instance of the builtin type 'list' (line 313)
    list_31636 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 313, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 313)
    # Adding element type (line 313)
    # Getting the type of 'localization_arg' (line 313)
    localization_arg_31637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 24), 'localization_arg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_31636, localization_arg_31637)
    # Adding element type (line 313)
    # Getting the type of 'type_of_self_arg' (line 313)
    type_of_self_arg_31638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 42), 'type_of_self_arg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_31636, type_of_self_arg_31638)
    # Adding element type (line 313)
    # Getting the type of 'type_store_arg' (line 313)
    type_store_arg_31639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 60), 'type_store_arg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_31636, type_store_arg_31639)
    # Adding element type (line 313)
    # Getting the type of 'function_name_arg' (line 313)
    function_name_arg_31640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 76), 'function_name_arg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_31636, function_name_arg_31640)
    # Adding element type (line 313)
    # Getting the type of 'param_names_list_arg' (line 313)
    param_names_list_arg_31641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 95), 'param_names_list_arg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_31636, param_names_list_arg_31641)
    # Adding element type (line 313)
    # Getting the type of 'varargs_param_name' (line 314)
    varargs_param_name_31642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 24), 'varargs_param_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_31636, varargs_param_name_31642)
    # Adding element type (line 313)
    # Getting the type of 'kwargs_param_name' (line 314)
    kwargs_param_name_31643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 44), 'kwargs_param_name', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_31636, kwargs_param_name_31643)
    # Adding element type (line 313)
    # Getting the type of 'call_defaults' (line 314)
    call_defaults_31644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 63), 'call_defaults', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_31636, call_defaults_31644)
    # Adding element type (line 313)
    # Getting the type of 'call_varargs' (line 314)
    call_varargs_31645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 78), 'call_varargs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_31636, call_varargs_31645)
    # Adding element type (line 313)
    # Getting the type of 'call_kwargs' (line 314)
    call_kwargs_31646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 92), 'call_kwargs', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 313, 23), list_31636, call_kwargs_31646)
    
    # Processing the call keyword arguments (line 312)
    kwargs_31647 = {}
    # Getting the type of 'create_call' (line 312)
    create_call_31634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 11), 'create_call', False)
    # Calling create_call(args, kwargs) (line 312)
    create_call_call_result_31648 = invoke(stypy.reporting.localization.Localization(__file__, 312, 11), create_call_31634, *[func_31635, list_31636], **kwargs_31647)
    
    # Assigning a type to the variable 'call' (line 312)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 312, 4), 'call', create_call_call_result_31648)
    
    # Assigning a Call to a Name (line 316):
    
    # Call to create_Assign(...): (line 316)
    # Processing the call arguments (line 316)
    # Getting the type of 'args_test_resul' (line 316)
    args_test_resul_31651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 46), 'args_test_resul', False)
    # Getting the type of 'call' (line 316)
    call_31652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 63), 'call', False)
    # Processing the call keyword arguments (line 316)
    kwargs_31653 = {}
    # Getting the type of 'core_language_copy' (line 316)
    core_language_copy_31649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 13), 'core_language_copy', False)
    # Obtaining the member 'create_Assign' of a type (line 316)
    create_Assign_31650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 13), core_language_copy_31649, 'create_Assign')
    # Calling create_Assign(args, kwargs) (line 316)
    create_Assign_call_result_31654 = invoke(stypy.reporting.localization.Localization(__file__, 316, 13), create_Assign_31650, *[args_test_resul_31651, call_31652], **kwargs_31653)
    
    # Assigning a type to the variable 'assign' (line 316)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 316, 4), 'assign', create_Assign_call_result_31654)
    
    # Assigning a Call to a Name (line 319):
    
    # Call to create_Name(...): (line 319)
    # Processing the call arguments (line 319)
    str_31657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 319, 53), 'str', 'arguments')
    # Processing the call keyword arguments (line 319)
    kwargs_31658 = {}
    # Getting the type of 'core_language_copy' (line 319)
    core_language_copy_31655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 319, 22), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 319)
    create_Name_31656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 319, 22), core_language_copy_31655, 'create_Name')
    # Calling create_Name(args, kwargs) (line 319)
    create_Name_call_result_31659 = invoke(stypy.reporting.localization.Localization(__file__, 319, 22), create_Name_31656, *[str_31657], **kwargs_31658)
    
    # Assigning a type to the variable 'argument_errors' (line 319)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 319, 4), 'argument_errors', create_Name_call_result_31659)
    
    # Assigning a Call to a Name (line 320):
    
    # Call to create_Name(...): (line 320)
    # Processing the call arguments (line 320)
    str_31662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 320, 51), 'str', 'is_error_type')
    # Processing the call keyword arguments (line 320)
    kwargs_31663 = {}
    # Getting the type of 'core_language_copy' (line 320)
    core_language_copy_31660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 20), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 320)
    create_Name_31661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 20), core_language_copy_31660, 'create_Name')
    # Calling create_Name(args, kwargs) (line 320)
    create_Name_call_result_31664 = invoke(stypy.reporting.localization.Localization(__file__, 320, 20), create_Name_31661, *[str_31662], **kwargs_31663)
    
    # Assigning a type to the variable 'is_error_type' (line 320)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 320, 4), 'is_error_type', create_Name_call_result_31664)
    
    # Assigning a Call to a Name (line 321):
    
    # Call to create_call(...): (line 321)
    # Processing the call arguments (line 321)
    # Getting the type of 'is_error_type' (line 321)
    is_error_type_31666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 26), 'is_error_type', False)
    # Getting the type of 'argument_errors' (line 321)
    argument_errors_31667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 41), 'argument_errors', False)
    # Processing the call keyword arguments (line 321)
    kwargs_31668 = {}
    # Getting the type of 'create_call' (line 321)
    create_call_31665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 14), 'create_call', False)
    # Calling create_call(args, kwargs) (line 321)
    create_call_call_result_31669 = invoke(stypy.reporting.localization.Localization(__file__, 321, 14), create_call_31665, *[is_error_type_31666, argument_errors_31667], **kwargs_31668)
    
    # Assigning a type to the variable 'if_test' (line 321)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 4), 'if_test', create_call_call_result_31669)
    
    # Call to is_constructor(...): (line 323)
    # Processing the call arguments (line 323)
    # Getting the type of 'function_def_node' (line 323)
    function_def_node_31671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 22), 'function_def_node', False)
    # Processing the call keyword arguments (line 323)
    kwargs_31672 = {}
    # Getting the type of 'is_constructor' (line 323)
    is_constructor_31670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 7), 'is_constructor', False)
    # Calling is_constructor(args, kwargs) (line 323)
    is_constructor_call_result_31673 = invoke(stypy.reporting.localization.Localization(__file__, 323, 7), is_constructor_31670, *[function_def_node_31671], **kwargs_31672)
    
    # Testing if the type of an if condition is none (line 323)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 323, 4), is_constructor_call_result_31673):
        pass
    else:
        
        # Testing the type of an if condition (line 323)
        if_condition_31674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 323, 4), is_constructor_call_result_31673)
        # Assigning a type to the variable 'if_condition_31674' (line 323)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 4), 'if_condition_31674', if_condition_31674)
        # SSA begins for if statement (line 323)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 324):
        # Getting the type of 'None' (line 324)
        None_31675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 26), 'None')
        # Assigning a type to the variable 'argument_errors' (line 324)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 324, 8), 'argument_errors', None_31675)
        # SSA join for if statement (line 323)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Name (line 326):
    
    # Obtaining an instance of the builtin type 'list' (line 326)
    list_31676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 326, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 326)
    # Adding element type (line 326)
    
    # Call to create_context_unset(...): (line 326)
    # Processing the call keyword arguments (line 326)
    kwargs_31678 = {}
    # Getting the type of 'create_context_unset' (line 326)
    create_context_unset_31677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 12), 'create_context_unset', False)
    # Calling create_context_unset(args, kwargs) (line 326)
    create_context_unset_call_result_31679 = invoke(stypy.reporting.localization.Localization(__file__, 326, 12), create_context_unset_31677, *[], **kwargs_31678)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 11), list_31676, create_context_unset_call_result_31679)
    # Adding element type (line 326)
    
    # Call to create_return(...): (line 326)
    # Processing the call arguments (line 326)
    # Getting the type of 'argument_errors' (line 326)
    argument_errors_31681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 50), 'argument_errors', False)
    # Processing the call keyword arguments (line 326)
    kwargs_31682 = {}
    # Getting the type of 'create_return' (line 326)
    create_return_31680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 36), 'create_return', False)
    # Calling create_return(args, kwargs) (line 326)
    create_return_call_result_31683 = invoke(stypy.reporting.localization.Localization(__file__, 326, 36), create_return_31680, *[argument_errors_31681], **kwargs_31682)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 326, 11), list_31676, create_return_call_result_31683)
    
    # Assigning a type to the variable 'body' (line 326)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 4), 'body', list_31676)
    
    # Assigning a Call to a Name (line 327):
    
    # Call to create_if(...): (line 327)
    # Processing the call arguments (line 327)
    # Getting the type of 'if_test' (line 327)
    if_test_31686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 48), 'if_test', False)
    # Getting the type of 'body' (line 327)
    body_31687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 57), 'body', False)
    # Processing the call keyword arguments (line 327)
    kwargs_31688 = {}
    # Getting the type of 'conditional_statements_copy' (line 327)
    conditional_statements_copy_31684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 10), 'conditional_statements_copy', False)
    # Obtaining the member 'create_if' of a type (line 327)
    create_if_31685 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 327, 10), conditional_statements_copy_31684, 'create_if')
    # Calling create_if(args, kwargs) (line 327)
    create_if_call_result_31689 = invoke(stypy.reporting.localization.Localization(__file__, 327, 10), create_if_31685, *[if_test_31686, body_31687], **kwargs_31688)
    
    # Assigning a type to the variable 'if_' (line 327)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 4), 'if_', create_if_call_result_31689)
    
    # Obtaining an instance of the builtin type 'list' (line 329)
    list_31690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 329)
    # Adding element type (line 329)
    # Getting the type of 'assign' (line 329)
    assign_31691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 12), 'assign')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 11), list_31690, assign_31691)
    # Adding element type (line 329)
    # Getting the type of 'if_' (line 329)
    if__31692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 329, 20), 'if_')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 329, 11), list_31690, if__31692)
    
    # Assigning a type to the variable 'stypy_return_type' (line 329)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 329, 4), 'stypy_return_type', list_31690)
    
    # ################# End of 'create_arg_number_test(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_arg_number_test' in the type store
    # Getting the type of 'stypy_return_type' (line 258)
    stypy_return_type_31693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31693)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_arg_number_test'
    return stypy_return_type_31693

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

    str_31694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, (-1)), 'str', '\n    Creates a variable to store a lambda function definition\n\n    :param function_name: Name of the lambda function\n    :param lambda_call: Lambda function\n    :param lineno: Line\n    :param col_offset: Column\n    :return: Statements to create the lambda function type\n    ')
    
    # Assigning a Call to a Name (line 349):
    
    # Call to create_Name(...): (line 349)
    # Processing the call arguments (line 349)
    # Getting the type of 'lambda_call' (line 349)
    lambda_call_31697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 46), 'lambda_call', False)
    # Processing the call keyword arguments (line 349)
    kwargs_31698 = {}
    # Getting the type of 'core_language_copy' (line 349)
    core_language_copy_31695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 15), 'core_language_copy', False)
    # Obtaining the member 'create_Name' of a type (line 349)
    create_Name_31696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 349, 15), core_language_copy_31695, 'create_Name')
    # Calling create_Name(args, kwargs) (line 349)
    create_Name_call_result_31699 = invoke(stypy.reporting.localization.Localization(__file__, 349, 15), create_Name_31696, *[lambda_call_31697], **kwargs_31698)
    
    # Assigning a type to the variable 'call_arg' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'call_arg', create_Name_call_result_31699)
    
    # Assigning a Call to a Name (line 351):
    
    # Call to create_set_type_of(...): (line 351)
    # Processing the call arguments (line 351)
    # Getting the type of 'function_name' (line 351)
    function_name_31702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 61), 'function_name', False)
    # Getting the type of 'call_arg' (line 351)
    call_arg_31703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 76), 'call_arg', False)
    # Getting the type of 'lineno' (line 351)
    lineno_31704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 86), 'lineno', False)
    # Getting the type of 'col_offset' (line 351)
    col_offset_31705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 94), 'col_offset', False)
    # Processing the call keyword arguments (line 351)
    kwargs_31706 = {}
    # Getting the type of 'stypy_functions_copy' (line 351)
    stypy_functions_copy_31700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 21), 'stypy_functions_copy', False)
    # Obtaining the member 'create_set_type_of' of a type (line 351)
    create_set_type_of_31701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 351, 21), stypy_functions_copy_31700, 'create_set_type_of')
    # Calling create_set_type_of(args, kwargs) (line 351)
    create_set_type_of_call_result_31707 = invoke(stypy.reporting.localization.Localization(__file__, 351, 21), create_set_type_of_31701, *[function_name_31702, call_arg_31703, lineno_31704, col_offset_31705], **kwargs_31706)
    
    # Assigning a type to the variable 'set_type_stmts' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'set_type_stmts', create_set_type_of_call_result_31707)
    
    # Call to flatten_lists(...): (line 354)
    # Processing the call arguments (line 354)
    # Getting the type of 'set_type_stmts' (line 354)
    set_type_stmts_31710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 46), 'set_type_stmts', False)
    # Processing the call keyword arguments (line 354)
    kwargs_31711 = {}
    # Getting the type of 'stypy_functions_copy' (line 354)
    stypy_functions_copy_31708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 11), 'stypy_functions_copy', False)
    # Obtaining the member 'flatten_lists' of a type (line 354)
    flatten_lists_31709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 354, 11), stypy_functions_copy_31708, 'flatten_lists')
    # Calling flatten_lists(args, kwargs) (line 354)
    flatten_lists_call_result_31712 = invoke(stypy.reporting.localization.Localization(__file__, 354, 11), flatten_lists_31709, *[set_type_stmts_31710], **kwargs_31711)
    
    # Assigning a type to the variable 'stypy_return_type' (line 354)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 4), 'stypy_return_type', flatten_lists_call_result_31712)
    
    # ################# End of 'create_type_for_lambda_function(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_type_for_lambda_function' in the type store
    # Getting the type of 'stypy_return_type' (line 332)
    stypy_return_type_31713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31713)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_type_for_lambda_function'
    return stypy_return_type_31713

# Assigning a type to the variable 'create_type_for_lambda_function' (line 332)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 332, 0), 'create_type_for_lambda_function', create_type_for_lambda_function)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
