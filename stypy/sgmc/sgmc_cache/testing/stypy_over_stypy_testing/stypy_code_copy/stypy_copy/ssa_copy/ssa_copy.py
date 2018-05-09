
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy.type_store_copy.typestore_copy import TypeStore
2: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
3: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
4: 
5: '''
6: Implementation of the SSA algorithm to calculate types of variables when dealing with branches in source code (ifs,
7: loops, ...)
8: '''
9: 
10: 
11: # TODO: Remove?
12: # from stypy.type_store.function_context import FunctionContext
13: 
14: # ############## SSA FOR CLAUSES WITH AN OPTIONAL ELSE BRANCH (IF, FOR, WHILE...) ###############
15: #
16: # def __join_annotations(function_context_a, function_context_b):
17: #     # Curiously, declaring a "global" in one of the branches avoids the potential unreferenced variable error for all
18: # of them, so
19: #     # we simply add the variables of both branches.
20: #     a_annotations = function_context_a.annotation_record
21: #     if (function_context_b is None):
22: #         b_annotations = dict()
23: #     else:
24: #         b_annotations = function_context_b.annotation_record.annotation_dict
25: #
26: #     for (line, annotations) in b_annotations.items():
27: #         for annotation in annotations:
28: #             a_annotations.annotate_type(line, annotation[2], annotation[0], annotation[1])
29: #
30: #     return a_annotations
31: 
32: 
33: def __join_globals(function_context_if, function_context_else):
34:     '''
35:     Join the global variables placed in two function contexts
36:     :param function_context_if: Function context
37:     :param function_context_else: Function context
38:     :return: The first function context with the globals of both of them
39:     '''
40:     # Curiously, declaring a "global" in one of the branches avoids the potential unreferenced variable error for all
41:     # of them, so we simply add the variables of both branches.
42:     if_globals = function_context_if.global_vars
43:     if function_context_else is None:
44:         else_globals = []
45:     else:
46:         else_globals = function_context_else.global_vars
47: 
48:     for var in else_globals:
49:         if var not in if_globals:
50:             if_globals.append(var)
51: 
52:     return if_globals
53: 
54: 
55: def __ssa_join_with_else_function_context(function_context_previous, function_context_if, function_context_else):
56:     '''
57:     Helper function of the SSA implementation of an if-else structure, used with each function context in the type
58:     store
59:     :param function_context_previous: Function context
60:     :param function_context_if: Function context
61:     :param function_context_else: Function context
62:     :return: A dictionary with names of variables and its joined types
63:     '''
64:     type_dict = {}
65: 
66:     if function_context_else is None:
67:         function_context_else = []  # Only the if branch is present
68: 
69:     for var_name in function_context_if:
70:         if var_name in function_context_else:
71:             # Variable defined in if and else body
72:             type_dict[var_name] = union_type_copy.UnionType.add(function_context_if[var_name],
73:                                                            function_context_else[var_name])
74:         else:
75:             # Variable defined in if and in the previous context
76:             if var_name in function_context_previous:
77:                 # Variable defined in if body (the previous type is then considered)
78:                 type_dict[var_name] = union_type_copy.UnionType.add(function_context_previous[var_name],
79:                                                                function_context_if[var_name])
80:             else:
81:                 # Variable defined in if body, but did not existed in the previous type store (it could be not defined)
82:                 type_dict[var_name] = union_type_copy.UnionType.add(function_context_if[var_name], UndefinedType())
83: 
84:     for var_name in function_context_else:
85:         if var_name in function_context_if:
86:             continue  # Already processed (above)
87:         else:
88:             # Variable defined in the else body, but not in the if body
89:             if var_name in function_context_previous:
90:                 # Variable defined in else (the previous one is considered)
91:                 type_dict[var_name] = union_type_copy.UnionType.add(function_context_previous[var_name],
92:                                                                function_context_else[var_name])
93:             else:
94:                 # Variable defined in else body, but did not existed in the previous type store (it could be not
95:                 # defined)
96:                 type_dict[var_name] = union_type_copy.UnionType.add(function_context_else[var_name], UndefinedType())
97: 
98:     # type_store_previous does not need to be iterated because it is included in the if and else stores
99:     return type_dict
100: 
101: 
102: def ssa_join_with_else_branch(type_store_previous, type_store_if, type_store_else):
103:     '''
104:     Implements the SSA algorithm with the type stores of an if-else structure
105:     :param type_store_previous: Type store
106:     :param type_store_if: Function context
107:     :param type_store_else:
108:     :return:
109:     '''
110:     # Join the variables of the previous, the if and the else branches type stores into a single dict
111:     joined_type_store = TypeStore(type_store_previous.program_name)
112:     joined_type_store.last_function_contexts = type_store_previous.last_function_contexts
113:     joined_type_store.context_stack = []  # Need to empty it, as a default function context is created when initializing
114:     for i in range(len(type_store_previous.context_stack)):
115:         # Only an if branch?
116:         if type_store_else is None:
117:             function_context_else = None
118:         else:
119:             function_context_else = type_store_else[i]
120: 
121:         joined_context_dict = __ssa_join_with_else_function_context(type_store_previous[i], type_store_if[i],
122:                                                                     function_context_else)
123: 
124:         # joined_f_context = FunctionContext(type_store_previous[i].function_name)
125:         joined_f_context = type_store_previous[i].copy()
126:         joined_f_context.types_of = joined_context_dict
127:         joined_f_context.global_vars = __join_globals(type_store_if[i], function_context_else)
128:         joined_f_context.annotation_record = type_store_if[
129:             i].annotation_record  # __join_annotations(type_store_if[i], function_context_else[i])
130: 
131:         joined_type_store.context_stack.append(joined_f_context)
132: 
133:     return joined_type_store
134: 
135: 
136: # ############## SSA FOR EXCEPTION SENTENCES ###############
137: 
138: def __join_except_branches_function_context(function_context_previous, function_context_new):
139:     '''
140:     Helper function to join variables of function contexts that belong to different except
141:     blocks
142:     :param function_context_previous: Function context
143:     :param function_context_new: Function context
144:     :return: A dictionary with names of variables and its joined types
145:     '''
146:     type_dict = {}
147: 
148:     for var_name in function_context_previous:
149:         if var_name in function_context_new:
150:             # Variable defined in both function contexts
151:             type_dict[var_name] = union_type_copy.UnionType.add(function_context_previous[var_name],
152:                                                            function_context_new[var_name])
153:         else:
154:             # Variable defined in previous but not on new function context
155:             type_dict[var_name] = function_context_previous[var_name]
156: 
157:     for var_name in function_context_new:
158:         if var_name in function_context_previous:
159:             # Variable defined in both function contexts
160:             type_dict[var_name] = union_type_copy.UnionType.add(function_context_new[var_name],
161:                                                            function_context_previous[var_name])
162:         else:
163:             # Variable defined in new but not on previous function context
164:             type_dict[var_name] = function_context_new[var_name]
165: 
166:     # type_store_previous does not need to be iterated because it is included in the if and else stores
167:     return type_dict
168: 
169: 
170: def __join_except_branches(type_store_previous, type_store_new):
171:     '''
172:     SSA algorithm to join type stores of different except branches
173:     :param type_store_previous: Type store
174:     :param type_store_new: Type store
175:     :return:
176:     '''
177:     joined_type_store = TypeStore(type_store_previous.program_name)
178:     joined_type_store.last_function_contexts = type_store_previous.last_function_contexts
179:     joined_type_store.context_stack = []  # Need to empty it, as a default function context is created when initializing
180:     for i in range(len(type_store_previous.context_stack)):
181:         joined_context_dict = __join_except_branches_function_context(type_store_previous[i], type_store_new[i])
182:         # joined_f_context = FunctionContext(type_store_previous[i].function_name)
183:         joined_f_context = type_store_previous[i].copy()
184:         joined_f_context.types_of = joined_context_dict
185:         joined_f_context.global_vars = __join_globals(type_store_previous[i], type_store_new[i])
186:         joined_f_context.annotation_record = type_store_previous[
187:             i].annotation_record  # __join_annotations(type_store_previous[i], type_store_new[i])
188: 
189:         joined_type_store.context_stack.append(joined_f_context)
190: 
191:     return joined_type_store
192: 
193: 
194: def __join_finally_function_context(function_context_previous, function_context_finally):
195:     '''
196:     Join the variables of a function context on a finally block with a function context of the joined type store
197:      of all the except branches in an exception clause
198:     :param function_context_previous: Function context
199:     :param function_context_finally: Function context
200:     :return: A dictionary with names of variables and its joined types
201:     '''
202:     type_dict = {}
203: 
204:     for var_name in function_context_previous:
205:         if var_name in function_context_finally:
206:             # Variable defined in both function contexts
207:             type_dict[var_name] = function_context_finally[var_name]
208:         else:
209:             # Variable defined in previous but not on new function context
210:             pass
211: 
212:     for var_name in function_context_finally:
213:         if var_name in function_context_previous:
214:             # Variable defined in both function contexts
215:             pass  # Already covered
216:         else:
217:             # Variable defined in new but not on previous function context
218:             type_dict[var_name] = function_context_finally[var_name]
219: 
220:     # type_store_previous does not need to be iterated because it is included in the if and else stores
221:     return type_dict
222: 
223: 
224: def __join_finally_branch(type_store_exception_block, type_store_finally):
225:     '''
226:     Join the type stores of a finally branch and the joined type store of all except branches in a exception handling
227:      block
228:     :param type_store_exception_block: Type store
229:     :param type_store_finally: Type store
230:     :return:
231:     '''
232:     joined_type_store = TypeStore(type_store_exception_block.program_name)
233:     joined_type_store.last_function_contexts = type_store_exception_block.last_function_contexts
234:     joined_type_store.context_stack = []  # Need to empty it, as a default function context is created when initializing
235:     for i in range(len(type_store_exception_block.context_stack)):
236:         joined_context_dict = __join_finally_function_context(type_store_exception_block[i], type_store_finally[i])
237:         # joined_f_context = FunctionContext(type_store_exception_block[i].function_name)
238:         joined_f_context = type_store_exception_block[i].copy()
239:         joined_f_context.types_of = joined_context_dict
240:         joined_f_context.global_vars = __join_globals(type_store_exception_block[i], type_store_finally[i])
241:         joined_f_context.annotation_record = type_store_exception_block[
242:             i].annotation_record  # __join_annotations(type_store_exception_block[i], type_store_finally[i])
243: 
244:         joined_type_store.context_stack.append(joined_f_context)
245: 
246:     return joined_type_store
247: 
248: 
249: def __join_try_except_function_context(function_context_previous, function_context_try, function_context_except):
250:     '''
251:     Implements the SSA algorithm in try-except blocks, dealing with function contexts.
252: 
253:     :param function_context_previous: Function context
254:     :param function_context_try: Function context
255:     :param function_context_except: Function context
256:     :return: A dictionary with names of variables and its joined types
257:     '''
258:     type_dict = {}
259: 
260:     for var_name in function_context_try:
261:         if var_name in function_context_except:
262:             # Variable defined in if and else body
263:             type_dict[var_name] = union_type_copy.UnionType.add(function_context_try[var_name],
264:                                                            function_context_except[var_name])
265:             if var_name not in function_context_previous:
266:                 type_dict[var_name] = union_type_copy.UnionType.add(type_dict[var_name], UndefinedType())
267:         else:
268:             # Variable defined in if but not else body
269:             if var_name in function_context_previous:
270:                 # Variable defined in if body (the previous type is then considered)
271:                 type_dict[var_name] = union_type_copy.UnionType.add(function_context_previous[var_name],
272:                                                                function_context_try[var_name])
273:             else:
274:                 # Variable defined in if body, but did not existed in the previous type store (it could be not defined)
275:                 type_dict[var_name] = union_type_copy.UnionType.add(function_context_try[var_name], UndefinedType())
276: 
277:     for var_name in function_context_except:
278:         if var_name in function_context_try:
279:             continue  # Already processed (above)
280:         else:
281:             # Variable defined in the else body, but not in the if body
282:             if var_name in function_context_previous:
283:                 # Variable defined in else (the previous one is considered)
284:                 type_dict[var_name] = union_type_copy.UnionType.add(function_context_previous[var_name],
285:                                                                function_context_except[var_name])
286:             else:
287:                 # Variable defined in else body, but did not existed in the previous type store (it could be not
288:                 # defined)
289:                 type_dict[var_name] = union_type_copy.UnionType.add(function_context_except[var_name], UndefinedType())
290: 
291:     # type_store_previous does not need to be iterated because it is included in the if and else stores
292:     return type_dict
293: 
294: 
295: def __join__try_except(type_store_previous, type_store_posttry, type_store_excepts):
296:     '''
297:     SSA Algotihm implementation for type stores in a try-except block
298:     :param type_store_previous: Type store
299:     :param type_store_posttry: Type store
300:     :param type_store_excepts: Type store
301:     :return:
302:     '''
303:     joined_type_store = TypeStore(type_store_previous.program_name)
304:     joined_type_store.last_function_contexts = type_store_previous.last_function_contexts
305:     joined_type_store.context_stack = []  # Need to empty it, as a default function context is created when initializing
306:     for i in range(len(type_store_previous.context_stack)):
307:         joined_context_dict = __join_try_except_function_context(type_store_previous[i], type_store_posttry[i],
308:                                                                  type_store_excepts[i])
309:         # joined_f_context = FunctionContext(type_store_previous[i].function_name)
310:         joined_f_context = type_store_previous[i].copy()
311:         joined_f_context.types_of = joined_context_dict
312:         joined_f_context.global_vars = __join_globals(type_store_posttry[i], type_store_excepts[i])
313:         joined_f_context.annotation_record = type_store_posttry[
314:             i].annotation_record  # __join_annotations(type_store_posttry[i], type_store_excepts[i])
315: 
316:         joined_type_store.context_stack.append(joined_f_context)
317: 
318:     return joined_type_store
319: 
320: 
321: def join_exception_block(type_store_pretry, type_store_posttry, type_store_finally=None, *type_store_except_branches):
322:     '''
323:     Implements the SSA algorithm for a full try-except-finally block, calling previous function
324:     :param type_store_pretry: Type store
325:     :param type_store_posttry: Type store
326:     :param type_store_finally: Type store
327:     :param type_store_except_branches: Type store
328:     :return:
329:     '''
330:     # Join the variables of the previous, the if and the else branches type stores into a single dict
331:     # types_dict = __join_if_else_function_context(type_store_previous, type_store_if, type_store_else)
332: 
333:     joined_type_store = TypeStore(type_store_pretry.program_name)
334:     joined_type_store.last_function_contexts = type_store_pretry.last_function_contexts
335:     joined_type_store.context_stack = []  # Need to empty it, as a default function context is created when initializing
336: 
337:     # Process all except branches to leave a single type store. else branch is treated as an additional except branch.
338:     if len(type_store_except_branches) == 1:
339:         type_store_excepts = type_store_except_branches[0]
340:     else:
341:         cont = 1
342:         type_store_excepts = type_store_except_branches[0]
343:         while cont < len(type_store_except_branches):
344:             type_store_excepts = __join_except_branches(type_store_excepts, type_store_except_branches[cont])
345:             cont += 1
346: 
347:     # Join the pre exception block type store with the try branch and the union of all the except branches
348:     joined_context_dict = __join__try_except(type_store_pretry, type_store_posttry,
349:                                              type_store_excepts)
350: 
351:     # Finally is special because it overwrites the type of already defined variables
352:     if type_store_finally is not None:
353:         joined_context_dict = __join_finally_branch(joined_context_dict, type_store_finally)
354: 
355:     return joined_context_dict
356: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from stypy_copy.type_store_copy.typestore_copy import TypeStore' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/ssa_copy/')
import_20485 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.type_store_copy.typestore_copy')

if (type(import_20485) is not StypyTypeError):

    if (import_20485 != 'pyd_module'):
        __import__(import_20485)
        sys_modules_20486 = sys.modules[import_20485]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.type_store_copy.typestore_copy', sys_modules_20486.module_type_store, module_type_store, ['TypeStore'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_20486, sys_modules_20486.module_type_store, module_type_store)
    else:
        from stypy_copy.type_store_copy.typestore_copy import TypeStore

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.type_store_copy.typestore_copy', None, module_type_store, ['TypeStore'], [TypeStore])

else:
    # Assigning a type to the variable 'stypy_copy.type_store_copy.typestore_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.type_store_copy.typestore_copy', import_20485)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/ssa_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/ssa_copy/')
import_20487 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_20487) is not StypyTypeError):

    if (import_20487 != 'pyd_module'):
        __import__(import_20487)
        sys_modules_20488 = sys.modules[import_20487]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_20488.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_20488, sys_modules_20488.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_20487)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/ssa_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/ssa_copy/')
import_20489 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy')

if (type(import_20489) is not StypyTypeError):

    if (import_20489 != 'pyd_module'):
        __import__(import_20489)
        sys_modules_20490 = sys.modules[import_20489]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', sys_modules_20490.module_type_store, module_type_store, ['UndefinedType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_20490, sys_modules_20490.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', None, module_type_store, ['UndefinedType'], [UndefinedType])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', import_20489)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/ssa_copy/')

str_20491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', '\nImplementation of the SSA algorithm to calculate types of variables when dealing with branches in source code (ifs,\nloops, ...)\n')

@norecursion
def __join_globals(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__join_globals'
    module_type_store = module_type_store.open_function_context('__join_globals', 33, 0, False)
    
    # Passed parameters checking function
    __join_globals.stypy_localization = localization
    __join_globals.stypy_type_of_self = None
    __join_globals.stypy_type_store = module_type_store
    __join_globals.stypy_function_name = '__join_globals'
    __join_globals.stypy_param_names_list = ['function_context_if', 'function_context_else']
    __join_globals.stypy_varargs_param_name = None
    __join_globals.stypy_kwargs_param_name = None
    __join_globals.stypy_call_defaults = defaults
    __join_globals.stypy_call_varargs = varargs
    __join_globals.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__join_globals', ['function_context_if', 'function_context_else'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__join_globals', localization, ['function_context_if', 'function_context_else'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__join_globals(...)' code ##################

    str_20492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'str', '\n    Join the global variables placed in two function contexts\n    :param function_context_if: Function context\n    :param function_context_else: Function context\n    :return: The first function context with the globals of both of them\n    ')
    
    # Assigning a Attribute to a Name (line 42):
    # Getting the type of 'function_context_if' (line 42)
    function_context_if_20493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'function_context_if')
    # Obtaining the member 'global_vars' of a type (line 42)
    global_vars_20494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 17), function_context_if_20493, 'global_vars')
    # Assigning a type to the variable 'if_globals' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'if_globals', global_vars_20494)
    
    # Type idiom detected: calculating its left and rigth part (line 43)
    # Getting the type of 'function_context_else' (line 43)
    function_context_else_20495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 7), 'function_context_else')
    # Getting the type of 'None' (line 43)
    None_20496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'None')
    
    (may_be_20497, more_types_in_union_20498) = may_be_none(function_context_else_20495, None_20496)

    if may_be_20497:

        if more_types_in_union_20498:
            # Runtime conditional SSA (line 43)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Name (line 44):
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_20499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        
        # Assigning a type to the variable 'else_globals' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'else_globals', list_20499)

        if more_types_in_union_20498:
            # Runtime conditional SSA for else branch (line 43)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_20497) or more_types_in_union_20498):
        
        # Assigning a Attribute to a Name (line 46):
        # Getting the type of 'function_context_else' (line 46)
        function_context_else_20500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'function_context_else')
        # Obtaining the member 'global_vars' of a type (line 46)
        global_vars_20501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 23), function_context_else_20500, 'global_vars')
        # Assigning a type to the variable 'else_globals' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'else_globals', global_vars_20501)

        if (may_be_20497 and more_types_in_union_20498):
            # SSA join for if statement (line 43)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'else_globals' (line 48)
    else_globals_20502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'else_globals')
    # Assigning a type to the variable 'else_globals_20502' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'else_globals_20502', else_globals_20502)
    # Testing if the for loop is going to be iterated (line 48)
    # Testing the type of a for loop iterable (line 48)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 4), else_globals_20502)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 48, 4), else_globals_20502):
        # Getting the type of the for loop variable (line 48)
        for_loop_var_20503 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 4), else_globals_20502)
        # Assigning a type to the variable 'var' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'var', for_loop_var_20503)
        # SSA begins for a for statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var' (line 49)
        var_20504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'var')
        # Getting the type of 'if_globals' (line 49)
        if_globals_20505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'if_globals')
        # Applying the binary operator 'notin' (line 49)
        result_contains_20506 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 11), 'notin', var_20504, if_globals_20505)
        
        # Testing if the type of an if condition is none (line 49)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 49, 8), result_contains_20506):
            pass
        else:
            
            # Testing the type of an if condition (line 49)
            if_condition_20507 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 8), result_contains_20506)
            # Assigning a type to the variable 'if_condition_20507' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'if_condition_20507', if_condition_20507)
            # SSA begins for if statement (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 50)
            # Processing the call arguments (line 50)
            # Getting the type of 'var' (line 50)
            var_20510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 30), 'var', False)
            # Processing the call keyword arguments (line 50)
            kwargs_20511 = {}
            # Getting the type of 'if_globals' (line 50)
            if_globals_20508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'if_globals', False)
            # Obtaining the member 'append' of a type (line 50)
            append_20509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), if_globals_20508, 'append')
            # Calling append(args, kwargs) (line 50)
            append_call_result_20512 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), append_20509, *[var_20510], **kwargs_20511)
            
            # SSA join for if statement (line 49)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'if_globals' (line 52)
    if_globals_20513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'if_globals')
    # Assigning a type to the variable 'stypy_return_type' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type', if_globals_20513)
    
    # ################# End of '__join_globals(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join_globals' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_20514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20514)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join_globals'
    return stypy_return_type_20514

# Assigning a type to the variable '__join_globals' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), '__join_globals', __join_globals)

@norecursion
def __ssa_join_with_else_function_context(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__ssa_join_with_else_function_context'
    module_type_store = module_type_store.open_function_context('__ssa_join_with_else_function_context', 55, 0, False)
    
    # Passed parameters checking function
    __ssa_join_with_else_function_context.stypy_localization = localization
    __ssa_join_with_else_function_context.stypy_type_of_self = None
    __ssa_join_with_else_function_context.stypy_type_store = module_type_store
    __ssa_join_with_else_function_context.stypy_function_name = '__ssa_join_with_else_function_context'
    __ssa_join_with_else_function_context.stypy_param_names_list = ['function_context_previous', 'function_context_if', 'function_context_else']
    __ssa_join_with_else_function_context.stypy_varargs_param_name = None
    __ssa_join_with_else_function_context.stypy_kwargs_param_name = None
    __ssa_join_with_else_function_context.stypy_call_defaults = defaults
    __ssa_join_with_else_function_context.stypy_call_varargs = varargs
    __ssa_join_with_else_function_context.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__ssa_join_with_else_function_context', ['function_context_previous', 'function_context_if', 'function_context_else'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__ssa_join_with_else_function_context', localization, ['function_context_previous', 'function_context_if', 'function_context_else'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__ssa_join_with_else_function_context(...)' code ##################

    str_20515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, (-1)), 'str', '\n    Helper function of the SSA implementation of an if-else structure, used with each function context in the type\n    store\n    :param function_context_previous: Function context\n    :param function_context_if: Function context\n    :param function_context_else: Function context\n    :return: A dictionary with names of variables and its joined types\n    ')
    
    # Assigning a Dict to a Name (line 64):
    
    # Obtaining an instance of the builtin type 'dict' (line 64)
    dict_20516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 64)
    
    # Assigning a type to the variable 'type_dict' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'type_dict', dict_20516)
    
    # Type idiom detected: calculating its left and rigth part (line 66)
    # Getting the type of 'function_context_else' (line 66)
    function_context_else_20517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'function_context_else')
    # Getting the type of 'None' (line 66)
    None_20518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'None')
    
    (may_be_20519, more_types_in_union_20520) = may_be_none(function_context_else_20517, None_20518)

    if may_be_20519:

        if more_types_in_union_20520:
            # Runtime conditional SSA (line 66)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Name (line 67):
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_20521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        
        # Assigning a type to the variable 'function_context_else' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'function_context_else', list_20521)

        if more_types_in_union_20520:
            # SSA join for if statement (line 66)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'function_context_if' (line 69)
    function_context_if_20522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'function_context_if')
    # Assigning a type to the variable 'function_context_if_20522' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'function_context_if_20522', function_context_if_20522)
    # Testing if the for loop is going to be iterated (line 69)
    # Testing the type of a for loop iterable (line 69)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 69, 4), function_context_if_20522)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 69, 4), function_context_if_20522):
        # Getting the type of the for loop variable (line 69)
        for_loop_var_20523 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 69, 4), function_context_if_20522)
        # Assigning a type to the variable 'var_name' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'var_name', for_loop_var_20523)
        # SSA begins for a for statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 70)
        var_name_20524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'var_name')
        # Getting the type of 'function_context_else' (line 70)
        function_context_else_20525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'function_context_else')
        # Applying the binary operator 'in' (line 70)
        result_contains_20526 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 11), 'in', var_name_20524, function_context_else_20525)
        
        # Testing if the type of an if condition is none (line 70)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 8), result_contains_20526):
            
            # Getting the type of 'var_name' (line 76)
            var_name_20543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 76)
            function_context_previous_20544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 76)
            result_contains_20545 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 15), 'in', var_name_20543, function_context_previous_20544)
            
            # Testing if the type of an if condition is none (line 76)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 12), result_contains_20545):
                
                # Assigning a Call to a Subscript (line 82):
                
                # Call to add(...): (line 82)
                # Processing the call arguments (line 82)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 82)
                var_name_20565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 88), 'var_name', False)
                # Getting the type of 'function_context_if' (line 82)
                function_context_if_20566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 68), 'function_context_if', False)
                # Obtaining the member '__getitem__' of a type (line 82)
                getitem___20567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 68), function_context_if_20566, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 82)
                subscript_call_result_20568 = invoke(stypy.reporting.localization.Localization(__file__, 82, 68), getitem___20567, var_name_20565)
                
                
                # Call to UndefinedType(...): (line 82)
                # Processing the call keyword arguments (line 82)
                kwargs_20570 = {}
                # Getting the type of 'UndefinedType' (line 82)
                UndefinedType_20569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 99), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 82)
                UndefinedType_call_result_20571 = invoke(stypy.reporting.localization.Localization(__file__, 82, 99), UndefinedType_20569, *[], **kwargs_20570)
                
                # Processing the call keyword arguments (line 82)
                kwargs_20572 = {}
                # Getting the type of 'union_type_copy' (line 82)
                union_type_copy_20562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 82)
                UnionType_20563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), union_type_copy_20562, 'UnionType')
                # Obtaining the member 'add' of a type (line 82)
                add_20564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), UnionType_20563, 'add')
                # Calling add(args, kwargs) (line 82)
                add_call_result_20573 = invoke(stypy.reporting.localization.Localization(__file__, 82, 38), add_20564, *[subscript_call_result_20568, UndefinedType_call_result_20571], **kwargs_20572)
                
                # Getting the type of 'type_dict' (line 82)
                type_dict_20574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'type_dict')
                # Getting the type of 'var_name' (line 82)
                var_name_20575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'var_name')
                # Storing an element on a container (line 82)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 16), type_dict_20574, (var_name_20575, add_call_result_20573))
            else:
                
                # Testing the type of an if condition (line 76)
                if_condition_20546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 12), result_contains_20545)
                # Assigning a type to the variable 'if_condition_20546' (line 76)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'if_condition_20546', if_condition_20546)
                # SSA begins for if statement (line 76)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 78):
                
                # Call to add(...): (line 78)
                # Processing the call arguments (line 78)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 78)
                var_name_20550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 78)
                function_context_previous_20551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 78)
                getitem___20552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 68), function_context_previous_20551, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                subscript_call_result_20553 = invoke(stypy.reporting.localization.Localization(__file__, 78, 68), getitem___20552, var_name_20550)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 79)
                var_name_20554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 83), 'var_name', False)
                # Getting the type of 'function_context_if' (line 79)
                function_context_if_20555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 63), 'function_context_if', False)
                # Obtaining the member '__getitem__' of a type (line 79)
                getitem___20556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 63), function_context_if_20555, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 79)
                subscript_call_result_20557 = invoke(stypy.reporting.localization.Localization(__file__, 79, 63), getitem___20556, var_name_20554)
                
                # Processing the call keyword arguments (line 78)
                kwargs_20558 = {}
                # Getting the type of 'union_type_copy' (line 78)
                union_type_copy_20547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 78)
                UnionType_20548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 38), union_type_copy_20547, 'UnionType')
                # Obtaining the member 'add' of a type (line 78)
                add_20549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 38), UnionType_20548, 'add')
                # Calling add(args, kwargs) (line 78)
                add_call_result_20559 = invoke(stypy.reporting.localization.Localization(__file__, 78, 38), add_20549, *[subscript_call_result_20553, subscript_call_result_20557], **kwargs_20558)
                
                # Getting the type of 'type_dict' (line 78)
                type_dict_20560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'type_dict')
                # Getting the type of 'var_name' (line 78)
                var_name_20561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'var_name')
                # Storing an element on a container (line 78)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 16), type_dict_20560, (var_name_20561, add_call_result_20559))
                # SSA branch for the else part of an if statement (line 76)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 82):
                
                # Call to add(...): (line 82)
                # Processing the call arguments (line 82)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 82)
                var_name_20565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 88), 'var_name', False)
                # Getting the type of 'function_context_if' (line 82)
                function_context_if_20566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 68), 'function_context_if', False)
                # Obtaining the member '__getitem__' of a type (line 82)
                getitem___20567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 68), function_context_if_20566, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 82)
                subscript_call_result_20568 = invoke(stypy.reporting.localization.Localization(__file__, 82, 68), getitem___20567, var_name_20565)
                
                
                # Call to UndefinedType(...): (line 82)
                # Processing the call keyword arguments (line 82)
                kwargs_20570 = {}
                # Getting the type of 'UndefinedType' (line 82)
                UndefinedType_20569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 99), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 82)
                UndefinedType_call_result_20571 = invoke(stypy.reporting.localization.Localization(__file__, 82, 99), UndefinedType_20569, *[], **kwargs_20570)
                
                # Processing the call keyword arguments (line 82)
                kwargs_20572 = {}
                # Getting the type of 'union_type_copy' (line 82)
                union_type_copy_20562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 82)
                UnionType_20563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), union_type_copy_20562, 'UnionType')
                # Obtaining the member 'add' of a type (line 82)
                add_20564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), UnionType_20563, 'add')
                # Calling add(args, kwargs) (line 82)
                add_call_result_20573 = invoke(stypy.reporting.localization.Localization(__file__, 82, 38), add_20564, *[subscript_call_result_20568, UndefinedType_call_result_20571], **kwargs_20572)
                
                # Getting the type of 'type_dict' (line 82)
                type_dict_20574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'type_dict')
                # Getting the type of 'var_name' (line 82)
                var_name_20575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'var_name')
                # Storing an element on a container (line 82)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 16), type_dict_20574, (var_name_20575, add_call_result_20573))
                # SSA join for if statement (line 76)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 70)
            if_condition_20527 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 8), result_contains_20526)
            # Assigning a type to the variable 'if_condition_20527' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'if_condition_20527', if_condition_20527)
            # SSA begins for if statement (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 72):
            
            # Call to add(...): (line 72)
            # Processing the call arguments (line 72)
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 72)
            var_name_20531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 84), 'var_name', False)
            # Getting the type of 'function_context_if' (line 72)
            function_context_if_20532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 64), 'function_context_if', False)
            # Obtaining the member '__getitem__' of a type (line 72)
            getitem___20533 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 64), function_context_if_20532, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 72)
            subscript_call_result_20534 = invoke(stypy.reporting.localization.Localization(__file__, 72, 64), getitem___20533, var_name_20531)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 73)
            var_name_20535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 81), 'var_name', False)
            # Getting the type of 'function_context_else' (line 73)
            function_context_else_20536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 59), 'function_context_else', False)
            # Obtaining the member '__getitem__' of a type (line 73)
            getitem___20537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 59), function_context_else_20536, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 73)
            subscript_call_result_20538 = invoke(stypy.reporting.localization.Localization(__file__, 73, 59), getitem___20537, var_name_20535)
            
            # Processing the call keyword arguments (line 72)
            kwargs_20539 = {}
            # Getting the type of 'union_type_copy' (line 72)
            union_type_copy_20528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 34), 'union_type_copy', False)
            # Obtaining the member 'UnionType' of a type (line 72)
            UnionType_20529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 34), union_type_copy_20528, 'UnionType')
            # Obtaining the member 'add' of a type (line 72)
            add_20530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 34), UnionType_20529, 'add')
            # Calling add(args, kwargs) (line 72)
            add_call_result_20540 = invoke(stypy.reporting.localization.Localization(__file__, 72, 34), add_20530, *[subscript_call_result_20534, subscript_call_result_20538], **kwargs_20539)
            
            # Getting the type of 'type_dict' (line 72)
            type_dict_20541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'type_dict')
            # Getting the type of 'var_name' (line 72)
            var_name_20542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'var_name')
            # Storing an element on a container (line 72)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 12), type_dict_20541, (var_name_20542, add_call_result_20540))
            # SSA branch for the else part of an if statement (line 70)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'var_name' (line 76)
            var_name_20543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 76)
            function_context_previous_20544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 76)
            result_contains_20545 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 15), 'in', var_name_20543, function_context_previous_20544)
            
            # Testing if the type of an if condition is none (line 76)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 12), result_contains_20545):
                
                # Assigning a Call to a Subscript (line 82):
                
                # Call to add(...): (line 82)
                # Processing the call arguments (line 82)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 82)
                var_name_20565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 88), 'var_name', False)
                # Getting the type of 'function_context_if' (line 82)
                function_context_if_20566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 68), 'function_context_if', False)
                # Obtaining the member '__getitem__' of a type (line 82)
                getitem___20567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 68), function_context_if_20566, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 82)
                subscript_call_result_20568 = invoke(stypy.reporting.localization.Localization(__file__, 82, 68), getitem___20567, var_name_20565)
                
                
                # Call to UndefinedType(...): (line 82)
                # Processing the call keyword arguments (line 82)
                kwargs_20570 = {}
                # Getting the type of 'UndefinedType' (line 82)
                UndefinedType_20569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 99), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 82)
                UndefinedType_call_result_20571 = invoke(stypy.reporting.localization.Localization(__file__, 82, 99), UndefinedType_20569, *[], **kwargs_20570)
                
                # Processing the call keyword arguments (line 82)
                kwargs_20572 = {}
                # Getting the type of 'union_type_copy' (line 82)
                union_type_copy_20562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 82)
                UnionType_20563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), union_type_copy_20562, 'UnionType')
                # Obtaining the member 'add' of a type (line 82)
                add_20564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), UnionType_20563, 'add')
                # Calling add(args, kwargs) (line 82)
                add_call_result_20573 = invoke(stypy.reporting.localization.Localization(__file__, 82, 38), add_20564, *[subscript_call_result_20568, UndefinedType_call_result_20571], **kwargs_20572)
                
                # Getting the type of 'type_dict' (line 82)
                type_dict_20574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'type_dict')
                # Getting the type of 'var_name' (line 82)
                var_name_20575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'var_name')
                # Storing an element on a container (line 82)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 16), type_dict_20574, (var_name_20575, add_call_result_20573))
            else:
                
                # Testing the type of an if condition (line 76)
                if_condition_20546 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 12), result_contains_20545)
                # Assigning a type to the variable 'if_condition_20546' (line 76)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'if_condition_20546', if_condition_20546)
                # SSA begins for if statement (line 76)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 78):
                
                # Call to add(...): (line 78)
                # Processing the call arguments (line 78)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 78)
                var_name_20550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 78)
                function_context_previous_20551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 78)
                getitem___20552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 68), function_context_previous_20551, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                subscript_call_result_20553 = invoke(stypy.reporting.localization.Localization(__file__, 78, 68), getitem___20552, var_name_20550)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 79)
                var_name_20554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 83), 'var_name', False)
                # Getting the type of 'function_context_if' (line 79)
                function_context_if_20555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 63), 'function_context_if', False)
                # Obtaining the member '__getitem__' of a type (line 79)
                getitem___20556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 63), function_context_if_20555, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 79)
                subscript_call_result_20557 = invoke(stypy.reporting.localization.Localization(__file__, 79, 63), getitem___20556, var_name_20554)
                
                # Processing the call keyword arguments (line 78)
                kwargs_20558 = {}
                # Getting the type of 'union_type_copy' (line 78)
                union_type_copy_20547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 78)
                UnionType_20548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 38), union_type_copy_20547, 'UnionType')
                # Obtaining the member 'add' of a type (line 78)
                add_20549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 38), UnionType_20548, 'add')
                # Calling add(args, kwargs) (line 78)
                add_call_result_20559 = invoke(stypy.reporting.localization.Localization(__file__, 78, 38), add_20549, *[subscript_call_result_20553, subscript_call_result_20557], **kwargs_20558)
                
                # Getting the type of 'type_dict' (line 78)
                type_dict_20560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'type_dict')
                # Getting the type of 'var_name' (line 78)
                var_name_20561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'var_name')
                # Storing an element on a container (line 78)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 16), type_dict_20560, (var_name_20561, add_call_result_20559))
                # SSA branch for the else part of an if statement (line 76)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 82):
                
                # Call to add(...): (line 82)
                # Processing the call arguments (line 82)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 82)
                var_name_20565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 88), 'var_name', False)
                # Getting the type of 'function_context_if' (line 82)
                function_context_if_20566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 68), 'function_context_if', False)
                # Obtaining the member '__getitem__' of a type (line 82)
                getitem___20567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 68), function_context_if_20566, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 82)
                subscript_call_result_20568 = invoke(stypy.reporting.localization.Localization(__file__, 82, 68), getitem___20567, var_name_20565)
                
                
                # Call to UndefinedType(...): (line 82)
                # Processing the call keyword arguments (line 82)
                kwargs_20570 = {}
                # Getting the type of 'UndefinedType' (line 82)
                UndefinedType_20569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 99), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 82)
                UndefinedType_call_result_20571 = invoke(stypy.reporting.localization.Localization(__file__, 82, 99), UndefinedType_20569, *[], **kwargs_20570)
                
                # Processing the call keyword arguments (line 82)
                kwargs_20572 = {}
                # Getting the type of 'union_type_copy' (line 82)
                union_type_copy_20562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 82)
                UnionType_20563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), union_type_copy_20562, 'UnionType')
                # Obtaining the member 'add' of a type (line 82)
                add_20564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), UnionType_20563, 'add')
                # Calling add(args, kwargs) (line 82)
                add_call_result_20573 = invoke(stypy.reporting.localization.Localization(__file__, 82, 38), add_20564, *[subscript_call_result_20568, UndefinedType_call_result_20571], **kwargs_20572)
                
                # Getting the type of 'type_dict' (line 82)
                type_dict_20574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'type_dict')
                # Getting the type of 'var_name' (line 82)
                var_name_20575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'var_name')
                # Storing an element on a container (line 82)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 16), type_dict_20574, (var_name_20575, add_call_result_20573))
                # SSA join for if statement (line 76)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 70)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'function_context_else' (line 84)
    function_context_else_20576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'function_context_else')
    # Assigning a type to the variable 'function_context_else_20576' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'function_context_else_20576', function_context_else_20576)
    # Testing if the for loop is going to be iterated (line 84)
    # Testing the type of a for loop iterable (line 84)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 4), function_context_else_20576)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 4), function_context_else_20576):
        # Getting the type of the for loop variable (line 84)
        for_loop_var_20577 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 4), function_context_else_20576)
        # Assigning a type to the variable 'var_name' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'var_name', for_loop_var_20577)
        # SSA begins for a for statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 85)
        var_name_20578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'var_name')
        # Getting the type of 'function_context_if' (line 85)
        function_context_if_20579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'function_context_if')
        # Applying the binary operator 'in' (line 85)
        result_contains_20580 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 11), 'in', var_name_20578, function_context_if_20579)
        
        # Testing if the type of an if condition is none (line 85)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 8), result_contains_20580):
            
            # Getting the type of 'var_name' (line 89)
            var_name_20582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 89)
            function_context_previous_20583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 89)
            result_contains_20584 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 15), 'in', var_name_20582, function_context_previous_20583)
            
            # Testing if the type of an if condition is none (line 89)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 12), result_contains_20584):
                
                # Assigning a Call to a Subscript (line 96):
                
                # Call to add(...): (line 96)
                # Processing the call arguments (line 96)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 96)
                var_name_20604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 90), 'var_name', False)
                # Getting the type of 'function_context_else' (line 96)
                function_context_else_20605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 68), 'function_context_else', False)
                # Obtaining the member '__getitem__' of a type (line 96)
                getitem___20606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 68), function_context_else_20605, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 96)
                subscript_call_result_20607 = invoke(stypy.reporting.localization.Localization(__file__, 96, 68), getitem___20606, var_name_20604)
                
                
                # Call to UndefinedType(...): (line 96)
                # Processing the call keyword arguments (line 96)
                kwargs_20609 = {}
                # Getting the type of 'UndefinedType' (line 96)
                UndefinedType_20608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 101), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 96)
                UndefinedType_call_result_20610 = invoke(stypy.reporting.localization.Localization(__file__, 96, 101), UndefinedType_20608, *[], **kwargs_20609)
                
                # Processing the call keyword arguments (line 96)
                kwargs_20611 = {}
                # Getting the type of 'union_type_copy' (line 96)
                union_type_copy_20601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 96)
                UnionType_20602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), union_type_copy_20601, 'UnionType')
                # Obtaining the member 'add' of a type (line 96)
                add_20603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), UnionType_20602, 'add')
                # Calling add(args, kwargs) (line 96)
                add_call_result_20612 = invoke(stypy.reporting.localization.Localization(__file__, 96, 38), add_20603, *[subscript_call_result_20607, UndefinedType_call_result_20610], **kwargs_20611)
                
                # Getting the type of 'type_dict' (line 96)
                type_dict_20613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'type_dict')
                # Getting the type of 'var_name' (line 96)
                var_name_20614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'var_name')
                # Storing an element on a container (line 96)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 16), type_dict_20613, (var_name_20614, add_call_result_20612))
            else:
                
                # Testing the type of an if condition (line 89)
                if_condition_20585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 12), result_contains_20584)
                # Assigning a type to the variable 'if_condition_20585' (line 89)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'if_condition_20585', if_condition_20585)
                # SSA begins for if statement (line 89)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 91):
                
                # Call to add(...): (line 91)
                # Processing the call arguments (line 91)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 91)
                var_name_20589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 91)
                function_context_previous_20590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 91)
                getitem___20591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 68), function_context_previous_20590, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 91)
                subscript_call_result_20592 = invoke(stypy.reporting.localization.Localization(__file__, 91, 68), getitem___20591, var_name_20589)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 92)
                var_name_20593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 85), 'var_name', False)
                # Getting the type of 'function_context_else' (line 92)
                function_context_else_20594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 63), 'function_context_else', False)
                # Obtaining the member '__getitem__' of a type (line 92)
                getitem___20595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 63), function_context_else_20594, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 92)
                subscript_call_result_20596 = invoke(stypy.reporting.localization.Localization(__file__, 92, 63), getitem___20595, var_name_20593)
                
                # Processing the call keyword arguments (line 91)
                kwargs_20597 = {}
                # Getting the type of 'union_type_copy' (line 91)
                union_type_copy_20586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 91)
                UnionType_20587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 38), union_type_copy_20586, 'UnionType')
                # Obtaining the member 'add' of a type (line 91)
                add_20588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 38), UnionType_20587, 'add')
                # Calling add(args, kwargs) (line 91)
                add_call_result_20598 = invoke(stypy.reporting.localization.Localization(__file__, 91, 38), add_20588, *[subscript_call_result_20592, subscript_call_result_20596], **kwargs_20597)
                
                # Getting the type of 'type_dict' (line 91)
                type_dict_20599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'type_dict')
                # Getting the type of 'var_name' (line 91)
                var_name_20600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'var_name')
                # Storing an element on a container (line 91)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 16), type_dict_20599, (var_name_20600, add_call_result_20598))
                # SSA branch for the else part of an if statement (line 89)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 96):
                
                # Call to add(...): (line 96)
                # Processing the call arguments (line 96)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 96)
                var_name_20604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 90), 'var_name', False)
                # Getting the type of 'function_context_else' (line 96)
                function_context_else_20605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 68), 'function_context_else', False)
                # Obtaining the member '__getitem__' of a type (line 96)
                getitem___20606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 68), function_context_else_20605, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 96)
                subscript_call_result_20607 = invoke(stypy.reporting.localization.Localization(__file__, 96, 68), getitem___20606, var_name_20604)
                
                
                # Call to UndefinedType(...): (line 96)
                # Processing the call keyword arguments (line 96)
                kwargs_20609 = {}
                # Getting the type of 'UndefinedType' (line 96)
                UndefinedType_20608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 101), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 96)
                UndefinedType_call_result_20610 = invoke(stypy.reporting.localization.Localization(__file__, 96, 101), UndefinedType_20608, *[], **kwargs_20609)
                
                # Processing the call keyword arguments (line 96)
                kwargs_20611 = {}
                # Getting the type of 'union_type_copy' (line 96)
                union_type_copy_20601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 96)
                UnionType_20602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), union_type_copy_20601, 'UnionType')
                # Obtaining the member 'add' of a type (line 96)
                add_20603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), UnionType_20602, 'add')
                # Calling add(args, kwargs) (line 96)
                add_call_result_20612 = invoke(stypy.reporting.localization.Localization(__file__, 96, 38), add_20603, *[subscript_call_result_20607, UndefinedType_call_result_20610], **kwargs_20611)
                
                # Getting the type of 'type_dict' (line 96)
                type_dict_20613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'type_dict')
                # Getting the type of 'var_name' (line 96)
                var_name_20614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'var_name')
                # Storing an element on a container (line 96)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 16), type_dict_20613, (var_name_20614, add_call_result_20612))
                # SSA join for if statement (line 89)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 85)
            if_condition_20581 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 8), result_contains_20580)
            # Assigning a type to the variable 'if_condition_20581' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'if_condition_20581', if_condition_20581)
            # SSA begins for if statement (line 85)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA branch for the else part of an if statement (line 85)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'var_name' (line 89)
            var_name_20582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 89)
            function_context_previous_20583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 89)
            result_contains_20584 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 15), 'in', var_name_20582, function_context_previous_20583)
            
            # Testing if the type of an if condition is none (line 89)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 12), result_contains_20584):
                
                # Assigning a Call to a Subscript (line 96):
                
                # Call to add(...): (line 96)
                # Processing the call arguments (line 96)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 96)
                var_name_20604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 90), 'var_name', False)
                # Getting the type of 'function_context_else' (line 96)
                function_context_else_20605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 68), 'function_context_else', False)
                # Obtaining the member '__getitem__' of a type (line 96)
                getitem___20606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 68), function_context_else_20605, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 96)
                subscript_call_result_20607 = invoke(stypy.reporting.localization.Localization(__file__, 96, 68), getitem___20606, var_name_20604)
                
                
                # Call to UndefinedType(...): (line 96)
                # Processing the call keyword arguments (line 96)
                kwargs_20609 = {}
                # Getting the type of 'UndefinedType' (line 96)
                UndefinedType_20608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 101), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 96)
                UndefinedType_call_result_20610 = invoke(stypy.reporting.localization.Localization(__file__, 96, 101), UndefinedType_20608, *[], **kwargs_20609)
                
                # Processing the call keyword arguments (line 96)
                kwargs_20611 = {}
                # Getting the type of 'union_type_copy' (line 96)
                union_type_copy_20601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 96)
                UnionType_20602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), union_type_copy_20601, 'UnionType')
                # Obtaining the member 'add' of a type (line 96)
                add_20603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), UnionType_20602, 'add')
                # Calling add(args, kwargs) (line 96)
                add_call_result_20612 = invoke(stypy.reporting.localization.Localization(__file__, 96, 38), add_20603, *[subscript_call_result_20607, UndefinedType_call_result_20610], **kwargs_20611)
                
                # Getting the type of 'type_dict' (line 96)
                type_dict_20613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'type_dict')
                # Getting the type of 'var_name' (line 96)
                var_name_20614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'var_name')
                # Storing an element on a container (line 96)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 16), type_dict_20613, (var_name_20614, add_call_result_20612))
            else:
                
                # Testing the type of an if condition (line 89)
                if_condition_20585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 12), result_contains_20584)
                # Assigning a type to the variable 'if_condition_20585' (line 89)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'if_condition_20585', if_condition_20585)
                # SSA begins for if statement (line 89)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 91):
                
                # Call to add(...): (line 91)
                # Processing the call arguments (line 91)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 91)
                var_name_20589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 91)
                function_context_previous_20590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 91)
                getitem___20591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 68), function_context_previous_20590, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 91)
                subscript_call_result_20592 = invoke(stypy.reporting.localization.Localization(__file__, 91, 68), getitem___20591, var_name_20589)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 92)
                var_name_20593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 85), 'var_name', False)
                # Getting the type of 'function_context_else' (line 92)
                function_context_else_20594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 63), 'function_context_else', False)
                # Obtaining the member '__getitem__' of a type (line 92)
                getitem___20595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 63), function_context_else_20594, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 92)
                subscript_call_result_20596 = invoke(stypy.reporting.localization.Localization(__file__, 92, 63), getitem___20595, var_name_20593)
                
                # Processing the call keyword arguments (line 91)
                kwargs_20597 = {}
                # Getting the type of 'union_type_copy' (line 91)
                union_type_copy_20586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 91)
                UnionType_20587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 38), union_type_copy_20586, 'UnionType')
                # Obtaining the member 'add' of a type (line 91)
                add_20588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 38), UnionType_20587, 'add')
                # Calling add(args, kwargs) (line 91)
                add_call_result_20598 = invoke(stypy.reporting.localization.Localization(__file__, 91, 38), add_20588, *[subscript_call_result_20592, subscript_call_result_20596], **kwargs_20597)
                
                # Getting the type of 'type_dict' (line 91)
                type_dict_20599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'type_dict')
                # Getting the type of 'var_name' (line 91)
                var_name_20600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'var_name')
                # Storing an element on a container (line 91)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 16), type_dict_20599, (var_name_20600, add_call_result_20598))
                # SSA branch for the else part of an if statement (line 89)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 96):
                
                # Call to add(...): (line 96)
                # Processing the call arguments (line 96)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 96)
                var_name_20604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 90), 'var_name', False)
                # Getting the type of 'function_context_else' (line 96)
                function_context_else_20605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 68), 'function_context_else', False)
                # Obtaining the member '__getitem__' of a type (line 96)
                getitem___20606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 68), function_context_else_20605, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 96)
                subscript_call_result_20607 = invoke(stypy.reporting.localization.Localization(__file__, 96, 68), getitem___20606, var_name_20604)
                
                
                # Call to UndefinedType(...): (line 96)
                # Processing the call keyword arguments (line 96)
                kwargs_20609 = {}
                # Getting the type of 'UndefinedType' (line 96)
                UndefinedType_20608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 101), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 96)
                UndefinedType_call_result_20610 = invoke(stypy.reporting.localization.Localization(__file__, 96, 101), UndefinedType_20608, *[], **kwargs_20609)
                
                # Processing the call keyword arguments (line 96)
                kwargs_20611 = {}
                # Getting the type of 'union_type_copy' (line 96)
                union_type_copy_20601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 96)
                UnionType_20602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), union_type_copy_20601, 'UnionType')
                # Obtaining the member 'add' of a type (line 96)
                add_20603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), UnionType_20602, 'add')
                # Calling add(args, kwargs) (line 96)
                add_call_result_20612 = invoke(stypy.reporting.localization.Localization(__file__, 96, 38), add_20603, *[subscript_call_result_20607, UndefinedType_call_result_20610], **kwargs_20611)
                
                # Getting the type of 'type_dict' (line 96)
                type_dict_20613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'type_dict')
                # Getting the type of 'var_name' (line 96)
                var_name_20614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'var_name')
                # Storing an element on a container (line 96)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 16), type_dict_20613, (var_name_20614, add_call_result_20612))
                # SSA join for if statement (line 89)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 85)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'type_dict' (line 99)
    type_dict_20615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'type_dict')
    # Assigning a type to the variable 'stypy_return_type' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type', type_dict_20615)
    
    # ################# End of '__ssa_join_with_else_function_context(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__ssa_join_with_else_function_context' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_20616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20616)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__ssa_join_with_else_function_context'
    return stypy_return_type_20616

# Assigning a type to the variable '__ssa_join_with_else_function_context' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), '__ssa_join_with_else_function_context', __ssa_join_with_else_function_context)

@norecursion
def ssa_join_with_else_branch(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ssa_join_with_else_branch'
    module_type_store = module_type_store.open_function_context('ssa_join_with_else_branch', 102, 0, False)
    
    # Passed parameters checking function
    ssa_join_with_else_branch.stypy_localization = localization
    ssa_join_with_else_branch.stypy_type_of_self = None
    ssa_join_with_else_branch.stypy_type_store = module_type_store
    ssa_join_with_else_branch.stypy_function_name = 'ssa_join_with_else_branch'
    ssa_join_with_else_branch.stypy_param_names_list = ['type_store_previous', 'type_store_if', 'type_store_else']
    ssa_join_with_else_branch.stypy_varargs_param_name = None
    ssa_join_with_else_branch.stypy_kwargs_param_name = None
    ssa_join_with_else_branch.stypy_call_defaults = defaults
    ssa_join_with_else_branch.stypy_call_varargs = varargs
    ssa_join_with_else_branch.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ssa_join_with_else_branch', ['type_store_previous', 'type_store_if', 'type_store_else'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ssa_join_with_else_branch', localization, ['type_store_previous', 'type_store_if', 'type_store_else'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ssa_join_with_else_branch(...)' code ##################

    str_20617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', '\n    Implements the SSA algorithm with the type stores of an if-else structure\n    :param type_store_previous: Type store\n    :param type_store_if: Function context\n    :param type_store_else:\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 111):
    
    # Call to TypeStore(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'type_store_previous' (line 111)
    type_store_previous_20619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'type_store_previous', False)
    # Obtaining the member 'program_name' of a type (line 111)
    program_name_20620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 34), type_store_previous_20619, 'program_name')
    # Processing the call keyword arguments (line 111)
    kwargs_20621 = {}
    # Getting the type of 'TypeStore' (line 111)
    TypeStore_20618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'TypeStore', False)
    # Calling TypeStore(args, kwargs) (line 111)
    TypeStore_call_result_20622 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), TypeStore_20618, *[program_name_20620], **kwargs_20621)
    
    # Assigning a type to the variable 'joined_type_store' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'joined_type_store', TypeStore_call_result_20622)
    
    # Assigning a Attribute to a Attribute (line 112):
    # Getting the type of 'type_store_previous' (line 112)
    type_store_previous_20623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 47), 'type_store_previous')
    # Obtaining the member 'last_function_contexts' of a type (line 112)
    last_function_contexts_20624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 47), type_store_previous_20623, 'last_function_contexts')
    # Getting the type of 'joined_type_store' (line 112)
    joined_type_store_20625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'joined_type_store')
    # Setting the type of the member 'last_function_contexts' of a type (line 112)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 4), joined_type_store_20625, 'last_function_contexts', last_function_contexts_20624)
    
    # Assigning a List to a Attribute (line 113):
    
    # Obtaining an instance of the builtin type 'list' (line 113)
    list_20626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 113)
    
    # Getting the type of 'joined_type_store' (line 113)
    joined_type_store_20627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'joined_type_store')
    # Setting the type of the member 'context_stack' of a type (line 113)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 4), joined_type_store_20627, 'context_stack', list_20626)
    
    
    # Call to range(...): (line 114)
    # Processing the call arguments (line 114)
    
    # Call to len(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'type_store_previous' (line 114)
    type_store_previous_20630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'type_store_previous', False)
    # Obtaining the member 'context_stack' of a type (line 114)
    context_stack_20631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 23), type_store_previous_20630, 'context_stack')
    # Processing the call keyword arguments (line 114)
    kwargs_20632 = {}
    # Getting the type of 'len' (line 114)
    len_20629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 19), 'len', False)
    # Calling len(args, kwargs) (line 114)
    len_call_result_20633 = invoke(stypy.reporting.localization.Localization(__file__, 114, 19), len_20629, *[context_stack_20631], **kwargs_20632)
    
    # Processing the call keyword arguments (line 114)
    kwargs_20634 = {}
    # Getting the type of 'range' (line 114)
    range_20628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'range', False)
    # Calling range(args, kwargs) (line 114)
    range_call_result_20635 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), range_20628, *[len_call_result_20633], **kwargs_20634)
    
    # Assigning a type to the variable 'range_call_result_20635' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'range_call_result_20635', range_call_result_20635)
    # Testing if the for loop is going to be iterated (line 114)
    # Testing the type of a for loop iterable (line 114)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 114, 4), range_call_result_20635)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 114, 4), range_call_result_20635):
        # Getting the type of the for loop variable (line 114)
        for_loop_var_20636 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 114, 4), range_call_result_20635)
        # Assigning a type to the variable 'i' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'i', for_loop_var_20636)
        # SSA begins for a for statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 116)
        # Getting the type of 'type_store_else' (line 116)
        type_store_else_20637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'type_store_else')
        # Getting the type of 'None' (line 116)
        None_20638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 30), 'None')
        
        (may_be_20639, more_types_in_union_20640) = may_be_none(type_store_else_20637, None_20638)

        if may_be_20639:

            if more_types_in_union_20640:
                # Runtime conditional SSA (line 116)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 117):
            # Getting the type of 'None' (line 117)
            None_20641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 36), 'None')
            # Assigning a type to the variable 'function_context_else' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'function_context_else', None_20641)

            if more_types_in_union_20640:
                # Runtime conditional SSA for else branch (line 116)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_20639) or more_types_in_union_20640):
            
            # Assigning a Subscript to a Name (line 119):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 119)
            i_20642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 52), 'i')
            # Getting the type of 'type_store_else' (line 119)
            type_store_else_20643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 36), 'type_store_else')
            # Obtaining the member '__getitem__' of a type (line 119)
            getitem___20644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 36), type_store_else_20643, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 119)
            subscript_call_result_20645 = invoke(stypy.reporting.localization.Localization(__file__, 119, 36), getitem___20644, i_20642)
            
            # Assigning a type to the variable 'function_context_else' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'function_context_else', subscript_call_result_20645)

            if (may_be_20639 and more_types_in_union_20640):
                # SSA join for if statement (line 116)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 121):
        
        # Call to __ssa_join_with_else_function_context(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 121)
        i_20647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 88), 'i', False)
        # Getting the type of 'type_store_previous' (line 121)
        type_store_previous_20648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 68), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___20649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 68), type_store_previous_20648, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_20650 = invoke(stypy.reporting.localization.Localization(__file__, 121, 68), getitem___20649, i_20647)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 121)
        i_20651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 106), 'i', False)
        # Getting the type of 'type_store_if' (line 121)
        type_store_if_20652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 92), 'type_store_if', False)
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___20653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 92), type_store_if_20652, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_20654 = invoke(stypy.reporting.localization.Localization(__file__, 121, 92), getitem___20653, i_20651)
        
        # Getting the type of 'function_context_else' (line 122)
        function_context_else_20655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 68), 'function_context_else', False)
        # Processing the call keyword arguments (line 121)
        kwargs_20656 = {}
        # Getting the type of '__ssa_join_with_else_function_context' (line 121)
        ssa_join_with_else_function_context_20646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 30), '__ssa_join_with_else_function_context', False)
        # Calling __ssa_join_with_else_function_context(args, kwargs) (line 121)
        ssa_join_with_else_function_context_call_result_20657 = invoke(stypy.reporting.localization.Localization(__file__, 121, 30), ssa_join_with_else_function_context_20646, *[subscript_call_result_20650, subscript_call_result_20654, function_context_else_20655], **kwargs_20656)
        
        # Assigning a type to the variable 'joined_context_dict' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'joined_context_dict', ssa_join_with_else_function_context_call_result_20657)
        
        # Assigning a Call to a Name (line 125):
        
        # Call to copy(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_20663 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 125)
        i_20658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 47), 'i', False)
        # Getting the type of 'type_store_previous' (line 125)
        type_store_previous_20659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___20660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 27), type_store_previous_20659, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_20661 = invoke(stypy.reporting.localization.Localization(__file__, 125, 27), getitem___20660, i_20658)
        
        # Obtaining the member 'copy' of a type (line 125)
        copy_20662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 27), subscript_call_result_20661, 'copy')
        # Calling copy(args, kwargs) (line 125)
        copy_call_result_20664 = invoke(stypy.reporting.localization.Localization(__file__, 125, 27), copy_20662, *[], **kwargs_20663)
        
        # Assigning a type to the variable 'joined_f_context' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'joined_f_context', copy_call_result_20664)
        
        # Assigning a Name to a Attribute (line 126):
        # Getting the type of 'joined_context_dict' (line 126)
        joined_context_dict_20665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'joined_context_dict')
        # Getting the type of 'joined_f_context' (line 126)
        joined_f_context_20666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'joined_f_context')
        # Setting the type of the member 'types_of' of a type (line 126)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), joined_f_context_20666, 'types_of', joined_context_dict_20665)
        
        # Assigning a Call to a Attribute (line 127):
        
        # Call to __join_globals(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 127)
        i_20668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 68), 'i', False)
        # Getting the type of 'type_store_if' (line 127)
        type_store_if_20669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 54), 'type_store_if', False)
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___20670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 54), type_store_if_20669, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_20671 = invoke(stypy.reporting.localization.Localization(__file__, 127, 54), getitem___20670, i_20668)
        
        # Getting the type of 'function_context_else' (line 127)
        function_context_else_20672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 72), 'function_context_else', False)
        # Processing the call keyword arguments (line 127)
        kwargs_20673 = {}
        # Getting the type of '__join_globals' (line 127)
        join_globals_20667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 39), '__join_globals', False)
        # Calling __join_globals(args, kwargs) (line 127)
        join_globals_call_result_20674 = invoke(stypy.reporting.localization.Localization(__file__, 127, 39), join_globals_20667, *[subscript_call_result_20671, function_context_else_20672], **kwargs_20673)
        
        # Getting the type of 'joined_f_context' (line 127)
        joined_f_context_20675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'joined_f_context')
        # Setting the type of the member 'global_vars' of a type (line 127)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), joined_f_context_20675, 'global_vars', join_globals_call_result_20674)
        
        # Assigning a Attribute to a Attribute (line 128):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 129)
        i_20676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'i')
        # Getting the type of 'type_store_if' (line 128)
        type_store_if_20677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 45), 'type_store_if')
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___20678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 45), type_store_if_20677, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_20679 = invoke(stypy.reporting.localization.Localization(__file__, 128, 45), getitem___20678, i_20676)
        
        # Obtaining the member 'annotation_record' of a type (line 128)
        annotation_record_20680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 45), subscript_call_result_20679, 'annotation_record')
        # Getting the type of 'joined_f_context' (line 128)
        joined_f_context_20681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'joined_f_context')
        # Setting the type of the member 'annotation_record' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), joined_f_context_20681, 'annotation_record', annotation_record_20680)
        
        # Call to append(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'joined_f_context' (line 131)
        joined_f_context_20685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 47), 'joined_f_context', False)
        # Processing the call keyword arguments (line 131)
        kwargs_20686 = {}
        # Getting the type of 'joined_type_store' (line 131)
        joined_type_store_20682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'joined_type_store', False)
        # Obtaining the member 'context_stack' of a type (line 131)
        context_stack_20683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), joined_type_store_20682, 'context_stack')
        # Obtaining the member 'append' of a type (line 131)
        append_20684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), context_stack_20683, 'append')
        # Calling append(args, kwargs) (line 131)
        append_call_result_20687 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), append_20684, *[joined_f_context_20685], **kwargs_20686)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'joined_type_store' (line 133)
    joined_type_store_20688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'joined_type_store')
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type', joined_type_store_20688)
    
    # ################# End of 'ssa_join_with_else_branch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ssa_join_with_else_branch' in the type store
    # Getting the type of 'stypy_return_type' (line 102)
    stypy_return_type_20689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20689)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ssa_join_with_else_branch'
    return stypy_return_type_20689

# Assigning a type to the variable 'ssa_join_with_else_branch' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'ssa_join_with_else_branch', ssa_join_with_else_branch)

@norecursion
def __join_except_branches_function_context(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__join_except_branches_function_context'
    module_type_store = module_type_store.open_function_context('__join_except_branches_function_context', 138, 0, False)
    
    # Passed parameters checking function
    __join_except_branches_function_context.stypy_localization = localization
    __join_except_branches_function_context.stypy_type_of_self = None
    __join_except_branches_function_context.stypy_type_store = module_type_store
    __join_except_branches_function_context.stypy_function_name = '__join_except_branches_function_context'
    __join_except_branches_function_context.stypy_param_names_list = ['function_context_previous', 'function_context_new']
    __join_except_branches_function_context.stypy_varargs_param_name = None
    __join_except_branches_function_context.stypy_kwargs_param_name = None
    __join_except_branches_function_context.stypy_call_defaults = defaults
    __join_except_branches_function_context.stypy_call_varargs = varargs
    __join_except_branches_function_context.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__join_except_branches_function_context', ['function_context_previous', 'function_context_new'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__join_except_branches_function_context', localization, ['function_context_previous', 'function_context_new'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__join_except_branches_function_context(...)' code ##################

    str_20690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, (-1)), 'str', '\n    Helper function to join variables of function contexts that belong to different except\n    blocks\n    :param function_context_previous: Function context\n    :param function_context_new: Function context\n    :return: A dictionary with names of variables and its joined types\n    ')
    
    # Assigning a Dict to a Name (line 146):
    
    # Obtaining an instance of the builtin type 'dict' (line 146)
    dict_20691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 146)
    
    # Assigning a type to the variable 'type_dict' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'type_dict', dict_20691)
    
    # Getting the type of 'function_context_previous' (line 148)
    function_context_previous_20692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'function_context_previous')
    # Assigning a type to the variable 'function_context_previous_20692' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'function_context_previous_20692', function_context_previous_20692)
    # Testing if the for loop is going to be iterated (line 148)
    # Testing the type of a for loop iterable (line 148)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 148, 4), function_context_previous_20692)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 148, 4), function_context_previous_20692):
        # Getting the type of the for loop variable (line 148)
        for_loop_var_20693 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 148, 4), function_context_previous_20692)
        # Assigning a type to the variable 'var_name' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'var_name', for_loop_var_20693)
        # SSA begins for a for statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 149)
        var_name_20694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'var_name')
        # Getting the type of 'function_context_new' (line 149)
        function_context_new_20695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), 'function_context_new')
        # Applying the binary operator 'in' (line 149)
        result_contains_20696 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), 'in', var_name_20694, function_context_new_20695)
        
        # Testing if the type of an if condition is none (line 149)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 149, 8), result_contains_20696):
            
            # Assigning a Subscript to a Subscript (line 155):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 155)
            var_name_20713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 60), 'var_name')
            # Getting the type of 'function_context_previous' (line 155)
            function_context_previous_20714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'function_context_previous')
            # Obtaining the member '__getitem__' of a type (line 155)
            getitem___20715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 34), function_context_previous_20714, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 155)
            subscript_call_result_20716 = invoke(stypy.reporting.localization.Localization(__file__, 155, 34), getitem___20715, var_name_20713)
            
            # Getting the type of 'type_dict' (line 155)
            type_dict_20717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'type_dict')
            # Getting the type of 'var_name' (line 155)
            var_name_20718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'var_name')
            # Storing an element on a container (line 155)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 12), type_dict_20717, (var_name_20718, subscript_call_result_20716))
        else:
            
            # Testing the type of an if condition (line 149)
            if_condition_20697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), result_contains_20696)
            # Assigning a type to the variable 'if_condition_20697' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_20697', if_condition_20697)
            # SSA begins for if statement (line 149)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 151):
            
            # Call to add(...): (line 151)
            # Processing the call arguments (line 151)
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 151)
            var_name_20701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 90), 'var_name', False)
            # Getting the type of 'function_context_previous' (line 151)
            function_context_previous_20702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 64), 'function_context_previous', False)
            # Obtaining the member '__getitem__' of a type (line 151)
            getitem___20703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 64), function_context_previous_20702, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 151)
            subscript_call_result_20704 = invoke(stypy.reporting.localization.Localization(__file__, 151, 64), getitem___20703, var_name_20701)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 152)
            var_name_20705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 80), 'var_name', False)
            # Getting the type of 'function_context_new' (line 152)
            function_context_new_20706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 59), 'function_context_new', False)
            # Obtaining the member '__getitem__' of a type (line 152)
            getitem___20707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 59), function_context_new_20706, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 152)
            subscript_call_result_20708 = invoke(stypy.reporting.localization.Localization(__file__, 152, 59), getitem___20707, var_name_20705)
            
            # Processing the call keyword arguments (line 151)
            kwargs_20709 = {}
            # Getting the type of 'union_type_copy' (line 151)
            union_type_copy_20698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'union_type_copy', False)
            # Obtaining the member 'UnionType' of a type (line 151)
            UnionType_20699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 34), union_type_copy_20698, 'UnionType')
            # Obtaining the member 'add' of a type (line 151)
            add_20700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 34), UnionType_20699, 'add')
            # Calling add(args, kwargs) (line 151)
            add_call_result_20710 = invoke(stypy.reporting.localization.Localization(__file__, 151, 34), add_20700, *[subscript_call_result_20704, subscript_call_result_20708], **kwargs_20709)
            
            # Getting the type of 'type_dict' (line 151)
            type_dict_20711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'type_dict')
            # Getting the type of 'var_name' (line 151)
            var_name_20712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 22), 'var_name')
            # Storing an element on a container (line 151)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 12), type_dict_20711, (var_name_20712, add_call_result_20710))
            # SSA branch for the else part of an if statement (line 149)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Subscript to a Subscript (line 155):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 155)
            var_name_20713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 60), 'var_name')
            # Getting the type of 'function_context_previous' (line 155)
            function_context_previous_20714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'function_context_previous')
            # Obtaining the member '__getitem__' of a type (line 155)
            getitem___20715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 34), function_context_previous_20714, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 155)
            subscript_call_result_20716 = invoke(stypy.reporting.localization.Localization(__file__, 155, 34), getitem___20715, var_name_20713)
            
            # Getting the type of 'type_dict' (line 155)
            type_dict_20717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'type_dict')
            # Getting the type of 'var_name' (line 155)
            var_name_20718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'var_name')
            # Storing an element on a container (line 155)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 12), type_dict_20717, (var_name_20718, subscript_call_result_20716))
            # SSA join for if statement (line 149)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'function_context_new' (line 157)
    function_context_new_20719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'function_context_new')
    # Assigning a type to the variable 'function_context_new_20719' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'function_context_new_20719', function_context_new_20719)
    # Testing if the for loop is going to be iterated (line 157)
    # Testing the type of a for loop iterable (line 157)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 4), function_context_new_20719)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 157, 4), function_context_new_20719):
        # Getting the type of the for loop variable (line 157)
        for_loop_var_20720 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 4), function_context_new_20719)
        # Assigning a type to the variable 'var_name' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'var_name', for_loop_var_20720)
        # SSA begins for a for statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 158)
        var_name_20721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'var_name')
        # Getting the type of 'function_context_previous' (line 158)
        function_context_previous_20722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 23), 'function_context_previous')
        # Applying the binary operator 'in' (line 158)
        result_contains_20723 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 11), 'in', var_name_20721, function_context_previous_20722)
        
        # Testing if the type of an if condition is none (line 158)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 158, 8), result_contains_20723):
            
            # Assigning a Subscript to a Subscript (line 164):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 164)
            var_name_20740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 55), 'var_name')
            # Getting the type of 'function_context_new' (line 164)
            function_context_new_20741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'function_context_new')
            # Obtaining the member '__getitem__' of a type (line 164)
            getitem___20742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 34), function_context_new_20741, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 164)
            subscript_call_result_20743 = invoke(stypy.reporting.localization.Localization(__file__, 164, 34), getitem___20742, var_name_20740)
            
            # Getting the type of 'type_dict' (line 164)
            type_dict_20744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'type_dict')
            # Getting the type of 'var_name' (line 164)
            var_name_20745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 22), 'var_name')
            # Storing an element on a container (line 164)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 12), type_dict_20744, (var_name_20745, subscript_call_result_20743))
        else:
            
            # Testing the type of an if condition (line 158)
            if_condition_20724 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), result_contains_20723)
            # Assigning a type to the variable 'if_condition_20724' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_20724', if_condition_20724)
            # SSA begins for if statement (line 158)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 160):
            
            # Call to add(...): (line 160)
            # Processing the call arguments (line 160)
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 160)
            var_name_20728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 85), 'var_name', False)
            # Getting the type of 'function_context_new' (line 160)
            function_context_new_20729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 64), 'function_context_new', False)
            # Obtaining the member '__getitem__' of a type (line 160)
            getitem___20730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 64), function_context_new_20729, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 160)
            subscript_call_result_20731 = invoke(stypy.reporting.localization.Localization(__file__, 160, 64), getitem___20730, var_name_20728)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 161)
            var_name_20732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 85), 'var_name', False)
            # Getting the type of 'function_context_previous' (line 161)
            function_context_previous_20733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 59), 'function_context_previous', False)
            # Obtaining the member '__getitem__' of a type (line 161)
            getitem___20734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 59), function_context_previous_20733, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 161)
            subscript_call_result_20735 = invoke(stypy.reporting.localization.Localization(__file__, 161, 59), getitem___20734, var_name_20732)
            
            # Processing the call keyword arguments (line 160)
            kwargs_20736 = {}
            # Getting the type of 'union_type_copy' (line 160)
            union_type_copy_20725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 34), 'union_type_copy', False)
            # Obtaining the member 'UnionType' of a type (line 160)
            UnionType_20726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 34), union_type_copy_20725, 'UnionType')
            # Obtaining the member 'add' of a type (line 160)
            add_20727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 34), UnionType_20726, 'add')
            # Calling add(args, kwargs) (line 160)
            add_call_result_20737 = invoke(stypy.reporting.localization.Localization(__file__, 160, 34), add_20727, *[subscript_call_result_20731, subscript_call_result_20735], **kwargs_20736)
            
            # Getting the type of 'type_dict' (line 160)
            type_dict_20738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'type_dict')
            # Getting the type of 'var_name' (line 160)
            var_name_20739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 22), 'var_name')
            # Storing an element on a container (line 160)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 12), type_dict_20738, (var_name_20739, add_call_result_20737))
            # SSA branch for the else part of an if statement (line 158)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Subscript to a Subscript (line 164):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 164)
            var_name_20740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 55), 'var_name')
            # Getting the type of 'function_context_new' (line 164)
            function_context_new_20741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'function_context_new')
            # Obtaining the member '__getitem__' of a type (line 164)
            getitem___20742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 34), function_context_new_20741, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 164)
            subscript_call_result_20743 = invoke(stypy.reporting.localization.Localization(__file__, 164, 34), getitem___20742, var_name_20740)
            
            # Getting the type of 'type_dict' (line 164)
            type_dict_20744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'type_dict')
            # Getting the type of 'var_name' (line 164)
            var_name_20745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 22), 'var_name')
            # Storing an element on a container (line 164)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 12), type_dict_20744, (var_name_20745, subscript_call_result_20743))
            # SSA join for if statement (line 158)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'type_dict' (line 167)
    type_dict_20746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'type_dict')
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type', type_dict_20746)
    
    # ################# End of '__join_except_branches_function_context(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join_except_branches_function_context' in the type store
    # Getting the type of 'stypy_return_type' (line 138)
    stypy_return_type_20747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20747)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join_except_branches_function_context'
    return stypy_return_type_20747

# Assigning a type to the variable '__join_except_branches_function_context' (line 138)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), '__join_except_branches_function_context', __join_except_branches_function_context)

@norecursion
def __join_except_branches(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__join_except_branches'
    module_type_store = module_type_store.open_function_context('__join_except_branches', 170, 0, False)
    
    # Passed parameters checking function
    __join_except_branches.stypy_localization = localization
    __join_except_branches.stypy_type_of_self = None
    __join_except_branches.stypy_type_store = module_type_store
    __join_except_branches.stypy_function_name = '__join_except_branches'
    __join_except_branches.stypy_param_names_list = ['type_store_previous', 'type_store_new']
    __join_except_branches.stypy_varargs_param_name = None
    __join_except_branches.stypy_kwargs_param_name = None
    __join_except_branches.stypy_call_defaults = defaults
    __join_except_branches.stypy_call_varargs = varargs
    __join_except_branches.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__join_except_branches', ['type_store_previous', 'type_store_new'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__join_except_branches', localization, ['type_store_previous', 'type_store_new'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__join_except_branches(...)' code ##################

    str_20748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, (-1)), 'str', '\n    SSA algorithm to join type stores of different except branches\n    :param type_store_previous: Type store\n    :param type_store_new: Type store\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 177):
    
    # Call to TypeStore(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'type_store_previous' (line 177)
    type_store_previous_20750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 34), 'type_store_previous', False)
    # Obtaining the member 'program_name' of a type (line 177)
    program_name_20751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 34), type_store_previous_20750, 'program_name')
    # Processing the call keyword arguments (line 177)
    kwargs_20752 = {}
    # Getting the type of 'TypeStore' (line 177)
    TypeStore_20749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'TypeStore', False)
    # Calling TypeStore(args, kwargs) (line 177)
    TypeStore_call_result_20753 = invoke(stypy.reporting.localization.Localization(__file__, 177, 24), TypeStore_20749, *[program_name_20751], **kwargs_20752)
    
    # Assigning a type to the variable 'joined_type_store' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'joined_type_store', TypeStore_call_result_20753)
    
    # Assigning a Attribute to a Attribute (line 178):
    # Getting the type of 'type_store_previous' (line 178)
    type_store_previous_20754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 47), 'type_store_previous')
    # Obtaining the member 'last_function_contexts' of a type (line 178)
    last_function_contexts_20755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 47), type_store_previous_20754, 'last_function_contexts')
    # Getting the type of 'joined_type_store' (line 178)
    joined_type_store_20756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'joined_type_store')
    # Setting the type of the member 'last_function_contexts' of a type (line 178)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 4), joined_type_store_20756, 'last_function_contexts', last_function_contexts_20755)
    
    # Assigning a List to a Attribute (line 179):
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_20757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    
    # Getting the type of 'joined_type_store' (line 179)
    joined_type_store_20758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'joined_type_store')
    # Setting the type of the member 'context_stack' of a type (line 179)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 4), joined_type_store_20758, 'context_stack', list_20757)
    
    
    # Call to range(...): (line 180)
    # Processing the call arguments (line 180)
    
    # Call to len(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'type_store_previous' (line 180)
    type_store_previous_20761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'type_store_previous', False)
    # Obtaining the member 'context_stack' of a type (line 180)
    context_stack_20762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 23), type_store_previous_20761, 'context_stack')
    # Processing the call keyword arguments (line 180)
    kwargs_20763 = {}
    # Getting the type of 'len' (line 180)
    len_20760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'len', False)
    # Calling len(args, kwargs) (line 180)
    len_call_result_20764 = invoke(stypy.reporting.localization.Localization(__file__, 180, 19), len_20760, *[context_stack_20762], **kwargs_20763)
    
    # Processing the call keyword arguments (line 180)
    kwargs_20765 = {}
    # Getting the type of 'range' (line 180)
    range_20759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 13), 'range', False)
    # Calling range(args, kwargs) (line 180)
    range_call_result_20766 = invoke(stypy.reporting.localization.Localization(__file__, 180, 13), range_20759, *[len_call_result_20764], **kwargs_20765)
    
    # Assigning a type to the variable 'range_call_result_20766' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'range_call_result_20766', range_call_result_20766)
    # Testing if the for loop is going to be iterated (line 180)
    # Testing the type of a for loop iterable (line 180)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 180, 4), range_call_result_20766)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 180, 4), range_call_result_20766):
        # Getting the type of the for loop variable (line 180)
        for_loop_var_20767 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 180, 4), range_call_result_20766)
        # Assigning a type to the variable 'i' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'i', for_loop_var_20767)
        # SSA begins for a for statement (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 181):
        
        # Call to __join_except_branches_function_context(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 181)
        i_20769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 90), 'i', False)
        # Getting the type of 'type_store_previous' (line 181)
        type_store_previous_20770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 70), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___20771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 70), type_store_previous_20770, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_20772 = invoke(stypy.reporting.localization.Localization(__file__, 181, 70), getitem___20771, i_20769)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 181)
        i_20773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 109), 'i', False)
        # Getting the type of 'type_store_new' (line 181)
        type_store_new_20774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 94), 'type_store_new', False)
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___20775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 94), type_store_new_20774, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_20776 = invoke(stypy.reporting.localization.Localization(__file__, 181, 94), getitem___20775, i_20773)
        
        # Processing the call keyword arguments (line 181)
        kwargs_20777 = {}
        # Getting the type of '__join_except_branches_function_context' (line 181)
        join_except_branches_function_context_20768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), '__join_except_branches_function_context', False)
        # Calling __join_except_branches_function_context(args, kwargs) (line 181)
        join_except_branches_function_context_call_result_20778 = invoke(stypy.reporting.localization.Localization(__file__, 181, 30), join_except_branches_function_context_20768, *[subscript_call_result_20772, subscript_call_result_20776], **kwargs_20777)
        
        # Assigning a type to the variable 'joined_context_dict' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'joined_context_dict', join_except_branches_function_context_call_result_20778)
        
        # Assigning a Call to a Name (line 183):
        
        # Call to copy(...): (line 183)
        # Processing the call keyword arguments (line 183)
        kwargs_20784 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 183)
        i_20779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 47), 'i', False)
        # Getting the type of 'type_store_previous' (line 183)
        type_store_previous_20780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 27), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___20781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 27), type_store_previous_20780, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_20782 = invoke(stypy.reporting.localization.Localization(__file__, 183, 27), getitem___20781, i_20779)
        
        # Obtaining the member 'copy' of a type (line 183)
        copy_20783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 27), subscript_call_result_20782, 'copy')
        # Calling copy(args, kwargs) (line 183)
        copy_call_result_20785 = invoke(stypy.reporting.localization.Localization(__file__, 183, 27), copy_20783, *[], **kwargs_20784)
        
        # Assigning a type to the variable 'joined_f_context' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'joined_f_context', copy_call_result_20785)
        
        # Assigning a Name to a Attribute (line 184):
        # Getting the type of 'joined_context_dict' (line 184)
        joined_context_dict_20786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'joined_context_dict')
        # Getting the type of 'joined_f_context' (line 184)
        joined_f_context_20787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'joined_f_context')
        # Setting the type of the member 'types_of' of a type (line 184)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), joined_f_context_20787, 'types_of', joined_context_dict_20786)
        
        # Assigning a Call to a Attribute (line 185):
        
        # Call to __join_globals(...): (line 185)
        # Processing the call arguments (line 185)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 185)
        i_20789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 74), 'i', False)
        # Getting the type of 'type_store_previous' (line 185)
        type_store_previous_20790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 54), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___20791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 54), type_store_previous_20790, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_20792 = invoke(stypy.reporting.localization.Localization(__file__, 185, 54), getitem___20791, i_20789)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 185)
        i_20793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 93), 'i', False)
        # Getting the type of 'type_store_new' (line 185)
        type_store_new_20794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 78), 'type_store_new', False)
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___20795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 78), type_store_new_20794, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_20796 = invoke(stypy.reporting.localization.Localization(__file__, 185, 78), getitem___20795, i_20793)
        
        # Processing the call keyword arguments (line 185)
        kwargs_20797 = {}
        # Getting the type of '__join_globals' (line 185)
        join_globals_20788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 39), '__join_globals', False)
        # Calling __join_globals(args, kwargs) (line 185)
        join_globals_call_result_20798 = invoke(stypy.reporting.localization.Localization(__file__, 185, 39), join_globals_20788, *[subscript_call_result_20792, subscript_call_result_20796], **kwargs_20797)
        
        # Getting the type of 'joined_f_context' (line 185)
        joined_f_context_20799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'joined_f_context')
        # Setting the type of the member 'global_vars' of a type (line 185)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), joined_f_context_20799, 'global_vars', join_globals_call_result_20798)
        
        # Assigning a Attribute to a Attribute (line 186):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 187)
        i_20800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'i')
        # Getting the type of 'type_store_previous' (line 186)
        type_store_previous_20801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 45), 'type_store_previous')
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___20802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 45), type_store_previous_20801, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_20803 = invoke(stypy.reporting.localization.Localization(__file__, 186, 45), getitem___20802, i_20800)
        
        # Obtaining the member 'annotation_record' of a type (line 186)
        annotation_record_20804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 45), subscript_call_result_20803, 'annotation_record')
        # Getting the type of 'joined_f_context' (line 186)
        joined_f_context_20805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'joined_f_context')
        # Setting the type of the member 'annotation_record' of a type (line 186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), joined_f_context_20805, 'annotation_record', annotation_record_20804)
        
        # Call to append(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'joined_f_context' (line 189)
        joined_f_context_20809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 47), 'joined_f_context', False)
        # Processing the call keyword arguments (line 189)
        kwargs_20810 = {}
        # Getting the type of 'joined_type_store' (line 189)
        joined_type_store_20806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'joined_type_store', False)
        # Obtaining the member 'context_stack' of a type (line 189)
        context_stack_20807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), joined_type_store_20806, 'context_stack')
        # Obtaining the member 'append' of a type (line 189)
        append_20808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), context_stack_20807, 'append')
        # Calling append(args, kwargs) (line 189)
        append_call_result_20811 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), append_20808, *[joined_f_context_20809], **kwargs_20810)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'joined_type_store' (line 191)
    joined_type_store_20812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'joined_type_store')
    # Assigning a type to the variable 'stypy_return_type' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type', joined_type_store_20812)
    
    # ################# End of '__join_except_branches(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join_except_branches' in the type store
    # Getting the type of 'stypy_return_type' (line 170)
    stypy_return_type_20813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20813)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join_except_branches'
    return stypy_return_type_20813

# Assigning a type to the variable '__join_except_branches' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), '__join_except_branches', __join_except_branches)

@norecursion
def __join_finally_function_context(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__join_finally_function_context'
    module_type_store = module_type_store.open_function_context('__join_finally_function_context', 194, 0, False)
    
    # Passed parameters checking function
    __join_finally_function_context.stypy_localization = localization
    __join_finally_function_context.stypy_type_of_self = None
    __join_finally_function_context.stypy_type_store = module_type_store
    __join_finally_function_context.stypy_function_name = '__join_finally_function_context'
    __join_finally_function_context.stypy_param_names_list = ['function_context_previous', 'function_context_finally']
    __join_finally_function_context.stypy_varargs_param_name = None
    __join_finally_function_context.stypy_kwargs_param_name = None
    __join_finally_function_context.stypy_call_defaults = defaults
    __join_finally_function_context.stypy_call_varargs = varargs
    __join_finally_function_context.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__join_finally_function_context', ['function_context_previous', 'function_context_finally'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__join_finally_function_context', localization, ['function_context_previous', 'function_context_finally'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__join_finally_function_context(...)' code ##################

    str_20814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, (-1)), 'str', '\n    Join the variables of a function context on a finally block with a function context of the joined type store\n     of all the except branches in an exception clause\n    :param function_context_previous: Function context\n    :param function_context_finally: Function context\n    :return: A dictionary with names of variables and its joined types\n    ')
    
    # Assigning a Dict to a Name (line 202):
    
    # Obtaining an instance of the builtin type 'dict' (line 202)
    dict_20815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 202)
    
    # Assigning a type to the variable 'type_dict' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'type_dict', dict_20815)
    
    # Getting the type of 'function_context_previous' (line 204)
    function_context_previous_20816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'function_context_previous')
    # Assigning a type to the variable 'function_context_previous_20816' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'function_context_previous_20816', function_context_previous_20816)
    # Testing if the for loop is going to be iterated (line 204)
    # Testing the type of a for loop iterable (line 204)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 204, 4), function_context_previous_20816)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 204, 4), function_context_previous_20816):
        # Getting the type of the for loop variable (line 204)
        for_loop_var_20817 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 204, 4), function_context_previous_20816)
        # Assigning a type to the variable 'var_name' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'var_name', for_loop_var_20817)
        # SSA begins for a for statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 205)
        var_name_20818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'var_name')
        # Getting the type of 'function_context_finally' (line 205)
        function_context_finally_20819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 23), 'function_context_finally')
        # Applying the binary operator 'in' (line 205)
        result_contains_20820 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), 'in', var_name_20818, function_context_finally_20819)
        
        # Testing if the type of an if condition is none (line 205)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 205, 8), result_contains_20820):
            pass
        else:
            
            # Testing the type of an if condition (line 205)
            if_condition_20821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), result_contains_20820)
            # Assigning a type to the variable 'if_condition_20821' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_20821', if_condition_20821)
            # SSA begins for if statement (line 205)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Subscript (line 207):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 207)
            var_name_20822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 59), 'var_name')
            # Getting the type of 'function_context_finally' (line 207)
            function_context_finally_20823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 34), 'function_context_finally')
            # Obtaining the member '__getitem__' of a type (line 207)
            getitem___20824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 34), function_context_finally_20823, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 207)
            subscript_call_result_20825 = invoke(stypy.reporting.localization.Localization(__file__, 207, 34), getitem___20824, var_name_20822)
            
            # Getting the type of 'type_dict' (line 207)
            type_dict_20826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'type_dict')
            # Getting the type of 'var_name' (line 207)
            var_name_20827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 22), 'var_name')
            # Storing an element on a container (line 207)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 12), type_dict_20826, (var_name_20827, subscript_call_result_20825))
            # SSA branch for the else part of an if statement (line 205)
            module_type_store.open_ssa_branch('else')
            pass
            # SSA join for if statement (line 205)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'function_context_finally' (line 212)
    function_context_finally_20828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'function_context_finally')
    # Assigning a type to the variable 'function_context_finally_20828' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'function_context_finally_20828', function_context_finally_20828)
    # Testing if the for loop is going to be iterated (line 212)
    # Testing the type of a for loop iterable (line 212)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 212, 4), function_context_finally_20828)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 212, 4), function_context_finally_20828):
        # Getting the type of the for loop variable (line 212)
        for_loop_var_20829 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 212, 4), function_context_finally_20828)
        # Assigning a type to the variable 'var_name' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'var_name', for_loop_var_20829)
        # SSA begins for a for statement (line 212)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 213)
        var_name_20830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'var_name')
        # Getting the type of 'function_context_previous' (line 213)
        function_context_previous_20831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 23), 'function_context_previous')
        # Applying the binary operator 'in' (line 213)
        result_contains_20832 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 11), 'in', var_name_20830, function_context_previous_20831)
        
        # Testing if the type of an if condition is none (line 213)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 213, 8), result_contains_20832):
            
            # Assigning a Subscript to a Subscript (line 218):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 218)
            var_name_20834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 59), 'var_name')
            # Getting the type of 'function_context_finally' (line 218)
            function_context_finally_20835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 34), 'function_context_finally')
            # Obtaining the member '__getitem__' of a type (line 218)
            getitem___20836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 34), function_context_finally_20835, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 218)
            subscript_call_result_20837 = invoke(stypy.reporting.localization.Localization(__file__, 218, 34), getitem___20836, var_name_20834)
            
            # Getting the type of 'type_dict' (line 218)
            type_dict_20838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'type_dict')
            # Getting the type of 'var_name' (line 218)
            var_name_20839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'var_name')
            # Storing an element on a container (line 218)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 12), type_dict_20838, (var_name_20839, subscript_call_result_20837))
        else:
            
            # Testing the type of an if condition (line 213)
            if_condition_20833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 8), result_contains_20832)
            # Assigning a type to the variable 'if_condition_20833' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'if_condition_20833', if_condition_20833)
            # SSA begins for if statement (line 213)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA branch for the else part of an if statement (line 213)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Subscript to a Subscript (line 218):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 218)
            var_name_20834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 59), 'var_name')
            # Getting the type of 'function_context_finally' (line 218)
            function_context_finally_20835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 34), 'function_context_finally')
            # Obtaining the member '__getitem__' of a type (line 218)
            getitem___20836 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 34), function_context_finally_20835, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 218)
            subscript_call_result_20837 = invoke(stypy.reporting.localization.Localization(__file__, 218, 34), getitem___20836, var_name_20834)
            
            # Getting the type of 'type_dict' (line 218)
            type_dict_20838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'type_dict')
            # Getting the type of 'var_name' (line 218)
            var_name_20839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'var_name')
            # Storing an element on a container (line 218)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 12), type_dict_20838, (var_name_20839, subscript_call_result_20837))
            # SSA join for if statement (line 213)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'type_dict' (line 221)
    type_dict_20840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'type_dict')
    # Assigning a type to the variable 'stypy_return_type' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type', type_dict_20840)
    
    # ################# End of '__join_finally_function_context(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join_finally_function_context' in the type store
    # Getting the type of 'stypy_return_type' (line 194)
    stypy_return_type_20841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20841)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join_finally_function_context'
    return stypy_return_type_20841

# Assigning a type to the variable '__join_finally_function_context' (line 194)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), '__join_finally_function_context', __join_finally_function_context)

@norecursion
def __join_finally_branch(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__join_finally_branch'
    module_type_store = module_type_store.open_function_context('__join_finally_branch', 224, 0, False)
    
    # Passed parameters checking function
    __join_finally_branch.stypy_localization = localization
    __join_finally_branch.stypy_type_of_self = None
    __join_finally_branch.stypy_type_store = module_type_store
    __join_finally_branch.stypy_function_name = '__join_finally_branch'
    __join_finally_branch.stypy_param_names_list = ['type_store_exception_block', 'type_store_finally']
    __join_finally_branch.stypy_varargs_param_name = None
    __join_finally_branch.stypy_kwargs_param_name = None
    __join_finally_branch.stypy_call_defaults = defaults
    __join_finally_branch.stypy_call_varargs = varargs
    __join_finally_branch.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__join_finally_branch', ['type_store_exception_block', 'type_store_finally'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__join_finally_branch', localization, ['type_store_exception_block', 'type_store_finally'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__join_finally_branch(...)' code ##################

    str_20842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, (-1)), 'str', '\n    Join the type stores of a finally branch and the joined type store of all except branches in a exception handling\n     block\n    :param type_store_exception_block: Type store\n    :param type_store_finally: Type store\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 232):
    
    # Call to TypeStore(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'type_store_exception_block' (line 232)
    type_store_exception_block_20844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 34), 'type_store_exception_block', False)
    # Obtaining the member 'program_name' of a type (line 232)
    program_name_20845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 34), type_store_exception_block_20844, 'program_name')
    # Processing the call keyword arguments (line 232)
    kwargs_20846 = {}
    # Getting the type of 'TypeStore' (line 232)
    TypeStore_20843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'TypeStore', False)
    # Calling TypeStore(args, kwargs) (line 232)
    TypeStore_call_result_20847 = invoke(stypy.reporting.localization.Localization(__file__, 232, 24), TypeStore_20843, *[program_name_20845], **kwargs_20846)
    
    # Assigning a type to the variable 'joined_type_store' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'joined_type_store', TypeStore_call_result_20847)
    
    # Assigning a Attribute to a Attribute (line 233):
    # Getting the type of 'type_store_exception_block' (line 233)
    type_store_exception_block_20848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 47), 'type_store_exception_block')
    # Obtaining the member 'last_function_contexts' of a type (line 233)
    last_function_contexts_20849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 47), type_store_exception_block_20848, 'last_function_contexts')
    # Getting the type of 'joined_type_store' (line 233)
    joined_type_store_20850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'joined_type_store')
    # Setting the type of the member 'last_function_contexts' of a type (line 233)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 4), joined_type_store_20850, 'last_function_contexts', last_function_contexts_20849)
    
    # Assigning a List to a Attribute (line 234):
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_20851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    
    # Getting the type of 'joined_type_store' (line 234)
    joined_type_store_20852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'joined_type_store')
    # Setting the type of the member 'context_stack' of a type (line 234)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 4), joined_type_store_20852, 'context_stack', list_20851)
    
    
    # Call to range(...): (line 235)
    # Processing the call arguments (line 235)
    
    # Call to len(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'type_store_exception_block' (line 235)
    type_store_exception_block_20855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'type_store_exception_block', False)
    # Obtaining the member 'context_stack' of a type (line 235)
    context_stack_20856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 23), type_store_exception_block_20855, 'context_stack')
    # Processing the call keyword arguments (line 235)
    kwargs_20857 = {}
    # Getting the type of 'len' (line 235)
    len_20854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'len', False)
    # Calling len(args, kwargs) (line 235)
    len_call_result_20858 = invoke(stypy.reporting.localization.Localization(__file__, 235, 19), len_20854, *[context_stack_20856], **kwargs_20857)
    
    # Processing the call keyword arguments (line 235)
    kwargs_20859 = {}
    # Getting the type of 'range' (line 235)
    range_20853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'range', False)
    # Calling range(args, kwargs) (line 235)
    range_call_result_20860 = invoke(stypy.reporting.localization.Localization(__file__, 235, 13), range_20853, *[len_call_result_20858], **kwargs_20859)
    
    # Assigning a type to the variable 'range_call_result_20860' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'range_call_result_20860', range_call_result_20860)
    # Testing if the for loop is going to be iterated (line 235)
    # Testing the type of a for loop iterable (line 235)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 235, 4), range_call_result_20860)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 235, 4), range_call_result_20860):
        # Getting the type of the for loop variable (line 235)
        for_loop_var_20861 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 235, 4), range_call_result_20860)
        # Assigning a type to the variable 'i' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'i', for_loop_var_20861)
        # SSA begins for a for statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 236):
        
        # Call to __join_finally_function_context(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 236)
        i_20863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 89), 'i', False)
        # Getting the type of 'type_store_exception_block' (line 236)
        type_store_exception_block_20864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 62), 'type_store_exception_block', False)
        # Obtaining the member '__getitem__' of a type (line 236)
        getitem___20865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 62), type_store_exception_block_20864, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 236)
        subscript_call_result_20866 = invoke(stypy.reporting.localization.Localization(__file__, 236, 62), getitem___20865, i_20863)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 236)
        i_20867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 112), 'i', False)
        # Getting the type of 'type_store_finally' (line 236)
        type_store_finally_20868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 93), 'type_store_finally', False)
        # Obtaining the member '__getitem__' of a type (line 236)
        getitem___20869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 93), type_store_finally_20868, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 236)
        subscript_call_result_20870 = invoke(stypy.reporting.localization.Localization(__file__, 236, 93), getitem___20869, i_20867)
        
        # Processing the call keyword arguments (line 236)
        kwargs_20871 = {}
        # Getting the type of '__join_finally_function_context' (line 236)
        join_finally_function_context_20862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 30), '__join_finally_function_context', False)
        # Calling __join_finally_function_context(args, kwargs) (line 236)
        join_finally_function_context_call_result_20872 = invoke(stypy.reporting.localization.Localization(__file__, 236, 30), join_finally_function_context_20862, *[subscript_call_result_20866, subscript_call_result_20870], **kwargs_20871)
        
        # Assigning a type to the variable 'joined_context_dict' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'joined_context_dict', join_finally_function_context_call_result_20872)
        
        # Assigning a Call to a Name (line 238):
        
        # Call to copy(...): (line 238)
        # Processing the call keyword arguments (line 238)
        kwargs_20878 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 238)
        i_20873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 54), 'i', False)
        # Getting the type of 'type_store_exception_block' (line 238)
        type_store_exception_block_20874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 27), 'type_store_exception_block', False)
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___20875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 27), type_store_exception_block_20874, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 238)
        subscript_call_result_20876 = invoke(stypy.reporting.localization.Localization(__file__, 238, 27), getitem___20875, i_20873)
        
        # Obtaining the member 'copy' of a type (line 238)
        copy_20877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 27), subscript_call_result_20876, 'copy')
        # Calling copy(args, kwargs) (line 238)
        copy_call_result_20879 = invoke(stypy.reporting.localization.Localization(__file__, 238, 27), copy_20877, *[], **kwargs_20878)
        
        # Assigning a type to the variable 'joined_f_context' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'joined_f_context', copy_call_result_20879)
        
        # Assigning a Name to a Attribute (line 239):
        # Getting the type of 'joined_context_dict' (line 239)
        joined_context_dict_20880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 36), 'joined_context_dict')
        # Getting the type of 'joined_f_context' (line 239)
        joined_f_context_20881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'joined_f_context')
        # Setting the type of the member 'types_of' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), joined_f_context_20881, 'types_of', joined_context_dict_20880)
        
        # Assigning a Call to a Attribute (line 240):
        
        # Call to __join_globals(...): (line 240)
        # Processing the call arguments (line 240)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 240)
        i_20883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 81), 'i', False)
        # Getting the type of 'type_store_exception_block' (line 240)
        type_store_exception_block_20884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 54), 'type_store_exception_block', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___20885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 54), type_store_exception_block_20884, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 240)
        subscript_call_result_20886 = invoke(stypy.reporting.localization.Localization(__file__, 240, 54), getitem___20885, i_20883)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 240)
        i_20887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 104), 'i', False)
        # Getting the type of 'type_store_finally' (line 240)
        type_store_finally_20888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 85), 'type_store_finally', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___20889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 85), type_store_finally_20888, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 240)
        subscript_call_result_20890 = invoke(stypy.reporting.localization.Localization(__file__, 240, 85), getitem___20889, i_20887)
        
        # Processing the call keyword arguments (line 240)
        kwargs_20891 = {}
        # Getting the type of '__join_globals' (line 240)
        join_globals_20882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 39), '__join_globals', False)
        # Calling __join_globals(args, kwargs) (line 240)
        join_globals_call_result_20892 = invoke(stypy.reporting.localization.Localization(__file__, 240, 39), join_globals_20882, *[subscript_call_result_20886, subscript_call_result_20890], **kwargs_20891)
        
        # Getting the type of 'joined_f_context' (line 240)
        joined_f_context_20893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'joined_f_context')
        # Setting the type of the member 'global_vars' of a type (line 240)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), joined_f_context_20893, 'global_vars', join_globals_call_result_20892)
        
        # Assigning a Attribute to a Attribute (line 241):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 242)
        i_20894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'i')
        # Getting the type of 'type_store_exception_block' (line 241)
        type_store_exception_block_20895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 45), 'type_store_exception_block')
        # Obtaining the member '__getitem__' of a type (line 241)
        getitem___20896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 45), type_store_exception_block_20895, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 241)
        subscript_call_result_20897 = invoke(stypy.reporting.localization.Localization(__file__, 241, 45), getitem___20896, i_20894)
        
        # Obtaining the member 'annotation_record' of a type (line 241)
        annotation_record_20898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 45), subscript_call_result_20897, 'annotation_record')
        # Getting the type of 'joined_f_context' (line 241)
        joined_f_context_20899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'joined_f_context')
        # Setting the type of the member 'annotation_record' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), joined_f_context_20899, 'annotation_record', annotation_record_20898)
        
        # Call to append(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'joined_f_context' (line 244)
        joined_f_context_20903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 47), 'joined_f_context', False)
        # Processing the call keyword arguments (line 244)
        kwargs_20904 = {}
        # Getting the type of 'joined_type_store' (line 244)
        joined_type_store_20900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'joined_type_store', False)
        # Obtaining the member 'context_stack' of a type (line 244)
        context_stack_20901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), joined_type_store_20900, 'context_stack')
        # Obtaining the member 'append' of a type (line 244)
        append_20902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), context_stack_20901, 'append')
        # Calling append(args, kwargs) (line 244)
        append_call_result_20905 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), append_20902, *[joined_f_context_20903], **kwargs_20904)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'joined_type_store' (line 246)
    joined_type_store_20906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'joined_type_store')
    # Assigning a type to the variable 'stypy_return_type' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type', joined_type_store_20906)
    
    # ################# End of '__join_finally_branch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join_finally_branch' in the type store
    # Getting the type of 'stypy_return_type' (line 224)
    stypy_return_type_20907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_20907)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join_finally_branch'
    return stypy_return_type_20907

# Assigning a type to the variable '__join_finally_branch' (line 224)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), '__join_finally_branch', __join_finally_branch)

@norecursion
def __join_try_except_function_context(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__join_try_except_function_context'
    module_type_store = module_type_store.open_function_context('__join_try_except_function_context', 249, 0, False)
    
    # Passed parameters checking function
    __join_try_except_function_context.stypy_localization = localization
    __join_try_except_function_context.stypy_type_of_self = None
    __join_try_except_function_context.stypy_type_store = module_type_store
    __join_try_except_function_context.stypy_function_name = '__join_try_except_function_context'
    __join_try_except_function_context.stypy_param_names_list = ['function_context_previous', 'function_context_try', 'function_context_except']
    __join_try_except_function_context.stypy_varargs_param_name = None
    __join_try_except_function_context.stypy_kwargs_param_name = None
    __join_try_except_function_context.stypy_call_defaults = defaults
    __join_try_except_function_context.stypy_call_varargs = varargs
    __join_try_except_function_context.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__join_try_except_function_context', ['function_context_previous', 'function_context_try', 'function_context_except'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__join_try_except_function_context', localization, ['function_context_previous', 'function_context_try', 'function_context_except'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__join_try_except_function_context(...)' code ##################

    str_20908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, (-1)), 'str', '\n    Implements the SSA algorithm in try-except blocks, dealing with function contexts.\n\n    :param function_context_previous: Function context\n    :param function_context_try: Function context\n    :param function_context_except: Function context\n    :return: A dictionary with names of variables and its joined types\n    ')
    
    # Assigning a Dict to a Name (line 258):
    
    # Obtaining an instance of the builtin type 'dict' (line 258)
    dict_20909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 258)
    
    # Assigning a type to the variable 'type_dict' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'type_dict', dict_20909)
    
    # Getting the type of 'function_context_try' (line 260)
    function_context_try_20910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), 'function_context_try')
    # Assigning a type to the variable 'function_context_try_20910' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'function_context_try_20910', function_context_try_20910)
    # Testing if the for loop is going to be iterated (line 260)
    # Testing the type of a for loop iterable (line 260)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 260, 4), function_context_try_20910)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 260, 4), function_context_try_20910):
        # Getting the type of the for loop variable (line 260)
        for_loop_var_20911 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 260, 4), function_context_try_20910)
        # Assigning a type to the variable 'var_name' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'var_name', for_loop_var_20911)
        # SSA begins for a for statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 261)
        var_name_20912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'var_name')
        # Getting the type of 'function_context_except' (line 261)
        function_context_except_20913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'function_context_except')
        # Applying the binary operator 'in' (line 261)
        result_contains_20914 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), 'in', var_name_20912, function_context_except_20913)
        
        # Testing if the type of an if condition is none (line 261)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 261, 8), result_contains_20914):
            
            # Getting the type of 'var_name' (line 269)
            var_name_20949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 269)
            function_context_previous_20950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 269)
            result_contains_20951 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 15), 'in', var_name_20949, function_context_previous_20950)
            
            # Testing if the type of an if condition is none (line 269)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 269, 12), result_contains_20951):
                
                # Assigning a Call to a Subscript (line 275):
                
                # Call to add(...): (line 275)
                # Processing the call arguments (line 275)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 275)
                var_name_20971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 89), 'var_name', False)
                # Getting the type of 'function_context_try' (line 275)
                function_context_try_20972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 68), 'function_context_try', False)
                # Obtaining the member '__getitem__' of a type (line 275)
                getitem___20973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 68), function_context_try_20972, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 275)
                subscript_call_result_20974 = invoke(stypy.reporting.localization.Localization(__file__, 275, 68), getitem___20973, var_name_20971)
                
                
                # Call to UndefinedType(...): (line 275)
                # Processing the call keyword arguments (line 275)
                kwargs_20976 = {}
                # Getting the type of 'UndefinedType' (line 275)
                UndefinedType_20975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 100), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 275)
                UndefinedType_call_result_20977 = invoke(stypy.reporting.localization.Localization(__file__, 275, 100), UndefinedType_20975, *[], **kwargs_20976)
                
                # Processing the call keyword arguments (line 275)
                kwargs_20978 = {}
                # Getting the type of 'union_type_copy' (line 275)
                union_type_copy_20968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 275)
                UnionType_20969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), union_type_copy_20968, 'UnionType')
                # Obtaining the member 'add' of a type (line 275)
                add_20970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), UnionType_20969, 'add')
                # Calling add(args, kwargs) (line 275)
                add_call_result_20979 = invoke(stypy.reporting.localization.Localization(__file__, 275, 38), add_20970, *[subscript_call_result_20974, UndefinedType_call_result_20977], **kwargs_20978)
                
                # Getting the type of 'type_dict' (line 275)
                type_dict_20980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'type_dict')
                # Getting the type of 'var_name' (line 275)
                var_name_20981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'var_name')
                # Storing an element on a container (line 275)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 16), type_dict_20980, (var_name_20981, add_call_result_20979))
            else:
                
                # Testing the type of an if condition (line 269)
                if_condition_20952 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 12), result_contains_20951)
                # Assigning a type to the variable 'if_condition_20952' (line 269)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'if_condition_20952', if_condition_20952)
                # SSA begins for if statement (line 269)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 271):
                
                # Call to add(...): (line 271)
                # Processing the call arguments (line 271)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 271)
                var_name_20956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 271)
                function_context_previous_20957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 271)
                getitem___20958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 68), function_context_previous_20957, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 271)
                subscript_call_result_20959 = invoke(stypy.reporting.localization.Localization(__file__, 271, 68), getitem___20958, var_name_20956)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 272)
                var_name_20960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 84), 'var_name', False)
                # Getting the type of 'function_context_try' (line 272)
                function_context_try_20961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 63), 'function_context_try', False)
                # Obtaining the member '__getitem__' of a type (line 272)
                getitem___20962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 63), function_context_try_20961, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 272)
                subscript_call_result_20963 = invoke(stypy.reporting.localization.Localization(__file__, 272, 63), getitem___20962, var_name_20960)
                
                # Processing the call keyword arguments (line 271)
                kwargs_20964 = {}
                # Getting the type of 'union_type_copy' (line 271)
                union_type_copy_20953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 271)
                UnionType_20954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 38), union_type_copy_20953, 'UnionType')
                # Obtaining the member 'add' of a type (line 271)
                add_20955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 38), UnionType_20954, 'add')
                # Calling add(args, kwargs) (line 271)
                add_call_result_20965 = invoke(stypy.reporting.localization.Localization(__file__, 271, 38), add_20955, *[subscript_call_result_20959, subscript_call_result_20963], **kwargs_20964)
                
                # Getting the type of 'type_dict' (line 271)
                type_dict_20966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'type_dict')
                # Getting the type of 'var_name' (line 271)
                var_name_20967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 26), 'var_name')
                # Storing an element on a container (line 271)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 16), type_dict_20966, (var_name_20967, add_call_result_20965))
                # SSA branch for the else part of an if statement (line 269)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 275):
                
                # Call to add(...): (line 275)
                # Processing the call arguments (line 275)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 275)
                var_name_20971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 89), 'var_name', False)
                # Getting the type of 'function_context_try' (line 275)
                function_context_try_20972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 68), 'function_context_try', False)
                # Obtaining the member '__getitem__' of a type (line 275)
                getitem___20973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 68), function_context_try_20972, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 275)
                subscript_call_result_20974 = invoke(stypy.reporting.localization.Localization(__file__, 275, 68), getitem___20973, var_name_20971)
                
                
                # Call to UndefinedType(...): (line 275)
                # Processing the call keyword arguments (line 275)
                kwargs_20976 = {}
                # Getting the type of 'UndefinedType' (line 275)
                UndefinedType_20975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 100), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 275)
                UndefinedType_call_result_20977 = invoke(stypy.reporting.localization.Localization(__file__, 275, 100), UndefinedType_20975, *[], **kwargs_20976)
                
                # Processing the call keyword arguments (line 275)
                kwargs_20978 = {}
                # Getting the type of 'union_type_copy' (line 275)
                union_type_copy_20968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 275)
                UnionType_20969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), union_type_copy_20968, 'UnionType')
                # Obtaining the member 'add' of a type (line 275)
                add_20970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), UnionType_20969, 'add')
                # Calling add(args, kwargs) (line 275)
                add_call_result_20979 = invoke(stypy.reporting.localization.Localization(__file__, 275, 38), add_20970, *[subscript_call_result_20974, UndefinedType_call_result_20977], **kwargs_20978)
                
                # Getting the type of 'type_dict' (line 275)
                type_dict_20980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'type_dict')
                # Getting the type of 'var_name' (line 275)
                var_name_20981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'var_name')
                # Storing an element on a container (line 275)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 16), type_dict_20980, (var_name_20981, add_call_result_20979))
                # SSA join for if statement (line 269)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 261)
            if_condition_20915 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), result_contains_20914)
            # Assigning a type to the variable 'if_condition_20915' (line 261)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_20915', if_condition_20915)
            # SSA begins for if statement (line 261)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 263):
            
            # Call to add(...): (line 263)
            # Processing the call arguments (line 263)
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 263)
            var_name_20919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 85), 'var_name', False)
            # Getting the type of 'function_context_try' (line 263)
            function_context_try_20920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 64), 'function_context_try', False)
            # Obtaining the member '__getitem__' of a type (line 263)
            getitem___20921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 64), function_context_try_20920, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 263)
            subscript_call_result_20922 = invoke(stypy.reporting.localization.Localization(__file__, 263, 64), getitem___20921, var_name_20919)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 264)
            var_name_20923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 83), 'var_name', False)
            # Getting the type of 'function_context_except' (line 264)
            function_context_except_20924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 59), 'function_context_except', False)
            # Obtaining the member '__getitem__' of a type (line 264)
            getitem___20925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 59), function_context_except_20924, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 264)
            subscript_call_result_20926 = invoke(stypy.reporting.localization.Localization(__file__, 264, 59), getitem___20925, var_name_20923)
            
            # Processing the call keyword arguments (line 263)
            kwargs_20927 = {}
            # Getting the type of 'union_type_copy' (line 263)
            union_type_copy_20916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 34), 'union_type_copy', False)
            # Obtaining the member 'UnionType' of a type (line 263)
            UnionType_20917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 34), union_type_copy_20916, 'UnionType')
            # Obtaining the member 'add' of a type (line 263)
            add_20918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 34), UnionType_20917, 'add')
            # Calling add(args, kwargs) (line 263)
            add_call_result_20928 = invoke(stypy.reporting.localization.Localization(__file__, 263, 34), add_20918, *[subscript_call_result_20922, subscript_call_result_20926], **kwargs_20927)
            
            # Getting the type of 'type_dict' (line 263)
            type_dict_20929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'type_dict')
            # Getting the type of 'var_name' (line 263)
            var_name_20930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 22), 'var_name')
            # Storing an element on a container (line 263)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 12), type_dict_20929, (var_name_20930, add_call_result_20928))
            
            # Getting the type of 'var_name' (line 265)
            var_name_20931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 265)
            function_context_previous_20932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 31), 'function_context_previous')
            # Applying the binary operator 'notin' (line 265)
            result_contains_20933 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 15), 'notin', var_name_20931, function_context_previous_20932)
            
            # Testing if the type of an if condition is none (line 265)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 265, 12), result_contains_20933):
                pass
            else:
                
                # Testing the type of an if condition (line 265)
                if_condition_20934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 12), result_contains_20933)
                # Assigning a type to the variable 'if_condition_20934' (line 265)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'if_condition_20934', if_condition_20934)
                # SSA begins for if statement (line 265)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 266):
                
                # Call to add(...): (line 266)
                # Processing the call arguments (line 266)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 266)
                var_name_20938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 78), 'var_name', False)
                # Getting the type of 'type_dict' (line 266)
                type_dict_20939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 68), 'type_dict', False)
                # Obtaining the member '__getitem__' of a type (line 266)
                getitem___20940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 68), type_dict_20939, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 266)
                subscript_call_result_20941 = invoke(stypy.reporting.localization.Localization(__file__, 266, 68), getitem___20940, var_name_20938)
                
                
                # Call to UndefinedType(...): (line 266)
                # Processing the call keyword arguments (line 266)
                kwargs_20943 = {}
                # Getting the type of 'UndefinedType' (line 266)
                UndefinedType_20942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 89), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 266)
                UndefinedType_call_result_20944 = invoke(stypy.reporting.localization.Localization(__file__, 266, 89), UndefinedType_20942, *[], **kwargs_20943)
                
                # Processing the call keyword arguments (line 266)
                kwargs_20945 = {}
                # Getting the type of 'union_type_copy' (line 266)
                union_type_copy_20935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 266)
                UnionType_20936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 38), union_type_copy_20935, 'UnionType')
                # Obtaining the member 'add' of a type (line 266)
                add_20937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 38), UnionType_20936, 'add')
                # Calling add(args, kwargs) (line 266)
                add_call_result_20946 = invoke(stypy.reporting.localization.Localization(__file__, 266, 38), add_20937, *[subscript_call_result_20941, UndefinedType_call_result_20944], **kwargs_20945)
                
                # Getting the type of 'type_dict' (line 266)
                type_dict_20947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'type_dict')
                # Getting the type of 'var_name' (line 266)
                var_name_20948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 26), 'var_name')
                # Storing an element on a container (line 266)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 16), type_dict_20947, (var_name_20948, add_call_result_20946))
                # SSA join for if statement (line 265)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA branch for the else part of an if statement (line 261)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'var_name' (line 269)
            var_name_20949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 269)
            function_context_previous_20950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 269)
            result_contains_20951 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 15), 'in', var_name_20949, function_context_previous_20950)
            
            # Testing if the type of an if condition is none (line 269)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 269, 12), result_contains_20951):
                
                # Assigning a Call to a Subscript (line 275):
                
                # Call to add(...): (line 275)
                # Processing the call arguments (line 275)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 275)
                var_name_20971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 89), 'var_name', False)
                # Getting the type of 'function_context_try' (line 275)
                function_context_try_20972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 68), 'function_context_try', False)
                # Obtaining the member '__getitem__' of a type (line 275)
                getitem___20973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 68), function_context_try_20972, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 275)
                subscript_call_result_20974 = invoke(stypy.reporting.localization.Localization(__file__, 275, 68), getitem___20973, var_name_20971)
                
                
                # Call to UndefinedType(...): (line 275)
                # Processing the call keyword arguments (line 275)
                kwargs_20976 = {}
                # Getting the type of 'UndefinedType' (line 275)
                UndefinedType_20975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 100), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 275)
                UndefinedType_call_result_20977 = invoke(stypy.reporting.localization.Localization(__file__, 275, 100), UndefinedType_20975, *[], **kwargs_20976)
                
                # Processing the call keyword arguments (line 275)
                kwargs_20978 = {}
                # Getting the type of 'union_type_copy' (line 275)
                union_type_copy_20968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 275)
                UnionType_20969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), union_type_copy_20968, 'UnionType')
                # Obtaining the member 'add' of a type (line 275)
                add_20970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), UnionType_20969, 'add')
                # Calling add(args, kwargs) (line 275)
                add_call_result_20979 = invoke(stypy.reporting.localization.Localization(__file__, 275, 38), add_20970, *[subscript_call_result_20974, UndefinedType_call_result_20977], **kwargs_20978)
                
                # Getting the type of 'type_dict' (line 275)
                type_dict_20980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'type_dict')
                # Getting the type of 'var_name' (line 275)
                var_name_20981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'var_name')
                # Storing an element on a container (line 275)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 16), type_dict_20980, (var_name_20981, add_call_result_20979))
            else:
                
                # Testing the type of an if condition (line 269)
                if_condition_20952 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 12), result_contains_20951)
                # Assigning a type to the variable 'if_condition_20952' (line 269)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'if_condition_20952', if_condition_20952)
                # SSA begins for if statement (line 269)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 271):
                
                # Call to add(...): (line 271)
                # Processing the call arguments (line 271)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 271)
                var_name_20956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 271)
                function_context_previous_20957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 271)
                getitem___20958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 68), function_context_previous_20957, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 271)
                subscript_call_result_20959 = invoke(stypy.reporting.localization.Localization(__file__, 271, 68), getitem___20958, var_name_20956)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 272)
                var_name_20960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 84), 'var_name', False)
                # Getting the type of 'function_context_try' (line 272)
                function_context_try_20961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 63), 'function_context_try', False)
                # Obtaining the member '__getitem__' of a type (line 272)
                getitem___20962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 63), function_context_try_20961, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 272)
                subscript_call_result_20963 = invoke(stypy.reporting.localization.Localization(__file__, 272, 63), getitem___20962, var_name_20960)
                
                # Processing the call keyword arguments (line 271)
                kwargs_20964 = {}
                # Getting the type of 'union_type_copy' (line 271)
                union_type_copy_20953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 271)
                UnionType_20954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 38), union_type_copy_20953, 'UnionType')
                # Obtaining the member 'add' of a type (line 271)
                add_20955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 38), UnionType_20954, 'add')
                # Calling add(args, kwargs) (line 271)
                add_call_result_20965 = invoke(stypy.reporting.localization.Localization(__file__, 271, 38), add_20955, *[subscript_call_result_20959, subscript_call_result_20963], **kwargs_20964)
                
                # Getting the type of 'type_dict' (line 271)
                type_dict_20966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'type_dict')
                # Getting the type of 'var_name' (line 271)
                var_name_20967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 26), 'var_name')
                # Storing an element on a container (line 271)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 16), type_dict_20966, (var_name_20967, add_call_result_20965))
                # SSA branch for the else part of an if statement (line 269)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 275):
                
                # Call to add(...): (line 275)
                # Processing the call arguments (line 275)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 275)
                var_name_20971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 89), 'var_name', False)
                # Getting the type of 'function_context_try' (line 275)
                function_context_try_20972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 68), 'function_context_try', False)
                # Obtaining the member '__getitem__' of a type (line 275)
                getitem___20973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 68), function_context_try_20972, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 275)
                subscript_call_result_20974 = invoke(stypy.reporting.localization.Localization(__file__, 275, 68), getitem___20973, var_name_20971)
                
                
                # Call to UndefinedType(...): (line 275)
                # Processing the call keyword arguments (line 275)
                kwargs_20976 = {}
                # Getting the type of 'UndefinedType' (line 275)
                UndefinedType_20975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 100), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 275)
                UndefinedType_call_result_20977 = invoke(stypy.reporting.localization.Localization(__file__, 275, 100), UndefinedType_20975, *[], **kwargs_20976)
                
                # Processing the call keyword arguments (line 275)
                kwargs_20978 = {}
                # Getting the type of 'union_type_copy' (line 275)
                union_type_copy_20968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 275)
                UnionType_20969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), union_type_copy_20968, 'UnionType')
                # Obtaining the member 'add' of a type (line 275)
                add_20970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), UnionType_20969, 'add')
                # Calling add(args, kwargs) (line 275)
                add_call_result_20979 = invoke(stypy.reporting.localization.Localization(__file__, 275, 38), add_20970, *[subscript_call_result_20974, UndefinedType_call_result_20977], **kwargs_20978)
                
                # Getting the type of 'type_dict' (line 275)
                type_dict_20980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'type_dict')
                # Getting the type of 'var_name' (line 275)
                var_name_20981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'var_name')
                # Storing an element on a container (line 275)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 16), type_dict_20980, (var_name_20981, add_call_result_20979))
                # SSA join for if statement (line 269)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 261)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'function_context_except' (line 277)
    function_context_except_20982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 'function_context_except')
    # Assigning a type to the variable 'function_context_except_20982' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'function_context_except_20982', function_context_except_20982)
    # Testing if the for loop is going to be iterated (line 277)
    # Testing the type of a for loop iterable (line 277)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 277, 4), function_context_except_20982)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 277, 4), function_context_except_20982):
        # Getting the type of the for loop variable (line 277)
        for_loop_var_20983 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 277, 4), function_context_except_20982)
        # Assigning a type to the variable 'var_name' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'var_name', for_loop_var_20983)
        # SSA begins for a for statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 278)
        var_name_20984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), 'var_name')
        # Getting the type of 'function_context_try' (line 278)
        function_context_try_20985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 23), 'function_context_try')
        # Applying the binary operator 'in' (line 278)
        result_contains_20986 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 11), 'in', var_name_20984, function_context_try_20985)
        
        # Testing if the type of an if condition is none (line 278)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 278, 8), result_contains_20986):
            
            # Getting the type of 'var_name' (line 282)
            var_name_20988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 282)
            function_context_previous_20989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 282)
            result_contains_20990 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 15), 'in', var_name_20988, function_context_previous_20989)
            
            # Testing if the type of an if condition is none (line 282)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 282, 12), result_contains_20990):
                
                # Assigning a Call to a Subscript (line 289):
                
                # Call to add(...): (line 289)
                # Processing the call arguments (line 289)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 289)
                var_name_21010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 92), 'var_name', False)
                # Getting the type of 'function_context_except' (line 289)
                function_context_except_21011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 68), 'function_context_except', False)
                # Obtaining the member '__getitem__' of a type (line 289)
                getitem___21012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 68), function_context_except_21011, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 289)
                subscript_call_result_21013 = invoke(stypy.reporting.localization.Localization(__file__, 289, 68), getitem___21012, var_name_21010)
                
                
                # Call to UndefinedType(...): (line 289)
                # Processing the call keyword arguments (line 289)
                kwargs_21015 = {}
                # Getting the type of 'UndefinedType' (line 289)
                UndefinedType_21014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 103), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 289)
                UndefinedType_call_result_21016 = invoke(stypy.reporting.localization.Localization(__file__, 289, 103), UndefinedType_21014, *[], **kwargs_21015)
                
                # Processing the call keyword arguments (line 289)
                kwargs_21017 = {}
                # Getting the type of 'union_type_copy' (line 289)
                union_type_copy_21007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 289)
                UnionType_21008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), union_type_copy_21007, 'UnionType')
                # Obtaining the member 'add' of a type (line 289)
                add_21009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), UnionType_21008, 'add')
                # Calling add(args, kwargs) (line 289)
                add_call_result_21018 = invoke(stypy.reporting.localization.Localization(__file__, 289, 38), add_21009, *[subscript_call_result_21013, UndefinedType_call_result_21016], **kwargs_21017)
                
                # Getting the type of 'type_dict' (line 289)
                type_dict_21019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'type_dict')
                # Getting the type of 'var_name' (line 289)
                var_name_21020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'var_name')
                # Storing an element on a container (line 289)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 16), type_dict_21019, (var_name_21020, add_call_result_21018))
            else:
                
                # Testing the type of an if condition (line 282)
                if_condition_20991 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 12), result_contains_20990)
                # Assigning a type to the variable 'if_condition_20991' (line 282)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'if_condition_20991', if_condition_20991)
                # SSA begins for if statement (line 282)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 284):
                
                # Call to add(...): (line 284)
                # Processing the call arguments (line 284)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 284)
                var_name_20995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 284)
                function_context_previous_20996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 284)
                getitem___20997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 68), function_context_previous_20996, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 284)
                subscript_call_result_20998 = invoke(stypy.reporting.localization.Localization(__file__, 284, 68), getitem___20997, var_name_20995)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 285)
                var_name_20999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 87), 'var_name', False)
                # Getting the type of 'function_context_except' (line 285)
                function_context_except_21000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 63), 'function_context_except', False)
                # Obtaining the member '__getitem__' of a type (line 285)
                getitem___21001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 63), function_context_except_21000, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 285)
                subscript_call_result_21002 = invoke(stypy.reporting.localization.Localization(__file__, 285, 63), getitem___21001, var_name_20999)
                
                # Processing the call keyword arguments (line 284)
                kwargs_21003 = {}
                # Getting the type of 'union_type_copy' (line 284)
                union_type_copy_20992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 284)
                UnionType_20993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 38), union_type_copy_20992, 'UnionType')
                # Obtaining the member 'add' of a type (line 284)
                add_20994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 38), UnionType_20993, 'add')
                # Calling add(args, kwargs) (line 284)
                add_call_result_21004 = invoke(stypy.reporting.localization.Localization(__file__, 284, 38), add_20994, *[subscript_call_result_20998, subscript_call_result_21002], **kwargs_21003)
                
                # Getting the type of 'type_dict' (line 284)
                type_dict_21005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'type_dict')
                # Getting the type of 'var_name' (line 284)
                var_name_21006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 26), 'var_name')
                # Storing an element on a container (line 284)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 16), type_dict_21005, (var_name_21006, add_call_result_21004))
                # SSA branch for the else part of an if statement (line 282)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 289):
                
                # Call to add(...): (line 289)
                # Processing the call arguments (line 289)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 289)
                var_name_21010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 92), 'var_name', False)
                # Getting the type of 'function_context_except' (line 289)
                function_context_except_21011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 68), 'function_context_except', False)
                # Obtaining the member '__getitem__' of a type (line 289)
                getitem___21012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 68), function_context_except_21011, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 289)
                subscript_call_result_21013 = invoke(stypy.reporting.localization.Localization(__file__, 289, 68), getitem___21012, var_name_21010)
                
                
                # Call to UndefinedType(...): (line 289)
                # Processing the call keyword arguments (line 289)
                kwargs_21015 = {}
                # Getting the type of 'UndefinedType' (line 289)
                UndefinedType_21014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 103), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 289)
                UndefinedType_call_result_21016 = invoke(stypy.reporting.localization.Localization(__file__, 289, 103), UndefinedType_21014, *[], **kwargs_21015)
                
                # Processing the call keyword arguments (line 289)
                kwargs_21017 = {}
                # Getting the type of 'union_type_copy' (line 289)
                union_type_copy_21007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 289)
                UnionType_21008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), union_type_copy_21007, 'UnionType')
                # Obtaining the member 'add' of a type (line 289)
                add_21009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), UnionType_21008, 'add')
                # Calling add(args, kwargs) (line 289)
                add_call_result_21018 = invoke(stypy.reporting.localization.Localization(__file__, 289, 38), add_21009, *[subscript_call_result_21013, UndefinedType_call_result_21016], **kwargs_21017)
                
                # Getting the type of 'type_dict' (line 289)
                type_dict_21019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'type_dict')
                # Getting the type of 'var_name' (line 289)
                var_name_21020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'var_name')
                # Storing an element on a container (line 289)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 16), type_dict_21019, (var_name_21020, add_call_result_21018))
                # SSA join for if statement (line 282)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 278)
            if_condition_20987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 8), result_contains_20986)
            # Assigning a type to the variable 'if_condition_20987' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'if_condition_20987', if_condition_20987)
            # SSA begins for if statement (line 278)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA branch for the else part of an if statement (line 278)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'var_name' (line 282)
            var_name_20988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 282)
            function_context_previous_20989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 282)
            result_contains_20990 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 15), 'in', var_name_20988, function_context_previous_20989)
            
            # Testing if the type of an if condition is none (line 282)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 282, 12), result_contains_20990):
                
                # Assigning a Call to a Subscript (line 289):
                
                # Call to add(...): (line 289)
                # Processing the call arguments (line 289)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 289)
                var_name_21010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 92), 'var_name', False)
                # Getting the type of 'function_context_except' (line 289)
                function_context_except_21011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 68), 'function_context_except', False)
                # Obtaining the member '__getitem__' of a type (line 289)
                getitem___21012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 68), function_context_except_21011, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 289)
                subscript_call_result_21013 = invoke(stypy.reporting.localization.Localization(__file__, 289, 68), getitem___21012, var_name_21010)
                
                
                # Call to UndefinedType(...): (line 289)
                # Processing the call keyword arguments (line 289)
                kwargs_21015 = {}
                # Getting the type of 'UndefinedType' (line 289)
                UndefinedType_21014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 103), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 289)
                UndefinedType_call_result_21016 = invoke(stypy.reporting.localization.Localization(__file__, 289, 103), UndefinedType_21014, *[], **kwargs_21015)
                
                # Processing the call keyword arguments (line 289)
                kwargs_21017 = {}
                # Getting the type of 'union_type_copy' (line 289)
                union_type_copy_21007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 289)
                UnionType_21008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), union_type_copy_21007, 'UnionType')
                # Obtaining the member 'add' of a type (line 289)
                add_21009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), UnionType_21008, 'add')
                # Calling add(args, kwargs) (line 289)
                add_call_result_21018 = invoke(stypy.reporting.localization.Localization(__file__, 289, 38), add_21009, *[subscript_call_result_21013, UndefinedType_call_result_21016], **kwargs_21017)
                
                # Getting the type of 'type_dict' (line 289)
                type_dict_21019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'type_dict')
                # Getting the type of 'var_name' (line 289)
                var_name_21020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'var_name')
                # Storing an element on a container (line 289)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 16), type_dict_21019, (var_name_21020, add_call_result_21018))
            else:
                
                # Testing the type of an if condition (line 282)
                if_condition_20991 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 12), result_contains_20990)
                # Assigning a type to the variable 'if_condition_20991' (line 282)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'if_condition_20991', if_condition_20991)
                # SSA begins for if statement (line 282)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 284):
                
                # Call to add(...): (line 284)
                # Processing the call arguments (line 284)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 284)
                var_name_20995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 284)
                function_context_previous_20996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 284)
                getitem___20997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 68), function_context_previous_20996, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 284)
                subscript_call_result_20998 = invoke(stypy.reporting.localization.Localization(__file__, 284, 68), getitem___20997, var_name_20995)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 285)
                var_name_20999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 87), 'var_name', False)
                # Getting the type of 'function_context_except' (line 285)
                function_context_except_21000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 63), 'function_context_except', False)
                # Obtaining the member '__getitem__' of a type (line 285)
                getitem___21001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 63), function_context_except_21000, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 285)
                subscript_call_result_21002 = invoke(stypy.reporting.localization.Localization(__file__, 285, 63), getitem___21001, var_name_20999)
                
                # Processing the call keyword arguments (line 284)
                kwargs_21003 = {}
                # Getting the type of 'union_type_copy' (line 284)
                union_type_copy_20992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 284)
                UnionType_20993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 38), union_type_copy_20992, 'UnionType')
                # Obtaining the member 'add' of a type (line 284)
                add_20994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 38), UnionType_20993, 'add')
                # Calling add(args, kwargs) (line 284)
                add_call_result_21004 = invoke(stypy.reporting.localization.Localization(__file__, 284, 38), add_20994, *[subscript_call_result_20998, subscript_call_result_21002], **kwargs_21003)
                
                # Getting the type of 'type_dict' (line 284)
                type_dict_21005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'type_dict')
                # Getting the type of 'var_name' (line 284)
                var_name_21006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 26), 'var_name')
                # Storing an element on a container (line 284)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 16), type_dict_21005, (var_name_21006, add_call_result_21004))
                # SSA branch for the else part of an if statement (line 282)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 289):
                
                # Call to add(...): (line 289)
                # Processing the call arguments (line 289)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 289)
                var_name_21010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 92), 'var_name', False)
                # Getting the type of 'function_context_except' (line 289)
                function_context_except_21011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 68), 'function_context_except', False)
                # Obtaining the member '__getitem__' of a type (line 289)
                getitem___21012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 68), function_context_except_21011, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 289)
                subscript_call_result_21013 = invoke(stypy.reporting.localization.Localization(__file__, 289, 68), getitem___21012, var_name_21010)
                
                
                # Call to UndefinedType(...): (line 289)
                # Processing the call keyword arguments (line 289)
                kwargs_21015 = {}
                # Getting the type of 'UndefinedType' (line 289)
                UndefinedType_21014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 103), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 289)
                UndefinedType_call_result_21016 = invoke(stypy.reporting.localization.Localization(__file__, 289, 103), UndefinedType_21014, *[], **kwargs_21015)
                
                # Processing the call keyword arguments (line 289)
                kwargs_21017 = {}
                # Getting the type of 'union_type_copy' (line 289)
                union_type_copy_21007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 289)
                UnionType_21008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), union_type_copy_21007, 'UnionType')
                # Obtaining the member 'add' of a type (line 289)
                add_21009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), UnionType_21008, 'add')
                # Calling add(args, kwargs) (line 289)
                add_call_result_21018 = invoke(stypy.reporting.localization.Localization(__file__, 289, 38), add_21009, *[subscript_call_result_21013, UndefinedType_call_result_21016], **kwargs_21017)
                
                # Getting the type of 'type_dict' (line 289)
                type_dict_21019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'type_dict')
                # Getting the type of 'var_name' (line 289)
                var_name_21020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'var_name')
                # Storing an element on a container (line 289)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 16), type_dict_21019, (var_name_21020, add_call_result_21018))
                # SSA join for if statement (line 282)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 278)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'type_dict' (line 292)
    type_dict_21021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 11), 'type_dict')
    # Assigning a type to the variable 'stypy_return_type' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type', type_dict_21021)
    
    # ################# End of '__join_try_except_function_context(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join_try_except_function_context' in the type store
    # Getting the type of 'stypy_return_type' (line 249)
    stypy_return_type_21022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21022)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join_try_except_function_context'
    return stypy_return_type_21022

# Assigning a type to the variable '__join_try_except_function_context' (line 249)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), '__join_try_except_function_context', __join_try_except_function_context)

@norecursion
def __join__try_except(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__join__try_except'
    module_type_store = module_type_store.open_function_context('__join__try_except', 295, 0, False)
    
    # Passed parameters checking function
    __join__try_except.stypy_localization = localization
    __join__try_except.stypy_type_of_self = None
    __join__try_except.stypy_type_store = module_type_store
    __join__try_except.stypy_function_name = '__join__try_except'
    __join__try_except.stypy_param_names_list = ['type_store_previous', 'type_store_posttry', 'type_store_excepts']
    __join__try_except.stypy_varargs_param_name = None
    __join__try_except.stypy_kwargs_param_name = None
    __join__try_except.stypy_call_defaults = defaults
    __join__try_except.stypy_call_varargs = varargs
    __join__try_except.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__join__try_except', ['type_store_previous', 'type_store_posttry', 'type_store_excepts'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__join__try_except', localization, ['type_store_previous', 'type_store_posttry', 'type_store_excepts'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__join__try_except(...)' code ##################

    str_21023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, (-1)), 'str', '\n    SSA Algotihm implementation for type stores in a try-except block\n    :param type_store_previous: Type store\n    :param type_store_posttry: Type store\n    :param type_store_excepts: Type store\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 303):
    
    # Call to TypeStore(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'type_store_previous' (line 303)
    type_store_previous_21025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 34), 'type_store_previous', False)
    # Obtaining the member 'program_name' of a type (line 303)
    program_name_21026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 34), type_store_previous_21025, 'program_name')
    # Processing the call keyword arguments (line 303)
    kwargs_21027 = {}
    # Getting the type of 'TypeStore' (line 303)
    TypeStore_21024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'TypeStore', False)
    # Calling TypeStore(args, kwargs) (line 303)
    TypeStore_call_result_21028 = invoke(stypy.reporting.localization.Localization(__file__, 303, 24), TypeStore_21024, *[program_name_21026], **kwargs_21027)
    
    # Assigning a type to the variable 'joined_type_store' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'joined_type_store', TypeStore_call_result_21028)
    
    # Assigning a Attribute to a Attribute (line 304):
    # Getting the type of 'type_store_previous' (line 304)
    type_store_previous_21029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 47), 'type_store_previous')
    # Obtaining the member 'last_function_contexts' of a type (line 304)
    last_function_contexts_21030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 47), type_store_previous_21029, 'last_function_contexts')
    # Getting the type of 'joined_type_store' (line 304)
    joined_type_store_21031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'joined_type_store')
    # Setting the type of the member 'last_function_contexts' of a type (line 304)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 4), joined_type_store_21031, 'last_function_contexts', last_function_contexts_21030)
    
    # Assigning a List to a Attribute (line 305):
    
    # Obtaining an instance of the builtin type 'list' (line 305)
    list_21032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 305)
    
    # Getting the type of 'joined_type_store' (line 305)
    joined_type_store_21033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'joined_type_store')
    # Setting the type of the member 'context_stack' of a type (line 305)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 4), joined_type_store_21033, 'context_stack', list_21032)
    
    
    # Call to range(...): (line 306)
    # Processing the call arguments (line 306)
    
    # Call to len(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'type_store_previous' (line 306)
    type_store_previous_21036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 23), 'type_store_previous', False)
    # Obtaining the member 'context_stack' of a type (line 306)
    context_stack_21037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 23), type_store_previous_21036, 'context_stack')
    # Processing the call keyword arguments (line 306)
    kwargs_21038 = {}
    # Getting the type of 'len' (line 306)
    len_21035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'len', False)
    # Calling len(args, kwargs) (line 306)
    len_call_result_21039 = invoke(stypy.reporting.localization.Localization(__file__, 306, 19), len_21035, *[context_stack_21037], **kwargs_21038)
    
    # Processing the call keyword arguments (line 306)
    kwargs_21040 = {}
    # Getting the type of 'range' (line 306)
    range_21034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 13), 'range', False)
    # Calling range(args, kwargs) (line 306)
    range_call_result_21041 = invoke(stypy.reporting.localization.Localization(__file__, 306, 13), range_21034, *[len_call_result_21039], **kwargs_21040)
    
    # Assigning a type to the variable 'range_call_result_21041' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'range_call_result_21041', range_call_result_21041)
    # Testing if the for loop is going to be iterated (line 306)
    # Testing the type of a for loop iterable (line 306)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 306, 4), range_call_result_21041)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 306, 4), range_call_result_21041):
        # Getting the type of the for loop variable (line 306)
        for_loop_var_21042 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 306, 4), range_call_result_21041)
        # Assigning a type to the variable 'i' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'i', for_loop_var_21042)
        # SSA begins for a for statement (line 306)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 307):
        
        # Call to __join_try_except_function_context(...): (line 307)
        # Processing the call arguments (line 307)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 307)
        i_21044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 85), 'i', False)
        # Getting the type of 'type_store_previous' (line 307)
        type_store_previous_21045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 65), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___21046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 65), type_store_previous_21045, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_21047 = invoke(stypy.reporting.localization.Localization(__file__, 307, 65), getitem___21046, i_21044)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 307)
        i_21048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 108), 'i', False)
        # Getting the type of 'type_store_posttry' (line 307)
        type_store_posttry_21049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 89), 'type_store_posttry', False)
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___21050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 89), type_store_posttry_21049, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_21051 = invoke(stypy.reporting.localization.Localization(__file__, 307, 89), getitem___21050, i_21048)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 308)
        i_21052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 84), 'i', False)
        # Getting the type of 'type_store_excepts' (line 308)
        type_store_excepts_21053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 65), 'type_store_excepts', False)
        # Obtaining the member '__getitem__' of a type (line 308)
        getitem___21054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 65), type_store_excepts_21053, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 308)
        subscript_call_result_21055 = invoke(stypy.reporting.localization.Localization(__file__, 308, 65), getitem___21054, i_21052)
        
        # Processing the call keyword arguments (line 307)
        kwargs_21056 = {}
        # Getting the type of '__join_try_except_function_context' (line 307)
        join_try_except_function_context_21043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 30), '__join_try_except_function_context', False)
        # Calling __join_try_except_function_context(args, kwargs) (line 307)
        join_try_except_function_context_call_result_21057 = invoke(stypy.reporting.localization.Localization(__file__, 307, 30), join_try_except_function_context_21043, *[subscript_call_result_21047, subscript_call_result_21051, subscript_call_result_21055], **kwargs_21056)
        
        # Assigning a type to the variable 'joined_context_dict' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'joined_context_dict', join_try_except_function_context_call_result_21057)
        
        # Assigning a Call to a Name (line 310):
        
        # Call to copy(...): (line 310)
        # Processing the call keyword arguments (line 310)
        kwargs_21063 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 310)
        i_21058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 47), 'i', False)
        # Getting the type of 'type_store_previous' (line 310)
        type_store_previous_21059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 27), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___21060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 27), type_store_previous_21059, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_21061 = invoke(stypy.reporting.localization.Localization(__file__, 310, 27), getitem___21060, i_21058)
        
        # Obtaining the member 'copy' of a type (line 310)
        copy_21062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 27), subscript_call_result_21061, 'copy')
        # Calling copy(args, kwargs) (line 310)
        copy_call_result_21064 = invoke(stypy.reporting.localization.Localization(__file__, 310, 27), copy_21062, *[], **kwargs_21063)
        
        # Assigning a type to the variable 'joined_f_context' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'joined_f_context', copy_call_result_21064)
        
        # Assigning a Name to a Attribute (line 311):
        # Getting the type of 'joined_context_dict' (line 311)
        joined_context_dict_21065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 36), 'joined_context_dict')
        # Getting the type of 'joined_f_context' (line 311)
        joined_f_context_21066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'joined_f_context')
        # Setting the type of the member 'types_of' of a type (line 311)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), joined_f_context_21066, 'types_of', joined_context_dict_21065)
        
        # Assigning a Call to a Attribute (line 312):
        
        # Call to __join_globals(...): (line 312)
        # Processing the call arguments (line 312)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 312)
        i_21068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 73), 'i', False)
        # Getting the type of 'type_store_posttry' (line 312)
        type_store_posttry_21069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 54), 'type_store_posttry', False)
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___21070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 54), type_store_posttry_21069, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_21071 = invoke(stypy.reporting.localization.Localization(__file__, 312, 54), getitem___21070, i_21068)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 312)
        i_21072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 96), 'i', False)
        # Getting the type of 'type_store_excepts' (line 312)
        type_store_excepts_21073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 77), 'type_store_excepts', False)
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___21074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 77), type_store_excepts_21073, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_21075 = invoke(stypy.reporting.localization.Localization(__file__, 312, 77), getitem___21074, i_21072)
        
        # Processing the call keyword arguments (line 312)
        kwargs_21076 = {}
        # Getting the type of '__join_globals' (line 312)
        join_globals_21067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 39), '__join_globals', False)
        # Calling __join_globals(args, kwargs) (line 312)
        join_globals_call_result_21077 = invoke(stypy.reporting.localization.Localization(__file__, 312, 39), join_globals_21067, *[subscript_call_result_21071, subscript_call_result_21075], **kwargs_21076)
        
        # Getting the type of 'joined_f_context' (line 312)
        joined_f_context_21078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'joined_f_context')
        # Setting the type of the member 'global_vars' of a type (line 312)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), joined_f_context_21078, 'global_vars', join_globals_call_result_21077)
        
        # Assigning a Attribute to a Attribute (line 313):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 314)
        i_21079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'i')
        # Getting the type of 'type_store_posttry' (line 313)
        type_store_posttry_21080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 45), 'type_store_posttry')
        # Obtaining the member '__getitem__' of a type (line 313)
        getitem___21081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 45), type_store_posttry_21080, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 313)
        subscript_call_result_21082 = invoke(stypy.reporting.localization.Localization(__file__, 313, 45), getitem___21081, i_21079)
        
        # Obtaining the member 'annotation_record' of a type (line 313)
        annotation_record_21083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 45), subscript_call_result_21082, 'annotation_record')
        # Getting the type of 'joined_f_context' (line 313)
        joined_f_context_21084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'joined_f_context')
        # Setting the type of the member 'annotation_record' of a type (line 313)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), joined_f_context_21084, 'annotation_record', annotation_record_21083)
        
        # Call to append(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'joined_f_context' (line 316)
        joined_f_context_21088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 47), 'joined_f_context', False)
        # Processing the call keyword arguments (line 316)
        kwargs_21089 = {}
        # Getting the type of 'joined_type_store' (line 316)
        joined_type_store_21085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'joined_type_store', False)
        # Obtaining the member 'context_stack' of a type (line 316)
        context_stack_21086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), joined_type_store_21085, 'context_stack')
        # Obtaining the member 'append' of a type (line 316)
        append_21087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), context_stack_21086, 'append')
        # Calling append(args, kwargs) (line 316)
        append_call_result_21090 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), append_21087, *[joined_f_context_21088], **kwargs_21089)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'joined_type_store' (line 318)
    joined_type_store_21091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 'joined_type_store')
    # Assigning a type to the variable 'stypy_return_type' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type', joined_type_store_21091)
    
    # ################# End of '__join__try_except(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join__try_except' in the type store
    # Getting the type of 'stypy_return_type' (line 295)
    stypy_return_type_21092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21092)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join__try_except'
    return stypy_return_type_21092

# Assigning a type to the variable '__join__try_except' (line 295)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), '__join__try_except', __join__try_except)

@norecursion
def join_exception_block(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 321)
    None_21093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 83), 'None')
    defaults = [None_21093]
    # Create a new context for function 'join_exception_block'
    module_type_store = module_type_store.open_function_context('join_exception_block', 321, 0, False)
    
    # Passed parameters checking function
    join_exception_block.stypy_localization = localization
    join_exception_block.stypy_type_of_self = None
    join_exception_block.stypy_type_store = module_type_store
    join_exception_block.stypy_function_name = 'join_exception_block'
    join_exception_block.stypy_param_names_list = ['type_store_pretry', 'type_store_posttry', 'type_store_finally']
    join_exception_block.stypy_varargs_param_name = 'type_store_except_branches'
    join_exception_block.stypy_kwargs_param_name = None
    join_exception_block.stypy_call_defaults = defaults
    join_exception_block.stypy_call_varargs = varargs
    join_exception_block.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'join_exception_block', ['type_store_pretry', 'type_store_posttry', 'type_store_finally'], 'type_store_except_branches', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'join_exception_block', localization, ['type_store_pretry', 'type_store_posttry', 'type_store_finally'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'join_exception_block(...)' code ##################

    str_21094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, (-1)), 'str', '\n    Implements the SSA algorithm for a full try-except-finally block, calling previous function\n    :param type_store_pretry: Type store\n    :param type_store_posttry: Type store\n    :param type_store_finally: Type store\n    :param type_store_except_branches: Type store\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 333):
    
    # Call to TypeStore(...): (line 333)
    # Processing the call arguments (line 333)
    # Getting the type of 'type_store_pretry' (line 333)
    type_store_pretry_21096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 34), 'type_store_pretry', False)
    # Obtaining the member 'program_name' of a type (line 333)
    program_name_21097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 34), type_store_pretry_21096, 'program_name')
    # Processing the call keyword arguments (line 333)
    kwargs_21098 = {}
    # Getting the type of 'TypeStore' (line 333)
    TypeStore_21095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 24), 'TypeStore', False)
    # Calling TypeStore(args, kwargs) (line 333)
    TypeStore_call_result_21099 = invoke(stypy.reporting.localization.Localization(__file__, 333, 24), TypeStore_21095, *[program_name_21097], **kwargs_21098)
    
    # Assigning a type to the variable 'joined_type_store' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'joined_type_store', TypeStore_call_result_21099)
    
    # Assigning a Attribute to a Attribute (line 334):
    # Getting the type of 'type_store_pretry' (line 334)
    type_store_pretry_21100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 47), 'type_store_pretry')
    # Obtaining the member 'last_function_contexts' of a type (line 334)
    last_function_contexts_21101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 47), type_store_pretry_21100, 'last_function_contexts')
    # Getting the type of 'joined_type_store' (line 334)
    joined_type_store_21102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'joined_type_store')
    # Setting the type of the member 'last_function_contexts' of a type (line 334)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 4), joined_type_store_21102, 'last_function_contexts', last_function_contexts_21101)
    
    # Assigning a List to a Attribute (line 335):
    
    # Obtaining an instance of the builtin type 'list' (line 335)
    list_21103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 335)
    
    # Getting the type of 'joined_type_store' (line 335)
    joined_type_store_21104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'joined_type_store')
    # Setting the type of the member 'context_stack' of a type (line 335)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 4), joined_type_store_21104, 'context_stack', list_21103)
    
    
    # Call to len(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'type_store_except_branches' (line 338)
    type_store_except_branches_21106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 11), 'type_store_except_branches', False)
    # Processing the call keyword arguments (line 338)
    kwargs_21107 = {}
    # Getting the type of 'len' (line 338)
    len_21105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 7), 'len', False)
    # Calling len(args, kwargs) (line 338)
    len_call_result_21108 = invoke(stypy.reporting.localization.Localization(__file__, 338, 7), len_21105, *[type_store_except_branches_21106], **kwargs_21107)
    
    int_21109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 42), 'int')
    # Applying the binary operator '==' (line 338)
    result_eq_21110 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 7), '==', len_call_result_21108, int_21109)
    
    # Testing if the type of an if condition is none (line 338)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 338, 4), result_eq_21110):
        
        # Assigning a Num to a Name (line 341):
        int_21116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 15), 'int')
        # Assigning a type to the variable 'cont' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'cont', int_21116)
        
        # Assigning a Subscript to a Name (line 342):
        
        # Obtaining the type of the subscript
        int_21117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 56), 'int')
        # Getting the type of 'type_store_except_branches' (line 342)
        type_store_except_branches_21118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 29), 'type_store_except_branches')
        # Obtaining the member '__getitem__' of a type (line 342)
        getitem___21119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 29), type_store_except_branches_21118, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 342)
        subscript_call_result_21120 = invoke(stypy.reporting.localization.Localization(__file__, 342, 29), getitem___21119, int_21117)
        
        # Assigning a type to the variable 'type_store_excepts' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'type_store_excepts', subscript_call_result_21120)
        
        
        # Getting the type of 'cont' (line 343)
        cont_21121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 14), 'cont')
        
        # Call to len(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'type_store_except_branches' (line 343)
        type_store_except_branches_21123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'type_store_except_branches', False)
        # Processing the call keyword arguments (line 343)
        kwargs_21124 = {}
        # Getting the type of 'len' (line 343)
        len_21122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 21), 'len', False)
        # Calling len(args, kwargs) (line 343)
        len_call_result_21125 = invoke(stypy.reporting.localization.Localization(__file__, 343, 21), len_21122, *[type_store_except_branches_21123], **kwargs_21124)
        
        # Applying the binary operator '<' (line 343)
        result_lt_21126 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 14), '<', cont_21121, len_call_result_21125)
        
        # Assigning a type to the variable 'result_lt_21126' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'result_lt_21126', result_lt_21126)
        # Testing if the while is going to be iterated (line 343)
        # Testing the type of an if condition (line 343)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 8), result_lt_21126)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 343, 8), result_lt_21126):
            # SSA begins for while statement (line 343)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Call to a Name (line 344):
            
            # Call to __join_except_branches(...): (line 344)
            # Processing the call arguments (line 344)
            # Getting the type of 'type_store_excepts' (line 344)
            type_store_excepts_21128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 56), 'type_store_excepts', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'cont' (line 344)
            cont_21129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 103), 'cont', False)
            # Getting the type of 'type_store_except_branches' (line 344)
            type_store_except_branches_21130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 76), 'type_store_except_branches', False)
            # Obtaining the member '__getitem__' of a type (line 344)
            getitem___21131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 76), type_store_except_branches_21130, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 344)
            subscript_call_result_21132 = invoke(stypy.reporting.localization.Localization(__file__, 344, 76), getitem___21131, cont_21129)
            
            # Processing the call keyword arguments (line 344)
            kwargs_21133 = {}
            # Getting the type of '__join_except_branches' (line 344)
            join_except_branches_21127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 33), '__join_except_branches', False)
            # Calling __join_except_branches(args, kwargs) (line 344)
            join_except_branches_call_result_21134 = invoke(stypy.reporting.localization.Localization(__file__, 344, 33), join_except_branches_21127, *[type_store_excepts_21128, subscript_call_result_21132], **kwargs_21133)
            
            # Assigning a type to the variable 'type_store_excepts' (line 344)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'type_store_excepts', join_except_branches_call_result_21134)
            
            # Getting the type of 'cont' (line 345)
            cont_21135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'cont')
            int_21136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 20), 'int')
            # Applying the binary operator '+=' (line 345)
            result_iadd_21137 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 12), '+=', cont_21135, int_21136)
            # Assigning a type to the variable 'cont' (line 345)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'cont', result_iadd_21137)
            
            # SSA join for while statement (line 343)
            module_type_store = module_type_store.join_ssa_context()

        
    else:
        
        # Testing the type of an if condition (line 338)
        if_condition_21111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 4), result_eq_21110)
        # Assigning a type to the variable 'if_condition_21111' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'if_condition_21111', if_condition_21111)
        # SSA begins for if statement (line 338)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 339):
        
        # Obtaining the type of the subscript
        int_21112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 56), 'int')
        # Getting the type of 'type_store_except_branches' (line 339)
        type_store_except_branches_21113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 29), 'type_store_except_branches')
        # Obtaining the member '__getitem__' of a type (line 339)
        getitem___21114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 29), type_store_except_branches_21113, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 339)
        subscript_call_result_21115 = invoke(stypy.reporting.localization.Localization(__file__, 339, 29), getitem___21114, int_21112)
        
        # Assigning a type to the variable 'type_store_excepts' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'type_store_excepts', subscript_call_result_21115)
        # SSA branch for the else part of an if statement (line 338)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 341):
        int_21116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 15), 'int')
        # Assigning a type to the variable 'cont' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'cont', int_21116)
        
        # Assigning a Subscript to a Name (line 342):
        
        # Obtaining the type of the subscript
        int_21117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 56), 'int')
        # Getting the type of 'type_store_except_branches' (line 342)
        type_store_except_branches_21118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 29), 'type_store_except_branches')
        # Obtaining the member '__getitem__' of a type (line 342)
        getitem___21119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 29), type_store_except_branches_21118, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 342)
        subscript_call_result_21120 = invoke(stypy.reporting.localization.Localization(__file__, 342, 29), getitem___21119, int_21117)
        
        # Assigning a type to the variable 'type_store_excepts' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'type_store_excepts', subscript_call_result_21120)
        
        
        # Getting the type of 'cont' (line 343)
        cont_21121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 14), 'cont')
        
        # Call to len(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'type_store_except_branches' (line 343)
        type_store_except_branches_21123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'type_store_except_branches', False)
        # Processing the call keyword arguments (line 343)
        kwargs_21124 = {}
        # Getting the type of 'len' (line 343)
        len_21122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 21), 'len', False)
        # Calling len(args, kwargs) (line 343)
        len_call_result_21125 = invoke(stypy.reporting.localization.Localization(__file__, 343, 21), len_21122, *[type_store_except_branches_21123], **kwargs_21124)
        
        # Applying the binary operator '<' (line 343)
        result_lt_21126 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 14), '<', cont_21121, len_call_result_21125)
        
        # Assigning a type to the variable 'result_lt_21126' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'result_lt_21126', result_lt_21126)
        # Testing if the while is going to be iterated (line 343)
        # Testing the type of an if condition (line 343)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 8), result_lt_21126)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 343, 8), result_lt_21126):
            # SSA begins for while statement (line 343)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Call to a Name (line 344):
            
            # Call to __join_except_branches(...): (line 344)
            # Processing the call arguments (line 344)
            # Getting the type of 'type_store_excepts' (line 344)
            type_store_excepts_21128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 56), 'type_store_excepts', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'cont' (line 344)
            cont_21129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 103), 'cont', False)
            # Getting the type of 'type_store_except_branches' (line 344)
            type_store_except_branches_21130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 76), 'type_store_except_branches', False)
            # Obtaining the member '__getitem__' of a type (line 344)
            getitem___21131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 76), type_store_except_branches_21130, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 344)
            subscript_call_result_21132 = invoke(stypy.reporting.localization.Localization(__file__, 344, 76), getitem___21131, cont_21129)
            
            # Processing the call keyword arguments (line 344)
            kwargs_21133 = {}
            # Getting the type of '__join_except_branches' (line 344)
            join_except_branches_21127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 33), '__join_except_branches', False)
            # Calling __join_except_branches(args, kwargs) (line 344)
            join_except_branches_call_result_21134 = invoke(stypy.reporting.localization.Localization(__file__, 344, 33), join_except_branches_21127, *[type_store_excepts_21128, subscript_call_result_21132], **kwargs_21133)
            
            # Assigning a type to the variable 'type_store_excepts' (line 344)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'type_store_excepts', join_except_branches_call_result_21134)
            
            # Getting the type of 'cont' (line 345)
            cont_21135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'cont')
            int_21136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 20), 'int')
            # Applying the binary operator '+=' (line 345)
            result_iadd_21137 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 12), '+=', cont_21135, int_21136)
            # Assigning a type to the variable 'cont' (line 345)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'cont', result_iadd_21137)
            
            # SSA join for while statement (line 343)
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 338)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 348):
    
    # Call to __join__try_except(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'type_store_pretry' (line 348)
    type_store_pretry_21139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 45), 'type_store_pretry', False)
    # Getting the type of 'type_store_posttry' (line 348)
    type_store_posttry_21140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 64), 'type_store_posttry', False)
    # Getting the type of 'type_store_excepts' (line 349)
    type_store_excepts_21141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 45), 'type_store_excepts', False)
    # Processing the call keyword arguments (line 348)
    kwargs_21142 = {}
    # Getting the type of '__join__try_except' (line 348)
    join__try_except_21138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 26), '__join__try_except', False)
    # Calling __join__try_except(args, kwargs) (line 348)
    join__try_except_call_result_21143 = invoke(stypy.reporting.localization.Localization(__file__, 348, 26), join__try_except_21138, *[type_store_pretry_21139, type_store_posttry_21140, type_store_excepts_21141], **kwargs_21142)
    
    # Assigning a type to the variable 'joined_context_dict' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'joined_context_dict', join__try_except_call_result_21143)
    
    # Type idiom detected: calculating its left and rigth part (line 352)
    # Getting the type of 'type_store_finally' (line 352)
    type_store_finally_21144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'type_store_finally')
    # Getting the type of 'None' (line 352)
    None_21145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 33), 'None')
    
    (may_be_21146, more_types_in_union_21147) = may_not_be_none(type_store_finally_21144, None_21145)

    if may_be_21146:

        if more_types_in_union_21147:
            # Runtime conditional SSA (line 352)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 353):
        
        # Call to __join_finally_branch(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'joined_context_dict' (line 353)
        joined_context_dict_21149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 52), 'joined_context_dict', False)
        # Getting the type of 'type_store_finally' (line 353)
        type_store_finally_21150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 73), 'type_store_finally', False)
        # Processing the call keyword arguments (line 353)
        kwargs_21151 = {}
        # Getting the type of '__join_finally_branch' (line 353)
        join_finally_branch_21148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 30), '__join_finally_branch', False)
        # Calling __join_finally_branch(args, kwargs) (line 353)
        join_finally_branch_call_result_21152 = invoke(stypy.reporting.localization.Localization(__file__, 353, 30), join_finally_branch_21148, *[joined_context_dict_21149, type_store_finally_21150], **kwargs_21151)
        
        # Assigning a type to the variable 'joined_context_dict' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'joined_context_dict', join_finally_branch_call_result_21152)

        if more_types_in_union_21147:
            # SSA join for if statement (line 352)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'joined_context_dict' (line 355)
    joined_context_dict_21153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 11), 'joined_context_dict')
    # Assigning a type to the variable 'stypy_return_type' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'stypy_return_type', joined_context_dict_21153)
    
    # ################# End of 'join_exception_block(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'join_exception_block' in the type store
    # Getting the type of 'stypy_return_type' (line 321)
    stypy_return_type_21154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21154)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'join_exception_block'
    return stypy_return_type_21154

# Assigning a type to the variable 'join_exception_block' (line 321)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), 'join_exception_block', join_exception_block)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
