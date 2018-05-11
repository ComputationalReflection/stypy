
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from ..type_store_copy.typestore_copy import TypeStore
2: from ..python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
3: from ..python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
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

# 'from testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.typestore_copy import TypeStore' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/ssa_copy/')
import_16501 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.typestore_copy')

if (type(import_16501) is not StypyTypeError):

    if (import_16501 != 'pyd_module'):
        __import__(import_16501)
        sys_modules_16502 = sys.modules[import_16501]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.typestore_copy', sys_modules_16502.module_type_store, module_type_store, ['TypeStore'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_16502, sys_modules_16502.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.typestore_copy import TypeStore

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.typestore_copy', None, module_type_store, ['TypeStore'], [TypeStore])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.typestore_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.type_store_copy.typestore_copy', import_16501)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/ssa_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/ssa_copy/')
import_16503 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_16503) is not StypyTypeError):

    if (import_16503 != 'pyd_module'):
        __import__(import_16503)
        sys_modules_16504 = sys.modules[import_16503]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_16504.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_16504, sys_modules_16504.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_16503)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/ssa_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/ssa_copy/')
import_16505 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy')

if (type(import_16505) is not StypyTypeError):

    if (import_16505 != 'pyd_module'):
        __import__(import_16505)
        sys_modules_16506 = sys.modules[import_16505]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', sys_modules_16506.module_type_store, module_type_store, ['UndefinedType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_16506, sys_modules_16506.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', None, module_type_store, ['UndefinedType'], [UndefinedType])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', import_16505)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/ssa_copy/')

str_16507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', '\nImplementation of the SSA algorithm to calculate types of variables when dealing with branches in source code (ifs,\nloops, ...)\n')

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

    str_16508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'str', '\n    Join the global variables placed in two function contexts\n    :param function_context_if: Function context\n    :param function_context_else: Function context\n    :return: The first function context with the globals of both of them\n    ')
    
    # Assigning a Attribute to a Name (line 42):
    # Getting the type of 'function_context_if' (line 42)
    function_context_if_16509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'function_context_if')
    # Obtaining the member 'global_vars' of a type (line 42)
    global_vars_16510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 17), function_context_if_16509, 'global_vars')
    # Assigning a type to the variable 'if_globals' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'if_globals', global_vars_16510)
    
    # Type idiom detected: calculating its left and rigth part (line 43)
    # Getting the type of 'function_context_else' (line 43)
    function_context_else_16511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 7), 'function_context_else')
    # Getting the type of 'None' (line 43)
    None_16512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'None')
    
    (may_be_16513, more_types_in_union_16514) = may_be_none(function_context_else_16511, None_16512)

    if may_be_16513:

        if more_types_in_union_16514:
            # Runtime conditional SSA (line 43)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Name (line 44):
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_16515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        
        # Assigning a type to the variable 'else_globals' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'else_globals', list_16515)

        if more_types_in_union_16514:
            # Runtime conditional SSA for else branch (line 43)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_16513) or more_types_in_union_16514):
        
        # Assigning a Attribute to a Name (line 46):
        # Getting the type of 'function_context_else' (line 46)
        function_context_else_16516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'function_context_else')
        # Obtaining the member 'global_vars' of a type (line 46)
        global_vars_16517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 23), function_context_else_16516, 'global_vars')
        # Assigning a type to the variable 'else_globals' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'else_globals', global_vars_16517)

        if (may_be_16513 and more_types_in_union_16514):
            # SSA join for if statement (line 43)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'else_globals' (line 48)
    else_globals_16518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'else_globals')
    # Assigning a type to the variable 'else_globals_16518' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'else_globals_16518', else_globals_16518)
    # Testing if the for loop is going to be iterated (line 48)
    # Testing the type of a for loop iterable (line 48)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 48, 4), else_globals_16518)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 48, 4), else_globals_16518):
        # Getting the type of the for loop variable (line 48)
        for_loop_var_16519 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 48, 4), else_globals_16518)
        # Assigning a type to the variable 'var' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'var', for_loop_var_16519)
        # SSA begins for a for statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var' (line 49)
        var_16520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'var')
        # Getting the type of 'if_globals' (line 49)
        if_globals_16521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'if_globals')
        # Applying the binary operator 'notin' (line 49)
        result_contains_16522 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 11), 'notin', var_16520, if_globals_16521)
        
        # Testing if the type of an if condition is none (line 49)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 49, 8), result_contains_16522):
            pass
        else:
            
            # Testing the type of an if condition (line 49)
            if_condition_16523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 8), result_contains_16522)
            # Assigning a type to the variable 'if_condition_16523' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'if_condition_16523', if_condition_16523)
            # SSA begins for if statement (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 50)
            # Processing the call arguments (line 50)
            # Getting the type of 'var' (line 50)
            var_16526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 30), 'var', False)
            # Processing the call keyword arguments (line 50)
            kwargs_16527 = {}
            # Getting the type of 'if_globals' (line 50)
            if_globals_16524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'if_globals', False)
            # Obtaining the member 'append' of a type (line 50)
            append_16525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), if_globals_16524, 'append')
            # Calling append(args, kwargs) (line 50)
            append_call_result_16528 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), append_16525, *[var_16526], **kwargs_16527)
            
            # SSA join for if statement (line 49)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'if_globals' (line 52)
    if_globals_16529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'if_globals')
    # Assigning a type to the variable 'stypy_return_type' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type', if_globals_16529)
    
    # ################# End of '__join_globals(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join_globals' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_16530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16530)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join_globals'
    return stypy_return_type_16530

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

    str_16531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, (-1)), 'str', '\n    Helper function of the SSA implementation of an if-else structure, used with each function context in the type\n    store\n    :param function_context_previous: Function context\n    :param function_context_if: Function context\n    :param function_context_else: Function context\n    :return: A dictionary with names of variables and its joined types\n    ')
    
    # Assigning a Dict to a Name (line 64):
    
    # Obtaining an instance of the builtin type 'dict' (line 64)
    dict_16532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 64)
    
    # Assigning a type to the variable 'type_dict' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'type_dict', dict_16532)
    
    # Type idiom detected: calculating its left and rigth part (line 66)
    # Getting the type of 'function_context_else' (line 66)
    function_context_else_16533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'function_context_else')
    # Getting the type of 'None' (line 66)
    None_16534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'None')
    
    (may_be_16535, more_types_in_union_16536) = may_be_none(function_context_else_16533, None_16534)

    if may_be_16535:

        if more_types_in_union_16536:
            # Runtime conditional SSA (line 66)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Name (line 67):
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_16537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        
        # Assigning a type to the variable 'function_context_else' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'function_context_else', list_16537)

        if more_types_in_union_16536:
            # SSA join for if statement (line 66)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'function_context_if' (line 69)
    function_context_if_16538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'function_context_if')
    # Assigning a type to the variable 'function_context_if_16538' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'function_context_if_16538', function_context_if_16538)
    # Testing if the for loop is going to be iterated (line 69)
    # Testing the type of a for loop iterable (line 69)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 69, 4), function_context_if_16538)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 69, 4), function_context_if_16538):
        # Getting the type of the for loop variable (line 69)
        for_loop_var_16539 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 69, 4), function_context_if_16538)
        # Assigning a type to the variable 'var_name' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'var_name', for_loop_var_16539)
        # SSA begins for a for statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 70)
        var_name_16540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'var_name')
        # Getting the type of 'function_context_else' (line 70)
        function_context_else_16541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'function_context_else')
        # Applying the binary operator 'in' (line 70)
        result_contains_16542 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 11), 'in', var_name_16540, function_context_else_16541)
        
        # Testing if the type of an if condition is none (line 70)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 8), result_contains_16542):
            
            # Getting the type of 'var_name' (line 76)
            var_name_16559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 76)
            function_context_previous_16560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 76)
            result_contains_16561 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 15), 'in', var_name_16559, function_context_previous_16560)
            
            # Testing if the type of an if condition is none (line 76)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 12), result_contains_16561):
                
                # Assigning a Call to a Subscript (line 82):
                
                # Call to add(...): (line 82)
                # Processing the call arguments (line 82)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 82)
                var_name_16581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 88), 'var_name', False)
                # Getting the type of 'function_context_if' (line 82)
                function_context_if_16582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 68), 'function_context_if', False)
                # Obtaining the member '__getitem__' of a type (line 82)
                getitem___16583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 68), function_context_if_16582, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 82)
                subscript_call_result_16584 = invoke(stypy.reporting.localization.Localization(__file__, 82, 68), getitem___16583, var_name_16581)
                
                
                # Call to UndefinedType(...): (line 82)
                # Processing the call keyword arguments (line 82)
                kwargs_16586 = {}
                # Getting the type of 'UndefinedType' (line 82)
                UndefinedType_16585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 99), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 82)
                UndefinedType_call_result_16587 = invoke(stypy.reporting.localization.Localization(__file__, 82, 99), UndefinedType_16585, *[], **kwargs_16586)
                
                # Processing the call keyword arguments (line 82)
                kwargs_16588 = {}
                # Getting the type of 'union_type_copy' (line 82)
                union_type_copy_16578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 82)
                UnionType_16579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), union_type_copy_16578, 'UnionType')
                # Obtaining the member 'add' of a type (line 82)
                add_16580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), UnionType_16579, 'add')
                # Calling add(args, kwargs) (line 82)
                add_call_result_16589 = invoke(stypy.reporting.localization.Localization(__file__, 82, 38), add_16580, *[subscript_call_result_16584, UndefinedType_call_result_16587], **kwargs_16588)
                
                # Getting the type of 'type_dict' (line 82)
                type_dict_16590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'type_dict')
                # Getting the type of 'var_name' (line 82)
                var_name_16591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'var_name')
                # Storing an element on a container (line 82)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 16), type_dict_16590, (var_name_16591, add_call_result_16589))
            else:
                
                # Testing the type of an if condition (line 76)
                if_condition_16562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 12), result_contains_16561)
                # Assigning a type to the variable 'if_condition_16562' (line 76)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'if_condition_16562', if_condition_16562)
                # SSA begins for if statement (line 76)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 78):
                
                # Call to add(...): (line 78)
                # Processing the call arguments (line 78)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 78)
                var_name_16566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 78)
                function_context_previous_16567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 78)
                getitem___16568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 68), function_context_previous_16567, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                subscript_call_result_16569 = invoke(stypy.reporting.localization.Localization(__file__, 78, 68), getitem___16568, var_name_16566)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 79)
                var_name_16570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 83), 'var_name', False)
                # Getting the type of 'function_context_if' (line 79)
                function_context_if_16571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 63), 'function_context_if', False)
                # Obtaining the member '__getitem__' of a type (line 79)
                getitem___16572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 63), function_context_if_16571, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 79)
                subscript_call_result_16573 = invoke(stypy.reporting.localization.Localization(__file__, 79, 63), getitem___16572, var_name_16570)
                
                # Processing the call keyword arguments (line 78)
                kwargs_16574 = {}
                # Getting the type of 'union_type_copy' (line 78)
                union_type_copy_16563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 78)
                UnionType_16564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 38), union_type_copy_16563, 'UnionType')
                # Obtaining the member 'add' of a type (line 78)
                add_16565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 38), UnionType_16564, 'add')
                # Calling add(args, kwargs) (line 78)
                add_call_result_16575 = invoke(stypy.reporting.localization.Localization(__file__, 78, 38), add_16565, *[subscript_call_result_16569, subscript_call_result_16573], **kwargs_16574)
                
                # Getting the type of 'type_dict' (line 78)
                type_dict_16576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'type_dict')
                # Getting the type of 'var_name' (line 78)
                var_name_16577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'var_name')
                # Storing an element on a container (line 78)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 16), type_dict_16576, (var_name_16577, add_call_result_16575))
                # SSA branch for the else part of an if statement (line 76)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 82):
                
                # Call to add(...): (line 82)
                # Processing the call arguments (line 82)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 82)
                var_name_16581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 88), 'var_name', False)
                # Getting the type of 'function_context_if' (line 82)
                function_context_if_16582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 68), 'function_context_if', False)
                # Obtaining the member '__getitem__' of a type (line 82)
                getitem___16583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 68), function_context_if_16582, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 82)
                subscript_call_result_16584 = invoke(stypy.reporting.localization.Localization(__file__, 82, 68), getitem___16583, var_name_16581)
                
                
                # Call to UndefinedType(...): (line 82)
                # Processing the call keyword arguments (line 82)
                kwargs_16586 = {}
                # Getting the type of 'UndefinedType' (line 82)
                UndefinedType_16585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 99), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 82)
                UndefinedType_call_result_16587 = invoke(stypy.reporting.localization.Localization(__file__, 82, 99), UndefinedType_16585, *[], **kwargs_16586)
                
                # Processing the call keyword arguments (line 82)
                kwargs_16588 = {}
                # Getting the type of 'union_type_copy' (line 82)
                union_type_copy_16578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 82)
                UnionType_16579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), union_type_copy_16578, 'UnionType')
                # Obtaining the member 'add' of a type (line 82)
                add_16580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), UnionType_16579, 'add')
                # Calling add(args, kwargs) (line 82)
                add_call_result_16589 = invoke(stypy.reporting.localization.Localization(__file__, 82, 38), add_16580, *[subscript_call_result_16584, UndefinedType_call_result_16587], **kwargs_16588)
                
                # Getting the type of 'type_dict' (line 82)
                type_dict_16590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'type_dict')
                # Getting the type of 'var_name' (line 82)
                var_name_16591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'var_name')
                # Storing an element on a container (line 82)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 16), type_dict_16590, (var_name_16591, add_call_result_16589))
                # SSA join for if statement (line 76)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 70)
            if_condition_16543 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 8), result_contains_16542)
            # Assigning a type to the variable 'if_condition_16543' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'if_condition_16543', if_condition_16543)
            # SSA begins for if statement (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 72):
            
            # Call to add(...): (line 72)
            # Processing the call arguments (line 72)
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 72)
            var_name_16547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 84), 'var_name', False)
            # Getting the type of 'function_context_if' (line 72)
            function_context_if_16548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 64), 'function_context_if', False)
            # Obtaining the member '__getitem__' of a type (line 72)
            getitem___16549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 64), function_context_if_16548, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 72)
            subscript_call_result_16550 = invoke(stypy.reporting.localization.Localization(__file__, 72, 64), getitem___16549, var_name_16547)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 73)
            var_name_16551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 81), 'var_name', False)
            # Getting the type of 'function_context_else' (line 73)
            function_context_else_16552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 59), 'function_context_else', False)
            # Obtaining the member '__getitem__' of a type (line 73)
            getitem___16553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 59), function_context_else_16552, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 73)
            subscript_call_result_16554 = invoke(stypy.reporting.localization.Localization(__file__, 73, 59), getitem___16553, var_name_16551)
            
            # Processing the call keyword arguments (line 72)
            kwargs_16555 = {}
            # Getting the type of 'union_type_copy' (line 72)
            union_type_copy_16544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 34), 'union_type_copy', False)
            # Obtaining the member 'UnionType' of a type (line 72)
            UnionType_16545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 34), union_type_copy_16544, 'UnionType')
            # Obtaining the member 'add' of a type (line 72)
            add_16546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 34), UnionType_16545, 'add')
            # Calling add(args, kwargs) (line 72)
            add_call_result_16556 = invoke(stypy.reporting.localization.Localization(__file__, 72, 34), add_16546, *[subscript_call_result_16550, subscript_call_result_16554], **kwargs_16555)
            
            # Getting the type of 'type_dict' (line 72)
            type_dict_16557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'type_dict')
            # Getting the type of 'var_name' (line 72)
            var_name_16558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'var_name')
            # Storing an element on a container (line 72)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 12), type_dict_16557, (var_name_16558, add_call_result_16556))
            # SSA branch for the else part of an if statement (line 70)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'var_name' (line 76)
            var_name_16559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 76)
            function_context_previous_16560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 76)
            result_contains_16561 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 15), 'in', var_name_16559, function_context_previous_16560)
            
            # Testing if the type of an if condition is none (line 76)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 12), result_contains_16561):
                
                # Assigning a Call to a Subscript (line 82):
                
                # Call to add(...): (line 82)
                # Processing the call arguments (line 82)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 82)
                var_name_16581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 88), 'var_name', False)
                # Getting the type of 'function_context_if' (line 82)
                function_context_if_16582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 68), 'function_context_if', False)
                # Obtaining the member '__getitem__' of a type (line 82)
                getitem___16583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 68), function_context_if_16582, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 82)
                subscript_call_result_16584 = invoke(stypy.reporting.localization.Localization(__file__, 82, 68), getitem___16583, var_name_16581)
                
                
                # Call to UndefinedType(...): (line 82)
                # Processing the call keyword arguments (line 82)
                kwargs_16586 = {}
                # Getting the type of 'UndefinedType' (line 82)
                UndefinedType_16585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 99), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 82)
                UndefinedType_call_result_16587 = invoke(stypy.reporting.localization.Localization(__file__, 82, 99), UndefinedType_16585, *[], **kwargs_16586)
                
                # Processing the call keyword arguments (line 82)
                kwargs_16588 = {}
                # Getting the type of 'union_type_copy' (line 82)
                union_type_copy_16578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 82)
                UnionType_16579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), union_type_copy_16578, 'UnionType')
                # Obtaining the member 'add' of a type (line 82)
                add_16580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), UnionType_16579, 'add')
                # Calling add(args, kwargs) (line 82)
                add_call_result_16589 = invoke(stypy.reporting.localization.Localization(__file__, 82, 38), add_16580, *[subscript_call_result_16584, UndefinedType_call_result_16587], **kwargs_16588)
                
                # Getting the type of 'type_dict' (line 82)
                type_dict_16590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'type_dict')
                # Getting the type of 'var_name' (line 82)
                var_name_16591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'var_name')
                # Storing an element on a container (line 82)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 16), type_dict_16590, (var_name_16591, add_call_result_16589))
            else:
                
                # Testing the type of an if condition (line 76)
                if_condition_16562 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 12), result_contains_16561)
                # Assigning a type to the variable 'if_condition_16562' (line 76)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'if_condition_16562', if_condition_16562)
                # SSA begins for if statement (line 76)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 78):
                
                # Call to add(...): (line 78)
                # Processing the call arguments (line 78)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 78)
                var_name_16566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 78)
                function_context_previous_16567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 78)
                getitem___16568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 68), function_context_previous_16567, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                subscript_call_result_16569 = invoke(stypy.reporting.localization.Localization(__file__, 78, 68), getitem___16568, var_name_16566)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 79)
                var_name_16570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 83), 'var_name', False)
                # Getting the type of 'function_context_if' (line 79)
                function_context_if_16571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 63), 'function_context_if', False)
                # Obtaining the member '__getitem__' of a type (line 79)
                getitem___16572 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 63), function_context_if_16571, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 79)
                subscript_call_result_16573 = invoke(stypy.reporting.localization.Localization(__file__, 79, 63), getitem___16572, var_name_16570)
                
                # Processing the call keyword arguments (line 78)
                kwargs_16574 = {}
                # Getting the type of 'union_type_copy' (line 78)
                union_type_copy_16563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 78)
                UnionType_16564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 38), union_type_copy_16563, 'UnionType')
                # Obtaining the member 'add' of a type (line 78)
                add_16565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 38), UnionType_16564, 'add')
                # Calling add(args, kwargs) (line 78)
                add_call_result_16575 = invoke(stypy.reporting.localization.Localization(__file__, 78, 38), add_16565, *[subscript_call_result_16569, subscript_call_result_16573], **kwargs_16574)
                
                # Getting the type of 'type_dict' (line 78)
                type_dict_16576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'type_dict')
                # Getting the type of 'var_name' (line 78)
                var_name_16577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'var_name')
                # Storing an element on a container (line 78)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 16), type_dict_16576, (var_name_16577, add_call_result_16575))
                # SSA branch for the else part of an if statement (line 76)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 82):
                
                # Call to add(...): (line 82)
                # Processing the call arguments (line 82)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 82)
                var_name_16581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 88), 'var_name', False)
                # Getting the type of 'function_context_if' (line 82)
                function_context_if_16582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 68), 'function_context_if', False)
                # Obtaining the member '__getitem__' of a type (line 82)
                getitem___16583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 68), function_context_if_16582, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 82)
                subscript_call_result_16584 = invoke(stypy.reporting.localization.Localization(__file__, 82, 68), getitem___16583, var_name_16581)
                
                
                # Call to UndefinedType(...): (line 82)
                # Processing the call keyword arguments (line 82)
                kwargs_16586 = {}
                # Getting the type of 'UndefinedType' (line 82)
                UndefinedType_16585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 99), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 82)
                UndefinedType_call_result_16587 = invoke(stypy.reporting.localization.Localization(__file__, 82, 99), UndefinedType_16585, *[], **kwargs_16586)
                
                # Processing the call keyword arguments (line 82)
                kwargs_16588 = {}
                # Getting the type of 'union_type_copy' (line 82)
                union_type_copy_16578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 82)
                UnionType_16579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), union_type_copy_16578, 'UnionType')
                # Obtaining the member 'add' of a type (line 82)
                add_16580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 38), UnionType_16579, 'add')
                # Calling add(args, kwargs) (line 82)
                add_call_result_16589 = invoke(stypy.reporting.localization.Localization(__file__, 82, 38), add_16580, *[subscript_call_result_16584, UndefinedType_call_result_16587], **kwargs_16588)
                
                # Getting the type of 'type_dict' (line 82)
                type_dict_16590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'type_dict')
                # Getting the type of 'var_name' (line 82)
                var_name_16591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'var_name')
                # Storing an element on a container (line 82)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 16), type_dict_16590, (var_name_16591, add_call_result_16589))
                # SSA join for if statement (line 76)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 70)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'function_context_else' (line 84)
    function_context_else_16592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'function_context_else')
    # Assigning a type to the variable 'function_context_else_16592' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'function_context_else_16592', function_context_else_16592)
    # Testing if the for loop is going to be iterated (line 84)
    # Testing the type of a for loop iterable (line 84)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 4), function_context_else_16592)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 4), function_context_else_16592):
        # Getting the type of the for loop variable (line 84)
        for_loop_var_16593 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 4), function_context_else_16592)
        # Assigning a type to the variable 'var_name' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'var_name', for_loop_var_16593)
        # SSA begins for a for statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 85)
        var_name_16594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'var_name')
        # Getting the type of 'function_context_if' (line 85)
        function_context_if_16595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'function_context_if')
        # Applying the binary operator 'in' (line 85)
        result_contains_16596 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 11), 'in', var_name_16594, function_context_if_16595)
        
        # Testing if the type of an if condition is none (line 85)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 8), result_contains_16596):
            
            # Getting the type of 'var_name' (line 89)
            var_name_16598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 89)
            function_context_previous_16599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 89)
            result_contains_16600 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 15), 'in', var_name_16598, function_context_previous_16599)
            
            # Testing if the type of an if condition is none (line 89)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 12), result_contains_16600):
                
                # Assigning a Call to a Subscript (line 96):
                
                # Call to add(...): (line 96)
                # Processing the call arguments (line 96)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 96)
                var_name_16620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 90), 'var_name', False)
                # Getting the type of 'function_context_else' (line 96)
                function_context_else_16621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 68), 'function_context_else', False)
                # Obtaining the member '__getitem__' of a type (line 96)
                getitem___16622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 68), function_context_else_16621, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 96)
                subscript_call_result_16623 = invoke(stypy.reporting.localization.Localization(__file__, 96, 68), getitem___16622, var_name_16620)
                
                
                # Call to UndefinedType(...): (line 96)
                # Processing the call keyword arguments (line 96)
                kwargs_16625 = {}
                # Getting the type of 'UndefinedType' (line 96)
                UndefinedType_16624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 101), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 96)
                UndefinedType_call_result_16626 = invoke(stypy.reporting.localization.Localization(__file__, 96, 101), UndefinedType_16624, *[], **kwargs_16625)
                
                # Processing the call keyword arguments (line 96)
                kwargs_16627 = {}
                # Getting the type of 'union_type_copy' (line 96)
                union_type_copy_16617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 96)
                UnionType_16618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), union_type_copy_16617, 'UnionType')
                # Obtaining the member 'add' of a type (line 96)
                add_16619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), UnionType_16618, 'add')
                # Calling add(args, kwargs) (line 96)
                add_call_result_16628 = invoke(stypy.reporting.localization.Localization(__file__, 96, 38), add_16619, *[subscript_call_result_16623, UndefinedType_call_result_16626], **kwargs_16627)
                
                # Getting the type of 'type_dict' (line 96)
                type_dict_16629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'type_dict')
                # Getting the type of 'var_name' (line 96)
                var_name_16630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'var_name')
                # Storing an element on a container (line 96)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 16), type_dict_16629, (var_name_16630, add_call_result_16628))
            else:
                
                # Testing the type of an if condition (line 89)
                if_condition_16601 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 12), result_contains_16600)
                # Assigning a type to the variable 'if_condition_16601' (line 89)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'if_condition_16601', if_condition_16601)
                # SSA begins for if statement (line 89)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 91):
                
                # Call to add(...): (line 91)
                # Processing the call arguments (line 91)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 91)
                var_name_16605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 91)
                function_context_previous_16606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 91)
                getitem___16607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 68), function_context_previous_16606, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 91)
                subscript_call_result_16608 = invoke(stypy.reporting.localization.Localization(__file__, 91, 68), getitem___16607, var_name_16605)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 92)
                var_name_16609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 85), 'var_name', False)
                # Getting the type of 'function_context_else' (line 92)
                function_context_else_16610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 63), 'function_context_else', False)
                # Obtaining the member '__getitem__' of a type (line 92)
                getitem___16611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 63), function_context_else_16610, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 92)
                subscript_call_result_16612 = invoke(stypy.reporting.localization.Localization(__file__, 92, 63), getitem___16611, var_name_16609)
                
                # Processing the call keyword arguments (line 91)
                kwargs_16613 = {}
                # Getting the type of 'union_type_copy' (line 91)
                union_type_copy_16602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 91)
                UnionType_16603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 38), union_type_copy_16602, 'UnionType')
                # Obtaining the member 'add' of a type (line 91)
                add_16604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 38), UnionType_16603, 'add')
                # Calling add(args, kwargs) (line 91)
                add_call_result_16614 = invoke(stypy.reporting.localization.Localization(__file__, 91, 38), add_16604, *[subscript_call_result_16608, subscript_call_result_16612], **kwargs_16613)
                
                # Getting the type of 'type_dict' (line 91)
                type_dict_16615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'type_dict')
                # Getting the type of 'var_name' (line 91)
                var_name_16616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'var_name')
                # Storing an element on a container (line 91)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 16), type_dict_16615, (var_name_16616, add_call_result_16614))
                # SSA branch for the else part of an if statement (line 89)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 96):
                
                # Call to add(...): (line 96)
                # Processing the call arguments (line 96)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 96)
                var_name_16620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 90), 'var_name', False)
                # Getting the type of 'function_context_else' (line 96)
                function_context_else_16621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 68), 'function_context_else', False)
                # Obtaining the member '__getitem__' of a type (line 96)
                getitem___16622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 68), function_context_else_16621, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 96)
                subscript_call_result_16623 = invoke(stypy.reporting.localization.Localization(__file__, 96, 68), getitem___16622, var_name_16620)
                
                
                # Call to UndefinedType(...): (line 96)
                # Processing the call keyword arguments (line 96)
                kwargs_16625 = {}
                # Getting the type of 'UndefinedType' (line 96)
                UndefinedType_16624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 101), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 96)
                UndefinedType_call_result_16626 = invoke(stypy.reporting.localization.Localization(__file__, 96, 101), UndefinedType_16624, *[], **kwargs_16625)
                
                # Processing the call keyword arguments (line 96)
                kwargs_16627 = {}
                # Getting the type of 'union_type_copy' (line 96)
                union_type_copy_16617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 96)
                UnionType_16618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), union_type_copy_16617, 'UnionType')
                # Obtaining the member 'add' of a type (line 96)
                add_16619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), UnionType_16618, 'add')
                # Calling add(args, kwargs) (line 96)
                add_call_result_16628 = invoke(stypy.reporting.localization.Localization(__file__, 96, 38), add_16619, *[subscript_call_result_16623, UndefinedType_call_result_16626], **kwargs_16627)
                
                # Getting the type of 'type_dict' (line 96)
                type_dict_16629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'type_dict')
                # Getting the type of 'var_name' (line 96)
                var_name_16630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'var_name')
                # Storing an element on a container (line 96)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 16), type_dict_16629, (var_name_16630, add_call_result_16628))
                # SSA join for if statement (line 89)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 85)
            if_condition_16597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 8), result_contains_16596)
            # Assigning a type to the variable 'if_condition_16597' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'if_condition_16597', if_condition_16597)
            # SSA begins for if statement (line 85)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA branch for the else part of an if statement (line 85)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'var_name' (line 89)
            var_name_16598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 89)
            function_context_previous_16599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 89)
            result_contains_16600 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 15), 'in', var_name_16598, function_context_previous_16599)
            
            # Testing if the type of an if condition is none (line 89)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 12), result_contains_16600):
                
                # Assigning a Call to a Subscript (line 96):
                
                # Call to add(...): (line 96)
                # Processing the call arguments (line 96)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 96)
                var_name_16620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 90), 'var_name', False)
                # Getting the type of 'function_context_else' (line 96)
                function_context_else_16621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 68), 'function_context_else', False)
                # Obtaining the member '__getitem__' of a type (line 96)
                getitem___16622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 68), function_context_else_16621, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 96)
                subscript_call_result_16623 = invoke(stypy.reporting.localization.Localization(__file__, 96, 68), getitem___16622, var_name_16620)
                
                
                # Call to UndefinedType(...): (line 96)
                # Processing the call keyword arguments (line 96)
                kwargs_16625 = {}
                # Getting the type of 'UndefinedType' (line 96)
                UndefinedType_16624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 101), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 96)
                UndefinedType_call_result_16626 = invoke(stypy.reporting.localization.Localization(__file__, 96, 101), UndefinedType_16624, *[], **kwargs_16625)
                
                # Processing the call keyword arguments (line 96)
                kwargs_16627 = {}
                # Getting the type of 'union_type_copy' (line 96)
                union_type_copy_16617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 96)
                UnionType_16618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), union_type_copy_16617, 'UnionType')
                # Obtaining the member 'add' of a type (line 96)
                add_16619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), UnionType_16618, 'add')
                # Calling add(args, kwargs) (line 96)
                add_call_result_16628 = invoke(stypy.reporting.localization.Localization(__file__, 96, 38), add_16619, *[subscript_call_result_16623, UndefinedType_call_result_16626], **kwargs_16627)
                
                # Getting the type of 'type_dict' (line 96)
                type_dict_16629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'type_dict')
                # Getting the type of 'var_name' (line 96)
                var_name_16630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'var_name')
                # Storing an element on a container (line 96)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 16), type_dict_16629, (var_name_16630, add_call_result_16628))
            else:
                
                # Testing the type of an if condition (line 89)
                if_condition_16601 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 12), result_contains_16600)
                # Assigning a type to the variable 'if_condition_16601' (line 89)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'if_condition_16601', if_condition_16601)
                # SSA begins for if statement (line 89)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 91):
                
                # Call to add(...): (line 91)
                # Processing the call arguments (line 91)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 91)
                var_name_16605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 91)
                function_context_previous_16606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 91)
                getitem___16607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 68), function_context_previous_16606, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 91)
                subscript_call_result_16608 = invoke(stypy.reporting.localization.Localization(__file__, 91, 68), getitem___16607, var_name_16605)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 92)
                var_name_16609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 85), 'var_name', False)
                # Getting the type of 'function_context_else' (line 92)
                function_context_else_16610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 63), 'function_context_else', False)
                # Obtaining the member '__getitem__' of a type (line 92)
                getitem___16611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 63), function_context_else_16610, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 92)
                subscript_call_result_16612 = invoke(stypy.reporting.localization.Localization(__file__, 92, 63), getitem___16611, var_name_16609)
                
                # Processing the call keyword arguments (line 91)
                kwargs_16613 = {}
                # Getting the type of 'union_type_copy' (line 91)
                union_type_copy_16602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 91)
                UnionType_16603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 38), union_type_copy_16602, 'UnionType')
                # Obtaining the member 'add' of a type (line 91)
                add_16604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 38), UnionType_16603, 'add')
                # Calling add(args, kwargs) (line 91)
                add_call_result_16614 = invoke(stypy.reporting.localization.Localization(__file__, 91, 38), add_16604, *[subscript_call_result_16608, subscript_call_result_16612], **kwargs_16613)
                
                # Getting the type of 'type_dict' (line 91)
                type_dict_16615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'type_dict')
                # Getting the type of 'var_name' (line 91)
                var_name_16616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 26), 'var_name')
                # Storing an element on a container (line 91)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 16), type_dict_16615, (var_name_16616, add_call_result_16614))
                # SSA branch for the else part of an if statement (line 89)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 96):
                
                # Call to add(...): (line 96)
                # Processing the call arguments (line 96)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 96)
                var_name_16620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 90), 'var_name', False)
                # Getting the type of 'function_context_else' (line 96)
                function_context_else_16621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 68), 'function_context_else', False)
                # Obtaining the member '__getitem__' of a type (line 96)
                getitem___16622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 68), function_context_else_16621, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 96)
                subscript_call_result_16623 = invoke(stypy.reporting.localization.Localization(__file__, 96, 68), getitem___16622, var_name_16620)
                
                
                # Call to UndefinedType(...): (line 96)
                # Processing the call keyword arguments (line 96)
                kwargs_16625 = {}
                # Getting the type of 'UndefinedType' (line 96)
                UndefinedType_16624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 101), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 96)
                UndefinedType_call_result_16626 = invoke(stypy.reporting.localization.Localization(__file__, 96, 101), UndefinedType_16624, *[], **kwargs_16625)
                
                # Processing the call keyword arguments (line 96)
                kwargs_16627 = {}
                # Getting the type of 'union_type_copy' (line 96)
                union_type_copy_16617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 96)
                UnionType_16618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), union_type_copy_16617, 'UnionType')
                # Obtaining the member 'add' of a type (line 96)
                add_16619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 38), UnionType_16618, 'add')
                # Calling add(args, kwargs) (line 96)
                add_call_result_16628 = invoke(stypy.reporting.localization.Localization(__file__, 96, 38), add_16619, *[subscript_call_result_16623, UndefinedType_call_result_16626], **kwargs_16627)
                
                # Getting the type of 'type_dict' (line 96)
                type_dict_16629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'type_dict')
                # Getting the type of 'var_name' (line 96)
                var_name_16630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'var_name')
                # Storing an element on a container (line 96)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 16), type_dict_16629, (var_name_16630, add_call_result_16628))
                # SSA join for if statement (line 89)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 85)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'type_dict' (line 99)
    type_dict_16631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'type_dict')
    # Assigning a type to the variable 'stypy_return_type' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type', type_dict_16631)
    
    # ################# End of '__ssa_join_with_else_function_context(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__ssa_join_with_else_function_context' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_16632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16632)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__ssa_join_with_else_function_context'
    return stypy_return_type_16632

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

    str_16633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, (-1)), 'str', '\n    Implements the SSA algorithm with the type stores of an if-else structure\n    :param type_store_previous: Type store\n    :param type_store_if: Function context\n    :param type_store_else:\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 111):
    
    # Call to TypeStore(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'type_store_previous' (line 111)
    type_store_previous_16635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'type_store_previous', False)
    # Obtaining the member 'program_name' of a type (line 111)
    program_name_16636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 34), type_store_previous_16635, 'program_name')
    # Processing the call keyword arguments (line 111)
    kwargs_16637 = {}
    # Getting the type of 'TypeStore' (line 111)
    TypeStore_16634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'TypeStore', False)
    # Calling TypeStore(args, kwargs) (line 111)
    TypeStore_call_result_16638 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), TypeStore_16634, *[program_name_16636], **kwargs_16637)
    
    # Assigning a type to the variable 'joined_type_store' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'joined_type_store', TypeStore_call_result_16638)
    
    # Assigning a Attribute to a Attribute (line 112):
    # Getting the type of 'type_store_previous' (line 112)
    type_store_previous_16639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 47), 'type_store_previous')
    # Obtaining the member 'last_function_contexts' of a type (line 112)
    last_function_contexts_16640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 47), type_store_previous_16639, 'last_function_contexts')
    # Getting the type of 'joined_type_store' (line 112)
    joined_type_store_16641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'joined_type_store')
    # Setting the type of the member 'last_function_contexts' of a type (line 112)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 4), joined_type_store_16641, 'last_function_contexts', last_function_contexts_16640)
    
    # Assigning a List to a Attribute (line 113):
    
    # Obtaining an instance of the builtin type 'list' (line 113)
    list_16642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 113)
    
    # Getting the type of 'joined_type_store' (line 113)
    joined_type_store_16643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'joined_type_store')
    # Setting the type of the member 'context_stack' of a type (line 113)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 4), joined_type_store_16643, 'context_stack', list_16642)
    
    
    # Call to range(...): (line 114)
    # Processing the call arguments (line 114)
    
    # Call to len(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'type_store_previous' (line 114)
    type_store_previous_16646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'type_store_previous', False)
    # Obtaining the member 'context_stack' of a type (line 114)
    context_stack_16647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 23), type_store_previous_16646, 'context_stack')
    # Processing the call keyword arguments (line 114)
    kwargs_16648 = {}
    # Getting the type of 'len' (line 114)
    len_16645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 19), 'len', False)
    # Calling len(args, kwargs) (line 114)
    len_call_result_16649 = invoke(stypy.reporting.localization.Localization(__file__, 114, 19), len_16645, *[context_stack_16647], **kwargs_16648)
    
    # Processing the call keyword arguments (line 114)
    kwargs_16650 = {}
    # Getting the type of 'range' (line 114)
    range_16644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 13), 'range', False)
    # Calling range(args, kwargs) (line 114)
    range_call_result_16651 = invoke(stypy.reporting.localization.Localization(__file__, 114, 13), range_16644, *[len_call_result_16649], **kwargs_16650)
    
    # Assigning a type to the variable 'range_call_result_16651' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'range_call_result_16651', range_call_result_16651)
    # Testing if the for loop is going to be iterated (line 114)
    # Testing the type of a for loop iterable (line 114)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 114, 4), range_call_result_16651)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 114, 4), range_call_result_16651):
        # Getting the type of the for loop variable (line 114)
        for_loop_var_16652 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 114, 4), range_call_result_16651)
        # Assigning a type to the variable 'i' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'i', for_loop_var_16652)
        # SSA begins for a for statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 116)
        # Getting the type of 'type_store_else' (line 116)
        type_store_else_16653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'type_store_else')
        # Getting the type of 'None' (line 116)
        None_16654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 30), 'None')
        
        (may_be_16655, more_types_in_union_16656) = may_be_none(type_store_else_16653, None_16654)

        if may_be_16655:

            if more_types_in_union_16656:
                # Runtime conditional SSA (line 116)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 117):
            # Getting the type of 'None' (line 117)
            None_16657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 36), 'None')
            # Assigning a type to the variable 'function_context_else' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'function_context_else', None_16657)

            if more_types_in_union_16656:
                # Runtime conditional SSA for else branch (line 116)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_16655) or more_types_in_union_16656):
            
            # Assigning a Subscript to a Name (line 119):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 119)
            i_16658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 52), 'i')
            # Getting the type of 'type_store_else' (line 119)
            type_store_else_16659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 36), 'type_store_else')
            # Obtaining the member '__getitem__' of a type (line 119)
            getitem___16660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 36), type_store_else_16659, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 119)
            subscript_call_result_16661 = invoke(stypy.reporting.localization.Localization(__file__, 119, 36), getitem___16660, i_16658)
            
            # Assigning a type to the variable 'function_context_else' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'function_context_else', subscript_call_result_16661)

            if (may_be_16655 and more_types_in_union_16656):
                # SSA join for if statement (line 116)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 121):
        
        # Call to __ssa_join_with_else_function_context(...): (line 121)
        # Processing the call arguments (line 121)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 121)
        i_16663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 88), 'i', False)
        # Getting the type of 'type_store_previous' (line 121)
        type_store_previous_16664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 68), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___16665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 68), type_store_previous_16664, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_16666 = invoke(stypy.reporting.localization.Localization(__file__, 121, 68), getitem___16665, i_16663)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 121)
        i_16667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 106), 'i', False)
        # Getting the type of 'type_store_if' (line 121)
        type_store_if_16668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 92), 'type_store_if', False)
        # Obtaining the member '__getitem__' of a type (line 121)
        getitem___16669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 92), type_store_if_16668, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 121)
        subscript_call_result_16670 = invoke(stypy.reporting.localization.Localization(__file__, 121, 92), getitem___16669, i_16667)
        
        # Getting the type of 'function_context_else' (line 122)
        function_context_else_16671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 68), 'function_context_else', False)
        # Processing the call keyword arguments (line 121)
        kwargs_16672 = {}
        # Getting the type of '__ssa_join_with_else_function_context' (line 121)
        ssa_join_with_else_function_context_16662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 30), '__ssa_join_with_else_function_context', False)
        # Calling __ssa_join_with_else_function_context(args, kwargs) (line 121)
        ssa_join_with_else_function_context_call_result_16673 = invoke(stypy.reporting.localization.Localization(__file__, 121, 30), ssa_join_with_else_function_context_16662, *[subscript_call_result_16666, subscript_call_result_16670, function_context_else_16671], **kwargs_16672)
        
        # Assigning a type to the variable 'joined_context_dict' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'joined_context_dict', ssa_join_with_else_function_context_call_result_16673)
        
        # Assigning a Call to a Name (line 125):
        
        # Call to copy(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_16679 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 125)
        i_16674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 47), 'i', False)
        # Getting the type of 'type_store_previous' (line 125)
        type_store_previous_16675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 125)
        getitem___16676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 27), type_store_previous_16675, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 125)
        subscript_call_result_16677 = invoke(stypy.reporting.localization.Localization(__file__, 125, 27), getitem___16676, i_16674)
        
        # Obtaining the member 'copy' of a type (line 125)
        copy_16678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 27), subscript_call_result_16677, 'copy')
        # Calling copy(args, kwargs) (line 125)
        copy_call_result_16680 = invoke(stypy.reporting.localization.Localization(__file__, 125, 27), copy_16678, *[], **kwargs_16679)
        
        # Assigning a type to the variable 'joined_f_context' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'joined_f_context', copy_call_result_16680)
        
        # Assigning a Name to a Attribute (line 126):
        # Getting the type of 'joined_context_dict' (line 126)
        joined_context_dict_16681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'joined_context_dict')
        # Getting the type of 'joined_f_context' (line 126)
        joined_f_context_16682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'joined_f_context')
        # Setting the type of the member 'types_of' of a type (line 126)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), joined_f_context_16682, 'types_of', joined_context_dict_16681)
        
        # Assigning a Call to a Attribute (line 127):
        
        # Call to __join_globals(...): (line 127)
        # Processing the call arguments (line 127)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 127)
        i_16684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 68), 'i', False)
        # Getting the type of 'type_store_if' (line 127)
        type_store_if_16685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 54), 'type_store_if', False)
        # Obtaining the member '__getitem__' of a type (line 127)
        getitem___16686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 54), type_store_if_16685, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
        subscript_call_result_16687 = invoke(stypy.reporting.localization.Localization(__file__, 127, 54), getitem___16686, i_16684)
        
        # Getting the type of 'function_context_else' (line 127)
        function_context_else_16688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 72), 'function_context_else', False)
        # Processing the call keyword arguments (line 127)
        kwargs_16689 = {}
        # Getting the type of '__join_globals' (line 127)
        join_globals_16683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 39), '__join_globals', False)
        # Calling __join_globals(args, kwargs) (line 127)
        join_globals_call_result_16690 = invoke(stypy.reporting.localization.Localization(__file__, 127, 39), join_globals_16683, *[subscript_call_result_16687, function_context_else_16688], **kwargs_16689)
        
        # Getting the type of 'joined_f_context' (line 127)
        joined_f_context_16691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'joined_f_context')
        # Setting the type of the member 'global_vars' of a type (line 127)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), joined_f_context_16691, 'global_vars', join_globals_call_result_16690)
        
        # Assigning a Attribute to a Attribute (line 128):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 129)
        i_16692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'i')
        # Getting the type of 'type_store_if' (line 128)
        type_store_if_16693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 45), 'type_store_if')
        # Obtaining the member '__getitem__' of a type (line 128)
        getitem___16694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 45), type_store_if_16693, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
        subscript_call_result_16695 = invoke(stypy.reporting.localization.Localization(__file__, 128, 45), getitem___16694, i_16692)
        
        # Obtaining the member 'annotation_record' of a type (line 128)
        annotation_record_16696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 45), subscript_call_result_16695, 'annotation_record')
        # Getting the type of 'joined_f_context' (line 128)
        joined_f_context_16697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'joined_f_context')
        # Setting the type of the member 'annotation_record' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), joined_f_context_16697, 'annotation_record', annotation_record_16696)
        
        # Call to append(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'joined_f_context' (line 131)
        joined_f_context_16701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 47), 'joined_f_context', False)
        # Processing the call keyword arguments (line 131)
        kwargs_16702 = {}
        # Getting the type of 'joined_type_store' (line 131)
        joined_type_store_16698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'joined_type_store', False)
        # Obtaining the member 'context_stack' of a type (line 131)
        context_stack_16699 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), joined_type_store_16698, 'context_stack')
        # Obtaining the member 'append' of a type (line 131)
        append_16700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), context_stack_16699, 'append')
        # Calling append(args, kwargs) (line 131)
        append_call_result_16703 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), append_16700, *[joined_f_context_16701], **kwargs_16702)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'joined_type_store' (line 133)
    joined_type_store_16704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'joined_type_store')
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type', joined_type_store_16704)
    
    # ################# End of 'ssa_join_with_else_branch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ssa_join_with_else_branch' in the type store
    # Getting the type of 'stypy_return_type' (line 102)
    stypy_return_type_16705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16705)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ssa_join_with_else_branch'
    return stypy_return_type_16705

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

    str_16706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, (-1)), 'str', '\n    Helper function to join variables of function contexts that belong to different except\n    blocks\n    :param function_context_previous: Function context\n    :param function_context_new: Function context\n    :return: A dictionary with names of variables and its joined types\n    ')
    
    # Assigning a Dict to a Name (line 146):
    
    # Obtaining an instance of the builtin type 'dict' (line 146)
    dict_16707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 146)
    
    # Assigning a type to the variable 'type_dict' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'type_dict', dict_16707)
    
    # Getting the type of 'function_context_previous' (line 148)
    function_context_previous_16708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'function_context_previous')
    # Assigning a type to the variable 'function_context_previous_16708' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'function_context_previous_16708', function_context_previous_16708)
    # Testing if the for loop is going to be iterated (line 148)
    # Testing the type of a for loop iterable (line 148)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 148, 4), function_context_previous_16708)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 148, 4), function_context_previous_16708):
        # Getting the type of the for loop variable (line 148)
        for_loop_var_16709 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 148, 4), function_context_previous_16708)
        # Assigning a type to the variable 'var_name' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'var_name', for_loop_var_16709)
        # SSA begins for a for statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 149)
        var_name_16710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'var_name')
        # Getting the type of 'function_context_new' (line 149)
        function_context_new_16711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 23), 'function_context_new')
        # Applying the binary operator 'in' (line 149)
        result_contains_16712 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), 'in', var_name_16710, function_context_new_16711)
        
        # Testing if the type of an if condition is none (line 149)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 149, 8), result_contains_16712):
            
            # Assigning a Subscript to a Subscript (line 155):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 155)
            var_name_16729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 60), 'var_name')
            # Getting the type of 'function_context_previous' (line 155)
            function_context_previous_16730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'function_context_previous')
            # Obtaining the member '__getitem__' of a type (line 155)
            getitem___16731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 34), function_context_previous_16730, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 155)
            subscript_call_result_16732 = invoke(stypy.reporting.localization.Localization(__file__, 155, 34), getitem___16731, var_name_16729)
            
            # Getting the type of 'type_dict' (line 155)
            type_dict_16733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'type_dict')
            # Getting the type of 'var_name' (line 155)
            var_name_16734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'var_name')
            # Storing an element on a container (line 155)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 12), type_dict_16733, (var_name_16734, subscript_call_result_16732))
        else:
            
            # Testing the type of an if condition (line 149)
            if_condition_16713 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), result_contains_16712)
            # Assigning a type to the variable 'if_condition_16713' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_16713', if_condition_16713)
            # SSA begins for if statement (line 149)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 151):
            
            # Call to add(...): (line 151)
            # Processing the call arguments (line 151)
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 151)
            var_name_16717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 90), 'var_name', False)
            # Getting the type of 'function_context_previous' (line 151)
            function_context_previous_16718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 64), 'function_context_previous', False)
            # Obtaining the member '__getitem__' of a type (line 151)
            getitem___16719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 64), function_context_previous_16718, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 151)
            subscript_call_result_16720 = invoke(stypy.reporting.localization.Localization(__file__, 151, 64), getitem___16719, var_name_16717)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 152)
            var_name_16721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 80), 'var_name', False)
            # Getting the type of 'function_context_new' (line 152)
            function_context_new_16722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 59), 'function_context_new', False)
            # Obtaining the member '__getitem__' of a type (line 152)
            getitem___16723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 59), function_context_new_16722, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 152)
            subscript_call_result_16724 = invoke(stypy.reporting.localization.Localization(__file__, 152, 59), getitem___16723, var_name_16721)
            
            # Processing the call keyword arguments (line 151)
            kwargs_16725 = {}
            # Getting the type of 'union_type_copy' (line 151)
            union_type_copy_16714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'union_type_copy', False)
            # Obtaining the member 'UnionType' of a type (line 151)
            UnionType_16715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 34), union_type_copy_16714, 'UnionType')
            # Obtaining the member 'add' of a type (line 151)
            add_16716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 34), UnionType_16715, 'add')
            # Calling add(args, kwargs) (line 151)
            add_call_result_16726 = invoke(stypy.reporting.localization.Localization(__file__, 151, 34), add_16716, *[subscript_call_result_16720, subscript_call_result_16724], **kwargs_16725)
            
            # Getting the type of 'type_dict' (line 151)
            type_dict_16727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'type_dict')
            # Getting the type of 'var_name' (line 151)
            var_name_16728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 22), 'var_name')
            # Storing an element on a container (line 151)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 12), type_dict_16727, (var_name_16728, add_call_result_16726))
            # SSA branch for the else part of an if statement (line 149)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Subscript to a Subscript (line 155):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 155)
            var_name_16729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 60), 'var_name')
            # Getting the type of 'function_context_previous' (line 155)
            function_context_previous_16730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'function_context_previous')
            # Obtaining the member '__getitem__' of a type (line 155)
            getitem___16731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 34), function_context_previous_16730, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 155)
            subscript_call_result_16732 = invoke(stypy.reporting.localization.Localization(__file__, 155, 34), getitem___16731, var_name_16729)
            
            # Getting the type of 'type_dict' (line 155)
            type_dict_16733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'type_dict')
            # Getting the type of 'var_name' (line 155)
            var_name_16734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'var_name')
            # Storing an element on a container (line 155)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 12), type_dict_16733, (var_name_16734, subscript_call_result_16732))
            # SSA join for if statement (line 149)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'function_context_new' (line 157)
    function_context_new_16735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'function_context_new')
    # Assigning a type to the variable 'function_context_new_16735' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'function_context_new_16735', function_context_new_16735)
    # Testing if the for loop is going to be iterated (line 157)
    # Testing the type of a for loop iterable (line 157)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 157, 4), function_context_new_16735)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 157, 4), function_context_new_16735):
        # Getting the type of the for loop variable (line 157)
        for_loop_var_16736 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 157, 4), function_context_new_16735)
        # Assigning a type to the variable 'var_name' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'var_name', for_loop_var_16736)
        # SSA begins for a for statement (line 157)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 158)
        var_name_16737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 11), 'var_name')
        # Getting the type of 'function_context_previous' (line 158)
        function_context_previous_16738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 23), 'function_context_previous')
        # Applying the binary operator 'in' (line 158)
        result_contains_16739 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 11), 'in', var_name_16737, function_context_previous_16738)
        
        # Testing if the type of an if condition is none (line 158)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 158, 8), result_contains_16739):
            
            # Assigning a Subscript to a Subscript (line 164):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 164)
            var_name_16756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 55), 'var_name')
            # Getting the type of 'function_context_new' (line 164)
            function_context_new_16757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'function_context_new')
            # Obtaining the member '__getitem__' of a type (line 164)
            getitem___16758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 34), function_context_new_16757, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 164)
            subscript_call_result_16759 = invoke(stypy.reporting.localization.Localization(__file__, 164, 34), getitem___16758, var_name_16756)
            
            # Getting the type of 'type_dict' (line 164)
            type_dict_16760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'type_dict')
            # Getting the type of 'var_name' (line 164)
            var_name_16761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 22), 'var_name')
            # Storing an element on a container (line 164)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 12), type_dict_16760, (var_name_16761, subscript_call_result_16759))
        else:
            
            # Testing the type of an if condition (line 158)
            if_condition_16740 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), result_contains_16739)
            # Assigning a type to the variable 'if_condition_16740' (line 158)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_16740', if_condition_16740)
            # SSA begins for if statement (line 158)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 160):
            
            # Call to add(...): (line 160)
            # Processing the call arguments (line 160)
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 160)
            var_name_16744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 85), 'var_name', False)
            # Getting the type of 'function_context_new' (line 160)
            function_context_new_16745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 64), 'function_context_new', False)
            # Obtaining the member '__getitem__' of a type (line 160)
            getitem___16746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 64), function_context_new_16745, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 160)
            subscript_call_result_16747 = invoke(stypy.reporting.localization.Localization(__file__, 160, 64), getitem___16746, var_name_16744)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 161)
            var_name_16748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 85), 'var_name', False)
            # Getting the type of 'function_context_previous' (line 161)
            function_context_previous_16749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 59), 'function_context_previous', False)
            # Obtaining the member '__getitem__' of a type (line 161)
            getitem___16750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 59), function_context_previous_16749, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 161)
            subscript_call_result_16751 = invoke(stypy.reporting.localization.Localization(__file__, 161, 59), getitem___16750, var_name_16748)
            
            # Processing the call keyword arguments (line 160)
            kwargs_16752 = {}
            # Getting the type of 'union_type_copy' (line 160)
            union_type_copy_16741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 34), 'union_type_copy', False)
            # Obtaining the member 'UnionType' of a type (line 160)
            UnionType_16742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 34), union_type_copy_16741, 'UnionType')
            # Obtaining the member 'add' of a type (line 160)
            add_16743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 34), UnionType_16742, 'add')
            # Calling add(args, kwargs) (line 160)
            add_call_result_16753 = invoke(stypy.reporting.localization.Localization(__file__, 160, 34), add_16743, *[subscript_call_result_16747, subscript_call_result_16751], **kwargs_16752)
            
            # Getting the type of 'type_dict' (line 160)
            type_dict_16754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'type_dict')
            # Getting the type of 'var_name' (line 160)
            var_name_16755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 22), 'var_name')
            # Storing an element on a container (line 160)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 160, 12), type_dict_16754, (var_name_16755, add_call_result_16753))
            # SSA branch for the else part of an if statement (line 158)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Subscript to a Subscript (line 164):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 164)
            var_name_16756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 55), 'var_name')
            # Getting the type of 'function_context_new' (line 164)
            function_context_new_16757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 34), 'function_context_new')
            # Obtaining the member '__getitem__' of a type (line 164)
            getitem___16758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 34), function_context_new_16757, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 164)
            subscript_call_result_16759 = invoke(stypy.reporting.localization.Localization(__file__, 164, 34), getitem___16758, var_name_16756)
            
            # Getting the type of 'type_dict' (line 164)
            type_dict_16760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 12), 'type_dict')
            # Getting the type of 'var_name' (line 164)
            var_name_16761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 22), 'var_name')
            # Storing an element on a container (line 164)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 12), type_dict_16760, (var_name_16761, subscript_call_result_16759))
            # SSA join for if statement (line 158)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'type_dict' (line 167)
    type_dict_16762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 11), 'type_dict')
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type', type_dict_16762)
    
    # ################# End of '__join_except_branches_function_context(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join_except_branches_function_context' in the type store
    # Getting the type of 'stypy_return_type' (line 138)
    stypy_return_type_16763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16763)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join_except_branches_function_context'
    return stypy_return_type_16763

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

    str_16764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, (-1)), 'str', '\n    SSA algorithm to join type stores of different except branches\n    :param type_store_previous: Type store\n    :param type_store_new: Type store\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 177):
    
    # Call to TypeStore(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'type_store_previous' (line 177)
    type_store_previous_16766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 34), 'type_store_previous', False)
    # Obtaining the member 'program_name' of a type (line 177)
    program_name_16767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 34), type_store_previous_16766, 'program_name')
    # Processing the call keyword arguments (line 177)
    kwargs_16768 = {}
    # Getting the type of 'TypeStore' (line 177)
    TypeStore_16765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 24), 'TypeStore', False)
    # Calling TypeStore(args, kwargs) (line 177)
    TypeStore_call_result_16769 = invoke(stypy.reporting.localization.Localization(__file__, 177, 24), TypeStore_16765, *[program_name_16767], **kwargs_16768)
    
    # Assigning a type to the variable 'joined_type_store' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'joined_type_store', TypeStore_call_result_16769)
    
    # Assigning a Attribute to a Attribute (line 178):
    # Getting the type of 'type_store_previous' (line 178)
    type_store_previous_16770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 47), 'type_store_previous')
    # Obtaining the member 'last_function_contexts' of a type (line 178)
    last_function_contexts_16771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 47), type_store_previous_16770, 'last_function_contexts')
    # Getting the type of 'joined_type_store' (line 178)
    joined_type_store_16772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'joined_type_store')
    # Setting the type of the member 'last_function_contexts' of a type (line 178)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 4), joined_type_store_16772, 'last_function_contexts', last_function_contexts_16771)
    
    # Assigning a List to a Attribute (line 179):
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_16773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    
    # Getting the type of 'joined_type_store' (line 179)
    joined_type_store_16774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'joined_type_store')
    # Setting the type of the member 'context_stack' of a type (line 179)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 4), joined_type_store_16774, 'context_stack', list_16773)
    
    
    # Call to range(...): (line 180)
    # Processing the call arguments (line 180)
    
    # Call to len(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'type_store_previous' (line 180)
    type_store_previous_16777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'type_store_previous', False)
    # Obtaining the member 'context_stack' of a type (line 180)
    context_stack_16778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 23), type_store_previous_16777, 'context_stack')
    # Processing the call keyword arguments (line 180)
    kwargs_16779 = {}
    # Getting the type of 'len' (line 180)
    len_16776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'len', False)
    # Calling len(args, kwargs) (line 180)
    len_call_result_16780 = invoke(stypy.reporting.localization.Localization(__file__, 180, 19), len_16776, *[context_stack_16778], **kwargs_16779)
    
    # Processing the call keyword arguments (line 180)
    kwargs_16781 = {}
    # Getting the type of 'range' (line 180)
    range_16775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 13), 'range', False)
    # Calling range(args, kwargs) (line 180)
    range_call_result_16782 = invoke(stypy.reporting.localization.Localization(__file__, 180, 13), range_16775, *[len_call_result_16780], **kwargs_16781)
    
    # Assigning a type to the variable 'range_call_result_16782' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'range_call_result_16782', range_call_result_16782)
    # Testing if the for loop is going to be iterated (line 180)
    # Testing the type of a for loop iterable (line 180)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 180, 4), range_call_result_16782)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 180, 4), range_call_result_16782):
        # Getting the type of the for loop variable (line 180)
        for_loop_var_16783 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 180, 4), range_call_result_16782)
        # Assigning a type to the variable 'i' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'i', for_loop_var_16783)
        # SSA begins for a for statement (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 181):
        
        # Call to __join_except_branches_function_context(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 181)
        i_16785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 90), 'i', False)
        # Getting the type of 'type_store_previous' (line 181)
        type_store_previous_16786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 70), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___16787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 70), type_store_previous_16786, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_16788 = invoke(stypy.reporting.localization.Localization(__file__, 181, 70), getitem___16787, i_16785)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 181)
        i_16789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 109), 'i', False)
        # Getting the type of 'type_store_new' (line 181)
        type_store_new_16790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 94), 'type_store_new', False)
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___16791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 94), type_store_new_16790, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_16792 = invoke(stypy.reporting.localization.Localization(__file__, 181, 94), getitem___16791, i_16789)
        
        # Processing the call keyword arguments (line 181)
        kwargs_16793 = {}
        # Getting the type of '__join_except_branches_function_context' (line 181)
        join_except_branches_function_context_16784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), '__join_except_branches_function_context', False)
        # Calling __join_except_branches_function_context(args, kwargs) (line 181)
        join_except_branches_function_context_call_result_16794 = invoke(stypy.reporting.localization.Localization(__file__, 181, 30), join_except_branches_function_context_16784, *[subscript_call_result_16788, subscript_call_result_16792], **kwargs_16793)
        
        # Assigning a type to the variable 'joined_context_dict' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'joined_context_dict', join_except_branches_function_context_call_result_16794)
        
        # Assigning a Call to a Name (line 183):
        
        # Call to copy(...): (line 183)
        # Processing the call keyword arguments (line 183)
        kwargs_16800 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 183)
        i_16795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 47), 'i', False)
        # Getting the type of 'type_store_previous' (line 183)
        type_store_previous_16796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 27), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___16797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 27), type_store_previous_16796, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_16798 = invoke(stypy.reporting.localization.Localization(__file__, 183, 27), getitem___16797, i_16795)
        
        # Obtaining the member 'copy' of a type (line 183)
        copy_16799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 27), subscript_call_result_16798, 'copy')
        # Calling copy(args, kwargs) (line 183)
        copy_call_result_16801 = invoke(stypy.reporting.localization.Localization(__file__, 183, 27), copy_16799, *[], **kwargs_16800)
        
        # Assigning a type to the variable 'joined_f_context' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'joined_f_context', copy_call_result_16801)
        
        # Assigning a Name to a Attribute (line 184):
        # Getting the type of 'joined_context_dict' (line 184)
        joined_context_dict_16802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'joined_context_dict')
        # Getting the type of 'joined_f_context' (line 184)
        joined_f_context_16803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'joined_f_context')
        # Setting the type of the member 'types_of' of a type (line 184)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), joined_f_context_16803, 'types_of', joined_context_dict_16802)
        
        # Assigning a Call to a Attribute (line 185):
        
        # Call to __join_globals(...): (line 185)
        # Processing the call arguments (line 185)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 185)
        i_16805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 74), 'i', False)
        # Getting the type of 'type_store_previous' (line 185)
        type_store_previous_16806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 54), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___16807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 54), type_store_previous_16806, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_16808 = invoke(stypy.reporting.localization.Localization(__file__, 185, 54), getitem___16807, i_16805)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 185)
        i_16809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 93), 'i', False)
        # Getting the type of 'type_store_new' (line 185)
        type_store_new_16810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 78), 'type_store_new', False)
        # Obtaining the member '__getitem__' of a type (line 185)
        getitem___16811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 78), type_store_new_16810, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 185)
        subscript_call_result_16812 = invoke(stypy.reporting.localization.Localization(__file__, 185, 78), getitem___16811, i_16809)
        
        # Processing the call keyword arguments (line 185)
        kwargs_16813 = {}
        # Getting the type of '__join_globals' (line 185)
        join_globals_16804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 39), '__join_globals', False)
        # Calling __join_globals(args, kwargs) (line 185)
        join_globals_call_result_16814 = invoke(stypy.reporting.localization.Localization(__file__, 185, 39), join_globals_16804, *[subscript_call_result_16808, subscript_call_result_16812], **kwargs_16813)
        
        # Getting the type of 'joined_f_context' (line 185)
        joined_f_context_16815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'joined_f_context')
        # Setting the type of the member 'global_vars' of a type (line 185)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), joined_f_context_16815, 'global_vars', join_globals_call_result_16814)
        
        # Assigning a Attribute to a Attribute (line 186):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 187)
        i_16816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 12), 'i')
        # Getting the type of 'type_store_previous' (line 186)
        type_store_previous_16817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 45), 'type_store_previous')
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___16818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 45), type_store_previous_16817, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_16819 = invoke(stypy.reporting.localization.Localization(__file__, 186, 45), getitem___16818, i_16816)
        
        # Obtaining the member 'annotation_record' of a type (line 186)
        annotation_record_16820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 45), subscript_call_result_16819, 'annotation_record')
        # Getting the type of 'joined_f_context' (line 186)
        joined_f_context_16821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'joined_f_context')
        # Setting the type of the member 'annotation_record' of a type (line 186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), joined_f_context_16821, 'annotation_record', annotation_record_16820)
        
        # Call to append(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'joined_f_context' (line 189)
        joined_f_context_16825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 47), 'joined_f_context', False)
        # Processing the call keyword arguments (line 189)
        kwargs_16826 = {}
        # Getting the type of 'joined_type_store' (line 189)
        joined_type_store_16822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'joined_type_store', False)
        # Obtaining the member 'context_stack' of a type (line 189)
        context_stack_16823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), joined_type_store_16822, 'context_stack')
        # Obtaining the member 'append' of a type (line 189)
        append_16824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), context_stack_16823, 'append')
        # Calling append(args, kwargs) (line 189)
        append_call_result_16827 = invoke(stypy.reporting.localization.Localization(__file__, 189, 8), append_16824, *[joined_f_context_16825], **kwargs_16826)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'joined_type_store' (line 191)
    joined_type_store_16828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'joined_type_store')
    # Assigning a type to the variable 'stypy_return_type' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type', joined_type_store_16828)
    
    # ################# End of '__join_except_branches(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join_except_branches' in the type store
    # Getting the type of 'stypy_return_type' (line 170)
    stypy_return_type_16829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16829)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join_except_branches'
    return stypy_return_type_16829

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

    str_16830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, (-1)), 'str', '\n    Join the variables of a function context on a finally block with a function context of the joined type store\n     of all the except branches in an exception clause\n    :param function_context_previous: Function context\n    :param function_context_finally: Function context\n    :return: A dictionary with names of variables and its joined types\n    ')
    
    # Assigning a Dict to a Name (line 202):
    
    # Obtaining an instance of the builtin type 'dict' (line 202)
    dict_16831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 202)
    
    # Assigning a type to the variable 'type_dict' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'type_dict', dict_16831)
    
    # Getting the type of 'function_context_previous' (line 204)
    function_context_previous_16832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 20), 'function_context_previous')
    # Assigning a type to the variable 'function_context_previous_16832' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'function_context_previous_16832', function_context_previous_16832)
    # Testing if the for loop is going to be iterated (line 204)
    # Testing the type of a for loop iterable (line 204)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 204, 4), function_context_previous_16832)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 204, 4), function_context_previous_16832):
        # Getting the type of the for loop variable (line 204)
        for_loop_var_16833 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 204, 4), function_context_previous_16832)
        # Assigning a type to the variable 'var_name' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'var_name', for_loop_var_16833)
        # SSA begins for a for statement (line 204)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 205)
        var_name_16834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'var_name')
        # Getting the type of 'function_context_finally' (line 205)
        function_context_finally_16835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 23), 'function_context_finally')
        # Applying the binary operator 'in' (line 205)
        result_contains_16836 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), 'in', var_name_16834, function_context_finally_16835)
        
        # Testing if the type of an if condition is none (line 205)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 205, 8), result_contains_16836):
            pass
        else:
            
            # Testing the type of an if condition (line 205)
            if_condition_16837 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), result_contains_16836)
            # Assigning a type to the variable 'if_condition_16837' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_16837', if_condition_16837)
            # SSA begins for if statement (line 205)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Subscript (line 207):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 207)
            var_name_16838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 59), 'var_name')
            # Getting the type of 'function_context_finally' (line 207)
            function_context_finally_16839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 34), 'function_context_finally')
            # Obtaining the member '__getitem__' of a type (line 207)
            getitem___16840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 34), function_context_finally_16839, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 207)
            subscript_call_result_16841 = invoke(stypy.reporting.localization.Localization(__file__, 207, 34), getitem___16840, var_name_16838)
            
            # Getting the type of 'type_dict' (line 207)
            type_dict_16842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'type_dict')
            # Getting the type of 'var_name' (line 207)
            var_name_16843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 22), 'var_name')
            # Storing an element on a container (line 207)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 12), type_dict_16842, (var_name_16843, subscript_call_result_16841))
            # SSA branch for the else part of an if statement (line 205)
            module_type_store.open_ssa_branch('else')
            pass
            # SSA join for if statement (line 205)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'function_context_finally' (line 212)
    function_context_finally_16844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'function_context_finally')
    # Assigning a type to the variable 'function_context_finally_16844' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'function_context_finally_16844', function_context_finally_16844)
    # Testing if the for loop is going to be iterated (line 212)
    # Testing the type of a for loop iterable (line 212)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 212, 4), function_context_finally_16844)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 212, 4), function_context_finally_16844):
        # Getting the type of the for loop variable (line 212)
        for_loop_var_16845 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 212, 4), function_context_finally_16844)
        # Assigning a type to the variable 'var_name' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'var_name', for_loop_var_16845)
        # SSA begins for a for statement (line 212)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 213)
        var_name_16846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'var_name')
        # Getting the type of 'function_context_previous' (line 213)
        function_context_previous_16847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 23), 'function_context_previous')
        # Applying the binary operator 'in' (line 213)
        result_contains_16848 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 11), 'in', var_name_16846, function_context_previous_16847)
        
        # Testing if the type of an if condition is none (line 213)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 213, 8), result_contains_16848):
            
            # Assigning a Subscript to a Subscript (line 218):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 218)
            var_name_16850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 59), 'var_name')
            # Getting the type of 'function_context_finally' (line 218)
            function_context_finally_16851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 34), 'function_context_finally')
            # Obtaining the member '__getitem__' of a type (line 218)
            getitem___16852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 34), function_context_finally_16851, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 218)
            subscript_call_result_16853 = invoke(stypy.reporting.localization.Localization(__file__, 218, 34), getitem___16852, var_name_16850)
            
            # Getting the type of 'type_dict' (line 218)
            type_dict_16854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'type_dict')
            # Getting the type of 'var_name' (line 218)
            var_name_16855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'var_name')
            # Storing an element on a container (line 218)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 12), type_dict_16854, (var_name_16855, subscript_call_result_16853))
        else:
            
            # Testing the type of an if condition (line 213)
            if_condition_16849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 8), result_contains_16848)
            # Assigning a type to the variable 'if_condition_16849' (line 213)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'if_condition_16849', if_condition_16849)
            # SSA begins for if statement (line 213)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA branch for the else part of an if statement (line 213)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Subscript to a Subscript (line 218):
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 218)
            var_name_16850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 59), 'var_name')
            # Getting the type of 'function_context_finally' (line 218)
            function_context_finally_16851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 34), 'function_context_finally')
            # Obtaining the member '__getitem__' of a type (line 218)
            getitem___16852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 34), function_context_finally_16851, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 218)
            subscript_call_result_16853 = invoke(stypy.reporting.localization.Localization(__file__, 218, 34), getitem___16852, var_name_16850)
            
            # Getting the type of 'type_dict' (line 218)
            type_dict_16854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'type_dict')
            # Getting the type of 'var_name' (line 218)
            var_name_16855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'var_name')
            # Storing an element on a container (line 218)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 12), type_dict_16854, (var_name_16855, subscript_call_result_16853))
            # SSA join for if statement (line 213)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'type_dict' (line 221)
    type_dict_16856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'type_dict')
    # Assigning a type to the variable 'stypy_return_type' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'stypy_return_type', type_dict_16856)
    
    # ################# End of '__join_finally_function_context(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join_finally_function_context' in the type store
    # Getting the type of 'stypy_return_type' (line 194)
    stypy_return_type_16857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16857)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join_finally_function_context'
    return stypy_return_type_16857

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

    str_16858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, (-1)), 'str', '\n    Join the type stores of a finally branch and the joined type store of all except branches in a exception handling\n     block\n    :param type_store_exception_block: Type store\n    :param type_store_finally: Type store\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 232):
    
    # Call to TypeStore(...): (line 232)
    # Processing the call arguments (line 232)
    # Getting the type of 'type_store_exception_block' (line 232)
    type_store_exception_block_16860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 34), 'type_store_exception_block', False)
    # Obtaining the member 'program_name' of a type (line 232)
    program_name_16861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 34), type_store_exception_block_16860, 'program_name')
    # Processing the call keyword arguments (line 232)
    kwargs_16862 = {}
    # Getting the type of 'TypeStore' (line 232)
    TypeStore_16859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'TypeStore', False)
    # Calling TypeStore(args, kwargs) (line 232)
    TypeStore_call_result_16863 = invoke(stypy.reporting.localization.Localization(__file__, 232, 24), TypeStore_16859, *[program_name_16861], **kwargs_16862)
    
    # Assigning a type to the variable 'joined_type_store' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'joined_type_store', TypeStore_call_result_16863)
    
    # Assigning a Attribute to a Attribute (line 233):
    # Getting the type of 'type_store_exception_block' (line 233)
    type_store_exception_block_16864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 47), 'type_store_exception_block')
    # Obtaining the member 'last_function_contexts' of a type (line 233)
    last_function_contexts_16865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 47), type_store_exception_block_16864, 'last_function_contexts')
    # Getting the type of 'joined_type_store' (line 233)
    joined_type_store_16866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 4), 'joined_type_store')
    # Setting the type of the member 'last_function_contexts' of a type (line 233)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 4), joined_type_store_16866, 'last_function_contexts', last_function_contexts_16865)
    
    # Assigning a List to a Attribute (line 234):
    
    # Obtaining an instance of the builtin type 'list' (line 234)
    list_16867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 234)
    
    # Getting the type of 'joined_type_store' (line 234)
    joined_type_store_16868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'joined_type_store')
    # Setting the type of the member 'context_stack' of a type (line 234)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 4), joined_type_store_16868, 'context_stack', list_16867)
    
    
    # Call to range(...): (line 235)
    # Processing the call arguments (line 235)
    
    # Call to len(...): (line 235)
    # Processing the call arguments (line 235)
    # Getting the type of 'type_store_exception_block' (line 235)
    type_store_exception_block_16871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'type_store_exception_block', False)
    # Obtaining the member 'context_stack' of a type (line 235)
    context_stack_16872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 23), type_store_exception_block_16871, 'context_stack')
    # Processing the call keyword arguments (line 235)
    kwargs_16873 = {}
    # Getting the type of 'len' (line 235)
    len_16870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 19), 'len', False)
    # Calling len(args, kwargs) (line 235)
    len_call_result_16874 = invoke(stypy.reporting.localization.Localization(__file__, 235, 19), len_16870, *[context_stack_16872], **kwargs_16873)
    
    # Processing the call keyword arguments (line 235)
    kwargs_16875 = {}
    # Getting the type of 'range' (line 235)
    range_16869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 13), 'range', False)
    # Calling range(args, kwargs) (line 235)
    range_call_result_16876 = invoke(stypy.reporting.localization.Localization(__file__, 235, 13), range_16869, *[len_call_result_16874], **kwargs_16875)
    
    # Assigning a type to the variable 'range_call_result_16876' (line 235)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'range_call_result_16876', range_call_result_16876)
    # Testing if the for loop is going to be iterated (line 235)
    # Testing the type of a for loop iterable (line 235)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 235, 4), range_call_result_16876)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 235, 4), range_call_result_16876):
        # Getting the type of the for loop variable (line 235)
        for_loop_var_16877 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 235, 4), range_call_result_16876)
        # Assigning a type to the variable 'i' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'i', for_loop_var_16877)
        # SSA begins for a for statement (line 235)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 236):
        
        # Call to __join_finally_function_context(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 236)
        i_16879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 89), 'i', False)
        # Getting the type of 'type_store_exception_block' (line 236)
        type_store_exception_block_16880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 62), 'type_store_exception_block', False)
        # Obtaining the member '__getitem__' of a type (line 236)
        getitem___16881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 62), type_store_exception_block_16880, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 236)
        subscript_call_result_16882 = invoke(stypy.reporting.localization.Localization(__file__, 236, 62), getitem___16881, i_16879)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 236)
        i_16883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 112), 'i', False)
        # Getting the type of 'type_store_finally' (line 236)
        type_store_finally_16884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 93), 'type_store_finally', False)
        # Obtaining the member '__getitem__' of a type (line 236)
        getitem___16885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 93), type_store_finally_16884, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 236)
        subscript_call_result_16886 = invoke(stypy.reporting.localization.Localization(__file__, 236, 93), getitem___16885, i_16883)
        
        # Processing the call keyword arguments (line 236)
        kwargs_16887 = {}
        # Getting the type of '__join_finally_function_context' (line 236)
        join_finally_function_context_16878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 30), '__join_finally_function_context', False)
        # Calling __join_finally_function_context(args, kwargs) (line 236)
        join_finally_function_context_call_result_16888 = invoke(stypy.reporting.localization.Localization(__file__, 236, 30), join_finally_function_context_16878, *[subscript_call_result_16882, subscript_call_result_16886], **kwargs_16887)
        
        # Assigning a type to the variable 'joined_context_dict' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'joined_context_dict', join_finally_function_context_call_result_16888)
        
        # Assigning a Call to a Name (line 238):
        
        # Call to copy(...): (line 238)
        # Processing the call keyword arguments (line 238)
        kwargs_16894 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 238)
        i_16889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 54), 'i', False)
        # Getting the type of 'type_store_exception_block' (line 238)
        type_store_exception_block_16890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 27), 'type_store_exception_block', False)
        # Obtaining the member '__getitem__' of a type (line 238)
        getitem___16891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 27), type_store_exception_block_16890, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 238)
        subscript_call_result_16892 = invoke(stypy.reporting.localization.Localization(__file__, 238, 27), getitem___16891, i_16889)
        
        # Obtaining the member 'copy' of a type (line 238)
        copy_16893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 27), subscript_call_result_16892, 'copy')
        # Calling copy(args, kwargs) (line 238)
        copy_call_result_16895 = invoke(stypy.reporting.localization.Localization(__file__, 238, 27), copy_16893, *[], **kwargs_16894)
        
        # Assigning a type to the variable 'joined_f_context' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'joined_f_context', copy_call_result_16895)
        
        # Assigning a Name to a Attribute (line 239):
        # Getting the type of 'joined_context_dict' (line 239)
        joined_context_dict_16896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 36), 'joined_context_dict')
        # Getting the type of 'joined_f_context' (line 239)
        joined_f_context_16897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'joined_f_context')
        # Setting the type of the member 'types_of' of a type (line 239)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), joined_f_context_16897, 'types_of', joined_context_dict_16896)
        
        # Assigning a Call to a Attribute (line 240):
        
        # Call to __join_globals(...): (line 240)
        # Processing the call arguments (line 240)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 240)
        i_16899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 81), 'i', False)
        # Getting the type of 'type_store_exception_block' (line 240)
        type_store_exception_block_16900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 54), 'type_store_exception_block', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___16901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 54), type_store_exception_block_16900, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 240)
        subscript_call_result_16902 = invoke(stypy.reporting.localization.Localization(__file__, 240, 54), getitem___16901, i_16899)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 240)
        i_16903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 104), 'i', False)
        # Getting the type of 'type_store_finally' (line 240)
        type_store_finally_16904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 85), 'type_store_finally', False)
        # Obtaining the member '__getitem__' of a type (line 240)
        getitem___16905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 85), type_store_finally_16904, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 240)
        subscript_call_result_16906 = invoke(stypy.reporting.localization.Localization(__file__, 240, 85), getitem___16905, i_16903)
        
        # Processing the call keyword arguments (line 240)
        kwargs_16907 = {}
        # Getting the type of '__join_globals' (line 240)
        join_globals_16898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 39), '__join_globals', False)
        # Calling __join_globals(args, kwargs) (line 240)
        join_globals_call_result_16908 = invoke(stypy.reporting.localization.Localization(__file__, 240, 39), join_globals_16898, *[subscript_call_result_16902, subscript_call_result_16906], **kwargs_16907)
        
        # Getting the type of 'joined_f_context' (line 240)
        joined_f_context_16909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'joined_f_context')
        # Setting the type of the member 'global_vars' of a type (line 240)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 8), joined_f_context_16909, 'global_vars', join_globals_call_result_16908)
        
        # Assigning a Attribute to a Attribute (line 241):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 242)
        i_16910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'i')
        # Getting the type of 'type_store_exception_block' (line 241)
        type_store_exception_block_16911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 45), 'type_store_exception_block')
        # Obtaining the member '__getitem__' of a type (line 241)
        getitem___16912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 45), type_store_exception_block_16911, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 241)
        subscript_call_result_16913 = invoke(stypy.reporting.localization.Localization(__file__, 241, 45), getitem___16912, i_16910)
        
        # Obtaining the member 'annotation_record' of a type (line 241)
        annotation_record_16914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 45), subscript_call_result_16913, 'annotation_record')
        # Getting the type of 'joined_f_context' (line 241)
        joined_f_context_16915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'joined_f_context')
        # Setting the type of the member 'annotation_record' of a type (line 241)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 8), joined_f_context_16915, 'annotation_record', annotation_record_16914)
        
        # Call to append(...): (line 244)
        # Processing the call arguments (line 244)
        # Getting the type of 'joined_f_context' (line 244)
        joined_f_context_16919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 47), 'joined_f_context', False)
        # Processing the call keyword arguments (line 244)
        kwargs_16920 = {}
        # Getting the type of 'joined_type_store' (line 244)
        joined_type_store_16916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'joined_type_store', False)
        # Obtaining the member 'context_stack' of a type (line 244)
        context_stack_16917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), joined_type_store_16916, 'context_stack')
        # Obtaining the member 'append' of a type (line 244)
        append_16918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 8), context_stack_16917, 'append')
        # Calling append(args, kwargs) (line 244)
        append_call_result_16921 = invoke(stypy.reporting.localization.Localization(__file__, 244, 8), append_16918, *[joined_f_context_16919], **kwargs_16920)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'joined_type_store' (line 246)
    joined_type_store_16922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 11), 'joined_type_store')
    # Assigning a type to the variable 'stypy_return_type' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type', joined_type_store_16922)
    
    # ################# End of '__join_finally_branch(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join_finally_branch' in the type store
    # Getting the type of 'stypy_return_type' (line 224)
    stypy_return_type_16923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16923)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join_finally_branch'
    return stypy_return_type_16923

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

    str_16924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, (-1)), 'str', '\n    Implements the SSA algorithm in try-except blocks, dealing with function contexts.\n\n    :param function_context_previous: Function context\n    :param function_context_try: Function context\n    :param function_context_except: Function context\n    :return: A dictionary with names of variables and its joined types\n    ')
    
    # Assigning a Dict to a Name (line 258):
    
    # Obtaining an instance of the builtin type 'dict' (line 258)
    dict_16925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 16), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 258)
    
    # Assigning a type to the variable 'type_dict' (line 258)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 4), 'type_dict', dict_16925)
    
    # Getting the type of 'function_context_try' (line 260)
    function_context_try_16926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 20), 'function_context_try')
    # Assigning a type to the variable 'function_context_try_16926' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'function_context_try_16926', function_context_try_16926)
    # Testing if the for loop is going to be iterated (line 260)
    # Testing the type of a for loop iterable (line 260)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 260, 4), function_context_try_16926)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 260, 4), function_context_try_16926):
        # Getting the type of the for loop variable (line 260)
        for_loop_var_16927 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 260, 4), function_context_try_16926)
        # Assigning a type to the variable 'var_name' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'var_name', for_loop_var_16927)
        # SSA begins for a for statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 261)
        var_name_16928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 11), 'var_name')
        # Getting the type of 'function_context_except' (line 261)
        function_context_except_16929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 23), 'function_context_except')
        # Applying the binary operator 'in' (line 261)
        result_contains_16930 = python_operator(stypy.reporting.localization.Localization(__file__, 261, 11), 'in', var_name_16928, function_context_except_16929)
        
        # Testing if the type of an if condition is none (line 261)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 261, 8), result_contains_16930):
            
            # Getting the type of 'var_name' (line 269)
            var_name_16965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 269)
            function_context_previous_16966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 269)
            result_contains_16967 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 15), 'in', var_name_16965, function_context_previous_16966)
            
            # Testing if the type of an if condition is none (line 269)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 269, 12), result_contains_16967):
                
                # Assigning a Call to a Subscript (line 275):
                
                # Call to add(...): (line 275)
                # Processing the call arguments (line 275)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 275)
                var_name_16987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 89), 'var_name', False)
                # Getting the type of 'function_context_try' (line 275)
                function_context_try_16988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 68), 'function_context_try', False)
                # Obtaining the member '__getitem__' of a type (line 275)
                getitem___16989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 68), function_context_try_16988, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 275)
                subscript_call_result_16990 = invoke(stypy.reporting.localization.Localization(__file__, 275, 68), getitem___16989, var_name_16987)
                
                
                # Call to UndefinedType(...): (line 275)
                # Processing the call keyword arguments (line 275)
                kwargs_16992 = {}
                # Getting the type of 'UndefinedType' (line 275)
                UndefinedType_16991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 100), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 275)
                UndefinedType_call_result_16993 = invoke(stypy.reporting.localization.Localization(__file__, 275, 100), UndefinedType_16991, *[], **kwargs_16992)
                
                # Processing the call keyword arguments (line 275)
                kwargs_16994 = {}
                # Getting the type of 'union_type_copy' (line 275)
                union_type_copy_16984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 275)
                UnionType_16985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), union_type_copy_16984, 'UnionType')
                # Obtaining the member 'add' of a type (line 275)
                add_16986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), UnionType_16985, 'add')
                # Calling add(args, kwargs) (line 275)
                add_call_result_16995 = invoke(stypy.reporting.localization.Localization(__file__, 275, 38), add_16986, *[subscript_call_result_16990, UndefinedType_call_result_16993], **kwargs_16994)
                
                # Getting the type of 'type_dict' (line 275)
                type_dict_16996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'type_dict')
                # Getting the type of 'var_name' (line 275)
                var_name_16997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'var_name')
                # Storing an element on a container (line 275)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 16), type_dict_16996, (var_name_16997, add_call_result_16995))
            else:
                
                # Testing the type of an if condition (line 269)
                if_condition_16968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 12), result_contains_16967)
                # Assigning a type to the variable 'if_condition_16968' (line 269)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'if_condition_16968', if_condition_16968)
                # SSA begins for if statement (line 269)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 271):
                
                # Call to add(...): (line 271)
                # Processing the call arguments (line 271)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 271)
                var_name_16972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 271)
                function_context_previous_16973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 271)
                getitem___16974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 68), function_context_previous_16973, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 271)
                subscript_call_result_16975 = invoke(stypy.reporting.localization.Localization(__file__, 271, 68), getitem___16974, var_name_16972)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 272)
                var_name_16976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 84), 'var_name', False)
                # Getting the type of 'function_context_try' (line 272)
                function_context_try_16977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 63), 'function_context_try', False)
                # Obtaining the member '__getitem__' of a type (line 272)
                getitem___16978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 63), function_context_try_16977, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 272)
                subscript_call_result_16979 = invoke(stypy.reporting.localization.Localization(__file__, 272, 63), getitem___16978, var_name_16976)
                
                # Processing the call keyword arguments (line 271)
                kwargs_16980 = {}
                # Getting the type of 'union_type_copy' (line 271)
                union_type_copy_16969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 271)
                UnionType_16970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 38), union_type_copy_16969, 'UnionType')
                # Obtaining the member 'add' of a type (line 271)
                add_16971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 38), UnionType_16970, 'add')
                # Calling add(args, kwargs) (line 271)
                add_call_result_16981 = invoke(stypy.reporting.localization.Localization(__file__, 271, 38), add_16971, *[subscript_call_result_16975, subscript_call_result_16979], **kwargs_16980)
                
                # Getting the type of 'type_dict' (line 271)
                type_dict_16982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'type_dict')
                # Getting the type of 'var_name' (line 271)
                var_name_16983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 26), 'var_name')
                # Storing an element on a container (line 271)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 16), type_dict_16982, (var_name_16983, add_call_result_16981))
                # SSA branch for the else part of an if statement (line 269)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 275):
                
                # Call to add(...): (line 275)
                # Processing the call arguments (line 275)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 275)
                var_name_16987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 89), 'var_name', False)
                # Getting the type of 'function_context_try' (line 275)
                function_context_try_16988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 68), 'function_context_try', False)
                # Obtaining the member '__getitem__' of a type (line 275)
                getitem___16989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 68), function_context_try_16988, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 275)
                subscript_call_result_16990 = invoke(stypy.reporting.localization.Localization(__file__, 275, 68), getitem___16989, var_name_16987)
                
                
                # Call to UndefinedType(...): (line 275)
                # Processing the call keyword arguments (line 275)
                kwargs_16992 = {}
                # Getting the type of 'UndefinedType' (line 275)
                UndefinedType_16991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 100), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 275)
                UndefinedType_call_result_16993 = invoke(stypy.reporting.localization.Localization(__file__, 275, 100), UndefinedType_16991, *[], **kwargs_16992)
                
                # Processing the call keyword arguments (line 275)
                kwargs_16994 = {}
                # Getting the type of 'union_type_copy' (line 275)
                union_type_copy_16984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 275)
                UnionType_16985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), union_type_copy_16984, 'UnionType')
                # Obtaining the member 'add' of a type (line 275)
                add_16986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), UnionType_16985, 'add')
                # Calling add(args, kwargs) (line 275)
                add_call_result_16995 = invoke(stypy.reporting.localization.Localization(__file__, 275, 38), add_16986, *[subscript_call_result_16990, UndefinedType_call_result_16993], **kwargs_16994)
                
                # Getting the type of 'type_dict' (line 275)
                type_dict_16996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'type_dict')
                # Getting the type of 'var_name' (line 275)
                var_name_16997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'var_name')
                # Storing an element on a container (line 275)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 16), type_dict_16996, (var_name_16997, add_call_result_16995))
                # SSA join for if statement (line 269)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 261)
            if_condition_16931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 261, 8), result_contains_16930)
            # Assigning a type to the variable 'if_condition_16931' (line 261)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'if_condition_16931', if_condition_16931)
            # SSA begins for if statement (line 261)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 263):
            
            # Call to add(...): (line 263)
            # Processing the call arguments (line 263)
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 263)
            var_name_16935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 85), 'var_name', False)
            # Getting the type of 'function_context_try' (line 263)
            function_context_try_16936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 64), 'function_context_try', False)
            # Obtaining the member '__getitem__' of a type (line 263)
            getitem___16937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 64), function_context_try_16936, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 263)
            subscript_call_result_16938 = invoke(stypy.reporting.localization.Localization(__file__, 263, 64), getitem___16937, var_name_16935)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'var_name' (line 264)
            var_name_16939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 83), 'var_name', False)
            # Getting the type of 'function_context_except' (line 264)
            function_context_except_16940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 59), 'function_context_except', False)
            # Obtaining the member '__getitem__' of a type (line 264)
            getitem___16941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 59), function_context_except_16940, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 264)
            subscript_call_result_16942 = invoke(stypy.reporting.localization.Localization(__file__, 264, 59), getitem___16941, var_name_16939)
            
            # Processing the call keyword arguments (line 263)
            kwargs_16943 = {}
            # Getting the type of 'union_type_copy' (line 263)
            union_type_copy_16932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 34), 'union_type_copy', False)
            # Obtaining the member 'UnionType' of a type (line 263)
            UnionType_16933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 34), union_type_copy_16932, 'UnionType')
            # Obtaining the member 'add' of a type (line 263)
            add_16934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 34), UnionType_16933, 'add')
            # Calling add(args, kwargs) (line 263)
            add_call_result_16944 = invoke(stypy.reporting.localization.Localization(__file__, 263, 34), add_16934, *[subscript_call_result_16938, subscript_call_result_16942], **kwargs_16943)
            
            # Getting the type of 'type_dict' (line 263)
            type_dict_16945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 12), 'type_dict')
            # Getting the type of 'var_name' (line 263)
            var_name_16946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 22), 'var_name')
            # Storing an element on a container (line 263)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 263, 12), type_dict_16945, (var_name_16946, add_call_result_16944))
            
            # Getting the type of 'var_name' (line 265)
            var_name_16947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 265)
            function_context_previous_16948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 31), 'function_context_previous')
            # Applying the binary operator 'notin' (line 265)
            result_contains_16949 = python_operator(stypy.reporting.localization.Localization(__file__, 265, 15), 'notin', var_name_16947, function_context_previous_16948)
            
            # Testing if the type of an if condition is none (line 265)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 265, 12), result_contains_16949):
                pass
            else:
                
                # Testing the type of an if condition (line 265)
                if_condition_16950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 265, 12), result_contains_16949)
                # Assigning a type to the variable 'if_condition_16950' (line 265)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 12), 'if_condition_16950', if_condition_16950)
                # SSA begins for if statement (line 265)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 266):
                
                # Call to add(...): (line 266)
                # Processing the call arguments (line 266)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 266)
                var_name_16954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 78), 'var_name', False)
                # Getting the type of 'type_dict' (line 266)
                type_dict_16955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 68), 'type_dict', False)
                # Obtaining the member '__getitem__' of a type (line 266)
                getitem___16956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 68), type_dict_16955, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 266)
                subscript_call_result_16957 = invoke(stypy.reporting.localization.Localization(__file__, 266, 68), getitem___16956, var_name_16954)
                
                
                # Call to UndefinedType(...): (line 266)
                # Processing the call keyword arguments (line 266)
                kwargs_16959 = {}
                # Getting the type of 'UndefinedType' (line 266)
                UndefinedType_16958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 89), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 266)
                UndefinedType_call_result_16960 = invoke(stypy.reporting.localization.Localization(__file__, 266, 89), UndefinedType_16958, *[], **kwargs_16959)
                
                # Processing the call keyword arguments (line 266)
                kwargs_16961 = {}
                # Getting the type of 'union_type_copy' (line 266)
                union_type_copy_16951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 266)
                UnionType_16952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 38), union_type_copy_16951, 'UnionType')
                # Obtaining the member 'add' of a type (line 266)
                add_16953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 38), UnionType_16952, 'add')
                # Calling add(args, kwargs) (line 266)
                add_call_result_16962 = invoke(stypy.reporting.localization.Localization(__file__, 266, 38), add_16953, *[subscript_call_result_16957, UndefinedType_call_result_16960], **kwargs_16961)
                
                # Getting the type of 'type_dict' (line 266)
                type_dict_16963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 16), 'type_dict')
                # Getting the type of 'var_name' (line 266)
                var_name_16964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 26), 'var_name')
                # Storing an element on a container (line 266)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 266, 16), type_dict_16963, (var_name_16964, add_call_result_16962))
                # SSA join for if statement (line 265)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA branch for the else part of an if statement (line 261)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'var_name' (line 269)
            var_name_16965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 269)
            function_context_previous_16966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 269)
            result_contains_16967 = python_operator(stypy.reporting.localization.Localization(__file__, 269, 15), 'in', var_name_16965, function_context_previous_16966)
            
            # Testing if the type of an if condition is none (line 269)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 269, 12), result_contains_16967):
                
                # Assigning a Call to a Subscript (line 275):
                
                # Call to add(...): (line 275)
                # Processing the call arguments (line 275)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 275)
                var_name_16987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 89), 'var_name', False)
                # Getting the type of 'function_context_try' (line 275)
                function_context_try_16988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 68), 'function_context_try', False)
                # Obtaining the member '__getitem__' of a type (line 275)
                getitem___16989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 68), function_context_try_16988, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 275)
                subscript_call_result_16990 = invoke(stypy.reporting.localization.Localization(__file__, 275, 68), getitem___16989, var_name_16987)
                
                
                # Call to UndefinedType(...): (line 275)
                # Processing the call keyword arguments (line 275)
                kwargs_16992 = {}
                # Getting the type of 'UndefinedType' (line 275)
                UndefinedType_16991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 100), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 275)
                UndefinedType_call_result_16993 = invoke(stypy.reporting.localization.Localization(__file__, 275, 100), UndefinedType_16991, *[], **kwargs_16992)
                
                # Processing the call keyword arguments (line 275)
                kwargs_16994 = {}
                # Getting the type of 'union_type_copy' (line 275)
                union_type_copy_16984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 275)
                UnionType_16985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), union_type_copy_16984, 'UnionType')
                # Obtaining the member 'add' of a type (line 275)
                add_16986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), UnionType_16985, 'add')
                # Calling add(args, kwargs) (line 275)
                add_call_result_16995 = invoke(stypy.reporting.localization.Localization(__file__, 275, 38), add_16986, *[subscript_call_result_16990, UndefinedType_call_result_16993], **kwargs_16994)
                
                # Getting the type of 'type_dict' (line 275)
                type_dict_16996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'type_dict')
                # Getting the type of 'var_name' (line 275)
                var_name_16997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'var_name')
                # Storing an element on a container (line 275)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 16), type_dict_16996, (var_name_16997, add_call_result_16995))
            else:
                
                # Testing the type of an if condition (line 269)
                if_condition_16968 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 269, 12), result_contains_16967)
                # Assigning a type to the variable 'if_condition_16968' (line 269)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 269, 12), 'if_condition_16968', if_condition_16968)
                # SSA begins for if statement (line 269)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 271):
                
                # Call to add(...): (line 271)
                # Processing the call arguments (line 271)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 271)
                var_name_16972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 271)
                function_context_previous_16973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 271)
                getitem___16974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 68), function_context_previous_16973, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 271)
                subscript_call_result_16975 = invoke(stypy.reporting.localization.Localization(__file__, 271, 68), getitem___16974, var_name_16972)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 272)
                var_name_16976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 84), 'var_name', False)
                # Getting the type of 'function_context_try' (line 272)
                function_context_try_16977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 63), 'function_context_try', False)
                # Obtaining the member '__getitem__' of a type (line 272)
                getitem___16978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 63), function_context_try_16977, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 272)
                subscript_call_result_16979 = invoke(stypy.reporting.localization.Localization(__file__, 272, 63), getitem___16978, var_name_16976)
                
                # Processing the call keyword arguments (line 271)
                kwargs_16980 = {}
                # Getting the type of 'union_type_copy' (line 271)
                union_type_copy_16969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 271)
                UnionType_16970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 38), union_type_copy_16969, 'UnionType')
                # Obtaining the member 'add' of a type (line 271)
                add_16971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 38), UnionType_16970, 'add')
                # Calling add(args, kwargs) (line 271)
                add_call_result_16981 = invoke(stypy.reporting.localization.Localization(__file__, 271, 38), add_16971, *[subscript_call_result_16975, subscript_call_result_16979], **kwargs_16980)
                
                # Getting the type of 'type_dict' (line 271)
                type_dict_16982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 16), 'type_dict')
                # Getting the type of 'var_name' (line 271)
                var_name_16983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 26), 'var_name')
                # Storing an element on a container (line 271)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 271, 16), type_dict_16982, (var_name_16983, add_call_result_16981))
                # SSA branch for the else part of an if statement (line 269)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 275):
                
                # Call to add(...): (line 275)
                # Processing the call arguments (line 275)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 275)
                var_name_16987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 89), 'var_name', False)
                # Getting the type of 'function_context_try' (line 275)
                function_context_try_16988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 68), 'function_context_try', False)
                # Obtaining the member '__getitem__' of a type (line 275)
                getitem___16989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 68), function_context_try_16988, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 275)
                subscript_call_result_16990 = invoke(stypy.reporting.localization.Localization(__file__, 275, 68), getitem___16989, var_name_16987)
                
                
                # Call to UndefinedType(...): (line 275)
                # Processing the call keyword arguments (line 275)
                kwargs_16992 = {}
                # Getting the type of 'UndefinedType' (line 275)
                UndefinedType_16991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 100), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 275)
                UndefinedType_call_result_16993 = invoke(stypy.reporting.localization.Localization(__file__, 275, 100), UndefinedType_16991, *[], **kwargs_16992)
                
                # Processing the call keyword arguments (line 275)
                kwargs_16994 = {}
                # Getting the type of 'union_type_copy' (line 275)
                union_type_copy_16984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 275)
                UnionType_16985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), union_type_copy_16984, 'UnionType')
                # Obtaining the member 'add' of a type (line 275)
                add_16986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 38), UnionType_16985, 'add')
                # Calling add(args, kwargs) (line 275)
                add_call_result_16995 = invoke(stypy.reporting.localization.Localization(__file__, 275, 38), add_16986, *[subscript_call_result_16990, UndefinedType_call_result_16993], **kwargs_16994)
                
                # Getting the type of 'type_dict' (line 275)
                type_dict_16996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 16), 'type_dict')
                # Getting the type of 'var_name' (line 275)
                var_name_16997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 26), 'var_name')
                # Storing an element on a container (line 275)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 275, 16), type_dict_16996, (var_name_16997, add_call_result_16995))
                # SSA join for if statement (line 269)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 261)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'function_context_except' (line 277)
    function_context_except_16998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 277, 20), 'function_context_except')
    # Assigning a type to the variable 'function_context_except_16998' (line 277)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'function_context_except_16998', function_context_except_16998)
    # Testing if the for loop is going to be iterated (line 277)
    # Testing the type of a for loop iterable (line 277)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 277, 4), function_context_except_16998)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 277, 4), function_context_except_16998):
        # Getting the type of the for loop variable (line 277)
        for_loop_var_16999 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 277, 4), function_context_except_16998)
        # Assigning a type to the variable 'var_name' (line 277)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 277, 4), 'var_name', for_loop_var_16999)
        # SSA begins for a for statement (line 277)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'var_name' (line 278)
        var_name_17000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), 'var_name')
        # Getting the type of 'function_context_try' (line 278)
        function_context_try_17001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 23), 'function_context_try')
        # Applying the binary operator 'in' (line 278)
        result_contains_17002 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 11), 'in', var_name_17000, function_context_try_17001)
        
        # Testing if the type of an if condition is none (line 278)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 278, 8), result_contains_17002):
            
            # Getting the type of 'var_name' (line 282)
            var_name_17004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 282)
            function_context_previous_17005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 282)
            result_contains_17006 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 15), 'in', var_name_17004, function_context_previous_17005)
            
            # Testing if the type of an if condition is none (line 282)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 282, 12), result_contains_17006):
                
                # Assigning a Call to a Subscript (line 289):
                
                # Call to add(...): (line 289)
                # Processing the call arguments (line 289)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 289)
                var_name_17026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 92), 'var_name', False)
                # Getting the type of 'function_context_except' (line 289)
                function_context_except_17027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 68), 'function_context_except', False)
                # Obtaining the member '__getitem__' of a type (line 289)
                getitem___17028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 68), function_context_except_17027, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 289)
                subscript_call_result_17029 = invoke(stypy.reporting.localization.Localization(__file__, 289, 68), getitem___17028, var_name_17026)
                
                
                # Call to UndefinedType(...): (line 289)
                # Processing the call keyword arguments (line 289)
                kwargs_17031 = {}
                # Getting the type of 'UndefinedType' (line 289)
                UndefinedType_17030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 103), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 289)
                UndefinedType_call_result_17032 = invoke(stypy.reporting.localization.Localization(__file__, 289, 103), UndefinedType_17030, *[], **kwargs_17031)
                
                # Processing the call keyword arguments (line 289)
                kwargs_17033 = {}
                # Getting the type of 'union_type_copy' (line 289)
                union_type_copy_17023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 289)
                UnionType_17024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), union_type_copy_17023, 'UnionType')
                # Obtaining the member 'add' of a type (line 289)
                add_17025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), UnionType_17024, 'add')
                # Calling add(args, kwargs) (line 289)
                add_call_result_17034 = invoke(stypy.reporting.localization.Localization(__file__, 289, 38), add_17025, *[subscript_call_result_17029, UndefinedType_call_result_17032], **kwargs_17033)
                
                # Getting the type of 'type_dict' (line 289)
                type_dict_17035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'type_dict')
                # Getting the type of 'var_name' (line 289)
                var_name_17036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'var_name')
                # Storing an element on a container (line 289)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 16), type_dict_17035, (var_name_17036, add_call_result_17034))
            else:
                
                # Testing the type of an if condition (line 282)
                if_condition_17007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 12), result_contains_17006)
                # Assigning a type to the variable 'if_condition_17007' (line 282)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'if_condition_17007', if_condition_17007)
                # SSA begins for if statement (line 282)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 284):
                
                # Call to add(...): (line 284)
                # Processing the call arguments (line 284)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 284)
                var_name_17011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 284)
                function_context_previous_17012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 284)
                getitem___17013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 68), function_context_previous_17012, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 284)
                subscript_call_result_17014 = invoke(stypy.reporting.localization.Localization(__file__, 284, 68), getitem___17013, var_name_17011)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 285)
                var_name_17015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 87), 'var_name', False)
                # Getting the type of 'function_context_except' (line 285)
                function_context_except_17016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 63), 'function_context_except', False)
                # Obtaining the member '__getitem__' of a type (line 285)
                getitem___17017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 63), function_context_except_17016, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 285)
                subscript_call_result_17018 = invoke(stypy.reporting.localization.Localization(__file__, 285, 63), getitem___17017, var_name_17015)
                
                # Processing the call keyword arguments (line 284)
                kwargs_17019 = {}
                # Getting the type of 'union_type_copy' (line 284)
                union_type_copy_17008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 284)
                UnionType_17009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 38), union_type_copy_17008, 'UnionType')
                # Obtaining the member 'add' of a type (line 284)
                add_17010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 38), UnionType_17009, 'add')
                # Calling add(args, kwargs) (line 284)
                add_call_result_17020 = invoke(stypy.reporting.localization.Localization(__file__, 284, 38), add_17010, *[subscript_call_result_17014, subscript_call_result_17018], **kwargs_17019)
                
                # Getting the type of 'type_dict' (line 284)
                type_dict_17021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'type_dict')
                # Getting the type of 'var_name' (line 284)
                var_name_17022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 26), 'var_name')
                # Storing an element on a container (line 284)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 16), type_dict_17021, (var_name_17022, add_call_result_17020))
                # SSA branch for the else part of an if statement (line 282)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 289):
                
                # Call to add(...): (line 289)
                # Processing the call arguments (line 289)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 289)
                var_name_17026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 92), 'var_name', False)
                # Getting the type of 'function_context_except' (line 289)
                function_context_except_17027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 68), 'function_context_except', False)
                # Obtaining the member '__getitem__' of a type (line 289)
                getitem___17028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 68), function_context_except_17027, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 289)
                subscript_call_result_17029 = invoke(stypy.reporting.localization.Localization(__file__, 289, 68), getitem___17028, var_name_17026)
                
                
                # Call to UndefinedType(...): (line 289)
                # Processing the call keyword arguments (line 289)
                kwargs_17031 = {}
                # Getting the type of 'UndefinedType' (line 289)
                UndefinedType_17030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 103), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 289)
                UndefinedType_call_result_17032 = invoke(stypy.reporting.localization.Localization(__file__, 289, 103), UndefinedType_17030, *[], **kwargs_17031)
                
                # Processing the call keyword arguments (line 289)
                kwargs_17033 = {}
                # Getting the type of 'union_type_copy' (line 289)
                union_type_copy_17023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 289)
                UnionType_17024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), union_type_copy_17023, 'UnionType')
                # Obtaining the member 'add' of a type (line 289)
                add_17025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), UnionType_17024, 'add')
                # Calling add(args, kwargs) (line 289)
                add_call_result_17034 = invoke(stypy.reporting.localization.Localization(__file__, 289, 38), add_17025, *[subscript_call_result_17029, UndefinedType_call_result_17032], **kwargs_17033)
                
                # Getting the type of 'type_dict' (line 289)
                type_dict_17035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'type_dict')
                # Getting the type of 'var_name' (line 289)
                var_name_17036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'var_name')
                # Storing an element on a container (line 289)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 16), type_dict_17035, (var_name_17036, add_call_result_17034))
                # SSA join for if statement (line 282)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 278)
            if_condition_17003 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 8), result_contains_17002)
            # Assigning a type to the variable 'if_condition_17003' (line 278)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 8), 'if_condition_17003', if_condition_17003)
            # SSA begins for if statement (line 278)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA branch for the else part of an if statement (line 278)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'var_name' (line 282)
            var_name_17004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 15), 'var_name')
            # Getting the type of 'function_context_previous' (line 282)
            function_context_previous_17005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 27), 'function_context_previous')
            # Applying the binary operator 'in' (line 282)
            result_contains_17006 = python_operator(stypy.reporting.localization.Localization(__file__, 282, 15), 'in', var_name_17004, function_context_previous_17005)
            
            # Testing if the type of an if condition is none (line 282)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 282, 12), result_contains_17006):
                
                # Assigning a Call to a Subscript (line 289):
                
                # Call to add(...): (line 289)
                # Processing the call arguments (line 289)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 289)
                var_name_17026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 92), 'var_name', False)
                # Getting the type of 'function_context_except' (line 289)
                function_context_except_17027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 68), 'function_context_except', False)
                # Obtaining the member '__getitem__' of a type (line 289)
                getitem___17028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 68), function_context_except_17027, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 289)
                subscript_call_result_17029 = invoke(stypy.reporting.localization.Localization(__file__, 289, 68), getitem___17028, var_name_17026)
                
                
                # Call to UndefinedType(...): (line 289)
                # Processing the call keyword arguments (line 289)
                kwargs_17031 = {}
                # Getting the type of 'UndefinedType' (line 289)
                UndefinedType_17030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 103), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 289)
                UndefinedType_call_result_17032 = invoke(stypy.reporting.localization.Localization(__file__, 289, 103), UndefinedType_17030, *[], **kwargs_17031)
                
                # Processing the call keyword arguments (line 289)
                kwargs_17033 = {}
                # Getting the type of 'union_type_copy' (line 289)
                union_type_copy_17023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 289)
                UnionType_17024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), union_type_copy_17023, 'UnionType')
                # Obtaining the member 'add' of a type (line 289)
                add_17025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), UnionType_17024, 'add')
                # Calling add(args, kwargs) (line 289)
                add_call_result_17034 = invoke(stypy.reporting.localization.Localization(__file__, 289, 38), add_17025, *[subscript_call_result_17029, UndefinedType_call_result_17032], **kwargs_17033)
                
                # Getting the type of 'type_dict' (line 289)
                type_dict_17035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'type_dict')
                # Getting the type of 'var_name' (line 289)
                var_name_17036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'var_name')
                # Storing an element on a container (line 289)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 16), type_dict_17035, (var_name_17036, add_call_result_17034))
            else:
                
                # Testing the type of an if condition (line 282)
                if_condition_17007 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 282, 12), result_contains_17006)
                # Assigning a type to the variable 'if_condition_17007' (line 282)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'if_condition_17007', if_condition_17007)
                # SSA begins for if statement (line 282)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 284):
                
                # Call to add(...): (line 284)
                # Processing the call arguments (line 284)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 284)
                var_name_17011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 94), 'var_name', False)
                # Getting the type of 'function_context_previous' (line 284)
                function_context_previous_17012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 68), 'function_context_previous', False)
                # Obtaining the member '__getitem__' of a type (line 284)
                getitem___17013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 68), function_context_previous_17012, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 284)
                subscript_call_result_17014 = invoke(stypy.reporting.localization.Localization(__file__, 284, 68), getitem___17013, var_name_17011)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 285)
                var_name_17015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 87), 'var_name', False)
                # Getting the type of 'function_context_except' (line 285)
                function_context_except_17016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 63), 'function_context_except', False)
                # Obtaining the member '__getitem__' of a type (line 285)
                getitem___17017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 63), function_context_except_17016, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 285)
                subscript_call_result_17018 = invoke(stypy.reporting.localization.Localization(__file__, 285, 63), getitem___17017, var_name_17015)
                
                # Processing the call keyword arguments (line 284)
                kwargs_17019 = {}
                # Getting the type of 'union_type_copy' (line 284)
                union_type_copy_17008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 284)
                UnionType_17009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 38), union_type_copy_17008, 'UnionType')
                # Obtaining the member 'add' of a type (line 284)
                add_17010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 284, 38), UnionType_17009, 'add')
                # Calling add(args, kwargs) (line 284)
                add_call_result_17020 = invoke(stypy.reporting.localization.Localization(__file__, 284, 38), add_17010, *[subscript_call_result_17014, subscript_call_result_17018], **kwargs_17019)
                
                # Getting the type of 'type_dict' (line 284)
                type_dict_17021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 16), 'type_dict')
                # Getting the type of 'var_name' (line 284)
                var_name_17022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 26), 'var_name')
                # Storing an element on a container (line 284)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 284, 16), type_dict_17021, (var_name_17022, add_call_result_17020))
                # SSA branch for the else part of an if statement (line 282)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Subscript (line 289):
                
                # Call to add(...): (line 289)
                # Processing the call arguments (line 289)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var_name' (line 289)
                var_name_17026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 92), 'var_name', False)
                # Getting the type of 'function_context_except' (line 289)
                function_context_except_17027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 68), 'function_context_except', False)
                # Obtaining the member '__getitem__' of a type (line 289)
                getitem___17028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 68), function_context_except_17027, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 289)
                subscript_call_result_17029 = invoke(stypy.reporting.localization.Localization(__file__, 289, 68), getitem___17028, var_name_17026)
                
                
                # Call to UndefinedType(...): (line 289)
                # Processing the call keyword arguments (line 289)
                kwargs_17031 = {}
                # Getting the type of 'UndefinedType' (line 289)
                UndefinedType_17030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 103), 'UndefinedType', False)
                # Calling UndefinedType(args, kwargs) (line 289)
                UndefinedType_call_result_17032 = invoke(stypy.reporting.localization.Localization(__file__, 289, 103), UndefinedType_17030, *[], **kwargs_17031)
                
                # Processing the call keyword arguments (line 289)
                kwargs_17033 = {}
                # Getting the type of 'union_type_copy' (line 289)
                union_type_copy_17023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 38), 'union_type_copy', False)
                # Obtaining the member 'UnionType' of a type (line 289)
                UnionType_17024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), union_type_copy_17023, 'UnionType')
                # Obtaining the member 'add' of a type (line 289)
                add_17025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 38), UnionType_17024, 'add')
                # Calling add(args, kwargs) (line 289)
                add_call_result_17034 = invoke(stypy.reporting.localization.Localization(__file__, 289, 38), add_17025, *[subscript_call_result_17029, UndefinedType_call_result_17032], **kwargs_17033)
                
                # Getting the type of 'type_dict' (line 289)
                type_dict_17035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 16), 'type_dict')
                # Getting the type of 'var_name' (line 289)
                var_name_17036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'var_name')
                # Storing an element on a container (line 289)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 16), type_dict_17035, (var_name_17036, add_call_result_17034))
                # SSA join for if statement (line 282)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 278)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'type_dict' (line 292)
    type_dict_17037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 11), 'type_dict')
    # Assigning a type to the variable 'stypy_return_type' (line 292)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'stypy_return_type', type_dict_17037)
    
    # ################# End of '__join_try_except_function_context(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join_try_except_function_context' in the type store
    # Getting the type of 'stypy_return_type' (line 249)
    stypy_return_type_17038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17038)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join_try_except_function_context'
    return stypy_return_type_17038

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

    str_17039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, (-1)), 'str', '\n    SSA Algotihm implementation for type stores in a try-except block\n    :param type_store_previous: Type store\n    :param type_store_posttry: Type store\n    :param type_store_excepts: Type store\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 303):
    
    # Call to TypeStore(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'type_store_previous' (line 303)
    type_store_previous_17041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 34), 'type_store_previous', False)
    # Obtaining the member 'program_name' of a type (line 303)
    program_name_17042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 34), type_store_previous_17041, 'program_name')
    # Processing the call keyword arguments (line 303)
    kwargs_17043 = {}
    # Getting the type of 'TypeStore' (line 303)
    TypeStore_17040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 24), 'TypeStore', False)
    # Calling TypeStore(args, kwargs) (line 303)
    TypeStore_call_result_17044 = invoke(stypy.reporting.localization.Localization(__file__, 303, 24), TypeStore_17040, *[program_name_17042], **kwargs_17043)
    
    # Assigning a type to the variable 'joined_type_store' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'joined_type_store', TypeStore_call_result_17044)
    
    # Assigning a Attribute to a Attribute (line 304):
    # Getting the type of 'type_store_previous' (line 304)
    type_store_previous_17045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 47), 'type_store_previous')
    # Obtaining the member 'last_function_contexts' of a type (line 304)
    last_function_contexts_17046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 47), type_store_previous_17045, 'last_function_contexts')
    # Getting the type of 'joined_type_store' (line 304)
    joined_type_store_17047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'joined_type_store')
    # Setting the type of the member 'last_function_contexts' of a type (line 304)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 4), joined_type_store_17047, 'last_function_contexts', last_function_contexts_17046)
    
    # Assigning a List to a Attribute (line 305):
    
    # Obtaining an instance of the builtin type 'list' (line 305)
    list_17048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 305)
    
    # Getting the type of 'joined_type_store' (line 305)
    joined_type_store_17049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'joined_type_store')
    # Setting the type of the member 'context_stack' of a type (line 305)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 4), joined_type_store_17049, 'context_stack', list_17048)
    
    
    # Call to range(...): (line 306)
    # Processing the call arguments (line 306)
    
    # Call to len(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'type_store_previous' (line 306)
    type_store_previous_17052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 23), 'type_store_previous', False)
    # Obtaining the member 'context_stack' of a type (line 306)
    context_stack_17053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 23), type_store_previous_17052, 'context_stack')
    # Processing the call keyword arguments (line 306)
    kwargs_17054 = {}
    # Getting the type of 'len' (line 306)
    len_17051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 19), 'len', False)
    # Calling len(args, kwargs) (line 306)
    len_call_result_17055 = invoke(stypy.reporting.localization.Localization(__file__, 306, 19), len_17051, *[context_stack_17053], **kwargs_17054)
    
    # Processing the call keyword arguments (line 306)
    kwargs_17056 = {}
    # Getting the type of 'range' (line 306)
    range_17050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 13), 'range', False)
    # Calling range(args, kwargs) (line 306)
    range_call_result_17057 = invoke(stypy.reporting.localization.Localization(__file__, 306, 13), range_17050, *[len_call_result_17055], **kwargs_17056)
    
    # Assigning a type to the variable 'range_call_result_17057' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'range_call_result_17057', range_call_result_17057)
    # Testing if the for loop is going to be iterated (line 306)
    # Testing the type of a for loop iterable (line 306)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 306, 4), range_call_result_17057)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 306, 4), range_call_result_17057):
        # Getting the type of the for loop variable (line 306)
        for_loop_var_17058 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 306, 4), range_call_result_17057)
        # Assigning a type to the variable 'i' (line 306)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'i', for_loop_var_17058)
        # SSA begins for a for statement (line 306)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 307):
        
        # Call to __join_try_except_function_context(...): (line 307)
        # Processing the call arguments (line 307)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 307)
        i_17060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 85), 'i', False)
        # Getting the type of 'type_store_previous' (line 307)
        type_store_previous_17061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 65), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___17062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 65), type_store_previous_17061, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_17063 = invoke(stypy.reporting.localization.Localization(__file__, 307, 65), getitem___17062, i_17060)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 307)
        i_17064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 108), 'i', False)
        # Getting the type of 'type_store_posttry' (line 307)
        type_store_posttry_17065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 89), 'type_store_posttry', False)
        # Obtaining the member '__getitem__' of a type (line 307)
        getitem___17066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 89), type_store_posttry_17065, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 307)
        subscript_call_result_17067 = invoke(stypy.reporting.localization.Localization(__file__, 307, 89), getitem___17066, i_17064)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 308)
        i_17068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 84), 'i', False)
        # Getting the type of 'type_store_excepts' (line 308)
        type_store_excepts_17069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 65), 'type_store_excepts', False)
        # Obtaining the member '__getitem__' of a type (line 308)
        getitem___17070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 65), type_store_excepts_17069, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 308)
        subscript_call_result_17071 = invoke(stypy.reporting.localization.Localization(__file__, 308, 65), getitem___17070, i_17068)
        
        # Processing the call keyword arguments (line 307)
        kwargs_17072 = {}
        # Getting the type of '__join_try_except_function_context' (line 307)
        join_try_except_function_context_17059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 30), '__join_try_except_function_context', False)
        # Calling __join_try_except_function_context(args, kwargs) (line 307)
        join_try_except_function_context_call_result_17073 = invoke(stypy.reporting.localization.Localization(__file__, 307, 30), join_try_except_function_context_17059, *[subscript_call_result_17063, subscript_call_result_17067, subscript_call_result_17071], **kwargs_17072)
        
        # Assigning a type to the variable 'joined_context_dict' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'joined_context_dict', join_try_except_function_context_call_result_17073)
        
        # Assigning a Call to a Name (line 310):
        
        # Call to copy(...): (line 310)
        # Processing the call keyword arguments (line 310)
        kwargs_17079 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 310)
        i_17074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 47), 'i', False)
        # Getting the type of 'type_store_previous' (line 310)
        type_store_previous_17075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 27), 'type_store_previous', False)
        # Obtaining the member '__getitem__' of a type (line 310)
        getitem___17076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 27), type_store_previous_17075, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 310)
        subscript_call_result_17077 = invoke(stypy.reporting.localization.Localization(__file__, 310, 27), getitem___17076, i_17074)
        
        # Obtaining the member 'copy' of a type (line 310)
        copy_17078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 27), subscript_call_result_17077, 'copy')
        # Calling copy(args, kwargs) (line 310)
        copy_call_result_17080 = invoke(stypy.reporting.localization.Localization(__file__, 310, 27), copy_17078, *[], **kwargs_17079)
        
        # Assigning a type to the variable 'joined_f_context' (line 310)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 8), 'joined_f_context', copy_call_result_17080)
        
        # Assigning a Name to a Attribute (line 311):
        # Getting the type of 'joined_context_dict' (line 311)
        joined_context_dict_17081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 36), 'joined_context_dict')
        # Getting the type of 'joined_f_context' (line 311)
        joined_f_context_17082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 311, 8), 'joined_f_context')
        # Setting the type of the member 'types_of' of a type (line 311)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 311, 8), joined_f_context_17082, 'types_of', joined_context_dict_17081)
        
        # Assigning a Call to a Attribute (line 312):
        
        # Call to __join_globals(...): (line 312)
        # Processing the call arguments (line 312)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 312)
        i_17084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 73), 'i', False)
        # Getting the type of 'type_store_posttry' (line 312)
        type_store_posttry_17085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 54), 'type_store_posttry', False)
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___17086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 54), type_store_posttry_17085, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_17087 = invoke(stypy.reporting.localization.Localization(__file__, 312, 54), getitem___17086, i_17084)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 312)
        i_17088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 96), 'i', False)
        # Getting the type of 'type_store_excepts' (line 312)
        type_store_excepts_17089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 77), 'type_store_excepts', False)
        # Obtaining the member '__getitem__' of a type (line 312)
        getitem___17090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 77), type_store_excepts_17089, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 312)
        subscript_call_result_17091 = invoke(stypy.reporting.localization.Localization(__file__, 312, 77), getitem___17090, i_17088)
        
        # Processing the call keyword arguments (line 312)
        kwargs_17092 = {}
        # Getting the type of '__join_globals' (line 312)
        join_globals_17083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 39), '__join_globals', False)
        # Calling __join_globals(args, kwargs) (line 312)
        join_globals_call_result_17093 = invoke(stypy.reporting.localization.Localization(__file__, 312, 39), join_globals_17083, *[subscript_call_result_17087, subscript_call_result_17091], **kwargs_17092)
        
        # Getting the type of 'joined_f_context' (line 312)
        joined_f_context_17094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 8), 'joined_f_context')
        # Setting the type of the member 'global_vars' of a type (line 312)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 8), joined_f_context_17094, 'global_vars', join_globals_call_result_17093)
        
        # Assigning a Attribute to a Attribute (line 313):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 314)
        i_17095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 12), 'i')
        # Getting the type of 'type_store_posttry' (line 313)
        type_store_posttry_17096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 45), 'type_store_posttry')
        # Obtaining the member '__getitem__' of a type (line 313)
        getitem___17097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 45), type_store_posttry_17096, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 313)
        subscript_call_result_17098 = invoke(stypy.reporting.localization.Localization(__file__, 313, 45), getitem___17097, i_17095)
        
        # Obtaining the member 'annotation_record' of a type (line 313)
        annotation_record_17099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 45), subscript_call_result_17098, 'annotation_record')
        # Getting the type of 'joined_f_context' (line 313)
        joined_f_context_17100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 8), 'joined_f_context')
        # Setting the type of the member 'annotation_record' of a type (line 313)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 8), joined_f_context_17100, 'annotation_record', annotation_record_17099)
        
        # Call to append(...): (line 316)
        # Processing the call arguments (line 316)
        # Getting the type of 'joined_f_context' (line 316)
        joined_f_context_17104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 47), 'joined_f_context', False)
        # Processing the call keyword arguments (line 316)
        kwargs_17105 = {}
        # Getting the type of 'joined_type_store' (line 316)
        joined_type_store_17101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 316, 8), 'joined_type_store', False)
        # Obtaining the member 'context_stack' of a type (line 316)
        context_stack_17102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), joined_type_store_17101, 'context_stack')
        # Obtaining the member 'append' of a type (line 316)
        append_17103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 316, 8), context_stack_17102, 'append')
        # Calling append(args, kwargs) (line 316)
        append_call_result_17106 = invoke(stypy.reporting.localization.Localization(__file__, 316, 8), append_17103, *[joined_f_context_17104], **kwargs_17105)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'joined_type_store' (line 318)
    joined_type_store_17107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 11), 'joined_type_store')
    # Assigning a type to the variable 'stypy_return_type' (line 318)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 4), 'stypy_return_type', joined_type_store_17107)
    
    # ################# End of '__join__try_except(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__join__try_except' in the type store
    # Getting the type of 'stypy_return_type' (line 295)
    stypy_return_type_17108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17108)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__join__try_except'
    return stypy_return_type_17108

# Assigning a type to the variable '__join__try_except' (line 295)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 0), '__join__try_except', __join__try_except)

@norecursion
def join_exception_block(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 321)
    None_17109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 83), 'None')
    defaults = [None_17109]
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

    str_17110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 329, (-1)), 'str', '\n    Implements the SSA algorithm for a full try-except-finally block, calling previous function\n    :param type_store_pretry: Type store\n    :param type_store_posttry: Type store\n    :param type_store_finally: Type store\n    :param type_store_except_branches: Type store\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 333):
    
    # Call to TypeStore(...): (line 333)
    # Processing the call arguments (line 333)
    # Getting the type of 'type_store_pretry' (line 333)
    type_store_pretry_17112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 34), 'type_store_pretry', False)
    # Obtaining the member 'program_name' of a type (line 333)
    program_name_17113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 34), type_store_pretry_17112, 'program_name')
    # Processing the call keyword arguments (line 333)
    kwargs_17114 = {}
    # Getting the type of 'TypeStore' (line 333)
    TypeStore_17111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 24), 'TypeStore', False)
    # Calling TypeStore(args, kwargs) (line 333)
    TypeStore_call_result_17115 = invoke(stypy.reporting.localization.Localization(__file__, 333, 24), TypeStore_17111, *[program_name_17113], **kwargs_17114)
    
    # Assigning a type to the variable 'joined_type_store' (line 333)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'joined_type_store', TypeStore_call_result_17115)
    
    # Assigning a Attribute to a Attribute (line 334):
    # Getting the type of 'type_store_pretry' (line 334)
    type_store_pretry_17116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 47), 'type_store_pretry')
    # Obtaining the member 'last_function_contexts' of a type (line 334)
    last_function_contexts_17117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 47), type_store_pretry_17116, 'last_function_contexts')
    # Getting the type of 'joined_type_store' (line 334)
    joined_type_store_17118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 4), 'joined_type_store')
    # Setting the type of the member 'last_function_contexts' of a type (line 334)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 4), joined_type_store_17118, 'last_function_contexts', last_function_contexts_17117)
    
    # Assigning a List to a Attribute (line 335):
    
    # Obtaining an instance of the builtin type 'list' (line 335)
    list_17119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 335, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 335)
    
    # Getting the type of 'joined_type_store' (line 335)
    joined_type_store_17120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 4), 'joined_type_store')
    # Setting the type of the member 'context_stack' of a type (line 335)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 4), joined_type_store_17120, 'context_stack', list_17119)
    
    
    # Call to len(...): (line 338)
    # Processing the call arguments (line 338)
    # Getting the type of 'type_store_except_branches' (line 338)
    type_store_except_branches_17122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 11), 'type_store_except_branches', False)
    # Processing the call keyword arguments (line 338)
    kwargs_17123 = {}
    # Getting the type of 'len' (line 338)
    len_17121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 7), 'len', False)
    # Calling len(args, kwargs) (line 338)
    len_call_result_17124 = invoke(stypy.reporting.localization.Localization(__file__, 338, 7), len_17121, *[type_store_except_branches_17122], **kwargs_17123)
    
    int_17125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 338, 42), 'int')
    # Applying the binary operator '==' (line 338)
    result_eq_17126 = python_operator(stypy.reporting.localization.Localization(__file__, 338, 7), '==', len_call_result_17124, int_17125)
    
    # Testing if the type of an if condition is none (line 338)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 338, 4), result_eq_17126):
        
        # Assigning a Num to a Name (line 341):
        int_17132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 15), 'int')
        # Assigning a type to the variable 'cont' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'cont', int_17132)
        
        # Assigning a Subscript to a Name (line 342):
        
        # Obtaining the type of the subscript
        int_17133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 56), 'int')
        # Getting the type of 'type_store_except_branches' (line 342)
        type_store_except_branches_17134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 29), 'type_store_except_branches')
        # Obtaining the member '__getitem__' of a type (line 342)
        getitem___17135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 29), type_store_except_branches_17134, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 342)
        subscript_call_result_17136 = invoke(stypy.reporting.localization.Localization(__file__, 342, 29), getitem___17135, int_17133)
        
        # Assigning a type to the variable 'type_store_excepts' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'type_store_excepts', subscript_call_result_17136)
        
        
        # Getting the type of 'cont' (line 343)
        cont_17137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 14), 'cont')
        
        # Call to len(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'type_store_except_branches' (line 343)
        type_store_except_branches_17139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'type_store_except_branches', False)
        # Processing the call keyword arguments (line 343)
        kwargs_17140 = {}
        # Getting the type of 'len' (line 343)
        len_17138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 21), 'len', False)
        # Calling len(args, kwargs) (line 343)
        len_call_result_17141 = invoke(stypy.reporting.localization.Localization(__file__, 343, 21), len_17138, *[type_store_except_branches_17139], **kwargs_17140)
        
        # Applying the binary operator '<' (line 343)
        result_lt_17142 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 14), '<', cont_17137, len_call_result_17141)
        
        # Assigning a type to the variable 'result_lt_17142' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'result_lt_17142', result_lt_17142)
        # Testing if the while is going to be iterated (line 343)
        # Testing the type of an if condition (line 343)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 8), result_lt_17142)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 343, 8), result_lt_17142):
            # SSA begins for while statement (line 343)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Call to a Name (line 344):
            
            # Call to __join_except_branches(...): (line 344)
            # Processing the call arguments (line 344)
            # Getting the type of 'type_store_excepts' (line 344)
            type_store_excepts_17144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 56), 'type_store_excepts', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'cont' (line 344)
            cont_17145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 103), 'cont', False)
            # Getting the type of 'type_store_except_branches' (line 344)
            type_store_except_branches_17146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 76), 'type_store_except_branches', False)
            # Obtaining the member '__getitem__' of a type (line 344)
            getitem___17147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 76), type_store_except_branches_17146, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 344)
            subscript_call_result_17148 = invoke(stypy.reporting.localization.Localization(__file__, 344, 76), getitem___17147, cont_17145)
            
            # Processing the call keyword arguments (line 344)
            kwargs_17149 = {}
            # Getting the type of '__join_except_branches' (line 344)
            join_except_branches_17143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 33), '__join_except_branches', False)
            # Calling __join_except_branches(args, kwargs) (line 344)
            join_except_branches_call_result_17150 = invoke(stypy.reporting.localization.Localization(__file__, 344, 33), join_except_branches_17143, *[type_store_excepts_17144, subscript_call_result_17148], **kwargs_17149)
            
            # Assigning a type to the variable 'type_store_excepts' (line 344)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'type_store_excepts', join_except_branches_call_result_17150)
            
            # Getting the type of 'cont' (line 345)
            cont_17151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'cont')
            int_17152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 20), 'int')
            # Applying the binary operator '+=' (line 345)
            result_iadd_17153 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 12), '+=', cont_17151, int_17152)
            # Assigning a type to the variable 'cont' (line 345)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'cont', result_iadd_17153)
            
            # SSA join for while statement (line 343)
            module_type_store = module_type_store.join_ssa_context()

        
    else:
        
        # Testing the type of an if condition (line 338)
        if_condition_17127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 338, 4), result_eq_17126)
        # Assigning a type to the variable 'if_condition_17127' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 4), 'if_condition_17127', if_condition_17127)
        # SSA begins for if statement (line 338)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 339):
        
        # Obtaining the type of the subscript
        int_17128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 56), 'int')
        # Getting the type of 'type_store_except_branches' (line 339)
        type_store_except_branches_17129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 29), 'type_store_except_branches')
        # Obtaining the member '__getitem__' of a type (line 339)
        getitem___17130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 29), type_store_except_branches_17129, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 339)
        subscript_call_result_17131 = invoke(stypy.reporting.localization.Localization(__file__, 339, 29), getitem___17130, int_17128)
        
        # Assigning a type to the variable 'type_store_excepts' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'type_store_excepts', subscript_call_result_17131)
        # SSA branch for the else part of an if statement (line 338)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 341):
        int_17132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 341, 15), 'int')
        # Assigning a type to the variable 'cont' (line 341)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 341, 8), 'cont', int_17132)
        
        # Assigning a Subscript to a Name (line 342):
        
        # Obtaining the type of the subscript
        int_17133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 342, 56), 'int')
        # Getting the type of 'type_store_except_branches' (line 342)
        type_store_except_branches_17134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 29), 'type_store_except_branches')
        # Obtaining the member '__getitem__' of a type (line 342)
        getitem___17135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 342, 29), type_store_except_branches_17134, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 342)
        subscript_call_result_17136 = invoke(stypy.reporting.localization.Localization(__file__, 342, 29), getitem___17135, int_17133)
        
        # Assigning a type to the variable 'type_store_excepts' (line 342)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 8), 'type_store_excepts', subscript_call_result_17136)
        
        
        # Getting the type of 'cont' (line 343)
        cont_17137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 14), 'cont')
        
        # Call to len(...): (line 343)
        # Processing the call arguments (line 343)
        # Getting the type of 'type_store_except_branches' (line 343)
        type_store_except_branches_17139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 25), 'type_store_except_branches', False)
        # Processing the call keyword arguments (line 343)
        kwargs_17140 = {}
        # Getting the type of 'len' (line 343)
        len_17138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 343, 21), 'len', False)
        # Calling len(args, kwargs) (line 343)
        len_call_result_17141 = invoke(stypy.reporting.localization.Localization(__file__, 343, 21), len_17138, *[type_store_except_branches_17139], **kwargs_17140)
        
        # Applying the binary operator '<' (line 343)
        result_lt_17142 = python_operator(stypy.reporting.localization.Localization(__file__, 343, 14), '<', cont_17137, len_call_result_17141)
        
        # Assigning a type to the variable 'result_lt_17142' (line 343)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 343, 8), 'result_lt_17142', result_lt_17142)
        # Testing if the while is going to be iterated (line 343)
        # Testing the type of an if condition (line 343)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 343, 8), result_lt_17142)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 343, 8), result_lt_17142):
            # SSA begins for while statement (line 343)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a Call to a Name (line 344):
            
            # Call to __join_except_branches(...): (line 344)
            # Processing the call arguments (line 344)
            # Getting the type of 'type_store_excepts' (line 344)
            type_store_excepts_17144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 56), 'type_store_excepts', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'cont' (line 344)
            cont_17145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 103), 'cont', False)
            # Getting the type of 'type_store_except_branches' (line 344)
            type_store_except_branches_17146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 76), 'type_store_except_branches', False)
            # Obtaining the member '__getitem__' of a type (line 344)
            getitem___17147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 344, 76), type_store_except_branches_17146, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 344)
            subscript_call_result_17148 = invoke(stypy.reporting.localization.Localization(__file__, 344, 76), getitem___17147, cont_17145)
            
            # Processing the call keyword arguments (line 344)
            kwargs_17149 = {}
            # Getting the type of '__join_except_branches' (line 344)
            join_except_branches_17143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 344, 33), '__join_except_branches', False)
            # Calling __join_except_branches(args, kwargs) (line 344)
            join_except_branches_call_result_17150 = invoke(stypy.reporting.localization.Localization(__file__, 344, 33), join_except_branches_17143, *[type_store_excepts_17144, subscript_call_result_17148], **kwargs_17149)
            
            # Assigning a type to the variable 'type_store_excepts' (line 344)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 344, 12), 'type_store_excepts', join_except_branches_call_result_17150)
            
            # Getting the type of 'cont' (line 345)
            cont_17151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'cont')
            int_17152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 345, 20), 'int')
            # Applying the binary operator '+=' (line 345)
            result_iadd_17153 = python_operator(stypy.reporting.localization.Localization(__file__, 345, 12), '+=', cont_17151, int_17152)
            # Assigning a type to the variable 'cont' (line 345)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 345, 12), 'cont', result_iadd_17153)
            
            # SSA join for while statement (line 343)
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 338)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 348):
    
    # Call to __join__try_except(...): (line 348)
    # Processing the call arguments (line 348)
    # Getting the type of 'type_store_pretry' (line 348)
    type_store_pretry_17155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 45), 'type_store_pretry', False)
    # Getting the type of 'type_store_posttry' (line 348)
    type_store_posttry_17156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 64), 'type_store_posttry', False)
    # Getting the type of 'type_store_excepts' (line 349)
    type_store_excepts_17157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 45), 'type_store_excepts', False)
    # Processing the call keyword arguments (line 348)
    kwargs_17158 = {}
    # Getting the type of '__join__try_except' (line 348)
    join__try_except_17154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 348, 26), '__join__try_except', False)
    # Calling __join__try_except(args, kwargs) (line 348)
    join__try_except_call_result_17159 = invoke(stypy.reporting.localization.Localization(__file__, 348, 26), join__try_except_17154, *[type_store_pretry_17155, type_store_posttry_17156, type_store_excepts_17157], **kwargs_17158)
    
    # Assigning a type to the variable 'joined_context_dict' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'joined_context_dict', join__try_except_call_result_17159)
    
    # Type idiom detected: calculating its left and rigth part (line 352)
    # Getting the type of 'type_store_finally' (line 352)
    type_store_finally_17160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 4), 'type_store_finally')
    # Getting the type of 'None' (line 352)
    None_17161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 352, 33), 'None')
    
    (may_be_17162, more_types_in_union_17163) = may_not_be_none(type_store_finally_17160, None_17161)

    if may_be_17162:

        if more_types_in_union_17163:
            # Runtime conditional SSA (line 352)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 353):
        
        # Call to __join_finally_branch(...): (line 353)
        # Processing the call arguments (line 353)
        # Getting the type of 'joined_context_dict' (line 353)
        joined_context_dict_17165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 52), 'joined_context_dict', False)
        # Getting the type of 'type_store_finally' (line 353)
        type_store_finally_17166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 73), 'type_store_finally', False)
        # Processing the call keyword arguments (line 353)
        kwargs_17167 = {}
        # Getting the type of '__join_finally_branch' (line 353)
        join_finally_branch_17164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 30), '__join_finally_branch', False)
        # Calling __join_finally_branch(args, kwargs) (line 353)
        join_finally_branch_call_result_17168 = invoke(stypy.reporting.localization.Localization(__file__, 353, 30), join_finally_branch_17164, *[joined_context_dict_17165, type_store_finally_17166], **kwargs_17167)
        
        # Assigning a type to the variable 'joined_context_dict' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 8), 'joined_context_dict', join_finally_branch_call_result_17168)

        if more_types_in_union_17163:
            # SSA join for if statement (line 352)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'joined_context_dict' (line 355)
    joined_context_dict_17169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 11), 'joined_context_dict')
    # Assigning a type to the variable 'stypy_return_type' (line 355)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 4), 'stypy_return_type', joined_context_dict_17169)
    
    # ################# End of 'join_exception_block(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'join_exception_block' in the type store
    # Getting the type of 'stypy_return_type' (line 321)
    stypy_return_type_17170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17170)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'join_exception_block'
    return stypy_return_type_17170

# Assigning a type to the variable 'join_exception_block' (line 321)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 321, 0), 'join_exception_block', join_exception_block)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
