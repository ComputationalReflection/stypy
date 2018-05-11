
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import inspect
2: 
3: from ...python_lib_copy.member_call_copy.handlers_copy import type_rule_call_handler_copy
4: from ...python_lib_copy.member_call_copy.handlers_copy import fake_param_values_call_handler_copy
5: from ...python_lib_copy.member_call_copy.handlers_copy import user_callables_call_handler_copy
6: from ...python_lib_copy.member_call_copy.type_modifiers_copy import file_type_modifier_copy
7: from arguments_unfolding_copy import *
8: from call_handlers_helper_methods_copy import *
9: 
10: '''
11: Call handlers are the entities we use to perform calls to type inference code. There are several call handlers, as
12: the call strategy is different depending on the origin of the code to be called:
13: 
14: - Rule-based call handlers: This is used with Python library modules and functions.
15: Some of these elements may have a rule file associated. This rule file indicates the accepted
16: parameters for this call and it expected return type depending on this parameters. This is the most powerful call
17: handler, as the rules we developed allows a wide range of type checking options that may be used to ensure valid
18: calls. However, rule files have to be developed for each Python module, and while we plan to develop rule files
19: for each one of them on a semi-automatic way, this is the last part of the stypy development process, which means
20: that not every module will have one. If no rule file is present, other call handler will take care of the call.
21: 
22: Type rules are read from a directory structure inside the library, so we can add them on a later stage of development
23: without changing stypy source code.
24: 
25: - User callables call handler: The existence of a rule-based call handler is justified by the inability to have the
26: code of Python library functions, as most of them are developed in C and the source code cannot be obtained anyway.
27: However, user-coded .py files are processed and converted to a type inference equivalent program. The conversion
28: of callable entities transform them to a callable form composed by two parameters: a list of variable arguments and
29: a list of keyword arguments (def converted_func(*args, **kwargs)) that are handled by the type inference code. This
30: call handler is the responsible of passing the parameters in this form, so we can call type inference code easily.
31: 
32: - Fake param values call handler: The last-resort call handler, used in those Python library modules with no current
33: type rule file and external third-party code that cannot be transformed to type inference code because source code
34: is not available. Calls to this type of code from type inference code will pass types instead of values to the call.
35:  For example, if we find in our program the call library_function_with_no_source_code(3, "hi") the type inference
36:  code we generate will turn this to library_function_with_no_source_code(*[int, str], **{}). As this call is not valid
37:  (the called function cannot be transformed to a type inference equivalent), this call handler obtains default
38:  predefined fake values for each passed type and phisically call the function with them in order to obtain a result.
39:  The type of this result is later returned to the type inference code. This is the functionality of this call handler.
40:  Note that this dynamically obtain the type of a call by performing the call, causing the execution of part of the
41:  real program instead of the type-inference equivalent, which is not optimal. However, it allows us to test a much
42:  wider array of programs initially, even if they use libraries and code that do not have the source available and
43:  have no type rule file attached to it. It is our goal, with time to rely on this call handler as less as possible.
44:  Note that if the passed type has an associated value, this value will be used instead of the default fake one. However,
45:  as we said, type values are only calculated in very limited cases.
46: '''
47: 
48: # We want the type-rule call handler instance available
49: rule_based_call_handler = type_rule_call_handler_copy.TypeRuleCallHandler()
50: 
51: '''
52: Here we register, ordered by priority, those classes that handle member calls using different strategies to obtain
53: the return type of a callable that we described previously, once the type or the input parameters are obtained. Note
54: that all call handlers are singletons, stateless classes.
55: '''
56: registered_call_handlers = [
57:     rule_based_call_handler,
58:     user_callables_call_handler_copy.UserCallablesCallHandler(),
59:     fake_param_values_call_handler_copy.FakeParamValuesCallHandler(),
60: ]
61: 
62: '''
63: A type modifier is an special class that is associated with type-rule call handler, complementing its functionality.
64: Although the rules we developed are able to express the return type of a Python library call function in a lot of
65: cases, there are cases when they are not enough to accurately express the shape of the return type of a function.
66: This is true when the return type is a collection of a certain type, for example. This is when a type modifier is
67: used: once a type rule has been used to determine that the call is valid, a type modifier associated to this call
68: is later called with the passed parameters to obtain a proper, more accurate return type than the expressed by the rule.
69: Note that not every Python library callable will have a type modifier associated. In fact most of them will not have
70: one, as this is only used to improve type inference on certain specific callables, whose rule files are not enough for
71: that. If a certain callable has both a rule file return type and a type modifier return type, the latter takes
72: precedence.
73: 
74: Only a type modifier is present at the moment: The one that dynamically reads type modifier functions for a Python
75: (.py) source file. Type modifiers are read from a directory structure inside the library, so we can add them on a
76:  later stage of development without changing stypy source code. Although only one type modifier is present, we
77:  developed this system to add more in the future, should the necessity arise.
78: '''
79: registered_type_modifiers = [
80:     file_type_modifier_copy.FileTypeModifier(),
81: ]
82: 
83: 
84: def get_param_arity(proxy_obj, callable_):
85:     '''
86:     Uses python introspection over the callable element to try to guess how many parameters can be passed to the
87:     callable. If it is not possible (Python library functions do not have this data), we use the type rule call
88:     handler to try to obtain them. If all fails, -1 is returned. This function also determines if the callable
89:     uses a variable list of arguments.
90:     :param proxy_obj: TypeInferenceProxy representing the callable
91:     :param callable_: Python callable entity
92:     :return: list of maximum passable arguments, has varargs tuple
93:     '''
94:     # Callable entity with metadata
95:     if hasattr(callable_, "im_func"):
96:         argspec = inspect.getargspec(callable_)
97:         real_args = len(
98:             argspec.args) - 2  # callable_.im_func.func_code.co_argcount - 2 #Do not consider localization and self
99:         has_varargs = argspec.varargs is not None
100:         return [real_args], has_varargs
101:     else:
102:         if rule_based_call_handler.applies_to(proxy_obj, callable_):
103:             return rule_based_call_handler.get_parameter_arity(proxy_obj, callable_)
104: 
105:     return [-1], False  # Unknown parameter number
106: 
107: 
108: def perform_call(proxy_obj, callable_, localization, *args, **kwargs):
109:     '''
110:     Perform the type inference of the call to the callable entity, using the passed arguments and a suitable
111:     call handler to resolve the call (see above).
112: 
113:     :param proxy_obj: TypeInferenceProxy representing the callable
114:     :param callable_: Python callable entity
115:     :param localization: Caller information
116:     :param args: named arguments plus variable list of arguments
117:     :param kwargs: keyword arguments plus defaults
118:     :return: The return type of the called element
119:     '''
120: 
121:     # Obtain the type of the arguments as a modifiable list
122:     arg_types = list(args)
123:     kwarg_types = kwargs
124: 
125:     # TODO: Remove?
126:     # arg_types = get_arg_types(args)
127:     # Obtain the types of the keyword arguments
128:     # kwarg_types = get_kwarg_types(kwargs)
129: 
130:     # Initialize variables
131:     unfolded_arg_tuples = None
132:     return_type = None
133:     found_valid_call = False
134:     found_errors = []
135:     found_type_errors = False
136: 
137:     try:
138:         # Process call handlers in order
139:         for call_handler in registered_call_handlers:
140:             # Use the first call handler that declares that can handle this callable
141:             if call_handler.applies_to(proxy_obj, callable_):
142:                 # When calling the callable element, the type of some parameters might be undefined (not initialized
143:                 # to any value in the preceding code). This function check this fact and substitute the Undefined
144:                 # parameters by suitable type errors. It also creates warnings if the undefined type is inside a
145:                 # union type, removing the undefined type from the union afterwards.
146:                 arg_types, kwarg_types = check_undefined_type_within_parameters(localization,
147:                                                                                 format_call(callable_, arg_types,
148:                                                                                             kwarg_types),
149:                                                                                 *arg_types, **kwarg_types)
150: 
151:                 # Is this a callable that has been converted to an equivalent type inference function?
152:                 if isinstance(call_handler, user_callables_call_handler_copy.UserCallablesCallHandler):
153:                     # Invoke the applicable call handler
154:                     ret = call_handler(proxy_obj, localization, callable_, *arg_types, **kwarg_types)
155:                     if not isinstance(ret, TypeError):
156:                         # Not an error? accumulate the return type in a return union type
157:                         found_valid_call = True
158:                         return_type = ret
159:                     else:
160:                         # Store found errors
161:                         found_errors.append(ret)
162:                 else:
163:                     # If we reach this point, it means that we have to use type rules or fake param values call handlers
164:                     # These call handlers do not handle union types, only concrete ones. Therefore, if the passed
165:                     # arguments have union types, these arguments have to be unfolded: each union type is separated
166:                     # into its stored types and a new parameter list is formed with each one of them. For example,
167:                     # if the parameters are ((int \/ str), float), this is generated:
168:                     # [
169:                     #   (int, float),
170:                     #   (str, float)
171:                     # ]
172:                     #
173:                     # This process is repeated with any union type found among parameters on any position, so at the
174:                     # end we have multiple possible parameter lists all with no union types present. Later on, all
175:                     # the possible parameter lists are checked with the call handler, and results of all of them are
176:                     # collected to obtain the final call result.
177: 
178:                     if unfolded_arg_tuples is None:
179:                         # Unfold union types found in arguments and/or keyword arguments to use proper python types
180:                         # in functions calls.
181:                         unfolded_arg_tuples = unfold_arguments(*arg_types, **kwarg_types)
182: 
183:                     # Use each possible combination of union type components found in args / kwargs
184:                     for tuple_ in unfolded_arg_tuples:
185:                         # If this parameter tuple contain a type error, do no type inference with it
186:                         if exist_a_type_error_within_parameters(localization, *tuple_[0], **tuple_[1]):
187:                             found_type_errors = True
188:                             continue
189: 
190:                         # Call the call handler with no union type
191:                         ret = call_handler(proxy_obj, localization, callable_, *tuple_[0], **tuple_[1])
192:                         if not isinstance(ret, TypeError):
193:                             # Not an error? accumulate the return type in a return union type. Call is possible with
194:                             # at least one combination of parameters.
195:                             found_valid_call = True
196: 
197:                             # As the call is possible with this parameter tuple, we must check possible type modifiers.
198:                             # Its return type prevails over the returned by the call handler
199:                             for modifier in registered_type_modifiers:
200:                                 if modifier.applies_to(proxy_obj, callable_):
201:                                     if inspect.ismethod(callable_) or inspect.ismethoddescriptor(callable_):
202:                                         # Are we calling with a type variable instead with a type instance?
203:                                         if not proxy_obj.parent_proxy.is_type_instance():
204:                                             # Invoke the modifier with the appropriate parameters
205:                                             ret = modifier(tuple_[0][0], localization, callable_, *tuple_[0][1:],
206:                                                            **tuple_[1])
207:                                             break
208:                                     # Invoke the modifier with the appropriate parameters
209:                                     ret = modifier(proxy_obj, localization, callable_, *tuple_[0], **tuple_[1])
210:                                     break
211: 
212:                             # Add the return type of the type rule or the modifier (if one is available for this call)
213:                             # to the final return type
214:                             return_type = union_type_copy.UnionType.add(return_type, ret)
215:                         else:
216:                             # Store found errors
217:                             found_errors.append(ret)
218: 
219:                 # Could we found any valid combination? Then return the union type that represent all the possible
220:                 # return types of all the valid parameter combinations and convert the call errors to warnings
221:                 if found_valid_call:
222:                     for error in found_errors:
223:                         error.turn_to_warning()
224:                     return return_type
225:                 else:
226:                     # No possible combination of parameters is possible? Remove all errors and return a single one
227:                     # with an appropriate message
228: 
229:                     # Only one error? then return it as the cause
230:                     if len(found_errors) == 1:
231:                         return found_errors[0]
232: 
233:                     # Multiple error found? We tried to return a compendium of all the obtained error messages, but
234:                     # we found that they were not clear or even misleading, as the user don't have to be aware of what
235:                     # a union type is. So, in the end we decided to remove all the generated errors and return a single
236:                     # one with a generic descriptive message and a pretty-print of the call.
237:                     for error in found_errors:
238:                         TypeError.remove_error_msg(error)
239: 
240:                     call_str = format_call(callable_, arg_types, kwarg_types)
241:                     if found_type_errors:
242:                         msg = "Type errors found among the types of the call parameters"
243:                     else:
244:                         msg = "The called entity do not accept any of these parameters"
245: 
246:                     return TypeError(localization, "{0}: {1}".format(call_str, msg))
247: 
248:     except Exception as e:
249:         # This may indicate an stypy bug
250:         return TypeError(localization, "An error was produced when invoking '{0}' with arguments [{1}]: {2}".format(
251:             callable_, list(arg_types) + list(kwarg_types.values()), e))
252: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import inspect' statement (line 1)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy import type_rule_call_handler_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_5178 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy')

if (type(import_5178) is not StypyTypeError):

    if (import_5178 != 'pyd_module'):
        __import__(import_5178)
        sys_modules_5179 = sys.modules[import_5178]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy', sys_modules_5179.module_type_store, module_type_store, ['type_rule_call_handler_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_5179, sys_modules_5179.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy import type_rule_call_handler_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy', None, module_type_store, ['type_rule_call_handler_copy'], [type_rule_call_handler_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy', import_5178)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy import fake_param_values_call_handler_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_5180 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy')

if (type(import_5180) is not StypyTypeError):

    if (import_5180 != 'pyd_module'):
        __import__(import_5180)
        sys_modules_5181 = sys.modules[import_5180]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy', sys_modules_5181.module_type_store, module_type_store, ['fake_param_values_call_handler_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_5181, sys_modules_5181.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy import fake_param_values_call_handler_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy', None, module_type_store, ['fake_param_values_call_handler_copy'], [fake_param_values_call_handler_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy', import_5180)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy import user_callables_call_handler_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_5182 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy')

if (type(import_5182) is not StypyTypeError):

    if (import_5182 != 'pyd_module'):
        __import__(import_5182)
        sys_modules_5183 = sys.modules[import_5182]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy', sys_modules_5183.module_type_store, module_type_store, ['user_callables_call_handler_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_5183, sys_modules_5183.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy import user_callables_call_handler_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy', None, module_type_store, ['user_callables_call_handler_copy'], [user_callables_call_handler_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy', import_5182)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy import file_type_modifier_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_5184 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy')

if (type(import_5184) is not StypyTypeError):

    if (import_5184 != 'pyd_module'):
        __import__(import_5184)
        sys_modules_5185 = sys.modules[import_5184]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy', sys_modules_5185.module_type_store, module_type_store, ['file_type_modifier_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_5185, sys_modules_5185.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy import file_type_modifier_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy', None, module_type_store, ['file_type_modifier_copy'], [file_type_modifier_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy', import_5184)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from arguments_unfolding_copy import ' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_5186 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'arguments_unfolding_copy')

if (type(import_5186) is not StypyTypeError):

    if (import_5186 != 'pyd_module'):
        __import__(import_5186)
        sys_modules_5187 = sys.modules[import_5186]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'arguments_unfolding_copy', sys_modules_5187.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_5187, sys_modules_5187.module_type_store, module_type_store)
    else:
        from arguments_unfolding_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'arguments_unfolding_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'arguments_unfolding_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'arguments_unfolding_copy', import_5186)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from call_handlers_helper_methods_copy import ' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_5188 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'call_handlers_helper_methods_copy')

if (type(import_5188) is not StypyTypeError):

    if (import_5188 != 'pyd_module'):
        __import__(import_5188)
        sys_modules_5189 = sys.modules[import_5188]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'call_handlers_helper_methods_copy', sys_modules_5189.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_5189, sys_modules_5189.module_type_store, module_type_store)
    else:
        from call_handlers_helper_methods_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'call_handlers_helper_methods_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'call_handlers_helper_methods_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'call_handlers_helper_methods_copy', import_5188)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

str_5190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, (-1)), 'str', '\nCall handlers are the entities we use to perform calls to type inference code. There are several call handlers, as\nthe call strategy is different depending on the origin of the code to be called:\n\n- Rule-based call handlers: This is used with Python library modules and functions.\nSome of these elements may have a rule file associated. This rule file indicates the accepted\nparameters for this call and it expected return type depending on this parameters. This is the most powerful call\nhandler, as the rules we developed allows a wide range of type checking options that may be used to ensure valid\ncalls. However, rule files have to be developed for each Python module, and while we plan to develop rule files\nfor each one of them on a semi-automatic way, this is the last part of the stypy development process, which means\nthat not every module will have one. If no rule file is present, other call handler will take care of the call.\n\nType rules are read from a directory structure inside the library, so we can add them on a later stage of development\nwithout changing stypy source code.\n\n- User callables call handler: The existence of a rule-based call handler is justified by the inability to have the\ncode of Python library functions, as most of them are developed in C and the source code cannot be obtained anyway.\nHowever, user-coded .py files are processed and converted to a type inference equivalent program. The conversion\nof callable entities transform them to a callable form composed by two parameters: a list of variable arguments and\na list of keyword arguments (def converted_func(*args, **kwargs)) that are handled by the type inference code. This\ncall handler is the responsible of passing the parameters in this form, so we can call type inference code easily.\n\n- Fake param values call handler: The last-resort call handler, used in those Python library modules with no current\ntype rule file and external third-party code that cannot be transformed to type inference code because source code\nis not available. Calls to this type of code from type inference code will pass types instead of values to the call.\n For example, if we find in our program the call library_function_with_no_source_code(3, "hi") the type inference\n code we generate will turn this to library_function_with_no_source_code(*[int, str], **{}). As this call is not valid\n (the called function cannot be transformed to a type inference equivalent), this call handler obtains default\n predefined fake values for each passed type and phisically call the function with them in order to obtain a result.\n The type of this result is later returned to the type inference code. This is the functionality of this call handler.\n Note that this dynamically obtain the type of a call by performing the call, causing the execution of part of the\n real program instead of the type-inference equivalent, which is not optimal. However, it allows us to test a much\n wider array of programs initially, even if they use libraries and code that do not have the source available and\n have no type rule file attached to it. It is our goal, with time to rely on this call handler as less as possible.\n Note that if the passed type has an associated value, this value will be used instead of the default fake one. However,\n as we said, type values are only calculated in very limited cases.\n')

# Assigning a Call to a Name (line 49):

# Assigning a Call to a Name (line 49):

# Call to TypeRuleCallHandler(...): (line 49)
# Processing the call keyword arguments (line 49)
kwargs_5193 = {}
# Getting the type of 'type_rule_call_handler_copy' (line 49)
type_rule_call_handler_copy_5191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'type_rule_call_handler_copy', False)
# Obtaining the member 'TypeRuleCallHandler' of a type (line 49)
TypeRuleCallHandler_5192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 26), type_rule_call_handler_copy_5191, 'TypeRuleCallHandler')
# Calling TypeRuleCallHandler(args, kwargs) (line 49)
TypeRuleCallHandler_call_result_5194 = invoke(stypy.reporting.localization.Localization(__file__, 49, 26), TypeRuleCallHandler_5192, *[], **kwargs_5193)

# Assigning a type to the variable 'rule_based_call_handler' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'rule_based_call_handler', TypeRuleCallHandler_call_result_5194)
str_5195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', '\nHere we register, ordered by priority, those classes that handle member calls using different strategies to obtain\nthe return type of a callable that we described previously, once the type or the input parameters are obtained. Note\nthat all call handlers are singletons, stateless classes.\n')

# Assigning a List to a Name (line 56):

# Assigning a List to a Name (line 56):

# Obtaining an instance of the builtin type 'list' (line 56)
list_5196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 56)
# Adding element type (line 56)
# Getting the type of 'rule_based_call_handler' (line 57)
rule_based_call_handler_5197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'rule_based_call_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 27), list_5196, rule_based_call_handler_5197)
# Adding element type (line 56)

# Call to UserCallablesCallHandler(...): (line 58)
# Processing the call keyword arguments (line 58)
kwargs_5200 = {}
# Getting the type of 'user_callables_call_handler_copy' (line 58)
user_callables_call_handler_copy_5198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'user_callables_call_handler_copy', False)
# Obtaining the member 'UserCallablesCallHandler' of a type (line 58)
UserCallablesCallHandler_5199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 4), user_callables_call_handler_copy_5198, 'UserCallablesCallHandler')
# Calling UserCallablesCallHandler(args, kwargs) (line 58)
UserCallablesCallHandler_call_result_5201 = invoke(stypy.reporting.localization.Localization(__file__, 58, 4), UserCallablesCallHandler_5199, *[], **kwargs_5200)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 27), list_5196, UserCallablesCallHandler_call_result_5201)
# Adding element type (line 56)

# Call to FakeParamValuesCallHandler(...): (line 59)
# Processing the call keyword arguments (line 59)
kwargs_5204 = {}
# Getting the type of 'fake_param_values_call_handler_copy' (line 59)
fake_param_values_call_handler_copy_5202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'fake_param_values_call_handler_copy', False)
# Obtaining the member 'FakeParamValuesCallHandler' of a type (line 59)
FakeParamValuesCallHandler_5203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 4), fake_param_values_call_handler_copy_5202, 'FakeParamValuesCallHandler')
# Calling FakeParamValuesCallHandler(args, kwargs) (line 59)
FakeParamValuesCallHandler_call_result_5205 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), FakeParamValuesCallHandler_5203, *[], **kwargs_5204)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 27), list_5196, FakeParamValuesCallHandler_call_result_5205)

# Assigning a type to the variable 'registered_call_handlers' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'registered_call_handlers', list_5196)
str_5206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', '\nA type modifier is an special class that is associated with type-rule call handler, complementing its functionality.\nAlthough the rules we developed are able to express the return type of a Python library call function in a lot of\ncases, there are cases when they are not enough to accurately express the shape of the return type of a function.\nThis is true when the return type is a collection of a certain type, for example. This is when a type modifier is\nused: once a type rule has been used to determine that the call is valid, a type modifier associated to this call\nis later called with the passed parameters to obtain a proper, more accurate return type than the expressed by the rule.\nNote that not every Python library callable will have a type modifier associated. In fact most of them will not have\none, as this is only used to improve type inference on certain specific callables, whose rule files are not enough for\nthat. If a certain callable has both a rule file return type and a type modifier return type, the latter takes\nprecedence.\n\nOnly a type modifier is present at the moment: The one that dynamically reads type modifier functions for a Python\n(.py) source file. Type modifiers are read from a directory structure inside the library, so we can add them on a\n later stage of development without changing stypy source code. Although only one type modifier is present, we\n developed this system to add more in the future, should the necessity arise.\n')

# Assigning a List to a Name (line 79):

# Assigning a List to a Name (line 79):

# Obtaining an instance of the builtin type 'list' (line 79)
list_5207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 79)
# Adding element type (line 79)

# Call to FileTypeModifier(...): (line 80)
# Processing the call keyword arguments (line 80)
kwargs_5210 = {}
# Getting the type of 'file_type_modifier_copy' (line 80)
file_type_modifier_copy_5208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'file_type_modifier_copy', False)
# Obtaining the member 'FileTypeModifier' of a type (line 80)
FileTypeModifier_5209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 4), file_type_modifier_copy_5208, 'FileTypeModifier')
# Calling FileTypeModifier(args, kwargs) (line 80)
FileTypeModifier_call_result_5211 = invoke(stypy.reporting.localization.Localization(__file__, 80, 4), FileTypeModifier_5209, *[], **kwargs_5210)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 28), list_5207, FileTypeModifier_call_result_5211)

# Assigning a type to the variable 'registered_type_modifiers' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'registered_type_modifiers', list_5207)

@norecursion
def get_param_arity(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_param_arity'
    module_type_store = module_type_store.open_function_context('get_param_arity', 84, 0, False)
    
    # Passed parameters checking function
    get_param_arity.stypy_localization = localization
    get_param_arity.stypy_type_of_self = None
    get_param_arity.stypy_type_store = module_type_store
    get_param_arity.stypy_function_name = 'get_param_arity'
    get_param_arity.stypy_param_names_list = ['proxy_obj', 'callable_']
    get_param_arity.stypy_varargs_param_name = None
    get_param_arity.stypy_kwargs_param_name = None
    get_param_arity.stypy_call_defaults = defaults
    get_param_arity.stypy_call_varargs = varargs
    get_param_arity.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_param_arity', ['proxy_obj', 'callable_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_param_arity', localization, ['proxy_obj', 'callable_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_param_arity(...)' code ##################

    str_5212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, (-1)), 'str', '\n    Uses python introspection over the callable element to try to guess how many parameters can be passed to the\n    callable. If it is not possible (Python library functions do not have this data), we use the type rule call\n    handler to try to obtain them. If all fails, -1 is returned. This function also determines if the callable\n    uses a variable list of arguments.\n    :param proxy_obj: TypeInferenceProxy representing the callable\n    :param callable_: Python callable entity\n    :return: list of maximum passable arguments, has varargs tuple\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 95)
    str_5213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 26), 'str', 'im_func')
    # Getting the type of 'callable_' (line 95)
    callable__5214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'callable_')
    
    (may_be_5215, more_types_in_union_5216) = may_provide_member(str_5213, callable__5214)

    if may_be_5215:

        if more_types_in_union_5216:
            # Runtime conditional SSA (line 95)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'callable_' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'callable_', remove_not_member_provider_from_union(callable__5214, 'im_func'))
        
        # Assigning a Call to a Name (line 96):
        
        # Assigning a Call to a Name (line 96):
        
        # Call to getargspec(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'callable_' (line 96)
        callable__5219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 37), 'callable_', False)
        # Processing the call keyword arguments (line 96)
        kwargs_5220 = {}
        # Getting the type of 'inspect' (line 96)
        inspect_5217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'inspect', False)
        # Obtaining the member 'getargspec' of a type (line 96)
        getargspec_5218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 18), inspect_5217, 'getargspec')
        # Calling getargspec(args, kwargs) (line 96)
        getargspec_call_result_5221 = invoke(stypy.reporting.localization.Localization(__file__, 96, 18), getargspec_5218, *[callable__5219], **kwargs_5220)
        
        # Assigning a type to the variable 'argspec' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'argspec', getargspec_call_result_5221)
        
        # Assigning a BinOp to a Name (line 97):
        
        # Assigning a BinOp to a Name (line 97):
        
        # Call to len(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'argspec' (line 98)
        argspec_5223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'argspec', False)
        # Obtaining the member 'args' of a type (line 98)
        args_5224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), argspec_5223, 'args')
        # Processing the call keyword arguments (line 97)
        kwargs_5225 = {}
        # Getting the type of 'len' (line 97)
        len_5222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'len', False)
        # Calling len(args, kwargs) (line 97)
        len_call_result_5226 = invoke(stypy.reporting.localization.Localization(__file__, 97, 20), len_5222, *[args_5224], **kwargs_5225)
        
        int_5227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 28), 'int')
        # Applying the binary operator '-' (line 97)
        result_sub_5228 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 20), '-', len_call_result_5226, int_5227)
        
        # Assigning a type to the variable 'real_args' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'real_args', result_sub_5228)
        
        # Assigning a Compare to a Name (line 99):
        
        # Assigning a Compare to a Name (line 99):
        
        # Getting the type of 'argspec' (line 99)
        argspec_5229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'argspec')
        # Obtaining the member 'varargs' of a type (line 99)
        varargs_5230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 22), argspec_5229, 'varargs')
        # Getting the type of 'None' (line 99)
        None_5231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 45), 'None')
        # Applying the binary operator 'isnot' (line 99)
        result_is_not_5232 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 22), 'isnot', varargs_5230, None_5231)
        
        # Assigning a type to the variable 'has_varargs' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'has_varargs', result_is_not_5232)
        
        # Obtaining an instance of the builtin type 'tuple' (line 100)
        tuple_5233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 100)
        # Adding element type (line 100)
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_5234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        # Adding element type (line 100)
        # Getting the type of 'real_args' (line 100)
        real_args_5235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'real_args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 15), list_5234, real_args_5235)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 15), tuple_5233, list_5234)
        # Adding element type (line 100)
        # Getting the type of 'has_varargs' (line 100)
        has_varargs_5236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 28), 'has_varargs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 15), tuple_5233, has_varargs_5236)
        
        # Assigning a type to the variable 'stypy_return_type' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'stypy_return_type', tuple_5233)

        if more_types_in_union_5216:
            # Runtime conditional SSA for else branch (line 95)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_5215) or more_types_in_union_5216):
        # Assigning a type to the variable 'callable_' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'callable_', remove_member_provider_from_union(callable__5214, 'im_func'))
        
        # Call to applies_to(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'proxy_obj' (line 102)
        proxy_obj_5239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 46), 'proxy_obj', False)
        # Getting the type of 'callable_' (line 102)
        callable__5240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 57), 'callable_', False)
        # Processing the call keyword arguments (line 102)
        kwargs_5241 = {}
        # Getting the type of 'rule_based_call_handler' (line 102)
        rule_based_call_handler_5237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'rule_based_call_handler', False)
        # Obtaining the member 'applies_to' of a type (line 102)
        applies_to_5238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 11), rule_based_call_handler_5237, 'applies_to')
        # Calling applies_to(args, kwargs) (line 102)
        applies_to_call_result_5242 = invoke(stypy.reporting.localization.Localization(__file__, 102, 11), applies_to_5238, *[proxy_obj_5239, callable__5240], **kwargs_5241)
        
        # Testing if the type of an if condition is none (line 102)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 8), applies_to_call_result_5242):
            pass
        else:
            
            # Testing the type of an if condition (line 102)
            if_condition_5243 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), applies_to_call_result_5242)
            # Assigning a type to the variable 'if_condition_5243' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'if_condition_5243', if_condition_5243)
            # SSA begins for if statement (line 102)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to get_parameter_arity(...): (line 103)
            # Processing the call arguments (line 103)
            # Getting the type of 'proxy_obj' (line 103)
            proxy_obj_5246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 63), 'proxy_obj', False)
            # Getting the type of 'callable_' (line 103)
            callable__5247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 74), 'callable_', False)
            # Processing the call keyword arguments (line 103)
            kwargs_5248 = {}
            # Getting the type of 'rule_based_call_handler' (line 103)
            rule_based_call_handler_5244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'rule_based_call_handler', False)
            # Obtaining the member 'get_parameter_arity' of a type (line 103)
            get_parameter_arity_5245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 19), rule_based_call_handler_5244, 'get_parameter_arity')
            # Calling get_parameter_arity(args, kwargs) (line 103)
            get_parameter_arity_call_result_5249 = invoke(stypy.reporting.localization.Localization(__file__, 103, 19), get_parameter_arity_5245, *[proxy_obj_5246, callable__5247], **kwargs_5248)
            
            # Assigning a type to the variable 'stypy_return_type' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'stypy_return_type', get_parameter_arity_call_result_5249)
            # SSA join for if statement (line 102)
            module_type_store = module_type_store.join_ssa_context()
            


        if (may_be_5215 and more_types_in_union_5216):
            # SSA join for if statement (line 95)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 105)
    tuple_5250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 105)
    # Adding element type (line 105)
    
    # Obtaining an instance of the builtin type 'list' (line 105)
    list_5251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 105)
    # Adding element type (line 105)
    int_5252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 11), list_5251, int_5252)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 11), tuple_5250, list_5251)
    # Adding element type (line 105)
    # Getting the type of 'False' (line 105)
    False_5253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 17), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 11), tuple_5250, False_5253)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', tuple_5250)
    
    # ################# End of 'get_param_arity(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_param_arity' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_5254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5254)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_param_arity'
    return stypy_return_type_5254

# Assigning a type to the variable 'get_param_arity' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'get_param_arity', get_param_arity)

@norecursion
def perform_call(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'perform_call'
    module_type_store = module_type_store.open_function_context('perform_call', 108, 0, False)
    
    # Passed parameters checking function
    perform_call.stypy_localization = localization
    perform_call.stypy_type_of_self = None
    perform_call.stypy_type_store = module_type_store
    perform_call.stypy_function_name = 'perform_call'
    perform_call.stypy_param_names_list = ['proxy_obj', 'callable_', 'localization']
    perform_call.stypy_varargs_param_name = 'args'
    perform_call.stypy_kwargs_param_name = 'kwargs'
    perform_call.stypy_call_defaults = defaults
    perform_call.stypy_call_varargs = varargs
    perform_call.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'perform_call', ['proxy_obj', 'callable_', 'localization'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'perform_call', localization, ['proxy_obj', 'callable_', 'localization'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'perform_call(...)' code ##################

    str_5255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, (-1)), 'str', '\n    Perform the type inference of the call to the callable entity, using the passed arguments and a suitable\n    call handler to resolve the call (see above).\n\n    :param proxy_obj: TypeInferenceProxy representing the callable\n    :param callable_: Python callable entity\n    :param localization: Caller information\n    :param args: named arguments plus variable list of arguments\n    :param kwargs: keyword arguments plus defaults\n    :return: The return type of the called element\n    ')
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to list(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'args' (line 122)
    args_5257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'args', False)
    # Processing the call keyword arguments (line 122)
    kwargs_5258 = {}
    # Getting the type of 'list' (line 122)
    list_5256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'list', False)
    # Calling list(args, kwargs) (line 122)
    list_call_result_5259 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), list_5256, *[args_5257], **kwargs_5258)
    
    # Assigning a type to the variable 'arg_types' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'arg_types', list_call_result_5259)
    
    # Assigning a Name to a Name (line 123):
    
    # Assigning a Name to a Name (line 123):
    # Getting the type of 'kwargs' (line 123)
    kwargs_5260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 18), 'kwargs')
    # Assigning a type to the variable 'kwarg_types' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'kwarg_types', kwargs_5260)
    
    # Assigning a Name to a Name (line 131):
    
    # Assigning a Name to a Name (line 131):
    # Getting the type of 'None' (line 131)
    None_5261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'None')
    # Assigning a type to the variable 'unfolded_arg_tuples' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'unfolded_arg_tuples', None_5261)
    
    # Assigning a Name to a Name (line 132):
    
    # Assigning a Name to a Name (line 132):
    # Getting the type of 'None' (line 132)
    None_5262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 18), 'None')
    # Assigning a type to the variable 'return_type' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'return_type', None_5262)
    
    # Assigning a Name to a Name (line 133):
    
    # Assigning a Name to a Name (line 133):
    # Getting the type of 'False' (line 133)
    False_5263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'False')
    # Assigning a type to the variable 'found_valid_call' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'found_valid_call', False_5263)
    
    # Assigning a List to a Name (line 134):
    
    # Assigning a List to a Name (line 134):
    
    # Obtaining an instance of the builtin type 'list' (line 134)
    list_5264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 134)
    
    # Assigning a type to the variable 'found_errors' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'found_errors', list_5264)
    
    # Assigning a Name to a Name (line 135):
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'False' (line 135)
    False_5265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'False')
    # Assigning a type to the variable 'found_type_errors' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'found_type_errors', False_5265)
    
    
    # SSA begins for try-except statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Getting the type of 'registered_call_handlers' (line 139)
    registered_call_handlers_5266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'registered_call_handlers')
    # Assigning a type to the variable 'registered_call_handlers_5266' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'registered_call_handlers_5266', registered_call_handlers_5266)
    # Testing if the for loop is going to be iterated (line 139)
    # Testing the type of a for loop iterable (line 139)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 139, 8), registered_call_handlers_5266)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 139, 8), registered_call_handlers_5266):
        # Getting the type of the for loop variable (line 139)
        for_loop_var_5267 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 139, 8), registered_call_handlers_5266)
        # Assigning a type to the variable 'call_handler' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'call_handler', for_loop_var_5267)
        # SSA begins for a for statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to applies_to(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'proxy_obj' (line 141)
        proxy_obj_5270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 39), 'proxy_obj', False)
        # Getting the type of 'callable_' (line 141)
        callable__5271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 50), 'callable_', False)
        # Processing the call keyword arguments (line 141)
        kwargs_5272 = {}
        # Getting the type of 'call_handler' (line 141)
        call_handler_5268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'call_handler', False)
        # Obtaining the member 'applies_to' of a type (line 141)
        applies_to_5269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), call_handler_5268, 'applies_to')
        # Calling applies_to(args, kwargs) (line 141)
        applies_to_call_result_5273 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), applies_to_5269, *[proxy_obj_5270, callable__5271], **kwargs_5272)
        
        # Testing if the type of an if condition is none (line 141)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 141, 12), applies_to_call_result_5273):
            pass
        else:
            
            # Testing the type of an if condition (line 141)
            if_condition_5274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 12), applies_to_call_result_5273)
            # Assigning a type to the variable 'if_condition_5274' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'if_condition_5274', if_condition_5274)
            # SSA begins for if statement (line 141)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Tuple (line 146):
            
            # Assigning a Call to a Name:
            
            # Call to check_undefined_type_within_parameters(...): (line 146)
            # Processing the call arguments (line 146)
            # Getting the type of 'localization' (line 146)
            localization_5276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 80), 'localization', False)
            
            # Call to format_call(...): (line 147)
            # Processing the call arguments (line 147)
            # Getting the type of 'callable_' (line 147)
            callable__5278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 92), 'callable_', False)
            # Getting the type of 'arg_types' (line 147)
            arg_types_5279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 103), 'arg_types', False)
            # Getting the type of 'kwarg_types' (line 148)
            kwarg_types_5280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 92), 'kwarg_types', False)
            # Processing the call keyword arguments (line 147)
            kwargs_5281 = {}
            # Getting the type of 'format_call' (line 147)
            format_call_5277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 80), 'format_call', False)
            # Calling format_call(args, kwargs) (line 147)
            format_call_call_result_5282 = invoke(stypy.reporting.localization.Localization(__file__, 147, 80), format_call_5277, *[callable__5278, arg_types_5279, kwarg_types_5280], **kwargs_5281)
            
            # Getting the type of 'arg_types' (line 149)
            arg_types_5283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 81), 'arg_types', False)
            # Processing the call keyword arguments (line 146)
            # Getting the type of 'kwarg_types' (line 149)
            kwarg_types_5284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 94), 'kwarg_types', False)
            kwargs_5285 = {'kwarg_types_5284': kwarg_types_5284}
            # Getting the type of 'check_undefined_type_within_parameters' (line 146)
            check_undefined_type_within_parameters_5275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 41), 'check_undefined_type_within_parameters', False)
            # Calling check_undefined_type_within_parameters(args, kwargs) (line 146)
            check_undefined_type_within_parameters_call_result_5286 = invoke(stypy.reporting.localization.Localization(__file__, 146, 41), check_undefined_type_within_parameters_5275, *[localization_5276, format_call_call_result_5282, arg_types_5283], **kwargs_5285)
            
            # Assigning a type to the variable 'call_assignment_5175' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_5175', check_undefined_type_within_parameters_call_result_5286)
            
            # Assigning a Call to a Name (line 146):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_5175' (line 146)
            call_assignment_5175_5287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_5175', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_5288 = stypy_get_value_from_tuple(call_assignment_5175_5287, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_5176' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_5176', stypy_get_value_from_tuple_call_result_5288)
            
            # Assigning a Name to a Name (line 146):
            # Getting the type of 'call_assignment_5176' (line 146)
            call_assignment_5176_5289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_5176')
            # Assigning a type to the variable 'arg_types' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'arg_types', call_assignment_5176_5289)
            
            # Assigning a Call to a Name (line 146):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_5175' (line 146)
            call_assignment_5175_5290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_5175', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_5291 = stypy_get_value_from_tuple(call_assignment_5175_5290, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_5177' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_5177', stypy_get_value_from_tuple_call_result_5291)
            
            # Assigning a Name to a Name (line 146):
            # Getting the type of 'call_assignment_5177' (line 146)
            call_assignment_5177_5292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_5177')
            # Assigning a type to the variable 'kwarg_types' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'kwarg_types', call_assignment_5177_5292)
            
            # Call to isinstance(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'call_handler' (line 152)
            call_handler_5294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'call_handler', False)
            # Getting the type of 'user_callables_call_handler_copy' (line 152)
            user_callables_call_handler_copy_5295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 44), 'user_callables_call_handler_copy', False)
            # Obtaining the member 'UserCallablesCallHandler' of a type (line 152)
            UserCallablesCallHandler_5296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 44), user_callables_call_handler_copy_5295, 'UserCallablesCallHandler')
            # Processing the call keyword arguments (line 152)
            kwargs_5297 = {}
            # Getting the type of 'isinstance' (line 152)
            isinstance_5293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 152)
            isinstance_call_result_5298 = invoke(stypy.reporting.localization.Localization(__file__, 152, 19), isinstance_5293, *[call_handler_5294, UserCallablesCallHandler_5296], **kwargs_5297)
            
            # Testing if the type of an if condition is none (line 152)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 152, 16), isinstance_call_result_5298):
                
                # Type idiom detected: calculating its left and rigth part (line 178)
                # Getting the type of 'unfolded_arg_tuples' (line 178)
                unfolded_arg_tuples_5319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'unfolded_arg_tuples')
                # Getting the type of 'None' (line 178)
                None_5320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 46), 'None')
                
                (may_be_5321, more_types_in_union_5322) = may_be_none(unfolded_arg_tuples_5319, None_5320)

                if may_be_5321:

                    if more_types_in_union_5322:
                        # Runtime conditional SSA (line 178)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Assigning a Call to a Name (line 181):
                    
                    # Assigning a Call to a Name (line 181):
                    
                    # Call to unfold_arguments(...): (line 181)
                    # Getting the type of 'arg_types' (line 181)
                    arg_types_5324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 64), 'arg_types', False)
                    # Processing the call keyword arguments (line 181)
                    # Getting the type of 'kwarg_types' (line 181)
                    kwarg_types_5325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 77), 'kwarg_types', False)
                    kwargs_5326 = {'kwarg_types_5325': kwarg_types_5325}
                    # Getting the type of 'unfold_arguments' (line 181)
                    unfold_arguments_5323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 46), 'unfold_arguments', False)
                    # Calling unfold_arguments(args, kwargs) (line 181)
                    unfold_arguments_call_result_5327 = invoke(stypy.reporting.localization.Localization(__file__, 181, 46), unfold_arguments_5323, *[arg_types_5324], **kwargs_5326)
                    
                    # Assigning a type to the variable 'unfolded_arg_tuples' (line 181)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'unfolded_arg_tuples', unfold_arguments_call_result_5327)

                    if more_types_in_union_5322:
                        # SSA join for if statement (line 178)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                # Getting the type of 'unfolded_arg_tuples' (line 184)
                unfolded_arg_tuples_5328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'unfolded_arg_tuples')
                # Assigning a type to the variable 'unfolded_arg_tuples_5328' (line 184)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'unfolded_arg_tuples_5328', unfolded_arg_tuples_5328)
                # Testing if the for loop is going to be iterated (line 184)
                # Testing the type of a for loop iterable (line 184)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 184, 20), unfolded_arg_tuples_5328)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 184, 20), unfolded_arg_tuples_5328):
                    # Getting the type of the for loop variable (line 184)
                    for_loop_var_5329 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 184, 20), unfolded_arg_tuples_5328)
                    # Assigning a type to the variable 'tuple_' (line 184)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'tuple_', for_loop_var_5329)
                    # SSA begins for a for statement (line 184)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to exist_a_type_error_within_parameters(...): (line 186)
                    # Processing the call arguments (line 186)
                    # Getting the type of 'localization' (line 186)
                    localization_5331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 64), 'localization', False)
                    
                    # Obtaining the type of the subscript
                    int_5332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 86), 'int')
                    # Getting the type of 'tuple_' (line 186)
                    tuple__5333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 79), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 186)
                    getitem___5334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 79), tuple__5333, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                    subscript_call_result_5335 = invoke(stypy.reporting.localization.Localization(__file__, 186, 79), getitem___5334, int_5332)
                    
                    # Processing the call keyword arguments (line 186)
                    
                    # Obtaining the type of the subscript
                    int_5336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 99), 'int')
                    # Getting the type of 'tuple_' (line 186)
                    tuple__5337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 92), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 186)
                    getitem___5338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 92), tuple__5337, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                    subscript_call_result_5339 = invoke(stypy.reporting.localization.Localization(__file__, 186, 92), getitem___5338, int_5336)
                    
                    kwargs_5340 = {'subscript_call_result_5339': subscript_call_result_5339}
                    # Getting the type of 'exist_a_type_error_within_parameters' (line 186)
                    exist_a_type_error_within_parameters_5330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'exist_a_type_error_within_parameters', False)
                    # Calling exist_a_type_error_within_parameters(args, kwargs) (line 186)
                    exist_a_type_error_within_parameters_call_result_5341 = invoke(stypy.reporting.localization.Localization(__file__, 186, 27), exist_a_type_error_within_parameters_5330, *[localization_5331, subscript_call_result_5335], **kwargs_5340)
                    
                    # Testing if the type of an if condition is none (line 186)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 186, 24), exist_a_type_error_within_parameters_call_result_5341):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 186)
                        if_condition_5342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 24), exist_a_type_error_within_parameters_call_result_5341)
                        # Assigning a type to the variable 'if_condition_5342' (line 186)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'if_condition_5342', if_condition_5342)
                        # SSA begins for if statement (line 186)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Name (line 187):
                        
                        # Assigning a Name to a Name (line 187):
                        # Getting the type of 'True' (line 187)
                        True_5343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 48), 'True')
                        # Assigning a type to the variable 'found_type_errors' (line 187)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'found_type_errors', True_5343)
                        # SSA join for if statement (line 186)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Assigning a Call to a Name (line 191):
                    
                    # Assigning a Call to a Name (line 191):
                    
                    # Call to call_handler(...): (line 191)
                    # Processing the call arguments (line 191)
                    # Getting the type of 'proxy_obj' (line 191)
                    proxy_obj_5345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 43), 'proxy_obj', False)
                    # Getting the type of 'localization' (line 191)
                    localization_5346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 54), 'localization', False)
                    # Getting the type of 'callable_' (line 191)
                    callable__5347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 68), 'callable_', False)
                    
                    # Obtaining the type of the subscript
                    int_5348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 87), 'int')
                    # Getting the type of 'tuple_' (line 191)
                    tuple__5349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 80), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 191)
                    getitem___5350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 80), tuple__5349, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
                    subscript_call_result_5351 = invoke(stypy.reporting.localization.Localization(__file__, 191, 80), getitem___5350, int_5348)
                    
                    # Processing the call keyword arguments (line 191)
                    
                    # Obtaining the type of the subscript
                    int_5352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 100), 'int')
                    # Getting the type of 'tuple_' (line 191)
                    tuple__5353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 93), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 191)
                    getitem___5354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 93), tuple__5353, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
                    subscript_call_result_5355 = invoke(stypy.reporting.localization.Localization(__file__, 191, 93), getitem___5354, int_5352)
                    
                    kwargs_5356 = {'subscript_call_result_5355': subscript_call_result_5355}
                    # Getting the type of 'call_handler' (line 191)
                    call_handler_5344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'call_handler', False)
                    # Calling call_handler(args, kwargs) (line 191)
                    call_handler_call_result_5357 = invoke(stypy.reporting.localization.Localization(__file__, 191, 30), call_handler_5344, *[proxy_obj_5345, localization_5346, callable__5347, subscript_call_result_5351], **kwargs_5356)
                    
                    # Assigning a type to the variable 'ret' (line 191)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 'ret', call_handler_call_result_5357)
                    
                    # Type idiom detected: calculating its left and rigth part (line 192)
                    # Getting the type of 'TypeError' (line 192)
                    TypeError_5358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 47), 'TypeError')
                    # Getting the type of 'ret' (line 192)
                    ret_5359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 42), 'ret')
                    
                    (may_be_5360, more_types_in_union_5361) = may_not_be_subtype(TypeError_5358, ret_5359)

                    if may_be_5360:

                        if more_types_in_union_5361:
                            # Runtime conditional SSA (line 192)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'ret' (line 192)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'ret', remove_subtype_from_union(ret_5359, TypeError))
                        
                        # Assigning a Name to a Name (line 195):
                        
                        # Assigning a Name to a Name (line 195):
                        # Getting the type of 'True' (line 195)
                        True_5362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 47), 'True')
                        # Assigning a type to the variable 'found_valid_call' (line 195)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 28), 'found_valid_call', True_5362)
                        
                        # Getting the type of 'registered_type_modifiers' (line 199)
                        registered_type_modifiers_5363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 44), 'registered_type_modifiers')
                        # Assigning a type to the variable 'registered_type_modifiers_5363' (line 199)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'registered_type_modifiers_5363', registered_type_modifiers_5363)
                        # Testing if the for loop is going to be iterated (line 199)
                        # Testing the type of a for loop iterable (line 199)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 199, 28), registered_type_modifiers_5363)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 199, 28), registered_type_modifiers_5363):
                            # Getting the type of the for loop variable (line 199)
                            for_loop_var_5364 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 199, 28), registered_type_modifiers_5363)
                            # Assigning a type to the variable 'modifier' (line 199)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'modifier', for_loop_var_5364)
                            # SSA begins for a for statement (line 199)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Call to applies_to(...): (line 200)
                            # Processing the call arguments (line 200)
                            # Getting the type of 'proxy_obj' (line 200)
                            proxy_obj_5367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 55), 'proxy_obj', False)
                            # Getting the type of 'callable_' (line 200)
                            callable__5368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 66), 'callable_', False)
                            # Processing the call keyword arguments (line 200)
                            kwargs_5369 = {}
                            # Getting the type of 'modifier' (line 200)
                            modifier_5365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 35), 'modifier', False)
                            # Obtaining the member 'applies_to' of a type (line 200)
                            applies_to_5366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 35), modifier_5365, 'applies_to')
                            # Calling applies_to(args, kwargs) (line 200)
                            applies_to_call_result_5370 = invoke(stypy.reporting.localization.Localization(__file__, 200, 35), applies_to_5366, *[proxy_obj_5367, callable__5368], **kwargs_5369)
                            
                            # Testing if the type of an if condition is none (line 200)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 200, 32), applies_to_call_result_5370):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 200)
                                if_condition_5371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 32), applies_to_call_result_5370)
                                # Assigning a type to the variable 'if_condition_5371' (line 200)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'if_condition_5371', if_condition_5371)
                                # SSA begins for if statement (line 200)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Evaluating a boolean operation
                                
                                # Call to ismethod(...): (line 201)
                                # Processing the call arguments (line 201)
                                # Getting the type of 'callable_' (line 201)
                                callable__5374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 56), 'callable_', False)
                                # Processing the call keyword arguments (line 201)
                                kwargs_5375 = {}
                                # Getting the type of 'inspect' (line 201)
                                inspect_5372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 39), 'inspect', False)
                                # Obtaining the member 'ismethod' of a type (line 201)
                                ismethod_5373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 39), inspect_5372, 'ismethod')
                                # Calling ismethod(args, kwargs) (line 201)
                                ismethod_call_result_5376 = invoke(stypy.reporting.localization.Localization(__file__, 201, 39), ismethod_5373, *[callable__5374], **kwargs_5375)
                                
                                
                                # Call to ismethoddescriptor(...): (line 201)
                                # Processing the call arguments (line 201)
                                # Getting the type of 'callable_' (line 201)
                                callable__5379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 97), 'callable_', False)
                                # Processing the call keyword arguments (line 201)
                                kwargs_5380 = {}
                                # Getting the type of 'inspect' (line 201)
                                inspect_5377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 70), 'inspect', False)
                                # Obtaining the member 'ismethoddescriptor' of a type (line 201)
                                ismethoddescriptor_5378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 70), inspect_5377, 'ismethoddescriptor')
                                # Calling ismethoddescriptor(args, kwargs) (line 201)
                                ismethoddescriptor_call_result_5381 = invoke(stypy.reporting.localization.Localization(__file__, 201, 70), ismethoddescriptor_5378, *[callable__5379], **kwargs_5380)
                                
                                # Applying the binary operator 'or' (line 201)
                                result_or_keyword_5382 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 39), 'or', ismethod_call_result_5376, ismethoddescriptor_call_result_5381)
                                
                                # Testing if the type of an if condition is none (line 201)

                                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 201, 36), result_or_keyword_5382):
                                    pass
                                else:
                                    
                                    # Testing the type of an if condition (line 201)
                                    if_condition_5383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 36), result_or_keyword_5382)
                                    # Assigning a type to the variable 'if_condition_5383' (line 201)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 36), 'if_condition_5383', if_condition_5383)
                                    # SSA begins for if statement (line 201)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                    
                                    
                                    # Call to is_type_instance(...): (line 203)
                                    # Processing the call keyword arguments (line 203)
                                    kwargs_5387 = {}
                                    # Getting the type of 'proxy_obj' (line 203)
                                    proxy_obj_5384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 47), 'proxy_obj', False)
                                    # Obtaining the member 'parent_proxy' of a type (line 203)
                                    parent_proxy_5385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 47), proxy_obj_5384, 'parent_proxy')
                                    # Obtaining the member 'is_type_instance' of a type (line 203)
                                    is_type_instance_5386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 47), parent_proxy_5385, 'is_type_instance')
                                    # Calling is_type_instance(args, kwargs) (line 203)
                                    is_type_instance_call_result_5388 = invoke(stypy.reporting.localization.Localization(__file__, 203, 47), is_type_instance_5386, *[], **kwargs_5387)
                                    
                                    # Applying the 'not' unary operator (line 203)
                                    result_not__5389 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 43), 'not', is_type_instance_call_result_5388)
                                    
                                    # Testing if the type of an if condition is none (line 203)

                                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 203, 40), result_not__5389):
                                        pass
                                    else:
                                        
                                        # Testing the type of an if condition (line 203)
                                        if_condition_5390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 40), result_not__5389)
                                        # Assigning a type to the variable 'if_condition_5390' (line 203)
                                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 40), 'if_condition_5390', if_condition_5390)
                                        # SSA begins for if statement (line 203)
                                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                        
                                        # Assigning a Call to a Name (line 205):
                                        
                                        # Assigning a Call to a Name (line 205):
                                        
                                        # Call to modifier(...): (line 205)
                                        # Processing the call arguments (line 205)
                                        
                                        # Obtaining the type of the subscript
                                        int_5392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 69), 'int')
                                        
                                        # Obtaining the type of the subscript
                                        int_5393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 66), 'int')
                                        # Getting the type of 'tuple_' (line 205)
                                        tuple__5394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 59), 'tuple_', False)
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 59), tuple__5394, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5396 = invoke(stypy.reporting.localization.Localization(__file__, 205, 59), getitem___5395, int_5393)
                                        
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 59), subscript_call_result_5396, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5398 = invoke(stypy.reporting.localization.Localization(__file__, 205, 59), getitem___5397, int_5392)
                                        
                                        # Getting the type of 'localization' (line 205)
                                        localization_5399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 73), 'localization', False)
                                        # Getting the type of 'callable_' (line 205)
                                        callable__5400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 87), 'callable_', False)
                                        
                                        # Obtaining the type of the subscript
                                        int_5401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 109), 'int')
                                        slice_5402 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 205, 99), int_5401, None, None)
                                        
                                        # Obtaining the type of the subscript
                                        int_5403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 106), 'int')
                                        # Getting the type of 'tuple_' (line 205)
                                        tuple__5404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 99), 'tuple_', False)
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 99), tuple__5404, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5406 = invoke(stypy.reporting.localization.Localization(__file__, 205, 99), getitem___5405, int_5403)
                                        
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 99), subscript_call_result_5406, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5408 = invoke(stypy.reporting.localization.Localization(__file__, 205, 99), getitem___5407, slice_5402)
                                        
                                        # Processing the call keyword arguments (line 205)
                                        
                                        # Obtaining the type of the subscript
                                        int_5409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 68), 'int')
                                        # Getting the type of 'tuple_' (line 206)
                                        tuple__5410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 61), 'tuple_', False)
                                        # Obtaining the member '__getitem__' of a type (line 206)
                                        getitem___5411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 61), tuple__5410, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
                                        subscript_call_result_5412 = invoke(stypy.reporting.localization.Localization(__file__, 206, 61), getitem___5411, int_5409)
                                        
                                        kwargs_5413 = {'subscript_call_result_5412': subscript_call_result_5412}
                                        # Getting the type of 'modifier' (line 205)
                                        modifier_5391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 50), 'modifier', False)
                                        # Calling modifier(args, kwargs) (line 205)
                                        modifier_call_result_5414 = invoke(stypy.reporting.localization.Localization(__file__, 205, 50), modifier_5391, *[subscript_call_result_5398, localization_5399, callable__5400, subscript_call_result_5408], **kwargs_5413)
                                        
                                        # Assigning a type to the variable 'ret' (line 205)
                                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 44), 'ret', modifier_call_result_5414)
                                        # SSA join for if statement (line 203)
                                        module_type_store = module_type_store.join_ssa_context()
                                        

                                    # SSA join for if statement (line 201)
                                    module_type_store = module_type_store.join_ssa_context()
                                    

                                
                                # Assigning a Call to a Name (line 209):
                                
                                # Assigning a Call to a Name (line 209):
                                
                                # Call to modifier(...): (line 209)
                                # Processing the call arguments (line 209)
                                # Getting the type of 'proxy_obj' (line 209)
                                proxy_obj_5416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 51), 'proxy_obj', False)
                                # Getting the type of 'localization' (line 209)
                                localization_5417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 62), 'localization', False)
                                # Getting the type of 'callable_' (line 209)
                                callable__5418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 76), 'callable_', False)
                                
                                # Obtaining the type of the subscript
                                int_5419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 95), 'int')
                                # Getting the type of 'tuple_' (line 209)
                                tuple__5420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 88), 'tuple_', False)
                                # Obtaining the member '__getitem__' of a type (line 209)
                                getitem___5421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 88), tuple__5420, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 209)
                                subscript_call_result_5422 = invoke(stypy.reporting.localization.Localization(__file__, 209, 88), getitem___5421, int_5419)
                                
                                # Processing the call keyword arguments (line 209)
                                
                                # Obtaining the type of the subscript
                                int_5423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 108), 'int')
                                # Getting the type of 'tuple_' (line 209)
                                tuple__5424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 101), 'tuple_', False)
                                # Obtaining the member '__getitem__' of a type (line 209)
                                getitem___5425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 101), tuple__5424, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 209)
                                subscript_call_result_5426 = invoke(stypy.reporting.localization.Localization(__file__, 209, 101), getitem___5425, int_5423)
                                
                                kwargs_5427 = {'subscript_call_result_5426': subscript_call_result_5426}
                                # Getting the type of 'modifier' (line 209)
                                modifier_5415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 42), 'modifier', False)
                                # Calling modifier(args, kwargs) (line 209)
                                modifier_call_result_5428 = invoke(stypy.reporting.localization.Localization(__file__, 209, 42), modifier_5415, *[proxy_obj_5416, localization_5417, callable__5418, subscript_call_result_5422], **kwargs_5427)
                                
                                # Assigning a type to the variable 'ret' (line 209)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 36), 'ret', modifier_call_result_5428)
                                # SSA join for if statement (line 200)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        
                        # Assigning a Call to a Name (line 214):
                        
                        # Assigning a Call to a Name (line 214):
                        
                        # Call to add(...): (line 214)
                        # Processing the call arguments (line 214)
                        # Getting the type of 'return_type' (line 214)
                        return_type_5432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 72), 'return_type', False)
                        # Getting the type of 'ret' (line 214)
                        ret_5433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 85), 'ret', False)
                        # Processing the call keyword arguments (line 214)
                        kwargs_5434 = {}
                        # Getting the type of 'union_type_copy' (line 214)
                        union_type_copy_5429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 42), 'union_type_copy', False)
                        # Obtaining the member 'UnionType' of a type (line 214)
                        UnionType_5430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 42), union_type_copy_5429, 'UnionType')
                        # Obtaining the member 'add' of a type (line 214)
                        add_5431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 42), UnionType_5430, 'add')
                        # Calling add(args, kwargs) (line 214)
                        add_call_result_5435 = invoke(stypy.reporting.localization.Localization(__file__, 214, 42), add_5431, *[return_type_5432, ret_5433], **kwargs_5434)
                        
                        # Assigning a type to the variable 'return_type' (line 214)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'return_type', add_call_result_5435)

                        if more_types_in_union_5361:
                            # Runtime conditional SSA for else branch (line 192)
                            module_type_store.open_ssa_branch('idiom else')



                    if ((not may_be_5360) or more_types_in_union_5361):
                        # Assigning a type to the variable 'ret' (line 192)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'ret', remove_not_subtype_from_union(ret_5359, TypeError))
                        
                        # Call to append(...): (line 217)
                        # Processing the call arguments (line 217)
                        # Getting the type of 'ret' (line 217)
                        ret_5438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 48), 'ret', False)
                        # Processing the call keyword arguments (line 217)
                        kwargs_5439 = {}
                        # Getting the type of 'found_errors' (line 217)
                        found_errors_5436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 28), 'found_errors', False)
                        # Obtaining the member 'append' of a type (line 217)
                        append_5437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 28), found_errors_5436, 'append')
                        # Calling append(args, kwargs) (line 217)
                        append_call_result_5440 = invoke(stypy.reporting.localization.Localization(__file__, 217, 28), append_5437, *[ret_5438], **kwargs_5439)
                        

                        if (may_be_5360 and more_types_in_union_5361):
                            # SSA join for if statement (line 192)
                            module_type_store = module_type_store.join_ssa_context()


                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
            else:
                
                # Testing the type of an if condition (line 152)
                if_condition_5299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 16), isinstance_call_result_5298)
                # Assigning a type to the variable 'if_condition_5299' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'if_condition_5299', if_condition_5299)
                # SSA begins for if statement (line 152)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 154):
                
                # Assigning a Call to a Name (line 154):
                
                # Call to call_handler(...): (line 154)
                # Processing the call arguments (line 154)
                # Getting the type of 'proxy_obj' (line 154)
                proxy_obj_5301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 39), 'proxy_obj', False)
                # Getting the type of 'localization' (line 154)
                localization_5302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 50), 'localization', False)
                # Getting the type of 'callable_' (line 154)
                callable__5303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 64), 'callable_', False)
                # Getting the type of 'arg_types' (line 154)
                arg_types_5304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 76), 'arg_types', False)
                # Processing the call keyword arguments (line 154)
                # Getting the type of 'kwarg_types' (line 154)
                kwarg_types_5305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 89), 'kwarg_types', False)
                kwargs_5306 = {'kwarg_types_5305': kwarg_types_5305}
                # Getting the type of 'call_handler' (line 154)
                call_handler_5300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'call_handler', False)
                # Calling call_handler(args, kwargs) (line 154)
                call_handler_call_result_5307 = invoke(stypy.reporting.localization.Localization(__file__, 154, 26), call_handler_5300, *[proxy_obj_5301, localization_5302, callable__5303, arg_types_5304], **kwargs_5306)
                
                # Assigning a type to the variable 'ret' (line 154)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'ret', call_handler_call_result_5307)
                
                # Type idiom detected: calculating its left and rigth part (line 155)
                # Getting the type of 'TypeError' (line 155)
                TypeError_5308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 43), 'TypeError')
                # Getting the type of 'ret' (line 155)
                ret_5309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 38), 'ret')
                
                (may_be_5310, more_types_in_union_5311) = may_not_be_subtype(TypeError_5308, ret_5309)

                if may_be_5310:

                    if more_types_in_union_5311:
                        # Runtime conditional SSA (line 155)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'ret' (line 155)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'ret', remove_subtype_from_union(ret_5309, TypeError))
                    
                    # Assigning a Name to a Name (line 157):
                    
                    # Assigning a Name to a Name (line 157):
                    # Getting the type of 'True' (line 157)
                    True_5312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 43), 'True')
                    # Assigning a type to the variable 'found_valid_call' (line 157)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'found_valid_call', True_5312)
                    
                    # Assigning a Name to a Name (line 158):
                    
                    # Assigning a Name to a Name (line 158):
                    # Getting the type of 'ret' (line 158)
                    ret_5313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 38), 'ret')
                    # Assigning a type to the variable 'return_type' (line 158)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'return_type', ret_5313)

                    if more_types_in_union_5311:
                        # Runtime conditional SSA for else branch (line 155)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_5310) or more_types_in_union_5311):
                    # Assigning a type to the variable 'ret' (line 155)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'ret', remove_not_subtype_from_union(ret_5309, TypeError))
                    
                    # Call to append(...): (line 161)
                    # Processing the call arguments (line 161)
                    # Getting the type of 'ret' (line 161)
                    ret_5316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 44), 'ret', False)
                    # Processing the call keyword arguments (line 161)
                    kwargs_5317 = {}
                    # Getting the type of 'found_errors' (line 161)
                    found_errors_5314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'found_errors', False)
                    # Obtaining the member 'append' of a type (line 161)
                    append_5315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 24), found_errors_5314, 'append')
                    # Calling append(args, kwargs) (line 161)
                    append_call_result_5318 = invoke(stypy.reporting.localization.Localization(__file__, 161, 24), append_5315, *[ret_5316], **kwargs_5317)
                    

                    if (may_be_5310 and more_types_in_union_5311):
                        # SSA join for if statement (line 155)
                        module_type_store = module_type_store.join_ssa_context()


                
                # SSA branch for the else part of an if statement (line 152)
                module_type_store.open_ssa_branch('else')
                
                # Type idiom detected: calculating its left and rigth part (line 178)
                # Getting the type of 'unfolded_arg_tuples' (line 178)
                unfolded_arg_tuples_5319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'unfolded_arg_tuples')
                # Getting the type of 'None' (line 178)
                None_5320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 46), 'None')
                
                (may_be_5321, more_types_in_union_5322) = may_be_none(unfolded_arg_tuples_5319, None_5320)

                if may_be_5321:

                    if more_types_in_union_5322:
                        # Runtime conditional SSA (line 178)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Assigning a Call to a Name (line 181):
                    
                    # Assigning a Call to a Name (line 181):
                    
                    # Call to unfold_arguments(...): (line 181)
                    # Getting the type of 'arg_types' (line 181)
                    arg_types_5324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 64), 'arg_types', False)
                    # Processing the call keyword arguments (line 181)
                    # Getting the type of 'kwarg_types' (line 181)
                    kwarg_types_5325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 77), 'kwarg_types', False)
                    kwargs_5326 = {'kwarg_types_5325': kwarg_types_5325}
                    # Getting the type of 'unfold_arguments' (line 181)
                    unfold_arguments_5323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 46), 'unfold_arguments', False)
                    # Calling unfold_arguments(args, kwargs) (line 181)
                    unfold_arguments_call_result_5327 = invoke(stypy.reporting.localization.Localization(__file__, 181, 46), unfold_arguments_5323, *[arg_types_5324], **kwargs_5326)
                    
                    # Assigning a type to the variable 'unfolded_arg_tuples' (line 181)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'unfolded_arg_tuples', unfold_arguments_call_result_5327)

                    if more_types_in_union_5322:
                        # SSA join for if statement (line 178)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                # Getting the type of 'unfolded_arg_tuples' (line 184)
                unfolded_arg_tuples_5328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'unfolded_arg_tuples')
                # Assigning a type to the variable 'unfolded_arg_tuples_5328' (line 184)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'unfolded_arg_tuples_5328', unfolded_arg_tuples_5328)
                # Testing if the for loop is going to be iterated (line 184)
                # Testing the type of a for loop iterable (line 184)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 184, 20), unfolded_arg_tuples_5328)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 184, 20), unfolded_arg_tuples_5328):
                    # Getting the type of the for loop variable (line 184)
                    for_loop_var_5329 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 184, 20), unfolded_arg_tuples_5328)
                    # Assigning a type to the variable 'tuple_' (line 184)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'tuple_', for_loop_var_5329)
                    # SSA begins for a for statement (line 184)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to exist_a_type_error_within_parameters(...): (line 186)
                    # Processing the call arguments (line 186)
                    # Getting the type of 'localization' (line 186)
                    localization_5331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 64), 'localization', False)
                    
                    # Obtaining the type of the subscript
                    int_5332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 86), 'int')
                    # Getting the type of 'tuple_' (line 186)
                    tuple__5333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 79), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 186)
                    getitem___5334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 79), tuple__5333, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                    subscript_call_result_5335 = invoke(stypy.reporting.localization.Localization(__file__, 186, 79), getitem___5334, int_5332)
                    
                    # Processing the call keyword arguments (line 186)
                    
                    # Obtaining the type of the subscript
                    int_5336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 99), 'int')
                    # Getting the type of 'tuple_' (line 186)
                    tuple__5337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 92), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 186)
                    getitem___5338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 92), tuple__5337, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                    subscript_call_result_5339 = invoke(stypy.reporting.localization.Localization(__file__, 186, 92), getitem___5338, int_5336)
                    
                    kwargs_5340 = {'subscript_call_result_5339': subscript_call_result_5339}
                    # Getting the type of 'exist_a_type_error_within_parameters' (line 186)
                    exist_a_type_error_within_parameters_5330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'exist_a_type_error_within_parameters', False)
                    # Calling exist_a_type_error_within_parameters(args, kwargs) (line 186)
                    exist_a_type_error_within_parameters_call_result_5341 = invoke(stypy.reporting.localization.Localization(__file__, 186, 27), exist_a_type_error_within_parameters_5330, *[localization_5331, subscript_call_result_5335], **kwargs_5340)
                    
                    # Testing if the type of an if condition is none (line 186)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 186, 24), exist_a_type_error_within_parameters_call_result_5341):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 186)
                        if_condition_5342 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 24), exist_a_type_error_within_parameters_call_result_5341)
                        # Assigning a type to the variable 'if_condition_5342' (line 186)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'if_condition_5342', if_condition_5342)
                        # SSA begins for if statement (line 186)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Name (line 187):
                        
                        # Assigning a Name to a Name (line 187):
                        # Getting the type of 'True' (line 187)
                        True_5343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 48), 'True')
                        # Assigning a type to the variable 'found_type_errors' (line 187)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'found_type_errors', True_5343)
                        # SSA join for if statement (line 186)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Assigning a Call to a Name (line 191):
                    
                    # Assigning a Call to a Name (line 191):
                    
                    # Call to call_handler(...): (line 191)
                    # Processing the call arguments (line 191)
                    # Getting the type of 'proxy_obj' (line 191)
                    proxy_obj_5345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 43), 'proxy_obj', False)
                    # Getting the type of 'localization' (line 191)
                    localization_5346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 54), 'localization', False)
                    # Getting the type of 'callable_' (line 191)
                    callable__5347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 68), 'callable_', False)
                    
                    # Obtaining the type of the subscript
                    int_5348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 87), 'int')
                    # Getting the type of 'tuple_' (line 191)
                    tuple__5349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 80), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 191)
                    getitem___5350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 80), tuple__5349, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
                    subscript_call_result_5351 = invoke(stypy.reporting.localization.Localization(__file__, 191, 80), getitem___5350, int_5348)
                    
                    # Processing the call keyword arguments (line 191)
                    
                    # Obtaining the type of the subscript
                    int_5352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 100), 'int')
                    # Getting the type of 'tuple_' (line 191)
                    tuple__5353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 93), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 191)
                    getitem___5354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 93), tuple__5353, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
                    subscript_call_result_5355 = invoke(stypy.reporting.localization.Localization(__file__, 191, 93), getitem___5354, int_5352)
                    
                    kwargs_5356 = {'subscript_call_result_5355': subscript_call_result_5355}
                    # Getting the type of 'call_handler' (line 191)
                    call_handler_5344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'call_handler', False)
                    # Calling call_handler(args, kwargs) (line 191)
                    call_handler_call_result_5357 = invoke(stypy.reporting.localization.Localization(__file__, 191, 30), call_handler_5344, *[proxy_obj_5345, localization_5346, callable__5347, subscript_call_result_5351], **kwargs_5356)
                    
                    # Assigning a type to the variable 'ret' (line 191)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 'ret', call_handler_call_result_5357)
                    
                    # Type idiom detected: calculating its left and rigth part (line 192)
                    # Getting the type of 'TypeError' (line 192)
                    TypeError_5358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 47), 'TypeError')
                    # Getting the type of 'ret' (line 192)
                    ret_5359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 42), 'ret')
                    
                    (may_be_5360, more_types_in_union_5361) = may_not_be_subtype(TypeError_5358, ret_5359)

                    if may_be_5360:

                        if more_types_in_union_5361:
                            # Runtime conditional SSA (line 192)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'ret' (line 192)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'ret', remove_subtype_from_union(ret_5359, TypeError))
                        
                        # Assigning a Name to a Name (line 195):
                        
                        # Assigning a Name to a Name (line 195):
                        # Getting the type of 'True' (line 195)
                        True_5362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 47), 'True')
                        # Assigning a type to the variable 'found_valid_call' (line 195)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 28), 'found_valid_call', True_5362)
                        
                        # Getting the type of 'registered_type_modifiers' (line 199)
                        registered_type_modifiers_5363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 44), 'registered_type_modifiers')
                        # Assigning a type to the variable 'registered_type_modifiers_5363' (line 199)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'registered_type_modifiers_5363', registered_type_modifiers_5363)
                        # Testing if the for loop is going to be iterated (line 199)
                        # Testing the type of a for loop iterable (line 199)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 199, 28), registered_type_modifiers_5363)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 199, 28), registered_type_modifiers_5363):
                            # Getting the type of the for loop variable (line 199)
                            for_loop_var_5364 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 199, 28), registered_type_modifiers_5363)
                            # Assigning a type to the variable 'modifier' (line 199)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'modifier', for_loop_var_5364)
                            # SSA begins for a for statement (line 199)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Call to applies_to(...): (line 200)
                            # Processing the call arguments (line 200)
                            # Getting the type of 'proxy_obj' (line 200)
                            proxy_obj_5367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 55), 'proxy_obj', False)
                            # Getting the type of 'callable_' (line 200)
                            callable__5368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 66), 'callable_', False)
                            # Processing the call keyword arguments (line 200)
                            kwargs_5369 = {}
                            # Getting the type of 'modifier' (line 200)
                            modifier_5365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 35), 'modifier', False)
                            # Obtaining the member 'applies_to' of a type (line 200)
                            applies_to_5366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 35), modifier_5365, 'applies_to')
                            # Calling applies_to(args, kwargs) (line 200)
                            applies_to_call_result_5370 = invoke(stypy.reporting.localization.Localization(__file__, 200, 35), applies_to_5366, *[proxy_obj_5367, callable__5368], **kwargs_5369)
                            
                            # Testing if the type of an if condition is none (line 200)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 200, 32), applies_to_call_result_5370):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 200)
                                if_condition_5371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 32), applies_to_call_result_5370)
                                # Assigning a type to the variable 'if_condition_5371' (line 200)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'if_condition_5371', if_condition_5371)
                                # SSA begins for if statement (line 200)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Evaluating a boolean operation
                                
                                # Call to ismethod(...): (line 201)
                                # Processing the call arguments (line 201)
                                # Getting the type of 'callable_' (line 201)
                                callable__5374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 56), 'callable_', False)
                                # Processing the call keyword arguments (line 201)
                                kwargs_5375 = {}
                                # Getting the type of 'inspect' (line 201)
                                inspect_5372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 39), 'inspect', False)
                                # Obtaining the member 'ismethod' of a type (line 201)
                                ismethod_5373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 39), inspect_5372, 'ismethod')
                                # Calling ismethod(args, kwargs) (line 201)
                                ismethod_call_result_5376 = invoke(stypy.reporting.localization.Localization(__file__, 201, 39), ismethod_5373, *[callable__5374], **kwargs_5375)
                                
                                
                                # Call to ismethoddescriptor(...): (line 201)
                                # Processing the call arguments (line 201)
                                # Getting the type of 'callable_' (line 201)
                                callable__5379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 97), 'callable_', False)
                                # Processing the call keyword arguments (line 201)
                                kwargs_5380 = {}
                                # Getting the type of 'inspect' (line 201)
                                inspect_5377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 70), 'inspect', False)
                                # Obtaining the member 'ismethoddescriptor' of a type (line 201)
                                ismethoddescriptor_5378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 70), inspect_5377, 'ismethoddescriptor')
                                # Calling ismethoddescriptor(args, kwargs) (line 201)
                                ismethoddescriptor_call_result_5381 = invoke(stypy.reporting.localization.Localization(__file__, 201, 70), ismethoddescriptor_5378, *[callable__5379], **kwargs_5380)
                                
                                # Applying the binary operator 'or' (line 201)
                                result_or_keyword_5382 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 39), 'or', ismethod_call_result_5376, ismethoddescriptor_call_result_5381)
                                
                                # Testing if the type of an if condition is none (line 201)

                                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 201, 36), result_or_keyword_5382):
                                    pass
                                else:
                                    
                                    # Testing the type of an if condition (line 201)
                                    if_condition_5383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 36), result_or_keyword_5382)
                                    # Assigning a type to the variable 'if_condition_5383' (line 201)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 36), 'if_condition_5383', if_condition_5383)
                                    # SSA begins for if statement (line 201)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                    
                                    
                                    # Call to is_type_instance(...): (line 203)
                                    # Processing the call keyword arguments (line 203)
                                    kwargs_5387 = {}
                                    # Getting the type of 'proxy_obj' (line 203)
                                    proxy_obj_5384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 47), 'proxy_obj', False)
                                    # Obtaining the member 'parent_proxy' of a type (line 203)
                                    parent_proxy_5385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 47), proxy_obj_5384, 'parent_proxy')
                                    # Obtaining the member 'is_type_instance' of a type (line 203)
                                    is_type_instance_5386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 47), parent_proxy_5385, 'is_type_instance')
                                    # Calling is_type_instance(args, kwargs) (line 203)
                                    is_type_instance_call_result_5388 = invoke(stypy.reporting.localization.Localization(__file__, 203, 47), is_type_instance_5386, *[], **kwargs_5387)
                                    
                                    # Applying the 'not' unary operator (line 203)
                                    result_not__5389 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 43), 'not', is_type_instance_call_result_5388)
                                    
                                    # Testing if the type of an if condition is none (line 203)

                                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 203, 40), result_not__5389):
                                        pass
                                    else:
                                        
                                        # Testing the type of an if condition (line 203)
                                        if_condition_5390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 40), result_not__5389)
                                        # Assigning a type to the variable 'if_condition_5390' (line 203)
                                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 40), 'if_condition_5390', if_condition_5390)
                                        # SSA begins for if statement (line 203)
                                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                        
                                        # Assigning a Call to a Name (line 205):
                                        
                                        # Assigning a Call to a Name (line 205):
                                        
                                        # Call to modifier(...): (line 205)
                                        # Processing the call arguments (line 205)
                                        
                                        # Obtaining the type of the subscript
                                        int_5392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 69), 'int')
                                        
                                        # Obtaining the type of the subscript
                                        int_5393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 66), 'int')
                                        # Getting the type of 'tuple_' (line 205)
                                        tuple__5394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 59), 'tuple_', False)
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 59), tuple__5394, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5396 = invoke(stypy.reporting.localization.Localization(__file__, 205, 59), getitem___5395, int_5393)
                                        
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 59), subscript_call_result_5396, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5398 = invoke(stypy.reporting.localization.Localization(__file__, 205, 59), getitem___5397, int_5392)
                                        
                                        # Getting the type of 'localization' (line 205)
                                        localization_5399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 73), 'localization', False)
                                        # Getting the type of 'callable_' (line 205)
                                        callable__5400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 87), 'callable_', False)
                                        
                                        # Obtaining the type of the subscript
                                        int_5401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 109), 'int')
                                        slice_5402 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 205, 99), int_5401, None, None)
                                        
                                        # Obtaining the type of the subscript
                                        int_5403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 106), 'int')
                                        # Getting the type of 'tuple_' (line 205)
                                        tuple__5404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 99), 'tuple_', False)
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 99), tuple__5404, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5406 = invoke(stypy.reporting.localization.Localization(__file__, 205, 99), getitem___5405, int_5403)
                                        
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 99), subscript_call_result_5406, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5408 = invoke(stypy.reporting.localization.Localization(__file__, 205, 99), getitem___5407, slice_5402)
                                        
                                        # Processing the call keyword arguments (line 205)
                                        
                                        # Obtaining the type of the subscript
                                        int_5409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 68), 'int')
                                        # Getting the type of 'tuple_' (line 206)
                                        tuple__5410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 61), 'tuple_', False)
                                        # Obtaining the member '__getitem__' of a type (line 206)
                                        getitem___5411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 61), tuple__5410, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
                                        subscript_call_result_5412 = invoke(stypy.reporting.localization.Localization(__file__, 206, 61), getitem___5411, int_5409)
                                        
                                        kwargs_5413 = {'subscript_call_result_5412': subscript_call_result_5412}
                                        # Getting the type of 'modifier' (line 205)
                                        modifier_5391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 50), 'modifier', False)
                                        # Calling modifier(args, kwargs) (line 205)
                                        modifier_call_result_5414 = invoke(stypy.reporting.localization.Localization(__file__, 205, 50), modifier_5391, *[subscript_call_result_5398, localization_5399, callable__5400, subscript_call_result_5408], **kwargs_5413)
                                        
                                        # Assigning a type to the variable 'ret' (line 205)
                                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 44), 'ret', modifier_call_result_5414)
                                        # SSA join for if statement (line 203)
                                        module_type_store = module_type_store.join_ssa_context()
                                        

                                    # SSA join for if statement (line 201)
                                    module_type_store = module_type_store.join_ssa_context()
                                    

                                
                                # Assigning a Call to a Name (line 209):
                                
                                # Assigning a Call to a Name (line 209):
                                
                                # Call to modifier(...): (line 209)
                                # Processing the call arguments (line 209)
                                # Getting the type of 'proxy_obj' (line 209)
                                proxy_obj_5416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 51), 'proxy_obj', False)
                                # Getting the type of 'localization' (line 209)
                                localization_5417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 62), 'localization', False)
                                # Getting the type of 'callable_' (line 209)
                                callable__5418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 76), 'callable_', False)
                                
                                # Obtaining the type of the subscript
                                int_5419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 95), 'int')
                                # Getting the type of 'tuple_' (line 209)
                                tuple__5420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 88), 'tuple_', False)
                                # Obtaining the member '__getitem__' of a type (line 209)
                                getitem___5421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 88), tuple__5420, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 209)
                                subscript_call_result_5422 = invoke(stypy.reporting.localization.Localization(__file__, 209, 88), getitem___5421, int_5419)
                                
                                # Processing the call keyword arguments (line 209)
                                
                                # Obtaining the type of the subscript
                                int_5423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 108), 'int')
                                # Getting the type of 'tuple_' (line 209)
                                tuple__5424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 101), 'tuple_', False)
                                # Obtaining the member '__getitem__' of a type (line 209)
                                getitem___5425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 101), tuple__5424, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 209)
                                subscript_call_result_5426 = invoke(stypy.reporting.localization.Localization(__file__, 209, 101), getitem___5425, int_5423)
                                
                                kwargs_5427 = {'subscript_call_result_5426': subscript_call_result_5426}
                                # Getting the type of 'modifier' (line 209)
                                modifier_5415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 42), 'modifier', False)
                                # Calling modifier(args, kwargs) (line 209)
                                modifier_call_result_5428 = invoke(stypy.reporting.localization.Localization(__file__, 209, 42), modifier_5415, *[proxy_obj_5416, localization_5417, callable__5418, subscript_call_result_5422], **kwargs_5427)
                                
                                # Assigning a type to the variable 'ret' (line 209)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 36), 'ret', modifier_call_result_5428)
                                # SSA join for if statement (line 200)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        
                        # Assigning a Call to a Name (line 214):
                        
                        # Assigning a Call to a Name (line 214):
                        
                        # Call to add(...): (line 214)
                        # Processing the call arguments (line 214)
                        # Getting the type of 'return_type' (line 214)
                        return_type_5432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 72), 'return_type', False)
                        # Getting the type of 'ret' (line 214)
                        ret_5433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 85), 'ret', False)
                        # Processing the call keyword arguments (line 214)
                        kwargs_5434 = {}
                        # Getting the type of 'union_type_copy' (line 214)
                        union_type_copy_5429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 42), 'union_type_copy', False)
                        # Obtaining the member 'UnionType' of a type (line 214)
                        UnionType_5430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 42), union_type_copy_5429, 'UnionType')
                        # Obtaining the member 'add' of a type (line 214)
                        add_5431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 42), UnionType_5430, 'add')
                        # Calling add(args, kwargs) (line 214)
                        add_call_result_5435 = invoke(stypy.reporting.localization.Localization(__file__, 214, 42), add_5431, *[return_type_5432, ret_5433], **kwargs_5434)
                        
                        # Assigning a type to the variable 'return_type' (line 214)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'return_type', add_call_result_5435)

                        if more_types_in_union_5361:
                            # Runtime conditional SSA for else branch (line 192)
                            module_type_store.open_ssa_branch('idiom else')



                    if ((not may_be_5360) or more_types_in_union_5361):
                        # Assigning a type to the variable 'ret' (line 192)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'ret', remove_not_subtype_from_union(ret_5359, TypeError))
                        
                        # Call to append(...): (line 217)
                        # Processing the call arguments (line 217)
                        # Getting the type of 'ret' (line 217)
                        ret_5438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 48), 'ret', False)
                        # Processing the call keyword arguments (line 217)
                        kwargs_5439 = {}
                        # Getting the type of 'found_errors' (line 217)
                        found_errors_5436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 28), 'found_errors', False)
                        # Obtaining the member 'append' of a type (line 217)
                        append_5437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 28), found_errors_5436, 'append')
                        # Calling append(args, kwargs) (line 217)
                        append_call_result_5440 = invoke(stypy.reporting.localization.Localization(__file__, 217, 28), append_5437, *[ret_5438], **kwargs_5439)
                        

                        if (may_be_5360 and more_types_in_union_5361):
                            # SSA join for if statement (line 192)
                            module_type_store = module_type_store.join_ssa_context()


                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for if statement (line 152)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'found_valid_call' (line 221)
            found_valid_call_5441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), 'found_valid_call')
            # Testing if the type of an if condition is none (line 221)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 221, 16), found_valid_call_5441):
                
                
                # Call to len(...): (line 230)
                # Processing the call arguments (line 230)
                # Getting the type of 'found_errors' (line 230)
                found_errors_5451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 27), 'found_errors', False)
                # Processing the call keyword arguments (line 230)
                kwargs_5452 = {}
                # Getting the type of 'len' (line 230)
                len_5450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'len', False)
                # Calling len(args, kwargs) (line 230)
                len_call_result_5453 = invoke(stypy.reporting.localization.Localization(__file__, 230, 23), len_5450, *[found_errors_5451], **kwargs_5452)
                
                int_5454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 44), 'int')
                # Applying the binary operator '==' (line 230)
                result_eq_5455 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 23), '==', len_call_result_5453, int_5454)
                
                # Testing if the type of an if condition is none (line 230)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 230, 20), result_eq_5455):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 230)
                    if_condition_5456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 20), result_eq_5455)
                    # Assigning a type to the variable 'if_condition_5456' (line 230)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'if_condition_5456', if_condition_5456)
                    # SSA begins for if statement (line 230)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Obtaining the type of the subscript
                    int_5457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 44), 'int')
                    # Getting the type of 'found_errors' (line 231)
                    found_errors_5458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 31), 'found_errors')
                    # Obtaining the member '__getitem__' of a type (line 231)
                    getitem___5459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 31), found_errors_5458, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 231)
                    subscript_call_result_5460 = invoke(stypy.reporting.localization.Localization(__file__, 231, 31), getitem___5459, int_5457)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 231)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'stypy_return_type', subscript_call_result_5460)
                    # SSA join for if statement (line 230)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'found_errors' (line 237)
                found_errors_5461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 'found_errors')
                # Assigning a type to the variable 'found_errors_5461' (line 237)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'found_errors_5461', found_errors_5461)
                # Testing if the for loop is going to be iterated (line 237)
                # Testing the type of a for loop iterable (line 237)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 237, 20), found_errors_5461)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 237, 20), found_errors_5461):
                    # Getting the type of the for loop variable (line 237)
                    for_loop_var_5462 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 237, 20), found_errors_5461)
                    # Assigning a type to the variable 'error' (line 237)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'error', for_loop_var_5462)
                    # SSA begins for a for statement (line 237)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to remove_error_msg(...): (line 238)
                    # Processing the call arguments (line 238)
                    # Getting the type of 'error' (line 238)
                    error_5465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 51), 'error', False)
                    # Processing the call keyword arguments (line 238)
                    kwargs_5466 = {}
                    # Getting the type of 'TypeError' (line 238)
                    TypeError_5463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'TypeError', False)
                    # Obtaining the member 'remove_error_msg' of a type (line 238)
                    remove_error_msg_5464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 24), TypeError_5463, 'remove_error_msg')
                    # Calling remove_error_msg(args, kwargs) (line 238)
                    remove_error_msg_call_result_5467 = invoke(stypy.reporting.localization.Localization(__file__, 238, 24), remove_error_msg_5464, *[error_5465], **kwargs_5466)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a Call to a Name (line 240):
                
                # Assigning a Call to a Name (line 240):
                
                # Call to format_call(...): (line 240)
                # Processing the call arguments (line 240)
                # Getting the type of 'callable_' (line 240)
                callable__5469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 43), 'callable_', False)
                # Getting the type of 'arg_types' (line 240)
                arg_types_5470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 54), 'arg_types', False)
                # Getting the type of 'kwarg_types' (line 240)
                kwarg_types_5471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 65), 'kwarg_types', False)
                # Processing the call keyword arguments (line 240)
                kwargs_5472 = {}
                # Getting the type of 'format_call' (line 240)
                format_call_5468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'format_call', False)
                # Calling format_call(args, kwargs) (line 240)
                format_call_call_result_5473 = invoke(stypy.reporting.localization.Localization(__file__, 240, 31), format_call_5468, *[callable__5469, arg_types_5470, kwarg_types_5471], **kwargs_5472)
                
                # Assigning a type to the variable 'call_str' (line 240)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'call_str', format_call_call_result_5473)
                # Getting the type of 'found_type_errors' (line 241)
                found_type_errors_5474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 'found_type_errors')
                # Testing if the type of an if condition is none (line 241)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 241, 20), found_type_errors_5474):
                    
                    # Assigning a Str to a Name (line 244):
                    
                    # Assigning a Str to a Name (line 244):
                    str_5477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'str', 'The called entity do not accept any of these parameters')
                    # Assigning a type to the variable 'msg' (line 244)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'msg', str_5477)
                else:
                    
                    # Testing the type of an if condition (line 241)
                    if_condition_5475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 20), found_type_errors_5474)
                    # Assigning a type to the variable 'if_condition_5475' (line 241)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'if_condition_5475', if_condition_5475)
                    # SSA begins for if statement (line 241)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Name (line 242):
                    
                    # Assigning a Str to a Name (line 242):
                    str_5476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 30), 'str', 'Type errors found among the types of the call parameters')
                    # Assigning a type to the variable 'msg' (line 242)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'msg', str_5476)
                    # SSA branch for the else part of an if statement (line 241)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Str to a Name (line 244):
                    
                    # Assigning a Str to a Name (line 244):
                    str_5477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'str', 'The called entity do not accept any of these parameters')
                    # Assigning a type to the variable 'msg' (line 244)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'msg', str_5477)
                    # SSA join for if statement (line 241)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to TypeError(...): (line 246)
                # Processing the call arguments (line 246)
                # Getting the type of 'localization' (line 246)
                localization_5479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 37), 'localization', False)
                
                # Call to format(...): (line 246)
                # Processing the call arguments (line 246)
                # Getting the type of 'call_str' (line 246)
                call_str_5482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 69), 'call_str', False)
                # Getting the type of 'msg' (line 246)
                msg_5483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 79), 'msg', False)
                # Processing the call keyword arguments (line 246)
                kwargs_5484 = {}
                str_5480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 51), 'str', '{0}: {1}')
                # Obtaining the member 'format' of a type (line 246)
                format_5481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 51), str_5480, 'format')
                # Calling format(args, kwargs) (line 246)
                format_call_result_5485 = invoke(stypy.reporting.localization.Localization(__file__, 246, 51), format_5481, *[call_str_5482, msg_5483], **kwargs_5484)
                
                # Processing the call keyword arguments (line 246)
                kwargs_5486 = {}
                # Getting the type of 'TypeError' (line 246)
                TypeError_5478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 27), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 246)
                TypeError_call_result_5487 = invoke(stypy.reporting.localization.Localization(__file__, 246, 27), TypeError_5478, *[localization_5479, format_call_result_5485], **kwargs_5486)
                
                # Assigning a type to the variable 'stypy_return_type' (line 246)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'stypy_return_type', TypeError_call_result_5487)
            else:
                
                # Testing the type of an if condition (line 221)
                if_condition_5442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 16), found_valid_call_5441)
                # Assigning a type to the variable 'if_condition_5442' (line 221)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'if_condition_5442', if_condition_5442)
                # SSA begins for if statement (line 221)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'found_errors' (line 222)
                found_errors_5443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 33), 'found_errors')
                # Assigning a type to the variable 'found_errors_5443' (line 222)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'found_errors_5443', found_errors_5443)
                # Testing if the for loop is going to be iterated (line 222)
                # Testing the type of a for loop iterable (line 222)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 222, 20), found_errors_5443)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 222, 20), found_errors_5443):
                    # Getting the type of the for loop variable (line 222)
                    for_loop_var_5444 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 222, 20), found_errors_5443)
                    # Assigning a type to the variable 'error' (line 222)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'error', for_loop_var_5444)
                    # SSA begins for a for statement (line 222)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to turn_to_warning(...): (line 223)
                    # Processing the call keyword arguments (line 223)
                    kwargs_5447 = {}
                    # Getting the type of 'error' (line 223)
                    error_5445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 24), 'error', False)
                    # Obtaining the member 'turn_to_warning' of a type (line 223)
                    turn_to_warning_5446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 24), error_5445, 'turn_to_warning')
                    # Calling turn_to_warning(args, kwargs) (line 223)
                    turn_to_warning_call_result_5448 = invoke(stypy.reporting.localization.Localization(__file__, 223, 24), turn_to_warning_5446, *[], **kwargs_5447)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # Getting the type of 'return_type' (line 224)
                return_type_5449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'return_type')
                # Assigning a type to the variable 'stypy_return_type' (line 224)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'stypy_return_type', return_type_5449)
                # SSA branch for the else part of an if statement (line 221)
                module_type_store.open_ssa_branch('else')
                
                
                # Call to len(...): (line 230)
                # Processing the call arguments (line 230)
                # Getting the type of 'found_errors' (line 230)
                found_errors_5451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 27), 'found_errors', False)
                # Processing the call keyword arguments (line 230)
                kwargs_5452 = {}
                # Getting the type of 'len' (line 230)
                len_5450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'len', False)
                # Calling len(args, kwargs) (line 230)
                len_call_result_5453 = invoke(stypy.reporting.localization.Localization(__file__, 230, 23), len_5450, *[found_errors_5451], **kwargs_5452)
                
                int_5454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 44), 'int')
                # Applying the binary operator '==' (line 230)
                result_eq_5455 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 23), '==', len_call_result_5453, int_5454)
                
                # Testing if the type of an if condition is none (line 230)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 230, 20), result_eq_5455):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 230)
                    if_condition_5456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 20), result_eq_5455)
                    # Assigning a type to the variable 'if_condition_5456' (line 230)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'if_condition_5456', if_condition_5456)
                    # SSA begins for if statement (line 230)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Obtaining the type of the subscript
                    int_5457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 44), 'int')
                    # Getting the type of 'found_errors' (line 231)
                    found_errors_5458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 31), 'found_errors')
                    # Obtaining the member '__getitem__' of a type (line 231)
                    getitem___5459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 31), found_errors_5458, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 231)
                    subscript_call_result_5460 = invoke(stypy.reporting.localization.Localization(__file__, 231, 31), getitem___5459, int_5457)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 231)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'stypy_return_type', subscript_call_result_5460)
                    # SSA join for if statement (line 230)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'found_errors' (line 237)
                found_errors_5461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 'found_errors')
                # Assigning a type to the variable 'found_errors_5461' (line 237)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'found_errors_5461', found_errors_5461)
                # Testing if the for loop is going to be iterated (line 237)
                # Testing the type of a for loop iterable (line 237)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 237, 20), found_errors_5461)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 237, 20), found_errors_5461):
                    # Getting the type of the for loop variable (line 237)
                    for_loop_var_5462 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 237, 20), found_errors_5461)
                    # Assigning a type to the variable 'error' (line 237)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'error', for_loop_var_5462)
                    # SSA begins for a for statement (line 237)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to remove_error_msg(...): (line 238)
                    # Processing the call arguments (line 238)
                    # Getting the type of 'error' (line 238)
                    error_5465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 51), 'error', False)
                    # Processing the call keyword arguments (line 238)
                    kwargs_5466 = {}
                    # Getting the type of 'TypeError' (line 238)
                    TypeError_5463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'TypeError', False)
                    # Obtaining the member 'remove_error_msg' of a type (line 238)
                    remove_error_msg_5464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 24), TypeError_5463, 'remove_error_msg')
                    # Calling remove_error_msg(args, kwargs) (line 238)
                    remove_error_msg_call_result_5467 = invoke(stypy.reporting.localization.Localization(__file__, 238, 24), remove_error_msg_5464, *[error_5465], **kwargs_5466)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a Call to a Name (line 240):
                
                # Assigning a Call to a Name (line 240):
                
                # Call to format_call(...): (line 240)
                # Processing the call arguments (line 240)
                # Getting the type of 'callable_' (line 240)
                callable__5469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 43), 'callable_', False)
                # Getting the type of 'arg_types' (line 240)
                arg_types_5470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 54), 'arg_types', False)
                # Getting the type of 'kwarg_types' (line 240)
                kwarg_types_5471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 65), 'kwarg_types', False)
                # Processing the call keyword arguments (line 240)
                kwargs_5472 = {}
                # Getting the type of 'format_call' (line 240)
                format_call_5468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'format_call', False)
                # Calling format_call(args, kwargs) (line 240)
                format_call_call_result_5473 = invoke(stypy.reporting.localization.Localization(__file__, 240, 31), format_call_5468, *[callable__5469, arg_types_5470, kwarg_types_5471], **kwargs_5472)
                
                # Assigning a type to the variable 'call_str' (line 240)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'call_str', format_call_call_result_5473)
                # Getting the type of 'found_type_errors' (line 241)
                found_type_errors_5474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 'found_type_errors')
                # Testing if the type of an if condition is none (line 241)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 241, 20), found_type_errors_5474):
                    
                    # Assigning a Str to a Name (line 244):
                    
                    # Assigning a Str to a Name (line 244):
                    str_5477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'str', 'The called entity do not accept any of these parameters')
                    # Assigning a type to the variable 'msg' (line 244)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'msg', str_5477)
                else:
                    
                    # Testing the type of an if condition (line 241)
                    if_condition_5475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 20), found_type_errors_5474)
                    # Assigning a type to the variable 'if_condition_5475' (line 241)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'if_condition_5475', if_condition_5475)
                    # SSA begins for if statement (line 241)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Name (line 242):
                    
                    # Assigning a Str to a Name (line 242):
                    str_5476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 30), 'str', 'Type errors found among the types of the call parameters')
                    # Assigning a type to the variable 'msg' (line 242)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'msg', str_5476)
                    # SSA branch for the else part of an if statement (line 241)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Str to a Name (line 244):
                    
                    # Assigning a Str to a Name (line 244):
                    str_5477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'str', 'The called entity do not accept any of these parameters')
                    # Assigning a type to the variable 'msg' (line 244)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'msg', str_5477)
                    # SSA join for if statement (line 241)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to TypeError(...): (line 246)
                # Processing the call arguments (line 246)
                # Getting the type of 'localization' (line 246)
                localization_5479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 37), 'localization', False)
                
                # Call to format(...): (line 246)
                # Processing the call arguments (line 246)
                # Getting the type of 'call_str' (line 246)
                call_str_5482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 69), 'call_str', False)
                # Getting the type of 'msg' (line 246)
                msg_5483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 79), 'msg', False)
                # Processing the call keyword arguments (line 246)
                kwargs_5484 = {}
                str_5480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 51), 'str', '{0}: {1}')
                # Obtaining the member 'format' of a type (line 246)
                format_5481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 51), str_5480, 'format')
                # Calling format(args, kwargs) (line 246)
                format_call_result_5485 = invoke(stypy.reporting.localization.Localization(__file__, 246, 51), format_5481, *[call_str_5482, msg_5483], **kwargs_5484)
                
                # Processing the call keyword arguments (line 246)
                kwargs_5486 = {}
                # Getting the type of 'TypeError' (line 246)
                TypeError_5478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 27), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 246)
                TypeError_call_result_5487 = invoke(stypy.reporting.localization.Localization(__file__, 246, 27), TypeError_5478, *[localization_5479, format_call_result_5485], **kwargs_5486)
                
                # Assigning a type to the variable 'stypy_return_type' (line 246)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'stypy_return_type', TypeError_call_result_5487)
                # SSA join for if statement (line 221)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 141)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # SSA branch for the except part of a try statement (line 137)
    # SSA branch for the except 'Exception' branch of a try statement (line 137)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 248)
    Exception_5488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 11), 'Exception')
    # Assigning a type to the variable 'e' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'e', Exception_5488)
    
    # Call to TypeError(...): (line 250)
    # Processing the call arguments (line 250)
    # Getting the type of 'localization' (line 250)
    localization_5490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 25), 'localization', False)
    
    # Call to format(...): (line 250)
    # Processing the call arguments (line 250)
    # Getting the type of 'callable_' (line 251)
    callable__5493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'callable_', False)
    
    # Call to list(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'arg_types' (line 251)
    arg_types_5495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 28), 'arg_types', False)
    # Processing the call keyword arguments (line 251)
    kwargs_5496 = {}
    # Getting the type of 'list' (line 251)
    list_5494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'list', False)
    # Calling list(args, kwargs) (line 251)
    list_call_result_5497 = invoke(stypy.reporting.localization.Localization(__file__, 251, 23), list_5494, *[arg_types_5495], **kwargs_5496)
    
    
    # Call to list(...): (line 251)
    # Processing the call arguments (line 251)
    
    # Call to values(...): (line 251)
    # Processing the call keyword arguments (line 251)
    kwargs_5501 = {}
    # Getting the type of 'kwarg_types' (line 251)
    kwarg_types_5499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 46), 'kwarg_types', False)
    # Obtaining the member 'values' of a type (line 251)
    values_5500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 46), kwarg_types_5499, 'values')
    # Calling values(args, kwargs) (line 251)
    values_call_result_5502 = invoke(stypy.reporting.localization.Localization(__file__, 251, 46), values_5500, *[], **kwargs_5501)
    
    # Processing the call keyword arguments (line 251)
    kwargs_5503 = {}
    # Getting the type of 'list' (line 251)
    list_5498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 41), 'list', False)
    # Calling list(args, kwargs) (line 251)
    list_call_result_5504 = invoke(stypy.reporting.localization.Localization(__file__, 251, 41), list_5498, *[values_call_result_5502], **kwargs_5503)
    
    # Applying the binary operator '+' (line 251)
    result_add_5505 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 23), '+', list_call_result_5497, list_call_result_5504)
    
    # Getting the type of 'e' (line 251)
    e_5506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 69), 'e', False)
    # Processing the call keyword arguments (line 250)
    kwargs_5507 = {}
    str_5491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 39), 'str', "An error was produced when invoking '{0}' with arguments [{1}]: {2}")
    # Obtaining the member 'format' of a type (line 250)
    format_5492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 39), str_5491, 'format')
    # Calling format(args, kwargs) (line 250)
    format_call_result_5508 = invoke(stypy.reporting.localization.Localization(__file__, 250, 39), format_5492, *[callable__5493, result_add_5505, e_5506], **kwargs_5507)
    
    # Processing the call keyword arguments (line 250)
    kwargs_5509 = {}
    # Getting the type of 'TypeError' (line 250)
    TypeError_5489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 250)
    TypeError_call_result_5510 = invoke(stypy.reporting.localization.Localization(__file__, 250, 15), TypeError_5489, *[localization_5490, format_call_result_5508], **kwargs_5509)
    
    # Assigning a type to the variable 'stypy_return_type' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'stypy_return_type', TypeError_call_result_5510)
    # SSA join for try-except statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'perform_call(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'perform_call' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_5511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5511)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'perform_call'
    return stypy_return_type_5511

# Assigning a type to the variable 'perform_call' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'perform_call', perform_call)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
