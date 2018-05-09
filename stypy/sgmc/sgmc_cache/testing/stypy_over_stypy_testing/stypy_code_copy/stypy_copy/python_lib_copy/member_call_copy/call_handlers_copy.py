
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import inspect
2: 
3: from stypy_copy.python_lib_copy.member_call_copy.handlers_copy import type_rule_call_handler_copy
4: from stypy_copy.python_lib_copy.member_call_copy.handlers_copy import fake_param_values_call_handler_copy
5: from stypy_copy.python_lib_copy.member_call_copy.handlers_copy import user_callables_call_handler_copy
6: from stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy import file_type_modifier_copy
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

# 'from stypy_copy.python_lib_copy.member_call_copy.handlers_copy import type_rule_call_handler_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_4892 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy')

if (type(import_4892) is not StypyTypeError):

    if (import_4892 != 'pyd_module'):
        __import__(import_4892)
        sys_modules_4893 = sys.modules[import_4892]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy', sys_modules_4893.module_type_store, module_type_store, ['type_rule_call_handler_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_4893, sys_modules_4893.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.member_call_copy.handlers_copy import type_rule_call_handler_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy', None, module_type_store, ['type_rule_call_handler_copy'], [type_rule_call_handler_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy', import_4892)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from stypy_copy.python_lib_copy.member_call_copy.handlers_copy import fake_param_values_call_handler_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_4894 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy')

if (type(import_4894) is not StypyTypeError):

    if (import_4894 != 'pyd_module'):
        __import__(import_4894)
        sys_modules_4895 = sys.modules[import_4894]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy', sys_modules_4895.module_type_store, module_type_store, ['fake_param_values_call_handler_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_4895, sys_modules_4895.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.member_call_copy.handlers_copy import fake_param_values_call_handler_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy', None, module_type_store, ['fake_param_values_call_handler_copy'], [fake_param_values_call_handler_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy', import_4894)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from stypy_copy.python_lib_copy.member_call_copy.handlers_copy import user_callables_call_handler_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_4896 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy')

if (type(import_4896) is not StypyTypeError):

    if (import_4896 != 'pyd_module'):
        __import__(import_4896)
        sys_modules_4897 = sys.modules[import_4896]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy', sys_modules_4897.module_type_store, module_type_store, ['user_callables_call_handler_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_4897, sys_modules_4897.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.member_call_copy.handlers_copy import user_callables_call_handler_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy', None, module_type_store, ['user_callables_call_handler_copy'], [user_callables_call_handler_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy', import_4896)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy import file_type_modifier_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_4898 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy')

if (type(import_4898) is not StypyTypeError):

    if (import_4898 != 'pyd_module'):
        __import__(import_4898)
        sys_modules_4899 = sys.modules[import_4898]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy', sys_modules_4899.module_type_store, module_type_store, ['file_type_modifier_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_4899, sys_modules_4899.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy import file_type_modifier_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy', None, module_type_store, ['file_type_modifier_copy'], [file_type_modifier_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.member_call_copy.type_modifiers_copy', import_4898)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from arguments_unfolding_copy import ' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_4900 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'arguments_unfolding_copy')

if (type(import_4900) is not StypyTypeError):

    if (import_4900 != 'pyd_module'):
        __import__(import_4900)
        sys_modules_4901 = sys.modules[import_4900]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'arguments_unfolding_copy', sys_modules_4901.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_4901, sys_modules_4901.module_type_store, module_type_store)
    else:
        from arguments_unfolding_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'arguments_unfolding_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'arguments_unfolding_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'arguments_unfolding_copy', import_4900)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from call_handlers_helper_methods_copy import ' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')
import_4902 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'call_handlers_helper_methods_copy')

if (type(import_4902) is not StypyTypeError):

    if (import_4902 != 'pyd_module'):
        __import__(import_4902)
        sys_modules_4903 = sys.modules[import_4902]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'call_handlers_helper_methods_copy', sys_modules_4903.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_4903, sys_modules_4903.module_type_store, module_type_store)
    else:
        from call_handlers_helper_methods_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'call_handlers_helper_methods_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'call_handlers_helper_methods_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'call_handlers_helper_methods_copy', import_4902)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/')

str_4904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, (-1)), 'str', '\nCall handlers are the entities we use to perform calls to type inference code. There are several call handlers, as\nthe call strategy is different depending on the origin of the code to be called:\n\n- Rule-based call handlers: This is used with Python library modules and functions.\nSome of these elements may have a rule file associated. This rule file indicates the accepted\nparameters for this call and it expected return type depending on this parameters. This is the most powerful call\nhandler, as the rules we developed allows a wide range of type checking options that may be used to ensure valid\ncalls. However, rule files have to be developed for each Python module, and while we plan to develop rule files\nfor each one of them on a semi-automatic way, this is the last part of the stypy development process, which means\nthat not every module will have one. If no rule file is present, other call handler will take care of the call.\n\nType rules are read from a directory structure inside the library, so we can add them on a later stage of development\nwithout changing stypy source code.\n\n- User callables call handler: The existence of a rule-based call handler is justified by the inability to have the\ncode of Python library functions, as most of them are developed in C and the source code cannot be obtained anyway.\nHowever, user-coded .py files are processed and converted to a type inference equivalent program. The conversion\nof callable entities transform them to a callable form composed by two parameters: a list of variable arguments and\na list of keyword arguments (def converted_func(*args, **kwargs)) that are handled by the type inference code. This\ncall handler is the responsible of passing the parameters in this form, so we can call type inference code easily.\n\n- Fake param values call handler: The last-resort call handler, used in those Python library modules with no current\ntype rule file and external third-party code that cannot be transformed to type inference code because source code\nis not available. Calls to this type of code from type inference code will pass types instead of values to the call.\n For example, if we find in our program the call library_function_with_no_source_code(3, "hi") the type inference\n code we generate will turn this to library_function_with_no_source_code(*[int, str], **{}). As this call is not valid\n (the called function cannot be transformed to a type inference equivalent), this call handler obtains default\n predefined fake values for each passed type and phisically call the function with them in order to obtain a result.\n The type of this result is later returned to the type inference code. This is the functionality of this call handler.\n Note that this dynamically obtain the type of a call by performing the call, causing the execution of part of the\n real program instead of the type-inference equivalent, which is not optimal. However, it allows us to test a much\n wider array of programs initially, even if they use libraries and code that do not have the source available and\n have no type rule file attached to it. It is our goal, with time to rely on this call handler as less as possible.\n Note that if the passed type has an associated value, this value will be used instead of the default fake one. However,\n as we said, type values are only calculated in very limited cases.\n')

# Assigning a Call to a Name (line 49):

# Assigning a Call to a Name (line 49):

# Call to TypeRuleCallHandler(...): (line 49)
# Processing the call keyword arguments (line 49)
kwargs_4907 = {}
# Getting the type of 'type_rule_call_handler_copy' (line 49)
type_rule_call_handler_copy_4905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'type_rule_call_handler_copy', False)
# Obtaining the member 'TypeRuleCallHandler' of a type (line 49)
TypeRuleCallHandler_4906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 26), type_rule_call_handler_copy_4905, 'TypeRuleCallHandler')
# Calling TypeRuleCallHandler(args, kwargs) (line 49)
TypeRuleCallHandler_call_result_4908 = invoke(stypy.reporting.localization.Localization(__file__, 49, 26), TypeRuleCallHandler_4906, *[], **kwargs_4907)

# Assigning a type to the variable 'rule_based_call_handler' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'rule_based_call_handler', TypeRuleCallHandler_call_result_4908)
str_4909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', '\nHere we register, ordered by priority, those classes that handle member calls using different strategies to obtain\nthe return type of a callable that we described previously, once the type or the input parameters are obtained. Note\nthat all call handlers are singletons, stateless classes.\n')

# Assigning a List to a Name (line 56):

# Assigning a List to a Name (line 56):

# Obtaining an instance of the builtin type 'list' (line 56)
list_4910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 27), 'list')
# Adding type elements to the builtin type 'list' instance (line 56)
# Adding element type (line 56)
# Getting the type of 'rule_based_call_handler' (line 57)
rule_based_call_handler_4911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'rule_based_call_handler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 27), list_4910, rule_based_call_handler_4911)
# Adding element type (line 56)

# Call to UserCallablesCallHandler(...): (line 58)
# Processing the call keyword arguments (line 58)
kwargs_4914 = {}
# Getting the type of 'user_callables_call_handler_copy' (line 58)
user_callables_call_handler_copy_4912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'user_callables_call_handler_copy', False)
# Obtaining the member 'UserCallablesCallHandler' of a type (line 58)
UserCallablesCallHandler_4913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 4), user_callables_call_handler_copy_4912, 'UserCallablesCallHandler')
# Calling UserCallablesCallHandler(args, kwargs) (line 58)
UserCallablesCallHandler_call_result_4915 = invoke(stypy.reporting.localization.Localization(__file__, 58, 4), UserCallablesCallHandler_4913, *[], **kwargs_4914)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 27), list_4910, UserCallablesCallHandler_call_result_4915)
# Adding element type (line 56)

# Call to FakeParamValuesCallHandler(...): (line 59)
# Processing the call keyword arguments (line 59)
kwargs_4918 = {}
# Getting the type of 'fake_param_values_call_handler_copy' (line 59)
fake_param_values_call_handler_copy_4916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'fake_param_values_call_handler_copy', False)
# Obtaining the member 'FakeParamValuesCallHandler' of a type (line 59)
FakeParamValuesCallHandler_4917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 4), fake_param_values_call_handler_copy_4916, 'FakeParamValuesCallHandler')
# Calling FakeParamValuesCallHandler(args, kwargs) (line 59)
FakeParamValuesCallHandler_call_result_4919 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), FakeParamValuesCallHandler_4917, *[], **kwargs_4918)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 27), list_4910, FakeParamValuesCallHandler_call_result_4919)

# Assigning a type to the variable 'registered_call_handlers' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'registered_call_handlers', list_4910)
str_4920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', '\nA type modifier is an special class that is associated with type-rule call handler, complementing its functionality.\nAlthough the rules we developed are able to express the return type of a Python library call function in a lot of\ncases, there are cases when they are not enough to accurately express the shape of the return type of a function.\nThis is true when the return type is a collection of a certain type, for example. This is when a type modifier is\nused: once a type rule has been used to determine that the call is valid, a type modifier associated to this call\nis later called with the passed parameters to obtain a proper, more accurate return type than the expressed by the rule.\nNote that not every Python library callable will have a type modifier associated. In fact most of them will not have\none, as this is only used to improve type inference on certain specific callables, whose rule files are not enough for\nthat. If a certain callable has both a rule file return type and a type modifier return type, the latter takes\nprecedence.\n\nOnly a type modifier is present at the moment: The one that dynamically reads type modifier functions for a Python\n(.py) source file. Type modifiers are read from a directory structure inside the library, so we can add them on a\n later stage of development without changing stypy source code. Although only one type modifier is present, we\n developed this system to add more in the future, should the necessity arise.\n')

# Assigning a List to a Name (line 79):

# Assigning a List to a Name (line 79):

# Obtaining an instance of the builtin type 'list' (line 79)
list_4921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 79)
# Adding element type (line 79)

# Call to FileTypeModifier(...): (line 80)
# Processing the call keyword arguments (line 80)
kwargs_4924 = {}
# Getting the type of 'file_type_modifier_copy' (line 80)
file_type_modifier_copy_4922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'file_type_modifier_copy', False)
# Obtaining the member 'FileTypeModifier' of a type (line 80)
FileTypeModifier_4923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 4), file_type_modifier_copy_4922, 'FileTypeModifier')
# Calling FileTypeModifier(args, kwargs) (line 80)
FileTypeModifier_call_result_4925 = invoke(stypy.reporting.localization.Localization(__file__, 80, 4), FileTypeModifier_4923, *[], **kwargs_4924)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 28), list_4921, FileTypeModifier_call_result_4925)

# Assigning a type to the variable 'registered_type_modifiers' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'registered_type_modifiers', list_4921)

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

    str_4926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, (-1)), 'str', '\n    Uses python introspection over the callable element to try to guess how many parameters can be passed to the\n    callable. If it is not possible (Python library functions do not have this data), we use the type rule call\n    handler to try to obtain them. If all fails, -1 is returned. This function also determines if the callable\n    uses a variable list of arguments.\n    :param proxy_obj: TypeInferenceProxy representing the callable\n    :param callable_: Python callable entity\n    :return: list of maximum passable arguments, has varargs tuple\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 95)
    str_4927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 26), 'str', 'im_func')
    # Getting the type of 'callable_' (line 95)
    callable__4928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'callable_')
    
    (may_be_4929, more_types_in_union_4930) = may_provide_member(str_4927, callable__4928)

    if may_be_4929:

        if more_types_in_union_4930:
            # Runtime conditional SSA (line 95)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'callable_' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'callable_', remove_not_member_provider_from_union(callable__4928, 'im_func'))
        
        # Assigning a Call to a Name (line 96):
        
        # Assigning a Call to a Name (line 96):
        
        # Call to getargspec(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'callable_' (line 96)
        callable__4933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 37), 'callable_', False)
        # Processing the call keyword arguments (line 96)
        kwargs_4934 = {}
        # Getting the type of 'inspect' (line 96)
        inspect_4931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'inspect', False)
        # Obtaining the member 'getargspec' of a type (line 96)
        getargspec_4932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 18), inspect_4931, 'getargspec')
        # Calling getargspec(args, kwargs) (line 96)
        getargspec_call_result_4935 = invoke(stypy.reporting.localization.Localization(__file__, 96, 18), getargspec_4932, *[callable__4933], **kwargs_4934)
        
        # Assigning a type to the variable 'argspec' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'argspec', getargspec_call_result_4935)
        
        # Assigning a BinOp to a Name (line 97):
        
        # Assigning a BinOp to a Name (line 97):
        
        # Call to len(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'argspec' (line 98)
        argspec_4937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'argspec', False)
        # Obtaining the member 'args' of a type (line 98)
        args_4938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), argspec_4937, 'args')
        # Processing the call keyword arguments (line 97)
        kwargs_4939 = {}
        # Getting the type of 'len' (line 97)
        len_4936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'len', False)
        # Calling len(args, kwargs) (line 97)
        len_call_result_4940 = invoke(stypy.reporting.localization.Localization(__file__, 97, 20), len_4936, *[args_4938], **kwargs_4939)
        
        int_4941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 28), 'int')
        # Applying the binary operator '-' (line 97)
        result_sub_4942 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 20), '-', len_call_result_4940, int_4941)
        
        # Assigning a type to the variable 'real_args' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'real_args', result_sub_4942)
        
        # Assigning a Compare to a Name (line 99):
        
        # Assigning a Compare to a Name (line 99):
        
        # Getting the type of 'argspec' (line 99)
        argspec_4943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 22), 'argspec')
        # Obtaining the member 'varargs' of a type (line 99)
        varargs_4944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 22), argspec_4943, 'varargs')
        # Getting the type of 'None' (line 99)
        None_4945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 45), 'None')
        # Applying the binary operator 'isnot' (line 99)
        result_is_not_4946 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 22), 'isnot', varargs_4944, None_4945)
        
        # Assigning a type to the variable 'has_varargs' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'has_varargs', result_is_not_4946)
        
        # Obtaining an instance of the builtin type 'tuple' (line 100)
        tuple_4947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 100)
        # Adding element type (line 100)
        
        # Obtaining an instance of the builtin type 'list' (line 100)
        list_4948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 100)
        # Adding element type (line 100)
        # Getting the type of 'real_args' (line 100)
        real_args_4949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'real_args')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 15), list_4948, real_args_4949)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 15), tuple_4947, list_4948)
        # Adding element type (line 100)
        # Getting the type of 'has_varargs' (line 100)
        has_varargs_4950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 28), 'has_varargs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 15), tuple_4947, has_varargs_4950)
        
        # Assigning a type to the variable 'stypy_return_type' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'stypy_return_type', tuple_4947)

        if more_types_in_union_4930:
            # Runtime conditional SSA for else branch (line 95)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_4929) or more_types_in_union_4930):
        # Assigning a type to the variable 'callable_' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'callable_', remove_member_provider_from_union(callable__4928, 'im_func'))
        
        # Call to applies_to(...): (line 102)
        # Processing the call arguments (line 102)
        # Getting the type of 'proxy_obj' (line 102)
        proxy_obj_4953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 46), 'proxy_obj', False)
        # Getting the type of 'callable_' (line 102)
        callable__4954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 57), 'callable_', False)
        # Processing the call keyword arguments (line 102)
        kwargs_4955 = {}
        # Getting the type of 'rule_based_call_handler' (line 102)
        rule_based_call_handler_4951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'rule_based_call_handler', False)
        # Obtaining the member 'applies_to' of a type (line 102)
        applies_to_4952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 11), rule_based_call_handler_4951, 'applies_to')
        # Calling applies_to(args, kwargs) (line 102)
        applies_to_call_result_4956 = invoke(stypy.reporting.localization.Localization(__file__, 102, 11), applies_to_4952, *[proxy_obj_4953, callable__4954], **kwargs_4955)
        
        # Testing if the type of an if condition is none (line 102)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 102, 8), applies_to_call_result_4956):
            pass
        else:
            
            # Testing the type of an if condition (line 102)
            if_condition_4957 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), applies_to_call_result_4956)
            # Assigning a type to the variable 'if_condition_4957' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'if_condition_4957', if_condition_4957)
            # SSA begins for if statement (line 102)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to get_parameter_arity(...): (line 103)
            # Processing the call arguments (line 103)
            # Getting the type of 'proxy_obj' (line 103)
            proxy_obj_4960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 63), 'proxy_obj', False)
            # Getting the type of 'callable_' (line 103)
            callable__4961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 74), 'callable_', False)
            # Processing the call keyword arguments (line 103)
            kwargs_4962 = {}
            # Getting the type of 'rule_based_call_handler' (line 103)
            rule_based_call_handler_4958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'rule_based_call_handler', False)
            # Obtaining the member 'get_parameter_arity' of a type (line 103)
            get_parameter_arity_4959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 19), rule_based_call_handler_4958, 'get_parameter_arity')
            # Calling get_parameter_arity(args, kwargs) (line 103)
            get_parameter_arity_call_result_4963 = invoke(stypy.reporting.localization.Localization(__file__, 103, 19), get_parameter_arity_4959, *[proxy_obj_4960, callable__4961], **kwargs_4962)
            
            # Assigning a type to the variable 'stypy_return_type' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'stypy_return_type', get_parameter_arity_call_result_4963)
            # SSA join for if statement (line 102)
            module_type_store = module_type_store.join_ssa_context()
            


        if (may_be_4929 and more_types_in_union_4930):
            # SSA join for if statement (line 95)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 105)
    tuple_4964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 105)
    # Adding element type (line 105)
    
    # Obtaining an instance of the builtin type 'list' (line 105)
    list_4965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 105)
    # Adding element type (line 105)
    int_4966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 11), list_4965, int_4966)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 11), tuple_4964, list_4965)
    # Adding element type (line 105)
    # Getting the type of 'False' (line 105)
    False_4967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 17), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 11), tuple_4964, False_4967)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', tuple_4964)
    
    # ################# End of 'get_param_arity(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_param_arity' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_4968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4968)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_param_arity'
    return stypy_return_type_4968

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

    str_4969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, (-1)), 'str', '\n    Perform the type inference of the call to the callable entity, using the passed arguments and a suitable\n    call handler to resolve the call (see above).\n\n    :param proxy_obj: TypeInferenceProxy representing the callable\n    :param callable_: Python callable entity\n    :param localization: Caller information\n    :param args: named arguments plus variable list of arguments\n    :param kwargs: keyword arguments plus defaults\n    :return: The return type of the called element\n    ')
    
    # Assigning a Call to a Name (line 122):
    
    # Assigning a Call to a Name (line 122):
    
    # Call to list(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'args' (line 122)
    args_4971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 21), 'args', False)
    # Processing the call keyword arguments (line 122)
    kwargs_4972 = {}
    # Getting the type of 'list' (line 122)
    list_4970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'list', False)
    # Calling list(args, kwargs) (line 122)
    list_call_result_4973 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), list_4970, *[args_4971], **kwargs_4972)
    
    # Assigning a type to the variable 'arg_types' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'arg_types', list_call_result_4973)
    
    # Assigning a Name to a Name (line 123):
    
    # Assigning a Name to a Name (line 123):
    # Getting the type of 'kwargs' (line 123)
    kwargs_4974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 18), 'kwargs')
    # Assigning a type to the variable 'kwarg_types' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'kwarg_types', kwargs_4974)
    
    # Assigning a Name to a Name (line 131):
    
    # Assigning a Name to a Name (line 131):
    # Getting the type of 'None' (line 131)
    None_4975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 26), 'None')
    # Assigning a type to the variable 'unfolded_arg_tuples' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'unfolded_arg_tuples', None_4975)
    
    # Assigning a Name to a Name (line 132):
    
    # Assigning a Name to a Name (line 132):
    # Getting the type of 'None' (line 132)
    None_4976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 18), 'None')
    # Assigning a type to the variable 'return_type' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'return_type', None_4976)
    
    # Assigning a Name to a Name (line 133):
    
    # Assigning a Name to a Name (line 133):
    # Getting the type of 'False' (line 133)
    False_4977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'False')
    # Assigning a type to the variable 'found_valid_call' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'found_valid_call', False_4977)
    
    # Assigning a List to a Name (line 134):
    
    # Assigning a List to a Name (line 134):
    
    # Obtaining an instance of the builtin type 'list' (line 134)
    list_4978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 134)
    
    # Assigning a type to the variable 'found_errors' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'found_errors', list_4978)
    
    # Assigning a Name to a Name (line 135):
    
    # Assigning a Name to a Name (line 135):
    # Getting the type of 'False' (line 135)
    False_4979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'False')
    # Assigning a type to the variable 'found_type_errors' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'found_type_errors', False_4979)
    
    
    # SSA begins for try-except statement (line 137)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Getting the type of 'registered_call_handlers' (line 139)
    registered_call_handlers_4980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'registered_call_handlers')
    # Assigning a type to the variable 'registered_call_handlers_4980' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'registered_call_handlers_4980', registered_call_handlers_4980)
    # Testing if the for loop is going to be iterated (line 139)
    # Testing the type of a for loop iterable (line 139)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 139, 8), registered_call_handlers_4980)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 139, 8), registered_call_handlers_4980):
        # Getting the type of the for loop variable (line 139)
        for_loop_var_4981 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 139, 8), registered_call_handlers_4980)
        # Assigning a type to the variable 'call_handler' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'call_handler', for_loop_var_4981)
        # SSA begins for a for statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to applies_to(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'proxy_obj' (line 141)
        proxy_obj_4984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 39), 'proxy_obj', False)
        # Getting the type of 'callable_' (line 141)
        callable__4985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 50), 'callable_', False)
        # Processing the call keyword arguments (line 141)
        kwargs_4986 = {}
        # Getting the type of 'call_handler' (line 141)
        call_handler_4982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'call_handler', False)
        # Obtaining the member 'applies_to' of a type (line 141)
        applies_to_4983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), call_handler_4982, 'applies_to')
        # Calling applies_to(args, kwargs) (line 141)
        applies_to_call_result_4987 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), applies_to_4983, *[proxy_obj_4984, callable__4985], **kwargs_4986)
        
        # Testing if the type of an if condition is none (line 141)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 141, 12), applies_to_call_result_4987):
            pass
        else:
            
            # Testing the type of an if condition (line 141)
            if_condition_4988 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 12), applies_to_call_result_4987)
            # Assigning a type to the variable 'if_condition_4988' (line 141)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'if_condition_4988', if_condition_4988)
            # SSA begins for if statement (line 141)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Tuple (line 146):
            
            # Assigning a Call to a Name:
            
            # Call to check_undefined_type_within_parameters(...): (line 146)
            # Processing the call arguments (line 146)
            # Getting the type of 'localization' (line 146)
            localization_4990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 80), 'localization', False)
            
            # Call to format_call(...): (line 147)
            # Processing the call arguments (line 147)
            # Getting the type of 'callable_' (line 147)
            callable__4992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 92), 'callable_', False)
            # Getting the type of 'arg_types' (line 147)
            arg_types_4993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 103), 'arg_types', False)
            # Getting the type of 'kwarg_types' (line 148)
            kwarg_types_4994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 92), 'kwarg_types', False)
            # Processing the call keyword arguments (line 147)
            kwargs_4995 = {}
            # Getting the type of 'format_call' (line 147)
            format_call_4991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 80), 'format_call', False)
            # Calling format_call(args, kwargs) (line 147)
            format_call_call_result_4996 = invoke(stypy.reporting.localization.Localization(__file__, 147, 80), format_call_4991, *[callable__4992, arg_types_4993, kwarg_types_4994], **kwargs_4995)
            
            # Getting the type of 'arg_types' (line 149)
            arg_types_4997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 81), 'arg_types', False)
            # Processing the call keyword arguments (line 146)
            # Getting the type of 'kwarg_types' (line 149)
            kwarg_types_4998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 94), 'kwarg_types', False)
            kwargs_4999 = {'kwarg_types_4998': kwarg_types_4998}
            # Getting the type of 'check_undefined_type_within_parameters' (line 146)
            check_undefined_type_within_parameters_4989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 41), 'check_undefined_type_within_parameters', False)
            # Calling check_undefined_type_within_parameters(args, kwargs) (line 146)
            check_undefined_type_within_parameters_call_result_5000 = invoke(stypy.reporting.localization.Localization(__file__, 146, 41), check_undefined_type_within_parameters_4989, *[localization_4990, format_call_call_result_4996, arg_types_4997], **kwargs_4999)
            
            # Assigning a type to the variable 'call_assignment_4889' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_4889', check_undefined_type_within_parameters_call_result_5000)
            
            # Assigning a Call to a Name (line 146):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4889' (line 146)
            call_assignment_4889_5001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_4889', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_5002 = stypy_get_value_from_tuple(call_assignment_4889_5001, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_4890' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_4890', stypy_get_value_from_tuple_call_result_5002)
            
            # Assigning a Name to a Name (line 146):
            # Getting the type of 'call_assignment_4890' (line 146)
            call_assignment_4890_5003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_4890')
            # Assigning a type to the variable 'arg_types' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'arg_types', call_assignment_4890_5003)
            
            # Assigning a Call to a Name (line 146):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_4889' (line 146)
            call_assignment_4889_5004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_4889', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_5005 = stypy_get_value_from_tuple(call_assignment_4889_5004, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_4891' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_4891', stypy_get_value_from_tuple_call_result_5005)
            
            # Assigning a Name to a Name (line 146):
            # Getting the type of 'call_assignment_4891' (line 146)
            call_assignment_4891_5006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'call_assignment_4891')
            # Assigning a type to the variable 'kwarg_types' (line 146)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 27), 'kwarg_types', call_assignment_4891_5006)
            
            # Call to isinstance(...): (line 152)
            # Processing the call arguments (line 152)
            # Getting the type of 'call_handler' (line 152)
            call_handler_5008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 30), 'call_handler', False)
            # Getting the type of 'user_callables_call_handler_copy' (line 152)
            user_callables_call_handler_copy_5009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 44), 'user_callables_call_handler_copy', False)
            # Obtaining the member 'UserCallablesCallHandler' of a type (line 152)
            UserCallablesCallHandler_5010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 44), user_callables_call_handler_copy_5009, 'UserCallablesCallHandler')
            # Processing the call keyword arguments (line 152)
            kwargs_5011 = {}
            # Getting the type of 'isinstance' (line 152)
            isinstance_5007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 152)
            isinstance_call_result_5012 = invoke(stypy.reporting.localization.Localization(__file__, 152, 19), isinstance_5007, *[call_handler_5008, UserCallablesCallHandler_5010], **kwargs_5011)
            
            # Testing if the type of an if condition is none (line 152)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 152, 16), isinstance_call_result_5012):
                
                # Type idiom detected: calculating its left and rigth part (line 178)
                # Getting the type of 'unfolded_arg_tuples' (line 178)
                unfolded_arg_tuples_5033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'unfolded_arg_tuples')
                # Getting the type of 'None' (line 178)
                None_5034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 46), 'None')
                
                (may_be_5035, more_types_in_union_5036) = may_be_none(unfolded_arg_tuples_5033, None_5034)

                if may_be_5035:

                    if more_types_in_union_5036:
                        # Runtime conditional SSA (line 178)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Assigning a Call to a Name (line 181):
                    
                    # Assigning a Call to a Name (line 181):
                    
                    # Call to unfold_arguments(...): (line 181)
                    # Getting the type of 'arg_types' (line 181)
                    arg_types_5038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 64), 'arg_types', False)
                    # Processing the call keyword arguments (line 181)
                    # Getting the type of 'kwarg_types' (line 181)
                    kwarg_types_5039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 77), 'kwarg_types', False)
                    kwargs_5040 = {'kwarg_types_5039': kwarg_types_5039}
                    # Getting the type of 'unfold_arguments' (line 181)
                    unfold_arguments_5037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 46), 'unfold_arguments', False)
                    # Calling unfold_arguments(args, kwargs) (line 181)
                    unfold_arguments_call_result_5041 = invoke(stypy.reporting.localization.Localization(__file__, 181, 46), unfold_arguments_5037, *[arg_types_5038], **kwargs_5040)
                    
                    # Assigning a type to the variable 'unfolded_arg_tuples' (line 181)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'unfolded_arg_tuples', unfold_arguments_call_result_5041)

                    if more_types_in_union_5036:
                        # SSA join for if statement (line 178)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                # Getting the type of 'unfolded_arg_tuples' (line 184)
                unfolded_arg_tuples_5042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'unfolded_arg_tuples')
                # Assigning a type to the variable 'unfolded_arg_tuples_5042' (line 184)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'unfolded_arg_tuples_5042', unfolded_arg_tuples_5042)
                # Testing if the for loop is going to be iterated (line 184)
                # Testing the type of a for loop iterable (line 184)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 184, 20), unfolded_arg_tuples_5042)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 184, 20), unfolded_arg_tuples_5042):
                    # Getting the type of the for loop variable (line 184)
                    for_loop_var_5043 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 184, 20), unfolded_arg_tuples_5042)
                    # Assigning a type to the variable 'tuple_' (line 184)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'tuple_', for_loop_var_5043)
                    # SSA begins for a for statement (line 184)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to exist_a_type_error_within_parameters(...): (line 186)
                    # Processing the call arguments (line 186)
                    # Getting the type of 'localization' (line 186)
                    localization_5045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 64), 'localization', False)
                    
                    # Obtaining the type of the subscript
                    int_5046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 86), 'int')
                    # Getting the type of 'tuple_' (line 186)
                    tuple__5047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 79), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 186)
                    getitem___5048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 79), tuple__5047, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                    subscript_call_result_5049 = invoke(stypy.reporting.localization.Localization(__file__, 186, 79), getitem___5048, int_5046)
                    
                    # Processing the call keyword arguments (line 186)
                    
                    # Obtaining the type of the subscript
                    int_5050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 99), 'int')
                    # Getting the type of 'tuple_' (line 186)
                    tuple__5051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 92), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 186)
                    getitem___5052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 92), tuple__5051, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                    subscript_call_result_5053 = invoke(stypy.reporting.localization.Localization(__file__, 186, 92), getitem___5052, int_5050)
                    
                    kwargs_5054 = {'subscript_call_result_5053': subscript_call_result_5053}
                    # Getting the type of 'exist_a_type_error_within_parameters' (line 186)
                    exist_a_type_error_within_parameters_5044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'exist_a_type_error_within_parameters', False)
                    # Calling exist_a_type_error_within_parameters(args, kwargs) (line 186)
                    exist_a_type_error_within_parameters_call_result_5055 = invoke(stypy.reporting.localization.Localization(__file__, 186, 27), exist_a_type_error_within_parameters_5044, *[localization_5045, subscript_call_result_5049], **kwargs_5054)
                    
                    # Testing if the type of an if condition is none (line 186)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 186, 24), exist_a_type_error_within_parameters_call_result_5055):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 186)
                        if_condition_5056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 24), exist_a_type_error_within_parameters_call_result_5055)
                        # Assigning a type to the variable 'if_condition_5056' (line 186)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'if_condition_5056', if_condition_5056)
                        # SSA begins for if statement (line 186)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Name (line 187):
                        
                        # Assigning a Name to a Name (line 187):
                        # Getting the type of 'True' (line 187)
                        True_5057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 48), 'True')
                        # Assigning a type to the variable 'found_type_errors' (line 187)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'found_type_errors', True_5057)
                        # SSA join for if statement (line 186)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Assigning a Call to a Name (line 191):
                    
                    # Assigning a Call to a Name (line 191):
                    
                    # Call to call_handler(...): (line 191)
                    # Processing the call arguments (line 191)
                    # Getting the type of 'proxy_obj' (line 191)
                    proxy_obj_5059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 43), 'proxy_obj', False)
                    # Getting the type of 'localization' (line 191)
                    localization_5060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 54), 'localization', False)
                    # Getting the type of 'callable_' (line 191)
                    callable__5061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 68), 'callable_', False)
                    
                    # Obtaining the type of the subscript
                    int_5062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 87), 'int')
                    # Getting the type of 'tuple_' (line 191)
                    tuple__5063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 80), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 191)
                    getitem___5064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 80), tuple__5063, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
                    subscript_call_result_5065 = invoke(stypy.reporting.localization.Localization(__file__, 191, 80), getitem___5064, int_5062)
                    
                    # Processing the call keyword arguments (line 191)
                    
                    # Obtaining the type of the subscript
                    int_5066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 100), 'int')
                    # Getting the type of 'tuple_' (line 191)
                    tuple__5067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 93), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 191)
                    getitem___5068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 93), tuple__5067, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
                    subscript_call_result_5069 = invoke(stypy.reporting.localization.Localization(__file__, 191, 93), getitem___5068, int_5066)
                    
                    kwargs_5070 = {'subscript_call_result_5069': subscript_call_result_5069}
                    # Getting the type of 'call_handler' (line 191)
                    call_handler_5058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'call_handler', False)
                    # Calling call_handler(args, kwargs) (line 191)
                    call_handler_call_result_5071 = invoke(stypy.reporting.localization.Localization(__file__, 191, 30), call_handler_5058, *[proxy_obj_5059, localization_5060, callable__5061, subscript_call_result_5065], **kwargs_5070)
                    
                    # Assigning a type to the variable 'ret' (line 191)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 'ret', call_handler_call_result_5071)
                    
                    # Type idiom detected: calculating its left and rigth part (line 192)
                    # Getting the type of 'TypeError' (line 192)
                    TypeError_5072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 47), 'TypeError')
                    # Getting the type of 'ret' (line 192)
                    ret_5073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 42), 'ret')
                    
                    (may_be_5074, more_types_in_union_5075) = may_not_be_subtype(TypeError_5072, ret_5073)

                    if may_be_5074:

                        if more_types_in_union_5075:
                            # Runtime conditional SSA (line 192)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'ret' (line 192)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'ret', remove_subtype_from_union(ret_5073, TypeError))
                        
                        # Assigning a Name to a Name (line 195):
                        
                        # Assigning a Name to a Name (line 195):
                        # Getting the type of 'True' (line 195)
                        True_5076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 47), 'True')
                        # Assigning a type to the variable 'found_valid_call' (line 195)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 28), 'found_valid_call', True_5076)
                        
                        # Getting the type of 'registered_type_modifiers' (line 199)
                        registered_type_modifiers_5077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 44), 'registered_type_modifiers')
                        # Assigning a type to the variable 'registered_type_modifiers_5077' (line 199)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'registered_type_modifiers_5077', registered_type_modifiers_5077)
                        # Testing if the for loop is going to be iterated (line 199)
                        # Testing the type of a for loop iterable (line 199)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 199, 28), registered_type_modifiers_5077)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 199, 28), registered_type_modifiers_5077):
                            # Getting the type of the for loop variable (line 199)
                            for_loop_var_5078 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 199, 28), registered_type_modifiers_5077)
                            # Assigning a type to the variable 'modifier' (line 199)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'modifier', for_loop_var_5078)
                            # SSA begins for a for statement (line 199)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Call to applies_to(...): (line 200)
                            # Processing the call arguments (line 200)
                            # Getting the type of 'proxy_obj' (line 200)
                            proxy_obj_5081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 55), 'proxy_obj', False)
                            # Getting the type of 'callable_' (line 200)
                            callable__5082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 66), 'callable_', False)
                            # Processing the call keyword arguments (line 200)
                            kwargs_5083 = {}
                            # Getting the type of 'modifier' (line 200)
                            modifier_5079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 35), 'modifier', False)
                            # Obtaining the member 'applies_to' of a type (line 200)
                            applies_to_5080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 35), modifier_5079, 'applies_to')
                            # Calling applies_to(args, kwargs) (line 200)
                            applies_to_call_result_5084 = invoke(stypy.reporting.localization.Localization(__file__, 200, 35), applies_to_5080, *[proxy_obj_5081, callable__5082], **kwargs_5083)
                            
                            # Testing if the type of an if condition is none (line 200)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 200, 32), applies_to_call_result_5084):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 200)
                                if_condition_5085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 32), applies_to_call_result_5084)
                                # Assigning a type to the variable 'if_condition_5085' (line 200)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'if_condition_5085', if_condition_5085)
                                # SSA begins for if statement (line 200)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Evaluating a boolean operation
                                
                                # Call to ismethod(...): (line 201)
                                # Processing the call arguments (line 201)
                                # Getting the type of 'callable_' (line 201)
                                callable__5088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 56), 'callable_', False)
                                # Processing the call keyword arguments (line 201)
                                kwargs_5089 = {}
                                # Getting the type of 'inspect' (line 201)
                                inspect_5086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 39), 'inspect', False)
                                # Obtaining the member 'ismethod' of a type (line 201)
                                ismethod_5087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 39), inspect_5086, 'ismethod')
                                # Calling ismethod(args, kwargs) (line 201)
                                ismethod_call_result_5090 = invoke(stypy.reporting.localization.Localization(__file__, 201, 39), ismethod_5087, *[callable__5088], **kwargs_5089)
                                
                                
                                # Call to ismethoddescriptor(...): (line 201)
                                # Processing the call arguments (line 201)
                                # Getting the type of 'callable_' (line 201)
                                callable__5093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 97), 'callable_', False)
                                # Processing the call keyword arguments (line 201)
                                kwargs_5094 = {}
                                # Getting the type of 'inspect' (line 201)
                                inspect_5091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 70), 'inspect', False)
                                # Obtaining the member 'ismethoddescriptor' of a type (line 201)
                                ismethoddescriptor_5092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 70), inspect_5091, 'ismethoddescriptor')
                                # Calling ismethoddescriptor(args, kwargs) (line 201)
                                ismethoddescriptor_call_result_5095 = invoke(stypy.reporting.localization.Localization(__file__, 201, 70), ismethoddescriptor_5092, *[callable__5093], **kwargs_5094)
                                
                                # Applying the binary operator 'or' (line 201)
                                result_or_keyword_5096 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 39), 'or', ismethod_call_result_5090, ismethoddescriptor_call_result_5095)
                                
                                # Testing if the type of an if condition is none (line 201)

                                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 201, 36), result_or_keyword_5096):
                                    pass
                                else:
                                    
                                    # Testing the type of an if condition (line 201)
                                    if_condition_5097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 36), result_or_keyword_5096)
                                    # Assigning a type to the variable 'if_condition_5097' (line 201)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 36), 'if_condition_5097', if_condition_5097)
                                    # SSA begins for if statement (line 201)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                    
                                    
                                    # Call to is_type_instance(...): (line 203)
                                    # Processing the call keyword arguments (line 203)
                                    kwargs_5101 = {}
                                    # Getting the type of 'proxy_obj' (line 203)
                                    proxy_obj_5098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 47), 'proxy_obj', False)
                                    # Obtaining the member 'parent_proxy' of a type (line 203)
                                    parent_proxy_5099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 47), proxy_obj_5098, 'parent_proxy')
                                    # Obtaining the member 'is_type_instance' of a type (line 203)
                                    is_type_instance_5100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 47), parent_proxy_5099, 'is_type_instance')
                                    # Calling is_type_instance(args, kwargs) (line 203)
                                    is_type_instance_call_result_5102 = invoke(stypy.reporting.localization.Localization(__file__, 203, 47), is_type_instance_5100, *[], **kwargs_5101)
                                    
                                    # Applying the 'not' unary operator (line 203)
                                    result_not__5103 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 43), 'not', is_type_instance_call_result_5102)
                                    
                                    # Testing if the type of an if condition is none (line 203)

                                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 203, 40), result_not__5103):
                                        pass
                                    else:
                                        
                                        # Testing the type of an if condition (line 203)
                                        if_condition_5104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 40), result_not__5103)
                                        # Assigning a type to the variable 'if_condition_5104' (line 203)
                                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 40), 'if_condition_5104', if_condition_5104)
                                        # SSA begins for if statement (line 203)
                                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                        
                                        # Assigning a Call to a Name (line 205):
                                        
                                        # Assigning a Call to a Name (line 205):
                                        
                                        # Call to modifier(...): (line 205)
                                        # Processing the call arguments (line 205)
                                        
                                        # Obtaining the type of the subscript
                                        int_5106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 69), 'int')
                                        
                                        # Obtaining the type of the subscript
                                        int_5107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 66), 'int')
                                        # Getting the type of 'tuple_' (line 205)
                                        tuple__5108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 59), 'tuple_', False)
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 59), tuple__5108, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5110 = invoke(stypy.reporting.localization.Localization(__file__, 205, 59), getitem___5109, int_5107)
                                        
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 59), subscript_call_result_5110, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5112 = invoke(stypy.reporting.localization.Localization(__file__, 205, 59), getitem___5111, int_5106)
                                        
                                        # Getting the type of 'localization' (line 205)
                                        localization_5113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 73), 'localization', False)
                                        # Getting the type of 'callable_' (line 205)
                                        callable__5114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 87), 'callable_', False)
                                        
                                        # Obtaining the type of the subscript
                                        int_5115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 109), 'int')
                                        slice_5116 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 205, 99), int_5115, None, None)
                                        
                                        # Obtaining the type of the subscript
                                        int_5117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 106), 'int')
                                        # Getting the type of 'tuple_' (line 205)
                                        tuple__5118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 99), 'tuple_', False)
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 99), tuple__5118, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5120 = invoke(stypy.reporting.localization.Localization(__file__, 205, 99), getitem___5119, int_5117)
                                        
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 99), subscript_call_result_5120, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5122 = invoke(stypy.reporting.localization.Localization(__file__, 205, 99), getitem___5121, slice_5116)
                                        
                                        # Processing the call keyword arguments (line 205)
                                        
                                        # Obtaining the type of the subscript
                                        int_5123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 68), 'int')
                                        # Getting the type of 'tuple_' (line 206)
                                        tuple__5124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 61), 'tuple_', False)
                                        # Obtaining the member '__getitem__' of a type (line 206)
                                        getitem___5125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 61), tuple__5124, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
                                        subscript_call_result_5126 = invoke(stypy.reporting.localization.Localization(__file__, 206, 61), getitem___5125, int_5123)
                                        
                                        kwargs_5127 = {'subscript_call_result_5126': subscript_call_result_5126}
                                        # Getting the type of 'modifier' (line 205)
                                        modifier_5105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 50), 'modifier', False)
                                        # Calling modifier(args, kwargs) (line 205)
                                        modifier_call_result_5128 = invoke(stypy.reporting.localization.Localization(__file__, 205, 50), modifier_5105, *[subscript_call_result_5112, localization_5113, callable__5114, subscript_call_result_5122], **kwargs_5127)
                                        
                                        # Assigning a type to the variable 'ret' (line 205)
                                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 44), 'ret', modifier_call_result_5128)
                                        # SSA join for if statement (line 203)
                                        module_type_store = module_type_store.join_ssa_context()
                                        

                                    # SSA join for if statement (line 201)
                                    module_type_store = module_type_store.join_ssa_context()
                                    

                                
                                # Assigning a Call to a Name (line 209):
                                
                                # Assigning a Call to a Name (line 209):
                                
                                # Call to modifier(...): (line 209)
                                # Processing the call arguments (line 209)
                                # Getting the type of 'proxy_obj' (line 209)
                                proxy_obj_5130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 51), 'proxy_obj', False)
                                # Getting the type of 'localization' (line 209)
                                localization_5131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 62), 'localization', False)
                                # Getting the type of 'callable_' (line 209)
                                callable__5132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 76), 'callable_', False)
                                
                                # Obtaining the type of the subscript
                                int_5133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 95), 'int')
                                # Getting the type of 'tuple_' (line 209)
                                tuple__5134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 88), 'tuple_', False)
                                # Obtaining the member '__getitem__' of a type (line 209)
                                getitem___5135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 88), tuple__5134, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 209)
                                subscript_call_result_5136 = invoke(stypy.reporting.localization.Localization(__file__, 209, 88), getitem___5135, int_5133)
                                
                                # Processing the call keyword arguments (line 209)
                                
                                # Obtaining the type of the subscript
                                int_5137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 108), 'int')
                                # Getting the type of 'tuple_' (line 209)
                                tuple__5138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 101), 'tuple_', False)
                                # Obtaining the member '__getitem__' of a type (line 209)
                                getitem___5139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 101), tuple__5138, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 209)
                                subscript_call_result_5140 = invoke(stypy.reporting.localization.Localization(__file__, 209, 101), getitem___5139, int_5137)
                                
                                kwargs_5141 = {'subscript_call_result_5140': subscript_call_result_5140}
                                # Getting the type of 'modifier' (line 209)
                                modifier_5129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 42), 'modifier', False)
                                # Calling modifier(args, kwargs) (line 209)
                                modifier_call_result_5142 = invoke(stypy.reporting.localization.Localization(__file__, 209, 42), modifier_5129, *[proxy_obj_5130, localization_5131, callable__5132, subscript_call_result_5136], **kwargs_5141)
                                
                                # Assigning a type to the variable 'ret' (line 209)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 36), 'ret', modifier_call_result_5142)
                                # SSA join for if statement (line 200)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        
                        # Assigning a Call to a Name (line 214):
                        
                        # Assigning a Call to a Name (line 214):
                        
                        # Call to add(...): (line 214)
                        # Processing the call arguments (line 214)
                        # Getting the type of 'return_type' (line 214)
                        return_type_5146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 72), 'return_type', False)
                        # Getting the type of 'ret' (line 214)
                        ret_5147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 85), 'ret', False)
                        # Processing the call keyword arguments (line 214)
                        kwargs_5148 = {}
                        # Getting the type of 'union_type_copy' (line 214)
                        union_type_copy_5143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 42), 'union_type_copy', False)
                        # Obtaining the member 'UnionType' of a type (line 214)
                        UnionType_5144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 42), union_type_copy_5143, 'UnionType')
                        # Obtaining the member 'add' of a type (line 214)
                        add_5145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 42), UnionType_5144, 'add')
                        # Calling add(args, kwargs) (line 214)
                        add_call_result_5149 = invoke(stypy.reporting.localization.Localization(__file__, 214, 42), add_5145, *[return_type_5146, ret_5147], **kwargs_5148)
                        
                        # Assigning a type to the variable 'return_type' (line 214)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'return_type', add_call_result_5149)

                        if more_types_in_union_5075:
                            # Runtime conditional SSA for else branch (line 192)
                            module_type_store.open_ssa_branch('idiom else')



                    if ((not may_be_5074) or more_types_in_union_5075):
                        # Assigning a type to the variable 'ret' (line 192)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'ret', remove_not_subtype_from_union(ret_5073, TypeError))
                        
                        # Call to append(...): (line 217)
                        # Processing the call arguments (line 217)
                        # Getting the type of 'ret' (line 217)
                        ret_5152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 48), 'ret', False)
                        # Processing the call keyword arguments (line 217)
                        kwargs_5153 = {}
                        # Getting the type of 'found_errors' (line 217)
                        found_errors_5150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 28), 'found_errors', False)
                        # Obtaining the member 'append' of a type (line 217)
                        append_5151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 28), found_errors_5150, 'append')
                        # Calling append(args, kwargs) (line 217)
                        append_call_result_5154 = invoke(stypy.reporting.localization.Localization(__file__, 217, 28), append_5151, *[ret_5152], **kwargs_5153)
                        

                        if (may_be_5074 and more_types_in_union_5075):
                            # SSA join for if statement (line 192)
                            module_type_store = module_type_store.join_ssa_context()


                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
            else:
                
                # Testing the type of an if condition (line 152)
                if_condition_5013 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 16), isinstance_call_result_5012)
                # Assigning a type to the variable 'if_condition_5013' (line 152)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 16), 'if_condition_5013', if_condition_5013)
                # SSA begins for if statement (line 152)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 154):
                
                # Assigning a Call to a Name (line 154):
                
                # Call to call_handler(...): (line 154)
                # Processing the call arguments (line 154)
                # Getting the type of 'proxy_obj' (line 154)
                proxy_obj_5015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 39), 'proxy_obj', False)
                # Getting the type of 'localization' (line 154)
                localization_5016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 50), 'localization', False)
                # Getting the type of 'callable_' (line 154)
                callable__5017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 64), 'callable_', False)
                # Getting the type of 'arg_types' (line 154)
                arg_types_5018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 76), 'arg_types', False)
                # Processing the call keyword arguments (line 154)
                # Getting the type of 'kwarg_types' (line 154)
                kwarg_types_5019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 89), 'kwarg_types', False)
                kwargs_5020 = {'kwarg_types_5019': kwarg_types_5019}
                # Getting the type of 'call_handler' (line 154)
                call_handler_5014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 26), 'call_handler', False)
                # Calling call_handler(args, kwargs) (line 154)
                call_handler_call_result_5021 = invoke(stypy.reporting.localization.Localization(__file__, 154, 26), call_handler_5014, *[proxy_obj_5015, localization_5016, callable__5017, arg_types_5018], **kwargs_5020)
                
                # Assigning a type to the variable 'ret' (line 154)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 20), 'ret', call_handler_call_result_5021)
                
                # Type idiom detected: calculating its left and rigth part (line 155)
                # Getting the type of 'TypeError' (line 155)
                TypeError_5022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 43), 'TypeError')
                # Getting the type of 'ret' (line 155)
                ret_5023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 38), 'ret')
                
                (may_be_5024, more_types_in_union_5025) = may_not_be_subtype(TypeError_5022, ret_5023)

                if may_be_5024:

                    if more_types_in_union_5025:
                        # Runtime conditional SSA (line 155)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'ret' (line 155)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'ret', remove_subtype_from_union(ret_5023, TypeError))
                    
                    # Assigning a Name to a Name (line 157):
                    
                    # Assigning a Name to a Name (line 157):
                    # Getting the type of 'True' (line 157)
                    True_5026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 43), 'True')
                    # Assigning a type to the variable 'found_valid_call' (line 157)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'found_valid_call', True_5026)
                    
                    # Assigning a Name to a Name (line 158):
                    
                    # Assigning a Name to a Name (line 158):
                    # Getting the type of 'ret' (line 158)
                    ret_5027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 38), 'ret')
                    # Assigning a type to the variable 'return_type' (line 158)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 24), 'return_type', ret_5027)

                    if more_types_in_union_5025:
                        # Runtime conditional SSA for else branch (line 155)
                        module_type_store.open_ssa_branch('idiom else')



                if ((not may_be_5024) or more_types_in_union_5025):
                    # Assigning a type to the variable 'ret' (line 155)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 20), 'ret', remove_not_subtype_from_union(ret_5023, TypeError))
                    
                    # Call to append(...): (line 161)
                    # Processing the call arguments (line 161)
                    # Getting the type of 'ret' (line 161)
                    ret_5030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 44), 'ret', False)
                    # Processing the call keyword arguments (line 161)
                    kwargs_5031 = {}
                    # Getting the type of 'found_errors' (line 161)
                    found_errors_5028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'found_errors', False)
                    # Obtaining the member 'append' of a type (line 161)
                    append_5029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 24), found_errors_5028, 'append')
                    # Calling append(args, kwargs) (line 161)
                    append_call_result_5032 = invoke(stypy.reporting.localization.Localization(__file__, 161, 24), append_5029, *[ret_5030], **kwargs_5031)
                    

                    if (may_be_5024 and more_types_in_union_5025):
                        # SSA join for if statement (line 155)
                        module_type_store = module_type_store.join_ssa_context()


                
                # SSA branch for the else part of an if statement (line 152)
                module_type_store.open_ssa_branch('else')
                
                # Type idiom detected: calculating its left and rigth part (line 178)
                # Getting the type of 'unfolded_arg_tuples' (line 178)
                unfolded_arg_tuples_5033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'unfolded_arg_tuples')
                # Getting the type of 'None' (line 178)
                None_5034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 46), 'None')
                
                (may_be_5035, more_types_in_union_5036) = may_be_none(unfolded_arg_tuples_5033, None_5034)

                if may_be_5035:

                    if more_types_in_union_5036:
                        # Runtime conditional SSA (line 178)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    
                    # Assigning a Call to a Name (line 181):
                    
                    # Assigning a Call to a Name (line 181):
                    
                    # Call to unfold_arguments(...): (line 181)
                    # Getting the type of 'arg_types' (line 181)
                    arg_types_5038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 64), 'arg_types', False)
                    # Processing the call keyword arguments (line 181)
                    # Getting the type of 'kwarg_types' (line 181)
                    kwarg_types_5039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 77), 'kwarg_types', False)
                    kwargs_5040 = {'kwarg_types_5039': kwarg_types_5039}
                    # Getting the type of 'unfold_arguments' (line 181)
                    unfold_arguments_5037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 46), 'unfold_arguments', False)
                    # Calling unfold_arguments(args, kwargs) (line 181)
                    unfold_arguments_call_result_5041 = invoke(stypy.reporting.localization.Localization(__file__, 181, 46), unfold_arguments_5037, *[arg_types_5038], **kwargs_5040)
                    
                    # Assigning a type to the variable 'unfolded_arg_tuples' (line 181)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 24), 'unfolded_arg_tuples', unfold_arguments_call_result_5041)

                    if more_types_in_union_5036:
                        # SSA join for if statement (line 178)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                # Getting the type of 'unfolded_arg_tuples' (line 184)
                unfolded_arg_tuples_5042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 34), 'unfolded_arg_tuples')
                # Assigning a type to the variable 'unfolded_arg_tuples_5042' (line 184)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'unfolded_arg_tuples_5042', unfolded_arg_tuples_5042)
                # Testing if the for loop is going to be iterated (line 184)
                # Testing the type of a for loop iterable (line 184)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 184, 20), unfolded_arg_tuples_5042)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 184, 20), unfolded_arg_tuples_5042):
                    # Getting the type of the for loop variable (line 184)
                    for_loop_var_5043 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 184, 20), unfolded_arg_tuples_5042)
                    # Assigning a type to the variable 'tuple_' (line 184)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'tuple_', for_loop_var_5043)
                    # SSA begins for a for statement (line 184)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to exist_a_type_error_within_parameters(...): (line 186)
                    # Processing the call arguments (line 186)
                    # Getting the type of 'localization' (line 186)
                    localization_5045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 64), 'localization', False)
                    
                    # Obtaining the type of the subscript
                    int_5046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 86), 'int')
                    # Getting the type of 'tuple_' (line 186)
                    tuple__5047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 79), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 186)
                    getitem___5048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 79), tuple__5047, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                    subscript_call_result_5049 = invoke(stypy.reporting.localization.Localization(__file__, 186, 79), getitem___5048, int_5046)
                    
                    # Processing the call keyword arguments (line 186)
                    
                    # Obtaining the type of the subscript
                    int_5050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 99), 'int')
                    # Getting the type of 'tuple_' (line 186)
                    tuple__5051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 92), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 186)
                    getitem___5052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 92), tuple__5051, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                    subscript_call_result_5053 = invoke(stypy.reporting.localization.Localization(__file__, 186, 92), getitem___5052, int_5050)
                    
                    kwargs_5054 = {'subscript_call_result_5053': subscript_call_result_5053}
                    # Getting the type of 'exist_a_type_error_within_parameters' (line 186)
                    exist_a_type_error_within_parameters_5044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 27), 'exist_a_type_error_within_parameters', False)
                    # Calling exist_a_type_error_within_parameters(args, kwargs) (line 186)
                    exist_a_type_error_within_parameters_call_result_5055 = invoke(stypy.reporting.localization.Localization(__file__, 186, 27), exist_a_type_error_within_parameters_5044, *[localization_5045, subscript_call_result_5049], **kwargs_5054)
                    
                    # Testing if the type of an if condition is none (line 186)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 186, 24), exist_a_type_error_within_parameters_call_result_5055):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 186)
                        if_condition_5056 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 24), exist_a_type_error_within_parameters_call_result_5055)
                        # Assigning a type to the variable 'if_condition_5056' (line 186)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'if_condition_5056', if_condition_5056)
                        # SSA begins for if statement (line 186)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Name (line 187):
                        
                        # Assigning a Name to a Name (line 187):
                        # Getting the type of 'True' (line 187)
                        True_5057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 48), 'True')
                        # Assigning a type to the variable 'found_type_errors' (line 187)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 28), 'found_type_errors', True_5057)
                        # SSA join for if statement (line 186)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    
                    # Assigning a Call to a Name (line 191):
                    
                    # Assigning a Call to a Name (line 191):
                    
                    # Call to call_handler(...): (line 191)
                    # Processing the call arguments (line 191)
                    # Getting the type of 'proxy_obj' (line 191)
                    proxy_obj_5059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 43), 'proxy_obj', False)
                    # Getting the type of 'localization' (line 191)
                    localization_5060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 54), 'localization', False)
                    # Getting the type of 'callable_' (line 191)
                    callable__5061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 68), 'callable_', False)
                    
                    # Obtaining the type of the subscript
                    int_5062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 87), 'int')
                    # Getting the type of 'tuple_' (line 191)
                    tuple__5063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 80), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 191)
                    getitem___5064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 80), tuple__5063, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
                    subscript_call_result_5065 = invoke(stypy.reporting.localization.Localization(__file__, 191, 80), getitem___5064, int_5062)
                    
                    # Processing the call keyword arguments (line 191)
                    
                    # Obtaining the type of the subscript
                    int_5066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 100), 'int')
                    # Getting the type of 'tuple_' (line 191)
                    tuple__5067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 93), 'tuple_', False)
                    # Obtaining the member '__getitem__' of a type (line 191)
                    getitem___5068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 93), tuple__5067, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 191)
                    subscript_call_result_5069 = invoke(stypy.reporting.localization.Localization(__file__, 191, 93), getitem___5068, int_5066)
                    
                    kwargs_5070 = {'subscript_call_result_5069': subscript_call_result_5069}
                    # Getting the type of 'call_handler' (line 191)
                    call_handler_5058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'call_handler', False)
                    # Calling call_handler(args, kwargs) (line 191)
                    call_handler_call_result_5071 = invoke(stypy.reporting.localization.Localization(__file__, 191, 30), call_handler_5058, *[proxy_obj_5059, localization_5060, callable__5061, subscript_call_result_5065], **kwargs_5070)
                    
                    # Assigning a type to the variable 'ret' (line 191)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 24), 'ret', call_handler_call_result_5071)
                    
                    # Type idiom detected: calculating its left and rigth part (line 192)
                    # Getting the type of 'TypeError' (line 192)
                    TypeError_5072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 47), 'TypeError')
                    # Getting the type of 'ret' (line 192)
                    ret_5073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 42), 'ret')
                    
                    (may_be_5074, more_types_in_union_5075) = may_not_be_subtype(TypeError_5072, ret_5073)

                    if may_be_5074:

                        if more_types_in_union_5075:
                            # Runtime conditional SSA (line 192)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        # Assigning a type to the variable 'ret' (line 192)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'ret', remove_subtype_from_union(ret_5073, TypeError))
                        
                        # Assigning a Name to a Name (line 195):
                        
                        # Assigning a Name to a Name (line 195):
                        # Getting the type of 'True' (line 195)
                        True_5076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 47), 'True')
                        # Assigning a type to the variable 'found_valid_call' (line 195)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 28), 'found_valid_call', True_5076)
                        
                        # Getting the type of 'registered_type_modifiers' (line 199)
                        registered_type_modifiers_5077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 44), 'registered_type_modifiers')
                        # Assigning a type to the variable 'registered_type_modifiers_5077' (line 199)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'registered_type_modifiers_5077', registered_type_modifiers_5077)
                        # Testing if the for loop is going to be iterated (line 199)
                        # Testing the type of a for loop iterable (line 199)
                        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 199, 28), registered_type_modifiers_5077)

                        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 199, 28), registered_type_modifiers_5077):
                            # Getting the type of the for loop variable (line 199)
                            for_loop_var_5078 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 199, 28), registered_type_modifiers_5077)
                            # Assigning a type to the variable 'modifier' (line 199)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 28), 'modifier', for_loop_var_5078)
                            # SSA begins for a for statement (line 199)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                            
                            # Call to applies_to(...): (line 200)
                            # Processing the call arguments (line 200)
                            # Getting the type of 'proxy_obj' (line 200)
                            proxy_obj_5081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 55), 'proxy_obj', False)
                            # Getting the type of 'callable_' (line 200)
                            callable__5082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 66), 'callable_', False)
                            # Processing the call keyword arguments (line 200)
                            kwargs_5083 = {}
                            # Getting the type of 'modifier' (line 200)
                            modifier_5079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 35), 'modifier', False)
                            # Obtaining the member 'applies_to' of a type (line 200)
                            applies_to_5080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 35), modifier_5079, 'applies_to')
                            # Calling applies_to(args, kwargs) (line 200)
                            applies_to_call_result_5084 = invoke(stypy.reporting.localization.Localization(__file__, 200, 35), applies_to_5080, *[proxy_obj_5081, callable__5082], **kwargs_5083)
                            
                            # Testing if the type of an if condition is none (line 200)

                            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 200, 32), applies_to_call_result_5084):
                                pass
                            else:
                                
                                # Testing the type of an if condition (line 200)
                                if_condition_5085 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 32), applies_to_call_result_5084)
                                # Assigning a type to the variable 'if_condition_5085' (line 200)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 32), 'if_condition_5085', if_condition_5085)
                                # SSA begins for if statement (line 200)
                                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                
                                # Evaluating a boolean operation
                                
                                # Call to ismethod(...): (line 201)
                                # Processing the call arguments (line 201)
                                # Getting the type of 'callable_' (line 201)
                                callable__5088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 56), 'callable_', False)
                                # Processing the call keyword arguments (line 201)
                                kwargs_5089 = {}
                                # Getting the type of 'inspect' (line 201)
                                inspect_5086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 39), 'inspect', False)
                                # Obtaining the member 'ismethod' of a type (line 201)
                                ismethod_5087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 39), inspect_5086, 'ismethod')
                                # Calling ismethod(args, kwargs) (line 201)
                                ismethod_call_result_5090 = invoke(stypy.reporting.localization.Localization(__file__, 201, 39), ismethod_5087, *[callable__5088], **kwargs_5089)
                                
                                
                                # Call to ismethoddescriptor(...): (line 201)
                                # Processing the call arguments (line 201)
                                # Getting the type of 'callable_' (line 201)
                                callable__5093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 97), 'callable_', False)
                                # Processing the call keyword arguments (line 201)
                                kwargs_5094 = {}
                                # Getting the type of 'inspect' (line 201)
                                inspect_5091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 70), 'inspect', False)
                                # Obtaining the member 'ismethoddescriptor' of a type (line 201)
                                ismethoddescriptor_5092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 70), inspect_5091, 'ismethoddescriptor')
                                # Calling ismethoddescriptor(args, kwargs) (line 201)
                                ismethoddescriptor_call_result_5095 = invoke(stypy.reporting.localization.Localization(__file__, 201, 70), ismethoddescriptor_5092, *[callable__5093], **kwargs_5094)
                                
                                # Applying the binary operator 'or' (line 201)
                                result_or_keyword_5096 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 39), 'or', ismethod_call_result_5090, ismethoddescriptor_call_result_5095)
                                
                                # Testing if the type of an if condition is none (line 201)

                                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 201, 36), result_or_keyword_5096):
                                    pass
                                else:
                                    
                                    # Testing the type of an if condition (line 201)
                                    if_condition_5097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 36), result_or_keyword_5096)
                                    # Assigning a type to the variable 'if_condition_5097' (line 201)
                                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 36), 'if_condition_5097', if_condition_5097)
                                    # SSA begins for if statement (line 201)
                                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                    
                                    
                                    # Call to is_type_instance(...): (line 203)
                                    # Processing the call keyword arguments (line 203)
                                    kwargs_5101 = {}
                                    # Getting the type of 'proxy_obj' (line 203)
                                    proxy_obj_5098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 47), 'proxy_obj', False)
                                    # Obtaining the member 'parent_proxy' of a type (line 203)
                                    parent_proxy_5099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 47), proxy_obj_5098, 'parent_proxy')
                                    # Obtaining the member 'is_type_instance' of a type (line 203)
                                    is_type_instance_5100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 47), parent_proxy_5099, 'is_type_instance')
                                    # Calling is_type_instance(args, kwargs) (line 203)
                                    is_type_instance_call_result_5102 = invoke(stypy.reporting.localization.Localization(__file__, 203, 47), is_type_instance_5100, *[], **kwargs_5101)
                                    
                                    # Applying the 'not' unary operator (line 203)
                                    result_not__5103 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 43), 'not', is_type_instance_call_result_5102)
                                    
                                    # Testing if the type of an if condition is none (line 203)

                                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 203, 40), result_not__5103):
                                        pass
                                    else:
                                        
                                        # Testing the type of an if condition (line 203)
                                        if_condition_5104 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 40), result_not__5103)
                                        # Assigning a type to the variable 'if_condition_5104' (line 203)
                                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 40), 'if_condition_5104', if_condition_5104)
                                        # SSA begins for if statement (line 203)
                                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                                        
                                        # Assigning a Call to a Name (line 205):
                                        
                                        # Assigning a Call to a Name (line 205):
                                        
                                        # Call to modifier(...): (line 205)
                                        # Processing the call arguments (line 205)
                                        
                                        # Obtaining the type of the subscript
                                        int_5106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 69), 'int')
                                        
                                        # Obtaining the type of the subscript
                                        int_5107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 66), 'int')
                                        # Getting the type of 'tuple_' (line 205)
                                        tuple__5108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 59), 'tuple_', False)
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 59), tuple__5108, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5110 = invoke(stypy.reporting.localization.Localization(__file__, 205, 59), getitem___5109, int_5107)
                                        
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 59), subscript_call_result_5110, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5112 = invoke(stypy.reporting.localization.Localization(__file__, 205, 59), getitem___5111, int_5106)
                                        
                                        # Getting the type of 'localization' (line 205)
                                        localization_5113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 73), 'localization', False)
                                        # Getting the type of 'callable_' (line 205)
                                        callable__5114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 87), 'callable_', False)
                                        
                                        # Obtaining the type of the subscript
                                        int_5115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 109), 'int')
                                        slice_5116 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 205, 99), int_5115, None, None)
                                        
                                        # Obtaining the type of the subscript
                                        int_5117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 106), 'int')
                                        # Getting the type of 'tuple_' (line 205)
                                        tuple__5118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 99), 'tuple_', False)
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 99), tuple__5118, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5120 = invoke(stypy.reporting.localization.Localization(__file__, 205, 99), getitem___5119, int_5117)
                                        
                                        # Obtaining the member '__getitem__' of a type (line 205)
                                        getitem___5121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 99), subscript_call_result_5120, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
                                        subscript_call_result_5122 = invoke(stypy.reporting.localization.Localization(__file__, 205, 99), getitem___5121, slice_5116)
                                        
                                        # Processing the call keyword arguments (line 205)
                                        
                                        # Obtaining the type of the subscript
                                        int_5123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 68), 'int')
                                        # Getting the type of 'tuple_' (line 206)
                                        tuple__5124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 61), 'tuple_', False)
                                        # Obtaining the member '__getitem__' of a type (line 206)
                                        getitem___5125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 61), tuple__5124, '__getitem__')
                                        # Calling the subscript (__getitem__) to obtain the elements type (line 206)
                                        subscript_call_result_5126 = invoke(stypy.reporting.localization.Localization(__file__, 206, 61), getitem___5125, int_5123)
                                        
                                        kwargs_5127 = {'subscript_call_result_5126': subscript_call_result_5126}
                                        # Getting the type of 'modifier' (line 205)
                                        modifier_5105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 50), 'modifier', False)
                                        # Calling modifier(args, kwargs) (line 205)
                                        modifier_call_result_5128 = invoke(stypy.reporting.localization.Localization(__file__, 205, 50), modifier_5105, *[subscript_call_result_5112, localization_5113, callable__5114, subscript_call_result_5122], **kwargs_5127)
                                        
                                        # Assigning a type to the variable 'ret' (line 205)
                                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 44), 'ret', modifier_call_result_5128)
                                        # SSA join for if statement (line 203)
                                        module_type_store = module_type_store.join_ssa_context()
                                        

                                    # SSA join for if statement (line 201)
                                    module_type_store = module_type_store.join_ssa_context()
                                    

                                
                                # Assigning a Call to a Name (line 209):
                                
                                # Assigning a Call to a Name (line 209):
                                
                                # Call to modifier(...): (line 209)
                                # Processing the call arguments (line 209)
                                # Getting the type of 'proxy_obj' (line 209)
                                proxy_obj_5130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 51), 'proxy_obj', False)
                                # Getting the type of 'localization' (line 209)
                                localization_5131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 62), 'localization', False)
                                # Getting the type of 'callable_' (line 209)
                                callable__5132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 76), 'callable_', False)
                                
                                # Obtaining the type of the subscript
                                int_5133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 95), 'int')
                                # Getting the type of 'tuple_' (line 209)
                                tuple__5134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 88), 'tuple_', False)
                                # Obtaining the member '__getitem__' of a type (line 209)
                                getitem___5135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 88), tuple__5134, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 209)
                                subscript_call_result_5136 = invoke(stypy.reporting.localization.Localization(__file__, 209, 88), getitem___5135, int_5133)
                                
                                # Processing the call keyword arguments (line 209)
                                
                                # Obtaining the type of the subscript
                                int_5137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 108), 'int')
                                # Getting the type of 'tuple_' (line 209)
                                tuple__5138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 101), 'tuple_', False)
                                # Obtaining the member '__getitem__' of a type (line 209)
                                getitem___5139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 101), tuple__5138, '__getitem__')
                                # Calling the subscript (__getitem__) to obtain the elements type (line 209)
                                subscript_call_result_5140 = invoke(stypy.reporting.localization.Localization(__file__, 209, 101), getitem___5139, int_5137)
                                
                                kwargs_5141 = {'subscript_call_result_5140': subscript_call_result_5140}
                                # Getting the type of 'modifier' (line 209)
                                modifier_5129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 42), 'modifier', False)
                                # Calling modifier(args, kwargs) (line 209)
                                modifier_call_result_5142 = invoke(stypy.reporting.localization.Localization(__file__, 209, 42), modifier_5129, *[proxy_obj_5130, localization_5131, callable__5132, subscript_call_result_5136], **kwargs_5141)
                                
                                # Assigning a type to the variable 'ret' (line 209)
                                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 36), 'ret', modifier_call_result_5142)
                                # SSA join for if statement (line 200)
                                module_type_store = module_type_store.join_ssa_context()
                                

                            # SSA join for a for statement
                            module_type_store = module_type_store.join_ssa_context()

                        
                        
                        # Assigning a Call to a Name (line 214):
                        
                        # Assigning a Call to a Name (line 214):
                        
                        # Call to add(...): (line 214)
                        # Processing the call arguments (line 214)
                        # Getting the type of 'return_type' (line 214)
                        return_type_5146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 72), 'return_type', False)
                        # Getting the type of 'ret' (line 214)
                        ret_5147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 85), 'ret', False)
                        # Processing the call keyword arguments (line 214)
                        kwargs_5148 = {}
                        # Getting the type of 'union_type_copy' (line 214)
                        union_type_copy_5143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 42), 'union_type_copy', False)
                        # Obtaining the member 'UnionType' of a type (line 214)
                        UnionType_5144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 42), union_type_copy_5143, 'UnionType')
                        # Obtaining the member 'add' of a type (line 214)
                        add_5145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 42), UnionType_5144, 'add')
                        # Calling add(args, kwargs) (line 214)
                        add_call_result_5149 = invoke(stypy.reporting.localization.Localization(__file__, 214, 42), add_5145, *[return_type_5146, ret_5147], **kwargs_5148)
                        
                        # Assigning a type to the variable 'return_type' (line 214)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 28), 'return_type', add_call_result_5149)

                        if more_types_in_union_5075:
                            # Runtime conditional SSA for else branch (line 192)
                            module_type_store.open_ssa_branch('idiom else')



                    if ((not may_be_5074) or more_types_in_union_5075):
                        # Assigning a type to the variable 'ret' (line 192)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 24), 'ret', remove_not_subtype_from_union(ret_5073, TypeError))
                        
                        # Call to append(...): (line 217)
                        # Processing the call arguments (line 217)
                        # Getting the type of 'ret' (line 217)
                        ret_5152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 48), 'ret', False)
                        # Processing the call keyword arguments (line 217)
                        kwargs_5153 = {}
                        # Getting the type of 'found_errors' (line 217)
                        found_errors_5150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 28), 'found_errors', False)
                        # Obtaining the member 'append' of a type (line 217)
                        append_5151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 28), found_errors_5150, 'append')
                        # Calling append(args, kwargs) (line 217)
                        append_call_result_5154 = invoke(stypy.reporting.localization.Localization(__file__, 217, 28), append_5151, *[ret_5152], **kwargs_5153)
                        

                        if (may_be_5074 and more_types_in_union_5075):
                            # SSA join for if statement (line 192)
                            module_type_store = module_type_store.join_ssa_context()


                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for if statement (line 152)
                module_type_store = module_type_store.join_ssa_context()
                

            # Getting the type of 'found_valid_call' (line 221)
            found_valid_call_5155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), 'found_valid_call')
            # Testing if the type of an if condition is none (line 221)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 221, 16), found_valid_call_5155):
                
                
                # Call to len(...): (line 230)
                # Processing the call arguments (line 230)
                # Getting the type of 'found_errors' (line 230)
                found_errors_5165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 27), 'found_errors', False)
                # Processing the call keyword arguments (line 230)
                kwargs_5166 = {}
                # Getting the type of 'len' (line 230)
                len_5164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'len', False)
                # Calling len(args, kwargs) (line 230)
                len_call_result_5167 = invoke(stypy.reporting.localization.Localization(__file__, 230, 23), len_5164, *[found_errors_5165], **kwargs_5166)
                
                int_5168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 44), 'int')
                # Applying the binary operator '==' (line 230)
                result_eq_5169 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 23), '==', len_call_result_5167, int_5168)
                
                # Testing if the type of an if condition is none (line 230)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 230, 20), result_eq_5169):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 230)
                    if_condition_5170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 20), result_eq_5169)
                    # Assigning a type to the variable 'if_condition_5170' (line 230)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'if_condition_5170', if_condition_5170)
                    # SSA begins for if statement (line 230)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Obtaining the type of the subscript
                    int_5171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 44), 'int')
                    # Getting the type of 'found_errors' (line 231)
                    found_errors_5172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 31), 'found_errors')
                    # Obtaining the member '__getitem__' of a type (line 231)
                    getitem___5173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 31), found_errors_5172, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 231)
                    subscript_call_result_5174 = invoke(stypy.reporting.localization.Localization(__file__, 231, 31), getitem___5173, int_5171)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 231)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'stypy_return_type', subscript_call_result_5174)
                    # SSA join for if statement (line 230)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'found_errors' (line 237)
                found_errors_5175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 'found_errors')
                # Assigning a type to the variable 'found_errors_5175' (line 237)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'found_errors_5175', found_errors_5175)
                # Testing if the for loop is going to be iterated (line 237)
                # Testing the type of a for loop iterable (line 237)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 237, 20), found_errors_5175)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 237, 20), found_errors_5175):
                    # Getting the type of the for loop variable (line 237)
                    for_loop_var_5176 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 237, 20), found_errors_5175)
                    # Assigning a type to the variable 'error' (line 237)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'error', for_loop_var_5176)
                    # SSA begins for a for statement (line 237)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to remove_error_msg(...): (line 238)
                    # Processing the call arguments (line 238)
                    # Getting the type of 'error' (line 238)
                    error_5179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 51), 'error', False)
                    # Processing the call keyword arguments (line 238)
                    kwargs_5180 = {}
                    # Getting the type of 'TypeError' (line 238)
                    TypeError_5177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'TypeError', False)
                    # Obtaining the member 'remove_error_msg' of a type (line 238)
                    remove_error_msg_5178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 24), TypeError_5177, 'remove_error_msg')
                    # Calling remove_error_msg(args, kwargs) (line 238)
                    remove_error_msg_call_result_5181 = invoke(stypy.reporting.localization.Localization(__file__, 238, 24), remove_error_msg_5178, *[error_5179], **kwargs_5180)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a Call to a Name (line 240):
                
                # Assigning a Call to a Name (line 240):
                
                # Call to format_call(...): (line 240)
                # Processing the call arguments (line 240)
                # Getting the type of 'callable_' (line 240)
                callable__5183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 43), 'callable_', False)
                # Getting the type of 'arg_types' (line 240)
                arg_types_5184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 54), 'arg_types', False)
                # Getting the type of 'kwarg_types' (line 240)
                kwarg_types_5185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 65), 'kwarg_types', False)
                # Processing the call keyword arguments (line 240)
                kwargs_5186 = {}
                # Getting the type of 'format_call' (line 240)
                format_call_5182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'format_call', False)
                # Calling format_call(args, kwargs) (line 240)
                format_call_call_result_5187 = invoke(stypy.reporting.localization.Localization(__file__, 240, 31), format_call_5182, *[callable__5183, arg_types_5184, kwarg_types_5185], **kwargs_5186)
                
                # Assigning a type to the variable 'call_str' (line 240)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'call_str', format_call_call_result_5187)
                # Getting the type of 'found_type_errors' (line 241)
                found_type_errors_5188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 'found_type_errors')
                # Testing if the type of an if condition is none (line 241)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 241, 20), found_type_errors_5188):
                    
                    # Assigning a Str to a Name (line 244):
                    
                    # Assigning a Str to a Name (line 244):
                    str_5191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'str', 'The called entity do not accept any of these parameters')
                    # Assigning a type to the variable 'msg' (line 244)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'msg', str_5191)
                else:
                    
                    # Testing the type of an if condition (line 241)
                    if_condition_5189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 20), found_type_errors_5188)
                    # Assigning a type to the variable 'if_condition_5189' (line 241)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'if_condition_5189', if_condition_5189)
                    # SSA begins for if statement (line 241)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Name (line 242):
                    
                    # Assigning a Str to a Name (line 242):
                    str_5190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 30), 'str', 'Type errors found among the types of the call parameters')
                    # Assigning a type to the variable 'msg' (line 242)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'msg', str_5190)
                    # SSA branch for the else part of an if statement (line 241)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Str to a Name (line 244):
                    
                    # Assigning a Str to a Name (line 244):
                    str_5191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'str', 'The called entity do not accept any of these parameters')
                    # Assigning a type to the variable 'msg' (line 244)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'msg', str_5191)
                    # SSA join for if statement (line 241)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to TypeError(...): (line 246)
                # Processing the call arguments (line 246)
                # Getting the type of 'localization' (line 246)
                localization_5193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 37), 'localization', False)
                
                # Call to format(...): (line 246)
                # Processing the call arguments (line 246)
                # Getting the type of 'call_str' (line 246)
                call_str_5196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 69), 'call_str', False)
                # Getting the type of 'msg' (line 246)
                msg_5197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 79), 'msg', False)
                # Processing the call keyword arguments (line 246)
                kwargs_5198 = {}
                str_5194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 51), 'str', '{0}: {1}')
                # Obtaining the member 'format' of a type (line 246)
                format_5195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 51), str_5194, 'format')
                # Calling format(args, kwargs) (line 246)
                format_call_result_5199 = invoke(stypy.reporting.localization.Localization(__file__, 246, 51), format_5195, *[call_str_5196, msg_5197], **kwargs_5198)
                
                # Processing the call keyword arguments (line 246)
                kwargs_5200 = {}
                # Getting the type of 'TypeError' (line 246)
                TypeError_5192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 27), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 246)
                TypeError_call_result_5201 = invoke(stypy.reporting.localization.Localization(__file__, 246, 27), TypeError_5192, *[localization_5193, format_call_result_5199], **kwargs_5200)
                
                # Assigning a type to the variable 'stypy_return_type' (line 246)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'stypy_return_type', TypeError_call_result_5201)
            else:
                
                # Testing the type of an if condition (line 221)
                if_condition_5156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 16), found_valid_call_5155)
                # Assigning a type to the variable 'if_condition_5156' (line 221)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'if_condition_5156', if_condition_5156)
                # SSA begins for if statement (line 221)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'found_errors' (line 222)
                found_errors_5157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 33), 'found_errors')
                # Assigning a type to the variable 'found_errors_5157' (line 222)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'found_errors_5157', found_errors_5157)
                # Testing if the for loop is going to be iterated (line 222)
                # Testing the type of a for loop iterable (line 222)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 222, 20), found_errors_5157)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 222, 20), found_errors_5157):
                    # Getting the type of the for loop variable (line 222)
                    for_loop_var_5158 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 222, 20), found_errors_5157)
                    # Assigning a type to the variable 'error' (line 222)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'error', for_loop_var_5158)
                    # SSA begins for a for statement (line 222)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to turn_to_warning(...): (line 223)
                    # Processing the call keyword arguments (line 223)
                    kwargs_5161 = {}
                    # Getting the type of 'error' (line 223)
                    error_5159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 24), 'error', False)
                    # Obtaining the member 'turn_to_warning' of a type (line 223)
                    turn_to_warning_5160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 24), error_5159, 'turn_to_warning')
                    # Calling turn_to_warning(args, kwargs) (line 223)
                    turn_to_warning_call_result_5162 = invoke(stypy.reporting.localization.Localization(__file__, 223, 24), turn_to_warning_5160, *[], **kwargs_5161)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # Getting the type of 'return_type' (line 224)
                return_type_5163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'return_type')
                # Assigning a type to the variable 'stypy_return_type' (line 224)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 20), 'stypy_return_type', return_type_5163)
                # SSA branch for the else part of an if statement (line 221)
                module_type_store.open_ssa_branch('else')
                
                
                # Call to len(...): (line 230)
                # Processing the call arguments (line 230)
                # Getting the type of 'found_errors' (line 230)
                found_errors_5165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 27), 'found_errors', False)
                # Processing the call keyword arguments (line 230)
                kwargs_5166 = {}
                # Getting the type of 'len' (line 230)
                len_5164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 'len', False)
                # Calling len(args, kwargs) (line 230)
                len_call_result_5167 = invoke(stypy.reporting.localization.Localization(__file__, 230, 23), len_5164, *[found_errors_5165], **kwargs_5166)
                
                int_5168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 230, 44), 'int')
                # Applying the binary operator '==' (line 230)
                result_eq_5169 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 23), '==', len_call_result_5167, int_5168)
                
                # Testing if the type of an if condition is none (line 230)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 230, 20), result_eq_5169):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 230)
                    if_condition_5170 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 230, 20), result_eq_5169)
                    # Assigning a type to the variable 'if_condition_5170' (line 230)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 20), 'if_condition_5170', if_condition_5170)
                    # SSA begins for if statement (line 230)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Obtaining the type of the subscript
                    int_5171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 44), 'int')
                    # Getting the type of 'found_errors' (line 231)
                    found_errors_5172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 31), 'found_errors')
                    # Obtaining the member '__getitem__' of a type (line 231)
                    getitem___5173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 31), found_errors_5172, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 231)
                    subscript_call_result_5174 = invoke(stypy.reporting.localization.Localization(__file__, 231, 31), getitem___5173, int_5171)
                    
                    # Assigning a type to the variable 'stypy_return_type' (line 231)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 24), 'stypy_return_type', subscript_call_result_5174)
                    # SSA join for if statement (line 230)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'found_errors' (line 237)
                found_errors_5175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 33), 'found_errors')
                # Assigning a type to the variable 'found_errors_5175' (line 237)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'found_errors_5175', found_errors_5175)
                # Testing if the for loop is going to be iterated (line 237)
                # Testing the type of a for loop iterable (line 237)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 237, 20), found_errors_5175)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 237, 20), found_errors_5175):
                    # Getting the type of the for loop variable (line 237)
                    for_loop_var_5176 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 237, 20), found_errors_5175)
                    # Assigning a type to the variable 'error' (line 237)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'error', for_loop_var_5176)
                    # SSA begins for a for statement (line 237)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to remove_error_msg(...): (line 238)
                    # Processing the call arguments (line 238)
                    # Getting the type of 'error' (line 238)
                    error_5179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 51), 'error', False)
                    # Processing the call keyword arguments (line 238)
                    kwargs_5180 = {}
                    # Getting the type of 'TypeError' (line 238)
                    TypeError_5177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 24), 'TypeError', False)
                    # Obtaining the member 'remove_error_msg' of a type (line 238)
                    remove_error_msg_5178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 24), TypeError_5177, 'remove_error_msg')
                    # Calling remove_error_msg(args, kwargs) (line 238)
                    remove_error_msg_call_result_5181 = invoke(stypy.reporting.localization.Localization(__file__, 238, 24), remove_error_msg_5178, *[error_5179], **kwargs_5180)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a Call to a Name (line 240):
                
                # Assigning a Call to a Name (line 240):
                
                # Call to format_call(...): (line 240)
                # Processing the call arguments (line 240)
                # Getting the type of 'callable_' (line 240)
                callable__5183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 43), 'callable_', False)
                # Getting the type of 'arg_types' (line 240)
                arg_types_5184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 54), 'arg_types', False)
                # Getting the type of 'kwarg_types' (line 240)
                kwarg_types_5185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 65), 'kwarg_types', False)
                # Processing the call keyword arguments (line 240)
                kwargs_5186 = {}
                # Getting the type of 'format_call' (line 240)
                format_call_5182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 31), 'format_call', False)
                # Calling format_call(args, kwargs) (line 240)
                format_call_call_result_5187 = invoke(stypy.reporting.localization.Localization(__file__, 240, 31), format_call_5182, *[callable__5183, arg_types_5184, kwarg_types_5185], **kwargs_5186)
                
                # Assigning a type to the variable 'call_str' (line 240)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'call_str', format_call_call_result_5187)
                # Getting the type of 'found_type_errors' (line 241)
                found_type_errors_5188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 23), 'found_type_errors')
                # Testing if the type of an if condition is none (line 241)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 241, 20), found_type_errors_5188):
                    
                    # Assigning a Str to a Name (line 244):
                    
                    # Assigning a Str to a Name (line 244):
                    str_5191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'str', 'The called entity do not accept any of these parameters')
                    # Assigning a type to the variable 'msg' (line 244)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'msg', str_5191)
                else:
                    
                    # Testing the type of an if condition (line 241)
                    if_condition_5189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 20), found_type_errors_5188)
                    # Assigning a type to the variable 'if_condition_5189' (line 241)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'if_condition_5189', if_condition_5189)
                    # SSA begins for if statement (line 241)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Str to a Name (line 242):
                    
                    # Assigning a Str to a Name (line 242):
                    str_5190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 30), 'str', 'Type errors found among the types of the call parameters')
                    # Assigning a type to the variable 'msg' (line 242)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 24), 'msg', str_5190)
                    # SSA branch for the else part of an if statement (line 241)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Str to a Name (line 244):
                    
                    # Assigning a Str to a Name (line 244):
                    str_5191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'str', 'The called entity do not accept any of these parameters')
                    # Assigning a type to the variable 'msg' (line 244)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'msg', str_5191)
                    # SSA join for if statement (line 241)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Call to TypeError(...): (line 246)
                # Processing the call arguments (line 246)
                # Getting the type of 'localization' (line 246)
                localization_5193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 37), 'localization', False)
                
                # Call to format(...): (line 246)
                # Processing the call arguments (line 246)
                # Getting the type of 'call_str' (line 246)
                call_str_5196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 69), 'call_str', False)
                # Getting the type of 'msg' (line 246)
                msg_5197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 79), 'msg', False)
                # Processing the call keyword arguments (line 246)
                kwargs_5198 = {}
                str_5194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 51), 'str', '{0}: {1}')
                # Obtaining the member 'format' of a type (line 246)
                format_5195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 51), str_5194, 'format')
                # Calling format(args, kwargs) (line 246)
                format_call_result_5199 = invoke(stypy.reporting.localization.Localization(__file__, 246, 51), format_5195, *[call_str_5196, msg_5197], **kwargs_5198)
                
                # Processing the call keyword arguments (line 246)
                kwargs_5200 = {}
                # Getting the type of 'TypeError' (line 246)
                TypeError_5192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 27), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 246)
                TypeError_call_result_5201 = invoke(stypy.reporting.localization.Localization(__file__, 246, 27), TypeError_5192, *[localization_5193, format_call_result_5199], **kwargs_5200)
                
                # Assigning a type to the variable 'stypy_return_type' (line 246)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 20), 'stypy_return_type', TypeError_call_result_5201)
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
    Exception_5202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 11), 'Exception')
    # Assigning a type to the variable 'e' (line 248)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 4), 'e', Exception_5202)
    
    # Call to TypeError(...): (line 250)
    # Processing the call arguments (line 250)
    # Getting the type of 'localization' (line 250)
    localization_5204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 25), 'localization', False)
    
    # Call to format(...): (line 250)
    # Processing the call arguments (line 250)
    # Getting the type of 'callable_' (line 251)
    callable__5207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'callable_', False)
    
    # Call to list(...): (line 251)
    # Processing the call arguments (line 251)
    # Getting the type of 'arg_types' (line 251)
    arg_types_5209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 28), 'arg_types', False)
    # Processing the call keyword arguments (line 251)
    kwargs_5210 = {}
    # Getting the type of 'list' (line 251)
    list_5208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 23), 'list', False)
    # Calling list(args, kwargs) (line 251)
    list_call_result_5211 = invoke(stypy.reporting.localization.Localization(__file__, 251, 23), list_5208, *[arg_types_5209], **kwargs_5210)
    
    
    # Call to list(...): (line 251)
    # Processing the call arguments (line 251)
    
    # Call to values(...): (line 251)
    # Processing the call keyword arguments (line 251)
    kwargs_5215 = {}
    # Getting the type of 'kwarg_types' (line 251)
    kwarg_types_5213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 46), 'kwarg_types', False)
    # Obtaining the member 'values' of a type (line 251)
    values_5214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 46), kwarg_types_5213, 'values')
    # Calling values(args, kwargs) (line 251)
    values_call_result_5216 = invoke(stypy.reporting.localization.Localization(__file__, 251, 46), values_5214, *[], **kwargs_5215)
    
    # Processing the call keyword arguments (line 251)
    kwargs_5217 = {}
    # Getting the type of 'list' (line 251)
    list_5212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 41), 'list', False)
    # Calling list(args, kwargs) (line 251)
    list_call_result_5218 = invoke(stypy.reporting.localization.Localization(__file__, 251, 41), list_5212, *[values_call_result_5216], **kwargs_5217)
    
    # Applying the binary operator '+' (line 251)
    result_add_5219 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 23), '+', list_call_result_5211, list_call_result_5218)
    
    # Getting the type of 'e' (line 251)
    e_5220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 69), 'e', False)
    # Processing the call keyword arguments (line 250)
    kwargs_5221 = {}
    str_5205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 39), 'str', "An error was produced when invoking '{0}' with arguments [{1}]: {2}")
    # Obtaining the member 'format' of a type (line 250)
    format_5206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 39), str_5205, 'format')
    # Calling format(args, kwargs) (line 250)
    format_call_result_5222 = invoke(stypy.reporting.localization.Localization(__file__, 250, 39), format_5206, *[callable__5207, result_add_5219, e_5220], **kwargs_5221)
    
    # Processing the call keyword arguments (line 250)
    kwargs_5223 = {}
    # Getting the type of 'TypeError' (line 250)
    TypeError_5203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 250)
    TypeError_call_result_5224 = invoke(stypy.reporting.localization.Localization(__file__, 250, 15), TypeError_5203, *[localization_5204, format_call_result_5222], **kwargs_5223)
    
    # Assigning a type to the variable 'stypy_return_type' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'stypy_return_type', TypeError_call_result_5224)
    # SSA join for try-except statement (line 137)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'perform_call(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'perform_call' in the type store
    # Getting the type of 'stypy_return_type' (line 108)
    stypy_return_type_5225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5225)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'perform_call'
    return stypy_return_type_5225

# Assigning a type to the variable 'perform_call' (line 108)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 0), 'perform_call', perform_call)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
