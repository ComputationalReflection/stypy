
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import sys
2: 
3: from errors_copy.type_error_copy import TypeError
4: from errors_copy.type_warning_copy import TypeWarning
5: from errors_copy.unsupported_features_copy import create_unsupported_python_feature_message
6: from code_generation_copy.type_inference_programs_copy.python_operators_copy import *
7: from python_lib_copy.module_imports_copy import python_imports_copy
8: from python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
9: from python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions
10: from python_lib_copy.type_rules_copy.type_groups_copy import type_group_generator_copy, type_groups_copy
11: from python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
12: 
13: '''
14: This file contains the stypy API that can be called inside the type inference generated programs source code.
15: These functions will be used to interact with stypy, extract type information and other necessary operations when
16: generating type inference code.
17: '''
18: 
19: '''
20: An object containing the Python __builtins__ module, containing the type inference functions for each Python builtin
21: '''
22: builtin_module = python_imports_copy.get_module_from_sys_cache('__builtin__')
23: 
24: 
25: def get_builtin_type(localization, type_name, value=UndefinedType):
26:     '''
27:     Obtains a Python builtin type instance to represent the type of an object in Python. Optionally, a value for
28:     this object can be specified. Values for objects are not much used within the current version of stypy, but
29:     they are stored for future enhancements. Currently, values, if present, are taken into account for the hasattr,
30:     setattr and getattr builtin functions.
31: 
32:     :param localization: Caller information
33:     :param type_name: Name of the Python type to be created ("int", "float"...)
34:     :param value: Optional value for this type. Value must be of the speficied type. The function does not check this.
35:     :return: A TypeInferenceProxy representing the specified type or a TypeError if the specified type do not exist
36:     '''
37: 
38:     # Type "NoneType" has an special treatment
39:     if "NoneType" in type_name:
40:         return python_imports_copy.import_from(localization, "None")
41: 
42:     # Python uses more builtin types than those defined in the types package. We created an special object to hold
43:     # an instance of each one of them. This ensures that this instance is returned.
44:     if hasattr(ExtraTypeDefinitions, type_name):
45:         ret_type = getattr(ExtraTypeDefinitions, type_name)
46:     else:
47:         # Type from the Python __builtins__ module
48:         ret_type = builtin_module.get_type_of_member(localization, type_name)
49: 
50:     # By default, types represent instances of these types (not type names)
51:     ret_type.set_type_instance(True)
52: 
53:     # Assign value if present
54:     if value is not UndefinedType:
55:         ret_type.set_value(value)
56: 
57:     return ret_type
58: 
59: 
60: def get_python_api_type(localization, module_name, type_name):
61:     '''
62:     This function can obtain any type name for any Python module that have it declared. This way we can access
63:     non-builtin types such as those declared on the time module and so on, provided they exist within the specified
64:     module
65:     :param localization: Caller information
66:     :param module_name: Module name
67:     :param type_name: Type name within the module
68:     :return: A TypeInferenceProxy for the specified type or a TypeError if the requested type do not exist
69:     '''
70:     # Import the module
71:     module = python_imports_copy.import_python_module(localization, module_name)
72:     if isinstance(module, TypeError):
73:         return module
74:     # Return the type declared as a member of the module
75:     return module.get_type_of_member(localization, type_name)
76: 
77: 
78: def import_elements_from_external_module(localization, imported_module_name, dest_type_store,
79:                                          *elements):
80:     '''
81:     This function imports all the declared public members of a user-defined or Python library module into the specified
82:     type store
83:     It modules the from <module> import <element1>, <element2>, ... or * sentences and also the import <module> sentence
84:     :param localization: Caller information
85:     :param main_module_path: Path of the module to import, i. e. path of the .py file of the module
86:     :param imported_module_name: Name of the module
87:     :param dest_type_store: Type store to add the module elements
88:     :param elements: A variable list of arguments with the elements to import. The value '*' means all elements. No
89:     value models the "import <module>" sentence
90:     :return: None or a TypeError if the requested type do not exist
91:     '''
92:     return python_imports_copy.import_elements_from_external_module(localization, imported_module_name,
93:                                                                dest_type_store, sys.path,
94:                                                                *elements)
95: 
96: 
97: def import_from(localization, member_name, module_name="__builtin__"):
98:     '''
99:     Imports a single member from a module. If no module is specified, the builtin module is used instead. Models the
100:     "from <module> import <member>" sentence, being a sort version of the import_elements_from_external_module function
101:     but only for Python library modules
102:     :param localization: Caller information
103:     :param member_name: Member to import
104:     :param module_name: Python library module that contains the member or nothing to use the __builtins__ module
105:     :return: A TypeInferenceProxy for the specified member or a TypeError if the requested element do not exist
106:     '''
107:     return python_imports_copy.import_from(localization, member_name, module_name)
108: 
109: 
110: def import_module(localization, module_name="__builtin__"):
111:     '''
112:     Import a full Python library module (models the "import <module>" sentence for Python library modules
113:     :param localization: Caller information
114:     :param module_name: Module to import
115:     :return: A TypeInferenceProxy for the specified module or a TypeError if the requested module do not exist
116:     '''
117:     return python_imports_copy.import_python_module(localization, module_name)
118: 
119: 
120: # This is a clone of the "operator" module that is used when invoking builtin operators. This is used to separate
121: # the "operator" Python module from the Python language operators implementation, because although its implementation
122: # is initially the same, builtin operators are not modifiable (as opposed to the ones offered by the operator module).
123: # This variable follows a Singleton pattern.
124: builtin_operators_module = None
125: 
126: 
127: def load_builtin_operators_module():
128:     '''
129:     Loads the builtin Python operators logic that model the Python operator behavior, as a clone of the "operator"
130:     Python library module, that initially holds the same behavior for each operator. Once initially loaded, this logic
131:     cannot be altered (in Python we cannot overload the '+' operator behavior for builtin types, but we can modify the
132:     behavior of the equivalent operator.add function).
133:     :return: The behavior of the Python operators
134:     '''
135:     global builtin_operators_module
136: 
137:     # First time calling an operator? Load operator logic
138:     if builtin_operators_module is None:
139:         operator_module = python_imports_copy.import_python_module(None, 'operator')
140:         builtin_operators_module = operator_module.clone()
141:         builtin_operators_module.name = "builtin_operators"
142: 
143:     return builtin_operators_module
144: 
145: 
146: forced_operator_result_checks = [
147:     (['lt', 'gt', 'lte', 'gte', 'le', 'ge'], type_group_generator_copy.Integer),
148: ]
149: 
150: 
151: def operator(localization, operator_symbol, *arguments):
152:     '''
153:     Handles all the invokations to Python operators of the type inference program.
154:     :param localization: Caller information
155:     :param operator_symbol: Operator symbol ('+', '-',...). Symbols instead of operator names ('add', 'sub', ...)
156:     are used in the generated type inference program to improve readability
157:     :param arguments: Variable list of arguments of the operator
158:     :return: Return type of the operator call
159:     '''
160:     global builtin_operators_module
161: 
162:     load_builtin_operators_module()
163: 
164:     try:
165:         # Test that this is a valid operator
166:         operator_name = operator_symbol_to_name(operator_symbol)
167:     except:
168:         # If not a valid operator, return a type error
169:         return TypeError(localization, "Unrecognized operator: {0}".format(operator_symbol))
170: 
171:     # Obtain the operator call from the operator module
172:     operator_call = builtin_operators_module.get_type_of_member(localization, operator_name)
173: 
174:     # PATCH: This specific operator reverses the argument order
175:     if operator_name == 'contains':
176:         arguments = tuple(reversed(arguments))
177: 
178:     # Invoke the operator and return its result type
179:     result = operator_call.invoke(localization, *arguments)
180:     for check_tuple in forced_operator_result_checks:
181:         if operator_name in check_tuple[0]:
182:             if check_tuple[1] == result:
183:                 return result
184:             else:
185:                 return TypeError(localization,
186:                                  "Operator {0} did not return an {1}".format(operator_name, check_tuple[1]))
187:     return result
188: 
189: 
190: def unsupported_python_feature(localization, feature, description=""):
191:     '''
192:     This is called when the type inference program uses not yet supported by stypy Python feature
193:     :param localization: Caller information
194:     :param feature: Feature name
195:     :param description: Message to give to the user
196:     :return: A specific TypeError for this kind of problem
197:     '''
198:     create_unsupported_python_feature_message(localization, feature, description)
199: 
200: 
201: def ensure_var_of_types(localization, var, var_description, *type_names):
202:     '''
203:     This function is used to be sure that an specific var is of one of the specified types. This function is used
204:     by type inference programs when a variable must be of a collection of specific types for the program to be
205:     correct, which can happen in certain situations such as if conditions or loop tests.
206:     :param localization: Caller information
207:     :param var: Variable to test (TypeInferenceProxy)
208:     :param var_description: Description of the purpose of the tested variable, to show in a potential TypeError
209:     :param type_names: Accepted type names
210:     :return: None or a TypeError if the variable do not have a suitable type
211:     '''
212:     # TODO: What happens when a var has the DynamicType or UndefinedType type?
213:     python_type = var.get_python_type()
214:     for type_name in type_names:
215:         if type_name is str:
216:             type_obj = eval("types." + type_name)
217:         else:
218:             type_obj = type_name
219: 
220:         if python_type is type_obj:
221:             return  # Suitable type found, end.
222: 
223:     return TypeError(localization, var_description + " must be of one of the following types: " + str(type_names))
224: 
225: 
226: def ensure_var_has_members(localization, var, var_description, *member_names):
227:     '''
228:     This function is used to make sure that a certain variable has an specific set of members, which may be needed
229:     when generating some type inference code that needs an specific structure o a certain object
230:     :param localization: Caller information
231:     :param var: Variable to test (TypeInferenceProxy)
232:     :param var_description: Description of the purpose of the tested variable, to show in a potential TypeError
233:     :param member_names: List of members that the type of the variable must have to be valid.
234:     :return: None or a TypeError if the variable do not have all passed members
235:     '''
236:     python_type = var.get_python_entity()
237:     for type_name in member_names:
238:         if not hasattr(python_type, type_name):
239:             TypeError(localization, var_description + " must have all of these members: " + str(member_names))
240:             return False
241: 
242:     return True
243: 
244: 
245: def __slice_bounds_checking(bound):
246:     if bound is None:
247:         return [None], []
248: 
249:     if isinstance(bound, union_type_copy.UnionType):
250:         types_to_check = bound.types
251:     else:
252:         types_to_check = [bound]
253: 
254:     right_types = []
255:     wrong_types = []
256:     for type_ in types_to_check:
257:         if type_group_generator_copy.Integer == type_ or type_groups_copy.CastsToIndex == type_:
258:             right_types.append(type_)
259:         else:
260:             wrong_types.append(type_)
261: 
262:     return right_types, wrong_types
263: 
264: 
265: def ensure_slice_bounds(localization, lower, upper, step):
266:     '''
267:     Check the parameters of a created slice to make sure that the slice have correct bounds. If not, it returns a
268:     silent TypeError, as the specific problem (invalid lower, upper or step parameter is reported separately)
269:     :param localization: Caller information
270:     :param lower: Lower bound of the slice or None
271:     :param upper: Upper bound of the slice or None
272:     :param step: Step of the slice or None
273:     :return: A slice object or a TypeError is its parameters are invalid
274:     '''
275:     error = False
276:     r, w = __slice_bounds_checking(lower)
277: 
278:     if len(w) > 0 and len(r) > 0:
279:         TypeWarning(localization, "Some of the possible types of the lower bound of the slice ({0}) are invalid".
280:                     format(lower))
281:     if len(w) > 0 and len(r) == 0:
282:         TypeError(localization, "The type of the lower bound of the slice ({0}) is invalid".format(lower))
283:         error = True
284: 
285:     r, w = __slice_bounds_checking(upper)
286:     if len(w) > 0 and len(r) > 0:
287:         TypeWarning(localization, "Some of the possible types of the upper bound of the slice ({0}) are invalid".
288:                     format(upper))
289:     if len(w) > 0 and len(r) == 0:
290:         TypeError(localization, "The type of the upper bound of the slice ({0}) is invalid".format(upper))
291:         error = True
292: 
293:     r, w = __slice_bounds_checking(step)
294:     if len(w) > 0 and len(r) > 0:
295:         TypeWarning(localization, "Some of the possible types of the step of the slice ({0}) are invalid".
296:                     format(step))
297:     if len(w) > 0 and len(r) == 0:
298:         TypeError(localization, "The type of the step of the slice ({0}) is invalid".format(step))
299:         error = True
300: 
301:     if not error:
302:         return get_builtin_type(localization, 'slice')
303:     else:
304:         return TypeError(localization, "Type error when specifying slice bounds", prints_msg=False)
305: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import sys' statement (line 1)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from errors_copy.type_error_copy import TypeError' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_302 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'errors_copy.type_error_copy')

if (type(import_302) is not StypyTypeError):

    if (import_302 != 'pyd_module'):
        __import__(import_302)
        sys_modules_303 = sys.modules[import_302]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'errors_copy.type_error_copy', sys_modules_303.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_303, sys_modules_303.module_type_store, module_type_store)
    else:
        from errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'errors_copy.type_error_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'errors_copy.type_error_copy', import_302)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from errors_copy.type_warning_copy import TypeWarning' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_304 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'errors_copy.type_warning_copy')

if (type(import_304) is not StypyTypeError):

    if (import_304 != 'pyd_module'):
        __import__(import_304)
        sys_modules_305 = sys.modules[import_304]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'errors_copy.type_warning_copy', sys_modules_305.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_305, sys_modules_305.module_type_store, module_type_store)
    else:
        from errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'errors_copy.type_warning_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'errors_copy.type_warning_copy', import_304)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from errors_copy.unsupported_features_copy import create_unsupported_python_feature_message' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_306 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'errors_copy.unsupported_features_copy')

if (type(import_306) is not StypyTypeError):

    if (import_306 != 'pyd_module'):
        __import__(import_306)
        sys_modules_307 = sys.modules[import_306]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'errors_copy.unsupported_features_copy', sys_modules_307.module_type_store, module_type_store, ['create_unsupported_python_feature_message'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_307, sys_modules_307.module_type_store, module_type_store)
    else:
        from errors_copy.unsupported_features_copy import create_unsupported_python_feature_message

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'errors_copy.unsupported_features_copy', None, module_type_store, ['create_unsupported_python_feature_message'], [create_unsupported_python_feature_message])

else:
    # Assigning a type to the variable 'errors_copy.unsupported_features_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'errors_copy.unsupported_features_copy', import_306)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from code_generation_copy.type_inference_programs_copy.python_operators_copy import ' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_308 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'code_generation_copy.type_inference_programs_copy.python_operators_copy')

if (type(import_308) is not StypyTypeError):

    if (import_308 != 'pyd_module'):
        __import__(import_308)
        sys_modules_309 = sys.modules[import_308]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'code_generation_copy.type_inference_programs_copy.python_operators_copy', sys_modules_309.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_309, sys_modules_309.module_type_store, module_type_store)
    else:
        from code_generation_copy.type_inference_programs_copy.python_operators_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'code_generation_copy.type_inference_programs_copy.python_operators_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'code_generation_copy.type_inference_programs_copy.python_operators_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'code_generation_copy.type_inference_programs_copy.python_operators_copy', import_308)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from python_lib_copy.module_imports_copy import python_imports_copy' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_310 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'python_lib_copy.module_imports_copy')

if (type(import_310) is not StypyTypeError):

    if (import_310 != 'pyd_module'):
        __import__(import_310)
        sys_modules_311 = sys.modules[import_310]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'python_lib_copy.module_imports_copy', sys_modules_311.module_type_store, module_type_store, ['python_imports_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_311, sys_modules_311.module_type_store, module_type_store)
    else:
        from python_lib_copy.module_imports_copy import python_imports_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'python_lib_copy.module_imports_copy', None, module_type_store, ['python_imports_copy'], [python_imports_copy])

else:
    # Assigning a type to the variable 'python_lib_copy.module_imports_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'python_lib_copy.module_imports_copy', import_310)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_312 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy')

if (type(import_312) is not StypyTypeError):

    if (import_312 != 'pyd_module'):
        __import__(import_312)
        sys_modules_313 = sys.modules[import_312]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', sys_modules_313.module_type_store, module_type_store, ['UndefinedType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_313, sys_modules_313.module_type_store, module_type_store)
    else:
        from python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', None, module_type_store, ['UndefinedType'], [UndefinedType])

else:
    # Assigning a type to the variable 'python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', import_312)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_314 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy')

if (type(import_314) is not StypyTypeError):

    if (import_314 != 'pyd_module'):
        __import__(import_314)
        sys_modules_315 = sys.modules[import_314]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', sys_modules_315.module_type_store, module_type_store, ['ExtraTypeDefinitions'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_315, sys_modules_315.module_type_store, module_type_store)
    else:
        from python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', None, module_type_store, ['ExtraTypeDefinitions'], [ExtraTypeDefinitions])

else:
    # Assigning a type to the variable 'python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', import_314)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from python_lib_copy.type_rules_copy.type_groups_copy import type_group_generator_copy, type_groups_copy' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_316 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'python_lib_copy.type_rules_copy.type_groups_copy')

if (type(import_316) is not StypyTypeError):

    if (import_316 != 'pyd_module'):
        __import__(import_316)
        sys_modules_317 = sys.modules[import_316]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'python_lib_copy.type_rules_copy.type_groups_copy', sys_modules_317.module_type_store, module_type_store, ['type_group_generator_copy', 'type_groups_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_317, sys_modules_317.module_type_store, module_type_store)
    else:
        from python_lib_copy.type_rules_copy.type_groups_copy import type_group_generator_copy, type_groups_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'python_lib_copy.type_rules_copy.type_groups_copy', None, module_type_store, ['type_group_generator_copy', 'type_groups_copy'], [type_group_generator_copy, type_groups_copy])

else:
    # Assigning a type to the variable 'python_lib_copy.type_rules_copy.type_groups_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'python_lib_copy.type_rules_copy.type_groups_copy', import_316)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 11)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')
import_318 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_318) is not StypyTypeError):

    if (import_318 != 'pyd_module'):
        __import__(import_318)
        sys_modules_319 = sys.modules[import_318]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'python_lib_copy.python_types_copy.type_inference_copy', sys_modules_319.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_319, sys_modules_319.module_type_store, module_type_store)
    else:
        from python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'python_lib_copy.python_types_copy.type_inference_copy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'python_lib_copy.python_types_copy.type_inference_copy', import_318)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/')

str_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\nThis file contains the stypy API that can be called inside the type inference generated programs source code.\nThese functions will be used to interact with stypy, extract type information and other necessary operations when\ngenerating type inference code.\n')
str_321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, (-1)), 'str', '\nAn object containing the Python __builtins__ module, containing the type inference functions for each Python builtin\n')

# Assigning a Call to a Name (line 22):

# Assigning a Call to a Name (line 22):

# Call to get_module_from_sys_cache(...): (line 22)
# Processing the call arguments (line 22)
str_324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 63), 'str', '__builtin__')
# Processing the call keyword arguments (line 22)
kwargs_325 = {}
# Getting the type of 'python_imports_copy' (line 22)
python_imports_copy_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'python_imports_copy', False)
# Obtaining the member 'get_module_from_sys_cache' of a type (line 22)
get_module_from_sys_cache_323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), python_imports_copy_322, 'get_module_from_sys_cache')
# Calling get_module_from_sys_cache(args, kwargs) (line 22)
get_module_from_sys_cache_call_result_326 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), get_module_from_sys_cache_323, *[str_324], **kwargs_325)

# Assigning a type to the variable 'builtin_module' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'builtin_module', get_module_from_sys_cache_call_result_326)

@norecursion
def get_builtin_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'UndefinedType' (line 25)
    UndefinedType_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 52), 'UndefinedType')
    defaults = [UndefinedType_327]
    # Create a new context for function 'get_builtin_type'
    module_type_store = module_type_store.open_function_context('get_builtin_type', 25, 0, False)
    
    # Passed parameters checking function
    get_builtin_type.stypy_localization = localization
    get_builtin_type.stypy_type_of_self = None
    get_builtin_type.stypy_type_store = module_type_store
    get_builtin_type.stypy_function_name = 'get_builtin_type'
    get_builtin_type.stypy_param_names_list = ['localization', 'type_name', 'value']
    get_builtin_type.stypy_varargs_param_name = None
    get_builtin_type.stypy_kwargs_param_name = None
    get_builtin_type.stypy_call_defaults = defaults
    get_builtin_type.stypy_call_varargs = varargs
    get_builtin_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_builtin_type', ['localization', 'type_name', 'value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_builtin_type', localization, ['localization', 'type_name', 'value'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_builtin_type(...)' code ##################

    str_328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', '\n    Obtains a Python builtin type instance to represent the type of an object in Python. Optionally, a value for\n    this object can be specified. Values for objects are not much used within the current version of stypy, but\n    they are stored for future enhancements. Currently, values, if present, are taken into account for the hasattr,\n    setattr and getattr builtin functions.\n\n    :param localization: Caller information\n    :param type_name: Name of the Python type to be created ("int", "float"...)\n    :param value: Optional value for this type. Value must be of the speficied type. The function does not check this.\n    :return: A TypeInferenceProxy representing the specified type or a TypeError if the specified type do not exist\n    ')
    
    str_329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 7), 'str', 'NoneType')
    # Getting the type of 'type_name' (line 39)
    type_name_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'type_name')
    # Applying the binary operator 'in' (line 39)
    result_contains_331 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 7), 'in', str_329, type_name_330)
    
    # Testing if the type of an if condition is none (line 39)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 39, 4), result_contains_331):
        pass
    else:
        
        # Testing the type of an if condition (line 39)
        if_condition_332 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 4), result_contains_331)
        # Assigning a type to the variable 'if_condition_332' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'if_condition_332', if_condition_332)
        # SSA begins for if statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to import_from(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'localization' (line 40)
        localization_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 47), 'localization', False)
        str_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 61), 'str', 'None')
        # Processing the call keyword arguments (line 40)
        kwargs_337 = {}
        # Getting the type of 'python_imports_copy' (line 40)
        python_imports_copy_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'python_imports_copy', False)
        # Obtaining the member 'import_from' of a type (line 40)
        import_from_334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), python_imports_copy_333, 'import_from')
        # Calling import_from(args, kwargs) (line 40)
        import_from_call_result_338 = invoke(stypy.reporting.localization.Localization(__file__, 40, 15), import_from_334, *[localization_335, str_336], **kwargs_337)
        
        # Assigning a type to the variable 'stypy_return_type' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'stypy_return_type', import_from_call_result_338)
        # SSA join for if statement (line 39)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to hasattr(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'ExtraTypeDefinitions' (line 44)
    ExtraTypeDefinitions_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'ExtraTypeDefinitions', False)
    # Getting the type of 'type_name' (line 44)
    type_name_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 37), 'type_name', False)
    # Processing the call keyword arguments (line 44)
    kwargs_342 = {}
    # Getting the type of 'hasattr' (line 44)
    hasattr_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 7), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 44)
    hasattr_call_result_343 = invoke(stypy.reporting.localization.Localization(__file__, 44, 7), hasattr_339, *[ExtraTypeDefinitions_340, type_name_341], **kwargs_342)
    
    # Testing if the type of an if condition is none (line 44)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 4), hasattr_call_result_343):
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to get_type_of_member(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'localization' (line 48)
        localization_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 53), 'localization', False)
        # Getting the type of 'type_name' (line 48)
        type_name_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 67), 'type_name', False)
        # Processing the call keyword arguments (line 48)
        kwargs_354 = {}
        # Getting the type of 'builtin_module' (line 48)
        builtin_module_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'builtin_module', False)
        # Obtaining the member 'get_type_of_member' of a type (line 48)
        get_type_of_member_351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), builtin_module_350, 'get_type_of_member')
        # Calling get_type_of_member(args, kwargs) (line 48)
        get_type_of_member_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), get_type_of_member_351, *[localization_352, type_name_353], **kwargs_354)
        
        # Assigning a type to the variable 'ret_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'ret_type', get_type_of_member_call_result_355)
    else:
        
        # Testing the type of an if condition (line 44)
        if_condition_344 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 4), hasattr_call_result_343)
        # Assigning a type to the variable 'if_condition_344' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'if_condition_344', if_condition_344)
        # SSA begins for if statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to getattr(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'ExtraTypeDefinitions' (line 45)
        ExtraTypeDefinitions_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'ExtraTypeDefinitions', False)
        # Getting the type of 'type_name' (line 45)
        type_name_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 49), 'type_name', False)
        # Processing the call keyword arguments (line 45)
        kwargs_348 = {}
        # Getting the type of 'getattr' (line 45)
        getattr_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 45)
        getattr_call_result_349 = invoke(stypy.reporting.localization.Localization(__file__, 45, 19), getattr_345, *[ExtraTypeDefinitions_346, type_name_347], **kwargs_348)
        
        # Assigning a type to the variable 'ret_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'ret_type', getattr_call_result_349)
        # SSA branch for the else part of an if statement (line 44)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to get_type_of_member(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'localization' (line 48)
        localization_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 53), 'localization', False)
        # Getting the type of 'type_name' (line 48)
        type_name_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 67), 'type_name', False)
        # Processing the call keyword arguments (line 48)
        kwargs_354 = {}
        # Getting the type of 'builtin_module' (line 48)
        builtin_module_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'builtin_module', False)
        # Obtaining the member 'get_type_of_member' of a type (line 48)
        get_type_of_member_351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), builtin_module_350, 'get_type_of_member')
        # Calling get_type_of_member(args, kwargs) (line 48)
        get_type_of_member_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), get_type_of_member_351, *[localization_352, type_name_353], **kwargs_354)
        
        # Assigning a type to the variable 'ret_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'ret_type', get_type_of_member_call_result_355)
        # SSA join for if statement (line 44)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to set_type_instance(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'True' (line 51)
    True_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'True', False)
    # Processing the call keyword arguments (line 51)
    kwargs_359 = {}
    # Getting the type of 'ret_type' (line 51)
    ret_type_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'ret_type', False)
    # Obtaining the member 'set_type_instance' of a type (line 51)
    set_type_instance_357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), ret_type_356, 'set_type_instance')
    # Calling set_type_instance(args, kwargs) (line 51)
    set_type_instance_call_result_360 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), set_type_instance_357, *[True_358], **kwargs_359)
    
    
    # Getting the type of 'value' (line 54)
    value_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 7), 'value')
    # Getting the type of 'UndefinedType' (line 54)
    UndefinedType_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'UndefinedType')
    # Applying the binary operator 'isnot' (line 54)
    result_is_not_363 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), 'isnot', value_361, UndefinedType_362)
    
    # Testing if the type of an if condition is none (line 54)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 54, 4), result_is_not_363):
        pass
    else:
        
        # Testing the type of an if condition (line 54)
        if_condition_364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 4), result_is_not_363)
        # Assigning a type to the variable 'if_condition_364' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'if_condition_364', if_condition_364)
        # SSA begins for if statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_value(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'value' (line 55)
        value_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'value', False)
        # Processing the call keyword arguments (line 55)
        kwargs_368 = {}
        # Getting the type of 'ret_type' (line 55)
        ret_type_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'ret_type', False)
        # Obtaining the member 'set_value' of a type (line 55)
        set_value_366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), ret_type_365, 'set_value')
        # Calling set_value(args, kwargs) (line 55)
        set_value_call_result_369 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), set_value_366, *[value_367], **kwargs_368)
        
        # SSA join for if statement (line 54)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'ret_type' (line 57)
    ret_type_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'ret_type')
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type', ret_type_370)
    
    # ################# End of 'get_builtin_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_builtin_type' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_371)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_builtin_type'
    return stypy_return_type_371

# Assigning a type to the variable 'get_builtin_type' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'get_builtin_type', get_builtin_type)

@norecursion
def get_python_api_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_python_api_type'
    module_type_store = module_type_store.open_function_context('get_python_api_type', 60, 0, False)
    
    # Passed parameters checking function
    get_python_api_type.stypy_localization = localization
    get_python_api_type.stypy_type_of_self = None
    get_python_api_type.stypy_type_store = module_type_store
    get_python_api_type.stypy_function_name = 'get_python_api_type'
    get_python_api_type.stypy_param_names_list = ['localization', 'module_name', 'type_name']
    get_python_api_type.stypy_varargs_param_name = None
    get_python_api_type.stypy_kwargs_param_name = None
    get_python_api_type.stypy_call_defaults = defaults
    get_python_api_type.stypy_call_varargs = varargs
    get_python_api_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_python_api_type', ['localization', 'module_name', 'type_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_python_api_type', localization, ['localization', 'module_name', 'type_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_python_api_type(...)' code ##################

    str_372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, (-1)), 'str', '\n    This function can obtain any type name for any Python module that have it declared. This way we can access\n    non-builtin types such as those declared on the time module and so on, provided they exist within the specified\n    module\n    :param localization: Caller information\n    :param module_name: Module name\n    :param type_name: Type name within the module\n    :return: A TypeInferenceProxy for the specified type or a TypeError if the requested type do not exist\n    ')
    
    # Assigning a Call to a Name (line 71):
    
    # Assigning a Call to a Name (line 71):
    
    # Call to import_python_module(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'localization' (line 71)
    localization_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 54), 'localization', False)
    # Getting the type of 'module_name' (line 71)
    module_name_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 68), 'module_name', False)
    # Processing the call keyword arguments (line 71)
    kwargs_377 = {}
    # Getting the type of 'python_imports_copy' (line 71)
    python_imports_copy_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 13), 'python_imports_copy', False)
    # Obtaining the member 'import_python_module' of a type (line 71)
    import_python_module_374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 13), python_imports_copy_373, 'import_python_module')
    # Calling import_python_module(args, kwargs) (line 71)
    import_python_module_call_result_378 = invoke(stypy.reporting.localization.Localization(__file__, 71, 13), import_python_module_374, *[localization_375, module_name_376], **kwargs_377)
    
    # Assigning a type to the variable 'module' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'module', import_python_module_call_result_378)
    
    # Type idiom detected: calculating its left and rigth part (line 72)
    # Getting the type of 'TypeError' (line 72)
    TypeError_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'TypeError')
    # Getting the type of 'module' (line 72)
    module_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'module')
    
    (may_be_381, more_types_in_union_382) = may_be_subtype(TypeError_379, module_380)

    if may_be_381:

        if more_types_in_union_382:
            # Runtime conditional SSA (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'module' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'module', remove_not_subtype_from_union(module_380, TypeError))
        # Getting the type of 'module' (line 73)
        module_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'module')
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type', module_383)

        if more_types_in_union_382:
            # SSA join for if statement (line 72)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to get_type_of_member(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'localization' (line 75)
    localization_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 37), 'localization', False)
    # Getting the type of 'type_name' (line 75)
    type_name_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 51), 'type_name', False)
    # Processing the call keyword arguments (line 75)
    kwargs_388 = {}
    # Getting the type of 'module' (line 75)
    module_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'module', False)
    # Obtaining the member 'get_type_of_member' of a type (line 75)
    get_type_of_member_385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 11), module_384, 'get_type_of_member')
    # Calling get_type_of_member(args, kwargs) (line 75)
    get_type_of_member_call_result_389 = invoke(stypy.reporting.localization.Localization(__file__, 75, 11), get_type_of_member_385, *[localization_386, type_name_387], **kwargs_388)
    
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type', get_type_of_member_call_result_389)
    
    # ################# End of 'get_python_api_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_python_api_type' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_390)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_python_api_type'
    return stypy_return_type_390

# Assigning a type to the variable 'get_python_api_type' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'get_python_api_type', get_python_api_type)

@norecursion
def import_elements_from_external_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'import_elements_from_external_module'
    module_type_store = module_type_store.open_function_context('import_elements_from_external_module', 78, 0, False)
    
    # Passed parameters checking function
    import_elements_from_external_module.stypy_localization = localization
    import_elements_from_external_module.stypy_type_of_self = None
    import_elements_from_external_module.stypy_type_store = module_type_store
    import_elements_from_external_module.stypy_function_name = 'import_elements_from_external_module'
    import_elements_from_external_module.stypy_param_names_list = ['localization', 'imported_module_name', 'dest_type_store']
    import_elements_from_external_module.stypy_varargs_param_name = 'elements'
    import_elements_from_external_module.stypy_kwargs_param_name = None
    import_elements_from_external_module.stypy_call_defaults = defaults
    import_elements_from_external_module.stypy_call_varargs = varargs
    import_elements_from_external_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'import_elements_from_external_module', ['localization', 'imported_module_name', 'dest_type_store'], 'elements', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'import_elements_from_external_module', localization, ['localization', 'imported_module_name', 'dest_type_store'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'import_elements_from_external_module(...)' code ##################

    str_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, (-1)), 'str', '\n    This function imports all the declared public members of a user-defined or Python library module into the specified\n    type store\n    It modules the from <module> import <element1>, <element2>, ... or * sentences and also the import <module> sentence\n    :param localization: Caller information\n    :param main_module_path: Path of the module to import, i. e. path of the .py file of the module\n    :param imported_module_name: Name of the module\n    :param dest_type_store: Type store to add the module elements\n    :param elements: A variable list of arguments with the elements to import. The value \'*\' means all elements. No\n    value models the "import <module>" sentence\n    :return: None or a TypeError if the requested type do not exist\n    ')
    
    # Call to import_elements_from_external_module(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'localization' (line 92)
    localization_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 68), 'localization', False)
    # Getting the type of 'imported_module_name' (line 92)
    imported_module_name_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 82), 'imported_module_name', False)
    # Getting the type of 'dest_type_store' (line 93)
    dest_type_store_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 63), 'dest_type_store', False)
    # Getting the type of 'sys' (line 93)
    sys_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 80), 'sys', False)
    # Obtaining the member 'path' of a type (line 93)
    path_398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 80), sys_397, 'path')
    # Getting the type of 'elements' (line 94)
    elements_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 64), 'elements', False)
    # Processing the call keyword arguments (line 92)
    kwargs_400 = {}
    # Getting the type of 'python_imports_copy' (line 92)
    python_imports_copy_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'python_imports_copy', False)
    # Obtaining the member 'import_elements_from_external_module' of a type (line 92)
    import_elements_from_external_module_393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 11), python_imports_copy_392, 'import_elements_from_external_module')
    # Calling import_elements_from_external_module(args, kwargs) (line 92)
    import_elements_from_external_module_call_result_401 = invoke(stypy.reporting.localization.Localization(__file__, 92, 11), import_elements_from_external_module_393, *[localization_394, imported_module_name_395, dest_type_store_396, path_398, elements_399], **kwargs_400)
    
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', import_elements_from_external_module_call_result_401)
    
    # ################# End of 'import_elements_from_external_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'import_elements_from_external_module' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_402)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'import_elements_from_external_module'
    return stypy_return_type_402

# Assigning a type to the variable 'import_elements_from_external_module' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'import_elements_from_external_module', import_elements_from_external_module)

@norecursion
def import_from(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 55), 'str', '__builtin__')
    defaults = [str_403]
    # Create a new context for function 'import_from'
    module_type_store = module_type_store.open_function_context('import_from', 97, 0, False)
    
    # Passed parameters checking function
    import_from.stypy_localization = localization
    import_from.stypy_type_of_self = None
    import_from.stypy_type_store = module_type_store
    import_from.stypy_function_name = 'import_from'
    import_from.stypy_param_names_list = ['localization', 'member_name', 'module_name']
    import_from.stypy_varargs_param_name = None
    import_from.stypy_kwargs_param_name = None
    import_from.stypy_call_defaults = defaults
    import_from.stypy_call_varargs = varargs
    import_from.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'import_from', ['localization', 'member_name', 'module_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'import_from', localization, ['localization', 'member_name', 'module_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'import_from(...)' code ##################

    str_404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, (-1)), 'str', '\n    Imports a single member from a module. If no module is specified, the builtin module is used instead. Models the\n    "from <module> import <member>" sentence, being a sort version of the import_elements_from_external_module function\n    but only for Python library modules\n    :param localization: Caller information\n    :param member_name: Member to import\n    :param module_name: Python library module that contains the member or nothing to use the __builtins__ module\n    :return: A TypeInferenceProxy for the specified member or a TypeError if the requested element do not exist\n    ')
    
    # Call to import_from(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'localization' (line 107)
    localization_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 43), 'localization', False)
    # Getting the type of 'member_name' (line 107)
    member_name_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 57), 'member_name', False)
    # Getting the type of 'module_name' (line 107)
    module_name_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 70), 'module_name', False)
    # Processing the call keyword arguments (line 107)
    kwargs_410 = {}
    # Getting the type of 'python_imports_copy' (line 107)
    python_imports_copy_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'python_imports_copy', False)
    # Obtaining the member 'import_from' of a type (line 107)
    import_from_406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 11), python_imports_copy_405, 'import_from')
    # Calling import_from(args, kwargs) (line 107)
    import_from_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 107, 11), import_from_406, *[localization_407, member_name_408, module_name_409], **kwargs_410)
    
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type', import_from_call_result_411)
    
    # ################# End of 'import_from(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'import_from' in the type store
    # Getting the type of 'stypy_return_type' (line 97)
    stypy_return_type_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_412)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'import_from'
    return stypy_return_type_412

# Assigning a type to the variable 'import_from' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'import_from', import_from)

@norecursion
def import_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 44), 'str', '__builtin__')
    defaults = [str_413]
    # Create a new context for function 'import_module'
    module_type_store = module_type_store.open_function_context('import_module', 110, 0, False)
    
    # Passed parameters checking function
    import_module.stypy_localization = localization
    import_module.stypy_type_of_self = None
    import_module.stypy_type_store = module_type_store
    import_module.stypy_function_name = 'import_module'
    import_module.stypy_param_names_list = ['localization', 'module_name']
    import_module.stypy_varargs_param_name = None
    import_module.stypy_kwargs_param_name = None
    import_module.stypy_call_defaults = defaults
    import_module.stypy_call_varargs = varargs
    import_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'import_module', ['localization', 'module_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'import_module', localization, ['localization', 'module_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'import_module(...)' code ##################

    str_414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, (-1)), 'str', '\n    Import a full Python library module (models the "import <module>" sentence for Python library modules\n    :param localization: Caller information\n    :param module_name: Module to import\n    :return: A TypeInferenceProxy for the specified module or a TypeError if the requested module do not exist\n    ')
    
    # Call to import_python_module(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'localization' (line 117)
    localization_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 52), 'localization', False)
    # Getting the type of 'module_name' (line 117)
    module_name_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 66), 'module_name', False)
    # Processing the call keyword arguments (line 117)
    kwargs_419 = {}
    # Getting the type of 'python_imports_copy' (line 117)
    python_imports_copy_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'python_imports_copy', False)
    # Obtaining the member 'import_python_module' of a type (line 117)
    import_python_module_416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 11), python_imports_copy_415, 'import_python_module')
    # Calling import_python_module(args, kwargs) (line 117)
    import_python_module_call_result_420 = invoke(stypy.reporting.localization.Localization(__file__, 117, 11), import_python_module_416, *[localization_417, module_name_418], **kwargs_419)
    
    # Assigning a type to the variable 'stypy_return_type' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type', import_python_module_call_result_420)
    
    # ################# End of 'import_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'import_module' in the type store
    # Getting the type of 'stypy_return_type' (line 110)
    stypy_return_type_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_421)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'import_module'
    return stypy_return_type_421

# Assigning a type to the variable 'import_module' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'import_module', import_module)

# Assigning a Name to a Name (line 124):

# Assigning a Name to a Name (line 124):
# Getting the type of 'None' (line 124)
None_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'None')
# Assigning a type to the variable 'builtin_operators_module' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'builtin_operators_module', None_422)

@norecursion
def load_builtin_operators_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'load_builtin_operators_module'
    module_type_store = module_type_store.open_function_context('load_builtin_operators_module', 127, 0, False)
    
    # Passed parameters checking function
    load_builtin_operators_module.stypy_localization = localization
    load_builtin_operators_module.stypy_type_of_self = None
    load_builtin_operators_module.stypy_type_store = module_type_store
    load_builtin_operators_module.stypy_function_name = 'load_builtin_operators_module'
    load_builtin_operators_module.stypy_param_names_list = []
    load_builtin_operators_module.stypy_varargs_param_name = None
    load_builtin_operators_module.stypy_kwargs_param_name = None
    load_builtin_operators_module.stypy_call_defaults = defaults
    load_builtin_operators_module.stypy_call_varargs = varargs
    load_builtin_operators_module.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'load_builtin_operators_module', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'load_builtin_operators_module', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'load_builtin_operators_module(...)' code ##################

    str_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, (-1)), 'str', '\n    Loads the builtin Python operators logic that model the Python operator behavior, as a clone of the "operator"\n    Python library module, that initially holds the same behavior for each operator. Once initially loaded, this logic\n    cannot be altered (in Python we cannot overload the \'+\' operator behavior for builtin types, but we can modify the\n    behavior of the equivalent operator.add function).\n    :return: The behavior of the Python operators\n    ')
    # Marking variables as global (line 135)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 135, 4), 'builtin_operators_module')
    
    # Type idiom detected: calculating its left and rigth part (line 138)
    # Getting the type of 'builtin_operators_module' (line 138)
    builtin_operators_module_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 7), 'builtin_operators_module')
    # Getting the type of 'None' (line 138)
    None_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 35), 'None')
    
    (may_be_426, more_types_in_union_427) = may_be_none(builtin_operators_module_424, None_425)

    if may_be_426:

        if more_types_in_union_427:
            # Runtime conditional SSA (line 138)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to import_python_module(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'None' (line 139)
        None_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 67), 'None', False)
        str_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 73), 'str', 'operator')
        # Processing the call keyword arguments (line 139)
        kwargs_432 = {}
        # Getting the type of 'python_imports_copy' (line 139)
        python_imports_copy_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 26), 'python_imports_copy', False)
        # Obtaining the member 'import_python_module' of a type (line 139)
        import_python_module_429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 26), python_imports_copy_428, 'import_python_module')
        # Calling import_python_module(args, kwargs) (line 139)
        import_python_module_call_result_433 = invoke(stypy.reporting.localization.Localization(__file__, 139, 26), import_python_module_429, *[None_430, str_431], **kwargs_432)
        
        # Assigning a type to the variable 'operator_module' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'operator_module', import_python_module_call_result_433)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to clone(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_436 = {}
        # Getting the type of 'operator_module' (line 140)
        operator_module_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 35), 'operator_module', False)
        # Obtaining the member 'clone' of a type (line 140)
        clone_435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 35), operator_module_434, 'clone')
        # Calling clone(args, kwargs) (line 140)
        clone_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 140, 35), clone_435, *[], **kwargs_436)
        
        # Assigning a type to the variable 'builtin_operators_module' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'builtin_operators_module', clone_call_result_437)
        
        # Assigning a Str to a Attribute (line 141):
        
        # Assigning a Str to a Attribute (line 141):
        str_438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 40), 'str', 'builtin_operators')
        # Getting the type of 'builtin_operators_module' (line 141)
        builtin_operators_module_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'builtin_operators_module')
        # Setting the type of the member 'name' of a type (line 141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), builtin_operators_module_439, 'name', str_438)

        if more_types_in_union_427:
            # SSA join for if statement (line 138)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'builtin_operators_module' (line 143)
    builtin_operators_module_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'builtin_operators_module')
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type', builtin_operators_module_440)
    
    # ################# End of 'load_builtin_operators_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'load_builtin_operators_module' in the type store
    # Getting the type of 'stypy_return_type' (line 127)
    stypy_return_type_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_441)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'load_builtin_operators_module'
    return stypy_return_type_441

# Assigning a type to the variable 'load_builtin_operators_module' (line 127)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'load_builtin_operators_module', load_builtin_operators_module)

# Assigning a List to a Name (line 146):

# Assigning a List to a Name (line 146):

# Obtaining an instance of the builtin type 'list' (line 146)
list_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 146)
# Adding element type (line 146)

# Obtaining an instance of the builtin type 'tuple' (line 147)
tuple_443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 147)
# Adding element type (line 147)

# Obtaining an instance of the builtin type 'list' (line 147)
list_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 147)
# Adding element type (line 147)
str_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 6), 'str', 'lt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), list_444, str_445)
# Adding element type (line 147)
str_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 12), 'str', 'gt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), list_444, str_446)
# Adding element type (line 147)
str_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 18), 'str', 'lte')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), list_444, str_447)
# Adding element type (line 147)
str_448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 25), 'str', 'gte')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), list_444, str_448)
# Adding element type (line 147)
str_449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 32), 'str', 'le')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), list_444, str_449)
# Adding element type (line 147)
str_450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 38), 'str', 'ge')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), list_444, str_450)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), tuple_443, list_444)
# Adding element type (line 147)
# Getting the type of 'type_group_generator_copy' (line 147)
type_group_generator_copy_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 45), 'type_group_generator_copy')
# Obtaining the member 'Integer' of a type (line 147)
Integer_452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 45), type_group_generator_copy_451, 'Integer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), tuple_443, Integer_452)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 32), list_442, tuple_443)

# Assigning a type to the variable 'forced_operator_result_checks' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'forced_operator_result_checks', list_442)

@norecursion
def operator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'operator'
    module_type_store = module_type_store.open_function_context('operator', 151, 0, False)
    
    # Passed parameters checking function
    operator.stypy_localization = localization
    operator.stypy_type_of_self = None
    operator.stypy_type_store = module_type_store
    operator.stypy_function_name = 'operator'
    operator.stypy_param_names_list = ['localization', 'operator_symbol']
    operator.stypy_varargs_param_name = 'arguments'
    operator.stypy_kwargs_param_name = None
    operator.stypy_call_defaults = defaults
    operator.stypy_call_varargs = varargs
    operator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'operator', ['localization', 'operator_symbol'], 'arguments', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'operator', localization, ['localization', 'operator_symbol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'operator(...)' code ##################

    str_453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, (-1)), 'str', "\n    Handles all the invokations to Python operators of the type inference program.\n    :param localization: Caller information\n    :param operator_symbol: Operator symbol ('+', '-',...). Symbols instead of operator names ('add', 'sub', ...)\n    are used in the generated type inference program to improve readability\n    :param arguments: Variable list of arguments of the operator\n    :return: Return type of the operator call\n    ")
    # Marking variables as global (line 160)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 160, 4), 'builtin_operators_module')
    
    # Call to load_builtin_operators_module(...): (line 162)
    # Processing the call keyword arguments (line 162)
    kwargs_455 = {}
    # Getting the type of 'load_builtin_operators_module' (line 162)
    load_builtin_operators_module_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'load_builtin_operators_module', False)
    # Calling load_builtin_operators_module(args, kwargs) (line 162)
    load_builtin_operators_module_call_result_456 = invoke(stypy.reporting.localization.Localization(__file__, 162, 4), load_builtin_operators_module_454, *[], **kwargs_455)
    
    
    
    # SSA begins for try-except statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to operator_symbol_to_name(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'operator_symbol' (line 166)
    operator_symbol_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 48), 'operator_symbol', False)
    # Processing the call keyword arguments (line 166)
    kwargs_459 = {}
    # Getting the type of 'operator_symbol_to_name' (line 166)
    operator_symbol_to_name_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'operator_symbol_to_name', False)
    # Calling operator_symbol_to_name(args, kwargs) (line 166)
    operator_symbol_to_name_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 166, 24), operator_symbol_to_name_457, *[operator_symbol_458], **kwargs_459)
    
    # Assigning a type to the variable 'operator_name' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'operator_name', operator_symbol_to_name_call_result_460)
    # SSA branch for the except part of a try statement (line 164)
    # SSA branch for the except '<any exception>' branch of a try statement (line 164)
    module_type_store.open_ssa_branch('except')
    
    # Call to TypeError(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'localization' (line 169)
    localization_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 25), 'localization', False)
    
    # Call to format(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'operator_symbol' (line 169)
    operator_symbol_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 75), 'operator_symbol', False)
    # Processing the call keyword arguments (line 169)
    kwargs_466 = {}
    str_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 39), 'str', 'Unrecognized operator: {0}')
    # Obtaining the member 'format' of a type (line 169)
    format_464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 39), str_463, 'format')
    # Calling format(args, kwargs) (line 169)
    format_call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 169, 39), format_464, *[operator_symbol_465], **kwargs_466)
    
    # Processing the call keyword arguments (line 169)
    kwargs_468 = {}
    # Getting the type of 'TypeError' (line 169)
    TypeError_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 169)
    TypeError_call_result_469 = invoke(stypy.reporting.localization.Localization(__file__, 169, 15), TypeError_461, *[localization_462, format_call_result_467], **kwargs_468)
    
    # Assigning a type to the variable 'stypy_return_type' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stypy_return_type', TypeError_call_result_469)
    # SSA join for try-except statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 172):
    
    # Assigning a Call to a Name (line 172):
    
    # Call to get_type_of_member(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'localization' (line 172)
    localization_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 64), 'localization', False)
    # Getting the type of 'operator_name' (line 172)
    operator_name_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 78), 'operator_name', False)
    # Processing the call keyword arguments (line 172)
    kwargs_474 = {}
    # Getting the type of 'builtin_operators_module' (line 172)
    builtin_operators_module_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'builtin_operators_module', False)
    # Obtaining the member 'get_type_of_member' of a type (line 172)
    get_type_of_member_471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 20), builtin_operators_module_470, 'get_type_of_member')
    # Calling get_type_of_member(args, kwargs) (line 172)
    get_type_of_member_call_result_475 = invoke(stypy.reporting.localization.Localization(__file__, 172, 20), get_type_of_member_471, *[localization_472, operator_name_473], **kwargs_474)
    
    # Assigning a type to the variable 'operator_call' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'operator_call', get_type_of_member_call_result_475)
    
    # Getting the type of 'operator_name' (line 175)
    operator_name_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 7), 'operator_name')
    str_477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 24), 'str', 'contains')
    # Applying the binary operator '==' (line 175)
    result_eq_478 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 7), '==', operator_name_476, str_477)
    
    # Testing if the type of an if condition is none (line 175)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 175, 4), result_eq_478):
        pass
    else:
        
        # Testing the type of an if condition (line 175)
        if_condition_479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 4), result_eq_478)
        # Assigning a type to the variable 'if_condition_479' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'if_condition_479', if_condition_479)
        # SSA begins for if statement (line 175)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to tuple(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Call to reversed(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'arguments' (line 176)
        arguments_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 35), 'arguments', False)
        # Processing the call keyword arguments (line 176)
        kwargs_483 = {}
        # Getting the type of 'reversed' (line 176)
        reversed_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'reversed', False)
        # Calling reversed(args, kwargs) (line 176)
        reversed_call_result_484 = invoke(stypy.reporting.localization.Localization(__file__, 176, 26), reversed_481, *[arguments_482], **kwargs_483)
        
        # Processing the call keyword arguments (line 176)
        kwargs_485 = {}
        # Getting the type of 'tuple' (line 176)
        tuple_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 176)
        tuple_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 176, 20), tuple_480, *[reversed_call_result_484], **kwargs_485)
        
        # Assigning a type to the variable 'arguments' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'arguments', tuple_call_result_486)
        # SSA join for if statement (line 175)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 179):
    
    # Assigning a Call to a Name (line 179):
    
    # Call to invoke(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'localization' (line 179)
    localization_489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'localization', False)
    # Getting the type of 'arguments' (line 179)
    arguments_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 49), 'arguments', False)
    # Processing the call keyword arguments (line 179)
    kwargs_491 = {}
    # Getting the type of 'operator_call' (line 179)
    operator_call_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 'operator_call', False)
    # Obtaining the member 'invoke' of a type (line 179)
    invoke_488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 13), operator_call_487, 'invoke')
    # Calling invoke(args, kwargs) (line 179)
    invoke_call_result_492 = invoke(stypy.reporting.localization.Localization(__file__, 179, 13), invoke_488, *[localization_489, arguments_490], **kwargs_491)
    
    # Assigning a type to the variable 'result' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'result', invoke_call_result_492)
    
    # Getting the type of 'forced_operator_result_checks' (line 180)
    forced_operator_result_checks_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'forced_operator_result_checks')
    # Assigning a type to the variable 'forced_operator_result_checks_493' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'forced_operator_result_checks_493', forced_operator_result_checks_493)
    # Testing if the for loop is going to be iterated (line 180)
    # Testing the type of a for loop iterable (line 180)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 180, 4), forced_operator_result_checks_493)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 180, 4), forced_operator_result_checks_493):
        # Getting the type of the for loop variable (line 180)
        for_loop_var_494 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 180, 4), forced_operator_result_checks_493)
        # Assigning a type to the variable 'check_tuple' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'check_tuple', for_loop_var_494)
        # SSA begins for a for statement (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'operator_name' (line 181)
        operator_name_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'operator_name')
        
        # Obtaining the type of the subscript
        int_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 40), 'int')
        # Getting the type of 'check_tuple' (line 181)
        check_tuple_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'check_tuple')
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 28), check_tuple_497, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_499 = invoke(stypy.reporting.localization.Localization(__file__, 181, 28), getitem___498, int_496)
        
        # Applying the binary operator 'in' (line 181)
        result_contains_500 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 11), 'in', operator_name_495, subscript_call_result_499)
        
        # Testing if the type of an if condition is none (line 181)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 181, 8), result_contains_500):
            pass
        else:
            
            # Testing the type of an if condition (line 181)
            if_condition_501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 8), result_contains_500)
            # Assigning a type to the variable 'if_condition_501' (line 181)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'if_condition_501', if_condition_501)
            # SSA begins for if statement (line 181)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Obtaining the type of the subscript
            int_502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 27), 'int')
            # Getting the type of 'check_tuple' (line 182)
            check_tuple_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'check_tuple')
            # Obtaining the member '__getitem__' of a type (line 182)
            getitem___504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), check_tuple_503, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 182)
            subscript_call_result_505 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), getitem___504, int_502)
            
            # Getting the type of 'result' (line 182)
            result_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 33), 'result')
            # Applying the binary operator '==' (line 182)
            result_eq_507 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 15), '==', subscript_call_result_505, result_506)
            
            # Testing if the type of an if condition is none (line 182)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 182, 12), result_eq_507):
                
                # Call to TypeError(...): (line 185)
                # Processing the call arguments (line 185)
                # Getting the type of 'localization' (line 185)
                localization_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 33), 'localization', False)
                
                # Call to format(...): (line 186)
                # Processing the call arguments (line 186)
                # Getting the type of 'operator_name' (line 186)
                operator_name_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 77), 'operator_name', False)
                
                # Obtaining the type of the subscript
                int_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 104), 'int')
                # Getting the type of 'check_tuple' (line 186)
                check_tuple_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 92), 'check_tuple', False)
                # Obtaining the member '__getitem__' of a type (line 186)
                getitem___517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 92), check_tuple_516, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                subscript_call_result_518 = invoke(stypy.reporting.localization.Localization(__file__, 186, 92), getitem___517, int_515)
                
                # Processing the call keyword arguments (line 186)
                kwargs_519 = {}
                str_512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 33), 'str', 'Operator {0} did not return an {1}')
                # Obtaining the member 'format' of a type (line 186)
                format_513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 33), str_512, 'format')
                # Calling format(args, kwargs) (line 186)
                format_call_result_520 = invoke(stypy.reporting.localization.Localization(__file__, 186, 33), format_513, *[operator_name_514, subscript_call_result_518], **kwargs_519)
                
                # Processing the call keyword arguments (line 185)
                kwargs_521 = {}
                # Getting the type of 'TypeError' (line 185)
                TypeError_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 185)
                TypeError_call_result_522 = invoke(stypy.reporting.localization.Localization(__file__, 185, 23), TypeError_510, *[localization_511, format_call_result_520], **kwargs_521)
                
                # Assigning a type to the variable 'stypy_return_type' (line 185)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'stypy_return_type', TypeError_call_result_522)
            else:
                
                # Testing the type of an if condition (line 182)
                if_condition_508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 12), result_eq_507)
                # Assigning a type to the variable 'if_condition_508' (line 182)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'if_condition_508', if_condition_508)
                # SSA begins for if statement (line 182)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'result' (line 183)
                result_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 23), 'result')
                # Assigning a type to the variable 'stypy_return_type' (line 183)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'stypy_return_type', result_509)
                # SSA branch for the else part of an if statement (line 182)
                module_type_store.open_ssa_branch('else')
                
                # Call to TypeError(...): (line 185)
                # Processing the call arguments (line 185)
                # Getting the type of 'localization' (line 185)
                localization_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 33), 'localization', False)
                
                # Call to format(...): (line 186)
                # Processing the call arguments (line 186)
                # Getting the type of 'operator_name' (line 186)
                operator_name_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 77), 'operator_name', False)
                
                # Obtaining the type of the subscript
                int_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 104), 'int')
                # Getting the type of 'check_tuple' (line 186)
                check_tuple_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 92), 'check_tuple', False)
                # Obtaining the member '__getitem__' of a type (line 186)
                getitem___517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 92), check_tuple_516, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                subscript_call_result_518 = invoke(stypy.reporting.localization.Localization(__file__, 186, 92), getitem___517, int_515)
                
                # Processing the call keyword arguments (line 186)
                kwargs_519 = {}
                str_512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 33), 'str', 'Operator {0} did not return an {1}')
                # Obtaining the member 'format' of a type (line 186)
                format_513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 33), str_512, 'format')
                # Calling format(args, kwargs) (line 186)
                format_call_result_520 = invoke(stypy.reporting.localization.Localization(__file__, 186, 33), format_513, *[operator_name_514, subscript_call_result_518], **kwargs_519)
                
                # Processing the call keyword arguments (line 185)
                kwargs_521 = {}
                # Getting the type of 'TypeError' (line 185)
                TypeError_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 185)
                TypeError_call_result_522 = invoke(stypy.reporting.localization.Localization(__file__, 185, 23), TypeError_510, *[localization_511, format_call_result_520], **kwargs_521)
                
                # Assigning a type to the variable 'stypy_return_type' (line 185)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'stypy_return_type', TypeError_call_result_522)
                # SSA join for if statement (line 182)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 181)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'result' (line 187)
    result_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type', result_523)
    
    # ################# End of 'operator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'operator' in the type store
    # Getting the type of 'stypy_return_type' (line 151)
    stypy_return_type_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_524)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'operator'
    return stypy_return_type_524

# Assigning a type to the variable 'operator' (line 151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'operator', operator)

@norecursion
def unsupported_python_feature(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 66), 'str', '')
    defaults = [str_525]
    # Create a new context for function 'unsupported_python_feature'
    module_type_store = module_type_store.open_function_context('unsupported_python_feature', 190, 0, False)
    
    # Passed parameters checking function
    unsupported_python_feature.stypy_localization = localization
    unsupported_python_feature.stypy_type_of_self = None
    unsupported_python_feature.stypy_type_store = module_type_store
    unsupported_python_feature.stypy_function_name = 'unsupported_python_feature'
    unsupported_python_feature.stypy_param_names_list = ['localization', 'feature', 'description']
    unsupported_python_feature.stypy_varargs_param_name = None
    unsupported_python_feature.stypy_kwargs_param_name = None
    unsupported_python_feature.stypy_call_defaults = defaults
    unsupported_python_feature.stypy_call_varargs = varargs
    unsupported_python_feature.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unsupported_python_feature', ['localization', 'feature', 'description'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unsupported_python_feature', localization, ['localization', 'feature', 'description'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unsupported_python_feature(...)' code ##################

    str_526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, (-1)), 'str', '\n    This is called when the type inference program uses not yet supported by stypy Python feature\n    :param localization: Caller information\n    :param feature: Feature name\n    :param description: Message to give to the user\n    :return: A specific TypeError for this kind of problem\n    ')
    
    # Call to create_unsupported_python_feature_message(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'localization' (line 198)
    localization_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 46), 'localization', False)
    # Getting the type of 'feature' (line 198)
    feature_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 60), 'feature', False)
    # Getting the type of 'description' (line 198)
    description_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 69), 'description', False)
    # Processing the call keyword arguments (line 198)
    kwargs_531 = {}
    # Getting the type of 'create_unsupported_python_feature_message' (line 198)
    create_unsupported_python_feature_message_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'create_unsupported_python_feature_message', False)
    # Calling create_unsupported_python_feature_message(args, kwargs) (line 198)
    create_unsupported_python_feature_message_call_result_532 = invoke(stypy.reporting.localization.Localization(__file__, 198, 4), create_unsupported_python_feature_message_527, *[localization_528, feature_529, description_530], **kwargs_531)
    
    
    # ################# End of 'unsupported_python_feature(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unsupported_python_feature' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_533)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unsupported_python_feature'
    return stypy_return_type_533

# Assigning a type to the variable 'unsupported_python_feature' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'unsupported_python_feature', unsupported_python_feature)

@norecursion
def ensure_var_of_types(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ensure_var_of_types'
    module_type_store = module_type_store.open_function_context('ensure_var_of_types', 201, 0, False)
    
    # Passed parameters checking function
    ensure_var_of_types.stypy_localization = localization
    ensure_var_of_types.stypy_type_of_self = None
    ensure_var_of_types.stypy_type_store = module_type_store
    ensure_var_of_types.stypy_function_name = 'ensure_var_of_types'
    ensure_var_of_types.stypy_param_names_list = ['localization', 'var', 'var_description']
    ensure_var_of_types.stypy_varargs_param_name = 'type_names'
    ensure_var_of_types.stypy_kwargs_param_name = None
    ensure_var_of_types.stypy_call_defaults = defaults
    ensure_var_of_types.stypy_call_varargs = varargs
    ensure_var_of_types.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ensure_var_of_types', ['localization', 'var', 'var_description'], 'type_names', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ensure_var_of_types', localization, ['localization', 'var', 'var_description'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ensure_var_of_types(...)' code ##################

    str_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, (-1)), 'str', '\n    This function is used to be sure that an specific var is of one of the specified types. This function is used\n    by type inference programs when a variable must be of a collection of specific types for the program to be\n    correct, which can happen in certain situations such as if conditions or loop tests.\n    :param localization: Caller information\n    :param var: Variable to test (TypeInferenceProxy)\n    :param var_description: Description of the purpose of the tested variable, to show in a potential TypeError\n    :param type_names: Accepted type names\n    :return: None or a TypeError if the variable do not have a suitable type\n    ')
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to get_python_type(...): (line 213)
    # Processing the call keyword arguments (line 213)
    kwargs_537 = {}
    # Getting the type of 'var' (line 213)
    var_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 18), 'var', False)
    # Obtaining the member 'get_python_type' of a type (line 213)
    get_python_type_536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 18), var_535, 'get_python_type')
    # Calling get_python_type(args, kwargs) (line 213)
    get_python_type_call_result_538 = invoke(stypy.reporting.localization.Localization(__file__, 213, 18), get_python_type_536, *[], **kwargs_537)
    
    # Assigning a type to the variable 'python_type' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'python_type', get_python_type_call_result_538)
    
    # Getting the type of 'type_names' (line 214)
    type_names_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'type_names')
    # Assigning a type to the variable 'type_names_539' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'type_names_539', type_names_539)
    # Testing if the for loop is going to be iterated (line 214)
    # Testing the type of a for loop iterable (line 214)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 214, 4), type_names_539)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 214, 4), type_names_539):
        # Getting the type of the for loop variable (line 214)
        for_loop_var_540 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 214, 4), type_names_539)
        # Assigning a type to the variable 'type_name' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'type_name', for_loop_var_540)
        # SSA begins for a for statement (line 214)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'type_name' (line 215)
        type_name_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'type_name')
        # Getting the type of 'str' (line 215)
        str_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'str')
        # Applying the binary operator 'is' (line 215)
        result_is__543 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 11), 'is', type_name_541, str_542)
        
        # Testing if the type of an if condition is none (line 215)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 215, 8), result_is__543):
            
            # Assigning a Name to a Name (line 218):
            
            # Assigning a Name to a Name (line 218):
            # Getting the type of 'type_name' (line 218)
            type_name_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'type_name')
            # Assigning a type to the variable 'type_obj' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'type_obj', type_name_551)
        else:
            
            # Testing the type of an if condition (line 215)
            if_condition_544 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), result_is__543)
            # Assigning a type to the variable 'if_condition_544' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_544', if_condition_544)
            # SSA begins for if statement (line 215)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 216):
            
            # Assigning a Call to a Name (line 216):
            
            # Call to eval(...): (line 216)
            # Processing the call arguments (line 216)
            str_546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 28), 'str', 'types.')
            # Getting the type of 'type_name' (line 216)
            type_name_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'type_name', False)
            # Applying the binary operator '+' (line 216)
            result_add_548 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 28), '+', str_546, type_name_547)
            
            # Processing the call keyword arguments (line 216)
            kwargs_549 = {}
            # Getting the type of 'eval' (line 216)
            eval_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 23), 'eval', False)
            # Calling eval(args, kwargs) (line 216)
            eval_call_result_550 = invoke(stypy.reporting.localization.Localization(__file__, 216, 23), eval_545, *[result_add_548], **kwargs_549)
            
            # Assigning a type to the variable 'type_obj' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'type_obj', eval_call_result_550)
            # SSA branch for the else part of an if statement (line 215)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 218):
            
            # Assigning a Name to a Name (line 218):
            # Getting the type of 'type_name' (line 218)
            type_name_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'type_name')
            # Assigning a type to the variable 'type_obj' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'type_obj', type_name_551)
            # SSA join for if statement (line 215)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'python_type' (line 220)
        python_type_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 11), 'python_type')
        # Getting the type of 'type_obj' (line 220)
        type_obj_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 26), 'type_obj')
        # Applying the binary operator 'is' (line 220)
        result_is__554 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 11), 'is', python_type_552, type_obj_553)
        
        # Testing if the type of an if condition is none (line 220)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 8), result_is__554):
            pass
        else:
            
            # Testing the type of an if condition (line 220)
            if_condition_555 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 8), result_is__554)
            # Assigning a type to the variable 'if_condition_555' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'if_condition_555', if_condition_555)
            # SSA begins for if statement (line 220)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Assigning a type to the variable 'stypy_return_type' (line 221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'stypy_return_type', types.NoneType)
            # SSA join for if statement (line 220)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to TypeError(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'localization' (line 223)
    localization_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'localization', False)
    # Getting the type of 'var_description' (line 223)
    var_description_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 35), 'var_description', False)
    str_559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 53), 'str', ' must be of one of the following types: ')
    # Applying the binary operator '+' (line 223)
    result_add_560 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 35), '+', var_description_558, str_559)
    
    
    # Call to str(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'type_names' (line 223)
    type_names_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 102), 'type_names', False)
    # Processing the call keyword arguments (line 223)
    kwargs_563 = {}
    # Getting the type of 'str' (line 223)
    str_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 98), 'str', False)
    # Calling str(args, kwargs) (line 223)
    str_call_result_564 = invoke(stypy.reporting.localization.Localization(__file__, 223, 98), str_561, *[type_names_562], **kwargs_563)
    
    # Applying the binary operator '+' (line 223)
    result_add_565 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 96), '+', result_add_560, str_call_result_564)
    
    # Processing the call keyword arguments (line 223)
    kwargs_566 = {}
    # Getting the type of 'TypeError' (line 223)
    TypeError_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 223)
    TypeError_call_result_567 = invoke(stypy.reporting.localization.Localization(__file__, 223, 11), TypeError_556, *[localization_557, result_add_565], **kwargs_566)
    
    # Assigning a type to the variable 'stypy_return_type' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type', TypeError_call_result_567)
    
    # ################# End of 'ensure_var_of_types(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ensure_var_of_types' in the type store
    # Getting the type of 'stypy_return_type' (line 201)
    stypy_return_type_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_568)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ensure_var_of_types'
    return stypy_return_type_568

# Assigning a type to the variable 'ensure_var_of_types' (line 201)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'ensure_var_of_types', ensure_var_of_types)

@norecursion
def ensure_var_has_members(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ensure_var_has_members'
    module_type_store = module_type_store.open_function_context('ensure_var_has_members', 226, 0, False)
    
    # Passed parameters checking function
    ensure_var_has_members.stypy_localization = localization
    ensure_var_has_members.stypy_type_of_self = None
    ensure_var_has_members.stypy_type_store = module_type_store
    ensure_var_has_members.stypy_function_name = 'ensure_var_has_members'
    ensure_var_has_members.stypy_param_names_list = ['localization', 'var', 'var_description']
    ensure_var_has_members.stypy_varargs_param_name = 'member_names'
    ensure_var_has_members.stypy_kwargs_param_name = None
    ensure_var_has_members.stypy_call_defaults = defaults
    ensure_var_has_members.stypy_call_varargs = varargs
    ensure_var_has_members.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ensure_var_has_members', ['localization', 'var', 'var_description'], 'member_names', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ensure_var_has_members', localization, ['localization', 'var', 'var_description'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ensure_var_has_members(...)' code ##################

    str_569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, (-1)), 'str', '\n    This function is used to make sure that a certain variable has an specific set of members, which may be needed\n    when generating some type inference code that needs an specific structure o a certain object\n    :param localization: Caller information\n    :param var: Variable to test (TypeInferenceProxy)\n    :param var_description: Description of the purpose of the tested variable, to show in a potential TypeError\n    :param member_names: List of members that the type of the variable must have to be valid.\n    :return: None or a TypeError if the variable do not have all passed members\n    ')
    
    # Assigning a Call to a Name (line 236):
    
    # Assigning a Call to a Name (line 236):
    
    # Call to get_python_entity(...): (line 236)
    # Processing the call keyword arguments (line 236)
    kwargs_572 = {}
    # Getting the type of 'var' (line 236)
    var_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 18), 'var', False)
    # Obtaining the member 'get_python_entity' of a type (line 236)
    get_python_entity_571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 18), var_570, 'get_python_entity')
    # Calling get_python_entity(args, kwargs) (line 236)
    get_python_entity_call_result_573 = invoke(stypy.reporting.localization.Localization(__file__, 236, 18), get_python_entity_571, *[], **kwargs_572)
    
    # Assigning a type to the variable 'python_type' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'python_type', get_python_entity_call_result_573)
    
    # Getting the type of 'member_names' (line 237)
    member_names_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 21), 'member_names')
    # Assigning a type to the variable 'member_names_574' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'member_names_574', member_names_574)
    # Testing if the for loop is going to be iterated (line 237)
    # Testing the type of a for loop iterable (line 237)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 237, 4), member_names_574)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 237, 4), member_names_574):
        # Getting the type of the for loop variable (line 237)
        for_loop_var_575 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 237, 4), member_names_574)
        # Assigning a type to the variable 'type_name' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'type_name', for_loop_var_575)
        # SSA begins for a for statement (line 237)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to hasattr(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'python_type' (line 238)
        python_type_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 23), 'python_type', False)
        # Getting the type of 'type_name' (line 238)
        type_name_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 36), 'type_name', False)
        # Processing the call keyword arguments (line 238)
        kwargs_579 = {}
        # Getting the type of 'hasattr' (line 238)
        hasattr_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 238)
        hasattr_call_result_580 = invoke(stypy.reporting.localization.Localization(__file__, 238, 15), hasattr_576, *[python_type_577, type_name_578], **kwargs_579)
        
        # Applying the 'not' unary operator (line 238)
        result_not__581 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 11), 'not', hasattr_call_result_580)
        
        # Testing if the type of an if condition is none (line 238)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 238, 8), result_not__581):
            pass
        else:
            
            # Testing the type of an if condition (line 238)
            if_condition_582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), result_not__581)
            # Assigning a type to the variable 'if_condition_582' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_582', if_condition_582)
            # SSA begins for if statement (line 238)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 239)
            # Processing the call arguments (line 239)
            # Getting the type of 'localization' (line 239)
            localization_584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 22), 'localization', False)
            # Getting the type of 'var_description' (line 239)
            var_description_585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 36), 'var_description', False)
            str_586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 54), 'str', ' must have all of these members: ')
            # Applying the binary operator '+' (line 239)
            result_add_587 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 36), '+', var_description_585, str_586)
            
            
            # Call to str(...): (line 239)
            # Processing the call arguments (line 239)
            # Getting the type of 'member_names' (line 239)
            member_names_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 96), 'member_names', False)
            # Processing the call keyword arguments (line 239)
            kwargs_590 = {}
            # Getting the type of 'str' (line 239)
            str_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 92), 'str', False)
            # Calling str(args, kwargs) (line 239)
            str_call_result_591 = invoke(stypy.reporting.localization.Localization(__file__, 239, 92), str_588, *[member_names_589], **kwargs_590)
            
            # Applying the binary operator '+' (line 239)
            result_add_592 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 90), '+', result_add_587, str_call_result_591)
            
            # Processing the call keyword arguments (line 239)
            kwargs_593 = {}
            # Getting the type of 'TypeError' (line 239)
            TypeError_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 239)
            TypeError_call_result_594 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), TypeError_583, *[localization_584, result_add_592], **kwargs_593)
            
            # Getting the type of 'False' (line 240)
            False_595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'stypy_return_type', False_595)
            # SSA join for if statement (line 238)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 242)
    True_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'stypy_return_type', True_596)
    
    # ################# End of 'ensure_var_has_members(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ensure_var_has_members' in the type store
    # Getting the type of 'stypy_return_type' (line 226)
    stypy_return_type_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_597)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ensure_var_has_members'
    return stypy_return_type_597

# Assigning a type to the variable 'ensure_var_has_members' (line 226)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 0), 'ensure_var_has_members', ensure_var_has_members)

@norecursion
def __slice_bounds_checking(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__slice_bounds_checking'
    module_type_store = module_type_store.open_function_context('__slice_bounds_checking', 245, 0, False)
    
    # Passed parameters checking function
    __slice_bounds_checking.stypy_localization = localization
    __slice_bounds_checking.stypy_type_of_self = None
    __slice_bounds_checking.stypy_type_store = module_type_store
    __slice_bounds_checking.stypy_function_name = '__slice_bounds_checking'
    __slice_bounds_checking.stypy_param_names_list = ['bound']
    __slice_bounds_checking.stypy_varargs_param_name = None
    __slice_bounds_checking.stypy_kwargs_param_name = None
    __slice_bounds_checking.stypy_call_defaults = defaults
    __slice_bounds_checking.stypy_call_varargs = varargs
    __slice_bounds_checking.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__slice_bounds_checking', ['bound'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__slice_bounds_checking', localization, ['bound'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__slice_bounds_checking(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 246)
    # Getting the type of 'bound' (line 246)
    bound_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 7), 'bound')
    # Getting the type of 'None' (line 246)
    None_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'None')
    
    (may_be_600, more_types_in_union_601) = may_be_none(bound_598, None_599)

    if may_be_600:

        if more_types_in_union_601:
            # Runtime conditional SSA (line 246)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Obtaining an instance of the builtin type 'tuple' (line 247)
        tuple_602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 247)
        # Adding element type (line 247)
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        # Adding element type (line 247)
        # Getting the type of 'None' (line 247)
        None_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 15), list_603, None_604)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 15), tuple_602, list_603)
        # Adding element type (line 247)
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 15), tuple_602, list_605)
        
        # Assigning a type to the variable 'stypy_return_type' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'stypy_return_type', tuple_602)

        if more_types_in_union_601:
            # SSA join for if statement (line 246)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'bound' (line 246)
    bound_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'bound')
    # Assigning a type to the variable 'bound' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'bound', remove_type_from_union(bound_606, types.NoneType))
    
    # Call to isinstance(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'bound' (line 249)
    bound_608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 18), 'bound', False)
    # Getting the type of 'union_type_copy' (line 249)
    union_type_copy_609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 25), 'union_type_copy', False)
    # Obtaining the member 'UnionType' of a type (line 249)
    UnionType_610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 25), union_type_copy_609, 'UnionType')
    # Processing the call keyword arguments (line 249)
    kwargs_611 = {}
    # Getting the type of 'isinstance' (line 249)
    isinstance_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 249)
    isinstance_call_result_612 = invoke(stypy.reporting.localization.Localization(__file__, 249, 7), isinstance_607, *[bound_608, UnionType_610], **kwargs_611)
    
    # Testing if the type of an if condition is none (line 249)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 249, 4), isinstance_call_result_612):
        
        # Assigning a List to a Name (line 252):
        
        # Assigning a List to a Name (line 252):
        
        # Obtaining an instance of the builtin type 'list' (line 252)
        list_616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 252)
        # Adding element type (line 252)
        # Getting the type of 'bound' (line 252)
        bound_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 26), 'bound')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 25), list_616, bound_617)
        
        # Assigning a type to the variable 'types_to_check' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'types_to_check', list_616)
    else:
        
        # Testing the type of an if condition (line 249)
        if_condition_613 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 4), isinstance_call_result_612)
        # Assigning a type to the variable 'if_condition_613' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'if_condition_613', if_condition_613)
        # SSA begins for if statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 250):
        
        # Assigning a Attribute to a Name (line 250):
        # Getting the type of 'bound' (line 250)
        bound_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 25), 'bound')
        # Obtaining the member 'types' of a type (line 250)
        types_615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 25), bound_614, 'types')
        # Assigning a type to the variable 'types_to_check' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'types_to_check', types_615)
        # SSA branch for the else part of an if statement (line 249)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 252):
        
        # Assigning a List to a Name (line 252):
        
        # Obtaining an instance of the builtin type 'list' (line 252)
        list_616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 252)
        # Adding element type (line 252)
        # Getting the type of 'bound' (line 252)
        bound_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 26), 'bound')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 25), list_616, bound_617)
        
        # Assigning a type to the variable 'types_to_check' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'types_to_check', list_616)
        # SSA join for if statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Name (line 254):
    
    # Assigning a List to a Name (line 254):
    
    # Obtaining an instance of the builtin type 'list' (line 254)
    list_618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 254)
    
    # Assigning a type to the variable 'right_types' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'right_types', list_618)
    
    # Assigning a List to a Name (line 255):
    
    # Assigning a List to a Name (line 255):
    
    # Obtaining an instance of the builtin type 'list' (line 255)
    list_619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 255)
    
    # Assigning a type to the variable 'wrong_types' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'wrong_types', list_619)
    
    # Getting the type of 'types_to_check' (line 256)
    types_to_check_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 17), 'types_to_check')
    # Assigning a type to the variable 'types_to_check_620' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'types_to_check_620', types_to_check_620)
    # Testing if the for loop is going to be iterated (line 256)
    # Testing the type of a for loop iterable (line 256)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 256, 4), types_to_check_620)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 256, 4), types_to_check_620):
        # Getting the type of the for loop variable (line 256)
        for_loop_var_621 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 256, 4), types_to_check_620)
        # Assigning a type to the variable 'type_' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'type_', for_loop_var_621)
        # SSA begins for a for statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'type_group_generator_copy' (line 257)
        type_group_generator_copy_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 'type_group_generator_copy')
        # Obtaining the member 'Integer' of a type (line 257)
        Integer_623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 11), type_group_generator_copy_622, 'Integer')
        # Getting the type of 'type_' (line 257)
        type__624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 48), 'type_')
        # Applying the binary operator '==' (line 257)
        result_eq_625 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 11), '==', Integer_623, type__624)
        
        
        # Getting the type of 'type_groups_copy' (line 257)
        type_groups_copy_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 57), 'type_groups_copy')
        # Obtaining the member 'CastsToIndex' of a type (line 257)
        CastsToIndex_627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 57), type_groups_copy_626, 'CastsToIndex')
        # Getting the type of 'type_' (line 257)
        type__628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 90), 'type_')
        # Applying the binary operator '==' (line 257)
        result_eq_629 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 57), '==', CastsToIndex_627, type__628)
        
        # Applying the binary operator 'or' (line 257)
        result_or_keyword_630 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 11), 'or', result_eq_625, result_eq_629)
        
        # Testing if the type of an if condition is none (line 257)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 257, 8), result_or_keyword_630):
            
            # Call to append(...): (line 260)
            # Processing the call arguments (line 260)
            # Getting the type of 'type_' (line 260)
            type__639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 31), 'type_', False)
            # Processing the call keyword arguments (line 260)
            kwargs_640 = {}
            # Getting the type of 'wrong_types' (line 260)
            wrong_types_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'wrong_types', False)
            # Obtaining the member 'append' of a type (line 260)
            append_638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), wrong_types_637, 'append')
            # Calling append(args, kwargs) (line 260)
            append_call_result_641 = invoke(stypy.reporting.localization.Localization(__file__, 260, 12), append_638, *[type__639], **kwargs_640)
            
        else:
            
            # Testing the type of an if condition (line 257)
            if_condition_631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 8), result_or_keyword_630)
            # Assigning a type to the variable 'if_condition_631' (line 257)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'if_condition_631', if_condition_631)
            # SSA begins for if statement (line 257)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 258)
            # Processing the call arguments (line 258)
            # Getting the type of 'type_' (line 258)
            type__634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 31), 'type_', False)
            # Processing the call keyword arguments (line 258)
            kwargs_635 = {}
            # Getting the type of 'right_types' (line 258)
            right_types_632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'right_types', False)
            # Obtaining the member 'append' of a type (line 258)
            append_633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 12), right_types_632, 'append')
            # Calling append(args, kwargs) (line 258)
            append_call_result_636 = invoke(stypy.reporting.localization.Localization(__file__, 258, 12), append_633, *[type__634], **kwargs_635)
            
            # SSA branch for the else part of an if statement (line 257)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 260)
            # Processing the call arguments (line 260)
            # Getting the type of 'type_' (line 260)
            type__639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 31), 'type_', False)
            # Processing the call keyword arguments (line 260)
            kwargs_640 = {}
            # Getting the type of 'wrong_types' (line 260)
            wrong_types_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'wrong_types', False)
            # Obtaining the member 'append' of a type (line 260)
            append_638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), wrong_types_637, 'append')
            # Calling append(args, kwargs) (line 260)
            append_call_result_641 = invoke(stypy.reporting.localization.Localization(__file__, 260, 12), append_638, *[type__639], **kwargs_640)
            
            # SSA join for if statement (line 257)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 262)
    tuple_642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 262)
    # Adding element type (line 262)
    # Getting the type of 'right_types' (line 262)
    right_types_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 11), 'right_types')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 11), tuple_642, right_types_643)
    # Adding element type (line 262)
    # Getting the type of 'wrong_types' (line 262)
    wrong_types_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'wrong_types')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 11), tuple_642, wrong_types_644)
    
    # Assigning a type to the variable 'stypy_return_type' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'stypy_return_type', tuple_642)
    
    # ################# End of '__slice_bounds_checking(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__slice_bounds_checking' in the type store
    # Getting the type of 'stypy_return_type' (line 245)
    stypy_return_type_645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_645)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__slice_bounds_checking'
    return stypy_return_type_645

# Assigning a type to the variable '__slice_bounds_checking' (line 245)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 0), '__slice_bounds_checking', __slice_bounds_checking)

@norecursion
def ensure_slice_bounds(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ensure_slice_bounds'
    module_type_store = module_type_store.open_function_context('ensure_slice_bounds', 265, 0, False)
    
    # Passed parameters checking function
    ensure_slice_bounds.stypy_localization = localization
    ensure_slice_bounds.stypy_type_of_self = None
    ensure_slice_bounds.stypy_type_store = module_type_store
    ensure_slice_bounds.stypy_function_name = 'ensure_slice_bounds'
    ensure_slice_bounds.stypy_param_names_list = ['localization', 'lower', 'upper', 'step']
    ensure_slice_bounds.stypy_varargs_param_name = None
    ensure_slice_bounds.stypy_kwargs_param_name = None
    ensure_slice_bounds.stypy_call_defaults = defaults
    ensure_slice_bounds.stypy_call_varargs = varargs
    ensure_slice_bounds.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ensure_slice_bounds', ['localization', 'lower', 'upper', 'step'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ensure_slice_bounds', localization, ['localization', 'lower', 'upper', 'step'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ensure_slice_bounds(...)' code ##################

    str_646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, (-1)), 'str', '\n    Check the parameters of a created slice to make sure that the slice have correct bounds. If not, it returns a\n    silent TypeError, as the specific problem (invalid lower, upper or step parameter is reported separately)\n    :param localization: Caller information\n    :param lower: Lower bound of the slice or None\n    :param upper: Upper bound of the slice or None\n    :param step: Step of the slice or None\n    :return: A slice object or a TypeError is its parameters are invalid\n    ')
    
    # Assigning a Name to a Name (line 275):
    
    # Assigning a Name to a Name (line 275):
    # Getting the type of 'False' (line 275)
    False_647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'False')
    # Assigning a type to the variable 'error' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'error', False_647)
    
    # Assigning a Call to a Tuple (line 276):
    
    # Assigning a Call to a Name:
    
    # Call to __slice_bounds_checking(...): (line 276)
    # Processing the call arguments (line 276)
    # Getting the type of 'lower' (line 276)
    lower_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 35), 'lower', False)
    # Processing the call keyword arguments (line 276)
    kwargs_650 = {}
    # Getting the type of '__slice_bounds_checking' (line 276)
    slice_bounds_checking_648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 11), '__slice_bounds_checking', False)
    # Calling __slice_bounds_checking(args, kwargs) (line 276)
    slice_bounds_checking_call_result_651 = invoke(stypy.reporting.localization.Localization(__file__, 276, 11), slice_bounds_checking_648, *[lower_649], **kwargs_650)
    
    # Assigning a type to the variable 'call_assignment_293' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_293', slice_bounds_checking_call_result_651)
    
    # Assigning a Call to a Name (line 276):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_293' (line 276)
    call_assignment_293_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_293', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_653 = stypy_get_value_from_tuple(call_assignment_293_652, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_294' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_294', stypy_get_value_from_tuple_call_result_653)
    
    # Assigning a Name to a Name (line 276):
    # Getting the type of 'call_assignment_294' (line 276)
    call_assignment_294_654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_294')
    # Assigning a type to the variable 'r' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'r', call_assignment_294_654)
    
    # Assigning a Call to a Name (line 276):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_293' (line 276)
    call_assignment_293_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_293', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_656 = stypy_get_value_from_tuple(call_assignment_293_655, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_295' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_295', stypy_get_value_from_tuple_call_result_656)
    
    # Assigning a Name to a Name (line 276):
    # Getting the type of 'call_assignment_295' (line 276)
    call_assignment_295_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_295')
    # Assigning a type to the variable 'w' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 7), 'w', call_assignment_295_657)
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'w' (line 278)
    w_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), 'w', False)
    # Processing the call keyword arguments (line 278)
    kwargs_660 = {}
    # Getting the type of 'len' (line 278)
    len_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 7), 'len', False)
    # Calling len(args, kwargs) (line 278)
    len_call_result_661 = invoke(stypy.reporting.localization.Localization(__file__, 278, 7), len_658, *[w_659], **kwargs_660)
    
    int_662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 16), 'int')
    # Applying the binary operator '>' (line 278)
    result_gt_663 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 7), '>', len_call_result_661, int_662)
    
    
    
    # Call to len(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'r' (line 278)
    r_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 26), 'r', False)
    # Processing the call keyword arguments (line 278)
    kwargs_666 = {}
    # Getting the type of 'len' (line 278)
    len_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 22), 'len', False)
    # Calling len(args, kwargs) (line 278)
    len_call_result_667 = invoke(stypy.reporting.localization.Localization(__file__, 278, 22), len_664, *[r_665], **kwargs_666)
    
    int_668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 31), 'int')
    # Applying the binary operator '>' (line 278)
    result_gt_669 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 22), '>', len_call_result_667, int_668)
    
    # Applying the binary operator 'and' (line 278)
    result_and_keyword_670 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 7), 'and', result_gt_663, result_gt_669)
    
    # Testing if the type of an if condition is none (line 278)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 278, 4), result_and_keyword_670):
        pass
    else:
        
        # Testing the type of an if condition (line 278)
        if_condition_671 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 4), result_and_keyword_670)
        # Assigning a type to the variable 'if_condition_671' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'if_condition_671', if_condition_671)
        # SSA begins for if statement (line 278)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeWarning(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'localization' (line 279)
        localization_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'localization', False)
        
        # Call to format(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'lower' (line 280)
        lower_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 27), 'lower', False)
        # Processing the call keyword arguments (line 279)
        kwargs_677 = {}
        str_674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 34), 'str', 'Some of the possible types of the lower bound of the slice ({0}) are invalid')
        # Obtaining the member 'format' of a type (line 279)
        format_675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 34), str_674, 'format')
        # Calling format(args, kwargs) (line 279)
        format_call_result_678 = invoke(stypy.reporting.localization.Localization(__file__, 279, 34), format_675, *[lower_676], **kwargs_677)
        
        # Processing the call keyword arguments (line 279)
        kwargs_679 = {}
        # Getting the type of 'TypeWarning' (line 279)
        TypeWarning_672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'TypeWarning', False)
        # Calling TypeWarning(args, kwargs) (line 279)
        TypeWarning_call_result_680 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), TypeWarning_672, *[localization_673, format_call_result_678], **kwargs_679)
        
        # SSA join for if statement (line 278)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 281)
    # Processing the call arguments (line 281)
    # Getting the type of 'w' (line 281)
    w_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 11), 'w', False)
    # Processing the call keyword arguments (line 281)
    kwargs_683 = {}
    # Getting the type of 'len' (line 281)
    len_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 7), 'len', False)
    # Calling len(args, kwargs) (line 281)
    len_call_result_684 = invoke(stypy.reporting.localization.Localization(__file__, 281, 7), len_681, *[w_682], **kwargs_683)
    
    int_685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 16), 'int')
    # Applying the binary operator '>' (line 281)
    result_gt_686 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 7), '>', len_call_result_684, int_685)
    
    
    
    # Call to len(...): (line 281)
    # Processing the call arguments (line 281)
    # Getting the type of 'r' (line 281)
    r_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 26), 'r', False)
    # Processing the call keyword arguments (line 281)
    kwargs_689 = {}
    # Getting the type of 'len' (line 281)
    len_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 22), 'len', False)
    # Calling len(args, kwargs) (line 281)
    len_call_result_690 = invoke(stypy.reporting.localization.Localization(__file__, 281, 22), len_687, *[r_688], **kwargs_689)
    
    int_691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 32), 'int')
    # Applying the binary operator '==' (line 281)
    result_eq_692 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 22), '==', len_call_result_690, int_691)
    
    # Applying the binary operator 'and' (line 281)
    result_and_keyword_693 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 7), 'and', result_gt_686, result_eq_692)
    
    # Testing if the type of an if condition is none (line 281)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 281, 4), result_and_keyword_693):
        pass
    else:
        
        # Testing the type of an if condition (line 281)
        if_condition_694 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 4), result_and_keyword_693)
        # Assigning a type to the variable 'if_condition_694' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'if_condition_694', if_condition_694)
        # SSA begins for if statement (line 281)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'localization' (line 282)
        localization_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), 'localization', False)
        
        # Call to format(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'lower' (line 282)
        lower_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 99), 'lower', False)
        # Processing the call keyword arguments (line 282)
        kwargs_700 = {}
        str_697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 32), 'str', 'The type of the lower bound of the slice ({0}) is invalid')
        # Obtaining the member 'format' of a type (line 282)
        format_698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 32), str_697, 'format')
        # Calling format(args, kwargs) (line 282)
        format_call_result_701 = invoke(stypy.reporting.localization.Localization(__file__, 282, 32), format_698, *[lower_699], **kwargs_700)
        
        # Processing the call keyword arguments (line 282)
        kwargs_702 = {}
        # Getting the type of 'TypeError' (line 282)
        TypeError_695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 282)
        TypeError_call_result_703 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), TypeError_695, *[localization_696, format_call_result_701], **kwargs_702)
        
        
        # Assigning a Name to a Name (line 283):
        
        # Assigning a Name to a Name (line 283):
        # Getting the type of 'True' (line 283)
        True_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'True')
        # Assigning a type to the variable 'error' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'error', True_704)
        # SSA join for if statement (line 281)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Tuple (line 285):
    
    # Assigning a Call to a Name:
    
    # Call to __slice_bounds_checking(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'upper' (line 285)
    upper_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 35), 'upper', False)
    # Processing the call keyword arguments (line 285)
    kwargs_707 = {}
    # Getting the type of '__slice_bounds_checking' (line 285)
    slice_bounds_checking_705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), '__slice_bounds_checking', False)
    # Calling __slice_bounds_checking(args, kwargs) (line 285)
    slice_bounds_checking_call_result_708 = invoke(stypy.reporting.localization.Localization(__file__, 285, 11), slice_bounds_checking_705, *[upper_706], **kwargs_707)
    
    # Assigning a type to the variable 'call_assignment_296' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_296', slice_bounds_checking_call_result_708)
    
    # Assigning a Call to a Name (line 285):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_296' (line 285)
    call_assignment_296_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_296', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_710 = stypy_get_value_from_tuple(call_assignment_296_709, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_297' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_297', stypy_get_value_from_tuple_call_result_710)
    
    # Assigning a Name to a Name (line 285):
    # Getting the type of 'call_assignment_297' (line 285)
    call_assignment_297_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_297')
    # Assigning a type to the variable 'r' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'r', call_assignment_297_711)
    
    # Assigning a Call to a Name (line 285):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_296' (line 285)
    call_assignment_296_712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_296', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_713 = stypy_get_value_from_tuple(call_assignment_296_712, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_298' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_298', stypy_get_value_from_tuple_call_result_713)
    
    # Assigning a Name to a Name (line 285):
    # Getting the type of 'call_assignment_298' (line 285)
    call_assignment_298_714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_298')
    # Assigning a type to the variable 'w' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 7), 'w', call_assignment_298_714)
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'w' (line 286)
    w_716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 11), 'w', False)
    # Processing the call keyword arguments (line 286)
    kwargs_717 = {}
    # Getting the type of 'len' (line 286)
    len_715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 7), 'len', False)
    # Calling len(args, kwargs) (line 286)
    len_call_result_718 = invoke(stypy.reporting.localization.Localization(__file__, 286, 7), len_715, *[w_716], **kwargs_717)
    
    int_719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 16), 'int')
    # Applying the binary operator '>' (line 286)
    result_gt_720 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 7), '>', len_call_result_718, int_719)
    
    
    
    # Call to len(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'r' (line 286)
    r_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 26), 'r', False)
    # Processing the call keyword arguments (line 286)
    kwargs_723 = {}
    # Getting the type of 'len' (line 286)
    len_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'len', False)
    # Calling len(args, kwargs) (line 286)
    len_call_result_724 = invoke(stypy.reporting.localization.Localization(__file__, 286, 22), len_721, *[r_722], **kwargs_723)
    
    int_725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 31), 'int')
    # Applying the binary operator '>' (line 286)
    result_gt_726 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 22), '>', len_call_result_724, int_725)
    
    # Applying the binary operator 'and' (line 286)
    result_and_keyword_727 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 7), 'and', result_gt_720, result_gt_726)
    
    # Testing if the type of an if condition is none (line 286)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 286, 4), result_and_keyword_727):
        pass
    else:
        
        # Testing the type of an if condition (line 286)
        if_condition_728 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 4), result_and_keyword_727)
        # Assigning a type to the variable 'if_condition_728' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'if_condition_728', if_condition_728)
        # SSA begins for if statement (line 286)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeWarning(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'localization' (line 287)
        localization_730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'localization', False)
        
        # Call to format(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'upper' (line 288)
        upper_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 27), 'upper', False)
        # Processing the call keyword arguments (line 287)
        kwargs_734 = {}
        str_731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 34), 'str', 'Some of the possible types of the upper bound of the slice ({0}) are invalid')
        # Obtaining the member 'format' of a type (line 287)
        format_732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 34), str_731, 'format')
        # Calling format(args, kwargs) (line 287)
        format_call_result_735 = invoke(stypy.reporting.localization.Localization(__file__, 287, 34), format_732, *[upper_733], **kwargs_734)
        
        # Processing the call keyword arguments (line 287)
        kwargs_736 = {}
        # Getting the type of 'TypeWarning' (line 287)
        TypeWarning_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'TypeWarning', False)
        # Calling TypeWarning(args, kwargs) (line 287)
        TypeWarning_call_result_737 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), TypeWarning_729, *[localization_730, format_call_result_735], **kwargs_736)
        
        # SSA join for if statement (line 286)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'w' (line 289)
    w_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 11), 'w', False)
    # Processing the call keyword arguments (line 289)
    kwargs_740 = {}
    # Getting the type of 'len' (line 289)
    len_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 7), 'len', False)
    # Calling len(args, kwargs) (line 289)
    len_call_result_741 = invoke(stypy.reporting.localization.Localization(__file__, 289, 7), len_738, *[w_739], **kwargs_740)
    
    int_742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 16), 'int')
    # Applying the binary operator '>' (line 289)
    result_gt_743 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 7), '>', len_call_result_741, int_742)
    
    
    
    # Call to len(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'r' (line 289)
    r_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'r', False)
    # Processing the call keyword arguments (line 289)
    kwargs_746 = {}
    # Getting the type of 'len' (line 289)
    len_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 22), 'len', False)
    # Calling len(args, kwargs) (line 289)
    len_call_result_747 = invoke(stypy.reporting.localization.Localization(__file__, 289, 22), len_744, *[r_745], **kwargs_746)
    
    int_748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 32), 'int')
    # Applying the binary operator '==' (line 289)
    result_eq_749 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 22), '==', len_call_result_747, int_748)
    
    # Applying the binary operator 'and' (line 289)
    result_and_keyword_750 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 7), 'and', result_gt_743, result_eq_749)
    
    # Testing if the type of an if condition is none (line 289)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 289, 4), result_and_keyword_750):
        pass
    else:
        
        # Testing the type of an if condition (line 289)
        if_condition_751 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 4), result_and_keyword_750)
        # Assigning a type to the variable 'if_condition_751' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'if_condition_751', if_condition_751)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'localization' (line 290)
        localization_753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 18), 'localization', False)
        
        # Call to format(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'upper' (line 290)
        upper_756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 99), 'upper', False)
        # Processing the call keyword arguments (line 290)
        kwargs_757 = {}
        str_754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 32), 'str', 'The type of the upper bound of the slice ({0}) is invalid')
        # Obtaining the member 'format' of a type (line 290)
        format_755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 32), str_754, 'format')
        # Calling format(args, kwargs) (line 290)
        format_call_result_758 = invoke(stypy.reporting.localization.Localization(__file__, 290, 32), format_755, *[upper_756], **kwargs_757)
        
        # Processing the call keyword arguments (line 290)
        kwargs_759 = {}
        # Getting the type of 'TypeError' (line 290)
        TypeError_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 290)
        TypeError_call_result_760 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), TypeError_752, *[localization_753, format_call_result_758], **kwargs_759)
        
        
        # Assigning a Name to a Name (line 291):
        
        # Assigning a Name to a Name (line 291):
        # Getting the type of 'True' (line 291)
        True_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'True')
        # Assigning a type to the variable 'error' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'error', True_761)
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Tuple (line 293):
    
    # Assigning a Call to a Name:
    
    # Call to __slice_bounds_checking(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'step' (line 293)
    step_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 35), 'step', False)
    # Processing the call keyword arguments (line 293)
    kwargs_764 = {}
    # Getting the type of '__slice_bounds_checking' (line 293)
    slice_bounds_checking_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 11), '__slice_bounds_checking', False)
    # Calling __slice_bounds_checking(args, kwargs) (line 293)
    slice_bounds_checking_call_result_765 = invoke(stypy.reporting.localization.Localization(__file__, 293, 11), slice_bounds_checking_762, *[step_763], **kwargs_764)
    
    # Assigning a type to the variable 'call_assignment_299' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_299', slice_bounds_checking_call_result_765)
    
    # Assigning a Call to a Name (line 293):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_299' (line 293)
    call_assignment_299_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_299', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_767 = stypy_get_value_from_tuple(call_assignment_299_766, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_300' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_300', stypy_get_value_from_tuple_call_result_767)
    
    # Assigning a Name to a Name (line 293):
    # Getting the type of 'call_assignment_300' (line 293)
    call_assignment_300_768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_300')
    # Assigning a type to the variable 'r' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'r', call_assignment_300_768)
    
    # Assigning a Call to a Name (line 293):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_299' (line 293)
    call_assignment_299_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_299', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_770 = stypy_get_value_from_tuple(call_assignment_299_769, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_301' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_301', stypy_get_value_from_tuple_call_result_770)
    
    # Assigning a Name to a Name (line 293):
    # Getting the type of 'call_assignment_301' (line 293)
    call_assignment_301_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_301')
    # Assigning a type to the variable 'w' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 7), 'w', call_assignment_301_771)
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'w' (line 294)
    w_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 11), 'w', False)
    # Processing the call keyword arguments (line 294)
    kwargs_774 = {}
    # Getting the type of 'len' (line 294)
    len_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 7), 'len', False)
    # Calling len(args, kwargs) (line 294)
    len_call_result_775 = invoke(stypy.reporting.localization.Localization(__file__, 294, 7), len_772, *[w_773], **kwargs_774)
    
    int_776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 16), 'int')
    # Applying the binary operator '>' (line 294)
    result_gt_777 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 7), '>', len_call_result_775, int_776)
    
    
    
    # Call to len(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'r' (line 294)
    r_779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 26), 'r', False)
    # Processing the call keyword arguments (line 294)
    kwargs_780 = {}
    # Getting the type of 'len' (line 294)
    len_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'len', False)
    # Calling len(args, kwargs) (line 294)
    len_call_result_781 = invoke(stypy.reporting.localization.Localization(__file__, 294, 22), len_778, *[r_779], **kwargs_780)
    
    int_782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 31), 'int')
    # Applying the binary operator '>' (line 294)
    result_gt_783 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 22), '>', len_call_result_781, int_782)
    
    # Applying the binary operator 'and' (line 294)
    result_and_keyword_784 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 7), 'and', result_gt_777, result_gt_783)
    
    # Testing if the type of an if condition is none (line 294)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 294, 4), result_and_keyword_784):
        pass
    else:
        
        # Testing the type of an if condition (line 294)
        if_condition_785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 4), result_and_keyword_784)
        # Assigning a type to the variable 'if_condition_785' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'if_condition_785', if_condition_785)
        # SSA begins for if statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeWarning(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'localization' (line 295)
        localization_787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'localization', False)
        
        # Call to format(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'step' (line 296)
        step_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 27), 'step', False)
        # Processing the call keyword arguments (line 295)
        kwargs_791 = {}
        str_788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 34), 'str', 'Some of the possible types of the step of the slice ({0}) are invalid')
        # Obtaining the member 'format' of a type (line 295)
        format_789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 34), str_788, 'format')
        # Calling format(args, kwargs) (line 295)
        format_call_result_792 = invoke(stypy.reporting.localization.Localization(__file__, 295, 34), format_789, *[step_790], **kwargs_791)
        
        # Processing the call keyword arguments (line 295)
        kwargs_793 = {}
        # Getting the type of 'TypeWarning' (line 295)
        TypeWarning_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'TypeWarning', False)
        # Calling TypeWarning(args, kwargs) (line 295)
        TypeWarning_call_result_794 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), TypeWarning_786, *[localization_787, format_call_result_792], **kwargs_793)
        
        # SSA join for if statement (line 294)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'w' (line 297)
    w_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 11), 'w', False)
    # Processing the call keyword arguments (line 297)
    kwargs_797 = {}
    # Getting the type of 'len' (line 297)
    len_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 7), 'len', False)
    # Calling len(args, kwargs) (line 297)
    len_call_result_798 = invoke(stypy.reporting.localization.Localization(__file__, 297, 7), len_795, *[w_796], **kwargs_797)
    
    int_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 16), 'int')
    # Applying the binary operator '>' (line 297)
    result_gt_800 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 7), '>', len_call_result_798, int_799)
    
    
    
    # Call to len(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'r' (line 297)
    r_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 26), 'r', False)
    # Processing the call keyword arguments (line 297)
    kwargs_803 = {}
    # Getting the type of 'len' (line 297)
    len_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 22), 'len', False)
    # Calling len(args, kwargs) (line 297)
    len_call_result_804 = invoke(stypy.reporting.localization.Localization(__file__, 297, 22), len_801, *[r_802], **kwargs_803)
    
    int_805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 32), 'int')
    # Applying the binary operator '==' (line 297)
    result_eq_806 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 22), '==', len_call_result_804, int_805)
    
    # Applying the binary operator 'and' (line 297)
    result_and_keyword_807 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 7), 'and', result_gt_800, result_eq_806)
    
    # Testing if the type of an if condition is none (line 297)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 297, 4), result_and_keyword_807):
        pass
    else:
        
        # Testing the type of an if condition (line 297)
        if_condition_808 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 4), result_and_keyword_807)
        # Assigning a type to the variable 'if_condition_808' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'if_condition_808', if_condition_808)
        # SSA begins for if statement (line 297)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'localization' (line 298)
        localization_810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 18), 'localization', False)
        
        # Call to format(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'step' (line 298)
        step_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 92), 'step', False)
        # Processing the call keyword arguments (line 298)
        kwargs_814 = {}
        str_811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 32), 'str', 'The type of the step of the slice ({0}) is invalid')
        # Obtaining the member 'format' of a type (line 298)
        format_812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 32), str_811, 'format')
        # Calling format(args, kwargs) (line 298)
        format_call_result_815 = invoke(stypy.reporting.localization.Localization(__file__, 298, 32), format_812, *[step_813], **kwargs_814)
        
        # Processing the call keyword arguments (line 298)
        kwargs_816 = {}
        # Getting the type of 'TypeError' (line 298)
        TypeError_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 298)
        TypeError_call_result_817 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), TypeError_809, *[localization_810, format_call_result_815], **kwargs_816)
        
        
        # Assigning a Name to a Name (line 299):
        
        # Assigning a Name to a Name (line 299):
        # Getting the type of 'True' (line 299)
        True_818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'True')
        # Assigning a type to the variable 'error' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'error', True_818)
        # SSA join for if statement (line 297)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'error' (line 301)
    error_819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 11), 'error')
    # Applying the 'not' unary operator (line 301)
    result_not__820 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 7), 'not', error_819)
    
    # Testing if the type of an if condition is none (line 301)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 301, 4), result_not__820):
        
        # Call to TypeError(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'localization' (line 304)
        localization_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'localization', False)
        str_829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 39), 'str', 'Type error when specifying slice bounds')
        # Processing the call keyword arguments (line 304)
        # Getting the type of 'False' (line 304)
        False_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 93), 'False', False)
        keyword_831 = False_830
        kwargs_832 = {'prints_msg': keyword_831}
        # Getting the type of 'TypeError' (line 304)
        TypeError_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 304)
        TypeError_call_result_833 = invoke(stypy.reporting.localization.Localization(__file__, 304, 15), TypeError_827, *[localization_828, str_829], **kwargs_832)
        
        # Assigning a type to the variable 'stypy_return_type' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'stypy_return_type', TypeError_call_result_833)
    else:
        
        # Testing the type of an if condition (line 301)
        if_condition_821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 4), result_not__820)
        # Assigning a type to the variable 'if_condition_821' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'if_condition_821', if_condition_821)
        # SSA begins for if statement (line 301)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to get_builtin_type(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'localization' (line 302)
        localization_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 32), 'localization', False)
        str_824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 46), 'str', 'slice')
        # Processing the call keyword arguments (line 302)
        kwargs_825 = {}
        # Getting the type of 'get_builtin_type' (line 302)
        get_builtin_type_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'get_builtin_type', False)
        # Calling get_builtin_type(args, kwargs) (line 302)
        get_builtin_type_call_result_826 = invoke(stypy.reporting.localization.Localization(__file__, 302, 15), get_builtin_type_822, *[localization_823, str_824], **kwargs_825)
        
        # Assigning a type to the variable 'stypy_return_type' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'stypy_return_type', get_builtin_type_call_result_826)
        # SSA branch for the else part of an if statement (line 301)
        module_type_store.open_ssa_branch('else')
        
        # Call to TypeError(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'localization' (line 304)
        localization_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'localization', False)
        str_829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 39), 'str', 'Type error when specifying slice bounds')
        # Processing the call keyword arguments (line 304)
        # Getting the type of 'False' (line 304)
        False_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 93), 'False', False)
        keyword_831 = False_830
        kwargs_832 = {'prints_msg': keyword_831}
        # Getting the type of 'TypeError' (line 304)
        TypeError_827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 304)
        TypeError_call_result_833 = invoke(stypy.reporting.localization.Localization(__file__, 304, 15), TypeError_827, *[localization_828, str_829], **kwargs_832)
        
        # Assigning a type to the variable 'stypy_return_type' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'stypy_return_type', TypeError_call_result_833)
        # SSA join for if statement (line 301)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'ensure_slice_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ensure_slice_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 265)
    stypy_return_type_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_834)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ensure_slice_bounds'
    return stypy_return_type_834

# Assigning a type to the variable 'ensure_slice_bounds' (line 265)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), 'ensure_slice_bounds', ensure_slice_bounds)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
