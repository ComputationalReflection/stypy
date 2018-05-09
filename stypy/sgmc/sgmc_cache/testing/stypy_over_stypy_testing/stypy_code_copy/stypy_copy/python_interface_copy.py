
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import sys
2: 
3: from stypy_copy.errors_copy.type_error_copy import TypeError
4: from stypy_copy.errors_copy.type_warning_copy import TypeWarning
5: from stypy_copy.errors_copy.unsupported_features_copy import create_unsupported_python_feature_message
6: from stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy import *
7: from stypy_copy.python_lib_copy.module_imports_copy import python_imports_copy
8: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
9: from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions
10: from stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy import type_group_generator_copy, type_groups_copy
11: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
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

# 'from stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_16 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_error_copy')

if (type(import_16) is not StypyTypeError):

    if (import_16 != 'pyd_module'):
        __import__(import_16)
        sys_modules_17 = sys.modules[import_16]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_error_copy', sys_modules_17.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_17, sys_modules_17.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_error_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_error_copy', import_16)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from stypy_copy.errors_copy.type_warning_copy import TypeWarning' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_18 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_warning_copy')

if (type(import_18) is not StypyTypeError):

    if (import_18 != 'pyd_module'):
        __import__(import_18)
        sys_modules_19 = sys.modules[import_18]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_warning_copy', sys_modules_19.module_type_store, module_type_store, ['TypeWarning'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_19, sys_modules_19.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_warning_copy import TypeWarning

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_warning_copy', None, module_type_store, ['TypeWarning'], [TypeWarning])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_warning_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_warning_copy', import_18)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from stypy_copy.errors_copy.unsupported_features_copy import create_unsupported_python_feature_message' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_20 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.errors_copy.unsupported_features_copy')

if (type(import_20) is not StypyTypeError):

    if (import_20 != 'pyd_module'):
        __import__(import_20)
        sys_modules_21 = sys.modules[import_20]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.errors_copy.unsupported_features_copy', sys_modules_21.module_type_store, module_type_store, ['create_unsupported_python_feature_message'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_21, sys_modules_21.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.unsupported_features_copy import create_unsupported_python_feature_message

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.errors_copy.unsupported_features_copy', None, module_type_store, ['create_unsupported_python_feature_message'], [create_unsupported_python_feature_message])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.unsupported_features_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.errors_copy.unsupported_features_copy', import_20)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy import ' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_22 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy')

if (type(import_22) is not StypyTypeError):

    if (import_22 != 'pyd_module'):
        __import__(import_22)
        sys_modules_23 = sys.modules[import_22]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy', sys_modules_23.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_23, sys_modules_23.module_type_store, module_type_store)
    else:
        from stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.code_generation_copy.type_inference_programs_copy.python_operators_copy', import_22)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from stypy_copy.python_lib_copy.module_imports_copy import python_imports_copy' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_24 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.python_lib_copy.module_imports_copy')

if (type(import_24) is not StypyTypeError):

    if (import_24 != 'pyd_module'):
        __import__(import_24)
        sys_modules_25 = sys.modules[import_24]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.python_lib_copy.module_imports_copy', sys_modules_25.module_type_store, module_type_store, ['python_imports_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_25, sys_modules_25.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.module_imports_copy import python_imports_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.python_lib_copy.module_imports_copy', None, module_type_store, ['python_imports_copy'], [python_imports_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.module_imports_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.python_lib_copy.module_imports_copy', import_24)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_26 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy')

if (type(import_26) is not StypyTypeError):

    if (import_26 != 'pyd_module'):
        __import__(import_26)
        sys_modules_27 = sys.modules[import_26]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', sys_modules_27.module_type_store, module_type_store, ['UndefinedType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_27, sys_modules_27.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', None, module_type_store, ['UndefinedType'], [UndefinedType])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', import_26)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_28 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy')

if (type(import_28) is not StypyTypeError):

    if (import_28 != 'pyd_module'):
        __import__(import_28)
        sys_modules_29 = sys.modules[import_28]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', sys_modules_29.module_type_store, module_type_store, ['ExtraTypeDefinitions'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_29, sys_modules_29.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ExtraTypeDefinitions

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', None, module_type_store, ['ExtraTypeDefinitions'], [ExtraTypeDefinitions])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', import_28)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy import type_group_generator_copy, type_groups_copy' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_30 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy')

if (type(import_30) is not StypyTypeError):

    if (import_30 != 'pyd_module'):
        __import__(import_30)
        sys_modules_31 = sys.modules[import_30]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy', sys_modules_31.module_type_store, module_type_store, ['type_group_generator_copy', 'type_groups_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_31, sys_modules_31.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy import type_group_generator_copy, type_groups_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy', None, module_type_store, ['type_group_generator_copy', 'type_groups_copy'], [type_group_generator_copy, type_groups_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.python_lib_copy.type_rules_copy.type_groups_copy', import_30)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 11)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')
import_32 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_32) is not StypyTypeError):

    if (import_32 != 'pyd_module'):
        __import__(import_32)
        sys_modules_33 = sys.modules[import_32]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_33.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_33, sys_modules_33.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_32)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/')

str_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\nThis file contains the stypy API that can be called inside the type inference generated programs source code.\nThese functions will be used to interact with stypy, extract type information and other necessary operations when\ngenerating type inference code.\n')
str_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, (-1)), 'str', '\nAn object containing the Python __builtins__ module, containing the type inference functions for each Python builtin\n')

# Assigning a Call to a Name (line 22):

# Assigning a Call to a Name (line 22):

# Call to get_module_from_sys_cache(...): (line 22)
# Processing the call arguments (line 22)
str_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 63), 'str', '__builtin__')
# Processing the call keyword arguments (line 22)
kwargs_39 = {}
# Getting the type of 'python_imports_copy' (line 22)
python_imports_copy_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'python_imports_copy', False)
# Obtaining the member 'get_module_from_sys_cache' of a type (line 22)
get_module_from_sys_cache_37 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), python_imports_copy_36, 'get_module_from_sys_cache')
# Calling get_module_from_sys_cache(args, kwargs) (line 22)
get_module_from_sys_cache_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), get_module_from_sys_cache_37, *[str_38], **kwargs_39)

# Assigning a type to the variable 'builtin_module' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'builtin_module', get_module_from_sys_cache_call_result_40)

@norecursion
def get_builtin_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'UndefinedType' (line 25)
    UndefinedType_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 52), 'UndefinedType')
    defaults = [UndefinedType_41]
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

    str_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', '\n    Obtains a Python builtin type instance to represent the type of an object in Python. Optionally, a value for\n    this object can be specified. Values for objects are not much used within the current version of stypy, but\n    they are stored for future enhancements. Currently, values, if present, are taken into account for the hasattr,\n    setattr and getattr builtin functions.\n\n    :param localization: Caller information\n    :param type_name: Name of the Python type to be created ("int", "float"...)\n    :param value: Optional value for this type. Value must be of the speficied type. The function does not check this.\n    :return: A TypeInferenceProxy representing the specified type or a TypeError if the specified type do not exist\n    ')
    
    str_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 7), 'str', 'NoneType')
    # Getting the type of 'type_name' (line 39)
    type_name_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'type_name')
    # Applying the binary operator 'in' (line 39)
    result_contains_45 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 7), 'in', str_43, type_name_44)
    
    # Testing if the type of an if condition is none (line 39)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 39, 4), result_contains_45):
        pass
    else:
        
        # Testing the type of an if condition (line 39)
        if_condition_46 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 4), result_contains_45)
        # Assigning a type to the variable 'if_condition_46' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'if_condition_46', if_condition_46)
        # SSA begins for if statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to import_from(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'localization' (line 40)
        localization_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 47), 'localization', False)
        str_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 61), 'str', 'None')
        # Processing the call keyword arguments (line 40)
        kwargs_51 = {}
        # Getting the type of 'python_imports_copy' (line 40)
        python_imports_copy_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'python_imports_copy', False)
        # Obtaining the member 'import_from' of a type (line 40)
        import_from_48 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), python_imports_copy_47, 'import_from')
        # Calling import_from(args, kwargs) (line 40)
        import_from_call_result_52 = invoke(stypy.reporting.localization.Localization(__file__, 40, 15), import_from_48, *[localization_49, str_50], **kwargs_51)
        
        # Assigning a type to the variable 'stypy_return_type' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'stypy_return_type', import_from_call_result_52)
        # SSA join for if statement (line 39)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to hasattr(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'ExtraTypeDefinitions' (line 44)
    ExtraTypeDefinitions_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'ExtraTypeDefinitions', False)
    # Getting the type of 'type_name' (line 44)
    type_name_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 37), 'type_name', False)
    # Processing the call keyword arguments (line 44)
    kwargs_56 = {}
    # Getting the type of 'hasattr' (line 44)
    hasattr_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 7), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 44)
    hasattr_call_result_57 = invoke(stypy.reporting.localization.Localization(__file__, 44, 7), hasattr_53, *[ExtraTypeDefinitions_54, type_name_55], **kwargs_56)
    
    # Testing if the type of an if condition is none (line 44)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 4), hasattr_call_result_57):
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to get_type_of_member(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'localization' (line 48)
        localization_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 53), 'localization', False)
        # Getting the type of 'type_name' (line 48)
        type_name_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 67), 'type_name', False)
        # Processing the call keyword arguments (line 48)
        kwargs_68 = {}
        # Getting the type of 'builtin_module' (line 48)
        builtin_module_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'builtin_module', False)
        # Obtaining the member 'get_type_of_member' of a type (line 48)
        get_type_of_member_65 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), builtin_module_64, 'get_type_of_member')
        # Calling get_type_of_member(args, kwargs) (line 48)
        get_type_of_member_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), get_type_of_member_65, *[localization_66, type_name_67], **kwargs_68)
        
        # Assigning a type to the variable 'ret_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'ret_type', get_type_of_member_call_result_69)
    else:
        
        # Testing the type of an if condition (line 44)
        if_condition_58 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 4), hasattr_call_result_57)
        # Assigning a type to the variable 'if_condition_58' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'if_condition_58', if_condition_58)
        # SSA begins for if statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 45):
        
        # Assigning a Call to a Name (line 45):
        
        # Call to getattr(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'ExtraTypeDefinitions' (line 45)
        ExtraTypeDefinitions_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'ExtraTypeDefinitions', False)
        # Getting the type of 'type_name' (line 45)
        type_name_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 49), 'type_name', False)
        # Processing the call keyword arguments (line 45)
        kwargs_62 = {}
        # Getting the type of 'getattr' (line 45)
        getattr_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'getattr', False)
        # Calling getattr(args, kwargs) (line 45)
        getattr_call_result_63 = invoke(stypy.reporting.localization.Localization(__file__, 45, 19), getattr_59, *[ExtraTypeDefinitions_60, type_name_61], **kwargs_62)
        
        # Assigning a type to the variable 'ret_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'ret_type', getattr_call_result_63)
        # SSA branch for the else part of an if statement (line 44)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to get_type_of_member(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'localization' (line 48)
        localization_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 53), 'localization', False)
        # Getting the type of 'type_name' (line 48)
        type_name_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 67), 'type_name', False)
        # Processing the call keyword arguments (line 48)
        kwargs_68 = {}
        # Getting the type of 'builtin_module' (line 48)
        builtin_module_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'builtin_module', False)
        # Obtaining the member 'get_type_of_member' of a type (line 48)
        get_type_of_member_65 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), builtin_module_64, 'get_type_of_member')
        # Calling get_type_of_member(args, kwargs) (line 48)
        get_type_of_member_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), get_type_of_member_65, *[localization_66, type_name_67], **kwargs_68)
        
        # Assigning a type to the variable 'ret_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'ret_type', get_type_of_member_call_result_69)
        # SSA join for if statement (line 44)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to set_type_instance(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'True' (line 51)
    True_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'True', False)
    # Processing the call keyword arguments (line 51)
    kwargs_73 = {}
    # Getting the type of 'ret_type' (line 51)
    ret_type_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'ret_type', False)
    # Obtaining the member 'set_type_instance' of a type (line 51)
    set_type_instance_71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), ret_type_70, 'set_type_instance')
    # Calling set_type_instance(args, kwargs) (line 51)
    set_type_instance_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), set_type_instance_71, *[True_72], **kwargs_73)
    
    
    # Getting the type of 'value' (line 54)
    value_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 7), 'value')
    # Getting the type of 'UndefinedType' (line 54)
    UndefinedType_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'UndefinedType')
    # Applying the binary operator 'isnot' (line 54)
    result_is_not_77 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), 'isnot', value_75, UndefinedType_76)
    
    # Testing if the type of an if condition is none (line 54)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 54, 4), result_is_not_77):
        pass
    else:
        
        # Testing the type of an if condition (line 54)
        if_condition_78 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 4), result_is_not_77)
        # Assigning a type to the variable 'if_condition_78' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'if_condition_78', if_condition_78)
        # SSA begins for if statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_value(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'value' (line 55)
        value_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'value', False)
        # Processing the call keyword arguments (line 55)
        kwargs_82 = {}
        # Getting the type of 'ret_type' (line 55)
        ret_type_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'ret_type', False)
        # Obtaining the member 'set_value' of a type (line 55)
        set_value_80 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), ret_type_79, 'set_value')
        # Calling set_value(args, kwargs) (line 55)
        set_value_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), set_value_80, *[value_81], **kwargs_82)
        
        # SSA join for if statement (line 54)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'ret_type' (line 57)
    ret_type_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'ret_type')
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type', ret_type_84)
    
    # ################# End of 'get_builtin_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_builtin_type' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_85)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_builtin_type'
    return stypy_return_type_85

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

    str_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, (-1)), 'str', '\n    This function can obtain any type name for any Python module that have it declared. This way we can access\n    non-builtin types such as those declared on the time module and so on, provided they exist within the specified\n    module\n    :param localization: Caller information\n    :param module_name: Module name\n    :param type_name: Type name within the module\n    :return: A TypeInferenceProxy for the specified type or a TypeError if the requested type do not exist\n    ')
    
    # Assigning a Call to a Name (line 71):
    
    # Assigning a Call to a Name (line 71):
    
    # Call to import_python_module(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'localization' (line 71)
    localization_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 54), 'localization', False)
    # Getting the type of 'module_name' (line 71)
    module_name_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 68), 'module_name', False)
    # Processing the call keyword arguments (line 71)
    kwargs_91 = {}
    # Getting the type of 'python_imports_copy' (line 71)
    python_imports_copy_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 13), 'python_imports_copy', False)
    # Obtaining the member 'import_python_module' of a type (line 71)
    import_python_module_88 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 13), python_imports_copy_87, 'import_python_module')
    # Calling import_python_module(args, kwargs) (line 71)
    import_python_module_call_result_92 = invoke(stypy.reporting.localization.Localization(__file__, 71, 13), import_python_module_88, *[localization_89, module_name_90], **kwargs_91)
    
    # Assigning a type to the variable 'module' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'module', import_python_module_call_result_92)
    
    # Type idiom detected: calculating its left and rigth part (line 72)
    # Getting the type of 'TypeError' (line 72)
    TypeError_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'TypeError')
    # Getting the type of 'module' (line 72)
    module_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 18), 'module')
    
    (may_be_95, more_types_in_union_96) = may_be_subtype(TypeError_93, module_94)

    if may_be_95:

        if more_types_in_union_96:
            # Runtime conditional SSA (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'module' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'module', remove_not_subtype_from_union(module_94, TypeError))
        # Getting the type of 'module' (line 73)
        module_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'module')
        # Assigning a type to the variable 'stypy_return_type' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'stypy_return_type', module_97)

        if more_types_in_union_96:
            # SSA join for if statement (line 72)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to get_type_of_member(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'localization' (line 75)
    localization_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 37), 'localization', False)
    # Getting the type of 'type_name' (line 75)
    type_name_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 51), 'type_name', False)
    # Processing the call keyword arguments (line 75)
    kwargs_102 = {}
    # Getting the type of 'module' (line 75)
    module_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'module', False)
    # Obtaining the member 'get_type_of_member' of a type (line 75)
    get_type_of_member_99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 11), module_98, 'get_type_of_member')
    # Calling get_type_of_member(args, kwargs) (line 75)
    get_type_of_member_call_result_103 = invoke(stypy.reporting.localization.Localization(__file__, 75, 11), get_type_of_member_99, *[localization_100, type_name_101], **kwargs_102)
    
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type', get_type_of_member_call_result_103)
    
    # ################# End of 'get_python_api_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_python_api_type' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_104)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_python_api_type'
    return stypy_return_type_104

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

    str_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, (-1)), 'str', '\n    This function imports all the declared public members of a user-defined or Python library module into the specified\n    type store\n    It modules the from <module> import <element1>, <element2>, ... or * sentences and also the import <module> sentence\n    :param localization: Caller information\n    :param main_module_path: Path of the module to import, i. e. path of the .py file of the module\n    :param imported_module_name: Name of the module\n    :param dest_type_store: Type store to add the module elements\n    :param elements: A variable list of arguments with the elements to import. The value \'*\' means all elements. No\n    value models the "import <module>" sentence\n    :return: None or a TypeError if the requested type do not exist\n    ')
    
    # Call to import_elements_from_external_module(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'localization' (line 92)
    localization_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 68), 'localization', False)
    # Getting the type of 'imported_module_name' (line 92)
    imported_module_name_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 82), 'imported_module_name', False)
    # Getting the type of 'dest_type_store' (line 93)
    dest_type_store_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 63), 'dest_type_store', False)
    # Getting the type of 'sys' (line 93)
    sys_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 80), 'sys', False)
    # Obtaining the member 'path' of a type (line 93)
    path_112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 80), sys_111, 'path')
    # Getting the type of 'elements' (line 94)
    elements_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 64), 'elements', False)
    # Processing the call keyword arguments (line 92)
    kwargs_114 = {}
    # Getting the type of 'python_imports_copy' (line 92)
    python_imports_copy_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'python_imports_copy', False)
    # Obtaining the member 'import_elements_from_external_module' of a type (line 92)
    import_elements_from_external_module_107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 11), python_imports_copy_106, 'import_elements_from_external_module')
    # Calling import_elements_from_external_module(args, kwargs) (line 92)
    import_elements_from_external_module_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 92, 11), import_elements_from_external_module_107, *[localization_108, imported_module_name_109, dest_type_store_110, path_112, elements_113], **kwargs_114)
    
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', import_elements_from_external_module_call_result_115)
    
    # ################# End of 'import_elements_from_external_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'import_elements_from_external_module' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_116)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'import_elements_from_external_module'
    return stypy_return_type_116

# Assigning a type to the variable 'import_elements_from_external_module' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'import_elements_from_external_module', import_elements_from_external_module)

@norecursion
def import_from(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 55), 'str', '__builtin__')
    defaults = [str_117]
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

    str_118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, (-1)), 'str', '\n    Imports a single member from a module. If no module is specified, the builtin module is used instead. Models the\n    "from <module> import <member>" sentence, being a sort version of the import_elements_from_external_module function\n    but only for Python library modules\n    :param localization: Caller information\n    :param member_name: Member to import\n    :param module_name: Python library module that contains the member or nothing to use the __builtins__ module\n    :return: A TypeInferenceProxy for the specified member or a TypeError if the requested element do not exist\n    ')
    
    # Call to import_from(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 'localization' (line 107)
    localization_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 43), 'localization', False)
    # Getting the type of 'member_name' (line 107)
    member_name_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 57), 'member_name', False)
    # Getting the type of 'module_name' (line 107)
    module_name_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 70), 'module_name', False)
    # Processing the call keyword arguments (line 107)
    kwargs_124 = {}
    # Getting the type of 'python_imports_copy' (line 107)
    python_imports_copy_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'python_imports_copy', False)
    # Obtaining the member 'import_from' of a type (line 107)
    import_from_120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 11), python_imports_copy_119, 'import_from')
    # Calling import_from(args, kwargs) (line 107)
    import_from_call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 107, 11), import_from_120, *[localization_121, member_name_122, module_name_123], **kwargs_124)
    
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type', import_from_call_result_125)
    
    # ################# End of 'import_from(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'import_from' in the type store
    # Getting the type of 'stypy_return_type' (line 97)
    stypy_return_type_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'import_from'
    return stypy_return_type_126

# Assigning a type to the variable 'import_from' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'import_from', import_from)

@norecursion
def import_module(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 44), 'str', '__builtin__')
    defaults = [str_127]
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

    str_128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, (-1)), 'str', '\n    Import a full Python library module (models the "import <module>" sentence for Python library modules\n    :param localization: Caller information\n    :param module_name: Module to import\n    :return: A TypeInferenceProxy for the specified module or a TypeError if the requested module do not exist\n    ')
    
    # Call to import_python_module(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'localization' (line 117)
    localization_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 52), 'localization', False)
    # Getting the type of 'module_name' (line 117)
    module_name_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 66), 'module_name', False)
    # Processing the call keyword arguments (line 117)
    kwargs_133 = {}
    # Getting the type of 'python_imports_copy' (line 117)
    python_imports_copy_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 11), 'python_imports_copy', False)
    # Obtaining the member 'import_python_module' of a type (line 117)
    import_python_module_130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 11), python_imports_copy_129, 'import_python_module')
    # Calling import_python_module(args, kwargs) (line 117)
    import_python_module_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 117, 11), import_python_module_130, *[localization_131, module_name_132], **kwargs_133)
    
    # Assigning a type to the variable 'stypy_return_type' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type', import_python_module_call_result_134)
    
    # ################# End of 'import_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'import_module' in the type store
    # Getting the type of 'stypy_return_type' (line 110)
    stypy_return_type_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_135)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'import_module'
    return stypy_return_type_135

# Assigning a type to the variable 'import_module' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'import_module', import_module)

# Assigning a Name to a Name (line 124):

# Assigning a Name to a Name (line 124):
# Getting the type of 'None' (line 124)
None_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'None')
# Assigning a type to the variable 'builtin_operators_module' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'builtin_operators_module', None_136)

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

    str_137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, (-1)), 'str', '\n    Loads the builtin Python operators logic that model the Python operator behavior, as a clone of the "operator"\n    Python library module, that initially holds the same behavior for each operator. Once initially loaded, this logic\n    cannot be altered (in Python we cannot overload the \'+\' operator behavior for builtin types, but we can modify the\n    behavior of the equivalent operator.add function).\n    :return: The behavior of the Python operators\n    ')
    # Marking variables as global (line 135)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 135, 4), 'builtin_operators_module')
    
    # Type idiom detected: calculating its left and rigth part (line 138)
    # Getting the type of 'builtin_operators_module' (line 138)
    builtin_operators_module_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 7), 'builtin_operators_module')
    # Getting the type of 'None' (line 138)
    None_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 35), 'None')
    
    (may_be_140, more_types_in_union_141) = may_be_none(builtin_operators_module_138, None_139)

    if may_be_140:

        if more_types_in_union_141:
            # Runtime conditional SSA (line 138)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to import_python_module(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'None' (line 139)
        None_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 67), 'None', False)
        str_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 73), 'str', 'operator')
        # Processing the call keyword arguments (line 139)
        kwargs_146 = {}
        # Getting the type of 'python_imports_copy' (line 139)
        python_imports_copy_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 26), 'python_imports_copy', False)
        # Obtaining the member 'import_python_module' of a type (line 139)
        import_python_module_143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 26), python_imports_copy_142, 'import_python_module')
        # Calling import_python_module(args, kwargs) (line 139)
        import_python_module_call_result_147 = invoke(stypy.reporting.localization.Localization(__file__, 139, 26), import_python_module_143, *[None_144, str_145], **kwargs_146)
        
        # Assigning a type to the variable 'operator_module' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'operator_module', import_python_module_call_result_147)
        
        # Assigning a Call to a Name (line 140):
        
        # Assigning a Call to a Name (line 140):
        
        # Call to clone(...): (line 140)
        # Processing the call keyword arguments (line 140)
        kwargs_150 = {}
        # Getting the type of 'operator_module' (line 140)
        operator_module_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 35), 'operator_module', False)
        # Obtaining the member 'clone' of a type (line 140)
        clone_149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 35), operator_module_148, 'clone')
        # Calling clone(args, kwargs) (line 140)
        clone_call_result_151 = invoke(stypy.reporting.localization.Localization(__file__, 140, 35), clone_149, *[], **kwargs_150)
        
        # Assigning a type to the variable 'builtin_operators_module' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'builtin_operators_module', clone_call_result_151)
        
        # Assigning a Str to a Attribute (line 141):
        
        # Assigning a Str to a Attribute (line 141):
        str_152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 40), 'str', 'builtin_operators')
        # Getting the type of 'builtin_operators_module' (line 141)
        builtin_operators_module_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'builtin_operators_module')
        # Setting the type of the member 'name' of a type (line 141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), builtin_operators_module_153, 'name', str_152)

        if more_types_in_union_141:
            # SSA join for if statement (line 138)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'builtin_operators_module' (line 143)
    builtin_operators_module_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'builtin_operators_module')
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type', builtin_operators_module_154)
    
    # ################# End of 'load_builtin_operators_module(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'load_builtin_operators_module' in the type store
    # Getting the type of 'stypy_return_type' (line 127)
    stypy_return_type_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_155)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'load_builtin_operators_module'
    return stypy_return_type_155

# Assigning a type to the variable 'load_builtin_operators_module' (line 127)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'load_builtin_operators_module', load_builtin_operators_module)

# Assigning a List to a Name (line 146):

# Assigning a List to a Name (line 146):

# Obtaining an instance of the builtin type 'list' (line 146)
list_156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 32), 'list')
# Adding type elements to the builtin type 'list' instance (line 146)
# Adding element type (line 146)

# Obtaining an instance of the builtin type 'tuple' (line 147)
tuple_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 5), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 147)
# Adding element type (line 147)

# Obtaining an instance of the builtin type 'list' (line 147)
list_158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 147)
# Adding element type (line 147)
str_159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 6), 'str', 'lt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), list_158, str_159)
# Adding element type (line 147)
str_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 12), 'str', 'gt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), list_158, str_160)
# Adding element type (line 147)
str_161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 18), 'str', 'lte')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), list_158, str_161)
# Adding element type (line 147)
str_162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 25), 'str', 'gte')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), list_158, str_162)
# Adding element type (line 147)
str_163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 32), 'str', 'le')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), list_158, str_163)
# Adding element type (line 147)
str_164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 38), 'str', 'ge')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), list_158, str_164)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), tuple_157, list_158)
# Adding element type (line 147)
# Getting the type of 'type_group_generator_copy' (line 147)
type_group_generator_copy_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 45), 'type_group_generator_copy')
# Obtaining the member 'Integer' of a type (line 147)
Integer_166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 45), type_group_generator_copy_165, 'Integer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 5), tuple_157, Integer_166)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 32), list_156, tuple_157)

# Assigning a type to the variable 'forced_operator_result_checks' (line 146)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 0), 'forced_operator_result_checks', list_156)

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

    str_167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, (-1)), 'str', "\n    Handles all the invokations to Python operators of the type inference program.\n    :param localization: Caller information\n    :param operator_symbol: Operator symbol ('+', '-',...). Symbols instead of operator names ('add', 'sub', ...)\n    are used in the generated type inference program to improve readability\n    :param arguments: Variable list of arguments of the operator\n    :return: Return type of the operator call\n    ")
    # Marking variables as global (line 160)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 160, 4), 'builtin_operators_module')
    
    # Call to load_builtin_operators_module(...): (line 162)
    # Processing the call keyword arguments (line 162)
    kwargs_169 = {}
    # Getting the type of 'load_builtin_operators_module' (line 162)
    load_builtin_operators_module_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'load_builtin_operators_module', False)
    # Calling load_builtin_operators_module(args, kwargs) (line 162)
    load_builtin_operators_module_call_result_170 = invoke(stypy.reporting.localization.Localization(__file__, 162, 4), load_builtin_operators_module_168, *[], **kwargs_169)
    
    
    
    # SSA begins for try-except statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 166):
    
    # Assigning a Call to a Name (line 166):
    
    # Call to operator_symbol_to_name(...): (line 166)
    # Processing the call arguments (line 166)
    # Getting the type of 'operator_symbol' (line 166)
    operator_symbol_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 48), 'operator_symbol', False)
    # Processing the call keyword arguments (line 166)
    kwargs_173 = {}
    # Getting the type of 'operator_symbol_to_name' (line 166)
    operator_symbol_to_name_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'operator_symbol_to_name', False)
    # Calling operator_symbol_to_name(args, kwargs) (line 166)
    operator_symbol_to_name_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 166, 24), operator_symbol_to_name_171, *[operator_symbol_172], **kwargs_173)
    
    # Assigning a type to the variable 'operator_name' (line 166)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'operator_name', operator_symbol_to_name_call_result_174)
    # SSA branch for the except part of a try statement (line 164)
    # SSA branch for the except '<any exception>' branch of a try statement (line 164)
    module_type_store.open_ssa_branch('except')
    
    # Call to TypeError(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'localization' (line 169)
    localization_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 25), 'localization', False)
    
    # Call to format(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'operator_symbol' (line 169)
    operator_symbol_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 75), 'operator_symbol', False)
    # Processing the call keyword arguments (line 169)
    kwargs_180 = {}
    str_177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 39), 'str', 'Unrecognized operator: {0}')
    # Obtaining the member 'format' of a type (line 169)
    format_178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 39), str_177, 'format')
    # Calling format(args, kwargs) (line 169)
    format_call_result_181 = invoke(stypy.reporting.localization.Localization(__file__, 169, 39), format_178, *[operator_symbol_179], **kwargs_180)
    
    # Processing the call keyword arguments (line 169)
    kwargs_182 = {}
    # Getting the type of 'TypeError' (line 169)
    TypeError_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 169)
    TypeError_call_result_183 = invoke(stypy.reporting.localization.Localization(__file__, 169, 15), TypeError_175, *[localization_176, format_call_result_181], **kwargs_182)
    
    # Assigning a type to the variable 'stypy_return_type' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stypy_return_type', TypeError_call_result_183)
    # SSA join for try-except statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 172):
    
    # Assigning a Call to a Name (line 172):
    
    # Call to get_type_of_member(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'localization' (line 172)
    localization_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 64), 'localization', False)
    # Getting the type of 'operator_name' (line 172)
    operator_name_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 78), 'operator_name', False)
    # Processing the call keyword arguments (line 172)
    kwargs_188 = {}
    # Getting the type of 'builtin_operators_module' (line 172)
    builtin_operators_module_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'builtin_operators_module', False)
    # Obtaining the member 'get_type_of_member' of a type (line 172)
    get_type_of_member_185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 20), builtin_operators_module_184, 'get_type_of_member')
    # Calling get_type_of_member(args, kwargs) (line 172)
    get_type_of_member_call_result_189 = invoke(stypy.reporting.localization.Localization(__file__, 172, 20), get_type_of_member_185, *[localization_186, operator_name_187], **kwargs_188)
    
    # Assigning a type to the variable 'operator_call' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'operator_call', get_type_of_member_call_result_189)
    
    # Getting the type of 'operator_name' (line 175)
    operator_name_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 7), 'operator_name')
    str_191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 24), 'str', 'contains')
    # Applying the binary operator '==' (line 175)
    result_eq_192 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 7), '==', operator_name_190, str_191)
    
    # Testing if the type of an if condition is none (line 175)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 175, 4), result_eq_192):
        pass
    else:
        
        # Testing the type of an if condition (line 175)
        if_condition_193 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 175, 4), result_eq_192)
        # Assigning a type to the variable 'if_condition_193' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'if_condition_193', if_condition_193)
        # SSA begins for if statement (line 175)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to tuple(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Call to reversed(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'arguments' (line 176)
        arguments_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 35), 'arguments', False)
        # Processing the call keyword arguments (line 176)
        kwargs_197 = {}
        # Getting the type of 'reversed' (line 176)
        reversed_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 26), 'reversed', False)
        # Calling reversed(args, kwargs) (line 176)
        reversed_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 176, 26), reversed_195, *[arguments_196], **kwargs_197)
        
        # Processing the call keyword arguments (line 176)
        kwargs_199 = {}
        # Getting the type of 'tuple' (line 176)
        tuple_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 20), 'tuple', False)
        # Calling tuple(args, kwargs) (line 176)
        tuple_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 176, 20), tuple_194, *[reversed_call_result_198], **kwargs_199)
        
        # Assigning a type to the variable 'arguments' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'arguments', tuple_call_result_200)
        # SSA join for if statement (line 175)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 179):
    
    # Assigning a Call to a Name (line 179):
    
    # Call to invoke(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'localization' (line 179)
    localization_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'localization', False)
    # Getting the type of 'arguments' (line 179)
    arguments_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 49), 'arguments', False)
    # Processing the call keyword arguments (line 179)
    kwargs_205 = {}
    # Getting the type of 'operator_call' (line 179)
    operator_call_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 13), 'operator_call', False)
    # Obtaining the member 'invoke' of a type (line 179)
    invoke_202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 13), operator_call_201, 'invoke')
    # Calling invoke(args, kwargs) (line 179)
    invoke_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 179, 13), invoke_202, *[localization_203, arguments_204], **kwargs_205)
    
    # Assigning a type to the variable 'result' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'result', invoke_call_result_206)
    
    # Getting the type of 'forced_operator_result_checks' (line 180)
    forced_operator_result_checks_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'forced_operator_result_checks')
    # Assigning a type to the variable 'forced_operator_result_checks_207' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'forced_operator_result_checks_207', forced_operator_result_checks_207)
    # Testing if the for loop is going to be iterated (line 180)
    # Testing the type of a for loop iterable (line 180)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 180, 4), forced_operator_result_checks_207)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 180, 4), forced_operator_result_checks_207):
        # Getting the type of the for loop variable (line 180)
        for_loop_var_208 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 180, 4), forced_operator_result_checks_207)
        # Assigning a type to the variable 'check_tuple' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'check_tuple', for_loop_var_208)
        # SSA begins for a for statement (line 180)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'operator_name' (line 181)
        operator_name_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 11), 'operator_name')
        
        # Obtaining the type of the subscript
        int_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 40), 'int')
        # Getting the type of 'check_tuple' (line 181)
        check_tuple_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 28), 'check_tuple')
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 28), check_tuple_211, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 181, 28), getitem___212, int_210)
        
        # Applying the binary operator 'in' (line 181)
        result_contains_214 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 11), 'in', operator_name_209, subscript_call_result_213)
        
        # Testing if the type of an if condition is none (line 181)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 181, 8), result_contains_214):
            pass
        else:
            
            # Testing the type of an if condition (line 181)
            if_condition_215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 181, 8), result_contains_214)
            # Assigning a type to the variable 'if_condition_215' (line 181)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'if_condition_215', if_condition_215)
            # SSA begins for if statement (line 181)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Obtaining the type of the subscript
            int_216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 27), 'int')
            # Getting the type of 'check_tuple' (line 182)
            check_tuple_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'check_tuple')
            # Obtaining the member '__getitem__' of a type (line 182)
            getitem___218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 15), check_tuple_217, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 182)
            subscript_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), getitem___218, int_216)
            
            # Getting the type of 'result' (line 182)
            result_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 33), 'result')
            # Applying the binary operator '==' (line 182)
            result_eq_221 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 15), '==', subscript_call_result_219, result_220)
            
            # Testing if the type of an if condition is none (line 182)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 182, 12), result_eq_221):
                
                # Call to TypeError(...): (line 185)
                # Processing the call arguments (line 185)
                # Getting the type of 'localization' (line 185)
                localization_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 33), 'localization', False)
                
                # Call to format(...): (line 186)
                # Processing the call arguments (line 186)
                # Getting the type of 'operator_name' (line 186)
                operator_name_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 77), 'operator_name', False)
                
                # Obtaining the type of the subscript
                int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 104), 'int')
                # Getting the type of 'check_tuple' (line 186)
                check_tuple_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 92), 'check_tuple', False)
                # Obtaining the member '__getitem__' of a type (line 186)
                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 92), check_tuple_230, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 186, 92), getitem___231, int_229)
                
                # Processing the call keyword arguments (line 186)
                kwargs_233 = {}
                str_226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 33), 'str', 'Operator {0} did not return an {1}')
                # Obtaining the member 'format' of a type (line 186)
                format_227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 33), str_226, 'format')
                # Calling format(args, kwargs) (line 186)
                format_call_result_234 = invoke(stypy.reporting.localization.Localization(__file__, 186, 33), format_227, *[operator_name_228, subscript_call_result_232], **kwargs_233)
                
                # Processing the call keyword arguments (line 185)
                kwargs_235 = {}
                # Getting the type of 'TypeError' (line 185)
                TypeError_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 185)
                TypeError_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 185, 23), TypeError_224, *[localization_225, format_call_result_234], **kwargs_235)
                
                # Assigning a type to the variable 'stypy_return_type' (line 185)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'stypy_return_type', TypeError_call_result_236)
            else:
                
                # Testing the type of an if condition (line 182)
                if_condition_222 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 12), result_eq_221)
                # Assigning a type to the variable 'if_condition_222' (line 182)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'if_condition_222', if_condition_222)
                # SSA begins for if statement (line 182)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'result' (line 183)
                result_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 23), 'result')
                # Assigning a type to the variable 'stypy_return_type' (line 183)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'stypy_return_type', result_223)
                # SSA branch for the else part of an if statement (line 182)
                module_type_store.open_ssa_branch('else')
                
                # Call to TypeError(...): (line 185)
                # Processing the call arguments (line 185)
                # Getting the type of 'localization' (line 185)
                localization_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 33), 'localization', False)
                
                # Call to format(...): (line 186)
                # Processing the call arguments (line 186)
                # Getting the type of 'operator_name' (line 186)
                operator_name_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 77), 'operator_name', False)
                
                # Obtaining the type of the subscript
                int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 104), 'int')
                # Getting the type of 'check_tuple' (line 186)
                check_tuple_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 92), 'check_tuple', False)
                # Obtaining the member '__getitem__' of a type (line 186)
                getitem___231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 92), check_tuple_230, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 186)
                subscript_call_result_232 = invoke(stypy.reporting.localization.Localization(__file__, 186, 92), getitem___231, int_229)
                
                # Processing the call keyword arguments (line 186)
                kwargs_233 = {}
                str_226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 33), 'str', 'Operator {0} did not return an {1}')
                # Obtaining the member 'format' of a type (line 186)
                format_227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 33), str_226, 'format')
                # Calling format(args, kwargs) (line 186)
                format_call_result_234 = invoke(stypy.reporting.localization.Localization(__file__, 186, 33), format_227, *[operator_name_228, subscript_call_result_232], **kwargs_233)
                
                # Processing the call keyword arguments (line 185)
                kwargs_235 = {}
                # Getting the type of 'TypeError' (line 185)
                TypeError_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 23), 'TypeError', False)
                # Calling TypeError(args, kwargs) (line 185)
                TypeError_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 185, 23), TypeError_224, *[localization_225, format_call_result_234], **kwargs_235)
                
                # Assigning a type to the variable 'stypy_return_type' (line 185)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'stypy_return_type', TypeError_call_result_236)
                # SSA join for if statement (line 182)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 181)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'result' (line 187)
    result_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type', result_237)
    
    # ################# End of 'operator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'operator' in the type store
    # Getting the type of 'stypy_return_type' (line 151)
    stypy_return_type_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_238)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'operator'
    return stypy_return_type_238

# Assigning a type to the variable 'operator' (line 151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'operator', operator)

@norecursion
def unsupported_python_feature(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 66), 'str', '')
    defaults = [str_239]
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

    str_240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, (-1)), 'str', '\n    This is called when the type inference program uses not yet supported by stypy Python feature\n    :param localization: Caller information\n    :param feature: Feature name\n    :param description: Message to give to the user\n    :return: A specific TypeError for this kind of problem\n    ')
    
    # Call to create_unsupported_python_feature_message(...): (line 198)
    # Processing the call arguments (line 198)
    # Getting the type of 'localization' (line 198)
    localization_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 46), 'localization', False)
    # Getting the type of 'feature' (line 198)
    feature_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 60), 'feature', False)
    # Getting the type of 'description' (line 198)
    description_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 69), 'description', False)
    # Processing the call keyword arguments (line 198)
    kwargs_245 = {}
    # Getting the type of 'create_unsupported_python_feature_message' (line 198)
    create_unsupported_python_feature_message_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'create_unsupported_python_feature_message', False)
    # Calling create_unsupported_python_feature_message(args, kwargs) (line 198)
    create_unsupported_python_feature_message_call_result_246 = invoke(stypy.reporting.localization.Localization(__file__, 198, 4), create_unsupported_python_feature_message_241, *[localization_242, feature_243, description_244], **kwargs_245)
    
    
    # ################# End of 'unsupported_python_feature(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unsupported_python_feature' in the type store
    # Getting the type of 'stypy_return_type' (line 190)
    stypy_return_type_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_247)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unsupported_python_feature'
    return stypy_return_type_247

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

    str_248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, (-1)), 'str', '\n    This function is used to be sure that an specific var is of one of the specified types. This function is used\n    by type inference programs when a variable must be of a collection of specific types for the program to be\n    correct, which can happen in certain situations such as if conditions or loop tests.\n    :param localization: Caller information\n    :param var: Variable to test (TypeInferenceProxy)\n    :param var_description: Description of the purpose of the tested variable, to show in a potential TypeError\n    :param type_names: Accepted type names\n    :return: None or a TypeError if the variable do not have a suitable type\n    ')
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to get_python_type(...): (line 213)
    # Processing the call keyword arguments (line 213)
    kwargs_251 = {}
    # Getting the type of 'var' (line 213)
    var_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 18), 'var', False)
    # Obtaining the member 'get_python_type' of a type (line 213)
    get_python_type_250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 18), var_249, 'get_python_type')
    # Calling get_python_type(args, kwargs) (line 213)
    get_python_type_call_result_252 = invoke(stypy.reporting.localization.Localization(__file__, 213, 18), get_python_type_250, *[], **kwargs_251)
    
    # Assigning a type to the variable 'python_type' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'python_type', get_python_type_call_result_252)
    
    # Getting the type of 'type_names' (line 214)
    type_names_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'type_names')
    # Assigning a type to the variable 'type_names_253' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'type_names_253', type_names_253)
    # Testing if the for loop is going to be iterated (line 214)
    # Testing the type of a for loop iterable (line 214)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 214, 4), type_names_253)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 214, 4), type_names_253):
        # Getting the type of the for loop variable (line 214)
        for_loop_var_254 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 214, 4), type_names_253)
        # Assigning a type to the variable 'type_name' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'type_name', for_loop_var_254)
        # SSA begins for a for statement (line 214)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'type_name' (line 215)
        type_name_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 11), 'type_name')
        # Getting the type of 'str' (line 215)
        str_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 24), 'str')
        # Applying the binary operator 'is' (line 215)
        result_is__257 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 11), 'is', type_name_255, str_256)
        
        # Testing if the type of an if condition is none (line 215)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 215, 8), result_is__257):
            
            # Assigning a Name to a Name (line 218):
            
            # Assigning a Name to a Name (line 218):
            # Getting the type of 'type_name' (line 218)
            type_name_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'type_name')
            # Assigning a type to the variable 'type_obj' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'type_obj', type_name_265)
        else:
            
            # Testing the type of an if condition (line 215)
            if_condition_258 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), result_is__257)
            # Assigning a type to the variable 'if_condition_258' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_258', if_condition_258)
            # SSA begins for if statement (line 215)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 216):
            
            # Assigning a Call to a Name (line 216):
            
            # Call to eval(...): (line 216)
            # Processing the call arguments (line 216)
            str_260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 28), 'str', 'types.')
            # Getting the type of 'type_name' (line 216)
            type_name_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'type_name', False)
            # Applying the binary operator '+' (line 216)
            result_add_262 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 28), '+', str_260, type_name_261)
            
            # Processing the call keyword arguments (line 216)
            kwargs_263 = {}
            # Getting the type of 'eval' (line 216)
            eval_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 23), 'eval', False)
            # Calling eval(args, kwargs) (line 216)
            eval_call_result_264 = invoke(stypy.reporting.localization.Localization(__file__, 216, 23), eval_259, *[result_add_262], **kwargs_263)
            
            # Assigning a type to the variable 'type_obj' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'type_obj', eval_call_result_264)
            # SSA branch for the else part of an if statement (line 215)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Name (line 218):
            
            # Assigning a Name to a Name (line 218):
            # Getting the type of 'type_name' (line 218)
            type_name_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 23), 'type_name')
            # Assigning a type to the variable 'type_obj' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'type_obj', type_name_265)
            # SSA join for if statement (line 215)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'python_type' (line 220)
        python_type_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 11), 'python_type')
        # Getting the type of 'type_obj' (line 220)
        type_obj_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 26), 'type_obj')
        # Applying the binary operator 'is' (line 220)
        result_is__268 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 11), 'is', python_type_266, type_obj_267)
        
        # Testing if the type of an if condition is none (line 220)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 220, 8), result_is__268):
            pass
        else:
            
            # Testing the type of an if condition (line 220)
            if_condition_269 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 220, 8), result_is__268)
            # Assigning a type to the variable 'if_condition_269' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'if_condition_269', if_condition_269)
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
    localization_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 21), 'localization', False)
    # Getting the type of 'var_description' (line 223)
    var_description_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 35), 'var_description', False)
    str_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 53), 'str', ' must be of one of the following types: ')
    # Applying the binary operator '+' (line 223)
    result_add_274 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 35), '+', var_description_272, str_273)
    
    
    # Call to str(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'type_names' (line 223)
    type_names_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 102), 'type_names', False)
    # Processing the call keyword arguments (line 223)
    kwargs_277 = {}
    # Getting the type of 'str' (line 223)
    str_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 98), 'str', False)
    # Calling str(args, kwargs) (line 223)
    str_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 223, 98), str_275, *[type_names_276], **kwargs_277)
    
    # Applying the binary operator '+' (line 223)
    result_add_279 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 96), '+', result_add_274, str_call_result_278)
    
    # Processing the call keyword arguments (line 223)
    kwargs_280 = {}
    # Getting the type of 'TypeError' (line 223)
    TypeError_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 11), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 223)
    TypeError_call_result_281 = invoke(stypy.reporting.localization.Localization(__file__, 223, 11), TypeError_270, *[localization_271, result_add_279], **kwargs_280)
    
    # Assigning a type to the variable 'stypy_return_type' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'stypy_return_type', TypeError_call_result_281)
    
    # ################# End of 'ensure_var_of_types(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ensure_var_of_types' in the type store
    # Getting the type of 'stypy_return_type' (line 201)
    stypy_return_type_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_282)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ensure_var_of_types'
    return stypy_return_type_282

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

    str_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, (-1)), 'str', '\n    This function is used to make sure that a certain variable has an specific set of members, which may be needed\n    when generating some type inference code that needs an specific structure o a certain object\n    :param localization: Caller information\n    :param var: Variable to test (TypeInferenceProxy)\n    :param var_description: Description of the purpose of the tested variable, to show in a potential TypeError\n    :param member_names: List of members that the type of the variable must have to be valid.\n    :return: None or a TypeError if the variable do not have all passed members\n    ')
    
    # Assigning a Call to a Name (line 236):
    
    # Assigning a Call to a Name (line 236):
    
    # Call to get_python_entity(...): (line 236)
    # Processing the call keyword arguments (line 236)
    kwargs_286 = {}
    # Getting the type of 'var' (line 236)
    var_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 18), 'var', False)
    # Obtaining the member 'get_python_entity' of a type (line 236)
    get_python_entity_285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 18), var_284, 'get_python_entity')
    # Calling get_python_entity(args, kwargs) (line 236)
    get_python_entity_call_result_287 = invoke(stypy.reporting.localization.Localization(__file__, 236, 18), get_python_entity_285, *[], **kwargs_286)
    
    # Assigning a type to the variable 'python_type' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'python_type', get_python_entity_call_result_287)
    
    # Getting the type of 'member_names' (line 237)
    member_names_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 21), 'member_names')
    # Assigning a type to the variable 'member_names_288' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'member_names_288', member_names_288)
    # Testing if the for loop is going to be iterated (line 237)
    # Testing the type of a for loop iterable (line 237)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 237, 4), member_names_288)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 237, 4), member_names_288):
        # Getting the type of the for loop variable (line 237)
        for_loop_var_289 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 237, 4), member_names_288)
        # Assigning a type to the variable 'type_name' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 4), 'type_name', for_loop_var_289)
        # SSA begins for a for statement (line 237)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to hasattr(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'python_type' (line 238)
        python_type_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 23), 'python_type', False)
        # Getting the type of 'type_name' (line 238)
        type_name_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 36), 'type_name', False)
        # Processing the call keyword arguments (line 238)
        kwargs_293 = {}
        # Getting the type of 'hasattr' (line 238)
        hasattr_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 238)
        hasattr_call_result_294 = invoke(stypy.reporting.localization.Localization(__file__, 238, 15), hasattr_290, *[python_type_291, type_name_292], **kwargs_293)
        
        # Applying the 'not' unary operator (line 238)
        result_not__295 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 11), 'not', hasattr_call_result_294)
        
        # Testing if the type of an if condition is none (line 238)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 238, 8), result_not__295):
            pass
        else:
            
            # Testing the type of an if condition (line 238)
            if_condition_296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 238, 8), result_not__295)
            # Assigning a type to the variable 'if_condition_296' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'if_condition_296', if_condition_296)
            # SSA begins for if statement (line 238)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to TypeError(...): (line 239)
            # Processing the call arguments (line 239)
            # Getting the type of 'localization' (line 239)
            localization_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 22), 'localization', False)
            # Getting the type of 'var_description' (line 239)
            var_description_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 36), 'var_description', False)
            str_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 54), 'str', ' must have all of these members: ')
            # Applying the binary operator '+' (line 239)
            result_add_301 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 36), '+', var_description_299, str_300)
            
            
            # Call to str(...): (line 239)
            # Processing the call arguments (line 239)
            # Getting the type of 'member_names' (line 239)
            member_names_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 96), 'member_names', False)
            # Processing the call keyword arguments (line 239)
            kwargs_304 = {}
            # Getting the type of 'str' (line 239)
            str_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 92), 'str', False)
            # Calling str(args, kwargs) (line 239)
            str_call_result_305 = invoke(stypy.reporting.localization.Localization(__file__, 239, 92), str_302, *[member_names_303], **kwargs_304)
            
            # Applying the binary operator '+' (line 239)
            result_add_306 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 90), '+', result_add_301, str_call_result_305)
            
            # Processing the call keyword arguments (line 239)
            kwargs_307 = {}
            # Getting the type of 'TypeError' (line 239)
            TypeError_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 239)
            TypeError_call_result_308 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), TypeError_297, *[localization_298, result_add_306], **kwargs_307)
            
            # Getting the type of 'False' (line 240)
            False_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'stypy_return_type', False_309)
            # SSA join for if statement (line 238)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 242)
    True_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 242)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'stypy_return_type', True_310)
    
    # ################# End of 'ensure_var_has_members(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ensure_var_has_members' in the type store
    # Getting the type of 'stypy_return_type' (line 226)
    stypy_return_type_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_311)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ensure_var_has_members'
    return stypy_return_type_311

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
    bound_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 7), 'bound')
    # Getting the type of 'None' (line 246)
    None_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 16), 'None')
    
    (may_be_314, more_types_in_union_315) = may_be_none(bound_312, None_313)

    if may_be_314:

        if more_types_in_union_315:
            # Runtime conditional SSA (line 246)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Obtaining an instance of the builtin type 'tuple' (line 247)
        tuple_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 247)
        # Adding element type (line 247)
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        # Adding element type (line 247)
        # Getting the type of 'None' (line 247)
        None_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 15), list_317, None_318)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 15), tuple_316, list_317)
        # Adding element type (line 247)
        
        # Obtaining an instance of the builtin type 'list' (line 247)
        list_319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 247)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 247, 15), tuple_316, list_319)
        
        # Assigning a type to the variable 'stypy_return_type' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'stypy_return_type', tuple_316)

        if more_types_in_union_315:
            # SSA join for if statement (line 246)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'bound' (line 246)
    bound_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'bound')
    # Assigning a type to the variable 'bound' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'bound', remove_type_from_union(bound_320, types.NoneType))
    
    # Call to isinstance(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'bound' (line 249)
    bound_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 18), 'bound', False)
    # Getting the type of 'union_type_copy' (line 249)
    union_type_copy_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 25), 'union_type_copy', False)
    # Obtaining the member 'UnionType' of a type (line 249)
    UnionType_324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 25), union_type_copy_323, 'UnionType')
    # Processing the call keyword arguments (line 249)
    kwargs_325 = {}
    # Getting the type of 'isinstance' (line 249)
    isinstance_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 249)
    isinstance_call_result_326 = invoke(stypy.reporting.localization.Localization(__file__, 249, 7), isinstance_321, *[bound_322, UnionType_324], **kwargs_325)
    
    # Testing if the type of an if condition is none (line 249)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 249, 4), isinstance_call_result_326):
        
        # Assigning a List to a Name (line 252):
        
        # Assigning a List to a Name (line 252):
        
        # Obtaining an instance of the builtin type 'list' (line 252)
        list_330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 252)
        # Adding element type (line 252)
        # Getting the type of 'bound' (line 252)
        bound_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 26), 'bound')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 25), list_330, bound_331)
        
        # Assigning a type to the variable 'types_to_check' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'types_to_check', list_330)
    else:
        
        # Testing the type of an if condition (line 249)
        if_condition_327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 249, 4), isinstance_call_result_326)
        # Assigning a type to the variable 'if_condition_327' (line 249)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'if_condition_327', if_condition_327)
        # SSA begins for if statement (line 249)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 250):
        
        # Assigning a Attribute to a Name (line 250):
        # Getting the type of 'bound' (line 250)
        bound_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 25), 'bound')
        # Obtaining the member 'types' of a type (line 250)
        types_329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 25), bound_328, 'types')
        # Assigning a type to the variable 'types_to_check' (line 250)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'types_to_check', types_329)
        # SSA branch for the else part of an if statement (line 249)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Name (line 252):
        
        # Assigning a List to a Name (line 252):
        
        # Obtaining an instance of the builtin type 'list' (line 252)
        list_330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 252)
        # Adding element type (line 252)
        # Getting the type of 'bound' (line 252)
        bound_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 26), 'bound')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 25), list_330, bound_331)
        
        # Assigning a type to the variable 'types_to_check' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'types_to_check', list_330)
        # SSA join for if statement (line 249)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Name (line 254):
    
    # Assigning a List to a Name (line 254):
    
    # Obtaining an instance of the builtin type 'list' (line 254)
    list_332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 254)
    
    # Assigning a type to the variable 'right_types' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'right_types', list_332)
    
    # Assigning a List to a Name (line 255):
    
    # Assigning a List to a Name (line 255):
    
    # Obtaining an instance of the builtin type 'list' (line 255)
    list_333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 255)
    
    # Assigning a type to the variable 'wrong_types' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'wrong_types', list_333)
    
    # Getting the type of 'types_to_check' (line 256)
    types_to_check_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 17), 'types_to_check')
    # Assigning a type to the variable 'types_to_check_334' (line 256)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'types_to_check_334', types_to_check_334)
    # Testing if the for loop is going to be iterated (line 256)
    # Testing the type of a for loop iterable (line 256)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 256, 4), types_to_check_334)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 256, 4), types_to_check_334):
        # Getting the type of the for loop variable (line 256)
        for_loop_var_335 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 256, 4), types_to_check_334)
        # Assigning a type to the variable 'type_' (line 256)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'type_', for_loop_var_335)
        # SSA begins for a for statement (line 256)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'type_group_generator_copy' (line 257)
        type_group_generator_copy_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 11), 'type_group_generator_copy')
        # Obtaining the member 'Integer' of a type (line 257)
        Integer_337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 11), type_group_generator_copy_336, 'Integer')
        # Getting the type of 'type_' (line 257)
        type__338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 48), 'type_')
        # Applying the binary operator '==' (line 257)
        result_eq_339 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 11), '==', Integer_337, type__338)
        
        
        # Getting the type of 'type_groups_copy' (line 257)
        type_groups_copy_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 57), 'type_groups_copy')
        # Obtaining the member 'CastsToIndex' of a type (line 257)
        CastsToIndex_341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 57), type_groups_copy_340, 'CastsToIndex')
        # Getting the type of 'type_' (line 257)
        type__342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 90), 'type_')
        # Applying the binary operator '==' (line 257)
        result_eq_343 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 57), '==', CastsToIndex_341, type__342)
        
        # Applying the binary operator 'or' (line 257)
        result_or_keyword_344 = python_operator(stypy.reporting.localization.Localization(__file__, 257, 11), 'or', result_eq_339, result_eq_343)
        
        # Testing if the type of an if condition is none (line 257)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 257, 8), result_or_keyword_344):
            
            # Call to append(...): (line 260)
            # Processing the call arguments (line 260)
            # Getting the type of 'type_' (line 260)
            type__353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 31), 'type_', False)
            # Processing the call keyword arguments (line 260)
            kwargs_354 = {}
            # Getting the type of 'wrong_types' (line 260)
            wrong_types_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'wrong_types', False)
            # Obtaining the member 'append' of a type (line 260)
            append_352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), wrong_types_351, 'append')
            # Calling append(args, kwargs) (line 260)
            append_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 260, 12), append_352, *[type__353], **kwargs_354)
            
        else:
            
            # Testing the type of an if condition (line 257)
            if_condition_345 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 257, 8), result_or_keyword_344)
            # Assigning a type to the variable 'if_condition_345' (line 257)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'if_condition_345', if_condition_345)
            # SSA begins for if statement (line 257)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 258)
            # Processing the call arguments (line 258)
            # Getting the type of 'type_' (line 258)
            type__348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 31), 'type_', False)
            # Processing the call keyword arguments (line 258)
            kwargs_349 = {}
            # Getting the type of 'right_types' (line 258)
            right_types_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'right_types', False)
            # Obtaining the member 'append' of a type (line 258)
            append_347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 12), right_types_346, 'append')
            # Calling append(args, kwargs) (line 258)
            append_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 258, 12), append_347, *[type__348], **kwargs_349)
            
            # SSA branch for the else part of an if statement (line 257)
            module_type_store.open_ssa_branch('else')
            
            # Call to append(...): (line 260)
            # Processing the call arguments (line 260)
            # Getting the type of 'type_' (line 260)
            type__353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 31), 'type_', False)
            # Processing the call keyword arguments (line 260)
            kwargs_354 = {}
            # Getting the type of 'wrong_types' (line 260)
            wrong_types_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 12), 'wrong_types', False)
            # Obtaining the member 'append' of a type (line 260)
            append_352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 12), wrong_types_351, 'append')
            # Calling append(args, kwargs) (line 260)
            append_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 260, 12), append_352, *[type__353], **kwargs_354)
            
            # SSA join for if statement (line 257)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 262)
    tuple_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 262, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 262)
    # Adding element type (line 262)
    # Getting the type of 'right_types' (line 262)
    right_types_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 11), 'right_types')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 11), tuple_356, right_types_357)
    # Adding element type (line 262)
    # Getting the type of 'wrong_types' (line 262)
    wrong_types_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 24), 'wrong_types')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 262, 11), tuple_356, wrong_types_358)
    
    # Assigning a type to the variable 'stypy_return_type' (line 262)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 4), 'stypy_return_type', tuple_356)
    
    # ################# End of '__slice_bounds_checking(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__slice_bounds_checking' in the type store
    # Getting the type of 'stypy_return_type' (line 245)
    stypy_return_type_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_359)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__slice_bounds_checking'
    return stypy_return_type_359

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

    str_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, (-1)), 'str', '\n    Check the parameters of a created slice to make sure that the slice have correct bounds. If not, it returns a\n    silent TypeError, as the specific problem (invalid lower, upper or step parameter is reported separately)\n    :param localization: Caller information\n    :param lower: Lower bound of the slice or None\n    :param upper: Upper bound of the slice or None\n    :param step: Step of the slice or None\n    :return: A slice object or a TypeError is its parameters are invalid\n    ')
    
    # Assigning a Name to a Name (line 275):
    
    # Assigning a Name to a Name (line 275):
    # Getting the type of 'False' (line 275)
    False_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'False')
    # Assigning a type to the variable 'error' (line 275)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'error', False_361)
    
    # Assigning a Call to a Tuple (line 276):
    
    # Assigning a Call to a Name:
    
    # Call to __slice_bounds_checking(...): (line 276)
    # Processing the call arguments (line 276)
    # Getting the type of 'lower' (line 276)
    lower_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 35), 'lower', False)
    # Processing the call keyword arguments (line 276)
    kwargs_364 = {}
    # Getting the type of '__slice_bounds_checking' (line 276)
    slice_bounds_checking_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 11), '__slice_bounds_checking', False)
    # Calling __slice_bounds_checking(args, kwargs) (line 276)
    slice_bounds_checking_call_result_365 = invoke(stypy.reporting.localization.Localization(__file__, 276, 11), slice_bounds_checking_362, *[lower_363], **kwargs_364)
    
    # Assigning a type to the variable 'call_assignment_7' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_7', slice_bounds_checking_call_result_365)
    
    # Assigning a Call to a Name (line 276):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_7' (line 276)
    call_assignment_7_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_7', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_367 = stypy_get_value_from_tuple(call_assignment_7_366, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_8' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_8', stypy_get_value_from_tuple_call_result_367)
    
    # Assigning a Name to a Name (line 276):
    # Getting the type of 'call_assignment_8' (line 276)
    call_assignment_8_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_8')
    # Assigning a type to the variable 'r' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'r', call_assignment_8_368)
    
    # Assigning a Call to a Name (line 276):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_7' (line 276)
    call_assignment_7_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_7', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_370 = stypy_get_value_from_tuple(call_assignment_7_369, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_9' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_9', stypy_get_value_from_tuple_call_result_370)
    
    # Assigning a Name to a Name (line 276):
    # Getting the type of 'call_assignment_9' (line 276)
    call_assignment_9_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 4), 'call_assignment_9')
    # Assigning a type to the variable 'w' (line 276)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 7), 'w', call_assignment_9_371)
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'w' (line 278)
    w_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), 'w', False)
    # Processing the call keyword arguments (line 278)
    kwargs_374 = {}
    # Getting the type of 'len' (line 278)
    len_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 7), 'len', False)
    # Calling len(args, kwargs) (line 278)
    len_call_result_375 = invoke(stypy.reporting.localization.Localization(__file__, 278, 7), len_372, *[w_373], **kwargs_374)
    
    int_376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 16), 'int')
    # Applying the binary operator '>' (line 278)
    result_gt_377 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 7), '>', len_call_result_375, int_376)
    
    
    
    # Call to len(...): (line 278)
    # Processing the call arguments (line 278)
    # Getting the type of 'r' (line 278)
    r_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 26), 'r', False)
    # Processing the call keyword arguments (line 278)
    kwargs_380 = {}
    # Getting the type of 'len' (line 278)
    len_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 22), 'len', False)
    # Calling len(args, kwargs) (line 278)
    len_call_result_381 = invoke(stypy.reporting.localization.Localization(__file__, 278, 22), len_378, *[r_379], **kwargs_380)
    
    int_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 31), 'int')
    # Applying the binary operator '>' (line 278)
    result_gt_383 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 22), '>', len_call_result_381, int_382)
    
    # Applying the binary operator 'and' (line 278)
    result_and_keyword_384 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 7), 'and', result_gt_377, result_gt_383)
    
    # Testing if the type of an if condition is none (line 278)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 278, 4), result_and_keyword_384):
        pass
    else:
        
        # Testing the type of an if condition (line 278)
        if_condition_385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 278, 4), result_and_keyword_384)
        # Assigning a type to the variable 'if_condition_385' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'if_condition_385', if_condition_385)
        # SSA begins for if statement (line 278)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeWarning(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'localization' (line 279)
        localization_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 20), 'localization', False)
        
        # Call to format(...): (line 279)
        # Processing the call arguments (line 279)
        # Getting the type of 'lower' (line 280)
        lower_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 27), 'lower', False)
        # Processing the call keyword arguments (line 279)
        kwargs_391 = {}
        str_388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 34), 'str', 'Some of the possible types of the lower bound of the slice ({0}) are invalid')
        # Obtaining the member 'format' of a type (line 279)
        format_389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 34), str_388, 'format')
        # Calling format(args, kwargs) (line 279)
        format_call_result_392 = invoke(stypy.reporting.localization.Localization(__file__, 279, 34), format_389, *[lower_390], **kwargs_391)
        
        # Processing the call keyword arguments (line 279)
        kwargs_393 = {}
        # Getting the type of 'TypeWarning' (line 279)
        TypeWarning_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 8), 'TypeWarning', False)
        # Calling TypeWarning(args, kwargs) (line 279)
        TypeWarning_call_result_394 = invoke(stypy.reporting.localization.Localization(__file__, 279, 8), TypeWarning_386, *[localization_387, format_call_result_392], **kwargs_393)
        
        # SSA join for if statement (line 278)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 281)
    # Processing the call arguments (line 281)
    # Getting the type of 'w' (line 281)
    w_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 11), 'w', False)
    # Processing the call keyword arguments (line 281)
    kwargs_397 = {}
    # Getting the type of 'len' (line 281)
    len_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 7), 'len', False)
    # Calling len(args, kwargs) (line 281)
    len_call_result_398 = invoke(stypy.reporting.localization.Localization(__file__, 281, 7), len_395, *[w_396], **kwargs_397)
    
    int_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 16), 'int')
    # Applying the binary operator '>' (line 281)
    result_gt_400 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 7), '>', len_call_result_398, int_399)
    
    
    
    # Call to len(...): (line 281)
    # Processing the call arguments (line 281)
    # Getting the type of 'r' (line 281)
    r_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 26), 'r', False)
    # Processing the call keyword arguments (line 281)
    kwargs_403 = {}
    # Getting the type of 'len' (line 281)
    len_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 281, 22), 'len', False)
    # Calling len(args, kwargs) (line 281)
    len_call_result_404 = invoke(stypy.reporting.localization.Localization(__file__, 281, 22), len_401, *[r_402], **kwargs_403)
    
    int_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 32), 'int')
    # Applying the binary operator '==' (line 281)
    result_eq_406 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 22), '==', len_call_result_404, int_405)
    
    # Applying the binary operator 'and' (line 281)
    result_and_keyword_407 = python_operator(stypy.reporting.localization.Localization(__file__, 281, 7), 'and', result_gt_400, result_eq_406)
    
    # Testing if the type of an if condition is none (line 281)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 281, 4), result_and_keyword_407):
        pass
    else:
        
        # Testing the type of an if condition (line 281)
        if_condition_408 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 281, 4), result_and_keyword_407)
        # Assigning a type to the variable 'if_condition_408' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 4), 'if_condition_408', if_condition_408)
        # SSA begins for if statement (line 281)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'localization' (line 282)
        localization_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 18), 'localization', False)
        
        # Call to format(...): (line 282)
        # Processing the call arguments (line 282)
        # Getting the type of 'lower' (line 282)
        lower_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 99), 'lower', False)
        # Processing the call keyword arguments (line 282)
        kwargs_414 = {}
        str_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 32), 'str', 'The type of the lower bound of the slice ({0}) is invalid')
        # Obtaining the member 'format' of a type (line 282)
        format_412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 282, 32), str_411, 'format')
        # Calling format(args, kwargs) (line 282)
        format_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 282, 32), format_412, *[lower_413], **kwargs_414)
        
        # Processing the call keyword arguments (line 282)
        kwargs_416 = {}
        # Getting the type of 'TypeError' (line 282)
        TypeError_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 282)
        TypeError_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 282, 8), TypeError_409, *[localization_410, format_call_result_415], **kwargs_416)
        
        
        # Assigning a Name to a Name (line 283):
        
        # Assigning a Name to a Name (line 283):
        # Getting the type of 'True' (line 283)
        True_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'True')
        # Assigning a type to the variable 'error' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'error', True_418)
        # SSA join for if statement (line 281)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Tuple (line 285):
    
    # Assigning a Call to a Name:
    
    # Call to __slice_bounds_checking(...): (line 285)
    # Processing the call arguments (line 285)
    # Getting the type of 'upper' (line 285)
    upper_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 35), 'upper', False)
    # Processing the call keyword arguments (line 285)
    kwargs_421 = {}
    # Getting the type of '__slice_bounds_checking' (line 285)
    slice_bounds_checking_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 11), '__slice_bounds_checking', False)
    # Calling __slice_bounds_checking(args, kwargs) (line 285)
    slice_bounds_checking_call_result_422 = invoke(stypy.reporting.localization.Localization(__file__, 285, 11), slice_bounds_checking_419, *[upper_420], **kwargs_421)
    
    # Assigning a type to the variable 'call_assignment_10' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_10', slice_bounds_checking_call_result_422)
    
    # Assigning a Call to a Name (line 285):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_10' (line 285)
    call_assignment_10_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_10', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_424 = stypy_get_value_from_tuple(call_assignment_10_423, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_11' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_11', stypy_get_value_from_tuple_call_result_424)
    
    # Assigning a Name to a Name (line 285):
    # Getting the type of 'call_assignment_11' (line 285)
    call_assignment_11_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_11')
    # Assigning a type to the variable 'r' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'r', call_assignment_11_425)
    
    # Assigning a Call to a Name (line 285):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_10' (line 285)
    call_assignment_10_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_10', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_427 = stypy_get_value_from_tuple(call_assignment_10_426, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_12' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_12', stypy_get_value_from_tuple_call_result_427)
    
    # Assigning a Name to a Name (line 285):
    # Getting the type of 'call_assignment_12' (line 285)
    call_assignment_12_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'call_assignment_12')
    # Assigning a type to the variable 'w' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 7), 'w', call_assignment_12_428)
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'w' (line 286)
    w_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 11), 'w', False)
    # Processing the call keyword arguments (line 286)
    kwargs_431 = {}
    # Getting the type of 'len' (line 286)
    len_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 7), 'len', False)
    # Calling len(args, kwargs) (line 286)
    len_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 286, 7), len_429, *[w_430], **kwargs_431)
    
    int_433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 16), 'int')
    # Applying the binary operator '>' (line 286)
    result_gt_434 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 7), '>', len_call_result_432, int_433)
    
    
    
    # Call to len(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'r' (line 286)
    r_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 26), 'r', False)
    # Processing the call keyword arguments (line 286)
    kwargs_437 = {}
    # Getting the type of 'len' (line 286)
    len_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 22), 'len', False)
    # Calling len(args, kwargs) (line 286)
    len_call_result_438 = invoke(stypy.reporting.localization.Localization(__file__, 286, 22), len_435, *[r_436], **kwargs_437)
    
    int_439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 31), 'int')
    # Applying the binary operator '>' (line 286)
    result_gt_440 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 22), '>', len_call_result_438, int_439)
    
    # Applying the binary operator 'and' (line 286)
    result_and_keyword_441 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 7), 'and', result_gt_434, result_gt_440)
    
    # Testing if the type of an if condition is none (line 286)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 286, 4), result_and_keyword_441):
        pass
    else:
        
        # Testing the type of an if condition (line 286)
        if_condition_442 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 286, 4), result_and_keyword_441)
        # Assigning a type to the variable 'if_condition_442' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'if_condition_442', if_condition_442)
        # SSA begins for if statement (line 286)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeWarning(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'localization' (line 287)
        localization_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 20), 'localization', False)
        
        # Call to format(...): (line 287)
        # Processing the call arguments (line 287)
        # Getting the type of 'upper' (line 288)
        upper_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 27), 'upper', False)
        # Processing the call keyword arguments (line 287)
        kwargs_448 = {}
        str_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 34), 'str', 'Some of the possible types of the upper bound of the slice ({0}) are invalid')
        # Obtaining the member 'format' of a type (line 287)
        format_446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 34), str_445, 'format')
        # Calling format(args, kwargs) (line 287)
        format_call_result_449 = invoke(stypy.reporting.localization.Localization(__file__, 287, 34), format_446, *[upper_447], **kwargs_448)
        
        # Processing the call keyword arguments (line 287)
        kwargs_450 = {}
        # Getting the type of 'TypeWarning' (line 287)
        TypeWarning_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 8), 'TypeWarning', False)
        # Calling TypeWarning(args, kwargs) (line 287)
        TypeWarning_call_result_451 = invoke(stypy.reporting.localization.Localization(__file__, 287, 8), TypeWarning_443, *[localization_444, format_call_result_449], **kwargs_450)
        
        # SSA join for if statement (line 286)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'w' (line 289)
    w_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 11), 'w', False)
    # Processing the call keyword arguments (line 289)
    kwargs_454 = {}
    # Getting the type of 'len' (line 289)
    len_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 7), 'len', False)
    # Calling len(args, kwargs) (line 289)
    len_call_result_455 = invoke(stypy.reporting.localization.Localization(__file__, 289, 7), len_452, *[w_453], **kwargs_454)
    
    int_456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 16), 'int')
    # Applying the binary operator '>' (line 289)
    result_gt_457 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 7), '>', len_call_result_455, int_456)
    
    
    
    # Call to len(...): (line 289)
    # Processing the call arguments (line 289)
    # Getting the type of 'r' (line 289)
    r_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'r', False)
    # Processing the call keyword arguments (line 289)
    kwargs_460 = {}
    # Getting the type of 'len' (line 289)
    len_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 22), 'len', False)
    # Calling len(args, kwargs) (line 289)
    len_call_result_461 = invoke(stypy.reporting.localization.Localization(__file__, 289, 22), len_458, *[r_459], **kwargs_460)
    
    int_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 32), 'int')
    # Applying the binary operator '==' (line 289)
    result_eq_463 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 22), '==', len_call_result_461, int_462)
    
    # Applying the binary operator 'and' (line 289)
    result_and_keyword_464 = python_operator(stypy.reporting.localization.Localization(__file__, 289, 7), 'and', result_gt_457, result_eq_463)
    
    # Testing if the type of an if condition is none (line 289)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 289, 4), result_and_keyword_464):
        pass
    else:
        
        # Testing the type of an if condition (line 289)
        if_condition_465 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 289, 4), result_and_keyword_464)
        # Assigning a type to the variable 'if_condition_465' (line 289)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'if_condition_465', if_condition_465)
        # SSA begins for if statement (line 289)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'localization' (line 290)
        localization_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 18), 'localization', False)
        
        # Call to format(...): (line 290)
        # Processing the call arguments (line 290)
        # Getting the type of 'upper' (line 290)
        upper_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 99), 'upper', False)
        # Processing the call keyword arguments (line 290)
        kwargs_471 = {}
        str_468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 32), 'str', 'The type of the upper bound of the slice ({0}) is invalid')
        # Obtaining the member 'format' of a type (line 290)
        format_469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 290, 32), str_468, 'format')
        # Calling format(args, kwargs) (line 290)
        format_call_result_472 = invoke(stypy.reporting.localization.Localization(__file__, 290, 32), format_469, *[upper_470], **kwargs_471)
        
        # Processing the call keyword arguments (line 290)
        kwargs_473 = {}
        # Getting the type of 'TypeError' (line 290)
        TypeError_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 290)
        TypeError_call_result_474 = invoke(stypy.reporting.localization.Localization(__file__, 290, 8), TypeError_466, *[localization_467, format_call_result_472], **kwargs_473)
        
        
        # Assigning a Name to a Name (line 291):
        
        # Assigning a Name to a Name (line 291):
        # Getting the type of 'True' (line 291)
        True_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 16), 'True')
        # Assigning a type to the variable 'error' (line 291)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 8), 'error', True_475)
        # SSA join for if statement (line 289)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Tuple (line 293):
    
    # Assigning a Call to a Name:
    
    # Call to __slice_bounds_checking(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'step' (line 293)
    step_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 35), 'step', False)
    # Processing the call keyword arguments (line 293)
    kwargs_478 = {}
    # Getting the type of '__slice_bounds_checking' (line 293)
    slice_bounds_checking_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 11), '__slice_bounds_checking', False)
    # Calling __slice_bounds_checking(args, kwargs) (line 293)
    slice_bounds_checking_call_result_479 = invoke(stypy.reporting.localization.Localization(__file__, 293, 11), slice_bounds_checking_476, *[step_477], **kwargs_478)
    
    # Assigning a type to the variable 'call_assignment_13' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_13', slice_bounds_checking_call_result_479)
    
    # Assigning a Call to a Name (line 293):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_13' (line 293)
    call_assignment_13_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_13', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_481 = stypy_get_value_from_tuple(call_assignment_13_480, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_14' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_14', stypy_get_value_from_tuple_call_result_481)
    
    # Assigning a Name to a Name (line 293):
    # Getting the type of 'call_assignment_14' (line 293)
    call_assignment_14_482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_14')
    # Assigning a type to the variable 'r' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'r', call_assignment_14_482)
    
    # Assigning a Call to a Name (line 293):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_13' (line 293)
    call_assignment_13_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_13', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_484 = stypy_get_value_from_tuple(call_assignment_13_483, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_15' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_15', stypy_get_value_from_tuple_call_result_484)
    
    # Assigning a Name to a Name (line 293):
    # Getting the type of 'call_assignment_15' (line 293)
    call_assignment_15_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'call_assignment_15')
    # Assigning a type to the variable 'w' (line 293)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 7), 'w', call_assignment_15_485)
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'w' (line 294)
    w_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 11), 'w', False)
    # Processing the call keyword arguments (line 294)
    kwargs_488 = {}
    # Getting the type of 'len' (line 294)
    len_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 7), 'len', False)
    # Calling len(args, kwargs) (line 294)
    len_call_result_489 = invoke(stypy.reporting.localization.Localization(__file__, 294, 7), len_486, *[w_487], **kwargs_488)
    
    int_490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 16), 'int')
    # Applying the binary operator '>' (line 294)
    result_gt_491 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 7), '>', len_call_result_489, int_490)
    
    
    
    # Call to len(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'r' (line 294)
    r_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 26), 'r', False)
    # Processing the call keyword arguments (line 294)
    kwargs_494 = {}
    # Getting the type of 'len' (line 294)
    len_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'len', False)
    # Calling len(args, kwargs) (line 294)
    len_call_result_495 = invoke(stypy.reporting.localization.Localization(__file__, 294, 22), len_492, *[r_493], **kwargs_494)
    
    int_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 31), 'int')
    # Applying the binary operator '>' (line 294)
    result_gt_497 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 22), '>', len_call_result_495, int_496)
    
    # Applying the binary operator 'and' (line 294)
    result_and_keyword_498 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 7), 'and', result_gt_491, result_gt_497)
    
    # Testing if the type of an if condition is none (line 294)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 294, 4), result_and_keyword_498):
        pass
    else:
        
        # Testing the type of an if condition (line 294)
        if_condition_499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 294, 4), result_and_keyword_498)
        # Assigning a type to the variable 'if_condition_499' (line 294)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'if_condition_499', if_condition_499)
        # SSA begins for if statement (line 294)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeWarning(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'localization' (line 295)
        localization_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 20), 'localization', False)
        
        # Call to format(...): (line 295)
        # Processing the call arguments (line 295)
        # Getting the type of 'step' (line 296)
        step_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 27), 'step', False)
        # Processing the call keyword arguments (line 295)
        kwargs_505 = {}
        str_502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 295, 34), 'str', 'Some of the possible types of the step of the slice ({0}) are invalid')
        # Obtaining the member 'format' of a type (line 295)
        format_503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 34), str_502, 'format')
        # Calling format(args, kwargs) (line 295)
        format_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 295, 34), format_503, *[step_504], **kwargs_505)
        
        # Processing the call keyword arguments (line 295)
        kwargs_507 = {}
        # Getting the type of 'TypeWarning' (line 295)
        TypeWarning_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 8), 'TypeWarning', False)
        # Calling TypeWarning(args, kwargs) (line 295)
        TypeWarning_call_result_508 = invoke(stypy.reporting.localization.Localization(__file__, 295, 8), TypeWarning_500, *[localization_501, format_call_result_506], **kwargs_507)
        
        # SSA join for if statement (line 294)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'w' (line 297)
    w_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 11), 'w', False)
    # Processing the call keyword arguments (line 297)
    kwargs_511 = {}
    # Getting the type of 'len' (line 297)
    len_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 7), 'len', False)
    # Calling len(args, kwargs) (line 297)
    len_call_result_512 = invoke(stypy.reporting.localization.Localization(__file__, 297, 7), len_509, *[w_510], **kwargs_511)
    
    int_513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 16), 'int')
    # Applying the binary operator '>' (line 297)
    result_gt_514 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 7), '>', len_call_result_512, int_513)
    
    
    
    # Call to len(...): (line 297)
    # Processing the call arguments (line 297)
    # Getting the type of 'r' (line 297)
    r_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 26), 'r', False)
    # Processing the call keyword arguments (line 297)
    kwargs_517 = {}
    # Getting the type of 'len' (line 297)
    len_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 22), 'len', False)
    # Calling len(args, kwargs) (line 297)
    len_call_result_518 = invoke(stypy.reporting.localization.Localization(__file__, 297, 22), len_515, *[r_516], **kwargs_517)
    
    int_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 297, 32), 'int')
    # Applying the binary operator '==' (line 297)
    result_eq_520 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 22), '==', len_call_result_518, int_519)
    
    # Applying the binary operator 'and' (line 297)
    result_and_keyword_521 = python_operator(stypy.reporting.localization.Localization(__file__, 297, 7), 'and', result_gt_514, result_eq_520)
    
    # Testing if the type of an if condition is none (line 297)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 297, 4), result_and_keyword_521):
        pass
    else:
        
        # Testing the type of an if condition (line 297)
        if_condition_522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 297, 4), result_and_keyword_521)
        # Assigning a type to the variable 'if_condition_522' (line 297)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 297, 4), 'if_condition_522', if_condition_522)
        # SSA begins for if statement (line 297)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to TypeError(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'localization' (line 298)
        localization_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 18), 'localization', False)
        
        # Call to format(...): (line 298)
        # Processing the call arguments (line 298)
        # Getting the type of 'step' (line 298)
        step_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 92), 'step', False)
        # Processing the call keyword arguments (line 298)
        kwargs_528 = {}
        str_525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 298, 32), 'str', 'The type of the step of the slice ({0}) is invalid')
        # Obtaining the member 'format' of a type (line 298)
        format_526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 32), str_525, 'format')
        # Calling format(args, kwargs) (line 298)
        format_call_result_529 = invoke(stypy.reporting.localization.Localization(__file__, 298, 32), format_526, *[step_527], **kwargs_528)
        
        # Processing the call keyword arguments (line 298)
        kwargs_530 = {}
        # Getting the type of 'TypeError' (line 298)
        TypeError_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 8), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 298)
        TypeError_call_result_531 = invoke(stypy.reporting.localization.Localization(__file__, 298, 8), TypeError_523, *[localization_524, format_call_result_529], **kwargs_530)
        
        
        # Assigning a Name to a Name (line 299):
        
        # Assigning a Name to a Name (line 299):
        # Getting the type of 'True' (line 299)
        True_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'True')
        # Assigning a type to the variable 'error' (line 299)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 299, 8), 'error', True_532)
        # SSA join for if statement (line 297)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'error' (line 301)
    error_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 11), 'error')
    # Applying the 'not' unary operator (line 301)
    result_not__534 = python_operator(stypy.reporting.localization.Localization(__file__, 301, 7), 'not', error_533)
    
    # Testing if the type of an if condition is none (line 301)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 301, 4), result_not__534):
        
        # Call to TypeError(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'localization' (line 304)
        localization_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'localization', False)
        str_543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 39), 'str', 'Type error when specifying slice bounds')
        # Processing the call keyword arguments (line 304)
        # Getting the type of 'False' (line 304)
        False_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 93), 'False', False)
        keyword_545 = False_544
        kwargs_546 = {'prints_msg': keyword_545}
        # Getting the type of 'TypeError' (line 304)
        TypeError_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 304)
        TypeError_call_result_547 = invoke(stypy.reporting.localization.Localization(__file__, 304, 15), TypeError_541, *[localization_542, str_543], **kwargs_546)
        
        # Assigning a type to the variable 'stypy_return_type' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'stypy_return_type', TypeError_call_result_547)
    else:
        
        # Testing the type of an if condition (line 301)
        if_condition_535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 301, 4), result_not__534)
        # Assigning a type to the variable 'if_condition_535' (line 301)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'if_condition_535', if_condition_535)
        # SSA begins for if statement (line 301)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to get_builtin_type(...): (line 302)
        # Processing the call arguments (line 302)
        # Getting the type of 'localization' (line 302)
        localization_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 32), 'localization', False)
        str_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 46), 'str', 'slice')
        # Processing the call keyword arguments (line 302)
        kwargs_539 = {}
        # Getting the type of 'get_builtin_type' (line 302)
        get_builtin_type_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 15), 'get_builtin_type', False)
        # Calling get_builtin_type(args, kwargs) (line 302)
        get_builtin_type_call_result_540 = invoke(stypy.reporting.localization.Localization(__file__, 302, 15), get_builtin_type_536, *[localization_537, str_538], **kwargs_539)
        
        # Assigning a type to the variable 'stypy_return_type' (line 302)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 8), 'stypy_return_type', get_builtin_type_call_result_540)
        # SSA branch for the else part of an if statement (line 301)
        module_type_store.open_ssa_branch('else')
        
        # Call to TypeError(...): (line 304)
        # Processing the call arguments (line 304)
        # Getting the type of 'localization' (line 304)
        localization_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 25), 'localization', False)
        str_543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 39), 'str', 'Type error when specifying slice bounds')
        # Processing the call keyword arguments (line 304)
        # Getting the type of 'False' (line 304)
        False_544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 93), 'False', False)
        keyword_545 = False_544
        kwargs_546 = {'prints_msg': keyword_545}
        # Getting the type of 'TypeError' (line 304)
        TypeError_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 304)
        TypeError_call_result_547 = invoke(stypy.reporting.localization.Localization(__file__, 304, 15), TypeError_541, *[localization_542, str_543], **kwargs_546)
        
        # Assigning a type to the variable 'stypy_return_type' (line 304)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'stypy_return_type', TypeError_call_result_547)
        # SSA join for if statement (line 301)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'ensure_slice_bounds(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ensure_slice_bounds' in the type store
    # Getting the type of 'stypy_return_type' (line 265)
    stypy_return_type_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_548)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ensure_slice_bounds'
    return stypy_return_type_548

# Assigning a type to the variable 'ensure_slice_bounds' (line 265)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 0), 'ensure_slice_bounds', ensure_slice_bounds)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
