
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import inspect
2: ## AQUI: INCLUIR UNION_TYPE_COPY CUELGA POR RECURSION PROBLEM
3: from ....errors_copy.type_error_copy import TypeError
4: from ....python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy import RecursionType
5: import type_equivalence_copy
6: from ....python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
7: from ....python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
8: 
9: 
10: '''
11: Several functions to identify what kind of type is a Python object
12: '''
13: 
14: 
15: # -------------
16: # Runtime type inspection
17: # -------------
18: 
19: # TODO: Remove?
20: # def is_type_store(obj_type):
21: #     return isinstance(obj_type, typestore.TypeStore)
22: 
23: #
24: # def is_object(obj_type):
25: #     if is_union_type(obj_type) or is_error_type(obj_type) or is_type_store(obj_type):
26: #         return False
27: #     return inspect.isclass(type(obj_type))
28: 
29: 
30: def is_class(class_type):
31:     '''
32:     Determines if class_type is a Python class
33: 
34:     :param class_type: Type to test
35:     :return: bool
36:     '''
37:     if is_union_type(class_type) or is_error_type(class_type):
38:         return False
39:     return inspect.isclass(class_type)
40: 
41: 
42: def is_union_type(the_type):
43:     '''
44:     Determines if the_type is a UnionType
45:     :param the_type: Type to test
46:     :return: bool
47:     '''
48:     return isinstance(the_type, union_type_copy.UnionType)
49: 
50: 
51: def is_undefined_type(the_type):
52:     '''
53:     Determines if the_type is an UndefinedType
54:     :param the_type: Type to test
55:     :return: bool
56:     '''
57:     return the_type == UndefinedType
58: 
59: 
60: def is_error_type(the_type):
61:     '''
62:     Determines if the_type is an ErrorType
63:     :param the_type: Type to test
64:     :return: bool
65:     '''
66:     return isinstance(the_type, TypeError)
67: 
68: 
69: def is_recursion_type(the_type):
70:     '''
71:     Determines if the_type is a RecursionType
72:     :param the_type: Type to test
73:     :return: bool
74:     '''
75:     return isinstance(the_type, RecursionType)
76: 
77: 
78: def is_property(the_type):
79:     '''
80:     Determines if the_type is a Python property
81:     :param the_type: Type to test
82:     :return: bool
83:     '''
84:     return isinstance(the_type, property)
85: 
86: # TODO: Remove?
87: # def __get_member_value(localization, member, type_of_obj, field_name):
88: #     # member = type_of_obj.__dict__[field_name]
89: #     if is_property(member):
90: #         return member.fget(localization, type_of_obj)
91: #     return member
92: #
93: #
94: # def get_type_of_member(localization, type_of_obj, field_name):
95: #     field_name = get_member_name(field_name)
96: #     return_type = __get_type_of_member(localization, type_of_obj, field_name)
97: #     if return_type is None:
98: #         return TypeError(localization, "The object does not provide a field named '%s'" % field_name)
99: #     else:
100: #         return __get_member_value(localization, return_type, type_of_obj, field_name)
101: #
102: #
103: # def __get_type_of_member_class_hierarchy(localization, type_of_obj, field_name):
104: #     if field_name in type_of_obj.__dict__:
105: #         # return __get_member_value (line, column, type_of_obj, field_name)
106: #         return type_of_obj.__dict__[field_name]
107: #     else:
108: #         for class_ in type_of_obj.__bases__:
109: #             return __get_type_of_member_class_hierarchy(localization, class_, field_name)
110: #         return None
111: #
112: #
113: # def __get_type_of_member(localization, type_of_obj, field_name):
114: #     if is_error_type(type_of_obj):
115: #         return type_of_obj
116: #
117: #     if inspect.ismodule(type_of_obj):
118: #         if field_name == "__class__":
119: #             return type_of_obj
120: #         if field_name in type_of_obj.__dict__:
121: #             return type_of_obj.__dict__[field_name]
122: #
123: #     if is_class(type_of_obj):
124: #         return __get_type_of_member_class_hierarchy(localization, type_of_obj, field_name)
125: #
126: #     if is_object(type_of_obj):
127: #         if field_name in type_of_obj.__dict__:
128: #             return type_of_obj.__dict__[field_name]
129: #         if field_name == "__class__":
130: #             return type_of_obj.__class__
131: #
132: #         return __get_type_of_member_class_hierarchy(localization, type_of_obj.__class__, field_name)
133: #
134: #     if is_union_type(type_of_obj):
135: #         inferred_types = []
136: #         error_types = []
137: #         for t in type_of_obj.types:
138: #             inferred_type = __get_type_of_member(localization, t, field_name)
139: #             #print "inferred_type = ", inferred_type
140: #             if not inferred_type is None:
141: #                 inferred_types.append(inferred_type)
142: #             else:
143: #                 error_types.append(t)
144: #         if len(inferred_types) == 0:
145: #             return None  # compiler error (no object provides the field)
146: #         if len(error_types) > 0:
147: #             Warning(localization, "The object may not provide a field named '%s'" % field_name)
148: #         inferred_type = None
149: #         for t in inferred_types:
150: #             inferred_type = stypy.python_lib.python_types.type_inference.union_type.UnionType.add(inferred_type, t)
151: #         return inferred_type
152: #
153: #     if is_type_store(type_of_obj):  # For modules
154: #         if field_name == "__class__":
155: #             return type_of_obj
156: #         return type_of_obj.get_type_of(localization, field_name)
157: #
158: #     return None  # compiler error
159: #
160: #
161: # def set_type_of_member(localization, type_of_obj, field_name, type_of_field):
162: #     field_name = get_member_name(field_name)
163: #
164: #     if is_class(type_of_obj):
165: #         member = __get_type_of_member(localization, type_of_obj, field_name)
166: #         if is_property(member):
167: #             member.fset(localization, type_of_obj, type_of_field)
168: #             return
169: #
170: #         type_of_obj.__dict__[field_name] = type_of_field
171: #         return
172: #
173: #     if is_object(type_of_obj):
174: #         member = __get_type_of_member(localization, type_of_obj, field_name)
175: #
176: #         if is_property(member):
177: #             member.fset(localization, type_of_obj, type_of_field)
178: #             return
179: #
180: #         type_of_obj.__dict__[field_name] = type_of_field
181: #         return
182: #
183: #     if is_type_store(type_of_obj):  # For modules
184: #         type_of_obj.set_type_of(localization, field_name, type_of_field)
185: #
186: #
187: # def invoke_member(localization, member, *args, **kwargs):
188: #     owner = None
189: #     if len(args) > 0:
190: #         owner = args[0]
191: #
192: #     if isinstance(owner, typestore.TypeStore) or inspect.ismodule(owner):
193: #         return member(localization, *args[1:], **kwargs)
194: #     else:
195: #         if inspect.isfunction(member) or inspect.isclass(member):
196: #             return member(localization, *args, **kwargs)
197: #         if type(owner) is types.InstanceType or type(owner) is types.ClassType:
198: #             return member(localization, owner, *args, **kwargs)
199: 
200: 
201: # # --------------------
202: # # Subtyping
203: # # --------------------
204: #
205: # def is_subtype(type1, type2):
206: #     if type_equivalence.equivalent_types(type1, type2):
207: #         return True
208: #     if type_equivalence.equivalent_types(type1, int) and type_equivalence.equivalent_types(type2, float):
209: #         return True
210: #     if is_union_type(type1):
211: #         for each_type in type1.types:
212: #             if not is_subtype(each_type, type2):
213: #                 return False
214: #         return True
215: #     if is_union_type(type2):
216: #         for each_type in type2.types:
217: #             if not is_subtype(type1, each_type):
218: #                 return False
219: #         return True
220: 

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

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_introspection_copy/')
import_14240 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_14240) is not StypyTypeError):

    if (import_14240 != 'pyd_module'):
        __import__(import_14240)
        sys_modules_14241 = sys.modules[import_14240]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_14241.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_14241, sys_modules_14241.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_14240)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_introspection_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy import RecursionType' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_introspection_copy/')
import_14242 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy')

if (type(import_14242) is not StypyTypeError):

    if (import_14242 != 'pyd_module'):
        __import__(import_14242)
        sys_modules_14243 = sys.modules[import_14242]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy', sys_modules_14243.module_type_store, module_type_store, ['RecursionType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_14243, sys_modules_14243.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy import RecursionType

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy', None, module_type_store, ['RecursionType'], [RecursionType])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.no_recursion_copy', import_14242)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_introspection_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import type_equivalence_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_introspection_copy/')
import_14244 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'type_equivalence_copy')

if (type(import_14244) is not StypyTypeError):

    if (import_14244 != 'pyd_module'):
        __import__(import_14244)
        sys_modules_14245 = sys.modules[import_14244]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'type_equivalence_copy', sys_modules_14245.module_type_store, module_type_store)
    else:
        import type_equivalence_copy

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'type_equivalence_copy', type_equivalence_copy, module_type_store)

else:
    # Assigning a type to the variable 'type_equivalence_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'type_equivalence_copy', import_14244)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_introspection_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_introspection_copy/')
import_14246 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_14246) is not StypyTypeError):

    if (import_14246 != 'pyd_module'):
        __import__(import_14246)
        sys_modules_14247 = sys.modules[import_14246]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_14247.module_type_store, module_type_store, ['union_type_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_14247, sys_modules_14247.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import union_type_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['union_type_copy'], [union_type_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_14246)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_introspection_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_introspection_copy/')
import_14248 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy')

if (type(import_14248) is not StypyTypeError):

    if (import_14248 != 'pyd_module'):
        __import__(import_14248)
        sys_modules_14249 = sys.modules[import_14248]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', sys_modules_14249.module_type_store, module_type_store, ['UndefinedType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_14249, sys_modules_14249.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', None, module_type_store, ['UndefinedType'], [UndefinedType])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', import_14248)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_introspection_copy/')

str_14250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, (-1)), 'str', '\nSeveral functions to identify what kind of type is a Python object\n')

@norecursion
def is_class(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_class'
    module_type_store = module_type_store.open_function_context('is_class', 30, 0, False)
    
    # Passed parameters checking function
    is_class.stypy_localization = localization
    is_class.stypy_type_of_self = None
    is_class.stypy_type_store = module_type_store
    is_class.stypy_function_name = 'is_class'
    is_class.stypy_param_names_list = ['class_type']
    is_class.stypy_varargs_param_name = None
    is_class.stypy_kwargs_param_name = None
    is_class.stypy_call_defaults = defaults
    is_class.stypy_call_varargs = varargs
    is_class.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_class', ['class_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_class', localization, ['class_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_class(...)' code ##################

    str_14251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', '\n    Determines if class_type is a Python class\n\n    :param class_type: Type to test\n    :return: bool\n    ')
    
    # Evaluating a boolean operation
    
    # Call to is_union_type(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'class_type' (line 37)
    class_type_14253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'class_type', False)
    # Processing the call keyword arguments (line 37)
    kwargs_14254 = {}
    # Getting the type of 'is_union_type' (line 37)
    is_union_type_14252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 7), 'is_union_type', False)
    # Calling is_union_type(args, kwargs) (line 37)
    is_union_type_call_result_14255 = invoke(stypy.reporting.localization.Localization(__file__, 37, 7), is_union_type_14252, *[class_type_14253], **kwargs_14254)
    
    
    # Call to is_error_type(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'class_type' (line 37)
    class_type_14257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 50), 'class_type', False)
    # Processing the call keyword arguments (line 37)
    kwargs_14258 = {}
    # Getting the type of 'is_error_type' (line 37)
    is_error_type_14256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 36), 'is_error_type', False)
    # Calling is_error_type(args, kwargs) (line 37)
    is_error_type_call_result_14259 = invoke(stypy.reporting.localization.Localization(__file__, 37, 36), is_error_type_14256, *[class_type_14257], **kwargs_14258)
    
    # Applying the binary operator 'or' (line 37)
    result_or_keyword_14260 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 7), 'or', is_union_type_call_result_14255, is_error_type_call_result_14259)
    
    # Testing if the type of an if condition is none (line 37)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 37, 4), result_or_keyword_14260):
        pass
    else:
        
        # Testing the type of an if condition (line 37)
        if_condition_14261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 4), result_or_keyword_14260)
        # Assigning a type to the variable 'if_condition_14261' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'if_condition_14261', if_condition_14261)
        # SSA begins for if statement (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 38)
        False_14262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', False_14262)
        # SSA join for if statement (line 37)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to isclass(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'class_type' (line 39)
    class_type_14265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'class_type', False)
    # Processing the call keyword arguments (line 39)
    kwargs_14266 = {}
    # Getting the type of 'inspect' (line 39)
    inspect_14263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'inspect', False)
    # Obtaining the member 'isclass' of a type (line 39)
    isclass_14264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 11), inspect_14263, 'isclass')
    # Calling isclass(args, kwargs) (line 39)
    isclass_call_result_14267 = invoke(stypy.reporting.localization.Localization(__file__, 39, 11), isclass_14264, *[class_type_14265], **kwargs_14266)
    
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type', isclass_call_result_14267)
    
    # ################# End of 'is_class(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_class' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_14268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14268)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_class'
    return stypy_return_type_14268

# Assigning a type to the variable 'is_class' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'is_class', is_class)

@norecursion
def is_union_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_union_type'
    module_type_store = module_type_store.open_function_context('is_union_type', 42, 0, False)
    
    # Passed parameters checking function
    is_union_type.stypy_localization = localization
    is_union_type.stypy_type_of_self = None
    is_union_type.stypy_type_store = module_type_store
    is_union_type.stypy_function_name = 'is_union_type'
    is_union_type.stypy_param_names_list = ['the_type']
    is_union_type.stypy_varargs_param_name = None
    is_union_type.stypy_kwargs_param_name = None
    is_union_type.stypy_call_defaults = defaults
    is_union_type.stypy_call_varargs = varargs
    is_union_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_union_type', ['the_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_union_type', localization, ['the_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_union_type(...)' code ##################

    str_14269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, (-1)), 'str', '\n    Determines if the_type is a UnionType\n    :param the_type: Type to test\n    :return: bool\n    ')
    
    # Call to isinstance(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'the_type' (line 48)
    the_type_14271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'the_type', False)
    # Getting the type of 'union_type_copy' (line 48)
    union_type_copy_14272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 32), 'union_type_copy', False)
    # Obtaining the member 'UnionType' of a type (line 48)
    UnionType_14273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 32), union_type_copy_14272, 'UnionType')
    # Processing the call keyword arguments (line 48)
    kwargs_14274 = {}
    # Getting the type of 'isinstance' (line 48)
    isinstance_14270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 48)
    isinstance_call_result_14275 = invoke(stypy.reporting.localization.Localization(__file__, 48, 11), isinstance_14270, *[the_type_14271, UnionType_14273], **kwargs_14274)
    
    # Assigning a type to the variable 'stypy_return_type' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type', isinstance_call_result_14275)
    
    # ################# End of 'is_union_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_union_type' in the type store
    # Getting the type of 'stypy_return_type' (line 42)
    stypy_return_type_14276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14276)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_union_type'
    return stypy_return_type_14276

# Assigning a type to the variable 'is_union_type' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'is_union_type', is_union_type)

@norecursion
def is_undefined_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_undefined_type'
    module_type_store = module_type_store.open_function_context('is_undefined_type', 51, 0, False)
    
    # Passed parameters checking function
    is_undefined_type.stypy_localization = localization
    is_undefined_type.stypy_type_of_self = None
    is_undefined_type.stypy_type_store = module_type_store
    is_undefined_type.stypy_function_name = 'is_undefined_type'
    is_undefined_type.stypy_param_names_list = ['the_type']
    is_undefined_type.stypy_varargs_param_name = None
    is_undefined_type.stypy_kwargs_param_name = None
    is_undefined_type.stypy_call_defaults = defaults
    is_undefined_type.stypy_call_varargs = varargs
    is_undefined_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_undefined_type', ['the_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_undefined_type', localization, ['the_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_undefined_type(...)' code ##################

    str_14277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n    Determines if the_type is an UndefinedType\n    :param the_type: Type to test\n    :return: bool\n    ')
    
    # Getting the type of 'the_type' (line 57)
    the_type_14278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'the_type')
    # Getting the type of 'UndefinedType' (line 57)
    UndefinedType_14279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'UndefinedType')
    # Applying the binary operator '==' (line 57)
    result_eq_14280 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 11), '==', the_type_14278, UndefinedType_14279)
    
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type', result_eq_14280)
    
    # ################# End of 'is_undefined_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_undefined_type' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_14281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14281)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_undefined_type'
    return stypy_return_type_14281

# Assigning a type to the variable 'is_undefined_type' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'is_undefined_type', is_undefined_type)

@norecursion
def is_error_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_error_type'
    module_type_store = module_type_store.open_function_context('is_error_type', 60, 0, False)
    
    # Passed parameters checking function
    is_error_type.stypy_localization = localization
    is_error_type.stypy_type_of_self = None
    is_error_type.stypy_type_store = module_type_store
    is_error_type.stypy_function_name = 'is_error_type'
    is_error_type.stypy_param_names_list = ['the_type']
    is_error_type.stypy_varargs_param_name = None
    is_error_type.stypy_kwargs_param_name = None
    is_error_type.stypy_call_defaults = defaults
    is_error_type.stypy_call_varargs = varargs
    is_error_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_error_type', ['the_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_error_type', localization, ['the_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_error_type(...)' code ##################

    str_14282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, (-1)), 'str', '\n    Determines if the_type is an ErrorType\n    :param the_type: Type to test\n    :return: bool\n    ')
    
    # Call to isinstance(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'the_type' (line 66)
    the_type_14284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'the_type', False)
    # Getting the type of 'TypeError' (line 66)
    TypeError_14285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'TypeError', False)
    # Processing the call keyword arguments (line 66)
    kwargs_14286 = {}
    # Getting the type of 'isinstance' (line 66)
    isinstance_14283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 66)
    isinstance_call_result_14287 = invoke(stypy.reporting.localization.Localization(__file__, 66, 11), isinstance_14283, *[the_type_14284, TypeError_14285], **kwargs_14286)
    
    # Assigning a type to the variable 'stypy_return_type' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type', isinstance_call_result_14287)
    
    # ################# End of 'is_error_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_error_type' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_14288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14288)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_error_type'
    return stypy_return_type_14288

# Assigning a type to the variable 'is_error_type' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'is_error_type', is_error_type)

@norecursion
def is_recursion_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_recursion_type'
    module_type_store = module_type_store.open_function_context('is_recursion_type', 69, 0, False)
    
    # Passed parameters checking function
    is_recursion_type.stypy_localization = localization
    is_recursion_type.stypy_type_of_self = None
    is_recursion_type.stypy_type_store = module_type_store
    is_recursion_type.stypy_function_name = 'is_recursion_type'
    is_recursion_type.stypy_param_names_list = ['the_type']
    is_recursion_type.stypy_varargs_param_name = None
    is_recursion_type.stypy_kwargs_param_name = None
    is_recursion_type.stypy_call_defaults = defaults
    is_recursion_type.stypy_call_varargs = varargs
    is_recursion_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_recursion_type', ['the_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_recursion_type', localization, ['the_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_recursion_type(...)' code ##################

    str_14289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, (-1)), 'str', '\n    Determines if the_type is a RecursionType\n    :param the_type: Type to test\n    :return: bool\n    ')
    
    # Call to isinstance(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'the_type' (line 75)
    the_type_14291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'the_type', False)
    # Getting the type of 'RecursionType' (line 75)
    RecursionType_14292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'RecursionType', False)
    # Processing the call keyword arguments (line 75)
    kwargs_14293 = {}
    # Getting the type of 'isinstance' (line 75)
    isinstance_14290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 75)
    isinstance_call_result_14294 = invoke(stypy.reporting.localization.Localization(__file__, 75, 11), isinstance_14290, *[the_type_14291, RecursionType_14292], **kwargs_14293)
    
    # Assigning a type to the variable 'stypy_return_type' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type', isinstance_call_result_14294)
    
    # ################# End of 'is_recursion_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_recursion_type' in the type store
    # Getting the type of 'stypy_return_type' (line 69)
    stypy_return_type_14295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14295)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_recursion_type'
    return stypy_return_type_14295

# Assigning a type to the variable 'is_recursion_type' (line 69)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 0), 'is_recursion_type', is_recursion_type)

@norecursion
def is_property(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_property'
    module_type_store = module_type_store.open_function_context('is_property', 78, 0, False)
    
    # Passed parameters checking function
    is_property.stypy_localization = localization
    is_property.stypy_type_of_self = None
    is_property.stypy_type_store = module_type_store
    is_property.stypy_function_name = 'is_property'
    is_property.stypy_param_names_list = ['the_type']
    is_property.stypy_varargs_param_name = None
    is_property.stypy_kwargs_param_name = None
    is_property.stypy_call_defaults = defaults
    is_property.stypy_call_varargs = varargs
    is_property.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_property', ['the_type'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_property', localization, ['the_type'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_property(...)' code ##################

    str_14296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, (-1)), 'str', '\n    Determines if the_type is a Python property\n    :param the_type: Type to test\n    :return: bool\n    ')
    
    # Call to isinstance(...): (line 84)
    # Processing the call arguments (line 84)
    # Getting the type of 'the_type' (line 84)
    the_type_14298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 22), 'the_type', False)
    # Getting the type of 'property' (line 84)
    property_14299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'property', False)
    # Processing the call keyword arguments (line 84)
    kwargs_14300 = {}
    # Getting the type of 'isinstance' (line 84)
    isinstance_14297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 84)
    isinstance_call_result_14301 = invoke(stypy.reporting.localization.Localization(__file__, 84, 11), isinstance_14297, *[the_type_14298, property_14299], **kwargs_14300)
    
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type', isinstance_call_result_14301)
    
    # ################# End of 'is_property(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_property' in the type store
    # Getting the type of 'stypy_return_type' (line 78)
    stypy_return_type_14302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14302)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_property'
    return stypy_return_type_14302

# Assigning a type to the variable 'is_property' (line 78)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 0), 'is_property', is_property)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
