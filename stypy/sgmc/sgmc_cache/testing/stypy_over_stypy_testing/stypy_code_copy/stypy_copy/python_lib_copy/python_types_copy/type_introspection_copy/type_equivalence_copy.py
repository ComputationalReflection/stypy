
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type
2: 
3: '''
4: This file implements an algorithm to compare types by its structure
5: '''
6: 
7: # TODO: Remove?
8: # --------------------
9: # Type Equivalence
10: # --------------------
11: 
12: 
13: # def equivalent_types(type1, type2):
14: #     '''Type equivalence is much more complex; we start with similar identity'''
15: #     if runtime_type_inspection.is_union_type(type1):
16: #         return __are_equivalent_union_types(type1, type2)
17: #     if runtime_type_inspection.is_union_type(type2):
18: #         return __are_equivalent_union_types(type1, type2)
19: #
20: #     # #Two dictionaries are equal is their element and index type are the same and their type mappings
21: #     # if isinstance(type1, python_data_structures.PythonDictionary):
22: #     #
23: #     # if isinstance(type1, python_data_structures.PythonIndexableDataStructure):
24: #
25: #     return type1 == type2
26: 
27: 
28: # def __are_equivalent_union_types(type1, type2):
29: #     if not (runtime_type_inspection.is_union_type(type1)):
30: #         return False
31: #     if not (runtime_type_inspection.is_union_type(type2)):
32: #         return False
33: #     if len(type1.types) != len(type2.types):
34: #         return False
35: #     types2 = list(type2.types)
36: #     for t1 in type1.types:
37: #         t2_index = find_equivalent_type(t1, types2)
38: #         if t2_index == -1:
39: #             return False
40: #         del types2[t2_index]
41: #     return True
42: 
43: 
44: # def find_equivalent_type(type, type_list):
45: #     for i in range(len(type_list)):
46: #         if equivalent_types(type, type_list[i]):
47: #             return i
48: #     return -1
49: 
50: # non_comparable_members = ['__call__', '__delattr__', '__format__']
51: 
52: def structural_equivalence(type1, type2, exclude_special_properties=True):
53:     '''
54:     Test if two types are structurally equal, optionally excluding special properties that should have been compared
55:     previously. This method is used by the TypeInferenceProxy __eq__ method.
56: 
57:     :param type1: Type to compare its structure
58:     :param type2: Type to compare its structure
59:     :param exclude_special_properties: Do not compare the value of certain special hardcoded properties, that have
60:     been processed previously in the TypeInferenceProxy __eq__ method.
61:     :return: bool
62:     '''
63:     type1_members = dir(type1)
64:     type2_members = dir(type2)
65: 
66:     # We do not consider values if only one of the compared types has one
67:     value_in_type1 = 'value' in type1_members
68:     value_in_type2 = 'value' in type1_members
69: 
70:     if value_in_type1 and not value_in_type2:
71:         type1_members.remove('value')
72: 
73:     if value_in_type2 and not value_in_type1:
74:         type1_members.remove('value')
75: 
76:     same_structure = type1_members == type2_members
77:     if not same_structure:
78:         return False
79: 
80:     for member in type1_members:
81:         if exclude_special_properties:
82:             if member in Type.special_properties_for_equality:
83:                 continue
84: 
85:         # try:
86:         member1 = getattr(type1, member)
87:         member2 = getattr(type2, member)
88: 
89:         # If both are wrapper types, we compare it
90:         if isinstance(member1, Type):  # and isinstance(member2, Type):
91:             if not member1.get_python_type() == member2.get_python_type():
92:                 return False
93:         else:
94:             # Else we compare its types
95:             if not type(member1) == type(member2):
96:                 return False
97:                 # except:
98:                 #     return False
99: 
100:                 # for member in type1_members:
101:                 #     if exclude_special_properties:
102:                 #         if member in Type.special_properties_for_equality:
103:                 #             continue
104:                 #     member1 = getattr(type1, member)
105:                 #     member2 = getattr(type2, member)
106:                 #     try:
107:                 #         # If both are wrapper types, we compare it
108:                 #         if not member1.get_python_type() == member2.get_python_type():
109:                 #             return False
110:                 #     except:
111:                 #         # Else we compare its types
112:                 #         if not type(member1) == type(member2):
113:                 #             return False
114: 
115:                 # try:
116:                 #     member1 = getattr(type1, member)
117:                 #     member2 = getattr(type2, member)
118:                 #
119:                 #     #If both are wrapper types, we compare it
120:                 #     if isinstance(member1, Type) and isinstance(member2, Type):
121:                 #         if not member1.get_python_type() == member2.get_python_type():
122:                 #             return False
123:                 #     else:
124:                 #         #Else we compare its types
125:                 #         if not type(member1) == type(member2):
126:                 #             return False
127:                 # except:
128:                 #     return False
129: 
130:     # for member in type2_members:
131:     #     # if member in non_comparable_members:
132:     #     #     continue
133:     #     try:
134:     #         member1 = getattr(type1, member)
135:     #         member2 = getattr(type2, member)
136:     #
137:     #         #If both are types, we compare it
138:     #         if isinstance(member1, Type) and isinstance(member2, Type):
139:     #             if not member1.get_python_type() == member2.get_python_type():
140:     #                 return False
141:     #         else:
142:     #             #Else we compare its types
143:     #             if not type(member1) == type(member2):
144:     #                 return False
145:     #         # if not getattr(type1, member) == getattr(type2, member):
146:     #         #     return False
147:     #     except:
148:     #         return False
149: 
150:     return True
151: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_introspection_copy/')
import_2201 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy')

if (type(import_2201) is not StypyTypeError):

    if (import_2201 != 'pyd_module'):
        __import__(import_2201)
        sys_modules_2202 = sys.modules[import_2201]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', sys_modules_2202.module_type_store, module_type_store, ['Type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_2202, sys_modules_2202.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', None, module_type_store, ['Type'], [Type])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', import_2201)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_introspection_copy/')

str_2203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', '\nThis file implements an algorithm to compare types by its structure\n')

@norecursion
def structural_equivalence(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 52)
    True_2204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 68), 'True')
    defaults = [True_2204]
    # Create a new context for function 'structural_equivalence'
    module_type_store = module_type_store.open_function_context('structural_equivalence', 52, 0, False)
    
    # Passed parameters checking function
    structural_equivalence.stypy_localization = localization
    structural_equivalence.stypy_type_of_self = None
    structural_equivalence.stypy_type_store = module_type_store
    structural_equivalence.stypy_function_name = 'structural_equivalence'
    structural_equivalence.stypy_param_names_list = ['type1', 'type2', 'exclude_special_properties']
    structural_equivalence.stypy_varargs_param_name = None
    structural_equivalence.stypy_kwargs_param_name = None
    structural_equivalence.stypy_call_defaults = defaults
    structural_equivalence.stypy_call_varargs = varargs
    structural_equivalence.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'structural_equivalence', ['type1', 'type2', 'exclude_special_properties'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'structural_equivalence', localization, ['type1', 'type2', 'exclude_special_properties'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'structural_equivalence(...)' code ##################

    str_2205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', '\n    Test if two types are structurally equal, optionally excluding special properties that should have been compared\n    previously. This method is used by the TypeInferenceProxy __eq__ method.\n\n    :param type1: Type to compare its structure\n    :param type2: Type to compare its structure\n    :param exclude_special_properties: Do not compare the value of certain special hardcoded properties, that have\n    been processed previously in the TypeInferenceProxy __eq__ method.\n    :return: bool\n    ')
    
    # Assigning a Call to a Name (line 63):
    
    # Call to dir(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'type1' (line 63)
    type1_2207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'type1', False)
    # Processing the call keyword arguments (line 63)
    kwargs_2208 = {}
    # Getting the type of 'dir' (line 63)
    dir_2206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'dir', False)
    # Calling dir(args, kwargs) (line 63)
    dir_call_result_2209 = invoke(stypy.reporting.localization.Localization(__file__, 63, 20), dir_2206, *[type1_2207], **kwargs_2208)
    
    # Assigning a type to the variable 'type1_members' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'type1_members', dir_call_result_2209)
    
    # Assigning a Call to a Name (line 64):
    
    # Call to dir(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'type2' (line 64)
    type2_2211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'type2', False)
    # Processing the call keyword arguments (line 64)
    kwargs_2212 = {}
    # Getting the type of 'dir' (line 64)
    dir_2210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'dir', False)
    # Calling dir(args, kwargs) (line 64)
    dir_call_result_2213 = invoke(stypy.reporting.localization.Localization(__file__, 64, 20), dir_2210, *[type2_2211], **kwargs_2212)
    
    # Assigning a type to the variable 'type2_members' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'type2_members', dir_call_result_2213)
    
    # Assigning a Compare to a Name (line 67):
    
    str_2214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 21), 'str', 'value')
    # Getting the type of 'type1_members' (line 67)
    type1_members_2215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 32), 'type1_members')
    # Applying the binary operator 'in' (line 67)
    result_contains_2216 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 21), 'in', str_2214, type1_members_2215)
    
    # Assigning a type to the variable 'value_in_type1' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'value_in_type1', result_contains_2216)
    
    # Assigning a Compare to a Name (line 68):
    
    str_2217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'str', 'value')
    # Getting the type of 'type1_members' (line 68)
    type1_members_2218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 32), 'type1_members')
    # Applying the binary operator 'in' (line 68)
    result_contains_2219 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 21), 'in', str_2217, type1_members_2218)
    
    # Assigning a type to the variable 'value_in_type2' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'value_in_type2', result_contains_2219)
    
    # Evaluating a boolean operation
    # Getting the type of 'value_in_type1' (line 70)
    value_in_type1_2220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 7), 'value_in_type1')
    
    # Getting the type of 'value_in_type2' (line 70)
    value_in_type2_2221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 30), 'value_in_type2')
    # Applying the 'not' unary operator (line 70)
    result_not__2222 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 26), 'not', value_in_type2_2221)
    
    # Applying the binary operator 'and' (line 70)
    result_and_keyword_2223 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 7), 'and', value_in_type1_2220, result_not__2222)
    
    # Testing if the type of an if condition is none (line 70)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 4), result_and_keyword_2223):
        pass
    else:
        
        # Testing the type of an if condition (line 70)
        if_condition_2224 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 4), result_and_keyword_2223)
        # Assigning a type to the variable 'if_condition_2224' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'if_condition_2224', if_condition_2224)
        # SSA begins for if statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove(...): (line 71)
        # Processing the call arguments (line 71)
        str_2227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'str', 'value')
        # Processing the call keyword arguments (line 71)
        kwargs_2228 = {}
        # Getting the type of 'type1_members' (line 71)
        type1_members_2225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'type1_members', False)
        # Obtaining the member 'remove' of a type (line 71)
        remove_2226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), type1_members_2225, 'remove')
        # Calling remove(args, kwargs) (line 71)
        remove_call_result_2229 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), remove_2226, *[str_2227], **kwargs_2228)
        
        # SSA join for if statement (line 70)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Evaluating a boolean operation
    # Getting the type of 'value_in_type2' (line 73)
    value_in_type2_2230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 7), 'value_in_type2')
    
    # Getting the type of 'value_in_type1' (line 73)
    value_in_type1_2231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'value_in_type1')
    # Applying the 'not' unary operator (line 73)
    result_not__2232 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 26), 'not', value_in_type1_2231)
    
    # Applying the binary operator 'and' (line 73)
    result_and_keyword_2233 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 7), 'and', value_in_type2_2230, result_not__2232)
    
    # Testing if the type of an if condition is none (line 73)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 4), result_and_keyword_2233):
        pass
    else:
        
        # Testing the type of an if condition (line 73)
        if_condition_2234 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 4), result_and_keyword_2233)
        # Assigning a type to the variable 'if_condition_2234' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'if_condition_2234', if_condition_2234)
        # SSA begins for if statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove(...): (line 74)
        # Processing the call arguments (line 74)
        str_2237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 29), 'str', 'value')
        # Processing the call keyword arguments (line 74)
        kwargs_2238 = {}
        # Getting the type of 'type1_members' (line 74)
        type1_members_2235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'type1_members', False)
        # Obtaining the member 'remove' of a type (line 74)
        remove_2236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 8), type1_members_2235, 'remove')
        # Calling remove(args, kwargs) (line 74)
        remove_call_result_2239 = invoke(stypy.reporting.localization.Localization(__file__, 74, 8), remove_2236, *[str_2237], **kwargs_2238)
        
        # SSA join for if statement (line 73)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Compare to a Name (line 76):
    
    # Getting the type of 'type1_members' (line 76)
    type1_members_2240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 21), 'type1_members')
    # Getting the type of 'type2_members' (line 76)
    type2_members_2241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 38), 'type2_members')
    # Applying the binary operator '==' (line 76)
    result_eq_2242 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 21), '==', type1_members_2240, type2_members_2241)
    
    # Assigning a type to the variable 'same_structure' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'same_structure', result_eq_2242)
    
    # Getting the type of 'same_structure' (line 77)
    same_structure_2243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'same_structure')
    # Applying the 'not' unary operator (line 77)
    result_not__2244 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 7), 'not', same_structure_2243)
    
    # Testing if the type of an if condition is none (line 77)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 4), result_not__2244):
        pass
    else:
        
        # Testing the type of an if condition (line 77)
        if_condition_2245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 4), result_not__2244)
        # Assigning a type to the variable 'if_condition_2245' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'if_condition_2245', if_condition_2245)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 78)
        False_2246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'stypy_return_type', False_2246)
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'type1_members' (line 80)
    type1_members_2247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'type1_members')
    # Assigning a type to the variable 'type1_members_2247' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'type1_members_2247', type1_members_2247)
    # Testing if the for loop is going to be iterated (line 80)
    # Testing the type of a for loop iterable (line 80)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 80, 4), type1_members_2247)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 80, 4), type1_members_2247):
        # Getting the type of the for loop variable (line 80)
        for_loop_var_2248 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 80, 4), type1_members_2247)
        # Assigning a type to the variable 'member' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'member', for_loop_var_2248)
        # SSA begins for a for statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Getting the type of 'exclude_special_properties' (line 81)
        exclude_special_properties_2249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'exclude_special_properties')
        # Testing if the type of an if condition is none (line 81)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 81, 8), exclude_special_properties_2249):
            pass
        else:
            
            # Testing the type of an if condition (line 81)
            if_condition_2250 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), exclude_special_properties_2249)
            # Assigning a type to the variable 'if_condition_2250' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_2250', if_condition_2250)
            # SSA begins for if statement (line 81)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'member' (line 82)
            member_2251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'member')
            # Getting the type of 'Type' (line 82)
            Type_2252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'Type')
            # Obtaining the member 'special_properties_for_equality' of a type (line 82)
            special_properties_for_equality_2253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 25), Type_2252, 'special_properties_for_equality')
            # Applying the binary operator 'in' (line 82)
            result_contains_2254 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 15), 'in', member_2251, special_properties_for_equality_2253)
            
            # Testing if the type of an if condition is none (line 82)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 82, 12), result_contains_2254):
                pass
            else:
                
                # Testing the type of an if condition (line 82)
                if_condition_2255 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 12), result_contains_2254)
                # Assigning a type to the variable 'if_condition_2255' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'if_condition_2255', if_condition_2255)
                # SSA begins for if statement (line 82)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 82)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 81)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 86):
        
        # Call to getattr(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'type1' (line 86)
        type1_2257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 26), 'type1', False)
        # Getting the type of 'member' (line 86)
        member_2258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 33), 'member', False)
        # Processing the call keyword arguments (line 86)
        kwargs_2259 = {}
        # Getting the type of 'getattr' (line 86)
        getattr_2256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 86)
        getattr_call_result_2260 = invoke(stypy.reporting.localization.Localization(__file__, 86, 18), getattr_2256, *[type1_2257, member_2258], **kwargs_2259)
        
        # Assigning a type to the variable 'member1' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'member1', getattr_call_result_2260)
        
        # Assigning a Call to a Name (line 87):
        
        # Call to getattr(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'type2' (line 87)
        type2_2262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 26), 'type2', False)
        # Getting the type of 'member' (line 87)
        member_2263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 33), 'member', False)
        # Processing the call keyword arguments (line 87)
        kwargs_2264 = {}
        # Getting the type of 'getattr' (line 87)
        getattr_2261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 18), 'getattr', False)
        # Calling getattr(args, kwargs) (line 87)
        getattr_call_result_2265 = invoke(stypy.reporting.localization.Localization(__file__, 87, 18), getattr_2261, *[type2_2262, member_2263], **kwargs_2264)
        
        # Assigning a type to the variable 'member2' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'member2', getattr_call_result_2265)
        
        # Call to isinstance(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'member1' (line 90)
        member1_2267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 22), 'member1', False)
        # Getting the type of 'Type' (line 90)
        Type_2268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 31), 'Type', False)
        # Processing the call keyword arguments (line 90)
        kwargs_2269 = {}
        # Getting the type of 'isinstance' (line 90)
        isinstance_2266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 90)
        isinstance_call_result_2270 = invoke(stypy.reporting.localization.Localization(__file__, 90, 11), isinstance_2266, *[member1_2267, Type_2268], **kwargs_2269)
        
        # Testing if the type of an if condition is none (line 90)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 90, 8), isinstance_call_result_2270):
            
            # Type idiom detected: calculating its left and rigth part (line 95)
            # Getting the type of 'member1' (line 95)
            member1_2284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'member1')
            
            # Call to type(...): (line 95)
            # Processing the call arguments (line 95)
            # Getting the type of 'member2' (line 95)
            member2_2286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 41), 'member2', False)
            # Processing the call keyword arguments (line 95)
            kwargs_2287 = {}
            # Getting the type of 'type' (line 95)
            type_2285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'type', False)
            # Calling type(args, kwargs) (line 95)
            type_call_result_2288 = invoke(stypy.reporting.localization.Localization(__file__, 95, 36), type_2285, *[member2_2286], **kwargs_2287)
            
            
            (may_be_2289, more_types_in_union_2290) = may_not_be_type(member1_2284, type_call_result_2288)

            if may_be_2289:

                if more_types_in_union_2290:
                    # Runtime conditional SSA (line 95)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'member1' (line 95)
                member1_2291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'member1')
                # Assigning a type to the variable 'member1' (line 95)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'member1', remove_type_from_union(member1_2291, type_call_result_2288))
                # Getting the type of 'False' (line 96)
                False_2292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 96)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'stypy_return_type', False_2292)

                if more_types_in_union_2290:
                    # SSA join for if statement (line 95)
                    module_type_store = module_type_store.join_ssa_context()


            
        else:
            
            # Testing the type of an if condition (line 90)
            if_condition_2271 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 8), isinstance_call_result_2270)
            # Assigning a type to the variable 'if_condition_2271' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'if_condition_2271', if_condition_2271)
            # SSA begins for if statement (line 90)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            
            # Call to get_python_type(...): (line 91)
            # Processing the call keyword arguments (line 91)
            kwargs_2274 = {}
            # Getting the type of 'member1' (line 91)
            member1_2272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'member1', False)
            # Obtaining the member 'get_python_type' of a type (line 91)
            get_python_type_2273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 19), member1_2272, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 91)
            get_python_type_call_result_2275 = invoke(stypy.reporting.localization.Localization(__file__, 91, 19), get_python_type_2273, *[], **kwargs_2274)
            
            
            # Call to get_python_type(...): (line 91)
            # Processing the call keyword arguments (line 91)
            kwargs_2278 = {}
            # Getting the type of 'member2' (line 91)
            member2_2276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 48), 'member2', False)
            # Obtaining the member 'get_python_type' of a type (line 91)
            get_python_type_2277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 48), member2_2276, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 91)
            get_python_type_call_result_2279 = invoke(stypy.reporting.localization.Localization(__file__, 91, 48), get_python_type_2277, *[], **kwargs_2278)
            
            # Applying the binary operator '==' (line 91)
            result_eq_2280 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 19), '==', get_python_type_call_result_2275, get_python_type_call_result_2279)
            
            # Applying the 'not' unary operator (line 91)
            result_not__2281 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 15), 'not', result_eq_2280)
            
            # Testing if the type of an if condition is none (line 91)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 91, 12), result_not__2281):
                pass
            else:
                
                # Testing the type of an if condition (line 91)
                if_condition_2282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 12), result_not__2281)
                # Assigning a type to the variable 'if_condition_2282' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'if_condition_2282', if_condition_2282)
                # SSA begins for if statement (line 91)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 92)
                False_2283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 92)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'stypy_return_type', False_2283)
                # SSA join for if statement (line 91)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA branch for the else part of an if statement (line 90)
            module_type_store.open_ssa_branch('else')
            
            # Type idiom detected: calculating its left and rigth part (line 95)
            # Getting the type of 'member1' (line 95)
            member1_2284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'member1')
            
            # Call to type(...): (line 95)
            # Processing the call arguments (line 95)
            # Getting the type of 'member2' (line 95)
            member2_2286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 41), 'member2', False)
            # Processing the call keyword arguments (line 95)
            kwargs_2287 = {}
            # Getting the type of 'type' (line 95)
            type_2285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'type', False)
            # Calling type(args, kwargs) (line 95)
            type_call_result_2288 = invoke(stypy.reporting.localization.Localization(__file__, 95, 36), type_2285, *[member2_2286], **kwargs_2287)
            
            
            (may_be_2289, more_types_in_union_2290) = may_not_be_type(member1_2284, type_call_result_2288)

            if may_be_2289:

                if more_types_in_union_2290:
                    # Runtime conditional SSA (line 95)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'member1' (line 95)
                member1_2291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'member1')
                # Assigning a type to the variable 'member1' (line 95)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'member1', remove_type_from_union(member1_2291, type_call_result_2288))
                # Getting the type of 'False' (line 96)
                False_2292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 96)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'stypy_return_type', False_2292)

                if more_types_in_union_2290:
                    # SSA join for if statement (line 95)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 90)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 150)
    True_2293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type', True_2293)
    
    # ################# End of 'structural_equivalence(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'structural_equivalence' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_2294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2294)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'structural_equivalence'
    return stypy_return_type_2294

# Assigning a type to the variable 'structural_equivalence' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'structural_equivalence', structural_equivalence)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
