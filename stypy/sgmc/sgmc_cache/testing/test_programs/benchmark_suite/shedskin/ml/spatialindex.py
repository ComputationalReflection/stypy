
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #  MiniLight Python : minimal global illumination renderer
2: #
3: #  Copyright (c) 2007-2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
4: #  http://www.hxa7241.org/
5: 
6: 
7: from triangle import Triangle, TOLERANCE
8: from vector3f import Vector3f_seq, Vector3f_scalar, MAX
9: 
10: MAX_LEVELS = 44
11: MAX_ITEMS  =  8
12: 
13: class SpatialIndex(object):
14: 
15:     def __init__(self, vect, bound, items, level=0):
16:         if vect:
17:             for item in items:
18:                 item.bound = item.get_bound()
19:             bound = vect.as_list() * 2
20:             for item in items:
21:                 for j in range(6):
22:                     if (bound[j] > item.bound[j]) ^ (j > 2):
23:                         bound[j] = item.bound[j]
24:             size = max((Vector3f_seq(bound[3:6]) - Vector3f_seq(bound[0:3])).as_list())
25:             self.bound = bound[0:3] + (Vector3f_seq(bound[3:6]).clamped(Vector3f_seq(bound[0:3]) + Vector3f_scalar(size), MAX)).as_list()
26:         else:
27:             self.bound = bound
28:         self.is_branch = len(items) > MAX_ITEMS and level < MAX_LEVELS - 1
29:         if self.is_branch:
30:             q1 = 0
31:             self.vector = [None] * 8
32:             for s in range(8):
33:                 sub_bound = []
34:                 for j in range(6):
35:                     m = j % 3
36:                     if (((s >> m) & 1) != 0) ^ (j > 2):
37:                         sub_bound.append((self.bound[m] + self.bound[m + 3]) * 0.5)
38:                     else:
39:                         sub_bound.append(self.bound[j])
40:                 sub_items = []
41:                 for item in items:
42:                     if item.bound[3] >= sub_bound[0] and item.bound[0] < sub_bound[3] and \
43:                        item.bound[4] >= sub_bound[1] and item.bound[1] < sub_bound[4] and \
44:                        item.bound[5] >= sub_bound[2] and item.bound[2] < sub_bound[5]:
45:                            sub_items.append(item)
46:                 q1 += 1 if len(sub_items) == len(items) else 0
47:                 q2 = (sub_bound[3] - sub_bound[0]) < (TOLERANCE * 4.0)
48:                 if len(sub_items) > 0:
49:                     self.vector[s] = SpatialIndex(None, sub_bound, sub_items, MAX_LEVELS if q1 > 1 or q2 else level + 1)
50:         else:
51:             self.items = items
52: 
53:     def get_intersection(self, ray_origin, ray_direction, last_hit, start=None):
54:         start = start if start else ray_origin
55:         hit_object = hit_position = None
56:         b0, b1, b2, b3, b4, b5 = self.bound
57:         if self.is_branch:
58:             sub_cell = 1 if start.x >= (b0+b3) * 0.5 else 0
59:             if start.y >= (b1+b4) * 0.5:
60:                 sub_cell |= 2
61:             if start.z >= (b2+b5) * 0.5:
62:                 sub_cell |= 4
63:             cell_position = start
64:             while True:
65:                 if self.vector[sub_cell] != None:
66:                     hit_object, hit_position = self.vector[sub_cell].get_intersection(ray_origin, ray_direction, last_hit, cell_position)
67:                     if hit_object != None:
68:                         break
69:                 step = 1.797e308
70:                 axis = 0
71:                 for i in range(3):
72:                     high = (sub_cell >> i) & 1
73:                     face = self.bound[i + high * 3] if (ray_direction[i] < 0.0) ^ (0 != high) else (self.bound[i] + self.bound[i + 3]) * 0.5
74:                     try:
75:                         distance = (face - ray_origin[i]) / ray_direction[i]
76:                     except:
77:                         distance = float(1e30000)
78:                     if distance <= step:
79:                         step = distance
80:                         axis = i
81:                 if (((sub_cell >> axis) & 1) == 1) ^ (ray_direction[axis] < 0.0):
82:                     break
83:                 cell_position = ray_origin + ray_direction * step
84:                 sub_cell = sub_cell ^ (1 << axis)
85:         else:
86:             nearest_distance = 1.797e308
87:             for item in self.items:
88:                 if item != last_hit:
89:                     distance = item.get_intersection(ray_origin, ray_direction)
90:                     if 0.0 <= distance < nearest_distance:
91:                         hit = ray_origin + ray_direction * distance
92:                         if (b0 - hit.x <= TOLERANCE) and \
93:                            (hit.x - b3 <= TOLERANCE) and \
94:                            (b1 - hit.y <= TOLERANCE) and \
95:                            (hit.y - b4 <= TOLERANCE) and \
96:                            (b2 - hit.z <= TOLERANCE) and \
97:                            (hit.z - b5 <= TOLERANCE):
98:                                hit_object = item
99:                                hit_position = hit
100:                                nearest_distance = distance
101:         return hit_object, hit_position
102: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from triangle import Triangle, TOLERANCE' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_1090 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'triangle')

if (type(import_1090) is not StypyTypeError):

    if (import_1090 != 'pyd_module'):
        __import__(import_1090)
        sys_modules_1091 = sys.modules[import_1090]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'triangle', sys_modules_1091.module_type_store, module_type_store, ['Triangle', 'TOLERANCE'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_1091, sys_modules_1091.module_type_store, module_type_store)
    else:
        from triangle import Triangle, TOLERANCE

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'triangle', None, module_type_store, ['Triangle', 'TOLERANCE'], [Triangle, TOLERANCE])

else:
    # Assigning a type to the variable 'triangle' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'triangle', import_1090)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from vector3f import Vector3f_seq, Vector3f_scalar, MAX' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_1092 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f')

if (type(import_1092) is not StypyTypeError):

    if (import_1092 != 'pyd_module'):
        __import__(import_1092)
        sys_modules_1093 = sys.modules[import_1092]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f', sys_modules_1093.module_type_store, module_type_store, ['Vector3f_seq', 'Vector3f_scalar', 'MAX'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_1093, sys_modules_1093.module_type_store, module_type_store)
    else:
        from vector3f import Vector3f_seq, Vector3f_scalar, MAX

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f', None, module_type_store, ['Vector3f_seq', 'Vector3f_scalar', 'MAX'], [Vector3f_seq, Vector3f_scalar, MAX])

else:
    # Assigning a type to the variable 'vector3f' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f', import_1092)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')


# Assigning a Num to a Name (line 10):

# Assigning a Num to a Name (line 10):
int_1094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'int')
# Assigning a type to the variable 'MAX_LEVELS' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'MAX_LEVELS', int_1094)

# Assigning a Num to a Name (line 11):

# Assigning a Num to a Name (line 11):
int_1095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'int')
# Assigning a type to the variable 'MAX_ITEMS' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'MAX_ITEMS', int_1095)
# Declaration of the 'SpatialIndex' class

class SpatialIndex(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_1096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 49), 'int')
        defaults = [int_1096]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SpatialIndex.__init__', ['vect', 'bound', 'items', 'level'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['vect', 'bound', 'items', 'level'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        # Getting the type of 'vect' (line 16)
        vect_1097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'vect')
        # Testing if the type of an if condition is none (line 16)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 16, 8), vect_1097):
            
            # Assigning a Name to a Attribute (line 27):
            
            # Assigning a Name to a Attribute (line 27):
            # Getting the type of 'bound' (line 27)
            bound_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'bound')
            # Getting the type of 'self' (line 27)
            self_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'self')
            # Setting the type of the member 'bound' of a type (line 27)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), self_1205, 'bound', bound_1204)
        else:
            
            # Testing the type of an if condition (line 16)
            if_condition_1098 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 16, 8), vect_1097)
            # Assigning a type to the variable 'if_condition_1098' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'if_condition_1098', if_condition_1098)
            # SSA begins for if statement (line 16)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'items' (line 17)
            items_1099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'items')
            # Testing if the for loop is going to be iterated (line 17)
            # Testing the type of a for loop iterable (line 17)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 17, 12), items_1099)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 17, 12), items_1099):
                # Getting the type of the for loop variable (line 17)
                for_loop_var_1100 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 17, 12), items_1099)
                # Assigning a type to the variable 'item' (line 17)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'item', for_loop_var_1100)
                # SSA begins for a for statement (line 17)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Attribute (line 18):
                
                # Assigning a Call to a Attribute (line 18):
                
                # Call to get_bound(...): (line 18)
                # Processing the call keyword arguments (line 18)
                kwargs_1103 = {}
                # Getting the type of 'item' (line 18)
                item_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 29), 'item', False)
                # Obtaining the member 'get_bound' of a type (line 18)
                get_bound_1102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 29), item_1101, 'get_bound')
                # Calling get_bound(args, kwargs) (line 18)
                get_bound_call_result_1104 = invoke(stypy.reporting.localization.Localization(__file__, 18, 29), get_bound_1102, *[], **kwargs_1103)
                
                # Getting the type of 'item' (line 18)
                item_1105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'item')
                # Setting the type of the member 'bound' of a type (line 18)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 16), item_1105, 'bound', get_bound_call_result_1104)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a BinOp to a Name (line 19):
            
            # Assigning a BinOp to a Name (line 19):
            
            # Call to as_list(...): (line 19)
            # Processing the call keyword arguments (line 19)
            kwargs_1108 = {}
            # Getting the type of 'vect' (line 19)
            vect_1106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'vect', False)
            # Obtaining the member 'as_list' of a type (line 19)
            as_list_1107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 20), vect_1106, 'as_list')
            # Calling as_list(args, kwargs) (line 19)
            as_list_call_result_1109 = invoke(stypy.reporting.localization.Localization(__file__, 19, 20), as_list_1107, *[], **kwargs_1108)
            
            int_1110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 37), 'int')
            # Applying the binary operator '*' (line 19)
            result_mul_1111 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 20), '*', as_list_call_result_1109, int_1110)
            
            # Assigning a type to the variable 'bound' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'bound', result_mul_1111)
            
            # Getting the type of 'items' (line 20)
            items_1112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 24), 'items')
            # Testing if the for loop is going to be iterated (line 20)
            # Testing the type of a for loop iterable (line 20)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 20, 12), items_1112)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 20, 12), items_1112):
                # Getting the type of the for loop variable (line 20)
                for_loop_var_1113 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 20, 12), items_1112)
                # Assigning a type to the variable 'item' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'item', for_loop_var_1113)
                # SSA begins for a for statement (line 20)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 21)
                # Processing the call arguments (line 21)
                int_1115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 31), 'int')
                # Processing the call keyword arguments (line 21)
                kwargs_1116 = {}
                # Getting the type of 'range' (line 21)
                range_1114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 25), 'range', False)
                # Calling range(args, kwargs) (line 21)
                range_call_result_1117 = invoke(stypy.reporting.localization.Localization(__file__, 21, 25), range_1114, *[int_1115], **kwargs_1116)
                
                # Testing if the for loop is going to be iterated (line 21)
                # Testing the type of a for loop iterable (line 21)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 21, 16), range_call_result_1117)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 21, 16), range_call_result_1117):
                    # Getting the type of the for loop variable (line 21)
                    for_loop_var_1118 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 21, 16), range_call_result_1117)
                    # Assigning a type to the variable 'j' (line 21)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'j', for_loop_var_1118)
                    # SSA begins for a for statement (line 21)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'j' (line 22)
                    j_1119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), 'j')
                    # Getting the type of 'bound' (line 22)
                    bound_1120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'bound')
                    # Obtaining the member '__getitem__' of a type (line 22)
                    getitem___1121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 24), bound_1120, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
                    subscript_call_result_1122 = invoke(stypy.reporting.localization.Localization(__file__, 22, 24), getitem___1121, j_1119)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'j' (line 22)
                    j_1123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 46), 'j')
                    # Getting the type of 'item' (line 22)
                    item_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 35), 'item')
                    # Obtaining the member 'bound' of a type (line 22)
                    bound_1125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 35), item_1124, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 22)
                    getitem___1126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 35), bound_1125, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
                    subscript_call_result_1127 = invoke(stypy.reporting.localization.Localization(__file__, 22, 35), getitem___1126, j_1123)
                    
                    # Applying the binary operator '>' (line 22)
                    result_gt_1128 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 24), '>', subscript_call_result_1122, subscript_call_result_1127)
                    
                    
                    # Getting the type of 'j' (line 22)
                    j_1129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 53), 'j')
                    int_1130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 57), 'int')
                    # Applying the binary operator '>' (line 22)
                    result_gt_1131 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 53), '>', j_1129, int_1130)
                    
                    # Applying the binary operator '^' (line 22)
                    result_xor_1132 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 23), '^', result_gt_1128, result_gt_1131)
                    
                    # Testing if the type of an if condition is none (line 22)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 22, 20), result_xor_1132):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 22)
                        if_condition_1133 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 20), result_xor_1132)
                        # Assigning a type to the variable 'if_condition_1133' (line 22)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'if_condition_1133', if_condition_1133)
                        # SSA begins for if statement (line 22)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Subscript to a Subscript (line 23):
                        
                        # Assigning a Subscript to a Subscript (line 23):
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'j' (line 23)
                        j_1134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 46), 'j')
                        # Getting the type of 'item' (line 23)
                        item_1135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 35), 'item')
                        # Obtaining the member 'bound' of a type (line 23)
                        bound_1136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 35), item_1135, 'bound')
                        # Obtaining the member '__getitem__' of a type (line 23)
                        getitem___1137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 35), bound_1136, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 23)
                        subscript_call_result_1138 = invoke(stypy.reporting.localization.Localization(__file__, 23, 35), getitem___1137, j_1134)
                        
                        # Getting the type of 'bound' (line 23)
                        bound_1139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'bound')
                        # Getting the type of 'j' (line 23)
                        j_1140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 'j')
                        # Storing an element on a container (line 23)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 24), bound_1139, (j_1140, subscript_call_result_1138))
                        # SSA join for if statement (line 22)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Call to a Name (line 24):
            
            # Assigning a Call to a Name (line 24):
            
            # Call to max(...): (line 24)
            # Processing the call arguments (line 24)
            
            # Call to as_list(...): (line 24)
            # Processing the call keyword arguments (line 24)
            kwargs_1162 = {}
            
            # Call to Vector3f_seq(...): (line 24)
            # Processing the call arguments (line 24)
            
            # Obtaining the type of the subscript
            int_1143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 43), 'int')
            int_1144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 45), 'int')
            slice_1145 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 24, 37), int_1143, int_1144, None)
            # Getting the type of 'bound' (line 24)
            bound_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 37), 'bound', False)
            # Obtaining the member '__getitem__' of a type (line 24)
            getitem___1147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 37), bound_1146, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 24)
            subscript_call_result_1148 = invoke(stypy.reporting.localization.Localization(__file__, 24, 37), getitem___1147, slice_1145)
            
            # Processing the call keyword arguments (line 24)
            kwargs_1149 = {}
            # Getting the type of 'Vector3f_seq' (line 24)
            Vector3f_seq_1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'Vector3f_seq', False)
            # Calling Vector3f_seq(args, kwargs) (line 24)
            Vector3f_seq_call_result_1150 = invoke(stypy.reporting.localization.Localization(__file__, 24, 24), Vector3f_seq_1142, *[subscript_call_result_1148], **kwargs_1149)
            
            
            # Call to Vector3f_seq(...): (line 24)
            # Processing the call arguments (line 24)
            
            # Obtaining the type of the subscript
            int_1152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 70), 'int')
            int_1153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 72), 'int')
            slice_1154 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 24, 64), int_1152, int_1153, None)
            # Getting the type of 'bound' (line 24)
            bound_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 64), 'bound', False)
            # Obtaining the member '__getitem__' of a type (line 24)
            getitem___1156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 64), bound_1155, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 24)
            subscript_call_result_1157 = invoke(stypy.reporting.localization.Localization(__file__, 24, 64), getitem___1156, slice_1154)
            
            # Processing the call keyword arguments (line 24)
            kwargs_1158 = {}
            # Getting the type of 'Vector3f_seq' (line 24)
            Vector3f_seq_1151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 51), 'Vector3f_seq', False)
            # Calling Vector3f_seq(args, kwargs) (line 24)
            Vector3f_seq_call_result_1159 = invoke(stypy.reporting.localization.Localization(__file__, 24, 51), Vector3f_seq_1151, *[subscript_call_result_1157], **kwargs_1158)
            
            # Applying the binary operator '-' (line 24)
            result_sub_1160 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 24), '-', Vector3f_seq_call_result_1150, Vector3f_seq_call_result_1159)
            
            # Obtaining the member 'as_list' of a type (line 24)
            as_list_1161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 24), result_sub_1160, 'as_list')
            # Calling as_list(args, kwargs) (line 24)
            as_list_call_result_1163 = invoke(stypy.reporting.localization.Localization(__file__, 24, 24), as_list_1161, *[], **kwargs_1162)
            
            # Processing the call keyword arguments (line 24)
            kwargs_1164 = {}
            # Getting the type of 'max' (line 24)
            max_1141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'max', False)
            # Calling max(args, kwargs) (line 24)
            max_call_result_1165 = invoke(stypy.reporting.localization.Localization(__file__, 24, 19), max_1141, *[as_list_call_result_1163], **kwargs_1164)
            
            # Assigning a type to the variable 'size' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'size', max_call_result_1165)
            
            # Assigning a BinOp to a Attribute (line 25):
            
            # Assigning a BinOp to a Attribute (line 25):
            
            # Obtaining the type of the subscript
            int_1166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'int')
            int_1167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 33), 'int')
            slice_1168 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 25), int_1166, int_1167, None)
            # Getting the type of 'bound' (line 25)
            bound_1169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'bound')
            # Obtaining the member '__getitem__' of a type (line 25)
            getitem___1170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 25), bound_1169, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 25)
            subscript_call_result_1171 = invoke(stypy.reporting.localization.Localization(__file__, 25, 25), getitem___1170, slice_1168)
            
            
            # Call to as_list(...): (line 25)
            # Processing the call keyword arguments (line 25)
            kwargs_1200 = {}
            
            # Call to clamped(...): (line 25)
            # Processing the call arguments (line 25)
            
            # Call to Vector3f_seq(...): (line 25)
            # Processing the call arguments (line 25)
            
            # Obtaining the type of the subscript
            int_1183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 91), 'int')
            int_1184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 93), 'int')
            slice_1185 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 85), int_1183, int_1184, None)
            # Getting the type of 'bound' (line 25)
            bound_1186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 85), 'bound', False)
            # Obtaining the member '__getitem__' of a type (line 25)
            getitem___1187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 85), bound_1186, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 25)
            subscript_call_result_1188 = invoke(stypy.reporting.localization.Localization(__file__, 25, 85), getitem___1187, slice_1185)
            
            # Processing the call keyword arguments (line 25)
            kwargs_1189 = {}
            # Getting the type of 'Vector3f_seq' (line 25)
            Vector3f_seq_1182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 72), 'Vector3f_seq', False)
            # Calling Vector3f_seq(args, kwargs) (line 25)
            Vector3f_seq_call_result_1190 = invoke(stypy.reporting.localization.Localization(__file__, 25, 72), Vector3f_seq_1182, *[subscript_call_result_1188], **kwargs_1189)
            
            
            # Call to Vector3f_scalar(...): (line 25)
            # Processing the call arguments (line 25)
            # Getting the type of 'size' (line 25)
            size_1192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 115), 'size', False)
            # Processing the call keyword arguments (line 25)
            kwargs_1193 = {}
            # Getting the type of 'Vector3f_scalar' (line 25)
            Vector3f_scalar_1191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 99), 'Vector3f_scalar', False)
            # Calling Vector3f_scalar(args, kwargs) (line 25)
            Vector3f_scalar_call_result_1194 = invoke(stypy.reporting.localization.Localization(__file__, 25, 99), Vector3f_scalar_1191, *[size_1192], **kwargs_1193)
            
            # Applying the binary operator '+' (line 25)
            result_add_1195 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 72), '+', Vector3f_seq_call_result_1190, Vector3f_scalar_call_result_1194)
            
            # Getting the type of 'MAX' (line 25)
            MAX_1196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 122), 'MAX', False)
            # Processing the call keyword arguments (line 25)
            kwargs_1197 = {}
            
            # Call to Vector3f_seq(...): (line 25)
            # Processing the call arguments (line 25)
            
            # Obtaining the type of the subscript
            int_1173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 58), 'int')
            int_1174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 60), 'int')
            slice_1175 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 52), int_1173, int_1174, None)
            # Getting the type of 'bound' (line 25)
            bound_1176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 52), 'bound', False)
            # Obtaining the member '__getitem__' of a type (line 25)
            getitem___1177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 52), bound_1176, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 25)
            subscript_call_result_1178 = invoke(stypy.reporting.localization.Localization(__file__, 25, 52), getitem___1177, slice_1175)
            
            # Processing the call keyword arguments (line 25)
            kwargs_1179 = {}
            # Getting the type of 'Vector3f_seq' (line 25)
            Vector3f_seq_1172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'Vector3f_seq', False)
            # Calling Vector3f_seq(args, kwargs) (line 25)
            Vector3f_seq_call_result_1180 = invoke(stypy.reporting.localization.Localization(__file__, 25, 39), Vector3f_seq_1172, *[subscript_call_result_1178], **kwargs_1179)
            
            # Obtaining the member 'clamped' of a type (line 25)
            clamped_1181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 39), Vector3f_seq_call_result_1180, 'clamped')
            # Calling clamped(args, kwargs) (line 25)
            clamped_call_result_1198 = invoke(stypy.reporting.localization.Localization(__file__, 25, 39), clamped_1181, *[result_add_1195, MAX_1196], **kwargs_1197)
            
            # Obtaining the member 'as_list' of a type (line 25)
            as_list_1199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 39), clamped_call_result_1198, 'as_list')
            # Calling as_list(args, kwargs) (line 25)
            as_list_call_result_1201 = invoke(stypy.reporting.localization.Localization(__file__, 25, 39), as_list_1199, *[], **kwargs_1200)
            
            # Applying the binary operator '+' (line 25)
            result_add_1202 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 25), '+', subscript_call_result_1171, as_list_call_result_1201)
            
            # Getting the type of 'self' (line 25)
            self_1203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'self')
            # Setting the type of the member 'bound' of a type (line 25)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), self_1203, 'bound', result_add_1202)
            # SSA branch for the else part of an if statement (line 16)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Attribute (line 27):
            
            # Assigning a Name to a Attribute (line 27):
            # Getting the type of 'bound' (line 27)
            bound_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'bound')
            # Getting the type of 'self' (line 27)
            self_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'self')
            # Setting the type of the member 'bound' of a type (line 27)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), self_1205, 'bound', bound_1204)
            # SSA join for if statement (line 16)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BoolOp to a Attribute (line 28):
        
        # Assigning a BoolOp to a Attribute (line 28):
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'items' (line 28)
        items_1207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'items', False)
        # Processing the call keyword arguments (line 28)
        kwargs_1208 = {}
        # Getting the type of 'len' (line 28)
        len_1206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 25), 'len', False)
        # Calling len(args, kwargs) (line 28)
        len_call_result_1209 = invoke(stypy.reporting.localization.Localization(__file__, 28, 25), len_1206, *[items_1207], **kwargs_1208)
        
        # Getting the type of 'MAX_ITEMS' (line 28)
        MAX_ITEMS_1210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 38), 'MAX_ITEMS')
        # Applying the binary operator '>' (line 28)
        result_gt_1211 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 25), '>', len_call_result_1209, MAX_ITEMS_1210)
        
        
        # Getting the type of 'level' (line 28)
        level_1212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 52), 'level')
        # Getting the type of 'MAX_LEVELS' (line 28)
        MAX_LEVELS_1213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 60), 'MAX_LEVELS')
        int_1214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 73), 'int')
        # Applying the binary operator '-' (line 28)
        result_sub_1215 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 60), '-', MAX_LEVELS_1213, int_1214)
        
        # Applying the binary operator '<' (line 28)
        result_lt_1216 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 52), '<', level_1212, result_sub_1215)
        
        # Applying the binary operator 'and' (line 28)
        result_and_keyword_1217 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 25), 'and', result_gt_1211, result_lt_1216)
        
        # Getting the type of 'self' (line 28)
        self_1218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'is_branch' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_1218, 'is_branch', result_and_keyword_1217)
        # Getting the type of 'self' (line 29)
        self_1219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'self')
        # Obtaining the member 'is_branch' of a type (line 29)
        is_branch_1220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 11), self_1219, 'is_branch')
        # Testing if the type of an if condition is none (line 29)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 8), is_branch_1220):
            
            # Assigning a Name to a Attribute (line 51):
            
            # Assigning a Name to a Attribute (line 51):
            # Getting the type of 'items' (line 51)
            items_1409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'items')
            # Getting the type of 'self' (line 51)
            self_1410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
            # Setting the type of the member 'items' of a type (line 51)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_1410, 'items', items_1409)
        else:
            
            # Testing the type of an if condition (line 29)
            if_condition_1221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 8), is_branch_1220)
            # Assigning a type to the variable 'if_condition_1221' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'if_condition_1221', if_condition_1221)
            # SSA begins for if statement (line 29)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Name (line 30):
            
            # Assigning a Num to a Name (line 30):
            int_1222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'int')
            # Assigning a type to the variable 'q1' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'q1', int_1222)
            
            # Assigning a BinOp to a Attribute (line 31):
            
            # Assigning a BinOp to a Attribute (line 31):
            
            # Obtaining an instance of the builtin type 'list' (line 31)
            list_1223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'list')
            # Adding type elements to the builtin type 'list' instance (line 31)
            # Adding element type (line 31)
            # Getting the type of 'None' (line 31)
            None_1224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 27), 'None')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 26), list_1223, None_1224)
            
            int_1225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'int')
            # Applying the binary operator '*' (line 31)
            result_mul_1226 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 26), '*', list_1223, int_1225)
            
            # Getting the type of 'self' (line 31)
            self_1227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'self')
            # Setting the type of the member 'vector' of a type (line 31)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), self_1227, 'vector', result_mul_1226)
            
            
            # Call to range(...): (line 32)
            # Processing the call arguments (line 32)
            int_1229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'int')
            # Processing the call keyword arguments (line 32)
            kwargs_1230 = {}
            # Getting the type of 'range' (line 32)
            range_1228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 21), 'range', False)
            # Calling range(args, kwargs) (line 32)
            range_call_result_1231 = invoke(stypy.reporting.localization.Localization(__file__, 32, 21), range_1228, *[int_1229], **kwargs_1230)
            
            # Testing if the for loop is going to be iterated (line 32)
            # Testing the type of a for loop iterable (line 32)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 32, 12), range_call_result_1231)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 32, 12), range_call_result_1231):
                # Getting the type of the for loop variable (line 32)
                for_loop_var_1232 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 32, 12), range_call_result_1231)
                # Assigning a type to the variable 's' (line 32)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 's', for_loop_var_1232)
                # SSA begins for a for statement (line 32)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a List to a Name (line 33):
                
                # Assigning a List to a Name (line 33):
                
                # Obtaining an instance of the builtin type 'list' (line 33)
                list_1233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'list')
                # Adding type elements to the builtin type 'list' instance (line 33)
                
                # Assigning a type to the variable 'sub_bound' (line 33)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'sub_bound', list_1233)
                
                
                # Call to range(...): (line 34)
                # Processing the call arguments (line 34)
                int_1235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'int')
                # Processing the call keyword arguments (line 34)
                kwargs_1236 = {}
                # Getting the type of 'range' (line 34)
                range_1234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'range', False)
                # Calling range(args, kwargs) (line 34)
                range_call_result_1237 = invoke(stypy.reporting.localization.Localization(__file__, 34, 25), range_1234, *[int_1235], **kwargs_1236)
                
                # Testing if the for loop is going to be iterated (line 34)
                # Testing the type of a for loop iterable (line 34)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 16), range_call_result_1237)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 34, 16), range_call_result_1237):
                    # Getting the type of the for loop variable (line 34)
                    for_loop_var_1238 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 16), range_call_result_1237)
                    # Assigning a type to the variable 'j' (line 34)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'j', for_loop_var_1238)
                    # SSA begins for a for statement (line 34)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a BinOp to a Name (line 35):
                    
                    # Assigning a BinOp to a Name (line 35):
                    # Getting the type of 'j' (line 35)
                    j_1239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'j')
                    int_1240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'int')
                    # Applying the binary operator '%' (line 35)
                    result_mod_1241 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 24), '%', j_1239, int_1240)
                    
                    # Assigning a type to the variable 'm' (line 35)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'm', result_mod_1241)
                    
                    # Getting the type of 's' (line 36)
                    s_1242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 26), 's')
                    # Getting the type of 'm' (line 36)
                    m_1243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'm')
                    # Applying the binary operator '>>' (line 36)
                    result_rshift_1244 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 26), '>>', s_1242, m_1243)
                    
                    int_1245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 36), 'int')
                    # Applying the binary operator '&' (line 36)
                    result_and__1246 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 25), '&', result_rshift_1244, int_1245)
                    
                    int_1247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 42), 'int')
                    # Applying the binary operator '!=' (line 36)
                    result_ne_1248 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 24), '!=', result_and__1246, int_1247)
                    
                    
                    # Getting the type of 'j' (line 36)
                    j_1249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 48), 'j')
                    int_1250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 52), 'int')
                    # Applying the binary operator '>' (line 36)
                    result_gt_1251 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 48), '>', j_1249, int_1250)
                    
                    # Applying the binary operator '^' (line 36)
                    result_xor_1252 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 23), '^', result_ne_1248, result_gt_1251)
                    
                    # Testing if the type of an if condition is none (line 36)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 36, 20), result_xor_1252):
                        
                        # Call to append(...): (line 39)
                        # Processing the call arguments (line 39)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'j' (line 39)
                        j_1275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 52), 'j', False)
                        # Getting the type of 'self' (line 39)
                        self_1276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 41), 'self', False)
                        # Obtaining the member 'bound' of a type (line 39)
                        bound_1277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 41), self_1276, 'bound')
                        # Obtaining the member '__getitem__' of a type (line 39)
                        getitem___1278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 41), bound_1277, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
                        subscript_call_result_1279 = invoke(stypy.reporting.localization.Localization(__file__, 39, 41), getitem___1278, j_1275)
                        
                        # Processing the call keyword arguments (line 39)
                        kwargs_1280 = {}
                        # Getting the type of 'sub_bound' (line 39)
                        sub_bound_1273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 24), 'sub_bound', False)
                        # Obtaining the member 'append' of a type (line 39)
                        append_1274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 24), sub_bound_1273, 'append')
                        # Calling append(args, kwargs) (line 39)
                        append_call_result_1281 = invoke(stypy.reporting.localization.Localization(__file__, 39, 24), append_1274, *[subscript_call_result_1279], **kwargs_1280)
                        
                    else:
                        
                        # Testing the type of an if condition (line 36)
                        if_condition_1253 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 20), result_xor_1252)
                        # Assigning a type to the variable 'if_condition_1253' (line 36)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'if_condition_1253', if_condition_1253)
                        # SSA begins for if statement (line 36)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 37)
                        # Processing the call arguments (line 37)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'm' (line 37)
                        m_1256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 53), 'm', False)
                        # Getting the type of 'self' (line 37)
                        self_1257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 42), 'self', False)
                        # Obtaining the member 'bound' of a type (line 37)
                        bound_1258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 42), self_1257, 'bound')
                        # Obtaining the member '__getitem__' of a type (line 37)
                        getitem___1259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 42), bound_1258, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 37)
                        subscript_call_result_1260 = invoke(stypy.reporting.localization.Localization(__file__, 37, 42), getitem___1259, m_1256)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'm' (line 37)
                        m_1261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 69), 'm', False)
                        int_1262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 73), 'int')
                        # Applying the binary operator '+' (line 37)
                        result_add_1263 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 69), '+', m_1261, int_1262)
                        
                        # Getting the type of 'self' (line 37)
                        self_1264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 58), 'self', False)
                        # Obtaining the member 'bound' of a type (line 37)
                        bound_1265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 58), self_1264, 'bound')
                        # Obtaining the member '__getitem__' of a type (line 37)
                        getitem___1266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 58), bound_1265, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 37)
                        subscript_call_result_1267 = invoke(stypy.reporting.localization.Localization(__file__, 37, 58), getitem___1266, result_add_1263)
                        
                        # Applying the binary operator '+' (line 37)
                        result_add_1268 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 42), '+', subscript_call_result_1260, subscript_call_result_1267)
                        
                        float_1269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 79), 'float')
                        # Applying the binary operator '*' (line 37)
                        result_mul_1270 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 41), '*', result_add_1268, float_1269)
                        
                        # Processing the call keyword arguments (line 37)
                        kwargs_1271 = {}
                        # Getting the type of 'sub_bound' (line 37)
                        sub_bound_1254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 24), 'sub_bound', False)
                        # Obtaining the member 'append' of a type (line 37)
                        append_1255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 24), sub_bound_1254, 'append')
                        # Calling append(args, kwargs) (line 37)
                        append_call_result_1272 = invoke(stypy.reporting.localization.Localization(__file__, 37, 24), append_1255, *[result_mul_1270], **kwargs_1271)
                        
                        # SSA branch for the else part of an if statement (line 36)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to append(...): (line 39)
                        # Processing the call arguments (line 39)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'j' (line 39)
                        j_1275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 52), 'j', False)
                        # Getting the type of 'self' (line 39)
                        self_1276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 41), 'self', False)
                        # Obtaining the member 'bound' of a type (line 39)
                        bound_1277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 41), self_1276, 'bound')
                        # Obtaining the member '__getitem__' of a type (line 39)
                        getitem___1278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 41), bound_1277, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
                        subscript_call_result_1279 = invoke(stypy.reporting.localization.Localization(__file__, 39, 41), getitem___1278, j_1275)
                        
                        # Processing the call keyword arguments (line 39)
                        kwargs_1280 = {}
                        # Getting the type of 'sub_bound' (line 39)
                        sub_bound_1273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 24), 'sub_bound', False)
                        # Obtaining the member 'append' of a type (line 39)
                        append_1274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 24), sub_bound_1273, 'append')
                        # Calling append(args, kwargs) (line 39)
                        append_call_result_1281 = invoke(stypy.reporting.localization.Localization(__file__, 39, 24), append_1274, *[subscript_call_result_1279], **kwargs_1280)
                        
                        # SSA join for if statement (line 36)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a List to a Name (line 40):
                
                # Assigning a List to a Name (line 40):
                
                # Obtaining an instance of the builtin type 'list' (line 40)
                list_1282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 28), 'list')
                # Adding type elements to the builtin type 'list' instance (line 40)
                
                # Assigning a type to the variable 'sub_items' (line 40)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'sub_items', list_1282)
                
                # Getting the type of 'items' (line 41)
                items_1283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'items')
                # Testing if the for loop is going to be iterated (line 41)
                # Testing the type of a for loop iterable (line 41)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 16), items_1283)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 41, 16), items_1283):
                    # Getting the type of the for loop variable (line 41)
                    for_loop_var_1284 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 16), items_1283)
                    # Assigning a type to the variable 'item' (line 41)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'item', for_loop_var_1284)
                    # SSA begins for a for statement (line 41)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Evaluating a boolean operation
                    
                    
                    # Obtaining the type of the subscript
                    int_1285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 34), 'int')
                    # Getting the type of 'item' (line 42)
                    item_1286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'item')
                    # Obtaining the member 'bound' of a type (line 42)
                    bound_1287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 23), item_1286, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 42)
                    getitem___1288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 23), bound_1287, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                    subscript_call_result_1289 = invoke(stypy.reporting.localization.Localization(__file__, 42, 23), getitem___1288, int_1285)
                    
                    
                    # Obtaining the type of the subscript
                    int_1290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 50), 'int')
                    # Getting the type of 'sub_bound' (line 42)
                    sub_bound_1291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 40), 'sub_bound')
                    # Obtaining the member '__getitem__' of a type (line 42)
                    getitem___1292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 40), sub_bound_1291, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                    subscript_call_result_1293 = invoke(stypy.reporting.localization.Localization(__file__, 42, 40), getitem___1292, int_1290)
                    
                    # Applying the binary operator '>=' (line 42)
                    result_ge_1294 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), '>=', subscript_call_result_1289, subscript_call_result_1293)
                    
                    
                    
                    # Obtaining the type of the subscript
                    int_1295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 68), 'int')
                    # Getting the type of 'item' (line 42)
                    item_1296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 57), 'item')
                    # Obtaining the member 'bound' of a type (line 42)
                    bound_1297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 57), item_1296, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 42)
                    getitem___1298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 57), bound_1297, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                    subscript_call_result_1299 = invoke(stypy.reporting.localization.Localization(__file__, 42, 57), getitem___1298, int_1295)
                    
                    
                    # Obtaining the type of the subscript
                    int_1300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 83), 'int')
                    # Getting the type of 'sub_bound' (line 42)
                    sub_bound_1301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 73), 'sub_bound')
                    # Obtaining the member '__getitem__' of a type (line 42)
                    getitem___1302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 73), sub_bound_1301, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                    subscript_call_result_1303 = invoke(stypy.reporting.localization.Localization(__file__, 42, 73), getitem___1302, int_1300)
                    
                    # Applying the binary operator '<' (line 42)
                    result_lt_1304 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 57), '<', subscript_call_result_1299, subscript_call_result_1303)
                    
                    # Applying the binary operator 'and' (line 42)
                    result_and_keyword_1305 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), 'and', result_ge_1294, result_lt_1304)
                    
                    
                    # Obtaining the type of the subscript
                    int_1306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'int')
                    # Getting the type of 'item' (line 43)
                    item_1307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'item')
                    # Obtaining the member 'bound' of a type (line 43)
                    bound_1308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 23), item_1307, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 43)
                    getitem___1309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 23), bound_1308, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
                    subscript_call_result_1310 = invoke(stypy.reporting.localization.Localization(__file__, 43, 23), getitem___1309, int_1306)
                    
                    
                    # Obtaining the type of the subscript
                    int_1311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 50), 'int')
                    # Getting the type of 'sub_bound' (line 43)
                    sub_bound_1312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 40), 'sub_bound')
                    # Obtaining the member '__getitem__' of a type (line 43)
                    getitem___1313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 40), sub_bound_1312, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
                    subscript_call_result_1314 = invoke(stypy.reporting.localization.Localization(__file__, 43, 40), getitem___1313, int_1311)
                    
                    # Applying the binary operator '>=' (line 43)
                    result_ge_1315 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 23), '>=', subscript_call_result_1310, subscript_call_result_1314)
                    
                    # Applying the binary operator 'and' (line 42)
                    result_and_keyword_1316 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), 'and', result_and_keyword_1305, result_ge_1315)
                    
                    
                    # Obtaining the type of the subscript
                    int_1317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 68), 'int')
                    # Getting the type of 'item' (line 43)
                    item_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 57), 'item')
                    # Obtaining the member 'bound' of a type (line 43)
                    bound_1319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 57), item_1318, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 43)
                    getitem___1320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 57), bound_1319, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
                    subscript_call_result_1321 = invoke(stypy.reporting.localization.Localization(__file__, 43, 57), getitem___1320, int_1317)
                    
                    
                    # Obtaining the type of the subscript
                    int_1322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 83), 'int')
                    # Getting the type of 'sub_bound' (line 43)
                    sub_bound_1323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 73), 'sub_bound')
                    # Obtaining the member '__getitem__' of a type (line 43)
                    getitem___1324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 73), sub_bound_1323, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
                    subscript_call_result_1325 = invoke(stypy.reporting.localization.Localization(__file__, 43, 73), getitem___1324, int_1322)
                    
                    # Applying the binary operator '<' (line 43)
                    result_lt_1326 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 57), '<', subscript_call_result_1321, subscript_call_result_1325)
                    
                    # Applying the binary operator 'and' (line 42)
                    result_and_keyword_1327 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), 'and', result_and_keyword_1316, result_lt_1326)
                    
                    
                    # Obtaining the type of the subscript
                    int_1328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'int')
                    # Getting the type of 'item' (line 44)
                    item_1329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'item')
                    # Obtaining the member 'bound' of a type (line 44)
                    bound_1330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 23), item_1329, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___1331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 23), bound_1330, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_1332 = invoke(stypy.reporting.localization.Localization(__file__, 44, 23), getitem___1331, int_1328)
                    
                    
                    # Obtaining the type of the subscript
                    int_1333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 50), 'int')
                    # Getting the type of 'sub_bound' (line 44)
                    sub_bound_1334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 40), 'sub_bound')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___1335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 40), sub_bound_1334, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_1336 = invoke(stypy.reporting.localization.Localization(__file__, 44, 40), getitem___1335, int_1333)
                    
                    # Applying the binary operator '>=' (line 44)
                    result_ge_1337 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 23), '>=', subscript_call_result_1332, subscript_call_result_1336)
                    
                    # Applying the binary operator 'and' (line 42)
                    result_and_keyword_1338 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), 'and', result_and_keyword_1327, result_ge_1337)
                    
                    
                    # Obtaining the type of the subscript
                    int_1339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 68), 'int')
                    # Getting the type of 'item' (line 44)
                    item_1340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 57), 'item')
                    # Obtaining the member 'bound' of a type (line 44)
                    bound_1341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 57), item_1340, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___1342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 57), bound_1341, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_1343 = invoke(stypy.reporting.localization.Localization(__file__, 44, 57), getitem___1342, int_1339)
                    
                    
                    # Obtaining the type of the subscript
                    int_1344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 83), 'int')
                    # Getting the type of 'sub_bound' (line 44)
                    sub_bound_1345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 73), 'sub_bound')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___1346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 73), sub_bound_1345, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_1347 = invoke(stypy.reporting.localization.Localization(__file__, 44, 73), getitem___1346, int_1344)
                    
                    # Applying the binary operator '<' (line 44)
                    result_lt_1348 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 57), '<', subscript_call_result_1343, subscript_call_result_1347)
                    
                    # Applying the binary operator 'and' (line 42)
                    result_and_keyword_1349 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), 'and', result_and_keyword_1338, result_lt_1348)
                    
                    # Testing if the type of an if condition is none (line 42)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 20), result_and_keyword_1349):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 42)
                        if_condition_1350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 20), result_and_keyword_1349)
                        # Assigning a type to the variable 'if_condition_1350' (line 42)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'if_condition_1350', if_condition_1350)
                        # SSA begins for if statement (line 42)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 45)
                        # Processing the call arguments (line 45)
                        # Getting the type of 'item' (line 45)
                        item_1353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 44), 'item', False)
                        # Processing the call keyword arguments (line 45)
                        kwargs_1354 = {}
                        # Getting the type of 'sub_items' (line 45)
                        sub_items_1351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'sub_items', False)
                        # Obtaining the member 'append' of a type (line 45)
                        append_1352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 27), sub_items_1351, 'append')
                        # Calling append(args, kwargs) (line 45)
                        append_call_result_1355 = invoke(stypy.reporting.localization.Localization(__file__, 45, 27), append_1352, *[item_1353], **kwargs_1354)
                        
                        # SSA join for if statement (line 42)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Getting the type of 'q1' (line 46)
                q1_1356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'q1')
                
                
                
                # Call to len(...): (line 46)
                # Processing the call arguments (line 46)
                # Getting the type of 'sub_items' (line 46)
                sub_items_1358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'sub_items', False)
                # Processing the call keyword arguments (line 46)
                kwargs_1359 = {}
                # Getting the type of 'len' (line 46)
                len_1357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'len', False)
                # Calling len(args, kwargs) (line 46)
                len_call_result_1360 = invoke(stypy.reporting.localization.Localization(__file__, 46, 27), len_1357, *[sub_items_1358], **kwargs_1359)
                
                
                # Call to len(...): (line 46)
                # Processing the call arguments (line 46)
                # Getting the type of 'items' (line 46)
                items_1362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 49), 'items', False)
                # Processing the call keyword arguments (line 46)
                kwargs_1363 = {}
                # Getting the type of 'len' (line 46)
                len_1361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 45), 'len', False)
                # Calling len(args, kwargs) (line 46)
                len_call_result_1364 = invoke(stypy.reporting.localization.Localization(__file__, 46, 45), len_1361, *[items_1362], **kwargs_1363)
                
                # Applying the binary operator '==' (line 46)
                result_eq_1365 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 27), '==', len_call_result_1360, len_call_result_1364)
                
                # Testing the type of an if expression (line 46)
                is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 22), result_eq_1365)
                # SSA begins for if expression (line 46)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                int_1366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'int')
                # SSA branch for the else part of an if expression (line 46)
                module_type_store.open_ssa_branch('if expression else')
                int_1367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 61), 'int')
                # SSA join for if expression (line 46)
                module_type_store = module_type_store.join_ssa_context()
                if_exp_1368 = union_type.UnionType.add(int_1366, int_1367)
                
                # Applying the binary operator '+=' (line 46)
                result_iadd_1369 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 16), '+=', q1_1356, if_exp_1368)
                # Assigning a type to the variable 'q1' (line 46)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'q1', result_iadd_1369)
                
                
                # Assigning a Compare to a Name (line 47):
                
                # Assigning a Compare to a Name (line 47):
                
                
                # Obtaining the type of the subscript
                int_1370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 32), 'int')
                # Getting the type of 'sub_bound' (line 47)
                sub_bound_1371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'sub_bound')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___1372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 22), sub_bound_1371, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_1373 = invoke(stypy.reporting.localization.Localization(__file__, 47, 22), getitem___1372, int_1370)
                
                
                # Obtaining the type of the subscript
                int_1374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 47), 'int')
                # Getting the type of 'sub_bound' (line 47)
                sub_bound_1375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'sub_bound')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___1376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 37), sub_bound_1375, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_1377 = invoke(stypy.reporting.localization.Localization(__file__, 47, 37), getitem___1376, int_1374)
                
                # Applying the binary operator '-' (line 47)
                result_sub_1378 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 22), '-', subscript_call_result_1373, subscript_call_result_1377)
                
                # Getting the type of 'TOLERANCE' (line 47)
                TOLERANCE_1379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 54), 'TOLERANCE')
                float_1380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 66), 'float')
                # Applying the binary operator '*' (line 47)
                result_mul_1381 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 54), '*', TOLERANCE_1379, float_1380)
                
                # Applying the binary operator '<' (line 47)
                result_lt_1382 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 21), '<', result_sub_1378, result_mul_1381)
                
                # Assigning a type to the variable 'q2' (line 47)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'q2', result_lt_1382)
                
                
                # Call to len(...): (line 48)
                # Processing the call arguments (line 48)
                # Getting the type of 'sub_items' (line 48)
                sub_items_1384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'sub_items', False)
                # Processing the call keyword arguments (line 48)
                kwargs_1385 = {}
                # Getting the type of 'len' (line 48)
                len_1383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'len', False)
                # Calling len(args, kwargs) (line 48)
                len_call_result_1386 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), len_1383, *[sub_items_1384], **kwargs_1385)
                
                int_1387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 36), 'int')
                # Applying the binary operator '>' (line 48)
                result_gt_1388 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 19), '>', len_call_result_1386, int_1387)
                
                # Testing if the type of an if condition is none (line 48)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 48, 16), result_gt_1388):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 48)
                    if_condition_1389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 16), result_gt_1388)
                    # Assigning a type to the variable 'if_condition_1389' (line 48)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'if_condition_1389', if_condition_1389)
                    # SSA begins for if statement (line 48)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Subscript (line 49):
                    
                    # Assigning a Call to a Subscript (line 49):
                    
                    # Call to SpatialIndex(...): (line 49)
                    # Processing the call arguments (line 49)
                    # Getting the type of 'None' (line 49)
                    None_1391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 50), 'None', False)
                    # Getting the type of 'sub_bound' (line 49)
                    sub_bound_1392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 56), 'sub_bound', False)
                    # Getting the type of 'sub_items' (line 49)
                    sub_items_1393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 67), 'sub_items', False)
                    
                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'q1' (line 49)
                    q1_1394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 92), 'q1', False)
                    int_1395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 97), 'int')
                    # Applying the binary operator '>' (line 49)
                    result_gt_1396 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 92), '>', q1_1394, int_1395)
                    
                    # Getting the type of 'q2' (line 49)
                    q2_1397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 102), 'q2', False)
                    # Applying the binary operator 'or' (line 49)
                    result_or_keyword_1398 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 92), 'or', result_gt_1396, q2_1397)
                    
                    # Testing the type of an if expression (line 49)
                    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 78), result_or_keyword_1398)
                    # SSA begins for if expression (line 49)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                    # Getting the type of 'MAX_LEVELS' (line 49)
                    MAX_LEVELS_1399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 78), 'MAX_LEVELS', False)
                    # SSA branch for the else part of an if expression (line 49)
                    module_type_store.open_ssa_branch('if expression else')
                    # Getting the type of 'level' (line 49)
                    level_1400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 110), 'level', False)
                    int_1401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 118), 'int')
                    # Applying the binary operator '+' (line 49)
                    result_add_1402 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 110), '+', level_1400, int_1401)
                    
                    # SSA join for if expression (line 49)
                    module_type_store = module_type_store.join_ssa_context()
                    if_exp_1403 = union_type.UnionType.add(MAX_LEVELS_1399, result_add_1402)
                    
                    # Processing the call keyword arguments (line 49)
                    kwargs_1404 = {}
                    # Getting the type of 'SpatialIndex' (line 49)
                    SpatialIndex_1390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 37), 'SpatialIndex', False)
                    # Calling SpatialIndex(args, kwargs) (line 49)
                    SpatialIndex_call_result_1405 = invoke(stypy.reporting.localization.Localization(__file__, 49, 37), SpatialIndex_1390, *[None_1391, sub_bound_1392, sub_items_1393, if_exp_1403], **kwargs_1404)
                    
                    # Getting the type of 'self' (line 49)
                    self_1406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'self')
                    # Obtaining the member 'vector' of a type (line 49)
                    vector_1407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 20), self_1406, 'vector')
                    # Getting the type of 's' (line 49)
                    s_1408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 32), 's')
                    # Storing an element on a container (line 49)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 20), vector_1407, (s_1408, SpatialIndex_call_result_1405))
                    # SSA join for if statement (line 48)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA branch for the else part of an if statement (line 29)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Attribute (line 51):
            
            # Assigning a Name to a Attribute (line 51):
            # Getting the type of 'items' (line 51)
            items_1409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'items')
            # Getting the type of 'self' (line 51)
            self_1410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
            # Setting the type of the member 'items' of a type (line 51)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_1410, 'items', items_1409)
            # SSA join for if statement (line 29)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_intersection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 53)
        None_1411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 74), 'None')
        defaults = [None_1411]
        # Create a new context for function 'get_intersection'
        module_type_store = module_type_store.open_function_context('get_intersection', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SpatialIndex.get_intersection.__dict__.__setitem__('stypy_localization', localization)
        SpatialIndex.get_intersection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SpatialIndex.get_intersection.__dict__.__setitem__('stypy_type_store', module_type_store)
        SpatialIndex.get_intersection.__dict__.__setitem__('stypy_function_name', 'SpatialIndex.get_intersection')
        SpatialIndex.get_intersection.__dict__.__setitem__('stypy_param_names_list', ['ray_origin', 'ray_direction', 'last_hit', 'start'])
        SpatialIndex.get_intersection.__dict__.__setitem__('stypy_varargs_param_name', None)
        SpatialIndex.get_intersection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SpatialIndex.get_intersection.__dict__.__setitem__('stypy_call_defaults', defaults)
        SpatialIndex.get_intersection.__dict__.__setitem__('stypy_call_varargs', varargs)
        SpatialIndex.get_intersection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SpatialIndex.get_intersection.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SpatialIndex.get_intersection', ['ray_origin', 'ray_direction', 'last_hit', 'start'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_intersection', localization, ['ray_origin', 'ray_direction', 'last_hit', 'start'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_intersection(...)' code ##################

        
        # Assigning a IfExp to a Name (line 54):
        
        # Assigning a IfExp to a Name (line 54):
        
        # Getting the type of 'start' (line 54)
        start_1412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'start')
        # Testing the type of an if expression (line 54)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 16), start_1412)
        # SSA begins for if expression (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'start' (line 54)
        start_1413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'start')
        # SSA branch for the else part of an if expression (line 54)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'ray_origin' (line 54)
        ray_origin_1414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 36), 'ray_origin')
        # SSA join for if expression (line 54)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1415 = union_type.UnionType.add(start_1413, ray_origin_1414)
        
        # Assigning a type to the variable 'start' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'start', if_exp_1415)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Name to a Name (line 55):
        # Getting the type of 'None' (line 55)
        None_1416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 36), 'None')
        # Assigning a type to the variable 'hit_position' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'hit_position', None_1416)
        
        # Assigning a Name to a Name (line 55):
        # Getting the type of 'hit_position' (line 55)
        hit_position_1417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'hit_position')
        # Assigning a type to the variable 'hit_object' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'hit_object', hit_position_1417)
        
        # Assigning a Attribute to a Tuple (line 56):
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_1418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
        # Getting the type of 'self' (line 56)
        self_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'self')
        # Obtaining the member 'bound' of a type (line 56)
        bound_1420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 33), self_1419, 'bound')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___1421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), bound_1420, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_1422 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___1421, int_1418)
        
        # Assigning a type to the variable 'tuple_var_assignment_1081' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1081', subscript_call_result_1422)
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_1423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
        # Getting the type of 'self' (line 56)
        self_1424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'self')
        # Obtaining the member 'bound' of a type (line 56)
        bound_1425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 33), self_1424, 'bound')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___1426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), bound_1425, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_1427 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___1426, int_1423)
        
        # Assigning a type to the variable 'tuple_var_assignment_1082' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1082', subscript_call_result_1427)
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_1428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
        # Getting the type of 'self' (line 56)
        self_1429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'self')
        # Obtaining the member 'bound' of a type (line 56)
        bound_1430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 33), self_1429, 'bound')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___1431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), bound_1430, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_1432 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___1431, int_1428)
        
        # Assigning a type to the variable 'tuple_var_assignment_1083' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1083', subscript_call_result_1432)
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_1433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
        # Getting the type of 'self' (line 56)
        self_1434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'self')
        # Obtaining the member 'bound' of a type (line 56)
        bound_1435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 33), self_1434, 'bound')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___1436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), bound_1435, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_1437 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___1436, int_1433)
        
        # Assigning a type to the variable 'tuple_var_assignment_1084' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1084', subscript_call_result_1437)
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_1438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
        # Getting the type of 'self' (line 56)
        self_1439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'self')
        # Obtaining the member 'bound' of a type (line 56)
        bound_1440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 33), self_1439, 'bound')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___1441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), bound_1440, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_1442 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___1441, int_1438)
        
        # Assigning a type to the variable 'tuple_var_assignment_1085' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1085', subscript_call_result_1442)
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_1443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
        # Getting the type of 'self' (line 56)
        self_1444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'self')
        # Obtaining the member 'bound' of a type (line 56)
        bound_1445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 33), self_1444, 'bound')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___1446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), bound_1445, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_1447 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___1446, int_1443)
        
        # Assigning a type to the variable 'tuple_var_assignment_1086' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1086', subscript_call_result_1447)
        
        # Assigning a Name to a Name (line 56):
        # Getting the type of 'tuple_var_assignment_1081' (line 56)
        tuple_var_assignment_1081_1448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1081')
        # Assigning a type to the variable 'b0' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'b0', tuple_var_assignment_1081_1448)
        
        # Assigning a Name to a Name (line 56):
        # Getting the type of 'tuple_var_assignment_1082' (line 56)
        tuple_var_assignment_1082_1449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1082')
        # Assigning a type to the variable 'b1' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'b1', tuple_var_assignment_1082_1449)
        
        # Assigning a Name to a Name (line 56):
        # Getting the type of 'tuple_var_assignment_1083' (line 56)
        tuple_var_assignment_1083_1450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1083')
        # Assigning a type to the variable 'b2' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'b2', tuple_var_assignment_1083_1450)
        
        # Assigning a Name to a Name (line 56):
        # Getting the type of 'tuple_var_assignment_1084' (line 56)
        tuple_var_assignment_1084_1451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1084')
        # Assigning a type to the variable 'b3' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'b3', tuple_var_assignment_1084_1451)
        
        # Assigning a Name to a Name (line 56):
        # Getting the type of 'tuple_var_assignment_1085' (line 56)
        tuple_var_assignment_1085_1452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1085')
        # Assigning a type to the variable 'b4' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'b4', tuple_var_assignment_1085_1452)
        
        # Assigning a Name to a Name (line 56):
        # Getting the type of 'tuple_var_assignment_1086' (line 56)
        tuple_var_assignment_1086_1453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1086')
        # Assigning a type to the variable 'b5' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 28), 'b5', tuple_var_assignment_1086_1453)
        # Getting the type of 'self' (line 57)
        self_1454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'self')
        # Obtaining the member 'is_branch' of a type (line 57)
        is_branch_1455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), self_1454, 'is_branch')
        # Testing if the type of an if condition is none (line 57)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 8), is_branch_1455):
            
            # Assigning a Num to a Name (line 86):
            
            # Assigning a Num to a Name (line 86):
            float_1623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'float')
            # Assigning a type to the variable 'nearest_distance' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'nearest_distance', float_1623)
            
            # Getting the type of 'self' (line 87)
            self_1624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'self')
            # Obtaining the member 'items' of a type (line 87)
            items_1625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 24), self_1624, 'items')
            # Testing if the for loop is going to be iterated (line 87)
            # Testing the type of a for loop iterable (line 87)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 12), items_1625)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 87, 12), items_1625):
                # Getting the type of the for loop variable (line 87)
                for_loop_var_1626 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 12), items_1625)
                # Assigning a type to the variable 'item' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'item', for_loop_var_1626)
                # SSA begins for a for statement (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'item' (line 88)
                item_1627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'item')
                # Getting the type of 'last_hit' (line 88)
                last_hit_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 'last_hit')
                # Applying the binary operator '!=' (line 88)
                result_ne_1629 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 19), '!=', item_1627, last_hit_1628)
                
                # Testing if the type of an if condition is none (line 88)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 16), result_ne_1629):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 88)
                    if_condition_1630 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 16), result_ne_1629)
                    # Assigning a type to the variable 'if_condition_1630' (line 88)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'if_condition_1630', if_condition_1630)
                    # SSA begins for if statement (line 88)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Call to get_intersection(...): (line 89)
                    # Processing the call arguments (line 89)
                    # Getting the type of 'ray_origin' (line 89)
                    ray_origin_1633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 53), 'ray_origin', False)
                    # Getting the type of 'ray_direction' (line 89)
                    ray_direction_1634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 65), 'ray_direction', False)
                    # Processing the call keyword arguments (line 89)
                    kwargs_1635 = {}
                    # Getting the type of 'item' (line 89)
                    item_1631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'item', False)
                    # Obtaining the member 'get_intersection' of a type (line 89)
                    get_intersection_1632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 31), item_1631, 'get_intersection')
                    # Calling get_intersection(args, kwargs) (line 89)
                    get_intersection_call_result_1636 = invoke(stypy.reporting.localization.Localization(__file__, 89, 31), get_intersection_1632, *[ray_origin_1633, ray_direction_1634], **kwargs_1635)
                    
                    # Assigning a type to the variable 'distance' (line 89)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'distance', get_intersection_call_result_1636)
                    
                    float_1637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'float')
                    # Getting the type of 'distance' (line 90)
                    distance_1638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'distance')
                    # Applying the binary operator '<=' (line 90)
                    result_le_1639 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '<=', float_1637, distance_1638)
                    # Getting the type of 'nearest_distance' (line 90)
                    nearest_distance_1640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 41), 'nearest_distance')
                    # Applying the binary operator '<' (line 90)
                    result_lt_1641 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '<', distance_1638, nearest_distance_1640)
                    # Applying the binary operator '&' (line 90)
                    result_and__1642 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '&', result_le_1639, result_lt_1641)
                    
                    # Testing if the type of an if condition is none (line 90)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 90, 20), result_and__1642):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 90)
                        if_condition_1643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 20), result_and__1642)
                        # Assigning a type to the variable 'if_condition_1643' (line 90)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'if_condition_1643', if_condition_1643)
                        # SSA begins for if statement (line 90)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 91):
                        
                        # Assigning a BinOp to a Name (line 91):
                        # Getting the type of 'ray_origin' (line 91)
                        ray_origin_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 30), 'ray_origin')
                        # Getting the type of 'ray_direction' (line 91)
                        ray_direction_1645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'ray_direction')
                        # Getting the type of 'distance' (line 91)
                        distance_1646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 59), 'distance')
                        # Applying the binary operator '*' (line 91)
                        result_mul_1647 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 43), '*', ray_direction_1645, distance_1646)
                        
                        # Applying the binary operator '+' (line 91)
                        result_add_1648 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 30), '+', ray_origin_1644, result_mul_1647)
                        
                        # Assigning a type to the variable 'hit' (line 91)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'hit', result_add_1648)
                        
                        # Evaluating a boolean operation
                        
                        # Getting the type of 'b0' (line 92)
                        b0_1649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'b0')
                        # Getting the type of 'hit' (line 92)
                        hit_1650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'hit')
                        # Obtaining the member 'x' of a type (line 92)
                        x_1651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 33), hit_1650, 'x')
                        # Applying the binary operator '-' (line 92)
                        result_sub_1652 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 28), '-', b0_1649, x_1651)
                        
                        # Getting the type of 'TOLERANCE' (line 92)
                        TOLERANCE_1653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 92)
                        result_le_1654 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 28), '<=', result_sub_1652, TOLERANCE_1653)
                        
                        
                        # Getting the type of 'hit' (line 93)
                        hit_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'hit')
                        # Obtaining the member 'x' of a type (line 93)
                        x_1656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 28), hit_1655, 'x')
                        # Getting the type of 'b3' (line 93)
                        b3_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 36), 'b3')
                        # Applying the binary operator '-' (line 93)
                        result_sub_1658 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 28), '-', x_1656, b3_1657)
                        
                        # Getting the type of 'TOLERANCE' (line 93)
                        TOLERANCE_1659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 93)
                        result_le_1660 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 28), '<=', result_sub_1658, TOLERANCE_1659)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1661 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_le_1654, result_le_1660)
                        
                        # Getting the type of 'b1' (line 94)
                        b1_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'b1')
                        # Getting the type of 'hit' (line 94)
                        hit_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'hit')
                        # Obtaining the member 'y' of a type (line 94)
                        y_1664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 33), hit_1663, 'y')
                        # Applying the binary operator '-' (line 94)
                        result_sub_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 28), '-', b1_1662, y_1664)
                        
                        # Getting the type of 'TOLERANCE' (line 94)
                        TOLERANCE_1666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 94)
                        result_le_1667 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 28), '<=', result_sub_1665, TOLERANCE_1666)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1668 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1661, result_le_1667)
                        
                        # Getting the type of 'hit' (line 95)
                        hit_1669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'hit')
                        # Obtaining the member 'y' of a type (line 95)
                        y_1670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 28), hit_1669, 'y')
                        # Getting the type of 'b4' (line 95)
                        b4_1671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'b4')
                        # Applying the binary operator '-' (line 95)
                        result_sub_1672 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 28), '-', y_1670, b4_1671)
                        
                        # Getting the type of 'TOLERANCE' (line 95)
                        TOLERANCE_1673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 95)
                        result_le_1674 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 28), '<=', result_sub_1672, TOLERANCE_1673)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1675 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1668, result_le_1674)
                        
                        # Getting the type of 'b2' (line 96)
                        b2_1676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 'b2')
                        # Getting the type of 'hit' (line 96)
                        hit_1677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'hit')
                        # Obtaining the member 'z' of a type (line 96)
                        z_1678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 33), hit_1677, 'z')
                        # Applying the binary operator '-' (line 96)
                        result_sub_1679 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 28), '-', b2_1676, z_1678)
                        
                        # Getting the type of 'TOLERANCE' (line 96)
                        TOLERANCE_1680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 96)
                        result_le_1681 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 28), '<=', result_sub_1679, TOLERANCE_1680)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1682 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1675, result_le_1681)
                        
                        # Getting the type of 'hit' (line 97)
                        hit_1683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'hit')
                        # Obtaining the member 'z' of a type (line 97)
                        z_1684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 28), hit_1683, 'z')
                        # Getting the type of 'b5' (line 97)
                        b5_1685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 36), 'b5')
                        # Applying the binary operator '-' (line 97)
                        result_sub_1686 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 28), '-', z_1684, b5_1685)
                        
                        # Getting the type of 'TOLERANCE' (line 97)
                        TOLERANCE_1687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 97)
                        result_le_1688 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 28), '<=', result_sub_1686, TOLERANCE_1687)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1689 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1682, result_le_1688)
                        
                        # Testing if the type of an if condition is none (line 92)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 92, 24), result_and_keyword_1689):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 92)
                            if_condition_1690 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 24), result_and_keyword_1689)
                            # Assigning a type to the variable 'if_condition_1690' (line 92)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'if_condition_1690', if_condition_1690)
                            # SSA begins for if statement (line 92)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Name (line 98):
                            
                            # Assigning a Name to a Name (line 98):
                            # Getting the type of 'item' (line 98)
                            item_1691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 44), 'item')
                            # Assigning a type to the variable 'hit_object' (line 98)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'hit_object', item_1691)
                            
                            # Assigning a Name to a Name (line 99):
                            
                            # Assigning a Name to a Name (line 99):
                            # Getting the type of 'hit' (line 99)
                            hit_1692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 46), 'hit')
                            # Assigning a type to the variable 'hit_position' (line 99)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 31), 'hit_position', hit_1692)
                            
                            # Assigning a Name to a Name (line 100):
                            
                            # Assigning a Name to a Name (line 100):
                            # Getting the type of 'distance' (line 100)
                            distance_1693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 50), 'distance')
                            # Assigning a type to the variable 'nearest_distance' (line 100)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'nearest_distance', distance_1693)
                            # SSA join for if statement (line 92)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 90)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 88)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
        else:
            
            # Testing the type of an if condition (line 57)
            if_condition_1456 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 8), is_branch_1455)
            # Assigning a type to the variable 'if_condition_1456' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'if_condition_1456', if_condition_1456)
            # SSA begins for if statement (line 57)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a IfExp to a Name (line 58):
            
            # Assigning a IfExp to a Name (line 58):
            
            
            # Getting the type of 'start' (line 58)
            start_1457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'start')
            # Obtaining the member 'x' of a type (line 58)
            x_1458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 28), start_1457, 'x')
            # Getting the type of 'b0' (line 58)
            b0_1459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 40), 'b0')
            # Getting the type of 'b3' (line 58)
            b3_1460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'b3')
            # Applying the binary operator '+' (line 58)
            result_add_1461 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 40), '+', b0_1459, b3_1460)
            
            float_1462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 49), 'float')
            # Applying the binary operator '*' (line 58)
            result_mul_1463 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 39), '*', result_add_1461, float_1462)
            
            # Applying the binary operator '>=' (line 58)
            result_ge_1464 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 28), '>=', x_1458, result_mul_1463)
            
            # Testing the type of an if expression (line 58)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 23), result_ge_1464)
            # SSA begins for if expression (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            int_1465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'int')
            # SSA branch for the else part of an if expression (line 58)
            module_type_store.open_ssa_branch('if expression else')
            int_1466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 58), 'int')
            # SSA join for if expression (line 58)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_1467 = union_type.UnionType.add(int_1465, int_1466)
            
            # Assigning a type to the variable 'sub_cell' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'sub_cell', if_exp_1467)
            
            # Getting the type of 'start' (line 59)
            start_1468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'start')
            # Obtaining the member 'y' of a type (line 59)
            y_1469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), start_1468, 'y')
            # Getting the type of 'b1' (line 59)
            b1_1470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 27), 'b1')
            # Getting the type of 'b4' (line 59)
            b4_1471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 30), 'b4')
            # Applying the binary operator '+' (line 59)
            result_add_1472 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 27), '+', b1_1470, b4_1471)
            
            float_1473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'float')
            # Applying the binary operator '*' (line 59)
            result_mul_1474 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 26), '*', result_add_1472, float_1473)
            
            # Applying the binary operator '>=' (line 59)
            result_ge_1475 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 15), '>=', y_1469, result_mul_1474)
            
            # Testing if the type of an if condition is none (line 59)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 12), result_ge_1475):
                pass
            else:
                
                # Testing the type of an if condition (line 59)
                if_condition_1476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 12), result_ge_1475)
                # Assigning a type to the variable 'if_condition_1476' (line 59)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'if_condition_1476', if_condition_1476)
                # SSA begins for if statement (line 59)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'sub_cell' (line 60)
                sub_cell_1477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'sub_cell')
                int_1478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'int')
                # Applying the binary operator '|=' (line 60)
                result_ior_1479 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 16), '|=', sub_cell_1477, int_1478)
                # Assigning a type to the variable 'sub_cell' (line 60)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'sub_cell', result_ior_1479)
                
                # SSA join for if statement (line 59)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'start' (line 61)
            start_1480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'start')
            # Obtaining the member 'z' of a type (line 61)
            z_1481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), start_1480, 'z')
            # Getting the type of 'b2' (line 61)
            b2_1482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 27), 'b2')
            # Getting the type of 'b5' (line 61)
            b5_1483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'b5')
            # Applying the binary operator '+' (line 61)
            result_add_1484 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 27), '+', b2_1482, b5_1483)
            
            float_1485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 36), 'float')
            # Applying the binary operator '*' (line 61)
            result_mul_1486 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 26), '*', result_add_1484, float_1485)
            
            # Applying the binary operator '>=' (line 61)
            result_ge_1487 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 15), '>=', z_1481, result_mul_1486)
            
            # Testing if the type of an if condition is none (line 61)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 61, 12), result_ge_1487):
                pass
            else:
                
                # Testing the type of an if condition (line 61)
                if_condition_1488 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 12), result_ge_1487)
                # Assigning a type to the variable 'if_condition_1488' (line 61)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'if_condition_1488', if_condition_1488)
                # SSA begins for if statement (line 61)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'sub_cell' (line 62)
                sub_cell_1489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'sub_cell')
                int_1490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 28), 'int')
                # Applying the binary operator '|=' (line 62)
                result_ior_1491 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 16), '|=', sub_cell_1489, int_1490)
                # Assigning a type to the variable 'sub_cell' (line 62)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'sub_cell', result_ior_1491)
                
                # SSA join for if statement (line 61)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Name to a Name (line 63):
            
            # Assigning a Name to a Name (line 63):
            # Getting the type of 'start' (line 63)
            start_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'start')
            # Assigning a type to the variable 'cell_position' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'cell_position', start_1492)
            
            # Getting the type of 'True' (line 64)
            True_1493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'True')
            # Testing if the while is going to be iterated (line 64)
            # Testing the type of an if condition (line 64)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 12), True_1493)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 64, 12), True_1493):
                # SSA begins for while statement (line 64)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'sub_cell' (line 65)
                sub_cell_1494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 31), 'sub_cell')
                # Getting the type of 'self' (line 65)
                self_1495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'self')
                # Obtaining the member 'vector' of a type (line 65)
                vector_1496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), self_1495, 'vector')
                # Obtaining the member '__getitem__' of a type (line 65)
                getitem___1497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), vector_1496, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 65)
                subscript_call_result_1498 = invoke(stypy.reporting.localization.Localization(__file__, 65, 19), getitem___1497, sub_cell_1494)
                
                # Getting the type of 'None' (line 65)
                None_1499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 44), 'None')
                # Applying the binary operator '!=' (line 65)
                result_ne_1500 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 19), '!=', subscript_call_result_1498, None_1499)
                
                # Testing if the type of an if condition is none (line 65)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 65, 16), result_ne_1500):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 65)
                    if_condition_1501 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 16), result_ne_1500)
                    # Assigning a type to the variable 'if_condition_1501' (line 65)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'if_condition_1501', if_condition_1501)
                    # SSA begins for if statement (line 65)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Tuple (line 66):
                    
                    # Assigning a Call to a Name:
                    
                    # Call to get_intersection(...): (line 66)
                    # Processing the call arguments (line 66)
                    # Getting the type of 'ray_origin' (line 66)
                    ray_origin_1508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 86), 'ray_origin', False)
                    # Getting the type of 'ray_direction' (line 66)
                    ray_direction_1509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 98), 'ray_direction', False)
                    # Getting the type of 'last_hit' (line 66)
                    last_hit_1510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 113), 'last_hit', False)
                    # Getting the type of 'cell_position' (line 66)
                    cell_position_1511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 123), 'cell_position', False)
                    # Processing the call keyword arguments (line 66)
                    kwargs_1512 = {}
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'sub_cell' (line 66)
                    sub_cell_1502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 59), 'sub_cell', False)
                    # Getting the type of 'self' (line 66)
                    self_1503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 47), 'self', False)
                    # Obtaining the member 'vector' of a type (line 66)
                    vector_1504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 47), self_1503, 'vector')
                    # Obtaining the member '__getitem__' of a type (line 66)
                    getitem___1505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 47), vector_1504, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
                    subscript_call_result_1506 = invoke(stypy.reporting.localization.Localization(__file__, 66, 47), getitem___1505, sub_cell_1502)
                    
                    # Obtaining the member 'get_intersection' of a type (line 66)
                    get_intersection_1507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 47), subscript_call_result_1506, 'get_intersection')
                    # Calling get_intersection(args, kwargs) (line 66)
                    get_intersection_call_result_1513 = invoke(stypy.reporting.localization.Localization(__file__, 66, 47), get_intersection_1507, *[ray_origin_1508, ray_direction_1509, last_hit_1510, cell_position_1511], **kwargs_1512)
                    
                    # Assigning a type to the variable 'call_assignment_1087' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1087', get_intersection_call_result_1513)
                    
                    # Assigning a Call to a Name (line 66):
                    
                    # Call to __getitem__(...):
                    # Processing the call arguments
                    int_1516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'int')
                    # Processing the call keyword arguments
                    kwargs_1517 = {}
                    # Getting the type of 'call_assignment_1087' (line 66)
                    call_assignment_1087_1514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1087', False)
                    # Obtaining the member '__getitem__' of a type (line 66)
                    getitem___1515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 20), call_assignment_1087_1514, '__getitem__')
                    # Calling __getitem__(args, kwargs)
                    getitem___call_result_1518 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1515, *[int_1516], **kwargs_1517)
                    
                    # Assigning a type to the variable 'call_assignment_1088' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1088', getitem___call_result_1518)
                    
                    # Assigning a Name to a Name (line 66):
                    # Getting the type of 'call_assignment_1088' (line 66)
                    call_assignment_1088_1519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1088')
                    # Assigning a type to the variable 'hit_object' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'hit_object', call_assignment_1088_1519)
                    
                    # Assigning a Call to a Name (line 66):
                    
                    # Call to __getitem__(...):
                    # Processing the call arguments
                    int_1522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 20), 'int')
                    # Processing the call keyword arguments
                    kwargs_1523 = {}
                    # Getting the type of 'call_assignment_1087' (line 66)
                    call_assignment_1087_1520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1087', False)
                    # Obtaining the member '__getitem__' of a type (line 66)
                    getitem___1521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 20), call_assignment_1087_1520, '__getitem__')
                    # Calling __getitem__(args, kwargs)
                    getitem___call_result_1524 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1521, *[int_1522], **kwargs_1523)
                    
                    # Assigning a type to the variable 'call_assignment_1089' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1089', getitem___call_result_1524)
                    
                    # Assigning a Name to a Name (line 66):
                    # Getting the type of 'call_assignment_1089' (line 66)
                    call_assignment_1089_1525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1089')
                    # Assigning a type to the variable 'hit_position' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'hit_position', call_assignment_1089_1525)
                    
                    # Getting the type of 'hit_object' (line 67)
                    hit_object_1526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'hit_object')
                    # Getting the type of 'None' (line 67)
                    None_1527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 37), 'None')
                    # Applying the binary operator '!=' (line 67)
                    result_ne_1528 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 23), '!=', hit_object_1526, None_1527)
                    
                    # Testing if the type of an if condition is none (line 67)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 20), result_ne_1528):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 67)
                        if_condition_1529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 20), result_ne_1528)
                        # Assigning a type to the variable 'if_condition_1529' (line 67)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'if_condition_1529', if_condition_1529)
                        # SSA begins for if statement (line 67)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # SSA join for if statement (line 67)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 65)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Num to a Name (line 69):
                
                # Assigning a Num to a Name (line 69):
                float_1530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'float')
                # Assigning a type to the variable 'step' (line 69)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'step', float_1530)
                
                # Assigning a Num to a Name (line 70):
                
                # Assigning a Num to a Name (line 70):
                int_1531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'int')
                # Assigning a type to the variable 'axis' (line 70)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'axis', int_1531)
                
                
                # Call to range(...): (line 71)
                # Processing the call arguments (line 71)
                int_1533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 31), 'int')
                # Processing the call keyword arguments (line 71)
                kwargs_1534 = {}
                # Getting the type of 'range' (line 71)
                range_1532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'range', False)
                # Calling range(args, kwargs) (line 71)
                range_call_result_1535 = invoke(stypy.reporting.localization.Localization(__file__, 71, 25), range_1532, *[int_1533], **kwargs_1534)
                
                # Testing if the for loop is going to be iterated (line 71)
                # Testing the type of a for loop iterable (line 71)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 71, 16), range_call_result_1535)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 71, 16), range_call_result_1535):
                    # Getting the type of the for loop variable (line 71)
                    for_loop_var_1536 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 71, 16), range_call_result_1535)
                    # Assigning a type to the variable 'i' (line 71)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'i', for_loop_var_1536)
                    # SSA begins for a for statement (line 71)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a BinOp to a Name (line 72):
                    
                    # Assigning a BinOp to a Name (line 72):
                    # Getting the type of 'sub_cell' (line 72)
                    sub_cell_1537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'sub_cell')
                    # Getting the type of 'i' (line 72)
                    i_1538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 40), 'i')
                    # Applying the binary operator '>>' (line 72)
                    result_rshift_1539 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 28), '>>', sub_cell_1537, i_1538)
                    
                    int_1540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 45), 'int')
                    # Applying the binary operator '&' (line 72)
                    result_and__1541 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 27), '&', result_rshift_1539, int_1540)
                    
                    # Assigning a type to the variable 'high' (line 72)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'high', result_and__1541)
                    
                    # Assigning a IfExp to a Name (line 73):
                    
                    # Assigning a IfExp to a Name (line 73):
                    
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 73)
                    i_1542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 70), 'i')
                    # Getting the type of 'ray_direction' (line 73)
                    ray_direction_1543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 56), 'ray_direction')
                    # Obtaining the member '__getitem__' of a type (line 73)
                    getitem___1544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 56), ray_direction_1543, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                    subscript_call_result_1545 = invoke(stypy.reporting.localization.Localization(__file__, 73, 56), getitem___1544, i_1542)
                    
                    float_1546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 75), 'float')
                    # Applying the binary operator '<' (line 73)
                    result_lt_1547 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 56), '<', subscript_call_result_1545, float_1546)
                    
                    
                    int_1548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 83), 'int')
                    # Getting the type of 'high' (line 73)
                    high_1549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 88), 'high')
                    # Applying the binary operator '!=' (line 73)
                    result_ne_1550 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 83), '!=', int_1548, high_1549)
                    
                    # Applying the binary operator '^' (line 73)
                    result_xor_1551 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 55), '^', result_lt_1547, result_ne_1550)
                    
                    # Testing the type of an if expression (line 73)
                    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 27), result_xor_1551)
                    # SSA begins for if expression (line 73)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 73)
                    i_1552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 38), 'i')
                    # Getting the type of 'high' (line 73)
                    high_1553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 42), 'high')
                    int_1554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 49), 'int')
                    # Applying the binary operator '*' (line 73)
                    result_mul_1555 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 42), '*', high_1553, int_1554)
                    
                    # Applying the binary operator '+' (line 73)
                    result_add_1556 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 38), '+', i_1552, result_mul_1555)
                    
                    # Getting the type of 'self' (line 73)
                    self_1557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 27), 'self')
                    # Obtaining the member 'bound' of a type (line 73)
                    bound_1558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 27), self_1557, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 73)
                    getitem___1559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 27), bound_1558, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                    subscript_call_result_1560 = invoke(stypy.reporting.localization.Localization(__file__, 73, 27), getitem___1559, result_add_1556)
                    
                    # SSA branch for the else part of an if expression (line 73)
                    module_type_store.open_ssa_branch('if expression else')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 73)
                    i_1561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 111), 'i')
                    # Getting the type of 'self' (line 73)
                    self_1562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 100), 'self')
                    # Obtaining the member 'bound' of a type (line 73)
                    bound_1563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 100), self_1562, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 73)
                    getitem___1564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 100), bound_1563, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                    subscript_call_result_1565 = invoke(stypy.reporting.localization.Localization(__file__, 73, 100), getitem___1564, i_1561)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 73)
                    i_1566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 127), 'i')
                    int_1567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 131), 'int')
                    # Applying the binary operator '+' (line 73)
                    result_add_1568 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 127), '+', i_1566, int_1567)
                    
                    # Getting the type of 'self' (line 73)
                    self_1569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 116), 'self')
                    # Obtaining the member 'bound' of a type (line 73)
                    bound_1570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 116), self_1569, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 73)
                    getitem___1571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 116), bound_1570, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                    subscript_call_result_1572 = invoke(stypy.reporting.localization.Localization(__file__, 73, 116), getitem___1571, result_add_1568)
                    
                    # Applying the binary operator '+' (line 73)
                    result_add_1573 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 100), '+', subscript_call_result_1565, subscript_call_result_1572)
                    
                    float_1574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 137), 'float')
                    # Applying the binary operator '*' (line 73)
                    result_mul_1575 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 99), '*', result_add_1573, float_1574)
                    
                    # SSA join for if expression (line 73)
                    module_type_store = module_type_store.join_ssa_context()
                    if_exp_1576 = union_type.UnionType.add(subscript_call_result_1560, result_mul_1575)
                    
                    # Assigning a type to the variable 'face' (line 73)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'face', if_exp_1576)
                    
                    
                    # SSA begins for try-except statement (line 74)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                    
                    # Assigning a BinOp to a Name (line 75):
                    
                    # Assigning a BinOp to a Name (line 75):
                    # Getting the type of 'face' (line 75)
                    face_1577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'face')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 75)
                    i_1578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 54), 'i')
                    # Getting the type of 'ray_origin' (line 75)
                    ray_origin_1579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 43), 'ray_origin')
                    # Obtaining the member '__getitem__' of a type (line 75)
                    getitem___1580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 43), ray_origin_1579, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                    subscript_call_result_1581 = invoke(stypy.reporting.localization.Localization(__file__, 75, 43), getitem___1580, i_1578)
                    
                    # Applying the binary operator '-' (line 75)
                    result_sub_1582 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 36), '-', face_1577, subscript_call_result_1581)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 75)
                    i_1583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 74), 'i')
                    # Getting the type of 'ray_direction' (line 75)
                    ray_direction_1584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 60), 'ray_direction')
                    # Obtaining the member '__getitem__' of a type (line 75)
                    getitem___1585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 60), ray_direction_1584, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                    subscript_call_result_1586 = invoke(stypy.reporting.localization.Localization(__file__, 75, 60), getitem___1585, i_1583)
                    
                    # Applying the binary operator 'div' (line 75)
                    result_div_1587 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 35), 'div', result_sub_1582, subscript_call_result_1586)
                    
                    # Assigning a type to the variable 'distance' (line 75)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'distance', result_div_1587)
                    # SSA branch for the except part of a try statement (line 74)
                    # SSA branch for the except '<any exception>' branch of a try statement (line 74)
                    module_type_store.open_ssa_branch('except')
                    
                    # Assigning a Call to a Name (line 77):
                    
                    # Assigning a Call to a Name (line 77):
                    
                    # Call to float(...): (line 77)
                    # Processing the call arguments (line 77)
                    float_1589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 41), 'float')
                    # Processing the call keyword arguments (line 77)
                    kwargs_1590 = {}
                    # Getting the type of 'float' (line 77)
                    float_1588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 35), 'float', False)
                    # Calling float(args, kwargs) (line 77)
                    float_call_result_1591 = invoke(stypy.reporting.localization.Localization(__file__, 77, 35), float_1588, *[float_1589], **kwargs_1590)
                    
                    # Assigning a type to the variable 'distance' (line 77)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'distance', float_call_result_1591)
                    # SSA join for try-except statement (line 74)
                    module_type_store = module_type_store.join_ssa_context()
                    
                    
                    # Getting the type of 'distance' (line 78)
                    distance_1592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'distance')
                    # Getting the type of 'step' (line 78)
                    step_1593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 35), 'step')
                    # Applying the binary operator '<=' (line 78)
                    result_le_1594 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 23), '<=', distance_1592, step_1593)
                    
                    # Testing if the type of an if condition is none (line 78)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 78, 20), result_le_1594):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 78)
                        if_condition_1595 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 20), result_le_1594)
                        # Assigning a type to the variable 'if_condition_1595' (line 78)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'if_condition_1595', if_condition_1595)
                        # SSA begins for if statement (line 78)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Name (line 79):
                        
                        # Assigning a Name to a Name (line 79):
                        # Getting the type of 'distance' (line 79)
                        distance_1596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 31), 'distance')
                        # Assigning a type to the variable 'step' (line 79)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'step', distance_1596)
                        
                        # Assigning a Name to a Name (line 80):
                        
                        # Assigning a Name to a Name (line 80):
                        # Getting the type of 'i' (line 80)
                        i_1597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'i')
                        # Assigning a type to the variable 'axis' (line 80)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'axis', i_1597)
                        # SSA join for if statement (line 78)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Getting the type of 'sub_cell' (line 81)
                sub_cell_1598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'sub_cell')
                # Getting the type of 'axis' (line 81)
                axis_1599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 34), 'axis')
                # Applying the binary operator '>>' (line 81)
                result_rshift_1600 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 22), '>>', sub_cell_1598, axis_1599)
                
                int_1601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 42), 'int')
                # Applying the binary operator '&' (line 81)
                result_and__1602 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 21), '&', result_rshift_1600, int_1601)
                
                int_1603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 48), 'int')
                # Applying the binary operator '==' (line 81)
                result_eq_1604 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 20), '==', result_and__1602, int_1603)
                
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'axis' (line 81)
                axis_1605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 68), 'axis')
                # Getting the type of 'ray_direction' (line 81)
                ray_direction_1606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 54), 'ray_direction')
                # Obtaining the member '__getitem__' of a type (line 81)
                getitem___1607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 54), ray_direction_1606, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 81)
                subscript_call_result_1608 = invoke(stypy.reporting.localization.Localization(__file__, 81, 54), getitem___1607, axis_1605)
                
                float_1609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 76), 'float')
                # Applying the binary operator '<' (line 81)
                result_lt_1610 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 54), '<', subscript_call_result_1608, float_1609)
                
                # Applying the binary operator '^' (line 81)
                result_xor_1611 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 19), '^', result_eq_1604, result_lt_1610)
                
                # Testing if the type of an if condition is none (line 81)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 81, 16), result_xor_1611):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 81)
                    if_condition_1612 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 16), result_xor_1611)
                    # Assigning a type to the variable 'if_condition_1612' (line 81)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'if_condition_1612', if_condition_1612)
                    # SSA begins for if statement (line 81)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # SSA join for if statement (line 81)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a BinOp to a Name (line 83):
                
                # Assigning a BinOp to a Name (line 83):
                # Getting the type of 'ray_origin' (line 83)
                ray_origin_1613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 32), 'ray_origin')
                # Getting the type of 'ray_direction' (line 83)
                ray_direction_1614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 45), 'ray_direction')
                # Getting the type of 'step' (line 83)
                step_1615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 61), 'step')
                # Applying the binary operator '*' (line 83)
                result_mul_1616 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 45), '*', ray_direction_1614, step_1615)
                
                # Applying the binary operator '+' (line 83)
                result_add_1617 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 32), '+', ray_origin_1613, result_mul_1616)
                
                # Assigning a type to the variable 'cell_position' (line 83)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'cell_position', result_add_1617)
                
                # Assigning a BinOp to a Name (line 84):
                
                # Assigning a BinOp to a Name (line 84):
                # Getting the type of 'sub_cell' (line 84)
                sub_cell_1618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'sub_cell')
                int_1619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 39), 'int')
                # Getting the type of 'axis' (line 84)
                axis_1620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 44), 'axis')
                # Applying the binary operator '<<' (line 84)
                result_lshift_1621 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 39), '<<', int_1619, axis_1620)
                
                # Applying the binary operator '^' (line 84)
                result_xor_1622 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 27), '^', sub_cell_1618, result_lshift_1621)
                
                # Assigning a type to the variable 'sub_cell' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'sub_cell', result_xor_1622)
                # SSA join for while statement (line 64)
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA branch for the else part of an if statement (line 57)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Num to a Name (line 86):
            
            # Assigning a Num to a Name (line 86):
            float_1623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'float')
            # Assigning a type to the variable 'nearest_distance' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'nearest_distance', float_1623)
            
            # Getting the type of 'self' (line 87)
            self_1624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'self')
            # Obtaining the member 'items' of a type (line 87)
            items_1625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 24), self_1624, 'items')
            # Testing if the for loop is going to be iterated (line 87)
            # Testing the type of a for loop iterable (line 87)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 12), items_1625)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 87, 12), items_1625):
                # Getting the type of the for loop variable (line 87)
                for_loop_var_1626 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 12), items_1625)
                # Assigning a type to the variable 'item' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'item', for_loop_var_1626)
                # SSA begins for a for statement (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'item' (line 88)
                item_1627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'item')
                # Getting the type of 'last_hit' (line 88)
                last_hit_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 'last_hit')
                # Applying the binary operator '!=' (line 88)
                result_ne_1629 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 19), '!=', item_1627, last_hit_1628)
                
                # Testing if the type of an if condition is none (line 88)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 16), result_ne_1629):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 88)
                    if_condition_1630 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 16), result_ne_1629)
                    # Assigning a type to the variable 'if_condition_1630' (line 88)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'if_condition_1630', if_condition_1630)
                    # SSA begins for if statement (line 88)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Call to get_intersection(...): (line 89)
                    # Processing the call arguments (line 89)
                    # Getting the type of 'ray_origin' (line 89)
                    ray_origin_1633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 53), 'ray_origin', False)
                    # Getting the type of 'ray_direction' (line 89)
                    ray_direction_1634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 65), 'ray_direction', False)
                    # Processing the call keyword arguments (line 89)
                    kwargs_1635 = {}
                    # Getting the type of 'item' (line 89)
                    item_1631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'item', False)
                    # Obtaining the member 'get_intersection' of a type (line 89)
                    get_intersection_1632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 31), item_1631, 'get_intersection')
                    # Calling get_intersection(args, kwargs) (line 89)
                    get_intersection_call_result_1636 = invoke(stypy.reporting.localization.Localization(__file__, 89, 31), get_intersection_1632, *[ray_origin_1633, ray_direction_1634], **kwargs_1635)
                    
                    # Assigning a type to the variable 'distance' (line 89)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'distance', get_intersection_call_result_1636)
                    
                    float_1637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'float')
                    # Getting the type of 'distance' (line 90)
                    distance_1638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'distance')
                    # Applying the binary operator '<=' (line 90)
                    result_le_1639 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '<=', float_1637, distance_1638)
                    # Getting the type of 'nearest_distance' (line 90)
                    nearest_distance_1640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 41), 'nearest_distance')
                    # Applying the binary operator '<' (line 90)
                    result_lt_1641 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '<', distance_1638, nearest_distance_1640)
                    # Applying the binary operator '&' (line 90)
                    result_and__1642 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '&', result_le_1639, result_lt_1641)
                    
                    # Testing if the type of an if condition is none (line 90)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 90, 20), result_and__1642):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 90)
                        if_condition_1643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 20), result_and__1642)
                        # Assigning a type to the variable 'if_condition_1643' (line 90)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'if_condition_1643', if_condition_1643)
                        # SSA begins for if statement (line 90)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 91):
                        
                        # Assigning a BinOp to a Name (line 91):
                        # Getting the type of 'ray_origin' (line 91)
                        ray_origin_1644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 30), 'ray_origin')
                        # Getting the type of 'ray_direction' (line 91)
                        ray_direction_1645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'ray_direction')
                        # Getting the type of 'distance' (line 91)
                        distance_1646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 59), 'distance')
                        # Applying the binary operator '*' (line 91)
                        result_mul_1647 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 43), '*', ray_direction_1645, distance_1646)
                        
                        # Applying the binary operator '+' (line 91)
                        result_add_1648 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 30), '+', ray_origin_1644, result_mul_1647)
                        
                        # Assigning a type to the variable 'hit' (line 91)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'hit', result_add_1648)
                        
                        # Evaluating a boolean operation
                        
                        # Getting the type of 'b0' (line 92)
                        b0_1649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'b0')
                        # Getting the type of 'hit' (line 92)
                        hit_1650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'hit')
                        # Obtaining the member 'x' of a type (line 92)
                        x_1651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 33), hit_1650, 'x')
                        # Applying the binary operator '-' (line 92)
                        result_sub_1652 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 28), '-', b0_1649, x_1651)
                        
                        # Getting the type of 'TOLERANCE' (line 92)
                        TOLERANCE_1653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 92)
                        result_le_1654 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 28), '<=', result_sub_1652, TOLERANCE_1653)
                        
                        
                        # Getting the type of 'hit' (line 93)
                        hit_1655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'hit')
                        # Obtaining the member 'x' of a type (line 93)
                        x_1656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 28), hit_1655, 'x')
                        # Getting the type of 'b3' (line 93)
                        b3_1657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 36), 'b3')
                        # Applying the binary operator '-' (line 93)
                        result_sub_1658 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 28), '-', x_1656, b3_1657)
                        
                        # Getting the type of 'TOLERANCE' (line 93)
                        TOLERANCE_1659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 93)
                        result_le_1660 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 28), '<=', result_sub_1658, TOLERANCE_1659)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1661 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_le_1654, result_le_1660)
                        
                        # Getting the type of 'b1' (line 94)
                        b1_1662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'b1')
                        # Getting the type of 'hit' (line 94)
                        hit_1663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'hit')
                        # Obtaining the member 'y' of a type (line 94)
                        y_1664 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 33), hit_1663, 'y')
                        # Applying the binary operator '-' (line 94)
                        result_sub_1665 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 28), '-', b1_1662, y_1664)
                        
                        # Getting the type of 'TOLERANCE' (line 94)
                        TOLERANCE_1666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 94)
                        result_le_1667 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 28), '<=', result_sub_1665, TOLERANCE_1666)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1668 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1661, result_le_1667)
                        
                        # Getting the type of 'hit' (line 95)
                        hit_1669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'hit')
                        # Obtaining the member 'y' of a type (line 95)
                        y_1670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 28), hit_1669, 'y')
                        # Getting the type of 'b4' (line 95)
                        b4_1671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'b4')
                        # Applying the binary operator '-' (line 95)
                        result_sub_1672 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 28), '-', y_1670, b4_1671)
                        
                        # Getting the type of 'TOLERANCE' (line 95)
                        TOLERANCE_1673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 95)
                        result_le_1674 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 28), '<=', result_sub_1672, TOLERANCE_1673)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1675 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1668, result_le_1674)
                        
                        # Getting the type of 'b2' (line 96)
                        b2_1676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 'b2')
                        # Getting the type of 'hit' (line 96)
                        hit_1677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'hit')
                        # Obtaining the member 'z' of a type (line 96)
                        z_1678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 33), hit_1677, 'z')
                        # Applying the binary operator '-' (line 96)
                        result_sub_1679 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 28), '-', b2_1676, z_1678)
                        
                        # Getting the type of 'TOLERANCE' (line 96)
                        TOLERANCE_1680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 96)
                        result_le_1681 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 28), '<=', result_sub_1679, TOLERANCE_1680)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1682 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1675, result_le_1681)
                        
                        # Getting the type of 'hit' (line 97)
                        hit_1683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'hit')
                        # Obtaining the member 'z' of a type (line 97)
                        z_1684 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 28), hit_1683, 'z')
                        # Getting the type of 'b5' (line 97)
                        b5_1685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 36), 'b5')
                        # Applying the binary operator '-' (line 97)
                        result_sub_1686 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 28), '-', z_1684, b5_1685)
                        
                        # Getting the type of 'TOLERANCE' (line 97)
                        TOLERANCE_1687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 97)
                        result_le_1688 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 28), '<=', result_sub_1686, TOLERANCE_1687)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1689 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1682, result_le_1688)
                        
                        # Testing if the type of an if condition is none (line 92)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 92, 24), result_and_keyword_1689):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 92)
                            if_condition_1690 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 24), result_and_keyword_1689)
                            # Assigning a type to the variable 'if_condition_1690' (line 92)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'if_condition_1690', if_condition_1690)
                            # SSA begins for if statement (line 92)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Name (line 98):
                            
                            # Assigning a Name to a Name (line 98):
                            # Getting the type of 'item' (line 98)
                            item_1691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 44), 'item')
                            # Assigning a type to the variable 'hit_object' (line 98)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'hit_object', item_1691)
                            
                            # Assigning a Name to a Name (line 99):
                            
                            # Assigning a Name to a Name (line 99):
                            # Getting the type of 'hit' (line 99)
                            hit_1692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 46), 'hit')
                            # Assigning a type to the variable 'hit_position' (line 99)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 31), 'hit_position', hit_1692)
                            
                            # Assigning a Name to a Name (line 100):
                            
                            # Assigning a Name to a Name (line 100):
                            # Getting the type of 'distance' (line 100)
                            distance_1693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 50), 'distance')
                            # Assigning a type to the variable 'nearest_distance' (line 100)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'nearest_distance', distance_1693)
                            # SSA join for if statement (line 92)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 90)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 88)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 57)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'tuple' (line 101)
        tuple_1694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 101)
        # Adding element type (line 101)
        # Getting the type of 'hit_object' (line 101)
        hit_object_1695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'hit_object')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 15), tuple_1694, hit_object_1695)
        # Adding element type (line 101)
        # Getting the type of 'hit_position' (line 101)
        hit_position_1696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'hit_position')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 15), tuple_1694, hit_position_1696)
        
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', tuple_1694)
        
        # ################# End of 'get_intersection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_intersection' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_1697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1697)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_intersection'
        return stypy_return_type_1697


# Assigning a type to the variable 'SpatialIndex' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'SpatialIndex', SpatialIndex)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
