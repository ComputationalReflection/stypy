
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
import_1051 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'triangle')

if (type(import_1051) is not StypyTypeError):

    if (import_1051 != 'pyd_module'):
        __import__(import_1051)
        sys_modules_1052 = sys.modules[import_1051]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'triangle', sys_modules_1052.module_type_store, module_type_store, ['Triangle', 'TOLERANCE'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_1052, sys_modules_1052.module_type_store, module_type_store)
    else:
        from triangle import Triangle, TOLERANCE

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'triangle', None, module_type_store, ['Triangle', 'TOLERANCE'], [Triangle, TOLERANCE])

else:
    # Assigning a type to the variable 'triangle' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'triangle', import_1051)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from vector3f import Vector3f_seq, Vector3f_scalar, MAX' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_1053 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f')

if (type(import_1053) is not StypyTypeError):

    if (import_1053 != 'pyd_module'):
        __import__(import_1053)
        sys_modules_1054 = sys.modules[import_1053]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f', sys_modules_1054.module_type_store, module_type_store, ['Vector3f_seq', 'Vector3f_scalar', 'MAX'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_1054, sys_modules_1054.module_type_store, module_type_store)
    else:
        from vector3f import Vector3f_seq, Vector3f_scalar, MAX

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f', None, module_type_store, ['Vector3f_seq', 'Vector3f_scalar', 'MAX'], [Vector3f_seq, Vector3f_scalar, MAX])

else:
    # Assigning a type to the variable 'vector3f' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'vector3f', import_1053)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')


# Assigning a Num to a Name (line 10):

# Assigning a Num to a Name (line 10):
int_1055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 13), 'int')
# Assigning a type to the variable 'MAX_LEVELS' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'MAX_LEVELS', int_1055)

# Assigning a Num to a Name (line 11):

# Assigning a Num to a Name (line 11):
int_1056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 14), 'int')
# Assigning a type to the variable 'MAX_ITEMS' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'MAX_ITEMS', int_1056)
# Declaration of the 'SpatialIndex' class

class SpatialIndex(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_1057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 49), 'int')
        defaults = [int_1057]
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
        vect_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'vect')
        # Testing if the type of an if condition is none (line 16)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 16, 8), vect_1058):
            
            # Assigning a Name to a Attribute (line 27):
            
            # Assigning a Name to a Attribute (line 27):
            # Getting the type of 'bound' (line 27)
            bound_1165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'bound')
            # Getting the type of 'self' (line 27)
            self_1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'self')
            # Setting the type of the member 'bound' of a type (line 27)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), self_1166, 'bound', bound_1165)
        else:
            
            # Testing the type of an if condition (line 16)
            if_condition_1059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 16, 8), vect_1058)
            # Assigning a type to the variable 'if_condition_1059' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'if_condition_1059', if_condition_1059)
            # SSA begins for if statement (line 16)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'items' (line 17)
            items_1060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'items')
            # Assigning a type to the variable 'items_1060' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'items_1060', items_1060)
            # Testing if the for loop is going to be iterated (line 17)
            # Testing the type of a for loop iterable (line 17)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 17, 12), items_1060)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 17, 12), items_1060):
                # Getting the type of the for loop variable (line 17)
                for_loop_var_1061 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 17, 12), items_1060)
                # Assigning a type to the variable 'item' (line 17)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'item', for_loop_var_1061)
                # SSA begins for a for statement (line 17)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Attribute (line 18):
                
                # Assigning a Call to a Attribute (line 18):
                
                # Call to get_bound(...): (line 18)
                # Processing the call keyword arguments (line 18)
                kwargs_1064 = {}
                # Getting the type of 'item' (line 18)
                item_1062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 29), 'item', False)
                # Obtaining the member 'get_bound' of a type (line 18)
                get_bound_1063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 29), item_1062, 'get_bound')
                # Calling get_bound(args, kwargs) (line 18)
                get_bound_call_result_1065 = invoke(stypy.reporting.localization.Localization(__file__, 18, 29), get_bound_1063, *[], **kwargs_1064)
                
                # Getting the type of 'item' (line 18)
                item_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'item')
                # Setting the type of the member 'bound' of a type (line 18)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 16), item_1066, 'bound', get_bound_call_result_1065)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a BinOp to a Name (line 19):
            
            # Assigning a BinOp to a Name (line 19):
            
            # Call to as_list(...): (line 19)
            # Processing the call keyword arguments (line 19)
            kwargs_1069 = {}
            # Getting the type of 'vect' (line 19)
            vect_1067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'vect', False)
            # Obtaining the member 'as_list' of a type (line 19)
            as_list_1068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 20), vect_1067, 'as_list')
            # Calling as_list(args, kwargs) (line 19)
            as_list_call_result_1070 = invoke(stypy.reporting.localization.Localization(__file__, 19, 20), as_list_1068, *[], **kwargs_1069)
            
            int_1071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 37), 'int')
            # Applying the binary operator '*' (line 19)
            result_mul_1072 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 20), '*', as_list_call_result_1070, int_1071)
            
            # Assigning a type to the variable 'bound' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'bound', result_mul_1072)
            
            # Getting the type of 'items' (line 20)
            items_1073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 24), 'items')
            # Assigning a type to the variable 'items_1073' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'items_1073', items_1073)
            # Testing if the for loop is going to be iterated (line 20)
            # Testing the type of a for loop iterable (line 20)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 20, 12), items_1073)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 20, 12), items_1073):
                # Getting the type of the for loop variable (line 20)
                for_loop_var_1074 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 20, 12), items_1073)
                # Assigning a type to the variable 'item' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'item', for_loop_var_1074)
                # SSA begins for a for statement (line 20)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                
                # Call to range(...): (line 21)
                # Processing the call arguments (line 21)
                int_1076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 31), 'int')
                # Processing the call keyword arguments (line 21)
                kwargs_1077 = {}
                # Getting the type of 'range' (line 21)
                range_1075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 25), 'range', False)
                # Calling range(args, kwargs) (line 21)
                range_call_result_1078 = invoke(stypy.reporting.localization.Localization(__file__, 21, 25), range_1075, *[int_1076], **kwargs_1077)
                
                # Assigning a type to the variable 'range_call_result_1078' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'range_call_result_1078', range_call_result_1078)
                # Testing if the for loop is going to be iterated (line 21)
                # Testing the type of a for loop iterable (line 21)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 21, 16), range_call_result_1078)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 21, 16), range_call_result_1078):
                    # Getting the type of the for loop variable (line 21)
                    for_loop_var_1079 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 21, 16), range_call_result_1078)
                    # Assigning a type to the variable 'j' (line 21)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'j', for_loop_var_1079)
                    # SSA begins for a for statement (line 21)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'j' (line 22)
                    j_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 30), 'j')
                    # Getting the type of 'bound' (line 22)
                    bound_1081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'bound')
                    # Obtaining the member '__getitem__' of a type (line 22)
                    getitem___1082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 24), bound_1081, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
                    subscript_call_result_1083 = invoke(stypy.reporting.localization.Localization(__file__, 22, 24), getitem___1082, j_1080)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'j' (line 22)
                    j_1084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 46), 'j')
                    # Getting the type of 'item' (line 22)
                    item_1085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 35), 'item')
                    # Obtaining the member 'bound' of a type (line 22)
                    bound_1086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 35), item_1085, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 22)
                    getitem___1087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 35), bound_1086, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
                    subscript_call_result_1088 = invoke(stypy.reporting.localization.Localization(__file__, 22, 35), getitem___1087, j_1084)
                    
                    # Applying the binary operator '>' (line 22)
                    result_gt_1089 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 24), '>', subscript_call_result_1083, subscript_call_result_1088)
                    
                    
                    # Getting the type of 'j' (line 22)
                    j_1090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 53), 'j')
                    int_1091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 57), 'int')
                    # Applying the binary operator '>' (line 22)
                    result_gt_1092 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 53), '>', j_1090, int_1091)
                    
                    # Applying the binary operator '^' (line 22)
                    result_xor_1093 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 23), '^', result_gt_1089, result_gt_1092)
                    
                    # Testing if the type of an if condition is none (line 22)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 22, 20), result_xor_1093):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 22)
                        if_condition_1094 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 20), result_xor_1093)
                        # Assigning a type to the variable 'if_condition_1094' (line 22)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'if_condition_1094', if_condition_1094)
                        # SSA begins for if statement (line 22)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Subscript to a Subscript (line 23):
                        
                        # Assigning a Subscript to a Subscript (line 23):
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'j' (line 23)
                        j_1095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 46), 'j')
                        # Getting the type of 'item' (line 23)
                        item_1096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 35), 'item')
                        # Obtaining the member 'bound' of a type (line 23)
                        bound_1097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 35), item_1096, 'bound')
                        # Obtaining the member '__getitem__' of a type (line 23)
                        getitem___1098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 35), bound_1097, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 23)
                        subscript_call_result_1099 = invoke(stypy.reporting.localization.Localization(__file__, 23, 35), getitem___1098, j_1095)
                        
                        # Getting the type of 'bound' (line 23)
                        bound_1100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 24), 'bound')
                        # Getting the type of 'j' (line 23)
                        j_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 'j')
                        # Storing an element on a container (line 23)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 24), bound_1100, (j_1101, subscript_call_result_1099))
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
            kwargs_1123 = {}
            
            # Call to Vector3f_seq(...): (line 24)
            # Processing the call arguments (line 24)
            
            # Obtaining the type of the subscript
            int_1104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 43), 'int')
            int_1105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 45), 'int')
            slice_1106 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 24, 37), int_1104, int_1105, None)
            # Getting the type of 'bound' (line 24)
            bound_1107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 37), 'bound', False)
            # Obtaining the member '__getitem__' of a type (line 24)
            getitem___1108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 37), bound_1107, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 24)
            subscript_call_result_1109 = invoke(stypy.reporting.localization.Localization(__file__, 24, 37), getitem___1108, slice_1106)
            
            # Processing the call keyword arguments (line 24)
            kwargs_1110 = {}
            # Getting the type of 'Vector3f_seq' (line 24)
            Vector3f_seq_1103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'Vector3f_seq', False)
            # Calling Vector3f_seq(args, kwargs) (line 24)
            Vector3f_seq_call_result_1111 = invoke(stypy.reporting.localization.Localization(__file__, 24, 24), Vector3f_seq_1103, *[subscript_call_result_1109], **kwargs_1110)
            
            
            # Call to Vector3f_seq(...): (line 24)
            # Processing the call arguments (line 24)
            
            # Obtaining the type of the subscript
            int_1113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 70), 'int')
            int_1114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 72), 'int')
            slice_1115 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 24, 64), int_1113, int_1114, None)
            # Getting the type of 'bound' (line 24)
            bound_1116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 64), 'bound', False)
            # Obtaining the member '__getitem__' of a type (line 24)
            getitem___1117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 64), bound_1116, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 24)
            subscript_call_result_1118 = invoke(stypy.reporting.localization.Localization(__file__, 24, 64), getitem___1117, slice_1115)
            
            # Processing the call keyword arguments (line 24)
            kwargs_1119 = {}
            # Getting the type of 'Vector3f_seq' (line 24)
            Vector3f_seq_1112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 51), 'Vector3f_seq', False)
            # Calling Vector3f_seq(args, kwargs) (line 24)
            Vector3f_seq_call_result_1120 = invoke(stypy.reporting.localization.Localization(__file__, 24, 51), Vector3f_seq_1112, *[subscript_call_result_1118], **kwargs_1119)
            
            # Applying the binary operator '-' (line 24)
            result_sub_1121 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 24), '-', Vector3f_seq_call_result_1111, Vector3f_seq_call_result_1120)
            
            # Obtaining the member 'as_list' of a type (line 24)
            as_list_1122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 24), result_sub_1121, 'as_list')
            # Calling as_list(args, kwargs) (line 24)
            as_list_call_result_1124 = invoke(stypy.reporting.localization.Localization(__file__, 24, 24), as_list_1122, *[], **kwargs_1123)
            
            # Processing the call keyword arguments (line 24)
            kwargs_1125 = {}
            # Getting the type of 'max' (line 24)
            max_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'max', False)
            # Calling max(args, kwargs) (line 24)
            max_call_result_1126 = invoke(stypy.reporting.localization.Localization(__file__, 24, 19), max_1102, *[as_list_call_result_1124], **kwargs_1125)
            
            # Assigning a type to the variable 'size' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'size', max_call_result_1126)
            
            # Assigning a BinOp to a Attribute (line 25):
            
            # Assigning a BinOp to a Attribute (line 25):
            
            # Obtaining the type of the subscript
            int_1127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 31), 'int')
            int_1128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 33), 'int')
            slice_1129 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 25), int_1127, int_1128, None)
            # Getting the type of 'bound' (line 25)
            bound_1130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'bound')
            # Obtaining the member '__getitem__' of a type (line 25)
            getitem___1131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 25), bound_1130, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 25)
            subscript_call_result_1132 = invoke(stypy.reporting.localization.Localization(__file__, 25, 25), getitem___1131, slice_1129)
            
            
            # Call to as_list(...): (line 25)
            # Processing the call keyword arguments (line 25)
            kwargs_1161 = {}
            
            # Call to clamped(...): (line 25)
            # Processing the call arguments (line 25)
            
            # Call to Vector3f_seq(...): (line 25)
            # Processing the call arguments (line 25)
            
            # Obtaining the type of the subscript
            int_1144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 91), 'int')
            int_1145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 93), 'int')
            slice_1146 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 85), int_1144, int_1145, None)
            # Getting the type of 'bound' (line 25)
            bound_1147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 85), 'bound', False)
            # Obtaining the member '__getitem__' of a type (line 25)
            getitem___1148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 85), bound_1147, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 25)
            subscript_call_result_1149 = invoke(stypy.reporting.localization.Localization(__file__, 25, 85), getitem___1148, slice_1146)
            
            # Processing the call keyword arguments (line 25)
            kwargs_1150 = {}
            # Getting the type of 'Vector3f_seq' (line 25)
            Vector3f_seq_1143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 72), 'Vector3f_seq', False)
            # Calling Vector3f_seq(args, kwargs) (line 25)
            Vector3f_seq_call_result_1151 = invoke(stypy.reporting.localization.Localization(__file__, 25, 72), Vector3f_seq_1143, *[subscript_call_result_1149], **kwargs_1150)
            
            
            # Call to Vector3f_scalar(...): (line 25)
            # Processing the call arguments (line 25)
            # Getting the type of 'size' (line 25)
            size_1153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 115), 'size', False)
            # Processing the call keyword arguments (line 25)
            kwargs_1154 = {}
            # Getting the type of 'Vector3f_scalar' (line 25)
            Vector3f_scalar_1152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 99), 'Vector3f_scalar', False)
            # Calling Vector3f_scalar(args, kwargs) (line 25)
            Vector3f_scalar_call_result_1155 = invoke(stypy.reporting.localization.Localization(__file__, 25, 99), Vector3f_scalar_1152, *[size_1153], **kwargs_1154)
            
            # Applying the binary operator '+' (line 25)
            result_add_1156 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 72), '+', Vector3f_seq_call_result_1151, Vector3f_scalar_call_result_1155)
            
            # Getting the type of 'MAX' (line 25)
            MAX_1157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 122), 'MAX', False)
            # Processing the call keyword arguments (line 25)
            kwargs_1158 = {}
            
            # Call to Vector3f_seq(...): (line 25)
            # Processing the call arguments (line 25)
            
            # Obtaining the type of the subscript
            int_1134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 58), 'int')
            int_1135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 60), 'int')
            slice_1136 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 25, 52), int_1134, int_1135, None)
            # Getting the type of 'bound' (line 25)
            bound_1137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 52), 'bound', False)
            # Obtaining the member '__getitem__' of a type (line 25)
            getitem___1138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 52), bound_1137, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 25)
            subscript_call_result_1139 = invoke(stypy.reporting.localization.Localization(__file__, 25, 52), getitem___1138, slice_1136)
            
            # Processing the call keyword arguments (line 25)
            kwargs_1140 = {}
            # Getting the type of 'Vector3f_seq' (line 25)
            Vector3f_seq_1133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'Vector3f_seq', False)
            # Calling Vector3f_seq(args, kwargs) (line 25)
            Vector3f_seq_call_result_1141 = invoke(stypy.reporting.localization.Localization(__file__, 25, 39), Vector3f_seq_1133, *[subscript_call_result_1139], **kwargs_1140)
            
            # Obtaining the member 'clamped' of a type (line 25)
            clamped_1142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 39), Vector3f_seq_call_result_1141, 'clamped')
            # Calling clamped(args, kwargs) (line 25)
            clamped_call_result_1159 = invoke(stypy.reporting.localization.Localization(__file__, 25, 39), clamped_1142, *[result_add_1156, MAX_1157], **kwargs_1158)
            
            # Obtaining the member 'as_list' of a type (line 25)
            as_list_1160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 39), clamped_call_result_1159, 'as_list')
            # Calling as_list(args, kwargs) (line 25)
            as_list_call_result_1162 = invoke(stypy.reporting.localization.Localization(__file__, 25, 39), as_list_1160, *[], **kwargs_1161)
            
            # Applying the binary operator '+' (line 25)
            result_add_1163 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 25), '+', subscript_call_result_1132, as_list_call_result_1162)
            
            # Getting the type of 'self' (line 25)
            self_1164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'self')
            # Setting the type of the member 'bound' of a type (line 25)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), self_1164, 'bound', result_add_1163)
            # SSA branch for the else part of an if statement (line 16)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Attribute (line 27):
            
            # Assigning a Name to a Attribute (line 27):
            # Getting the type of 'bound' (line 27)
            bound_1165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'bound')
            # Getting the type of 'self' (line 27)
            self_1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'self')
            # Setting the type of the member 'bound' of a type (line 27)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), self_1166, 'bound', bound_1165)
            # SSA join for if statement (line 16)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BoolOp to a Attribute (line 28):
        
        # Assigning a BoolOp to a Attribute (line 28):
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'items' (line 28)
        items_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'items', False)
        # Processing the call keyword arguments (line 28)
        kwargs_1169 = {}
        # Getting the type of 'len' (line 28)
        len_1167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 25), 'len', False)
        # Calling len(args, kwargs) (line 28)
        len_call_result_1170 = invoke(stypy.reporting.localization.Localization(__file__, 28, 25), len_1167, *[items_1168], **kwargs_1169)
        
        # Getting the type of 'MAX_ITEMS' (line 28)
        MAX_ITEMS_1171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 38), 'MAX_ITEMS')
        # Applying the binary operator '>' (line 28)
        result_gt_1172 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 25), '>', len_call_result_1170, MAX_ITEMS_1171)
        
        
        # Getting the type of 'level' (line 28)
        level_1173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 52), 'level')
        # Getting the type of 'MAX_LEVELS' (line 28)
        MAX_LEVELS_1174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 60), 'MAX_LEVELS')
        int_1175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 73), 'int')
        # Applying the binary operator '-' (line 28)
        result_sub_1176 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 60), '-', MAX_LEVELS_1174, int_1175)
        
        # Applying the binary operator '<' (line 28)
        result_lt_1177 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 52), '<', level_1173, result_sub_1176)
        
        # Applying the binary operator 'and' (line 28)
        result_and_keyword_1178 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 25), 'and', result_gt_1172, result_lt_1177)
        
        # Getting the type of 'self' (line 28)
        self_1179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'is_branch' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_1179, 'is_branch', result_and_keyword_1178)
        # Getting the type of 'self' (line 29)
        self_1180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'self')
        # Obtaining the member 'is_branch' of a type (line 29)
        is_branch_1181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 11), self_1180, 'is_branch')
        # Testing if the type of an if condition is none (line 29)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 8), is_branch_1181):
            
            # Assigning a Name to a Attribute (line 51):
            
            # Assigning a Name to a Attribute (line 51):
            # Getting the type of 'items' (line 51)
            items_1370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'items')
            # Getting the type of 'self' (line 51)
            self_1371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
            # Setting the type of the member 'items' of a type (line 51)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_1371, 'items', items_1370)
        else:
            
            # Testing the type of an if condition (line 29)
            if_condition_1182 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 8), is_branch_1181)
            # Assigning a type to the variable 'if_condition_1182' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'if_condition_1182', if_condition_1182)
            # SSA begins for if statement (line 29)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Num to a Name (line 30):
            
            # Assigning a Num to a Name (line 30):
            int_1183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 17), 'int')
            # Assigning a type to the variable 'q1' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'q1', int_1183)
            
            # Assigning a BinOp to a Attribute (line 31):
            
            # Assigning a BinOp to a Attribute (line 31):
            
            # Obtaining an instance of the builtin type 'list' (line 31)
            list_1184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'list')
            # Adding type elements to the builtin type 'list' instance (line 31)
            # Adding element type (line 31)
            # Getting the type of 'None' (line 31)
            None_1185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 27), 'None')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 26), list_1184, None_1185)
            
            int_1186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'int')
            # Applying the binary operator '*' (line 31)
            result_mul_1187 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 26), '*', list_1184, int_1186)
            
            # Getting the type of 'self' (line 31)
            self_1188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'self')
            # Setting the type of the member 'vector' of a type (line 31)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), self_1188, 'vector', result_mul_1187)
            
            
            # Call to range(...): (line 32)
            # Processing the call arguments (line 32)
            int_1190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'int')
            # Processing the call keyword arguments (line 32)
            kwargs_1191 = {}
            # Getting the type of 'range' (line 32)
            range_1189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 21), 'range', False)
            # Calling range(args, kwargs) (line 32)
            range_call_result_1192 = invoke(stypy.reporting.localization.Localization(__file__, 32, 21), range_1189, *[int_1190], **kwargs_1191)
            
            # Assigning a type to the variable 'range_call_result_1192' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'range_call_result_1192', range_call_result_1192)
            # Testing if the for loop is going to be iterated (line 32)
            # Testing the type of a for loop iterable (line 32)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 32, 12), range_call_result_1192)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 32, 12), range_call_result_1192):
                # Getting the type of the for loop variable (line 32)
                for_loop_var_1193 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 32, 12), range_call_result_1192)
                # Assigning a type to the variable 's' (line 32)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 's', for_loop_var_1193)
                # SSA begins for a for statement (line 32)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a List to a Name (line 33):
                
                # Assigning a List to a Name (line 33):
                
                # Obtaining an instance of the builtin type 'list' (line 33)
                list_1194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'list')
                # Adding type elements to the builtin type 'list' instance (line 33)
                
                # Assigning a type to the variable 'sub_bound' (line 33)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'sub_bound', list_1194)
                
                
                # Call to range(...): (line 34)
                # Processing the call arguments (line 34)
                int_1196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'int')
                # Processing the call keyword arguments (line 34)
                kwargs_1197 = {}
                # Getting the type of 'range' (line 34)
                range_1195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'range', False)
                # Calling range(args, kwargs) (line 34)
                range_call_result_1198 = invoke(stypy.reporting.localization.Localization(__file__, 34, 25), range_1195, *[int_1196], **kwargs_1197)
                
                # Assigning a type to the variable 'range_call_result_1198' (line 34)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'range_call_result_1198', range_call_result_1198)
                # Testing if the for loop is going to be iterated (line 34)
                # Testing the type of a for loop iterable (line 34)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 16), range_call_result_1198)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 34, 16), range_call_result_1198):
                    # Getting the type of the for loop variable (line 34)
                    for_loop_var_1199 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 16), range_call_result_1198)
                    # Assigning a type to the variable 'j' (line 34)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'j', for_loop_var_1199)
                    # SSA begins for a for statement (line 34)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a BinOp to a Name (line 35):
                    
                    # Assigning a BinOp to a Name (line 35):
                    # Getting the type of 'j' (line 35)
                    j_1200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'j')
                    int_1201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 28), 'int')
                    # Applying the binary operator '%' (line 35)
                    result_mod_1202 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 24), '%', j_1200, int_1201)
                    
                    # Assigning a type to the variable 'm' (line 35)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'm', result_mod_1202)
                    
                    # Getting the type of 's' (line 36)
                    s_1203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 26), 's')
                    # Getting the type of 'm' (line 36)
                    m_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'm')
                    # Applying the binary operator '>>' (line 36)
                    result_rshift_1205 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 26), '>>', s_1203, m_1204)
                    
                    int_1206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 36), 'int')
                    # Applying the binary operator '&' (line 36)
                    result_and__1207 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 25), '&', result_rshift_1205, int_1206)
                    
                    int_1208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 42), 'int')
                    # Applying the binary operator '!=' (line 36)
                    result_ne_1209 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 24), '!=', result_and__1207, int_1208)
                    
                    
                    # Getting the type of 'j' (line 36)
                    j_1210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 48), 'j')
                    int_1211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 52), 'int')
                    # Applying the binary operator '>' (line 36)
                    result_gt_1212 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 48), '>', j_1210, int_1211)
                    
                    # Applying the binary operator '^' (line 36)
                    result_xor_1213 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 23), '^', result_ne_1209, result_gt_1212)
                    
                    # Testing if the type of an if condition is none (line 36)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 36, 20), result_xor_1213):
                        
                        # Call to append(...): (line 39)
                        # Processing the call arguments (line 39)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'j' (line 39)
                        j_1236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 52), 'j', False)
                        # Getting the type of 'self' (line 39)
                        self_1237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 41), 'self', False)
                        # Obtaining the member 'bound' of a type (line 39)
                        bound_1238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 41), self_1237, 'bound')
                        # Obtaining the member '__getitem__' of a type (line 39)
                        getitem___1239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 41), bound_1238, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
                        subscript_call_result_1240 = invoke(stypy.reporting.localization.Localization(__file__, 39, 41), getitem___1239, j_1236)
                        
                        # Processing the call keyword arguments (line 39)
                        kwargs_1241 = {}
                        # Getting the type of 'sub_bound' (line 39)
                        sub_bound_1234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 24), 'sub_bound', False)
                        # Obtaining the member 'append' of a type (line 39)
                        append_1235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 24), sub_bound_1234, 'append')
                        # Calling append(args, kwargs) (line 39)
                        append_call_result_1242 = invoke(stypy.reporting.localization.Localization(__file__, 39, 24), append_1235, *[subscript_call_result_1240], **kwargs_1241)
                        
                    else:
                        
                        # Testing the type of an if condition (line 36)
                        if_condition_1214 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 20), result_xor_1213)
                        # Assigning a type to the variable 'if_condition_1214' (line 36)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'if_condition_1214', if_condition_1214)
                        # SSA begins for if statement (line 36)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 37)
                        # Processing the call arguments (line 37)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'm' (line 37)
                        m_1217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 53), 'm', False)
                        # Getting the type of 'self' (line 37)
                        self_1218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 42), 'self', False)
                        # Obtaining the member 'bound' of a type (line 37)
                        bound_1219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 42), self_1218, 'bound')
                        # Obtaining the member '__getitem__' of a type (line 37)
                        getitem___1220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 42), bound_1219, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 37)
                        subscript_call_result_1221 = invoke(stypy.reporting.localization.Localization(__file__, 37, 42), getitem___1220, m_1217)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'm' (line 37)
                        m_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 69), 'm', False)
                        int_1223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 73), 'int')
                        # Applying the binary operator '+' (line 37)
                        result_add_1224 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 69), '+', m_1222, int_1223)
                        
                        # Getting the type of 'self' (line 37)
                        self_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 58), 'self', False)
                        # Obtaining the member 'bound' of a type (line 37)
                        bound_1226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 58), self_1225, 'bound')
                        # Obtaining the member '__getitem__' of a type (line 37)
                        getitem___1227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 58), bound_1226, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 37)
                        subscript_call_result_1228 = invoke(stypy.reporting.localization.Localization(__file__, 37, 58), getitem___1227, result_add_1224)
                        
                        # Applying the binary operator '+' (line 37)
                        result_add_1229 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 42), '+', subscript_call_result_1221, subscript_call_result_1228)
                        
                        float_1230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 79), 'float')
                        # Applying the binary operator '*' (line 37)
                        result_mul_1231 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 41), '*', result_add_1229, float_1230)
                        
                        # Processing the call keyword arguments (line 37)
                        kwargs_1232 = {}
                        # Getting the type of 'sub_bound' (line 37)
                        sub_bound_1215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 24), 'sub_bound', False)
                        # Obtaining the member 'append' of a type (line 37)
                        append_1216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 24), sub_bound_1215, 'append')
                        # Calling append(args, kwargs) (line 37)
                        append_call_result_1233 = invoke(stypy.reporting.localization.Localization(__file__, 37, 24), append_1216, *[result_mul_1231], **kwargs_1232)
                        
                        # SSA branch for the else part of an if statement (line 36)
                        module_type_store.open_ssa_branch('else')
                        
                        # Call to append(...): (line 39)
                        # Processing the call arguments (line 39)
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'j' (line 39)
                        j_1236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 52), 'j', False)
                        # Getting the type of 'self' (line 39)
                        self_1237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 41), 'self', False)
                        # Obtaining the member 'bound' of a type (line 39)
                        bound_1238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 41), self_1237, 'bound')
                        # Obtaining the member '__getitem__' of a type (line 39)
                        getitem___1239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 41), bound_1238, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
                        subscript_call_result_1240 = invoke(stypy.reporting.localization.Localization(__file__, 39, 41), getitem___1239, j_1236)
                        
                        # Processing the call keyword arguments (line 39)
                        kwargs_1241 = {}
                        # Getting the type of 'sub_bound' (line 39)
                        sub_bound_1234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 24), 'sub_bound', False)
                        # Obtaining the member 'append' of a type (line 39)
                        append_1235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 24), sub_bound_1234, 'append')
                        # Calling append(args, kwargs) (line 39)
                        append_call_result_1242 = invoke(stypy.reporting.localization.Localization(__file__, 39, 24), append_1235, *[subscript_call_result_1240], **kwargs_1241)
                        
                        # SSA join for if statement (line 36)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Assigning a List to a Name (line 40):
                
                # Assigning a List to a Name (line 40):
                
                # Obtaining an instance of the builtin type 'list' (line 40)
                list_1243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 28), 'list')
                # Adding type elements to the builtin type 'list' instance (line 40)
                
                # Assigning a type to the variable 'sub_items' (line 40)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'sub_items', list_1243)
                
                # Getting the type of 'items' (line 41)
                items_1244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'items')
                # Assigning a type to the variable 'items_1244' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'items_1244', items_1244)
                # Testing if the for loop is going to be iterated (line 41)
                # Testing the type of a for loop iterable (line 41)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 16), items_1244)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 41, 16), items_1244):
                    # Getting the type of the for loop variable (line 41)
                    for_loop_var_1245 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 16), items_1244)
                    # Assigning a type to the variable 'item' (line 41)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'item', for_loop_var_1245)
                    # SSA begins for a for statement (line 41)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Evaluating a boolean operation
                    
                    
                    # Obtaining the type of the subscript
                    int_1246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 34), 'int')
                    # Getting the type of 'item' (line 42)
                    item_1247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'item')
                    # Obtaining the member 'bound' of a type (line 42)
                    bound_1248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 23), item_1247, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 42)
                    getitem___1249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 23), bound_1248, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                    subscript_call_result_1250 = invoke(stypy.reporting.localization.Localization(__file__, 42, 23), getitem___1249, int_1246)
                    
                    
                    # Obtaining the type of the subscript
                    int_1251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 50), 'int')
                    # Getting the type of 'sub_bound' (line 42)
                    sub_bound_1252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 40), 'sub_bound')
                    # Obtaining the member '__getitem__' of a type (line 42)
                    getitem___1253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 40), sub_bound_1252, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                    subscript_call_result_1254 = invoke(stypy.reporting.localization.Localization(__file__, 42, 40), getitem___1253, int_1251)
                    
                    # Applying the binary operator '>=' (line 42)
                    result_ge_1255 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), '>=', subscript_call_result_1250, subscript_call_result_1254)
                    
                    
                    
                    # Obtaining the type of the subscript
                    int_1256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 68), 'int')
                    # Getting the type of 'item' (line 42)
                    item_1257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 57), 'item')
                    # Obtaining the member 'bound' of a type (line 42)
                    bound_1258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 57), item_1257, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 42)
                    getitem___1259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 57), bound_1258, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                    subscript_call_result_1260 = invoke(stypy.reporting.localization.Localization(__file__, 42, 57), getitem___1259, int_1256)
                    
                    
                    # Obtaining the type of the subscript
                    int_1261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 83), 'int')
                    # Getting the type of 'sub_bound' (line 42)
                    sub_bound_1262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 73), 'sub_bound')
                    # Obtaining the member '__getitem__' of a type (line 42)
                    getitem___1263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 73), sub_bound_1262, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                    subscript_call_result_1264 = invoke(stypy.reporting.localization.Localization(__file__, 42, 73), getitem___1263, int_1261)
                    
                    # Applying the binary operator '<' (line 42)
                    result_lt_1265 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 57), '<', subscript_call_result_1260, subscript_call_result_1264)
                    
                    # Applying the binary operator 'and' (line 42)
                    result_and_keyword_1266 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), 'and', result_ge_1255, result_lt_1265)
                    
                    
                    # Obtaining the type of the subscript
                    int_1267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'int')
                    # Getting the type of 'item' (line 43)
                    item_1268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'item')
                    # Obtaining the member 'bound' of a type (line 43)
                    bound_1269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 23), item_1268, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 43)
                    getitem___1270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 23), bound_1269, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
                    subscript_call_result_1271 = invoke(stypy.reporting.localization.Localization(__file__, 43, 23), getitem___1270, int_1267)
                    
                    
                    # Obtaining the type of the subscript
                    int_1272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 50), 'int')
                    # Getting the type of 'sub_bound' (line 43)
                    sub_bound_1273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 40), 'sub_bound')
                    # Obtaining the member '__getitem__' of a type (line 43)
                    getitem___1274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 40), sub_bound_1273, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
                    subscript_call_result_1275 = invoke(stypy.reporting.localization.Localization(__file__, 43, 40), getitem___1274, int_1272)
                    
                    # Applying the binary operator '>=' (line 43)
                    result_ge_1276 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 23), '>=', subscript_call_result_1271, subscript_call_result_1275)
                    
                    # Applying the binary operator 'and' (line 42)
                    result_and_keyword_1277 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), 'and', result_and_keyword_1266, result_ge_1276)
                    
                    
                    # Obtaining the type of the subscript
                    int_1278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 68), 'int')
                    # Getting the type of 'item' (line 43)
                    item_1279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 57), 'item')
                    # Obtaining the member 'bound' of a type (line 43)
                    bound_1280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 57), item_1279, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 43)
                    getitem___1281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 57), bound_1280, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
                    subscript_call_result_1282 = invoke(stypy.reporting.localization.Localization(__file__, 43, 57), getitem___1281, int_1278)
                    
                    
                    # Obtaining the type of the subscript
                    int_1283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 83), 'int')
                    # Getting the type of 'sub_bound' (line 43)
                    sub_bound_1284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 73), 'sub_bound')
                    # Obtaining the member '__getitem__' of a type (line 43)
                    getitem___1285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 73), sub_bound_1284, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
                    subscript_call_result_1286 = invoke(stypy.reporting.localization.Localization(__file__, 43, 73), getitem___1285, int_1283)
                    
                    # Applying the binary operator '<' (line 43)
                    result_lt_1287 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 57), '<', subscript_call_result_1282, subscript_call_result_1286)
                    
                    # Applying the binary operator 'and' (line 42)
                    result_and_keyword_1288 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), 'and', result_and_keyword_1277, result_lt_1287)
                    
                    
                    # Obtaining the type of the subscript
                    int_1289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'int')
                    # Getting the type of 'item' (line 44)
                    item_1290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'item')
                    # Obtaining the member 'bound' of a type (line 44)
                    bound_1291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 23), item_1290, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___1292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 23), bound_1291, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_1293 = invoke(stypy.reporting.localization.Localization(__file__, 44, 23), getitem___1292, int_1289)
                    
                    
                    # Obtaining the type of the subscript
                    int_1294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 50), 'int')
                    # Getting the type of 'sub_bound' (line 44)
                    sub_bound_1295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 40), 'sub_bound')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___1296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 40), sub_bound_1295, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_1297 = invoke(stypy.reporting.localization.Localization(__file__, 44, 40), getitem___1296, int_1294)
                    
                    # Applying the binary operator '>=' (line 44)
                    result_ge_1298 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 23), '>=', subscript_call_result_1293, subscript_call_result_1297)
                    
                    # Applying the binary operator 'and' (line 42)
                    result_and_keyword_1299 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), 'and', result_and_keyword_1288, result_ge_1298)
                    
                    
                    # Obtaining the type of the subscript
                    int_1300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 68), 'int')
                    # Getting the type of 'item' (line 44)
                    item_1301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 57), 'item')
                    # Obtaining the member 'bound' of a type (line 44)
                    bound_1302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 57), item_1301, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___1303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 57), bound_1302, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_1304 = invoke(stypy.reporting.localization.Localization(__file__, 44, 57), getitem___1303, int_1300)
                    
                    
                    # Obtaining the type of the subscript
                    int_1305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 83), 'int')
                    # Getting the type of 'sub_bound' (line 44)
                    sub_bound_1306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 73), 'sub_bound')
                    # Obtaining the member '__getitem__' of a type (line 44)
                    getitem___1307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 73), sub_bound_1306, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                    subscript_call_result_1308 = invoke(stypy.reporting.localization.Localization(__file__, 44, 73), getitem___1307, int_1305)
                    
                    # Applying the binary operator '<' (line 44)
                    result_lt_1309 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 57), '<', subscript_call_result_1304, subscript_call_result_1308)
                    
                    # Applying the binary operator 'and' (line 42)
                    result_and_keyword_1310 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 23), 'and', result_and_keyword_1299, result_lt_1309)
                    
                    # Testing if the type of an if condition is none (line 42)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 20), result_and_keyword_1310):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 42)
                        if_condition_1311 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 20), result_and_keyword_1310)
                        # Assigning a type to the variable 'if_condition_1311' (line 42)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'if_condition_1311', if_condition_1311)
                        # SSA begins for if statement (line 42)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 45)
                        # Processing the call arguments (line 45)
                        # Getting the type of 'item' (line 45)
                        item_1314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 44), 'item', False)
                        # Processing the call keyword arguments (line 45)
                        kwargs_1315 = {}
                        # Getting the type of 'sub_items' (line 45)
                        sub_items_1312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'sub_items', False)
                        # Obtaining the member 'append' of a type (line 45)
                        append_1313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 27), sub_items_1312, 'append')
                        # Calling append(args, kwargs) (line 45)
                        append_call_result_1316 = invoke(stypy.reporting.localization.Localization(__file__, 45, 27), append_1313, *[item_1314], **kwargs_1315)
                        
                        # SSA join for if statement (line 42)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Getting the type of 'q1' (line 46)
                q1_1317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'q1')
                
                
                
                # Call to len(...): (line 46)
                # Processing the call arguments (line 46)
                # Getting the type of 'sub_items' (line 46)
                sub_items_1319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'sub_items', False)
                # Processing the call keyword arguments (line 46)
                kwargs_1320 = {}
                # Getting the type of 'len' (line 46)
                len_1318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'len', False)
                # Calling len(args, kwargs) (line 46)
                len_call_result_1321 = invoke(stypy.reporting.localization.Localization(__file__, 46, 27), len_1318, *[sub_items_1319], **kwargs_1320)
                
                
                # Call to len(...): (line 46)
                # Processing the call arguments (line 46)
                # Getting the type of 'items' (line 46)
                items_1323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 49), 'items', False)
                # Processing the call keyword arguments (line 46)
                kwargs_1324 = {}
                # Getting the type of 'len' (line 46)
                len_1322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 45), 'len', False)
                # Calling len(args, kwargs) (line 46)
                len_call_result_1325 = invoke(stypy.reporting.localization.Localization(__file__, 46, 45), len_1322, *[items_1323], **kwargs_1324)
                
                # Applying the binary operator '==' (line 46)
                result_eq_1326 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 27), '==', len_call_result_1321, len_call_result_1325)
                
                # Testing the type of an if expression (line 46)
                is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 22), result_eq_1326)
                # SSA begins for if expression (line 46)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                int_1327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'int')
                # SSA branch for the else part of an if expression (line 46)
                module_type_store.open_ssa_branch('if expression else')
                int_1328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 61), 'int')
                # SSA join for if expression (line 46)
                module_type_store = module_type_store.join_ssa_context()
                if_exp_1329 = union_type.UnionType.add(int_1327, int_1328)
                
                # Applying the binary operator '+=' (line 46)
                result_iadd_1330 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 16), '+=', q1_1317, if_exp_1329)
                # Assigning a type to the variable 'q1' (line 46)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'q1', result_iadd_1330)
                
                
                # Assigning a Compare to a Name (line 47):
                
                # Assigning a Compare to a Name (line 47):
                
                
                # Obtaining the type of the subscript
                int_1331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 32), 'int')
                # Getting the type of 'sub_bound' (line 47)
                sub_bound_1332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'sub_bound')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___1333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 22), sub_bound_1332, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_1334 = invoke(stypy.reporting.localization.Localization(__file__, 47, 22), getitem___1333, int_1331)
                
                
                # Obtaining the type of the subscript
                int_1335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 47), 'int')
                # Getting the type of 'sub_bound' (line 47)
                sub_bound_1336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'sub_bound')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___1337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 37), sub_bound_1336, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_1338 = invoke(stypy.reporting.localization.Localization(__file__, 47, 37), getitem___1337, int_1335)
                
                # Applying the binary operator '-' (line 47)
                result_sub_1339 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 22), '-', subscript_call_result_1334, subscript_call_result_1338)
                
                # Getting the type of 'TOLERANCE' (line 47)
                TOLERANCE_1340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 54), 'TOLERANCE')
                float_1341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 66), 'float')
                # Applying the binary operator '*' (line 47)
                result_mul_1342 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 54), '*', TOLERANCE_1340, float_1341)
                
                # Applying the binary operator '<' (line 47)
                result_lt_1343 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 21), '<', result_sub_1339, result_mul_1342)
                
                # Assigning a type to the variable 'q2' (line 47)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'q2', result_lt_1343)
                
                
                # Call to len(...): (line 48)
                # Processing the call arguments (line 48)
                # Getting the type of 'sub_items' (line 48)
                sub_items_1345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'sub_items', False)
                # Processing the call keyword arguments (line 48)
                kwargs_1346 = {}
                # Getting the type of 'len' (line 48)
                len_1344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'len', False)
                # Calling len(args, kwargs) (line 48)
                len_call_result_1347 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), len_1344, *[sub_items_1345], **kwargs_1346)
                
                int_1348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 36), 'int')
                # Applying the binary operator '>' (line 48)
                result_gt_1349 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 19), '>', len_call_result_1347, int_1348)
                
                # Testing if the type of an if condition is none (line 48)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 48, 16), result_gt_1349):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 48)
                    if_condition_1350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 16), result_gt_1349)
                    # Assigning a type to the variable 'if_condition_1350' (line 48)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'if_condition_1350', if_condition_1350)
                    # SSA begins for if statement (line 48)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Subscript (line 49):
                    
                    # Assigning a Call to a Subscript (line 49):
                    
                    # Call to SpatialIndex(...): (line 49)
                    # Processing the call arguments (line 49)
                    # Getting the type of 'None' (line 49)
                    None_1352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 50), 'None', False)
                    # Getting the type of 'sub_bound' (line 49)
                    sub_bound_1353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 56), 'sub_bound', False)
                    # Getting the type of 'sub_items' (line 49)
                    sub_items_1354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 67), 'sub_items', False)
                    
                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'q1' (line 49)
                    q1_1355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 92), 'q1', False)
                    int_1356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 97), 'int')
                    # Applying the binary operator '>' (line 49)
                    result_gt_1357 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 92), '>', q1_1355, int_1356)
                    
                    # Getting the type of 'q2' (line 49)
                    q2_1358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 102), 'q2', False)
                    # Applying the binary operator 'or' (line 49)
                    result_or_keyword_1359 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 92), 'or', result_gt_1357, q2_1358)
                    
                    # Testing the type of an if expression (line 49)
                    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 78), result_or_keyword_1359)
                    # SSA begins for if expression (line 49)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                    # Getting the type of 'MAX_LEVELS' (line 49)
                    MAX_LEVELS_1360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 78), 'MAX_LEVELS', False)
                    # SSA branch for the else part of an if expression (line 49)
                    module_type_store.open_ssa_branch('if expression else')
                    # Getting the type of 'level' (line 49)
                    level_1361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 110), 'level', False)
                    int_1362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 118), 'int')
                    # Applying the binary operator '+' (line 49)
                    result_add_1363 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 110), '+', level_1361, int_1362)
                    
                    # SSA join for if expression (line 49)
                    module_type_store = module_type_store.join_ssa_context()
                    if_exp_1364 = union_type.UnionType.add(MAX_LEVELS_1360, result_add_1363)
                    
                    # Processing the call keyword arguments (line 49)
                    kwargs_1365 = {}
                    # Getting the type of 'SpatialIndex' (line 49)
                    SpatialIndex_1351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 37), 'SpatialIndex', False)
                    # Calling SpatialIndex(args, kwargs) (line 49)
                    SpatialIndex_call_result_1366 = invoke(stypy.reporting.localization.Localization(__file__, 49, 37), SpatialIndex_1351, *[None_1352, sub_bound_1353, sub_items_1354, if_exp_1364], **kwargs_1365)
                    
                    # Getting the type of 'self' (line 49)
                    self_1367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 20), 'self')
                    # Obtaining the member 'vector' of a type (line 49)
                    vector_1368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 20), self_1367, 'vector')
                    # Getting the type of 's' (line 49)
                    s_1369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 32), 's')
                    # Storing an element on a container (line 49)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 20), vector_1368, (s_1369, SpatialIndex_call_result_1366))
                    # SSA join for if statement (line 48)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA branch for the else part of an if statement (line 29)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Name to a Attribute (line 51):
            
            # Assigning a Name to a Attribute (line 51):
            # Getting the type of 'items' (line 51)
            items_1370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'items')
            # Getting the type of 'self' (line 51)
            self_1371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'self')
            # Setting the type of the member 'items' of a type (line 51)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), self_1371, 'items', items_1370)
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
        None_1372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 74), 'None')
        defaults = [None_1372]
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
        start_1373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'start')
        # Testing the type of an if expression (line 54)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 16), start_1373)
        # SSA begins for if expression (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'start' (line 54)
        start_1374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'start')
        # SSA branch for the else part of an if expression (line 54)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'ray_origin' (line 54)
        ray_origin_1375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 36), 'ray_origin')
        # SSA join for if expression (line 54)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1376 = union_type.UnionType.add(start_1374, ray_origin_1375)
        
        # Assigning a type to the variable 'start' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'start', if_exp_1376)
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Name to a Name (line 55):
        # Getting the type of 'None' (line 55)
        None_1377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 36), 'None')
        # Assigning a type to the variable 'hit_position' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'hit_position', None_1377)
        
        # Assigning a Name to a Name (line 55):
        # Getting the type of 'hit_position' (line 55)
        hit_position_1378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'hit_position')
        # Assigning a type to the variable 'hit_object' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'hit_object', hit_position_1378)
        
        # Assigning a Attribute to a Tuple (line 56):
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_1379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
        # Getting the type of 'self' (line 56)
        self_1380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'self')
        # Obtaining the member 'bound' of a type (line 56)
        bound_1381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 33), self_1380, 'bound')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___1382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), bound_1381, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_1383 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___1382, int_1379)
        
        # Assigning a type to the variable 'tuple_var_assignment_1042' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1042', subscript_call_result_1383)
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_1384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
        # Getting the type of 'self' (line 56)
        self_1385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'self')
        # Obtaining the member 'bound' of a type (line 56)
        bound_1386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 33), self_1385, 'bound')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___1387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), bound_1386, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_1388 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___1387, int_1384)
        
        # Assigning a type to the variable 'tuple_var_assignment_1043' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1043', subscript_call_result_1388)
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_1389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
        # Getting the type of 'self' (line 56)
        self_1390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'self')
        # Obtaining the member 'bound' of a type (line 56)
        bound_1391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 33), self_1390, 'bound')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___1392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), bound_1391, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_1393 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___1392, int_1389)
        
        # Assigning a type to the variable 'tuple_var_assignment_1044' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1044', subscript_call_result_1393)
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_1394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
        # Getting the type of 'self' (line 56)
        self_1395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'self')
        # Obtaining the member 'bound' of a type (line 56)
        bound_1396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 33), self_1395, 'bound')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___1397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), bound_1396, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_1398 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___1397, int_1394)
        
        # Assigning a type to the variable 'tuple_var_assignment_1045' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1045', subscript_call_result_1398)
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_1399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
        # Getting the type of 'self' (line 56)
        self_1400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'self')
        # Obtaining the member 'bound' of a type (line 56)
        bound_1401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 33), self_1400, 'bound')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___1402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), bound_1401, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_1403 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___1402, int_1399)
        
        # Assigning a type to the variable 'tuple_var_assignment_1046' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1046', subscript_call_result_1403)
        
        # Assigning a Subscript to a Name (line 56):
        
        # Obtaining the type of the subscript
        int_1404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 8), 'int')
        # Getting the type of 'self' (line 56)
        self_1405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 33), 'self')
        # Obtaining the member 'bound' of a type (line 56)
        bound_1406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 33), self_1405, 'bound')
        # Obtaining the member '__getitem__' of a type (line 56)
        getitem___1407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), bound_1406, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 56)
        subscript_call_result_1408 = invoke(stypy.reporting.localization.Localization(__file__, 56, 8), getitem___1407, int_1404)
        
        # Assigning a type to the variable 'tuple_var_assignment_1047' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1047', subscript_call_result_1408)
        
        # Assigning a Name to a Name (line 56):
        # Getting the type of 'tuple_var_assignment_1042' (line 56)
        tuple_var_assignment_1042_1409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1042')
        # Assigning a type to the variable 'b0' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'b0', tuple_var_assignment_1042_1409)
        
        # Assigning a Name to a Name (line 56):
        # Getting the type of 'tuple_var_assignment_1043' (line 56)
        tuple_var_assignment_1043_1410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1043')
        # Assigning a type to the variable 'b1' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'b1', tuple_var_assignment_1043_1410)
        
        # Assigning a Name to a Name (line 56):
        # Getting the type of 'tuple_var_assignment_1044' (line 56)
        tuple_var_assignment_1044_1411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1044')
        # Assigning a type to the variable 'b2' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'b2', tuple_var_assignment_1044_1411)
        
        # Assigning a Name to a Name (line 56):
        # Getting the type of 'tuple_var_assignment_1045' (line 56)
        tuple_var_assignment_1045_1412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1045')
        # Assigning a type to the variable 'b3' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'b3', tuple_var_assignment_1045_1412)
        
        # Assigning a Name to a Name (line 56):
        # Getting the type of 'tuple_var_assignment_1046' (line 56)
        tuple_var_assignment_1046_1413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1046')
        # Assigning a type to the variable 'b4' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'b4', tuple_var_assignment_1046_1413)
        
        # Assigning a Name to a Name (line 56):
        # Getting the type of 'tuple_var_assignment_1047' (line 56)
        tuple_var_assignment_1047_1414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'tuple_var_assignment_1047')
        # Assigning a type to the variable 'b5' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 28), 'b5', tuple_var_assignment_1047_1414)
        # Getting the type of 'self' (line 57)
        self_1415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'self')
        # Obtaining the member 'is_branch' of a type (line 57)
        is_branch_1416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), self_1415, 'is_branch')
        # Testing if the type of an if condition is none (line 57)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 8), is_branch_1416):
            
            # Assigning a Num to a Name (line 86):
            
            # Assigning a Num to a Name (line 86):
            float_1578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'float')
            # Assigning a type to the variable 'nearest_distance' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'nearest_distance', float_1578)
            
            # Getting the type of 'self' (line 87)
            self_1579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'self')
            # Obtaining the member 'items' of a type (line 87)
            items_1580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 24), self_1579, 'items')
            # Assigning a type to the variable 'items_1580' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'items_1580', items_1580)
            # Testing if the for loop is going to be iterated (line 87)
            # Testing the type of a for loop iterable (line 87)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 12), items_1580)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 87, 12), items_1580):
                # Getting the type of the for loop variable (line 87)
                for_loop_var_1581 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 12), items_1580)
                # Assigning a type to the variable 'item' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'item', for_loop_var_1581)
                # SSA begins for a for statement (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'item' (line 88)
                item_1582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'item')
                # Getting the type of 'last_hit' (line 88)
                last_hit_1583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 'last_hit')
                # Applying the binary operator '!=' (line 88)
                result_ne_1584 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 19), '!=', item_1582, last_hit_1583)
                
                # Testing if the type of an if condition is none (line 88)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 16), result_ne_1584):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 88)
                    if_condition_1585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 16), result_ne_1584)
                    # Assigning a type to the variable 'if_condition_1585' (line 88)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'if_condition_1585', if_condition_1585)
                    # SSA begins for if statement (line 88)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Call to get_intersection(...): (line 89)
                    # Processing the call arguments (line 89)
                    # Getting the type of 'ray_origin' (line 89)
                    ray_origin_1588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 53), 'ray_origin', False)
                    # Getting the type of 'ray_direction' (line 89)
                    ray_direction_1589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 65), 'ray_direction', False)
                    # Processing the call keyword arguments (line 89)
                    kwargs_1590 = {}
                    # Getting the type of 'item' (line 89)
                    item_1586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'item', False)
                    # Obtaining the member 'get_intersection' of a type (line 89)
                    get_intersection_1587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 31), item_1586, 'get_intersection')
                    # Calling get_intersection(args, kwargs) (line 89)
                    get_intersection_call_result_1591 = invoke(stypy.reporting.localization.Localization(__file__, 89, 31), get_intersection_1587, *[ray_origin_1588, ray_direction_1589], **kwargs_1590)
                    
                    # Assigning a type to the variable 'distance' (line 89)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'distance', get_intersection_call_result_1591)
                    
                    float_1592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'float')
                    # Getting the type of 'distance' (line 90)
                    distance_1593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'distance')
                    # Applying the binary operator '<=' (line 90)
                    result_le_1594 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '<=', float_1592, distance_1593)
                    # Getting the type of 'nearest_distance' (line 90)
                    nearest_distance_1595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 41), 'nearest_distance')
                    # Applying the binary operator '<' (line 90)
                    result_lt_1596 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '<', distance_1593, nearest_distance_1595)
                    # Applying the binary operator '&' (line 90)
                    result_and__1597 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '&', result_le_1594, result_lt_1596)
                    
                    # Testing if the type of an if condition is none (line 90)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 90, 20), result_and__1597):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 90)
                        if_condition_1598 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 20), result_and__1597)
                        # Assigning a type to the variable 'if_condition_1598' (line 90)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'if_condition_1598', if_condition_1598)
                        # SSA begins for if statement (line 90)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 91):
                        
                        # Assigning a BinOp to a Name (line 91):
                        # Getting the type of 'ray_origin' (line 91)
                        ray_origin_1599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 30), 'ray_origin')
                        # Getting the type of 'ray_direction' (line 91)
                        ray_direction_1600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'ray_direction')
                        # Getting the type of 'distance' (line 91)
                        distance_1601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 59), 'distance')
                        # Applying the binary operator '*' (line 91)
                        result_mul_1602 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 43), '*', ray_direction_1600, distance_1601)
                        
                        # Applying the binary operator '+' (line 91)
                        result_add_1603 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 30), '+', ray_origin_1599, result_mul_1602)
                        
                        # Assigning a type to the variable 'hit' (line 91)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'hit', result_add_1603)
                        
                        # Evaluating a boolean operation
                        
                        # Getting the type of 'b0' (line 92)
                        b0_1604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'b0')
                        # Getting the type of 'hit' (line 92)
                        hit_1605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'hit')
                        # Obtaining the member 'x' of a type (line 92)
                        x_1606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 33), hit_1605, 'x')
                        # Applying the binary operator '-' (line 92)
                        result_sub_1607 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 28), '-', b0_1604, x_1606)
                        
                        # Getting the type of 'TOLERANCE' (line 92)
                        TOLERANCE_1608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 92)
                        result_le_1609 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 28), '<=', result_sub_1607, TOLERANCE_1608)
                        
                        
                        # Getting the type of 'hit' (line 93)
                        hit_1610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'hit')
                        # Obtaining the member 'x' of a type (line 93)
                        x_1611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 28), hit_1610, 'x')
                        # Getting the type of 'b3' (line 93)
                        b3_1612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 36), 'b3')
                        # Applying the binary operator '-' (line 93)
                        result_sub_1613 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 28), '-', x_1611, b3_1612)
                        
                        # Getting the type of 'TOLERANCE' (line 93)
                        TOLERANCE_1614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 93)
                        result_le_1615 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 28), '<=', result_sub_1613, TOLERANCE_1614)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1616 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_le_1609, result_le_1615)
                        
                        # Getting the type of 'b1' (line 94)
                        b1_1617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'b1')
                        # Getting the type of 'hit' (line 94)
                        hit_1618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'hit')
                        # Obtaining the member 'y' of a type (line 94)
                        y_1619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 33), hit_1618, 'y')
                        # Applying the binary operator '-' (line 94)
                        result_sub_1620 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 28), '-', b1_1617, y_1619)
                        
                        # Getting the type of 'TOLERANCE' (line 94)
                        TOLERANCE_1621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 94)
                        result_le_1622 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 28), '<=', result_sub_1620, TOLERANCE_1621)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1623 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1616, result_le_1622)
                        
                        # Getting the type of 'hit' (line 95)
                        hit_1624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'hit')
                        # Obtaining the member 'y' of a type (line 95)
                        y_1625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 28), hit_1624, 'y')
                        # Getting the type of 'b4' (line 95)
                        b4_1626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'b4')
                        # Applying the binary operator '-' (line 95)
                        result_sub_1627 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 28), '-', y_1625, b4_1626)
                        
                        # Getting the type of 'TOLERANCE' (line 95)
                        TOLERANCE_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 95)
                        result_le_1629 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 28), '<=', result_sub_1627, TOLERANCE_1628)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1630 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1623, result_le_1629)
                        
                        # Getting the type of 'b2' (line 96)
                        b2_1631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 'b2')
                        # Getting the type of 'hit' (line 96)
                        hit_1632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'hit')
                        # Obtaining the member 'z' of a type (line 96)
                        z_1633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 33), hit_1632, 'z')
                        # Applying the binary operator '-' (line 96)
                        result_sub_1634 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 28), '-', b2_1631, z_1633)
                        
                        # Getting the type of 'TOLERANCE' (line 96)
                        TOLERANCE_1635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 96)
                        result_le_1636 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 28), '<=', result_sub_1634, TOLERANCE_1635)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1637 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1630, result_le_1636)
                        
                        # Getting the type of 'hit' (line 97)
                        hit_1638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'hit')
                        # Obtaining the member 'z' of a type (line 97)
                        z_1639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 28), hit_1638, 'z')
                        # Getting the type of 'b5' (line 97)
                        b5_1640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 36), 'b5')
                        # Applying the binary operator '-' (line 97)
                        result_sub_1641 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 28), '-', z_1639, b5_1640)
                        
                        # Getting the type of 'TOLERANCE' (line 97)
                        TOLERANCE_1642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 97)
                        result_le_1643 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 28), '<=', result_sub_1641, TOLERANCE_1642)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1644 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1637, result_le_1643)
                        
                        # Testing if the type of an if condition is none (line 92)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 92, 24), result_and_keyword_1644):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 92)
                            if_condition_1645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 24), result_and_keyword_1644)
                            # Assigning a type to the variable 'if_condition_1645' (line 92)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'if_condition_1645', if_condition_1645)
                            # SSA begins for if statement (line 92)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Name (line 98):
                            
                            # Assigning a Name to a Name (line 98):
                            # Getting the type of 'item' (line 98)
                            item_1646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 44), 'item')
                            # Assigning a type to the variable 'hit_object' (line 98)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'hit_object', item_1646)
                            
                            # Assigning a Name to a Name (line 99):
                            
                            # Assigning a Name to a Name (line 99):
                            # Getting the type of 'hit' (line 99)
                            hit_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 46), 'hit')
                            # Assigning a type to the variable 'hit_position' (line 99)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 31), 'hit_position', hit_1647)
                            
                            # Assigning a Name to a Name (line 100):
                            
                            # Assigning a Name to a Name (line 100):
                            # Getting the type of 'distance' (line 100)
                            distance_1648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 50), 'distance')
                            # Assigning a type to the variable 'nearest_distance' (line 100)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'nearest_distance', distance_1648)
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
            if_condition_1417 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 8), is_branch_1416)
            # Assigning a type to the variable 'if_condition_1417' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'if_condition_1417', if_condition_1417)
            # SSA begins for if statement (line 57)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a IfExp to a Name (line 58):
            
            # Assigning a IfExp to a Name (line 58):
            
            
            # Getting the type of 'start' (line 58)
            start_1418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'start')
            # Obtaining the member 'x' of a type (line 58)
            x_1419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 28), start_1418, 'x')
            # Getting the type of 'b0' (line 58)
            b0_1420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 40), 'b0')
            # Getting the type of 'b3' (line 58)
            b3_1421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'b3')
            # Applying the binary operator '+' (line 58)
            result_add_1422 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 40), '+', b0_1420, b3_1421)
            
            float_1423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 49), 'float')
            # Applying the binary operator '*' (line 58)
            result_mul_1424 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 39), '*', result_add_1422, float_1423)
            
            # Applying the binary operator '>=' (line 58)
            result_ge_1425 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 28), '>=', x_1419, result_mul_1424)
            
            # Testing the type of an if expression (line 58)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 23), result_ge_1425)
            # SSA begins for if expression (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
            int_1426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'int')
            # SSA branch for the else part of an if expression (line 58)
            module_type_store.open_ssa_branch('if expression else')
            int_1427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 58), 'int')
            # SSA join for if expression (line 58)
            module_type_store = module_type_store.join_ssa_context()
            if_exp_1428 = union_type.UnionType.add(int_1426, int_1427)
            
            # Assigning a type to the variable 'sub_cell' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'sub_cell', if_exp_1428)
            
            # Getting the type of 'start' (line 59)
            start_1429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'start')
            # Obtaining the member 'y' of a type (line 59)
            y_1430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), start_1429, 'y')
            # Getting the type of 'b1' (line 59)
            b1_1431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 27), 'b1')
            # Getting the type of 'b4' (line 59)
            b4_1432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 30), 'b4')
            # Applying the binary operator '+' (line 59)
            result_add_1433 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 27), '+', b1_1431, b4_1432)
            
            float_1434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 36), 'float')
            # Applying the binary operator '*' (line 59)
            result_mul_1435 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 26), '*', result_add_1433, float_1434)
            
            # Applying the binary operator '>=' (line 59)
            result_ge_1436 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 15), '>=', y_1430, result_mul_1435)
            
            # Testing if the type of an if condition is none (line 59)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 12), result_ge_1436):
                pass
            else:
                
                # Testing the type of an if condition (line 59)
                if_condition_1437 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 12), result_ge_1436)
                # Assigning a type to the variable 'if_condition_1437' (line 59)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'if_condition_1437', if_condition_1437)
                # SSA begins for if statement (line 59)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'sub_cell' (line 60)
                sub_cell_1438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'sub_cell')
                int_1439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 28), 'int')
                # Applying the binary operator '|=' (line 60)
                result_ior_1440 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 16), '|=', sub_cell_1438, int_1439)
                # Assigning a type to the variable 'sub_cell' (line 60)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'sub_cell', result_ior_1440)
                
                # SSA join for if statement (line 59)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'start' (line 61)
            start_1441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'start')
            # Obtaining the member 'z' of a type (line 61)
            z_1442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), start_1441, 'z')
            # Getting the type of 'b2' (line 61)
            b2_1443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 27), 'b2')
            # Getting the type of 'b5' (line 61)
            b5_1444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'b5')
            # Applying the binary operator '+' (line 61)
            result_add_1445 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 27), '+', b2_1443, b5_1444)
            
            float_1446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 36), 'float')
            # Applying the binary operator '*' (line 61)
            result_mul_1447 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 26), '*', result_add_1445, float_1446)
            
            # Applying the binary operator '>=' (line 61)
            result_ge_1448 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 15), '>=', z_1442, result_mul_1447)
            
            # Testing if the type of an if condition is none (line 61)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 61, 12), result_ge_1448):
                pass
            else:
                
                # Testing the type of an if condition (line 61)
                if_condition_1449 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 12), result_ge_1448)
                # Assigning a type to the variable 'if_condition_1449' (line 61)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'if_condition_1449', if_condition_1449)
                # SSA begins for if statement (line 61)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'sub_cell' (line 62)
                sub_cell_1450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'sub_cell')
                int_1451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 28), 'int')
                # Applying the binary operator '|=' (line 62)
                result_ior_1452 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 16), '|=', sub_cell_1450, int_1451)
                # Assigning a type to the variable 'sub_cell' (line 62)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'sub_cell', result_ior_1452)
                
                # SSA join for if statement (line 61)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Name to a Name (line 63):
            
            # Assigning a Name to a Name (line 63):
            # Getting the type of 'start' (line 63)
            start_1453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 28), 'start')
            # Assigning a type to the variable 'cell_position' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'cell_position', start_1453)
            
            # Getting the type of 'True' (line 64)
            True_1454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 18), 'True')
            # Assigning a type to the variable 'True_1454' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'True_1454', True_1454)
            # Testing if the while is going to be iterated (line 64)
            # Testing the type of an if condition (line 64)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 12), True_1454)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 64, 12), True_1454):
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'sub_cell' (line 65)
                sub_cell_1455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 31), 'sub_cell')
                # Getting the type of 'self' (line 65)
                self_1456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'self')
                # Obtaining the member 'vector' of a type (line 65)
                vector_1457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), self_1456, 'vector')
                # Obtaining the member '__getitem__' of a type (line 65)
                getitem___1458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), vector_1457, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 65)
                subscript_call_result_1459 = invoke(stypy.reporting.localization.Localization(__file__, 65, 19), getitem___1458, sub_cell_1455)
                
                # Getting the type of 'None' (line 65)
                None_1460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 44), 'None')
                # Applying the binary operator '!=' (line 65)
                result_ne_1461 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 19), '!=', subscript_call_result_1459, None_1460)
                
                # Testing if the type of an if condition is none (line 65)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 65, 16), result_ne_1461):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 65)
                    if_condition_1462 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 16), result_ne_1461)
                    # Assigning a type to the variable 'if_condition_1462' (line 65)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'if_condition_1462', if_condition_1462)
                    # SSA begins for if statement (line 65)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Tuple (line 66):
                    
                    # Assigning a Call to a Name:
                    
                    # Call to get_intersection(...): (line 66)
                    # Processing the call arguments (line 66)
                    # Getting the type of 'ray_origin' (line 66)
                    ray_origin_1469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 86), 'ray_origin', False)
                    # Getting the type of 'ray_direction' (line 66)
                    ray_direction_1470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 98), 'ray_direction', False)
                    # Getting the type of 'last_hit' (line 66)
                    last_hit_1471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 113), 'last_hit', False)
                    # Getting the type of 'cell_position' (line 66)
                    cell_position_1472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 123), 'cell_position', False)
                    # Processing the call keyword arguments (line 66)
                    kwargs_1473 = {}
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'sub_cell' (line 66)
                    sub_cell_1463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 59), 'sub_cell', False)
                    # Getting the type of 'self' (line 66)
                    self_1464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 47), 'self', False)
                    # Obtaining the member 'vector' of a type (line 66)
                    vector_1465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 47), self_1464, 'vector')
                    # Obtaining the member '__getitem__' of a type (line 66)
                    getitem___1466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 47), vector_1465, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
                    subscript_call_result_1467 = invoke(stypy.reporting.localization.Localization(__file__, 66, 47), getitem___1466, sub_cell_1463)
                    
                    # Obtaining the member 'get_intersection' of a type (line 66)
                    get_intersection_1468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 47), subscript_call_result_1467, 'get_intersection')
                    # Calling get_intersection(args, kwargs) (line 66)
                    get_intersection_call_result_1474 = invoke(stypy.reporting.localization.Localization(__file__, 66, 47), get_intersection_1468, *[ray_origin_1469, ray_direction_1470, last_hit_1471, cell_position_1472], **kwargs_1473)
                    
                    # Assigning a type to the variable 'call_assignment_1048' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1048', get_intersection_call_result_1474)
                    
                    # Assigning a Call to a Name (line 66):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_1048' (line 66)
                    call_assignment_1048_1475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1048', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_1476 = stypy_get_value_from_tuple(call_assignment_1048_1475, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_1049' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1049', stypy_get_value_from_tuple_call_result_1476)
                    
                    # Assigning a Name to a Name (line 66):
                    # Getting the type of 'call_assignment_1049' (line 66)
                    call_assignment_1049_1477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1049')
                    # Assigning a type to the variable 'hit_object' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'hit_object', call_assignment_1049_1477)
                    
                    # Assigning a Call to a Name (line 66):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_1048' (line 66)
                    call_assignment_1048_1478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1048', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_1479 = stypy_get_value_from_tuple(call_assignment_1048_1478, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_1050' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1050', stypy_get_value_from_tuple_call_result_1479)
                    
                    # Assigning a Name to a Name (line 66):
                    # Getting the type of 'call_assignment_1050' (line 66)
                    call_assignment_1050_1480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'call_assignment_1050')
                    # Assigning a type to the variable 'hit_position' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'hit_position', call_assignment_1050_1480)
                    
                    # Getting the type of 'hit_object' (line 67)
                    hit_object_1481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'hit_object')
                    # Getting the type of 'None' (line 67)
                    None_1482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 37), 'None')
                    # Applying the binary operator '!=' (line 67)
                    result_ne_1483 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 23), '!=', hit_object_1481, None_1482)
                    
                    # Testing if the type of an if condition is none (line 67)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 20), result_ne_1483):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 67)
                        if_condition_1484 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 20), result_ne_1483)
                        # Assigning a type to the variable 'if_condition_1484' (line 67)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'if_condition_1484', if_condition_1484)
                        # SSA begins for if statement (line 67)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # SSA join for if statement (line 67)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 65)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Num to a Name (line 69):
                
                # Assigning a Num to a Name (line 69):
                float_1485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'float')
                # Assigning a type to the variable 'step' (line 69)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'step', float_1485)
                
                # Assigning a Num to a Name (line 70):
                
                # Assigning a Num to a Name (line 70):
                int_1486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'int')
                # Assigning a type to the variable 'axis' (line 70)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'axis', int_1486)
                
                
                # Call to range(...): (line 71)
                # Processing the call arguments (line 71)
                int_1488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 31), 'int')
                # Processing the call keyword arguments (line 71)
                kwargs_1489 = {}
                # Getting the type of 'range' (line 71)
                range_1487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 25), 'range', False)
                # Calling range(args, kwargs) (line 71)
                range_call_result_1490 = invoke(stypy.reporting.localization.Localization(__file__, 71, 25), range_1487, *[int_1488], **kwargs_1489)
                
                # Assigning a type to the variable 'range_call_result_1490' (line 71)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'range_call_result_1490', range_call_result_1490)
                # Testing if the for loop is going to be iterated (line 71)
                # Testing the type of a for loop iterable (line 71)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 71, 16), range_call_result_1490)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 71, 16), range_call_result_1490):
                    # Getting the type of the for loop variable (line 71)
                    for_loop_var_1491 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 71, 16), range_call_result_1490)
                    # Assigning a type to the variable 'i' (line 71)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'i', for_loop_var_1491)
                    # SSA begins for a for statement (line 71)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a BinOp to a Name (line 72):
                    
                    # Assigning a BinOp to a Name (line 72):
                    # Getting the type of 'sub_cell' (line 72)
                    sub_cell_1492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 28), 'sub_cell')
                    # Getting the type of 'i' (line 72)
                    i_1493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 40), 'i')
                    # Applying the binary operator '>>' (line 72)
                    result_rshift_1494 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 28), '>>', sub_cell_1492, i_1493)
                    
                    int_1495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 45), 'int')
                    # Applying the binary operator '&' (line 72)
                    result_and__1496 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 27), '&', result_rshift_1494, int_1495)
                    
                    # Assigning a type to the variable 'high' (line 72)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 20), 'high', result_and__1496)
                    
                    # Assigning a IfExp to a Name (line 73):
                    
                    # Assigning a IfExp to a Name (line 73):
                    
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 73)
                    i_1497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 70), 'i')
                    # Getting the type of 'ray_direction' (line 73)
                    ray_direction_1498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 56), 'ray_direction')
                    # Obtaining the member '__getitem__' of a type (line 73)
                    getitem___1499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 56), ray_direction_1498, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                    subscript_call_result_1500 = invoke(stypy.reporting.localization.Localization(__file__, 73, 56), getitem___1499, i_1497)
                    
                    float_1501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 75), 'float')
                    # Applying the binary operator '<' (line 73)
                    result_lt_1502 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 56), '<', subscript_call_result_1500, float_1501)
                    
                    
                    int_1503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 83), 'int')
                    # Getting the type of 'high' (line 73)
                    high_1504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 88), 'high')
                    # Applying the binary operator '!=' (line 73)
                    result_ne_1505 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 83), '!=', int_1503, high_1504)
                    
                    # Applying the binary operator '^' (line 73)
                    result_xor_1506 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 55), '^', result_lt_1502, result_ne_1505)
                    
                    # Testing the type of an if expression (line 73)
                    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 27), result_xor_1506)
                    # SSA begins for if expression (line 73)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 73)
                    i_1507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 38), 'i')
                    # Getting the type of 'high' (line 73)
                    high_1508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 42), 'high')
                    int_1509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 49), 'int')
                    # Applying the binary operator '*' (line 73)
                    result_mul_1510 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 42), '*', high_1508, int_1509)
                    
                    # Applying the binary operator '+' (line 73)
                    result_add_1511 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 38), '+', i_1507, result_mul_1510)
                    
                    # Getting the type of 'self' (line 73)
                    self_1512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 27), 'self')
                    # Obtaining the member 'bound' of a type (line 73)
                    bound_1513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 27), self_1512, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 73)
                    getitem___1514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 27), bound_1513, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                    subscript_call_result_1515 = invoke(stypy.reporting.localization.Localization(__file__, 73, 27), getitem___1514, result_add_1511)
                    
                    # SSA branch for the else part of an if expression (line 73)
                    module_type_store.open_ssa_branch('if expression else')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 73)
                    i_1516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 111), 'i')
                    # Getting the type of 'self' (line 73)
                    self_1517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 100), 'self')
                    # Obtaining the member 'bound' of a type (line 73)
                    bound_1518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 100), self_1517, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 73)
                    getitem___1519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 100), bound_1518, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                    subscript_call_result_1520 = invoke(stypy.reporting.localization.Localization(__file__, 73, 100), getitem___1519, i_1516)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 73)
                    i_1521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 127), 'i')
                    int_1522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 131), 'int')
                    # Applying the binary operator '+' (line 73)
                    result_add_1523 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 127), '+', i_1521, int_1522)
                    
                    # Getting the type of 'self' (line 73)
                    self_1524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 116), 'self')
                    # Obtaining the member 'bound' of a type (line 73)
                    bound_1525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 116), self_1524, 'bound')
                    # Obtaining the member '__getitem__' of a type (line 73)
                    getitem___1526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 116), bound_1525, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 73)
                    subscript_call_result_1527 = invoke(stypy.reporting.localization.Localization(__file__, 73, 116), getitem___1526, result_add_1523)
                    
                    # Applying the binary operator '+' (line 73)
                    result_add_1528 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 100), '+', subscript_call_result_1520, subscript_call_result_1527)
                    
                    float_1529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 137), 'float')
                    # Applying the binary operator '*' (line 73)
                    result_mul_1530 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 99), '*', result_add_1528, float_1529)
                    
                    # SSA join for if expression (line 73)
                    module_type_store = module_type_store.join_ssa_context()
                    if_exp_1531 = union_type.UnionType.add(subscript_call_result_1515, result_mul_1530)
                    
                    # Assigning a type to the variable 'face' (line 73)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'face', if_exp_1531)
                    
                    
                    # SSA begins for try-except statement (line 74)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                    
                    # Assigning a BinOp to a Name (line 75):
                    
                    # Assigning a BinOp to a Name (line 75):
                    # Getting the type of 'face' (line 75)
                    face_1532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 36), 'face')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 75)
                    i_1533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 54), 'i')
                    # Getting the type of 'ray_origin' (line 75)
                    ray_origin_1534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 43), 'ray_origin')
                    # Obtaining the member '__getitem__' of a type (line 75)
                    getitem___1535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 43), ray_origin_1534, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                    subscript_call_result_1536 = invoke(stypy.reporting.localization.Localization(__file__, 75, 43), getitem___1535, i_1533)
                    
                    # Applying the binary operator '-' (line 75)
                    result_sub_1537 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 36), '-', face_1532, subscript_call_result_1536)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'i' (line 75)
                    i_1538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 74), 'i')
                    # Getting the type of 'ray_direction' (line 75)
                    ray_direction_1539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 60), 'ray_direction')
                    # Obtaining the member '__getitem__' of a type (line 75)
                    getitem___1540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 60), ray_direction_1539, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                    subscript_call_result_1541 = invoke(stypy.reporting.localization.Localization(__file__, 75, 60), getitem___1540, i_1538)
                    
                    # Applying the binary operator 'div' (line 75)
                    result_div_1542 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 35), 'div', result_sub_1537, subscript_call_result_1541)
                    
                    # Assigning a type to the variable 'distance' (line 75)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'distance', result_div_1542)
                    # SSA branch for the except part of a try statement (line 74)
                    # SSA branch for the except '<any exception>' branch of a try statement (line 74)
                    module_type_store.open_ssa_branch('except')
                    
                    # Assigning a Call to a Name (line 77):
                    
                    # Assigning a Call to a Name (line 77):
                    
                    # Call to float(...): (line 77)
                    # Processing the call arguments (line 77)
                    float_1544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 41), 'float')
                    # Processing the call keyword arguments (line 77)
                    kwargs_1545 = {}
                    # Getting the type of 'float' (line 77)
                    float_1543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 35), 'float', False)
                    # Calling float(args, kwargs) (line 77)
                    float_call_result_1546 = invoke(stypy.reporting.localization.Localization(__file__, 77, 35), float_1543, *[float_1544], **kwargs_1545)
                    
                    # Assigning a type to the variable 'distance' (line 77)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'distance', float_call_result_1546)
                    # SSA join for try-except statement (line 74)
                    module_type_store = module_type_store.join_ssa_context()
                    
                    
                    # Getting the type of 'distance' (line 78)
                    distance_1547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'distance')
                    # Getting the type of 'step' (line 78)
                    step_1548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 35), 'step')
                    # Applying the binary operator '<=' (line 78)
                    result_le_1549 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 23), '<=', distance_1547, step_1548)
                    
                    # Testing if the type of an if condition is none (line 78)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 78, 20), result_le_1549):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 78)
                        if_condition_1550 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 20), result_le_1549)
                        # Assigning a type to the variable 'if_condition_1550' (line 78)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'if_condition_1550', if_condition_1550)
                        # SSA begins for if statement (line 78)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Name (line 79):
                        
                        # Assigning a Name to a Name (line 79):
                        # Getting the type of 'distance' (line 79)
                        distance_1551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 31), 'distance')
                        # Assigning a type to the variable 'step' (line 79)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'step', distance_1551)
                        
                        # Assigning a Name to a Name (line 80):
                        
                        # Assigning a Name to a Name (line 80):
                        # Getting the type of 'i' (line 80)
                        i_1552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'i')
                        # Assigning a type to the variable 'axis' (line 80)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'axis', i_1552)
                        # SSA join for if statement (line 78)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Getting the type of 'sub_cell' (line 81)
                sub_cell_1553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 22), 'sub_cell')
                # Getting the type of 'axis' (line 81)
                axis_1554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 34), 'axis')
                # Applying the binary operator '>>' (line 81)
                result_rshift_1555 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 22), '>>', sub_cell_1553, axis_1554)
                
                int_1556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 42), 'int')
                # Applying the binary operator '&' (line 81)
                result_and__1557 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 21), '&', result_rshift_1555, int_1556)
                
                int_1558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 48), 'int')
                # Applying the binary operator '==' (line 81)
                result_eq_1559 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 20), '==', result_and__1557, int_1558)
                
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'axis' (line 81)
                axis_1560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 68), 'axis')
                # Getting the type of 'ray_direction' (line 81)
                ray_direction_1561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 54), 'ray_direction')
                # Obtaining the member '__getitem__' of a type (line 81)
                getitem___1562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 54), ray_direction_1561, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 81)
                subscript_call_result_1563 = invoke(stypy.reporting.localization.Localization(__file__, 81, 54), getitem___1562, axis_1560)
                
                float_1564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 76), 'float')
                # Applying the binary operator '<' (line 81)
                result_lt_1565 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 54), '<', subscript_call_result_1563, float_1564)
                
                # Applying the binary operator '^' (line 81)
                result_xor_1566 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 19), '^', result_eq_1559, result_lt_1565)
                
                # Testing if the type of an if condition is none (line 81)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 81, 16), result_xor_1566):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 81)
                    if_condition_1567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 16), result_xor_1566)
                    # Assigning a type to the variable 'if_condition_1567' (line 81)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'if_condition_1567', if_condition_1567)
                    # SSA begins for if statement (line 81)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # SSA join for if statement (line 81)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a BinOp to a Name (line 83):
                
                # Assigning a BinOp to a Name (line 83):
                # Getting the type of 'ray_origin' (line 83)
                ray_origin_1568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 32), 'ray_origin')
                # Getting the type of 'ray_direction' (line 83)
                ray_direction_1569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 45), 'ray_direction')
                # Getting the type of 'step' (line 83)
                step_1570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 61), 'step')
                # Applying the binary operator '*' (line 83)
                result_mul_1571 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 45), '*', ray_direction_1569, step_1570)
                
                # Applying the binary operator '+' (line 83)
                result_add_1572 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 32), '+', ray_origin_1568, result_mul_1571)
                
                # Assigning a type to the variable 'cell_position' (line 83)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'cell_position', result_add_1572)
                
                # Assigning a BinOp to a Name (line 84):
                
                # Assigning a BinOp to a Name (line 84):
                # Getting the type of 'sub_cell' (line 84)
                sub_cell_1573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'sub_cell')
                int_1574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 39), 'int')
                # Getting the type of 'axis' (line 84)
                axis_1575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 44), 'axis')
                # Applying the binary operator '<<' (line 84)
                result_lshift_1576 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 39), '<<', int_1574, axis_1575)
                
                # Applying the binary operator '^' (line 84)
                result_xor_1577 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 27), '^', sub_cell_1573, result_lshift_1576)
                
                # Assigning a type to the variable 'sub_cell' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'sub_cell', result_xor_1577)

            
            # SSA branch for the else part of an if statement (line 57)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Num to a Name (line 86):
            
            # Assigning a Num to a Name (line 86):
            float_1578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 31), 'float')
            # Assigning a type to the variable 'nearest_distance' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'nearest_distance', float_1578)
            
            # Getting the type of 'self' (line 87)
            self_1579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'self')
            # Obtaining the member 'items' of a type (line 87)
            items_1580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 24), self_1579, 'items')
            # Assigning a type to the variable 'items_1580' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'items_1580', items_1580)
            # Testing if the for loop is going to be iterated (line 87)
            # Testing the type of a for loop iterable (line 87)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 12), items_1580)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 87, 12), items_1580):
                # Getting the type of the for loop variable (line 87)
                for_loop_var_1581 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 12), items_1580)
                # Assigning a type to the variable 'item' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'item', for_loop_var_1581)
                # SSA begins for a for statement (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'item' (line 88)
                item_1582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'item')
                # Getting the type of 'last_hit' (line 88)
                last_hit_1583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 27), 'last_hit')
                # Applying the binary operator '!=' (line 88)
                result_ne_1584 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 19), '!=', item_1582, last_hit_1583)
                
                # Testing if the type of an if condition is none (line 88)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 88, 16), result_ne_1584):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 88)
                    if_condition_1585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 16), result_ne_1584)
                    # Assigning a type to the variable 'if_condition_1585' (line 88)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'if_condition_1585', if_condition_1585)
                    # SSA begins for if statement (line 88)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Assigning a Call to a Name (line 89):
                    
                    # Call to get_intersection(...): (line 89)
                    # Processing the call arguments (line 89)
                    # Getting the type of 'ray_origin' (line 89)
                    ray_origin_1588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 53), 'ray_origin', False)
                    # Getting the type of 'ray_direction' (line 89)
                    ray_direction_1589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 65), 'ray_direction', False)
                    # Processing the call keyword arguments (line 89)
                    kwargs_1590 = {}
                    # Getting the type of 'item' (line 89)
                    item_1586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'item', False)
                    # Obtaining the member 'get_intersection' of a type (line 89)
                    get_intersection_1587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 31), item_1586, 'get_intersection')
                    # Calling get_intersection(args, kwargs) (line 89)
                    get_intersection_call_result_1591 = invoke(stypy.reporting.localization.Localization(__file__, 89, 31), get_intersection_1587, *[ray_origin_1588, ray_direction_1589], **kwargs_1590)
                    
                    # Assigning a type to the variable 'distance' (line 89)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 20), 'distance', get_intersection_call_result_1591)
                    
                    float_1592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'float')
                    # Getting the type of 'distance' (line 90)
                    distance_1593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 30), 'distance')
                    # Applying the binary operator '<=' (line 90)
                    result_le_1594 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '<=', float_1592, distance_1593)
                    # Getting the type of 'nearest_distance' (line 90)
                    nearest_distance_1595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 41), 'nearest_distance')
                    # Applying the binary operator '<' (line 90)
                    result_lt_1596 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '<', distance_1593, nearest_distance_1595)
                    # Applying the binary operator '&' (line 90)
                    result_and__1597 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 23), '&', result_le_1594, result_lt_1596)
                    
                    # Testing if the type of an if condition is none (line 90)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 90, 20), result_and__1597):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 90)
                        if_condition_1598 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 20), result_and__1597)
                        # Assigning a type to the variable 'if_condition_1598' (line 90)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'if_condition_1598', if_condition_1598)
                        # SSA begins for if statement (line 90)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 91):
                        
                        # Assigning a BinOp to a Name (line 91):
                        # Getting the type of 'ray_origin' (line 91)
                        ray_origin_1599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 30), 'ray_origin')
                        # Getting the type of 'ray_direction' (line 91)
                        ray_direction_1600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'ray_direction')
                        # Getting the type of 'distance' (line 91)
                        distance_1601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 59), 'distance')
                        # Applying the binary operator '*' (line 91)
                        result_mul_1602 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 43), '*', ray_direction_1600, distance_1601)
                        
                        # Applying the binary operator '+' (line 91)
                        result_add_1603 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 30), '+', ray_origin_1599, result_mul_1602)
                        
                        # Assigning a type to the variable 'hit' (line 91)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'hit', result_add_1603)
                        
                        # Evaluating a boolean operation
                        
                        # Getting the type of 'b0' (line 92)
                        b0_1604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'b0')
                        # Getting the type of 'hit' (line 92)
                        hit_1605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 33), 'hit')
                        # Obtaining the member 'x' of a type (line 92)
                        x_1606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 33), hit_1605, 'x')
                        # Applying the binary operator '-' (line 92)
                        result_sub_1607 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 28), '-', b0_1604, x_1606)
                        
                        # Getting the type of 'TOLERANCE' (line 92)
                        TOLERANCE_1608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 92)
                        result_le_1609 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 28), '<=', result_sub_1607, TOLERANCE_1608)
                        
                        
                        # Getting the type of 'hit' (line 93)
                        hit_1610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'hit')
                        # Obtaining the member 'x' of a type (line 93)
                        x_1611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 28), hit_1610, 'x')
                        # Getting the type of 'b3' (line 93)
                        b3_1612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 36), 'b3')
                        # Applying the binary operator '-' (line 93)
                        result_sub_1613 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 28), '-', x_1611, b3_1612)
                        
                        # Getting the type of 'TOLERANCE' (line 93)
                        TOLERANCE_1614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 93)
                        result_le_1615 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 28), '<=', result_sub_1613, TOLERANCE_1614)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1616 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_le_1609, result_le_1615)
                        
                        # Getting the type of 'b1' (line 94)
                        b1_1617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'b1')
                        # Getting the type of 'hit' (line 94)
                        hit_1618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'hit')
                        # Obtaining the member 'y' of a type (line 94)
                        y_1619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 33), hit_1618, 'y')
                        # Applying the binary operator '-' (line 94)
                        result_sub_1620 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 28), '-', b1_1617, y_1619)
                        
                        # Getting the type of 'TOLERANCE' (line 94)
                        TOLERANCE_1621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 94)
                        result_le_1622 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 28), '<=', result_sub_1620, TOLERANCE_1621)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1623 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1616, result_le_1622)
                        
                        # Getting the type of 'hit' (line 95)
                        hit_1624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'hit')
                        # Obtaining the member 'y' of a type (line 95)
                        y_1625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 28), hit_1624, 'y')
                        # Getting the type of 'b4' (line 95)
                        b4_1626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'b4')
                        # Applying the binary operator '-' (line 95)
                        result_sub_1627 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 28), '-', y_1625, b4_1626)
                        
                        # Getting the type of 'TOLERANCE' (line 95)
                        TOLERANCE_1628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 95)
                        result_le_1629 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 28), '<=', result_sub_1627, TOLERANCE_1628)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1630 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1623, result_le_1629)
                        
                        # Getting the type of 'b2' (line 96)
                        b2_1631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 28), 'b2')
                        # Getting the type of 'hit' (line 96)
                        hit_1632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'hit')
                        # Obtaining the member 'z' of a type (line 96)
                        z_1633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 33), hit_1632, 'z')
                        # Applying the binary operator '-' (line 96)
                        result_sub_1634 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 28), '-', b2_1631, z_1633)
                        
                        # Getting the type of 'TOLERANCE' (line 96)
                        TOLERANCE_1635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 96)
                        result_le_1636 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 28), '<=', result_sub_1634, TOLERANCE_1635)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1637 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1630, result_le_1636)
                        
                        # Getting the type of 'hit' (line 97)
                        hit_1638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'hit')
                        # Obtaining the member 'z' of a type (line 97)
                        z_1639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 28), hit_1638, 'z')
                        # Getting the type of 'b5' (line 97)
                        b5_1640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 36), 'b5')
                        # Applying the binary operator '-' (line 97)
                        result_sub_1641 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 28), '-', z_1639, b5_1640)
                        
                        # Getting the type of 'TOLERANCE' (line 97)
                        TOLERANCE_1642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 42), 'TOLERANCE')
                        # Applying the binary operator '<=' (line 97)
                        result_le_1643 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 28), '<=', result_sub_1641, TOLERANCE_1642)
                        
                        # Applying the binary operator 'and' (line 92)
                        result_and_keyword_1644 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 27), 'and', result_and_keyword_1637, result_le_1643)
                        
                        # Testing if the type of an if condition is none (line 92)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 92, 24), result_and_keyword_1644):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 92)
                            if_condition_1645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 24), result_and_keyword_1644)
                            # Assigning a type to the variable 'if_condition_1645' (line 92)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'if_condition_1645', if_condition_1645)
                            # SSA begins for if statement (line 92)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Name (line 98):
                            
                            # Assigning a Name to a Name (line 98):
                            # Getting the type of 'item' (line 98)
                            item_1646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 44), 'item')
                            # Assigning a type to the variable 'hit_object' (line 98)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'hit_object', item_1646)
                            
                            # Assigning a Name to a Name (line 99):
                            
                            # Assigning a Name to a Name (line 99):
                            # Getting the type of 'hit' (line 99)
                            hit_1647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 46), 'hit')
                            # Assigning a type to the variable 'hit_position' (line 99)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 31), 'hit_position', hit_1647)
                            
                            # Assigning a Name to a Name (line 100):
                            
                            # Assigning a Name to a Name (line 100):
                            # Getting the type of 'distance' (line 100)
                            distance_1648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 50), 'distance')
                            # Assigning a type to the variable 'nearest_distance' (line 100)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'nearest_distance', distance_1648)
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
        tuple_1649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 101)
        # Adding element type (line 101)
        # Getting the type of 'hit_object' (line 101)
        hit_object_1650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'hit_object')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 15), tuple_1649, hit_object_1650)
        # Adding element type (line 101)
        # Getting the type of 'hit_position' (line 101)
        hit_position_1651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'hit_position')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 15), tuple_1649, hit_position_1651)
        
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', tuple_1649)
        
        # ################# End of 'get_intersection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_intersection' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_1652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1652)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_intersection'
        return stypy_return_type_1652


# Assigning a type to the variable 'SpatialIndex' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'SpatialIndex', SpatialIndex)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
