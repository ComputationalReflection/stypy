
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #  MiniLight Python : minimal global illumination renderer
2: #
3: #  Copyright (c) 2007-2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
4: #  http://www.hxa7241.org/
5: 
6: 
7: from math import pi, tan
8: from random import random
9: from raytracer import RayTracer
10: from vector3f import Vector3f, Vector3f_str
11: 
12: import re
13: SEARCH = re.compile('(\(.+\))\s*(\(.+\))\s*(\S+)')
14: 
15: class Camera(object):
16: 
17:     def __init__(self, in_stream):
18:         for line in in_stream:
19:             if not line.isspace():
20:                 p, d, a = SEARCH.search(line).groups()
21:                 self.view_position = Vector3f_str(p)
22:                 self.view_direction = Vector3f_str(d).unitize()
23:                 if self.view_direction.is_zero():
24:                     self.view_direction = Vector3f(0.0, 0.0, 1.0)
25:                 self.view_angle = min(max(10.0, float(a)), 160.0) * (pi / 180.0)
26:                 self.right = Vector3f(0.0, 1.0, 0.0).cross(self.view_direction).unitize()
27:                 if self.right.is_zero():
28:                     self.up = Vector3f(0.0, 0.0, 1.0 if self.view_direction.y else -1.0)
29:                     self.right = self.up.cross(self.view_direction).unitize()
30:                 else:
31:                     self.up = self.view_direction.cross(self.right).unitize()
32:                 break
33: 
34:     def get_frame(self, scene, image):
35:         raytracer = RayTracer(scene)
36:         aspect = float(image.height) / float(image.width)
37:         for y in range(image.height):
38:             for x in range(image.width):
39:                 x_coefficient = ((x + random()) * 2.0 / image.width) - 1.0
40:                 y_coefficient = ((y + random()) * 2.0 / image.height) - 1.0
41:                 offset = self.right * x_coefficient + self.up * (y_coefficient * aspect)
42:                 sample_direction = (self.view_direction + (offset * tan(self.view_angle * 0.5))).unitize()
43:                 radiance = raytracer.get_radiance(self.view_position, sample_direction)
44:                 image.add_to_pixel(x, y, radiance)
45: 
46: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from math import pi, tan' statement (line 7)
try:
    from math import pi, tan

except:
    pi = UndefinedType
    tan = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'math', None, module_type_store, ['pi', 'tan'], [pi, tan])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from random import random' statement (line 8)
try:
    from random import random

except:
    random = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'random', None, module_type_store, ['random'], [random])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from raytracer import RayTracer' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_47 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'raytracer')

if (type(import_47) is not StypyTypeError):

    if (import_47 != 'pyd_module'):
        __import__(import_47)
        sys_modules_48 = sys.modules[import_47]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'raytracer', sys_modules_48.module_type_store, module_type_store, ['RayTracer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_48, sys_modules_48.module_type_store, module_type_store)
    else:
        from raytracer import RayTracer

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'raytracer', None, module_type_store, ['RayTracer'], [RayTracer])

else:
    # Assigning a type to the variable 'raytracer' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'raytracer', import_47)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from vector3f import Vector3f, Vector3f_str' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_49 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'vector3f')

if (type(import_49) is not StypyTypeError):

    if (import_49 != 'pyd_module'):
        __import__(import_49)
        sys_modules_50 = sys.modules[import_49]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'vector3f', sys_modules_50.module_type_store, module_type_store, ['Vector3f', 'Vector3f_str'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_50, sys_modules_50.module_type_store, module_type_store)
    else:
        from vector3f import Vector3f, Vector3f_str

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'vector3f', None, module_type_store, ['Vector3f', 'Vector3f_str'], [Vector3f, Vector3f_str])

else:
    # Assigning a type to the variable 'vector3f' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'vector3f', import_49)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import re' statement (line 12)
import re

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 're', re, module_type_store)


# Assigning a Call to a Name (line 13):

# Assigning a Call to a Name (line 13):

# Call to compile(...): (line 13)
# Processing the call arguments (line 13)
str_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'str', '(\\(.+\\))\\s*(\\(.+\\))\\s*(\\S+)')
# Processing the call keyword arguments (line 13)
kwargs_54 = {}
# Getting the type of 're' (line 13)
re_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 9), 're', False)
# Obtaining the member 'compile' of a type (line 13)
compile_52 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 9), re_51, 'compile')
# Calling compile(args, kwargs) (line 13)
compile_call_result_55 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), compile_52, *[str_53], **kwargs_54)

# Assigning a type to the variable 'SEARCH' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'SEARCH', compile_call_result_55)
# Declaration of the 'Camera' class

class Camera(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Camera.__init__', ['in_stream'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['in_stream'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Getting the type of 'in_stream' (line 18)
        in_stream_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'in_stream')
        # Assigning a type to the variable 'in_stream_56' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'in_stream_56', in_stream_56)
        # Testing if the for loop is going to be iterated (line 18)
        # Testing the type of a for loop iterable (line 18)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 18, 8), in_stream_56)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 18, 8), in_stream_56):
            # Getting the type of the for loop variable (line 18)
            for_loop_var_57 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 18, 8), in_stream_56)
            # Assigning a type to the variable 'line' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'line', for_loop_var_57)
            # SSA begins for a for statement (line 18)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to isspace(...): (line 19)
            # Processing the call keyword arguments (line 19)
            kwargs_60 = {}
            # Getting the type of 'line' (line 19)
            line_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'line', False)
            # Obtaining the member 'isspace' of a type (line 19)
            isspace_59 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), line_58, 'isspace')
            # Calling isspace(args, kwargs) (line 19)
            isspace_call_result_61 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), isspace_59, *[], **kwargs_60)
            
            # Applying the 'not' unary operator (line 19)
            result_not__62 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 15), 'not', isspace_call_result_61)
            
            # Testing if the type of an if condition is none (line 19)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 19, 12), result_not__62):
                pass
            else:
                
                # Testing the type of an if condition (line 19)
                if_condition_63 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 12), result_not__62)
                # Assigning a type to the variable 'if_condition_63' (line 19)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'if_condition_63', if_condition_63)
                # SSA begins for if statement (line 19)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 20):
                
                # Assigning a Call to a Name:
                
                # Call to groups(...): (line 20)
                # Processing the call keyword arguments (line 20)
                kwargs_70 = {}
                
                # Call to search(...): (line 20)
                # Processing the call arguments (line 20)
                # Getting the type of 'line' (line 20)
                line_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 40), 'line', False)
                # Processing the call keyword arguments (line 20)
                kwargs_67 = {}
                # Getting the type of 'SEARCH' (line 20)
                SEARCH_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 26), 'SEARCH', False)
                # Obtaining the member 'search' of a type (line 20)
                search_65 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), SEARCH_64, 'search')
                # Calling search(args, kwargs) (line 20)
                search_call_result_68 = invoke(stypy.reporting.localization.Localization(__file__, 20, 26), search_65, *[line_66], **kwargs_67)
                
                # Obtaining the member 'groups' of a type (line 20)
                groups_69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), search_call_result_68, 'groups')
                # Calling groups(args, kwargs) (line 20)
                groups_call_result_71 = invoke(stypy.reporting.localization.Localization(__file__, 20, 26), groups_69, *[], **kwargs_70)
                
                # Assigning a type to the variable 'call_assignment_43' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_43', groups_call_result_71)
                
                # Assigning a Call to a Name (line 20):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_43' (line 20)
                call_assignment_43_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_43', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_73 = stypy_get_value_from_tuple(call_assignment_43_72, 3, 0)
                
                # Assigning a type to the variable 'call_assignment_44' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_44', stypy_get_value_from_tuple_call_result_73)
                
                # Assigning a Name to a Name (line 20):
                # Getting the type of 'call_assignment_44' (line 20)
                call_assignment_44_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_44')
                # Assigning a type to the variable 'p' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'p', call_assignment_44_74)
                
                # Assigning a Call to a Name (line 20):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_43' (line 20)
                call_assignment_43_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_43', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_76 = stypy_get_value_from_tuple(call_assignment_43_75, 3, 1)
                
                # Assigning a type to the variable 'call_assignment_45' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_45', stypy_get_value_from_tuple_call_result_76)
                
                # Assigning a Name to a Name (line 20):
                # Getting the type of 'call_assignment_45' (line 20)
                call_assignment_45_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_45')
                # Assigning a type to the variable 'd' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'd', call_assignment_45_77)
                
                # Assigning a Call to a Name (line 20):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_43' (line 20)
                call_assignment_43_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_43', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_79 = stypy_get_value_from_tuple(call_assignment_43_78, 3, 2)
                
                # Assigning a type to the variable 'call_assignment_46' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_46', stypy_get_value_from_tuple_call_result_79)
                
                # Assigning a Name to a Name (line 20):
                # Getting the type of 'call_assignment_46' (line 20)
                call_assignment_46_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_46')
                # Assigning a type to the variable 'a' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'a', call_assignment_46_80)
                
                # Assigning a Call to a Attribute (line 21):
                
                # Assigning a Call to a Attribute (line 21):
                
                # Call to Vector3f_str(...): (line 21)
                # Processing the call arguments (line 21)
                # Getting the type of 'p' (line 21)
                p_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 50), 'p', False)
                # Processing the call keyword arguments (line 21)
                kwargs_83 = {}
                # Getting the type of 'Vector3f_str' (line 21)
                Vector3f_str_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 37), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 21)
                Vector3f_str_call_result_84 = invoke(stypy.reporting.localization.Localization(__file__, 21, 37), Vector3f_str_81, *[p_82], **kwargs_83)
                
                # Getting the type of 'self' (line 21)
                self_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'self')
                # Setting the type of the member 'view_position' of a type (line 21)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), self_85, 'view_position', Vector3f_str_call_result_84)
                
                # Assigning a Call to a Attribute (line 22):
                
                # Assigning a Call to a Attribute (line 22):
                
                # Call to unitize(...): (line 22)
                # Processing the call keyword arguments (line 22)
                kwargs_91 = {}
                
                # Call to Vector3f_str(...): (line 22)
                # Processing the call arguments (line 22)
                # Getting the type of 'd' (line 22)
                d_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 51), 'd', False)
                # Processing the call keyword arguments (line 22)
                kwargs_88 = {}
                # Getting the type of 'Vector3f_str' (line 22)
                Vector3f_str_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 38), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 22)
                Vector3f_str_call_result_89 = invoke(stypy.reporting.localization.Localization(__file__, 22, 38), Vector3f_str_86, *[d_87], **kwargs_88)
                
                # Obtaining the member 'unitize' of a type (line 22)
                unitize_90 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 38), Vector3f_str_call_result_89, 'unitize')
                # Calling unitize(args, kwargs) (line 22)
                unitize_call_result_92 = invoke(stypy.reporting.localization.Localization(__file__, 22, 38), unitize_90, *[], **kwargs_91)
                
                # Getting the type of 'self' (line 22)
                self_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'self')
                # Setting the type of the member 'view_direction' of a type (line 22)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), self_93, 'view_direction', unitize_call_result_92)
                
                # Call to is_zero(...): (line 23)
                # Processing the call keyword arguments (line 23)
                kwargs_97 = {}
                # Getting the type of 'self' (line 23)
                self_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'self', False)
                # Obtaining the member 'view_direction' of a type (line 23)
                view_direction_95 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 19), self_94, 'view_direction')
                # Obtaining the member 'is_zero' of a type (line 23)
                is_zero_96 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 19), view_direction_95, 'is_zero')
                # Calling is_zero(args, kwargs) (line 23)
                is_zero_call_result_98 = invoke(stypy.reporting.localization.Localization(__file__, 23, 19), is_zero_96, *[], **kwargs_97)
                
                # Testing if the type of an if condition is none (line 23)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 23, 16), is_zero_call_result_98):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 23)
                    if_condition_99 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 16), is_zero_call_result_98)
                    # Assigning a type to the variable 'if_condition_99' (line 23)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'if_condition_99', if_condition_99)
                    # SSA begins for if statement (line 23)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Attribute (line 24):
                    
                    # Assigning a Call to a Attribute (line 24):
                    
                    # Call to Vector3f(...): (line 24)
                    # Processing the call arguments (line 24)
                    float_101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 51), 'float')
                    float_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 56), 'float')
                    float_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 61), 'float')
                    # Processing the call keyword arguments (line 24)
                    kwargs_104 = {}
                    # Getting the type of 'Vector3f' (line 24)
                    Vector3f_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 42), 'Vector3f', False)
                    # Calling Vector3f(args, kwargs) (line 24)
                    Vector3f_call_result_105 = invoke(stypy.reporting.localization.Localization(__file__, 24, 42), Vector3f_100, *[float_101, float_102, float_103], **kwargs_104)
                    
                    # Getting the type of 'self' (line 24)
                    self_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'self')
                    # Setting the type of the member 'view_direction' of a type (line 24)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 20), self_106, 'view_direction', Vector3f_call_result_105)
                    # SSA join for if statement (line 23)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a BinOp to a Attribute (line 25):
                
                # Assigning a BinOp to a Attribute (line 25):
                
                # Call to min(...): (line 25)
                # Processing the call arguments (line 25)
                
                # Call to max(...): (line 25)
                # Processing the call arguments (line 25)
                float_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 42), 'float')
                
                # Call to float(...): (line 25)
                # Processing the call arguments (line 25)
                # Getting the type of 'a' (line 25)
                a_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 54), 'a', False)
                # Processing the call keyword arguments (line 25)
                kwargs_112 = {}
                # Getting the type of 'float' (line 25)
                float_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 48), 'float', False)
                # Calling float(args, kwargs) (line 25)
                float_call_result_113 = invoke(stypy.reporting.localization.Localization(__file__, 25, 48), float_110, *[a_111], **kwargs_112)
                
                # Processing the call keyword arguments (line 25)
                kwargs_114 = {}
                # Getting the type of 'max' (line 25)
                max_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 38), 'max', False)
                # Calling max(args, kwargs) (line 25)
                max_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 25, 38), max_108, *[float_109, float_call_result_113], **kwargs_114)
                
                float_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 59), 'float')
                # Processing the call keyword arguments (line 25)
                kwargs_117 = {}
                # Getting the type of 'min' (line 25)
                min_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 34), 'min', False)
                # Calling min(args, kwargs) (line 25)
                min_call_result_118 = invoke(stypy.reporting.localization.Localization(__file__, 25, 34), min_107, *[max_call_result_115, float_116], **kwargs_117)
                
                # Getting the type of 'pi' (line 25)
                pi_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 69), 'pi')
                float_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 74), 'float')
                # Applying the binary operator 'div' (line 25)
                result_div_121 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 69), 'div', pi_119, float_120)
                
                # Applying the binary operator '*' (line 25)
                result_mul_122 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 34), '*', min_call_result_118, result_div_121)
                
                # Getting the type of 'self' (line 25)
                self_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'self')
                # Setting the type of the member 'view_angle' of a type (line 25)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), self_123, 'view_angle', result_mul_122)
                
                # Assigning a Call to a Attribute (line 26):
                
                # Assigning a Call to a Attribute (line 26):
                
                # Call to unitize(...): (line 26)
                # Processing the call keyword arguments (line 26)
                kwargs_136 = {}
                
                # Call to cross(...): (line 26)
                # Processing the call arguments (line 26)
                # Getting the type of 'self' (line 26)
                self_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 59), 'self', False)
                # Obtaining the member 'view_direction' of a type (line 26)
                view_direction_132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 59), self_131, 'view_direction')
                # Processing the call keyword arguments (line 26)
                kwargs_133 = {}
                
                # Call to Vector3f(...): (line 26)
                # Processing the call arguments (line 26)
                float_125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 38), 'float')
                float_126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 43), 'float')
                float_127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 48), 'float')
                # Processing the call keyword arguments (line 26)
                kwargs_128 = {}
                # Getting the type of 'Vector3f' (line 26)
                Vector3f_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'Vector3f', False)
                # Calling Vector3f(args, kwargs) (line 26)
                Vector3f_call_result_129 = invoke(stypy.reporting.localization.Localization(__file__, 26, 29), Vector3f_124, *[float_125, float_126, float_127], **kwargs_128)
                
                # Obtaining the member 'cross' of a type (line 26)
                cross_130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 29), Vector3f_call_result_129, 'cross')
                # Calling cross(args, kwargs) (line 26)
                cross_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 26, 29), cross_130, *[view_direction_132], **kwargs_133)
                
                # Obtaining the member 'unitize' of a type (line 26)
                unitize_135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 29), cross_call_result_134, 'unitize')
                # Calling unitize(args, kwargs) (line 26)
                unitize_call_result_137 = invoke(stypy.reporting.localization.Localization(__file__, 26, 29), unitize_135, *[], **kwargs_136)
                
                # Getting the type of 'self' (line 26)
                self_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'self')
                # Setting the type of the member 'right' of a type (line 26)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), self_138, 'right', unitize_call_result_137)
                
                # Call to is_zero(...): (line 27)
                # Processing the call keyword arguments (line 27)
                kwargs_142 = {}
                # Getting the type of 'self' (line 27)
                self_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'self', False)
                # Obtaining the member 'right' of a type (line 27)
                right_140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 19), self_139, 'right')
                # Obtaining the member 'is_zero' of a type (line 27)
                is_zero_141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 19), right_140, 'is_zero')
                # Calling is_zero(args, kwargs) (line 27)
                is_zero_call_result_143 = invoke(stypy.reporting.localization.Localization(__file__, 27, 19), is_zero_141, *[], **kwargs_142)
                
                # Testing if the type of an if condition is none (line 27)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 27, 16), is_zero_call_result_143):
                    
                    # Assigning a Call to a Attribute (line 31):
                    
                    # Assigning a Call to a Attribute (line 31):
                    
                    # Call to unitize(...): (line 31)
                    # Processing the call keyword arguments (line 31)
                    kwargs_176 = {}
                    
                    # Call to cross(...): (line 31)
                    # Processing the call arguments (line 31)
                    # Getting the type of 'self' (line 31)
                    self_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 56), 'self', False)
                    # Obtaining the member 'right' of a type (line 31)
                    right_172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 56), self_171, 'right')
                    # Processing the call keyword arguments (line 31)
                    kwargs_173 = {}
                    # Getting the type of 'self' (line 31)
                    self_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'self', False)
                    # Obtaining the member 'view_direction' of a type (line 31)
                    view_direction_169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), self_168, 'view_direction')
                    # Obtaining the member 'cross' of a type (line 31)
                    cross_170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), view_direction_169, 'cross')
                    # Calling cross(args, kwargs) (line 31)
                    cross_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), cross_170, *[right_172], **kwargs_173)
                    
                    # Obtaining the member 'unitize' of a type (line 31)
                    unitize_175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), cross_call_result_174, 'unitize')
                    # Calling unitize(args, kwargs) (line 31)
                    unitize_call_result_177 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), unitize_175, *[], **kwargs_176)
                    
                    # Getting the type of 'self' (line 31)
                    self_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'self')
                    # Setting the type of the member 'up' of a type (line 31)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), self_178, 'up', unitize_call_result_177)
                else:
                    
                    # Testing the type of an if condition (line 27)
                    if_condition_144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 16), is_zero_call_result_143)
                    # Assigning a type to the variable 'if_condition_144' (line 27)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'if_condition_144', if_condition_144)
                    # SSA begins for if statement (line 27)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Attribute (line 28):
                    
                    # Assigning a Call to a Attribute (line 28):
                    
                    # Call to Vector3f(...): (line 28)
                    # Processing the call arguments (line 28)
                    float_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 39), 'float')
                    float_147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 44), 'float')
                    
                    # Getting the type of 'self' (line 28)
                    self_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 56), 'self', False)
                    # Obtaining the member 'view_direction' of a type (line 28)
                    view_direction_149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 56), self_148, 'view_direction')
                    # Obtaining the member 'y' of a type (line 28)
                    y_150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 56), view_direction_149, 'y')
                    # Testing the type of an if expression (line 28)
                    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 49), y_150)
                    # SSA begins for if expression (line 28)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                    float_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 49), 'float')
                    # SSA branch for the else part of an if expression (line 28)
                    module_type_store.open_ssa_branch('if expression else')
                    float_152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 83), 'float')
                    # SSA join for if expression (line 28)
                    module_type_store = module_type_store.join_ssa_context()
                    if_exp_153 = union_type.UnionType.add(float_151, float_152)
                    
                    # Processing the call keyword arguments (line 28)
                    kwargs_154 = {}
                    # Getting the type of 'Vector3f' (line 28)
                    Vector3f_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'Vector3f', False)
                    # Calling Vector3f(args, kwargs) (line 28)
                    Vector3f_call_result_155 = invoke(stypy.reporting.localization.Localization(__file__, 28, 30), Vector3f_145, *[float_146, float_147, if_exp_153], **kwargs_154)
                    
                    # Getting the type of 'self' (line 28)
                    self_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'self')
                    # Setting the type of the member 'up' of a type (line 28)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 20), self_156, 'up', Vector3f_call_result_155)
                    
                    # Assigning a Call to a Attribute (line 29):
                    
                    # Assigning a Call to a Attribute (line 29):
                    
                    # Call to unitize(...): (line 29)
                    # Processing the call keyword arguments (line 29)
                    kwargs_165 = {}
                    
                    # Call to cross(...): (line 29)
                    # Processing the call arguments (line 29)
                    # Getting the type of 'self' (line 29)
                    self_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 47), 'self', False)
                    # Obtaining the member 'view_direction' of a type (line 29)
                    view_direction_161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 47), self_160, 'view_direction')
                    # Processing the call keyword arguments (line 29)
                    kwargs_162 = {}
                    # Getting the type of 'self' (line 29)
                    self_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 33), 'self', False)
                    # Obtaining the member 'up' of a type (line 29)
                    up_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 33), self_157, 'up')
                    # Obtaining the member 'cross' of a type (line 29)
                    cross_159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 33), up_158, 'cross')
                    # Calling cross(args, kwargs) (line 29)
                    cross_call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 29, 33), cross_159, *[view_direction_161], **kwargs_162)
                    
                    # Obtaining the member 'unitize' of a type (line 29)
                    unitize_164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 33), cross_call_result_163, 'unitize')
                    # Calling unitize(args, kwargs) (line 29)
                    unitize_call_result_166 = invoke(stypy.reporting.localization.Localization(__file__, 29, 33), unitize_164, *[], **kwargs_165)
                    
                    # Getting the type of 'self' (line 29)
                    self_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'self')
                    # Setting the type of the member 'right' of a type (line 29)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 20), self_167, 'right', unitize_call_result_166)
                    # SSA branch for the else part of an if statement (line 27)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Attribute (line 31):
                    
                    # Assigning a Call to a Attribute (line 31):
                    
                    # Call to unitize(...): (line 31)
                    # Processing the call keyword arguments (line 31)
                    kwargs_176 = {}
                    
                    # Call to cross(...): (line 31)
                    # Processing the call arguments (line 31)
                    # Getting the type of 'self' (line 31)
                    self_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 56), 'self', False)
                    # Obtaining the member 'right' of a type (line 31)
                    right_172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 56), self_171, 'right')
                    # Processing the call keyword arguments (line 31)
                    kwargs_173 = {}
                    # Getting the type of 'self' (line 31)
                    self_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'self', False)
                    # Obtaining the member 'view_direction' of a type (line 31)
                    view_direction_169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), self_168, 'view_direction')
                    # Obtaining the member 'cross' of a type (line 31)
                    cross_170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), view_direction_169, 'cross')
                    # Calling cross(args, kwargs) (line 31)
                    cross_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), cross_170, *[right_172], **kwargs_173)
                    
                    # Obtaining the member 'unitize' of a type (line 31)
                    unitize_175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), cross_call_result_174, 'unitize')
                    # Calling unitize(args, kwargs) (line 31)
                    unitize_call_result_177 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), unitize_175, *[], **kwargs_176)
                    
                    # Getting the type of 'self' (line 31)
                    self_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'self')
                    # Setting the type of the member 'up' of a type (line 31)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), self_178, 'up', unitize_call_result_177)
                    # SSA join for if statement (line 27)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 19)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_frame(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_frame'
        module_type_store = module_type_store.open_function_context('get_frame', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Camera.get_frame.__dict__.__setitem__('stypy_localization', localization)
        Camera.get_frame.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Camera.get_frame.__dict__.__setitem__('stypy_type_store', module_type_store)
        Camera.get_frame.__dict__.__setitem__('stypy_function_name', 'Camera.get_frame')
        Camera.get_frame.__dict__.__setitem__('stypy_param_names_list', ['scene', 'image'])
        Camera.get_frame.__dict__.__setitem__('stypy_varargs_param_name', None)
        Camera.get_frame.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Camera.get_frame.__dict__.__setitem__('stypy_call_defaults', defaults)
        Camera.get_frame.__dict__.__setitem__('stypy_call_varargs', varargs)
        Camera.get_frame.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Camera.get_frame.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Camera.get_frame', ['scene', 'image'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_frame', localization, ['scene', 'image'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_frame(...)' code ##################

        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to RayTracer(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'scene' (line 35)
        scene_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 30), 'scene', False)
        # Processing the call keyword arguments (line 35)
        kwargs_181 = {}
        # Getting the type of 'RayTracer' (line 35)
        RayTracer_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'RayTracer', False)
        # Calling RayTracer(args, kwargs) (line 35)
        RayTracer_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 35, 20), RayTracer_179, *[scene_180], **kwargs_181)
        
        # Assigning a type to the variable 'raytracer' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'raytracer', RayTracer_call_result_182)
        
        # Assigning a BinOp to a Name (line 36):
        
        # Assigning a BinOp to a Name (line 36):
        
        # Call to float(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'image' (line 36)
        image_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'image', False)
        # Obtaining the member 'height' of a type (line 36)
        height_185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), image_184, 'height')
        # Processing the call keyword arguments (line 36)
        kwargs_186 = {}
        # Getting the type of 'float' (line 36)
        float_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'float', False)
        # Calling float(args, kwargs) (line 36)
        float_call_result_187 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), float_183, *[height_185], **kwargs_186)
        
        
        # Call to float(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'image' (line 36)
        image_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 45), 'image', False)
        # Obtaining the member 'width' of a type (line 36)
        width_190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 45), image_189, 'width')
        # Processing the call keyword arguments (line 36)
        kwargs_191 = {}
        # Getting the type of 'float' (line 36)
        float_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 39), 'float', False)
        # Calling float(args, kwargs) (line 36)
        float_call_result_192 = invoke(stypy.reporting.localization.Localization(__file__, 36, 39), float_188, *[width_190], **kwargs_191)
        
        # Applying the binary operator 'div' (line 36)
        result_div_193 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 17), 'div', float_call_result_187, float_call_result_192)
        
        # Assigning a type to the variable 'aspect' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'aspect', result_div_193)
        
        
        # Call to range(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'image' (line 37)
        image_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'image', False)
        # Obtaining the member 'height' of a type (line 37)
        height_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 23), image_195, 'height')
        # Processing the call keyword arguments (line 37)
        kwargs_197 = {}
        # Getting the type of 'range' (line 37)
        range_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'range', False)
        # Calling range(args, kwargs) (line 37)
        range_call_result_198 = invoke(stypy.reporting.localization.Localization(__file__, 37, 17), range_194, *[height_196], **kwargs_197)
        
        # Assigning a type to the variable 'range_call_result_198' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'range_call_result_198', range_call_result_198)
        # Testing if the for loop is going to be iterated (line 37)
        # Testing the type of a for loop iterable (line 37)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 8), range_call_result_198)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 37, 8), range_call_result_198):
            # Getting the type of the for loop variable (line 37)
            for_loop_var_199 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 8), range_call_result_198)
            # Assigning a type to the variable 'y' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'y', for_loop_var_199)
            # SSA begins for a for statement (line 37)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 38)
            # Processing the call arguments (line 38)
            # Getting the type of 'image' (line 38)
            image_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'image', False)
            # Obtaining the member 'width' of a type (line 38)
            width_202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 27), image_201, 'width')
            # Processing the call keyword arguments (line 38)
            kwargs_203 = {}
            # Getting the type of 'range' (line 38)
            range_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'range', False)
            # Calling range(args, kwargs) (line 38)
            range_call_result_204 = invoke(stypy.reporting.localization.Localization(__file__, 38, 21), range_200, *[width_202], **kwargs_203)
            
            # Assigning a type to the variable 'range_call_result_204' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'range_call_result_204', range_call_result_204)
            # Testing if the for loop is going to be iterated (line 38)
            # Testing the type of a for loop iterable (line 38)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 12), range_call_result_204)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 38, 12), range_call_result_204):
                # Getting the type of the for loop variable (line 38)
                for_loop_var_205 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 12), range_call_result_204)
                # Assigning a type to the variable 'x' (line 38)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'x', for_loop_var_205)
                # SSA begins for a for statement (line 38)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a BinOp to a Name (line 39):
                
                # Assigning a BinOp to a Name (line 39):
                # Getting the type of 'x' (line 39)
                x_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 34), 'x')
                
                # Call to random(...): (line 39)
                # Processing the call keyword arguments (line 39)
                kwargs_208 = {}
                # Getting the type of 'random' (line 39)
                random_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 38), 'random', False)
                # Calling random(args, kwargs) (line 39)
                random_call_result_209 = invoke(stypy.reporting.localization.Localization(__file__, 39, 38), random_207, *[], **kwargs_208)
                
                # Applying the binary operator '+' (line 39)
                result_add_210 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 34), '+', x_206, random_call_result_209)
                
                float_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 50), 'float')
                # Applying the binary operator '*' (line 39)
                result_mul_212 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 33), '*', result_add_210, float_211)
                
                # Getting the type of 'image' (line 39)
                image_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 56), 'image')
                # Obtaining the member 'width' of a type (line 39)
                width_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 56), image_213, 'width')
                # Applying the binary operator 'div' (line 39)
                result_div_215 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 54), 'div', result_mul_212, width_214)
                
                float_216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 71), 'float')
                # Applying the binary operator '-' (line 39)
                result_sub_217 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 32), '-', result_div_215, float_216)
                
                # Assigning a type to the variable 'x_coefficient' (line 39)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'x_coefficient', result_sub_217)
                
                # Assigning a BinOp to a Name (line 40):
                
                # Assigning a BinOp to a Name (line 40):
                # Getting the type of 'y' (line 40)
                y_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 34), 'y')
                
                # Call to random(...): (line 40)
                # Processing the call keyword arguments (line 40)
                kwargs_220 = {}
                # Getting the type of 'random' (line 40)
                random_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 38), 'random', False)
                # Calling random(args, kwargs) (line 40)
                random_call_result_221 = invoke(stypy.reporting.localization.Localization(__file__, 40, 38), random_219, *[], **kwargs_220)
                
                # Applying the binary operator '+' (line 40)
                result_add_222 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 34), '+', y_218, random_call_result_221)
                
                float_223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 50), 'float')
                # Applying the binary operator '*' (line 40)
                result_mul_224 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 33), '*', result_add_222, float_223)
                
                # Getting the type of 'image' (line 40)
                image_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 56), 'image')
                # Obtaining the member 'height' of a type (line 40)
                height_226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 56), image_225, 'height')
                # Applying the binary operator 'div' (line 40)
                result_div_227 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 54), 'div', result_mul_224, height_226)
                
                float_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 72), 'float')
                # Applying the binary operator '-' (line 40)
                result_sub_229 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 32), '-', result_div_227, float_228)
                
                # Assigning a type to the variable 'y_coefficient' (line 40)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'y_coefficient', result_sub_229)
                
                # Assigning a BinOp to a Name (line 41):
                
                # Assigning a BinOp to a Name (line 41):
                # Getting the type of 'self' (line 41)
                self_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'self')
                # Obtaining the member 'right' of a type (line 41)
                right_231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 25), self_230, 'right')
                # Getting the type of 'x_coefficient' (line 41)
                x_coefficient_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 38), 'x_coefficient')
                # Applying the binary operator '*' (line 41)
                result_mul_233 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 25), '*', right_231, x_coefficient_232)
                
                # Getting the type of 'self' (line 41)
                self_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 54), 'self')
                # Obtaining the member 'up' of a type (line 41)
                up_235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 54), self_234, 'up')
                # Getting the type of 'y_coefficient' (line 41)
                y_coefficient_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 65), 'y_coefficient')
                # Getting the type of 'aspect' (line 41)
                aspect_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 81), 'aspect')
                # Applying the binary operator '*' (line 41)
                result_mul_238 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 65), '*', y_coefficient_236, aspect_237)
                
                # Applying the binary operator '*' (line 41)
                result_mul_239 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 54), '*', up_235, result_mul_238)
                
                # Applying the binary operator '+' (line 41)
                result_add_240 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 25), '+', result_mul_233, result_mul_239)
                
                # Assigning a type to the variable 'offset' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'offset', result_add_240)
                
                # Assigning a Call to a Name (line 42):
                
                # Assigning a Call to a Name (line 42):
                
                # Call to unitize(...): (line 42)
                # Processing the call keyword arguments (line 42)
                kwargs_254 = {}
                # Getting the type of 'self' (line 42)
                self_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 36), 'self', False)
                # Obtaining the member 'view_direction' of a type (line 42)
                view_direction_242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 36), self_241, 'view_direction')
                # Getting the type of 'offset' (line 42)
                offset_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 59), 'offset', False)
                
                # Call to tan(...): (line 42)
                # Processing the call arguments (line 42)
                # Getting the type of 'self' (line 42)
                self_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 72), 'self', False)
                # Obtaining the member 'view_angle' of a type (line 42)
                view_angle_246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 72), self_245, 'view_angle')
                float_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 90), 'float')
                # Applying the binary operator '*' (line 42)
                result_mul_248 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 72), '*', view_angle_246, float_247)
                
                # Processing the call keyword arguments (line 42)
                kwargs_249 = {}
                # Getting the type of 'tan' (line 42)
                tan_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 68), 'tan', False)
                # Calling tan(args, kwargs) (line 42)
                tan_call_result_250 = invoke(stypy.reporting.localization.Localization(__file__, 42, 68), tan_244, *[result_mul_248], **kwargs_249)
                
                # Applying the binary operator '*' (line 42)
                result_mul_251 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 59), '*', offset_243, tan_call_result_250)
                
                # Applying the binary operator '+' (line 42)
                result_add_252 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 36), '+', view_direction_242, result_mul_251)
                
                # Obtaining the member 'unitize' of a type (line 42)
                unitize_253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 36), result_add_252, 'unitize')
                # Calling unitize(args, kwargs) (line 42)
                unitize_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 42, 36), unitize_253, *[], **kwargs_254)
                
                # Assigning a type to the variable 'sample_direction' (line 42)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'sample_direction', unitize_call_result_255)
                
                # Assigning a Call to a Name (line 43):
                
                # Assigning a Call to a Name (line 43):
                
                # Call to get_radiance(...): (line 43)
                # Processing the call arguments (line 43)
                # Getting the type of 'self' (line 43)
                self_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 50), 'self', False)
                # Obtaining the member 'view_position' of a type (line 43)
                view_position_259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 50), self_258, 'view_position')
                # Getting the type of 'sample_direction' (line 43)
                sample_direction_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 70), 'sample_direction', False)
                # Processing the call keyword arguments (line 43)
                kwargs_261 = {}
                # Getting the type of 'raytracer' (line 43)
                raytracer_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 27), 'raytracer', False)
                # Obtaining the member 'get_radiance' of a type (line 43)
                get_radiance_257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 27), raytracer_256, 'get_radiance')
                # Calling get_radiance(args, kwargs) (line 43)
                get_radiance_call_result_262 = invoke(stypy.reporting.localization.Localization(__file__, 43, 27), get_radiance_257, *[view_position_259, sample_direction_260], **kwargs_261)
                
                # Assigning a type to the variable 'radiance' (line 43)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'radiance', get_radiance_call_result_262)
                
                # Call to add_to_pixel(...): (line 44)
                # Processing the call arguments (line 44)
                # Getting the type of 'x' (line 44)
                x_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'x', False)
                # Getting the type of 'y' (line 44)
                y_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 38), 'y', False)
                # Getting the type of 'radiance' (line 44)
                radiance_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 41), 'radiance', False)
                # Processing the call keyword arguments (line 44)
                kwargs_268 = {}
                # Getting the type of 'image' (line 44)
                image_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'image', False)
                # Obtaining the member 'add_to_pixel' of a type (line 44)
                add_to_pixel_264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 16), image_263, 'add_to_pixel')
                # Calling add_to_pixel(args, kwargs) (line 44)
                add_to_pixel_call_result_269 = invoke(stypy.reporting.localization.Localization(__file__, 44, 16), add_to_pixel_264, *[x_265, y_266, radiance_267], **kwargs_268)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'get_frame(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_frame' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_270)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_frame'
        return stypy_return_type_270


# Assigning a type to the variable 'Camera' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'Camera', Camera)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
