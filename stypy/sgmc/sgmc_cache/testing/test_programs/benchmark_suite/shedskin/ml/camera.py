
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
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'int')
                # Processing the call keyword arguments
                kwargs_75 = {}
                # Getting the type of 'call_assignment_43' (line 20)
                call_assignment_43_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_43', False)
                # Obtaining the member '__getitem__' of a type (line 20)
                getitem___73 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 16), call_assignment_43_72, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_76 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___73, *[int_74], **kwargs_75)
                
                # Assigning a type to the variable 'call_assignment_44' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_44', getitem___call_result_76)
                
                # Assigning a Name to a Name (line 20):
                # Getting the type of 'call_assignment_44' (line 20)
                call_assignment_44_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_44')
                # Assigning a type to the variable 'p' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'p', call_assignment_44_77)
                
                # Assigning a Call to a Name (line 20):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'int')
                # Processing the call keyword arguments
                kwargs_81 = {}
                # Getting the type of 'call_assignment_43' (line 20)
                call_assignment_43_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_43', False)
                # Obtaining the member '__getitem__' of a type (line 20)
                getitem___79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 16), call_assignment_43_78, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___79, *[int_80], **kwargs_81)
                
                # Assigning a type to the variable 'call_assignment_45' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_45', getitem___call_result_82)
                
                # Assigning a Name to a Name (line 20):
                # Getting the type of 'call_assignment_45' (line 20)
                call_assignment_45_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_45')
                # Assigning a type to the variable 'd' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'd', call_assignment_45_83)
                
                # Assigning a Call to a Name (line 20):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'int')
                # Processing the call keyword arguments
                kwargs_87 = {}
                # Getting the type of 'call_assignment_43' (line 20)
                call_assignment_43_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_43', False)
                # Obtaining the member '__getitem__' of a type (line 20)
                getitem___85 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 16), call_assignment_43_84, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___85, *[int_86], **kwargs_87)
                
                # Assigning a type to the variable 'call_assignment_46' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_46', getitem___call_result_88)
                
                # Assigning a Name to a Name (line 20):
                # Getting the type of 'call_assignment_46' (line 20)
                call_assignment_46_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'call_assignment_46')
                # Assigning a type to the variable 'a' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'a', call_assignment_46_89)
                
                # Assigning a Call to a Attribute (line 21):
                
                # Assigning a Call to a Attribute (line 21):
                
                # Call to Vector3f_str(...): (line 21)
                # Processing the call arguments (line 21)
                # Getting the type of 'p' (line 21)
                p_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 50), 'p', False)
                # Processing the call keyword arguments (line 21)
                kwargs_92 = {}
                # Getting the type of 'Vector3f_str' (line 21)
                Vector3f_str_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 37), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 21)
                Vector3f_str_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 21, 37), Vector3f_str_90, *[p_91], **kwargs_92)
                
                # Getting the type of 'self' (line 21)
                self_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'self')
                # Setting the type of the member 'view_position' of a type (line 21)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), self_94, 'view_position', Vector3f_str_call_result_93)
                
                # Assigning a Call to a Attribute (line 22):
                
                # Assigning a Call to a Attribute (line 22):
                
                # Call to unitize(...): (line 22)
                # Processing the call keyword arguments (line 22)
                kwargs_100 = {}
                
                # Call to Vector3f_str(...): (line 22)
                # Processing the call arguments (line 22)
                # Getting the type of 'd' (line 22)
                d_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 51), 'd', False)
                # Processing the call keyword arguments (line 22)
                kwargs_97 = {}
                # Getting the type of 'Vector3f_str' (line 22)
                Vector3f_str_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 38), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 22)
                Vector3f_str_call_result_98 = invoke(stypy.reporting.localization.Localization(__file__, 22, 38), Vector3f_str_95, *[d_96], **kwargs_97)
                
                # Obtaining the member 'unitize' of a type (line 22)
                unitize_99 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 38), Vector3f_str_call_result_98, 'unitize')
                # Calling unitize(args, kwargs) (line 22)
                unitize_call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 22, 38), unitize_99, *[], **kwargs_100)
                
                # Getting the type of 'self' (line 22)
                self_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'self')
                # Setting the type of the member 'view_direction' of a type (line 22)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), self_102, 'view_direction', unitize_call_result_101)
                
                # Call to is_zero(...): (line 23)
                # Processing the call keyword arguments (line 23)
                kwargs_106 = {}
                # Getting the type of 'self' (line 23)
                self_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'self', False)
                # Obtaining the member 'view_direction' of a type (line 23)
                view_direction_104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 19), self_103, 'view_direction')
                # Obtaining the member 'is_zero' of a type (line 23)
                is_zero_105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 19), view_direction_104, 'is_zero')
                # Calling is_zero(args, kwargs) (line 23)
                is_zero_call_result_107 = invoke(stypy.reporting.localization.Localization(__file__, 23, 19), is_zero_105, *[], **kwargs_106)
                
                # Testing if the type of an if condition is none (line 23)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 23, 16), is_zero_call_result_107):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 23)
                    if_condition_108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 16), is_zero_call_result_107)
                    # Assigning a type to the variable 'if_condition_108' (line 23)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'if_condition_108', if_condition_108)
                    # SSA begins for if statement (line 23)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Attribute (line 24):
                    
                    # Assigning a Call to a Attribute (line 24):
                    
                    # Call to Vector3f(...): (line 24)
                    # Processing the call arguments (line 24)
                    float_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 51), 'float')
                    float_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 56), 'float')
                    float_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 61), 'float')
                    # Processing the call keyword arguments (line 24)
                    kwargs_113 = {}
                    # Getting the type of 'Vector3f' (line 24)
                    Vector3f_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 42), 'Vector3f', False)
                    # Calling Vector3f(args, kwargs) (line 24)
                    Vector3f_call_result_114 = invoke(stypy.reporting.localization.Localization(__file__, 24, 42), Vector3f_109, *[float_110, float_111, float_112], **kwargs_113)
                    
                    # Getting the type of 'self' (line 24)
                    self_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'self')
                    # Setting the type of the member 'view_direction' of a type (line 24)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 20), self_115, 'view_direction', Vector3f_call_result_114)
                    # SSA join for if statement (line 23)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a BinOp to a Attribute (line 25):
                
                # Assigning a BinOp to a Attribute (line 25):
                
                # Call to min(...): (line 25)
                # Processing the call arguments (line 25)
                
                # Call to max(...): (line 25)
                # Processing the call arguments (line 25)
                float_118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 42), 'float')
                
                # Call to float(...): (line 25)
                # Processing the call arguments (line 25)
                # Getting the type of 'a' (line 25)
                a_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 54), 'a', False)
                # Processing the call keyword arguments (line 25)
                kwargs_121 = {}
                # Getting the type of 'float' (line 25)
                float_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 48), 'float', False)
                # Calling float(args, kwargs) (line 25)
                float_call_result_122 = invoke(stypy.reporting.localization.Localization(__file__, 25, 48), float_119, *[a_120], **kwargs_121)
                
                # Processing the call keyword arguments (line 25)
                kwargs_123 = {}
                # Getting the type of 'max' (line 25)
                max_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 38), 'max', False)
                # Calling max(args, kwargs) (line 25)
                max_call_result_124 = invoke(stypy.reporting.localization.Localization(__file__, 25, 38), max_117, *[float_118, float_call_result_122], **kwargs_123)
                
                float_125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 59), 'float')
                # Processing the call keyword arguments (line 25)
                kwargs_126 = {}
                # Getting the type of 'min' (line 25)
                min_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 34), 'min', False)
                # Calling min(args, kwargs) (line 25)
                min_call_result_127 = invoke(stypy.reporting.localization.Localization(__file__, 25, 34), min_116, *[max_call_result_124, float_125], **kwargs_126)
                
                # Getting the type of 'pi' (line 25)
                pi_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 69), 'pi')
                float_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 74), 'float')
                # Applying the binary operator 'div' (line 25)
                result_div_130 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 69), 'div', pi_128, float_129)
                
                # Applying the binary operator '*' (line 25)
                result_mul_131 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 34), '*', min_call_result_127, result_div_130)
                
                # Getting the type of 'self' (line 25)
                self_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'self')
                # Setting the type of the member 'view_angle' of a type (line 25)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), self_132, 'view_angle', result_mul_131)
                
                # Assigning a Call to a Attribute (line 26):
                
                # Assigning a Call to a Attribute (line 26):
                
                # Call to unitize(...): (line 26)
                # Processing the call keyword arguments (line 26)
                kwargs_145 = {}
                
                # Call to cross(...): (line 26)
                # Processing the call arguments (line 26)
                # Getting the type of 'self' (line 26)
                self_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 59), 'self', False)
                # Obtaining the member 'view_direction' of a type (line 26)
                view_direction_141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 59), self_140, 'view_direction')
                # Processing the call keyword arguments (line 26)
                kwargs_142 = {}
                
                # Call to Vector3f(...): (line 26)
                # Processing the call arguments (line 26)
                float_134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 38), 'float')
                float_135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 43), 'float')
                float_136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 48), 'float')
                # Processing the call keyword arguments (line 26)
                kwargs_137 = {}
                # Getting the type of 'Vector3f' (line 26)
                Vector3f_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'Vector3f', False)
                # Calling Vector3f(args, kwargs) (line 26)
                Vector3f_call_result_138 = invoke(stypy.reporting.localization.Localization(__file__, 26, 29), Vector3f_133, *[float_134, float_135, float_136], **kwargs_137)
                
                # Obtaining the member 'cross' of a type (line 26)
                cross_139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 29), Vector3f_call_result_138, 'cross')
                # Calling cross(args, kwargs) (line 26)
                cross_call_result_143 = invoke(stypy.reporting.localization.Localization(__file__, 26, 29), cross_139, *[view_direction_141], **kwargs_142)
                
                # Obtaining the member 'unitize' of a type (line 26)
                unitize_144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 29), cross_call_result_143, 'unitize')
                # Calling unitize(args, kwargs) (line 26)
                unitize_call_result_146 = invoke(stypy.reporting.localization.Localization(__file__, 26, 29), unitize_144, *[], **kwargs_145)
                
                # Getting the type of 'self' (line 26)
                self_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'self')
                # Setting the type of the member 'right' of a type (line 26)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), self_147, 'right', unitize_call_result_146)
                
                # Call to is_zero(...): (line 27)
                # Processing the call keyword arguments (line 27)
                kwargs_151 = {}
                # Getting the type of 'self' (line 27)
                self_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'self', False)
                # Obtaining the member 'right' of a type (line 27)
                right_149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 19), self_148, 'right')
                # Obtaining the member 'is_zero' of a type (line 27)
                is_zero_150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 19), right_149, 'is_zero')
                # Calling is_zero(args, kwargs) (line 27)
                is_zero_call_result_152 = invoke(stypy.reporting.localization.Localization(__file__, 27, 19), is_zero_150, *[], **kwargs_151)
                
                # Testing if the type of an if condition is none (line 27)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 27, 16), is_zero_call_result_152):
                    
                    # Assigning a Call to a Attribute (line 31):
                    
                    # Assigning a Call to a Attribute (line 31):
                    
                    # Call to unitize(...): (line 31)
                    # Processing the call keyword arguments (line 31)
                    kwargs_185 = {}
                    
                    # Call to cross(...): (line 31)
                    # Processing the call arguments (line 31)
                    # Getting the type of 'self' (line 31)
                    self_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 56), 'self', False)
                    # Obtaining the member 'right' of a type (line 31)
                    right_181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 56), self_180, 'right')
                    # Processing the call keyword arguments (line 31)
                    kwargs_182 = {}
                    # Getting the type of 'self' (line 31)
                    self_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'self', False)
                    # Obtaining the member 'view_direction' of a type (line 31)
                    view_direction_178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), self_177, 'view_direction')
                    # Obtaining the member 'cross' of a type (line 31)
                    cross_179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), view_direction_178, 'cross')
                    # Calling cross(args, kwargs) (line 31)
                    cross_call_result_183 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), cross_179, *[right_181], **kwargs_182)
                    
                    # Obtaining the member 'unitize' of a type (line 31)
                    unitize_184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), cross_call_result_183, 'unitize')
                    # Calling unitize(args, kwargs) (line 31)
                    unitize_call_result_186 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), unitize_184, *[], **kwargs_185)
                    
                    # Getting the type of 'self' (line 31)
                    self_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'self')
                    # Setting the type of the member 'up' of a type (line 31)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), self_187, 'up', unitize_call_result_186)
                else:
                    
                    # Testing the type of an if condition (line 27)
                    if_condition_153 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 16), is_zero_call_result_152)
                    # Assigning a type to the variable 'if_condition_153' (line 27)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'if_condition_153', if_condition_153)
                    # SSA begins for if statement (line 27)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Attribute (line 28):
                    
                    # Assigning a Call to a Attribute (line 28):
                    
                    # Call to Vector3f(...): (line 28)
                    # Processing the call arguments (line 28)
                    float_155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 39), 'float')
                    float_156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 44), 'float')
                    
                    # Getting the type of 'self' (line 28)
                    self_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 56), 'self', False)
                    # Obtaining the member 'view_direction' of a type (line 28)
                    view_direction_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 56), self_157, 'view_direction')
                    # Obtaining the member 'y' of a type (line 28)
                    y_159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 56), view_direction_158, 'y')
                    # Testing the type of an if expression (line 28)
                    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 49), y_159)
                    # SSA begins for if expression (line 28)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                    float_160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 49), 'float')
                    # SSA branch for the else part of an if expression (line 28)
                    module_type_store.open_ssa_branch('if expression else')
                    float_161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 83), 'float')
                    # SSA join for if expression (line 28)
                    module_type_store = module_type_store.join_ssa_context()
                    if_exp_162 = union_type.UnionType.add(float_160, float_161)
                    
                    # Processing the call keyword arguments (line 28)
                    kwargs_163 = {}
                    # Getting the type of 'Vector3f' (line 28)
                    Vector3f_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'Vector3f', False)
                    # Calling Vector3f(args, kwargs) (line 28)
                    Vector3f_call_result_164 = invoke(stypy.reporting.localization.Localization(__file__, 28, 30), Vector3f_154, *[float_155, float_156, if_exp_162], **kwargs_163)
                    
                    # Getting the type of 'self' (line 28)
                    self_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'self')
                    # Setting the type of the member 'up' of a type (line 28)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 20), self_165, 'up', Vector3f_call_result_164)
                    
                    # Assigning a Call to a Attribute (line 29):
                    
                    # Assigning a Call to a Attribute (line 29):
                    
                    # Call to unitize(...): (line 29)
                    # Processing the call keyword arguments (line 29)
                    kwargs_174 = {}
                    
                    # Call to cross(...): (line 29)
                    # Processing the call arguments (line 29)
                    # Getting the type of 'self' (line 29)
                    self_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 47), 'self', False)
                    # Obtaining the member 'view_direction' of a type (line 29)
                    view_direction_170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 47), self_169, 'view_direction')
                    # Processing the call keyword arguments (line 29)
                    kwargs_171 = {}
                    # Getting the type of 'self' (line 29)
                    self_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 33), 'self', False)
                    # Obtaining the member 'up' of a type (line 29)
                    up_167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 33), self_166, 'up')
                    # Obtaining the member 'cross' of a type (line 29)
                    cross_168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 33), up_167, 'cross')
                    # Calling cross(args, kwargs) (line 29)
                    cross_call_result_172 = invoke(stypy.reporting.localization.Localization(__file__, 29, 33), cross_168, *[view_direction_170], **kwargs_171)
                    
                    # Obtaining the member 'unitize' of a type (line 29)
                    unitize_173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 33), cross_call_result_172, 'unitize')
                    # Calling unitize(args, kwargs) (line 29)
                    unitize_call_result_175 = invoke(stypy.reporting.localization.Localization(__file__, 29, 33), unitize_173, *[], **kwargs_174)
                    
                    # Getting the type of 'self' (line 29)
                    self_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'self')
                    # Setting the type of the member 'right' of a type (line 29)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 20), self_176, 'right', unitize_call_result_175)
                    # SSA branch for the else part of an if statement (line 27)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Attribute (line 31):
                    
                    # Assigning a Call to a Attribute (line 31):
                    
                    # Call to unitize(...): (line 31)
                    # Processing the call keyword arguments (line 31)
                    kwargs_185 = {}
                    
                    # Call to cross(...): (line 31)
                    # Processing the call arguments (line 31)
                    # Getting the type of 'self' (line 31)
                    self_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 56), 'self', False)
                    # Obtaining the member 'right' of a type (line 31)
                    right_181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 56), self_180, 'right')
                    # Processing the call keyword arguments (line 31)
                    kwargs_182 = {}
                    # Getting the type of 'self' (line 31)
                    self_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'self', False)
                    # Obtaining the member 'view_direction' of a type (line 31)
                    view_direction_178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), self_177, 'view_direction')
                    # Obtaining the member 'cross' of a type (line 31)
                    cross_179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), view_direction_178, 'cross')
                    # Calling cross(args, kwargs) (line 31)
                    cross_call_result_183 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), cross_179, *[right_181], **kwargs_182)
                    
                    # Obtaining the member 'unitize' of a type (line 31)
                    unitize_184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), cross_call_result_183, 'unitize')
                    # Calling unitize(args, kwargs) (line 31)
                    unitize_call_result_186 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), unitize_184, *[], **kwargs_185)
                    
                    # Getting the type of 'self' (line 31)
                    self_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'self')
                    # Setting the type of the member 'up' of a type (line 31)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), self_187, 'up', unitize_call_result_186)
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
        scene_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 30), 'scene', False)
        # Processing the call keyword arguments (line 35)
        kwargs_190 = {}
        # Getting the type of 'RayTracer' (line 35)
        RayTracer_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'RayTracer', False)
        # Calling RayTracer(args, kwargs) (line 35)
        RayTracer_call_result_191 = invoke(stypy.reporting.localization.Localization(__file__, 35, 20), RayTracer_188, *[scene_189], **kwargs_190)
        
        # Assigning a type to the variable 'raytracer' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'raytracer', RayTracer_call_result_191)
        
        # Assigning a BinOp to a Name (line 36):
        
        # Assigning a BinOp to a Name (line 36):
        
        # Call to float(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'image' (line 36)
        image_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'image', False)
        # Obtaining the member 'height' of a type (line 36)
        height_194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), image_193, 'height')
        # Processing the call keyword arguments (line 36)
        kwargs_195 = {}
        # Getting the type of 'float' (line 36)
        float_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'float', False)
        # Calling float(args, kwargs) (line 36)
        float_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), float_192, *[height_194], **kwargs_195)
        
        
        # Call to float(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'image' (line 36)
        image_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 45), 'image', False)
        # Obtaining the member 'width' of a type (line 36)
        width_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 45), image_198, 'width')
        # Processing the call keyword arguments (line 36)
        kwargs_200 = {}
        # Getting the type of 'float' (line 36)
        float_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 39), 'float', False)
        # Calling float(args, kwargs) (line 36)
        float_call_result_201 = invoke(stypy.reporting.localization.Localization(__file__, 36, 39), float_197, *[width_199], **kwargs_200)
        
        # Applying the binary operator 'div' (line 36)
        result_div_202 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 17), 'div', float_call_result_196, float_call_result_201)
        
        # Assigning a type to the variable 'aspect' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'aspect', result_div_202)
        
        
        # Call to range(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'image' (line 37)
        image_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'image', False)
        # Obtaining the member 'height' of a type (line 37)
        height_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 23), image_204, 'height')
        # Processing the call keyword arguments (line 37)
        kwargs_206 = {}
        # Getting the type of 'range' (line 37)
        range_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'range', False)
        # Calling range(args, kwargs) (line 37)
        range_call_result_207 = invoke(stypy.reporting.localization.Localization(__file__, 37, 17), range_203, *[height_205], **kwargs_206)
        
        # Testing if the for loop is going to be iterated (line 37)
        # Testing the type of a for loop iterable (line 37)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 8), range_call_result_207)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 37, 8), range_call_result_207):
            # Getting the type of the for loop variable (line 37)
            for_loop_var_208 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 8), range_call_result_207)
            # Assigning a type to the variable 'y' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'y', for_loop_var_208)
            # SSA begins for a for statement (line 37)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 38)
            # Processing the call arguments (line 38)
            # Getting the type of 'image' (line 38)
            image_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'image', False)
            # Obtaining the member 'width' of a type (line 38)
            width_211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 27), image_210, 'width')
            # Processing the call keyword arguments (line 38)
            kwargs_212 = {}
            # Getting the type of 'range' (line 38)
            range_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'range', False)
            # Calling range(args, kwargs) (line 38)
            range_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 38, 21), range_209, *[width_211], **kwargs_212)
            
            # Testing if the for loop is going to be iterated (line 38)
            # Testing the type of a for loop iterable (line 38)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 12), range_call_result_213)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 38, 12), range_call_result_213):
                # Getting the type of the for loop variable (line 38)
                for_loop_var_214 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 12), range_call_result_213)
                # Assigning a type to the variable 'x' (line 38)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'x', for_loop_var_214)
                # SSA begins for a for statement (line 38)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a BinOp to a Name (line 39):
                
                # Assigning a BinOp to a Name (line 39):
                # Getting the type of 'x' (line 39)
                x_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 34), 'x')
                
                # Call to random(...): (line 39)
                # Processing the call keyword arguments (line 39)
                kwargs_217 = {}
                # Getting the type of 'random' (line 39)
                random_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 38), 'random', False)
                # Calling random(args, kwargs) (line 39)
                random_call_result_218 = invoke(stypy.reporting.localization.Localization(__file__, 39, 38), random_216, *[], **kwargs_217)
                
                # Applying the binary operator '+' (line 39)
                result_add_219 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 34), '+', x_215, random_call_result_218)
                
                float_220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 50), 'float')
                # Applying the binary operator '*' (line 39)
                result_mul_221 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 33), '*', result_add_219, float_220)
                
                # Getting the type of 'image' (line 39)
                image_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 56), 'image')
                # Obtaining the member 'width' of a type (line 39)
                width_223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 56), image_222, 'width')
                # Applying the binary operator 'div' (line 39)
                result_div_224 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 54), 'div', result_mul_221, width_223)
                
                float_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 71), 'float')
                # Applying the binary operator '-' (line 39)
                result_sub_226 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 32), '-', result_div_224, float_225)
                
                # Assigning a type to the variable 'x_coefficient' (line 39)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'x_coefficient', result_sub_226)
                
                # Assigning a BinOp to a Name (line 40):
                
                # Assigning a BinOp to a Name (line 40):
                # Getting the type of 'y' (line 40)
                y_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 34), 'y')
                
                # Call to random(...): (line 40)
                # Processing the call keyword arguments (line 40)
                kwargs_229 = {}
                # Getting the type of 'random' (line 40)
                random_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 38), 'random', False)
                # Calling random(args, kwargs) (line 40)
                random_call_result_230 = invoke(stypy.reporting.localization.Localization(__file__, 40, 38), random_228, *[], **kwargs_229)
                
                # Applying the binary operator '+' (line 40)
                result_add_231 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 34), '+', y_227, random_call_result_230)
                
                float_232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 50), 'float')
                # Applying the binary operator '*' (line 40)
                result_mul_233 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 33), '*', result_add_231, float_232)
                
                # Getting the type of 'image' (line 40)
                image_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 56), 'image')
                # Obtaining the member 'height' of a type (line 40)
                height_235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 56), image_234, 'height')
                # Applying the binary operator 'div' (line 40)
                result_div_236 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 54), 'div', result_mul_233, height_235)
                
                float_237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 72), 'float')
                # Applying the binary operator '-' (line 40)
                result_sub_238 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 32), '-', result_div_236, float_237)
                
                # Assigning a type to the variable 'y_coefficient' (line 40)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'y_coefficient', result_sub_238)
                
                # Assigning a BinOp to a Name (line 41):
                
                # Assigning a BinOp to a Name (line 41):
                # Getting the type of 'self' (line 41)
                self_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'self')
                # Obtaining the member 'right' of a type (line 41)
                right_240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 25), self_239, 'right')
                # Getting the type of 'x_coefficient' (line 41)
                x_coefficient_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 38), 'x_coefficient')
                # Applying the binary operator '*' (line 41)
                result_mul_242 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 25), '*', right_240, x_coefficient_241)
                
                # Getting the type of 'self' (line 41)
                self_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 54), 'self')
                # Obtaining the member 'up' of a type (line 41)
                up_244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 54), self_243, 'up')
                # Getting the type of 'y_coefficient' (line 41)
                y_coefficient_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 65), 'y_coefficient')
                # Getting the type of 'aspect' (line 41)
                aspect_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 81), 'aspect')
                # Applying the binary operator '*' (line 41)
                result_mul_247 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 65), '*', y_coefficient_245, aspect_246)
                
                # Applying the binary operator '*' (line 41)
                result_mul_248 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 54), '*', up_244, result_mul_247)
                
                # Applying the binary operator '+' (line 41)
                result_add_249 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 25), '+', result_mul_242, result_mul_248)
                
                # Assigning a type to the variable 'offset' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'offset', result_add_249)
                
                # Assigning a Call to a Name (line 42):
                
                # Assigning a Call to a Name (line 42):
                
                # Call to unitize(...): (line 42)
                # Processing the call keyword arguments (line 42)
                kwargs_263 = {}
                # Getting the type of 'self' (line 42)
                self_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 36), 'self', False)
                # Obtaining the member 'view_direction' of a type (line 42)
                view_direction_251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 36), self_250, 'view_direction')
                # Getting the type of 'offset' (line 42)
                offset_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 59), 'offset', False)
                
                # Call to tan(...): (line 42)
                # Processing the call arguments (line 42)
                # Getting the type of 'self' (line 42)
                self_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 72), 'self', False)
                # Obtaining the member 'view_angle' of a type (line 42)
                view_angle_255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 72), self_254, 'view_angle')
                float_256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 90), 'float')
                # Applying the binary operator '*' (line 42)
                result_mul_257 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 72), '*', view_angle_255, float_256)
                
                # Processing the call keyword arguments (line 42)
                kwargs_258 = {}
                # Getting the type of 'tan' (line 42)
                tan_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 68), 'tan', False)
                # Calling tan(args, kwargs) (line 42)
                tan_call_result_259 = invoke(stypy.reporting.localization.Localization(__file__, 42, 68), tan_253, *[result_mul_257], **kwargs_258)
                
                # Applying the binary operator '*' (line 42)
                result_mul_260 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 59), '*', offset_252, tan_call_result_259)
                
                # Applying the binary operator '+' (line 42)
                result_add_261 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 36), '+', view_direction_251, result_mul_260)
                
                # Obtaining the member 'unitize' of a type (line 42)
                unitize_262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 36), result_add_261, 'unitize')
                # Calling unitize(args, kwargs) (line 42)
                unitize_call_result_264 = invoke(stypy.reporting.localization.Localization(__file__, 42, 36), unitize_262, *[], **kwargs_263)
                
                # Assigning a type to the variable 'sample_direction' (line 42)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'sample_direction', unitize_call_result_264)
                
                # Assigning a Call to a Name (line 43):
                
                # Assigning a Call to a Name (line 43):
                
                # Call to get_radiance(...): (line 43)
                # Processing the call arguments (line 43)
                # Getting the type of 'self' (line 43)
                self_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 50), 'self', False)
                # Obtaining the member 'view_position' of a type (line 43)
                view_position_268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 50), self_267, 'view_position')
                # Getting the type of 'sample_direction' (line 43)
                sample_direction_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 70), 'sample_direction', False)
                # Processing the call keyword arguments (line 43)
                kwargs_270 = {}
                # Getting the type of 'raytracer' (line 43)
                raytracer_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 27), 'raytracer', False)
                # Obtaining the member 'get_radiance' of a type (line 43)
                get_radiance_266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 27), raytracer_265, 'get_radiance')
                # Calling get_radiance(args, kwargs) (line 43)
                get_radiance_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 43, 27), get_radiance_266, *[view_position_268, sample_direction_269], **kwargs_270)
                
                # Assigning a type to the variable 'radiance' (line 43)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'radiance', get_radiance_call_result_271)
                
                # Call to add_to_pixel(...): (line 44)
                # Processing the call arguments (line 44)
                # Getting the type of 'x' (line 44)
                x_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'x', False)
                # Getting the type of 'y' (line 44)
                y_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 38), 'y', False)
                # Getting the type of 'radiance' (line 44)
                radiance_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 41), 'radiance', False)
                # Processing the call keyword arguments (line 44)
                kwargs_277 = {}
                # Getting the type of 'image' (line 44)
                image_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'image', False)
                # Obtaining the member 'add_to_pixel' of a type (line 44)
                add_to_pixel_273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 16), image_272, 'add_to_pixel')
                # Calling add_to_pixel(args, kwargs) (line 44)
                add_to_pixel_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 44, 16), add_to_pixel_273, *[x_274, y_275, radiance_276], **kwargs_277)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'get_frame(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_frame' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_frame'
        return stypy_return_type_279


# Assigning a type to the variable 'Camera' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'Camera', Camera)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
