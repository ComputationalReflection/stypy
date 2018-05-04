
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #  MiniLight Python : minimal global illumination renderer
2: #
3: #  Copyright (c) 2007-2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
4: #  http://www.hxa7241.org/
5: 
6: 
7: from random import choice
8: from spatialindex import SpatialIndex
9: from triangle import Triangle
10: from vector3f import Vector3f_str, ZERO, ONE, MAX
11: 
12: import re
13: SEARCH = re.compile('(\(.+\))\s*(\(.+\))')
14: 
15: MAX_TRIANGLES = 0x100000
16: 
17: class Scene(object):
18: 
19:     def __init__(self, in_stream, eye_position):
20:         for line in in_stream:
21:             if not line.isspace():
22:                 s, g = SEARCH.search(line).groups()
23:                 self.sky_emission = Vector3f_str(s).clamped(ZERO, MAX)
24:                 self.ground_reflection = Vector3f_str(g).clamped(ZERO, ONE)
25:                 self.triangles = []
26:                 try:
27:                     for i in range(MAX_TRIANGLES):
28:                         self.triangles.append(Triangle(in_stream))
29:                 except StopIteration:
30:                     pass
31:                 self.emitters = [triangle for triangle in self.triangles if not triangle.emitivity.is_zero() and triangle.area > 0.0]
32:                 self.index = SpatialIndex(eye_position, None, self.triangles)
33: 
34:                 break
35: 
36:     def get_intersection(self, ray_origin, ray_direction, last_hit):
37:         return self.index.get_intersection(ray_origin, ray_direction, last_hit)
38: 
39:     def get_emitter(self):
40:         emitter = None if len(self.emitters) == 0 else choice(self.emitters)
41:         return (emitter.get_sample_point() if emitter else ZERO), emitter
42: 
43:     def emitters_count(self):
44:         return len(self.emitters)
45: 
46:     def get_default_emission(self, back_direction):
47:         return self.sky_emission if back_direction.y < 0.0 else self.sky_emission.mul(self.ground_reflection)
48: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from random import choice' statement (line 7)
try:
    from random import choice

except:
    choice = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'random', None, module_type_store, ['choice'], [choice])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from spatialindex import SpatialIndex' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_893 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'spatialindex')

if (type(import_893) is not StypyTypeError):

    if (import_893 != 'pyd_module'):
        __import__(import_893)
        sys_modules_894 = sys.modules[import_893]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'spatialindex', sys_modules_894.module_type_store, module_type_store, ['SpatialIndex'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_894, sys_modules_894.module_type_store, module_type_store)
    else:
        from spatialindex import SpatialIndex

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'spatialindex', None, module_type_store, ['SpatialIndex'], [SpatialIndex])

else:
    # Assigning a type to the variable 'spatialindex' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'spatialindex', import_893)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from triangle import Triangle' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_895 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'triangle')

if (type(import_895) is not StypyTypeError):

    if (import_895 != 'pyd_module'):
        __import__(import_895)
        sys_modules_896 = sys.modules[import_895]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'triangle', sys_modules_896.module_type_store, module_type_store, ['Triangle'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_896, sys_modules_896.module_type_store, module_type_store)
    else:
        from triangle import Triangle

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'triangle', None, module_type_store, ['Triangle'], [Triangle])

else:
    # Assigning a type to the variable 'triangle' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'triangle', import_895)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from vector3f import Vector3f_str, ZERO, ONE, MAX' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_897 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'vector3f')

if (type(import_897) is not StypyTypeError):

    if (import_897 != 'pyd_module'):
        __import__(import_897)
        sys_modules_898 = sys.modules[import_897]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'vector3f', sys_modules_898.module_type_store, module_type_store, ['Vector3f_str', 'ZERO', 'ONE', 'MAX'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_898, sys_modules_898.module_type_store, module_type_store)
    else:
        from vector3f import Vector3f_str, ZERO, ONE, MAX

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'vector3f', None, module_type_store, ['Vector3f_str', 'ZERO', 'ONE', 'MAX'], [Vector3f_str, ZERO, ONE, MAX])

else:
    # Assigning a type to the variable 'vector3f' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'vector3f', import_897)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import re' statement (line 12)
import re

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 're', re, module_type_store)


# Assigning a Call to a Name (line 13):

# Assigning a Call to a Name (line 13):

# Call to compile(...): (line 13)
# Processing the call arguments (line 13)
str_901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'str', '(\\(.+\\))\\s*(\\(.+\\))')
# Processing the call keyword arguments (line 13)
kwargs_902 = {}
# Getting the type of 're' (line 13)
re_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 9), 're', False)
# Obtaining the member 'compile' of a type (line 13)
compile_900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 9), re_899, 'compile')
# Calling compile(args, kwargs) (line 13)
compile_call_result_903 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), compile_900, *[str_901], **kwargs_902)

# Assigning a type to the variable 'SEARCH' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'SEARCH', compile_call_result_903)

# Assigning a Num to a Name (line 15):

# Assigning a Num to a Name (line 15):
int_904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'int')
# Assigning a type to the variable 'MAX_TRIANGLES' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'MAX_TRIANGLES', int_904)
# Declaration of the 'Scene' class

class Scene(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Scene.__init__', ['in_stream', 'eye_position'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['in_stream', 'eye_position'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Getting the type of 'in_stream' (line 20)
        in_stream_905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'in_stream')
        # Assigning a type to the variable 'in_stream_905' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'in_stream_905', in_stream_905)
        # Testing if the for loop is going to be iterated (line 20)
        # Testing the type of a for loop iterable (line 20)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 20, 8), in_stream_905)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 20, 8), in_stream_905):
            # Getting the type of the for loop variable (line 20)
            for_loop_var_906 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 20, 8), in_stream_905)
            # Assigning a type to the variable 'line' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'line', for_loop_var_906)
            # SSA begins for a for statement (line 20)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to isspace(...): (line 21)
            # Processing the call keyword arguments (line 21)
            kwargs_909 = {}
            # Getting the type of 'line' (line 21)
            line_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'line', False)
            # Obtaining the member 'isspace' of a type (line 21)
            isspace_908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 19), line_907, 'isspace')
            # Calling isspace(args, kwargs) (line 21)
            isspace_call_result_910 = invoke(stypy.reporting.localization.Localization(__file__, 21, 19), isspace_908, *[], **kwargs_909)
            
            # Applying the 'not' unary operator (line 21)
            result_not__911 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 15), 'not', isspace_call_result_910)
            
            # Testing if the type of an if condition is none (line 21)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 21, 12), result_not__911):
                pass
            else:
                
                # Testing the type of an if condition (line 21)
                if_condition_912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 12), result_not__911)
                # Assigning a type to the variable 'if_condition_912' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'if_condition_912', if_condition_912)
                # SSA begins for if statement (line 21)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 22):
                
                # Assigning a Call to a Name:
                
                # Call to groups(...): (line 22)
                # Processing the call keyword arguments (line 22)
                kwargs_919 = {}
                
                # Call to search(...): (line 22)
                # Processing the call arguments (line 22)
                # Getting the type of 'line' (line 22)
                line_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 37), 'line', False)
                # Processing the call keyword arguments (line 22)
                kwargs_916 = {}
                # Getting the type of 'SEARCH' (line 22)
                SEARCH_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'SEARCH', False)
                # Obtaining the member 'search' of a type (line 22)
                search_914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 23), SEARCH_913, 'search')
                # Calling search(args, kwargs) (line 22)
                search_call_result_917 = invoke(stypy.reporting.localization.Localization(__file__, 22, 23), search_914, *[line_915], **kwargs_916)
                
                # Obtaining the member 'groups' of a type (line 22)
                groups_918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 23), search_call_result_917, 'groups')
                # Calling groups(args, kwargs) (line 22)
                groups_call_result_920 = invoke(stypy.reporting.localization.Localization(__file__, 22, 23), groups_918, *[], **kwargs_919)
                
                # Assigning a type to the variable 'call_assignment_890' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_890', groups_call_result_920)
                
                # Assigning a Call to a Name (line 22):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_890' (line 22)
                call_assignment_890_921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_890', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_922 = stypy_get_value_from_tuple(call_assignment_890_921, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_891' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_891', stypy_get_value_from_tuple_call_result_922)
                
                # Assigning a Name to a Name (line 22):
                # Getting the type of 'call_assignment_891' (line 22)
                call_assignment_891_923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_891')
                # Assigning a type to the variable 's' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 's', call_assignment_891_923)
                
                # Assigning a Call to a Name (line 22):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_890' (line 22)
                call_assignment_890_924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_890', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_925 = stypy_get_value_from_tuple(call_assignment_890_924, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_892' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_892', stypy_get_value_from_tuple_call_result_925)
                
                # Assigning a Name to a Name (line 22):
                # Getting the type of 'call_assignment_892' (line 22)
                call_assignment_892_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_892')
                # Assigning a type to the variable 'g' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'g', call_assignment_892_926)
                
                # Assigning a Call to a Attribute (line 23):
                
                # Assigning a Call to a Attribute (line 23):
                
                # Call to clamped(...): (line 23)
                # Processing the call arguments (line 23)
                # Getting the type of 'ZERO' (line 23)
                ZERO_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 60), 'ZERO', False)
                # Getting the type of 'MAX' (line 23)
                MAX_933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 66), 'MAX', False)
                # Processing the call keyword arguments (line 23)
                kwargs_934 = {}
                
                # Call to Vector3f_str(...): (line 23)
                # Processing the call arguments (line 23)
                # Getting the type of 's' (line 23)
                s_928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 49), 's', False)
                # Processing the call keyword arguments (line 23)
                kwargs_929 = {}
                # Getting the type of 'Vector3f_str' (line 23)
                Vector3f_str_927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 36), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 23)
                Vector3f_str_call_result_930 = invoke(stypy.reporting.localization.Localization(__file__, 23, 36), Vector3f_str_927, *[s_928], **kwargs_929)
                
                # Obtaining the member 'clamped' of a type (line 23)
                clamped_931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 36), Vector3f_str_call_result_930, 'clamped')
                # Calling clamped(args, kwargs) (line 23)
                clamped_call_result_935 = invoke(stypy.reporting.localization.Localization(__file__, 23, 36), clamped_931, *[ZERO_932, MAX_933], **kwargs_934)
                
                # Getting the type of 'self' (line 23)
                self_936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'self')
                # Setting the type of the member 'sky_emission' of a type (line 23)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 16), self_936, 'sky_emission', clamped_call_result_935)
                
                # Assigning a Call to a Attribute (line 24):
                
                # Assigning a Call to a Attribute (line 24):
                
                # Call to clamped(...): (line 24)
                # Processing the call arguments (line 24)
                # Getting the type of 'ZERO' (line 24)
                ZERO_942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 65), 'ZERO', False)
                # Getting the type of 'ONE' (line 24)
                ONE_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 71), 'ONE', False)
                # Processing the call keyword arguments (line 24)
                kwargs_944 = {}
                
                # Call to Vector3f_str(...): (line 24)
                # Processing the call arguments (line 24)
                # Getting the type of 'g' (line 24)
                g_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 54), 'g', False)
                # Processing the call keyword arguments (line 24)
                kwargs_939 = {}
                # Getting the type of 'Vector3f_str' (line 24)
                Vector3f_str_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 41), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 24)
                Vector3f_str_call_result_940 = invoke(stypy.reporting.localization.Localization(__file__, 24, 41), Vector3f_str_937, *[g_938], **kwargs_939)
                
                # Obtaining the member 'clamped' of a type (line 24)
                clamped_941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 41), Vector3f_str_call_result_940, 'clamped')
                # Calling clamped(args, kwargs) (line 24)
                clamped_call_result_945 = invoke(stypy.reporting.localization.Localization(__file__, 24, 41), clamped_941, *[ZERO_942, ONE_943], **kwargs_944)
                
                # Getting the type of 'self' (line 24)
                self_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'self')
                # Setting the type of the member 'ground_reflection' of a type (line 24)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), self_946, 'ground_reflection', clamped_call_result_945)
                
                # Assigning a List to a Attribute (line 25):
                
                # Assigning a List to a Attribute (line 25):
                
                # Obtaining an instance of the builtin type 'list' (line 25)
                list_947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 33), 'list')
                # Adding type elements to the builtin type 'list' instance (line 25)
                
                # Getting the type of 'self' (line 25)
                self_948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'self')
                # Setting the type of the member 'triangles' of a type (line 25)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), self_948, 'triangles', list_947)
                
                
                # SSA begins for try-except statement (line 26)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                
                
                # Call to range(...): (line 27)
                # Processing the call arguments (line 27)
                # Getting the type of 'MAX_TRIANGLES' (line 27)
                MAX_TRIANGLES_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 35), 'MAX_TRIANGLES', False)
                # Processing the call keyword arguments (line 27)
                kwargs_951 = {}
                # Getting the type of 'range' (line 27)
                range_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), 'range', False)
                # Calling range(args, kwargs) (line 27)
                range_call_result_952 = invoke(stypy.reporting.localization.Localization(__file__, 27, 29), range_949, *[MAX_TRIANGLES_950], **kwargs_951)
                
                # Assigning a type to the variable 'range_call_result_952' (line 27)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'range_call_result_952', range_call_result_952)
                # Testing if the for loop is going to be iterated (line 27)
                # Testing the type of a for loop iterable (line 27)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 20), range_call_result_952)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 27, 20), range_call_result_952):
                    # Getting the type of the for loop variable (line 27)
                    for_loop_var_953 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 20), range_call_result_952)
                    # Assigning a type to the variable 'i' (line 27)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'i', for_loop_var_953)
                    # SSA begins for a for statement (line 27)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to append(...): (line 28)
                    # Processing the call arguments (line 28)
                    
                    # Call to Triangle(...): (line 28)
                    # Processing the call arguments (line 28)
                    # Getting the type of 'in_stream' (line 28)
                    in_stream_958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 55), 'in_stream', False)
                    # Processing the call keyword arguments (line 28)
                    kwargs_959 = {}
                    # Getting the type of 'Triangle' (line 28)
                    Triangle_957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 46), 'Triangle', False)
                    # Calling Triangle(args, kwargs) (line 28)
                    Triangle_call_result_960 = invoke(stypy.reporting.localization.Localization(__file__, 28, 46), Triangle_957, *[in_stream_958], **kwargs_959)
                    
                    # Processing the call keyword arguments (line 28)
                    kwargs_961 = {}
                    # Getting the type of 'self' (line 28)
                    self_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'self', False)
                    # Obtaining the member 'triangles' of a type (line 28)
                    triangles_955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), self_954, 'triangles')
                    # Obtaining the member 'append' of a type (line 28)
                    append_956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), triangles_955, 'append')
                    # Calling append(args, kwargs) (line 28)
                    append_call_result_962 = invoke(stypy.reporting.localization.Localization(__file__, 28, 24), append_956, *[Triangle_call_result_960], **kwargs_961)
                    
                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                # SSA branch for the except part of a try statement (line 26)
                # SSA branch for the except 'StopIteration' branch of a try statement (line 26)
                module_type_store.open_ssa_branch('except')
                pass
                # SSA join for try-except statement (line 26)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a ListComp to a Attribute (line 31):
                
                # Assigning a ListComp to a Attribute (line 31):
                # Calculating list comprehension
                # Calculating comprehension expression
                # Getting the type of 'self' (line 31)
                self_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 58), 'self')
                # Obtaining the member 'triangles' of a type (line 31)
                triangles_976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 58), self_975, 'triangles')
                comprehension_977 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 33), triangles_976)
                # Assigning a type to the variable 'triangle' (line 31)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'triangle', comprehension_977)
                
                # Evaluating a boolean operation
                
                
                # Call to is_zero(...): (line 31)
                # Processing the call keyword arguments (line 31)
                kwargs_967 = {}
                # Getting the type of 'triangle' (line 31)
                triangle_964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 80), 'triangle', False)
                # Obtaining the member 'emitivity' of a type (line 31)
                emitivity_965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 80), triangle_964, 'emitivity')
                # Obtaining the member 'is_zero' of a type (line 31)
                is_zero_966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 80), emitivity_965, 'is_zero')
                # Calling is_zero(args, kwargs) (line 31)
                is_zero_call_result_968 = invoke(stypy.reporting.localization.Localization(__file__, 31, 80), is_zero_966, *[], **kwargs_967)
                
                # Applying the 'not' unary operator (line 31)
                result_not__969 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 76), 'not', is_zero_call_result_968)
                
                
                # Getting the type of 'triangle' (line 31)
                triangle_970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 113), 'triangle')
                # Obtaining the member 'area' of a type (line 31)
                area_971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 113), triangle_970, 'area')
                float_972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 129), 'float')
                # Applying the binary operator '>' (line 31)
                result_gt_973 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 113), '>', area_971, float_972)
                
                # Applying the binary operator 'and' (line 31)
                result_and_keyword_974 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 76), 'and', result_not__969, result_gt_973)
                
                # Getting the type of 'triangle' (line 31)
                triangle_963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'triangle')
                list_978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 33), 'list')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 33), list_978, triangle_963)
                # Getting the type of 'self' (line 31)
                self_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'self')
                # Setting the type of the member 'emitters' of a type (line 31)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), self_979, 'emitters', list_978)
                
                # Assigning a Call to a Attribute (line 32):
                
                # Assigning a Call to a Attribute (line 32):
                
                # Call to SpatialIndex(...): (line 32)
                # Processing the call arguments (line 32)
                # Getting the type of 'eye_position' (line 32)
                eye_position_981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 42), 'eye_position', False)
                # Getting the type of 'None' (line 32)
                None_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 56), 'None', False)
                # Getting the type of 'self' (line 32)
                self_983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 62), 'self', False)
                # Obtaining the member 'triangles' of a type (line 32)
                triangles_984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 62), self_983, 'triangles')
                # Processing the call keyword arguments (line 32)
                kwargs_985 = {}
                # Getting the type of 'SpatialIndex' (line 32)
                SpatialIndex_980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 29), 'SpatialIndex', False)
                # Calling SpatialIndex(args, kwargs) (line 32)
                SpatialIndex_call_result_986 = invoke(stypy.reporting.localization.Localization(__file__, 32, 29), SpatialIndex_980, *[eye_position_981, None_982, triangles_984], **kwargs_985)
                
                # Getting the type of 'self' (line 32)
                self_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'self')
                # Setting the type of the member 'index' of a type (line 32)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), self_987, 'index', SpatialIndex_call_result_986)
                # SSA join for if statement (line 21)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
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
        defaults = []
        # Create a new context for function 'get_intersection'
        module_type_store = module_type_store.open_function_context('get_intersection', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Scene.get_intersection.__dict__.__setitem__('stypy_localization', localization)
        Scene.get_intersection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Scene.get_intersection.__dict__.__setitem__('stypy_type_store', module_type_store)
        Scene.get_intersection.__dict__.__setitem__('stypy_function_name', 'Scene.get_intersection')
        Scene.get_intersection.__dict__.__setitem__('stypy_param_names_list', ['ray_origin', 'ray_direction', 'last_hit'])
        Scene.get_intersection.__dict__.__setitem__('stypy_varargs_param_name', None)
        Scene.get_intersection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Scene.get_intersection.__dict__.__setitem__('stypy_call_defaults', defaults)
        Scene.get_intersection.__dict__.__setitem__('stypy_call_varargs', varargs)
        Scene.get_intersection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Scene.get_intersection.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Scene.get_intersection', ['ray_origin', 'ray_direction', 'last_hit'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_intersection', localization, ['ray_origin', 'ray_direction', 'last_hit'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_intersection(...)' code ##################

        
        # Call to get_intersection(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'ray_origin' (line 37)
        ray_origin_991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 43), 'ray_origin', False)
        # Getting the type of 'ray_direction' (line 37)
        ray_direction_992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 55), 'ray_direction', False)
        # Getting the type of 'last_hit' (line 37)
        last_hit_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 70), 'last_hit', False)
        # Processing the call keyword arguments (line 37)
        kwargs_994 = {}
        # Getting the type of 'self' (line 37)
        self_988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'self', False)
        # Obtaining the member 'index' of a type (line 37)
        index_989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), self_988, 'index')
        # Obtaining the member 'get_intersection' of a type (line 37)
        get_intersection_990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), index_989, 'get_intersection')
        # Calling get_intersection(args, kwargs) (line 37)
        get_intersection_call_result_995 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), get_intersection_990, *[ray_origin_991, ray_direction_992, last_hit_993], **kwargs_994)
        
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'stypy_return_type', get_intersection_call_result_995)
        
        # ################# End of 'get_intersection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_intersection' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_996)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_intersection'
        return stypy_return_type_996


    @norecursion
    def get_emitter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_emitter'
        module_type_store = module_type_store.open_function_context('get_emitter', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Scene.get_emitter.__dict__.__setitem__('stypy_localization', localization)
        Scene.get_emitter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Scene.get_emitter.__dict__.__setitem__('stypy_type_store', module_type_store)
        Scene.get_emitter.__dict__.__setitem__('stypy_function_name', 'Scene.get_emitter')
        Scene.get_emitter.__dict__.__setitem__('stypy_param_names_list', [])
        Scene.get_emitter.__dict__.__setitem__('stypy_varargs_param_name', None)
        Scene.get_emitter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Scene.get_emitter.__dict__.__setitem__('stypy_call_defaults', defaults)
        Scene.get_emitter.__dict__.__setitem__('stypy_call_varargs', varargs)
        Scene.get_emitter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Scene.get_emitter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Scene.get_emitter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_emitter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_emitter(...)' code ##################

        
        # Assigning a IfExp to a Name (line 40):
        
        # Assigning a IfExp to a Name (line 40):
        
        
        
        # Call to len(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'self', False)
        # Obtaining the member 'emitters' of a type (line 40)
        emitters_999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 30), self_998, 'emitters')
        # Processing the call keyword arguments (line 40)
        kwargs_1000 = {}
        # Getting the type of 'len' (line 40)
        len_997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'len', False)
        # Calling len(args, kwargs) (line 40)
        len_call_result_1001 = invoke(stypy.reporting.localization.Localization(__file__, 40, 26), len_997, *[emitters_999], **kwargs_1000)
        
        int_1002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 48), 'int')
        # Applying the binary operator '==' (line 40)
        result_eq_1003 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 26), '==', len_call_result_1001, int_1002)
        
        # Testing the type of an if expression (line 40)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 18), result_eq_1003)
        # SSA begins for if expression (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'None' (line 40)
        None_1004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'None')
        # SSA branch for the else part of an if expression (line 40)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to choice(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 62), 'self', False)
        # Obtaining the member 'emitters' of a type (line 40)
        emitters_1007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 62), self_1006, 'emitters')
        # Processing the call keyword arguments (line 40)
        kwargs_1008 = {}
        # Getting the type of 'choice' (line 40)
        choice_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 55), 'choice', False)
        # Calling choice(args, kwargs) (line 40)
        choice_call_result_1009 = invoke(stypy.reporting.localization.Localization(__file__, 40, 55), choice_1005, *[emitters_1007], **kwargs_1008)
        
        # SSA join for if expression (line 40)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1010 = union_type.UnionType.add(None_1004, choice_call_result_1009)
        
        # Assigning a type to the variable 'emitter' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'emitter', if_exp_1010)
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_1011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        
        # Getting the type of 'emitter' (line 41)
        emitter_1012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 46), 'emitter')
        # Testing the type of an if expression (line 41)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 16), emitter_1012)
        # SSA begins for if expression (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to get_sample_point(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_1015 = {}
        # Getting the type of 'emitter' (line 41)
        emitter_1013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'emitter', False)
        # Obtaining the member 'get_sample_point' of a type (line 41)
        get_sample_point_1014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 16), emitter_1013, 'get_sample_point')
        # Calling get_sample_point(args, kwargs) (line 41)
        get_sample_point_call_result_1016 = invoke(stypy.reporting.localization.Localization(__file__, 41, 16), get_sample_point_1014, *[], **kwargs_1015)
        
        # SSA branch for the else part of an if expression (line 41)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'ZERO' (line 41)
        ZERO_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 59), 'ZERO')
        # SSA join for if expression (line 41)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1018 = union_type.UnionType.add(get_sample_point_call_result_1016, ZERO_1017)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 15), tuple_1011, if_exp_1018)
        # Adding element type (line 41)
        # Getting the type of 'emitter' (line 41)
        emitter_1019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 66), 'emitter')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 15), tuple_1011, emitter_1019)
        
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', tuple_1011)
        
        # ################# End of 'get_emitter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_emitter' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_1020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1020)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_emitter'
        return stypy_return_type_1020


    @norecursion
    def emitters_count(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'emitters_count'
        module_type_store = module_type_store.open_function_context('emitters_count', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Scene.emitters_count.__dict__.__setitem__('stypy_localization', localization)
        Scene.emitters_count.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Scene.emitters_count.__dict__.__setitem__('stypy_type_store', module_type_store)
        Scene.emitters_count.__dict__.__setitem__('stypy_function_name', 'Scene.emitters_count')
        Scene.emitters_count.__dict__.__setitem__('stypy_param_names_list', [])
        Scene.emitters_count.__dict__.__setitem__('stypy_varargs_param_name', None)
        Scene.emitters_count.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Scene.emitters_count.__dict__.__setitem__('stypy_call_defaults', defaults)
        Scene.emitters_count.__dict__.__setitem__('stypy_call_varargs', varargs)
        Scene.emitters_count.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Scene.emitters_count.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Scene.emitters_count', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'emitters_count', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'emitters_count(...)' code ##################

        
        # Call to len(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'self' (line 44)
        self_1022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'self', False)
        # Obtaining the member 'emitters' of a type (line 44)
        emitters_1023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 19), self_1022, 'emitters')
        # Processing the call keyword arguments (line 44)
        kwargs_1024 = {}
        # Getting the type of 'len' (line 44)
        len_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'len', False)
        # Calling len(args, kwargs) (line 44)
        len_call_result_1025 = invoke(stypy.reporting.localization.Localization(__file__, 44, 15), len_1021, *[emitters_1023], **kwargs_1024)
        
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', len_call_result_1025)
        
        # ################# End of 'emitters_count(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'emitters_count' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_1026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1026)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'emitters_count'
        return stypy_return_type_1026


    @norecursion
    def get_default_emission(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_default_emission'
        module_type_store = module_type_store.open_function_context('get_default_emission', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Scene.get_default_emission.__dict__.__setitem__('stypy_localization', localization)
        Scene.get_default_emission.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Scene.get_default_emission.__dict__.__setitem__('stypy_type_store', module_type_store)
        Scene.get_default_emission.__dict__.__setitem__('stypy_function_name', 'Scene.get_default_emission')
        Scene.get_default_emission.__dict__.__setitem__('stypy_param_names_list', ['back_direction'])
        Scene.get_default_emission.__dict__.__setitem__('stypy_varargs_param_name', None)
        Scene.get_default_emission.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Scene.get_default_emission.__dict__.__setitem__('stypy_call_defaults', defaults)
        Scene.get_default_emission.__dict__.__setitem__('stypy_call_varargs', varargs)
        Scene.get_default_emission.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Scene.get_default_emission.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Scene.get_default_emission', ['back_direction'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_default_emission', localization, ['back_direction'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_default_emission(...)' code ##################

        
        
        # Getting the type of 'back_direction' (line 47)
        back_direction_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 36), 'back_direction')
        # Obtaining the member 'y' of a type (line 47)
        y_1028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 36), back_direction_1027, 'y')
        float_1029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 55), 'float')
        # Applying the binary operator '<' (line 47)
        result_lt_1030 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 36), '<', y_1028, float_1029)
        
        # Testing the type of an if expression (line 47)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 15), result_lt_1030)
        # SSA begins for if expression (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'self' (line 47)
        self_1031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'self')
        # Obtaining the member 'sky_emission' of a type (line 47)
        sky_emission_1032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 15), self_1031, 'sky_emission')
        # SSA branch for the else part of an if expression (line 47)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to mul(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'self' (line 47)
        self_1036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 86), 'self', False)
        # Obtaining the member 'ground_reflection' of a type (line 47)
        ground_reflection_1037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 86), self_1036, 'ground_reflection')
        # Processing the call keyword arguments (line 47)
        kwargs_1038 = {}
        # Getting the type of 'self' (line 47)
        self_1033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 64), 'self', False)
        # Obtaining the member 'sky_emission' of a type (line 47)
        sky_emission_1034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 64), self_1033, 'sky_emission')
        # Obtaining the member 'mul' of a type (line 47)
        mul_1035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 64), sky_emission_1034, 'mul')
        # Calling mul(args, kwargs) (line 47)
        mul_call_result_1039 = invoke(stypy.reporting.localization.Localization(__file__, 47, 64), mul_1035, *[ground_reflection_1037], **kwargs_1038)
        
        # SSA join for if expression (line 47)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1040 = union_type.UnionType.add(sky_emission_1032, mul_call_result_1039)
        
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', if_exp_1040)
        
        # ################# End of 'get_default_emission(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_default_emission' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_1041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1041)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_default_emission'
        return stypy_return_type_1041


# Assigning a type to the variable 'Scene' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'Scene', Scene)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
