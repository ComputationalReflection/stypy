
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
import_926 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'spatialindex')

if (type(import_926) is not StypyTypeError):

    if (import_926 != 'pyd_module'):
        __import__(import_926)
        sys_modules_927 = sys.modules[import_926]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'spatialindex', sys_modules_927.module_type_store, module_type_store, ['SpatialIndex'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_927, sys_modules_927.module_type_store, module_type_store)
    else:
        from spatialindex import SpatialIndex

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'spatialindex', None, module_type_store, ['SpatialIndex'], [SpatialIndex])

else:
    # Assigning a type to the variable 'spatialindex' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'spatialindex', import_926)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from triangle import Triangle' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_928 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'triangle')

if (type(import_928) is not StypyTypeError):

    if (import_928 != 'pyd_module'):
        __import__(import_928)
        sys_modules_929 = sys.modules[import_928]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'triangle', sys_modules_929.module_type_store, module_type_store, ['Triangle'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_929, sys_modules_929.module_type_store, module_type_store)
    else:
        from triangle import Triangle

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'triangle', None, module_type_store, ['Triangle'], [Triangle])

else:
    # Assigning a type to the variable 'triangle' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'triangle', import_928)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from vector3f import Vector3f_str, ZERO, ONE, MAX' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_930 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'vector3f')

if (type(import_930) is not StypyTypeError):

    if (import_930 != 'pyd_module'):
        __import__(import_930)
        sys_modules_931 = sys.modules[import_930]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'vector3f', sys_modules_931.module_type_store, module_type_store, ['Vector3f_str', 'ZERO', 'ONE', 'MAX'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_931, sys_modules_931.module_type_store, module_type_store)
    else:
        from vector3f import Vector3f_str, ZERO, ONE, MAX

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'vector3f', None, module_type_store, ['Vector3f_str', 'ZERO', 'ONE', 'MAX'], [Vector3f_str, ZERO, ONE, MAX])

else:
    # Assigning a type to the variable 'vector3f' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'vector3f', import_930)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import re' statement (line 12)
import re

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 're', re, module_type_store)


# Assigning a Call to a Name (line 13):

# Assigning a Call to a Name (line 13):

# Call to compile(...): (line 13)
# Processing the call arguments (line 13)
str_934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'str', '(\\(.+\\))\\s*(\\(.+\\))')
# Processing the call keyword arguments (line 13)
kwargs_935 = {}
# Getting the type of 're' (line 13)
re_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 9), 're', False)
# Obtaining the member 'compile' of a type (line 13)
compile_933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 9), re_932, 'compile')
# Calling compile(args, kwargs) (line 13)
compile_call_result_936 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), compile_933, *[str_934], **kwargs_935)

# Assigning a type to the variable 'SEARCH' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'SEARCH', compile_call_result_936)

# Assigning a Num to a Name (line 15):

# Assigning a Num to a Name (line 15):
int_937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'int')
# Assigning a type to the variable 'MAX_TRIANGLES' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'MAX_TRIANGLES', int_937)
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
        in_stream_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'in_stream')
        # Testing if the for loop is going to be iterated (line 20)
        # Testing the type of a for loop iterable (line 20)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 20, 8), in_stream_938)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 20, 8), in_stream_938):
            # Getting the type of the for loop variable (line 20)
            for_loop_var_939 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 20, 8), in_stream_938)
            # Assigning a type to the variable 'line' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'line', for_loop_var_939)
            # SSA begins for a for statement (line 20)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to isspace(...): (line 21)
            # Processing the call keyword arguments (line 21)
            kwargs_942 = {}
            # Getting the type of 'line' (line 21)
            line_940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'line', False)
            # Obtaining the member 'isspace' of a type (line 21)
            isspace_941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 19), line_940, 'isspace')
            # Calling isspace(args, kwargs) (line 21)
            isspace_call_result_943 = invoke(stypy.reporting.localization.Localization(__file__, 21, 19), isspace_941, *[], **kwargs_942)
            
            # Applying the 'not' unary operator (line 21)
            result_not__944 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 15), 'not', isspace_call_result_943)
            
            # Testing if the type of an if condition is none (line 21)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 21, 12), result_not__944):
                pass
            else:
                
                # Testing the type of an if condition (line 21)
                if_condition_945 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 12), result_not__944)
                # Assigning a type to the variable 'if_condition_945' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'if_condition_945', if_condition_945)
                # SSA begins for if statement (line 21)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 22):
                
                # Assigning a Call to a Name:
                
                # Call to groups(...): (line 22)
                # Processing the call keyword arguments (line 22)
                kwargs_952 = {}
                
                # Call to search(...): (line 22)
                # Processing the call arguments (line 22)
                # Getting the type of 'line' (line 22)
                line_948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 37), 'line', False)
                # Processing the call keyword arguments (line 22)
                kwargs_949 = {}
                # Getting the type of 'SEARCH' (line 22)
                SEARCH_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'SEARCH', False)
                # Obtaining the member 'search' of a type (line 22)
                search_947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 23), SEARCH_946, 'search')
                # Calling search(args, kwargs) (line 22)
                search_call_result_950 = invoke(stypy.reporting.localization.Localization(__file__, 22, 23), search_947, *[line_948], **kwargs_949)
                
                # Obtaining the member 'groups' of a type (line 22)
                groups_951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 23), search_call_result_950, 'groups')
                # Calling groups(args, kwargs) (line 22)
                groups_call_result_953 = invoke(stypy.reporting.localization.Localization(__file__, 22, 23), groups_951, *[], **kwargs_952)
                
                # Assigning a type to the variable 'call_assignment_923' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_923', groups_call_result_953)
                
                # Assigning a Call to a Name (line 22):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
                # Processing the call keyword arguments
                kwargs_957 = {}
                # Getting the type of 'call_assignment_923' (line 22)
                call_assignment_923_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_923', False)
                # Obtaining the member '__getitem__' of a type (line 22)
                getitem___955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), call_assignment_923_954, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_958 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___955, *[int_956], **kwargs_957)
                
                # Assigning a type to the variable 'call_assignment_924' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_924', getitem___call_result_958)
                
                # Assigning a Name to a Name (line 22):
                # Getting the type of 'call_assignment_924' (line 22)
                call_assignment_924_959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_924')
                # Assigning a type to the variable 's' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 's', call_assignment_924_959)
                
                # Assigning a Call to a Name (line 22):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'int')
                # Processing the call keyword arguments
                kwargs_963 = {}
                # Getting the type of 'call_assignment_923' (line 22)
                call_assignment_923_960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_923', False)
                # Obtaining the member '__getitem__' of a type (line 22)
                getitem___961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), call_assignment_923_960, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_964 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___961, *[int_962], **kwargs_963)
                
                # Assigning a type to the variable 'call_assignment_925' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_925', getitem___call_result_964)
                
                # Assigning a Name to a Name (line 22):
                # Getting the type of 'call_assignment_925' (line 22)
                call_assignment_925_965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'call_assignment_925')
                # Assigning a type to the variable 'g' (line 22)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'g', call_assignment_925_965)
                
                # Assigning a Call to a Attribute (line 23):
                
                # Assigning a Call to a Attribute (line 23):
                
                # Call to clamped(...): (line 23)
                # Processing the call arguments (line 23)
                # Getting the type of 'ZERO' (line 23)
                ZERO_971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 60), 'ZERO', False)
                # Getting the type of 'MAX' (line 23)
                MAX_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 66), 'MAX', False)
                # Processing the call keyword arguments (line 23)
                kwargs_973 = {}
                
                # Call to Vector3f_str(...): (line 23)
                # Processing the call arguments (line 23)
                # Getting the type of 's' (line 23)
                s_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 49), 's', False)
                # Processing the call keyword arguments (line 23)
                kwargs_968 = {}
                # Getting the type of 'Vector3f_str' (line 23)
                Vector3f_str_966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 36), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 23)
                Vector3f_str_call_result_969 = invoke(stypy.reporting.localization.Localization(__file__, 23, 36), Vector3f_str_966, *[s_967], **kwargs_968)
                
                # Obtaining the member 'clamped' of a type (line 23)
                clamped_970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 36), Vector3f_str_call_result_969, 'clamped')
                # Calling clamped(args, kwargs) (line 23)
                clamped_call_result_974 = invoke(stypy.reporting.localization.Localization(__file__, 23, 36), clamped_970, *[ZERO_971, MAX_972], **kwargs_973)
                
                # Getting the type of 'self' (line 23)
                self_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'self')
                # Setting the type of the member 'sky_emission' of a type (line 23)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 16), self_975, 'sky_emission', clamped_call_result_974)
                
                # Assigning a Call to a Attribute (line 24):
                
                # Assigning a Call to a Attribute (line 24):
                
                # Call to clamped(...): (line 24)
                # Processing the call arguments (line 24)
                # Getting the type of 'ZERO' (line 24)
                ZERO_981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 65), 'ZERO', False)
                # Getting the type of 'ONE' (line 24)
                ONE_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 71), 'ONE', False)
                # Processing the call keyword arguments (line 24)
                kwargs_983 = {}
                
                # Call to Vector3f_str(...): (line 24)
                # Processing the call arguments (line 24)
                # Getting the type of 'g' (line 24)
                g_977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 54), 'g', False)
                # Processing the call keyword arguments (line 24)
                kwargs_978 = {}
                # Getting the type of 'Vector3f_str' (line 24)
                Vector3f_str_976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 41), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 24)
                Vector3f_str_call_result_979 = invoke(stypy.reporting.localization.Localization(__file__, 24, 41), Vector3f_str_976, *[g_977], **kwargs_978)
                
                # Obtaining the member 'clamped' of a type (line 24)
                clamped_980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 41), Vector3f_str_call_result_979, 'clamped')
                # Calling clamped(args, kwargs) (line 24)
                clamped_call_result_984 = invoke(stypy.reporting.localization.Localization(__file__, 24, 41), clamped_980, *[ZERO_981, ONE_982], **kwargs_983)
                
                # Getting the type of 'self' (line 24)
                self_985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'self')
                # Setting the type of the member 'ground_reflection' of a type (line 24)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), self_985, 'ground_reflection', clamped_call_result_984)
                
                # Assigning a List to a Attribute (line 25):
                
                # Assigning a List to a Attribute (line 25):
                
                # Obtaining an instance of the builtin type 'list' (line 25)
                list_986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 33), 'list')
                # Adding type elements to the builtin type 'list' instance (line 25)
                
                # Getting the type of 'self' (line 25)
                self_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'self')
                # Setting the type of the member 'triangles' of a type (line 25)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), self_987, 'triangles', list_986)
                
                
                # SSA begins for try-except statement (line 26)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
                
                
                # Call to range(...): (line 27)
                # Processing the call arguments (line 27)
                # Getting the type of 'MAX_TRIANGLES' (line 27)
                MAX_TRIANGLES_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 35), 'MAX_TRIANGLES', False)
                # Processing the call keyword arguments (line 27)
                kwargs_990 = {}
                # Getting the type of 'range' (line 27)
                range_988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), 'range', False)
                # Calling range(args, kwargs) (line 27)
                range_call_result_991 = invoke(stypy.reporting.localization.Localization(__file__, 27, 29), range_988, *[MAX_TRIANGLES_989], **kwargs_990)
                
                # Testing if the for loop is going to be iterated (line 27)
                # Testing the type of a for loop iterable (line 27)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 20), range_call_result_991)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 27, 20), range_call_result_991):
                    # Getting the type of the for loop variable (line 27)
                    for_loop_var_992 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 20), range_call_result_991)
                    # Assigning a type to the variable 'i' (line 27)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'i', for_loop_var_992)
                    # SSA begins for a for statement (line 27)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Call to append(...): (line 28)
                    # Processing the call arguments (line 28)
                    
                    # Call to Triangle(...): (line 28)
                    # Processing the call arguments (line 28)
                    # Getting the type of 'in_stream' (line 28)
                    in_stream_997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 55), 'in_stream', False)
                    # Processing the call keyword arguments (line 28)
                    kwargs_998 = {}
                    # Getting the type of 'Triangle' (line 28)
                    Triangle_996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 46), 'Triangle', False)
                    # Calling Triangle(args, kwargs) (line 28)
                    Triangle_call_result_999 = invoke(stypy.reporting.localization.Localization(__file__, 28, 46), Triangle_996, *[in_stream_997], **kwargs_998)
                    
                    # Processing the call keyword arguments (line 28)
                    kwargs_1000 = {}
                    # Getting the type of 'self' (line 28)
                    self_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'self', False)
                    # Obtaining the member 'triangles' of a type (line 28)
                    triangles_994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), self_993, 'triangles')
                    # Obtaining the member 'append' of a type (line 28)
                    append_995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), triangles_994, 'append')
                    # Calling append(args, kwargs) (line 28)
                    append_call_result_1001 = invoke(stypy.reporting.localization.Localization(__file__, 28, 24), append_995, *[Triangle_call_result_999], **kwargs_1000)
                    
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
                self_1014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 58), 'self')
                # Obtaining the member 'triangles' of a type (line 31)
                triangles_1015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 58), self_1014, 'triangles')
                comprehension_1016 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 33), triangles_1015)
                # Assigning a type to the variable 'triangle' (line 31)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'triangle', comprehension_1016)
                
                # Evaluating a boolean operation
                
                
                # Call to is_zero(...): (line 31)
                # Processing the call keyword arguments (line 31)
                kwargs_1006 = {}
                # Getting the type of 'triangle' (line 31)
                triangle_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 80), 'triangle', False)
                # Obtaining the member 'emitivity' of a type (line 31)
                emitivity_1004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 80), triangle_1003, 'emitivity')
                # Obtaining the member 'is_zero' of a type (line 31)
                is_zero_1005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 80), emitivity_1004, 'is_zero')
                # Calling is_zero(args, kwargs) (line 31)
                is_zero_call_result_1007 = invoke(stypy.reporting.localization.Localization(__file__, 31, 80), is_zero_1005, *[], **kwargs_1006)
                
                # Applying the 'not' unary operator (line 31)
                result_not__1008 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 76), 'not', is_zero_call_result_1007)
                
                
                # Getting the type of 'triangle' (line 31)
                triangle_1009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 113), 'triangle')
                # Obtaining the member 'area' of a type (line 31)
                area_1010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 113), triangle_1009, 'area')
                float_1011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 129), 'float')
                # Applying the binary operator '>' (line 31)
                result_gt_1012 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 113), '>', area_1010, float_1011)
                
                # Applying the binary operator 'and' (line 31)
                result_and_keyword_1013 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 76), 'and', result_not__1008, result_gt_1012)
                
                # Getting the type of 'triangle' (line 31)
                triangle_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'triangle')
                list_1017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 33), 'list')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 33), list_1017, triangle_1002)
                # Getting the type of 'self' (line 31)
                self_1018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'self')
                # Setting the type of the member 'emitters' of a type (line 31)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), self_1018, 'emitters', list_1017)
                
                # Assigning a Call to a Attribute (line 32):
                
                # Assigning a Call to a Attribute (line 32):
                
                # Call to SpatialIndex(...): (line 32)
                # Processing the call arguments (line 32)
                # Getting the type of 'eye_position' (line 32)
                eye_position_1020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 42), 'eye_position', False)
                # Getting the type of 'None' (line 32)
                None_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 56), 'None', False)
                # Getting the type of 'self' (line 32)
                self_1022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 62), 'self', False)
                # Obtaining the member 'triangles' of a type (line 32)
                triangles_1023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 62), self_1022, 'triangles')
                # Processing the call keyword arguments (line 32)
                kwargs_1024 = {}
                # Getting the type of 'SpatialIndex' (line 32)
                SpatialIndex_1019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 29), 'SpatialIndex', False)
                # Calling SpatialIndex(args, kwargs) (line 32)
                SpatialIndex_call_result_1025 = invoke(stypy.reporting.localization.Localization(__file__, 32, 29), SpatialIndex_1019, *[eye_position_1020, None_1021, triangles_1023], **kwargs_1024)
                
                # Getting the type of 'self' (line 32)
                self_1026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'self')
                # Setting the type of the member 'index' of a type (line 32)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), self_1026, 'index', SpatialIndex_call_result_1025)
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
        ray_origin_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 43), 'ray_origin', False)
        # Getting the type of 'ray_direction' (line 37)
        ray_direction_1031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 55), 'ray_direction', False)
        # Getting the type of 'last_hit' (line 37)
        last_hit_1032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 70), 'last_hit', False)
        # Processing the call keyword arguments (line 37)
        kwargs_1033 = {}
        # Getting the type of 'self' (line 37)
        self_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'self', False)
        # Obtaining the member 'index' of a type (line 37)
        index_1028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), self_1027, 'index')
        # Obtaining the member 'get_intersection' of a type (line 37)
        get_intersection_1029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), index_1028, 'get_intersection')
        # Calling get_intersection(args, kwargs) (line 37)
        get_intersection_call_result_1034 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), get_intersection_1029, *[ray_origin_1030, ray_direction_1031, last_hit_1032], **kwargs_1033)
        
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'stypy_return_type', get_intersection_call_result_1034)
        
        # ################# End of 'get_intersection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_intersection' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_1035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1035)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_intersection'
        return stypy_return_type_1035


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
        self_1037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 30), 'self', False)
        # Obtaining the member 'emitters' of a type (line 40)
        emitters_1038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 30), self_1037, 'emitters')
        # Processing the call keyword arguments (line 40)
        kwargs_1039 = {}
        # Getting the type of 'len' (line 40)
        len_1036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'len', False)
        # Calling len(args, kwargs) (line 40)
        len_call_result_1040 = invoke(stypy.reporting.localization.Localization(__file__, 40, 26), len_1036, *[emitters_1038], **kwargs_1039)
        
        int_1041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 48), 'int')
        # Applying the binary operator '==' (line 40)
        result_eq_1042 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 26), '==', len_call_result_1040, int_1041)
        
        # Testing the type of an if expression (line 40)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 18), result_eq_1042)
        # SSA begins for if expression (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'None' (line 40)
        None_1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'None')
        # SSA branch for the else part of an if expression (line 40)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to choice(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_1045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 62), 'self', False)
        # Obtaining the member 'emitters' of a type (line 40)
        emitters_1046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 62), self_1045, 'emitters')
        # Processing the call keyword arguments (line 40)
        kwargs_1047 = {}
        # Getting the type of 'choice' (line 40)
        choice_1044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 55), 'choice', False)
        # Calling choice(args, kwargs) (line 40)
        choice_call_result_1048 = invoke(stypy.reporting.localization.Localization(__file__, 40, 55), choice_1044, *[emitters_1046], **kwargs_1047)
        
        # SSA join for if expression (line 40)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1049 = union_type.UnionType.add(None_1043, choice_call_result_1048)
        
        # Assigning a type to the variable 'emitter' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'emitter', if_exp_1049)
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_1050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        
        # Getting the type of 'emitter' (line 41)
        emitter_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 46), 'emitter')
        # Testing the type of an if expression (line 41)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 16), emitter_1051)
        # SSA begins for if expression (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to get_sample_point(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_1054 = {}
        # Getting the type of 'emitter' (line 41)
        emitter_1052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'emitter', False)
        # Obtaining the member 'get_sample_point' of a type (line 41)
        get_sample_point_1053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 16), emitter_1052, 'get_sample_point')
        # Calling get_sample_point(args, kwargs) (line 41)
        get_sample_point_call_result_1055 = invoke(stypy.reporting.localization.Localization(__file__, 41, 16), get_sample_point_1053, *[], **kwargs_1054)
        
        # SSA branch for the else part of an if expression (line 41)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'ZERO' (line 41)
        ZERO_1056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 59), 'ZERO')
        # SSA join for if expression (line 41)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1057 = union_type.UnionType.add(get_sample_point_call_result_1055, ZERO_1056)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 15), tuple_1050, if_exp_1057)
        # Adding element type (line 41)
        # Getting the type of 'emitter' (line 41)
        emitter_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 66), 'emitter')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 15), tuple_1050, emitter_1058)
        
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', tuple_1050)
        
        # ################# End of 'get_emitter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_emitter' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_1059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1059)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_emitter'
        return stypy_return_type_1059


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
        self_1061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'self', False)
        # Obtaining the member 'emitters' of a type (line 44)
        emitters_1062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 19), self_1061, 'emitters')
        # Processing the call keyword arguments (line 44)
        kwargs_1063 = {}
        # Getting the type of 'len' (line 44)
        len_1060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'len', False)
        # Calling len(args, kwargs) (line 44)
        len_call_result_1064 = invoke(stypy.reporting.localization.Localization(__file__, 44, 15), len_1060, *[emitters_1062], **kwargs_1063)
        
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', len_call_result_1064)
        
        # ################# End of 'emitters_count(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'emitters_count' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1065)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'emitters_count'
        return stypy_return_type_1065


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
        back_direction_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 36), 'back_direction')
        # Obtaining the member 'y' of a type (line 47)
        y_1067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 36), back_direction_1066, 'y')
        float_1068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 55), 'float')
        # Applying the binary operator '<' (line 47)
        result_lt_1069 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 36), '<', y_1067, float_1068)
        
        # Testing the type of an if expression (line 47)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 15), result_lt_1069)
        # SSA begins for if expression (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        # Getting the type of 'self' (line 47)
        self_1070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'self')
        # Obtaining the member 'sky_emission' of a type (line 47)
        sky_emission_1071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 15), self_1070, 'sky_emission')
        # SSA branch for the else part of an if expression (line 47)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to mul(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'self' (line 47)
        self_1075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 86), 'self', False)
        # Obtaining the member 'ground_reflection' of a type (line 47)
        ground_reflection_1076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 86), self_1075, 'ground_reflection')
        # Processing the call keyword arguments (line 47)
        kwargs_1077 = {}
        # Getting the type of 'self' (line 47)
        self_1072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 64), 'self', False)
        # Obtaining the member 'sky_emission' of a type (line 47)
        sky_emission_1073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 64), self_1072, 'sky_emission')
        # Obtaining the member 'mul' of a type (line 47)
        mul_1074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 64), sky_emission_1073, 'mul')
        # Calling mul(args, kwargs) (line 47)
        mul_call_result_1078 = invoke(stypy.reporting.localization.Localization(__file__, 47, 64), mul_1074, *[ground_reflection_1076], **kwargs_1077)
        
        # SSA join for if expression (line 47)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_1079 = union_type.UnionType.add(sky_emission_1071, mul_call_result_1078)
        
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', if_exp_1079)
        
        # ################# End of 'get_default_emission(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_default_emission' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1080)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_default_emission'
        return stypy_return_type_1080


# Assigning a type to the variable 'Scene' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'Scene', Scene)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
