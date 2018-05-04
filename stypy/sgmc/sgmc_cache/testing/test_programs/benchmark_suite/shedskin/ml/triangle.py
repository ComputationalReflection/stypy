
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #  MiniLight Python : minimal global illumination renderer
2: #
3: #  Copyright (c) 2007-2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
4: #  http://www.hxa7241.org/
5: 
6: 
7: from math import sqrt
8: from random import random
9: from vector3f import Vector3f_str, ZERO, ONE, MAX
10: 
11: import re
12: SEARCH = re.compile('(\(.+\))\s*(\(.+\))\s*(\(.+\))\s*(\(.+\))\s*(\(.+\))')
13: 
14: TOLERANCE = 1.0 / 1024.0
15: 
16: class Triangle(object):
17: 
18:     def __init__(self, in_stream):
19:         for line in in_stream:
20:             if not line.isspace():
21:                 v0, v1, v2, r, e = SEARCH.search(line).groups()
22:                 self.vertexs = [Vector3f_str(v0), Vector3f_str(v1), Vector3f_str(v2)]
23:                 self.edge0 = Vector3f_str(v1) - Vector3f_str(v0)
24:                 self.edge3 = Vector3f_str(v2) - Vector3f_str(v0)
25:                 self.reflectivity = Vector3f_str(r).clamped(ZERO, ONE)
26:                 self.emitivity = Vector3f_str(e).clamped(ZERO, MAX)
27:                 edge1 = Vector3f_str(v2) - Vector3f_str(v1)
28:                 self.tangent = self.edge0.unitize()
29:                 self.normal = self.tangent.cross(edge1).unitize()
30:                 pa2 = self.edge0.cross(edge1)
31:                 self.area = sqrt(pa2.dot(pa2)) * 0.5
32:                 return
33:         raise StopIteration
34: 
35:     def get_bound(self):
36:         v2 = self.vertexs[2]
37:         bound = [v2.x, v2.y, v2.z, v2.x, v2.y, v2.z]
38:         for j in range(3):
39:             v0 = self.vertexs[0][j]
40:             v1 = self.vertexs[1][j]
41:             if v0 < v1:
42:                 if v0 < bound[j]:
43:                     bound[j] = v0
44:                 if v1 > bound[j + 3]:
45:                     bound[j + 3] = v1
46:             else:
47:                 if v1 < bound[j]:
48:                     bound[j] = v1
49:                 if v0 > bound[j + 3]:
50:                     bound[j + 3] = v0
51:             bound[j] -= (abs(bound[j]) + 1.0) * TOLERANCE
52:             bound[j + 3] += (abs(bound[j + 3]) + 1.0) * TOLERANCE
53:         return bound
54: 
55:     def get_intersection(self, ray_origin, ray_direction):
56:         e1x = self.edge0.x; e1y = self.edge0.y; e1z = self.edge0.z
57:         e2x = self.edge3.x; e2y = self.edge3.y; e2z = self.edge3.z
58:         pvx = ray_direction.y * e2z - ray_direction.z * e2y
59:         pvy = ray_direction.z * e2x - ray_direction.x * e2z
60:         pvz = ray_direction.x * e2y - ray_direction.y * e2x
61:         det = e1x * pvx + e1y * pvy + e1z * pvz
62:         if -0.000001 < det < 0.000001:
63:             return -1.0
64:         inv_det = 1.0 / det
65:         v0 = self.vertexs[0]
66:         tvx = ray_origin.x - v0.x
67:         tvy = ray_origin.y - v0.y
68:         tvz = ray_origin.z - v0.z
69:         u = (tvx * pvx + tvy * pvy + tvz * pvz) * inv_det
70:         if u < 0.0: 
71:             return -1.0
72:         elif u > 1.0: 
73:             return -1.0
74:         qvx = tvy * e1z - tvz * e1y
75:         qvy = tvz * e1x - tvx * e1z
76:         qvz = tvx * e1y - tvy * e1x
77:         v = (ray_direction.x * qvx + ray_direction.y * qvy + ray_direction.z * qvz) * inv_det
78:         if v < 0.0:
79:             return -1.0
80:         elif u + v > 1.0:
81:             return -1.0
82:         t = (e2x * qvx + e2y * qvy + e2z * qvz) * inv_det
83:         if t < 0.0:
84:             return -1.0
85:         return t
86: 
87:     def get_sample_point(self):
88:         sqr1 = sqrt(random())
89:         r2 = random()
90:         a = 1.0 - sqr1
91:         b = (1.0 - r2) * sqr1
92:         return self.edge0 * a + self.edge3 * b + self.vertexs[0]
93: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from math import sqrt' statement (line 7)
try:
    from math import sqrt

except:
    sqrt = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'math', None, module_type_store, ['sqrt'], [sqrt])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from random import random' statement (line 8)
try:
    from random import random

except:
    random = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'random', None, module_type_store, ['random'], [random])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from vector3f import Vector3f_str, ZERO, ONE, MAX' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')
import_1840 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'vector3f')

if (type(import_1840) is not StypyTypeError):

    if (import_1840 != 'pyd_module'):
        __import__(import_1840)
        sys_modules_1841 = sys.modules[import_1840]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'vector3f', sys_modules_1841.module_type_store, module_type_store, ['Vector3f_str', 'ZERO', 'ONE', 'MAX'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_1841, sys_modules_1841.module_type_store, module_type_store)
    else:
        from vector3f import Vector3f_str, ZERO, ONE, MAX

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'vector3f', None, module_type_store, ['Vector3f_str', 'ZERO', 'ONE', 'MAX'], [Vector3f_str, ZERO, ONE, MAX])

else:
    # Assigning a type to the variable 'vector3f' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'vector3f', import_1840)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import re' statement (line 11)
import re

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 're', re, module_type_store)


# Assigning a Call to a Name (line 12):

# Assigning a Call to a Name (line 12):

# Call to compile(...): (line 12)
# Processing the call arguments (line 12)
str_1844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 20), 'str', '(\\(.+\\))\\s*(\\(.+\\))\\s*(\\(.+\\))\\s*(\\(.+\\))\\s*(\\(.+\\))')
# Processing the call keyword arguments (line 12)
kwargs_1845 = {}
# Getting the type of 're' (line 12)
re_1842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 're', False)
# Obtaining the member 'compile' of a type (line 12)
compile_1843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 9), re_1842, 'compile')
# Calling compile(args, kwargs) (line 12)
compile_call_result_1846 = invoke(stypy.reporting.localization.Localization(__file__, 12, 9), compile_1843, *[str_1844], **kwargs_1845)

# Assigning a type to the variable 'SEARCH' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'SEARCH', compile_call_result_1846)

# Assigning a BinOp to a Name (line 14):

# Assigning a BinOp to a Name (line 14):
float_1847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'float')
float_1848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'float')
# Applying the binary operator 'div' (line 14)
result_div_1849 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 12), 'div', float_1847, float_1848)

# Assigning a type to the variable 'TOLERANCE' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'TOLERANCE', result_div_1849)
# Declaration of the 'Triangle' class

class Triangle(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Triangle.__init__', ['in_stream'], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'in_stream' (line 19)
        in_stream_1850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'in_stream')
        # Assigning a type to the variable 'in_stream_1850' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'in_stream_1850', in_stream_1850)
        # Testing if the for loop is going to be iterated (line 19)
        # Testing the type of a for loop iterable (line 19)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 8), in_stream_1850)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 19, 8), in_stream_1850):
            # Getting the type of the for loop variable (line 19)
            for_loop_var_1851 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 8), in_stream_1850)
            # Assigning a type to the variable 'line' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'line', for_loop_var_1851)
            # SSA begins for a for statement (line 19)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to isspace(...): (line 20)
            # Processing the call keyword arguments (line 20)
            kwargs_1854 = {}
            # Getting the type of 'line' (line 20)
            line_1852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'line', False)
            # Obtaining the member 'isspace' of a type (line 20)
            isspace_1853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 19), line_1852, 'isspace')
            # Calling isspace(args, kwargs) (line 20)
            isspace_call_result_1855 = invoke(stypy.reporting.localization.Localization(__file__, 20, 19), isspace_1853, *[], **kwargs_1854)
            
            # Applying the 'not' unary operator (line 20)
            result_not__1856 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 15), 'not', isspace_call_result_1855)
            
            # Testing if the type of an if condition is none (line 20)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 20, 12), result_not__1856):
                pass
            else:
                
                # Testing the type of an if condition (line 20)
                if_condition_1857 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 12), result_not__1856)
                # Assigning a type to the variable 'if_condition_1857' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'if_condition_1857', if_condition_1857)
                # SSA begins for if statement (line 20)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 21):
                
                # Assigning a Call to a Name:
                
                # Call to groups(...): (line 21)
                # Processing the call keyword arguments (line 21)
                kwargs_1864 = {}
                
                # Call to search(...): (line 21)
                # Processing the call arguments (line 21)
                # Getting the type of 'line' (line 21)
                line_1860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 49), 'line', False)
                # Processing the call keyword arguments (line 21)
                kwargs_1861 = {}
                # Getting the type of 'SEARCH' (line 21)
                SEARCH_1858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 35), 'SEARCH', False)
                # Obtaining the member 'search' of a type (line 21)
                search_1859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 35), SEARCH_1858, 'search')
                # Calling search(args, kwargs) (line 21)
                search_call_result_1862 = invoke(stypy.reporting.localization.Localization(__file__, 21, 35), search_1859, *[line_1860], **kwargs_1861)
                
                # Obtaining the member 'groups' of a type (line 21)
                groups_1863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 35), search_call_result_1862, 'groups')
                # Calling groups(args, kwargs) (line 21)
                groups_call_result_1865 = invoke(stypy.reporting.localization.Localization(__file__, 21, 35), groups_1863, *[], **kwargs_1864)
                
                # Assigning a type to the variable 'call_assignment_1834' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1834', groups_call_result_1865)
                
                # Assigning a Call to a Name (line 21):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_1834' (line 21)
                call_assignment_1834_1866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1834', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_1867 = stypy_get_value_from_tuple(call_assignment_1834_1866, 5, 0)
                
                # Assigning a type to the variable 'call_assignment_1835' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1835', stypy_get_value_from_tuple_call_result_1867)
                
                # Assigning a Name to a Name (line 21):
                # Getting the type of 'call_assignment_1835' (line 21)
                call_assignment_1835_1868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1835')
                # Assigning a type to the variable 'v0' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'v0', call_assignment_1835_1868)
                
                # Assigning a Call to a Name (line 21):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_1834' (line 21)
                call_assignment_1834_1869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1834', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_1870 = stypy_get_value_from_tuple(call_assignment_1834_1869, 5, 1)
                
                # Assigning a type to the variable 'call_assignment_1836' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1836', stypy_get_value_from_tuple_call_result_1870)
                
                # Assigning a Name to a Name (line 21):
                # Getting the type of 'call_assignment_1836' (line 21)
                call_assignment_1836_1871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1836')
                # Assigning a type to the variable 'v1' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'v1', call_assignment_1836_1871)
                
                # Assigning a Call to a Name (line 21):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_1834' (line 21)
                call_assignment_1834_1872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1834', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_1873 = stypy_get_value_from_tuple(call_assignment_1834_1872, 5, 2)
                
                # Assigning a type to the variable 'call_assignment_1837' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1837', stypy_get_value_from_tuple_call_result_1873)
                
                # Assigning a Name to a Name (line 21):
                # Getting the type of 'call_assignment_1837' (line 21)
                call_assignment_1837_1874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1837')
                # Assigning a type to the variable 'v2' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'v2', call_assignment_1837_1874)
                
                # Assigning a Call to a Name (line 21):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_1834' (line 21)
                call_assignment_1834_1875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1834', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_1876 = stypy_get_value_from_tuple(call_assignment_1834_1875, 5, 3)
                
                # Assigning a type to the variable 'call_assignment_1838' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1838', stypy_get_value_from_tuple_call_result_1876)
                
                # Assigning a Name to a Name (line 21):
                # Getting the type of 'call_assignment_1838' (line 21)
                call_assignment_1838_1877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1838')
                # Assigning a type to the variable 'r' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 28), 'r', call_assignment_1838_1877)
                
                # Assigning a Call to a Name (line 21):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_1834' (line 21)
                call_assignment_1834_1878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1834', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_1879 = stypy_get_value_from_tuple(call_assignment_1834_1878, 5, 4)
                
                # Assigning a type to the variable 'call_assignment_1839' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1839', stypy_get_value_from_tuple_call_result_1879)
                
                # Assigning a Name to a Name (line 21):
                # Getting the type of 'call_assignment_1839' (line 21)
                call_assignment_1839_1880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1839')
                # Assigning a type to the variable 'e' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 31), 'e', call_assignment_1839_1880)
                
                # Assigning a List to a Attribute (line 22):
                
                # Assigning a List to a Attribute (line 22):
                
                # Obtaining an instance of the builtin type 'list' (line 22)
                list_1881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 31), 'list')
                # Adding type elements to the builtin type 'list' instance (line 22)
                # Adding element type (line 22)
                
                # Call to Vector3f_str(...): (line 22)
                # Processing the call arguments (line 22)
                # Getting the type of 'v0' (line 22)
                v0_1883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 45), 'v0', False)
                # Processing the call keyword arguments (line 22)
                kwargs_1884 = {}
                # Getting the type of 'Vector3f_str' (line 22)
                Vector3f_str_1882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 32), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 22)
                Vector3f_str_call_result_1885 = invoke(stypy.reporting.localization.Localization(__file__, 22, 32), Vector3f_str_1882, *[v0_1883], **kwargs_1884)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 31), list_1881, Vector3f_str_call_result_1885)
                # Adding element type (line 22)
                
                # Call to Vector3f_str(...): (line 22)
                # Processing the call arguments (line 22)
                # Getting the type of 'v1' (line 22)
                v1_1887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 63), 'v1', False)
                # Processing the call keyword arguments (line 22)
                kwargs_1888 = {}
                # Getting the type of 'Vector3f_str' (line 22)
                Vector3f_str_1886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 50), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 22)
                Vector3f_str_call_result_1889 = invoke(stypy.reporting.localization.Localization(__file__, 22, 50), Vector3f_str_1886, *[v1_1887], **kwargs_1888)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 31), list_1881, Vector3f_str_call_result_1889)
                # Adding element type (line 22)
                
                # Call to Vector3f_str(...): (line 22)
                # Processing the call arguments (line 22)
                # Getting the type of 'v2' (line 22)
                v2_1891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 81), 'v2', False)
                # Processing the call keyword arguments (line 22)
                kwargs_1892 = {}
                # Getting the type of 'Vector3f_str' (line 22)
                Vector3f_str_1890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 68), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 22)
                Vector3f_str_call_result_1893 = invoke(stypy.reporting.localization.Localization(__file__, 22, 68), Vector3f_str_1890, *[v2_1891], **kwargs_1892)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 31), list_1881, Vector3f_str_call_result_1893)
                
                # Getting the type of 'self' (line 22)
                self_1894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'self')
                # Setting the type of the member 'vertexs' of a type (line 22)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), self_1894, 'vertexs', list_1881)
                
                # Assigning a BinOp to a Attribute (line 23):
                
                # Assigning a BinOp to a Attribute (line 23):
                
                # Call to Vector3f_str(...): (line 23)
                # Processing the call arguments (line 23)
                # Getting the type of 'v1' (line 23)
                v1_1896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 42), 'v1', False)
                # Processing the call keyword arguments (line 23)
                kwargs_1897 = {}
                # Getting the type of 'Vector3f_str' (line 23)
                Vector3f_str_1895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 29), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 23)
                Vector3f_str_call_result_1898 = invoke(stypy.reporting.localization.Localization(__file__, 23, 29), Vector3f_str_1895, *[v1_1896], **kwargs_1897)
                
                
                # Call to Vector3f_str(...): (line 23)
                # Processing the call arguments (line 23)
                # Getting the type of 'v0' (line 23)
                v0_1900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 61), 'v0', False)
                # Processing the call keyword arguments (line 23)
                kwargs_1901 = {}
                # Getting the type of 'Vector3f_str' (line 23)
                Vector3f_str_1899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 48), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 23)
                Vector3f_str_call_result_1902 = invoke(stypy.reporting.localization.Localization(__file__, 23, 48), Vector3f_str_1899, *[v0_1900], **kwargs_1901)
                
                # Applying the binary operator '-' (line 23)
                result_sub_1903 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 29), '-', Vector3f_str_call_result_1898, Vector3f_str_call_result_1902)
                
                # Getting the type of 'self' (line 23)
                self_1904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'self')
                # Setting the type of the member 'edge0' of a type (line 23)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 16), self_1904, 'edge0', result_sub_1903)
                
                # Assigning a BinOp to a Attribute (line 24):
                
                # Assigning a BinOp to a Attribute (line 24):
                
                # Call to Vector3f_str(...): (line 24)
                # Processing the call arguments (line 24)
                # Getting the type of 'v2' (line 24)
                v2_1906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 42), 'v2', False)
                # Processing the call keyword arguments (line 24)
                kwargs_1907 = {}
                # Getting the type of 'Vector3f_str' (line 24)
                Vector3f_str_1905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 24)
                Vector3f_str_call_result_1908 = invoke(stypy.reporting.localization.Localization(__file__, 24, 29), Vector3f_str_1905, *[v2_1906], **kwargs_1907)
                
                
                # Call to Vector3f_str(...): (line 24)
                # Processing the call arguments (line 24)
                # Getting the type of 'v0' (line 24)
                v0_1910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 61), 'v0', False)
                # Processing the call keyword arguments (line 24)
                kwargs_1911 = {}
                # Getting the type of 'Vector3f_str' (line 24)
                Vector3f_str_1909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 48), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 24)
                Vector3f_str_call_result_1912 = invoke(stypy.reporting.localization.Localization(__file__, 24, 48), Vector3f_str_1909, *[v0_1910], **kwargs_1911)
                
                # Applying the binary operator '-' (line 24)
                result_sub_1913 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 29), '-', Vector3f_str_call_result_1908, Vector3f_str_call_result_1912)
                
                # Getting the type of 'self' (line 24)
                self_1914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'self')
                # Setting the type of the member 'edge3' of a type (line 24)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), self_1914, 'edge3', result_sub_1913)
                
                # Assigning a Call to a Attribute (line 25):
                
                # Assigning a Call to a Attribute (line 25):
                
                # Call to clamped(...): (line 25)
                # Processing the call arguments (line 25)
                # Getting the type of 'ZERO' (line 25)
                ZERO_1920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 60), 'ZERO', False)
                # Getting the type of 'ONE' (line 25)
                ONE_1921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 66), 'ONE', False)
                # Processing the call keyword arguments (line 25)
                kwargs_1922 = {}
                
                # Call to Vector3f_str(...): (line 25)
                # Processing the call arguments (line 25)
                # Getting the type of 'r' (line 25)
                r_1916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 49), 'r', False)
                # Processing the call keyword arguments (line 25)
                kwargs_1917 = {}
                # Getting the type of 'Vector3f_str' (line 25)
                Vector3f_str_1915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 36), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 25)
                Vector3f_str_call_result_1918 = invoke(stypy.reporting.localization.Localization(__file__, 25, 36), Vector3f_str_1915, *[r_1916], **kwargs_1917)
                
                # Obtaining the member 'clamped' of a type (line 25)
                clamped_1919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 36), Vector3f_str_call_result_1918, 'clamped')
                # Calling clamped(args, kwargs) (line 25)
                clamped_call_result_1923 = invoke(stypy.reporting.localization.Localization(__file__, 25, 36), clamped_1919, *[ZERO_1920, ONE_1921], **kwargs_1922)
                
                # Getting the type of 'self' (line 25)
                self_1924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'self')
                # Setting the type of the member 'reflectivity' of a type (line 25)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), self_1924, 'reflectivity', clamped_call_result_1923)
                
                # Assigning a Call to a Attribute (line 26):
                
                # Assigning a Call to a Attribute (line 26):
                
                # Call to clamped(...): (line 26)
                # Processing the call arguments (line 26)
                # Getting the type of 'ZERO' (line 26)
                ZERO_1930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 57), 'ZERO', False)
                # Getting the type of 'MAX' (line 26)
                MAX_1931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 63), 'MAX', False)
                # Processing the call keyword arguments (line 26)
                kwargs_1932 = {}
                
                # Call to Vector3f_str(...): (line 26)
                # Processing the call arguments (line 26)
                # Getting the type of 'e' (line 26)
                e_1926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 46), 'e', False)
                # Processing the call keyword arguments (line 26)
                kwargs_1927 = {}
                # Getting the type of 'Vector3f_str' (line 26)
                Vector3f_str_1925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 33), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 26)
                Vector3f_str_call_result_1928 = invoke(stypy.reporting.localization.Localization(__file__, 26, 33), Vector3f_str_1925, *[e_1926], **kwargs_1927)
                
                # Obtaining the member 'clamped' of a type (line 26)
                clamped_1929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 33), Vector3f_str_call_result_1928, 'clamped')
                # Calling clamped(args, kwargs) (line 26)
                clamped_call_result_1933 = invoke(stypy.reporting.localization.Localization(__file__, 26, 33), clamped_1929, *[ZERO_1930, MAX_1931], **kwargs_1932)
                
                # Getting the type of 'self' (line 26)
                self_1934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'self')
                # Setting the type of the member 'emitivity' of a type (line 26)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), self_1934, 'emitivity', clamped_call_result_1933)
                
                # Assigning a BinOp to a Name (line 27):
                
                # Assigning a BinOp to a Name (line 27):
                
                # Call to Vector3f_str(...): (line 27)
                # Processing the call arguments (line 27)
                # Getting the type of 'v2' (line 27)
                v2_1936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 37), 'v2', False)
                # Processing the call keyword arguments (line 27)
                kwargs_1937 = {}
                # Getting the type of 'Vector3f_str' (line 27)
                Vector3f_str_1935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 27)
                Vector3f_str_call_result_1938 = invoke(stypy.reporting.localization.Localization(__file__, 27, 24), Vector3f_str_1935, *[v2_1936], **kwargs_1937)
                
                
                # Call to Vector3f_str(...): (line 27)
                # Processing the call arguments (line 27)
                # Getting the type of 'v1' (line 27)
                v1_1940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 56), 'v1', False)
                # Processing the call keyword arguments (line 27)
                kwargs_1941 = {}
                # Getting the type of 'Vector3f_str' (line 27)
                Vector3f_str_1939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 43), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 27)
                Vector3f_str_call_result_1942 = invoke(stypy.reporting.localization.Localization(__file__, 27, 43), Vector3f_str_1939, *[v1_1940], **kwargs_1941)
                
                # Applying the binary operator '-' (line 27)
                result_sub_1943 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 24), '-', Vector3f_str_call_result_1938, Vector3f_str_call_result_1942)
                
                # Assigning a type to the variable 'edge1' (line 27)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'edge1', result_sub_1943)
                
                # Assigning a Call to a Attribute (line 28):
                
                # Assigning a Call to a Attribute (line 28):
                
                # Call to unitize(...): (line 28)
                # Processing the call keyword arguments (line 28)
                kwargs_1947 = {}
                # Getting the type of 'self' (line 28)
                self_1944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 31), 'self', False)
                # Obtaining the member 'edge0' of a type (line 28)
                edge0_1945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 31), self_1944, 'edge0')
                # Obtaining the member 'unitize' of a type (line 28)
                unitize_1946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 31), edge0_1945, 'unitize')
                # Calling unitize(args, kwargs) (line 28)
                unitize_call_result_1948 = invoke(stypy.reporting.localization.Localization(__file__, 28, 31), unitize_1946, *[], **kwargs_1947)
                
                # Getting the type of 'self' (line 28)
                self_1949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'self')
                # Setting the type of the member 'tangent' of a type (line 28)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), self_1949, 'tangent', unitize_call_result_1948)
                
                # Assigning a Call to a Attribute (line 29):
                
                # Assigning a Call to a Attribute (line 29):
                
                # Call to unitize(...): (line 29)
                # Processing the call keyword arguments (line 29)
                kwargs_1957 = {}
                
                # Call to cross(...): (line 29)
                # Processing the call arguments (line 29)
                # Getting the type of 'edge1' (line 29)
                edge1_1953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 49), 'edge1', False)
                # Processing the call keyword arguments (line 29)
                kwargs_1954 = {}
                # Getting the type of 'self' (line 29)
                self_1950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 30), 'self', False)
                # Obtaining the member 'tangent' of a type (line 29)
                tangent_1951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 30), self_1950, 'tangent')
                # Obtaining the member 'cross' of a type (line 29)
                cross_1952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 30), tangent_1951, 'cross')
                # Calling cross(args, kwargs) (line 29)
                cross_call_result_1955 = invoke(stypy.reporting.localization.Localization(__file__, 29, 30), cross_1952, *[edge1_1953], **kwargs_1954)
                
                # Obtaining the member 'unitize' of a type (line 29)
                unitize_1956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 30), cross_call_result_1955, 'unitize')
                # Calling unitize(args, kwargs) (line 29)
                unitize_call_result_1958 = invoke(stypy.reporting.localization.Localization(__file__, 29, 30), unitize_1956, *[], **kwargs_1957)
                
                # Getting the type of 'self' (line 29)
                self_1959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'self')
                # Setting the type of the member 'normal' of a type (line 29)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 16), self_1959, 'normal', unitize_call_result_1958)
                
                # Assigning a Call to a Name (line 30):
                
                # Assigning a Call to a Name (line 30):
                
                # Call to cross(...): (line 30)
                # Processing the call arguments (line 30)
                # Getting the type of 'edge1' (line 30)
                edge1_1963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 39), 'edge1', False)
                # Processing the call keyword arguments (line 30)
                kwargs_1964 = {}
                # Getting the type of 'self' (line 30)
                self_1960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'self', False)
                # Obtaining the member 'edge0' of a type (line 30)
                edge0_1961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 22), self_1960, 'edge0')
                # Obtaining the member 'cross' of a type (line 30)
                cross_1962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 22), edge0_1961, 'cross')
                # Calling cross(args, kwargs) (line 30)
                cross_call_result_1965 = invoke(stypy.reporting.localization.Localization(__file__, 30, 22), cross_1962, *[edge1_1963], **kwargs_1964)
                
                # Assigning a type to the variable 'pa2' (line 30)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'pa2', cross_call_result_1965)
                
                # Assigning a BinOp to a Attribute (line 31):
                
                # Assigning a BinOp to a Attribute (line 31):
                
                # Call to sqrt(...): (line 31)
                # Processing the call arguments (line 31)
                
                # Call to dot(...): (line 31)
                # Processing the call arguments (line 31)
                # Getting the type of 'pa2' (line 31)
                pa2_1969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 41), 'pa2', False)
                # Processing the call keyword arguments (line 31)
                kwargs_1970 = {}
                # Getting the type of 'pa2' (line 31)
                pa2_1967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'pa2', False)
                # Obtaining the member 'dot' of a type (line 31)
                dot_1968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 33), pa2_1967, 'dot')
                # Calling dot(args, kwargs) (line 31)
                dot_call_result_1971 = invoke(stypy.reporting.localization.Localization(__file__, 31, 33), dot_1968, *[pa2_1969], **kwargs_1970)
                
                # Processing the call keyword arguments (line 31)
                kwargs_1972 = {}
                # Getting the type of 'sqrt' (line 31)
                sqrt_1966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 28), 'sqrt', False)
                # Calling sqrt(args, kwargs) (line 31)
                sqrt_call_result_1973 = invoke(stypy.reporting.localization.Localization(__file__, 31, 28), sqrt_1966, *[dot_call_result_1971], **kwargs_1972)
                
                float_1974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 49), 'float')
                # Applying the binary operator '*' (line 31)
                result_mul_1975 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 28), '*', sqrt_call_result_1973, float_1974)
                
                # Getting the type of 'self' (line 31)
                self_1976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'self')
                # Setting the type of the member 'area' of a type (line 31)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), self_1976, 'area', result_mul_1975)
                # Assigning a type to the variable 'stypy_return_type' (line 32)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'stypy_return_type', types.NoneType)
                # SSA join for if statement (line 20)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'StopIteration' (line 33)
        StopIteration_1977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'StopIteration')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 33, 8), StopIteration_1977, 'raise parameter', BaseException)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_bound(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_bound'
        module_type_store = module_type_store.open_function_context('get_bound', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Triangle.get_bound.__dict__.__setitem__('stypy_localization', localization)
        Triangle.get_bound.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Triangle.get_bound.__dict__.__setitem__('stypy_type_store', module_type_store)
        Triangle.get_bound.__dict__.__setitem__('stypy_function_name', 'Triangle.get_bound')
        Triangle.get_bound.__dict__.__setitem__('stypy_param_names_list', [])
        Triangle.get_bound.__dict__.__setitem__('stypy_varargs_param_name', None)
        Triangle.get_bound.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Triangle.get_bound.__dict__.__setitem__('stypy_call_defaults', defaults)
        Triangle.get_bound.__dict__.__setitem__('stypy_call_varargs', varargs)
        Triangle.get_bound.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Triangle.get_bound.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Triangle.get_bound', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_bound', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_bound(...)' code ##################

        
        # Assigning a Subscript to a Name (line 36):
        
        # Assigning a Subscript to a Name (line 36):
        
        # Obtaining the type of the subscript
        int_1978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 26), 'int')
        # Getting the type of 'self' (line 36)
        self_1979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'self')
        # Obtaining the member 'vertexs' of a type (line 36)
        vertexs_1980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), self_1979, 'vertexs')
        # Obtaining the member '__getitem__' of a type (line 36)
        getitem___1981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), vertexs_1980, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 36)
        subscript_call_result_1982 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), getitem___1981, int_1978)
        
        # Assigning a type to the variable 'v2' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'v2', subscript_call_result_1982)
        
        # Assigning a List to a Name (line 37):
        
        # Assigning a List to a Name (line 37):
        
        # Obtaining an instance of the builtin type 'list' (line 37)
        list_1983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 37)
        # Adding element type (line 37)
        # Getting the type of 'v2' (line 37)
        v2_1984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'v2')
        # Obtaining the member 'x' of a type (line 37)
        x_1985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 17), v2_1984, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_1983, x_1985)
        # Adding element type (line 37)
        # Getting the type of 'v2' (line 37)
        v2_1986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'v2')
        # Obtaining the member 'y' of a type (line 37)
        y_1987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 23), v2_1986, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_1983, y_1987)
        # Adding element type (line 37)
        # Getting the type of 'v2' (line 37)
        v2_1988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'v2')
        # Obtaining the member 'z' of a type (line 37)
        z_1989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 29), v2_1988, 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_1983, z_1989)
        # Adding element type (line 37)
        # Getting the type of 'v2' (line 37)
        v2_1990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 35), 'v2')
        # Obtaining the member 'x' of a type (line 37)
        x_1991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 35), v2_1990, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_1983, x_1991)
        # Adding element type (line 37)
        # Getting the type of 'v2' (line 37)
        v2_1992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 41), 'v2')
        # Obtaining the member 'y' of a type (line 37)
        y_1993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 41), v2_1992, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_1983, y_1993)
        # Adding element type (line 37)
        # Getting the type of 'v2' (line 37)
        v2_1994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 47), 'v2')
        # Obtaining the member 'z' of a type (line 37)
        z_1995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 47), v2_1994, 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_1983, z_1995)
        
        # Assigning a type to the variable 'bound' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'bound', list_1983)
        
        
        # Call to range(...): (line 38)
        # Processing the call arguments (line 38)
        int_1997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'int')
        # Processing the call keyword arguments (line 38)
        kwargs_1998 = {}
        # Getting the type of 'range' (line 38)
        range_1996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'range', False)
        # Calling range(args, kwargs) (line 38)
        range_call_result_1999 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), range_1996, *[int_1997], **kwargs_1998)
        
        # Assigning a type to the variable 'range_call_result_1999' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'range_call_result_1999', range_call_result_1999)
        # Testing if the for loop is going to be iterated (line 38)
        # Testing the type of a for loop iterable (line 38)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 8), range_call_result_1999)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 38, 8), range_call_result_1999):
            # Getting the type of the for loop variable (line 38)
            for_loop_var_2000 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 8), range_call_result_1999)
            # Assigning a type to the variable 'j' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'j', for_loop_var_2000)
            # SSA begins for a for statement (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 39):
            
            # Assigning a Subscript to a Name (line 39):
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 39)
            j_2001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 33), 'j')
            
            # Obtaining the type of the subscript
            int_2002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'int')
            # Getting the type of 'self' (line 39)
            self_2003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'self')
            # Obtaining the member 'vertexs' of a type (line 39)
            vertexs_2004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 17), self_2003, 'vertexs')
            # Obtaining the member '__getitem__' of a type (line 39)
            getitem___2005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 17), vertexs_2004, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 39)
            subscript_call_result_2006 = invoke(stypy.reporting.localization.Localization(__file__, 39, 17), getitem___2005, int_2002)
            
            # Obtaining the member '__getitem__' of a type (line 39)
            getitem___2007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 17), subscript_call_result_2006, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 39)
            subscript_call_result_2008 = invoke(stypy.reporting.localization.Localization(__file__, 39, 17), getitem___2007, j_2001)
            
            # Assigning a type to the variable 'v0' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'v0', subscript_call_result_2008)
            
            # Assigning a Subscript to a Name (line 40):
            
            # Assigning a Subscript to a Name (line 40):
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 40)
            j_2009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 33), 'j')
            
            # Obtaining the type of the subscript
            int_2010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 30), 'int')
            # Getting the type of 'self' (line 40)
            self_2011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'self')
            # Obtaining the member 'vertexs' of a type (line 40)
            vertexs_2012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 17), self_2011, 'vertexs')
            # Obtaining the member '__getitem__' of a type (line 40)
            getitem___2013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 17), vertexs_2012, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 40)
            subscript_call_result_2014 = invoke(stypy.reporting.localization.Localization(__file__, 40, 17), getitem___2013, int_2010)
            
            # Obtaining the member '__getitem__' of a type (line 40)
            getitem___2015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 17), subscript_call_result_2014, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 40)
            subscript_call_result_2016 = invoke(stypy.reporting.localization.Localization(__file__, 40, 17), getitem___2015, j_2009)
            
            # Assigning a type to the variable 'v1' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'v1', subscript_call_result_2016)
            
            # Getting the type of 'v0' (line 41)
            v0_2017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'v0')
            # Getting the type of 'v1' (line 41)
            v1_2018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'v1')
            # Applying the binary operator '<' (line 41)
            result_lt_2019 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), '<', v0_2017, v1_2018)
            
            # Testing if the type of an if condition is none (line 41)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 12), result_lt_2019):
                
                # Getting the type of 'v1' (line 47)
                v1_2045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'v1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 47)
                j_2046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'j')
                # Getting the type of 'bound' (line 47)
                bound_2047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'bound')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___2048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), bound_2047, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_2049 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___2048, j_2046)
                
                # Applying the binary operator '<' (line 47)
                result_lt_2050 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), '<', v1_2045, subscript_call_result_2049)
                
                # Testing if the type of an if condition is none (line 47)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 47, 16), result_lt_2050):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 47)
                    if_condition_2051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 16), result_lt_2050)
                    # Assigning a type to the variable 'if_condition_2051' (line 47)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'if_condition_2051', if_condition_2051)
                    # SSA begins for if statement (line 47)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 48):
                    
                    # Assigning a Name to a Subscript (line 48):
                    # Getting the type of 'v1' (line 48)
                    v1_2052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'v1')
                    # Getting the type of 'bound' (line 48)
                    bound_2053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'bound')
                    # Getting the type of 'j' (line 48)
                    j_2054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'j')
                    # Storing an element on a container (line 48)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 20), bound_2053, (j_2054, v1_2052))
                    # SSA join for if statement (line 47)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'v0' (line 49)
                v0_2055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'v0')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 49)
                j_2056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'j')
                int_2057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 34), 'int')
                # Applying the binary operator '+' (line 49)
                result_add_2058 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 30), '+', j_2056, int_2057)
                
                # Getting the type of 'bound' (line 49)
                bound_2059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'bound')
                # Obtaining the member '__getitem__' of a type (line 49)
                getitem___2060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 24), bound_2059, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                subscript_call_result_2061 = invoke(stypy.reporting.localization.Localization(__file__, 49, 24), getitem___2060, result_add_2058)
                
                # Applying the binary operator '>' (line 49)
                result_gt_2062 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 19), '>', v0_2055, subscript_call_result_2061)
                
                # Testing if the type of an if condition is none (line 49)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 49, 16), result_gt_2062):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 49)
                    if_condition_2063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 16), result_gt_2062)
                    # Assigning a type to the variable 'if_condition_2063' (line 49)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'if_condition_2063', if_condition_2063)
                    # SSA begins for if statement (line 49)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 50):
                    
                    # Assigning a Name to a Subscript (line 50):
                    # Getting the type of 'v0' (line 50)
                    v0_2064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'v0')
                    # Getting the type of 'bound' (line 50)
                    bound_2065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'bound')
                    # Getting the type of 'j' (line 50)
                    j_2066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'j')
                    int_2067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'int')
                    # Applying the binary operator '+' (line 50)
                    result_add_2068 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 26), '+', j_2066, int_2067)
                    
                    # Storing an element on a container (line 50)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), bound_2065, (result_add_2068, v0_2064))
                    # SSA join for if statement (line 49)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 41)
                if_condition_2020 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 12), result_lt_2019)
                # Assigning a type to the variable 'if_condition_2020' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'if_condition_2020', if_condition_2020)
                # SSA begins for if statement (line 41)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'v0' (line 42)
                v0_2021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'v0')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 42)
                j_2022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 30), 'j')
                # Getting the type of 'bound' (line 42)
                bound_2023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'bound')
                # Obtaining the member '__getitem__' of a type (line 42)
                getitem___2024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), bound_2023, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                subscript_call_result_2025 = invoke(stypy.reporting.localization.Localization(__file__, 42, 24), getitem___2024, j_2022)
                
                # Applying the binary operator '<' (line 42)
                result_lt_2026 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 19), '<', v0_2021, subscript_call_result_2025)
                
                # Testing if the type of an if condition is none (line 42)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 16), result_lt_2026):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 42)
                    if_condition_2027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 16), result_lt_2026)
                    # Assigning a type to the variable 'if_condition_2027' (line 42)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'if_condition_2027', if_condition_2027)
                    # SSA begins for if statement (line 42)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 43):
                    
                    # Assigning a Name to a Subscript (line 43):
                    # Getting the type of 'v0' (line 43)
                    v0_2028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 31), 'v0')
                    # Getting the type of 'bound' (line 43)
                    bound_2029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'bound')
                    # Getting the type of 'j' (line 43)
                    j_2030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 26), 'j')
                    # Storing an element on a container (line 43)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 20), bound_2029, (j_2030, v0_2028))
                    # SSA join for if statement (line 42)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'v1' (line 44)
                v1_2031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'v1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 44)
                j_2032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'j')
                int_2033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'int')
                # Applying the binary operator '+' (line 44)
                result_add_2034 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 30), '+', j_2032, int_2033)
                
                # Getting the type of 'bound' (line 44)
                bound_2035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'bound')
                # Obtaining the member '__getitem__' of a type (line 44)
                getitem___2036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 24), bound_2035, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                subscript_call_result_2037 = invoke(stypy.reporting.localization.Localization(__file__, 44, 24), getitem___2036, result_add_2034)
                
                # Applying the binary operator '>' (line 44)
                result_gt_2038 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 19), '>', v1_2031, subscript_call_result_2037)
                
                # Testing if the type of an if condition is none (line 44)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 16), result_gt_2038):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 44)
                    if_condition_2039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 16), result_gt_2038)
                    # Assigning a type to the variable 'if_condition_2039' (line 44)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'if_condition_2039', if_condition_2039)
                    # SSA begins for if statement (line 44)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 45):
                    
                    # Assigning a Name to a Subscript (line 45):
                    # Getting the type of 'v1' (line 45)
                    v1_2040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 35), 'v1')
                    # Getting the type of 'bound' (line 45)
                    bound_2041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'bound')
                    # Getting the type of 'j' (line 45)
                    j_2042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'j')
                    int_2043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 30), 'int')
                    # Applying the binary operator '+' (line 45)
                    result_add_2044 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 26), '+', j_2042, int_2043)
                    
                    # Storing an element on a container (line 45)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 20), bound_2041, (result_add_2044, v1_2040))
                    # SSA join for if statement (line 44)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 41)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'v1' (line 47)
                v1_2045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'v1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 47)
                j_2046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'j')
                # Getting the type of 'bound' (line 47)
                bound_2047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'bound')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___2048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), bound_2047, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_2049 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___2048, j_2046)
                
                # Applying the binary operator '<' (line 47)
                result_lt_2050 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), '<', v1_2045, subscript_call_result_2049)
                
                # Testing if the type of an if condition is none (line 47)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 47, 16), result_lt_2050):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 47)
                    if_condition_2051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 16), result_lt_2050)
                    # Assigning a type to the variable 'if_condition_2051' (line 47)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'if_condition_2051', if_condition_2051)
                    # SSA begins for if statement (line 47)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 48):
                    
                    # Assigning a Name to a Subscript (line 48):
                    # Getting the type of 'v1' (line 48)
                    v1_2052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'v1')
                    # Getting the type of 'bound' (line 48)
                    bound_2053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'bound')
                    # Getting the type of 'j' (line 48)
                    j_2054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'j')
                    # Storing an element on a container (line 48)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 20), bound_2053, (j_2054, v1_2052))
                    # SSA join for if statement (line 47)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'v0' (line 49)
                v0_2055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'v0')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 49)
                j_2056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'j')
                int_2057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 34), 'int')
                # Applying the binary operator '+' (line 49)
                result_add_2058 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 30), '+', j_2056, int_2057)
                
                # Getting the type of 'bound' (line 49)
                bound_2059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'bound')
                # Obtaining the member '__getitem__' of a type (line 49)
                getitem___2060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 24), bound_2059, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                subscript_call_result_2061 = invoke(stypy.reporting.localization.Localization(__file__, 49, 24), getitem___2060, result_add_2058)
                
                # Applying the binary operator '>' (line 49)
                result_gt_2062 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 19), '>', v0_2055, subscript_call_result_2061)
                
                # Testing if the type of an if condition is none (line 49)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 49, 16), result_gt_2062):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 49)
                    if_condition_2063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 16), result_gt_2062)
                    # Assigning a type to the variable 'if_condition_2063' (line 49)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'if_condition_2063', if_condition_2063)
                    # SSA begins for if statement (line 49)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 50):
                    
                    # Assigning a Name to a Subscript (line 50):
                    # Getting the type of 'v0' (line 50)
                    v0_2064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'v0')
                    # Getting the type of 'bound' (line 50)
                    bound_2065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'bound')
                    # Getting the type of 'j' (line 50)
                    j_2066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'j')
                    int_2067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'int')
                    # Applying the binary operator '+' (line 50)
                    result_add_2068 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 26), '+', j_2066, int_2067)
                    
                    # Storing an element on a container (line 50)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), bound_2065, (result_add_2068, v0_2064))
                    # SSA join for if statement (line 49)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 41)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'bound' (line 51)
            bound_2069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'bound')
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 51)
            j_2070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'j')
            # Getting the type of 'bound' (line 51)
            bound_2071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'bound')
            # Obtaining the member '__getitem__' of a type (line 51)
            getitem___2072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), bound_2071, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 51)
            subscript_call_result_2073 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), getitem___2072, j_2070)
            
            
            # Call to abs(...): (line 51)
            # Processing the call arguments (line 51)
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 51)
            j_2075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 35), 'j', False)
            # Getting the type of 'bound' (line 51)
            bound_2076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'bound', False)
            # Obtaining the member '__getitem__' of a type (line 51)
            getitem___2077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 29), bound_2076, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 51)
            subscript_call_result_2078 = invoke(stypy.reporting.localization.Localization(__file__, 51, 29), getitem___2077, j_2075)
            
            # Processing the call keyword arguments (line 51)
            kwargs_2079 = {}
            # Getting the type of 'abs' (line 51)
            abs_2074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'abs', False)
            # Calling abs(args, kwargs) (line 51)
            abs_call_result_2080 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), abs_2074, *[subscript_call_result_2078], **kwargs_2079)
            
            float_2081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 41), 'float')
            # Applying the binary operator '+' (line 51)
            result_add_2082 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 25), '+', abs_call_result_2080, float_2081)
            
            # Getting the type of 'TOLERANCE' (line 51)
            TOLERANCE_2083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 48), 'TOLERANCE')
            # Applying the binary operator '*' (line 51)
            result_mul_2084 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 24), '*', result_add_2082, TOLERANCE_2083)
            
            # Applying the binary operator '-=' (line 51)
            result_isub_2085 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 12), '-=', subscript_call_result_2073, result_mul_2084)
            # Getting the type of 'bound' (line 51)
            bound_2086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'bound')
            # Getting the type of 'j' (line 51)
            j_2087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'j')
            # Storing an element on a container (line 51)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 12), bound_2086, (j_2087, result_isub_2085))
            
            
            # Getting the type of 'bound' (line 52)
            bound_2088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'bound')
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 52)
            j_2089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'j')
            int_2090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 22), 'int')
            # Applying the binary operator '+' (line 52)
            result_add_2091 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 18), '+', j_2089, int_2090)
            
            # Getting the type of 'bound' (line 52)
            bound_2092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'bound')
            # Obtaining the member '__getitem__' of a type (line 52)
            getitem___2093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), bound_2092, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 52)
            subscript_call_result_2094 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), getitem___2093, result_add_2091)
            
            
            # Call to abs(...): (line 52)
            # Processing the call arguments (line 52)
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 52)
            j_2096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 39), 'j', False)
            int_2097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 43), 'int')
            # Applying the binary operator '+' (line 52)
            result_add_2098 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 39), '+', j_2096, int_2097)
            
            # Getting the type of 'bound' (line 52)
            bound_2099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 33), 'bound', False)
            # Obtaining the member '__getitem__' of a type (line 52)
            getitem___2100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 33), bound_2099, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 52)
            subscript_call_result_2101 = invoke(stypy.reporting.localization.Localization(__file__, 52, 33), getitem___2100, result_add_2098)
            
            # Processing the call keyword arguments (line 52)
            kwargs_2102 = {}
            # Getting the type of 'abs' (line 52)
            abs_2095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 29), 'abs', False)
            # Calling abs(args, kwargs) (line 52)
            abs_call_result_2103 = invoke(stypy.reporting.localization.Localization(__file__, 52, 29), abs_2095, *[subscript_call_result_2101], **kwargs_2102)
            
            float_2104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 49), 'float')
            # Applying the binary operator '+' (line 52)
            result_add_2105 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 29), '+', abs_call_result_2103, float_2104)
            
            # Getting the type of 'TOLERANCE' (line 52)
            TOLERANCE_2106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 56), 'TOLERANCE')
            # Applying the binary operator '*' (line 52)
            result_mul_2107 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 28), '*', result_add_2105, TOLERANCE_2106)
            
            # Applying the binary operator '+=' (line 52)
            result_iadd_2108 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 12), '+=', subscript_call_result_2094, result_mul_2107)
            # Getting the type of 'bound' (line 52)
            bound_2109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'bound')
            # Getting the type of 'j' (line 52)
            j_2110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'j')
            int_2111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 22), 'int')
            # Applying the binary operator '+' (line 52)
            result_add_2112 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 18), '+', j_2110, int_2111)
            
            # Storing an element on a container (line 52)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), bound_2109, (result_add_2112, result_iadd_2108))
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'bound' (line 53)
        bound_2113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'bound')
        # Assigning a type to the variable 'stypy_return_type' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', bound_2113)
        
        # ################# End of 'get_bound(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_bound' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_2114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2114)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_bound'
        return stypy_return_type_2114


    @norecursion
    def get_intersection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_intersection'
        module_type_store = module_type_store.open_function_context('get_intersection', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Triangle.get_intersection.__dict__.__setitem__('stypy_localization', localization)
        Triangle.get_intersection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Triangle.get_intersection.__dict__.__setitem__('stypy_type_store', module_type_store)
        Triangle.get_intersection.__dict__.__setitem__('stypy_function_name', 'Triangle.get_intersection')
        Triangle.get_intersection.__dict__.__setitem__('stypy_param_names_list', ['ray_origin', 'ray_direction'])
        Triangle.get_intersection.__dict__.__setitem__('stypy_varargs_param_name', None)
        Triangle.get_intersection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Triangle.get_intersection.__dict__.__setitem__('stypy_call_defaults', defaults)
        Triangle.get_intersection.__dict__.__setitem__('stypy_call_varargs', varargs)
        Triangle.get_intersection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Triangle.get_intersection.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Triangle.get_intersection', ['ray_origin', 'ray_direction'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_intersection', localization, ['ray_origin', 'ray_direction'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_intersection(...)' code ##################

        
        # Assigning a Attribute to a Name (line 56):
        
        # Assigning a Attribute to a Name (line 56):
        # Getting the type of 'self' (line 56)
        self_2115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'self')
        # Obtaining the member 'edge0' of a type (line 56)
        edge0_2116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 14), self_2115, 'edge0')
        # Obtaining the member 'x' of a type (line 56)
        x_2117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 14), edge0_2116, 'x')
        # Assigning a type to the variable 'e1x' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'e1x', x_2117)
        
        # Assigning a Attribute to a Name (line 56):
        
        # Assigning a Attribute to a Name (line 56):
        # Getting the type of 'self' (line 56)
        self_2118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 34), 'self')
        # Obtaining the member 'edge0' of a type (line 56)
        edge0_2119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 34), self_2118, 'edge0')
        # Obtaining the member 'y' of a type (line 56)
        y_2120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 34), edge0_2119, 'y')
        # Assigning a type to the variable 'e1y' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 28), 'e1y', y_2120)
        
        # Assigning a Attribute to a Name (line 56):
        
        # Assigning a Attribute to a Name (line 56):
        # Getting the type of 'self' (line 56)
        self_2121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 54), 'self')
        # Obtaining the member 'edge0' of a type (line 56)
        edge0_2122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 54), self_2121, 'edge0')
        # Obtaining the member 'z' of a type (line 56)
        z_2123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 54), edge0_2122, 'z')
        # Assigning a type to the variable 'e1z' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 48), 'e1z', z_2123)
        
        # Assigning a Attribute to a Name (line 57):
        
        # Assigning a Attribute to a Name (line 57):
        # Getting the type of 'self' (line 57)
        self_2124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'self')
        # Obtaining the member 'edge3' of a type (line 57)
        edge3_2125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 14), self_2124, 'edge3')
        # Obtaining the member 'x' of a type (line 57)
        x_2126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 14), edge3_2125, 'x')
        # Assigning a type to the variable 'e2x' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'e2x', x_2126)
        
        # Assigning a Attribute to a Name (line 57):
        
        # Assigning a Attribute to a Name (line 57):
        # Getting the type of 'self' (line 57)
        self_2127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'self')
        # Obtaining the member 'edge3' of a type (line 57)
        edge3_2128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 34), self_2127, 'edge3')
        # Obtaining the member 'y' of a type (line 57)
        y_2129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 34), edge3_2128, 'y')
        # Assigning a type to the variable 'e2y' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'e2y', y_2129)
        
        # Assigning a Attribute to a Name (line 57):
        
        # Assigning a Attribute to a Name (line 57):
        # Getting the type of 'self' (line 57)
        self_2130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 54), 'self')
        # Obtaining the member 'edge3' of a type (line 57)
        edge3_2131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 54), self_2130, 'edge3')
        # Obtaining the member 'z' of a type (line 57)
        z_2132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 54), edge3_2131, 'z')
        # Assigning a type to the variable 'e2z' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 48), 'e2z', z_2132)
        
        # Assigning a BinOp to a Name (line 58):
        
        # Assigning a BinOp to a Name (line 58):
        # Getting the type of 'ray_direction' (line 58)
        ray_direction_2133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 14), 'ray_direction')
        # Obtaining the member 'y' of a type (line 58)
        y_2134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 14), ray_direction_2133, 'y')
        # Getting the type of 'e2z' (line 58)
        e2z_2135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'e2z')
        # Applying the binary operator '*' (line 58)
        result_mul_2136 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 14), '*', y_2134, e2z_2135)
        
        # Getting the type of 'ray_direction' (line 58)
        ray_direction_2137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 38), 'ray_direction')
        # Obtaining the member 'z' of a type (line 58)
        z_2138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 38), ray_direction_2137, 'z')
        # Getting the type of 'e2y' (line 58)
        e2y_2139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 56), 'e2y')
        # Applying the binary operator '*' (line 58)
        result_mul_2140 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 38), '*', z_2138, e2y_2139)
        
        # Applying the binary operator '-' (line 58)
        result_sub_2141 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 14), '-', result_mul_2136, result_mul_2140)
        
        # Assigning a type to the variable 'pvx' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'pvx', result_sub_2141)
        
        # Assigning a BinOp to a Name (line 59):
        
        # Assigning a BinOp to a Name (line 59):
        # Getting the type of 'ray_direction' (line 59)
        ray_direction_2142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 14), 'ray_direction')
        # Obtaining the member 'z' of a type (line 59)
        z_2143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 14), ray_direction_2142, 'z')
        # Getting the type of 'e2x' (line 59)
        e2x_2144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'e2x')
        # Applying the binary operator '*' (line 59)
        result_mul_2145 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 14), '*', z_2143, e2x_2144)
        
        # Getting the type of 'ray_direction' (line 59)
        ray_direction_2146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 38), 'ray_direction')
        # Obtaining the member 'x' of a type (line 59)
        x_2147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 38), ray_direction_2146, 'x')
        # Getting the type of 'e2z' (line 59)
        e2z_2148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 56), 'e2z')
        # Applying the binary operator '*' (line 59)
        result_mul_2149 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 38), '*', x_2147, e2z_2148)
        
        # Applying the binary operator '-' (line 59)
        result_sub_2150 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 14), '-', result_mul_2145, result_mul_2149)
        
        # Assigning a type to the variable 'pvy' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'pvy', result_sub_2150)
        
        # Assigning a BinOp to a Name (line 60):
        
        # Assigning a BinOp to a Name (line 60):
        # Getting the type of 'ray_direction' (line 60)
        ray_direction_2151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'ray_direction')
        # Obtaining the member 'x' of a type (line 60)
        x_2152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 14), ray_direction_2151, 'x')
        # Getting the type of 'e2y' (line 60)
        e2y_2153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 32), 'e2y')
        # Applying the binary operator '*' (line 60)
        result_mul_2154 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 14), '*', x_2152, e2y_2153)
        
        # Getting the type of 'ray_direction' (line 60)
        ray_direction_2155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'ray_direction')
        # Obtaining the member 'y' of a type (line 60)
        y_2156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 38), ray_direction_2155, 'y')
        # Getting the type of 'e2x' (line 60)
        e2x_2157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 56), 'e2x')
        # Applying the binary operator '*' (line 60)
        result_mul_2158 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 38), '*', y_2156, e2x_2157)
        
        # Applying the binary operator '-' (line 60)
        result_sub_2159 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 14), '-', result_mul_2154, result_mul_2158)
        
        # Assigning a type to the variable 'pvz' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'pvz', result_sub_2159)
        
        # Assigning a BinOp to a Name (line 61):
        
        # Assigning a BinOp to a Name (line 61):
        # Getting the type of 'e1x' (line 61)
        e1x_2160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'e1x')
        # Getting the type of 'pvx' (line 61)
        pvx_2161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'pvx')
        # Applying the binary operator '*' (line 61)
        result_mul_2162 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 14), '*', e1x_2160, pvx_2161)
        
        # Getting the type of 'e1y' (line 61)
        e1y_2163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'e1y')
        # Getting the type of 'pvy' (line 61)
        pvy_2164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'pvy')
        # Applying the binary operator '*' (line 61)
        result_mul_2165 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 26), '*', e1y_2163, pvy_2164)
        
        # Applying the binary operator '+' (line 61)
        result_add_2166 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 14), '+', result_mul_2162, result_mul_2165)
        
        # Getting the type of 'e1z' (line 61)
        e1z_2167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'e1z')
        # Getting the type of 'pvz' (line 61)
        pvz_2168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 44), 'pvz')
        # Applying the binary operator '*' (line 61)
        result_mul_2169 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 38), '*', e1z_2167, pvz_2168)
        
        # Applying the binary operator '+' (line 61)
        result_add_2170 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 36), '+', result_add_2166, result_mul_2169)
        
        # Assigning a type to the variable 'det' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'det', result_add_2170)
        
        float_2171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 11), 'float')
        # Getting the type of 'det' (line 62)
        det_2172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'det')
        # Applying the binary operator '<' (line 62)
        result_lt_2173 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), '<', float_2171, det_2172)
        float_2174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 29), 'float')
        # Applying the binary operator '<' (line 62)
        result_lt_2175 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), '<', det_2172, float_2174)
        # Applying the binary operator '&' (line 62)
        result_and__2176 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), '&', result_lt_2173, result_lt_2175)
        
        # Testing if the type of an if condition is none (line 62)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 62, 8), result_and__2176):
            pass
        else:
            
            # Testing the type of an if condition (line 62)
            if_condition_2177 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 8), result_and__2176)
            # Assigning a type to the variable 'if_condition_2177' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'if_condition_2177', if_condition_2177)
            # SSA begins for if statement (line 62)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            float_2178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'stypy_return_type', float_2178)
            # SSA join for if statement (line 62)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 64):
        
        # Assigning a BinOp to a Name (line 64):
        float_2179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'float')
        # Getting the type of 'det' (line 64)
        det_2180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'det')
        # Applying the binary operator 'div' (line 64)
        result_div_2181 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 18), 'div', float_2179, det_2180)
        
        # Assigning a type to the variable 'inv_det' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'inv_det', result_div_2181)
        
        # Assigning a Subscript to a Name (line 65):
        
        # Assigning a Subscript to a Name (line 65):
        
        # Obtaining the type of the subscript
        int_2182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'int')
        # Getting the type of 'self' (line 65)
        self_2183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'self')
        # Obtaining the member 'vertexs' of a type (line 65)
        vertexs_2184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 13), self_2183, 'vertexs')
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___2185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 13), vertexs_2184, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_2186 = invoke(stypy.reporting.localization.Localization(__file__, 65, 13), getitem___2185, int_2182)
        
        # Assigning a type to the variable 'v0' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'v0', subscript_call_result_2186)
        
        # Assigning a BinOp to a Name (line 66):
        
        # Assigning a BinOp to a Name (line 66):
        # Getting the type of 'ray_origin' (line 66)
        ray_origin_2187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 14), 'ray_origin')
        # Obtaining the member 'x' of a type (line 66)
        x_2188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 14), ray_origin_2187, 'x')
        # Getting the type of 'v0' (line 66)
        v0_2189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'v0')
        # Obtaining the member 'x' of a type (line 66)
        x_2190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 29), v0_2189, 'x')
        # Applying the binary operator '-' (line 66)
        result_sub_2191 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 14), '-', x_2188, x_2190)
        
        # Assigning a type to the variable 'tvx' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'tvx', result_sub_2191)
        
        # Assigning a BinOp to a Name (line 67):
        
        # Assigning a BinOp to a Name (line 67):
        # Getting the type of 'ray_origin' (line 67)
        ray_origin_2192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 14), 'ray_origin')
        # Obtaining the member 'y' of a type (line 67)
        y_2193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 14), ray_origin_2192, 'y')
        # Getting the type of 'v0' (line 67)
        v0_2194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'v0')
        # Obtaining the member 'y' of a type (line 67)
        y_2195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 29), v0_2194, 'y')
        # Applying the binary operator '-' (line 67)
        result_sub_2196 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 14), '-', y_2193, y_2195)
        
        # Assigning a type to the variable 'tvy' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'tvy', result_sub_2196)
        
        # Assigning a BinOp to a Name (line 68):
        
        # Assigning a BinOp to a Name (line 68):
        # Getting the type of 'ray_origin' (line 68)
        ray_origin_2197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'ray_origin')
        # Obtaining the member 'z' of a type (line 68)
        z_2198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 14), ray_origin_2197, 'z')
        # Getting the type of 'v0' (line 68)
        v0_2199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 29), 'v0')
        # Obtaining the member 'z' of a type (line 68)
        z_2200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 29), v0_2199, 'z')
        # Applying the binary operator '-' (line 68)
        result_sub_2201 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 14), '-', z_2198, z_2200)
        
        # Assigning a type to the variable 'tvz' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tvz', result_sub_2201)
        
        # Assigning a BinOp to a Name (line 69):
        
        # Assigning a BinOp to a Name (line 69):
        # Getting the type of 'tvx' (line 69)
        tvx_2202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'tvx')
        # Getting the type of 'pvx' (line 69)
        pvx_2203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'pvx')
        # Applying the binary operator '*' (line 69)
        result_mul_2204 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '*', tvx_2202, pvx_2203)
        
        # Getting the type of 'tvy' (line 69)
        tvy_2205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'tvy')
        # Getting the type of 'pvy' (line 69)
        pvy_2206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 31), 'pvy')
        # Applying the binary operator '*' (line 69)
        result_mul_2207 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 25), '*', tvy_2205, pvy_2206)
        
        # Applying the binary operator '+' (line 69)
        result_add_2208 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '+', result_mul_2204, result_mul_2207)
        
        # Getting the type of 'tvz' (line 69)
        tvz_2209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 37), 'tvz')
        # Getting the type of 'pvz' (line 69)
        pvz_2210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), 'pvz')
        # Applying the binary operator '*' (line 69)
        result_mul_2211 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 37), '*', tvz_2209, pvz_2210)
        
        # Applying the binary operator '+' (line 69)
        result_add_2212 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 35), '+', result_add_2208, result_mul_2211)
        
        # Getting the type of 'inv_det' (line 69)
        inv_det_2213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 50), 'inv_det')
        # Applying the binary operator '*' (line 69)
        result_mul_2214 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 12), '*', result_add_2212, inv_det_2213)
        
        # Assigning a type to the variable 'u' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'u', result_mul_2214)
        
        # Getting the type of 'u' (line 70)
        u_2215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'u')
        float_2216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 15), 'float')
        # Applying the binary operator '<' (line 70)
        result_lt_2217 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 11), '<', u_2215, float_2216)
        
        # Testing if the type of an if condition is none (line 70)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 8), result_lt_2217):
            
            # Getting the type of 'u' (line 72)
            u_2220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'u')
            float_2221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 17), 'float')
            # Applying the binary operator '>' (line 72)
            result_gt_2222 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '>', u_2220, float_2221)
            
            # Testing if the type of an if condition is none (line 72)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 13), result_gt_2222):
                pass
            else:
                
                # Testing the type of an if condition (line 72)
                if_condition_2223 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_gt_2222)
                # Assigning a type to the variable 'if_condition_2223' (line 72)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_2223', if_condition_2223)
                # SSA begins for if statement (line 72)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                float_2224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'float')
                # Assigning a type to the variable 'stypy_return_type' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', float_2224)
                # SSA join for if statement (line 72)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 70)
            if_condition_2218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 8), result_lt_2217)
            # Assigning a type to the variable 'if_condition_2218' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'if_condition_2218', if_condition_2218)
            # SSA begins for if statement (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            float_2219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'stypy_return_type', float_2219)
            # SSA branch for the else part of an if statement (line 70)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'u' (line 72)
            u_2220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'u')
            float_2221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 17), 'float')
            # Applying the binary operator '>' (line 72)
            result_gt_2222 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '>', u_2220, float_2221)
            
            # Testing if the type of an if condition is none (line 72)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 13), result_gt_2222):
                pass
            else:
                
                # Testing the type of an if condition (line 72)
                if_condition_2223 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_gt_2222)
                # Assigning a type to the variable 'if_condition_2223' (line 72)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_2223', if_condition_2223)
                # SSA begins for if statement (line 72)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                float_2224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'float')
                # Assigning a type to the variable 'stypy_return_type' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', float_2224)
                # SSA join for if statement (line 72)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 70)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 74):
        
        # Assigning a BinOp to a Name (line 74):
        # Getting the type of 'tvy' (line 74)
        tvy_2225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 14), 'tvy')
        # Getting the type of 'e1z' (line 74)
        e1z_2226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'e1z')
        # Applying the binary operator '*' (line 74)
        result_mul_2227 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 14), '*', tvy_2225, e1z_2226)
        
        # Getting the type of 'tvz' (line 74)
        tvz_2228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'tvz')
        # Getting the type of 'e1y' (line 74)
        e1y_2229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 32), 'e1y')
        # Applying the binary operator '*' (line 74)
        result_mul_2230 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 26), '*', tvz_2228, e1y_2229)
        
        # Applying the binary operator '-' (line 74)
        result_sub_2231 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 14), '-', result_mul_2227, result_mul_2230)
        
        # Assigning a type to the variable 'qvx' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'qvx', result_sub_2231)
        
        # Assigning a BinOp to a Name (line 75):
        
        # Assigning a BinOp to a Name (line 75):
        # Getting the type of 'tvz' (line 75)
        tvz_2232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'tvz')
        # Getting the type of 'e1x' (line 75)
        e1x_2233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'e1x')
        # Applying the binary operator '*' (line 75)
        result_mul_2234 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 14), '*', tvz_2232, e1x_2233)
        
        # Getting the type of 'tvx' (line 75)
        tvx_2235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'tvx')
        # Getting the type of 'e1z' (line 75)
        e1z_2236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'e1z')
        # Applying the binary operator '*' (line 75)
        result_mul_2237 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 26), '*', tvx_2235, e1z_2236)
        
        # Applying the binary operator '-' (line 75)
        result_sub_2238 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 14), '-', result_mul_2234, result_mul_2237)
        
        # Assigning a type to the variable 'qvy' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'qvy', result_sub_2238)
        
        # Assigning a BinOp to a Name (line 76):
        
        # Assigning a BinOp to a Name (line 76):
        # Getting the type of 'tvx' (line 76)
        tvx_2239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'tvx')
        # Getting the type of 'e1y' (line 76)
        e1y_2240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'e1y')
        # Applying the binary operator '*' (line 76)
        result_mul_2241 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 14), '*', tvx_2239, e1y_2240)
        
        # Getting the type of 'tvy' (line 76)
        tvy_2242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'tvy')
        # Getting the type of 'e1x' (line 76)
        e1x_2243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 32), 'e1x')
        # Applying the binary operator '*' (line 76)
        result_mul_2244 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 26), '*', tvy_2242, e1x_2243)
        
        # Applying the binary operator '-' (line 76)
        result_sub_2245 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 14), '-', result_mul_2241, result_mul_2244)
        
        # Assigning a type to the variable 'qvz' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'qvz', result_sub_2245)
        
        # Assigning a BinOp to a Name (line 77):
        
        # Assigning a BinOp to a Name (line 77):
        # Getting the type of 'ray_direction' (line 77)
        ray_direction_2246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'ray_direction')
        # Obtaining the member 'x' of a type (line 77)
        x_2247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), ray_direction_2246, 'x')
        # Getting the type of 'qvx' (line 77)
        qvx_2248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 31), 'qvx')
        # Applying the binary operator '*' (line 77)
        result_mul_2249 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 13), '*', x_2247, qvx_2248)
        
        # Getting the type of 'ray_direction' (line 77)
        ray_direction_2250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), 'ray_direction')
        # Obtaining the member 'y' of a type (line 77)
        y_2251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 37), ray_direction_2250, 'y')
        # Getting the type of 'qvy' (line 77)
        qvy_2252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 55), 'qvy')
        # Applying the binary operator '*' (line 77)
        result_mul_2253 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 37), '*', y_2251, qvy_2252)
        
        # Applying the binary operator '+' (line 77)
        result_add_2254 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 13), '+', result_mul_2249, result_mul_2253)
        
        # Getting the type of 'ray_direction' (line 77)
        ray_direction_2255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 61), 'ray_direction')
        # Obtaining the member 'z' of a type (line 77)
        z_2256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 61), ray_direction_2255, 'z')
        # Getting the type of 'qvz' (line 77)
        qvz_2257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 79), 'qvz')
        # Applying the binary operator '*' (line 77)
        result_mul_2258 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 61), '*', z_2256, qvz_2257)
        
        # Applying the binary operator '+' (line 77)
        result_add_2259 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 59), '+', result_add_2254, result_mul_2258)
        
        # Getting the type of 'inv_det' (line 77)
        inv_det_2260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 86), 'inv_det')
        # Applying the binary operator '*' (line 77)
        result_mul_2261 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 12), '*', result_add_2259, inv_det_2260)
        
        # Assigning a type to the variable 'v' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'v', result_mul_2261)
        
        # Getting the type of 'v' (line 78)
        v_2262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'v')
        float_2263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 15), 'float')
        # Applying the binary operator '<' (line 78)
        result_lt_2264 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), '<', v_2262, float_2263)
        
        # Testing if the type of an if condition is none (line 78)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 78, 8), result_lt_2264):
            
            # Getting the type of 'u' (line 80)
            u_2267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'u')
            # Getting the type of 'v' (line 80)
            v_2268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'v')
            # Applying the binary operator '+' (line 80)
            result_add_2269 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '+', u_2267, v_2268)
            
            float_2270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'float')
            # Applying the binary operator '>' (line 80)
            result_gt_2271 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '>', result_add_2269, float_2270)
            
            # Testing if the type of an if condition is none (line 80)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 13), result_gt_2271):
                pass
            else:
                
                # Testing the type of an if condition (line 80)
                if_condition_2272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 13), result_gt_2271)
                # Assigning a type to the variable 'if_condition_2272' (line 80)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'if_condition_2272', if_condition_2272)
                # SSA begins for if statement (line 80)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                float_2273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 19), 'float')
                # Assigning a type to the variable 'stypy_return_type' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stypy_return_type', float_2273)
                # SSA join for if statement (line 80)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 78)
            if_condition_2265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_lt_2264)
            # Assigning a type to the variable 'if_condition_2265' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_2265', if_condition_2265)
            # SSA begins for if statement (line 78)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            float_2266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'stypy_return_type', float_2266)
            # SSA branch for the else part of an if statement (line 78)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'u' (line 80)
            u_2267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'u')
            # Getting the type of 'v' (line 80)
            v_2268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'v')
            # Applying the binary operator '+' (line 80)
            result_add_2269 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '+', u_2267, v_2268)
            
            float_2270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'float')
            # Applying the binary operator '>' (line 80)
            result_gt_2271 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '>', result_add_2269, float_2270)
            
            # Testing if the type of an if condition is none (line 80)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 13), result_gt_2271):
                pass
            else:
                
                # Testing the type of an if condition (line 80)
                if_condition_2272 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 13), result_gt_2271)
                # Assigning a type to the variable 'if_condition_2272' (line 80)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'if_condition_2272', if_condition_2272)
                # SSA begins for if statement (line 80)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                float_2273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 19), 'float')
                # Assigning a type to the variable 'stypy_return_type' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stypy_return_type', float_2273)
                # SSA join for if statement (line 80)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 78)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 82):
        
        # Assigning a BinOp to a Name (line 82):
        # Getting the type of 'e2x' (line 82)
        e2x_2274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'e2x')
        # Getting the type of 'qvx' (line 82)
        qvx_2275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'qvx')
        # Applying the binary operator '*' (line 82)
        result_mul_2276 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 13), '*', e2x_2274, qvx_2275)
        
        # Getting the type of 'e2y' (line 82)
        e2y_2277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'e2y')
        # Getting the type of 'qvy' (line 82)
        qvy_2278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 31), 'qvy')
        # Applying the binary operator '*' (line 82)
        result_mul_2279 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 25), '*', e2y_2277, qvy_2278)
        
        # Applying the binary operator '+' (line 82)
        result_add_2280 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 13), '+', result_mul_2276, result_mul_2279)
        
        # Getting the type of 'e2z' (line 82)
        e2z_2281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'e2z')
        # Getting the type of 'qvz' (line 82)
        qvz_2282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), 'qvz')
        # Applying the binary operator '*' (line 82)
        result_mul_2283 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 37), '*', e2z_2281, qvz_2282)
        
        # Applying the binary operator '+' (line 82)
        result_add_2284 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 35), '+', result_add_2280, result_mul_2283)
        
        # Getting the type of 'inv_det' (line 82)
        inv_det_2285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 50), 'inv_det')
        # Applying the binary operator '*' (line 82)
        result_mul_2286 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 12), '*', result_add_2284, inv_det_2285)
        
        # Assigning a type to the variable 't' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 't', result_mul_2286)
        
        # Getting the type of 't' (line 83)
        t_2287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 't')
        float_2288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'float')
        # Applying the binary operator '<' (line 83)
        result_lt_2289 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), '<', t_2287, float_2288)
        
        # Testing if the type of an if condition is none (line 83)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 8), result_lt_2289):
            pass
        else:
            
            # Testing the type of an if condition (line 83)
            if_condition_2290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 8), result_lt_2289)
            # Assigning a type to the variable 'if_condition_2290' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'if_condition_2290', if_condition_2290)
            # SSA begins for if statement (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            float_2291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'stypy_return_type', float_2291)
            # SSA join for if statement (line 83)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 85)
        t_2292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 't')
        # Assigning a type to the variable 'stypy_return_type' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', t_2292)
        
        # ################# End of 'get_intersection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_intersection' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_2293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2293)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_intersection'
        return stypy_return_type_2293


    @norecursion
    def get_sample_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_sample_point'
        module_type_store = module_type_store.open_function_context('get_sample_point', 87, 4, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Triangle.get_sample_point.__dict__.__setitem__('stypy_localization', localization)
        Triangle.get_sample_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Triangle.get_sample_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        Triangle.get_sample_point.__dict__.__setitem__('stypy_function_name', 'Triangle.get_sample_point')
        Triangle.get_sample_point.__dict__.__setitem__('stypy_param_names_list', [])
        Triangle.get_sample_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        Triangle.get_sample_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Triangle.get_sample_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        Triangle.get_sample_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        Triangle.get_sample_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Triangle.get_sample_point.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Triangle.get_sample_point', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_sample_point', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_sample_point(...)' code ##################

        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Call to sqrt(...): (line 88)
        # Processing the call arguments (line 88)
        
        # Call to random(...): (line 88)
        # Processing the call keyword arguments (line 88)
        kwargs_2296 = {}
        # Getting the type of 'random' (line 88)
        random_2295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'random', False)
        # Calling random(args, kwargs) (line 88)
        random_call_result_2297 = invoke(stypy.reporting.localization.Localization(__file__, 88, 20), random_2295, *[], **kwargs_2296)
        
        # Processing the call keyword arguments (line 88)
        kwargs_2298 = {}
        # Getting the type of 'sqrt' (line 88)
        sqrt_2294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 88)
        sqrt_call_result_2299 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), sqrt_2294, *[random_call_result_2297], **kwargs_2298)
        
        # Assigning a type to the variable 'sqr1' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'sqr1', sqrt_call_result_2299)
        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to random(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_2301 = {}
        # Getting the type of 'random' (line 89)
        random_2300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'random', False)
        # Calling random(args, kwargs) (line 89)
        random_call_result_2302 = invoke(stypy.reporting.localization.Localization(__file__, 89, 13), random_2300, *[], **kwargs_2301)
        
        # Assigning a type to the variable 'r2' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'r2', random_call_result_2302)
        
        # Assigning a BinOp to a Name (line 90):
        
        # Assigning a BinOp to a Name (line 90):
        float_2303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 12), 'float')
        # Getting the type of 'sqr1' (line 90)
        sqr1_2304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'sqr1')
        # Applying the binary operator '-' (line 90)
        result_sub_2305 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 12), '-', float_2303, sqr1_2304)
        
        # Assigning a type to the variable 'a' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'a', result_sub_2305)
        
        # Assigning a BinOp to a Name (line 91):
        
        # Assigning a BinOp to a Name (line 91):
        float_2306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 13), 'float')
        # Getting the type of 'r2' (line 91)
        r2_2307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'r2')
        # Applying the binary operator '-' (line 91)
        result_sub_2308 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 13), '-', float_2306, r2_2307)
        
        # Getting the type of 'sqr1' (line 91)
        sqr1_2309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'sqr1')
        # Applying the binary operator '*' (line 91)
        result_mul_2310 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 12), '*', result_sub_2308, sqr1_2309)
        
        # Assigning a type to the variable 'b' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'b', result_mul_2310)
        # Getting the type of 'self' (line 92)
        self_2311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'self')
        # Obtaining the member 'edge0' of a type (line 92)
        edge0_2312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 15), self_2311, 'edge0')
        # Getting the type of 'a' (line 92)
        a_2313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'a')
        # Applying the binary operator '*' (line 92)
        result_mul_2314 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 15), '*', edge0_2312, a_2313)
        
        # Getting the type of 'self' (line 92)
        self_2315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'self')
        # Obtaining the member 'edge3' of a type (line 92)
        edge3_2316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 32), self_2315, 'edge3')
        # Getting the type of 'b' (line 92)
        b_2317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 45), 'b')
        # Applying the binary operator '*' (line 92)
        result_mul_2318 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 32), '*', edge3_2316, b_2317)
        
        # Applying the binary operator '+' (line 92)
        result_add_2319 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 15), '+', result_mul_2314, result_mul_2318)
        
        
        # Obtaining the type of the subscript
        int_2320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 62), 'int')
        # Getting the type of 'self' (line 92)
        self_2321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 49), 'self')
        # Obtaining the member 'vertexs' of a type (line 92)
        vertexs_2322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 49), self_2321, 'vertexs')
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___2323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 49), vertexs_2322, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_2324 = invoke(stypy.reporting.localization.Localization(__file__, 92, 49), getitem___2323, int_2320)
        
        # Applying the binary operator '+' (line 92)
        result_add_2325 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 47), '+', result_add_2319, subscript_call_result_2324)
        
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', result_add_2325)
        
        # ################# End of 'get_sample_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_sample_point' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_2326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2326)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_sample_point'
        return stypy_return_type_2326


# Assigning a type to the variable 'Triangle' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'Triangle', Triangle)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
