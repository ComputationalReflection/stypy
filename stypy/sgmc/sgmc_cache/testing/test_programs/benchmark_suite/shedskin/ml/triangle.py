
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
import_1885 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'vector3f')

if (type(import_1885) is not StypyTypeError):

    if (import_1885 != 'pyd_module'):
        __import__(import_1885)
        sys_modules_1886 = sys.modules[import_1885]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'vector3f', sys_modules_1886.module_type_store, module_type_store, ['Vector3f_str', 'ZERO', 'ONE', 'MAX'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_1886, sys_modules_1886.module_type_store, module_type_store)
    else:
        from vector3f import Vector3f_str, ZERO, ONE, MAX

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'vector3f', None, module_type_store, ['Vector3f_str', 'ZERO', 'ONE', 'MAX'], [Vector3f_str, ZERO, ONE, MAX])

else:
    # Assigning a type to the variable 'vector3f' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'vector3f', import_1885)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/ml/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import re' statement (line 11)
import re

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 're', re, module_type_store)


# Assigning a Call to a Name (line 12):

# Assigning a Call to a Name (line 12):

# Call to compile(...): (line 12)
# Processing the call arguments (line 12)
str_1889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 20), 'str', '(\\(.+\\))\\s*(\\(.+\\))\\s*(\\(.+\\))\\s*(\\(.+\\))\\s*(\\(.+\\))')
# Processing the call keyword arguments (line 12)
kwargs_1890 = {}
# Getting the type of 're' (line 12)
re_1887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 're', False)
# Obtaining the member 'compile' of a type (line 12)
compile_1888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 9), re_1887, 'compile')
# Calling compile(args, kwargs) (line 12)
compile_call_result_1891 = invoke(stypy.reporting.localization.Localization(__file__, 12, 9), compile_1888, *[str_1889], **kwargs_1890)

# Assigning a type to the variable 'SEARCH' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'SEARCH', compile_call_result_1891)

# Assigning a BinOp to a Name (line 14):

# Assigning a BinOp to a Name (line 14):
float_1892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'float')
float_1893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'float')
# Applying the binary operator 'div' (line 14)
result_div_1894 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 12), 'div', float_1892, float_1893)

# Assigning a type to the variable 'TOLERANCE' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'TOLERANCE', result_div_1894)
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
        in_stream_1895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'in_stream')
        # Testing if the for loop is going to be iterated (line 19)
        # Testing the type of a for loop iterable (line 19)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 8), in_stream_1895)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 19, 8), in_stream_1895):
            # Getting the type of the for loop variable (line 19)
            for_loop_var_1896 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 8), in_stream_1895)
            # Assigning a type to the variable 'line' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'line', for_loop_var_1896)
            # SSA begins for a for statement (line 19)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to isspace(...): (line 20)
            # Processing the call keyword arguments (line 20)
            kwargs_1899 = {}
            # Getting the type of 'line' (line 20)
            line_1897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'line', False)
            # Obtaining the member 'isspace' of a type (line 20)
            isspace_1898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 19), line_1897, 'isspace')
            # Calling isspace(args, kwargs) (line 20)
            isspace_call_result_1900 = invoke(stypy.reporting.localization.Localization(__file__, 20, 19), isspace_1898, *[], **kwargs_1899)
            
            # Applying the 'not' unary operator (line 20)
            result_not__1901 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 15), 'not', isspace_call_result_1900)
            
            # Testing if the type of an if condition is none (line 20)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 20, 12), result_not__1901):
                pass
            else:
                
                # Testing the type of an if condition (line 20)
                if_condition_1902 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 12), result_not__1901)
                # Assigning a type to the variable 'if_condition_1902' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'if_condition_1902', if_condition_1902)
                # SSA begins for if statement (line 20)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 21):
                
                # Assigning a Call to a Name:
                
                # Call to groups(...): (line 21)
                # Processing the call keyword arguments (line 21)
                kwargs_1909 = {}
                
                # Call to search(...): (line 21)
                # Processing the call arguments (line 21)
                # Getting the type of 'line' (line 21)
                line_1905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 49), 'line', False)
                # Processing the call keyword arguments (line 21)
                kwargs_1906 = {}
                # Getting the type of 'SEARCH' (line 21)
                SEARCH_1903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 35), 'SEARCH', False)
                # Obtaining the member 'search' of a type (line 21)
                search_1904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 35), SEARCH_1903, 'search')
                # Calling search(args, kwargs) (line 21)
                search_call_result_1907 = invoke(stypy.reporting.localization.Localization(__file__, 21, 35), search_1904, *[line_1905], **kwargs_1906)
                
                # Obtaining the member 'groups' of a type (line 21)
                groups_1908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 35), search_call_result_1907, 'groups')
                # Calling groups(args, kwargs) (line 21)
                groups_call_result_1910 = invoke(stypy.reporting.localization.Localization(__file__, 21, 35), groups_1908, *[], **kwargs_1909)
                
                # Assigning a type to the variable 'call_assignment_1879' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1879', groups_call_result_1910)
                
                # Assigning a Call to a Name (line 21):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_1913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 16), 'int')
                # Processing the call keyword arguments
                kwargs_1914 = {}
                # Getting the type of 'call_assignment_1879' (line 21)
                call_assignment_1879_1911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1879', False)
                # Obtaining the member '__getitem__' of a type (line 21)
                getitem___1912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), call_assignment_1879_1911, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_1915 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1912, *[int_1913], **kwargs_1914)
                
                # Assigning a type to the variable 'call_assignment_1880' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1880', getitem___call_result_1915)
                
                # Assigning a Name to a Name (line 21):
                # Getting the type of 'call_assignment_1880' (line 21)
                call_assignment_1880_1916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1880')
                # Assigning a type to the variable 'v0' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'v0', call_assignment_1880_1916)
                
                # Assigning a Call to a Name (line 21):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_1919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 16), 'int')
                # Processing the call keyword arguments
                kwargs_1920 = {}
                # Getting the type of 'call_assignment_1879' (line 21)
                call_assignment_1879_1917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1879', False)
                # Obtaining the member '__getitem__' of a type (line 21)
                getitem___1918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), call_assignment_1879_1917, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_1921 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1918, *[int_1919], **kwargs_1920)
                
                # Assigning a type to the variable 'call_assignment_1881' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1881', getitem___call_result_1921)
                
                # Assigning a Name to a Name (line 21):
                # Getting the type of 'call_assignment_1881' (line 21)
                call_assignment_1881_1922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1881')
                # Assigning a type to the variable 'v1' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'v1', call_assignment_1881_1922)
                
                # Assigning a Call to a Name (line 21):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_1925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 16), 'int')
                # Processing the call keyword arguments
                kwargs_1926 = {}
                # Getting the type of 'call_assignment_1879' (line 21)
                call_assignment_1879_1923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1879', False)
                # Obtaining the member '__getitem__' of a type (line 21)
                getitem___1924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), call_assignment_1879_1923, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_1927 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1924, *[int_1925], **kwargs_1926)
                
                # Assigning a type to the variable 'call_assignment_1882' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1882', getitem___call_result_1927)
                
                # Assigning a Name to a Name (line 21):
                # Getting the type of 'call_assignment_1882' (line 21)
                call_assignment_1882_1928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1882')
                # Assigning a type to the variable 'v2' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 24), 'v2', call_assignment_1882_1928)
                
                # Assigning a Call to a Name (line 21):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_1931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 16), 'int')
                # Processing the call keyword arguments
                kwargs_1932 = {}
                # Getting the type of 'call_assignment_1879' (line 21)
                call_assignment_1879_1929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1879', False)
                # Obtaining the member '__getitem__' of a type (line 21)
                getitem___1930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), call_assignment_1879_1929, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_1933 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1930, *[int_1931], **kwargs_1932)
                
                # Assigning a type to the variable 'call_assignment_1883' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1883', getitem___call_result_1933)
                
                # Assigning a Name to a Name (line 21):
                # Getting the type of 'call_assignment_1883' (line 21)
                call_assignment_1883_1934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1883')
                # Assigning a type to the variable 'r' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 28), 'r', call_assignment_1883_1934)
                
                # Assigning a Call to a Name (line 21):
                
                # Call to __getitem__(...):
                # Processing the call arguments
                int_1937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 16), 'int')
                # Processing the call keyword arguments
                kwargs_1938 = {}
                # Getting the type of 'call_assignment_1879' (line 21)
                call_assignment_1879_1935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1879', False)
                # Obtaining the member '__getitem__' of a type (line 21)
                getitem___1936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), call_assignment_1879_1935, '__getitem__')
                # Calling __getitem__(args, kwargs)
                getitem___call_result_1939 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___1936, *[int_1937], **kwargs_1938)
                
                # Assigning a type to the variable 'call_assignment_1884' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1884', getitem___call_result_1939)
                
                # Assigning a Name to a Name (line 21):
                # Getting the type of 'call_assignment_1884' (line 21)
                call_assignment_1884_1940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'call_assignment_1884')
                # Assigning a type to the variable 'e' (line 21)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 31), 'e', call_assignment_1884_1940)
                
                # Assigning a List to a Attribute (line 22):
                
                # Assigning a List to a Attribute (line 22):
                
                # Obtaining an instance of the builtin type 'list' (line 22)
                list_1941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 31), 'list')
                # Adding type elements to the builtin type 'list' instance (line 22)
                # Adding element type (line 22)
                
                # Call to Vector3f_str(...): (line 22)
                # Processing the call arguments (line 22)
                # Getting the type of 'v0' (line 22)
                v0_1943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 45), 'v0', False)
                # Processing the call keyword arguments (line 22)
                kwargs_1944 = {}
                # Getting the type of 'Vector3f_str' (line 22)
                Vector3f_str_1942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 32), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 22)
                Vector3f_str_call_result_1945 = invoke(stypy.reporting.localization.Localization(__file__, 22, 32), Vector3f_str_1942, *[v0_1943], **kwargs_1944)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 31), list_1941, Vector3f_str_call_result_1945)
                # Adding element type (line 22)
                
                # Call to Vector3f_str(...): (line 22)
                # Processing the call arguments (line 22)
                # Getting the type of 'v1' (line 22)
                v1_1947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 63), 'v1', False)
                # Processing the call keyword arguments (line 22)
                kwargs_1948 = {}
                # Getting the type of 'Vector3f_str' (line 22)
                Vector3f_str_1946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 50), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 22)
                Vector3f_str_call_result_1949 = invoke(stypy.reporting.localization.Localization(__file__, 22, 50), Vector3f_str_1946, *[v1_1947], **kwargs_1948)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 31), list_1941, Vector3f_str_call_result_1949)
                # Adding element type (line 22)
                
                # Call to Vector3f_str(...): (line 22)
                # Processing the call arguments (line 22)
                # Getting the type of 'v2' (line 22)
                v2_1951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 81), 'v2', False)
                # Processing the call keyword arguments (line 22)
                kwargs_1952 = {}
                # Getting the type of 'Vector3f_str' (line 22)
                Vector3f_str_1950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 68), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 22)
                Vector3f_str_call_result_1953 = invoke(stypy.reporting.localization.Localization(__file__, 22, 68), Vector3f_str_1950, *[v2_1951], **kwargs_1952)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 31), list_1941, Vector3f_str_call_result_1953)
                
                # Getting the type of 'self' (line 22)
                self_1954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'self')
                # Setting the type of the member 'vertexs' of a type (line 22)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 16), self_1954, 'vertexs', list_1941)
                
                # Assigning a BinOp to a Attribute (line 23):
                
                # Assigning a BinOp to a Attribute (line 23):
                
                # Call to Vector3f_str(...): (line 23)
                # Processing the call arguments (line 23)
                # Getting the type of 'v1' (line 23)
                v1_1956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 42), 'v1', False)
                # Processing the call keyword arguments (line 23)
                kwargs_1957 = {}
                # Getting the type of 'Vector3f_str' (line 23)
                Vector3f_str_1955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 29), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 23)
                Vector3f_str_call_result_1958 = invoke(stypy.reporting.localization.Localization(__file__, 23, 29), Vector3f_str_1955, *[v1_1956], **kwargs_1957)
                
                
                # Call to Vector3f_str(...): (line 23)
                # Processing the call arguments (line 23)
                # Getting the type of 'v0' (line 23)
                v0_1960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 61), 'v0', False)
                # Processing the call keyword arguments (line 23)
                kwargs_1961 = {}
                # Getting the type of 'Vector3f_str' (line 23)
                Vector3f_str_1959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 48), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 23)
                Vector3f_str_call_result_1962 = invoke(stypy.reporting.localization.Localization(__file__, 23, 48), Vector3f_str_1959, *[v0_1960], **kwargs_1961)
                
                # Applying the binary operator '-' (line 23)
                result_sub_1963 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 29), '-', Vector3f_str_call_result_1958, Vector3f_str_call_result_1962)
                
                # Getting the type of 'self' (line 23)
                self_1964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'self')
                # Setting the type of the member 'edge0' of a type (line 23)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 16), self_1964, 'edge0', result_sub_1963)
                
                # Assigning a BinOp to a Attribute (line 24):
                
                # Assigning a BinOp to a Attribute (line 24):
                
                # Call to Vector3f_str(...): (line 24)
                # Processing the call arguments (line 24)
                # Getting the type of 'v2' (line 24)
                v2_1966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 42), 'v2', False)
                # Processing the call keyword arguments (line 24)
                kwargs_1967 = {}
                # Getting the type of 'Vector3f_str' (line 24)
                Vector3f_str_1965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 24)
                Vector3f_str_call_result_1968 = invoke(stypy.reporting.localization.Localization(__file__, 24, 29), Vector3f_str_1965, *[v2_1966], **kwargs_1967)
                
                
                # Call to Vector3f_str(...): (line 24)
                # Processing the call arguments (line 24)
                # Getting the type of 'v0' (line 24)
                v0_1970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 61), 'v0', False)
                # Processing the call keyword arguments (line 24)
                kwargs_1971 = {}
                # Getting the type of 'Vector3f_str' (line 24)
                Vector3f_str_1969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 48), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 24)
                Vector3f_str_call_result_1972 = invoke(stypy.reporting.localization.Localization(__file__, 24, 48), Vector3f_str_1969, *[v0_1970], **kwargs_1971)
                
                # Applying the binary operator '-' (line 24)
                result_sub_1973 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 29), '-', Vector3f_str_call_result_1968, Vector3f_str_call_result_1972)
                
                # Getting the type of 'self' (line 24)
                self_1974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'self')
                # Setting the type of the member 'edge3' of a type (line 24)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), self_1974, 'edge3', result_sub_1973)
                
                # Assigning a Call to a Attribute (line 25):
                
                # Assigning a Call to a Attribute (line 25):
                
                # Call to clamped(...): (line 25)
                # Processing the call arguments (line 25)
                # Getting the type of 'ZERO' (line 25)
                ZERO_1980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 60), 'ZERO', False)
                # Getting the type of 'ONE' (line 25)
                ONE_1981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 66), 'ONE', False)
                # Processing the call keyword arguments (line 25)
                kwargs_1982 = {}
                
                # Call to Vector3f_str(...): (line 25)
                # Processing the call arguments (line 25)
                # Getting the type of 'r' (line 25)
                r_1976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 49), 'r', False)
                # Processing the call keyword arguments (line 25)
                kwargs_1977 = {}
                # Getting the type of 'Vector3f_str' (line 25)
                Vector3f_str_1975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 36), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 25)
                Vector3f_str_call_result_1978 = invoke(stypy.reporting.localization.Localization(__file__, 25, 36), Vector3f_str_1975, *[r_1976], **kwargs_1977)
                
                # Obtaining the member 'clamped' of a type (line 25)
                clamped_1979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 36), Vector3f_str_call_result_1978, 'clamped')
                # Calling clamped(args, kwargs) (line 25)
                clamped_call_result_1983 = invoke(stypy.reporting.localization.Localization(__file__, 25, 36), clamped_1979, *[ZERO_1980, ONE_1981], **kwargs_1982)
                
                # Getting the type of 'self' (line 25)
                self_1984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'self')
                # Setting the type of the member 'reflectivity' of a type (line 25)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), self_1984, 'reflectivity', clamped_call_result_1983)
                
                # Assigning a Call to a Attribute (line 26):
                
                # Assigning a Call to a Attribute (line 26):
                
                # Call to clamped(...): (line 26)
                # Processing the call arguments (line 26)
                # Getting the type of 'ZERO' (line 26)
                ZERO_1990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 57), 'ZERO', False)
                # Getting the type of 'MAX' (line 26)
                MAX_1991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 63), 'MAX', False)
                # Processing the call keyword arguments (line 26)
                kwargs_1992 = {}
                
                # Call to Vector3f_str(...): (line 26)
                # Processing the call arguments (line 26)
                # Getting the type of 'e' (line 26)
                e_1986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 46), 'e', False)
                # Processing the call keyword arguments (line 26)
                kwargs_1987 = {}
                # Getting the type of 'Vector3f_str' (line 26)
                Vector3f_str_1985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 33), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 26)
                Vector3f_str_call_result_1988 = invoke(stypy.reporting.localization.Localization(__file__, 26, 33), Vector3f_str_1985, *[e_1986], **kwargs_1987)
                
                # Obtaining the member 'clamped' of a type (line 26)
                clamped_1989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 33), Vector3f_str_call_result_1988, 'clamped')
                # Calling clamped(args, kwargs) (line 26)
                clamped_call_result_1993 = invoke(stypy.reporting.localization.Localization(__file__, 26, 33), clamped_1989, *[ZERO_1990, MAX_1991], **kwargs_1992)
                
                # Getting the type of 'self' (line 26)
                self_1994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'self')
                # Setting the type of the member 'emitivity' of a type (line 26)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 16), self_1994, 'emitivity', clamped_call_result_1993)
                
                # Assigning a BinOp to a Name (line 27):
                
                # Assigning a BinOp to a Name (line 27):
                
                # Call to Vector3f_str(...): (line 27)
                # Processing the call arguments (line 27)
                # Getting the type of 'v2' (line 27)
                v2_1996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 37), 'v2', False)
                # Processing the call keyword arguments (line 27)
                kwargs_1997 = {}
                # Getting the type of 'Vector3f_str' (line 27)
                Vector3f_str_1995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 24), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 27)
                Vector3f_str_call_result_1998 = invoke(stypy.reporting.localization.Localization(__file__, 27, 24), Vector3f_str_1995, *[v2_1996], **kwargs_1997)
                
                
                # Call to Vector3f_str(...): (line 27)
                # Processing the call arguments (line 27)
                # Getting the type of 'v1' (line 27)
                v1_2000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 56), 'v1', False)
                # Processing the call keyword arguments (line 27)
                kwargs_2001 = {}
                # Getting the type of 'Vector3f_str' (line 27)
                Vector3f_str_1999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 43), 'Vector3f_str', False)
                # Calling Vector3f_str(args, kwargs) (line 27)
                Vector3f_str_call_result_2002 = invoke(stypy.reporting.localization.Localization(__file__, 27, 43), Vector3f_str_1999, *[v1_2000], **kwargs_2001)
                
                # Applying the binary operator '-' (line 27)
                result_sub_2003 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 24), '-', Vector3f_str_call_result_1998, Vector3f_str_call_result_2002)
                
                # Assigning a type to the variable 'edge1' (line 27)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'edge1', result_sub_2003)
                
                # Assigning a Call to a Attribute (line 28):
                
                # Assigning a Call to a Attribute (line 28):
                
                # Call to unitize(...): (line 28)
                # Processing the call keyword arguments (line 28)
                kwargs_2007 = {}
                # Getting the type of 'self' (line 28)
                self_2004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 31), 'self', False)
                # Obtaining the member 'edge0' of a type (line 28)
                edge0_2005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 31), self_2004, 'edge0')
                # Obtaining the member 'unitize' of a type (line 28)
                unitize_2006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 31), edge0_2005, 'unitize')
                # Calling unitize(args, kwargs) (line 28)
                unitize_call_result_2008 = invoke(stypy.reporting.localization.Localization(__file__, 28, 31), unitize_2006, *[], **kwargs_2007)
                
                # Getting the type of 'self' (line 28)
                self_2009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'self')
                # Setting the type of the member 'tangent' of a type (line 28)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), self_2009, 'tangent', unitize_call_result_2008)
                
                # Assigning a Call to a Attribute (line 29):
                
                # Assigning a Call to a Attribute (line 29):
                
                # Call to unitize(...): (line 29)
                # Processing the call keyword arguments (line 29)
                kwargs_2017 = {}
                
                # Call to cross(...): (line 29)
                # Processing the call arguments (line 29)
                # Getting the type of 'edge1' (line 29)
                edge1_2013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 49), 'edge1', False)
                # Processing the call keyword arguments (line 29)
                kwargs_2014 = {}
                # Getting the type of 'self' (line 29)
                self_2010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 30), 'self', False)
                # Obtaining the member 'tangent' of a type (line 29)
                tangent_2011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 30), self_2010, 'tangent')
                # Obtaining the member 'cross' of a type (line 29)
                cross_2012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 30), tangent_2011, 'cross')
                # Calling cross(args, kwargs) (line 29)
                cross_call_result_2015 = invoke(stypy.reporting.localization.Localization(__file__, 29, 30), cross_2012, *[edge1_2013], **kwargs_2014)
                
                # Obtaining the member 'unitize' of a type (line 29)
                unitize_2016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 30), cross_call_result_2015, 'unitize')
                # Calling unitize(args, kwargs) (line 29)
                unitize_call_result_2018 = invoke(stypy.reporting.localization.Localization(__file__, 29, 30), unitize_2016, *[], **kwargs_2017)
                
                # Getting the type of 'self' (line 29)
                self_2019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'self')
                # Setting the type of the member 'normal' of a type (line 29)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 16), self_2019, 'normal', unitize_call_result_2018)
                
                # Assigning a Call to a Name (line 30):
                
                # Assigning a Call to a Name (line 30):
                
                # Call to cross(...): (line 30)
                # Processing the call arguments (line 30)
                # Getting the type of 'edge1' (line 30)
                edge1_2023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 39), 'edge1', False)
                # Processing the call keyword arguments (line 30)
                kwargs_2024 = {}
                # Getting the type of 'self' (line 30)
                self_2020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'self', False)
                # Obtaining the member 'edge0' of a type (line 30)
                edge0_2021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 22), self_2020, 'edge0')
                # Obtaining the member 'cross' of a type (line 30)
                cross_2022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 22), edge0_2021, 'cross')
                # Calling cross(args, kwargs) (line 30)
                cross_call_result_2025 = invoke(stypy.reporting.localization.Localization(__file__, 30, 22), cross_2022, *[edge1_2023], **kwargs_2024)
                
                # Assigning a type to the variable 'pa2' (line 30)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'pa2', cross_call_result_2025)
                
                # Assigning a BinOp to a Attribute (line 31):
                
                # Assigning a BinOp to a Attribute (line 31):
                
                # Call to sqrt(...): (line 31)
                # Processing the call arguments (line 31)
                
                # Call to dot(...): (line 31)
                # Processing the call arguments (line 31)
                # Getting the type of 'pa2' (line 31)
                pa2_2029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 41), 'pa2', False)
                # Processing the call keyword arguments (line 31)
                kwargs_2030 = {}
                # Getting the type of 'pa2' (line 31)
                pa2_2027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'pa2', False)
                # Obtaining the member 'dot' of a type (line 31)
                dot_2028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 33), pa2_2027, 'dot')
                # Calling dot(args, kwargs) (line 31)
                dot_call_result_2031 = invoke(stypy.reporting.localization.Localization(__file__, 31, 33), dot_2028, *[pa2_2029], **kwargs_2030)
                
                # Processing the call keyword arguments (line 31)
                kwargs_2032 = {}
                # Getting the type of 'sqrt' (line 31)
                sqrt_2026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 28), 'sqrt', False)
                # Calling sqrt(args, kwargs) (line 31)
                sqrt_call_result_2033 = invoke(stypy.reporting.localization.Localization(__file__, 31, 28), sqrt_2026, *[dot_call_result_2031], **kwargs_2032)
                
                float_2034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 49), 'float')
                # Applying the binary operator '*' (line 31)
                result_mul_2035 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 28), '*', sqrt_call_result_2033, float_2034)
                
                # Getting the type of 'self' (line 31)
                self_2036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'self')
                # Setting the type of the member 'area' of a type (line 31)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), self_2036, 'area', result_mul_2035)
                # Assigning a type to the variable 'stypy_return_type' (line 32)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'stypy_return_type', types.NoneType)
                # SSA join for if statement (line 20)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'StopIteration' (line 33)
        StopIteration_2037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'StopIteration')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 33, 8), StopIteration_2037, 'raise parameter', BaseException)
        
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
        int_2038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 26), 'int')
        # Getting the type of 'self' (line 36)
        self_2039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'self')
        # Obtaining the member 'vertexs' of a type (line 36)
        vertexs_2040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), self_2039, 'vertexs')
        # Obtaining the member '__getitem__' of a type (line 36)
        getitem___2041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), vertexs_2040, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 36)
        subscript_call_result_2042 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), getitem___2041, int_2038)
        
        # Assigning a type to the variable 'v2' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'v2', subscript_call_result_2042)
        
        # Assigning a List to a Name (line 37):
        
        # Assigning a List to a Name (line 37):
        
        # Obtaining an instance of the builtin type 'list' (line 37)
        list_2043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 37)
        # Adding element type (line 37)
        # Getting the type of 'v2' (line 37)
        v2_2044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 17), 'v2')
        # Obtaining the member 'x' of a type (line 37)
        x_2045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 17), v2_2044, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_2043, x_2045)
        # Adding element type (line 37)
        # Getting the type of 'v2' (line 37)
        v2_2046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'v2')
        # Obtaining the member 'y' of a type (line 37)
        y_2047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 23), v2_2046, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_2043, y_2047)
        # Adding element type (line 37)
        # Getting the type of 'v2' (line 37)
        v2_2048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'v2')
        # Obtaining the member 'z' of a type (line 37)
        z_2049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 29), v2_2048, 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_2043, z_2049)
        # Adding element type (line 37)
        # Getting the type of 'v2' (line 37)
        v2_2050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 35), 'v2')
        # Obtaining the member 'x' of a type (line 37)
        x_2051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 35), v2_2050, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_2043, x_2051)
        # Adding element type (line 37)
        # Getting the type of 'v2' (line 37)
        v2_2052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 41), 'v2')
        # Obtaining the member 'y' of a type (line 37)
        y_2053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 41), v2_2052, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_2043, y_2053)
        # Adding element type (line 37)
        # Getting the type of 'v2' (line 37)
        v2_2054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 47), 'v2')
        # Obtaining the member 'z' of a type (line 37)
        z_2055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 47), v2_2054, 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 16), list_2043, z_2055)
        
        # Assigning a type to the variable 'bound' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'bound', list_2043)
        
        
        # Call to range(...): (line 38)
        # Processing the call arguments (line 38)
        int_2057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'int')
        # Processing the call keyword arguments (line 38)
        kwargs_2058 = {}
        # Getting the type of 'range' (line 38)
        range_2056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'range', False)
        # Calling range(args, kwargs) (line 38)
        range_call_result_2059 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), range_2056, *[int_2057], **kwargs_2058)
        
        # Testing if the for loop is going to be iterated (line 38)
        # Testing the type of a for loop iterable (line 38)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 8), range_call_result_2059)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 38, 8), range_call_result_2059):
            # Getting the type of the for loop variable (line 38)
            for_loop_var_2060 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 8), range_call_result_2059)
            # Assigning a type to the variable 'j' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'j', for_loop_var_2060)
            # SSA begins for a for statement (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 39):
            
            # Assigning a Subscript to a Name (line 39):
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 39)
            j_2061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 33), 'j')
            
            # Obtaining the type of the subscript
            int_2062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'int')
            # Getting the type of 'self' (line 39)
            self_2063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'self')
            # Obtaining the member 'vertexs' of a type (line 39)
            vertexs_2064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 17), self_2063, 'vertexs')
            # Obtaining the member '__getitem__' of a type (line 39)
            getitem___2065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 17), vertexs_2064, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 39)
            subscript_call_result_2066 = invoke(stypy.reporting.localization.Localization(__file__, 39, 17), getitem___2065, int_2062)
            
            # Obtaining the member '__getitem__' of a type (line 39)
            getitem___2067 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 17), subscript_call_result_2066, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 39)
            subscript_call_result_2068 = invoke(stypy.reporting.localization.Localization(__file__, 39, 17), getitem___2067, j_2061)
            
            # Assigning a type to the variable 'v0' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'v0', subscript_call_result_2068)
            
            # Assigning a Subscript to a Name (line 40):
            
            # Assigning a Subscript to a Name (line 40):
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 40)
            j_2069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 33), 'j')
            
            # Obtaining the type of the subscript
            int_2070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 30), 'int')
            # Getting the type of 'self' (line 40)
            self_2071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'self')
            # Obtaining the member 'vertexs' of a type (line 40)
            vertexs_2072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 17), self_2071, 'vertexs')
            # Obtaining the member '__getitem__' of a type (line 40)
            getitem___2073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 17), vertexs_2072, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 40)
            subscript_call_result_2074 = invoke(stypy.reporting.localization.Localization(__file__, 40, 17), getitem___2073, int_2070)
            
            # Obtaining the member '__getitem__' of a type (line 40)
            getitem___2075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 17), subscript_call_result_2074, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 40)
            subscript_call_result_2076 = invoke(stypy.reporting.localization.Localization(__file__, 40, 17), getitem___2075, j_2069)
            
            # Assigning a type to the variable 'v1' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'v1', subscript_call_result_2076)
            
            # Getting the type of 'v0' (line 41)
            v0_2077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'v0')
            # Getting the type of 'v1' (line 41)
            v1_2078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'v1')
            # Applying the binary operator '<' (line 41)
            result_lt_2079 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 15), '<', v0_2077, v1_2078)
            
            # Testing if the type of an if condition is none (line 41)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 12), result_lt_2079):
                
                # Getting the type of 'v1' (line 47)
                v1_2105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'v1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 47)
                j_2106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'j')
                # Getting the type of 'bound' (line 47)
                bound_2107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'bound')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___2108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), bound_2107, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_2109 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___2108, j_2106)
                
                # Applying the binary operator '<' (line 47)
                result_lt_2110 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), '<', v1_2105, subscript_call_result_2109)
                
                # Testing if the type of an if condition is none (line 47)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 47, 16), result_lt_2110):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 47)
                    if_condition_2111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 16), result_lt_2110)
                    # Assigning a type to the variable 'if_condition_2111' (line 47)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'if_condition_2111', if_condition_2111)
                    # SSA begins for if statement (line 47)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 48):
                    
                    # Assigning a Name to a Subscript (line 48):
                    # Getting the type of 'v1' (line 48)
                    v1_2112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'v1')
                    # Getting the type of 'bound' (line 48)
                    bound_2113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'bound')
                    # Getting the type of 'j' (line 48)
                    j_2114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'j')
                    # Storing an element on a container (line 48)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 20), bound_2113, (j_2114, v1_2112))
                    # SSA join for if statement (line 47)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'v0' (line 49)
                v0_2115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'v0')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 49)
                j_2116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'j')
                int_2117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 34), 'int')
                # Applying the binary operator '+' (line 49)
                result_add_2118 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 30), '+', j_2116, int_2117)
                
                # Getting the type of 'bound' (line 49)
                bound_2119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'bound')
                # Obtaining the member '__getitem__' of a type (line 49)
                getitem___2120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 24), bound_2119, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                subscript_call_result_2121 = invoke(stypy.reporting.localization.Localization(__file__, 49, 24), getitem___2120, result_add_2118)
                
                # Applying the binary operator '>' (line 49)
                result_gt_2122 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 19), '>', v0_2115, subscript_call_result_2121)
                
                # Testing if the type of an if condition is none (line 49)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 49, 16), result_gt_2122):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 49)
                    if_condition_2123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 16), result_gt_2122)
                    # Assigning a type to the variable 'if_condition_2123' (line 49)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'if_condition_2123', if_condition_2123)
                    # SSA begins for if statement (line 49)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 50):
                    
                    # Assigning a Name to a Subscript (line 50):
                    # Getting the type of 'v0' (line 50)
                    v0_2124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'v0')
                    # Getting the type of 'bound' (line 50)
                    bound_2125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'bound')
                    # Getting the type of 'j' (line 50)
                    j_2126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'j')
                    int_2127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'int')
                    # Applying the binary operator '+' (line 50)
                    result_add_2128 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 26), '+', j_2126, int_2127)
                    
                    # Storing an element on a container (line 50)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), bound_2125, (result_add_2128, v0_2124))
                    # SSA join for if statement (line 49)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 41)
                if_condition_2080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 12), result_lt_2079)
                # Assigning a type to the variable 'if_condition_2080' (line 41)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'if_condition_2080', if_condition_2080)
                # SSA begins for if statement (line 41)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'v0' (line 42)
                v0_2081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'v0')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 42)
                j_2082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 30), 'j')
                # Getting the type of 'bound' (line 42)
                bound_2083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'bound')
                # Obtaining the member '__getitem__' of a type (line 42)
                getitem___2084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), bound_2083, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 42)
                subscript_call_result_2085 = invoke(stypy.reporting.localization.Localization(__file__, 42, 24), getitem___2084, j_2082)
                
                # Applying the binary operator '<' (line 42)
                result_lt_2086 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 19), '<', v0_2081, subscript_call_result_2085)
                
                # Testing if the type of an if condition is none (line 42)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 16), result_lt_2086):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 42)
                    if_condition_2087 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 16), result_lt_2086)
                    # Assigning a type to the variable 'if_condition_2087' (line 42)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'if_condition_2087', if_condition_2087)
                    # SSA begins for if statement (line 42)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 43):
                    
                    # Assigning a Name to a Subscript (line 43):
                    # Getting the type of 'v0' (line 43)
                    v0_2088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 31), 'v0')
                    # Getting the type of 'bound' (line 43)
                    bound_2089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 20), 'bound')
                    # Getting the type of 'j' (line 43)
                    j_2090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 26), 'j')
                    # Storing an element on a container (line 43)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 20), bound_2089, (j_2090, v0_2088))
                    # SSA join for if statement (line 42)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'v1' (line 44)
                v1_2091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'v1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 44)
                j_2092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'j')
                int_2093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'int')
                # Applying the binary operator '+' (line 44)
                result_add_2094 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 30), '+', j_2092, int_2093)
                
                # Getting the type of 'bound' (line 44)
                bound_2095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'bound')
                # Obtaining the member '__getitem__' of a type (line 44)
                getitem___2096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 24), bound_2095, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 44)
                subscript_call_result_2097 = invoke(stypy.reporting.localization.Localization(__file__, 44, 24), getitem___2096, result_add_2094)
                
                # Applying the binary operator '>' (line 44)
                result_gt_2098 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 19), '>', v1_2091, subscript_call_result_2097)
                
                # Testing if the type of an if condition is none (line 44)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 16), result_gt_2098):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 44)
                    if_condition_2099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 16), result_gt_2098)
                    # Assigning a type to the variable 'if_condition_2099' (line 44)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'if_condition_2099', if_condition_2099)
                    # SSA begins for if statement (line 44)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 45):
                    
                    # Assigning a Name to a Subscript (line 45):
                    # Getting the type of 'v1' (line 45)
                    v1_2100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 35), 'v1')
                    # Getting the type of 'bound' (line 45)
                    bound_2101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'bound')
                    # Getting the type of 'j' (line 45)
                    j_2102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'j')
                    int_2103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 30), 'int')
                    # Applying the binary operator '+' (line 45)
                    result_add_2104 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 26), '+', j_2102, int_2103)
                    
                    # Storing an element on a container (line 45)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 20), bound_2101, (result_add_2104, v1_2100))
                    # SSA join for if statement (line 44)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA branch for the else part of an if statement (line 41)
                module_type_store.open_ssa_branch('else')
                
                # Getting the type of 'v1' (line 47)
                v1_2105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'v1')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 47)
                j_2106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 30), 'j')
                # Getting the type of 'bound' (line 47)
                bound_2107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'bound')
                # Obtaining the member '__getitem__' of a type (line 47)
                getitem___2108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), bound_2107, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 47)
                subscript_call_result_2109 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), getitem___2108, j_2106)
                
                # Applying the binary operator '<' (line 47)
                result_lt_2110 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), '<', v1_2105, subscript_call_result_2109)
                
                # Testing if the type of an if condition is none (line 47)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 47, 16), result_lt_2110):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 47)
                    if_condition_2111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 16), result_lt_2110)
                    # Assigning a type to the variable 'if_condition_2111' (line 47)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'if_condition_2111', if_condition_2111)
                    # SSA begins for if statement (line 47)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 48):
                    
                    # Assigning a Name to a Subscript (line 48):
                    # Getting the type of 'v1' (line 48)
                    v1_2112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 31), 'v1')
                    # Getting the type of 'bound' (line 48)
                    bound_2113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'bound')
                    # Getting the type of 'j' (line 48)
                    j_2114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 26), 'j')
                    # Storing an element on a container (line 48)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 20), bound_2113, (j_2114, v1_2112))
                    # SSA join for if statement (line 47)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'v0' (line 49)
                v0_2115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'v0')
                
                # Obtaining the type of the subscript
                # Getting the type of 'j' (line 49)
                j_2116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'j')
                int_2117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 34), 'int')
                # Applying the binary operator '+' (line 49)
                result_add_2118 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 30), '+', j_2116, int_2117)
                
                # Getting the type of 'bound' (line 49)
                bound_2119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'bound')
                # Obtaining the member '__getitem__' of a type (line 49)
                getitem___2120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 24), bound_2119, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 49)
                subscript_call_result_2121 = invoke(stypy.reporting.localization.Localization(__file__, 49, 24), getitem___2120, result_add_2118)
                
                # Applying the binary operator '>' (line 49)
                result_gt_2122 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 19), '>', v0_2115, subscript_call_result_2121)
                
                # Testing if the type of an if condition is none (line 49)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 49, 16), result_gt_2122):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 49)
                    if_condition_2123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 16), result_gt_2122)
                    # Assigning a type to the variable 'if_condition_2123' (line 49)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'if_condition_2123', if_condition_2123)
                    # SSA begins for if statement (line 49)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 50):
                    
                    # Assigning a Name to a Subscript (line 50):
                    # Getting the type of 'v0' (line 50)
                    v0_2124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 35), 'v0')
                    # Getting the type of 'bound' (line 50)
                    bound_2125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 20), 'bound')
                    # Getting the type of 'j' (line 50)
                    j_2126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'j')
                    int_2127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'int')
                    # Applying the binary operator '+' (line 50)
                    result_add_2128 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 26), '+', j_2126, int_2127)
                    
                    # Storing an element on a container (line 50)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 20), bound_2125, (result_add_2128, v0_2124))
                    # SSA join for if statement (line 49)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 41)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'bound' (line 51)
            bound_2129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'bound')
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 51)
            j_2130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'j')
            # Getting the type of 'bound' (line 51)
            bound_2131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'bound')
            # Obtaining the member '__getitem__' of a type (line 51)
            getitem___2132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), bound_2131, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 51)
            subscript_call_result_2133 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), getitem___2132, j_2130)
            
            
            # Call to abs(...): (line 51)
            # Processing the call arguments (line 51)
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 51)
            j_2135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 35), 'j', False)
            # Getting the type of 'bound' (line 51)
            bound_2136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'bound', False)
            # Obtaining the member '__getitem__' of a type (line 51)
            getitem___2137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 29), bound_2136, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 51)
            subscript_call_result_2138 = invoke(stypy.reporting.localization.Localization(__file__, 51, 29), getitem___2137, j_2135)
            
            # Processing the call keyword arguments (line 51)
            kwargs_2139 = {}
            # Getting the type of 'abs' (line 51)
            abs_2134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'abs', False)
            # Calling abs(args, kwargs) (line 51)
            abs_call_result_2140 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), abs_2134, *[subscript_call_result_2138], **kwargs_2139)
            
            float_2141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 41), 'float')
            # Applying the binary operator '+' (line 51)
            result_add_2142 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 25), '+', abs_call_result_2140, float_2141)
            
            # Getting the type of 'TOLERANCE' (line 51)
            TOLERANCE_2143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 48), 'TOLERANCE')
            # Applying the binary operator '*' (line 51)
            result_mul_2144 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 24), '*', result_add_2142, TOLERANCE_2143)
            
            # Applying the binary operator '-=' (line 51)
            result_isub_2145 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 12), '-=', subscript_call_result_2133, result_mul_2144)
            # Getting the type of 'bound' (line 51)
            bound_2146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'bound')
            # Getting the type of 'j' (line 51)
            j_2147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'j')
            # Storing an element on a container (line 51)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 12), bound_2146, (j_2147, result_isub_2145))
            
            
            # Getting the type of 'bound' (line 52)
            bound_2148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'bound')
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 52)
            j_2149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'j')
            int_2150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 22), 'int')
            # Applying the binary operator '+' (line 52)
            result_add_2151 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 18), '+', j_2149, int_2150)
            
            # Getting the type of 'bound' (line 52)
            bound_2152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'bound')
            # Obtaining the member '__getitem__' of a type (line 52)
            getitem___2153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), bound_2152, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 52)
            subscript_call_result_2154 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), getitem___2153, result_add_2151)
            
            
            # Call to abs(...): (line 52)
            # Processing the call arguments (line 52)
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 52)
            j_2156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 39), 'j', False)
            int_2157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 43), 'int')
            # Applying the binary operator '+' (line 52)
            result_add_2158 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 39), '+', j_2156, int_2157)
            
            # Getting the type of 'bound' (line 52)
            bound_2159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 33), 'bound', False)
            # Obtaining the member '__getitem__' of a type (line 52)
            getitem___2160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 33), bound_2159, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 52)
            subscript_call_result_2161 = invoke(stypy.reporting.localization.Localization(__file__, 52, 33), getitem___2160, result_add_2158)
            
            # Processing the call keyword arguments (line 52)
            kwargs_2162 = {}
            # Getting the type of 'abs' (line 52)
            abs_2155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 29), 'abs', False)
            # Calling abs(args, kwargs) (line 52)
            abs_call_result_2163 = invoke(stypy.reporting.localization.Localization(__file__, 52, 29), abs_2155, *[subscript_call_result_2161], **kwargs_2162)
            
            float_2164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 49), 'float')
            # Applying the binary operator '+' (line 52)
            result_add_2165 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 29), '+', abs_call_result_2163, float_2164)
            
            # Getting the type of 'TOLERANCE' (line 52)
            TOLERANCE_2166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 56), 'TOLERANCE')
            # Applying the binary operator '*' (line 52)
            result_mul_2167 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 28), '*', result_add_2165, TOLERANCE_2166)
            
            # Applying the binary operator '+=' (line 52)
            result_iadd_2168 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 12), '+=', subscript_call_result_2154, result_mul_2167)
            # Getting the type of 'bound' (line 52)
            bound_2169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'bound')
            # Getting the type of 'j' (line 52)
            j_2170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 18), 'j')
            int_2171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 22), 'int')
            # Applying the binary operator '+' (line 52)
            result_add_2172 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 18), '+', j_2170, int_2171)
            
            # Storing an element on a container (line 52)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), bound_2169, (result_add_2172, result_iadd_2168))
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'bound' (line 53)
        bound_2173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'bound')
        # Assigning a type to the variable 'stypy_return_type' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', bound_2173)
        
        # ################# End of 'get_bound(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_bound' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_2174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2174)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_bound'
        return stypy_return_type_2174


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
        self_2175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'self')
        # Obtaining the member 'edge0' of a type (line 56)
        edge0_2176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 14), self_2175, 'edge0')
        # Obtaining the member 'x' of a type (line 56)
        x_2177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 14), edge0_2176, 'x')
        # Assigning a type to the variable 'e1x' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'e1x', x_2177)
        
        # Assigning a Attribute to a Name (line 56):
        
        # Assigning a Attribute to a Name (line 56):
        # Getting the type of 'self' (line 56)
        self_2178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 34), 'self')
        # Obtaining the member 'edge0' of a type (line 56)
        edge0_2179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 34), self_2178, 'edge0')
        # Obtaining the member 'y' of a type (line 56)
        y_2180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 34), edge0_2179, 'y')
        # Assigning a type to the variable 'e1y' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 28), 'e1y', y_2180)
        
        # Assigning a Attribute to a Name (line 56):
        
        # Assigning a Attribute to a Name (line 56):
        # Getting the type of 'self' (line 56)
        self_2181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 54), 'self')
        # Obtaining the member 'edge0' of a type (line 56)
        edge0_2182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 54), self_2181, 'edge0')
        # Obtaining the member 'z' of a type (line 56)
        z_2183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 54), edge0_2182, 'z')
        # Assigning a type to the variable 'e1z' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 48), 'e1z', z_2183)
        
        # Assigning a Attribute to a Name (line 57):
        
        # Assigning a Attribute to a Name (line 57):
        # Getting the type of 'self' (line 57)
        self_2184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'self')
        # Obtaining the member 'edge3' of a type (line 57)
        edge3_2185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 14), self_2184, 'edge3')
        # Obtaining the member 'x' of a type (line 57)
        x_2186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 14), edge3_2185, 'x')
        # Assigning a type to the variable 'e2x' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'e2x', x_2186)
        
        # Assigning a Attribute to a Name (line 57):
        
        # Assigning a Attribute to a Name (line 57):
        # Getting the type of 'self' (line 57)
        self_2187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'self')
        # Obtaining the member 'edge3' of a type (line 57)
        edge3_2188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 34), self_2187, 'edge3')
        # Obtaining the member 'y' of a type (line 57)
        y_2189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 34), edge3_2188, 'y')
        # Assigning a type to the variable 'e2y' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'e2y', y_2189)
        
        # Assigning a Attribute to a Name (line 57):
        
        # Assigning a Attribute to a Name (line 57):
        # Getting the type of 'self' (line 57)
        self_2190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 54), 'self')
        # Obtaining the member 'edge3' of a type (line 57)
        edge3_2191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 54), self_2190, 'edge3')
        # Obtaining the member 'z' of a type (line 57)
        z_2192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 54), edge3_2191, 'z')
        # Assigning a type to the variable 'e2z' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 48), 'e2z', z_2192)
        
        # Assigning a BinOp to a Name (line 58):
        
        # Assigning a BinOp to a Name (line 58):
        # Getting the type of 'ray_direction' (line 58)
        ray_direction_2193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 14), 'ray_direction')
        # Obtaining the member 'y' of a type (line 58)
        y_2194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 14), ray_direction_2193, 'y')
        # Getting the type of 'e2z' (line 58)
        e2z_2195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'e2z')
        # Applying the binary operator '*' (line 58)
        result_mul_2196 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 14), '*', y_2194, e2z_2195)
        
        # Getting the type of 'ray_direction' (line 58)
        ray_direction_2197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 38), 'ray_direction')
        # Obtaining the member 'z' of a type (line 58)
        z_2198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 38), ray_direction_2197, 'z')
        # Getting the type of 'e2y' (line 58)
        e2y_2199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 56), 'e2y')
        # Applying the binary operator '*' (line 58)
        result_mul_2200 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 38), '*', z_2198, e2y_2199)
        
        # Applying the binary operator '-' (line 58)
        result_sub_2201 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 14), '-', result_mul_2196, result_mul_2200)
        
        # Assigning a type to the variable 'pvx' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'pvx', result_sub_2201)
        
        # Assigning a BinOp to a Name (line 59):
        
        # Assigning a BinOp to a Name (line 59):
        # Getting the type of 'ray_direction' (line 59)
        ray_direction_2202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 14), 'ray_direction')
        # Obtaining the member 'z' of a type (line 59)
        z_2203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 14), ray_direction_2202, 'z')
        # Getting the type of 'e2x' (line 59)
        e2x_2204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'e2x')
        # Applying the binary operator '*' (line 59)
        result_mul_2205 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 14), '*', z_2203, e2x_2204)
        
        # Getting the type of 'ray_direction' (line 59)
        ray_direction_2206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 38), 'ray_direction')
        # Obtaining the member 'x' of a type (line 59)
        x_2207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 38), ray_direction_2206, 'x')
        # Getting the type of 'e2z' (line 59)
        e2z_2208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 56), 'e2z')
        # Applying the binary operator '*' (line 59)
        result_mul_2209 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 38), '*', x_2207, e2z_2208)
        
        # Applying the binary operator '-' (line 59)
        result_sub_2210 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 14), '-', result_mul_2205, result_mul_2209)
        
        # Assigning a type to the variable 'pvy' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'pvy', result_sub_2210)
        
        # Assigning a BinOp to a Name (line 60):
        
        # Assigning a BinOp to a Name (line 60):
        # Getting the type of 'ray_direction' (line 60)
        ray_direction_2211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 14), 'ray_direction')
        # Obtaining the member 'x' of a type (line 60)
        x_2212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 14), ray_direction_2211, 'x')
        # Getting the type of 'e2y' (line 60)
        e2y_2213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 32), 'e2y')
        # Applying the binary operator '*' (line 60)
        result_mul_2214 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 14), '*', x_2212, e2y_2213)
        
        # Getting the type of 'ray_direction' (line 60)
        ray_direction_2215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'ray_direction')
        # Obtaining the member 'y' of a type (line 60)
        y_2216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 38), ray_direction_2215, 'y')
        # Getting the type of 'e2x' (line 60)
        e2x_2217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 56), 'e2x')
        # Applying the binary operator '*' (line 60)
        result_mul_2218 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 38), '*', y_2216, e2x_2217)
        
        # Applying the binary operator '-' (line 60)
        result_sub_2219 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 14), '-', result_mul_2214, result_mul_2218)
        
        # Assigning a type to the variable 'pvz' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'pvz', result_sub_2219)
        
        # Assigning a BinOp to a Name (line 61):
        
        # Assigning a BinOp to a Name (line 61):
        # Getting the type of 'e1x' (line 61)
        e1x_2220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'e1x')
        # Getting the type of 'pvx' (line 61)
        pvx_2221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'pvx')
        # Applying the binary operator '*' (line 61)
        result_mul_2222 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 14), '*', e1x_2220, pvx_2221)
        
        # Getting the type of 'e1y' (line 61)
        e1y_2223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'e1y')
        # Getting the type of 'pvy' (line 61)
        pvy_2224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 32), 'pvy')
        # Applying the binary operator '*' (line 61)
        result_mul_2225 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 26), '*', e1y_2223, pvy_2224)
        
        # Applying the binary operator '+' (line 61)
        result_add_2226 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 14), '+', result_mul_2222, result_mul_2225)
        
        # Getting the type of 'e1z' (line 61)
        e1z_2227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 38), 'e1z')
        # Getting the type of 'pvz' (line 61)
        pvz_2228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 44), 'pvz')
        # Applying the binary operator '*' (line 61)
        result_mul_2229 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 38), '*', e1z_2227, pvz_2228)
        
        # Applying the binary operator '+' (line 61)
        result_add_2230 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 36), '+', result_add_2226, result_mul_2229)
        
        # Assigning a type to the variable 'det' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'det', result_add_2230)
        
        float_2231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 11), 'float')
        # Getting the type of 'det' (line 62)
        det_2232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'det')
        # Applying the binary operator '<' (line 62)
        result_lt_2233 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), '<', float_2231, det_2232)
        float_2234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 29), 'float')
        # Applying the binary operator '<' (line 62)
        result_lt_2235 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), '<', det_2232, float_2234)
        # Applying the binary operator '&' (line 62)
        result_and__2236 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), '&', result_lt_2233, result_lt_2235)
        
        # Testing if the type of an if condition is none (line 62)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 62, 8), result_and__2236):
            pass
        else:
            
            # Testing the type of an if condition (line 62)
            if_condition_2237 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 8), result_and__2236)
            # Assigning a type to the variable 'if_condition_2237' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'if_condition_2237', if_condition_2237)
            # SSA begins for if statement (line 62)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            float_2238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'stypy_return_type', float_2238)
            # SSA join for if statement (line 62)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 64):
        
        # Assigning a BinOp to a Name (line 64):
        float_2239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 18), 'float')
        # Getting the type of 'det' (line 64)
        det_2240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'det')
        # Applying the binary operator 'div' (line 64)
        result_div_2241 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 18), 'div', float_2239, det_2240)
        
        # Assigning a type to the variable 'inv_det' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'inv_det', result_div_2241)
        
        # Assigning a Subscript to a Name (line 65):
        
        # Assigning a Subscript to a Name (line 65):
        
        # Obtaining the type of the subscript
        int_2242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 26), 'int')
        # Getting the type of 'self' (line 65)
        self_2243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'self')
        # Obtaining the member 'vertexs' of a type (line 65)
        vertexs_2244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 13), self_2243, 'vertexs')
        # Obtaining the member '__getitem__' of a type (line 65)
        getitem___2245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 13), vertexs_2244, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 65)
        subscript_call_result_2246 = invoke(stypy.reporting.localization.Localization(__file__, 65, 13), getitem___2245, int_2242)
        
        # Assigning a type to the variable 'v0' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'v0', subscript_call_result_2246)
        
        # Assigning a BinOp to a Name (line 66):
        
        # Assigning a BinOp to a Name (line 66):
        # Getting the type of 'ray_origin' (line 66)
        ray_origin_2247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 14), 'ray_origin')
        # Obtaining the member 'x' of a type (line 66)
        x_2248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 14), ray_origin_2247, 'x')
        # Getting the type of 'v0' (line 66)
        v0_2249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 29), 'v0')
        # Obtaining the member 'x' of a type (line 66)
        x_2250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 29), v0_2249, 'x')
        # Applying the binary operator '-' (line 66)
        result_sub_2251 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 14), '-', x_2248, x_2250)
        
        # Assigning a type to the variable 'tvx' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'tvx', result_sub_2251)
        
        # Assigning a BinOp to a Name (line 67):
        
        # Assigning a BinOp to a Name (line 67):
        # Getting the type of 'ray_origin' (line 67)
        ray_origin_2252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 14), 'ray_origin')
        # Obtaining the member 'y' of a type (line 67)
        y_2253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 14), ray_origin_2252, 'y')
        # Getting the type of 'v0' (line 67)
        v0_2254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'v0')
        # Obtaining the member 'y' of a type (line 67)
        y_2255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 29), v0_2254, 'y')
        # Applying the binary operator '-' (line 67)
        result_sub_2256 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 14), '-', y_2253, y_2255)
        
        # Assigning a type to the variable 'tvy' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'tvy', result_sub_2256)
        
        # Assigning a BinOp to a Name (line 68):
        
        # Assigning a BinOp to a Name (line 68):
        # Getting the type of 'ray_origin' (line 68)
        ray_origin_2257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'ray_origin')
        # Obtaining the member 'z' of a type (line 68)
        z_2258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 14), ray_origin_2257, 'z')
        # Getting the type of 'v0' (line 68)
        v0_2259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 29), 'v0')
        # Obtaining the member 'z' of a type (line 68)
        z_2260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 29), v0_2259, 'z')
        # Applying the binary operator '-' (line 68)
        result_sub_2261 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 14), '-', z_2258, z_2260)
        
        # Assigning a type to the variable 'tvz' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'tvz', result_sub_2261)
        
        # Assigning a BinOp to a Name (line 69):
        
        # Assigning a BinOp to a Name (line 69):
        # Getting the type of 'tvx' (line 69)
        tvx_2262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'tvx')
        # Getting the type of 'pvx' (line 69)
        pvx_2263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'pvx')
        # Applying the binary operator '*' (line 69)
        result_mul_2264 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '*', tvx_2262, pvx_2263)
        
        # Getting the type of 'tvy' (line 69)
        tvy_2265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'tvy')
        # Getting the type of 'pvy' (line 69)
        pvy_2266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 31), 'pvy')
        # Applying the binary operator '*' (line 69)
        result_mul_2267 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 25), '*', tvy_2265, pvy_2266)
        
        # Applying the binary operator '+' (line 69)
        result_add_2268 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '+', result_mul_2264, result_mul_2267)
        
        # Getting the type of 'tvz' (line 69)
        tvz_2269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 37), 'tvz')
        # Getting the type of 'pvz' (line 69)
        pvz_2270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), 'pvz')
        # Applying the binary operator '*' (line 69)
        result_mul_2271 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 37), '*', tvz_2269, pvz_2270)
        
        # Applying the binary operator '+' (line 69)
        result_add_2272 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 35), '+', result_add_2268, result_mul_2271)
        
        # Getting the type of 'inv_det' (line 69)
        inv_det_2273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 50), 'inv_det')
        # Applying the binary operator '*' (line 69)
        result_mul_2274 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 12), '*', result_add_2272, inv_det_2273)
        
        # Assigning a type to the variable 'u' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'u', result_mul_2274)
        
        # Getting the type of 'u' (line 70)
        u_2275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'u')
        float_2276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 15), 'float')
        # Applying the binary operator '<' (line 70)
        result_lt_2277 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 11), '<', u_2275, float_2276)
        
        # Testing if the type of an if condition is none (line 70)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 8), result_lt_2277):
            
            # Getting the type of 'u' (line 72)
            u_2280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'u')
            float_2281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 17), 'float')
            # Applying the binary operator '>' (line 72)
            result_gt_2282 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '>', u_2280, float_2281)
            
            # Testing if the type of an if condition is none (line 72)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 13), result_gt_2282):
                pass
            else:
                
                # Testing the type of an if condition (line 72)
                if_condition_2283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_gt_2282)
                # Assigning a type to the variable 'if_condition_2283' (line 72)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_2283', if_condition_2283)
                # SSA begins for if statement (line 72)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                float_2284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'float')
                # Assigning a type to the variable 'stypy_return_type' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', float_2284)
                # SSA join for if statement (line 72)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 70)
            if_condition_2278 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 8), result_lt_2277)
            # Assigning a type to the variable 'if_condition_2278' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'if_condition_2278', if_condition_2278)
            # SSA begins for if statement (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            float_2279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'stypy_return_type', float_2279)
            # SSA branch for the else part of an if statement (line 70)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'u' (line 72)
            u_2280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'u')
            float_2281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 17), 'float')
            # Applying the binary operator '>' (line 72)
            result_gt_2282 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 13), '>', u_2280, float_2281)
            
            # Testing if the type of an if condition is none (line 72)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 13), result_gt_2282):
                pass
            else:
                
                # Testing the type of an if condition (line 72)
                if_condition_2283 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 13), result_gt_2282)
                # Assigning a type to the variable 'if_condition_2283' (line 72)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'if_condition_2283', if_condition_2283)
                # SSA begins for if statement (line 72)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                float_2284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 19), 'float')
                # Assigning a type to the variable 'stypy_return_type' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'stypy_return_type', float_2284)
                # SSA join for if statement (line 72)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 70)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 74):
        
        # Assigning a BinOp to a Name (line 74):
        # Getting the type of 'tvy' (line 74)
        tvy_2285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 14), 'tvy')
        # Getting the type of 'e1z' (line 74)
        e1z_2286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'e1z')
        # Applying the binary operator '*' (line 74)
        result_mul_2287 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 14), '*', tvy_2285, e1z_2286)
        
        # Getting the type of 'tvz' (line 74)
        tvz_2288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'tvz')
        # Getting the type of 'e1y' (line 74)
        e1y_2289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 32), 'e1y')
        # Applying the binary operator '*' (line 74)
        result_mul_2290 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 26), '*', tvz_2288, e1y_2289)
        
        # Applying the binary operator '-' (line 74)
        result_sub_2291 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 14), '-', result_mul_2287, result_mul_2290)
        
        # Assigning a type to the variable 'qvx' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'qvx', result_sub_2291)
        
        # Assigning a BinOp to a Name (line 75):
        
        # Assigning a BinOp to a Name (line 75):
        # Getting the type of 'tvz' (line 75)
        tvz_2292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'tvz')
        # Getting the type of 'e1x' (line 75)
        e1x_2293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'e1x')
        # Applying the binary operator '*' (line 75)
        result_mul_2294 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 14), '*', tvz_2292, e1x_2293)
        
        # Getting the type of 'tvx' (line 75)
        tvx_2295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'tvx')
        # Getting the type of 'e1z' (line 75)
        e1z_2296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'e1z')
        # Applying the binary operator '*' (line 75)
        result_mul_2297 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 26), '*', tvx_2295, e1z_2296)
        
        # Applying the binary operator '-' (line 75)
        result_sub_2298 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 14), '-', result_mul_2294, result_mul_2297)
        
        # Assigning a type to the variable 'qvy' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'qvy', result_sub_2298)
        
        # Assigning a BinOp to a Name (line 76):
        
        # Assigning a BinOp to a Name (line 76):
        # Getting the type of 'tvx' (line 76)
        tvx_2299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 14), 'tvx')
        # Getting the type of 'e1y' (line 76)
        e1y_2300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'e1y')
        # Applying the binary operator '*' (line 76)
        result_mul_2301 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 14), '*', tvx_2299, e1y_2300)
        
        # Getting the type of 'tvy' (line 76)
        tvy_2302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'tvy')
        # Getting the type of 'e1x' (line 76)
        e1x_2303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 32), 'e1x')
        # Applying the binary operator '*' (line 76)
        result_mul_2304 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 26), '*', tvy_2302, e1x_2303)
        
        # Applying the binary operator '-' (line 76)
        result_sub_2305 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 14), '-', result_mul_2301, result_mul_2304)
        
        # Assigning a type to the variable 'qvz' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'qvz', result_sub_2305)
        
        # Assigning a BinOp to a Name (line 77):
        
        # Assigning a BinOp to a Name (line 77):
        # Getting the type of 'ray_direction' (line 77)
        ray_direction_2306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'ray_direction')
        # Obtaining the member 'x' of a type (line 77)
        x_2307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), ray_direction_2306, 'x')
        # Getting the type of 'qvx' (line 77)
        qvx_2308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 31), 'qvx')
        # Applying the binary operator '*' (line 77)
        result_mul_2309 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 13), '*', x_2307, qvx_2308)
        
        # Getting the type of 'ray_direction' (line 77)
        ray_direction_2310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), 'ray_direction')
        # Obtaining the member 'y' of a type (line 77)
        y_2311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 37), ray_direction_2310, 'y')
        # Getting the type of 'qvy' (line 77)
        qvy_2312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 55), 'qvy')
        # Applying the binary operator '*' (line 77)
        result_mul_2313 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 37), '*', y_2311, qvy_2312)
        
        # Applying the binary operator '+' (line 77)
        result_add_2314 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 13), '+', result_mul_2309, result_mul_2313)
        
        # Getting the type of 'ray_direction' (line 77)
        ray_direction_2315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 61), 'ray_direction')
        # Obtaining the member 'z' of a type (line 77)
        z_2316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 61), ray_direction_2315, 'z')
        # Getting the type of 'qvz' (line 77)
        qvz_2317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 79), 'qvz')
        # Applying the binary operator '*' (line 77)
        result_mul_2318 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 61), '*', z_2316, qvz_2317)
        
        # Applying the binary operator '+' (line 77)
        result_add_2319 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 59), '+', result_add_2314, result_mul_2318)
        
        # Getting the type of 'inv_det' (line 77)
        inv_det_2320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 86), 'inv_det')
        # Applying the binary operator '*' (line 77)
        result_mul_2321 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 12), '*', result_add_2319, inv_det_2320)
        
        # Assigning a type to the variable 'v' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'v', result_mul_2321)
        
        # Getting the type of 'v' (line 78)
        v_2322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'v')
        float_2323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 15), 'float')
        # Applying the binary operator '<' (line 78)
        result_lt_2324 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), '<', v_2322, float_2323)
        
        # Testing if the type of an if condition is none (line 78)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 78, 8), result_lt_2324):
            
            # Getting the type of 'u' (line 80)
            u_2327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'u')
            # Getting the type of 'v' (line 80)
            v_2328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'v')
            # Applying the binary operator '+' (line 80)
            result_add_2329 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '+', u_2327, v_2328)
            
            float_2330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'float')
            # Applying the binary operator '>' (line 80)
            result_gt_2331 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '>', result_add_2329, float_2330)
            
            # Testing if the type of an if condition is none (line 80)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 13), result_gt_2331):
                pass
            else:
                
                # Testing the type of an if condition (line 80)
                if_condition_2332 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 13), result_gt_2331)
                # Assigning a type to the variable 'if_condition_2332' (line 80)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'if_condition_2332', if_condition_2332)
                # SSA begins for if statement (line 80)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                float_2333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 19), 'float')
                # Assigning a type to the variable 'stypy_return_type' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stypy_return_type', float_2333)
                # SSA join for if statement (line 80)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 78)
            if_condition_2325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_lt_2324)
            # Assigning a type to the variable 'if_condition_2325' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_2325', if_condition_2325)
            # SSA begins for if statement (line 78)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            float_2326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'stypy_return_type', float_2326)
            # SSA branch for the else part of an if statement (line 78)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'u' (line 80)
            u_2327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'u')
            # Getting the type of 'v' (line 80)
            v_2328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'v')
            # Applying the binary operator '+' (line 80)
            result_add_2329 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '+', u_2327, v_2328)
            
            float_2330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'float')
            # Applying the binary operator '>' (line 80)
            result_gt_2331 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 13), '>', result_add_2329, float_2330)
            
            # Testing if the type of an if condition is none (line 80)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 13), result_gt_2331):
                pass
            else:
                
                # Testing the type of an if condition (line 80)
                if_condition_2332 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 13), result_gt_2331)
                # Assigning a type to the variable 'if_condition_2332' (line 80)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 13), 'if_condition_2332', if_condition_2332)
                # SSA begins for if statement (line 80)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                float_2333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 19), 'float')
                # Assigning a type to the variable 'stypy_return_type' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stypy_return_type', float_2333)
                # SSA join for if statement (line 80)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 78)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 82):
        
        # Assigning a BinOp to a Name (line 82):
        # Getting the type of 'e2x' (line 82)
        e2x_2334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'e2x')
        # Getting the type of 'qvx' (line 82)
        qvx_2335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'qvx')
        # Applying the binary operator '*' (line 82)
        result_mul_2336 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 13), '*', e2x_2334, qvx_2335)
        
        # Getting the type of 'e2y' (line 82)
        e2y_2337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 25), 'e2y')
        # Getting the type of 'qvy' (line 82)
        qvy_2338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 31), 'qvy')
        # Applying the binary operator '*' (line 82)
        result_mul_2339 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 25), '*', e2y_2337, qvy_2338)
        
        # Applying the binary operator '+' (line 82)
        result_add_2340 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 13), '+', result_mul_2336, result_mul_2339)
        
        # Getting the type of 'e2z' (line 82)
        e2z_2341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'e2z')
        # Getting the type of 'qvz' (line 82)
        qvz_2342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), 'qvz')
        # Applying the binary operator '*' (line 82)
        result_mul_2343 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 37), '*', e2z_2341, qvz_2342)
        
        # Applying the binary operator '+' (line 82)
        result_add_2344 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 35), '+', result_add_2340, result_mul_2343)
        
        # Getting the type of 'inv_det' (line 82)
        inv_det_2345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 50), 'inv_det')
        # Applying the binary operator '*' (line 82)
        result_mul_2346 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 12), '*', result_add_2344, inv_det_2345)
        
        # Assigning a type to the variable 't' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 't', result_mul_2346)
        
        # Getting the type of 't' (line 83)
        t_2347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 't')
        float_2348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'float')
        # Applying the binary operator '<' (line 83)
        result_lt_2349 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), '<', t_2347, float_2348)
        
        # Testing if the type of an if condition is none (line 83)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 8), result_lt_2349):
            pass
        else:
            
            # Testing the type of an if condition (line 83)
            if_condition_2350 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 8), result_lt_2349)
            # Assigning a type to the variable 'if_condition_2350' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'if_condition_2350', if_condition_2350)
            # SSA begins for if statement (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            float_2351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 19), 'float')
            # Assigning a type to the variable 'stypy_return_type' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'stypy_return_type', float_2351)
            # SSA join for if statement (line 83)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 't' (line 85)
        t_2352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 't')
        # Assigning a type to the variable 'stypy_return_type' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', t_2352)
        
        # ################# End of 'get_intersection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_intersection' in the type store
        # Getting the type of 'stypy_return_type' (line 55)
        stypy_return_type_2353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2353)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_intersection'
        return stypy_return_type_2353


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
        kwargs_2356 = {}
        # Getting the type of 'random' (line 88)
        random_2355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'random', False)
        # Calling random(args, kwargs) (line 88)
        random_call_result_2357 = invoke(stypy.reporting.localization.Localization(__file__, 88, 20), random_2355, *[], **kwargs_2356)
        
        # Processing the call keyword arguments (line 88)
        kwargs_2358 = {}
        # Getting the type of 'sqrt' (line 88)
        sqrt_2354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 88)
        sqrt_call_result_2359 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), sqrt_2354, *[random_call_result_2357], **kwargs_2358)
        
        # Assigning a type to the variable 'sqr1' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'sqr1', sqrt_call_result_2359)
        
        # Assigning a Call to a Name (line 89):
        
        # Assigning a Call to a Name (line 89):
        
        # Call to random(...): (line 89)
        # Processing the call keyword arguments (line 89)
        kwargs_2361 = {}
        # Getting the type of 'random' (line 89)
        random_2360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 13), 'random', False)
        # Calling random(args, kwargs) (line 89)
        random_call_result_2362 = invoke(stypy.reporting.localization.Localization(__file__, 89, 13), random_2360, *[], **kwargs_2361)
        
        # Assigning a type to the variable 'r2' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'r2', random_call_result_2362)
        
        # Assigning a BinOp to a Name (line 90):
        
        # Assigning a BinOp to a Name (line 90):
        float_2363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 12), 'float')
        # Getting the type of 'sqr1' (line 90)
        sqr1_2364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'sqr1')
        # Applying the binary operator '-' (line 90)
        result_sub_2365 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 12), '-', float_2363, sqr1_2364)
        
        # Assigning a type to the variable 'a' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'a', result_sub_2365)
        
        # Assigning a BinOp to a Name (line 91):
        
        # Assigning a BinOp to a Name (line 91):
        float_2366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 13), 'float')
        # Getting the type of 'r2' (line 91)
        r2_2367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'r2')
        # Applying the binary operator '-' (line 91)
        result_sub_2368 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 13), '-', float_2366, r2_2367)
        
        # Getting the type of 'sqr1' (line 91)
        sqr1_2369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'sqr1')
        # Applying the binary operator '*' (line 91)
        result_mul_2370 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 12), '*', result_sub_2368, sqr1_2369)
        
        # Assigning a type to the variable 'b' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'b', result_mul_2370)
        # Getting the type of 'self' (line 92)
        self_2371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 15), 'self')
        # Obtaining the member 'edge0' of a type (line 92)
        edge0_2372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 15), self_2371, 'edge0')
        # Getting the type of 'a' (line 92)
        a_2373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'a')
        # Applying the binary operator '*' (line 92)
        result_mul_2374 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 15), '*', edge0_2372, a_2373)
        
        # Getting the type of 'self' (line 92)
        self_2375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'self')
        # Obtaining the member 'edge3' of a type (line 92)
        edge3_2376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 32), self_2375, 'edge3')
        # Getting the type of 'b' (line 92)
        b_2377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 45), 'b')
        # Applying the binary operator '*' (line 92)
        result_mul_2378 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 32), '*', edge3_2376, b_2377)
        
        # Applying the binary operator '+' (line 92)
        result_add_2379 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 15), '+', result_mul_2374, result_mul_2378)
        
        
        # Obtaining the type of the subscript
        int_2380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 62), 'int')
        # Getting the type of 'self' (line 92)
        self_2381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 49), 'self')
        # Obtaining the member 'vertexs' of a type (line 92)
        vertexs_2382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 49), self_2381, 'vertexs')
        # Obtaining the member '__getitem__' of a type (line 92)
        getitem___2383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 49), vertexs_2382, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 92)
        subscript_call_result_2384 = invoke(stypy.reporting.localization.Localization(__file__, 92, 49), getitem___2383, int_2380)
        
        # Applying the binary operator '+' (line 92)
        result_add_2385 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 47), '+', result_add_2379, subscript_call_result_2384)
        
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'stypy_return_type', result_add_2385)
        
        # ################# End of 'get_sample_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_sample_point' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_2386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2386)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_sample_point'
        return stypy_return_type_2386


# Assigning a type to the variable 'Triangle' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'Triangle', Triangle)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
