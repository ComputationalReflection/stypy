
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://rosettacode.org/wiki/K-means%2B%2B_clustering
2: 
3: from math import pi, sin, cos
4: from random import random, choice
5: from copy import copy
6: 
7: FLOAT_MAX = 1e100
8: 
9: 
10: class Point:
11:     __slots__ = ["x", "y", "group"]
12: 
13:     def __init__(self, x=0.0, y=0.0, group=0):
14:         self.x, self.y, self.group = x, y, group
15: 
16: 
17: def generate_points(npoints, radius):
18:     points = [Point() for _ in xrange(npoints)]
19: 
20:     # note: this is not a uniform 2-d distribution
21:     for p in points:
22:         r = random() * radius
23:         ang = random() * 2 * pi
24:         p.x = r * cos(ang)
25:         p.y = r * sin(ang)
26: 
27:     return points
28: 
29: 
30: def sqr_distance_2D(a, b):
31:     return (a.x - b.x) ** 2 + (a.y - b.y) ** 2
32: 
33: 
34: def nearest_cluster_center(point, cluster_centers):
35:     '''Distance and index of the closest cluster center'''
36: 
37:     min_index = point.group
38:     min_dist = FLOAT_MAX
39: 
40:     for i, cc in enumerate(cluster_centers):
41:         d = sqr_distance_2D(cc, point)
42:         if min_dist > d:
43:             min_dist = d
44:             min_index = i
45: 
46:     return (min_index, min_dist)
47: 
48: 
49: def kpp(points, cluster_centers):
50:     cluster_centers[0] = copy(choice(points))
51:     d = [0.0 for _ in xrange(len(points))]
52: 
53:     for i in xrange(1, len(cluster_centers)):
54:         sum = 0
55:         for j, p in enumerate(points):
56:             d[j] = nearest_cluster_center(p, cluster_centers[:i])[1]
57:             sum += d[j]
58: 
59:         sum *= random()
60: 
61:         for j, di in enumerate(d):
62:             sum -= di
63:             if sum > 0:
64:                 continue
65:             cluster_centers[i] = copy(points[j])
66:             break
67: 
68:     for p in points:
69:         p.group = nearest_cluster_center(p, cluster_centers)[0]
70: 
71: 
72: def lloyd(points, nclusters):
73:     cluster_centers = [Point() for _ in xrange(nclusters)]
74: 
75:     # call k++ init
76:     kpp(points, cluster_centers)
77: 
78:     lenpts10 = len(points) >> 10
79: 
80:     changed = 0
81:     while True:
82:         # group element for centroids are used as counters
83:         for cc in cluster_centers:
84:             cc.x = 0
85:             cc.y = 0
86:             cc.group = 0
87: 
88:         for p in points:
89:             cluster_centers[p.group].group += 1
90:             cluster_centers[p.group].x += p.x
91:             cluster_centers[p.group].y += p.y
92: 
93:         for cc in cluster_centers:
94:             cc.x /= cc.group
95:             cc.y /= cc.group
96: 
97:         # find closest centroid of each PointPtr
98:         changed = 0
99:         for p in points:
100:             min_i = nearest_cluster_center(p, cluster_centers)[0]
101:             if min_i != p.group:
102:                 changed += 1
103:                 p.group = min_i
104: 
105:         # stop when 99.9% of points are good
106:         if changed <= lenpts10:
107:             break
108: 
109:     for i, cc in enumerate(cluster_centers):
110:         cc.group = i
111: 
112:     return cluster_centers
113: 
114: 
115: class Color:
116:     def __init__(self, r, g, b):
117:         self.r = r
118:         self.g = g
119:         self.b = b
120: 
121: 
122: def print_eps(points, cluster_centers, W=400, H=400):
123:     colors = []
124:     for i in xrange(len(cluster_centers)):
125:         colors.append(Color((3 * (i + 1) % 11) / 11.0,
126:                             (7 * i % 11) / 11.0,
127:                             (9 * i % 11) / 11.0))
128: 
129:     max_x = max_y = -FLOAT_MAX
130:     min_x = min_y = FLOAT_MAX
131: 
132:     for p in points:
133:         if max_x < p.x: max_x = p.x
134:         if min_x > p.x: min_x = p.x
135:         if max_y < p.y: max_y = p.y
136:         if min_y > p.y: min_y = p.y
137: 
138:     scale = min(W / (max_x - min_x),
139:                 H / (max_y - min_y))
140:     cx = (max_x + min_x) / 2
141:     cy = (max_y + min_y) / 2
142: 
143:     # print "%%!PS-Adobe-3.0\n%%%%BoundingBox: -5 -5 %d %d" % (W + 10, H + 10)
144: 
145:     # print ("/l {rlineto} def /m {rmoveto} def\n" +
146:     #       "/c { .25 sub exch .25 sub exch .5 0 360 arc fill } def\n" +
147:     #       "/s { moveto -2 0 m 2 2 l 2 -2 l -2 -2 l closepath " +
148:     #       "   gsave 1 setgray fill grestore gsave 3 setlinewidth" +
149:     #       " 1 setgray stroke grestore 0 setgray stroke }def")
150: 
151:     for i, cc in enumerate(cluster_centers):
152:         # print ("%g %g %g setrgbcolor" %
153:         #       (colors[i].r, colors[i].g, colors[i].b))
154: 
155:         for p in points:
156:             if p.group != i:
157:                 continue
158:             # print ("%.3f %.3f c" % ((p.x - cx) * scale + W / 2,
159:             #                        (p.y - cy) * scale + H / 2))
160: 
161:         # print ("\n0 setgray %g %g s" % ((cc.x - cx) * scale + W / 2,
162:         #                                (cc.y - cy) * scale + H / 2))
163: 
164:     # print "\n%%%%EOF"
165: 
166: 
167: def main():
168:     npoints = 30000
169:     k = 7  # # clusters
170: 
171:     points = generate_points(npoints, 10)
172:     cluster_centers = lloyd(points, k)
173:     print_eps(points, cluster_centers)
174: 
175: 
176: def run():
177:     main()
178:     return True
179: 
180: 
181: run()
182: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from math import pi, sin, cos' statement (line 3)
try:
    from math import pi, sin, cos

except:
    pi = UndefinedType
    sin = UndefinedType
    cos = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'math', None, module_type_store, ['pi', 'sin', 'cos'], [pi, sin, cos])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from random import random, choice' statement (line 4)
try:
    from random import random, choice

except:
    random = UndefinedType
    choice = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'random', None, module_type_store, ['random', 'choice'], [random, choice])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from copy import copy' statement (line 5)
try:
    from copy import copy

except:
    copy = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'copy', None, module_type_store, ['copy'], [copy])


# Assigning a Num to a Name (line 7):

# Assigning a Num to a Name (line 7):
float_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'float')
# Assigning a type to the variable 'FLOAT_MAX' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'FLOAT_MAX', float_4)
# Declaration of the 'Point' class

class Point:
    
    # Assigning a List to a Name (line 11):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'float')
        float_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 32), 'float')
        int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 43), 'int')
        defaults = [float_5, float_6, int_7]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Point.__init__', ['x', 'y', 'group'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'y', 'group'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Tuple to a Tuple (line 14):
        
        # Assigning a Name to a Name (line 14):
        # Getting the type of 'x' (line 14)
        x_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 37), 'x')
        # Assigning a type to the variable 'tuple_assignment_1' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'tuple_assignment_1', x_8)
        
        # Assigning a Name to a Name (line 14):
        # Getting the type of 'y' (line 14)
        y_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 40), 'y')
        # Assigning a type to the variable 'tuple_assignment_2' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'tuple_assignment_2', y_9)
        
        # Assigning a Name to a Name (line 14):
        # Getting the type of 'group' (line 14)
        group_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 43), 'group')
        # Assigning a type to the variable 'tuple_assignment_3' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'tuple_assignment_3', group_10)
        
        # Assigning a Name to a Attribute (line 14):
        # Getting the type of 'tuple_assignment_1' (line 14)
        tuple_assignment_1_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'tuple_assignment_1')
        # Getting the type of 'self' (line 14)
        self_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member 'x' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_12, 'x', tuple_assignment_1_11)
        
        # Assigning a Name to a Attribute (line 14):
        # Getting the type of 'tuple_assignment_2' (line 14)
        tuple_assignment_2_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'tuple_assignment_2')
        # Getting the type of 'self' (line 14)
        self_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'self')
        # Setting the type of the member 'y' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 16), self_14, 'y', tuple_assignment_2_13)
        
        # Assigning a Name to a Attribute (line 14):
        # Getting the type of 'tuple_assignment_3' (line 14)
        tuple_assignment_3_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'tuple_assignment_3')
        # Getting the type of 'self' (line 14)
        self_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 24), 'self')
        # Setting the type of the member 'group' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 24), self_16, 'group', tuple_assignment_3_15)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Point' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'Point', Point)

# Assigning a List to a Name (line 11):

# Obtaining an instance of the builtin type 'list' (line 11)
list_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'str', 'x')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 16), list_17, str_18)
# Adding element type (line 11)
str_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'str', 'y')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 16), list_17, str_19)
# Adding element type (line 11)
str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'str', 'group')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 16), list_17, str_20)

# Getting the type of 'Point'
Point_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Point')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Point_21, '__slots__', list_17)

@norecursion
def generate_points(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'generate_points'
    module_type_store = module_type_store.open_function_context('generate_points', 17, 0, False)
    
    # Passed parameters checking function
    generate_points.stypy_localization = localization
    generate_points.stypy_type_of_self = None
    generate_points.stypy_type_store = module_type_store
    generate_points.stypy_function_name = 'generate_points'
    generate_points.stypy_param_names_list = ['npoints', 'radius']
    generate_points.stypy_varargs_param_name = None
    generate_points.stypy_kwargs_param_name = None
    generate_points.stypy_call_defaults = defaults
    generate_points.stypy_call_varargs = varargs
    generate_points.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generate_points', ['npoints', 'radius'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generate_points', localization, ['npoints', 'radius'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generate_points(...)' code ##################

    
    # Assigning a ListComp to a Name (line 18):
    
    # Assigning a ListComp to a Name (line 18):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to xrange(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'npoints' (line 18)
    npoints_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 38), 'npoints', False)
    # Processing the call keyword arguments (line 18)
    kwargs_27 = {}
    # Getting the type of 'xrange' (line 18)
    xrange_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 31), 'xrange', False)
    # Calling xrange(args, kwargs) (line 18)
    xrange_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 18, 31), xrange_25, *[npoints_26], **kwargs_27)
    
    comprehension_29 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 14), xrange_call_result_28)
    # Assigning a type to the variable '_' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), '_', comprehension_29)
    
    # Call to Point(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_23 = {}
    # Getting the type of 'Point' (line 18)
    Point_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'Point', False)
    # Calling Point(args, kwargs) (line 18)
    Point_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), Point_22, *[], **kwargs_23)
    
    list_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 14), list_30, Point_call_result_24)
    # Assigning a type to the variable 'points' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'points', list_30)
    
    # Getting the type of 'points' (line 21)
    points_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 13), 'points')
    # Assigning a type to the variable 'points_31' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'points_31', points_31)
    # Testing if the for loop is going to be iterated (line 21)
    # Testing the type of a for loop iterable (line 21)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 21, 4), points_31)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 21, 4), points_31):
        # Getting the type of the for loop variable (line 21)
        for_loop_var_32 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 21, 4), points_31)
        # Assigning a type to the variable 'p' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'p', for_loop_var_32)
        # SSA begins for a for statement (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 22):
        
        # Assigning a BinOp to a Name (line 22):
        
        # Call to random(...): (line 22)
        # Processing the call keyword arguments (line 22)
        kwargs_34 = {}
        # Getting the type of 'random' (line 22)
        random_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'random', False)
        # Calling random(args, kwargs) (line 22)
        random_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), random_33, *[], **kwargs_34)
        
        # Getting the type of 'radius' (line 22)
        radius_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'radius')
        # Applying the binary operator '*' (line 22)
        result_mul_37 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), '*', random_call_result_35, radius_36)
        
        # Assigning a type to the variable 'r' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'r', result_mul_37)
        
        # Assigning a BinOp to a Name (line 23):
        
        # Assigning a BinOp to a Name (line 23):
        
        # Call to random(...): (line 23)
        # Processing the call keyword arguments (line 23)
        kwargs_39 = {}
        # Getting the type of 'random' (line 23)
        random_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'random', False)
        # Calling random(args, kwargs) (line 23)
        random_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 23, 14), random_38, *[], **kwargs_39)
        
        int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'int')
        # Applying the binary operator '*' (line 23)
        result_mul_42 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 14), '*', random_call_result_40, int_41)
        
        # Getting the type of 'pi' (line 23)
        pi_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 29), 'pi')
        # Applying the binary operator '*' (line 23)
        result_mul_44 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 27), '*', result_mul_42, pi_43)
        
        # Assigning a type to the variable 'ang' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'ang', result_mul_44)
        
        # Assigning a BinOp to a Attribute (line 24):
        
        # Assigning a BinOp to a Attribute (line 24):
        # Getting the type of 'r' (line 24)
        r_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 14), 'r')
        
        # Call to cos(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'ang' (line 24)
        ang_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 'ang', False)
        # Processing the call keyword arguments (line 24)
        kwargs_48 = {}
        # Getting the type of 'cos' (line 24)
        cos_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'cos', False)
        # Calling cos(args, kwargs) (line 24)
        cos_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 24, 18), cos_46, *[ang_47], **kwargs_48)
        
        # Applying the binary operator '*' (line 24)
        result_mul_50 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 14), '*', r_45, cos_call_result_49)
        
        # Getting the type of 'p' (line 24)
        p_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'p')
        # Setting the type of the member 'x' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), p_51, 'x', result_mul_50)
        
        # Assigning a BinOp to a Attribute (line 25):
        
        # Assigning a BinOp to a Attribute (line 25):
        # Getting the type of 'r' (line 25)
        r_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 'r')
        
        # Call to sin(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'ang' (line 25)
        ang_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 22), 'ang', False)
        # Processing the call keyword arguments (line 25)
        kwargs_55 = {}
        # Getting the type of 'sin' (line 25)
        sin_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'sin', False)
        # Calling sin(args, kwargs) (line 25)
        sin_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 25, 18), sin_53, *[ang_54], **kwargs_55)
        
        # Applying the binary operator '*' (line 25)
        result_mul_57 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 14), '*', r_52, sin_call_result_56)
        
        # Getting the type of 'p' (line 25)
        p_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'p')
        # Setting the type of the member 'y' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), p_58, 'y', result_mul_57)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'points' (line 27)
    points_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'points')
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type', points_59)
    
    # ################# End of 'generate_points(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generate_points' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_60)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generate_points'
    return stypy_return_type_60

# Assigning a type to the variable 'generate_points' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'generate_points', generate_points)

@norecursion
def sqr_distance_2D(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sqr_distance_2D'
    module_type_store = module_type_store.open_function_context('sqr_distance_2D', 30, 0, False)
    
    # Passed parameters checking function
    sqr_distance_2D.stypy_localization = localization
    sqr_distance_2D.stypy_type_of_self = None
    sqr_distance_2D.stypy_type_store = module_type_store
    sqr_distance_2D.stypy_function_name = 'sqr_distance_2D'
    sqr_distance_2D.stypy_param_names_list = ['a', 'b']
    sqr_distance_2D.stypy_varargs_param_name = None
    sqr_distance_2D.stypy_kwargs_param_name = None
    sqr_distance_2D.stypy_call_defaults = defaults
    sqr_distance_2D.stypy_call_varargs = varargs
    sqr_distance_2D.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sqr_distance_2D', ['a', 'b'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sqr_distance_2D', localization, ['a', 'b'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sqr_distance_2D(...)' code ##################

    # Getting the type of 'a' (line 31)
    a_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'a')
    # Obtaining the member 'x' of a type (line 31)
    x_62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 12), a_61, 'x')
    # Getting the type of 'b' (line 31)
    b_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'b')
    # Obtaining the member 'x' of a type (line 31)
    x_64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 18), b_63, 'x')
    # Applying the binary operator '-' (line 31)
    result_sub_65 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 12), '-', x_62, x_64)
    
    int_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 26), 'int')
    # Applying the binary operator '**' (line 31)
    result_pow_67 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), '**', result_sub_65, int_66)
    
    # Getting the type of 'a' (line 31)
    a_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'a')
    # Obtaining the member 'y' of a type (line 31)
    y_69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 31), a_68, 'y')
    # Getting the type of 'b' (line 31)
    b_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'b')
    # Obtaining the member 'y' of a type (line 31)
    y_71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 37), b_70, 'y')
    # Applying the binary operator '-' (line 31)
    result_sub_72 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 31), '-', y_69, y_71)
    
    int_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 45), 'int')
    # Applying the binary operator '**' (line 31)
    result_pow_74 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 30), '**', result_sub_72, int_73)
    
    # Applying the binary operator '+' (line 31)
    result_add_75 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), '+', result_pow_67, result_pow_74)
    
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type', result_add_75)
    
    # ################# End of 'sqr_distance_2D(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sqr_distance_2D' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_76)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sqr_distance_2D'
    return stypy_return_type_76

# Assigning a type to the variable 'sqr_distance_2D' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'sqr_distance_2D', sqr_distance_2D)

@norecursion
def nearest_cluster_center(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'nearest_cluster_center'
    module_type_store = module_type_store.open_function_context('nearest_cluster_center', 34, 0, False)
    
    # Passed parameters checking function
    nearest_cluster_center.stypy_localization = localization
    nearest_cluster_center.stypy_type_of_self = None
    nearest_cluster_center.stypy_type_store = module_type_store
    nearest_cluster_center.stypy_function_name = 'nearest_cluster_center'
    nearest_cluster_center.stypy_param_names_list = ['point', 'cluster_centers']
    nearest_cluster_center.stypy_varargs_param_name = None
    nearest_cluster_center.stypy_kwargs_param_name = None
    nearest_cluster_center.stypy_call_defaults = defaults
    nearest_cluster_center.stypy_call_varargs = varargs
    nearest_cluster_center.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'nearest_cluster_center', ['point', 'cluster_centers'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'nearest_cluster_center', localization, ['point', 'cluster_centers'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'nearest_cluster_center(...)' code ##################

    str_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'str', 'Distance and index of the closest cluster center')
    
    # Assigning a Attribute to a Name (line 37):
    
    # Assigning a Attribute to a Name (line 37):
    # Getting the type of 'point' (line 37)
    point_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'point')
    # Obtaining the member 'group' of a type (line 37)
    group_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 16), point_78, 'group')
    # Assigning a type to the variable 'min_index' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'min_index', group_79)
    
    # Assigning a Name to a Name (line 38):
    
    # Assigning a Name to a Name (line 38):
    # Getting the type of 'FLOAT_MAX' (line 38)
    FLOAT_MAX_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'FLOAT_MAX')
    # Assigning a type to the variable 'min_dist' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'min_dist', FLOAT_MAX_80)
    
    
    # Call to enumerate(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'cluster_centers' (line 40)
    cluster_centers_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'cluster_centers', False)
    # Processing the call keyword arguments (line 40)
    kwargs_83 = {}
    # Getting the type of 'enumerate' (line 40)
    enumerate_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 40)
    enumerate_call_result_84 = invoke(stypy.reporting.localization.Localization(__file__, 40, 17), enumerate_81, *[cluster_centers_82], **kwargs_83)
    
    # Assigning a type to the variable 'enumerate_call_result_84' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'enumerate_call_result_84', enumerate_call_result_84)
    # Testing if the for loop is going to be iterated (line 40)
    # Testing the type of a for loop iterable (line 40)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 4), enumerate_call_result_84)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 40, 4), enumerate_call_result_84):
        # Getting the type of the for loop variable (line 40)
        for_loop_var_85 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 4), enumerate_call_result_84)
        # Assigning a type to the variable 'i' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), for_loop_var_85, 2, 0))
        # Assigning a type to the variable 'cc' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'cc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 4), for_loop_var_85, 2, 1))
        # SSA begins for a for statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 41):
        
        # Assigning a Call to a Name (line 41):
        
        # Call to sqr_distance_2D(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'cc' (line 41)
        cc_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'cc', False)
        # Getting the type of 'point' (line 41)
        point_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 32), 'point', False)
        # Processing the call keyword arguments (line 41)
        kwargs_89 = {}
        # Getting the type of 'sqr_distance_2D' (line 41)
        sqr_distance_2D_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'sqr_distance_2D', False)
        # Calling sqr_distance_2D(args, kwargs) (line 41)
        sqr_distance_2D_call_result_90 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), sqr_distance_2D_86, *[cc_87, point_88], **kwargs_89)
        
        # Assigning a type to the variable 'd' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'd', sqr_distance_2D_call_result_90)
        
        # Getting the type of 'min_dist' (line 42)
        min_dist_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'min_dist')
        # Getting the type of 'd' (line 42)
        d_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'd')
        # Applying the binary operator '>' (line 42)
        result_gt_93 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 11), '>', min_dist_91, d_92)
        
        # Testing if the type of an if condition is none (line 42)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 8), result_gt_93):
            pass
        else:
            
            # Testing the type of an if condition (line 42)
            if_condition_94 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), result_gt_93)
            # Assigning a type to the variable 'if_condition_94' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_94', if_condition_94)
            # SSA begins for if statement (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 43):
            
            # Assigning a Name to a Name (line 43):
            # Getting the type of 'd' (line 43)
            d_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'd')
            # Assigning a type to the variable 'min_dist' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'min_dist', d_95)
            
            # Assigning a Name to a Name (line 44):
            
            # Assigning a Name to a Name (line 44):
            # Getting the type of 'i' (line 44)
            i_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'i')
            # Assigning a type to the variable 'min_index' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'min_index', i_96)
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    # Getting the type of 'min_index' (line 46)
    min_index_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'min_index')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), tuple_97, min_index_98)
    # Adding element type (line 46)
    # Getting the type of 'min_dist' (line 46)
    min_dist_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'min_dist')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), tuple_97, min_dist_99)
    
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type', tuple_97)
    
    # ################# End of 'nearest_cluster_center(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'nearest_cluster_center' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_100)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'nearest_cluster_center'
    return stypy_return_type_100

# Assigning a type to the variable 'nearest_cluster_center' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'nearest_cluster_center', nearest_cluster_center)

@norecursion
def kpp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'kpp'
    module_type_store = module_type_store.open_function_context('kpp', 49, 0, False)
    
    # Passed parameters checking function
    kpp.stypy_localization = localization
    kpp.stypy_type_of_self = None
    kpp.stypy_type_store = module_type_store
    kpp.stypy_function_name = 'kpp'
    kpp.stypy_param_names_list = ['points', 'cluster_centers']
    kpp.stypy_varargs_param_name = None
    kpp.stypy_kwargs_param_name = None
    kpp.stypy_call_defaults = defaults
    kpp.stypy_call_varargs = varargs
    kpp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'kpp', ['points', 'cluster_centers'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'kpp', localization, ['points', 'cluster_centers'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'kpp(...)' code ##################

    
    # Assigning a Call to a Subscript (line 50):
    
    # Assigning a Call to a Subscript (line 50):
    
    # Call to copy(...): (line 50)
    # Processing the call arguments (line 50)
    
    # Call to choice(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'points' (line 50)
    points_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 37), 'points', False)
    # Processing the call keyword arguments (line 50)
    kwargs_104 = {}
    # Getting the type of 'choice' (line 50)
    choice_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 30), 'choice', False)
    # Calling choice(args, kwargs) (line 50)
    choice_call_result_105 = invoke(stypy.reporting.localization.Localization(__file__, 50, 30), choice_102, *[points_103], **kwargs_104)
    
    # Processing the call keyword arguments (line 50)
    kwargs_106 = {}
    # Getting the type of 'copy' (line 50)
    copy_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'copy', False)
    # Calling copy(args, kwargs) (line 50)
    copy_call_result_107 = invoke(stypy.reporting.localization.Localization(__file__, 50, 25), copy_101, *[choice_call_result_105], **kwargs_106)
    
    # Getting the type of 'cluster_centers' (line 50)
    cluster_centers_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'cluster_centers')
    int_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 20), 'int')
    # Storing an element on a container (line 50)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 4), cluster_centers_108, (int_109, copy_call_result_107))
    
    # Assigning a ListComp to a Name (line 51):
    
    # Assigning a ListComp to a Name (line 51):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to xrange(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Call to len(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'points' (line 51)
    points_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 33), 'points', False)
    # Processing the call keyword arguments (line 51)
    kwargs_114 = {}
    # Getting the type of 'len' (line 51)
    len_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'len', False)
    # Calling len(args, kwargs) (line 51)
    len_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 51, 29), len_112, *[points_113], **kwargs_114)
    
    # Processing the call keyword arguments (line 51)
    kwargs_116 = {}
    # Getting the type of 'xrange' (line 51)
    xrange_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'xrange', False)
    # Calling xrange(args, kwargs) (line 51)
    xrange_call_result_117 = invoke(stypy.reporting.localization.Localization(__file__, 51, 22), xrange_111, *[len_call_result_115], **kwargs_116)
    
    comprehension_118 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), xrange_call_result_117)
    # Assigning a type to the variable '_' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 9), '_', comprehension_118)
    float_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'float')
    list_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 9), list_119, float_110)
    # Assigning a type to the variable 'd' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'd', list_119)
    
    
    # Call to xrange(...): (line 53)
    # Processing the call arguments (line 53)
    int_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 20), 'int')
    
    # Call to len(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'cluster_centers' (line 53)
    cluster_centers_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'cluster_centers', False)
    # Processing the call keyword arguments (line 53)
    kwargs_124 = {}
    # Getting the type of 'len' (line 53)
    len_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'len', False)
    # Calling len(args, kwargs) (line 53)
    len_call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 53, 23), len_122, *[cluster_centers_123], **kwargs_124)
    
    # Processing the call keyword arguments (line 53)
    kwargs_126 = {}
    # Getting the type of 'xrange' (line 53)
    xrange_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 53)
    xrange_call_result_127 = invoke(stypy.reporting.localization.Localization(__file__, 53, 13), xrange_120, *[int_121, len_call_result_125], **kwargs_126)
    
    # Assigning a type to the variable 'xrange_call_result_127' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'xrange_call_result_127', xrange_call_result_127)
    # Testing if the for loop is going to be iterated (line 53)
    # Testing the type of a for loop iterable (line 53)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 53, 4), xrange_call_result_127)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 53, 4), xrange_call_result_127):
        # Getting the type of the for loop variable (line 53)
        for_loop_var_128 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 53, 4), xrange_call_result_127)
        # Assigning a type to the variable 'i' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'i', for_loop_var_128)
        # SSA begins for a for statement (line 53)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Name (line 54):
        
        # Assigning a Num to a Name (line 54):
        int_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 14), 'int')
        # Assigning a type to the variable 'sum' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'sum', int_129)
        
        
        # Call to enumerate(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'points' (line 55)
        points_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 30), 'points', False)
        # Processing the call keyword arguments (line 55)
        kwargs_132 = {}
        # Getting the type of 'enumerate' (line 55)
        enumerate_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 55)
        enumerate_call_result_133 = invoke(stypy.reporting.localization.Localization(__file__, 55, 20), enumerate_130, *[points_131], **kwargs_132)
        
        # Assigning a type to the variable 'enumerate_call_result_133' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'enumerate_call_result_133', enumerate_call_result_133)
        # Testing if the for loop is going to be iterated (line 55)
        # Testing the type of a for loop iterable (line 55)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 55, 8), enumerate_call_result_133)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 55, 8), enumerate_call_result_133):
            # Getting the type of the for loop variable (line 55)
            for_loop_var_134 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 55, 8), enumerate_call_result_133)
            # Assigning a type to the variable 'j' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), for_loop_var_134, 2, 0))
            # Assigning a type to the variable 'p' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'p', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 8), for_loop_var_134, 2, 1))
            # SSA begins for a for statement (line 55)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Subscript (line 56):
            
            # Assigning a Subscript to a Subscript (line 56):
            
            # Obtaining the type of the subscript
            int_135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 66), 'int')
            
            # Call to nearest_cluster_center(...): (line 56)
            # Processing the call arguments (line 56)
            # Getting the type of 'p' (line 56)
            p_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 42), 'p', False)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 56)
            i_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 62), 'i', False)
            slice_139 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 45), None, i_138, None)
            # Getting the type of 'cluster_centers' (line 56)
            cluster_centers_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 45), 'cluster_centers', False)
            # Obtaining the member '__getitem__' of a type (line 56)
            getitem___141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 45), cluster_centers_140, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 56)
            subscript_call_result_142 = invoke(stypy.reporting.localization.Localization(__file__, 56, 45), getitem___141, slice_139)
            
            # Processing the call keyword arguments (line 56)
            kwargs_143 = {}
            # Getting the type of 'nearest_cluster_center' (line 56)
            nearest_cluster_center_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'nearest_cluster_center', False)
            # Calling nearest_cluster_center(args, kwargs) (line 56)
            nearest_cluster_center_call_result_144 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), nearest_cluster_center_136, *[p_137, subscript_call_result_142], **kwargs_143)
            
            # Obtaining the member '__getitem__' of a type (line 56)
            getitem___145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 19), nearest_cluster_center_call_result_144, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 56)
            subscript_call_result_146 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), getitem___145, int_135)
            
            # Getting the type of 'd' (line 56)
            d_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'd')
            # Getting the type of 'j' (line 56)
            j_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'j')
            # Storing an element on a container (line 56)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 12), d_147, (j_148, subscript_call_result_146))
            
            # Getting the type of 'sum' (line 57)
            sum_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'sum')
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 57)
            j_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'j')
            # Getting the type of 'd' (line 57)
            d_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 'd')
            # Obtaining the member '__getitem__' of a type (line 57)
            getitem___152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 19), d_151, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 57)
            subscript_call_result_153 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), getitem___152, j_150)
            
            # Applying the binary operator '+=' (line 57)
            result_iadd_154 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 12), '+=', sum_149, subscript_call_result_153)
            # Assigning a type to the variable 'sum' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'sum', result_iadd_154)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'sum' (line 59)
        sum_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'sum')
        
        # Call to random(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_157 = {}
        # Getting the type of 'random' (line 59)
        random_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'random', False)
        # Calling random(args, kwargs) (line 59)
        random_call_result_158 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), random_156, *[], **kwargs_157)
        
        # Applying the binary operator '*=' (line 59)
        result_imul_159 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 8), '*=', sum_155, random_call_result_158)
        # Assigning a type to the variable 'sum' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'sum', result_imul_159)
        
        
        
        # Call to enumerate(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'd' (line 61)
        d_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 31), 'd', False)
        # Processing the call keyword arguments (line 61)
        kwargs_162 = {}
        # Getting the type of 'enumerate' (line 61)
        enumerate_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 61)
        enumerate_call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 61, 21), enumerate_160, *[d_161], **kwargs_162)
        
        # Assigning a type to the variable 'enumerate_call_result_163' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'enumerate_call_result_163', enumerate_call_result_163)
        # Testing if the for loop is going to be iterated (line 61)
        # Testing the type of a for loop iterable (line 61)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 8), enumerate_call_result_163)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 61, 8), enumerate_call_result_163):
            # Getting the type of the for loop variable (line 61)
            for_loop_var_164 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 8), enumerate_call_result_163)
            # Assigning a type to the variable 'j' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), for_loop_var_164, 2, 0))
            # Assigning a type to the variable 'di' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'di', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), for_loop_var_164, 2, 1))
            # SSA begins for a for statement (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'sum' (line 62)
            sum_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'sum')
            # Getting the type of 'di' (line 62)
            di_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'di')
            # Applying the binary operator '-=' (line 62)
            result_isub_167 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 12), '-=', sum_165, di_166)
            # Assigning a type to the variable 'sum' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'sum', result_isub_167)
            
            
            # Getting the type of 'sum' (line 63)
            sum_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'sum')
            int_169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'int')
            # Applying the binary operator '>' (line 63)
            result_gt_170 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 15), '>', sum_168, int_169)
            
            # Testing if the type of an if condition is none (line 63)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 63, 12), result_gt_170):
                pass
            else:
                
                # Testing the type of an if condition (line 63)
                if_condition_171 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 12), result_gt_170)
                # Assigning a type to the variable 'if_condition_171' (line 63)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'if_condition_171', if_condition_171)
                # SSA begins for if statement (line 63)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 63)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Call to a Subscript (line 65):
            
            # Assigning a Call to a Subscript (line 65):
            
            # Call to copy(...): (line 65)
            # Processing the call arguments (line 65)
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 65)
            j_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 45), 'j', False)
            # Getting the type of 'points' (line 65)
            points_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 38), 'points', False)
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 38), points_174, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 65, 38), getitem___175, j_173)
            
            # Processing the call keyword arguments (line 65)
            kwargs_177 = {}
            # Getting the type of 'copy' (line 65)
            copy_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 33), 'copy', False)
            # Calling copy(args, kwargs) (line 65)
            copy_call_result_178 = invoke(stypy.reporting.localization.Localization(__file__, 65, 33), copy_172, *[subscript_call_result_176], **kwargs_177)
            
            # Getting the type of 'cluster_centers' (line 65)
            cluster_centers_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'cluster_centers')
            # Getting the type of 'i' (line 65)
            i_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'i')
            # Storing an element on a container (line 65)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 12), cluster_centers_179, (i_180, copy_call_result_178))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Getting the type of 'points' (line 68)
    points_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 13), 'points')
    # Assigning a type to the variable 'points_181' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'points_181', points_181)
    # Testing if the for loop is going to be iterated (line 68)
    # Testing the type of a for loop iterable (line 68)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 68, 4), points_181)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 68, 4), points_181):
        # Getting the type of the for loop variable (line 68)
        for_loop_var_182 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 68, 4), points_181)
        # Assigning a type to the variable 'p' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'p', for_loop_var_182)
        # SSA begins for a for statement (line 68)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Attribute (line 69):
        
        # Assigning a Subscript to a Attribute (line 69):
        
        # Obtaining the type of the subscript
        int_183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 61), 'int')
        
        # Call to nearest_cluster_center(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'p' (line 69)
        p_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 41), 'p', False)
        # Getting the type of 'cluster_centers' (line 69)
        cluster_centers_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 44), 'cluster_centers', False)
        # Processing the call keyword arguments (line 69)
        kwargs_187 = {}
        # Getting the type of 'nearest_cluster_center' (line 69)
        nearest_cluster_center_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 18), 'nearest_cluster_center', False)
        # Calling nearest_cluster_center(args, kwargs) (line 69)
        nearest_cluster_center_call_result_188 = invoke(stypy.reporting.localization.Localization(__file__, 69, 18), nearest_cluster_center_184, *[p_185, cluster_centers_186], **kwargs_187)
        
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 18), nearest_cluster_center_call_result_188, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_190 = invoke(stypy.reporting.localization.Localization(__file__, 69, 18), getitem___189, int_183)
        
        # Getting the type of 'p' (line 69)
        p_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'p')
        # Setting the type of the member 'group' of a type (line 69)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 8), p_191, 'group', subscript_call_result_190)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'kpp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'kpp' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_192)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'kpp'
    return stypy_return_type_192

# Assigning a type to the variable 'kpp' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'kpp', kpp)

@norecursion
def lloyd(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lloyd'
    module_type_store = module_type_store.open_function_context('lloyd', 72, 0, False)
    
    # Passed parameters checking function
    lloyd.stypy_localization = localization
    lloyd.stypy_type_of_self = None
    lloyd.stypy_type_store = module_type_store
    lloyd.stypy_function_name = 'lloyd'
    lloyd.stypy_param_names_list = ['points', 'nclusters']
    lloyd.stypy_varargs_param_name = None
    lloyd.stypy_kwargs_param_name = None
    lloyd.stypy_call_defaults = defaults
    lloyd.stypy_call_varargs = varargs
    lloyd.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lloyd', ['points', 'nclusters'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lloyd', localization, ['points', 'nclusters'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lloyd(...)' code ##################

    
    # Assigning a ListComp to a Name (line 73):
    
    # Assigning a ListComp to a Name (line 73):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to xrange(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'nclusters' (line 73)
    nclusters_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 47), 'nclusters', False)
    # Processing the call keyword arguments (line 73)
    kwargs_198 = {}
    # Getting the type of 'xrange' (line 73)
    xrange_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 40), 'xrange', False)
    # Calling xrange(args, kwargs) (line 73)
    xrange_call_result_199 = invoke(stypy.reporting.localization.Localization(__file__, 73, 40), xrange_196, *[nclusters_197], **kwargs_198)
    
    comprehension_200 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 23), xrange_call_result_199)
    # Assigning a type to the variable '_' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), '_', comprehension_200)
    
    # Call to Point(...): (line 73)
    # Processing the call keyword arguments (line 73)
    kwargs_194 = {}
    # Getting the type of 'Point' (line 73)
    Point_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'Point', False)
    # Calling Point(args, kwargs) (line 73)
    Point_call_result_195 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), Point_193, *[], **kwargs_194)
    
    list_201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 23), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 23), list_201, Point_call_result_195)
    # Assigning a type to the variable 'cluster_centers' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'cluster_centers', list_201)
    
    # Call to kpp(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'points' (line 76)
    points_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'points', False)
    # Getting the type of 'cluster_centers' (line 76)
    cluster_centers_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'cluster_centers', False)
    # Processing the call keyword arguments (line 76)
    kwargs_205 = {}
    # Getting the type of 'kpp' (line 76)
    kpp_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'kpp', False)
    # Calling kpp(args, kwargs) (line 76)
    kpp_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), kpp_202, *[points_203, cluster_centers_204], **kwargs_205)
    
    
    # Assigning a BinOp to a Name (line 78):
    
    # Assigning a BinOp to a Name (line 78):
    
    # Call to len(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'points' (line 78)
    points_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'points', False)
    # Processing the call keyword arguments (line 78)
    kwargs_209 = {}
    # Getting the type of 'len' (line 78)
    len_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'len', False)
    # Calling len(args, kwargs) (line 78)
    len_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 78, 15), len_207, *[points_208], **kwargs_209)
    
    int_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 30), 'int')
    # Applying the binary operator '>>' (line 78)
    result_rshift_212 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 15), '>>', len_call_result_210, int_211)
    
    # Assigning a type to the variable 'lenpts10' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'lenpts10', result_rshift_212)
    
    # Assigning a Num to a Name (line 80):
    
    # Assigning a Num to a Name (line 80):
    int_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 14), 'int')
    # Assigning a type to the variable 'changed' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'changed', int_213)
    
    # Getting the type of 'True' (line 81)
    True_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 10), 'True')
    # Assigning a type to the variable 'True_214' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'True_214', True_214)
    # Testing if the while is going to be iterated (line 81)
    # Testing the type of an if condition (line 81)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 4), True_214)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 81, 4), True_214):
        
        # Getting the type of 'cluster_centers' (line 83)
        cluster_centers_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 18), 'cluster_centers')
        # Assigning a type to the variable 'cluster_centers_215' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'cluster_centers_215', cluster_centers_215)
        # Testing if the for loop is going to be iterated (line 83)
        # Testing the type of a for loop iterable (line 83)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 83, 8), cluster_centers_215)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 83, 8), cluster_centers_215):
            # Getting the type of the for loop variable (line 83)
            for_loop_var_216 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 83, 8), cluster_centers_215)
            # Assigning a type to the variable 'cc' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'cc', for_loop_var_216)
            # SSA begins for a for statement (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Attribute (line 84):
            
            # Assigning a Num to a Attribute (line 84):
            int_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 19), 'int')
            # Getting the type of 'cc' (line 84)
            cc_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'cc')
            # Setting the type of the member 'x' of a type (line 84)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), cc_218, 'x', int_217)
            
            # Assigning a Num to a Attribute (line 85):
            
            # Assigning a Num to a Attribute (line 85):
            int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 19), 'int')
            # Getting the type of 'cc' (line 85)
            cc_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'cc')
            # Setting the type of the member 'y' of a type (line 85)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), cc_220, 'y', int_219)
            
            # Assigning a Num to a Attribute (line 86):
            
            # Assigning a Num to a Attribute (line 86):
            int_221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'int')
            # Getting the type of 'cc' (line 86)
            cc_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'cc')
            # Setting the type of the member 'group' of a type (line 86)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), cc_222, 'group', int_221)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'points' (line 88)
        points_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'points')
        # Assigning a type to the variable 'points_223' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'points_223', points_223)
        # Testing if the for loop is going to be iterated (line 88)
        # Testing the type of a for loop iterable (line 88)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 8), points_223)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 88, 8), points_223):
            # Getting the type of the for loop variable (line 88)
            for_loop_var_224 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 8), points_223)
            # Assigning a type to the variable 'p' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'p', for_loop_var_224)
            # SSA begins for a for statement (line 88)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'p' (line 89)
            p_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'p')
            # Obtaining the member 'group' of a type (line 89)
            group_226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 28), p_225, 'group')
            # Getting the type of 'cluster_centers' (line 89)
            cluster_centers_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'cluster_centers')
            # Obtaining the member '__getitem__' of a type (line 89)
            getitem___228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), cluster_centers_227, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 89)
            subscript_call_result_229 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), getitem___228, group_226)
            
            # Obtaining the member 'group' of a type (line 89)
            group_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), subscript_call_result_229, 'group')
            int_231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 46), 'int')
            # Applying the binary operator '+=' (line 89)
            result_iadd_232 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 12), '+=', group_230, int_231)
            
            # Obtaining the type of the subscript
            # Getting the type of 'p' (line 89)
            p_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'p')
            # Obtaining the member 'group' of a type (line 89)
            group_234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 28), p_233, 'group')
            # Getting the type of 'cluster_centers' (line 89)
            cluster_centers_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'cluster_centers')
            # Obtaining the member '__getitem__' of a type (line 89)
            getitem___236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), cluster_centers_235, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 89)
            subscript_call_result_237 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), getitem___236, group_234)
            
            # Setting the type of the member 'group' of a type (line 89)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 12), subscript_call_result_237, 'group', result_iadd_232)
            
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'p' (line 90)
            p_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'p')
            # Obtaining the member 'group' of a type (line 90)
            group_239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), p_238, 'group')
            # Getting the type of 'cluster_centers' (line 90)
            cluster_centers_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'cluster_centers')
            # Obtaining the member '__getitem__' of a type (line 90)
            getitem___241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), cluster_centers_240, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 90)
            subscript_call_result_242 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), getitem___241, group_239)
            
            # Obtaining the member 'x' of a type (line 90)
            x_243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), subscript_call_result_242, 'x')
            # Getting the type of 'p' (line 90)
            p_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 42), 'p')
            # Obtaining the member 'x' of a type (line 90)
            x_245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 42), p_244, 'x')
            # Applying the binary operator '+=' (line 90)
            result_iadd_246 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 12), '+=', x_243, x_245)
            
            # Obtaining the type of the subscript
            # Getting the type of 'p' (line 90)
            p_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'p')
            # Obtaining the member 'group' of a type (line 90)
            group_248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 28), p_247, 'group')
            # Getting the type of 'cluster_centers' (line 90)
            cluster_centers_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'cluster_centers')
            # Obtaining the member '__getitem__' of a type (line 90)
            getitem___250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), cluster_centers_249, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 90)
            subscript_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), getitem___250, group_248)
            
            # Setting the type of the member 'x' of a type (line 90)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), subscript_call_result_251, 'x', result_iadd_246)
            
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'p' (line 91)
            p_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'p')
            # Obtaining the member 'group' of a type (line 91)
            group_253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 28), p_252, 'group')
            # Getting the type of 'cluster_centers' (line 91)
            cluster_centers_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'cluster_centers')
            # Obtaining the member '__getitem__' of a type (line 91)
            getitem___255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), cluster_centers_254, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 91)
            subscript_call_result_256 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), getitem___255, group_253)
            
            # Obtaining the member 'y' of a type (line 91)
            y_257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), subscript_call_result_256, 'y')
            # Getting the type of 'p' (line 91)
            p_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 42), 'p')
            # Obtaining the member 'y' of a type (line 91)
            y_259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 42), p_258, 'y')
            # Applying the binary operator '+=' (line 91)
            result_iadd_260 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 12), '+=', y_257, y_259)
            
            # Obtaining the type of the subscript
            # Getting the type of 'p' (line 91)
            p_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'p')
            # Obtaining the member 'group' of a type (line 91)
            group_262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 28), p_261, 'group')
            # Getting the type of 'cluster_centers' (line 91)
            cluster_centers_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'cluster_centers')
            # Obtaining the member '__getitem__' of a type (line 91)
            getitem___264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), cluster_centers_263, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 91)
            subscript_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 91, 12), getitem___264, group_262)
            
            # Setting the type of the member 'y' of a type (line 91)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 12), subscript_call_result_265, 'y', result_iadd_260)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'cluster_centers' (line 93)
        cluster_centers_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 18), 'cluster_centers')
        # Assigning a type to the variable 'cluster_centers_266' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'cluster_centers_266', cluster_centers_266)
        # Testing if the for loop is going to be iterated (line 93)
        # Testing the type of a for loop iterable (line 93)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 93, 8), cluster_centers_266)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 93, 8), cluster_centers_266):
            # Getting the type of the for loop variable (line 93)
            for_loop_var_267 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 93, 8), cluster_centers_266)
            # Assigning a type to the variable 'cc' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'cc', for_loop_var_267)
            # SSA begins for a for statement (line 93)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'cc' (line 94)
            cc_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'cc')
            # Obtaining the member 'x' of a type (line 94)
            x_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), cc_268, 'x')
            # Getting the type of 'cc' (line 94)
            cc_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'cc')
            # Obtaining the member 'group' of a type (line 94)
            group_271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 20), cc_270, 'group')
            # Applying the binary operator 'div=' (line 94)
            result_div_272 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 12), 'div=', x_269, group_271)
            # Getting the type of 'cc' (line 94)
            cc_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'cc')
            # Setting the type of the member 'x' of a type (line 94)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), cc_273, 'x', result_div_272)
            
            
            # Getting the type of 'cc' (line 95)
            cc_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'cc')
            # Obtaining the member 'y' of a type (line 95)
            y_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), cc_274, 'y')
            # Getting the type of 'cc' (line 95)
            cc_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 20), 'cc')
            # Obtaining the member 'group' of a type (line 95)
            group_277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 20), cc_276, 'group')
            # Applying the binary operator 'div=' (line 95)
            result_div_278 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 12), 'div=', y_275, group_277)
            # Getting the type of 'cc' (line 95)
            cc_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'cc')
            # Setting the type of the member 'y' of a type (line 95)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), cc_279, 'y', result_div_278)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Num to a Name (line 98):
        
        # Assigning a Num to a Name (line 98):
        int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 18), 'int')
        # Assigning a type to the variable 'changed' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'changed', int_280)
        
        # Getting the type of 'points' (line 99)
        points_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 17), 'points')
        # Assigning a type to the variable 'points_281' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'points_281', points_281)
        # Testing if the for loop is going to be iterated (line 99)
        # Testing the type of a for loop iterable (line 99)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 8), points_281)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 99, 8), points_281):
            # Getting the type of the for loop variable (line 99)
            for_loop_var_282 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 8), points_281)
            # Assigning a type to the variable 'p' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'p', for_loop_var_282)
            # SSA begins for a for statement (line 99)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 100):
            
            # Assigning a Subscript to a Name (line 100):
            
            # Obtaining the type of the subscript
            int_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 63), 'int')
            
            # Call to nearest_cluster_center(...): (line 100)
            # Processing the call arguments (line 100)
            # Getting the type of 'p' (line 100)
            p_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 43), 'p', False)
            # Getting the type of 'cluster_centers' (line 100)
            cluster_centers_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 46), 'cluster_centers', False)
            # Processing the call keyword arguments (line 100)
            kwargs_287 = {}
            # Getting the type of 'nearest_cluster_center' (line 100)
            nearest_cluster_center_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'nearest_cluster_center', False)
            # Calling nearest_cluster_center(args, kwargs) (line 100)
            nearest_cluster_center_call_result_288 = invoke(stypy.reporting.localization.Localization(__file__, 100, 20), nearest_cluster_center_284, *[p_285, cluster_centers_286], **kwargs_287)
            
            # Obtaining the member '__getitem__' of a type (line 100)
            getitem___289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 20), nearest_cluster_center_call_result_288, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 100)
            subscript_call_result_290 = invoke(stypy.reporting.localization.Localization(__file__, 100, 20), getitem___289, int_283)
            
            # Assigning a type to the variable 'min_i' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'min_i', subscript_call_result_290)
            
            # Getting the type of 'min_i' (line 101)
            min_i_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'min_i')
            # Getting the type of 'p' (line 101)
            p_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'p')
            # Obtaining the member 'group' of a type (line 101)
            group_293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 24), p_292, 'group')
            # Applying the binary operator '!=' (line 101)
            result_ne_294 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 15), '!=', min_i_291, group_293)
            
            # Testing if the type of an if condition is none (line 101)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 101, 12), result_ne_294):
                pass
            else:
                
                # Testing the type of an if condition (line 101)
                if_condition_295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 12), result_ne_294)
                # Assigning a type to the variable 'if_condition_295' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'if_condition_295', if_condition_295)
                # SSA begins for if statement (line 101)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'changed' (line 102)
                changed_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'changed')
                int_297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 27), 'int')
                # Applying the binary operator '+=' (line 102)
                result_iadd_298 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 16), '+=', changed_296, int_297)
                # Assigning a type to the variable 'changed' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'changed', result_iadd_298)
                
                
                # Assigning a Name to a Attribute (line 103):
                
                # Assigning a Name to a Attribute (line 103):
                # Getting the type of 'min_i' (line 103)
                min_i_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 26), 'min_i')
                # Getting the type of 'p' (line 103)
                p_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'p')
                # Setting the type of the member 'group' of a type (line 103)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), p_300, 'group', min_i_299)
                # SSA join for if statement (line 101)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Getting the type of 'changed' (line 106)
        changed_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'changed')
        # Getting the type of 'lenpts10' (line 106)
        lenpts10_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 22), 'lenpts10')
        # Applying the binary operator '<=' (line 106)
        result_le_303 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 11), '<=', changed_301, lenpts10_302)
        
        # Testing if the type of an if condition is none (line 106)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 8), result_le_303):
            pass
        else:
            
            # Testing the type of an if condition (line 106)
            if_condition_304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), result_le_303)
            # Assigning a type to the variable 'if_condition_304' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_304', if_condition_304)
            # SSA begins for if statement (line 106)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 106)
            module_type_store = module_type_store.join_ssa_context()
            


    
    
    
    # Call to enumerate(...): (line 109)
    # Processing the call arguments (line 109)
    # Getting the type of 'cluster_centers' (line 109)
    cluster_centers_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'cluster_centers', False)
    # Processing the call keyword arguments (line 109)
    kwargs_307 = {}
    # Getting the type of 'enumerate' (line 109)
    enumerate_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 17), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 109)
    enumerate_call_result_308 = invoke(stypy.reporting.localization.Localization(__file__, 109, 17), enumerate_305, *[cluster_centers_306], **kwargs_307)
    
    # Assigning a type to the variable 'enumerate_call_result_308' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'enumerate_call_result_308', enumerate_call_result_308)
    # Testing if the for loop is going to be iterated (line 109)
    # Testing the type of a for loop iterable (line 109)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 109, 4), enumerate_call_result_308)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 109, 4), enumerate_call_result_308):
        # Getting the type of the for loop variable (line 109)
        for_loop_var_309 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 109, 4), enumerate_call_result_308)
        # Assigning a type to the variable 'i' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 4), for_loop_var_309, 2, 0))
        # Assigning a type to the variable 'cc' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'cc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 4), for_loop_var_309, 2, 1))
        # SSA begins for a for statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Attribute (line 110):
        
        # Assigning a Name to a Attribute (line 110):
        # Getting the type of 'i' (line 110)
        i_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'i')
        # Getting the type of 'cc' (line 110)
        cc_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'cc')
        # Setting the type of the member 'group' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), cc_311, 'group', i_310)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'cluster_centers' (line 112)
    cluster_centers_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'cluster_centers')
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type', cluster_centers_312)
    
    # ################# End of 'lloyd(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lloyd' in the type store
    # Getting the type of 'stypy_return_type' (line 72)
    stypy_return_type_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_313)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lloyd'
    return stypy_return_type_313

# Assigning a type to the variable 'lloyd' (line 72)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 0), 'lloyd', lloyd)
# Declaration of the 'Color' class

class Color:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Color.__init__', ['r', 'g', 'b'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['r', 'g', 'b'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 117):
        
        # Assigning a Name to a Attribute (line 117):
        # Getting the type of 'r' (line 117)
        r_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 17), 'r')
        # Getting the type of 'self' (line 117)
        self_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self')
        # Setting the type of the member 'r' of a type (line 117)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_315, 'r', r_314)
        
        # Assigning a Name to a Attribute (line 118):
        
        # Assigning a Name to a Attribute (line 118):
        # Getting the type of 'g' (line 118)
        g_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'g')
        # Getting the type of 'self' (line 118)
        self_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self')
        # Setting the type of the member 'g' of a type (line 118)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_317, 'g', g_316)
        
        # Assigning a Name to a Attribute (line 119):
        
        # Assigning a Name to a Attribute (line 119):
        # Getting the type of 'b' (line 119)
        b_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 17), 'b')
        # Getting the type of 'self' (line 119)
        self_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member 'b' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_319, 'b', b_318)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Color' (line 115)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'Color', Color)

@norecursion
def print_eps(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 41), 'int')
    int_321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 48), 'int')
    defaults = [int_320, int_321]
    # Create a new context for function 'print_eps'
    module_type_store = module_type_store.open_function_context('print_eps', 122, 0, False)
    
    # Passed parameters checking function
    print_eps.stypy_localization = localization
    print_eps.stypy_type_of_self = None
    print_eps.stypy_type_store = module_type_store
    print_eps.stypy_function_name = 'print_eps'
    print_eps.stypy_param_names_list = ['points', 'cluster_centers', 'W', 'H']
    print_eps.stypy_varargs_param_name = None
    print_eps.stypy_kwargs_param_name = None
    print_eps.stypy_call_defaults = defaults
    print_eps.stypy_call_varargs = varargs
    print_eps.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_eps', ['points', 'cluster_centers', 'W', 'H'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_eps', localization, ['points', 'cluster_centers', 'W', 'H'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_eps(...)' code ##################

    
    # Assigning a List to a Name (line 123):
    
    # Assigning a List to a Name (line 123):
    
    # Obtaining an instance of the builtin type 'list' (line 123)
    list_322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 123)
    
    # Assigning a type to the variable 'colors' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'colors', list_322)
    
    
    # Call to xrange(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Call to len(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'cluster_centers' (line 124)
    cluster_centers_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 24), 'cluster_centers', False)
    # Processing the call keyword arguments (line 124)
    kwargs_326 = {}
    # Getting the type of 'len' (line 124)
    len_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'len', False)
    # Calling len(args, kwargs) (line 124)
    len_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 124, 20), len_324, *[cluster_centers_325], **kwargs_326)
    
    # Processing the call keyword arguments (line 124)
    kwargs_328 = {}
    # Getting the type of 'xrange' (line 124)
    xrange_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 124)
    xrange_call_result_329 = invoke(stypy.reporting.localization.Localization(__file__, 124, 13), xrange_323, *[len_call_result_327], **kwargs_328)
    
    # Assigning a type to the variable 'xrange_call_result_329' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'xrange_call_result_329', xrange_call_result_329)
    # Testing if the for loop is going to be iterated (line 124)
    # Testing the type of a for loop iterable (line 124)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 124, 4), xrange_call_result_329)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 124, 4), xrange_call_result_329):
        # Getting the type of the for loop variable (line 124)
        for_loop_var_330 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 124, 4), xrange_call_result_329)
        # Assigning a type to the variable 'i' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'i', for_loop_var_330)
        # SSA begins for a for statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 125)
        # Processing the call arguments (line 125)
        
        # Call to Color(...): (line 125)
        # Processing the call arguments (line 125)
        int_334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 29), 'int')
        # Getting the type of 'i' (line 125)
        i_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 34), 'i', False)
        int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 38), 'int')
        # Applying the binary operator '+' (line 125)
        result_add_337 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 34), '+', i_335, int_336)
        
        # Applying the binary operator '*' (line 125)
        result_mul_338 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 29), '*', int_334, result_add_337)
        
        int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 43), 'int')
        # Applying the binary operator '%' (line 125)
        result_mod_340 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 41), '%', result_mul_338, int_339)
        
        float_341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 49), 'float')
        # Applying the binary operator 'div' (line 125)
        result_div_342 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 28), 'div', result_mod_340, float_341)
        
        int_343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 29), 'int')
        # Getting the type of 'i' (line 126)
        i_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 33), 'i', False)
        # Applying the binary operator '*' (line 126)
        result_mul_345 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 29), '*', int_343, i_344)
        
        int_346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 37), 'int')
        # Applying the binary operator '%' (line 126)
        result_mod_347 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 35), '%', result_mul_345, int_346)
        
        float_348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 43), 'float')
        # Applying the binary operator 'div' (line 126)
        result_div_349 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 28), 'div', result_mod_347, float_348)
        
        int_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 29), 'int')
        # Getting the type of 'i' (line 127)
        i_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 33), 'i', False)
        # Applying the binary operator '*' (line 127)
        result_mul_352 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 29), '*', int_350, i_351)
        
        int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 37), 'int')
        # Applying the binary operator '%' (line 127)
        result_mod_354 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 35), '%', result_mul_352, int_353)
        
        float_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 43), 'float')
        # Applying the binary operator 'div' (line 127)
        result_div_356 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 28), 'div', result_mod_354, float_355)
        
        # Processing the call keyword arguments (line 125)
        kwargs_357 = {}
        # Getting the type of 'Color' (line 125)
        Color_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 22), 'Color', False)
        # Calling Color(args, kwargs) (line 125)
        Color_call_result_358 = invoke(stypy.reporting.localization.Localization(__file__, 125, 22), Color_333, *[result_div_342, result_div_349, result_div_356], **kwargs_357)
        
        # Processing the call keyword arguments (line 125)
        kwargs_359 = {}
        # Getting the type of 'colors' (line 125)
        colors_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'colors', False)
        # Obtaining the member 'append' of a type (line 125)
        append_332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), colors_331, 'append')
        # Calling append(args, kwargs) (line 125)
        append_call_result_360 = invoke(stypy.reporting.localization.Localization(__file__, 125, 8), append_332, *[Color_call_result_358], **kwargs_359)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Multiple assignment of 2 elements.
    
    # Assigning a UnaryOp to a Name (line 129):
    
    # Getting the type of 'FLOAT_MAX' (line 129)
    FLOAT_MAX_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'FLOAT_MAX')
    # Applying the 'usub' unary operator (line 129)
    result___neg___362 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 20), 'usub', FLOAT_MAX_361)
    
    # Assigning a type to the variable 'max_y' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'max_y', result___neg___362)
    
    # Assigning a Name to a Name (line 129):
    # Getting the type of 'max_y' (line 129)
    max_y_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'max_y')
    # Assigning a type to the variable 'max_x' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'max_x', max_y_363)
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Name to a Name (line 130):
    # Getting the type of 'FLOAT_MAX' (line 130)
    FLOAT_MAX_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'FLOAT_MAX')
    # Assigning a type to the variable 'min_y' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'min_y', FLOAT_MAX_364)
    
    # Assigning a Name to a Name (line 130):
    # Getting the type of 'min_y' (line 130)
    min_y_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'min_y')
    # Assigning a type to the variable 'min_x' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'min_x', min_y_365)
    
    # Getting the type of 'points' (line 132)
    points_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 13), 'points')
    # Assigning a type to the variable 'points_366' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'points_366', points_366)
    # Testing if the for loop is going to be iterated (line 132)
    # Testing the type of a for loop iterable (line 132)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 132, 4), points_366)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 132, 4), points_366):
        # Getting the type of the for loop variable (line 132)
        for_loop_var_367 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 132, 4), points_366)
        # Assigning a type to the variable 'p' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'p', for_loop_var_367)
        # SSA begins for a for statement (line 132)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'max_x' (line 133)
        max_x_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'max_x')
        # Getting the type of 'p' (line 133)
        p_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'p')
        # Obtaining the member 'x' of a type (line 133)
        x_370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 19), p_369, 'x')
        # Applying the binary operator '<' (line 133)
        result_lt_371 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 11), '<', max_x_368, x_370)
        
        # Testing if the type of an if condition is none (line 133)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 133, 8), result_lt_371):
            pass
        else:
            
            # Testing the type of an if condition (line 133)
            if_condition_372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 8), result_lt_371)
            # Assigning a type to the variable 'if_condition_372' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'if_condition_372', if_condition_372)
            # SSA begins for if statement (line 133)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 133):
            
            # Assigning a Attribute to a Name (line 133):
            # Getting the type of 'p' (line 133)
            p_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 32), 'p')
            # Obtaining the member 'x' of a type (line 133)
            x_374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 32), p_373, 'x')
            # Assigning a type to the variable 'max_x' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'max_x', x_374)
            # SSA join for if statement (line 133)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'min_x' (line 134)
        min_x_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'min_x')
        # Getting the type of 'p' (line 134)
        p_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 19), 'p')
        # Obtaining the member 'x' of a type (line 134)
        x_377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 19), p_376, 'x')
        # Applying the binary operator '>' (line 134)
        result_gt_378 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 11), '>', min_x_375, x_377)
        
        # Testing if the type of an if condition is none (line 134)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 134, 8), result_gt_378):
            pass
        else:
            
            # Testing the type of an if condition (line 134)
            if_condition_379 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 8), result_gt_378)
            # Assigning a type to the variable 'if_condition_379' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'if_condition_379', if_condition_379)
            # SSA begins for if statement (line 134)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 134):
            
            # Assigning a Attribute to a Name (line 134):
            # Getting the type of 'p' (line 134)
            p_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 32), 'p')
            # Obtaining the member 'x' of a type (line 134)
            x_381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 32), p_380, 'x')
            # Assigning a type to the variable 'min_x' (line 134)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'min_x', x_381)
            # SSA join for if statement (line 134)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'max_y' (line 135)
        max_y_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 11), 'max_y')
        # Getting the type of 'p' (line 135)
        p_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'p')
        # Obtaining the member 'y' of a type (line 135)
        y_384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 19), p_383, 'y')
        # Applying the binary operator '<' (line 135)
        result_lt_385 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 11), '<', max_y_382, y_384)
        
        # Testing if the type of an if condition is none (line 135)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 135, 8), result_lt_385):
            pass
        else:
            
            # Testing the type of an if condition (line 135)
            if_condition_386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 8), result_lt_385)
            # Assigning a type to the variable 'if_condition_386' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'if_condition_386', if_condition_386)
            # SSA begins for if statement (line 135)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 135):
            
            # Assigning a Attribute to a Name (line 135):
            # Getting the type of 'p' (line 135)
            p_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 32), 'p')
            # Obtaining the member 'y' of a type (line 135)
            y_388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 32), p_387, 'y')
            # Assigning a type to the variable 'max_y' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'max_y', y_388)
            # SSA join for if statement (line 135)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'min_y' (line 136)
        min_y_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'min_y')
        # Getting the type of 'p' (line 136)
        p_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 19), 'p')
        # Obtaining the member 'y' of a type (line 136)
        y_391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 19), p_390, 'y')
        # Applying the binary operator '>' (line 136)
        result_gt_392 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 11), '>', min_y_389, y_391)
        
        # Testing if the type of an if condition is none (line 136)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 136, 8), result_gt_392):
            pass
        else:
            
            # Testing the type of an if condition (line 136)
            if_condition_393 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 136, 8), result_gt_392)
            # Assigning a type to the variable 'if_condition_393' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'if_condition_393', if_condition_393)
            # SSA begins for if statement (line 136)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 136):
            
            # Assigning a Attribute to a Name (line 136):
            # Getting the type of 'p' (line 136)
            p_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 32), 'p')
            # Obtaining the member 'y' of a type (line 136)
            y_395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 32), p_394, 'y')
            # Assigning a type to the variable 'min_y' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'min_y', y_395)
            # SSA join for if statement (line 136)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Call to a Name (line 138):
    
    # Assigning a Call to a Name (line 138):
    
    # Call to min(...): (line 138)
    # Processing the call arguments (line 138)
    # Getting the type of 'W' (line 138)
    W_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'W', False)
    # Getting the type of 'max_x' (line 138)
    max_x_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'max_x', False)
    # Getting the type of 'min_x' (line 138)
    min_x_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 29), 'min_x', False)
    # Applying the binary operator '-' (line 138)
    result_sub_400 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 21), '-', max_x_398, min_x_399)
    
    # Applying the binary operator 'div' (line 138)
    result_div_401 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 16), 'div', W_397, result_sub_400)
    
    # Getting the type of 'H' (line 139)
    H_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'H', False)
    # Getting the type of 'max_y' (line 139)
    max_y_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 21), 'max_y', False)
    # Getting the type of 'min_y' (line 139)
    min_y_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 29), 'min_y', False)
    # Applying the binary operator '-' (line 139)
    result_sub_405 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 21), '-', max_y_403, min_y_404)
    
    # Applying the binary operator 'div' (line 139)
    result_div_406 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 16), 'div', H_402, result_sub_405)
    
    # Processing the call keyword arguments (line 138)
    kwargs_407 = {}
    # Getting the type of 'min' (line 138)
    min_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'min', False)
    # Calling min(args, kwargs) (line 138)
    min_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), min_396, *[result_div_401, result_div_406], **kwargs_407)
    
    # Assigning a type to the variable 'scale' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'scale', min_call_result_408)
    
    # Assigning a BinOp to a Name (line 140):
    
    # Assigning a BinOp to a Name (line 140):
    # Getting the type of 'max_x' (line 140)
    max_x_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 10), 'max_x')
    # Getting the type of 'min_x' (line 140)
    min_x_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 18), 'min_x')
    # Applying the binary operator '+' (line 140)
    result_add_411 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 10), '+', max_x_409, min_x_410)
    
    int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 27), 'int')
    # Applying the binary operator 'div' (line 140)
    result_div_413 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 9), 'div', result_add_411, int_412)
    
    # Assigning a type to the variable 'cx' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'cx', result_div_413)
    
    # Assigning a BinOp to a Name (line 141):
    
    # Assigning a BinOp to a Name (line 141):
    # Getting the type of 'max_y' (line 141)
    max_y_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 10), 'max_y')
    # Getting the type of 'min_y' (line 141)
    min_y_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 18), 'min_y')
    # Applying the binary operator '+' (line 141)
    result_add_416 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 10), '+', max_y_414, min_y_415)
    
    int_417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 27), 'int')
    # Applying the binary operator 'div' (line 141)
    result_div_418 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 9), 'div', result_add_416, int_417)
    
    # Assigning a type to the variable 'cy' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'cy', result_div_418)
    
    
    # Call to enumerate(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'cluster_centers' (line 151)
    cluster_centers_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 27), 'cluster_centers', False)
    # Processing the call keyword arguments (line 151)
    kwargs_421 = {}
    # Getting the type of 'enumerate' (line 151)
    enumerate_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 17), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 151)
    enumerate_call_result_422 = invoke(stypy.reporting.localization.Localization(__file__, 151, 17), enumerate_419, *[cluster_centers_420], **kwargs_421)
    
    # Assigning a type to the variable 'enumerate_call_result_422' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'enumerate_call_result_422', enumerate_call_result_422)
    # Testing if the for loop is going to be iterated (line 151)
    # Testing the type of a for loop iterable (line 151)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 151, 4), enumerate_call_result_422)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 151, 4), enumerate_call_result_422):
        # Getting the type of the for loop variable (line 151)
        for_loop_var_423 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 151, 4), enumerate_call_result_422)
        # Assigning a type to the variable 'i' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 4), for_loop_var_423, 2, 0))
        # Assigning a type to the variable 'cc' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'cc', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 151, 4), for_loop_var_423, 2, 1))
        # SSA begins for a for statement (line 151)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'points' (line 155)
        points_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 17), 'points')
        # Assigning a type to the variable 'points_424' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'points_424', points_424)
        # Testing if the for loop is going to be iterated (line 155)
        # Testing the type of a for loop iterable (line 155)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 155, 8), points_424)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 155, 8), points_424):
            # Getting the type of the for loop variable (line 155)
            for_loop_var_425 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 155, 8), points_424)
            # Assigning a type to the variable 'p' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'p', for_loop_var_425)
            # SSA begins for a for statement (line 155)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'p' (line 156)
            p_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'p')
            # Obtaining the member 'group' of a type (line 156)
            group_427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 15), p_426, 'group')
            # Getting the type of 'i' (line 156)
            i_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 26), 'i')
            # Applying the binary operator '!=' (line 156)
            result_ne_429 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 15), '!=', group_427, i_428)
            
            # Testing if the type of an if condition is none (line 156)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 156, 12), result_ne_429):
                pass
            else:
                
                # Testing the type of an if condition (line 156)
                if_condition_430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 12), result_ne_429)
                # Assigning a type to the variable 'if_condition_430' (line 156)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'if_condition_430', if_condition_430)
                # SSA begins for if statement (line 156)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 156)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'print_eps(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_eps' in the type store
    # Getting the type of 'stypy_return_type' (line 122)
    stypy_return_type_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_431)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_eps'
    return stypy_return_type_431

# Assigning a type to the variable 'print_eps' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'print_eps', print_eps)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 167, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = []
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main(...)' code ##################

    
    # Assigning a Num to a Name (line 168):
    
    # Assigning a Num to a Name (line 168):
    int_432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 14), 'int')
    # Assigning a type to the variable 'npoints' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'npoints', int_432)
    
    # Assigning a Num to a Name (line 169):
    
    # Assigning a Num to a Name (line 169):
    int_433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 8), 'int')
    # Assigning a type to the variable 'k' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'k', int_433)
    
    # Assigning a Call to a Name (line 171):
    
    # Assigning a Call to a Name (line 171):
    
    # Call to generate_points(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'npoints' (line 171)
    npoints_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 29), 'npoints', False)
    int_436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 38), 'int')
    # Processing the call keyword arguments (line 171)
    kwargs_437 = {}
    # Getting the type of 'generate_points' (line 171)
    generate_points_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 13), 'generate_points', False)
    # Calling generate_points(args, kwargs) (line 171)
    generate_points_call_result_438 = invoke(stypy.reporting.localization.Localization(__file__, 171, 13), generate_points_434, *[npoints_435, int_436], **kwargs_437)
    
    # Assigning a type to the variable 'points' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'points', generate_points_call_result_438)
    
    # Assigning a Call to a Name (line 172):
    
    # Assigning a Call to a Name (line 172):
    
    # Call to lloyd(...): (line 172)
    # Processing the call arguments (line 172)
    # Getting the type of 'points' (line 172)
    points_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 28), 'points', False)
    # Getting the type of 'k' (line 172)
    k_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 36), 'k', False)
    # Processing the call keyword arguments (line 172)
    kwargs_442 = {}
    # Getting the type of 'lloyd' (line 172)
    lloyd_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 22), 'lloyd', False)
    # Calling lloyd(args, kwargs) (line 172)
    lloyd_call_result_443 = invoke(stypy.reporting.localization.Localization(__file__, 172, 22), lloyd_439, *[points_440, k_441], **kwargs_442)
    
    # Assigning a type to the variable 'cluster_centers' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'cluster_centers', lloyd_call_result_443)
    
    # Call to print_eps(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of 'points' (line 173)
    points_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 14), 'points', False)
    # Getting the type of 'cluster_centers' (line 173)
    cluster_centers_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'cluster_centers', False)
    # Processing the call keyword arguments (line 173)
    kwargs_447 = {}
    # Getting the type of 'print_eps' (line 173)
    print_eps_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'print_eps', False)
    # Calling print_eps(args, kwargs) (line 173)
    print_eps_call_result_448 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), print_eps_444, *[points_445, cluster_centers_446], **kwargs_447)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 167)
    stypy_return_type_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_449)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_449

# Assigning a type to the variable 'main' (line 167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 176, 0, False)
    
    # Passed parameters checking function
    run.stypy_localization = localization
    run.stypy_type_of_self = None
    run.stypy_type_store = module_type_store
    run.stypy_function_name = 'run'
    run.stypy_param_names_list = []
    run.stypy_varargs_param_name = None
    run.stypy_kwargs_param_name = None
    run.stypy_call_defaults = defaults
    run.stypy_call_varargs = varargs
    run.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run(...)' code ##################

    
    # Call to main(...): (line 177)
    # Processing the call keyword arguments (line 177)
    kwargs_451 = {}
    # Getting the type of 'main' (line 177)
    main_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 4), 'main', False)
    # Calling main(args, kwargs) (line 177)
    main_call_result_452 = invoke(stypy.reporting.localization.Localization(__file__, 177, 4), main_450, *[], **kwargs_451)
    
    # Getting the type of 'True' (line 178)
    True_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type', True_453)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_454)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_454

# Assigning a type to the variable 'run' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 0), 'run', run)

# Call to run(...): (line 181)
# Processing the call keyword arguments (line 181)
kwargs_456 = {}
# Getting the type of 'run' (line 181)
run_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 0), 'run', False)
# Calling run(args, kwargs) (line 181)
run_call_result_457 = invoke(stypy.reporting.localization.Localization(__file__, 181, 0), run_455, *[], **kwargs_456)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
