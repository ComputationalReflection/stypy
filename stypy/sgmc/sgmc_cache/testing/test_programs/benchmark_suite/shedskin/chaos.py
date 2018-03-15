
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #   Copyright (C) 2005 Carl Friedrich Bolz
2: 
3: #   MIT license
4: 
5: '''create chaosgame-like fractals
6: '''
7: 
8: import random
9: import math
10: random.seed(1234)
11: import sys
12: import time
13: import os
14: 
15: def Relative(path):
16:     return os.path.join(os.path.dirname(__file__), path)
17: 
18: class GVector(object):
19:     def __init__(self, x = 0, y = 0, z = 0):
20:         self.x = x
21:         self.y = y
22:         self.z = z
23: 
24:     def Mag(self):
25:         return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
26: 
27:     def dist(self, other):
28:         return math.sqrt((self.x - other.x) ** 2 +
29:                          (self.y - other.y) ** 2 +
30:                          (self.z - other.z) ** 2)
31: 
32:     def __add__(self, other):
33:         v = GVector(self.x + other.x, self.y + other.y, self.z + other.z)
34:         return v
35: 
36:     def __sub__(self, other):
37:         return self + other * -1
38: 
39:     def __mul__(self, other):
40:         v = GVector(self.x * other, self.y * other, self.z * other)
41:         return v
42:     
43:     def linear_combination(self, other, l1, l2):
44:         v = GVector(self.x * l1 + other.x * l2,
45:                     self.y * l1 + other.y * l2,
46:                     self.z * l1 + other.z * l2)
47:         return v
48: 
49:     def __str__(self):
50:         return "<%f, %f, %f>" % (self.x, self.y, self.z)
51: 
52:     def __repr__(self):
53:         return "GVector(%f, %f, %f)" % (self.x, self.y, self.z)
54: 
55: def GetKnots(points, degree):
56:     knots = [0] * degree + range(1, len(points) - degree)
57:     knots += [len(points) - degree] * degree
58:     return knots
59: 
60: class Spline(object):
61:     '''Class for representing B-Splines and NURBS of arbitrary degree'''
62:     def __init__(self, points, degree = 3, knots = None):
63:         '''Creates a Spline. points is a list of GVector, degree is the
64: degree of the Spline.'''
65:         if knots == None:
66:             self.knots = GetKnots(points, degree)
67:         else:
68:             if len(points) > len(knots) - degree + 1:
69:                 raise ValueError("too many control points")
70:             elif len(points) < len(knots) - degree + 1:
71:                 raise ValueError("not enough control points")
72:             last = knots[0]
73:             for cur in knots[1:]:
74:                 if cur < last:
75:                     raise ValueError( "knots not strictly increasing")
76:                 last = cur
77:             self.knots = knots
78:         self.points = points
79:         self.degree = degree
80: 
81:     def GetDomain(self):
82:         '''Returns the domain of the B-Spline'''
83:         return (self.knots[self.degree - 1],
84:                 self.knots[len(self.knots) - self.degree])
85: 
86:     def call(self, u):
87:         '''Calculates a point of the B-Spline using de Boors Algorithm'''
88:         dom = self.GetDomain()
89:         if u < dom[0] or u > dom[1]:
90:             raise ValueError("Function value not in domain")
91:         if u == dom[0]:
92:             return self.points[0]
93:         if u == dom[1]:
94:             return self.points[-1]
95:         I = self.GetIndex(u)
96:         d = [self.points[I - self.degree + 1 + ii]
97:              for ii in range(self.degree + 1)]
98:         U = self.knots
99:         for ik in range(1, self.degree + 1):
100:             for ii in range(I - self.degree + ik + 1, I + 2):
101:                 ua = U[ii + self.degree - ik]
102:                 ub = U[ii - 1]
103:                 co1 = (ua - u) / float(ua - ub)
104:                 co2 = (u - ub) / float(ua - ub)
105:                 index = ii - I + self.degree - ik - 1
106:                 d[index] = d[index].linear_combination(d[index + 1], co1, co2)
107:         return d[0]
108: 
109:     def GetIndex(self, u):
110:         dom = self.GetDomain()
111:         for ii in range(self.degree - 1, len(self.knots) - self.degree):
112:             if u >= self.knots[ii] and u < self.knots[ii + 1]:
113:                 I = ii
114:                 break
115:         else:
116:              I = dom[1] - 1
117:         return I
118: 
119:     def __len__(self):
120:         return len(self.points)
121: 
122:     def __repr__(self):
123:         return "Spline(%r, %r, %r)" % (self.points, self.degree, self.knots)
124: 
125:         
126: def save_im(im, fn):
127:     f = open(fn, "wb")
128:     magic = 'P6\n'
129:     maxval = 255
130:     w = len(im)
131:     h = len(im[0])
132:     f.write(magic)
133:     f.write('%i %i\n%i\n' % (w, h, maxval))
134:     for j in range(h):
135:         for i in range(w):
136:             val = im[i][j]
137:             c = val * 255
138:             f.write('%c%c%c' % (c, c, c))
139:     f.close()
140: 
141: class Chaosgame(object):
142:     def __init__(self, splines, thickness=0.1):
143:         self.splines = splines
144:         self.thickness = thickness
145:         self.minx = min([p.x for spl in splines for p in spl.points])
146:         self.miny = min([p.y for spl in splines for p in spl.points])
147:         self.maxx = max([p.x for spl in splines for p in spl.points])
148:         self.maxy = max([p.y for spl in splines for p in spl.points])
149:         self.height = self.maxy - self.miny
150:         self.width = self.maxx - self.minx
151:         self.num_trafos = []
152:         maxlength = thickness * self.width / float(self.height)
153:         for spl in splines:
154:             length = 0
155:             curr = spl.call(0)
156:             for i in range(1, 1000):
157:                 last = curr
158:                 t = 1.0 / 999 * i
159:                 curr = spl.call(t)
160:                 length += curr.dist(last)
161:             self.num_trafos.append(max(1, int(length / maxlength * 1.5)))
162:         self.num_total = reduce(lambda a,b: a+b, self.num_trafos, 0)
163: 
164: 
165:     def get_random_trafo(self):
166:         r = random.randrange(int(self.num_total) + 1)
167:         l = 0
168:         for i in range(len(self.num_trafos)):
169:             if r >= l and r < l + self.num_trafos[i]:
170:                 return i, random.randrange(self.num_trafos[i])
171:             l += self.num_trafos[i]
172:         return len(self.num_trafos) - 1, random.randrange(self.num_trafos[-1])
173: 
174:     def transform_point(self, point, trafo=None):
175:         x = (point.x - self.minx) / float(self.width)
176:         y = (point.y - self.miny) / float(self.height)
177:         if trafo is None:
178:             trafo = self.get_random_trafo()
179:         start, end = self.splines[trafo[0]].GetDomain()
180:         length = end - start
181:         seg_length = length / float(self.num_trafos[trafo[0]])
182:         t = start + seg_length * trafo[1] + seg_length * x
183:         basepoint = self.splines[trafo[0]].call(t)
184:         if t + 1.0/50000 > end:
185:             neighbour = self.splines[trafo[0]].call(t - 1.0/50000)
186:             derivative = neighbour - basepoint
187:         else:
188:             neighbour = self.splines[trafo[0]].call(t + 1.0/50000)
189:             derivative = basepoint - neighbour
190:         if derivative.Mag() != 0:
191:             basepoint.x += derivative.y / derivative.Mag() * (y - 0.5) * \
192:                            self.thickness
193:             basepoint.y += -derivative.x / derivative.Mag() * (y - 0.5) * \
194:                            self.thickness
195:         else:
196:             pass#print "r",
197:         self.truncate(basepoint)
198:         return basepoint
199: 
200:     def truncate(self, point):
201:         if point.x >= self.maxx:
202:             point.x = self.maxx
203:         if point.y >= self.maxy:
204:             point.y = self.maxy
205:         if point.x < self.minx:
206:             point.x = self.minx
207:         if point.y < self.miny:
208:             point.y = self.miny
209: 
210:     def create_image_chaos(self, w, h, name, n):
211:         im = [[1] * h for i in range(w)]
212:         point = GVector((self.maxx + self.minx) / 2.0,
213:                         (self.maxy + self.miny) / 2.0, 0)
214:         colored = 0
215:         times = []
216:         for _ in range(n):
217:             t1 = time.time()
218:             for i in xrange(5000):
219:                 point = self.transform_point(point)
220:                 x = (point.x - self.minx) / self.width * w
221:                 y = (point.y - self.miny) / self.height * h
222:                 x = int(x)
223:                 y = int(y)
224:                 if x == w:
225:                     x -= 1
226:                 if y == h:
227:                     y -= 1
228:                 im[x][h - y - 1] = 0
229:             t2 = time.time()
230:             times.append(t2 - t1)
231:         save_im(im, name)
232:         return times
233: 
234: 
235: def main(n):
236:     splines = [
237:         Spline([
238:             GVector(1.597350, 3.304460, 0.000000),
239:             GVector(1.575810, 4.123260, 0.000000),
240:             GVector(1.313210, 5.288350, 0.000000),
241:             GVector(1.618900, 5.329910, 0.000000),
242:             GVector(2.889940, 5.502700, 0.000000),
243:             GVector(2.373060, 4.381830, 0.000000),
244:             GVector(1.662000, 4.360280, 0.000000)],
245:             3, [0, 0, 0, 1, 1, 1, 2, 2, 2]),
246:         Spline([
247:             GVector(2.804500, 4.017350, 0.000000),
248:             GVector(2.550500, 3.525230, 0.000000),
249:             GVector(1.979010, 2.620360, 0.000000),
250:             GVector(1.979010, 2.620360, 0.000000)],
251:             3, [0, 0, 0, 1, 1, 1]),
252:         Spline([
253:             GVector(2.001670, 4.011320, 0.000000),
254:             GVector(2.335040, 3.312830, 0.000000),
255:             GVector(2.366800, 3.233460, 0.000000),
256:             GVector(2.366800, 3.233460, 0.000000)],
257:             3, [0, 0, 0, 1, 1, 1])
258:         ]
259:     c = Chaosgame(splines, 0.25)
260:     return c.create_image_chaos(1000, 1200, Relative("py.ppm"), n)
261: 
262: 
263: def run():
264:     main(50)
265:     return True
266: 
267: run()
268: 
269: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, (-1)), 'str', 'create chaosgame-like fractals\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import random' statement (line 8)
import random

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'random', random, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import math' statement (line 9)
import math

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'math', math, module_type_store)


# Call to seed(...): (line 10)
# Processing the call arguments (line 10)
int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 12), 'int')
# Processing the call keyword arguments (line 10)
kwargs_8 = {}
# Getting the type of 'random' (line 10)
random_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'random', False)
# Obtaining the member 'seed' of a type (line 10)
seed_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 0), random_5, 'seed')
# Calling seed(args, kwargs) (line 10)
seed_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 10, 0), seed_6, *[int_7], **kwargs_8)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import sys' statement (line 11)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import time' statement (line 12)
import time

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import os' statement (line 13)
import os

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'os', os, module_type_store)


@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 15, 0, False)
    
    # Passed parameters checking function
    Relative.stypy_localization = localization
    Relative.stypy_type_of_self = None
    Relative.stypy_type_store = module_type_store
    Relative.stypy_function_name = 'Relative'
    Relative.stypy_param_names_list = ['path']
    Relative.stypy_varargs_param_name = None
    Relative.stypy_kwargs_param_name = None
    Relative.stypy_call_defaults = defaults
    Relative.stypy_call_varargs = varargs
    Relative.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Relative', ['path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Relative', localization, ['path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Relative(...)' code ##################

    
    # Call to join(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Call to dirname(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of '__file__' (line 16)
    file___16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 40), '__file__', False)
    # Processing the call keyword arguments (line 16)
    kwargs_17 = {}
    # Getting the type of 'os' (line 16)
    os_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 16)
    path_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 24), os_13, 'path')
    # Obtaining the member 'dirname' of a type (line 16)
    dirname_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 24), path_14, 'dirname')
    # Calling dirname(args, kwargs) (line 16)
    dirname_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 16, 24), dirname_15, *[file___16], **kwargs_17)
    
    # Getting the type of 'path' (line 16)
    path_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 51), 'path', False)
    # Processing the call keyword arguments (line 16)
    kwargs_20 = {}
    # Getting the type of 'os' (line 16)
    os_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 16)
    path_11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 11), os_10, 'path')
    # Obtaining the member 'join' of a type (line 16)
    join_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 11), path_11, 'join')
    # Calling join(args, kwargs) (line 16)
    join_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 16, 11), join_12, *[dirname_call_result_18, path_19], **kwargs_20)
    
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type', join_call_result_21)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_22)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_22

# Assigning a type to the variable 'Relative' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'Relative', Relative)
# Declaration of the 'GVector' class

class GVector(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 27), 'int')
        int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 34), 'int')
        int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 41), 'int')
        defaults = [int_23, int_24, int_25]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.__init__', ['x', 'y', 'z'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x', 'y', 'z'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 20):
        
        # Assigning a Name to a Attribute (line 20):
        # Getting the type of 'x' (line 20)
        x_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'x')
        # Getting the type of 'self' (line 20)
        self_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self')
        # Setting the type of the member 'x' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_27, 'x', x_26)
        
        # Assigning a Name to a Attribute (line 21):
        
        # Assigning a Name to a Attribute (line 21):
        # Getting the type of 'y' (line 21)
        y_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'y')
        # Getting the type of 'self' (line 21)
        self_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self')
        # Setting the type of the member 'y' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_29, 'y', y_28)
        
        # Assigning a Name to a Attribute (line 22):
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'z' (line 22)
        z_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'z')
        # Getting the type of 'self' (line 22)
        self_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'z' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_31, 'z', z_30)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def Mag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'Mag'
        module_type_store = module_type_store.open_function_context('Mag', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.Mag.__dict__.__setitem__('stypy_localization', localization)
        GVector.Mag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.Mag.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.Mag.__dict__.__setitem__('stypy_function_name', 'GVector.Mag')
        GVector.Mag.__dict__.__setitem__('stypy_param_names_list', [])
        GVector.Mag.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.Mag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.Mag.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.Mag.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.Mag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.Mag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.Mag', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'Mag', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'Mag(...)' code ##################

        
        # Call to sqrt(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'self' (line 25)
        self_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'self', False)
        # Obtaining the member 'x' of a type (line 25)
        x_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 25), self_34, 'x')
        int_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 35), 'int')
        # Applying the binary operator '**' (line 25)
        result_pow_37 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 25), '**', x_35, int_36)
        
        # Getting the type of 'self' (line 25)
        self_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'self', False)
        # Obtaining the member 'y' of a type (line 25)
        y_39 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 39), self_38, 'y')
        int_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 49), 'int')
        # Applying the binary operator '**' (line 25)
        result_pow_41 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 39), '**', y_39, int_40)
        
        # Applying the binary operator '+' (line 25)
        result_add_42 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 25), '+', result_pow_37, result_pow_41)
        
        # Getting the type of 'self' (line 25)
        self_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 53), 'self', False)
        # Obtaining the member 'z' of a type (line 25)
        z_44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 53), self_43, 'z')
        int_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 63), 'int')
        # Applying the binary operator '**' (line 25)
        result_pow_46 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 53), '**', z_44, int_45)
        
        # Applying the binary operator '+' (line 25)
        result_add_47 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 51), '+', result_add_42, result_pow_46)
        
        # Processing the call keyword arguments (line 25)
        kwargs_48 = {}
        # Getting the type of 'math' (line 25)
        math_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 25)
        sqrt_33 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 15), math_32, 'sqrt')
        # Calling sqrt(args, kwargs) (line 25)
        sqrt_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), sqrt_33, *[result_add_47], **kwargs_48)
        
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', sqrt_call_result_49)
        
        # ################# End of 'Mag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'Mag' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_50)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'Mag'
        return stypy_return_type_50


    @norecursion
    def dist(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dist'
        module_type_store = module_type_store.open_function_context('dist', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.dist.__dict__.__setitem__('stypy_localization', localization)
        GVector.dist.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.dist.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.dist.__dict__.__setitem__('stypy_function_name', 'GVector.dist')
        GVector.dist.__dict__.__setitem__('stypy_param_names_list', ['other'])
        GVector.dist.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.dist.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.dist.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.dist.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.dist.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.dist.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.dist', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dist', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dist(...)' code ##################

        
        # Call to sqrt(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'self' (line 28)
        self_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'self', False)
        # Obtaining the member 'x' of a type (line 28)
        x_54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 26), self_53, 'x')
        # Getting the type of 'other' (line 28)
        other_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 35), 'other', False)
        # Obtaining the member 'x' of a type (line 28)
        x_56 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 35), other_55, 'x')
        # Applying the binary operator '-' (line 28)
        result_sub_57 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 26), '-', x_54, x_56)
        
        int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 47), 'int')
        # Applying the binary operator '**' (line 28)
        result_pow_59 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 25), '**', result_sub_57, int_58)
        
        # Getting the type of 'self' (line 29)
        self_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 26), 'self', False)
        # Obtaining the member 'y' of a type (line 29)
        y_61 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 26), self_60, 'y')
        # Getting the type of 'other' (line 29)
        other_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 35), 'other', False)
        # Obtaining the member 'y' of a type (line 29)
        y_63 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 35), other_62, 'y')
        # Applying the binary operator '-' (line 29)
        result_sub_64 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 26), '-', y_61, y_63)
        
        int_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 47), 'int')
        # Applying the binary operator '**' (line 29)
        result_pow_66 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 25), '**', result_sub_64, int_65)
        
        # Applying the binary operator '+' (line 28)
        result_add_67 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 25), '+', result_pow_59, result_pow_66)
        
        # Getting the type of 'self' (line 30)
        self_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'self', False)
        # Obtaining the member 'z' of a type (line 30)
        z_69 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 26), self_68, 'z')
        # Getting the type of 'other' (line 30)
        other_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 35), 'other', False)
        # Obtaining the member 'z' of a type (line 30)
        z_71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 35), other_70, 'z')
        # Applying the binary operator '-' (line 30)
        result_sub_72 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 26), '-', z_69, z_71)
        
        int_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 47), 'int')
        # Applying the binary operator '**' (line 30)
        result_pow_74 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 25), '**', result_sub_72, int_73)
        
        # Applying the binary operator '+' (line 29)
        result_add_75 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 49), '+', result_add_67, result_pow_74)
        
        # Processing the call keyword arguments (line 28)
        kwargs_76 = {}
        # Getting the type of 'math' (line 28)
        math_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'math', False)
        # Obtaining the member 'sqrt' of a type (line 28)
        sqrt_52 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 15), math_51, 'sqrt')
        # Calling sqrt(args, kwargs) (line 28)
        sqrt_call_result_77 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), sqrt_52, *[result_add_75], **kwargs_76)
        
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', sqrt_call_result_77)
        
        # ################# End of 'dist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dist' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_78)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dist'
        return stypy_return_type_78


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.__add__.__dict__.__setitem__('stypy_localization', localization)
        GVector.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.__add__.__dict__.__setitem__('stypy_function_name', 'GVector.__add__')
        GVector.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        GVector.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.__add__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add__(...)' code ##################

        
        # Assigning a Call to a Name (line 33):
        
        # Assigning a Call to a Name (line 33):
        
        # Call to GVector(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'self' (line 33)
        self_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'self', False)
        # Obtaining the member 'x' of a type (line 33)
        x_81 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 20), self_80, 'x')
        # Getting the type of 'other' (line 33)
        other_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'other', False)
        # Obtaining the member 'x' of a type (line 33)
        x_83 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 29), other_82, 'x')
        # Applying the binary operator '+' (line 33)
        result_add_84 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 20), '+', x_81, x_83)
        
        # Getting the type of 'self' (line 33)
        self_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 38), 'self', False)
        # Obtaining the member 'y' of a type (line 33)
        y_86 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 38), self_85, 'y')
        # Getting the type of 'other' (line 33)
        other_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 47), 'other', False)
        # Obtaining the member 'y' of a type (line 33)
        y_88 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 47), other_87, 'y')
        # Applying the binary operator '+' (line 33)
        result_add_89 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 38), '+', y_86, y_88)
        
        # Getting the type of 'self' (line 33)
        self_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 56), 'self', False)
        # Obtaining the member 'z' of a type (line 33)
        z_91 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 56), self_90, 'z')
        # Getting the type of 'other' (line 33)
        other_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 65), 'other', False)
        # Obtaining the member 'z' of a type (line 33)
        z_93 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 65), other_92, 'z')
        # Applying the binary operator '+' (line 33)
        result_add_94 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 56), '+', z_91, z_93)
        
        # Processing the call keyword arguments (line 33)
        kwargs_95 = {}
        # Getting the type of 'GVector' (line 33)
        GVector_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'GVector', False)
        # Calling GVector(args, kwargs) (line 33)
        GVector_call_result_96 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), GVector_79, *[result_add_84, result_add_89, result_add_94], **kwargs_95)
        
        # Assigning a type to the variable 'v' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'v', GVector_call_result_96)
        # Getting the type of 'v' (line 34)
        v_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'v')
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', v_97)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_98)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_98


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.__sub__.__dict__.__setitem__('stypy_localization', localization)
        GVector.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.__sub__.__dict__.__setitem__('stypy_function_name', 'GVector.__sub__')
        GVector.__sub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        GVector.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.__sub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__sub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__sub__(...)' code ##################

        # Getting the type of 'self' (line 37)
        self_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'self')
        # Getting the type of 'other' (line 37)
        other_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'other')
        int_101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 30), 'int')
        # Applying the binary operator '*' (line 37)
        result_mul_102 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 22), '*', other_100, int_101)
        
        # Applying the binary operator '+' (line 37)
        result_add_103 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 15), '+', self_99, result_mul_102)
        
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'stypy_return_type', result_add_103)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_104)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_104


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.__mul__.__dict__.__setitem__('stypy_localization', localization)
        GVector.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.__mul__.__dict__.__setitem__('stypy_function_name', 'GVector.__mul__')
        GVector.__mul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        GVector.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.__mul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mul__(...)' code ##################

        
        # Assigning a Call to a Name (line 40):
        
        # Assigning a Call to a Name (line 40):
        
        # Call to GVector(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'self', False)
        # Obtaining the member 'x' of a type (line 40)
        x_107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 20), self_106, 'x')
        # Getting the type of 'other' (line 40)
        other_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 29), 'other', False)
        # Applying the binary operator '*' (line 40)
        result_mul_109 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 20), '*', x_107, other_108)
        
        # Getting the type of 'self' (line 40)
        self_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 36), 'self', False)
        # Obtaining the member 'y' of a type (line 40)
        y_111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 36), self_110, 'y')
        # Getting the type of 'other' (line 40)
        other_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 45), 'other', False)
        # Applying the binary operator '*' (line 40)
        result_mul_113 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 36), '*', y_111, other_112)
        
        # Getting the type of 'self' (line 40)
        self_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 52), 'self', False)
        # Obtaining the member 'z' of a type (line 40)
        z_115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 52), self_114, 'z')
        # Getting the type of 'other' (line 40)
        other_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 61), 'other', False)
        # Applying the binary operator '*' (line 40)
        result_mul_117 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 52), '*', z_115, other_116)
        
        # Processing the call keyword arguments (line 40)
        kwargs_118 = {}
        # Getting the type of 'GVector' (line 40)
        GVector_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'GVector', False)
        # Calling GVector(args, kwargs) (line 40)
        GVector_call_result_119 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), GVector_105, *[result_mul_109, result_mul_113, result_mul_117], **kwargs_118)
        
        # Assigning a type to the variable 'v' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'v', GVector_call_result_119)
        # Getting the type of 'v' (line 41)
        v_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'v')
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', v_120)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_121)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_121


    @norecursion
    def linear_combination(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'linear_combination'
        module_type_store = module_type_store.open_function_context('linear_combination', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.linear_combination.__dict__.__setitem__('stypy_localization', localization)
        GVector.linear_combination.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.linear_combination.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.linear_combination.__dict__.__setitem__('stypy_function_name', 'GVector.linear_combination')
        GVector.linear_combination.__dict__.__setitem__('stypy_param_names_list', ['other', 'l1', 'l2'])
        GVector.linear_combination.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.linear_combination.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.linear_combination.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.linear_combination.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.linear_combination.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.linear_combination.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.linear_combination', ['other', 'l1', 'l2'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'linear_combination', localization, ['other', 'l1', 'l2'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'linear_combination(...)' code ##################

        
        # Assigning a Call to a Name (line 44):
        
        # Assigning a Call to a Name (line 44):
        
        # Call to GVector(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'self' (line 44)
        self_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'self', False)
        # Obtaining the member 'x' of a type (line 44)
        x_124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), self_123, 'x')
        # Getting the type of 'l1' (line 44)
        l1_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 29), 'l1', False)
        # Applying the binary operator '*' (line 44)
        result_mul_126 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 20), '*', x_124, l1_125)
        
        # Getting the type of 'other' (line 44)
        other_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 34), 'other', False)
        # Obtaining the member 'x' of a type (line 44)
        x_128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 34), other_127, 'x')
        # Getting the type of 'l2' (line 44)
        l2_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 44), 'l2', False)
        # Applying the binary operator '*' (line 44)
        result_mul_130 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 34), '*', x_128, l2_129)
        
        # Applying the binary operator '+' (line 44)
        result_add_131 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 20), '+', result_mul_126, result_mul_130)
        
        # Getting the type of 'self' (line 45)
        self_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'self', False)
        # Obtaining the member 'y' of a type (line 45)
        y_133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 20), self_132, 'y')
        # Getting the type of 'l1' (line 45)
        l1_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'l1', False)
        # Applying the binary operator '*' (line 45)
        result_mul_135 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 20), '*', y_133, l1_134)
        
        # Getting the type of 'other' (line 45)
        other_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'other', False)
        # Obtaining the member 'y' of a type (line 45)
        y_137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 34), other_136, 'y')
        # Getting the type of 'l2' (line 45)
        l2_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 44), 'l2', False)
        # Applying the binary operator '*' (line 45)
        result_mul_139 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 34), '*', y_137, l2_138)
        
        # Applying the binary operator '+' (line 45)
        result_add_140 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 20), '+', result_mul_135, result_mul_139)
        
        # Getting the type of 'self' (line 46)
        self_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'self', False)
        # Obtaining the member 'z' of a type (line 46)
        z_142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 20), self_141, 'z')
        # Getting the type of 'l1' (line 46)
        l1_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'l1', False)
        # Applying the binary operator '*' (line 46)
        result_mul_144 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 20), '*', z_142, l1_143)
        
        # Getting the type of 'other' (line 46)
        other_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'other', False)
        # Obtaining the member 'z' of a type (line 46)
        z_146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 34), other_145, 'z')
        # Getting the type of 'l2' (line 46)
        l2_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 44), 'l2', False)
        # Applying the binary operator '*' (line 46)
        result_mul_148 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 34), '*', z_146, l2_147)
        
        # Applying the binary operator '+' (line 46)
        result_add_149 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 20), '+', result_mul_144, result_mul_148)
        
        # Processing the call keyword arguments (line 44)
        kwargs_150 = {}
        # Getting the type of 'GVector' (line 44)
        GVector_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'GVector', False)
        # Calling GVector(args, kwargs) (line 44)
        GVector_call_result_151 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), GVector_122, *[result_add_131, result_add_140, result_add_149], **kwargs_150)
        
        # Assigning a type to the variable 'v' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'v', GVector_call_result_151)
        # Getting the type of 'v' (line 47)
        v_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'v')
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', v_152)
        
        # ################# End of 'linear_combination(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'linear_combination' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_153)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'linear_combination'
        return stypy_return_type_153


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        GVector.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.stypy__str__.__dict__.__setitem__('stypy_function_name', 'GVector.stypy__str__')
        GVector.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        GVector.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        str_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 15), 'str', '<%f, %f, %f>')
        
        # Obtaining an instance of the builtin type 'tuple' (line 50)
        tuple_155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 50)
        # Adding element type (line 50)
        # Getting the type of 'self' (line 50)
        self_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 33), 'self')
        # Obtaining the member 'x' of a type (line 50)
        x_157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 33), self_156, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 33), tuple_155, x_157)
        # Adding element type (line 50)
        # Getting the type of 'self' (line 50)
        self_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 41), 'self')
        # Obtaining the member 'y' of a type (line 50)
        y_159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 41), self_158, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 33), tuple_155, y_159)
        # Adding element type (line 50)
        # Getting the type of 'self' (line 50)
        self_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 49), 'self')
        # Obtaining the member 'z' of a type (line 50)
        z_161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 49), self_160, 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 33), tuple_155, z_161)
        
        # Applying the binary operator '%' (line 50)
        result_mod_162 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), '%', str_154, tuple_155)
        
        # Assigning a type to the variable 'stypy_return_type' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type', result_mod_162)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_163)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_163


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        GVector.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'GVector.stypy__repr__')
        GVector.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        GVector.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        GVector.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'GVector.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        str_164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 15), 'str', 'GVector(%f, %f, %f)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 53)
        tuple_165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 53)
        # Adding element type (line 53)
        # Getting the type of 'self' (line 53)
        self_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 40), 'self')
        # Obtaining the member 'x' of a type (line 53)
        x_167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 40), self_166, 'x')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 40), tuple_165, x_167)
        # Adding element type (line 53)
        # Getting the type of 'self' (line 53)
        self_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 48), 'self')
        # Obtaining the member 'y' of a type (line 53)
        y_169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 48), self_168, 'y')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 40), tuple_165, y_169)
        # Adding element type (line 53)
        # Getting the type of 'self' (line 53)
        self_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 56), 'self')
        # Obtaining the member 'z' of a type (line 53)
        z_171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 56), self_170, 'z')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 40), tuple_165, z_171)
        
        # Applying the binary operator '%' (line 53)
        result_mod_172 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 15), '%', str_164, tuple_165)
        
        # Assigning a type to the variable 'stypy_return_type' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', result_mod_172)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_173)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_173


# Assigning a type to the variable 'GVector' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'GVector', GVector)

@norecursion
def GetKnots(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'GetKnots'
    module_type_store = module_type_store.open_function_context('GetKnots', 55, 0, False)
    
    # Passed parameters checking function
    GetKnots.stypy_localization = localization
    GetKnots.stypy_type_of_self = None
    GetKnots.stypy_type_store = module_type_store
    GetKnots.stypy_function_name = 'GetKnots'
    GetKnots.stypy_param_names_list = ['points', 'degree']
    GetKnots.stypy_varargs_param_name = None
    GetKnots.stypy_kwargs_param_name = None
    GetKnots.stypy_call_defaults = defaults
    GetKnots.stypy_call_varargs = varargs
    GetKnots.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'GetKnots', ['points', 'degree'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'GetKnots', localization, ['points', 'degree'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'GetKnots(...)' code ##################

    
    # Assigning a BinOp to a Name (line 56):
    
    # Assigning a BinOp to a Name (line 56):
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 12), list_174, int_175)
    
    # Getting the type of 'degree' (line 56)
    degree_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'degree')
    # Applying the binary operator '*' (line 56)
    result_mul_177 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 12), '*', list_174, degree_176)
    
    
    # Call to range(...): (line 56)
    # Processing the call arguments (line 56)
    int_179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 33), 'int')
    
    # Call to len(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'points' (line 56)
    points_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 40), 'points', False)
    # Processing the call keyword arguments (line 56)
    kwargs_182 = {}
    # Getting the type of 'len' (line 56)
    len_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'len', False)
    # Calling len(args, kwargs) (line 56)
    len_call_result_183 = invoke(stypy.reporting.localization.Localization(__file__, 56, 36), len_180, *[points_181], **kwargs_182)
    
    # Getting the type of 'degree' (line 56)
    degree_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 50), 'degree', False)
    # Applying the binary operator '-' (line 56)
    result_sub_185 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 36), '-', len_call_result_183, degree_184)
    
    # Processing the call keyword arguments (line 56)
    kwargs_186 = {}
    # Getting the type of 'range' (line 56)
    range_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'range', False)
    # Calling range(args, kwargs) (line 56)
    range_call_result_187 = invoke(stypy.reporting.localization.Localization(__file__, 56, 27), range_178, *[int_179, result_sub_185], **kwargs_186)
    
    # Applying the binary operator '+' (line 56)
    result_add_188 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 12), '+', result_mul_177, range_call_result_187)
    
    # Assigning a type to the variable 'knots' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'knots', result_add_188)
    
    # Getting the type of 'knots' (line 57)
    knots_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'knots')
    
    # Obtaining an instance of the builtin type 'list' (line 57)
    list_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 57)
    # Adding element type (line 57)
    
    # Call to len(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'points' (line 57)
    points_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'points', False)
    # Processing the call keyword arguments (line 57)
    kwargs_193 = {}
    # Getting the type of 'len' (line 57)
    len_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 14), 'len', False)
    # Calling len(args, kwargs) (line 57)
    len_call_result_194 = invoke(stypy.reporting.localization.Localization(__file__, 57, 14), len_191, *[points_192], **kwargs_193)
    
    # Getting the type of 'degree' (line 57)
    degree_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'degree')
    # Applying the binary operator '-' (line 57)
    result_sub_196 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 14), '-', len_call_result_194, degree_195)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 13), list_190, result_sub_196)
    
    # Getting the type of 'degree' (line 57)
    degree_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 38), 'degree')
    # Applying the binary operator '*' (line 57)
    result_mul_198 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 13), '*', list_190, degree_197)
    
    # Applying the binary operator '+=' (line 57)
    result_iadd_199 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 4), '+=', knots_189, result_mul_198)
    # Assigning a type to the variable 'knots' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'knots', result_iadd_199)
    
    # Getting the type of 'knots' (line 58)
    knots_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'knots')
    # Assigning a type to the variable 'stypy_return_type' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type', knots_200)
    
    # ################# End of 'GetKnots(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'GetKnots' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_201)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'GetKnots'
    return stypy_return_type_201

# Assigning a type to the variable 'GetKnots' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'GetKnots', GetKnots)
# Declaration of the 'Spline' class

class Spline(object, ):
    str_202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'str', 'Class for representing B-Splines and NURBS of arbitrary degree')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 40), 'int')
        # Getting the type of 'None' (line 62)
        None_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 51), 'None')
        defaults = [int_203, None_204]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spline.__init__', ['points', 'degree', 'knots'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['points', 'degree', 'knots'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', 'Creates a Spline. points is a list of GVector, degree is the\ndegree of the Spline.')
        
        # Type idiom detected: calculating its left and rigth part (line 65)
        # Getting the type of 'knots' (line 65)
        knots_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'knots')
        # Getting the type of 'None' (line 65)
        None_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'None')
        
        (may_be_208, more_types_in_union_209) = may_be_none(knots_206, None_207)

        if may_be_208:

            if more_types_in_union_209:
                # Runtime conditional SSA (line 65)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 66):
            
            # Assigning a Call to a Attribute (line 66):
            
            # Call to GetKnots(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'points' (line 66)
            points_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 34), 'points', False)
            # Getting the type of 'degree' (line 66)
            degree_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 42), 'degree', False)
            # Processing the call keyword arguments (line 66)
            kwargs_213 = {}
            # Getting the type of 'GetKnots' (line 66)
            GetKnots_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 25), 'GetKnots', False)
            # Calling GetKnots(args, kwargs) (line 66)
            GetKnots_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 66, 25), GetKnots_210, *[points_211, degree_212], **kwargs_213)
            
            # Getting the type of 'self' (line 66)
            self_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'self')
            # Setting the type of the member 'knots' of a type (line 66)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), self_215, 'knots', GetKnots_call_result_214)

            if more_types_in_union_209:
                # Runtime conditional SSA for else branch (line 65)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_208) or more_types_in_union_209):
            
            
            # Call to len(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'points' (line 68)
            points_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'points', False)
            # Processing the call keyword arguments (line 68)
            kwargs_218 = {}
            # Getting the type of 'len' (line 68)
            len_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'len', False)
            # Calling len(args, kwargs) (line 68)
            len_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), len_216, *[points_217], **kwargs_218)
            
            
            # Call to len(...): (line 68)
            # Processing the call arguments (line 68)
            # Getting the type of 'knots' (line 68)
            knots_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 33), 'knots', False)
            # Processing the call keyword arguments (line 68)
            kwargs_222 = {}
            # Getting the type of 'len' (line 68)
            len_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 29), 'len', False)
            # Calling len(args, kwargs) (line 68)
            len_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 68, 29), len_220, *[knots_221], **kwargs_222)
            
            # Getting the type of 'degree' (line 68)
            degree_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 42), 'degree')
            # Applying the binary operator '-' (line 68)
            result_sub_225 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 29), '-', len_call_result_223, degree_224)
            
            int_226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 51), 'int')
            # Applying the binary operator '+' (line 68)
            result_add_227 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 49), '+', result_sub_225, int_226)
            
            # Applying the binary operator '>' (line 68)
            result_gt_228 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 15), '>', len_call_result_219, result_add_227)
            
            # Testing if the type of an if condition is none (line 68)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 68, 12), result_gt_228):
                
                
                # Call to len(...): (line 70)
                # Processing the call arguments (line 70)
                # Getting the type of 'points' (line 70)
                points_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'points', False)
                # Processing the call keyword arguments (line 70)
                kwargs_236 = {}
                # Getting the type of 'len' (line 70)
                len_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'len', False)
                # Calling len(args, kwargs) (line 70)
                len_call_result_237 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), len_234, *[points_235], **kwargs_236)
                
                
                # Call to len(...): (line 70)
                # Processing the call arguments (line 70)
                # Getting the type of 'knots' (line 70)
                knots_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'knots', False)
                # Processing the call keyword arguments (line 70)
                kwargs_240 = {}
                # Getting the type of 'len' (line 70)
                len_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 31), 'len', False)
                # Calling len(args, kwargs) (line 70)
                len_call_result_241 = invoke(stypy.reporting.localization.Localization(__file__, 70, 31), len_238, *[knots_239], **kwargs_240)
                
                # Getting the type of 'degree' (line 70)
                degree_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 44), 'degree')
                # Applying the binary operator '-' (line 70)
                result_sub_243 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 31), '-', len_call_result_241, degree_242)
                
                int_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 53), 'int')
                # Applying the binary operator '+' (line 70)
                result_add_245 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 51), '+', result_sub_243, int_244)
                
                # Applying the binary operator '<' (line 70)
                result_lt_246 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 17), '<', len_call_result_237, result_add_245)
                
                # Testing if the type of an if condition is none (line 70)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 17), result_lt_246):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 70)
                    if_condition_247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 17), result_lt_246)
                    # Assigning a type to the variable 'if_condition_247' (line 70)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'if_condition_247', if_condition_247)
                    # SSA begins for if statement (line 70)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to ValueError(...): (line 71)
                    # Processing the call arguments (line 71)
                    str_249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 33), 'str', 'not enough control points')
                    # Processing the call keyword arguments (line 71)
                    kwargs_250 = {}
                    # Getting the type of 'ValueError' (line 71)
                    ValueError_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'ValueError', False)
                    # Calling ValueError(args, kwargs) (line 71)
                    ValueError_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 71, 22), ValueError_248, *[str_249], **kwargs_250)
                    
                    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 71, 16), ValueError_call_result_251, 'raise parameter', BaseException)
                    # SSA join for if statement (line 70)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 68)
                if_condition_229 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 12), result_gt_228)
                # Assigning a type to the variable 'if_condition_229' (line 68)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'if_condition_229', if_condition_229)
                # SSA begins for if statement (line 68)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to ValueError(...): (line 69)
                # Processing the call arguments (line 69)
                str_231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 33), 'str', 'too many control points')
                # Processing the call keyword arguments (line 69)
                kwargs_232 = {}
                # Getting the type of 'ValueError' (line 69)
                ValueError_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'ValueError', False)
                # Calling ValueError(args, kwargs) (line 69)
                ValueError_call_result_233 = invoke(stypy.reporting.localization.Localization(__file__, 69, 22), ValueError_230, *[str_231], **kwargs_232)
                
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 69, 16), ValueError_call_result_233, 'raise parameter', BaseException)
                # SSA branch for the else part of an if statement (line 68)
                module_type_store.open_ssa_branch('else')
                
                
                # Call to len(...): (line 70)
                # Processing the call arguments (line 70)
                # Getting the type of 'points' (line 70)
                points_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'points', False)
                # Processing the call keyword arguments (line 70)
                kwargs_236 = {}
                # Getting the type of 'len' (line 70)
                len_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'len', False)
                # Calling len(args, kwargs) (line 70)
                len_call_result_237 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), len_234, *[points_235], **kwargs_236)
                
                
                # Call to len(...): (line 70)
                # Processing the call arguments (line 70)
                # Getting the type of 'knots' (line 70)
                knots_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'knots', False)
                # Processing the call keyword arguments (line 70)
                kwargs_240 = {}
                # Getting the type of 'len' (line 70)
                len_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 31), 'len', False)
                # Calling len(args, kwargs) (line 70)
                len_call_result_241 = invoke(stypy.reporting.localization.Localization(__file__, 70, 31), len_238, *[knots_239], **kwargs_240)
                
                # Getting the type of 'degree' (line 70)
                degree_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 44), 'degree')
                # Applying the binary operator '-' (line 70)
                result_sub_243 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 31), '-', len_call_result_241, degree_242)
                
                int_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 53), 'int')
                # Applying the binary operator '+' (line 70)
                result_add_245 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 51), '+', result_sub_243, int_244)
                
                # Applying the binary operator '<' (line 70)
                result_lt_246 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 17), '<', len_call_result_237, result_add_245)
                
                # Testing if the type of an if condition is none (line 70)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 17), result_lt_246):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 70)
                    if_condition_247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 17), result_lt_246)
                    # Assigning a type to the variable 'if_condition_247' (line 70)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 'if_condition_247', if_condition_247)
                    # SSA begins for if statement (line 70)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to ValueError(...): (line 71)
                    # Processing the call arguments (line 71)
                    str_249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 33), 'str', 'not enough control points')
                    # Processing the call keyword arguments (line 71)
                    kwargs_250 = {}
                    # Getting the type of 'ValueError' (line 71)
                    ValueError_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'ValueError', False)
                    # Calling ValueError(args, kwargs) (line 71)
                    ValueError_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 71, 22), ValueError_248, *[str_249], **kwargs_250)
                    
                    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 71, 16), ValueError_call_result_251, 'raise parameter', BaseException)
                    # SSA join for if statement (line 70)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 68)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a Subscript to a Name (line 72):
            
            # Assigning a Subscript to a Name (line 72):
            
            # Obtaining the type of the subscript
            int_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'int')
            # Getting the type of 'knots' (line 72)
            knots_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'knots')
            # Obtaining the member '__getitem__' of a type (line 72)
            getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 19), knots_253, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 72)
            subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 72, 19), getitem___254, int_252)
            
            # Assigning a type to the variable 'last' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'last', subscript_call_result_255)
            
            
            # Obtaining the type of the subscript
            int_256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 29), 'int')
            slice_257 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 73, 23), int_256, None, None)
            # Getting the type of 'knots' (line 73)
            knots_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'knots')
            # Obtaining the member '__getitem__' of a type (line 73)
            getitem___259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), knots_258, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 73)
            subscript_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), getitem___259, slice_257)
            
            # Assigning a type to the variable 'subscript_call_result_260' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'subscript_call_result_260', subscript_call_result_260)
            # Testing if the for loop is going to be iterated (line 73)
            # Testing the type of a for loop iterable (line 73)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 73, 12), subscript_call_result_260)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 73, 12), subscript_call_result_260):
                # Getting the type of the for loop variable (line 73)
                for_loop_var_261 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 73, 12), subscript_call_result_260)
                # Assigning a type to the variable 'cur' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'cur', for_loop_var_261)
                # SSA begins for a for statement (line 73)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'cur' (line 74)
                cur_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'cur')
                # Getting the type of 'last' (line 74)
                last_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 25), 'last')
                # Applying the binary operator '<' (line 74)
                result_lt_264 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 19), '<', cur_262, last_263)
                
                # Testing if the type of an if condition is none (line 74)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 16), result_lt_264):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 74)
                    if_condition_265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 16), result_lt_264)
                    # Assigning a type to the variable 'if_condition_265' (line 74)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'if_condition_265', if_condition_265)
                    # SSA begins for if statement (line 74)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to ValueError(...): (line 75)
                    # Processing the call arguments (line 75)
                    str_267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 38), 'str', 'knots not strictly increasing')
                    # Processing the call keyword arguments (line 75)
                    kwargs_268 = {}
                    # Getting the type of 'ValueError' (line 75)
                    ValueError_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'ValueError', False)
                    # Calling ValueError(args, kwargs) (line 75)
                    ValueError_call_result_269 = invoke(stypy.reporting.localization.Localization(__file__, 75, 26), ValueError_266, *[str_267], **kwargs_268)
                    
                    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 75, 20), ValueError_call_result_269, 'raise parameter', BaseException)
                    # SSA join for if statement (line 74)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Name to a Name (line 76):
                
                # Assigning a Name to a Name (line 76):
                # Getting the type of 'cur' (line 76)
                cur_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'cur')
                # Assigning a type to the variable 'last' (line 76)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'last', cur_270)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Name to a Attribute (line 77):
            
            # Assigning a Name to a Attribute (line 77):
            # Getting the type of 'knots' (line 77)
            knots_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'knots')
            # Getting the type of 'self' (line 77)
            self_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'self')
            # Setting the type of the member 'knots' of a type (line 77)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), self_272, 'knots', knots_271)

            if (may_be_208 and more_types_in_union_209):
                # SSA join for if statement (line 65)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 78):
        
        # Assigning a Name to a Attribute (line 78):
        # Getting the type of 'points' (line 78)
        points_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 22), 'points')
        # Getting the type of 'self' (line 78)
        self_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'self')
        # Setting the type of the member 'points' of a type (line 78)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 8), self_274, 'points', points_273)
        
        # Assigning a Name to a Attribute (line 79):
        
        # Assigning a Name to a Attribute (line 79):
        # Getting the type of 'degree' (line 79)
        degree_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), 'degree')
        # Getting the type of 'self' (line 79)
        self_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self')
        # Setting the type of the member 'degree' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_276, 'degree', degree_275)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def GetDomain(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'GetDomain'
        module_type_store = module_type_store.open_function_context('GetDomain', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spline.GetDomain.__dict__.__setitem__('stypy_localization', localization)
        Spline.GetDomain.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spline.GetDomain.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spline.GetDomain.__dict__.__setitem__('stypy_function_name', 'Spline.GetDomain')
        Spline.GetDomain.__dict__.__setitem__('stypy_param_names_list', [])
        Spline.GetDomain.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spline.GetDomain.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spline.GetDomain.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spline.GetDomain.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spline.GetDomain.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spline.GetDomain.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spline.GetDomain', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'GetDomain', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'GetDomain(...)' code ##################

        str_277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'str', 'Returns the domain of the B-Spline')
        
        # Obtaining an instance of the builtin type 'tuple' (line 83)
        tuple_278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 83)
        # Adding element type (line 83)
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 83)
        self_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'self')
        # Obtaining the member 'degree' of a type (line 83)
        degree_280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 27), self_279, 'degree')
        int_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 41), 'int')
        # Applying the binary operator '-' (line 83)
        result_sub_282 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 27), '-', degree_280, int_281)
        
        # Getting the type of 'self' (line 83)
        self_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'self')
        # Obtaining the member 'knots' of a type (line 83)
        knots_284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), self_283, 'knots')
        # Obtaining the member '__getitem__' of a type (line 83)
        getitem___285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 16), knots_284, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 83)
        subscript_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 83, 16), getitem___285, result_sub_282)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 16), tuple_278, subscript_call_result_286)
        # Adding element type (line 83)
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'self' (line 84)
        self_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 31), 'self', False)
        # Obtaining the member 'knots' of a type (line 84)
        knots_289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 31), self_288, 'knots')
        # Processing the call keyword arguments (line 84)
        kwargs_290 = {}
        # Getting the type of 'len' (line 84)
        len_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 27), 'len', False)
        # Calling len(args, kwargs) (line 84)
        len_call_result_291 = invoke(stypy.reporting.localization.Localization(__file__, 84, 27), len_287, *[knots_289], **kwargs_290)
        
        # Getting the type of 'self' (line 84)
        self_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 45), 'self')
        # Obtaining the member 'degree' of a type (line 84)
        degree_293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 45), self_292, 'degree')
        # Applying the binary operator '-' (line 84)
        result_sub_294 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 27), '-', len_call_result_291, degree_293)
        
        # Getting the type of 'self' (line 84)
        self_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'self')
        # Obtaining the member 'knots' of a type (line 84)
        knots_296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 16), self_295, 'knots')
        # Obtaining the member '__getitem__' of a type (line 84)
        getitem___297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 16), knots_296, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 84)
        subscript_call_result_298 = invoke(stypy.reporting.localization.Localization(__file__, 84, 16), getitem___297, result_sub_294)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 16), tuple_278, subscript_call_result_298)
        
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', tuple_278)
        
        # ################# End of 'GetDomain(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'GetDomain' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_299)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'GetDomain'
        return stypy_return_type_299


    @norecursion
    def call(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'call'
        module_type_store = module_type_store.open_function_context('call', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spline.call.__dict__.__setitem__('stypy_localization', localization)
        Spline.call.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spline.call.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spline.call.__dict__.__setitem__('stypy_function_name', 'Spline.call')
        Spline.call.__dict__.__setitem__('stypy_param_names_list', ['u'])
        Spline.call.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spline.call.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spline.call.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spline.call.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spline.call.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spline.call.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spline.call', ['u'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'call', localization, ['u'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'call(...)' code ##################

        str_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'str', 'Calculates a point of the B-Spline using de Boors Algorithm')
        
        # Assigning a Call to a Name (line 88):
        
        # Assigning a Call to a Name (line 88):
        
        # Call to GetDomain(...): (line 88)
        # Processing the call keyword arguments (line 88)
        kwargs_303 = {}
        # Getting the type of 'self' (line 88)
        self_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 14), 'self', False)
        # Obtaining the member 'GetDomain' of a type (line 88)
        GetDomain_302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 14), self_301, 'GetDomain')
        # Calling GetDomain(args, kwargs) (line 88)
        GetDomain_call_result_304 = invoke(stypy.reporting.localization.Localization(__file__, 88, 14), GetDomain_302, *[], **kwargs_303)
        
        # Assigning a type to the variable 'dom' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'dom', GetDomain_call_result_304)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'u' (line 89)
        u_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'u')
        
        # Obtaining the type of the subscript
        int_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 19), 'int')
        # Getting the type of 'dom' (line 89)
        dom_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'dom')
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 15), dom_307, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_309 = invoke(stypy.reporting.localization.Localization(__file__, 89, 15), getitem___308, int_306)
        
        # Applying the binary operator '<' (line 89)
        result_lt_310 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 11), '<', u_305, subscript_call_result_309)
        
        
        # Getting the type of 'u' (line 89)
        u_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'u')
        
        # Obtaining the type of the subscript
        int_312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 33), 'int')
        # Getting the type of 'dom' (line 89)
        dom_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 29), 'dom')
        # Obtaining the member '__getitem__' of a type (line 89)
        getitem___314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 29), dom_313, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 89)
        subscript_call_result_315 = invoke(stypy.reporting.localization.Localization(__file__, 89, 29), getitem___314, int_312)
        
        # Applying the binary operator '>' (line 89)
        result_gt_316 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 25), '>', u_311, subscript_call_result_315)
        
        # Applying the binary operator 'or' (line 89)
        result_or_keyword_317 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 11), 'or', result_lt_310, result_gt_316)
        
        # Testing if the type of an if condition is none (line 89)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 89, 8), result_or_keyword_317):
            pass
        else:
            
            # Testing the type of an if condition (line 89)
            if_condition_318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 8), result_or_keyword_317)
            # Assigning a type to the variable 'if_condition_318' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'if_condition_318', if_condition_318)
            # SSA begins for if statement (line 89)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to ValueError(...): (line 90)
            # Processing the call arguments (line 90)
            str_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 29), 'str', 'Function value not in domain')
            # Processing the call keyword arguments (line 90)
            kwargs_321 = {}
            # Getting the type of 'ValueError' (line 90)
            ValueError_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 90)
            ValueError_call_result_322 = invoke(stypy.reporting.localization.Localization(__file__, 90, 18), ValueError_319, *[str_320], **kwargs_321)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 90, 12), ValueError_call_result_322, 'raise parameter', BaseException)
            # SSA join for if statement (line 89)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'u' (line 91)
        u_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'u')
        
        # Obtaining the type of the subscript
        int_324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 20), 'int')
        # Getting the type of 'dom' (line 91)
        dom_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'dom')
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 16), dom_325, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 91, 16), getitem___326, int_324)
        
        # Applying the binary operator '==' (line 91)
        result_eq_328 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 11), '==', u_323, subscript_call_result_327)
        
        # Testing if the type of an if condition is none (line 91)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 91, 8), result_eq_328):
            pass
        else:
            
            # Testing the type of an if condition (line 91)
            if_condition_329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), result_eq_328)
            # Assigning a type to the variable 'if_condition_329' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_329', if_condition_329)
            # SSA begins for if statement (line 91)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            int_330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 31), 'int')
            # Getting the type of 'self' (line 92)
            self_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'self')
            # Obtaining the member 'points' of a type (line 92)
            points_332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), self_331, 'points')
            # Obtaining the member '__getitem__' of a type (line 92)
            getitem___333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 19), points_332, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 92)
            subscript_call_result_334 = invoke(stypy.reporting.localization.Localization(__file__, 92, 19), getitem___333, int_330)
            
            # Assigning a type to the variable 'stypy_return_type' (line 92)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'stypy_return_type', subscript_call_result_334)
            # SSA join for if statement (line 91)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'u' (line 93)
        u_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'u')
        
        # Obtaining the type of the subscript
        int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 20), 'int')
        # Getting the type of 'dom' (line 93)
        dom_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'dom')
        # Obtaining the member '__getitem__' of a type (line 93)
        getitem___338 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 16), dom_337, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 93)
        subscript_call_result_339 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), getitem___338, int_336)
        
        # Applying the binary operator '==' (line 93)
        result_eq_340 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 11), '==', u_335, subscript_call_result_339)
        
        # Testing if the type of an if condition is none (line 93)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 93, 8), result_eq_340):
            pass
        else:
            
            # Testing the type of an if condition (line 93)
            if_condition_341 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 8), result_eq_340)
            # Assigning a type to the variable 'if_condition_341' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'if_condition_341', if_condition_341)
            # SSA begins for if statement (line 93)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            int_342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 31), 'int')
            # Getting the type of 'self' (line 94)
            self_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'self')
            # Obtaining the member 'points' of a type (line 94)
            points_344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 19), self_343, 'points')
            # Obtaining the member '__getitem__' of a type (line 94)
            getitem___345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 19), points_344, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 94)
            subscript_call_result_346 = invoke(stypy.reporting.localization.Localization(__file__, 94, 19), getitem___345, int_342)
            
            # Assigning a type to the variable 'stypy_return_type' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'stypy_return_type', subscript_call_result_346)
            # SSA join for if statement (line 93)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to GetIndex(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'u' (line 95)
        u_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 26), 'u', False)
        # Processing the call keyword arguments (line 95)
        kwargs_350 = {}
        # Getting the type of 'self' (line 95)
        self_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'self', False)
        # Obtaining the member 'GetIndex' of a type (line 95)
        GetIndex_348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 12), self_347, 'GetIndex')
        # Calling GetIndex(args, kwargs) (line 95)
        GetIndex_call_result_351 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), GetIndex_348, *[u_349], **kwargs_350)
        
        # Assigning a type to the variable 'I' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'I', GetIndex_call_result_351)
        
        # Assigning a ListComp to a Name (line 96):
        
        # Assigning a ListComp to a Name (line 96):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'self' (line 97)
        self_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'self', False)
        # Obtaining the member 'degree' of a type (line 97)
        degree_366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 29), self_365, 'degree')
        int_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 43), 'int')
        # Applying the binary operator '+' (line 97)
        result_add_368 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 29), '+', degree_366, int_367)
        
        # Processing the call keyword arguments (line 97)
        kwargs_369 = {}
        # Getting the type of 'range' (line 97)
        range_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'range', False)
        # Calling range(args, kwargs) (line 97)
        range_call_result_370 = invoke(stypy.reporting.localization.Localization(__file__, 97, 23), range_364, *[result_add_368], **kwargs_369)
        
        comprehension_371 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 13), range_call_result_370)
        # Assigning a type to the variable 'ii' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'ii', comprehension_371)
        
        # Obtaining the type of the subscript
        # Getting the type of 'I' (line 96)
        I_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 25), 'I')
        # Getting the type of 'self' (line 96)
        self_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 29), 'self')
        # Obtaining the member 'degree' of a type (line 96)
        degree_354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 29), self_353, 'degree')
        # Applying the binary operator '-' (line 96)
        result_sub_355 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 25), '-', I_352, degree_354)
        
        int_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 43), 'int')
        # Applying the binary operator '+' (line 96)
        result_add_357 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 41), '+', result_sub_355, int_356)
        
        # Getting the type of 'ii' (line 96)
        ii_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 47), 'ii')
        # Applying the binary operator '+' (line 96)
        result_add_359 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 45), '+', result_add_357, ii_358)
        
        # Getting the type of 'self' (line 96)
        self_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'self')
        # Obtaining the member 'points' of a type (line 96)
        points_361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 13), self_360, 'points')
        # Obtaining the member '__getitem__' of a type (line 96)
        getitem___362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 13), points_361, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 96)
        subscript_call_result_363 = invoke(stypy.reporting.localization.Localization(__file__, 96, 13), getitem___362, result_add_359)
        
        list_372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 13), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 13), list_372, subscript_call_result_363)
        # Assigning a type to the variable 'd' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'd', list_372)
        
        # Assigning a Attribute to a Name (line 98):
        
        # Assigning a Attribute to a Name (line 98):
        # Getting the type of 'self' (line 98)
        self_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'self')
        # Obtaining the member 'knots' of a type (line 98)
        knots_374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), self_373, 'knots')
        # Assigning a type to the variable 'U' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'U', knots_374)
        
        
        # Call to range(...): (line 99)
        # Processing the call arguments (line 99)
        int_376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 24), 'int')
        # Getting the type of 'self' (line 99)
        self_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 27), 'self', False)
        # Obtaining the member 'degree' of a type (line 99)
        degree_378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 27), self_377, 'degree')
        int_379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 41), 'int')
        # Applying the binary operator '+' (line 99)
        result_add_380 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 27), '+', degree_378, int_379)
        
        # Processing the call keyword arguments (line 99)
        kwargs_381 = {}
        # Getting the type of 'range' (line 99)
        range_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'range', False)
        # Calling range(args, kwargs) (line 99)
        range_call_result_382 = invoke(stypy.reporting.localization.Localization(__file__, 99, 18), range_375, *[int_376, result_add_380], **kwargs_381)
        
        # Assigning a type to the variable 'range_call_result_382' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'range_call_result_382', range_call_result_382)
        # Testing if the for loop is going to be iterated (line 99)
        # Testing the type of a for loop iterable (line 99)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 8), range_call_result_382)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 99, 8), range_call_result_382):
            # Getting the type of the for loop variable (line 99)
            for_loop_var_383 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 8), range_call_result_382)
            # Assigning a type to the variable 'ik' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'ik', for_loop_var_383)
            # SSA begins for a for statement (line 99)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 100)
            # Processing the call arguments (line 100)
            # Getting the type of 'I' (line 100)
            I_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 28), 'I', False)
            # Getting the type of 'self' (line 100)
            self_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 32), 'self', False)
            # Obtaining the member 'degree' of a type (line 100)
            degree_387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 32), self_386, 'degree')
            # Applying the binary operator '-' (line 100)
            result_sub_388 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 28), '-', I_385, degree_387)
            
            # Getting the type of 'ik' (line 100)
            ik_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 46), 'ik', False)
            # Applying the binary operator '+' (line 100)
            result_add_390 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 44), '+', result_sub_388, ik_389)
            
            int_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 51), 'int')
            # Applying the binary operator '+' (line 100)
            result_add_392 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 49), '+', result_add_390, int_391)
            
            # Getting the type of 'I' (line 100)
            I_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 54), 'I', False)
            int_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 58), 'int')
            # Applying the binary operator '+' (line 100)
            result_add_395 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 54), '+', I_393, int_394)
            
            # Processing the call keyword arguments (line 100)
            kwargs_396 = {}
            # Getting the type of 'range' (line 100)
            range_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'range', False)
            # Calling range(args, kwargs) (line 100)
            range_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 100, 22), range_384, *[result_add_392, result_add_395], **kwargs_396)
            
            # Assigning a type to the variable 'range_call_result_397' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'range_call_result_397', range_call_result_397)
            # Testing if the for loop is going to be iterated (line 100)
            # Testing the type of a for loop iterable (line 100)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 100, 12), range_call_result_397)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 100, 12), range_call_result_397):
                # Getting the type of the for loop variable (line 100)
                for_loop_var_398 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 100, 12), range_call_result_397)
                # Assigning a type to the variable 'ii' (line 100)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'ii', for_loop_var_398)
                # SSA begins for a for statement (line 100)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Subscript to a Name (line 101):
                
                # Assigning a Subscript to a Name (line 101):
                
                # Obtaining the type of the subscript
                # Getting the type of 'ii' (line 101)
                ii_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 23), 'ii')
                # Getting the type of 'self' (line 101)
                self_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 28), 'self')
                # Obtaining the member 'degree' of a type (line 101)
                degree_401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 28), self_400, 'degree')
                # Applying the binary operator '+' (line 101)
                result_add_402 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 23), '+', ii_399, degree_401)
                
                # Getting the type of 'ik' (line 101)
                ik_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'ik')
                # Applying the binary operator '-' (line 101)
                result_sub_404 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 40), '-', result_add_402, ik_403)
                
                # Getting the type of 'U' (line 101)
                U_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'U')
                # Obtaining the member '__getitem__' of a type (line 101)
                getitem___406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 21), U_405, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 101)
                subscript_call_result_407 = invoke(stypy.reporting.localization.Localization(__file__, 101, 21), getitem___406, result_sub_404)
                
                # Assigning a type to the variable 'ua' (line 101)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'ua', subscript_call_result_407)
                
                # Assigning a Subscript to a Name (line 102):
                
                # Assigning a Subscript to a Name (line 102):
                
                # Obtaining the type of the subscript
                # Getting the type of 'ii' (line 102)
                ii_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'ii')
                int_409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 28), 'int')
                # Applying the binary operator '-' (line 102)
                result_sub_410 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 23), '-', ii_408, int_409)
                
                # Getting the type of 'U' (line 102)
                U_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 21), 'U')
                # Obtaining the member '__getitem__' of a type (line 102)
                getitem___412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 21), U_411, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 102)
                subscript_call_result_413 = invoke(stypy.reporting.localization.Localization(__file__, 102, 21), getitem___412, result_sub_410)
                
                # Assigning a type to the variable 'ub' (line 102)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'ub', subscript_call_result_413)
                
                # Assigning a BinOp to a Name (line 103):
                
                # Assigning a BinOp to a Name (line 103):
                # Getting the type of 'ua' (line 103)
                ua_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'ua')
                # Getting the type of 'u' (line 103)
                u_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 28), 'u')
                # Applying the binary operator '-' (line 103)
                result_sub_416 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 23), '-', ua_414, u_415)
                
                
                # Call to float(...): (line 103)
                # Processing the call arguments (line 103)
                # Getting the type of 'ua' (line 103)
                ua_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 39), 'ua', False)
                # Getting the type of 'ub' (line 103)
                ub_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 44), 'ub', False)
                # Applying the binary operator '-' (line 103)
                result_sub_420 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 39), '-', ua_418, ub_419)
                
                # Processing the call keyword arguments (line 103)
                kwargs_421 = {}
                # Getting the type of 'float' (line 103)
                float_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 33), 'float', False)
                # Calling float(args, kwargs) (line 103)
                float_call_result_422 = invoke(stypy.reporting.localization.Localization(__file__, 103, 33), float_417, *[result_sub_420], **kwargs_421)
                
                # Applying the binary operator 'div' (line 103)
                result_div_423 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 22), 'div', result_sub_416, float_call_result_422)
                
                # Assigning a type to the variable 'co1' (line 103)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'co1', result_div_423)
                
                # Assigning a BinOp to a Name (line 104):
                
                # Assigning a BinOp to a Name (line 104):
                # Getting the type of 'u' (line 104)
                u_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 23), 'u')
                # Getting the type of 'ub' (line 104)
                ub_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'ub')
                # Applying the binary operator '-' (line 104)
                result_sub_426 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 23), '-', u_424, ub_425)
                
                
                # Call to float(...): (line 104)
                # Processing the call arguments (line 104)
                # Getting the type of 'ua' (line 104)
                ua_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 39), 'ua', False)
                # Getting the type of 'ub' (line 104)
                ub_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 44), 'ub', False)
                # Applying the binary operator '-' (line 104)
                result_sub_430 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 39), '-', ua_428, ub_429)
                
                # Processing the call keyword arguments (line 104)
                kwargs_431 = {}
                # Getting the type of 'float' (line 104)
                float_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'float', False)
                # Calling float(args, kwargs) (line 104)
                float_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 104, 33), float_427, *[result_sub_430], **kwargs_431)
                
                # Applying the binary operator 'div' (line 104)
                result_div_433 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 22), 'div', result_sub_426, float_call_result_432)
                
                # Assigning a type to the variable 'co2' (line 104)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'co2', result_div_433)
                
                # Assigning a BinOp to a Name (line 105):
                
                # Assigning a BinOp to a Name (line 105):
                # Getting the type of 'ii' (line 105)
                ii_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 24), 'ii')
                # Getting the type of 'I' (line 105)
                I_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 29), 'I')
                # Applying the binary operator '-' (line 105)
                result_sub_436 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 24), '-', ii_434, I_435)
                
                # Getting the type of 'self' (line 105)
                self_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'self')
                # Obtaining the member 'degree' of a type (line 105)
                degree_438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 33), self_437, 'degree')
                # Applying the binary operator '+' (line 105)
                result_add_439 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 31), '+', result_sub_436, degree_438)
                
                # Getting the type of 'ik' (line 105)
                ik_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 47), 'ik')
                # Applying the binary operator '-' (line 105)
                result_sub_441 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 45), '-', result_add_439, ik_440)
                
                int_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 52), 'int')
                # Applying the binary operator '-' (line 105)
                result_sub_443 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 50), '-', result_sub_441, int_442)
                
                # Assigning a type to the variable 'index' (line 105)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'index', result_sub_443)
                
                # Assigning a Call to a Subscript (line 106):
                
                # Assigning a Call to a Subscript (line 106):
                
                # Call to linear_combination(...): (line 106)
                # Processing the call arguments (line 106)
                
                # Obtaining the type of the subscript
                # Getting the type of 'index' (line 106)
                index_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 57), 'index', False)
                int_450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 65), 'int')
                # Applying the binary operator '+' (line 106)
                result_add_451 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 57), '+', index_449, int_450)
                
                # Getting the type of 'd' (line 106)
                d_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 55), 'd', False)
                # Obtaining the member '__getitem__' of a type (line 106)
                getitem___453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 55), d_452, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 106)
                subscript_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 106, 55), getitem___453, result_add_451)
                
                # Getting the type of 'co1' (line 106)
                co1_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 69), 'co1', False)
                # Getting the type of 'co2' (line 106)
                co2_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 74), 'co2', False)
                # Processing the call keyword arguments (line 106)
                kwargs_457 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'index' (line 106)
                index_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'index', False)
                # Getting the type of 'd' (line 106)
                d_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'd', False)
                # Obtaining the member '__getitem__' of a type (line 106)
                getitem___446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 27), d_445, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 106)
                subscript_call_result_447 = invoke(stypy.reporting.localization.Localization(__file__, 106, 27), getitem___446, index_444)
                
                # Obtaining the member 'linear_combination' of a type (line 106)
                linear_combination_448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 27), subscript_call_result_447, 'linear_combination')
                # Calling linear_combination(args, kwargs) (line 106)
                linear_combination_call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 106, 27), linear_combination_448, *[subscript_call_result_454, co1_455, co2_456], **kwargs_457)
                
                # Getting the type of 'd' (line 106)
                d_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'd')
                # Getting the type of 'index' (line 106)
                index_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 18), 'index')
                # Storing an element on a container (line 106)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 16), d_459, (index_460, linear_combination_call_result_458))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining the type of the subscript
        int_461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 17), 'int')
        # Getting the type of 'd' (line 107)
        d_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'd')
        # Obtaining the member '__getitem__' of a type (line 107)
        getitem___463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 15), d_462, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 107)
        subscript_call_result_464 = invoke(stypy.reporting.localization.Localization(__file__, 107, 15), getitem___463, int_461)
        
        # Assigning a type to the variable 'stypy_return_type' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'stypy_return_type', subscript_call_result_464)
        
        # ################# End of 'call(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'call' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_465)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'call'
        return stypy_return_type_465


    @norecursion
    def GetIndex(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'GetIndex'
        module_type_store = module_type_store.open_function_context('GetIndex', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spline.GetIndex.__dict__.__setitem__('stypy_localization', localization)
        Spline.GetIndex.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spline.GetIndex.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spline.GetIndex.__dict__.__setitem__('stypy_function_name', 'Spline.GetIndex')
        Spline.GetIndex.__dict__.__setitem__('stypy_param_names_list', ['u'])
        Spline.GetIndex.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spline.GetIndex.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spline.GetIndex.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spline.GetIndex.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spline.GetIndex.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spline.GetIndex.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spline.GetIndex', ['u'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'GetIndex', localization, ['u'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'GetIndex(...)' code ##################

        
        # Assigning a Call to a Name (line 110):
        
        # Assigning a Call to a Name (line 110):
        
        # Call to GetDomain(...): (line 110)
        # Processing the call keyword arguments (line 110)
        kwargs_468 = {}
        # Getting the type of 'self' (line 110)
        self_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'self', False)
        # Obtaining the member 'GetDomain' of a type (line 110)
        GetDomain_467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 14), self_466, 'GetDomain')
        # Calling GetDomain(args, kwargs) (line 110)
        GetDomain_call_result_469 = invoke(stypy.reporting.localization.Localization(__file__, 110, 14), GetDomain_467, *[], **kwargs_468)
        
        # Assigning a type to the variable 'dom' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'dom', GetDomain_call_result_469)
        
        
        # Call to range(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'self' (line 111)
        self_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'self', False)
        # Obtaining the member 'degree' of a type (line 111)
        degree_472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), self_471, 'degree')
        int_473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 38), 'int')
        # Applying the binary operator '-' (line 111)
        result_sub_474 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 24), '-', degree_472, int_473)
        
        
        # Call to len(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'self' (line 111)
        self_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 45), 'self', False)
        # Obtaining the member 'knots' of a type (line 111)
        knots_477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 45), self_476, 'knots')
        # Processing the call keyword arguments (line 111)
        kwargs_478 = {}
        # Getting the type of 'len' (line 111)
        len_475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'len', False)
        # Calling len(args, kwargs) (line 111)
        len_call_result_479 = invoke(stypy.reporting.localization.Localization(__file__, 111, 41), len_475, *[knots_477], **kwargs_478)
        
        # Getting the type of 'self' (line 111)
        self_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 59), 'self', False)
        # Obtaining the member 'degree' of a type (line 111)
        degree_481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 59), self_480, 'degree')
        # Applying the binary operator '-' (line 111)
        result_sub_482 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 41), '-', len_call_result_479, degree_481)
        
        # Processing the call keyword arguments (line 111)
        kwargs_483 = {}
        # Getting the type of 'range' (line 111)
        range_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'range', False)
        # Calling range(args, kwargs) (line 111)
        range_call_result_484 = invoke(stypy.reporting.localization.Localization(__file__, 111, 18), range_470, *[result_sub_474, result_sub_482], **kwargs_483)
        
        # Assigning a type to the variable 'range_call_result_484' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'range_call_result_484', range_call_result_484)
        # Testing if the for loop is going to be iterated (line 111)
        # Testing the type of a for loop iterable (line 111)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 8), range_call_result_484)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 111, 8), range_call_result_484):
            # Getting the type of the for loop variable (line 111)
            for_loop_var_485 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 8), range_call_result_484)
            # Assigning a type to the variable 'ii' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'ii', for_loop_var_485)
            # SSA begins for a for statement (line 111)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Evaluating a boolean operation
            
            # Getting the type of 'u' (line 112)
            u_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'u')
            
            # Obtaining the type of the subscript
            # Getting the type of 'ii' (line 112)
            ii_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'ii')
            # Getting the type of 'self' (line 112)
            self_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'self')
            # Obtaining the member 'knots' of a type (line 112)
            knots_489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), self_488, 'knots')
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), knots_489, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_491 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), getitem___490, ii_487)
            
            # Applying the binary operator '>=' (line 112)
            result_ge_492 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 15), '>=', u_486, subscript_call_result_491)
            
            
            # Getting the type of 'u' (line 112)
            u_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 39), 'u')
            
            # Obtaining the type of the subscript
            # Getting the type of 'ii' (line 112)
            ii_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 54), 'ii')
            int_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 59), 'int')
            # Applying the binary operator '+' (line 112)
            result_add_496 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 54), '+', ii_494, int_495)
            
            # Getting the type of 'self' (line 112)
            self_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 43), 'self')
            # Obtaining the member 'knots' of a type (line 112)
            knots_498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 43), self_497, 'knots')
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 43), knots_498, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_500 = invoke(stypy.reporting.localization.Localization(__file__, 112, 43), getitem___499, result_add_496)
            
            # Applying the binary operator '<' (line 112)
            result_lt_501 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 39), '<', u_493, subscript_call_result_500)
            
            # Applying the binary operator 'and' (line 112)
            result_and_keyword_502 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 15), 'and', result_ge_492, result_lt_501)
            
            # Testing if the type of an if condition is none (line 112)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 112, 12), result_and_keyword_502):
                pass
            else:
                
                # Testing the type of an if condition (line 112)
                if_condition_503 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 12), result_and_keyword_502)
                # Assigning a type to the variable 'if_condition_503' (line 112)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'if_condition_503', if_condition_503)
                # SSA begins for if statement (line 112)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 113):
                
                # Assigning a Name to a Name (line 113):
                # Getting the type of 'ii' (line 113)
                ii_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'ii')
                # Assigning a type to the variable 'I' (line 113)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'I', ii_504)
                # SSA join for if statement (line 112)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA branch for the else part of a for statement (line 111)
            module_type_store.open_ssa_branch('for loop else')
            
            # Assigning a BinOp to a Name (line 116):
            
            # Assigning a BinOp to a Name (line 116):
            
            # Obtaining the type of the subscript
            int_505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'int')
            # Getting the type of 'dom' (line 116)
            dom_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'dom')
            # Obtaining the member '__getitem__' of a type (line 116)
            getitem___507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 17), dom_506, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 116)
            subscript_call_result_508 = invoke(stypy.reporting.localization.Localization(__file__, 116, 17), getitem___507, int_505)
            
            int_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'int')
            # Applying the binary operator '-' (line 116)
            result_sub_510 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 17), '-', subscript_call_result_508, int_509)
            
            # Assigning a type to the variable 'I' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 13), 'I', result_sub_510)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
        else:
            
            # Assigning a BinOp to a Name (line 116):
            
            # Assigning a BinOp to a Name (line 116):
            
            # Obtaining the type of the subscript
            int_505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 21), 'int')
            # Getting the type of 'dom' (line 116)
            dom_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 17), 'dom')
            # Obtaining the member '__getitem__' of a type (line 116)
            getitem___507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 17), dom_506, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 116)
            subscript_call_result_508 = invoke(stypy.reporting.localization.Localization(__file__, 116, 17), getitem___507, int_505)
            
            int_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 26), 'int')
            # Applying the binary operator '-' (line 116)
            result_sub_510 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 17), '-', subscript_call_result_508, int_509)
            
            # Assigning a type to the variable 'I' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 13), 'I', result_sub_510)

        
        # Getting the type of 'I' (line 117)
        I_511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'I')
        # Assigning a type to the variable 'stypy_return_type' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', I_511)
        
        # ################# End of 'GetIndex(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'GetIndex' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_512)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'GetIndex'
        return stypy_return_type_512


    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spline.__len__.__dict__.__setitem__('stypy_localization', localization)
        Spline.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spline.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spline.__len__.__dict__.__setitem__('stypy_function_name', 'Spline.__len__')
        Spline.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        Spline.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spline.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spline.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spline.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spline.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spline.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spline.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        
        # Call to len(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'self' (line 120)
        self_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'self', False)
        # Obtaining the member 'points' of a type (line 120)
        points_515 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 19), self_514, 'points')
        # Processing the call keyword arguments (line 120)
        kwargs_516 = {}
        # Getting the type of 'len' (line 120)
        len_513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'len', False)
        # Calling len(args, kwargs) (line 120)
        len_call_result_517 = invoke(stypy.reporting.localization.Localization(__file__, 120, 15), len_513, *[points_515], **kwargs_516)
        
        # Assigning a type to the variable 'stypy_return_type' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', len_call_result_517)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_518)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_518


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Spline.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Spline.stypy__repr__')
        Spline.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Spline.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Spline.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Spline.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        str_519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 15), 'str', 'Spline(%r, %r, %r)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 123)
        tuple_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 123)
        # Adding element type (line 123)
        # Getting the type of 'self' (line 123)
        self_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'self')
        # Obtaining the member 'points' of a type (line 123)
        points_522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 39), self_521, 'points')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 39), tuple_520, points_522)
        # Adding element type (line 123)
        # Getting the type of 'self' (line 123)
        self_523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 52), 'self')
        # Obtaining the member 'degree' of a type (line 123)
        degree_524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 52), self_523, 'degree')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 39), tuple_520, degree_524)
        # Adding element type (line 123)
        # Getting the type of 'self' (line 123)
        self_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 65), 'self')
        # Obtaining the member 'knots' of a type (line 123)
        knots_526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 65), self_525, 'knots')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 39), tuple_520, knots_526)
        
        # Applying the binary operator '%' (line 123)
        result_mod_527 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 15), '%', str_519, tuple_520)
        
        # Assigning a type to the variable 'stypy_return_type' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'stypy_return_type', result_mod_527)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_528)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_528


# Assigning a type to the variable 'Spline' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'Spline', Spline)

@norecursion
def save_im(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'save_im'
    module_type_store = module_type_store.open_function_context('save_im', 126, 0, False)
    
    # Passed parameters checking function
    save_im.stypy_localization = localization
    save_im.stypy_type_of_self = None
    save_im.stypy_type_store = module_type_store
    save_im.stypy_function_name = 'save_im'
    save_im.stypy_param_names_list = ['im', 'fn']
    save_im.stypy_varargs_param_name = None
    save_im.stypy_kwargs_param_name = None
    save_im.stypy_call_defaults = defaults
    save_im.stypy_call_varargs = varargs
    save_im.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'save_im', ['im', 'fn'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'save_im', localization, ['im', 'fn'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'save_im(...)' code ##################

    
    # Assigning a Call to a Name (line 127):
    
    # Assigning a Call to a Name (line 127):
    
    # Call to open(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'fn' (line 127)
    fn_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 13), 'fn', False)
    str_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'str', 'wb')
    # Processing the call keyword arguments (line 127)
    kwargs_532 = {}
    # Getting the type of 'open' (line 127)
    open_529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'open', False)
    # Calling open(args, kwargs) (line 127)
    open_call_result_533 = invoke(stypy.reporting.localization.Localization(__file__, 127, 8), open_529, *[fn_530, str_531], **kwargs_532)
    
    # Assigning a type to the variable 'f' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'f', open_call_result_533)
    
    # Assigning a Str to a Name (line 128):
    
    # Assigning a Str to a Name (line 128):
    str_534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 12), 'str', 'P6\n')
    # Assigning a type to the variable 'magic' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'magic', str_534)
    
    # Assigning a Num to a Name (line 129):
    
    # Assigning a Num to a Name (line 129):
    int_535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 13), 'int')
    # Assigning a type to the variable 'maxval' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'maxval', int_535)
    
    # Assigning a Call to a Name (line 130):
    
    # Assigning a Call to a Name (line 130):
    
    # Call to len(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'im' (line 130)
    im_537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'im', False)
    # Processing the call keyword arguments (line 130)
    kwargs_538 = {}
    # Getting the type of 'len' (line 130)
    len_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'len', False)
    # Calling len(args, kwargs) (line 130)
    len_call_result_539 = invoke(stypy.reporting.localization.Localization(__file__, 130, 8), len_536, *[im_537], **kwargs_538)
    
    # Assigning a type to the variable 'w' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'w', len_call_result_539)
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to len(...): (line 131)
    # Processing the call arguments (line 131)
    
    # Obtaining the type of the subscript
    int_541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 15), 'int')
    # Getting the type of 'im' (line 131)
    im_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'im', False)
    # Obtaining the member '__getitem__' of a type (line 131)
    getitem___543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 12), im_542, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 131)
    subscript_call_result_544 = invoke(stypy.reporting.localization.Localization(__file__, 131, 12), getitem___543, int_541)
    
    # Processing the call keyword arguments (line 131)
    kwargs_545 = {}
    # Getting the type of 'len' (line 131)
    len_540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'len', False)
    # Calling len(args, kwargs) (line 131)
    len_call_result_546 = invoke(stypy.reporting.localization.Localization(__file__, 131, 8), len_540, *[subscript_call_result_544], **kwargs_545)
    
    # Assigning a type to the variable 'h' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'h', len_call_result_546)
    
    # Call to write(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'magic' (line 132)
    magic_549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'magic', False)
    # Processing the call keyword arguments (line 132)
    kwargs_550 = {}
    # Getting the type of 'f' (line 132)
    f_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 132)
    write_548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 4), f_547, 'write')
    # Calling write(args, kwargs) (line 132)
    write_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 132, 4), write_548, *[magic_549], **kwargs_550)
    
    
    # Call to write(...): (line 133)
    # Processing the call arguments (line 133)
    str_554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 12), 'str', '%i %i\n%i\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 133)
    tuple_555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 133)
    # Adding element type (line 133)
    # Getting the type of 'w' (line 133)
    w_556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 29), 'w', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 29), tuple_555, w_556)
    # Adding element type (line 133)
    # Getting the type of 'h' (line 133)
    h_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 32), 'h', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 29), tuple_555, h_557)
    # Adding element type (line 133)
    # Getting the type of 'maxval' (line 133)
    maxval_558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 35), 'maxval', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 29), tuple_555, maxval_558)
    
    # Applying the binary operator '%' (line 133)
    result_mod_559 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 12), '%', str_554, tuple_555)
    
    # Processing the call keyword arguments (line 133)
    kwargs_560 = {}
    # Getting the type of 'f' (line 133)
    f_552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'f', False)
    # Obtaining the member 'write' of a type (line 133)
    write_553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 4), f_552, 'write')
    # Calling write(args, kwargs) (line 133)
    write_call_result_561 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), write_553, *[result_mod_559], **kwargs_560)
    
    
    
    # Call to range(...): (line 134)
    # Processing the call arguments (line 134)
    # Getting the type of 'h' (line 134)
    h_563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 19), 'h', False)
    # Processing the call keyword arguments (line 134)
    kwargs_564 = {}
    # Getting the type of 'range' (line 134)
    range_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 13), 'range', False)
    # Calling range(args, kwargs) (line 134)
    range_call_result_565 = invoke(stypy.reporting.localization.Localization(__file__, 134, 13), range_562, *[h_563], **kwargs_564)
    
    # Assigning a type to the variable 'range_call_result_565' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'range_call_result_565', range_call_result_565)
    # Testing if the for loop is going to be iterated (line 134)
    # Testing the type of a for loop iterable (line 134)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 134, 4), range_call_result_565)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 134, 4), range_call_result_565):
        # Getting the type of the for loop variable (line 134)
        for_loop_var_566 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 134, 4), range_call_result_565)
        # Assigning a type to the variable 'j' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'j', for_loop_var_566)
        # SSA begins for a for statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'w' (line 135)
        w_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 23), 'w', False)
        # Processing the call keyword arguments (line 135)
        kwargs_569 = {}
        # Getting the type of 'range' (line 135)
        range_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 17), 'range', False)
        # Calling range(args, kwargs) (line 135)
        range_call_result_570 = invoke(stypy.reporting.localization.Localization(__file__, 135, 17), range_567, *[w_568], **kwargs_569)
        
        # Assigning a type to the variable 'range_call_result_570' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'range_call_result_570', range_call_result_570)
        # Testing if the for loop is going to be iterated (line 135)
        # Testing the type of a for loop iterable (line 135)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 135, 8), range_call_result_570)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 135, 8), range_call_result_570):
            # Getting the type of the for loop variable (line 135)
            for_loop_var_571 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 135, 8), range_call_result_570)
            # Assigning a type to the variable 'i' (line 135)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'i', for_loop_var_571)
            # SSA begins for a for statement (line 135)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 136):
            
            # Assigning a Subscript to a Name (line 136):
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 136)
            j_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'j')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 136)
            i_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 21), 'i')
            # Getting the type of 'im' (line 136)
            im_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 18), 'im')
            # Obtaining the member '__getitem__' of a type (line 136)
            getitem___575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 18), im_574, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 136)
            subscript_call_result_576 = invoke(stypy.reporting.localization.Localization(__file__, 136, 18), getitem___575, i_573)
            
            # Obtaining the member '__getitem__' of a type (line 136)
            getitem___577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 18), subscript_call_result_576, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 136)
            subscript_call_result_578 = invoke(stypy.reporting.localization.Localization(__file__, 136, 18), getitem___577, j_572)
            
            # Assigning a type to the variable 'val' (line 136)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'val', subscript_call_result_578)
            
            # Assigning a BinOp to a Name (line 137):
            
            # Assigning a BinOp to a Name (line 137):
            # Getting the type of 'val' (line 137)
            val_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'val')
            int_580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 22), 'int')
            # Applying the binary operator '*' (line 137)
            result_mul_581 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 16), '*', val_579, int_580)
            
            # Assigning a type to the variable 'c' (line 137)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 12), 'c', result_mul_581)
            
            # Call to write(...): (line 138)
            # Processing the call arguments (line 138)
            str_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 20), 'str', '%c%c%c')
            
            # Obtaining an instance of the builtin type 'tuple' (line 138)
            tuple_585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 32), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 138)
            # Adding element type (line 138)
            # Getting the type of 'c' (line 138)
            c_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 32), 'c', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 32), tuple_585, c_586)
            # Adding element type (line 138)
            # Getting the type of 'c' (line 138)
            c_587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 35), 'c', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 32), tuple_585, c_587)
            # Adding element type (line 138)
            # Getting the type of 'c' (line 138)
            c_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 38), 'c', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 32), tuple_585, c_588)
            
            # Applying the binary operator '%' (line 138)
            result_mod_589 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 20), '%', str_584, tuple_585)
            
            # Processing the call keyword arguments (line 138)
            kwargs_590 = {}
            # Getting the type of 'f' (line 138)
            f_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'f', False)
            # Obtaining the member 'write' of a type (line 138)
            write_583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), f_582, 'write')
            # Calling write(args, kwargs) (line 138)
            write_call_result_591 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), write_583, *[result_mod_589], **kwargs_590)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to close(...): (line 139)
    # Processing the call keyword arguments (line 139)
    kwargs_594 = {}
    # Getting the type of 'f' (line 139)
    f_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'f', False)
    # Obtaining the member 'close' of a type (line 139)
    close_593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 4), f_592, 'close')
    # Calling close(args, kwargs) (line 139)
    close_call_result_595 = invoke(stypy.reporting.localization.Localization(__file__, 139, 4), close_593, *[], **kwargs_594)
    
    
    # ################# End of 'save_im(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'save_im' in the type store
    # Getting the type of 'stypy_return_type' (line 126)
    stypy_return_type_596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_596)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'save_im'
    return stypy_return_type_596

# Assigning a type to the variable 'save_im' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'save_im', save_im)
# Declaration of the 'Chaosgame' class

class Chaosgame(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        float_597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 42), 'float')
        defaults = [float_597]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 142, 4, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chaosgame.__init__', ['splines', 'thickness'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['splines', 'thickness'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 143):
        
        # Assigning a Name to a Attribute (line 143):
        # Getting the type of 'splines' (line 143)
        splines_598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'splines')
        # Getting the type of 'self' (line 143)
        self_599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'self')
        # Setting the type of the member 'splines' of a type (line 143)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), self_599, 'splines', splines_598)
        
        # Assigning a Name to a Attribute (line 144):
        
        # Assigning a Name to a Attribute (line 144):
        # Getting the type of 'thickness' (line 144)
        thickness_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'thickness')
        # Getting the type of 'self' (line 144)
        self_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'self')
        # Setting the type of the member 'thickness' of a type (line 144)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), self_601, 'thickness', thickness_600)
        
        # Assigning a Call to a Attribute (line 145):
        
        # Assigning a Call to a Attribute (line 145):
        
        # Call to min(...): (line 145)
        # Processing the call arguments (line 145)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'splines' (line 145)
        splines_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 40), 'splines', False)
        comprehension_606 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), splines_605)
        # Assigning a type to the variable 'spl' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'spl', comprehension_606)
        # Calculating comprehension expression
        # Getting the type of 'spl' (line 145)
        spl_607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 57), 'spl', False)
        # Obtaining the member 'points' of a type (line 145)
        points_608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 57), spl_607, 'points')
        comprehension_609 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), points_608)
        # Assigning a type to the variable 'p' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'p', comprehension_609)
        # Getting the type of 'p' (line 145)
        p_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'p', False)
        # Obtaining the member 'x' of a type (line 145)
        x_604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), p_603, 'x')
        list_610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 25), list_610, x_604)
        # Processing the call keyword arguments (line 145)
        kwargs_611 = {}
        # Getting the type of 'min' (line 145)
        min_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 20), 'min', False)
        # Calling min(args, kwargs) (line 145)
        min_call_result_612 = invoke(stypy.reporting.localization.Localization(__file__, 145, 20), min_602, *[list_610], **kwargs_611)
        
        # Getting the type of 'self' (line 145)
        self_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self')
        # Setting the type of the member 'minx' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_613, 'minx', min_call_result_612)
        
        # Assigning a Call to a Attribute (line 146):
        
        # Assigning a Call to a Attribute (line 146):
        
        # Call to min(...): (line 146)
        # Processing the call arguments (line 146)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'splines' (line 146)
        splines_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 40), 'splines', False)
        comprehension_618 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 25), splines_617)
        # Assigning a type to the variable 'spl' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'spl', comprehension_618)
        # Calculating comprehension expression
        # Getting the type of 'spl' (line 146)
        spl_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 57), 'spl', False)
        # Obtaining the member 'points' of a type (line 146)
        points_620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 57), spl_619, 'points')
        comprehension_621 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 25), points_620)
        # Assigning a type to the variable 'p' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'p', comprehension_621)
        # Getting the type of 'p' (line 146)
        p_615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'p', False)
        # Obtaining the member 'y' of a type (line 146)
        y_616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 25), p_615, 'y')
        list_622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 146, 25), list_622, y_616)
        # Processing the call keyword arguments (line 146)
        kwargs_623 = {}
        # Getting the type of 'min' (line 146)
        min_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 20), 'min', False)
        # Calling min(args, kwargs) (line 146)
        min_call_result_624 = invoke(stypy.reporting.localization.Localization(__file__, 146, 20), min_614, *[list_622], **kwargs_623)
        
        # Getting the type of 'self' (line 146)
        self_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self')
        # Setting the type of the member 'miny' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_625, 'miny', min_call_result_624)
        
        # Assigning a Call to a Attribute (line 147):
        
        # Assigning a Call to a Attribute (line 147):
        
        # Call to max(...): (line 147)
        # Processing the call arguments (line 147)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'splines' (line 147)
        splines_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 40), 'splines', False)
        comprehension_630 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 25), splines_629)
        # Assigning a type to the variable 'spl' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'spl', comprehension_630)
        # Calculating comprehension expression
        # Getting the type of 'spl' (line 147)
        spl_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 57), 'spl', False)
        # Obtaining the member 'points' of a type (line 147)
        points_632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 57), spl_631, 'points')
        comprehension_633 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 25), points_632)
        # Assigning a type to the variable 'p' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'p', comprehension_633)
        # Getting the type of 'p' (line 147)
        p_627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'p', False)
        # Obtaining the member 'x' of a type (line 147)
        x_628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 25), p_627, 'x')
        list_634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 25), list_634, x_628)
        # Processing the call keyword arguments (line 147)
        kwargs_635 = {}
        # Getting the type of 'max' (line 147)
        max_626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 20), 'max', False)
        # Calling max(args, kwargs) (line 147)
        max_call_result_636 = invoke(stypy.reporting.localization.Localization(__file__, 147, 20), max_626, *[list_634], **kwargs_635)
        
        # Getting the type of 'self' (line 147)
        self_637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'self')
        # Setting the type of the member 'maxx' of a type (line 147)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 8), self_637, 'maxx', max_call_result_636)
        
        # Assigning a Call to a Attribute (line 148):
        
        # Assigning a Call to a Attribute (line 148):
        
        # Call to max(...): (line 148)
        # Processing the call arguments (line 148)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'splines' (line 148)
        splines_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 40), 'splines', False)
        comprehension_642 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 25), splines_641)
        # Assigning a type to the variable 'spl' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'spl', comprehension_642)
        # Calculating comprehension expression
        # Getting the type of 'spl' (line 148)
        spl_643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 57), 'spl', False)
        # Obtaining the member 'points' of a type (line 148)
        points_644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 57), spl_643, 'points')
        comprehension_645 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 25), points_644)
        # Assigning a type to the variable 'p' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'p', comprehension_645)
        # Getting the type of 'p' (line 148)
        p_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 25), 'p', False)
        # Obtaining the member 'y' of a type (line 148)
        y_640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 25), p_639, 'y')
        list_646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 25), list_646, y_640)
        # Processing the call keyword arguments (line 148)
        kwargs_647 = {}
        # Getting the type of 'max' (line 148)
        max_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 20), 'max', False)
        # Calling max(args, kwargs) (line 148)
        max_call_result_648 = invoke(stypy.reporting.localization.Localization(__file__, 148, 20), max_638, *[list_646], **kwargs_647)
        
        # Getting the type of 'self' (line 148)
        self_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'self')
        # Setting the type of the member 'maxy' of a type (line 148)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 8), self_649, 'maxy', max_call_result_648)
        
        # Assigning a BinOp to a Attribute (line 149):
        
        # Assigning a BinOp to a Attribute (line 149):
        # Getting the type of 'self' (line 149)
        self_650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 22), 'self')
        # Obtaining the member 'maxy' of a type (line 149)
        maxy_651 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 22), self_650, 'maxy')
        # Getting the type of 'self' (line 149)
        self_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 34), 'self')
        # Obtaining the member 'miny' of a type (line 149)
        miny_653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 34), self_652, 'miny')
        # Applying the binary operator '-' (line 149)
        result_sub_654 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 22), '-', maxy_651, miny_653)
        
        # Getting the type of 'self' (line 149)
        self_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self')
        # Setting the type of the member 'height' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_655, 'height', result_sub_654)
        
        # Assigning a BinOp to a Attribute (line 150):
        
        # Assigning a BinOp to a Attribute (line 150):
        # Getting the type of 'self' (line 150)
        self_656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 21), 'self')
        # Obtaining the member 'maxx' of a type (line 150)
        maxx_657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 21), self_656, 'maxx')
        # Getting the type of 'self' (line 150)
        self_658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 33), 'self')
        # Obtaining the member 'minx' of a type (line 150)
        minx_659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 33), self_658, 'minx')
        # Applying the binary operator '-' (line 150)
        result_sub_660 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 21), '-', maxx_657, minx_659)
        
        # Getting the type of 'self' (line 150)
        self_661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'self')
        # Setting the type of the member 'width' of a type (line 150)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), self_661, 'width', result_sub_660)
        
        # Assigning a List to a Attribute (line 151):
        
        # Assigning a List to a Attribute (line 151):
        
        # Obtaining an instance of the builtin type 'list' (line 151)
        list_662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 26), 'list')
        # Adding type elements to the builtin type 'list' instance (line 151)
        
        # Getting the type of 'self' (line 151)
        self_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'self')
        # Setting the type of the member 'num_trafos' of a type (line 151)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 8), self_663, 'num_trafos', list_662)
        
        # Assigning a BinOp to a Name (line 152):
        
        # Assigning a BinOp to a Name (line 152):
        # Getting the type of 'thickness' (line 152)
        thickness_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'thickness')
        # Getting the type of 'self' (line 152)
        self_665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'self')
        # Obtaining the member 'width' of a type (line 152)
        width_666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 32), self_665, 'width')
        # Applying the binary operator '*' (line 152)
        result_mul_667 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 20), '*', thickness_664, width_666)
        
        
        # Call to float(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'self' (line 152)
        self_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 51), 'self', False)
        # Obtaining the member 'height' of a type (line 152)
        height_670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 51), self_669, 'height')
        # Processing the call keyword arguments (line 152)
        kwargs_671 = {}
        # Getting the type of 'float' (line 152)
        float_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 45), 'float', False)
        # Calling float(args, kwargs) (line 152)
        float_call_result_672 = invoke(stypy.reporting.localization.Localization(__file__, 152, 45), float_668, *[height_670], **kwargs_671)
        
        # Applying the binary operator 'div' (line 152)
        result_div_673 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 43), 'div', result_mul_667, float_call_result_672)
        
        # Assigning a type to the variable 'maxlength' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'maxlength', result_div_673)
        
        # Getting the type of 'splines' (line 153)
        splines_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'splines')
        # Assigning a type to the variable 'splines_674' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'splines_674', splines_674)
        # Testing if the for loop is going to be iterated (line 153)
        # Testing the type of a for loop iterable (line 153)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 153, 8), splines_674)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 153, 8), splines_674):
            # Getting the type of the for loop variable (line 153)
            for_loop_var_675 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 153, 8), splines_674)
            # Assigning a type to the variable 'spl' (line 153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'spl', for_loop_var_675)
            # SSA begins for a for statement (line 153)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Name (line 154):
            
            # Assigning a Num to a Name (line 154):
            int_676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 21), 'int')
            # Assigning a type to the variable 'length' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'length', int_676)
            
            # Assigning a Call to a Name (line 155):
            
            # Assigning a Call to a Name (line 155):
            
            # Call to call(...): (line 155)
            # Processing the call arguments (line 155)
            int_679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 28), 'int')
            # Processing the call keyword arguments (line 155)
            kwargs_680 = {}
            # Getting the type of 'spl' (line 155)
            spl_677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'spl', False)
            # Obtaining the member 'call' of a type (line 155)
            call_678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 19), spl_677, 'call')
            # Calling call(args, kwargs) (line 155)
            call_call_result_681 = invoke(stypy.reporting.localization.Localization(__file__, 155, 19), call_678, *[int_679], **kwargs_680)
            
            # Assigning a type to the variable 'curr' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'curr', call_call_result_681)
            
            
            # Call to range(...): (line 156)
            # Processing the call arguments (line 156)
            int_683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 27), 'int')
            int_684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 30), 'int')
            # Processing the call keyword arguments (line 156)
            kwargs_685 = {}
            # Getting the type of 'range' (line 156)
            range_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 21), 'range', False)
            # Calling range(args, kwargs) (line 156)
            range_call_result_686 = invoke(stypy.reporting.localization.Localization(__file__, 156, 21), range_682, *[int_683, int_684], **kwargs_685)
            
            # Assigning a type to the variable 'range_call_result_686' (line 156)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'range_call_result_686', range_call_result_686)
            # Testing if the for loop is going to be iterated (line 156)
            # Testing the type of a for loop iterable (line 156)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 156, 12), range_call_result_686)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 156, 12), range_call_result_686):
                # Getting the type of the for loop variable (line 156)
                for_loop_var_687 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 156, 12), range_call_result_686)
                # Assigning a type to the variable 'i' (line 156)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'i', for_loop_var_687)
                # SSA begins for a for statement (line 156)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Name to a Name (line 157):
                
                # Assigning a Name to a Name (line 157):
                # Getting the type of 'curr' (line 157)
                curr_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 23), 'curr')
                # Assigning a type to the variable 'last' (line 157)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 16), 'last', curr_688)
                
                # Assigning a BinOp to a Name (line 158):
                
                # Assigning a BinOp to a Name (line 158):
                float_689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 20), 'float')
                int_690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 26), 'int')
                # Applying the binary operator 'div' (line 158)
                result_div_691 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 20), 'div', float_689, int_690)
                
                # Getting the type of 'i' (line 158)
                i_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 32), 'i')
                # Applying the binary operator '*' (line 158)
                result_mul_693 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 30), '*', result_div_691, i_692)
                
                # Assigning a type to the variable 't' (line 158)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 't', result_mul_693)
                
                # Assigning a Call to a Name (line 159):
                
                # Assigning a Call to a Name (line 159):
                
                # Call to call(...): (line 159)
                # Processing the call arguments (line 159)
                # Getting the type of 't' (line 159)
                t_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 't', False)
                # Processing the call keyword arguments (line 159)
                kwargs_697 = {}
                # Getting the type of 'spl' (line 159)
                spl_694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 23), 'spl', False)
                # Obtaining the member 'call' of a type (line 159)
                call_695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 23), spl_694, 'call')
                # Calling call(args, kwargs) (line 159)
                call_call_result_698 = invoke(stypy.reporting.localization.Localization(__file__, 159, 23), call_695, *[t_696], **kwargs_697)
                
                # Assigning a type to the variable 'curr' (line 159)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 16), 'curr', call_call_result_698)
                
                # Getting the type of 'length' (line 160)
                length_699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'length')
                
                # Call to dist(...): (line 160)
                # Processing the call arguments (line 160)
                # Getting the type of 'last' (line 160)
                last_702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 36), 'last', False)
                # Processing the call keyword arguments (line 160)
                kwargs_703 = {}
                # Getting the type of 'curr' (line 160)
                curr_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 26), 'curr', False)
                # Obtaining the member 'dist' of a type (line 160)
                dist_701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 26), curr_700, 'dist')
                # Calling dist(args, kwargs) (line 160)
                dist_call_result_704 = invoke(stypy.reporting.localization.Localization(__file__, 160, 26), dist_701, *[last_702], **kwargs_703)
                
                # Applying the binary operator '+=' (line 160)
                result_iadd_705 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 16), '+=', length_699, dist_call_result_704)
                # Assigning a type to the variable 'length' (line 160)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 16), 'length', result_iadd_705)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to append(...): (line 161)
            # Processing the call arguments (line 161)
            
            # Call to max(...): (line 161)
            # Processing the call arguments (line 161)
            int_710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 39), 'int')
            
            # Call to int(...): (line 161)
            # Processing the call arguments (line 161)
            # Getting the type of 'length' (line 161)
            length_712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 46), 'length', False)
            # Getting the type of 'maxlength' (line 161)
            maxlength_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 55), 'maxlength', False)
            # Applying the binary operator 'div' (line 161)
            result_div_714 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 46), 'div', length_712, maxlength_713)
            
            float_715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 67), 'float')
            # Applying the binary operator '*' (line 161)
            result_mul_716 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 65), '*', result_div_714, float_715)
            
            # Processing the call keyword arguments (line 161)
            kwargs_717 = {}
            # Getting the type of 'int' (line 161)
            int_711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 42), 'int', False)
            # Calling int(args, kwargs) (line 161)
            int_call_result_718 = invoke(stypy.reporting.localization.Localization(__file__, 161, 42), int_711, *[result_mul_716], **kwargs_717)
            
            # Processing the call keyword arguments (line 161)
            kwargs_719 = {}
            # Getting the type of 'max' (line 161)
            max_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 35), 'max', False)
            # Calling max(args, kwargs) (line 161)
            max_call_result_720 = invoke(stypy.reporting.localization.Localization(__file__, 161, 35), max_709, *[int_710, int_call_result_718], **kwargs_719)
            
            # Processing the call keyword arguments (line 161)
            kwargs_721 = {}
            # Getting the type of 'self' (line 161)
            self_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 12), 'self', False)
            # Obtaining the member 'num_trafos' of a type (line 161)
            num_trafos_707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), self_706, 'num_trafos')
            # Obtaining the member 'append' of a type (line 161)
            append_708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 12), num_trafos_707, 'append')
            # Calling append(args, kwargs) (line 161)
            append_call_result_722 = invoke(stypy.reporting.localization.Localization(__file__, 161, 12), append_708, *[max_call_result_720], **kwargs_721)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Attribute (line 162):
        
        # Assigning a Call to a Attribute (line 162):
        
        # Call to reduce(...): (line 162)
        # Processing the call arguments (line 162)

        @norecursion
        def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_1'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 162, 32, True)
            # Passed parameters checking function
            _stypy_temp_lambda_1.stypy_localization = localization
            _stypy_temp_lambda_1.stypy_type_of_self = None
            _stypy_temp_lambda_1.stypy_type_store = module_type_store
            _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
            _stypy_temp_lambda_1.stypy_param_names_list = ['a', 'b']
            _stypy_temp_lambda_1.stypy_varargs_param_name = None
            _stypy_temp_lambda_1.stypy_kwargs_param_name = None
            _stypy_temp_lambda_1.stypy_call_defaults = defaults
            _stypy_temp_lambda_1.stypy_call_varargs = varargs
            _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['a', 'b'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_1', ['a', 'b'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'a' (line 162)
            a_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 44), 'a', False)
            # Getting the type of 'b' (line 162)
            b_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 46), 'b', False)
            # Applying the binary operator '+' (line 162)
            result_add_726 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 44), '+', a_724, b_725)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'stypy_return_type', result_add_726)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_1' in the type store
            # Getting the type of 'stypy_return_type' (line 162)
            stypy_return_type_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_727)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_1'
            return stypy_return_type_727

        # Assigning a type to the variable '_stypy_temp_lambda_1' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
        # Getting the type of '_stypy_temp_lambda_1' (line 162)
        _stypy_temp_lambda_1_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), '_stypy_temp_lambda_1')
        # Getting the type of 'self' (line 162)
        self_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 49), 'self', False)
        # Obtaining the member 'num_trafos' of a type (line 162)
        num_trafos_730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 49), self_729, 'num_trafos')
        int_731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 66), 'int')
        # Processing the call keyword arguments (line 162)
        kwargs_732 = {}
        # Getting the type of 'reduce' (line 162)
        reduce_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 25), 'reduce', False)
        # Calling reduce(args, kwargs) (line 162)
        reduce_call_result_733 = invoke(stypy.reporting.localization.Localization(__file__, 162, 25), reduce_723, *[_stypy_temp_lambda_1_728, num_trafos_730, int_731], **kwargs_732)
        
        # Getting the type of 'self' (line 162)
        self_734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self')
        # Setting the type of the member 'num_total' of a type (line 162)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_734, 'num_total', reduce_call_result_733)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_random_trafo(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_random_trafo'
        module_type_store = module_type_store.open_function_context('get_random_trafo', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_localization', localization)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_type_store', module_type_store)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_function_name', 'Chaosgame.get_random_trafo')
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_param_names_list', [])
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_varargs_param_name', None)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_call_defaults', defaults)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_call_varargs', varargs)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Chaosgame.get_random_trafo.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chaosgame.get_random_trafo', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_random_trafo', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_random_trafo(...)' code ##################

        
        # Assigning a Call to a Name (line 166):
        
        # Assigning a Call to a Name (line 166):
        
        # Call to randrange(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Call to int(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'self' (line 166)
        self_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 33), 'self', False)
        # Obtaining the member 'num_total' of a type (line 166)
        num_total_739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 33), self_738, 'num_total')
        # Processing the call keyword arguments (line 166)
        kwargs_740 = {}
        # Getting the type of 'int' (line 166)
        int_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 29), 'int', False)
        # Calling int(args, kwargs) (line 166)
        int_call_result_741 = invoke(stypy.reporting.localization.Localization(__file__, 166, 29), int_737, *[num_total_739], **kwargs_740)
        
        int_742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 51), 'int')
        # Applying the binary operator '+' (line 166)
        result_add_743 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 29), '+', int_call_result_741, int_742)
        
        # Processing the call keyword arguments (line 166)
        kwargs_744 = {}
        # Getting the type of 'random' (line 166)
        random_735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 12), 'random', False)
        # Obtaining the member 'randrange' of a type (line 166)
        randrange_736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 12), random_735, 'randrange')
        # Calling randrange(args, kwargs) (line 166)
        randrange_call_result_745 = invoke(stypy.reporting.localization.Localization(__file__, 166, 12), randrange_736, *[result_add_743], **kwargs_744)
        
        # Assigning a type to the variable 'r' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'r', randrange_call_result_745)
        
        # Assigning a Num to a Name (line 167):
        
        # Assigning a Num to a Name (line 167):
        int_746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 12), 'int')
        # Assigning a type to the variable 'l' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'l', int_746)
        
        
        # Call to range(...): (line 168)
        # Processing the call arguments (line 168)
        
        # Call to len(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'self' (line 168)
        self_749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'self', False)
        # Obtaining the member 'num_trafos' of a type (line 168)
        num_trafos_750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 27), self_749, 'num_trafos')
        # Processing the call keyword arguments (line 168)
        kwargs_751 = {}
        # Getting the type of 'len' (line 168)
        len_748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 23), 'len', False)
        # Calling len(args, kwargs) (line 168)
        len_call_result_752 = invoke(stypy.reporting.localization.Localization(__file__, 168, 23), len_748, *[num_trafos_750], **kwargs_751)
        
        # Processing the call keyword arguments (line 168)
        kwargs_753 = {}
        # Getting the type of 'range' (line 168)
        range_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 17), 'range', False)
        # Calling range(args, kwargs) (line 168)
        range_call_result_754 = invoke(stypy.reporting.localization.Localization(__file__, 168, 17), range_747, *[len_call_result_752], **kwargs_753)
        
        # Assigning a type to the variable 'range_call_result_754' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'range_call_result_754', range_call_result_754)
        # Testing if the for loop is going to be iterated (line 168)
        # Testing the type of a for loop iterable (line 168)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 168, 8), range_call_result_754)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 168, 8), range_call_result_754):
            # Getting the type of the for loop variable (line 168)
            for_loop_var_755 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 168, 8), range_call_result_754)
            # Assigning a type to the variable 'i' (line 168)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'i', for_loop_var_755)
            # SSA begins for a for statement (line 168)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Evaluating a boolean operation
            
            # Getting the type of 'r' (line 169)
            r_756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'r')
            # Getting the type of 'l' (line 169)
            l_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'l')
            # Applying the binary operator '>=' (line 169)
            result_ge_758 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 15), '>=', r_756, l_757)
            
            
            # Getting the type of 'r' (line 169)
            r_759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'r')
            # Getting the type of 'l' (line 169)
            l_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 30), 'l')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 169)
            i_761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 50), 'i')
            # Getting the type of 'self' (line 169)
            self_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'self')
            # Obtaining the member 'num_trafos' of a type (line 169)
            num_trafos_763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 34), self_762, 'num_trafos')
            # Obtaining the member '__getitem__' of a type (line 169)
            getitem___764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 34), num_trafos_763, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 169)
            subscript_call_result_765 = invoke(stypy.reporting.localization.Localization(__file__, 169, 34), getitem___764, i_761)
            
            # Applying the binary operator '+' (line 169)
            result_add_766 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 30), '+', l_760, subscript_call_result_765)
            
            # Applying the binary operator '<' (line 169)
            result_lt_767 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 26), '<', r_759, result_add_766)
            
            # Applying the binary operator 'and' (line 169)
            result_and_keyword_768 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 15), 'and', result_ge_758, result_lt_767)
            
            # Testing if the type of an if condition is none (line 169)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 169, 12), result_and_keyword_768):
                pass
            else:
                
                # Testing the type of an if condition (line 169)
                if_condition_769 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 169, 12), result_and_keyword_768)
                # Assigning a type to the variable 'if_condition_769' (line 169)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'if_condition_769', if_condition_769)
                # SSA begins for if statement (line 169)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Obtaining an instance of the builtin type 'tuple' (line 170)
                tuple_770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 170)
                # Adding element type (line 170)
                # Getting the type of 'i' (line 170)
                i_771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 23), 'i')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 23), tuple_770, i_771)
                # Adding element type (line 170)
                
                # Call to randrange(...): (line 170)
                # Processing the call arguments (line 170)
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 170)
                i_774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 59), 'i', False)
                # Getting the type of 'self' (line 170)
                self_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 43), 'self', False)
                # Obtaining the member 'num_trafos' of a type (line 170)
                num_trafos_776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 43), self_775, 'num_trafos')
                # Obtaining the member '__getitem__' of a type (line 170)
                getitem___777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 43), num_trafos_776, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 170)
                subscript_call_result_778 = invoke(stypy.reporting.localization.Localization(__file__, 170, 43), getitem___777, i_774)
                
                # Processing the call keyword arguments (line 170)
                kwargs_779 = {}
                # Getting the type of 'random' (line 170)
                random_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 26), 'random', False)
                # Obtaining the member 'randrange' of a type (line 170)
                randrange_773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 26), random_772, 'randrange')
                # Calling randrange(args, kwargs) (line 170)
                randrange_call_result_780 = invoke(stypy.reporting.localization.Localization(__file__, 170, 26), randrange_773, *[subscript_call_result_778], **kwargs_779)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 23), tuple_770, randrange_call_result_780)
                
                # Assigning a type to the variable 'stypy_return_type' (line 170)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'stypy_return_type', tuple_770)
                # SSA join for if statement (line 169)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'l' (line 171)
            l_781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'l')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 171)
            i_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 33), 'i')
            # Getting the type of 'self' (line 171)
            self_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 17), 'self')
            # Obtaining the member 'num_trafos' of a type (line 171)
            num_trafos_784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 17), self_783, 'num_trafos')
            # Obtaining the member '__getitem__' of a type (line 171)
            getitem___785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 17), num_trafos_784, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 171)
            subscript_call_result_786 = invoke(stypy.reporting.localization.Localization(__file__, 171, 17), getitem___785, i_782)
            
            # Applying the binary operator '+=' (line 171)
            result_iadd_787 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 12), '+=', l_781, subscript_call_result_786)
            # Assigning a type to the variable 'l' (line 171)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'l', result_iadd_787)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 172)
        tuple_788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 172)
        # Adding element type (line 172)
        
        # Call to len(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'self' (line 172)
        self_790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 19), 'self', False)
        # Obtaining the member 'num_trafos' of a type (line 172)
        num_trafos_791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 19), self_790, 'num_trafos')
        # Processing the call keyword arguments (line 172)
        kwargs_792 = {}
        # Getting the type of 'len' (line 172)
        len_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'len', False)
        # Calling len(args, kwargs) (line 172)
        len_call_result_793 = invoke(stypy.reporting.localization.Localization(__file__, 172, 15), len_789, *[num_trafos_791], **kwargs_792)
        
        int_794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 38), 'int')
        # Applying the binary operator '-' (line 172)
        result_sub_795 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 15), '-', len_call_result_793, int_794)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 15), tuple_788, result_sub_795)
        # Adding element type (line 172)
        
        # Call to randrange(...): (line 172)
        # Processing the call arguments (line 172)
        
        # Obtaining the type of the subscript
        int_798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 74), 'int')
        # Getting the type of 'self' (line 172)
        self_799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 58), 'self', False)
        # Obtaining the member 'num_trafos' of a type (line 172)
        num_trafos_800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 58), self_799, 'num_trafos')
        # Obtaining the member '__getitem__' of a type (line 172)
        getitem___801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 58), num_trafos_800, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 172)
        subscript_call_result_802 = invoke(stypy.reporting.localization.Localization(__file__, 172, 58), getitem___801, int_798)
        
        # Processing the call keyword arguments (line 172)
        kwargs_803 = {}
        # Getting the type of 'random' (line 172)
        random_796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 41), 'random', False)
        # Obtaining the member 'randrange' of a type (line 172)
        randrange_797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 41), random_796, 'randrange')
        # Calling randrange(args, kwargs) (line 172)
        randrange_call_result_804 = invoke(stypy.reporting.localization.Localization(__file__, 172, 41), randrange_797, *[subscript_call_result_802], **kwargs_803)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 15), tuple_788, randrange_call_result_804)
        
        # Assigning a type to the variable 'stypy_return_type' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'stypy_return_type', tuple_788)
        
        # ################# End of 'get_random_trafo(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_random_trafo' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_805)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_random_trafo'
        return stypy_return_type_805


    @norecursion
    def transform_point(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 174)
        None_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 43), 'None')
        defaults = [None_806]
        # Create a new context for function 'transform_point'
        module_type_store = module_type_store.open_function_context('transform_point', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Chaosgame.transform_point.__dict__.__setitem__('stypy_localization', localization)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_type_store', module_type_store)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_function_name', 'Chaosgame.transform_point')
        Chaosgame.transform_point.__dict__.__setitem__('stypy_param_names_list', ['point', 'trafo'])
        Chaosgame.transform_point.__dict__.__setitem__('stypy_varargs_param_name', None)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_call_defaults', defaults)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_call_varargs', varargs)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Chaosgame.transform_point.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chaosgame.transform_point', ['point', 'trafo'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'transform_point', localization, ['point', 'trafo'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'transform_point(...)' code ##################

        
        # Assigning a BinOp to a Name (line 175):
        
        # Assigning a BinOp to a Name (line 175):
        # Getting the type of 'point' (line 175)
        point_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 13), 'point')
        # Obtaining the member 'x' of a type (line 175)
        x_808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 13), point_807, 'x')
        # Getting the type of 'self' (line 175)
        self_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'self')
        # Obtaining the member 'minx' of a type (line 175)
        minx_810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 23), self_809, 'minx')
        # Applying the binary operator '-' (line 175)
        result_sub_811 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 13), '-', x_808, minx_810)
        
        
        # Call to float(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'self' (line 175)
        self_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 42), 'self', False)
        # Obtaining the member 'width' of a type (line 175)
        width_814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 42), self_813, 'width')
        # Processing the call keyword arguments (line 175)
        kwargs_815 = {}
        # Getting the type of 'float' (line 175)
        float_812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 36), 'float', False)
        # Calling float(args, kwargs) (line 175)
        float_call_result_816 = invoke(stypy.reporting.localization.Localization(__file__, 175, 36), float_812, *[width_814], **kwargs_815)
        
        # Applying the binary operator 'div' (line 175)
        result_div_817 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 12), 'div', result_sub_811, float_call_result_816)
        
        # Assigning a type to the variable 'x' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'x', result_div_817)
        
        # Assigning a BinOp to a Name (line 176):
        
        # Assigning a BinOp to a Name (line 176):
        # Getting the type of 'point' (line 176)
        point_818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 13), 'point')
        # Obtaining the member 'y' of a type (line 176)
        y_819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 13), point_818, 'y')
        # Getting the type of 'self' (line 176)
        self_820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 23), 'self')
        # Obtaining the member 'miny' of a type (line 176)
        miny_821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 23), self_820, 'miny')
        # Applying the binary operator '-' (line 176)
        result_sub_822 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 13), '-', y_819, miny_821)
        
        
        # Call to float(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'self' (line 176)
        self_824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 42), 'self', False)
        # Obtaining the member 'height' of a type (line 176)
        height_825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 42), self_824, 'height')
        # Processing the call keyword arguments (line 176)
        kwargs_826 = {}
        # Getting the type of 'float' (line 176)
        float_823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 36), 'float', False)
        # Calling float(args, kwargs) (line 176)
        float_call_result_827 = invoke(stypy.reporting.localization.Localization(__file__, 176, 36), float_823, *[height_825], **kwargs_826)
        
        # Applying the binary operator 'div' (line 176)
        result_div_828 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 12), 'div', result_sub_822, float_call_result_827)
        
        # Assigning a type to the variable 'y' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'y', result_div_828)
        
        # Type idiom detected: calculating its left and rigth part (line 177)
        # Getting the type of 'trafo' (line 177)
        trafo_829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 11), 'trafo')
        # Getting the type of 'None' (line 177)
        None_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 20), 'None')
        
        (may_be_831, more_types_in_union_832) = may_be_none(trafo_829, None_830)

        if may_be_831:

            if more_types_in_union_832:
                # Runtime conditional SSA (line 177)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 178):
            
            # Assigning a Call to a Name (line 178):
            
            # Call to get_random_trafo(...): (line 178)
            # Processing the call keyword arguments (line 178)
            kwargs_835 = {}
            # Getting the type of 'self' (line 178)
            self_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'self', False)
            # Obtaining the member 'get_random_trafo' of a type (line 178)
            get_random_trafo_834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 20), self_833, 'get_random_trafo')
            # Calling get_random_trafo(args, kwargs) (line 178)
            get_random_trafo_call_result_836 = invoke(stypy.reporting.localization.Localization(__file__, 178, 20), get_random_trafo_834, *[], **kwargs_835)
            
            # Assigning a type to the variable 'trafo' (line 178)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'trafo', get_random_trafo_call_result_836)

            if more_types_in_union_832:
                # SSA join for if statement (line 177)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Tuple (line 179):
        
        # Assigning a Call to a Name:
        
        # Call to GetDomain(...): (line 179)
        # Processing the call keyword arguments (line 179)
        kwargs_846 = {}
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 40), 'int')
        # Getting the type of 'trafo' (line 179)
        trafo_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 34), 'trafo', False)
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 34), trafo_838, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_840 = invoke(stypy.reporting.localization.Localization(__file__, 179, 34), getitem___839, int_837)
        
        # Getting the type of 'self' (line 179)
        self_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'self', False)
        # Obtaining the member 'splines' of a type (line 179)
        splines_842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 21), self_841, 'splines')
        # Obtaining the member '__getitem__' of a type (line 179)
        getitem___843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 21), splines_842, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 179)
        subscript_call_result_844 = invoke(stypy.reporting.localization.Localization(__file__, 179, 21), getitem___843, subscript_call_result_840)
        
        # Obtaining the member 'GetDomain' of a type (line 179)
        GetDomain_845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 21), subscript_call_result_844, 'GetDomain')
        # Calling GetDomain(args, kwargs) (line 179)
        GetDomain_call_result_847 = invoke(stypy.reporting.localization.Localization(__file__, 179, 21), GetDomain_845, *[], **kwargs_846)
        
        # Assigning a type to the variable 'call_assignment_1' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'call_assignment_1', GetDomain_call_result_847)
        
        # Assigning a Call to a Name (line 179):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 179)
        call_assignment_1_848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_849 = stypy_get_value_from_tuple(call_assignment_1_848, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_2' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'call_assignment_2', stypy_get_value_from_tuple_call_result_849)
        
        # Assigning a Name to a Name (line 179):
        # Getting the type of 'call_assignment_2' (line 179)
        call_assignment_2_850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'call_assignment_2')
        # Assigning a type to the variable 'start' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'start', call_assignment_2_850)
        
        # Assigning a Call to a Name (line 179):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_1' (line 179)
        call_assignment_1_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'call_assignment_1', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_852 = stypy_get_value_from_tuple(call_assignment_1_851, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_3' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'call_assignment_3', stypy_get_value_from_tuple_call_result_852)
        
        # Assigning a Name to a Name (line 179):
        # Getting the type of 'call_assignment_3' (line 179)
        call_assignment_3_853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'call_assignment_3')
        # Assigning a type to the variable 'end' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 15), 'end', call_assignment_3_853)
        
        # Assigning a BinOp to a Name (line 180):
        
        # Assigning a BinOp to a Name (line 180):
        # Getting the type of 'end' (line 180)
        end_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 17), 'end')
        # Getting the type of 'start' (line 180)
        start_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 23), 'start')
        # Applying the binary operator '-' (line 180)
        result_sub_856 = python_operator(stypy.reporting.localization.Localization(__file__, 180, 17), '-', end_854, start_855)
        
        # Assigning a type to the variable 'length' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'length', result_sub_856)
        
        # Assigning a BinOp to a Name (line 181):
        
        # Assigning a BinOp to a Name (line 181):
        # Getting the type of 'length' (line 181)
        length_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 21), 'length')
        
        # Call to float(...): (line 181)
        # Processing the call arguments (line 181)
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 58), 'int')
        # Getting the type of 'trafo' (line 181)
        trafo_860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 52), 'trafo', False)
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 52), trafo_860, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_862 = invoke(stypy.reporting.localization.Localization(__file__, 181, 52), getitem___861, int_859)
        
        # Getting the type of 'self' (line 181)
        self_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 36), 'self', False)
        # Obtaining the member 'num_trafos' of a type (line 181)
        num_trafos_864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 36), self_863, 'num_trafos')
        # Obtaining the member '__getitem__' of a type (line 181)
        getitem___865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 36), num_trafos_864, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 181)
        subscript_call_result_866 = invoke(stypy.reporting.localization.Localization(__file__, 181, 36), getitem___865, subscript_call_result_862)
        
        # Processing the call keyword arguments (line 181)
        kwargs_867 = {}
        # Getting the type of 'float' (line 181)
        float_858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), 'float', False)
        # Calling float(args, kwargs) (line 181)
        float_call_result_868 = invoke(stypy.reporting.localization.Localization(__file__, 181, 30), float_858, *[subscript_call_result_866], **kwargs_867)
        
        # Applying the binary operator 'div' (line 181)
        result_div_869 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 21), 'div', length_857, float_call_result_868)
        
        # Assigning a type to the variable 'seg_length' (line 181)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'seg_length', result_div_869)
        
        # Assigning a BinOp to a Name (line 182):
        
        # Assigning a BinOp to a Name (line 182):
        # Getting the type of 'start' (line 182)
        start_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'start')
        # Getting the type of 'seg_length' (line 182)
        seg_length_871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 20), 'seg_length')
        
        # Obtaining the type of the subscript
        int_872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 39), 'int')
        # Getting the type of 'trafo' (line 182)
        trafo_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 33), 'trafo')
        # Obtaining the member '__getitem__' of a type (line 182)
        getitem___874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 33), trafo_873, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 182)
        subscript_call_result_875 = invoke(stypy.reporting.localization.Localization(__file__, 182, 33), getitem___874, int_872)
        
        # Applying the binary operator '*' (line 182)
        result_mul_876 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 20), '*', seg_length_871, subscript_call_result_875)
        
        # Applying the binary operator '+' (line 182)
        result_add_877 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 12), '+', start_870, result_mul_876)
        
        # Getting the type of 'seg_length' (line 182)
        seg_length_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 44), 'seg_length')
        # Getting the type of 'x' (line 182)
        x_879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 57), 'x')
        # Applying the binary operator '*' (line 182)
        result_mul_880 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 44), '*', seg_length_878, x_879)
        
        # Applying the binary operator '+' (line 182)
        result_add_881 = python_operator(stypy.reporting.localization.Localization(__file__, 182, 42), '+', result_add_877, result_mul_880)
        
        # Assigning a type to the variable 't' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 't', result_add_881)
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to call(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 't' (line 183)
        t_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 48), 't', False)
        # Processing the call keyword arguments (line 183)
        kwargs_892 = {}
        
        # Obtaining the type of the subscript
        
        # Obtaining the type of the subscript
        int_882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 39), 'int')
        # Getting the type of 'trafo' (line 183)
        trafo_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 33), 'trafo', False)
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 33), trafo_883, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_885 = invoke(stypy.reporting.localization.Localization(__file__, 183, 33), getitem___884, int_882)
        
        # Getting the type of 'self' (line 183)
        self_886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 20), 'self', False)
        # Obtaining the member 'splines' of a type (line 183)
        splines_887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 20), self_886, 'splines')
        # Obtaining the member '__getitem__' of a type (line 183)
        getitem___888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 20), splines_887, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 183)
        subscript_call_result_889 = invoke(stypy.reporting.localization.Localization(__file__, 183, 20), getitem___888, subscript_call_result_885)
        
        # Obtaining the member 'call' of a type (line 183)
        call_890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 20), subscript_call_result_889, 'call')
        # Calling call(args, kwargs) (line 183)
        call_call_result_893 = invoke(stypy.reporting.localization.Localization(__file__, 183, 20), call_890, *[t_891], **kwargs_892)
        
        # Assigning a type to the variable 'basepoint' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'basepoint', call_call_result_893)
        
        # Getting the type of 't' (line 184)
        t_894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 't')
        float_895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 15), 'float')
        int_896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 19), 'int')
        # Applying the binary operator 'div' (line 184)
        result_div_897 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), 'div', float_895, int_896)
        
        # Applying the binary operator '+' (line 184)
        result_add_898 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 11), '+', t_894, result_div_897)
        
        # Getting the type of 'end' (line 184)
        end_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 27), 'end')
        # Applying the binary operator '>' (line 184)
        result_gt_900 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 11), '>', result_add_898, end_899)
        
        # Testing if the type of an if condition is none (line 184)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 184, 8), result_gt_900):
            
            # Assigning a Call to a Name (line 188):
            
            # Assigning a Call to a Name (line 188):
            
            # Call to call(...): (line 188)
            # Processing the call arguments (line 188)
            # Getting the type of 't' (line 188)
            t_930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 52), 't', False)
            float_931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 56), 'float')
            int_932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 60), 'int')
            # Applying the binary operator 'div' (line 188)
            result_div_933 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 56), 'div', float_931, int_932)
            
            # Applying the binary operator '+' (line 188)
            result_add_934 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 52), '+', t_930, result_div_933)
            
            # Processing the call keyword arguments (line 188)
            kwargs_935 = {}
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            int_921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 43), 'int')
            # Getting the type of 'trafo' (line 188)
            trafo_922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 37), 'trafo', False)
            # Obtaining the member '__getitem__' of a type (line 188)
            getitem___923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 37), trafo_922, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 188)
            subscript_call_result_924 = invoke(stypy.reporting.localization.Localization(__file__, 188, 37), getitem___923, int_921)
            
            # Getting the type of 'self' (line 188)
            self_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'self', False)
            # Obtaining the member 'splines' of a type (line 188)
            splines_926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), self_925, 'splines')
            # Obtaining the member '__getitem__' of a type (line 188)
            getitem___927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), splines_926, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 188)
            subscript_call_result_928 = invoke(stypy.reporting.localization.Localization(__file__, 188, 24), getitem___927, subscript_call_result_924)
            
            # Obtaining the member 'call' of a type (line 188)
            call_929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), subscript_call_result_928, 'call')
            # Calling call(args, kwargs) (line 188)
            call_call_result_936 = invoke(stypy.reporting.localization.Localization(__file__, 188, 24), call_929, *[result_add_934], **kwargs_935)
            
            # Assigning a type to the variable 'neighbour' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'neighbour', call_call_result_936)
            
            # Assigning a BinOp to a Name (line 189):
            
            # Assigning a BinOp to a Name (line 189):
            # Getting the type of 'basepoint' (line 189)
            basepoint_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'basepoint')
            # Getting the type of 'neighbour' (line 189)
            neighbour_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 37), 'neighbour')
            # Applying the binary operator '-' (line 189)
            result_sub_939 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 25), '-', basepoint_937, neighbour_938)
            
            # Assigning a type to the variable 'derivative' (line 189)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'derivative', result_sub_939)
        else:
            
            # Testing the type of an if condition (line 184)
            if_condition_901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 8), result_gt_900)
            # Assigning a type to the variable 'if_condition_901' (line 184)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'if_condition_901', if_condition_901)
            # SSA begins for if statement (line 184)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 185):
            
            # Assigning a Call to a Name (line 185):
            
            # Call to call(...): (line 185)
            # Processing the call arguments (line 185)
            # Getting the type of 't' (line 185)
            t_911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 52), 't', False)
            float_912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 56), 'float')
            int_913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 60), 'int')
            # Applying the binary operator 'div' (line 185)
            result_div_914 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 56), 'div', float_912, int_913)
            
            # Applying the binary operator '-' (line 185)
            result_sub_915 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 52), '-', t_911, result_div_914)
            
            # Processing the call keyword arguments (line 185)
            kwargs_916 = {}
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            int_902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 43), 'int')
            # Getting the type of 'trafo' (line 185)
            trafo_903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 37), 'trafo', False)
            # Obtaining the member '__getitem__' of a type (line 185)
            getitem___904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 37), trafo_903, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 185)
            subscript_call_result_905 = invoke(stypy.reporting.localization.Localization(__file__, 185, 37), getitem___904, int_902)
            
            # Getting the type of 'self' (line 185)
            self_906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 24), 'self', False)
            # Obtaining the member 'splines' of a type (line 185)
            splines_907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 24), self_906, 'splines')
            # Obtaining the member '__getitem__' of a type (line 185)
            getitem___908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 24), splines_907, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 185)
            subscript_call_result_909 = invoke(stypy.reporting.localization.Localization(__file__, 185, 24), getitem___908, subscript_call_result_905)
            
            # Obtaining the member 'call' of a type (line 185)
            call_910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 24), subscript_call_result_909, 'call')
            # Calling call(args, kwargs) (line 185)
            call_call_result_917 = invoke(stypy.reporting.localization.Localization(__file__, 185, 24), call_910, *[result_sub_915], **kwargs_916)
            
            # Assigning a type to the variable 'neighbour' (line 185)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'neighbour', call_call_result_917)
            
            # Assigning a BinOp to a Name (line 186):
            
            # Assigning a BinOp to a Name (line 186):
            # Getting the type of 'neighbour' (line 186)
            neighbour_918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 25), 'neighbour')
            # Getting the type of 'basepoint' (line 186)
            basepoint_919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 37), 'basepoint')
            # Applying the binary operator '-' (line 186)
            result_sub_920 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 25), '-', neighbour_918, basepoint_919)
            
            # Assigning a type to the variable 'derivative' (line 186)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 12), 'derivative', result_sub_920)
            # SSA branch for the else part of an if statement (line 184)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 188):
            
            # Assigning a Call to a Name (line 188):
            
            # Call to call(...): (line 188)
            # Processing the call arguments (line 188)
            # Getting the type of 't' (line 188)
            t_930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 52), 't', False)
            float_931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 56), 'float')
            int_932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 60), 'int')
            # Applying the binary operator 'div' (line 188)
            result_div_933 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 56), 'div', float_931, int_932)
            
            # Applying the binary operator '+' (line 188)
            result_add_934 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 52), '+', t_930, result_div_933)
            
            # Processing the call keyword arguments (line 188)
            kwargs_935 = {}
            
            # Obtaining the type of the subscript
            
            # Obtaining the type of the subscript
            int_921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 43), 'int')
            # Getting the type of 'trafo' (line 188)
            trafo_922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 37), 'trafo', False)
            # Obtaining the member '__getitem__' of a type (line 188)
            getitem___923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 37), trafo_922, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 188)
            subscript_call_result_924 = invoke(stypy.reporting.localization.Localization(__file__, 188, 37), getitem___923, int_921)
            
            # Getting the type of 'self' (line 188)
            self_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 24), 'self', False)
            # Obtaining the member 'splines' of a type (line 188)
            splines_926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), self_925, 'splines')
            # Obtaining the member '__getitem__' of a type (line 188)
            getitem___927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), splines_926, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 188)
            subscript_call_result_928 = invoke(stypy.reporting.localization.Localization(__file__, 188, 24), getitem___927, subscript_call_result_924)
            
            # Obtaining the member 'call' of a type (line 188)
            call_929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 24), subscript_call_result_928, 'call')
            # Calling call(args, kwargs) (line 188)
            call_call_result_936 = invoke(stypy.reporting.localization.Localization(__file__, 188, 24), call_929, *[result_add_934], **kwargs_935)
            
            # Assigning a type to the variable 'neighbour' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'neighbour', call_call_result_936)
            
            # Assigning a BinOp to a Name (line 189):
            
            # Assigning a BinOp to a Name (line 189):
            # Getting the type of 'basepoint' (line 189)
            basepoint_937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'basepoint')
            # Getting the type of 'neighbour' (line 189)
            neighbour_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 37), 'neighbour')
            # Applying the binary operator '-' (line 189)
            result_sub_939 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 25), '-', basepoint_937, neighbour_938)
            
            # Assigning a type to the variable 'derivative' (line 189)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'derivative', result_sub_939)
            # SSA join for if statement (line 184)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to Mag(...): (line 190)
        # Processing the call keyword arguments (line 190)
        kwargs_942 = {}
        # Getting the type of 'derivative' (line 190)
        derivative_940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'derivative', False)
        # Obtaining the member 'Mag' of a type (line 190)
        Mag_941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 11), derivative_940, 'Mag')
        # Calling Mag(args, kwargs) (line 190)
        Mag_call_result_943 = invoke(stypy.reporting.localization.Localization(__file__, 190, 11), Mag_941, *[], **kwargs_942)
        
        int_944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 31), 'int')
        # Applying the binary operator '!=' (line 190)
        result_ne_945 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 11), '!=', Mag_call_result_943, int_944)
        
        # Testing if the type of an if condition is none (line 190)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 190, 8), result_ne_945):
            pass
        else:
            
            # Testing the type of an if condition (line 190)
            if_condition_946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 8), result_ne_945)
            # Assigning a type to the variable 'if_condition_946' (line 190)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'if_condition_946', if_condition_946)
            # SSA begins for if statement (line 190)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'basepoint' (line 191)
            basepoint_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'basepoint')
            # Obtaining the member 'x' of a type (line 191)
            x_948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), basepoint_947, 'x')
            # Getting the type of 'derivative' (line 191)
            derivative_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 27), 'derivative')
            # Obtaining the member 'y' of a type (line 191)
            y_950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 27), derivative_949, 'y')
            
            # Call to Mag(...): (line 191)
            # Processing the call keyword arguments (line 191)
            kwargs_953 = {}
            # Getting the type of 'derivative' (line 191)
            derivative_951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 42), 'derivative', False)
            # Obtaining the member 'Mag' of a type (line 191)
            Mag_952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 42), derivative_951, 'Mag')
            # Calling Mag(args, kwargs) (line 191)
            Mag_call_result_954 = invoke(stypy.reporting.localization.Localization(__file__, 191, 42), Mag_952, *[], **kwargs_953)
            
            # Applying the binary operator 'div' (line 191)
            result_div_955 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 27), 'div', y_950, Mag_call_result_954)
            
            # Getting the type of 'y' (line 191)
            y_956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 62), 'y')
            float_957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 66), 'float')
            # Applying the binary operator '-' (line 191)
            result_sub_958 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 62), '-', y_956, float_957)
            
            # Applying the binary operator '*' (line 191)
            result_mul_959 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 59), '*', result_div_955, result_sub_958)
            
            # Getting the type of 'self' (line 192)
            self_960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 27), 'self')
            # Obtaining the member 'thickness' of a type (line 192)
            thickness_961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 27), self_960, 'thickness')
            # Applying the binary operator '*' (line 191)
            result_mul_962 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 71), '*', result_mul_959, thickness_961)
            
            # Applying the binary operator '+=' (line 191)
            result_iadd_963 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 12), '+=', x_948, result_mul_962)
            # Getting the type of 'basepoint' (line 191)
            basepoint_964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'basepoint')
            # Setting the type of the member 'x' of a type (line 191)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), basepoint_964, 'x', result_iadd_963)
            
            
            # Getting the type of 'basepoint' (line 193)
            basepoint_965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'basepoint')
            # Obtaining the member 'y' of a type (line 193)
            y_966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), basepoint_965, 'y')
            
            # Getting the type of 'derivative' (line 193)
            derivative_967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 28), 'derivative')
            # Obtaining the member 'x' of a type (line 193)
            x_968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 28), derivative_967, 'x')
            # Applying the 'usub' unary operator (line 193)
            result___neg___969 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 27), 'usub', x_968)
            
            
            # Call to Mag(...): (line 193)
            # Processing the call keyword arguments (line 193)
            kwargs_972 = {}
            # Getting the type of 'derivative' (line 193)
            derivative_970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 43), 'derivative', False)
            # Obtaining the member 'Mag' of a type (line 193)
            Mag_971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 43), derivative_970, 'Mag')
            # Calling Mag(args, kwargs) (line 193)
            Mag_call_result_973 = invoke(stypy.reporting.localization.Localization(__file__, 193, 43), Mag_971, *[], **kwargs_972)
            
            # Applying the binary operator 'div' (line 193)
            result_div_974 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 27), 'div', result___neg___969, Mag_call_result_973)
            
            # Getting the type of 'y' (line 193)
            y_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 63), 'y')
            float_976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 67), 'float')
            # Applying the binary operator '-' (line 193)
            result_sub_977 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 63), '-', y_975, float_976)
            
            # Applying the binary operator '*' (line 193)
            result_mul_978 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 60), '*', result_div_974, result_sub_977)
            
            # Getting the type of 'self' (line 194)
            self_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 27), 'self')
            # Obtaining the member 'thickness' of a type (line 194)
            thickness_980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 27), self_979, 'thickness')
            # Applying the binary operator '*' (line 193)
            result_mul_981 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 72), '*', result_mul_978, thickness_980)
            
            # Applying the binary operator '+=' (line 193)
            result_iadd_982 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 12), '+=', y_966, result_mul_981)
            # Getting the type of 'basepoint' (line 193)
            basepoint_983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'basepoint')
            # Setting the type of the member 'y' of a type (line 193)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 12), basepoint_983, 'y', result_iadd_982)
            
            # SSA branch for the else part of an if statement (line 190)
            module_type_store.open_ssa_branch('else')
            pass
            # SSA join for if statement (line 190)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to truncate(...): (line 197)
        # Processing the call arguments (line 197)
        # Getting the type of 'basepoint' (line 197)
        basepoint_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 22), 'basepoint', False)
        # Processing the call keyword arguments (line 197)
        kwargs_987 = {}
        # Getting the type of 'self' (line 197)
        self_984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'self', False)
        # Obtaining the member 'truncate' of a type (line 197)
        truncate_985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 8), self_984, 'truncate')
        # Calling truncate(args, kwargs) (line 197)
        truncate_call_result_988 = invoke(stypy.reporting.localization.Localization(__file__, 197, 8), truncate_985, *[basepoint_986], **kwargs_987)
        
        # Getting the type of 'basepoint' (line 198)
        basepoint_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'basepoint')
        # Assigning a type to the variable 'stypy_return_type' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'stypy_return_type', basepoint_989)
        
        # ################# End of 'transform_point(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'transform_point' in the type store
        # Getting the type of 'stypy_return_type' (line 174)
        stypy_return_type_990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_990)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'transform_point'
        return stypy_return_type_990


    @norecursion
    def truncate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'truncate'
        module_type_store = module_type_store.open_function_context('truncate', 200, 4, False)
        # Assigning a type to the variable 'self' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Chaosgame.truncate.__dict__.__setitem__('stypy_localization', localization)
        Chaosgame.truncate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Chaosgame.truncate.__dict__.__setitem__('stypy_type_store', module_type_store)
        Chaosgame.truncate.__dict__.__setitem__('stypy_function_name', 'Chaosgame.truncate')
        Chaosgame.truncate.__dict__.__setitem__('stypy_param_names_list', ['point'])
        Chaosgame.truncate.__dict__.__setitem__('stypy_varargs_param_name', None)
        Chaosgame.truncate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Chaosgame.truncate.__dict__.__setitem__('stypy_call_defaults', defaults)
        Chaosgame.truncate.__dict__.__setitem__('stypy_call_varargs', varargs)
        Chaosgame.truncate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Chaosgame.truncate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chaosgame.truncate', ['point'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'truncate', localization, ['point'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'truncate(...)' code ##################

        
        # Getting the type of 'point' (line 201)
        point_991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 11), 'point')
        # Obtaining the member 'x' of a type (line 201)
        x_992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 11), point_991, 'x')
        # Getting the type of 'self' (line 201)
        self_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 22), 'self')
        # Obtaining the member 'maxx' of a type (line 201)
        maxx_994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 22), self_993, 'maxx')
        # Applying the binary operator '>=' (line 201)
        result_ge_995 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 11), '>=', x_992, maxx_994)
        
        # Testing if the type of an if condition is none (line 201)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 201, 8), result_ge_995):
            pass
        else:
            
            # Testing the type of an if condition (line 201)
            if_condition_996 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 8), result_ge_995)
            # Assigning a type to the variable 'if_condition_996' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'if_condition_996', if_condition_996)
            # SSA begins for if statement (line 201)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 202):
            
            # Assigning a Attribute to a Attribute (line 202):
            # Getting the type of 'self' (line 202)
            self_997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 22), 'self')
            # Obtaining the member 'maxx' of a type (line 202)
            maxx_998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 22), self_997, 'maxx')
            # Getting the type of 'point' (line 202)
            point_999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'point')
            # Setting the type of the member 'x' of a type (line 202)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 12), point_999, 'x', maxx_998)
            # SSA join for if statement (line 201)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'point' (line 203)
        point_1000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'point')
        # Obtaining the member 'y' of a type (line 203)
        y_1001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 11), point_1000, 'y')
        # Getting the type of 'self' (line 203)
        self_1002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 22), 'self')
        # Obtaining the member 'maxy' of a type (line 203)
        maxy_1003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 22), self_1002, 'maxy')
        # Applying the binary operator '>=' (line 203)
        result_ge_1004 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 11), '>=', y_1001, maxy_1003)
        
        # Testing if the type of an if condition is none (line 203)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 203, 8), result_ge_1004):
            pass
        else:
            
            # Testing the type of an if condition (line 203)
            if_condition_1005 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 8), result_ge_1004)
            # Assigning a type to the variable 'if_condition_1005' (line 203)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'if_condition_1005', if_condition_1005)
            # SSA begins for if statement (line 203)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 204):
            
            # Assigning a Attribute to a Attribute (line 204):
            # Getting the type of 'self' (line 204)
            self_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 22), 'self')
            # Obtaining the member 'maxy' of a type (line 204)
            maxy_1007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 22), self_1006, 'maxy')
            # Getting the type of 'point' (line 204)
            point_1008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'point')
            # Setting the type of the member 'y' of a type (line 204)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), point_1008, 'y', maxy_1007)
            # SSA join for if statement (line 203)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'point' (line 205)
        point_1009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 11), 'point')
        # Obtaining the member 'x' of a type (line 205)
        x_1010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 11), point_1009, 'x')
        # Getting the type of 'self' (line 205)
        self_1011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 21), 'self')
        # Obtaining the member 'minx' of a type (line 205)
        minx_1012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 21), self_1011, 'minx')
        # Applying the binary operator '<' (line 205)
        result_lt_1013 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 11), '<', x_1010, minx_1012)
        
        # Testing if the type of an if condition is none (line 205)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 205, 8), result_lt_1013):
            pass
        else:
            
            # Testing the type of an if condition (line 205)
            if_condition_1014 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 8), result_lt_1013)
            # Assigning a type to the variable 'if_condition_1014' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'if_condition_1014', if_condition_1014)
            # SSA begins for if statement (line 205)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 206):
            
            # Assigning a Attribute to a Attribute (line 206):
            # Getting the type of 'self' (line 206)
            self_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 22), 'self')
            # Obtaining the member 'minx' of a type (line 206)
            minx_1016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 22), self_1015, 'minx')
            # Getting the type of 'point' (line 206)
            point_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'point')
            # Setting the type of the member 'x' of a type (line 206)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 12), point_1017, 'x', minx_1016)
            # SSA join for if statement (line 205)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'point' (line 207)
        point_1018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'point')
        # Obtaining the member 'y' of a type (line 207)
        y_1019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 11), point_1018, 'y')
        # Getting the type of 'self' (line 207)
        self_1020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 21), 'self')
        # Obtaining the member 'miny' of a type (line 207)
        miny_1021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 21), self_1020, 'miny')
        # Applying the binary operator '<' (line 207)
        result_lt_1022 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 11), '<', y_1019, miny_1021)
        
        # Testing if the type of an if condition is none (line 207)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 207, 8), result_lt_1022):
            pass
        else:
            
            # Testing the type of an if condition (line 207)
            if_condition_1023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 8), result_lt_1022)
            # Assigning a type to the variable 'if_condition_1023' (line 207)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'if_condition_1023', if_condition_1023)
            # SSA begins for if statement (line 207)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Attribute (line 208):
            
            # Assigning a Attribute to a Attribute (line 208):
            # Getting the type of 'self' (line 208)
            self_1024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 22), 'self')
            # Obtaining the member 'miny' of a type (line 208)
            miny_1025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 22), self_1024, 'miny')
            # Getting the type of 'point' (line 208)
            point_1026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'point')
            # Setting the type of the member 'y' of a type (line 208)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 12), point_1026, 'y', miny_1025)
            # SSA join for if statement (line 207)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'truncate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'truncate' in the type store
        # Getting the type of 'stypy_return_type' (line 200)
        stypy_return_type_1027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1027)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'truncate'
        return stypy_return_type_1027


    @norecursion
    def create_image_chaos(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_image_chaos'
        module_type_store = module_type_store.open_function_context('create_image_chaos', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_localization', localization)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_type_store', module_type_store)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_function_name', 'Chaosgame.create_image_chaos')
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_param_names_list', ['w', 'h', 'name', 'n'])
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_varargs_param_name', None)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_call_defaults', defaults)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_call_varargs', varargs)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Chaosgame.create_image_chaos.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chaosgame.create_image_chaos', ['w', 'h', 'name', 'n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_image_chaos', localization, ['w', 'h', 'name', 'n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_image_chaos(...)' code ##################

        
        # Assigning a ListComp to a Name (line 211):
        
        # Assigning a ListComp to a Name (line 211):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to range(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'w' (line 211)
        w_1033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 37), 'w', False)
        # Processing the call keyword arguments (line 211)
        kwargs_1034 = {}
        # Getting the type of 'range' (line 211)
        range_1032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 31), 'range', False)
        # Calling range(args, kwargs) (line 211)
        range_call_result_1035 = invoke(stypy.reporting.localization.Localization(__file__, 211, 31), range_1032, *[w_1033], **kwargs_1034)
        
        comprehension_1036 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), range_call_result_1035)
        # Assigning a type to the variable 'i' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 14), 'i', comprehension_1036)
        
        # Obtaining an instance of the builtin type 'list' (line 211)
        list_1028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 211)
        # Adding element type (line 211)
        int_1029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 15), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), list_1028, int_1029)
        
        # Getting the type of 'h' (line 211)
        h_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 20), 'h')
        # Applying the binary operator '*' (line 211)
        result_mul_1031 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 14), '*', list_1028, h_1030)
        
        list_1037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 14), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 211, 14), list_1037, result_mul_1031)
        # Assigning a type to the variable 'im' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'im', list_1037)
        
        # Assigning a Call to a Name (line 212):
        
        # Assigning a Call to a Name (line 212):
        
        # Call to GVector(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'self' (line 212)
        self_1039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 25), 'self', False)
        # Obtaining the member 'maxx' of a type (line 212)
        maxx_1040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 25), self_1039, 'maxx')
        # Getting the type of 'self' (line 212)
        self_1041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 37), 'self', False)
        # Obtaining the member 'minx' of a type (line 212)
        minx_1042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 37), self_1041, 'minx')
        # Applying the binary operator '+' (line 212)
        result_add_1043 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 25), '+', maxx_1040, minx_1042)
        
        float_1044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 50), 'float')
        # Applying the binary operator 'div' (line 212)
        result_div_1045 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 24), 'div', result_add_1043, float_1044)
        
        # Getting the type of 'self' (line 213)
        self_1046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 25), 'self', False)
        # Obtaining the member 'maxy' of a type (line 213)
        maxy_1047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 25), self_1046, 'maxy')
        # Getting the type of 'self' (line 213)
        self_1048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 37), 'self', False)
        # Obtaining the member 'miny' of a type (line 213)
        miny_1049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 37), self_1048, 'miny')
        # Applying the binary operator '+' (line 213)
        result_add_1050 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 25), '+', maxy_1047, miny_1049)
        
        float_1051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 50), 'float')
        # Applying the binary operator 'div' (line 213)
        result_div_1052 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 24), 'div', result_add_1050, float_1051)
        
        int_1053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 55), 'int')
        # Processing the call keyword arguments (line 212)
        kwargs_1054 = {}
        # Getting the type of 'GVector' (line 212)
        GVector_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'GVector', False)
        # Calling GVector(args, kwargs) (line 212)
        GVector_call_result_1055 = invoke(stypy.reporting.localization.Localization(__file__, 212, 16), GVector_1038, *[result_div_1045, result_div_1052, int_1053], **kwargs_1054)
        
        # Assigning a type to the variable 'point' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'point', GVector_call_result_1055)
        
        # Assigning a Num to a Name (line 214):
        
        # Assigning a Num to a Name (line 214):
        int_1056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 18), 'int')
        # Assigning a type to the variable 'colored' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'colored', int_1056)
        
        # Assigning a List to a Name (line 215):
        
        # Assigning a List to a Name (line 215):
        
        # Obtaining an instance of the builtin type 'list' (line 215)
        list_1057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 215)
        
        # Assigning a type to the variable 'times' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'times', list_1057)
        
        
        # Call to range(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'n' (line 216)
        n_1059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 23), 'n', False)
        # Processing the call keyword arguments (line 216)
        kwargs_1060 = {}
        # Getting the type of 'range' (line 216)
        range_1058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 17), 'range', False)
        # Calling range(args, kwargs) (line 216)
        range_call_result_1061 = invoke(stypy.reporting.localization.Localization(__file__, 216, 17), range_1058, *[n_1059], **kwargs_1060)
        
        # Assigning a type to the variable 'range_call_result_1061' (line 216)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'range_call_result_1061', range_call_result_1061)
        # Testing if the for loop is going to be iterated (line 216)
        # Testing the type of a for loop iterable (line 216)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 216, 8), range_call_result_1061)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 216, 8), range_call_result_1061):
            # Getting the type of the for loop variable (line 216)
            for_loop_var_1062 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 216, 8), range_call_result_1061)
            # Assigning a type to the variable '_' (line 216)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), '_', for_loop_var_1062)
            # SSA begins for a for statement (line 216)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 217):
            
            # Assigning a Call to a Name (line 217):
            
            # Call to time(...): (line 217)
            # Processing the call keyword arguments (line 217)
            kwargs_1065 = {}
            # Getting the type of 'time' (line 217)
            time_1063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 17), 'time', False)
            # Obtaining the member 'time' of a type (line 217)
            time_1064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 17), time_1063, 'time')
            # Calling time(args, kwargs) (line 217)
            time_call_result_1066 = invoke(stypy.reporting.localization.Localization(__file__, 217, 17), time_1064, *[], **kwargs_1065)
            
            # Assigning a type to the variable 't1' (line 217)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 't1', time_call_result_1066)
            
            
            # Call to xrange(...): (line 218)
            # Processing the call arguments (line 218)
            int_1068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 28), 'int')
            # Processing the call keyword arguments (line 218)
            kwargs_1069 = {}
            # Getting the type of 'xrange' (line 218)
            xrange_1067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 218)
            xrange_call_result_1070 = invoke(stypy.reporting.localization.Localization(__file__, 218, 21), xrange_1067, *[int_1068], **kwargs_1069)
            
            # Assigning a type to the variable 'xrange_call_result_1070' (line 218)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'xrange_call_result_1070', xrange_call_result_1070)
            # Testing if the for loop is going to be iterated (line 218)
            # Testing the type of a for loop iterable (line 218)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 218, 12), xrange_call_result_1070)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 218, 12), xrange_call_result_1070):
                # Getting the type of the for loop variable (line 218)
                for_loop_var_1071 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 218, 12), xrange_call_result_1070)
                # Assigning a type to the variable 'i' (line 218)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'i', for_loop_var_1071)
                # SSA begins for a for statement (line 218)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 219):
                
                # Assigning a Call to a Name (line 219):
                
                # Call to transform_point(...): (line 219)
                # Processing the call arguments (line 219)
                # Getting the type of 'point' (line 219)
                point_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 45), 'point', False)
                # Processing the call keyword arguments (line 219)
                kwargs_1075 = {}
                # Getting the type of 'self' (line 219)
                self_1072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 24), 'self', False)
                # Obtaining the member 'transform_point' of a type (line 219)
                transform_point_1073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 24), self_1072, 'transform_point')
                # Calling transform_point(args, kwargs) (line 219)
                transform_point_call_result_1076 = invoke(stypy.reporting.localization.Localization(__file__, 219, 24), transform_point_1073, *[point_1074], **kwargs_1075)
                
                # Assigning a type to the variable 'point' (line 219)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'point', transform_point_call_result_1076)
                
                # Assigning a BinOp to a Name (line 220):
                
                # Assigning a BinOp to a Name (line 220):
                # Getting the type of 'point' (line 220)
                point_1077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'point')
                # Obtaining the member 'x' of a type (line 220)
                x_1078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 21), point_1077, 'x')
                # Getting the type of 'self' (line 220)
                self_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 31), 'self')
                # Obtaining the member 'minx' of a type (line 220)
                minx_1080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 31), self_1079, 'minx')
                # Applying the binary operator '-' (line 220)
                result_sub_1081 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 21), '-', x_1078, minx_1080)
                
                # Getting the type of 'self' (line 220)
                self_1082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 44), 'self')
                # Obtaining the member 'width' of a type (line 220)
                width_1083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 44), self_1082, 'width')
                # Applying the binary operator 'div' (line 220)
                result_div_1084 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 20), 'div', result_sub_1081, width_1083)
                
                # Getting the type of 'w' (line 220)
                w_1085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 57), 'w')
                # Applying the binary operator '*' (line 220)
                result_mul_1086 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 55), '*', result_div_1084, w_1085)
                
                # Assigning a type to the variable 'x' (line 220)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'x', result_mul_1086)
                
                # Assigning a BinOp to a Name (line 221):
                
                # Assigning a BinOp to a Name (line 221):
                # Getting the type of 'point' (line 221)
                point_1087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 21), 'point')
                # Obtaining the member 'y' of a type (line 221)
                y_1088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 21), point_1087, 'y')
                # Getting the type of 'self' (line 221)
                self_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 31), 'self')
                # Obtaining the member 'miny' of a type (line 221)
                miny_1090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 31), self_1089, 'miny')
                # Applying the binary operator '-' (line 221)
                result_sub_1091 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 21), '-', y_1088, miny_1090)
                
                # Getting the type of 'self' (line 221)
                self_1092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 44), 'self')
                # Obtaining the member 'height' of a type (line 221)
                height_1093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 44), self_1092, 'height')
                # Applying the binary operator 'div' (line 221)
                result_div_1094 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 20), 'div', result_sub_1091, height_1093)
                
                # Getting the type of 'h' (line 221)
                h_1095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 58), 'h')
                # Applying the binary operator '*' (line 221)
                result_mul_1096 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 56), '*', result_div_1094, h_1095)
                
                # Assigning a type to the variable 'y' (line 221)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'y', result_mul_1096)
                
                # Assigning a Call to a Name (line 222):
                
                # Assigning a Call to a Name (line 222):
                
                # Call to int(...): (line 222)
                # Processing the call arguments (line 222)
                # Getting the type of 'x' (line 222)
                x_1098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 24), 'x', False)
                # Processing the call keyword arguments (line 222)
                kwargs_1099 = {}
                # Getting the type of 'int' (line 222)
                int_1097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'int', False)
                # Calling int(args, kwargs) (line 222)
                int_call_result_1100 = invoke(stypy.reporting.localization.Localization(__file__, 222, 20), int_1097, *[x_1098], **kwargs_1099)
                
                # Assigning a type to the variable 'x' (line 222)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 16), 'x', int_call_result_1100)
                
                # Assigning a Call to a Name (line 223):
                
                # Assigning a Call to a Name (line 223):
                
                # Call to int(...): (line 223)
                # Processing the call arguments (line 223)
                # Getting the type of 'y' (line 223)
                y_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 24), 'y', False)
                # Processing the call keyword arguments (line 223)
                kwargs_1103 = {}
                # Getting the type of 'int' (line 223)
                int_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'int', False)
                # Calling int(args, kwargs) (line 223)
                int_call_result_1104 = invoke(stypy.reporting.localization.Localization(__file__, 223, 20), int_1101, *[y_1102], **kwargs_1103)
                
                # Assigning a type to the variable 'y' (line 223)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'y', int_call_result_1104)
                
                # Getting the type of 'x' (line 224)
                x_1105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'x')
                # Getting the type of 'w' (line 224)
                w_1106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 24), 'w')
                # Applying the binary operator '==' (line 224)
                result_eq_1107 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 19), '==', x_1105, w_1106)
                
                # Testing if the type of an if condition is none (line 224)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 224, 16), result_eq_1107):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 224)
                    if_condition_1108 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 16), result_eq_1107)
                    # Assigning a type to the variable 'if_condition_1108' (line 224)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 16), 'if_condition_1108', if_condition_1108)
                    # SSA begins for if statement (line 224)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'x' (line 225)
                    x_1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'x')
                    int_1110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 25), 'int')
                    # Applying the binary operator '-=' (line 225)
                    result_isub_1111 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 20), '-=', x_1109, int_1110)
                    # Assigning a type to the variable 'x' (line 225)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 20), 'x', result_isub_1111)
                    
                    # SSA join for if statement (line 224)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Getting the type of 'y' (line 226)
                y_1112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'y')
                # Getting the type of 'h' (line 226)
                h_1113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'h')
                # Applying the binary operator '==' (line 226)
                result_eq_1114 = python_operator(stypy.reporting.localization.Localization(__file__, 226, 19), '==', y_1112, h_1113)
                
                # Testing if the type of an if condition is none (line 226)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 226, 16), result_eq_1114):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 226)
                    if_condition_1115 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 226, 16), result_eq_1114)
                    # Assigning a type to the variable 'if_condition_1115' (line 226)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 16), 'if_condition_1115', if_condition_1115)
                    # SSA begins for if statement (line 226)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Getting the type of 'y' (line 227)
                    y_1116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'y')
                    int_1117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 25), 'int')
                    # Applying the binary operator '-=' (line 227)
                    result_isub_1118 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 20), '-=', y_1116, int_1117)
                    # Assigning a type to the variable 'y' (line 227)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 20), 'y', result_isub_1118)
                    
                    # SSA join for if statement (line 226)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Num to a Subscript (line 228):
                
                # Assigning a Num to a Subscript (line 228):
                int_1119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 35), 'int')
                
                # Obtaining the type of the subscript
                # Getting the type of 'x' (line 228)
                x_1120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 19), 'x')
                # Getting the type of 'im' (line 228)
                im_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'im')
                # Obtaining the member '__getitem__' of a type (line 228)
                getitem___1122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), im_1121, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 228)
                subscript_call_result_1123 = invoke(stypy.reporting.localization.Localization(__file__, 228, 16), getitem___1122, x_1120)
                
                # Getting the type of 'h' (line 228)
                h_1124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 22), 'h')
                # Getting the type of 'y' (line 228)
                y_1125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 26), 'y')
                # Applying the binary operator '-' (line 228)
                result_sub_1126 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 22), '-', h_1124, y_1125)
                
                int_1127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 30), 'int')
                # Applying the binary operator '-' (line 228)
                result_sub_1128 = python_operator(stypy.reporting.localization.Localization(__file__, 228, 28), '-', result_sub_1126, int_1127)
                
                # Storing an element on a container (line 228)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 228, 16), subscript_call_result_1123, (result_sub_1128, int_1119))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Assigning a Call to a Name (line 229):
            
            # Assigning a Call to a Name (line 229):
            
            # Call to time(...): (line 229)
            # Processing the call keyword arguments (line 229)
            kwargs_1131 = {}
            # Getting the type of 'time' (line 229)
            time_1129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 17), 'time', False)
            # Obtaining the member 'time' of a type (line 229)
            time_1130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 17), time_1129, 'time')
            # Calling time(args, kwargs) (line 229)
            time_call_result_1132 = invoke(stypy.reporting.localization.Localization(__file__, 229, 17), time_1130, *[], **kwargs_1131)
            
            # Assigning a type to the variable 't2' (line 229)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 't2', time_call_result_1132)
            
            # Call to append(...): (line 230)
            # Processing the call arguments (line 230)
            # Getting the type of 't2' (line 230)
            t2_1135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 25), 't2', False)
            # Getting the type of 't1' (line 230)
            t1_1136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 30), 't1', False)
            # Applying the binary operator '-' (line 230)
            result_sub_1137 = python_operator(stypy.reporting.localization.Localization(__file__, 230, 25), '-', t2_1135, t1_1136)
            
            # Processing the call keyword arguments (line 230)
            kwargs_1138 = {}
            # Getting the type of 'times' (line 230)
            times_1133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 12), 'times', False)
            # Obtaining the member 'append' of a type (line 230)
            append_1134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 12), times_1133, 'append')
            # Calling append(args, kwargs) (line 230)
            append_call_result_1139 = invoke(stypy.reporting.localization.Localization(__file__, 230, 12), append_1134, *[result_sub_1137], **kwargs_1138)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to save_im(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'im' (line 231)
        im_1141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'im', False)
        # Getting the type of 'name' (line 231)
        name_1142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 20), 'name', False)
        # Processing the call keyword arguments (line 231)
        kwargs_1143 = {}
        # Getting the type of 'save_im' (line 231)
        save_im_1140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'save_im', False)
        # Calling save_im(args, kwargs) (line 231)
        save_im_call_result_1144 = invoke(stypy.reporting.localization.Localization(__file__, 231, 8), save_im_1140, *[im_1141, name_1142], **kwargs_1143)
        
        # Getting the type of 'times' (line 232)
        times_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'times')
        # Assigning a type to the variable 'stypy_return_type' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'stypy_return_type', times_1145)
        
        # ################# End of 'create_image_chaos(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_image_chaos' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1146)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_image_chaos'
        return stypy_return_type_1146


# Assigning a type to the variable 'Chaosgame' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'Chaosgame', Chaosgame)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 235, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = ['n']
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main(...)' code ##################

    
    # Assigning a List to a Name (line 236):
    
    # Assigning a List to a Name (line 236):
    
    # Obtaining an instance of the builtin type 'list' (line 236)
    list_1147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 236, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 236)
    # Adding element type (line 236)
    
    # Call to Spline(...): (line 237)
    # Processing the call arguments (line 237)
    
    # Obtaining an instance of the builtin type 'list' (line 237)
    list_1149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 237)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 238)
    # Processing the call arguments (line 238)
    float_1151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 20), 'float')
    float_1152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 30), 'float')
    float_1153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 40), 'float')
    # Processing the call keyword arguments (line 238)
    kwargs_1154 = {}
    # Getting the type of 'GVector' (line 238)
    GVector_1150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 238)
    GVector_call_result_1155 = invoke(stypy.reporting.localization.Localization(__file__, 238, 12), GVector_1150, *[float_1151, float_1152, float_1153], **kwargs_1154)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1149, GVector_call_result_1155)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 239)
    # Processing the call arguments (line 239)
    float_1157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 20), 'float')
    float_1158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 30), 'float')
    float_1159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 40), 'float')
    # Processing the call keyword arguments (line 239)
    kwargs_1160 = {}
    # Getting the type of 'GVector' (line 239)
    GVector_1156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 239)
    GVector_call_result_1161 = invoke(stypy.reporting.localization.Localization(__file__, 239, 12), GVector_1156, *[float_1157, float_1158, float_1159], **kwargs_1160)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1149, GVector_call_result_1161)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 240)
    # Processing the call arguments (line 240)
    float_1163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 20), 'float')
    float_1164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 30), 'float')
    float_1165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 40), 'float')
    # Processing the call keyword arguments (line 240)
    kwargs_1166 = {}
    # Getting the type of 'GVector' (line 240)
    GVector_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 240)
    GVector_call_result_1167 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), GVector_1162, *[float_1163, float_1164, float_1165], **kwargs_1166)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1149, GVector_call_result_1167)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 241)
    # Processing the call arguments (line 241)
    float_1169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 20), 'float')
    float_1170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 30), 'float')
    float_1171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 40), 'float')
    # Processing the call keyword arguments (line 241)
    kwargs_1172 = {}
    # Getting the type of 'GVector' (line 241)
    GVector_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 241)
    GVector_call_result_1173 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), GVector_1168, *[float_1169, float_1170, float_1171], **kwargs_1172)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1149, GVector_call_result_1173)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 242)
    # Processing the call arguments (line 242)
    float_1175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 20), 'float')
    float_1176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 30), 'float')
    float_1177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 40), 'float')
    # Processing the call keyword arguments (line 242)
    kwargs_1178 = {}
    # Getting the type of 'GVector' (line 242)
    GVector_1174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 242)
    GVector_call_result_1179 = invoke(stypy.reporting.localization.Localization(__file__, 242, 12), GVector_1174, *[float_1175, float_1176, float_1177], **kwargs_1178)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1149, GVector_call_result_1179)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 243)
    # Processing the call arguments (line 243)
    float_1181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 20), 'float')
    float_1182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 30), 'float')
    float_1183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 40), 'float')
    # Processing the call keyword arguments (line 243)
    kwargs_1184 = {}
    # Getting the type of 'GVector' (line 243)
    GVector_1180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 243)
    GVector_call_result_1185 = invoke(stypy.reporting.localization.Localization(__file__, 243, 12), GVector_1180, *[float_1181, float_1182, float_1183], **kwargs_1184)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1149, GVector_call_result_1185)
    # Adding element type (line 237)
    
    # Call to GVector(...): (line 244)
    # Processing the call arguments (line 244)
    float_1187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 20), 'float')
    float_1188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 30), 'float')
    float_1189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 40), 'float')
    # Processing the call keyword arguments (line 244)
    kwargs_1190 = {}
    # Getting the type of 'GVector' (line 244)
    GVector_1186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 244)
    GVector_call_result_1191 = invoke(stypy.reporting.localization.Localization(__file__, 244, 12), GVector_1186, *[float_1187, float_1188, float_1189], **kwargs_1190)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 237, 15), list_1149, GVector_call_result_1191)
    
    int_1192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 12), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 245)
    list_1193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 245)
    # Adding element type (line 245)
    int_1194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1193, int_1194)
    # Adding element type (line 245)
    int_1195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1193, int_1195)
    # Adding element type (line 245)
    int_1196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1193, int_1196)
    # Adding element type (line 245)
    int_1197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1193, int_1197)
    # Adding element type (line 245)
    int_1198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1193, int_1198)
    # Adding element type (line 245)
    int_1199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1193, int_1199)
    # Adding element type (line 245)
    int_1200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1193, int_1200)
    # Adding element type (line 245)
    int_1201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1193, int_1201)
    # Adding element type (line 245)
    int_1202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 245, 15), list_1193, int_1202)
    
    # Processing the call keyword arguments (line 237)
    kwargs_1203 = {}
    # Getting the type of 'Spline' (line 237)
    Spline_1148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'Spline', False)
    # Calling Spline(args, kwargs) (line 237)
    Spline_call_result_1204 = invoke(stypy.reporting.localization.Localization(__file__, 237, 8), Spline_1148, *[list_1149, int_1192, list_1193], **kwargs_1203)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 14), list_1147, Spline_call_result_1204)
    # Adding element type (line 236)
    
    # Call to Spline(...): (line 246)
    # Processing the call arguments (line 246)
    
    # Obtaining an instance of the builtin type 'list' (line 246)
    list_1206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 246, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 246)
    # Adding element type (line 246)
    
    # Call to GVector(...): (line 247)
    # Processing the call arguments (line 247)
    float_1208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 20), 'float')
    float_1209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 30), 'float')
    float_1210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 40), 'float')
    # Processing the call keyword arguments (line 247)
    kwargs_1211 = {}
    # Getting the type of 'GVector' (line 247)
    GVector_1207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 247)
    GVector_call_result_1212 = invoke(stypy.reporting.localization.Localization(__file__, 247, 12), GVector_1207, *[float_1208, float_1209, float_1210], **kwargs_1211)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 15), list_1206, GVector_call_result_1212)
    # Adding element type (line 246)
    
    # Call to GVector(...): (line 248)
    # Processing the call arguments (line 248)
    float_1214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 20), 'float')
    float_1215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 30), 'float')
    float_1216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 40), 'float')
    # Processing the call keyword arguments (line 248)
    kwargs_1217 = {}
    # Getting the type of 'GVector' (line 248)
    GVector_1213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 248)
    GVector_call_result_1218 = invoke(stypy.reporting.localization.Localization(__file__, 248, 12), GVector_1213, *[float_1214, float_1215, float_1216], **kwargs_1217)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 15), list_1206, GVector_call_result_1218)
    # Adding element type (line 246)
    
    # Call to GVector(...): (line 249)
    # Processing the call arguments (line 249)
    float_1220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 20), 'float')
    float_1221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 30), 'float')
    float_1222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 40), 'float')
    # Processing the call keyword arguments (line 249)
    kwargs_1223 = {}
    # Getting the type of 'GVector' (line 249)
    GVector_1219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 249)
    GVector_call_result_1224 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), GVector_1219, *[float_1220, float_1221, float_1222], **kwargs_1223)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 15), list_1206, GVector_call_result_1224)
    # Adding element type (line 246)
    
    # Call to GVector(...): (line 250)
    # Processing the call arguments (line 250)
    float_1226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 20), 'float')
    float_1227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 30), 'float')
    float_1228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 40), 'float')
    # Processing the call keyword arguments (line 250)
    kwargs_1229 = {}
    # Getting the type of 'GVector' (line 250)
    GVector_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 250)
    GVector_call_result_1230 = invoke(stypy.reporting.localization.Localization(__file__, 250, 12), GVector_1225, *[float_1226, float_1227, float_1228], **kwargs_1229)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 246, 15), list_1206, GVector_call_result_1230)
    
    int_1231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 12), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 251)
    list_1232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 251)
    # Adding element type (line 251)
    int_1233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 15), list_1232, int_1233)
    # Adding element type (line 251)
    int_1234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 15), list_1232, int_1234)
    # Adding element type (line 251)
    int_1235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 15), list_1232, int_1235)
    # Adding element type (line 251)
    int_1236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 15), list_1232, int_1236)
    # Adding element type (line 251)
    int_1237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 15), list_1232, int_1237)
    # Adding element type (line 251)
    int_1238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 251, 15), list_1232, int_1238)
    
    # Processing the call keyword arguments (line 246)
    kwargs_1239 = {}
    # Getting the type of 'Spline' (line 246)
    Spline_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'Spline', False)
    # Calling Spline(args, kwargs) (line 246)
    Spline_call_result_1240 = invoke(stypy.reporting.localization.Localization(__file__, 246, 8), Spline_1205, *[list_1206, int_1231, list_1232], **kwargs_1239)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 14), list_1147, Spline_call_result_1240)
    # Adding element type (line 236)
    
    # Call to Spline(...): (line 252)
    # Processing the call arguments (line 252)
    
    # Obtaining an instance of the builtin type 'list' (line 252)
    list_1242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 252)
    # Adding element type (line 252)
    
    # Call to GVector(...): (line 253)
    # Processing the call arguments (line 253)
    float_1244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 20), 'float')
    float_1245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 30), 'float')
    float_1246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 40), 'float')
    # Processing the call keyword arguments (line 253)
    kwargs_1247 = {}
    # Getting the type of 'GVector' (line 253)
    GVector_1243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 253)
    GVector_call_result_1248 = invoke(stypy.reporting.localization.Localization(__file__, 253, 12), GVector_1243, *[float_1244, float_1245, float_1246], **kwargs_1247)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 15), list_1242, GVector_call_result_1248)
    # Adding element type (line 252)
    
    # Call to GVector(...): (line 254)
    # Processing the call arguments (line 254)
    float_1250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 20), 'float')
    float_1251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 30), 'float')
    float_1252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 40), 'float')
    # Processing the call keyword arguments (line 254)
    kwargs_1253 = {}
    # Getting the type of 'GVector' (line 254)
    GVector_1249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 254)
    GVector_call_result_1254 = invoke(stypy.reporting.localization.Localization(__file__, 254, 12), GVector_1249, *[float_1250, float_1251, float_1252], **kwargs_1253)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 15), list_1242, GVector_call_result_1254)
    # Adding element type (line 252)
    
    # Call to GVector(...): (line 255)
    # Processing the call arguments (line 255)
    float_1256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 20), 'float')
    float_1257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 30), 'float')
    float_1258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 40), 'float')
    # Processing the call keyword arguments (line 255)
    kwargs_1259 = {}
    # Getting the type of 'GVector' (line 255)
    GVector_1255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 255)
    GVector_call_result_1260 = invoke(stypy.reporting.localization.Localization(__file__, 255, 12), GVector_1255, *[float_1256, float_1257, float_1258], **kwargs_1259)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 15), list_1242, GVector_call_result_1260)
    # Adding element type (line 252)
    
    # Call to GVector(...): (line 256)
    # Processing the call arguments (line 256)
    float_1262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 20), 'float')
    float_1263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 30), 'float')
    float_1264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 40), 'float')
    # Processing the call keyword arguments (line 256)
    kwargs_1265 = {}
    # Getting the type of 'GVector' (line 256)
    GVector_1261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 12), 'GVector', False)
    # Calling GVector(args, kwargs) (line 256)
    GVector_call_result_1266 = invoke(stypy.reporting.localization.Localization(__file__, 256, 12), GVector_1261, *[float_1262, float_1263, float_1264], **kwargs_1265)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 15), list_1242, GVector_call_result_1266)
    
    int_1267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 12), 'int')
    
    # Obtaining an instance of the builtin type 'list' (line 257)
    list_1268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 257)
    # Adding element type (line 257)
    int_1269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_1268, int_1269)
    # Adding element type (line 257)
    int_1270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_1268, int_1270)
    # Adding element type (line 257)
    int_1271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_1268, int_1271)
    # Adding element type (line 257)
    int_1272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_1268, int_1272)
    # Adding element type (line 257)
    int_1273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_1268, int_1273)
    # Adding element type (line 257)
    int_1274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 257, 15), list_1268, int_1274)
    
    # Processing the call keyword arguments (line 252)
    kwargs_1275 = {}
    # Getting the type of 'Spline' (line 252)
    Spline_1241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'Spline', False)
    # Calling Spline(args, kwargs) (line 252)
    Spline_call_result_1276 = invoke(stypy.reporting.localization.Localization(__file__, 252, 8), Spline_1241, *[list_1242, int_1267, list_1268], **kwargs_1275)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 236, 14), list_1147, Spline_call_result_1276)
    
    # Assigning a type to the variable 'splines' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 4), 'splines', list_1147)
    
    # Assigning a Call to a Name (line 259):
    
    # Assigning a Call to a Name (line 259):
    
    # Call to Chaosgame(...): (line 259)
    # Processing the call arguments (line 259)
    # Getting the type of 'splines' (line 259)
    splines_1278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'splines', False)
    float_1279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 27), 'float')
    # Processing the call keyword arguments (line 259)
    kwargs_1280 = {}
    # Getting the type of 'Chaosgame' (line 259)
    Chaosgame_1277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 8), 'Chaosgame', False)
    # Calling Chaosgame(args, kwargs) (line 259)
    Chaosgame_call_result_1281 = invoke(stypy.reporting.localization.Localization(__file__, 259, 8), Chaosgame_1277, *[splines_1278, float_1279], **kwargs_1280)
    
    # Assigning a type to the variable 'c' (line 259)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'c', Chaosgame_call_result_1281)
    
    # Call to create_image_chaos(...): (line 260)
    # Processing the call arguments (line 260)
    int_1284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 32), 'int')
    int_1285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 38), 'int')
    
    # Call to Relative(...): (line 260)
    # Processing the call arguments (line 260)
    str_1287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 53), 'str', 'py.ppm')
    # Processing the call keyword arguments (line 260)
    kwargs_1288 = {}
    # Getting the type of 'Relative' (line 260)
    Relative_1286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 44), 'Relative', False)
    # Calling Relative(args, kwargs) (line 260)
    Relative_call_result_1289 = invoke(stypy.reporting.localization.Localization(__file__, 260, 44), Relative_1286, *[str_1287], **kwargs_1288)
    
    # Getting the type of 'n' (line 260)
    n_1290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 64), 'n', False)
    # Processing the call keyword arguments (line 260)
    kwargs_1291 = {}
    # Getting the type of 'c' (line 260)
    c_1282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'c', False)
    # Obtaining the member 'create_image_chaos' of a type (line 260)
    create_image_chaos_1283 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 11), c_1282, 'create_image_chaos')
    # Calling create_image_chaos(args, kwargs) (line 260)
    create_image_chaos_call_result_1292 = invoke(stypy.reporting.localization.Localization(__file__, 260, 11), create_image_chaos_1283, *[int_1284, int_1285, Relative_call_result_1289, n_1290], **kwargs_1291)
    
    # Assigning a type to the variable 'stypy_return_type' (line 260)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type', create_image_chaos_call_result_1292)
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 235)
    stypy_return_type_1293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1293)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_1293

# Assigning a type to the variable 'main' (line 235)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 263, 0, False)
    
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

    
    # Call to main(...): (line 264)
    # Processing the call arguments (line 264)
    int_1295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 9), 'int')
    # Processing the call keyword arguments (line 264)
    kwargs_1296 = {}
    # Getting the type of 'main' (line 264)
    main_1294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 4), 'main', False)
    # Calling main(args, kwargs) (line 264)
    main_call_result_1297 = invoke(stypy.reporting.localization.Localization(__file__, 264, 4), main_1294, *[int_1295], **kwargs_1296)
    
    # Getting the type of 'True' (line 265)
    True_1298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'stypy_return_type', True_1298)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 263)
    stypy_return_type_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1299)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_1299

# Assigning a type to the variable 'run' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'run', run)

# Call to run(...): (line 267)
# Processing the call keyword arguments (line 267)
kwargs_1301 = {}
# Getting the type of 'run' (line 267)
run_1300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'run', False)
# Calling run(args, kwargs) (line 267)
run_call_result_1302 = invoke(stypy.reporting.localization.Localization(__file__, 267, 0), run_1300, *[], **kwargs_1301)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
