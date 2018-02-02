
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from random import random
2: from math import sqrt
3: import sys
4: 
5: # path tracer, (c) jonas wagner (http://29a.ch/)
6: # http://29a.ch/2010/5/17/path-tracing-a-cornell-box-in-javascript
7: # converted to Python by <anonymous>
8: 
9: ITERATIONS = 10  # should be much higher for good quality
10: 
11: 
12: class V3(object):
13:     def __init__(self, x_, y_, z_):
14:         self.x = x_
15:         self.y = y_
16:         self.z = z_
17: 
18:     def add(self, v):
19:         return V3(self.x + v.x, self.y + v.y, self.z + v.z)
20: 
21:     def iadd(self, v):
22:         self.x += v.x
23:         self.y += v.y
24:         self.z += v.z
25: 
26:     def sub(self, v):
27:         return V3(self.x - v.x, self.y - v.y, self.z - v.z)
28: 
29:     def subdot(self, v, u):
30:         return (self.x - v.x) * u.x + (self.y - v.y) * u.y + (self.z - v.z) * u.z
31: 
32:     def subdot2(self, v):
33:         return (self.x - v.x) ** 2 + (self.y - v.y) ** 2 + (self.z - v.z) ** 2
34: 
35:     def mul(self, v):
36:         return V3(self.x * v.x, self.y * v.y, self.z * v.z)
37: 
38:     def div(self, v):
39:         return V3(self.x / v.x, self.y / v.y, self.z / v.z)
40: 
41:     def muls(self, s):
42:         return V3(self.x * s, self.y * s, self.z * s)
43: 
44:     def divs(self, s):
45:         return self.muls(1.0 / s)
46: 
47:     def dot(self, v):
48:         return self.x * v.x + self.y * v.y + self.z * v.z
49: 
50:     def normalize(self):
51:         return self.divs(sqrt(self.dot(self)))
52: 
53: 
54: def getRandomNormalInHemisphere(v):
55:     '''
56:     This is my crude way of generating random normals in a hemisphere.
57:     In the first step I generate random vectors with components
58:     from -1 to 1. As this introduces a bias I discard all the points
59:     outside of the unit sphere. Now I've got a random normal vector.
60:     The last step is to mirror the poif it is in the wrong hemisphere.
61:     '''
62:     while True:
63:         v2 = V3(random() * 2.0 - 1.0,
64:                 random() * 2.0 - 1.0,
65:                 random() * 2.0 - 1.0)
66:         v2_dot = v2.dot(v2)
67:         if v2_dot <= 1.0:
68:             break
69: 
70:     # should only require about 1.9 iterations of average
71:     # v2 = v2.normalize()
72:     v2 = v2.divs(sqrt(v2_dot))
73: 
74:     # if the pois in the wrong hemisphere, mirror it
75:     if v2.dot(v) < 0.0:
76:         return v2.muls(-1)
77:     return v2
78: 
79: 
80: class Ray(object):
81:     def __init__(self, origin, direction):
82:         self.origin = origin
83:         self.direction = direction
84: 
85: 
86: class Camera(object):
87:     '''
88:     The camera is defined by an eyepo(origin) and three corners
89:     of the view plane (it's a rect in my case...)
90:     '''
91: 
92:     def __init__(self, origin, topleft, topright, bottomleft):
93:         self.origin = origin
94:         self.topleft = topleft
95:         self.topright = topleft
96:         self.bottomleft = bottomleft
97: 
98:         self.xd = topright.sub(topleft)
99:         self.yd = bottomleft.sub(topleft)
100: 
101:     def getRay(self, x, y):
102:         # poon screen plane
103:         p = self.topleft.add(self.xd.muls(x)).add(self.yd.muls(y))
104:         return Ray(self.origin, p.sub(self.origin).normalize())
105: 
106: 
107: class Sphere(object):
108:     def __init__(self, center, radius):
109:         self.center = center
110:         self.radius = radius
111:         self.radius2 = radius * radius
112: 
113:     # returns distance when ray intersects with sphere surface
114:     def intersect(self, ray):
115:         b = ray.origin.subdot(self.center, ray.direction)
116:         c = ray.origin.subdot2(self.center) - self.radius2
117:         d = b * b - c
118:         return (-b - sqrt(d)) if d > 0 else -1.0
119: 
120:     def getNormal(self, point):
121:         return point.sub(self.center).normalize()
122: 
123: 
124: class Material(object):
125:     def __init__(self, color, emission=None):
126:         self.color = color
127:         self.emission = V3(0.0, 0.0, 0.0) if emission is None else emission
128: 
129:     def bounce(self, ray, normal):
130:         return getRandomNormalInHemisphere(normal)
131: 
132: 
133: class Chrome(Material):
134:     def __init__(self, color):
135:         super(Chrome, self).__init__(color)
136: 
137:     def bounce(self, ray, normal):
138:         theta1 = abs(ray.direction.dot(normal))
139:         return ray.direction.add(normal.muls(theta1 * 2.0))
140: 
141: 
142: class Glass(Material):
143:     def __init__(self, color, ior, reflection):
144:         super(Glass, self).__init__(color)
145:         self.ior = ior
146:         self.reflection = reflection
147: 
148:     def bounce(self, ray, normal):
149:         theta1 = abs(ray.direction.dot(normal))
150:         if theta1 >= 0.0:
151:             internalIndex = self.ior
152:             externalIndex = 1.0
153:         else:
154:             internalIndex = 1.0
155:             externalIndex = self.ior
156:         eta = externalIndex / internalIndex
157:         theta2 = sqrt(1.0 - (eta * eta) * (1.0 - (theta1 * theta1)))
158:         rs = (externalIndex * theta1 - internalIndex * theta2) / (externalIndex * theta1 + internalIndex * theta2)
159:         rp = (internalIndex * theta1 - externalIndex * theta2) / (internalIndex * theta1 + externalIndex * theta2)
160:         reflectance = (rs * rs + rp * rp)
161:         # reflection
162:         if random() < reflectance + self.reflection:
163:             return ray.direction.add(normal.muls(theta1 * 2.0))
164:         # refraction
165:         return (ray.direction.add(normal.muls(theta1)).muls(eta).add(normal.muls(-theta2)))
166: 
167: 
168: class Body(object):
169:     def __init__(self, shape, material):
170:         self.shape = shape
171:         self.material = material
172: 
173: 
174: class Output(object):
175:     def __init__(self, width, height):
176:         self.width = width
177:         self.height = height
178: 
179: 
180: class Scene(object):
181:     def __init__(self, output, camera, objects):
182:         self.output = output
183:         self.camera = camera
184:         self.objects = objects
185: 
186: 
187: class Renderer(object):
188:     def __init__(self, scene):
189:         self.scene = scene
190:         self.buffer = [V3(0.0, 0.0, 0.0) for i in xrange(scene.output.width * scene.output.height)]
191: 
192:     def clearBuffer(self):
193:         for i in xrange(len(self.buffer)):
194:             self.buffer[i].x = 0.0
195:             self.buffer[i].y = 0.0
196:             self.buffer[i].z = 0.0
197: 
198:     def iterate(self):
199:         scene = self.scene
200:         w = scene.output.width
201:         h = scene.output.height
202:         i = 0
203:         # randomly jitter pixels so there is no aliasing
204:         y = random() / h
205:         ystep = 1.0 / h
206:         while y < 0.99999:
207:             x = random() / w
208:             xstep = 1.0 / w
209:             while x < 0.99999:
210:                 ray = scene.camera.getRay(x, y)
211:                 color = self.trace(ray, 0)
212:                 self.buffer[i].iadd(color)
213:                 i += 1
214:                 x += xstep
215:             y += ystep
216: 
217:     def trace(self, ray, n):
218:         mint = float("inf")
219: 
220:         # trace no more than 5 bounces
221:         if n > 4:
222:             return V3(0.0, 0.0, 0.0)
223: 
224:         hit = None
225: 
226:         for i in xrange(len(self.scene.objects)):
227:             o = self.scene.objects[i]
228:             t = o.shape.intersect(ray)
229:             if t > 0 and t <= mint:
230:                 mint = t
231:                 hit = o
232: 
233:         if hit is None:
234:             return V3(0.0, 0.0, 0.0)
235: 
236:         point = ray.origin.add(ray.direction.muls(mint))
237:         normal = hit.shape.getNormal(point)
238:         direction = hit.material.bounce(ray, normal)
239:         # if the ray is refractedmove the intersection poa bit in
240:         if direction.dot(ray.direction) > 0.0:
241:             point = ray.origin.add(ray.direction.muls(mint * 1.0000001))
242:             # otherwise move it out to prevent problems with floating point
243:             # accuracy
244:         else:
245:             point = ray.origin.add(ray.direction.muls(mint * 0.9999999))
246:         newray = Ray(point, direction)
247:         return self.trace(newray, n + 1).mul(hit.material.color).add(hit.material.emission)
248: 
249:     @staticmethod
250:     def cmap(x):
251:         return 0 if x < 0.0 else (255 if x > 1.0 else int(x * 255))
252: 
253:     # / Write image to PPM file
254:     def saveFrame(self, filename, nframe):
255:         fout = file(filename, "w")
256:         fout.write("P3\n%d %d\n%d\n" % (self.scene.output.width, self.scene.output.height, 255))
257:         for p in self.buffer:
258:             fout.write("%d %d %d\n" % (Renderer.cmap(p.x / nframe),
259:                                        Renderer.cmap(p.y / nframe),
260:                                        Renderer.cmap(p.z / nframe)))
261:         fout.close()
262: 
263: 
264: def run():
265:     width = 320
266:     height = 240
267: 
268:     scene = Scene(
269:         Output(width, height),
270: 
271:         Camera(
272:             V3(0.0, -0.5, 0.0),
273:             V3(-1.3, 1.0, 1.0),
274:             V3(1.3, 1.0, 1.0),
275:             V3(-1.3, 1.0, -1.0)
276:         ),
277: 
278:         [
279:             # glowing sphere
280:             # Body(Sphere(V3(0.0, 3.0, 0.0), 0.5), Material(V3(0.9, 0.9, 0.9), V3(1.5, 1.5, 1.5))),
281:             # glass sphere
282:             Body(Sphere(V3(1.0, 2.0, 0.0), 0.5), Glass(V3(1.00, 1.00, 1.00), 1.5, 0.1)),
283:             # chrome sphere
284:             Body(Sphere(V3(-1.1, 2.8, 0.0), 0.5), Chrome(V3(0.8, 0.8, 0.8))),
285:             # floor
286:             Body(Sphere(V3(0.0, 3.5, -10e6), 10e6 - 0.5), Material(V3(0.9, 0.9, 0.9))),
287:             # back
288:             Body(Sphere(V3(0.0, 10e6, 0.0), 10e6 - 4.5), Material(V3(0.9, 0.9, 0.9))),
289:             # left
290:             Body(Sphere(V3(-10e6, 3.5, 0.0), 10e6 - 1.9), Material(V3(0.9, 0.5, 0.5))),
291:             # right
292:             Body(Sphere(V3(10e6, 3.5, 0.0), 10e6 - 1.9), Material(V3(0.5, 0.5, 0.9))),
293:             # top light, the emmision should be close to that of warm sunlight (~5400k)
294:             Body(Sphere(V3(0.0, 0.0, 10e6), 10e6 - 2.5), Material(V3(0.0, 0.0, 0.0), V3(1.6, 1.47, 1.29))),
295:             # front
296:             Body(Sphere(V3(0.0, -10e6, 0.0), 10e6 - 2.5), Material(V3(0.9, 0.9, 0.9))),
297:         ]
298:     )
299: 
300:     renderer = Renderer(scene)
301: 
302:     nframe = 0
303:     for count in range(ITERATIONS):
304:         renderer.iterate()
305:         ##        sys.stdout.write('*')
306:         ##        sys.stdout.flush()
307:         nframe += 1
308: 
309:     renderer.saveFrame("pt.ppm", nframe)
310:     return True
311: 
312: 
313: run()
314: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from random import random' statement (line 1)
try:
    from random import random

except:
    random = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'random', None, module_type_store, ['random'], [random])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from math import sqrt' statement (line 2)
try:
    from math import sqrt

except:
    sqrt = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'math', None, module_type_store, ['sqrt'], [sqrt])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)


# Assigning a Num to a Name (line 9):
int_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 13), 'int')
# Assigning a type to the variable 'ITERATIONS' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'ITERATIONS', int_1)
# Declaration of the 'V3' class

class V3(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'V3.__init__', ['x_', 'y_', 'z_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['x_', 'y_', 'z_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 14):
        # Getting the type of 'x_' (line 14)
        x__2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'x_')
        # Getting the type of 'self' (line 14)
        self_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member 'x' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_3, 'x', x__2)
        
        # Assigning a Name to a Attribute (line 15):
        # Getting the type of 'y_' (line 15)
        y__4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 17), 'y_')
        # Getting the type of 'self' (line 15)
        self_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member 'y' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_5, 'y', y__4)
        
        # Assigning a Name to a Attribute (line 16):
        # Getting the type of 'z_' (line 16)
        z__6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'z_')
        # Getting the type of 'self' (line 16)
        self_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self')
        # Setting the type of the member 'z' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_7, 'z', z__6)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def add(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add'
        module_type_store = module_type_store.open_function_context('add', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        V3.add.__dict__.__setitem__('stypy_localization', localization)
        V3.add.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        V3.add.__dict__.__setitem__('stypy_type_store', module_type_store)
        V3.add.__dict__.__setitem__('stypy_function_name', 'V3.add')
        V3.add.__dict__.__setitem__('stypy_param_names_list', ['v'])
        V3.add.__dict__.__setitem__('stypy_varargs_param_name', None)
        V3.add.__dict__.__setitem__('stypy_kwargs_param_name', None)
        V3.add.__dict__.__setitem__('stypy_call_defaults', defaults)
        V3.add.__dict__.__setitem__('stypy_call_varargs', varargs)
        V3.add.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        V3.add.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'V3.add', ['v'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add', localization, ['v'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add(...)' code ##################

        
        # Call to V3(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'self' (line 19)
        self_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 18), 'self', False)
        # Obtaining the member 'x' of a type (line 19)
        x_10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 18), self_9, 'x')
        # Getting the type of 'v' (line 19)
        v_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'v', False)
        # Obtaining the member 'x' of a type (line 19)
        x_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 27), v_11, 'x')
        # Applying the binary operator '+' (line 19)
        result_add_13 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 18), '+', x_10, x_12)
        
        # Getting the type of 'self' (line 19)
        self_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 32), 'self', False)
        # Obtaining the member 'y' of a type (line 19)
        y_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 32), self_14, 'y')
        # Getting the type of 'v' (line 19)
        v_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 41), 'v', False)
        # Obtaining the member 'y' of a type (line 19)
        y_17 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 41), v_16, 'y')
        # Applying the binary operator '+' (line 19)
        result_add_18 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 32), '+', y_15, y_17)
        
        # Getting the type of 'self' (line 19)
        self_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 46), 'self', False)
        # Obtaining the member 'z' of a type (line 19)
        z_20 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 46), self_19, 'z')
        # Getting the type of 'v' (line 19)
        v_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 55), 'v', False)
        # Obtaining the member 'z' of a type (line 19)
        z_22 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 55), v_21, 'z')
        # Applying the binary operator '+' (line 19)
        result_add_23 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 46), '+', z_20, z_22)
        
        # Processing the call keyword arguments (line 19)
        kwargs_24 = {}
        # Getting the type of 'V3' (line 19)
        V3_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'V3', False)
        # Calling V3(args, kwargs) (line 19)
        V3_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), V3_8, *[result_add_13, result_add_18, result_add_23], **kwargs_24)
        
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type', V3_call_result_25)
        
        # ################# End of 'add(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_26)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add'
        return stypy_return_type_26


    @norecursion
    def iadd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'iadd'
        module_type_store = module_type_store.open_function_context('iadd', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        V3.iadd.__dict__.__setitem__('stypy_localization', localization)
        V3.iadd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        V3.iadd.__dict__.__setitem__('stypy_type_store', module_type_store)
        V3.iadd.__dict__.__setitem__('stypy_function_name', 'V3.iadd')
        V3.iadd.__dict__.__setitem__('stypy_param_names_list', ['v'])
        V3.iadd.__dict__.__setitem__('stypy_varargs_param_name', None)
        V3.iadd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        V3.iadd.__dict__.__setitem__('stypy_call_defaults', defaults)
        V3.iadd.__dict__.__setitem__('stypy_call_varargs', varargs)
        V3.iadd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        V3.iadd.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'V3.iadd', ['v'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'iadd', localization, ['v'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'iadd(...)' code ##################

        
        # Getting the type of 'self' (line 22)
        self_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Obtaining the member 'x' of a type (line 22)
        x_28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_27, 'x')
        # Getting the type of 'v' (line 22)
        v_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), 'v')
        # Obtaining the member 'x' of a type (line 22)
        x_30 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 18), v_29, 'x')
        # Applying the binary operator '+=' (line 22)
        result_iadd_31 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 8), '+=', x_28, x_30)
        # Getting the type of 'self' (line 22)
        self_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'x' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_32, 'x', result_iadd_31)
        
        
        # Getting the type of 'self' (line 23)
        self_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Obtaining the member 'y' of a type (line 23)
        y_34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_33, 'y')
        # Getting the type of 'v' (line 23)
        v_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 18), 'v')
        # Obtaining the member 'y' of a type (line 23)
        y_36 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 18), v_35, 'y')
        # Applying the binary operator '+=' (line 23)
        result_iadd_37 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 8), '+=', y_34, y_36)
        # Getting the type of 'self' (line 23)
        self_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'y' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_38, 'y', result_iadd_37)
        
        
        # Getting the type of 'self' (line 24)
        self_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Obtaining the member 'z' of a type (line 24)
        z_40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_39, 'z')
        # Getting the type of 'v' (line 24)
        v_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'v')
        # Obtaining the member 'z' of a type (line 24)
        z_42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 18), v_41, 'z')
        # Applying the binary operator '+=' (line 24)
        result_iadd_43 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 8), '+=', z_40, z_42)
        # Getting the type of 'self' (line 24)
        self_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Setting the type of the member 'z' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_44, 'z', result_iadd_43)
        
        
        # ################# End of 'iadd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'iadd' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_45)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'iadd'
        return stypy_return_type_45


    @norecursion
    def sub(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sub'
        module_type_store = module_type_store.open_function_context('sub', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        V3.sub.__dict__.__setitem__('stypy_localization', localization)
        V3.sub.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        V3.sub.__dict__.__setitem__('stypy_type_store', module_type_store)
        V3.sub.__dict__.__setitem__('stypy_function_name', 'V3.sub')
        V3.sub.__dict__.__setitem__('stypy_param_names_list', ['v'])
        V3.sub.__dict__.__setitem__('stypy_varargs_param_name', None)
        V3.sub.__dict__.__setitem__('stypy_kwargs_param_name', None)
        V3.sub.__dict__.__setitem__('stypy_call_defaults', defaults)
        V3.sub.__dict__.__setitem__('stypy_call_varargs', varargs)
        V3.sub.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        V3.sub.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'V3.sub', ['v'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sub', localization, ['v'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sub(...)' code ##################

        
        # Call to V3(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'self' (line 27)
        self_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'self', False)
        # Obtaining the member 'x' of a type (line 27)
        x_48 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 18), self_47, 'x')
        # Getting the type of 'v' (line 27)
        v_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 27), 'v', False)
        # Obtaining the member 'x' of a type (line 27)
        x_50 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 27), v_49, 'x')
        # Applying the binary operator '-' (line 27)
        result_sub_51 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 18), '-', x_48, x_50)
        
        # Getting the type of 'self' (line 27)
        self_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 32), 'self', False)
        # Obtaining the member 'y' of a type (line 27)
        y_53 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 32), self_52, 'y')
        # Getting the type of 'v' (line 27)
        v_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 41), 'v', False)
        # Obtaining the member 'y' of a type (line 27)
        y_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 41), v_54, 'y')
        # Applying the binary operator '-' (line 27)
        result_sub_56 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 32), '-', y_53, y_55)
        
        # Getting the type of 'self' (line 27)
        self_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 46), 'self', False)
        # Obtaining the member 'z' of a type (line 27)
        z_58 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 46), self_57, 'z')
        # Getting the type of 'v' (line 27)
        v_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 55), 'v', False)
        # Obtaining the member 'z' of a type (line 27)
        z_60 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 55), v_59, 'z')
        # Applying the binary operator '-' (line 27)
        result_sub_61 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 46), '-', z_58, z_60)
        
        # Processing the call keyword arguments (line 27)
        kwargs_62 = {}
        # Getting the type of 'V3' (line 27)
        V3_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'V3', False)
        # Calling V3(args, kwargs) (line 27)
        V3_call_result_63 = invoke(stypy.reporting.localization.Localization(__file__, 27, 15), V3_46, *[result_sub_51, result_sub_56, result_sub_61], **kwargs_62)
        
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', V3_call_result_63)
        
        # ################# End of 'sub(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sub' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_64)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sub'
        return stypy_return_type_64


    @norecursion
    def subdot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'subdot'
        module_type_store = module_type_store.open_function_context('subdot', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        V3.subdot.__dict__.__setitem__('stypy_localization', localization)
        V3.subdot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        V3.subdot.__dict__.__setitem__('stypy_type_store', module_type_store)
        V3.subdot.__dict__.__setitem__('stypy_function_name', 'V3.subdot')
        V3.subdot.__dict__.__setitem__('stypy_param_names_list', ['v', 'u'])
        V3.subdot.__dict__.__setitem__('stypy_varargs_param_name', None)
        V3.subdot.__dict__.__setitem__('stypy_kwargs_param_name', None)
        V3.subdot.__dict__.__setitem__('stypy_call_defaults', defaults)
        V3.subdot.__dict__.__setitem__('stypy_call_varargs', varargs)
        V3.subdot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        V3.subdot.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'V3.subdot', ['v', 'u'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'subdot', localization, ['v', 'u'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'subdot(...)' code ##################

        # Getting the type of 'self' (line 30)
        self_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'self')
        # Obtaining the member 'x' of a type (line 30)
        x_66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 16), self_65, 'x')
        # Getting the type of 'v' (line 30)
        v_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'v')
        # Obtaining the member 'x' of a type (line 30)
        x_68 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 25), v_67, 'x')
        # Applying the binary operator '-' (line 30)
        result_sub_69 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 16), '-', x_66, x_68)
        
        # Getting the type of 'u' (line 30)
        u_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), 'u')
        # Obtaining the member 'x' of a type (line 30)
        x_71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 32), u_70, 'x')
        # Applying the binary operator '*' (line 30)
        result_mul_72 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 15), '*', result_sub_69, x_71)
        
        # Getting the type of 'self' (line 30)
        self_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 39), 'self')
        # Obtaining the member 'y' of a type (line 30)
        y_74 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 39), self_73, 'y')
        # Getting the type of 'v' (line 30)
        v_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 48), 'v')
        # Obtaining the member 'y' of a type (line 30)
        y_76 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 48), v_75, 'y')
        # Applying the binary operator '-' (line 30)
        result_sub_77 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 39), '-', y_74, y_76)
        
        # Getting the type of 'u' (line 30)
        u_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 55), 'u')
        # Obtaining the member 'y' of a type (line 30)
        y_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 55), u_78, 'y')
        # Applying the binary operator '*' (line 30)
        result_mul_80 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 38), '*', result_sub_77, y_79)
        
        # Applying the binary operator '+' (line 30)
        result_add_81 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 15), '+', result_mul_72, result_mul_80)
        
        # Getting the type of 'self' (line 30)
        self_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 62), 'self')
        # Obtaining the member 'z' of a type (line 30)
        z_83 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 62), self_82, 'z')
        # Getting the type of 'v' (line 30)
        v_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 71), 'v')
        # Obtaining the member 'z' of a type (line 30)
        z_85 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 71), v_84, 'z')
        # Applying the binary operator '-' (line 30)
        result_sub_86 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 62), '-', z_83, z_85)
        
        # Getting the type of 'u' (line 30)
        u_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 78), 'u')
        # Obtaining the member 'z' of a type (line 30)
        z_88 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 78), u_87, 'z')
        # Applying the binary operator '*' (line 30)
        result_mul_89 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 61), '*', result_sub_86, z_88)
        
        # Applying the binary operator '+' (line 30)
        result_add_90 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 59), '+', result_add_81, result_mul_89)
        
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', result_add_90)
        
        # ################# End of 'subdot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'subdot' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_91)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'subdot'
        return stypy_return_type_91


    @norecursion
    def subdot2(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'subdot2'
        module_type_store = module_type_store.open_function_context('subdot2', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        V3.subdot2.__dict__.__setitem__('stypy_localization', localization)
        V3.subdot2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        V3.subdot2.__dict__.__setitem__('stypy_type_store', module_type_store)
        V3.subdot2.__dict__.__setitem__('stypy_function_name', 'V3.subdot2')
        V3.subdot2.__dict__.__setitem__('stypy_param_names_list', ['v'])
        V3.subdot2.__dict__.__setitem__('stypy_varargs_param_name', None)
        V3.subdot2.__dict__.__setitem__('stypy_kwargs_param_name', None)
        V3.subdot2.__dict__.__setitem__('stypy_call_defaults', defaults)
        V3.subdot2.__dict__.__setitem__('stypy_call_varargs', varargs)
        V3.subdot2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        V3.subdot2.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'V3.subdot2', ['v'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'subdot2', localization, ['v'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'subdot2(...)' code ##################

        # Getting the type of 'self' (line 33)
        self_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'self')
        # Obtaining the member 'x' of a type (line 33)
        x_93 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 16), self_92, 'x')
        # Getting the type of 'v' (line 33)
        v_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'v')
        # Obtaining the member 'x' of a type (line 33)
        x_95 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 25), v_94, 'x')
        # Applying the binary operator '-' (line 33)
        result_sub_96 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 16), '-', x_93, x_95)
        
        int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 33), 'int')
        # Applying the binary operator '**' (line 33)
        result_pow_98 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 15), '**', result_sub_96, int_97)
        
        # Getting the type of 'self' (line 33)
        self_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 38), 'self')
        # Obtaining the member 'y' of a type (line 33)
        y_100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 38), self_99, 'y')
        # Getting the type of 'v' (line 33)
        v_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 47), 'v')
        # Obtaining the member 'y' of a type (line 33)
        y_102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 47), v_101, 'y')
        # Applying the binary operator '-' (line 33)
        result_sub_103 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 38), '-', y_100, y_102)
        
        int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 55), 'int')
        # Applying the binary operator '**' (line 33)
        result_pow_105 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 37), '**', result_sub_103, int_104)
        
        # Applying the binary operator '+' (line 33)
        result_add_106 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 15), '+', result_pow_98, result_pow_105)
        
        # Getting the type of 'self' (line 33)
        self_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 60), 'self')
        # Obtaining the member 'z' of a type (line 33)
        z_108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 60), self_107, 'z')
        # Getting the type of 'v' (line 33)
        v_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 69), 'v')
        # Obtaining the member 'z' of a type (line 33)
        z_110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 69), v_109, 'z')
        # Applying the binary operator '-' (line 33)
        result_sub_111 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 60), '-', z_108, z_110)
        
        int_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 77), 'int')
        # Applying the binary operator '**' (line 33)
        result_pow_113 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 59), '**', result_sub_111, int_112)
        
        # Applying the binary operator '+' (line 33)
        result_add_114 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 57), '+', result_add_106, result_pow_113)
        
        # Assigning a type to the variable 'stypy_return_type' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', result_add_114)
        
        # ################# End of 'subdot2(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'subdot2' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'subdot2'
        return stypy_return_type_115


    @norecursion
    def mul(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mul'
        module_type_store = module_type_store.open_function_context('mul', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        V3.mul.__dict__.__setitem__('stypy_localization', localization)
        V3.mul.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        V3.mul.__dict__.__setitem__('stypy_type_store', module_type_store)
        V3.mul.__dict__.__setitem__('stypy_function_name', 'V3.mul')
        V3.mul.__dict__.__setitem__('stypy_param_names_list', ['v'])
        V3.mul.__dict__.__setitem__('stypy_varargs_param_name', None)
        V3.mul.__dict__.__setitem__('stypy_kwargs_param_name', None)
        V3.mul.__dict__.__setitem__('stypy_call_defaults', defaults)
        V3.mul.__dict__.__setitem__('stypy_call_varargs', varargs)
        V3.mul.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        V3.mul.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'V3.mul', ['v'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mul', localization, ['v'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mul(...)' code ##################

        
        # Call to V3(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 18), 'self', False)
        # Obtaining the member 'x' of a type (line 36)
        x_118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 18), self_117, 'x')
        # Getting the type of 'v' (line 36)
        v_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'v', False)
        # Obtaining the member 'x' of a type (line 36)
        x_120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 27), v_119, 'x')
        # Applying the binary operator '*' (line 36)
        result_mul_121 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 18), '*', x_118, x_120)
        
        # Getting the type of 'self' (line 36)
        self_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 32), 'self', False)
        # Obtaining the member 'y' of a type (line 36)
        y_123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 32), self_122, 'y')
        # Getting the type of 'v' (line 36)
        v_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 41), 'v', False)
        # Obtaining the member 'y' of a type (line 36)
        y_125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 41), v_124, 'y')
        # Applying the binary operator '*' (line 36)
        result_mul_126 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 32), '*', y_123, y_125)
        
        # Getting the type of 'self' (line 36)
        self_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 46), 'self', False)
        # Obtaining the member 'z' of a type (line 36)
        z_128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 46), self_127, 'z')
        # Getting the type of 'v' (line 36)
        v_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 55), 'v', False)
        # Obtaining the member 'z' of a type (line 36)
        z_130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 55), v_129, 'z')
        # Applying the binary operator '*' (line 36)
        result_mul_131 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 46), '*', z_128, z_130)
        
        # Processing the call keyword arguments (line 36)
        kwargs_132 = {}
        # Getting the type of 'V3' (line 36)
        V3_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'V3', False)
        # Calling V3(args, kwargs) (line 36)
        V3_call_result_133 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), V3_116, *[result_mul_121, result_mul_126, result_mul_131], **kwargs_132)
        
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', V3_call_result_133)
        
        # ################# End of 'mul(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mul' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mul'
        return stypy_return_type_134


    @norecursion
    def div(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'div'
        module_type_store = module_type_store.open_function_context('div', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        V3.div.__dict__.__setitem__('stypy_localization', localization)
        V3.div.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        V3.div.__dict__.__setitem__('stypy_type_store', module_type_store)
        V3.div.__dict__.__setitem__('stypy_function_name', 'V3.div')
        V3.div.__dict__.__setitem__('stypy_param_names_list', ['v'])
        V3.div.__dict__.__setitem__('stypy_varargs_param_name', None)
        V3.div.__dict__.__setitem__('stypy_kwargs_param_name', None)
        V3.div.__dict__.__setitem__('stypy_call_defaults', defaults)
        V3.div.__dict__.__setitem__('stypy_call_varargs', varargs)
        V3.div.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        V3.div.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'V3.div', ['v'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'div', localization, ['v'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'div(...)' code ##################

        
        # Call to V3(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'self' (line 39)
        self_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'self', False)
        # Obtaining the member 'x' of a type (line 39)
        x_137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 18), self_136, 'x')
        # Getting the type of 'v' (line 39)
        v_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'v', False)
        # Obtaining the member 'x' of a type (line 39)
        x_139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 27), v_138, 'x')
        # Applying the binary operator 'div' (line 39)
        result_div_140 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 18), 'div', x_137, x_139)
        
        # Getting the type of 'self' (line 39)
        self_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 32), 'self', False)
        # Obtaining the member 'y' of a type (line 39)
        y_142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 32), self_141, 'y')
        # Getting the type of 'v' (line 39)
        v_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 41), 'v', False)
        # Obtaining the member 'y' of a type (line 39)
        y_144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 41), v_143, 'y')
        # Applying the binary operator 'div' (line 39)
        result_div_145 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 32), 'div', y_142, y_144)
        
        # Getting the type of 'self' (line 39)
        self_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 46), 'self', False)
        # Obtaining the member 'z' of a type (line 39)
        z_147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 46), self_146, 'z')
        # Getting the type of 'v' (line 39)
        v_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 55), 'v', False)
        # Obtaining the member 'z' of a type (line 39)
        z_149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 55), v_148, 'z')
        # Applying the binary operator 'div' (line 39)
        result_div_150 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 46), 'div', z_147, z_149)
        
        # Processing the call keyword arguments (line 39)
        kwargs_151 = {}
        # Getting the type of 'V3' (line 39)
        V3_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'V3', False)
        # Calling V3(args, kwargs) (line 39)
        V3_call_result_152 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), V3_135, *[result_div_140, result_div_145, result_div_150], **kwargs_151)
        
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', V3_call_result_152)
        
        # ################# End of 'div(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'div' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_153)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'div'
        return stypy_return_type_153


    @norecursion
    def muls(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'muls'
        module_type_store = module_type_store.open_function_context('muls', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        V3.muls.__dict__.__setitem__('stypy_localization', localization)
        V3.muls.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        V3.muls.__dict__.__setitem__('stypy_type_store', module_type_store)
        V3.muls.__dict__.__setitem__('stypy_function_name', 'V3.muls')
        V3.muls.__dict__.__setitem__('stypy_param_names_list', ['s'])
        V3.muls.__dict__.__setitem__('stypy_varargs_param_name', None)
        V3.muls.__dict__.__setitem__('stypy_kwargs_param_name', None)
        V3.muls.__dict__.__setitem__('stypy_call_defaults', defaults)
        V3.muls.__dict__.__setitem__('stypy_call_varargs', varargs)
        V3.muls.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        V3.muls.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'V3.muls', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'muls', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'muls(...)' code ##################

        
        # Call to V3(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'self' (line 42)
        self_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), 'self', False)
        # Obtaining the member 'x' of a type (line 42)
        x_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 18), self_155, 'x')
        # Getting the type of 's' (line 42)
        s_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 's', False)
        # Applying the binary operator '*' (line 42)
        result_mul_158 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 18), '*', x_156, s_157)
        
        # Getting the type of 'self' (line 42)
        self_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 30), 'self', False)
        # Obtaining the member 'y' of a type (line 42)
        y_160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 30), self_159, 'y')
        # Getting the type of 's' (line 42)
        s_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 's', False)
        # Applying the binary operator '*' (line 42)
        result_mul_162 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 30), '*', y_160, s_161)
        
        # Getting the type of 'self' (line 42)
        self_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 42), 'self', False)
        # Obtaining the member 'z' of a type (line 42)
        z_164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 42), self_163, 'z')
        # Getting the type of 's' (line 42)
        s_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 51), 's', False)
        # Applying the binary operator '*' (line 42)
        result_mul_166 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 42), '*', z_164, s_165)
        
        # Processing the call keyword arguments (line 42)
        kwargs_167 = {}
        # Getting the type of 'V3' (line 42)
        V3_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'V3', False)
        # Calling V3(args, kwargs) (line 42)
        V3_call_result_168 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), V3_154, *[result_mul_158, result_mul_162, result_mul_166], **kwargs_167)
        
        # Assigning a type to the variable 'stypy_return_type' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', V3_call_result_168)
        
        # ################# End of 'muls(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'muls' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'muls'
        return stypy_return_type_169


    @norecursion
    def divs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'divs'
        module_type_store = module_type_store.open_function_context('divs', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        V3.divs.__dict__.__setitem__('stypy_localization', localization)
        V3.divs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        V3.divs.__dict__.__setitem__('stypy_type_store', module_type_store)
        V3.divs.__dict__.__setitem__('stypy_function_name', 'V3.divs')
        V3.divs.__dict__.__setitem__('stypy_param_names_list', ['s'])
        V3.divs.__dict__.__setitem__('stypy_varargs_param_name', None)
        V3.divs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        V3.divs.__dict__.__setitem__('stypy_call_defaults', defaults)
        V3.divs.__dict__.__setitem__('stypy_call_varargs', varargs)
        V3.divs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        V3.divs.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'V3.divs', ['s'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'divs', localization, ['s'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'divs(...)' code ##################

        
        # Call to muls(...): (line 45)
        # Processing the call arguments (line 45)
        float_172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'float')
        # Getting the type of 's' (line 45)
        s_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 's', False)
        # Applying the binary operator 'div' (line 45)
        result_div_174 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 25), 'div', float_172, s_173)
        
        # Processing the call keyword arguments (line 45)
        kwargs_175 = {}
        # Getting the type of 'self' (line 45)
        self_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'self', False)
        # Obtaining the member 'muls' of a type (line 45)
        muls_171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 15), self_170, 'muls')
        # Calling muls(args, kwargs) (line 45)
        muls_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 45, 15), muls_171, *[result_div_174], **kwargs_175)
        
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', muls_call_result_176)
        
        # ################# End of 'divs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'divs' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_177)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'divs'
        return stypy_return_type_177


    @norecursion
    def dot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dot'
        module_type_store = module_type_store.open_function_context('dot', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        V3.dot.__dict__.__setitem__('stypy_localization', localization)
        V3.dot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        V3.dot.__dict__.__setitem__('stypy_type_store', module_type_store)
        V3.dot.__dict__.__setitem__('stypy_function_name', 'V3.dot')
        V3.dot.__dict__.__setitem__('stypy_param_names_list', ['v'])
        V3.dot.__dict__.__setitem__('stypy_varargs_param_name', None)
        V3.dot.__dict__.__setitem__('stypy_kwargs_param_name', None)
        V3.dot.__dict__.__setitem__('stypy_call_defaults', defaults)
        V3.dot.__dict__.__setitem__('stypy_call_varargs', varargs)
        V3.dot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        V3.dot.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'V3.dot', ['v'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dot', localization, ['v'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dot(...)' code ##################

        # Getting the type of 'self' (line 48)
        self_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'self')
        # Obtaining the member 'x' of a type (line 48)
        x_179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 15), self_178, 'x')
        # Getting the type of 'v' (line 48)
        v_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'v')
        # Obtaining the member 'x' of a type (line 48)
        x_181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 24), v_180, 'x')
        # Applying the binary operator '*' (line 48)
        result_mul_182 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 15), '*', x_179, x_181)
        
        # Getting the type of 'self' (line 48)
        self_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 30), 'self')
        # Obtaining the member 'y' of a type (line 48)
        y_184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 30), self_183, 'y')
        # Getting the type of 'v' (line 48)
        v_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 39), 'v')
        # Obtaining the member 'y' of a type (line 48)
        y_186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 39), v_185, 'y')
        # Applying the binary operator '*' (line 48)
        result_mul_187 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 30), '*', y_184, y_186)
        
        # Applying the binary operator '+' (line 48)
        result_add_188 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 15), '+', result_mul_182, result_mul_187)
        
        # Getting the type of 'self' (line 48)
        self_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 45), 'self')
        # Obtaining the member 'z' of a type (line 48)
        z_190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 45), self_189, 'z')
        # Getting the type of 'v' (line 48)
        v_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 54), 'v')
        # Obtaining the member 'z' of a type (line 48)
        z_192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 54), v_191, 'z')
        # Applying the binary operator '*' (line 48)
        result_mul_193 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 45), '*', z_190, z_192)
        
        # Applying the binary operator '+' (line 48)
        result_add_194 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 43), '+', result_add_188, result_mul_193)
        
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', result_add_194)
        
        # ################# End of 'dot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dot' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_195)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dot'
        return stypy_return_type_195


    @norecursion
    def normalize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'normalize'
        module_type_store = module_type_store.open_function_context('normalize', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        V3.normalize.__dict__.__setitem__('stypy_localization', localization)
        V3.normalize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        V3.normalize.__dict__.__setitem__('stypy_type_store', module_type_store)
        V3.normalize.__dict__.__setitem__('stypy_function_name', 'V3.normalize')
        V3.normalize.__dict__.__setitem__('stypy_param_names_list', [])
        V3.normalize.__dict__.__setitem__('stypy_varargs_param_name', None)
        V3.normalize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        V3.normalize.__dict__.__setitem__('stypy_call_defaults', defaults)
        V3.normalize.__dict__.__setitem__('stypy_call_varargs', varargs)
        V3.normalize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        V3.normalize.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'V3.normalize', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'normalize', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'normalize(...)' code ##################

        
        # Call to divs(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to sqrt(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to dot(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'self' (line 51)
        self_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 39), 'self', False)
        # Processing the call keyword arguments (line 51)
        kwargs_202 = {}
        # Getting the type of 'self' (line 51)
        self_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'self', False)
        # Obtaining the member 'dot' of a type (line 51)
        dot_200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 30), self_199, 'dot')
        # Calling dot(args, kwargs) (line 51)
        dot_call_result_203 = invoke(stypy.reporting.localization.Localization(__file__, 51, 30), dot_200, *[self_201], **kwargs_202)
        
        # Processing the call keyword arguments (line 51)
        kwargs_204 = {}
        # Getting the type of 'sqrt' (line 51)
        sqrt_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 51)
        sqrt_call_result_205 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), sqrt_198, *[dot_call_result_203], **kwargs_204)
        
        # Processing the call keyword arguments (line 51)
        kwargs_206 = {}
        # Getting the type of 'self' (line 51)
        self_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'self', False)
        # Obtaining the member 'divs' of a type (line 51)
        divs_197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 15), self_196, 'divs')
        # Calling divs(args, kwargs) (line 51)
        divs_call_result_207 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), divs_197, *[sqrt_call_result_205], **kwargs_206)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', divs_call_result_207)
        
        # ################# End of 'normalize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'normalize' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_208)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'normalize'
        return stypy_return_type_208


# Assigning a type to the variable 'V3' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'V3', V3)

@norecursion
def getRandomNormalInHemisphere(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'getRandomNormalInHemisphere'
    module_type_store = module_type_store.open_function_context('getRandomNormalInHemisphere', 54, 0, False)
    
    # Passed parameters checking function
    getRandomNormalInHemisphere.stypy_localization = localization
    getRandomNormalInHemisphere.stypy_type_of_self = None
    getRandomNormalInHemisphere.stypy_type_store = module_type_store
    getRandomNormalInHemisphere.stypy_function_name = 'getRandomNormalInHemisphere'
    getRandomNormalInHemisphere.stypy_param_names_list = ['v']
    getRandomNormalInHemisphere.stypy_varargs_param_name = None
    getRandomNormalInHemisphere.stypy_kwargs_param_name = None
    getRandomNormalInHemisphere.stypy_call_defaults = defaults
    getRandomNormalInHemisphere.stypy_call_varargs = varargs
    getRandomNormalInHemisphere.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'getRandomNormalInHemisphere', ['v'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'getRandomNormalInHemisphere', localization, ['v'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'getRandomNormalInHemisphere(...)' code ##################

    str_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', "\n    This is my crude way of generating random normals in a hemisphere.\n    In the first step I generate random vectors with components\n    from -1 to 1. As this introduces a bias I discard all the points\n    outside of the unit sphere. Now I've got a random normal vector.\n    The last step is to mirror the poif it is in the wrong hemisphere.\n    ")
    
    # Getting the type of 'True' (line 62)
    True_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 10), 'True')
    # Testing if the while is going to be iterated (line 62)
    # Testing the type of an if condition (line 62)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 4), True_210)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 62, 4), True_210):
        # SSA begins for while statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 63):
        
        # Call to V3(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Call to random(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_213 = {}
        # Getting the type of 'random' (line 63)
        random_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'random', False)
        # Calling random(args, kwargs) (line 63)
        random_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), random_212, *[], **kwargs_213)
        
        float_215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'float')
        # Applying the binary operator '*' (line 63)
        result_mul_216 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 16), '*', random_call_result_214, float_215)
        
        float_217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 33), 'float')
        # Applying the binary operator '-' (line 63)
        result_sub_218 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 16), '-', result_mul_216, float_217)
        
        
        # Call to random(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_220 = {}
        # Getting the type of 'random' (line 64)
        random_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'random', False)
        # Calling random(args, kwargs) (line 64)
        random_call_result_221 = invoke(stypy.reporting.localization.Localization(__file__, 64, 16), random_219, *[], **kwargs_220)
        
        float_222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 27), 'float')
        # Applying the binary operator '*' (line 64)
        result_mul_223 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 16), '*', random_call_result_221, float_222)
        
        float_224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 33), 'float')
        # Applying the binary operator '-' (line 64)
        result_sub_225 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 16), '-', result_mul_223, float_224)
        
        
        # Call to random(...): (line 65)
        # Processing the call keyword arguments (line 65)
        kwargs_227 = {}
        # Getting the type of 'random' (line 65)
        random_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'random', False)
        # Calling random(args, kwargs) (line 65)
        random_call_result_228 = invoke(stypy.reporting.localization.Localization(__file__, 65, 16), random_226, *[], **kwargs_227)
        
        float_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 27), 'float')
        # Applying the binary operator '*' (line 65)
        result_mul_230 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 16), '*', random_call_result_228, float_229)
        
        float_231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 33), 'float')
        # Applying the binary operator '-' (line 65)
        result_sub_232 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 16), '-', result_mul_230, float_231)
        
        # Processing the call keyword arguments (line 63)
        kwargs_233 = {}
        # Getting the type of 'V3' (line 63)
        V3_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 13), 'V3', False)
        # Calling V3(args, kwargs) (line 63)
        V3_call_result_234 = invoke(stypy.reporting.localization.Localization(__file__, 63, 13), V3_211, *[result_sub_218, result_sub_225, result_sub_232], **kwargs_233)
        
        # Assigning a type to the variable 'v2' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'v2', V3_call_result_234)
        
        # Assigning a Call to a Name (line 66):
        
        # Call to dot(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'v2' (line 66)
        v2_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'v2', False)
        # Processing the call keyword arguments (line 66)
        kwargs_238 = {}
        # Getting the type of 'v2' (line 66)
        v2_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'v2', False)
        # Obtaining the member 'dot' of a type (line 66)
        dot_236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 17), v2_235, 'dot')
        # Calling dot(args, kwargs) (line 66)
        dot_call_result_239 = invoke(stypy.reporting.localization.Localization(__file__, 66, 17), dot_236, *[v2_237], **kwargs_238)
        
        # Assigning a type to the variable 'v2_dot' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'v2_dot', dot_call_result_239)
        
        # Getting the type of 'v2_dot' (line 67)
        v2_dot_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'v2_dot')
        float_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 21), 'float')
        # Applying the binary operator '<=' (line 67)
        result_le_242 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), '<=', v2_dot_240, float_241)
        
        # Testing if the type of an if condition is none (line 67)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 8), result_le_242):
            pass
        else:
            
            # Testing the type of an if condition (line 67)
            if_condition_243 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), result_le_242)
            # Assigning a type to the variable 'if_condition_243' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_243', if_condition_243)
            # SSA begins for if statement (line 67)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 67)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for while statement (line 62)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Call to a Name (line 72):
    
    # Call to divs(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Call to sqrt(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'v2_dot' (line 72)
    v2_dot_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'v2_dot', False)
    # Processing the call keyword arguments (line 72)
    kwargs_248 = {}
    # Getting the type of 'sqrt' (line 72)
    sqrt_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 72)
    sqrt_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 72, 17), sqrt_246, *[v2_dot_247], **kwargs_248)
    
    # Processing the call keyword arguments (line 72)
    kwargs_250 = {}
    # Getting the type of 'v2' (line 72)
    v2_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 9), 'v2', False)
    # Obtaining the member 'divs' of a type (line 72)
    divs_245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 9), v2_244, 'divs')
    # Calling divs(args, kwargs) (line 72)
    divs_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 72, 9), divs_245, *[sqrt_call_result_249], **kwargs_250)
    
    # Assigning a type to the variable 'v2' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'v2', divs_call_result_251)
    
    
    # Call to dot(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'v' (line 75)
    v_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'v', False)
    # Processing the call keyword arguments (line 75)
    kwargs_255 = {}
    # Getting the type of 'v2' (line 75)
    v2_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 7), 'v2', False)
    # Obtaining the member 'dot' of a type (line 75)
    dot_253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 7), v2_252, 'dot')
    # Calling dot(args, kwargs) (line 75)
    dot_call_result_256 = invoke(stypy.reporting.localization.Localization(__file__, 75, 7), dot_253, *[v_254], **kwargs_255)
    
    float_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 19), 'float')
    # Applying the binary operator '<' (line 75)
    result_lt_258 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 7), '<', dot_call_result_256, float_257)
    
    # Testing if the type of an if condition is none (line 75)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 4), result_lt_258):
        pass
    else:
        
        # Testing the type of an if condition (line 75)
        if_condition_259 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 4), result_lt_258)
        # Assigning a type to the variable 'if_condition_259' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'if_condition_259', if_condition_259)
        # SSA begins for if statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to muls(...): (line 76)
        # Processing the call arguments (line 76)
        int_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 23), 'int')
        # Processing the call keyword arguments (line 76)
        kwargs_263 = {}
        # Getting the type of 'v2' (line 76)
        v2_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'v2', False)
        # Obtaining the member 'muls' of a type (line 76)
        muls_261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 15), v2_260, 'muls')
        # Calling muls(args, kwargs) (line 76)
        muls_call_result_264 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), muls_261, *[int_262], **kwargs_263)
        
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', muls_call_result_264)
        # SSA join for if statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'v2' (line 77)
    v2_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'v2')
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type', v2_265)
    
    # ################# End of 'getRandomNormalInHemisphere(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'getRandomNormalInHemisphere' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_266)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'getRandomNormalInHemisphere'
    return stypy_return_type_266

# Assigning a type to the variable 'getRandomNormalInHemisphere' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'getRandomNormalInHemisphere', getRandomNormalInHemisphere)
# Declaration of the 'Ray' class

class Ray(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Ray.__init__', ['origin', 'direction'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['origin', 'direction'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 82):
        # Getting the type of 'origin' (line 82)
        origin_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 22), 'origin')
        # Getting the type of 'self' (line 82)
        self_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
        # Setting the type of the member 'origin' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_268, 'origin', origin_267)
        
        # Assigning a Name to a Attribute (line 83):
        # Getting the type of 'direction' (line 83)
        direction_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 25), 'direction')
        # Getting the type of 'self' (line 83)
        self_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self')
        # Setting the type of the member 'direction' of a type (line 83)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_270, 'direction', direction_269)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Ray' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'Ray', Ray)
# Declaration of the 'Camera' class

class Camera(object, ):
    str_271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, (-1)), 'str', "\n    The camera is defined by an eyepo(origin) and three corners\n    of the view plane (it's a rect in my case...)\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 92, 4, False)
        # Assigning a type to the variable 'self' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Camera.__init__', ['origin', 'topleft', 'topright', 'bottomleft'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['origin', 'topleft', 'topright', 'bottomleft'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 93):
        # Getting the type of 'origin' (line 93)
        origin_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'origin')
        # Getting the type of 'self' (line 93)
        self_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'self')
        # Setting the type of the member 'origin' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), self_273, 'origin', origin_272)
        
        # Assigning a Name to a Attribute (line 94):
        # Getting the type of 'topleft' (line 94)
        topleft_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 23), 'topleft')
        # Getting the type of 'self' (line 94)
        self_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member 'topleft' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_275, 'topleft', topleft_274)
        
        # Assigning a Name to a Attribute (line 95):
        # Getting the type of 'topleft' (line 95)
        topleft_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'topleft')
        # Getting the type of 'self' (line 95)
        self_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member 'topright' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_277, 'topright', topleft_276)
        
        # Assigning a Name to a Attribute (line 96):
        # Getting the type of 'bottomleft' (line 96)
        bottomleft_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'bottomleft')
        # Getting the type of 'self' (line 96)
        self_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self')
        # Setting the type of the member 'bottomleft' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_279, 'bottomleft', bottomleft_278)
        
        # Assigning a Call to a Attribute (line 98):
        
        # Call to sub(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'topleft' (line 98)
        topleft_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'topleft', False)
        # Processing the call keyword arguments (line 98)
        kwargs_283 = {}
        # Getting the type of 'topright' (line 98)
        topright_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 18), 'topright', False)
        # Obtaining the member 'sub' of a type (line 98)
        sub_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 18), topright_280, 'sub')
        # Calling sub(args, kwargs) (line 98)
        sub_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 98, 18), sub_281, *[topleft_282], **kwargs_283)
        
        # Getting the type of 'self' (line 98)
        self_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'self')
        # Setting the type of the member 'xd' of a type (line 98)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 8), self_285, 'xd', sub_call_result_284)
        
        # Assigning a Call to a Attribute (line 99):
        
        # Call to sub(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'topleft' (line 99)
        topleft_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 33), 'topleft', False)
        # Processing the call keyword arguments (line 99)
        kwargs_289 = {}
        # Getting the type of 'bottomleft' (line 99)
        bottomleft_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 18), 'bottomleft', False)
        # Obtaining the member 'sub' of a type (line 99)
        sub_287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 18), bottomleft_286, 'sub')
        # Calling sub(args, kwargs) (line 99)
        sub_call_result_290 = invoke(stypy.reporting.localization.Localization(__file__, 99, 18), sub_287, *[topleft_288], **kwargs_289)
        
        # Getting the type of 'self' (line 99)
        self_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'self')
        # Setting the type of the member 'yd' of a type (line 99)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 8), self_291, 'yd', sub_call_result_290)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def getRay(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getRay'
        module_type_store = module_type_store.open_function_context('getRay', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Camera.getRay.__dict__.__setitem__('stypy_localization', localization)
        Camera.getRay.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Camera.getRay.__dict__.__setitem__('stypy_type_store', module_type_store)
        Camera.getRay.__dict__.__setitem__('stypy_function_name', 'Camera.getRay')
        Camera.getRay.__dict__.__setitem__('stypy_param_names_list', ['x', 'y'])
        Camera.getRay.__dict__.__setitem__('stypy_varargs_param_name', None)
        Camera.getRay.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Camera.getRay.__dict__.__setitem__('stypy_call_defaults', defaults)
        Camera.getRay.__dict__.__setitem__('stypy_call_varargs', varargs)
        Camera.getRay.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Camera.getRay.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Camera.getRay', ['x', 'y'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getRay', localization, ['x', 'y'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getRay(...)' code ##################

        
        # Assigning a Call to a Name (line 103):
        
        # Call to add(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to muls(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'y' (line 103)
        y_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 63), 'y', False)
        # Processing the call keyword arguments (line 103)
        kwargs_308 = {}
        # Getting the type of 'self' (line 103)
        self_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 50), 'self', False)
        # Obtaining the member 'yd' of a type (line 103)
        yd_305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 50), self_304, 'yd')
        # Obtaining the member 'muls' of a type (line 103)
        muls_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 50), yd_305, 'muls')
        # Calling muls(args, kwargs) (line 103)
        muls_call_result_309 = invoke(stypy.reporting.localization.Localization(__file__, 103, 50), muls_306, *[y_307], **kwargs_308)
        
        # Processing the call keyword arguments (line 103)
        kwargs_310 = {}
        
        # Call to add(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to muls(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'x' (line 103)
        x_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 42), 'x', False)
        # Processing the call keyword arguments (line 103)
        kwargs_299 = {}
        # Getting the type of 'self' (line 103)
        self_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 29), 'self', False)
        # Obtaining the member 'xd' of a type (line 103)
        xd_296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 29), self_295, 'xd')
        # Obtaining the member 'muls' of a type (line 103)
        muls_297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 29), xd_296, 'muls')
        # Calling muls(args, kwargs) (line 103)
        muls_call_result_300 = invoke(stypy.reporting.localization.Localization(__file__, 103, 29), muls_297, *[x_298], **kwargs_299)
        
        # Processing the call keyword arguments (line 103)
        kwargs_301 = {}
        # Getting the type of 'self' (line 103)
        self_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self', False)
        # Obtaining the member 'topleft' of a type (line 103)
        topleft_293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_292, 'topleft')
        # Obtaining the member 'add' of a type (line 103)
        add_294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), topleft_293, 'add')
        # Calling add(args, kwargs) (line 103)
        add_call_result_302 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), add_294, *[muls_call_result_300], **kwargs_301)
        
        # Obtaining the member 'add' of a type (line 103)
        add_303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), add_call_result_302, 'add')
        # Calling add(args, kwargs) (line 103)
        add_call_result_311 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), add_303, *[muls_call_result_309], **kwargs_310)
        
        # Assigning a type to the variable 'p' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'p', add_call_result_311)
        
        # Call to Ray(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'self' (line 104)
        self_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'self', False)
        # Obtaining the member 'origin' of a type (line 104)
        origin_314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 19), self_313, 'origin')
        
        # Call to normalize(...): (line 104)
        # Processing the call keyword arguments (line 104)
        kwargs_322 = {}
        
        # Call to sub(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'self' (line 104)
        self_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 38), 'self', False)
        # Obtaining the member 'origin' of a type (line 104)
        origin_318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 38), self_317, 'origin')
        # Processing the call keyword arguments (line 104)
        kwargs_319 = {}
        # Getting the type of 'p' (line 104)
        p_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 32), 'p', False)
        # Obtaining the member 'sub' of a type (line 104)
        sub_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 32), p_315, 'sub')
        # Calling sub(args, kwargs) (line 104)
        sub_call_result_320 = invoke(stypy.reporting.localization.Localization(__file__, 104, 32), sub_316, *[origin_318], **kwargs_319)
        
        # Obtaining the member 'normalize' of a type (line 104)
        normalize_321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 32), sub_call_result_320, 'normalize')
        # Calling normalize(args, kwargs) (line 104)
        normalize_call_result_323 = invoke(stypy.reporting.localization.Localization(__file__, 104, 32), normalize_321, *[], **kwargs_322)
        
        # Processing the call keyword arguments (line 104)
        kwargs_324 = {}
        # Getting the type of 'Ray' (line 104)
        Ray_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'Ray', False)
        # Calling Ray(args, kwargs) (line 104)
        Ray_call_result_325 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), Ray_312, *[origin_314, normalize_call_result_323], **kwargs_324)
        
        # Assigning a type to the variable 'stypy_return_type' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'stypy_return_type', Ray_call_result_325)
        
        # ################# End of 'getRay(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getRay' in the type store
        # Getting the type of 'stypy_return_type' (line 101)
        stypy_return_type_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_326)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getRay'
        return stypy_return_type_326


# Assigning a type to the variable 'Camera' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'Camera', Camera)
# Declaration of the 'Sphere' class

class Sphere(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sphere.__init__', ['center', 'radius'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['center', 'radius'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 109):
        # Getting the type of 'center' (line 109)
        center_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 22), 'center')
        # Getting the type of 'self' (line 109)
        self_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self')
        # Setting the type of the member 'center' of a type (line 109)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_328, 'center', center_327)
        
        # Assigning a Name to a Attribute (line 110):
        # Getting the type of 'radius' (line 110)
        radius_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'radius')
        # Getting the type of 'self' (line 110)
        self_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'self')
        # Setting the type of the member 'radius' of a type (line 110)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), self_330, 'radius', radius_329)
        
        # Assigning a BinOp to a Attribute (line 111):
        # Getting the type of 'radius' (line 111)
        radius_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'radius')
        # Getting the type of 'radius' (line 111)
        radius_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 32), 'radius')
        # Applying the binary operator '*' (line 111)
        result_mul_333 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 23), '*', radius_331, radius_332)
        
        # Getting the type of 'self' (line 111)
        self_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self')
        # Setting the type of the member 'radius2' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_334, 'radius2', result_mul_333)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def intersect(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'intersect'
        module_type_store = module_type_store.open_function_context('intersect', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Sphere.intersect.__dict__.__setitem__('stypy_localization', localization)
        Sphere.intersect.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Sphere.intersect.__dict__.__setitem__('stypy_type_store', module_type_store)
        Sphere.intersect.__dict__.__setitem__('stypy_function_name', 'Sphere.intersect')
        Sphere.intersect.__dict__.__setitem__('stypy_param_names_list', ['ray'])
        Sphere.intersect.__dict__.__setitem__('stypy_varargs_param_name', None)
        Sphere.intersect.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Sphere.intersect.__dict__.__setitem__('stypy_call_defaults', defaults)
        Sphere.intersect.__dict__.__setitem__('stypy_call_varargs', varargs)
        Sphere.intersect.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Sphere.intersect.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sphere.intersect', ['ray'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'intersect', localization, ['ray'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'intersect(...)' code ##################

        
        # Assigning a Call to a Name (line 115):
        
        # Call to subdot(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'self' (line 115)
        self_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 30), 'self', False)
        # Obtaining the member 'center' of a type (line 115)
        center_339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 30), self_338, 'center')
        # Getting the type of 'ray' (line 115)
        ray_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 43), 'ray', False)
        # Obtaining the member 'direction' of a type (line 115)
        direction_341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 43), ray_340, 'direction')
        # Processing the call keyword arguments (line 115)
        kwargs_342 = {}
        # Getting the type of 'ray' (line 115)
        ray_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'ray', False)
        # Obtaining the member 'origin' of a type (line 115)
        origin_336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), ray_335, 'origin')
        # Obtaining the member 'subdot' of a type (line 115)
        subdot_337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), origin_336, 'subdot')
        # Calling subdot(args, kwargs) (line 115)
        subdot_call_result_343 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), subdot_337, *[center_339, direction_341], **kwargs_342)
        
        # Assigning a type to the variable 'b' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'b', subdot_call_result_343)
        
        # Assigning a BinOp to a Name (line 116):
        
        # Call to subdot2(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'self' (line 116)
        self_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'self', False)
        # Obtaining the member 'center' of a type (line 116)
        center_348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 31), self_347, 'center')
        # Processing the call keyword arguments (line 116)
        kwargs_349 = {}
        # Getting the type of 'ray' (line 116)
        ray_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'ray', False)
        # Obtaining the member 'origin' of a type (line 116)
        origin_345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), ray_344, 'origin')
        # Obtaining the member 'subdot2' of a type (line 116)
        subdot2_346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 12), origin_345, 'subdot2')
        # Calling subdot2(args, kwargs) (line 116)
        subdot2_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 116, 12), subdot2_346, *[center_348], **kwargs_349)
        
        # Getting the type of 'self' (line 116)
        self_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 46), 'self')
        # Obtaining the member 'radius2' of a type (line 116)
        radius2_352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 46), self_351, 'radius2')
        # Applying the binary operator '-' (line 116)
        result_sub_353 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 12), '-', subdot2_call_result_350, radius2_352)
        
        # Assigning a type to the variable 'c' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'c', result_sub_353)
        
        # Assigning a BinOp to a Name (line 117):
        # Getting the type of 'b' (line 117)
        b_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'b')
        # Getting the type of 'b' (line 117)
        b_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'b')
        # Applying the binary operator '*' (line 117)
        result_mul_356 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 12), '*', b_354, b_355)
        
        # Getting the type of 'c' (line 117)
        c_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 20), 'c')
        # Applying the binary operator '-' (line 117)
        result_sub_358 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 12), '-', result_mul_356, c_357)
        
        # Assigning a type to the variable 'd' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'd', result_sub_358)
        
        
        # Getting the type of 'd' (line 118)
        d_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), 'd')
        int_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 37), 'int')
        # Applying the binary operator '>' (line 118)
        result_gt_361 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 33), '>', d_359, int_360)
        
        # Testing the type of an if expression (line 118)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 15), result_gt_361)
        # SSA begins for if expression (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Getting the type of 'b' (line 118)
        b_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 17), 'b')
        # Applying the 'usub' unary operator (line 118)
        result___neg___363 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 16), 'usub', b_362)
        
        
        # Call to sqrt(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'd' (line 118)
        d_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'd', False)
        # Processing the call keyword arguments (line 118)
        kwargs_366 = {}
        # Getting the type of 'sqrt' (line 118)
        sqrt_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 21), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 118)
        sqrt_call_result_367 = invoke(stypy.reporting.localization.Localization(__file__, 118, 21), sqrt_364, *[d_365], **kwargs_366)
        
        # Applying the binary operator '-' (line 118)
        result_sub_368 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 16), '-', result___neg___363, sqrt_call_result_367)
        
        # SSA branch for the else part of an if expression (line 118)
        module_type_store.open_ssa_branch('if expression else')
        float_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 44), 'float')
        # SSA join for if expression (line 118)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_370 = union_type.UnionType.add(result_sub_368, float_369)
        
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', if_exp_370)
        
        # ################# End of 'intersect(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'intersect' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_371)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'intersect'
        return stypy_return_type_371


    @norecursion
    def getNormal(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getNormal'
        module_type_store = module_type_store.open_function_context('getNormal', 120, 4, False)
        # Assigning a type to the variable 'self' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Sphere.getNormal.__dict__.__setitem__('stypy_localization', localization)
        Sphere.getNormal.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Sphere.getNormal.__dict__.__setitem__('stypy_type_store', module_type_store)
        Sphere.getNormal.__dict__.__setitem__('stypy_function_name', 'Sphere.getNormal')
        Sphere.getNormal.__dict__.__setitem__('stypy_param_names_list', ['point'])
        Sphere.getNormal.__dict__.__setitem__('stypy_varargs_param_name', None)
        Sphere.getNormal.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Sphere.getNormal.__dict__.__setitem__('stypy_call_defaults', defaults)
        Sphere.getNormal.__dict__.__setitem__('stypy_call_varargs', varargs)
        Sphere.getNormal.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Sphere.getNormal.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Sphere.getNormal', ['point'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getNormal', localization, ['point'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getNormal(...)' code ##################

        
        # Call to normalize(...): (line 121)
        # Processing the call keyword arguments (line 121)
        kwargs_379 = {}
        
        # Call to sub(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'self' (line 121)
        self_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 25), 'self', False)
        # Obtaining the member 'center' of a type (line 121)
        center_375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 25), self_374, 'center')
        # Processing the call keyword arguments (line 121)
        kwargs_376 = {}
        # Getting the type of 'point' (line 121)
        point_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'point', False)
        # Obtaining the member 'sub' of a type (line 121)
        sub_373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), point_372, 'sub')
        # Calling sub(args, kwargs) (line 121)
        sub_call_result_377 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), sub_373, *[center_375], **kwargs_376)
        
        # Obtaining the member 'normalize' of a type (line 121)
        normalize_378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), sub_call_result_377, 'normalize')
        # Calling normalize(args, kwargs) (line 121)
        normalize_call_result_380 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), normalize_378, *[], **kwargs_379)
        
        # Assigning a type to the variable 'stypy_return_type' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'stypy_return_type', normalize_call_result_380)
        
        # ################# End of 'getNormal(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getNormal' in the type store
        # Getting the type of 'stypy_return_type' (line 120)
        stypy_return_type_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_381)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getNormal'
        return stypy_return_type_381


# Assigning a type to the variable 'Sphere' (line 107)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'Sphere', Sphere)
# Declaration of the 'Material' class

class Material(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 125)
        None_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 39), 'None')
        defaults = [None_382]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Material.__init__', ['color', 'emission'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['color', 'emission'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 126):
        # Getting the type of 'color' (line 126)
        color_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 21), 'color')
        # Getting the type of 'self' (line 126)
        self_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'self')
        # Setting the type of the member 'color' of a type (line 126)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), self_384, 'color', color_383)
        
        # Assigning a IfExp to a Attribute (line 127):
        
        
        # Getting the type of 'emission' (line 127)
        emission_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 45), 'emission')
        # Getting the type of 'None' (line 127)
        None_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 57), 'None')
        # Applying the binary operator 'is' (line 127)
        result_is__387 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 45), 'is', emission_385, None_386)
        
        # Testing the type of an if expression (line 127)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 24), result_is__387)
        # SSA begins for if expression (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        
        # Call to V3(...): (line 127)
        # Processing the call arguments (line 127)
        float_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 27), 'float')
        float_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 32), 'float')
        float_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 37), 'float')
        # Processing the call keyword arguments (line 127)
        kwargs_392 = {}
        # Getting the type of 'V3' (line 127)
        V3_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'V3', False)
        # Calling V3(args, kwargs) (line 127)
        V3_call_result_393 = invoke(stypy.reporting.localization.Localization(__file__, 127, 24), V3_388, *[float_389, float_390, float_391], **kwargs_392)
        
        # SSA branch for the else part of an if expression (line 127)
        module_type_store.open_ssa_branch('if expression else')
        # Getting the type of 'emission' (line 127)
        emission_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 67), 'emission')
        # SSA join for if expression (line 127)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_395 = union_type.UnionType.add(V3_call_result_393, emission_394)
        
        # Getting the type of 'self' (line 127)
        self_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'self')
        # Setting the type of the member 'emission' of a type (line 127)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 8), self_396, 'emission', if_exp_395)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def bounce(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'bounce'
        module_type_store = module_type_store.open_function_context('bounce', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Material.bounce.__dict__.__setitem__('stypy_localization', localization)
        Material.bounce.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Material.bounce.__dict__.__setitem__('stypy_type_store', module_type_store)
        Material.bounce.__dict__.__setitem__('stypy_function_name', 'Material.bounce')
        Material.bounce.__dict__.__setitem__('stypy_param_names_list', ['ray', 'normal'])
        Material.bounce.__dict__.__setitem__('stypy_varargs_param_name', None)
        Material.bounce.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Material.bounce.__dict__.__setitem__('stypy_call_defaults', defaults)
        Material.bounce.__dict__.__setitem__('stypy_call_varargs', varargs)
        Material.bounce.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Material.bounce.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Material.bounce', ['ray', 'normal'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bounce', localization, ['ray', 'normal'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bounce(...)' code ##################

        
        # Call to getRandomNormalInHemisphere(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'normal' (line 130)
        normal_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 43), 'normal', False)
        # Processing the call keyword arguments (line 130)
        kwargs_399 = {}
        # Getting the type of 'getRandomNormalInHemisphere' (line 130)
        getRandomNormalInHemisphere_397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'getRandomNormalInHemisphere', False)
        # Calling getRandomNormalInHemisphere(args, kwargs) (line 130)
        getRandomNormalInHemisphere_call_result_400 = invoke(stypy.reporting.localization.Localization(__file__, 130, 15), getRandomNormalInHemisphere_397, *[normal_398], **kwargs_399)
        
        # Assigning a type to the variable 'stypy_return_type' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'stypy_return_type', getRandomNormalInHemisphere_call_result_400)
        
        # ################# End of 'bounce(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bounce' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_401)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bounce'
        return stypy_return_type_401


# Assigning a type to the variable 'Material' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'Material', Material)
# Declaration of the 'Chrome' class
# Getting the type of 'Material' (line 133)
Material_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 13), 'Material')

class Chrome(Material_402, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 134, 4, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chrome.__init__', ['color'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['color'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'color' (line 135)
        color_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 37), 'color', False)
        # Processing the call keyword arguments (line 135)
        kwargs_410 = {}
        
        # Call to super(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'Chrome' (line 135)
        Chrome_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 14), 'Chrome', False)
        # Getting the type of 'self' (line 135)
        self_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'self', False)
        # Processing the call keyword arguments (line 135)
        kwargs_406 = {}
        # Getting the type of 'super' (line 135)
        super_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'super', False)
        # Calling super(args, kwargs) (line 135)
        super_call_result_407 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), super_403, *[Chrome_404, self_405], **kwargs_406)
        
        # Obtaining the member '__init__' of a type (line 135)
        init___408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 8), super_call_result_407, '__init__')
        # Calling __init__(args, kwargs) (line 135)
        init___call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), init___408, *[color_409], **kwargs_410)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def bounce(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'bounce'
        module_type_store = module_type_store.open_function_context('bounce', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Chrome.bounce.__dict__.__setitem__('stypy_localization', localization)
        Chrome.bounce.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Chrome.bounce.__dict__.__setitem__('stypy_type_store', module_type_store)
        Chrome.bounce.__dict__.__setitem__('stypy_function_name', 'Chrome.bounce')
        Chrome.bounce.__dict__.__setitem__('stypy_param_names_list', ['ray', 'normal'])
        Chrome.bounce.__dict__.__setitem__('stypy_varargs_param_name', None)
        Chrome.bounce.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Chrome.bounce.__dict__.__setitem__('stypy_call_defaults', defaults)
        Chrome.bounce.__dict__.__setitem__('stypy_call_varargs', varargs)
        Chrome.bounce.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Chrome.bounce.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Chrome.bounce', ['ray', 'normal'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bounce', localization, ['ray', 'normal'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bounce(...)' code ##################

        
        # Assigning a Call to a Name (line 138):
        
        # Call to abs(...): (line 138)
        # Processing the call arguments (line 138)
        
        # Call to dot(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'normal' (line 138)
        normal_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 39), 'normal', False)
        # Processing the call keyword arguments (line 138)
        kwargs_417 = {}
        # Getting the type of 'ray' (line 138)
        ray_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'ray', False)
        # Obtaining the member 'direction' of a type (line 138)
        direction_414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 21), ray_413, 'direction')
        # Obtaining the member 'dot' of a type (line 138)
        dot_415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 21), direction_414, 'dot')
        # Calling dot(args, kwargs) (line 138)
        dot_call_result_418 = invoke(stypy.reporting.localization.Localization(__file__, 138, 21), dot_415, *[normal_416], **kwargs_417)
        
        # Processing the call keyword arguments (line 138)
        kwargs_419 = {}
        # Getting the type of 'abs' (line 138)
        abs_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 17), 'abs', False)
        # Calling abs(args, kwargs) (line 138)
        abs_call_result_420 = invoke(stypy.reporting.localization.Localization(__file__, 138, 17), abs_412, *[dot_call_result_418], **kwargs_419)
        
        # Assigning a type to the variable 'theta1' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'theta1', abs_call_result_420)
        
        # Call to add(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Call to muls(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'theta1' (line 139)
        theta1_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 45), 'theta1', False)
        float_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 54), 'float')
        # Applying the binary operator '*' (line 139)
        result_mul_428 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 45), '*', theta1_426, float_427)
        
        # Processing the call keyword arguments (line 139)
        kwargs_429 = {}
        # Getting the type of 'normal' (line 139)
        normal_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 33), 'normal', False)
        # Obtaining the member 'muls' of a type (line 139)
        muls_425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 33), normal_424, 'muls')
        # Calling muls(args, kwargs) (line 139)
        muls_call_result_430 = invoke(stypy.reporting.localization.Localization(__file__, 139, 33), muls_425, *[result_mul_428], **kwargs_429)
        
        # Processing the call keyword arguments (line 139)
        kwargs_431 = {}
        # Getting the type of 'ray' (line 139)
        ray_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'ray', False)
        # Obtaining the member 'direction' of a type (line 139)
        direction_422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 15), ray_421, 'direction')
        # Obtaining the member 'add' of a type (line 139)
        add_423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 15), direction_422, 'add')
        # Calling add(args, kwargs) (line 139)
        add_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), add_423, *[muls_call_result_430], **kwargs_431)
        
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'stypy_return_type', add_call_result_432)
        
        # ################# End of 'bounce(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bounce' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_433)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bounce'
        return stypy_return_type_433


# Assigning a type to the variable 'Chrome' (line 133)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'Chrome', Chrome)
# Declaration of the 'Glass' class
# Getting the type of 'Material' (line 142)
Material_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'Material')

class Glass(Material_434, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Glass.__init__', ['color', 'ior', 'reflection'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['color', 'ior', 'reflection'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'color' (line 144)
        color_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 36), 'color', False)
        # Processing the call keyword arguments (line 144)
        kwargs_442 = {}
        
        # Call to super(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'Glass' (line 144)
        Glass_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 14), 'Glass', False)
        # Getting the type of 'self' (line 144)
        self_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 21), 'self', False)
        # Processing the call keyword arguments (line 144)
        kwargs_438 = {}
        # Getting the type of 'super' (line 144)
        super_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'super', False)
        # Calling super(args, kwargs) (line 144)
        super_call_result_439 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), super_435, *[Glass_436, self_437], **kwargs_438)
        
        # Obtaining the member '__init__' of a type (line 144)
        init___440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 8), super_call_result_439, '__init__')
        # Calling __init__(args, kwargs) (line 144)
        init___call_result_443 = invoke(stypy.reporting.localization.Localization(__file__, 144, 8), init___440, *[color_441], **kwargs_442)
        
        
        # Assigning a Name to a Attribute (line 145):
        # Getting the type of 'ior' (line 145)
        ior_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'ior')
        # Getting the type of 'self' (line 145)
        self_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'self')
        # Setting the type of the member 'ior' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), self_445, 'ior', ior_444)
        
        # Assigning a Name to a Attribute (line 146):
        # Getting the type of 'reflection' (line 146)
        reflection_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'reflection')
        # Getting the type of 'self' (line 146)
        self_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'self')
        # Setting the type of the member 'reflection' of a type (line 146)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 8), self_447, 'reflection', reflection_446)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def bounce(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'bounce'
        module_type_store = module_type_store.open_function_context('bounce', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Glass.bounce.__dict__.__setitem__('stypy_localization', localization)
        Glass.bounce.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Glass.bounce.__dict__.__setitem__('stypy_type_store', module_type_store)
        Glass.bounce.__dict__.__setitem__('stypy_function_name', 'Glass.bounce')
        Glass.bounce.__dict__.__setitem__('stypy_param_names_list', ['ray', 'normal'])
        Glass.bounce.__dict__.__setitem__('stypy_varargs_param_name', None)
        Glass.bounce.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Glass.bounce.__dict__.__setitem__('stypy_call_defaults', defaults)
        Glass.bounce.__dict__.__setitem__('stypy_call_varargs', varargs)
        Glass.bounce.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Glass.bounce.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Glass.bounce', ['ray', 'normal'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bounce', localization, ['ray', 'normal'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bounce(...)' code ##################

        
        # Assigning a Call to a Name (line 149):
        
        # Call to abs(...): (line 149)
        # Processing the call arguments (line 149)
        
        # Call to dot(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'normal' (line 149)
        normal_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 39), 'normal', False)
        # Processing the call keyword arguments (line 149)
        kwargs_453 = {}
        # Getting the type of 'ray' (line 149)
        ray_449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 21), 'ray', False)
        # Obtaining the member 'direction' of a type (line 149)
        direction_450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 21), ray_449, 'direction')
        # Obtaining the member 'dot' of a type (line 149)
        dot_451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 21), direction_450, 'dot')
        # Calling dot(args, kwargs) (line 149)
        dot_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 149, 21), dot_451, *[normal_452], **kwargs_453)
        
        # Processing the call keyword arguments (line 149)
        kwargs_455 = {}
        # Getting the type of 'abs' (line 149)
        abs_448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'abs', False)
        # Calling abs(args, kwargs) (line 149)
        abs_call_result_456 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), abs_448, *[dot_call_result_454], **kwargs_455)
        
        # Assigning a type to the variable 'theta1' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'theta1', abs_call_result_456)
        
        # Getting the type of 'theta1' (line 150)
        theta1_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'theta1')
        float_458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 21), 'float')
        # Applying the binary operator '>=' (line 150)
        result_ge_459 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 11), '>=', theta1_457, float_458)
        
        # Testing if the type of an if condition is none (line 150)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 150, 8), result_ge_459):
            
            # Assigning a Num to a Name (line 154):
            float_464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 28), 'float')
            # Assigning a type to the variable 'internalIndex' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'internalIndex', float_464)
            
            # Assigning a Attribute to a Name (line 155):
            # Getting the type of 'self' (line 155)
            self_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'self')
            # Obtaining the member 'ior' of a type (line 155)
            ior_466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 28), self_465, 'ior')
            # Assigning a type to the variable 'externalIndex' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'externalIndex', ior_466)
        else:
            
            # Testing the type of an if condition (line 150)
            if_condition_460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), result_ge_459)
            # Assigning a type to the variable 'if_condition_460' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_460', if_condition_460)
            # SSA begins for if statement (line 150)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 151):
            # Getting the type of 'self' (line 151)
            self_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 28), 'self')
            # Obtaining the member 'ior' of a type (line 151)
            ior_462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 28), self_461, 'ior')
            # Assigning a type to the variable 'internalIndex' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'internalIndex', ior_462)
            
            # Assigning a Num to a Name (line 152):
            float_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'float')
            # Assigning a type to the variable 'externalIndex' (line 152)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'externalIndex', float_463)
            # SSA branch for the else part of an if statement (line 150)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Num to a Name (line 154):
            float_464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 28), 'float')
            # Assigning a type to the variable 'internalIndex' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'internalIndex', float_464)
            
            # Assigning a Attribute to a Name (line 155):
            # Getting the type of 'self' (line 155)
            self_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 28), 'self')
            # Obtaining the member 'ior' of a type (line 155)
            ior_466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 28), self_465, 'ior')
            # Assigning a type to the variable 'externalIndex' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'externalIndex', ior_466)
            # SSA join for if statement (line 150)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 156):
        # Getting the type of 'externalIndex' (line 156)
        externalIndex_467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 14), 'externalIndex')
        # Getting the type of 'internalIndex' (line 156)
        internalIndex_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 30), 'internalIndex')
        # Applying the binary operator 'div' (line 156)
        result_div_469 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 14), 'div', externalIndex_467, internalIndex_468)
        
        # Assigning a type to the variable 'eta' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'eta', result_div_469)
        
        # Assigning a Call to a Name (line 157):
        
        # Call to sqrt(...): (line 157)
        # Processing the call arguments (line 157)
        float_471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 22), 'float')
        # Getting the type of 'eta' (line 157)
        eta_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 29), 'eta', False)
        # Getting the type of 'eta' (line 157)
        eta_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 35), 'eta', False)
        # Applying the binary operator '*' (line 157)
        result_mul_474 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 29), '*', eta_472, eta_473)
        
        float_475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 43), 'float')
        # Getting the type of 'theta1' (line 157)
        theta1_476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 50), 'theta1', False)
        # Getting the type of 'theta1' (line 157)
        theta1_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 59), 'theta1', False)
        # Applying the binary operator '*' (line 157)
        result_mul_478 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 50), '*', theta1_476, theta1_477)
        
        # Applying the binary operator '-' (line 157)
        result_sub_479 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 43), '-', float_475, result_mul_478)
        
        # Applying the binary operator '*' (line 157)
        result_mul_480 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 28), '*', result_mul_474, result_sub_479)
        
        # Applying the binary operator '-' (line 157)
        result_sub_481 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 22), '-', float_471, result_mul_480)
        
        # Processing the call keyword arguments (line 157)
        kwargs_482 = {}
        # Getting the type of 'sqrt' (line 157)
        sqrt_470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 17), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 157)
        sqrt_call_result_483 = invoke(stypy.reporting.localization.Localization(__file__, 157, 17), sqrt_470, *[result_sub_481], **kwargs_482)
        
        # Assigning a type to the variable 'theta2' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'theta2', sqrt_call_result_483)
        
        # Assigning a BinOp to a Name (line 158):
        # Getting the type of 'externalIndex' (line 158)
        externalIndex_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 14), 'externalIndex')
        # Getting the type of 'theta1' (line 158)
        theta1_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 30), 'theta1')
        # Applying the binary operator '*' (line 158)
        result_mul_486 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 14), '*', externalIndex_484, theta1_485)
        
        # Getting the type of 'internalIndex' (line 158)
        internalIndex_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 39), 'internalIndex')
        # Getting the type of 'theta2' (line 158)
        theta2_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 55), 'theta2')
        # Applying the binary operator '*' (line 158)
        result_mul_489 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 39), '*', internalIndex_487, theta2_488)
        
        # Applying the binary operator '-' (line 158)
        result_sub_490 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 14), '-', result_mul_486, result_mul_489)
        
        # Getting the type of 'externalIndex' (line 158)
        externalIndex_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 66), 'externalIndex')
        # Getting the type of 'theta1' (line 158)
        theta1_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 82), 'theta1')
        # Applying the binary operator '*' (line 158)
        result_mul_493 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 66), '*', externalIndex_491, theta1_492)
        
        # Getting the type of 'internalIndex' (line 158)
        internalIndex_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 91), 'internalIndex')
        # Getting the type of 'theta2' (line 158)
        theta2_495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 107), 'theta2')
        # Applying the binary operator '*' (line 158)
        result_mul_496 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 91), '*', internalIndex_494, theta2_495)
        
        # Applying the binary operator '+' (line 158)
        result_add_497 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 66), '+', result_mul_493, result_mul_496)
        
        # Applying the binary operator 'div' (line 158)
        result_div_498 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 13), 'div', result_sub_490, result_add_497)
        
        # Assigning a type to the variable 'rs' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'rs', result_div_498)
        
        # Assigning a BinOp to a Name (line 159):
        # Getting the type of 'internalIndex' (line 159)
        internalIndex_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 14), 'internalIndex')
        # Getting the type of 'theta1' (line 159)
        theta1_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 30), 'theta1')
        # Applying the binary operator '*' (line 159)
        result_mul_501 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 14), '*', internalIndex_499, theta1_500)
        
        # Getting the type of 'externalIndex' (line 159)
        externalIndex_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 39), 'externalIndex')
        # Getting the type of 'theta2' (line 159)
        theta2_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 55), 'theta2')
        # Applying the binary operator '*' (line 159)
        result_mul_504 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 39), '*', externalIndex_502, theta2_503)
        
        # Applying the binary operator '-' (line 159)
        result_sub_505 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 14), '-', result_mul_501, result_mul_504)
        
        # Getting the type of 'internalIndex' (line 159)
        internalIndex_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 66), 'internalIndex')
        # Getting the type of 'theta1' (line 159)
        theta1_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 82), 'theta1')
        # Applying the binary operator '*' (line 159)
        result_mul_508 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 66), '*', internalIndex_506, theta1_507)
        
        # Getting the type of 'externalIndex' (line 159)
        externalIndex_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 91), 'externalIndex')
        # Getting the type of 'theta2' (line 159)
        theta2_510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 107), 'theta2')
        # Applying the binary operator '*' (line 159)
        result_mul_511 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 91), '*', externalIndex_509, theta2_510)
        
        # Applying the binary operator '+' (line 159)
        result_add_512 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 66), '+', result_mul_508, result_mul_511)
        
        # Applying the binary operator 'div' (line 159)
        result_div_513 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 13), 'div', result_sub_505, result_add_512)
        
        # Assigning a type to the variable 'rp' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'rp', result_div_513)
        
        # Assigning a BinOp to a Name (line 160):
        # Getting the type of 'rs' (line 160)
        rs_514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 23), 'rs')
        # Getting the type of 'rs' (line 160)
        rs_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 28), 'rs')
        # Applying the binary operator '*' (line 160)
        result_mul_516 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 23), '*', rs_514, rs_515)
        
        # Getting the type of 'rp' (line 160)
        rp_517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 33), 'rp')
        # Getting the type of 'rp' (line 160)
        rp_518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 38), 'rp')
        # Applying the binary operator '*' (line 160)
        result_mul_519 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 33), '*', rp_517, rp_518)
        
        # Applying the binary operator '+' (line 160)
        result_add_520 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 23), '+', result_mul_516, result_mul_519)
        
        # Assigning a type to the variable 'reflectance' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'reflectance', result_add_520)
        
        
        # Call to random(...): (line 162)
        # Processing the call keyword arguments (line 162)
        kwargs_522 = {}
        # Getting the type of 'random' (line 162)
        random_521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'random', False)
        # Calling random(args, kwargs) (line 162)
        random_call_result_523 = invoke(stypy.reporting.localization.Localization(__file__, 162, 11), random_521, *[], **kwargs_522)
        
        # Getting the type of 'reflectance' (line 162)
        reflectance_524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 22), 'reflectance')
        # Getting the type of 'self' (line 162)
        self_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 36), 'self')
        # Obtaining the member 'reflection' of a type (line 162)
        reflection_526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 36), self_525, 'reflection')
        # Applying the binary operator '+' (line 162)
        result_add_527 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 22), '+', reflectance_524, reflection_526)
        
        # Applying the binary operator '<' (line 162)
        result_lt_528 = python_operator(stypy.reporting.localization.Localization(__file__, 162, 11), '<', random_call_result_523, result_add_527)
        
        # Testing if the type of an if condition is none (line 162)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 162, 8), result_lt_528):
            pass
        else:
            
            # Testing the type of an if condition (line 162)
            if_condition_529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 8), result_lt_528)
            # Assigning a type to the variable 'if_condition_529' (line 162)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'if_condition_529', if_condition_529)
            # SSA begins for if statement (line 162)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to add(...): (line 163)
            # Processing the call arguments (line 163)
            
            # Call to muls(...): (line 163)
            # Processing the call arguments (line 163)
            # Getting the type of 'theta1' (line 163)
            theta1_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 49), 'theta1', False)
            float_536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 58), 'float')
            # Applying the binary operator '*' (line 163)
            result_mul_537 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 49), '*', theta1_535, float_536)
            
            # Processing the call keyword arguments (line 163)
            kwargs_538 = {}
            # Getting the type of 'normal' (line 163)
            normal_533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 37), 'normal', False)
            # Obtaining the member 'muls' of a type (line 163)
            muls_534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 37), normal_533, 'muls')
            # Calling muls(args, kwargs) (line 163)
            muls_call_result_539 = invoke(stypy.reporting.localization.Localization(__file__, 163, 37), muls_534, *[result_mul_537], **kwargs_538)
            
            # Processing the call keyword arguments (line 163)
            kwargs_540 = {}
            # Getting the type of 'ray' (line 163)
            ray_530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'ray', False)
            # Obtaining the member 'direction' of a type (line 163)
            direction_531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), ray_530, 'direction')
            # Obtaining the member 'add' of a type (line 163)
            add_532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), direction_531, 'add')
            # Calling add(args, kwargs) (line 163)
            add_call_result_541 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), add_532, *[muls_call_result_539], **kwargs_540)
            
            # Assigning a type to the variable 'stypy_return_type' (line 163)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'stypy_return_type', add_call_result_541)
            # SSA join for if statement (line 162)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to add(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Call to muls(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Getting the type of 'theta2' (line 165)
        theta2_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 82), 'theta2', False)
        # Applying the 'usub' unary operator (line 165)
        result___neg___560 = python_operator(stypy.reporting.localization.Localization(__file__, 165, 81), 'usub', theta2_559)
        
        # Processing the call keyword arguments (line 165)
        kwargs_561 = {}
        # Getting the type of 'normal' (line 165)
        normal_557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 69), 'normal', False)
        # Obtaining the member 'muls' of a type (line 165)
        muls_558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 69), normal_557, 'muls')
        # Calling muls(args, kwargs) (line 165)
        muls_call_result_562 = invoke(stypy.reporting.localization.Localization(__file__, 165, 69), muls_558, *[result___neg___560], **kwargs_561)
        
        # Processing the call keyword arguments (line 165)
        kwargs_563 = {}
        
        # Call to muls(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'eta' (line 165)
        eta_553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 60), 'eta', False)
        # Processing the call keyword arguments (line 165)
        kwargs_554 = {}
        
        # Call to add(...): (line 165)
        # Processing the call arguments (line 165)
        
        # Call to muls(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'theta1' (line 165)
        theta1_547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 46), 'theta1', False)
        # Processing the call keyword arguments (line 165)
        kwargs_548 = {}
        # Getting the type of 'normal' (line 165)
        normal_545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 34), 'normal', False)
        # Obtaining the member 'muls' of a type (line 165)
        muls_546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 34), normal_545, 'muls')
        # Calling muls(args, kwargs) (line 165)
        muls_call_result_549 = invoke(stypy.reporting.localization.Localization(__file__, 165, 34), muls_546, *[theta1_547], **kwargs_548)
        
        # Processing the call keyword arguments (line 165)
        kwargs_550 = {}
        # Getting the type of 'ray' (line 165)
        ray_542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'ray', False)
        # Obtaining the member 'direction' of a type (line 165)
        direction_543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), ray_542, 'direction')
        # Obtaining the member 'add' of a type (line 165)
        add_544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), direction_543, 'add')
        # Calling add(args, kwargs) (line 165)
        add_call_result_551 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), add_544, *[muls_call_result_549], **kwargs_550)
        
        # Obtaining the member 'muls' of a type (line 165)
        muls_552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), add_call_result_551, 'muls')
        # Calling muls(args, kwargs) (line 165)
        muls_call_result_555 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), muls_552, *[eta_553], **kwargs_554)
        
        # Obtaining the member 'add' of a type (line 165)
        add_556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 16), muls_call_result_555, 'add')
        # Calling add(args, kwargs) (line 165)
        add_call_result_564 = invoke(stypy.reporting.localization.Localization(__file__, 165, 16), add_556, *[muls_call_result_562], **kwargs_563)
        
        # Assigning a type to the variable 'stypy_return_type' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'stypy_return_type', add_call_result_564)
        
        # ################# End of 'bounce(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bounce' in the type store
        # Getting the type of 'stypy_return_type' (line 148)
        stypy_return_type_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_565)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bounce'
        return stypy_return_type_565


# Assigning a type to the variable 'Glass' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'Glass', Glass)
# Declaration of the 'Body' class

class Body(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 169, 4, False)
        # Assigning a type to the variable 'self' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Body.__init__', ['shape', 'material'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['shape', 'material'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 170):
        # Getting the type of 'shape' (line 170)
        shape_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 21), 'shape')
        # Getting the type of 'self' (line 170)
        self_567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'self')
        # Setting the type of the member 'shape' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), self_567, 'shape', shape_566)
        
        # Assigning a Name to a Attribute (line 171):
        # Getting the type of 'material' (line 171)
        material_568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 24), 'material')
        # Getting the type of 'self' (line 171)
        self_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'self')
        # Setting the type of the member 'material' of a type (line 171)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 8), self_569, 'material', material_568)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Body' (line 168)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 0), 'Body', Body)
# Declaration of the 'Output' class

class Output(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 175, 4, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Output.__init__', ['width', 'height'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['width', 'height'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 176):
        # Getting the type of 'width' (line 176)
        width_570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 21), 'width')
        # Getting the type of 'self' (line 176)
        self_571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'self')
        # Setting the type of the member 'width' of a type (line 176)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 8), self_571, 'width', width_570)
        
        # Assigning a Name to a Attribute (line 177):
        # Getting the type of 'height' (line 177)
        height_572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 22), 'height')
        # Getting the type of 'self' (line 177)
        self_573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self')
        # Setting the type of the member 'height' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_573, 'height', height_572)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Output' (line 174)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 0), 'Output', Output)
# Declaration of the 'Scene' class

class Scene(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 181, 4, False)
        # Assigning a type to the variable 'self' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Scene.__init__', ['output', 'camera', 'objects'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['output', 'camera', 'objects'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 182):
        # Getting the type of 'output' (line 182)
        output_574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'output')
        # Getting the type of 'self' (line 182)
        self_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'self')
        # Setting the type of the member 'output' of a type (line 182)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 8), self_575, 'output', output_574)
        
        # Assigning a Name to a Attribute (line 183):
        # Getting the type of 'camera' (line 183)
        camera_576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'camera')
        # Getting the type of 'self' (line 183)
        self_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'self')
        # Setting the type of the member 'camera' of a type (line 183)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 8), self_577, 'camera', camera_576)
        
        # Assigning a Name to a Attribute (line 184):
        # Getting the type of 'objects' (line 184)
        objects_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 23), 'objects')
        # Getting the type of 'self' (line 184)
        self_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'self')
        # Setting the type of the member 'objects' of a type (line 184)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), self_579, 'objects', objects_578)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Scene' (line 180)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 0), 'Scene', Scene)
# Declaration of the 'Renderer' class

class Renderer(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 188, 4, False)
        # Assigning a type to the variable 'self' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Renderer.__init__', ['scene'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['scene'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 189):
        # Getting the type of 'scene' (line 189)
        scene_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), 'scene')
        # Getting the type of 'self' (line 189)
        self_581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'self')
        # Setting the type of the member 'scene' of a type (line 189)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 8), self_581, 'scene', scene_580)
        
        # Assigning a ListComp to a Attribute (line 190):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 190)
        # Processing the call arguments (line 190)
        # Getting the type of 'scene' (line 190)
        scene_589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 57), 'scene', False)
        # Obtaining the member 'output' of a type (line 190)
        output_590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 57), scene_589, 'output')
        # Obtaining the member 'width' of a type (line 190)
        width_591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 57), output_590, 'width')
        # Getting the type of 'scene' (line 190)
        scene_592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 78), 'scene', False)
        # Obtaining the member 'output' of a type (line 190)
        output_593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 78), scene_592, 'output')
        # Obtaining the member 'height' of a type (line 190)
        height_594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 78), output_593, 'height')
        # Applying the binary operator '*' (line 190)
        result_mul_595 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 57), '*', width_591, height_594)
        
        # Processing the call keyword arguments (line 190)
        kwargs_596 = {}
        # Getting the type of 'xrange' (line 190)
        xrange_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 50), 'xrange', False)
        # Calling xrange(args, kwargs) (line 190)
        xrange_call_result_597 = invoke(stypy.reporting.localization.Localization(__file__, 190, 50), xrange_588, *[result_mul_595], **kwargs_596)
        
        comprehension_598 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 23), xrange_call_result_597)
        # Assigning a type to the variable 'i' (line 190)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 23), 'i', comprehension_598)
        
        # Call to V3(...): (line 190)
        # Processing the call arguments (line 190)
        float_583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 26), 'float')
        float_584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 31), 'float')
        float_585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 36), 'float')
        # Processing the call keyword arguments (line 190)
        kwargs_586 = {}
        # Getting the type of 'V3' (line 190)
        V3_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 23), 'V3', False)
        # Calling V3(args, kwargs) (line 190)
        V3_call_result_587 = invoke(stypy.reporting.localization.Localization(__file__, 190, 23), V3_582, *[float_583, float_584, float_585], **kwargs_586)
        
        list_599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 23), list_599, V3_call_result_587)
        # Getting the type of 'self' (line 190)
        self_600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'self')
        # Setting the type of the member 'buffer' of a type (line 190)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), self_600, 'buffer', list_599)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def clearBuffer(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clearBuffer'
        module_type_store = module_type_store.open_function_context('clearBuffer', 192, 4, False)
        # Assigning a type to the variable 'self' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Renderer.clearBuffer.__dict__.__setitem__('stypy_localization', localization)
        Renderer.clearBuffer.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Renderer.clearBuffer.__dict__.__setitem__('stypy_type_store', module_type_store)
        Renderer.clearBuffer.__dict__.__setitem__('stypy_function_name', 'Renderer.clearBuffer')
        Renderer.clearBuffer.__dict__.__setitem__('stypy_param_names_list', [])
        Renderer.clearBuffer.__dict__.__setitem__('stypy_varargs_param_name', None)
        Renderer.clearBuffer.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Renderer.clearBuffer.__dict__.__setitem__('stypy_call_defaults', defaults)
        Renderer.clearBuffer.__dict__.__setitem__('stypy_call_varargs', varargs)
        Renderer.clearBuffer.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Renderer.clearBuffer.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Renderer.clearBuffer', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clearBuffer', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clearBuffer(...)' code ##################

        
        
        # Call to xrange(...): (line 193)
        # Processing the call arguments (line 193)
        
        # Call to len(...): (line 193)
        # Processing the call arguments (line 193)
        # Getting the type of 'self' (line 193)
        self_603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 28), 'self', False)
        # Obtaining the member 'buffer' of a type (line 193)
        buffer_604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 28), self_603, 'buffer')
        # Processing the call keyword arguments (line 193)
        kwargs_605 = {}
        # Getting the type of 'len' (line 193)
        len_602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 24), 'len', False)
        # Calling len(args, kwargs) (line 193)
        len_call_result_606 = invoke(stypy.reporting.localization.Localization(__file__, 193, 24), len_602, *[buffer_604], **kwargs_605)
        
        # Processing the call keyword arguments (line 193)
        kwargs_607 = {}
        # Getting the type of 'xrange' (line 193)
        xrange_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 193)
        xrange_call_result_608 = invoke(stypy.reporting.localization.Localization(__file__, 193, 17), xrange_601, *[len_call_result_606], **kwargs_607)
        
        # Testing if the for loop is going to be iterated (line 193)
        # Testing the type of a for loop iterable (line 193)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 193, 8), xrange_call_result_608)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 193, 8), xrange_call_result_608):
            # Getting the type of the for loop variable (line 193)
            for_loop_var_609 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 193, 8), xrange_call_result_608)
            # Assigning a type to the variable 'i' (line 193)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'i', for_loop_var_609)
            # SSA begins for a for statement (line 193)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Num to a Attribute (line 194):
            float_610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 31), 'float')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 194)
            i_611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 24), 'i')
            # Getting the type of 'self' (line 194)
            self_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 12), 'self')
            # Obtaining the member 'buffer' of a type (line 194)
            buffer_613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), self_612, 'buffer')
            # Obtaining the member '__getitem__' of a type (line 194)
            getitem___614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), buffer_613, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 194)
            subscript_call_result_615 = invoke(stypy.reporting.localization.Localization(__file__, 194, 12), getitem___614, i_611)
            
            # Setting the type of the member 'x' of a type (line 194)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 12), subscript_call_result_615, 'x', float_610)
            
            # Assigning a Num to a Attribute (line 195):
            float_616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 195, 31), 'float')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 195)
            i_617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 24), 'i')
            # Getting the type of 'self' (line 195)
            self_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'self')
            # Obtaining the member 'buffer' of a type (line 195)
            buffer_619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), self_618, 'buffer')
            # Obtaining the member '__getitem__' of a type (line 195)
            getitem___620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), buffer_619, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 195)
            subscript_call_result_621 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), getitem___620, i_617)
            
            # Setting the type of the member 'y' of a type (line 195)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), subscript_call_result_621, 'y', float_616)
            
            # Assigning a Num to a Attribute (line 196):
            float_622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 31), 'float')
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 196)
            i_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 24), 'i')
            # Getting the type of 'self' (line 196)
            self_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'self')
            # Obtaining the member 'buffer' of a type (line 196)
            buffer_625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), self_624, 'buffer')
            # Obtaining the member '__getitem__' of a type (line 196)
            getitem___626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), buffer_625, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 196)
            subscript_call_result_627 = invoke(stypy.reporting.localization.Localization(__file__, 196, 12), getitem___626, i_623)
            
            # Setting the type of the member 'z' of a type (line 196)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 12), subscript_call_result_627, 'z', float_622)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'clearBuffer(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clearBuffer' in the type store
        # Getting the type of 'stypy_return_type' (line 192)
        stypy_return_type_628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_628)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clearBuffer'
        return stypy_return_type_628


    @norecursion
    def iterate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'iterate'
        module_type_store = module_type_store.open_function_context('iterate', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Renderer.iterate.__dict__.__setitem__('stypy_localization', localization)
        Renderer.iterate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Renderer.iterate.__dict__.__setitem__('stypy_type_store', module_type_store)
        Renderer.iterate.__dict__.__setitem__('stypy_function_name', 'Renderer.iterate')
        Renderer.iterate.__dict__.__setitem__('stypy_param_names_list', [])
        Renderer.iterate.__dict__.__setitem__('stypy_varargs_param_name', None)
        Renderer.iterate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Renderer.iterate.__dict__.__setitem__('stypy_call_defaults', defaults)
        Renderer.iterate.__dict__.__setitem__('stypy_call_varargs', varargs)
        Renderer.iterate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Renderer.iterate.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Renderer.iterate', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'iterate', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'iterate(...)' code ##################

        
        # Assigning a Attribute to a Name (line 199):
        # Getting the type of 'self' (line 199)
        self_629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 16), 'self')
        # Obtaining the member 'scene' of a type (line 199)
        scene_630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 16), self_629, 'scene')
        # Assigning a type to the variable 'scene' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'scene', scene_630)
        
        # Assigning a Attribute to a Name (line 200):
        # Getting the type of 'scene' (line 200)
        scene_631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'scene')
        # Obtaining the member 'output' of a type (line 200)
        output_632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 12), scene_631, 'output')
        # Obtaining the member 'width' of a type (line 200)
        width_633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 12), output_632, 'width')
        # Assigning a type to the variable 'w' (line 200)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'w', width_633)
        
        # Assigning a Attribute to a Name (line 201):
        # Getting the type of 'scene' (line 201)
        scene_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'scene')
        # Obtaining the member 'output' of a type (line 201)
        output_635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 12), scene_634, 'output')
        # Obtaining the member 'height' of a type (line 201)
        height_636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 12), output_635, 'height')
        # Assigning a type to the variable 'h' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 8), 'h', height_636)
        
        # Assigning a Num to a Name (line 202):
        int_637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 12), 'int')
        # Assigning a type to the variable 'i' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'i', int_637)
        
        # Assigning a BinOp to a Name (line 204):
        
        # Call to random(...): (line 204)
        # Processing the call keyword arguments (line 204)
        kwargs_639 = {}
        # Getting the type of 'random' (line 204)
        random_638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'random', False)
        # Calling random(args, kwargs) (line 204)
        random_call_result_640 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), random_638, *[], **kwargs_639)
        
        # Getting the type of 'h' (line 204)
        h_641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 23), 'h')
        # Applying the binary operator 'div' (line 204)
        result_div_642 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 12), 'div', random_call_result_640, h_641)
        
        # Assigning a type to the variable 'y' (line 204)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'y', result_div_642)
        
        # Assigning a BinOp to a Name (line 205):
        float_643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 16), 'float')
        # Getting the type of 'h' (line 205)
        h_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 22), 'h')
        # Applying the binary operator 'div' (line 205)
        result_div_645 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 16), 'div', float_643, h_644)
        
        # Assigning a type to the variable 'ystep' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'ystep', result_div_645)
        
        
        # Getting the type of 'y' (line 206)
        y_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 14), 'y')
        float_647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 18), 'float')
        # Applying the binary operator '<' (line 206)
        result_lt_648 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 14), '<', y_646, float_647)
        
        # Testing if the while is going to be iterated (line 206)
        # Testing the type of an if condition (line 206)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 8), result_lt_648)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 206, 8), result_lt_648):
            # SSA begins for while statement (line 206)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
            
            # Assigning a BinOp to a Name (line 207):
            
            # Call to random(...): (line 207)
            # Processing the call keyword arguments (line 207)
            kwargs_650 = {}
            # Getting the type of 'random' (line 207)
            random_649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 16), 'random', False)
            # Calling random(args, kwargs) (line 207)
            random_call_result_651 = invoke(stypy.reporting.localization.Localization(__file__, 207, 16), random_649, *[], **kwargs_650)
            
            # Getting the type of 'w' (line 207)
            w_652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 27), 'w')
            # Applying the binary operator 'div' (line 207)
            result_div_653 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 16), 'div', random_call_result_651, w_652)
            
            # Assigning a type to the variable 'x' (line 207)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'x', result_div_653)
            
            # Assigning a BinOp to a Name (line 208):
            float_654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 20), 'float')
            # Getting the type of 'w' (line 208)
            w_655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 26), 'w')
            # Applying the binary operator 'div' (line 208)
            result_div_656 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 20), 'div', float_654, w_655)
            
            # Assigning a type to the variable 'xstep' (line 208)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 12), 'xstep', result_div_656)
            
            
            # Getting the type of 'x' (line 209)
            x_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 18), 'x')
            float_658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 22), 'float')
            # Applying the binary operator '<' (line 209)
            result_lt_659 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 18), '<', x_657, float_658)
            
            # Testing if the while is going to be iterated (line 209)
            # Testing the type of an if condition (line 209)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 12), result_lt_659)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 209, 12), result_lt_659):
                # SSA begins for while statement (line 209)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
                
                # Assigning a Call to a Name (line 210):
                
                # Call to getRay(...): (line 210)
                # Processing the call arguments (line 210)
                # Getting the type of 'x' (line 210)
                x_663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 42), 'x', False)
                # Getting the type of 'y' (line 210)
                y_664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 45), 'y', False)
                # Processing the call keyword arguments (line 210)
                kwargs_665 = {}
                # Getting the type of 'scene' (line 210)
                scene_660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'scene', False)
                # Obtaining the member 'camera' of a type (line 210)
                camera_661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 22), scene_660, 'camera')
                # Obtaining the member 'getRay' of a type (line 210)
                getRay_662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 22), camera_661, 'getRay')
                # Calling getRay(args, kwargs) (line 210)
                getRay_call_result_666 = invoke(stypy.reporting.localization.Localization(__file__, 210, 22), getRay_662, *[x_663, y_664], **kwargs_665)
                
                # Assigning a type to the variable 'ray' (line 210)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'ray', getRay_call_result_666)
                
                # Assigning a Call to a Name (line 211):
                
                # Call to trace(...): (line 211)
                # Processing the call arguments (line 211)
                # Getting the type of 'ray' (line 211)
                ray_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 35), 'ray', False)
                int_670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 40), 'int')
                # Processing the call keyword arguments (line 211)
                kwargs_671 = {}
                # Getting the type of 'self' (line 211)
                self_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 'self', False)
                # Obtaining the member 'trace' of a type (line 211)
                trace_668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 24), self_667, 'trace')
                # Calling trace(args, kwargs) (line 211)
                trace_call_result_672 = invoke(stypy.reporting.localization.Localization(__file__, 211, 24), trace_668, *[ray_669, int_670], **kwargs_671)
                
                # Assigning a type to the variable 'color' (line 211)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'color', trace_call_result_672)
                
                # Call to iadd(...): (line 212)
                # Processing the call arguments (line 212)
                # Getting the type of 'color' (line 212)
                color_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 36), 'color', False)
                # Processing the call keyword arguments (line 212)
                kwargs_680 = {}
                
                # Obtaining the type of the subscript
                # Getting the type of 'i' (line 212)
                i_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 28), 'i', False)
                # Getting the type of 'self' (line 212)
                self_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 16), 'self', False)
                # Obtaining the member 'buffer' of a type (line 212)
                buffer_675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 16), self_674, 'buffer')
                # Obtaining the member '__getitem__' of a type (line 212)
                getitem___676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 16), buffer_675, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 212)
                subscript_call_result_677 = invoke(stypy.reporting.localization.Localization(__file__, 212, 16), getitem___676, i_673)
                
                # Obtaining the member 'iadd' of a type (line 212)
                iadd_678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 16), subscript_call_result_677, 'iadd')
                # Calling iadd(args, kwargs) (line 212)
                iadd_call_result_681 = invoke(stypy.reporting.localization.Localization(__file__, 212, 16), iadd_678, *[color_679], **kwargs_680)
                
                
                # Getting the type of 'i' (line 213)
                i_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'i')
                int_683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 21), 'int')
                # Applying the binary operator '+=' (line 213)
                result_iadd_684 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 16), '+=', i_682, int_683)
                # Assigning a type to the variable 'i' (line 213)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'i', result_iadd_684)
                
                
                # Getting the type of 'x' (line 214)
                x_685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'x')
                # Getting the type of 'xstep' (line 214)
                xstep_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'xstep')
                # Applying the binary operator '+=' (line 214)
                result_iadd_687 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 16), '+=', x_685, xstep_686)
                # Assigning a type to the variable 'x' (line 214)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 16), 'x', result_iadd_687)
                
                # SSA join for while statement (line 209)
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Getting the type of 'y' (line 215)
            y_688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'y')
            # Getting the type of 'ystep' (line 215)
            ystep_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 17), 'ystep')
            # Applying the binary operator '+=' (line 215)
            result_iadd_690 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 12), '+=', y_688, ystep_689)
            # Assigning a type to the variable 'y' (line 215)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'y', result_iadd_690)
            
            # SSA join for while statement (line 206)
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'iterate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'iterate' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_691)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'iterate'
        return stypy_return_type_691


    @norecursion
    def trace(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'trace'
        module_type_store = module_type_store.open_function_context('trace', 217, 4, False)
        # Assigning a type to the variable 'self' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Renderer.trace.__dict__.__setitem__('stypy_localization', localization)
        Renderer.trace.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Renderer.trace.__dict__.__setitem__('stypy_type_store', module_type_store)
        Renderer.trace.__dict__.__setitem__('stypy_function_name', 'Renderer.trace')
        Renderer.trace.__dict__.__setitem__('stypy_param_names_list', ['ray', 'n'])
        Renderer.trace.__dict__.__setitem__('stypy_varargs_param_name', None)
        Renderer.trace.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Renderer.trace.__dict__.__setitem__('stypy_call_defaults', defaults)
        Renderer.trace.__dict__.__setitem__('stypy_call_varargs', varargs)
        Renderer.trace.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Renderer.trace.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Renderer.trace', ['ray', 'n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'trace', localization, ['ray', 'n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'trace(...)' code ##################

        
        # Assigning a Call to a Name (line 218):
        
        # Call to float(...): (line 218)
        # Processing the call arguments (line 218)
        str_693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 21), 'str', 'inf')
        # Processing the call keyword arguments (line 218)
        kwargs_694 = {}
        # Getting the type of 'float' (line 218)
        float_692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'float', False)
        # Calling float(args, kwargs) (line 218)
        float_call_result_695 = invoke(stypy.reporting.localization.Localization(__file__, 218, 15), float_692, *[str_693], **kwargs_694)
        
        # Assigning a type to the variable 'mint' (line 218)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'mint', float_call_result_695)
        
        # Getting the type of 'n' (line 221)
        n_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 11), 'n')
        int_697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 15), 'int')
        # Applying the binary operator '>' (line 221)
        result_gt_698 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 11), '>', n_696, int_697)
        
        # Testing if the type of an if condition is none (line 221)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 221, 8), result_gt_698):
            pass
        else:
            
            # Testing the type of an if condition (line 221)
            if_condition_699 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 8), result_gt_698)
            # Assigning a type to the variable 'if_condition_699' (line 221)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'if_condition_699', if_condition_699)
            # SSA begins for if statement (line 221)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to V3(...): (line 222)
            # Processing the call arguments (line 222)
            float_701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 22), 'float')
            float_702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 27), 'float')
            float_703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 32), 'float')
            # Processing the call keyword arguments (line 222)
            kwargs_704 = {}
            # Getting the type of 'V3' (line 222)
            V3_700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 19), 'V3', False)
            # Calling V3(args, kwargs) (line 222)
            V3_call_result_705 = invoke(stypy.reporting.localization.Localization(__file__, 222, 19), V3_700, *[float_701, float_702, float_703], **kwargs_704)
            
            # Assigning a type to the variable 'stypy_return_type' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'stypy_return_type', V3_call_result_705)
            # SSA join for if statement (line 221)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Name (line 224):
        # Getting the type of 'None' (line 224)
        None_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 14), 'None')
        # Assigning a type to the variable 'hit' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'hit', None_706)
        
        
        # Call to xrange(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Call to len(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'self' (line 226)
        self_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 28), 'self', False)
        # Obtaining the member 'scene' of a type (line 226)
        scene_710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 28), self_709, 'scene')
        # Obtaining the member 'objects' of a type (line 226)
        objects_711 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 28), scene_710, 'objects')
        # Processing the call keyword arguments (line 226)
        kwargs_712 = {}
        # Getting the type of 'len' (line 226)
        len_708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'len', False)
        # Calling len(args, kwargs) (line 226)
        len_call_result_713 = invoke(stypy.reporting.localization.Localization(__file__, 226, 24), len_708, *[objects_711], **kwargs_712)
        
        # Processing the call keyword arguments (line 226)
        kwargs_714 = {}
        # Getting the type of 'xrange' (line 226)
        xrange_707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 226)
        xrange_call_result_715 = invoke(stypy.reporting.localization.Localization(__file__, 226, 17), xrange_707, *[len_call_result_713], **kwargs_714)
        
        # Testing if the for loop is going to be iterated (line 226)
        # Testing the type of a for loop iterable (line 226)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 226, 8), xrange_call_result_715)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 226, 8), xrange_call_result_715):
            # Getting the type of the for loop variable (line 226)
            for_loop_var_716 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 226, 8), xrange_call_result_715)
            # Assigning a type to the variable 'i' (line 226)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'i', for_loop_var_716)
            # SSA begins for a for statement (line 226)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 227):
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 227)
            i_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 35), 'i')
            # Getting the type of 'self' (line 227)
            self_718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 16), 'self')
            # Obtaining the member 'scene' of a type (line 227)
            scene_719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), self_718, 'scene')
            # Obtaining the member 'objects' of a type (line 227)
            objects_720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), scene_719, 'objects')
            # Obtaining the member '__getitem__' of a type (line 227)
            getitem___721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 16), objects_720, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 227)
            subscript_call_result_722 = invoke(stypy.reporting.localization.Localization(__file__, 227, 16), getitem___721, i_717)
            
            # Assigning a type to the variable 'o' (line 227)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 12), 'o', subscript_call_result_722)
            
            # Assigning a Call to a Name (line 228):
            
            # Call to intersect(...): (line 228)
            # Processing the call arguments (line 228)
            # Getting the type of 'ray' (line 228)
            ray_726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 34), 'ray', False)
            # Processing the call keyword arguments (line 228)
            kwargs_727 = {}
            # Getting the type of 'o' (line 228)
            o_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 16), 'o', False)
            # Obtaining the member 'shape' of a type (line 228)
            shape_724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), o_723, 'shape')
            # Obtaining the member 'intersect' of a type (line 228)
            intersect_725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 16), shape_724, 'intersect')
            # Calling intersect(args, kwargs) (line 228)
            intersect_call_result_728 = invoke(stypy.reporting.localization.Localization(__file__, 228, 16), intersect_725, *[ray_726], **kwargs_727)
            
            # Assigning a type to the variable 't' (line 228)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 't', intersect_call_result_728)
            
            # Evaluating a boolean operation
            
            # Getting the type of 't' (line 229)
            t_729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 't')
            int_730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 19), 'int')
            # Applying the binary operator '>' (line 229)
            result_gt_731 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 15), '>', t_729, int_730)
            
            
            # Getting the type of 't' (line 229)
            t_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 25), 't')
            # Getting the type of 'mint' (line 229)
            mint_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 30), 'mint')
            # Applying the binary operator '<=' (line 229)
            result_le_734 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 25), '<=', t_732, mint_733)
            
            # Applying the binary operator 'and' (line 229)
            result_and_keyword_735 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 15), 'and', result_gt_731, result_le_734)
            
            # Testing if the type of an if condition is none (line 229)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 229, 12), result_and_keyword_735):
                pass
            else:
                
                # Testing the type of an if condition (line 229)
                if_condition_736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 12), result_and_keyword_735)
                # Assigning a type to the variable 'if_condition_736' (line 229)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 12), 'if_condition_736', if_condition_736)
                # SSA begins for if statement (line 229)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 230):
                # Getting the type of 't' (line 230)
                t_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 23), 't')
                # Assigning a type to the variable 'mint' (line 230)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 16), 'mint', t_737)
                
                # Assigning a Name to a Name (line 231):
                # Getting the type of 'o' (line 231)
                o_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 22), 'o')
                # Assigning a type to the variable 'hit' (line 231)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 16), 'hit', o_738)
                # SSA join for if statement (line 229)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Type idiom detected: calculating its left and rigth part (line 233)
        # Getting the type of 'hit' (line 233)
        hit_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 11), 'hit')
        # Getting the type of 'None' (line 233)
        None_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 18), 'None')
        
        (may_be_741, more_types_in_union_742) = may_be_none(hit_739, None_740)

        if may_be_741:

            if more_types_in_union_742:
                # Runtime conditional SSA (line 233)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to V3(...): (line 234)
            # Processing the call arguments (line 234)
            float_744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 22), 'float')
            float_745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 27), 'float')
            float_746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 32), 'float')
            # Processing the call keyword arguments (line 234)
            kwargs_747 = {}
            # Getting the type of 'V3' (line 234)
            V3_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 19), 'V3', False)
            # Calling V3(args, kwargs) (line 234)
            V3_call_result_748 = invoke(stypy.reporting.localization.Localization(__file__, 234, 19), V3_743, *[float_744, float_745, float_746], **kwargs_747)
            
            # Assigning a type to the variable 'stypy_return_type' (line 234)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'stypy_return_type', V3_call_result_748)

            if more_types_in_union_742:
                # SSA join for if statement (line 233)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 236):
        
        # Call to add(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Call to muls(...): (line 236)
        # Processing the call arguments (line 236)
        # Getting the type of 'mint' (line 236)
        mint_755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 50), 'mint', False)
        # Processing the call keyword arguments (line 236)
        kwargs_756 = {}
        # Getting the type of 'ray' (line 236)
        ray_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 31), 'ray', False)
        # Obtaining the member 'direction' of a type (line 236)
        direction_753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 31), ray_752, 'direction')
        # Obtaining the member 'muls' of a type (line 236)
        muls_754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 31), direction_753, 'muls')
        # Calling muls(args, kwargs) (line 236)
        muls_call_result_757 = invoke(stypy.reporting.localization.Localization(__file__, 236, 31), muls_754, *[mint_755], **kwargs_756)
        
        # Processing the call keyword arguments (line 236)
        kwargs_758 = {}
        # Getting the type of 'ray' (line 236)
        ray_749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 16), 'ray', False)
        # Obtaining the member 'origin' of a type (line 236)
        origin_750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), ray_749, 'origin')
        # Obtaining the member 'add' of a type (line 236)
        add_751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 16), origin_750, 'add')
        # Calling add(args, kwargs) (line 236)
        add_call_result_759 = invoke(stypy.reporting.localization.Localization(__file__, 236, 16), add_751, *[muls_call_result_757], **kwargs_758)
        
        # Assigning a type to the variable 'point' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'point', add_call_result_759)
        
        # Assigning a Call to a Name (line 237):
        
        # Call to getNormal(...): (line 237)
        # Processing the call arguments (line 237)
        # Getting the type of 'point' (line 237)
        point_763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 37), 'point', False)
        # Processing the call keyword arguments (line 237)
        kwargs_764 = {}
        # Getting the type of 'hit' (line 237)
        hit_760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 17), 'hit', False)
        # Obtaining the member 'shape' of a type (line 237)
        shape_761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 17), hit_760, 'shape')
        # Obtaining the member 'getNormal' of a type (line 237)
        getNormal_762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 17), shape_761, 'getNormal')
        # Calling getNormal(args, kwargs) (line 237)
        getNormal_call_result_765 = invoke(stypy.reporting.localization.Localization(__file__, 237, 17), getNormal_762, *[point_763], **kwargs_764)
        
        # Assigning a type to the variable 'normal' (line 237)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'normal', getNormal_call_result_765)
        
        # Assigning a Call to a Name (line 238):
        
        # Call to bounce(...): (line 238)
        # Processing the call arguments (line 238)
        # Getting the type of 'ray' (line 238)
        ray_769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 40), 'ray', False)
        # Getting the type of 'normal' (line 238)
        normal_770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 45), 'normal', False)
        # Processing the call keyword arguments (line 238)
        kwargs_771 = {}
        # Getting the type of 'hit' (line 238)
        hit_766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'hit', False)
        # Obtaining the member 'material' of a type (line 238)
        material_767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 20), hit_766, 'material')
        # Obtaining the member 'bounce' of a type (line 238)
        bounce_768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 20), material_767, 'bounce')
        # Calling bounce(args, kwargs) (line 238)
        bounce_call_result_772 = invoke(stypy.reporting.localization.Localization(__file__, 238, 20), bounce_768, *[ray_769, normal_770], **kwargs_771)
        
        # Assigning a type to the variable 'direction' (line 238)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 8), 'direction', bounce_call_result_772)
        
        
        # Call to dot(...): (line 240)
        # Processing the call arguments (line 240)
        # Getting the type of 'ray' (line 240)
        ray_775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 25), 'ray', False)
        # Obtaining the member 'direction' of a type (line 240)
        direction_776 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 25), ray_775, 'direction')
        # Processing the call keyword arguments (line 240)
        kwargs_777 = {}
        # Getting the type of 'direction' (line 240)
        direction_773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 11), 'direction', False)
        # Obtaining the member 'dot' of a type (line 240)
        dot_774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 11), direction_773, 'dot')
        # Calling dot(args, kwargs) (line 240)
        dot_call_result_778 = invoke(stypy.reporting.localization.Localization(__file__, 240, 11), dot_774, *[direction_776], **kwargs_777)
        
        float_779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 42), 'float')
        # Applying the binary operator '>' (line 240)
        result_gt_780 = python_operator(stypy.reporting.localization.Localization(__file__, 240, 11), '>', dot_call_result_778, float_779)
        
        # Testing if the type of an if condition is none (line 240)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 240, 8), result_gt_780):
            
            # Assigning a Call to a Name (line 245):
            
            # Call to add(...): (line 245)
            # Processing the call arguments (line 245)
            
            # Call to muls(...): (line 245)
            # Processing the call arguments (line 245)
            # Getting the type of 'mint' (line 245)
            mint_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 54), 'mint', False)
            float_802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 61), 'float')
            # Applying the binary operator '*' (line 245)
            result_mul_803 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 54), '*', mint_801, float_802)
            
            # Processing the call keyword arguments (line 245)
            kwargs_804 = {}
            # Getting the type of 'ray' (line 245)
            ray_798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 35), 'ray', False)
            # Obtaining the member 'direction' of a type (line 245)
            direction_799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 35), ray_798, 'direction')
            # Obtaining the member 'muls' of a type (line 245)
            muls_800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 35), direction_799, 'muls')
            # Calling muls(args, kwargs) (line 245)
            muls_call_result_805 = invoke(stypy.reporting.localization.Localization(__file__, 245, 35), muls_800, *[result_mul_803], **kwargs_804)
            
            # Processing the call keyword arguments (line 245)
            kwargs_806 = {}
            # Getting the type of 'ray' (line 245)
            ray_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'ray', False)
            # Obtaining the member 'origin' of a type (line 245)
            origin_796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 20), ray_795, 'origin')
            # Obtaining the member 'add' of a type (line 245)
            add_797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 20), origin_796, 'add')
            # Calling add(args, kwargs) (line 245)
            add_call_result_807 = invoke(stypy.reporting.localization.Localization(__file__, 245, 20), add_797, *[muls_call_result_805], **kwargs_806)
            
            # Assigning a type to the variable 'point' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'point', add_call_result_807)
        else:
            
            # Testing the type of an if condition (line 240)
            if_condition_781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 240, 8), result_gt_780)
            # Assigning a type to the variable 'if_condition_781' (line 240)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'if_condition_781', if_condition_781)
            # SSA begins for if statement (line 240)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 241):
            
            # Call to add(...): (line 241)
            # Processing the call arguments (line 241)
            
            # Call to muls(...): (line 241)
            # Processing the call arguments (line 241)
            # Getting the type of 'mint' (line 241)
            mint_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 54), 'mint', False)
            float_789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 61), 'float')
            # Applying the binary operator '*' (line 241)
            result_mul_790 = python_operator(stypy.reporting.localization.Localization(__file__, 241, 54), '*', mint_788, float_789)
            
            # Processing the call keyword arguments (line 241)
            kwargs_791 = {}
            # Getting the type of 'ray' (line 241)
            ray_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 35), 'ray', False)
            # Obtaining the member 'direction' of a type (line 241)
            direction_786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 35), ray_785, 'direction')
            # Obtaining the member 'muls' of a type (line 241)
            muls_787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 35), direction_786, 'muls')
            # Calling muls(args, kwargs) (line 241)
            muls_call_result_792 = invoke(stypy.reporting.localization.Localization(__file__, 241, 35), muls_787, *[result_mul_790], **kwargs_791)
            
            # Processing the call keyword arguments (line 241)
            kwargs_793 = {}
            # Getting the type of 'ray' (line 241)
            ray_782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 20), 'ray', False)
            # Obtaining the member 'origin' of a type (line 241)
            origin_783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 20), ray_782, 'origin')
            # Obtaining the member 'add' of a type (line 241)
            add_784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 20), origin_783, 'add')
            # Calling add(args, kwargs) (line 241)
            add_call_result_794 = invoke(stypy.reporting.localization.Localization(__file__, 241, 20), add_784, *[muls_call_result_792], **kwargs_793)
            
            # Assigning a type to the variable 'point' (line 241)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'point', add_call_result_794)
            # SSA branch for the else part of an if statement (line 240)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Name (line 245):
            
            # Call to add(...): (line 245)
            # Processing the call arguments (line 245)
            
            # Call to muls(...): (line 245)
            # Processing the call arguments (line 245)
            # Getting the type of 'mint' (line 245)
            mint_801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 54), 'mint', False)
            float_802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 61), 'float')
            # Applying the binary operator '*' (line 245)
            result_mul_803 = python_operator(stypy.reporting.localization.Localization(__file__, 245, 54), '*', mint_801, float_802)
            
            # Processing the call keyword arguments (line 245)
            kwargs_804 = {}
            # Getting the type of 'ray' (line 245)
            ray_798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 35), 'ray', False)
            # Obtaining the member 'direction' of a type (line 245)
            direction_799 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 35), ray_798, 'direction')
            # Obtaining the member 'muls' of a type (line 245)
            muls_800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 35), direction_799, 'muls')
            # Calling muls(args, kwargs) (line 245)
            muls_call_result_805 = invoke(stypy.reporting.localization.Localization(__file__, 245, 35), muls_800, *[result_mul_803], **kwargs_804)
            
            # Processing the call keyword arguments (line 245)
            kwargs_806 = {}
            # Getting the type of 'ray' (line 245)
            ray_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 245, 20), 'ray', False)
            # Obtaining the member 'origin' of a type (line 245)
            origin_796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 20), ray_795, 'origin')
            # Obtaining the member 'add' of a type (line 245)
            add_797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 245, 20), origin_796, 'add')
            # Calling add(args, kwargs) (line 245)
            add_call_result_807 = invoke(stypy.reporting.localization.Localization(__file__, 245, 20), add_797, *[muls_call_result_805], **kwargs_806)
            
            # Assigning a type to the variable 'point' (line 245)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 12), 'point', add_call_result_807)
            # SSA join for if statement (line 240)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 246):
        
        # Call to Ray(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'point' (line 246)
        point_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 21), 'point', False)
        # Getting the type of 'direction' (line 246)
        direction_810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 28), 'direction', False)
        # Processing the call keyword arguments (line 246)
        kwargs_811 = {}
        # Getting the type of 'Ray' (line 246)
        Ray_808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 17), 'Ray', False)
        # Calling Ray(args, kwargs) (line 246)
        Ray_call_result_812 = invoke(stypy.reporting.localization.Localization(__file__, 246, 17), Ray_808, *[point_809, direction_810], **kwargs_811)
        
        # Assigning a type to the variable 'newray' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'newray', Ray_call_result_812)
        
        # Call to add(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'hit' (line 247)
        hit_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 69), 'hit', False)
        # Obtaining the member 'material' of a type (line 247)
        material_829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 69), hit_828, 'material')
        # Obtaining the member 'emission' of a type (line 247)
        emission_830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 69), material_829, 'emission')
        # Processing the call keyword arguments (line 247)
        kwargs_831 = {}
        
        # Call to mul(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'hit' (line 247)
        hit_822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 45), 'hit', False)
        # Obtaining the member 'material' of a type (line 247)
        material_823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 45), hit_822, 'material')
        # Obtaining the member 'color' of a type (line 247)
        color_824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 45), material_823, 'color')
        # Processing the call keyword arguments (line 247)
        kwargs_825 = {}
        
        # Call to trace(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'newray' (line 247)
        newray_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 26), 'newray', False)
        # Getting the type of 'n' (line 247)
        n_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 34), 'n', False)
        int_817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 38), 'int')
        # Applying the binary operator '+' (line 247)
        result_add_818 = python_operator(stypy.reporting.localization.Localization(__file__, 247, 34), '+', n_816, int_817)
        
        # Processing the call keyword arguments (line 247)
        kwargs_819 = {}
        # Getting the type of 'self' (line 247)
        self_813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 15), 'self', False)
        # Obtaining the member 'trace' of a type (line 247)
        trace_814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 15), self_813, 'trace')
        # Calling trace(args, kwargs) (line 247)
        trace_call_result_820 = invoke(stypy.reporting.localization.Localization(__file__, 247, 15), trace_814, *[newray_815, result_add_818], **kwargs_819)
        
        # Obtaining the member 'mul' of a type (line 247)
        mul_821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 15), trace_call_result_820, 'mul')
        # Calling mul(args, kwargs) (line 247)
        mul_call_result_826 = invoke(stypy.reporting.localization.Localization(__file__, 247, 15), mul_821, *[color_824], **kwargs_825)
        
        # Obtaining the member 'add' of a type (line 247)
        add_827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 15), mul_call_result_826, 'add')
        # Calling add(args, kwargs) (line 247)
        add_call_result_832 = invoke(stypy.reporting.localization.Localization(__file__, 247, 15), add_827, *[emission_830], **kwargs_831)
        
        # Assigning a type to the variable 'stypy_return_type' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'stypy_return_type', add_call_result_832)
        
        # ################# End of 'trace(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'trace' in the type store
        # Getting the type of 'stypy_return_type' (line 217)
        stypy_return_type_833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_833)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'trace'
        return stypy_return_type_833


    @staticmethod
    @norecursion
    def cmap(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'cmap'
        module_type_store = module_type_store.open_function_context('cmap', 249, 4, False)
        
        # Passed parameters checking function
        Renderer.cmap.__dict__.__setitem__('stypy_localization', localization)
        Renderer.cmap.__dict__.__setitem__('stypy_type_of_self', None)
        Renderer.cmap.__dict__.__setitem__('stypy_type_store', module_type_store)
        Renderer.cmap.__dict__.__setitem__('stypy_function_name', 'cmap')
        Renderer.cmap.__dict__.__setitem__('stypy_param_names_list', ['x'])
        Renderer.cmap.__dict__.__setitem__('stypy_varargs_param_name', None)
        Renderer.cmap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Renderer.cmap.__dict__.__setitem__('stypy_call_defaults', defaults)
        Renderer.cmap.__dict__.__setitem__('stypy_call_varargs', varargs)
        Renderer.cmap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Renderer.cmap.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'cmap', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'cmap', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'cmap(...)' code ##################

        
        
        # Getting the type of 'x' (line 251)
        x_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'x')
        float_835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 24), 'float')
        # Applying the binary operator '<' (line 251)
        result_lt_836 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 20), '<', x_834, float_835)
        
        # Testing the type of an if expression (line 251)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 15), result_lt_836)
        # SSA begins for if expression (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        int_837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 15), 'int')
        # SSA branch for the else part of an if expression (line 251)
        module_type_store.open_ssa_branch('if expression else')
        
        
        # Getting the type of 'x' (line 251)
        x_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 41), 'x')
        float_839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 45), 'float')
        # Applying the binary operator '>' (line 251)
        result_gt_840 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 41), '>', x_838, float_839)
        
        # Testing the type of an if expression (line 251)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 34), result_gt_840)
        # SSA begins for if expression (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
        int_841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 34), 'int')
        # SSA branch for the else part of an if expression (line 251)
        module_type_store.open_ssa_branch('if expression else')
        
        # Call to int(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'x' (line 251)
        x_843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 58), 'x', False)
        int_844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 62), 'int')
        # Applying the binary operator '*' (line 251)
        result_mul_845 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 58), '*', x_843, int_844)
        
        # Processing the call keyword arguments (line 251)
        kwargs_846 = {}
        # Getting the type of 'int' (line 251)
        int_842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 54), 'int', False)
        # Calling int(args, kwargs) (line 251)
        int_call_result_847 = invoke(stypy.reporting.localization.Localization(__file__, 251, 54), int_842, *[result_mul_845], **kwargs_846)
        
        # SSA join for if expression (line 251)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_848 = union_type.UnionType.add(int_841, int_call_result_847)
        
        # SSA join for if expression (line 251)
        module_type_store = module_type_store.join_ssa_context()
        if_exp_849 = union_type.UnionType.add(int_837, if_exp_848)
        
        # Assigning a type to the variable 'stypy_return_type' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'stypy_return_type', if_exp_849)
        
        # ################# End of 'cmap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'cmap' in the type store
        # Getting the type of 'stypy_return_type' (line 249)
        stypy_return_type_850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_850)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'cmap'
        return stypy_return_type_850


    @norecursion
    def saveFrame(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'saveFrame'
        module_type_store = module_type_store.open_function_context('saveFrame', 254, 4, False)
        # Assigning a type to the variable 'self' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Renderer.saveFrame.__dict__.__setitem__('stypy_localization', localization)
        Renderer.saveFrame.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Renderer.saveFrame.__dict__.__setitem__('stypy_type_store', module_type_store)
        Renderer.saveFrame.__dict__.__setitem__('stypy_function_name', 'Renderer.saveFrame')
        Renderer.saveFrame.__dict__.__setitem__('stypy_param_names_list', ['filename', 'nframe'])
        Renderer.saveFrame.__dict__.__setitem__('stypy_varargs_param_name', None)
        Renderer.saveFrame.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Renderer.saveFrame.__dict__.__setitem__('stypy_call_defaults', defaults)
        Renderer.saveFrame.__dict__.__setitem__('stypy_call_varargs', varargs)
        Renderer.saveFrame.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Renderer.saveFrame.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Renderer.saveFrame', ['filename', 'nframe'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'saveFrame', localization, ['filename', 'nframe'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'saveFrame(...)' code ##################

        
        # Assigning a Call to a Name (line 255):
        
        # Call to file(...): (line 255)
        # Processing the call arguments (line 255)
        # Getting the type of 'filename' (line 255)
        filename_852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 20), 'filename', False)
        str_853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 30), 'str', 'w')
        # Processing the call keyword arguments (line 255)
        kwargs_854 = {}
        # Getting the type of 'file' (line 255)
        file_851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'file', False)
        # Calling file(args, kwargs) (line 255)
        file_call_result_855 = invoke(stypy.reporting.localization.Localization(__file__, 255, 15), file_851, *[filename_852, str_853], **kwargs_854)
        
        # Assigning a type to the variable 'fout' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 8), 'fout', file_call_result_855)
        
        # Call to write(...): (line 256)
        # Processing the call arguments (line 256)
        str_858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 19), 'str', 'P3\n%d %d\n%d\n')
        
        # Obtaining an instance of the builtin type 'tuple' (line 256)
        tuple_859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 40), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 256)
        # Adding element type (line 256)
        # Getting the type of 'self' (line 256)
        self_860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 40), 'self', False)
        # Obtaining the member 'scene' of a type (line 256)
        scene_861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 40), self_860, 'scene')
        # Obtaining the member 'output' of a type (line 256)
        output_862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 40), scene_861, 'output')
        # Obtaining the member 'width' of a type (line 256)
        width_863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 40), output_862, 'width')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 40), tuple_859, width_863)
        # Adding element type (line 256)
        # Getting the type of 'self' (line 256)
        self_864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 65), 'self', False)
        # Obtaining the member 'scene' of a type (line 256)
        scene_865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 65), self_864, 'scene')
        # Obtaining the member 'output' of a type (line 256)
        output_866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 65), scene_865, 'output')
        # Obtaining the member 'height' of a type (line 256)
        height_867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 65), output_866, 'height')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 40), tuple_859, height_867)
        # Adding element type (line 256)
        int_868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 256, 91), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 256, 40), tuple_859, int_868)
        
        # Applying the binary operator '%' (line 256)
        result_mod_869 = python_operator(stypy.reporting.localization.Localization(__file__, 256, 19), '%', str_858, tuple_859)
        
        # Processing the call keyword arguments (line 256)
        kwargs_870 = {}
        # Getting the type of 'fout' (line 256)
        fout_856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 8), 'fout', False)
        # Obtaining the member 'write' of a type (line 256)
        write_857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 8), fout_856, 'write')
        # Calling write(args, kwargs) (line 256)
        write_call_result_871 = invoke(stypy.reporting.localization.Localization(__file__, 256, 8), write_857, *[result_mod_869], **kwargs_870)
        
        
        # Getting the type of 'self' (line 257)
        self_872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 17), 'self')
        # Obtaining the member 'buffer' of a type (line 257)
        buffer_873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 17), self_872, 'buffer')
        # Testing if the for loop is going to be iterated (line 257)
        # Testing the type of a for loop iterable (line 257)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 257, 8), buffer_873)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 257, 8), buffer_873):
            # Getting the type of the for loop variable (line 257)
            for_loop_var_874 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 257, 8), buffer_873)
            # Assigning a type to the variable 'p' (line 257)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'p', for_loop_var_874)
            # SSA begins for a for statement (line 257)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to write(...): (line 258)
            # Processing the call arguments (line 258)
            str_877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 23), 'str', '%d %d %d\n')
            
            # Obtaining an instance of the builtin type 'tuple' (line 258)
            tuple_878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 39), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 258)
            # Adding element type (line 258)
            
            # Call to cmap(...): (line 258)
            # Processing the call arguments (line 258)
            # Getting the type of 'p' (line 258)
            p_881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 53), 'p', False)
            # Obtaining the member 'x' of a type (line 258)
            x_882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 53), p_881, 'x')
            # Getting the type of 'nframe' (line 258)
            nframe_883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 59), 'nframe', False)
            # Applying the binary operator 'div' (line 258)
            result_div_884 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 53), 'div', x_882, nframe_883)
            
            # Processing the call keyword arguments (line 258)
            kwargs_885 = {}
            # Getting the type of 'Renderer' (line 258)
            Renderer_879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 39), 'Renderer', False)
            # Obtaining the member 'cmap' of a type (line 258)
            cmap_880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 39), Renderer_879, 'cmap')
            # Calling cmap(args, kwargs) (line 258)
            cmap_call_result_886 = invoke(stypy.reporting.localization.Localization(__file__, 258, 39), cmap_880, *[result_div_884], **kwargs_885)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 39), tuple_878, cmap_call_result_886)
            # Adding element type (line 258)
            
            # Call to cmap(...): (line 259)
            # Processing the call arguments (line 259)
            # Getting the type of 'p' (line 259)
            p_889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 53), 'p', False)
            # Obtaining the member 'y' of a type (line 259)
            y_890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 53), p_889, 'y')
            # Getting the type of 'nframe' (line 259)
            nframe_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 59), 'nframe', False)
            # Applying the binary operator 'div' (line 259)
            result_div_892 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 53), 'div', y_890, nframe_891)
            
            # Processing the call keyword arguments (line 259)
            kwargs_893 = {}
            # Getting the type of 'Renderer' (line 259)
            Renderer_887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 39), 'Renderer', False)
            # Obtaining the member 'cmap' of a type (line 259)
            cmap_888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 39), Renderer_887, 'cmap')
            # Calling cmap(args, kwargs) (line 259)
            cmap_call_result_894 = invoke(stypy.reporting.localization.Localization(__file__, 259, 39), cmap_888, *[result_div_892], **kwargs_893)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 39), tuple_878, cmap_call_result_894)
            # Adding element type (line 258)
            
            # Call to cmap(...): (line 260)
            # Processing the call arguments (line 260)
            # Getting the type of 'p' (line 260)
            p_897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 53), 'p', False)
            # Obtaining the member 'z' of a type (line 260)
            z_898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 53), p_897, 'z')
            # Getting the type of 'nframe' (line 260)
            nframe_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 59), 'nframe', False)
            # Applying the binary operator 'div' (line 260)
            result_div_900 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 53), 'div', z_898, nframe_899)
            
            # Processing the call keyword arguments (line 260)
            kwargs_901 = {}
            # Getting the type of 'Renderer' (line 260)
            Renderer_895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 39), 'Renderer', False)
            # Obtaining the member 'cmap' of a type (line 260)
            cmap_896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 260, 39), Renderer_895, 'cmap')
            # Calling cmap(args, kwargs) (line 260)
            cmap_call_result_902 = invoke(stypy.reporting.localization.Localization(__file__, 260, 39), cmap_896, *[result_div_900], **kwargs_901)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 39), tuple_878, cmap_call_result_902)
            
            # Applying the binary operator '%' (line 258)
            result_mod_903 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 23), '%', str_877, tuple_878)
            
            # Processing the call keyword arguments (line 258)
            kwargs_904 = {}
            # Getting the type of 'fout' (line 258)
            fout_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 12), 'fout', False)
            # Obtaining the member 'write' of a type (line 258)
            write_876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 12), fout_875, 'write')
            # Calling write(args, kwargs) (line 258)
            write_call_result_905 = invoke(stypy.reporting.localization.Localization(__file__, 258, 12), write_876, *[result_mod_903], **kwargs_904)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to close(...): (line 261)
        # Processing the call keyword arguments (line 261)
        kwargs_908 = {}
        # Getting the type of 'fout' (line 261)
        fout_906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'fout', False)
        # Obtaining the member 'close' of a type (line 261)
        close_907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 8), fout_906, 'close')
        # Calling close(args, kwargs) (line 261)
        close_call_result_909 = invoke(stypy.reporting.localization.Localization(__file__, 261, 8), close_907, *[], **kwargs_908)
        
        
        # ################# End of 'saveFrame(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'saveFrame' in the type store
        # Getting the type of 'stypy_return_type' (line 254)
        stypy_return_type_910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_910)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'saveFrame'
        return stypy_return_type_910


# Assigning a type to the variable 'Renderer' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'Renderer', Renderer)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 264, 0, False)
    
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

    
    # Assigning a Num to a Name (line 265):
    int_911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 12), 'int')
    # Assigning a type to the variable 'width' (line 265)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'width', int_911)
    
    # Assigning a Num to a Name (line 266):
    int_912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 13), 'int')
    # Assigning a type to the variable 'height' (line 266)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 4), 'height', int_912)
    
    # Assigning a Call to a Name (line 268):
    
    # Call to Scene(...): (line 268)
    # Processing the call arguments (line 268)
    
    # Call to Output(...): (line 269)
    # Processing the call arguments (line 269)
    # Getting the type of 'width' (line 269)
    width_915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 15), 'width', False)
    # Getting the type of 'height' (line 269)
    height_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'height', False)
    # Processing the call keyword arguments (line 269)
    kwargs_917 = {}
    # Getting the type of 'Output' (line 269)
    Output_914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'Output', False)
    # Calling Output(args, kwargs) (line 269)
    Output_call_result_918 = invoke(stypy.reporting.localization.Localization(__file__, 269, 8), Output_914, *[width_915, height_916], **kwargs_917)
    
    
    # Call to Camera(...): (line 271)
    # Processing the call arguments (line 271)
    
    # Call to V3(...): (line 272)
    # Processing the call arguments (line 272)
    float_921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 15), 'float')
    float_922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 20), 'float')
    float_923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 272, 26), 'float')
    # Processing the call keyword arguments (line 272)
    kwargs_924 = {}
    # Getting the type of 'V3' (line 272)
    V3_920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 12), 'V3', False)
    # Calling V3(args, kwargs) (line 272)
    V3_call_result_925 = invoke(stypy.reporting.localization.Localization(__file__, 272, 12), V3_920, *[float_921, float_922, float_923], **kwargs_924)
    
    
    # Call to V3(...): (line 273)
    # Processing the call arguments (line 273)
    float_927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 15), 'float')
    float_928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 21), 'float')
    float_929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 273, 26), 'float')
    # Processing the call keyword arguments (line 273)
    kwargs_930 = {}
    # Getting the type of 'V3' (line 273)
    V3_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'V3', False)
    # Calling V3(args, kwargs) (line 273)
    V3_call_result_931 = invoke(stypy.reporting.localization.Localization(__file__, 273, 12), V3_926, *[float_927, float_928, float_929], **kwargs_930)
    
    
    # Call to V3(...): (line 274)
    # Processing the call arguments (line 274)
    float_933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 15), 'float')
    float_934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 20), 'float')
    float_935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 274, 25), 'float')
    # Processing the call keyword arguments (line 274)
    kwargs_936 = {}
    # Getting the type of 'V3' (line 274)
    V3_932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 12), 'V3', False)
    # Calling V3(args, kwargs) (line 274)
    V3_call_result_937 = invoke(stypy.reporting.localization.Localization(__file__, 274, 12), V3_932, *[float_933, float_934, float_935], **kwargs_936)
    
    
    # Call to V3(...): (line 275)
    # Processing the call arguments (line 275)
    float_939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 15), 'float')
    float_940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 21), 'float')
    float_941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 26), 'float')
    # Processing the call keyword arguments (line 275)
    kwargs_942 = {}
    # Getting the type of 'V3' (line 275)
    V3_938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'V3', False)
    # Calling V3(args, kwargs) (line 275)
    V3_call_result_943 = invoke(stypy.reporting.localization.Localization(__file__, 275, 12), V3_938, *[float_939, float_940, float_941], **kwargs_942)
    
    # Processing the call keyword arguments (line 271)
    kwargs_944 = {}
    # Getting the type of 'Camera' (line 271)
    Camera_919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'Camera', False)
    # Calling Camera(args, kwargs) (line 271)
    Camera_call_result_945 = invoke(stypy.reporting.localization.Localization(__file__, 271, 8), Camera_919, *[V3_call_result_925, V3_call_result_931, V3_call_result_937, V3_call_result_943], **kwargs_944)
    
    
    # Obtaining an instance of the builtin type 'list' (line 278)
    list_946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 278)
    # Adding element type (line 278)
    
    # Call to Body(...): (line 282)
    # Processing the call arguments (line 282)
    
    # Call to Sphere(...): (line 282)
    # Processing the call arguments (line 282)
    
    # Call to V3(...): (line 282)
    # Processing the call arguments (line 282)
    float_950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 27), 'float')
    float_951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 32), 'float')
    float_952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 37), 'float')
    # Processing the call keyword arguments (line 282)
    kwargs_953 = {}
    # Getting the type of 'V3' (line 282)
    V3_949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 24), 'V3', False)
    # Calling V3(args, kwargs) (line 282)
    V3_call_result_954 = invoke(stypy.reporting.localization.Localization(__file__, 282, 24), V3_949, *[float_950, float_951, float_952], **kwargs_953)
    
    float_955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 43), 'float')
    # Processing the call keyword arguments (line 282)
    kwargs_956 = {}
    # Getting the type of 'Sphere' (line 282)
    Sphere_948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 17), 'Sphere', False)
    # Calling Sphere(args, kwargs) (line 282)
    Sphere_call_result_957 = invoke(stypy.reporting.localization.Localization(__file__, 282, 17), Sphere_948, *[V3_call_result_954, float_955], **kwargs_956)
    
    
    # Call to Glass(...): (line 282)
    # Processing the call arguments (line 282)
    
    # Call to V3(...): (line 282)
    # Processing the call arguments (line 282)
    float_960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 58), 'float')
    float_961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 64), 'float')
    float_962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 70), 'float')
    # Processing the call keyword arguments (line 282)
    kwargs_963 = {}
    # Getting the type of 'V3' (line 282)
    V3_959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 55), 'V3', False)
    # Calling V3(args, kwargs) (line 282)
    V3_call_result_964 = invoke(stypy.reporting.localization.Localization(__file__, 282, 55), V3_959, *[float_960, float_961, float_962], **kwargs_963)
    
    float_965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 77), 'float')
    float_966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 282, 82), 'float')
    # Processing the call keyword arguments (line 282)
    kwargs_967 = {}
    # Getting the type of 'Glass' (line 282)
    Glass_958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 49), 'Glass', False)
    # Calling Glass(args, kwargs) (line 282)
    Glass_call_result_968 = invoke(stypy.reporting.localization.Localization(__file__, 282, 49), Glass_958, *[V3_call_result_964, float_965, float_966], **kwargs_967)
    
    # Processing the call keyword arguments (line 282)
    kwargs_969 = {}
    # Getting the type of 'Body' (line 282)
    Body_947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 12), 'Body', False)
    # Calling Body(args, kwargs) (line 282)
    Body_call_result_970 = invoke(stypy.reporting.localization.Localization(__file__, 282, 12), Body_947, *[Sphere_call_result_957, Glass_call_result_968], **kwargs_969)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 8), list_946, Body_call_result_970)
    # Adding element type (line 278)
    
    # Call to Body(...): (line 284)
    # Processing the call arguments (line 284)
    
    # Call to Sphere(...): (line 284)
    # Processing the call arguments (line 284)
    
    # Call to V3(...): (line 284)
    # Processing the call arguments (line 284)
    float_974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 27), 'float')
    float_975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 33), 'float')
    float_976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 38), 'float')
    # Processing the call keyword arguments (line 284)
    kwargs_977 = {}
    # Getting the type of 'V3' (line 284)
    V3_973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 24), 'V3', False)
    # Calling V3(args, kwargs) (line 284)
    V3_call_result_978 = invoke(stypy.reporting.localization.Localization(__file__, 284, 24), V3_973, *[float_974, float_975, float_976], **kwargs_977)
    
    float_979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 44), 'float')
    # Processing the call keyword arguments (line 284)
    kwargs_980 = {}
    # Getting the type of 'Sphere' (line 284)
    Sphere_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 17), 'Sphere', False)
    # Calling Sphere(args, kwargs) (line 284)
    Sphere_call_result_981 = invoke(stypy.reporting.localization.Localization(__file__, 284, 17), Sphere_972, *[V3_call_result_978, float_979], **kwargs_980)
    
    
    # Call to Chrome(...): (line 284)
    # Processing the call arguments (line 284)
    
    # Call to V3(...): (line 284)
    # Processing the call arguments (line 284)
    float_984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 60), 'float')
    float_985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 65), 'float')
    float_986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 70), 'float')
    # Processing the call keyword arguments (line 284)
    kwargs_987 = {}
    # Getting the type of 'V3' (line 284)
    V3_983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 57), 'V3', False)
    # Calling V3(args, kwargs) (line 284)
    V3_call_result_988 = invoke(stypy.reporting.localization.Localization(__file__, 284, 57), V3_983, *[float_984, float_985, float_986], **kwargs_987)
    
    # Processing the call keyword arguments (line 284)
    kwargs_989 = {}
    # Getting the type of 'Chrome' (line 284)
    Chrome_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 50), 'Chrome', False)
    # Calling Chrome(args, kwargs) (line 284)
    Chrome_call_result_990 = invoke(stypy.reporting.localization.Localization(__file__, 284, 50), Chrome_982, *[V3_call_result_988], **kwargs_989)
    
    # Processing the call keyword arguments (line 284)
    kwargs_991 = {}
    # Getting the type of 'Body' (line 284)
    Body_971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'Body', False)
    # Calling Body(args, kwargs) (line 284)
    Body_call_result_992 = invoke(stypy.reporting.localization.Localization(__file__, 284, 12), Body_971, *[Sphere_call_result_981, Chrome_call_result_990], **kwargs_991)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 8), list_946, Body_call_result_992)
    # Adding element type (line 278)
    
    # Call to Body(...): (line 286)
    # Processing the call arguments (line 286)
    
    # Call to Sphere(...): (line 286)
    # Processing the call arguments (line 286)
    
    # Call to V3(...): (line 286)
    # Processing the call arguments (line 286)
    float_996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 27), 'float')
    float_997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 32), 'float')
    float_998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 37), 'float')
    # Processing the call keyword arguments (line 286)
    kwargs_999 = {}
    # Getting the type of 'V3' (line 286)
    V3_995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 24), 'V3', False)
    # Calling V3(args, kwargs) (line 286)
    V3_call_result_1000 = invoke(stypy.reporting.localization.Localization(__file__, 286, 24), V3_995, *[float_996, float_997, float_998], **kwargs_999)
    
    float_1001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 45), 'float')
    float_1002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 52), 'float')
    # Applying the binary operator '-' (line 286)
    result_sub_1003 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 45), '-', float_1001, float_1002)
    
    # Processing the call keyword arguments (line 286)
    kwargs_1004 = {}
    # Getting the type of 'Sphere' (line 286)
    Sphere_994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 17), 'Sphere', False)
    # Calling Sphere(args, kwargs) (line 286)
    Sphere_call_result_1005 = invoke(stypy.reporting.localization.Localization(__file__, 286, 17), Sphere_994, *[V3_call_result_1000, result_sub_1003], **kwargs_1004)
    
    
    # Call to Material(...): (line 286)
    # Processing the call arguments (line 286)
    
    # Call to V3(...): (line 286)
    # Processing the call arguments (line 286)
    float_1008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 70), 'float')
    float_1009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 75), 'float')
    float_1010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 80), 'float')
    # Processing the call keyword arguments (line 286)
    kwargs_1011 = {}
    # Getting the type of 'V3' (line 286)
    V3_1007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 67), 'V3', False)
    # Calling V3(args, kwargs) (line 286)
    V3_call_result_1012 = invoke(stypy.reporting.localization.Localization(__file__, 286, 67), V3_1007, *[float_1008, float_1009, float_1010], **kwargs_1011)
    
    # Processing the call keyword arguments (line 286)
    kwargs_1013 = {}
    # Getting the type of 'Material' (line 286)
    Material_1006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 58), 'Material', False)
    # Calling Material(args, kwargs) (line 286)
    Material_call_result_1014 = invoke(stypy.reporting.localization.Localization(__file__, 286, 58), Material_1006, *[V3_call_result_1012], **kwargs_1013)
    
    # Processing the call keyword arguments (line 286)
    kwargs_1015 = {}
    # Getting the type of 'Body' (line 286)
    Body_993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 12), 'Body', False)
    # Calling Body(args, kwargs) (line 286)
    Body_call_result_1016 = invoke(stypy.reporting.localization.Localization(__file__, 286, 12), Body_993, *[Sphere_call_result_1005, Material_call_result_1014], **kwargs_1015)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 8), list_946, Body_call_result_1016)
    # Adding element type (line 278)
    
    # Call to Body(...): (line 288)
    # Processing the call arguments (line 288)
    
    # Call to Sphere(...): (line 288)
    # Processing the call arguments (line 288)
    
    # Call to V3(...): (line 288)
    # Processing the call arguments (line 288)
    float_1020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 27), 'float')
    float_1021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 32), 'float')
    float_1022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 38), 'float')
    # Processing the call keyword arguments (line 288)
    kwargs_1023 = {}
    # Getting the type of 'V3' (line 288)
    V3_1019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 24), 'V3', False)
    # Calling V3(args, kwargs) (line 288)
    V3_call_result_1024 = invoke(stypy.reporting.localization.Localization(__file__, 288, 24), V3_1019, *[float_1020, float_1021, float_1022], **kwargs_1023)
    
    float_1025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 44), 'float')
    float_1026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 51), 'float')
    # Applying the binary operator '-' (line 288)
    result_sub_1027 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 44), '-', float_1025, float_1026)
    
    # Processing the call keyword arguments (line 288)
    kwargs_1028 = {}
    # Getting the type of 'Sphere' (line 288)
    Sphere_1018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 17), 'Sphere', False)
    # Calling Sphere(args, kwargs) (line 288)
    Sphere_call_result_1029 = invoke(stypy.reporting.localization.Localization(__file__, 288, 17), Sphere_1018, *[V3_call_result_1024, result_sub_1027], **kwargs_1028)
    
    
    # Call to Material(...): (line 288)
    # Processing the call arguments (line 288)
    
    # Call to V3(...): (line 288)
    # Processing the call arguments (line 288)
    float_1032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 69), 'float')
    float_1033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 74), 'float')
    float_1034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 79), 'float')
    # Processing the call keyword arguments (line 288)
    kwargs_1035 = {}
    # Getting the type of 'V3' (line 288)
    V3_1031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 66), 'V3', False)
    # Calling V3(args, kwargs) (line 288)
    V3_call_result_1036 = invoke(stypy.reporting.localization.Localization(__file__, 288, 66), V3_1031, *[float_1032, float_1033, float_1034], **kwargs_1035)
    
    # Processing the call keyword arguments (line 288)
    kwargs_1037 = {}
    # Getting the type of 'Material' (line 288)
    Material_1030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 57), 'Material', False)
    # Calling Material(args, kwargs) (line 288)
    Material_call_result_1038 = invoke(stypy.reporting.localization.Localization(__file__, 288, 57), Material_1030, *[V3_call_result_1036], **kwargs_1037)
    
    # Processing the call keyword arguments (line 288)
    kwargs_1039 = {}
    # Getting the type of 'Body' (line 288)
    Body_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 12), 'Body', False)
    # Calling Body(args, kwargs) (line 288)
    Body_call_result_1040 = invoke(stypy.reporting.localization.Localization(__file__, 288, 12), Body_1017, *[Sphere_call_result_1029, Material_call_result_1038], **kwargs_1039)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 8), list_946, Body_call_result_1040)
    # Adding element type (line 278)
    
    # Call to Body(...): (line 290)
    # Processing the call arguments (line 290)
    
    # Call to Sphere(...): (line 290)
    # Processing the call arguments (line 290)
    
    # Call to V3(...): (line 290)
    # Processing the call arguments (line 290)
    float_1044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 27), 'float')
    float_1045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 34), 'float')
    float_1046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 39), 'float')
    # Processing the call keyword arguments (line 290)
    kwargs_1047 = {}
    # Getting the type of 'V3' (line 290)
    V3_1043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 24), 'V3', False)
    # Calling V3(args, kwargs) (line 290)
    V3_call_result_1048 = invoke(stypy.reporting.localization.Localization(__file__, 290, 24), V3_1043, *[float_1044, float_1045, float_1046], **kwargs_1047)
    
    float_1049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 45), 'float')
    float_1050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 52), 'float')
    # Applying the binary operator '-' (line 290)
    result_sub_1051 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 45), '-', float_1049, float_1050)
    
    # Processing the call keyword arguments (line 290)
    kwargs_1052 = {}
    # Getting the type of 'Sphere' (line 290)
    Sphere_1042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 17), 'Sphere', False)
    # Calling Sphere(args, kwargs) (line 290)
    Sphere_call_result_1053 = invoke(stypy.reporting.localization.Localization(__file__, 290, 17), Sphere_1042, *[V3_call_result_1048, result_sub_1051], **kwargs_1052)
    
    
    # Call to Material(...): (line 290)
    # Processing the call arguments (line 290)
    
    # Call to V3(...): (line 290)
    # Processing the call arguments (line 290)
    float_1056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 70), 'float')
    float_1057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 75), 'float')
    float_1058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 80), 'float')
    # Processing the call keyword arguments (line 290)
    kwargs_1059 = {}
    # Getting the type of 'V3' (line 290)
    V3_1055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 67), 'V3', False)
    # Calling V3(args, kwargs) (line 290)
    V3_call_result_1060 = invoke(stypy.reporting.localization.Localization(__file__, 290, 67), V3_1055, *[float_1056, float_1057, float_1058], **kwargs_1059)
    
    # Processing the call keyword arguments (line 290)
    kwargs_1061 = {}
    # Getting the type of 'Material' (line 290)
    Material_1054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 58), 'Material', False)
    # Calling Material(args, kwargs) (line 290)
    Material_call_result_1062 = invoke(stypy.reporting.localization.Localization(__file__, 290, 58), Material_1054, *[V3_call_result_1060], **kwargs_1061)
    
    # Processing the call keyword arguments (line 290)
    kwargs_1063 = {}
    # Getting the type of 'Body' (line 290)
    Body_1041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 12), 'Body', False)
    # Calling Body(args, kwargs) (line 290)
    Body_call_result_1064 = invoke(stypy.reporting.localization.Localization(__file__, 290, 12), Body_1041, *[Sphere_call_result_1053, Material_call_result_1062], **kwargs_1063)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 8), list_946, Body_call_result_1064)
    # Adding element type (line 278)
    
    # Call to Body(...): (line 292)
    # Processing the call arguments (line 292)
    
    # Call to Sphere(...): (line 292)
    # Processing the call arguments (line 292)
    
    # Call to V3(...): (line 292)
    # Processing the call arguments (line 292)
    float_1068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 27), 'float')
    float_1069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 33), 'float')
    float_1070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 38), 'float')
    # Processing the call keyword arguments (line 292)
    kwargs_1071 = {}
    # Getting the type of 'V3' (line 292)
    V3_1067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 24), 'V3', False)
    # Calling V3(args, kwargs) (line 292)
    V3_call_result_1072 = invoke(stypy.reporting.localization.Localization(__file__, 292, 24), V3_1067, *[float_1068, float_1069, float_1070], **kwargs_1071)
    
    float_1073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 44), 'float')
    float_1074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 51), 'float')
    # Applying the binary operator '-' (line 292)
    result_sub_1075 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 44), '-', float_1073, float_1074)
    
    # Processing the call keyword arguments (line 292)
    kwargs_1076 = {}
    # Getting the type of 'Sphere' (line 292)
    Sphere_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 17), 'Sphere', False)
    # Calling Sphere(args, kwargs) (line 292)
    Sphere_call_result_1077 = invoke(stypy.reporting.localization.Localization(__file__, 292, 17), Sphere_1066, *[V3_call_result_1072, result_sub_1075], **kwargs_1076)
    
    
    # Call to Material(...): (line 292)
    # Processing the call arguments (line 292)
    
    # Call to V3(...): (line 292)
    # Processing the call arguments (line 292)
    float_1080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 69), 'float')
    float_1081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 74), 'float')
    float_1082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 79), 'float')
    # Processing the call keyword arguments (line 292)
    kwargs_1083 = {}
    # Getting the type of 'V3' (line 292)
    V3_1079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 66), 'V3', False)
    # Calling V3(args, kwargs) (line 292)
    V3_call_result_1084 = invoke(stypy.reporting.localization.Localization(__file__, 292, 66), V3_1079, *[float_1080, float_1081, float_1082], **kwargs_1083)
    
    # Processing the call keyword arguments (line 292)
    kwargs_1085 = {}
    # Getting the type of 'Material' (line 292)
    Material_1078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 57), 'Material', False)
    # Calling Material(args, kwargs) (line 292)
    Material_call_result_1086 = invoke(stypy.reporting.localization.Localization(__file__, 292, 57), Material_1078, *[V3_call_result_1084], **kwargs_1085)
    
    # Processing the call keyword arguments (line 292)
    kwargs_1087 = {}
    # Getting the type of 'Body' (line 292)
    Body_1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 12), 'Body', False)
    # Calling Body(args, kwargs) (line 292)
    Body_call_result_1088 = invoke(stypy.reporting.localization.Localization(__file__, 292, 12), Body_1065, *[Sphere_call_result_1077, Material_call_result_1086], **kwargs_1087)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 8), list_946, Body_call_result_1088)
    # Adding element type (line 278)
    
    # Call to Body(...): (line 294)
    # Processing the call arguments (line 294)
    
    # Call to Sphere(...): (line 294)
    # Processing the call arguments (line 294)
    
    # Call to V3(...): (line 294)
    # Processing the call arguments (line 294)
    float_1092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 27), 'float')
    float_1093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 32), 'float')
    float_1094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 37), 'float')
    # Processing the call keyword arguments (line 294)
    kwargs_1095 = {}
    # Getting the type of 'V3' (line 294)
    V3_1091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 24), 'V3', False)
    # Calling V3(args, kwargs) (line 294)
    V3_call_result_1096 = invoke(stypy.reporting.localization.Localization(__file__, 294, 24), V3_1091, *[float_1092, float_1093, float_1094], **kwargs_1095)
    
    float_1097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 44), 'float')
    float_1098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 51), 'float')
    # Applying the binary operator '-' (line 294)
    result_sub_1099 = python_operator(stypy.reporting.localization.Localization(__file__, 294, 44), '-', float_1097, float_1098)
    
    # Processing the call keyword arguments (line 294)
    kwargs_1100 = {}
    # Getting the type of 'Sphere' (line 294)
    Sphere_1090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 17), 'Sphere', False)
    # Calling Sphere(args, kwargs) (line 294)
    Sphere_call_result_1101 = invoke(stypy.reporting.localization.Localization(__file__, 294, 17), Sphere_1090, *[V3_call_result_1096, result_sub_1099], **kwargs_1100)
    
    
    # Call to Material(...): (line 294)
    # Processing the call arguments (line 294)
    
    # Call to V3(...): (line 294)
    # Processing the call arguments (line 294)
    float_1104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 69), 'float')
    float_1105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 74), 'float')
    float_1106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 79), 'float')
    # Processing the call keyword arguments (line 294)
    kwargs_1107 = {}
    # Getting the type of 'V3' (line 294)
    V3_1103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 66), 'V3', False)
    # Calling V3(args, kwargs) (line 294)
    V3_call_result_1108 = invoke(stypy.reporting.localization.Localization(__file__, 294, 66), V3_1103, *[float_1104, float_1105, float_1106], **kwargs_1107)
    
    
    # Call to V3(...): (line 294)
    # Processing the call arguments (line 294)
    float_1110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 88), 'float')
    float_1111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 93), 'float')
    float_1112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 99), 'float')
    # Processing the call keyword arguments (line 294)
    kwargs_1113 = {}
    # Getting the type of 'V3' (line 294)
    V3_1109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 85), 'V3', False)
    # Calling V3(args, kwargs) (line 294)
    V3_call_result_1114 = invoke(stypy.reporting.localization.Localization(__file__, 294, 85), V3_1109, *[float_1110, float_1111, float_1112], **kwargs_1113)
    
    # Processing the call keyword arguments (line 294)
    kwargs_1115 = {}
    # Getting the type of 'Material' (line 294)
    Material_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 57), 'Material', False)
    # Calling Material(args, kwargs) (line 294)
    Material_call_result_1116 = invoke(stypy.reporting.localization.Localization(__file__, 294, 57), Material_1102, *[V3_call_result_1108, V3_call_result_1114], **kwargs_1115)
    
    # Processing the call keyword arguments (line 294)
    kwargs_1117 = {}
    # Getting the type of 'Body' (line 294)
    Body_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 12), 'Body', False)
    # Calling Body(args, kwargs) (line 294)
    Body_call_result_1118 = invoke(stypy.reporting.localization.Localization(__file__, 294, 12), Body_1089, *[Sphere_call_result_1101, Material_call_result_1116], **kwargs_1117)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 8), list_946, Body_call_result_1118)
    # Adding element type (line 278)
    
    # Call to Body(...): (line 296)
    # Processing the call arguments (line 296)
    
    # Call to Sphere(...): (line 296)
    # Processing the call arguments (line 296)
    
    # Call to V3(...): (line 296)
    # Processing the call arguments (line 296)
    float_1122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 27), 'float')
    float_1123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 32), 'float')
    float_1124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 39), 'float')
    # Processing the call keyword arguments (line 296)
    kwargs_1125 = {}
    # Getting the type of 'V3' (line 296)
    V3_1121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 24), 'V3', False)
    # Calling V3(args, kwargs) (line 296)
    V3_call_result_1126 = invoke(stypy.reporting.localization.Localization(__file__, 296, 24), V3_1121, *[float_1122, float_1123, float_1124], **kwargs_1125)
    
    float_1127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 45), 'float')
    float_1128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 52), 'float')
    # Applying the binary operator '-' (line 296)
    result_sub_1129 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 45), '-', float_1127, float_1128)
    
    # Processing the call keyword arguments (line 296)
    kwargs_1130 = {}
    # Getting the type of 'Sphere' (line 296)
    Sphere_1120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 17), 'Sphere', False)
    # Calling Sphere(args, kwargs) (line 296)
    Sphere_call_result_1131 = invoke(stypy.reporting.localization.Localization(__file__, 296, 17), Sphere_1120, *[V3_call_result_1126, result_sub_1129], **kwargs_1130)
    
    
    # Call to Material(...): (line 296)
    # Processing the call arguments (line 296)
    
    # Call to V3(...): (line 296)
    # Processing the call arguments (line 296)
    float_1134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 70), 'float')
    float_1135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 75), 'float')
    float_1136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 80), 'float')
    # Processing the call keyword arguments (line 296)
    kwargs_1137 = {}
    # Getting the type of 'V3' (line 296)
    V3_1133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 67), 'V3', False)
    # Calling V3(args, kwargs) (line 296)
    V3_call_result_1138 = invoke(stypy.reporting.localization.Localization(__file__, 296, 67), V3_1133, *[float_1134, float_1135, float_1136], **kwargs_1137)
    
    # Processing the call keyword arguments (line 296)
    kwargs_1139 = {}
    # Getting the type of 'Material' (line 296)
    Material_1132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 58), 'Material', False)
    # Calling Material(args, kwargs) (line 296)
    Material_call_result_1140 = invoke(stypy.reporting.localization.Localization(__file__, 296, 58), Material_1132, *[V3_call_result_1138], **kwargs_1139)
    
    # Processing the call keyword arguments (line 296)
    kwargs_1141 = {}
    # Getting the type of 'Body' (line 296)
    Body_1119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 12), 'Body', False)
    # Calling Body(args, kwargs) (line 296)
    Body_call_result_1142 = invoke(stypy.reporting.localization.Localization(__file__, 296, 12), Body_1119, *[Sphere_call_result_1131, Material_call_result_1140], **kwargs_1141)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 8), list_946, Body_call_result_1142)
    
    # Processing the call keyword arguments (line 268)
    kwargs_1143 = {}
    # Getting the type of 'Scene' (line 268)
    Scene_913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 12), 'Scene', False)
    # Calling Scene(args, kwargs) (line 268)
    Scene_call_result_1144 = invoke(stypy.reporting.localization.Localization(__file__, 268, 12), Scene_913, *[Output_call_result_918, Camera_call_result_945, list_946], **kwargs_1143)
    
    # Assigning a type to the variable 'scene' (line 268)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 268, 4), 'scene', Scene_call_result_1144)
    
    # Assigning a Call to a Name (line 300):
    
    # Call to Renderer(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 'scene' (line 300)
    scene_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 24), 'scene', False)
    # Processing the call keyword arguments (line 300)
    kwargs_1147 = {}
    # Getting the type of 'Renderer' (line 300)
    Renderer_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 15), 'Renderer', False)
    # Calling Renderer(args, kwargs) (line 300)
    Renderer_call_result_1148 = invoke(stypy.reporting.localization.Localization(__file__, 300, 15), Renderer_1145, *[scene_1146], **kwargs_1147)
    
    # Assigning a type to the variable 'renderer' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'renderer', Renderer_call_result_1148)
    
    # Assigning a Num to a Name (line 302):
    int_1149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 13), 'int')
    # Assigning a type to the variable 'nframe' (line 302)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'nframe', int_1149)
    
    
    # Call to range(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'ITERATIONS' (line 303)
    ITERATIONS_1151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 23), 'ITERATIONS', False)
    # Processing the call keyword arguments (line 303)
    kwargs_1152 = {}
    # Getting the type of 'range' (line 303)
    range_1150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 17), 'range', False)
    # Calling range(args, kwargs) (line 303)
    range_call_result_1153 = invoke(stypy.reporting.localization.Localization(__file__, 303, 17), range_1150, *[ITERATIONS_1151], **kwargs_1152)
    
    # Testing if the for loop is going to be iterated (line 303)
    # Testing the type of a for loop iterable (line 303)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 303, 4), range_call_result_1153)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 303, 4), range_call_result_1153):
        # Getting the type of the for loop variable (line 303)
        for_loop_var_1154 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 303, 4), range_call_result_1153)
        # Assigning a type to the variable 'count' (line 303)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'count', for_loop_var_1154)
        # SSA begins for a for statement (line 303)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to iterate(...): (line 304)
        # Processing the call keyword arguments (line 304)
        kwargs_1157 = {}
        # Getting the type of 'renderer' (line 304)
        renderer_1155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 8), 'renderer', False)
        # Obtaining the member 'iterate' of a type (line 304)
        iterate_1156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 8), renderer_1155, 'iterate')
        # Calling iterate(args, kwargs) (line 304)
        iterate_call_result_1158 = invoke(stypy.reporting.localization.Localization(__file__, 304, 8), iterate_1156, *[], **kwargs_1157)
        
        
        # Getting the type of 'nframe' (line 307)
        nframe_1159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'nframe')
        int_1160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 18), 'int')
        # Applying the binary operator '+=' (line 307)
        result_iadd_1161 = python_operator(stypy.reporting.localization.Localization(__file__, 307, 8), '+=', nframe_1159, int_1160)
        # Assigning a type to the variable 'nframe' (line 307)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 307, 8), 'nframe', result_iadd_1161)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to saveFrame(...): (line 309)
    # Processing the call arguments (line 309)
    str_1164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 309, 23), 'str', 'pt.ppm')
    # Getting the type of 'nframe' (line 309)
    nframe_1165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 33), 'nframe', False)
    # Processing the call keyword arguments (line 309)
    kwargs_1166 = {}
    # Getting the type of 'renderer' (line 309)
    renderer_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 4), 'renderer', False)
    # Obtaining the member 'saveFrame' of a type (line 309)
    saveFrame_1163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 4), renderer_1162, 'saveFrame')
    # Calling saveFrame(args, kwargs) (line 309)
    saveFrame_call_result_1167 = invoke(stypy.reporting.localization.Localization(__file__, 309, 4), saveFrame_1163, *[str_1164, nframe_1165], **kwargs_1166)
    
    # Getting the type of 'True' (line 310)
    True_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 310)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 4), 'stypy_return_type', True_1168)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 264)
    stypy_return_type_1169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1169)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_1169

# Assigning a type to the variable 'run' (line 264)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 0), 'run', run)

# Call to run(...): (line 313)
# Processing the call keyword arguments (line 313)
kwargs_1171 = {}
# Getting the type of 'run' (line 313)
run_1170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 0), 'run', False)
# Calling run(args, kwargs) (line 313)
run_call_result_1172 = invoke(stypy.reporting.localization.Localization(__file__, 313, 0), run_1170, *[], **kwargs_1171)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
