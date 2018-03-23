
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # (c) Dave Griffiths
2: # --- http://www.pawfal.org/index.php?page=PyGmy
3: #
4: # ray tracer :-) (see output test.ppm)
5: 
6: from math import sin, cos, sqrt
7: import random, sys
8: import os
9: 
10: 
11: def Relative(path):
12:     return os.path.join(os.path.dirname(__file__), path)
13: 
14: 
15: def sq(a):
16:     return a * a
17: 
18: 
19: def conv_value(col):
20:     if col >= 1.0:
21:         return "255"
22:     elif col <= 0.0:
23:         return "0"
24:     else:
25:         return str(int(col * 255.0))
26: 
27: 
28: class Shaderinfo:
29:     pass
30: 
31: 
32: class vec:
33:     def __init__(self, x, y, z):
34:         self.x = float(x)
35:         self.y = float(y)
36:         self.z = float(z)
37: 
38:     def __add__(self, other):
39:         return vec(self.x + other.x, self.y + other.y, self.z + other.z)
40: 
41:     def __sub__(self, other):
42:         return vec(self.x - other.x, self.y - other.y, self.z - other.z)
43: 
44:     def __mul__(self, amount):
45:         return vec(self.x * amount, self.y * amount, self.z * amount)
46: 
47:     def __div__(self, amount):
48:         return vec(self.x / amount, self.y / amount, self.z / amount)
49: 
50:     def __neg__(self):
51:         return vec(-self.x, -self.y, -self.z)
52: 
53:     def dot(self, other):
54:         return self.x * other.x + self.y * other.y + self.z * other.z
55: 
56:     def dist(self, other):
57:         return sqrt((other.x - self.x) * (other.x - self.x) +
58:                     (other.y - self.y) * (other.y - self.y) +
59:                     (other.z - self.z) * (other.z - self.z))
60: 
61:     def sq(self):
62:         return sq(self.x) + sq(self.y) + sq(self.z)
63: 
64:     def mag(self):
65:         return self.dist(vec(0.0, 0.0, 0.0))
66: 
67:     def norm(self):
68:         mag = self.mag()
69:         if mag != 0:
70:             self.x = self.x / mag
71:             self.y = self.y / mag
72:             self.z = self.z / mag
73: 
74:     def reflect(self, normal):
75:         vdn = self.dot(normal) * 2
76:         return self - normal * vdn
77: 
78: 
79: class line:
80:     def __init__(self, start, end):
81:         self.start = start
82:         self.end = end
83: 
84:     def vec(self):
85:         return self.end - self.start
86: 
87: 
88: class renderobject:
89:     def __init__(self, shader):
90:         self.shader = shader
91: 
92: 
93: class plane(renderobject):
94:     def __init__(self, plane, dist, shader):
95:         renderobject.__init__(self, shader)
96:         self.plane = plane
97:         self.dist = dist
98: 
99:     def intersect(self, l):
100:         vd = self.plane.dot(l.vec())
101:         if vd == 0:
102:             return "none", (vec(0.0, 0.0, 0.0), vec(0.0, 0.0, 0.0))
103:         v0 = -(self.plane.dot(l.start) + self.dist)
104:         t = v0 / vd
105:         if t < 0 or t > 1:
106:             return "none", (vec(0.0, 0.0, 0.0), vec(0.0, 0.0, 0.0))
107:         return "one", (l.start + (l.vec() * t), self.plane)
108: 
109: 
110: class sphere(renderobject):
111:     def __init__(self, pos, radius, shader):
112:         renderobject.__init__(self, shader)
113:         self.pos = pos
114:         self.radius = radius
115: 
116:     def intersect(self, l):
117:         lvec = l.vec()
118:         a = sq(lvec.x) + sq(lvec.y) + sq(lvec.z)
119: 
120:         b = 2 * (lvec.x * (l.start.x - self.pos.x) + lvec.y * (l.start.y - self.pos.y) + lvec.z * (
121:                 l.start.z - self.pos.z))
122: 
123:         c = self.pos.sq() + l.start.sq() - 2 * (
124:                 self.pos.x * l.start.x + self.pos.y * l.start.y + self.pos.z * l.start.z) - sq(self.radius)
125: 
126:         i = b * b - 4 * a * c
127: 
128:         intersectiontype = "none"
129:         pos = vec(0.0, 0.0, 0.0)
130:         norm = vec(0.0, 0.0, 0.0)
131:         t = 0.0
132: 
133:         if i > 0:
134:             if i == 0:
135:                 intersectiontype = "one"
136:                 t = -b / (2 * a)
137:             else:
138:                 intersectiontype = "two"
139:                 t = (-b - sqrt(b * b - 4 * a * c)) / (2 * a)
140: 
141:             if t > 0 and t < 1:
142:                 pos = l.start + lvec * t
143:                 norm = pos - self.pos
144:                 norm.norm()
145:             else:
146:                 intersectiontype = "none"
147: 
148:         return intersectiontype, (pos, norm)
149: 
150: 
151: class light:
152:     def checkshadow(self, obj, objects, l):
153:         for ob in objects:
154:             if ob is not obj:
155:                 intersects, (pos, norm) = ob.intersect(l)
156:                 if intersects is not "none":
157:                     return 1
158:         return 0
159: 
160: 
161: class parallellight(light):
162:     def __init__(self, direction, col):
163:         direction.norm()
164:         self.direction = direction
165:         self.col = col
166: 
167:     def inshadow(self, obj, objects, pos):
168:         l = line(pos, pos + self.direction * 1000.0)
169:         return self.checkshadow(obj, objects, l)
170: 
171:     def light(self, shaderinfo):
172:         if self.inshadow(shaderinfo.thisobj, shaderinfo.objects, shaderinfo.position):
173:             return vec(0.0, 0.0, 0.0)
174:         return self.col * self.direction.dot(shaderinfo.normal)
175: 
176: 
177: class pointlight(light):
178:     def __init__(self, position, col):
179:         self.position = position
180:         self.col = col
181: 
182:     def inshadow(self, obj, objects, pos):
183:         l = line(pos, self.position)
184:         return self.checkshadow(obj, objects, l)
185: 
186:     def light(self, shaderinfo):
187:         if self.inshadow(shaderinfo.thisobj, shaderinfo.objects, shaderinfo.position):
188:             return vec(0.0, 0.0, 0.0)
189:         direction = shaderinfo.position - self.position
190:         direction.norm()
191:         direction = -direction
192:         return self.col * direction.dot(shaderinfo.normal)
193: 
194: 
195: class shader:
196:     def getreflected(self, shaderinfo):
197:         depth = shaderinfo.depth
198:         col = vec(0.0, 0.0, 0.0)
199:         if depth > 0:
200:             lray = line(shaderinfo.ray.start, shaderinfo.ray.end)  # copy.copy(shaderinfo.ray)
201:             ray = lray.vec()
202:             normal = vec(shaderinfo.normal.x, shaderinfo.normal.y, shaderinfo.normal.z)  # copy.copy(shaderinfo.normal)
203: 
204:             ray = ray.reflect(normal)
205:             reflected = line(shaderinfo.position, shaderinfo.position + ray)
206:             obj = shaderinfo.thisobj
207:             objects = shaderinfo.objects
208: 
209:             newshaderinfo = Shaderinfo()  # copy.copy(shaderinfo) # XXX
210:             newshaderinfo.thisobj = shaderinfo.thisobj
211:             newshaderinfo.objects = shaderinfo.objects
212:             newshaderinfo.lights = shaderinfo.lights
213:             newshaderinfo.position = shaderinfo.position
214:             newshaderinfo.normal = shaderinfo.normal
215: 
216:             newshaderinfo.ray = reflected
217:             newshaderinfo.depth = depth - 1
218: 
219:             # todo - depth test
220:             for ob in objects:
221:                 if ob is not obj:
222:                     intersects, (position, normal) = ob.intersect(reflected)
223:                     if intersects is not "none":
224:                         newshaderinfo.thisobj = ob
225:                         newshaderinfo.position = position
226:                         newshaderinfo.normal = normal
227:                         col = col + ob.shader.shade(newshaderinfo)
228:         return col
229: 
230:     def isoccluded(self, ray, shaderinfo):
231:         dist = ray.mag()
232:         test = line(shaderinfo.position, shaderinfo.position + ray)
233:         obj = shaderinfo.thisobj
234:         objects = shaderinfo.objects
235:         # todo - depth test
236:         for ob in objects:
237:             if ob is not obj:
238:                 intersects, (position, normal) = ob.intersect(test)
239:                 if intersects is not "none":
240:                     return 1
241:         return 0
242: 
243:     def doocclusion(self, samples, shaderinfo):
244:         # not really very scientific, or good in any way...
245:         oc = 0.0
246:         for i in xrange(samples):
247:             ray = vec(float(random.randrange(-100, 100)), float(random.randrange(-100, 100)),
248:                       float(random.randrange(-100, 100)))
249:             ray.norm()
250:             ray = ray * 2.5
251:             if self.isoccluded(ray, shaderinfo):
252:                 oc = oc + 1
253:         oc = oc / float(samples)
254:         return 1 - oc
255: 
256:     def shade(self, shaderinfo):
257:         col = vec(0.0, 0.0, 0.0)
258:         for lite in shaderinfo.lights:
259:             col = col + lite.light(shaderinfo)
260:         return col
261: 
262: 
263: class world:
264:     def __init__(self, width, height):
265:         self.lights = []
266:         self.objects = []
267:         self.cameratype = "persp"
268:         self.width = width
269:         self.height = height
270:         self.backplane = 2000.0
271:         self.imageplane = 5.0
272:         self.aspect = self.width / float(self.height)
273: 
274:     def render(self, filename):
275:         out_file = file(filename, 'w')
276:         # PPM header
277:         print >> out_file, "P3"
278:         print >> out_file, self.width, self.height
279:         print >> out_file, "256"
280:         total = self.width * self.height
281:         count = 0
282: 
283:         for sy in xrange(self.height):
284:             pixel_line = []
285:             for sx in xrange(self.width):
286:                 x = 2 * (0.5 - sx / float(self.width)) * self.aspect
287:                 y = 2 * (0.5 - sy / float(self.height))
288:                 if self.cameratype == "ortho":
289:                     ray = line(vec(x, y, 0.0), vec(x, y, self.backplane))
290:                 else:
291:                     ray = line(vec(0.0, 0.0, 0.0), vec(x, y, self.imageplane))
292:                     ray.end = ray.end * self.backplane
293: 
294:                 col = vec(0.0, 0.0, 0.0)
295:                 depth = self.backplane
296:                 shaderinfo = Shaderinfo()  # {"ray":ray,"lights":self.lights,"objects":self.objects,"depth":2}
297:                 shaderinfo.ray = ray
298:                 shaderinfo.lights = self.lights
299:                 shaderinfo.objects = self.objects
300:                 shaderinfo.depth = 2
301: 
302:                 for obj in self.objects:
303:                     intersects, (position, normal) = obj.intersect(ray)
304:                     if intersects is not "none":
305:                         if position.z < depth and position.z > 0:
306:                             depth = position.z
307:                             shaderinfo.thisobj = obj
308:                             shaderinfo.position = position
309:                             shaderinfo.normal = normal
310:                             col = obj.shader.shade(shaderinfo)
311: 
312:                 pixel_line.append(conv_value(col.x))
313:                 pixel_line.append(conv_value(col.y))
314:                 pixel_line.append(conv_value(col.z))
315:                 count = count + 1
316: 
317:             print >> out_file, " ".join(pixel_line)
318:             percentstr = str(int((count / float(total)) * 100)) + "%"
319:             # print "" + percentstr
320:         out_file.close()
321: 
322: 
323: class everythingshader(shader):
324:     def shade(self, shaderinfo):
325:         col = shader.shade(self, shaderinfo)
326:         ref = self.getreflected(shaderinfo)
327:         col = col * 0.5 + ref * 0.5
328:         return col * self.doocclusion(10, shaderinfo)
329: 
330: 
331: class spotshader(shader):
332:     def shade(self, shaderinfo):
333:         col = shader.shade(self, shaderinfo)
334:         position = shaderinfo.position
335:         jitter = sin(position.x) + cos(position.z)
336:         if jitter > 0.5:
337:             col = col / 2
338:         ref = self.getreflected(shaderinfo)
339:         return ref * 0.5 + col * 0.5 * self.doocclusion(10, shaderinfo)
340: 
341: 
342: def main():
343:     ##    if len(sys.argv) == 3:
344:     ##        nx, ny = int(sys.argv[1]), int(sys.argv[2])
345:     ##    else:
346:     nx, ny = 160, 120
347:     w = world(nx, ny)
348:     numballs = 10.0
349:     offset = vec(0.0, -5.0, 55.0)
350:     rad = 12.0
351:     radperball = (2 * 3.141592) / numballs
352: 
353:     for i in xrange(int(numballs)):
354:         x = sin(0.3 + radperball * float(i)) * rad
355:         y = cos(0.3 + radperball * float(i)) * rad
356:         w.objects.append(sphere(vec(x, 0.0, y) + offset, 2.0, everythingshader()))
357: 
358:     w.objects.append(sphere(vec(3.0, 3.0, 0.0) + offset, 5.0, everythingshader()))
359:     w.objects.append(plane(vec(0.0, 1.0, 0.0), 7.0, spotshader()))
360:     w.lights.append(parallellight(vec(1.0, 1.0, -1.0), vec(0.3, 0.9, 0.1)))
361:     w.lights.append(pointlight(vec(5.0, 100.0, -5.0), vec(0.5, 0.5, 1.0)))
362: 
363:     w.render(Relative('test.ppm'))
364: 
365: 
366: def run():
367:     main()
368:     return True
369: 
370: 
371: run()
372: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from math import sin, cos, sqrt' statement (line 6)
try:
    from math import sin, cos, sqrt

except:
    sin = UndefinedType
    cos = UndefinedType
    sqrt = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'math', None, module_type_store, ['sin', 'cos', 'sqrt'], [sin, cos, sqrt])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# Multiple import statement. import random (1/2) (line 7)
import random

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'random', random, module_type_store)
# Multiple import statement. import sys (2/2) (line 7)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import os' statement (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)


@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 11, 0, False)
    
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

    
    # Call to join(...): (line 12)
    # Processing the call arguments (line 12)
    
    # Call to dirname(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of '__file__' (line 12)
    file___29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 40), '__file__', False)
    # Processing the call keyword arguments (line 12)
    kwargs_30 = {}
    # Getting the type of 'os' (line 12)
    os_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 12)
    path_27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 24), os_26, 'path')
    # Obtaining the member 'dirname' of a type (line 12)
    dirname_28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 24), path_27, 'dirname')
    # Calling dirname(args, kwargs) (line 12)
    dirname_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 12, 24), dirname_28, *[file___29], **kwargs_30)
    
    # Getting the type of 'path' (line 12)
    path_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 51), 'path', False)
    # Processing the call keyword arguments (line 12)
    kwargs_33 = {}
    # Getting the type of 'os' (line 12)
    os_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 12)
    path_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 11), os_23, 'path')
    # Obtaining the member 'join' of a type (line 12)
    join_25 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 11), path_24, 'join')
    # Calling join(args, kwargs) (line 12)
    join_call_result_34 = invoke(stypy.reporting.localization.Localization(__file__, 12, 11), join_25, *[dirname_call_result_31, path_32], **kwargs_33)
    
    # Assigning a type to the variable 'stypy_return_type' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type', join_call_result_34)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_35

# Assigning a type to the variable 'Relative' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'Relative', Relative)

@norecursion
def sq(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sq'
    module_type_store = module_type_store.open_function_context('sq', 15, 0, False)
    
    # Passed parameters checking function
    sq.stypy_localization = localization
    sq.stypy_type_of_self = None
    sq.stypy_type_store = module_type_store
    sq.stypy_function_name = 'sq'
    sq.stypy_param_names_list = ['a']
    sq.stypy_varargs_param_name = None
    sq.stypy_kwargs_param_name = None
    sq.stypy_call_defaults = defaults
    sq.stypy_call_varargs = varargs
    sq.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sq', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sq', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sq(...)' code ##################

    # Getting the type of 'a' (line 16)
    a_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'a')
    # Getting the type of 'a' (line 16)
    a_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'a')
    # Applying the binary operator '*' (line 16)
    result_mul_38 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 11), '*', a_36, a_37)
    
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type', result_mul_38)
    
    # ################# End of 'sq(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sq' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_39)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sq'
    return stypy_return_type_39

# Assigning a type to the variable 'sq' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'sq', sq)

@norecursion
def conv_value(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'conv_value'
    module_type_store = module_type_store.open_function_context('conv_value', 19, 0, False)
    
    # Passed parameters checking function
    conv_value.stypy_localization = localization
    conv_value.stypy_type_of_self = None
    conv_value.stypy_type_store = module_type_store
    conv_value.stypy_function_name = 'conv_value'
    conv_value.stypy_param_names_list = ['col']
    conv_value.stypy_varargs_param_name = None
    conv_value.stypy_kwargs_param_name = None
    conv_value.stypy_call_defaults = defaults
    conv_value.stypy_call_varargs = varargs
    conv_value.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'conv_value', ['col'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'conv_value', localization, ['col'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'conv_value(...)' code ##################

    
    # Getting the type of 'col' (line 20)
    col_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 7), 'col')
    float_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 14), 'float')
    # Applying the binary operator '>=' (line 20)
    result_ge_42 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 7), '>=', col_40, float_41)
    
    # Testing if the type of an if condition is none (line 20)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 20, 4), result_ge_42):
        
        # Getting the type of 'col' (line 22)
        col_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), 'col')
        float_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'float')
        # Applying the binary operator '<=' (line 22)
        result_le_47 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 9), '<=', col_45, float_46)
        
        # Testing if the type of an if condition is none (line 22)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 22, 9), result_le_47):
            
            # Call to str(...): (line 25)
            # Processing the call arguments (line 25)
            
            # Call to int(...): (line 25)
            # Processing the call arguments (line 25)
            # Getting the type of 'col' (line 25)
            col_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'col', False)
            float_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'float')
            # Applying the binary operator '*' (line 25)
            result_mul_54 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 23), '*', col_52, float_53)
            
            # Processing the call keyword arguments (line 25)
            kwargs_55 = {}
            # Getting the type of 'int' (line 25)
            int_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'int', False)
            # Calling int(args, kwargs) (line 25)
            int_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), int_51, *[result_mul_54], **kwargs_55)
            
            # Processing the call keyword arguments (line 25)
            kwargs_57 = {}
            # Getting the type of 'str' (line 25)
            str_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'str', False)
            # Calling str(args, kwargs) (line 25)
            str_call_result_58 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), str_50, *[int_call_result_56], **kwargs_57)
            
            # Assigning a type to the variable 'stypy_return_type' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', str_call_result_58)
        else:
            
            # Testing the type of an if condition (line 22)
            if_condition_48 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 9), result_le_47)
            # Assigning a type to the variable 'if_condition_48' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), 'if_condition_48', if_condition_48)
            # SSA begins for if statement (line 22)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'str', '0')
            # Assigning a type to the variable 'stypy_return_type' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', str_49)
            # SSA branch for the else part of an if statement (line 22)
            module_type_store.open_ssa_branch('else')
            
            # Call to str(...): (line 25)
            # Processing the call arguments (line 25)
            
            # Call to int(...): (line 25)
            # Processing the call arguments (line 25)
            # Getting the type of 'col' (line 25)
            col_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'col', False)
            float_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'float')
            # Applying the binary operator '*' (line 25)
            result_mul_54 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 23), '*', col_52, float_53)
            
            # Processing the call keyword arguments (line 25)
            kwargs_55 = {}
            # Getting the type of 'int' (line 25)
            int_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'int', False)
            # Calling int(args, kwargs) (line 25)
            int_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), int_51, *[result_mul_54], **kwargs_55)
            
            # Processing the call keyword arguments (line 25)
            kwargs_57 = {}
            # Getting the type of 'str' (line 25)
            str_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'str', False)
            # Calling str(args, kwargs) (line 25)
            str_call_result_58 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), str_50, *[int_call_result_56], **kwargs_57)
            
            # Assigning a type to the variable 'stypy_return_type' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', str_call_result_58)
            # SSA join for if statement (line 22)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 20)
        if_condition_43 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 4), result_ge_42)
        # Assigning a type to the variable 'if_condition_43' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'if_condition_43', if_condition_43)
        # SSA begins for if statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'str', '255')
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type', str_44)
        # SSA branch for the else part of an if statement (line 20)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'col' (line 22)
        col_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), 'col')
        float_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'float')
        # Applying the binary operator '<=' (line 22)
        result_le_47 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 9), '<=', col_45, float_46)
        
        # Testing if the type of an if condition is none (line 22)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 22, 9), result_le_47):
            
            # Call to str(...): (line 25)
            # Processing the call arguments (line 25)
            
            # Call to int(...): (line 25)
            # Processing the call arguments (line 25)
            # Getting the type of 'col' (line 25)
            col_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'col', False)
            float_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'float')
            # Applying the binary operator '*' (line 25)
            result_mul_54 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 23), '*', col_52, float_53)
            
            # Processing the call keyword arguments (line 25)
            kwargs_55 = {}
            # Getting the type of 'int' (line 25)
            int_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'int', False)
            # Calling int(args, kwargs) (line 25)
            int_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), int_51, *[result_mul_54], **kwargs_55)
            
            # Processing the call keyword arguments (line 25)
            kwargs_57 = {}
            # Getting the type of 'str' (line 25)
            str_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'str', False)
            # Calling str(args, kwargs) (line 25)
            str_call_result_58 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), str_50, *[int_call_result_56], **kwargs_57)
            
            # Assigning a type to the variable 'stypy_return_type' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', str_call_result_58)
        else:
            
            # Testing the type of an if condition (line 22)
            if_condition_48 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 9), result_le_47)
            # Assigning a type to the variable 'if_condition_48' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), 'if_condition_48', if_condition_48)
            # SSA begins for if statement (line 22)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 15), 'str', '0')
            # Assigning a type to the variable 'stypy_return_type' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'stypy_return_type', str_49)
            # SSA branch for the else part of an if statement (line 22)
            module_type_store.open_ssa_branch('else')
            
            # Call to str(...): (line 25)
            # Processing the call arguments (line 25)
            
            # Call to int(...): (line 25)
            # Processing the call arguments (line 25)
            # Getting the type of 'col' (line 25)
            col_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'col', False)
            float_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'float')
            # Applying the binary operator '*' (line 25)
            result_mul_54 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 23), '*', col_52, float_53)
            
            # Processing the call keyword arguments (line 25)
            kwargs_55 = {}
            # Getting the type of 'int' (line 25)
            int_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'int', False)
            # Calling int(args, kwargs) (line 25)
            int_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 25, 19), int_51, *[result_mul_54], **kwargs_55)
            
            # Processing the call keyword arguments (line 25)
            kwargs_57 = {}
            # Getting the type of 'str' (line 25)
            str_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'str', False)
            # Calling str(args, kwargs) (line 25)
            str_call_result_58 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), str_50, *[int_call_result_56], **kwargs_57)
            
            # Assigning a type to the variable 'stypy_return_type' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', str_call_result_58)
            # SSA join for if statement (line 22)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 20)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'conv_value(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'conv_value' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'conv_value'
    return stypy_return_type_59

# Assigning a type to the variable 'conv_value' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'conv_value', conv_value)
# Declaration of the 'Shaderinfo' class

class Shaderinfo:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 28, 0, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Shaderinfo.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Shaderinfo' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'Shaderinfo', Shaderinfo)
# Declaration of the 'vec' class

class vec:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vec.__init__', ['x', 'y', 'z'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 34):
        
        # Assigning a Call to a Attribute (line 34):
        
        # Assigning a Call to a Attribute (line 34):
        
        # Call to float(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'x' (line 34)
        x_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'x', False)
        # Processing the call keyword arguments (line 34)
        kwargs_62 = {}
        # Getting the type of 'float' (line 34)
        float_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'float', False)
        # Calling float(args, kwargs) (line 34)
        float_call_result_63 = invoke(stypy.reporting.localization.Localization(__file__, 34, 17), float_60, *[x_61], **kwargs_62)
        
        # Getting the type of 'self' (line 34)
        self_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'x' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_64, 'x', float_call_result_63)
        
        # Assigning a Call to a Attribute (line 35):
        
        # Assigning a Call to a Attribute (line 35):
        
        # Assigning a Call to a Attribute (line 35):
        
        # Call to float(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'y' (line 35)
        y_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'y', False)
        # Processing the call keyword arguments (line 35)
        kwargs_67 = {}
        # Getting the type of 'float' (line 35)
        float_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'float', False)
        # Calling float(args, kwargs) (line 35)
        float_call_result_68 = invoke(stypy.reporting.localization.Localization(__file__, 35, 17), float_65, *[y_66], **kwargs_67)
        
        # Getting the type of 'self' (line 35)
        self_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'y' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_69, 'y', float_call_result_68)
        
        # Assigning a Call to a Attribute (line 36):
        
        # Assigning a Call to a Attribute (line 36):
        
        # Assigning a Call to a Attribute (line 36):
        
        # Call to float(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'z' (line 36)
        z_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'z', False)
        # Processing the call keyword arguments (line 36)
        kwargs_72 = {}
        # Getting the type of 'float' (line 36)
        float_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'float', False)
        # Calling float(args, kwargs) (line 36)
        float_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 36, 17), float_70, *[z_71], **kwargs_72)
        
        # Getting the type of 'self' (line 36)
        self_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member 'z' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_74, 'z', float_call_result_73)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vec.__add__.__dict__.__setitem__('stypy_localization', localization)
        vec.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vec.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        vec.__add__.__dict__.__setitem__('stypy_function_name', 'vec.__add__')
        vec.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        vec.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        vec.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vec.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        vec.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        vec.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vec.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vec.__add__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Call to vec(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'self' (line 39)
        self_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'self', False)
        # Obtaining the member 'x' of a type (line 39)
        x_77 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 19), self_76, 'x')
        # Getting the type of 'other' (line 39)
        other_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 28), 'other', False)
        # Obtaining the member 'x' of a type (line 39)
        x_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 28), other_78, 'x')
        # Applying the binary operator '+' (line 39)
        result_add_80 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 19), '+', x_77, x_79)
        
        # Getting the type of 'self' (line 39)
        self_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 37), 'self', False)
        # Obtaining the member 'y' of a type (line 39)
        y_82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 37), self_81, 'y')
        # Getting the type of 'other' (line 39)
        other_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 46), 'other', False)
        # Obtaining the member 'y' of a type (line 39)
        y_84 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 46), other_83, 'y')
        # Applying the binary operator '+' (line 39)
        result_add_85 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 37), '+', y_82, y_84)
        
        # Getting the type of 'self' (line 39)
        self_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 55), 'self', False)
        # Obtaining the member 'z' of a type (line 39)
        z_87 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 55), self_86, 'z')
        # Getting the type of 'other' (line 39)
        other_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 64), 'other', False)
        # Obtaining the member 'z' of a type (line 39)
        z_89 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 64), other_88, 'z')
        # Applying the binary operator '+' (line 39)
        result_add_90 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 55), '+', z_87, z_89)
        
        # Processing the call keyword arguments (line 39)
        kwargs_91 = {}
        # Getting the type of 'vec' (line 39)
        vec_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'vec', False)
        # Calling vec(args, kwargs) (line 39)
        vec_call_result_92 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), vec_75, *[result_add_80, result_add_85, result_add_90], **kwargs_91)
        
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', vec_call_result_92)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_93)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_93


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vec.__sub__.__dict__.__setitem__('stypy_localization', localization)
        vec.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vec.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        vec.__sub__.__dict__.__setitem__('stypy_function_name', 'vec.__sub__')
        vec.__sub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        vec.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        vec.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vec.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        vec.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        vec.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vec.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vec.__sub__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Call to vec(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'self' (line 42)
        self_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'self', False)
        # Obtaining the member 'x' of a type (line 42)
        x_96 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 19), self_95, 'x')
        # Getting the type of 'other' (line 42)
        other_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'other', False)
        # Obtaining the member 'x' of a type (line 42)
        x_98 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 28), other_97, 'x')
        # Applying the binary operator '-' (line 42)
        result_sub_99 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 19), '-', x_96, x_98)
        
        # Getting the type of 'self' (line 42)
        self_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 37), 'self', False)
        # Obtaining the member 'y' of a type (line 42)
        y_101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 37), self_100, 'y')
        # Getting the type of 'other' (line 42)
        other_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 46), 'other', False)
        # Obtaining the member 'y' of a type (line 42)
        y_103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 46), other_102, 'y')
        # Applying the binary operator '-' (line 42)
        result_sub_104 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 37), '-', y_101, y_103)
        
        # Getting the type of 'self' (line 42)
        self_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 55), 'self', False)
        # Obtaining the member 'z' of a type (line 42)
        z_106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 55), self_105, 'z')
        # Getting the type of 'other' (line 42)
        other_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 64), 'other', False)
        # Obtaining the member 'z' of a type (line 42)
        z_108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 64), other_107, 'z')
        # Applying the binary operator '-' (line 42)
        result_sub_109 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 55), '-', z_106, z_108)
        
        # Processing the call keyword arguments (line 42)
        kwargs_110 = {}
        # Getting the type of 'vec' (line 42)
        vec_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'vec', False)
        # Calling vec(args, kwargs) (line 42)
        vec_call_result_111 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), vec_94, *[result_sub_99, result_sub_104, result_sub_109], **kwargs_110)
        
        # Assigning a type to the variable 'stypy_return_type' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', vec_call_result_111)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_112)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_112


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vec.__mul__.__dict__.__setitem__('stypy_localization', localization)
        vec.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vec.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        vec.__mul__.__dict__.__setitem__('stypy_function_name', 'vec.__mul__')
        vec.__mul__.__dict__.__setitem__('stypy_param_names_list', ['amount'])
        vec.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        vec.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vec.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        vec.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        vec.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vec.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vec.__mul__', ['amount'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mul__', localization, ['amount'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mul__(...)' code ##################

        
        # Call to vec(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'self' (line 45)
        self_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'self', False)
        # Obtaining the member 'x' of a type (line 45)
        x_115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 19), self_114, 'x')
        # Getting the type of 'amount' (line 45)
        amount_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 28), 'amount', False)
        # Applying the binary operator '*' (line 45)
        result_mul_117 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 19), '*', x_115, amount_116)
        
        # Getting the type of 'self' (line 45)
        self_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 36), 'self', False)
        # Obtaining the member 'y' of a type (line 45)
        y_119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 36), self_118, 'y')
        # Getting the type of 'amount' (line 45)
        amount_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 45), 'amount', False)
        # Applying the binary operator '*' (line 45)
        result_mul_121 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 36), '*', y_119, amount_120)
        
        # Getting the type of 'self' (line 45)
        self_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 53), 'self', False)
        # Obtaining the member 'z' of a type (line 45)
        z_123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 53), self_122, 'z')
        # Getting the type of 'amount' (line 45)
        amount_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 62), 'amount', False)
        # Applying the binary operator '*' (line 45)
        result_mul_125 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 53), '*', z_123, amount_124)
        
        # Processing the call keyword arguments (line 45)
        kwargs_126 = {}
        # Getting the type of 'vec' (line 45)
        vec_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'vec', False)
        # Calling vec(args, kwargs) (line 45)
        vec_call_result_127 = invoke(stypy.reporting.localization.Localization(__file__, 45, 15), vec_113, *[result_mul_117, result_mul_121, result_mul_125], **kwargs_126)
        
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', vec_call_result_127)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_128


    @norecursion
    def __div__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__div__'
        module_type_store = module_type_store.open_function_context('__div__', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vec.__div__.__dict__.__setitem__('stypy_localization', localization)
        vec.__div__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vec.__div__.__dict__.__setitem__('stypy_type_store', module_type_store)
        vec.__div__.__dict__.__setitem__('stypy_function_name', 'vec.__div__')
        vec.__div__.__dict__.__setitem__('stypy_param_names_list', ['amount'])
        vec.__div__.__dict__.__setitem__('stypy_varargs_param_name', None)
        vec.__div__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vec.__div__.__dict__.__setitem__('stypy_call_defaults', defaults)
        vec.__div__.__dict__.__setitem__('stypy_call_varargs', varargs)
        vec.__div__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vec.__div__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vec.__div__', ['amount'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__div__', localization, ['amount'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__div__(...)' code ##################

        
        # Call to vec(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'self' (line 48)
        self_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'self', False)
        # Obtaining the member 'x' of a type (line 48)
        x_131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), self_130, 'x')
        # Getting the type of 'amount' (line 48)
        amount_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 'amount', False)
        # Applying the binary operator 'div' (line 48)
        result_div_133 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 19), 'div', x_131, amount_132)
        
        # Getting the type of 'self' (line 48)
        self_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 36), 'self', False)
        # Obtaining the member 'y' of a type (line 48)
        y_135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 36), self_134, 'y')
        # Getting the type of 'amount' (line 48)
        amount_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 45), 'amount', False)
        # Applying the binary operator 'div' (line 48)
        result_div_137 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 36), 'div', y_135, amount_136)
        
        # Getting the type of 'self' (line 48)
        self_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 53), 'self', False)
        # Obtaining the member 'z' of a type (line 48)
        z_139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 53), self_138, 'z')
        # Getting the type of 'amount' (line 48)
        amount_140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 62), 'amount', False)
        # Applying the binary operator 'div' (line 48)
        result_div_141 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 53), 'div', z_139, amount_140)
        
        # Processing the call keyword arguments (line 48)
        kwargs_142 = {}
        # Getting the type of 'vec' (line 48)
        vec_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'vec', False)
        # Calling vec(args, kwargs) (line 48)
        vec_call_result_143 = invoke(stypy.reporting.localization.Localization(__file__, 48, 15), vec_129, *[result_div_133, result_div_137, result_div_141], **kwargs_142)
        
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', vec_call_result_143)
        
        # ################# End of '__div__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__div__' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_144)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__div__'
        return stypy_return_type_144


    @norecursion
    def __neg__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__neg__'
        module_type_store = module_type_store.open_function_context('__neg__', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vec.__neg__.__dict__.__setitem__('stypy_localization', localization)
        vec.__neg__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vec.__neg__.__dict__.__setitem__('stypy_type_store', module_type_store)
        vec.__neg__.__dict__.__setitem__('stypy_function_name', 'vec.__neg__')
        vec.__neg__.__dict__.__setitem__('stypy_param_names_list', [])
        vec.__neg__.__dict__.__setitem__('stypy_varargs_param_name', None)
        vec.__neg__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vec.__neg__.__dict__.__setitem__('stypy_call_defaults', defaults)
        vec.__neg__.__dict__.__setitem__('stypy_call_varargs', varargs)
        vec.__neg__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vec.__neg__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vec.__neg__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__neg__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__neg__(...)' code ##################

        
        # Call to vec(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Getting the type of 'self' (line 51)
        self_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'self', False)
        # Obtaining the member 'x' of a type (line 51)
        x_147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 20), self_146, 'x')
        # Applying the 'usub' unary operator (line 51)
        result___neg___148 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 19), 'usub', x_147)
        
        
        # Getting the type of 'self' (line 51)
        self_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'self', False)
        # Obtaining the member 'y' of a type (line 51)
        y_150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 29), self_149, 'y')
        # Applying the 'usub' unary operator (line 51)
        result___neg___151 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 28), 'usub', y_150)
        
        
        # Getting the type of 'self' (line 51)
        self_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 38), 'self', False)
        # Obtaining the member 'z' of a type (line 51)
        z_153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 38), self_152, 'z')
        # Applying the 'usub' unary operator (line 51)
        result___neg___154 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 37), 'usub', z_153)
        
        # Processing the call keyword arguments (line 51)
        kwargs_155 = {}
        # Getting the type of 'vec' (line 51)
        vec_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'vec', False)
        # Calling vec(args, kwargs) (line 51)
        vec_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), vec_145, *[result___neg___148, result___neg___151, result___neg___154], **kwargs_155)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', vec_call_result_156)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_157)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_157


    @norecursion
    def dot(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dot'
        module_type_store = module_type_store.open_function_context('dot', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vec.dot.__dict__.__setitem__('stypy_localization', localization)
        vec.dot.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vec.dot.__dict__.__setitem__('stypy_type_store', module_type_store)
        vec.dot.__dict__.__setitem__('stypy_function_name', 'vec.dot')
        vec.dot.__dict__.__setitem__('stypy_param_names_list', ['other'])
        vec.dot.__dict__.__setitem__('stypy_varargs_param_name', None)
        vec.dot.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vec.dot.__dict__.__setitem__('stypy_call_defaults', defaults)
        vec.dot.__dict__.__setitem__('stypy_call_varargs', varargs)
        vec.dot.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vec.dot.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vec.dot', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dot', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dot(...)' code ##################

        # Getting the type of 'self' (line 54)
        self_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'self')
        # Obtaining the member 'x' of a type (line 54)
        x_159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 15), self_158, 'x')
        # Getting the type of 'other' (line 54)
        other_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'other')
        # Obtaining the member 'x' of a type (line 54)
        x_161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), other_160, 'x')
        # Applying the binary operator '*' (line 54)
        result_mul_162 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 15), '*', x_159, x_161)
        
        # Getting the type of 'self' (line 54)
        self_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'self')
        # Obtaining the member 'y' of a type (line 54)
        y_164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 34), self_163, 'y')
        # Getting the type of 'other' (line 54)
        other_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 43), 'other')
        # Obtaining the member 'y' of a type (line 54)
        y_166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 43), other_165, 'y')
        # Applying the binary operator '*' (line 54)
        result_mul_167 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 34), '*', y_164, y_166)
        
        # Applying the binary operator '+' (line 54)
        result_add_168 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 15), '+', result_mul_162, result_mul_167)
        
        # Getting the type of 'self' (line 54)
        self_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 53), 'self')
        # Obtaining the member 'z' of a type (line 54)
        z_170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 53), self_169, 'z')
        # Getting the type of 'other' (line 54)
        other_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 62), 'other')
        # Obtaining the member 'z' of a type (line 54)
        z_172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 62), other_171, 'z')
        # Applying the binary operator '*' (line 54)
        result_mul_173 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 53), '*', z_170, z_172)
        
        # Applying the binary operator '+' (line 54)
        result_add_174 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 51), '+', result_add_168, result_mul_173)
        
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'stypy_return_type', result_add_174)
        
        # ################# End of 'dot(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dot' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_175)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dot'
        return stypy_return_type_175


    @norecursion
    def dist(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dist'
        module_type_store = module_type_store.open_function_context('dist', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vec.dist.__dict__.__setitem__('stypy_localization', localization)
        vec.dist.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vec.dist.__dict__.__setitem__('stypy_type_store', module_type_store)
        vec.dist.__dict__.__setitem__('stypy_function_name', 'vec.dist')
        vec.dist.__dict__.__setitem__('stypy_param_names_list', ['other'])
        vec.dist.__dict__.__setitem__('stypy_varargs_param_name', None)
        vec.dist.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vec.dist.__dict__.__setitem__('stypy_call_defaults', defaults)
        vec.dist.__dict__.__setitem__('stypy_call_varargs', varargs)
        vec.dist.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vec.dist.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vec.dist', ['other'], None, None, defaults, varargs, kwargs)

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

        
        # Call to sqrt(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'other' (line 57)
        other_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'other', False)
        # Obtaining the member 'x' of a type (line 57)
        x_178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 21), other_177, 'x')
        # Getting the type of 'self' (line 57)
        self_179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 31), 'self', False)
        # Obtaining the member 'x' of a type (line 57)
        x_180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 31), self_179, 'x')
        # Applying the binary operator '-' (line 57)
        result_sub_181 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 21), '-', x_178, x_180)
        
        # Getting the type of 'other' (line 57)
        other_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 42), 'other', False)
        # Obtaining the member 'x' of a type (line 57)
        x_183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 42), other_182, 'x')
        # Getting the type of 'self' (line 57)
        self_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 52), 'self', False)
        # Obtaining the member 'x' of a type (line 57)
        x_185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 52), self_184, 'x')
        # Applying the binary operator '-' (line 57)
        result_sub_186 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 42), '-', x_183, x_185)
        
        # Applying the binary operator '*' (line 57)
        result_mul_187 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 20), '*', result_sub_181, result_sub_186)
        
        # Getting the type of 'other' (line 58)
        other_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'other', False)
        # Obtaining the member 'y' of a type (line 58)
        y_189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 21), other_188, 'y')
        # Getting the type of 'self' (line 58)
        self_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 31), 'self', False)
        # Obtaining the member 'y' of a type (line 58)
        y_191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 31), self_190, 'y')
        # Applying the binary operator '-' (line 58)
        result_sub_192 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 21), '-', y_189, y_191)
        
        # Getting the type of 'other' (line 58)
        other_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 42), 'other', False)
        # Obtaining the member 'y' of a type (line 58)
        y_194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 42), other_193, 'y')
        # Getting the type of 'self' (line 58)
        self_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 52), 'self', False)
        # Obtaining the member 'y' of a type (line 58)
        y_196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 52), self_195, 'y')
        # Applying the binary operator '-' (line 58)
        result_sub_197 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 42), '-', y_194, y_196)
        
        # Applying the binary operator '*' (line 58)
        result_mul_198 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 20), '*', result_sub_192, result_sub_197)
        
        # Applying the binary operator '+' (line 57)
        result_add_199 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 20), '+', result_mul_187, result_mul_198)
        
        # Getting the type of 'other' (line 59)
        other_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'other', False)
        # Obtaining the member 'z' of a type (line 59)
        z_201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 21), other_200, 'z')
        # Getting the type of 'self' (line 59)
        self_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 31), 'self', False)
        # Obtaining the member 'z' of a type (line 59)
        z_203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 31), self_202, 'z')
        # Applying the binary operator '-' (line 59)
        result_sub_204 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 21), '-', z_201, z_203)
        
        # Getting the type of 'other' (line 59)
        other_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 42), 'other', False)
        # Obtaining the member 'z' of a type (line 59)
        z_206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 42), other_205, 'z')
        # Getting the type of 'self' (line 59)
        self_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 52), 'self', False)
        # Obtaining the member 'z' of a type (line 59)
        z_208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 52), self_207, 'z')
        # Applying the binary operator '-' (line 59)
        result_sub_209 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 42), '-', z_206, z_208)
        
        # Applying the binary operator '*' (line 59)
        result_mul_210 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 20), '*', result_sub_204, result_sub_209)
        
        # Applying the binary operator '+' (line 58)
        result_add_211 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 60), '+', result_add_199, result_mul_210)
        
        # Processing the call keyword arguments (line 57)
        kwargs_212 = {}
        # Getting the type of 'sqrt' (line 57)
        sqrt_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 57)
        sqrt_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), sqrt_176, *[result_add_211], **kwargs_212)
        
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', sqrt_call_result_213)
        
        # ################# End of 'dist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dist' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_214)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dist'
        return stypy_return_type_214


    @norecursion
    def sq(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'sq'
        module_type_store = module_type_store.open_function_context('sq', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vec.sq.__dict__.__setitem__('stypy_localization', localization)
        vec.sq.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vec.sq.__dict__.__setitem__('stypy_type_store', module_type_store)
        vec.sq.__dict__.__setitem__('stypy_function_name', 'vec.sq')
        vec.sq.__dict__.__setitem__('stypy_param_names_list', [])
        vec.sq.__dict__.__setitem__('stypy_varargs_param_name', None)
        vec.sq.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vec.sq.__dict__.__setitem__('stypy_call_defaults', defaults)
        vec.sq.__dict__.__setitem__('stypy_call_varargs', varargs)
        vec.sq.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vec.sq.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vec.sq', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'sq', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'sq(...)' code ##################

        
        # Call to sq(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 18), 'self', False)
        # Obtaining the member 'x' of a type (line 62)
        x_217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 18), self_216, 'x')
        # Processing the call keyword arguments (line 62)
        kwargs_218 = {}
        # Getting the type of 'sq' (line 62)
        sq_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'sq', False)
        # Calling sq(args, kwargs) (line 62)
        sq_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 62, 15), sq_215, *[x_217], **kwargs_218)
        
        
        # Call to sq(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 31), 'self', False)
        # Obtaining the member 'y' of a type (line 62)
        y_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 31), self_221, 'y')
        # Processing the call keyword arguments (line 62)
        kwargs_223 = {}
        # Getting the type of 'sq' (line 62)
        sq_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'sq', False)
        # Calling sq(args, kwargs) (line 62)
        sq_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 62, 28), sq_220, *[y_222], **kwargs_223)
        
        # Applying the binary operator '+' (line 62)
        result_add_225 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 15), '+', sq_call_result_219, sq_call_result_224)
        
        
        # Call to sq(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'self' (line 62)
        self_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 44), 'self', False)
        # Obtaining the member 'z' of a type (line 62)
        z_228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 44), self_227, 'z')
        # Processing the call keyword arguments (line 62)
        kwargs_229 = {}
        # Getting the type of 'sq' (line 62)
        sq_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 41), 'sq', False)
        # Calling sq(args, kwargs) (line 62)
        sq_call_result_230 = invoke(stypy.reporting.localization.Localization(__file__, 62, 41), sq_226, *[z_228], **kwargs_229)
        
        # Applying the binary operator '+' (line 62)
        result_add_231 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 39), '+', result_add_225, sq_call_result_230)
        
        # Assigning a type to the variable 'stypy_return_type' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'stypy_return_type', result_add_231)
        
        # ################# End of 'sq(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'sq' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_232)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'sq'
        return stypy_return_type_232


    @norecursion
    def mag(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mag'
        module_type_store = module_type_store.open_function_context('mag', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vec.mag.__dict__.__setitem__('stypy_localization', localization)
        vec.mag.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vec.mag.__dict__.__setitem__('stypy_type_store', module_type_store)
        vec.mag.__dict__.__setitem__('stypy_function_name', 'vec.mag')
        vec.mag.__dict__.__setitem__('stypy_param_names_list', [])
        vec.mag.__dict__.__setitem__('stypy_varargs_param_name', None)
        vec.mag.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vec.mag.__dict__.__setitem__('stypy_call_defaults', defaults)
        vec.mag.__dict__.__setitem__('stypy_call_varargs', varargs)
        vec.mag.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vec.mag.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vec.mag', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mag', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mag(...)' code ##################

        
        # Call to dist(...): (line 65)
        # Processing the call arguments (line 65)
        
        # Call to vec(...): (line 65)
        # Processing the call arguments (line 65)
        float_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'float')
        float_237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 34), 'float')
        float_238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 39), 'float')
        # Processing the call keyword arguments (line 65)
        kwargs_239 = {}
        # Getting the type of 'vec' (line 65)
        vec_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'vec', False)
        # Calling vec(args, kwargs) (line 65)
        vec_call_result_240 = invoke(stypy.reporting.localization.Localization(__file__, 65, 25), vec_235, *[float_236, float_237, float_238], **kwargs_239)
        
        # Processing the call keyword arguments (line 65)
        kwargs_241 = {}
        # Getting the type of 'self' (line 65)
        self_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'self', False)
        # Obtaining the member 'dist' of a type (line 65)
        dist_234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 15), self_233, 'dist')
        # Calling dist(args, kwargs) (line 65)
        dist_call_result_242 = invoke(stypy.reporting.localization.Localization(__file__, 65, 15), dist_234, *[vec_call_result_240], **kwargs_241)
        
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', dist_call_result_242)
        
        # ################# End of 'mag(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mag' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_243)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mag'
        return stypy_return_type_243


    @norecursion
    def norm(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'norm'
        module_type_store = module_type_store.open_function_context('norm', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vec.norm.__dict__.__setitem__('stypy_localization', localization)
        vec.norm.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vec.norm.__dict__.__setitem__('stypy_type_store', module_type_store)
        vec.norm.__dict__.__setitem__('stypy_function_name', 'vec.norm')
        vec.norm.__dict__.__setitem__('stypy_param_names_list', [])
        vec.norm.__dict__.__setitem__('stypy_varargs_param_name', None)
        vec.norm.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vec.norm.__dict__.__setitem__('stypy_call_defaults', defaults)
        vec.norm.__dict__.__setitem__('stypy_call_varargs', varargs)
        vec.norm.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vec.norm.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vec.norm', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'norm', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'norm(...)' code ##################

        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to mag(...): (line 68)
        # Processing the call keyword arguments (line 68)
        kwargs_246 = {}
        # Getting the type of 'self' (line 68)
        self_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'self', False)
        # Obtaining the member 'mag' of a type (line 68)
        mag_245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 14), self_244, 'mag')
        # Calling mag(args, kwargs) (line 68)
        mag_call_result_247 = invoke(stypy.reporting.localization.Localization(__file__, 68, 14), mag_245, *[], **kwargs_246)
        
        # Assigning a type to the variable 'mag' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'mag', mag_call_result_247)
        
        # Getting the type of 'mag' (line 69)
        mag_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'mag')
        int_249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 18), 'int')
        # Applying the binary operator '!=' (line 69)
        result_ne_250 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 11), '!=', mag_248, int_249)
        
        # Testing if the type of an if condition is none (line 69)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 8), result_ne_250):
            pass
        else:
            
            # Testing the type of an if condition (line 69)
            if_condition_251 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), result_ne_250)
            # Assigning a type to the variable 'if_condition_251' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_251', if_condition_251)
            # SSA begins for if statement (line 69)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Attribute (line 70):
            
            # Assigning a BinOp to a Attribute (line 70):
            
            # Assigning a BinOp to a Attribute (line 70):
            # Getting the type of 'self' (line 70)
            self_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'self')
            # Obtaining the member 'x' of a type (line 70)
            x_253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 21), self_252, 'x')
            # Getting the type of 'mag' (line 70)
            mag_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 30), 'mag')
            # Applying the binary operator 'div' (line 70)
            result_div_255 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 21), 'div', x_253, mag_254)
            
            # Getting the type of 'self' (line 70)
            self_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'self')
            # Setting the type of the member 'x' of a type (line 70)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), self_256, 'x', result_div_255)
            
            # Assigning a BinOp to a Attribute (line 71):
            
            # Assigning a BinOp to a Attribute (line 71):
            
            # Assigning a BinOp to a Attribute (line 71):
            # Getting the type of 'self' (line 71)
            self_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'self')
            # Obtaining the member 'y' of a type (line 71)
            y_258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 21), self_257, 'y')
            # Getting the type of 'mag' (line 71)
            mag_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 30), 'mag')
            # Applying the binary operator 'div' (line 71)
            result_div_260 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 21), 'div', y_258, mag_259)
            
            # Getting the type of 'self' (line 71)
            self_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'self')
            # Setting the type of the member 'y' of a type (line 71)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), self_261, 'y', result_div_260)
            
            # Assigning a BinOp to a Attribute (line 72):
            
            # Assigning a BinOp to a Attribute (line 72):
            
            # Assigning a BinOp to a Attribute (line 72):
            # Getting the type of 'self' (line 72)
            self_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 21), 'self')
            # Obtaining the member 'z' of a type (line 72)
            z_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 21), self_262, 'z')
            # Getting the type of 'mag' (line 72)
            mag_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 30), 'mag')
            # Applying the binary operator 'div' (line 72)
            result_div_265 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 21), 'div', z_263, mag_264)
            
            # Getting the type of 'self' (line 72)
            self_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'self')
            # Setting the type of the member 'z' of a type (line 72)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 12), self_266, 'z', result_div_265)
            # SSA join for if statement (line 69)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'norm(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'norm' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_267)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'norm'
        return stypy_return_type_267


    @norecursion
    def reflect(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reflect'
        module_type_store = module_type_store.open_function_context('reflect', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        vec.reflect.__dict__.__setitem__('stypy_localization', localization)
        vec.reflect.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        vec.reflect.__dict__.__setitem__('stypy_type_store', module_type_store)
        vec.reflect.__dict__.__setitem__('stypy_function_name', 'vec.reflect')
        vec.reflect.__dict__.__setitem__('stypy_param_names_list', ['normal'])
        vec.reflect.__dict__.__setitem__('stypy_varargs_param_name', None)
        vec.reflect.__dict__.__setitem__('stypy_kwargs_param_name', None)
        vec.reflect.__dict__.__setitem__('stypy_call_defaults', defaults)
        vec.reflect.__dict__.__setitem__('stypy_call_varargs', varargs)
        vec.reflect.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        vec.reflect.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'vec.reflect', ['normal'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reflect', localization, ['normal'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reflect(...)' code ##################

        
        # Assigning a BinOp to a Name (line 75):
        
        # Assigning a BinOp to a Name (line 75):
        
        # Assigning a BinOp to a Name (line 75):
        
        # Call to dot(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'normal' (line 75)
        normal_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'normal', False)
        # Processing the call keyword arguments (line 75)
        kwargs_271 = {}
        # Getting the type of 'self' (line 75)
        self_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 14), 'self', False)
        # Obtaining the member 'dot' of a type (line 75)
        dot_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 14), self_268, 'dot')
        # Calling dot(args, kwargs) (line 75)
        dot_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 75, 14), dot_269, *[normal_270], **kwargs_271)
        
        int_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 33), 'int')
        # Applying the binary operator '*' (line 75)
        result_mul_274 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 14), '*', dot_call_result_272, int_273)
        
        # Assigning a type to the variable 'vdn' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'vdn', result_mul_274)
        # Getting the type of 'self' (line 76)
        self_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'self')
        # Getting the type of 'normal' (line 76)
        normal_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 22), 'normal')
        # Getting the type of 'vdn' (line 76)
        vdn_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'vdn')
        # Applying the binary operator '*' (line 76)
        result_mul_278 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 22), '*', normal_276, vdn_277)
        
        # Applying the binary operator '-' (line 76)
        result_sub_279 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 15), '-', self_275, result_mul_278)
        
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', result_sub_279)
        
        # ################# End of 'reflect(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reflect' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_280)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reflect'
        return stypy_return_type_280


# Assigning a type to the variable 'vec' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'vec', vec)
# Declaration of the 'line' class

class line:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'line.__init__', ['start', 'end'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['start', 'end'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 81):
        
        # Assigning a Name to a Attribute (line 81):
        
        # Assigning a Name to a Attribute (line 81):
        # Getting the type of 'start' (line 81)
        start_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 21), 'start')
        # Getting the type of 'self' (line 81)
        self_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'self')
        # Setting the type of the member 'start' of a type (line 81)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 8), self_282, 'start', start_281)
        
        # Assigning a Name to a Attribute (line 82):
        
        # Assigning a Name to a Attribute (line 82):
        
        # Assigning a Name to a Attribute (line 82):
        # Getting the type of 'end' (line 82)
        end_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'end')
        # Getting the type of 'self' (line 82)
        self_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self')
        # Setting the type of the member 'end' of a type (line 82)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_284, 'end', end_283)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def vec(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'vec'
        module_type_store = module_type_store.open_function_context('vec', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        line.vec.__dict__.__setitem__('stypy_localization', localization)
        line.vec.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        line.vec.__dict__.__setitem__('stypy_type_store', module_type_store)
        line.vec.__dict__.__setitem__('stypy_function_name', 'line.vec')
        line.vec.__dict__.__setitem__('stypy_param_names_list', [])
        line.vec.__dict__.__setitem__('stypy_varargs_param_name', None)
        line.vec.__dict__.__setitem__('stypy_kwargs_param_name', None)
        line.vec.__dict__.__setitem__('stypy_call_defaults', defaults)
        line.vec.__dict__.__setitem__('stypy_call_varargs', varargs)
        line.vec.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        line.vec.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'line.vec', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'vec', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'vec(...)' code ##################

        # Getting the type of 'self' (line 85)
        self_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'self')
        # Obtaining the member 'end' of a type (line 85)
        end_286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 15), self_285, 'end')
        # Getting the type of 'self' (line 85)
        self_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'self')
        # Obtaining the member 'start' of a type (line 85)
        start_288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 26), self_287, 'start')
        # Applying the binary operator '-' (line 85)
        result_sub_289 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 15), '-', end_286, start_288)
        
        # Assigning a type to the variable 'stypy_return_type' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', result_sub_289)
        
        # ################# End of 'vec(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'vec' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_290)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'vec'
        return stypy_return_type_290


# Assigning a type to the variable 'line' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'line', line)
# Declaration of the 'renderobject' class

class renderobject:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'renderobject.__init__', ['shader'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['shader'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 90):
        
        # Assigning a Name to a Attribute (line 90):
        
        # Assigning a Name to a Attribute (line 90):
        # Getting the type of 'shader' (line 90)
        shader_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 22), 'shader')
        # Getting the type of 'self' (line 90)
        self_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member 'shader' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_292, 'shader', shader_291)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'renderobject' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'renderobject', renderobject)
# Declaration of the 'plane' class
# Getting the type of 'renderobject' (line 93)
renderobject_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'renderobject')

class plane(renderobject_293, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'plane.__init__', ['plane', 'dist', 'shader'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['plane', 'dist', 'shader'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'self' (line 95)
        self_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 30), 'self', False)
        # Getting the type of 'shader' (line 95)
        shader_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'shader', False)
        # Processing the call keyword arguments (line 95)
        kwargs_298 = {}
        # Getting the type of 'renderobject' (line 95)
        renderobject_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'renderobject', False)
        # Obtaining the member '__init__' of a type (line 95)
        init___295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), renderobject_294, '__init__')
        # Calling __init__(args, kwargs) (line 95)
        init___call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), init___295, *[self_296, shader_297], **kwargs_298)
        
        
        # Assigning a Name to a Attribute (line 96):
        
        # Assigning a Name to a Attribute (line 96):
        
        # Assigning a Name to a Attribute (line 96):
        # Getting the type of 'plane' (line 96)
        plane_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 'plane')
        # Getting the type of 'self' (line 96)
        self_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'self')
        # Setting the type of the member 'plane' of a type (line 96)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), self_301, 'plane', plane_300)
        
        # Assigning a Name to a Attribute (line 97):
        
        # Assigning a Name to a Attribute (line 97):
        
        # Assigning a Name to a Attribute (line 97):
        # Getting the type of 'dist' (line 97)
        dist_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'dist')
        # Getting the type of 'self' (line 97)
        self_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self')
        # Setting the type of the member 'dist' of a type (line 97)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_303, 'dist', dist_302)
        
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
        module_type_store = module_type_store.open_function_context('intersect', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        plane.intersect.__dict__.__setitem__('stypy_localization', localization)
        plane.intersect.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        plane.intersect.__dict__.__setitem__('stypy_type_store', module_type_store)
        plane.intersect.__dict__.__setitem__('stypy_function_name', 'plane.intersect')
        plane.intersect.__dict__.__setitem__('stypy_param_names_list', ['l'])
        plane.intersect.__dict__.__setitem__('stypy_varargs_param_name', None)
        plane.intersect.__dict__.__setitem__('stypy_kwargs_param_name', None)
        plane.intersect.__dict__.__setitem__('stypy_call_defaults', defaults)
        plane.intersect.__dict__.__setitem__('stypy_call_varargs', varargs)
        plane.intersect.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        plane.intersect.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'plane.intersect', ['l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'intersect', localization, ['l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'intersect(...)' code ##################

        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to dot(...): (line 100)
        # Processing the call arguments (line 100)
        
        # Call to vec(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_309 = {}
        # Getting the type of 'l' (line 100)
        l_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 28), 'l', False)
        # Obtaining the member 'vec' of a type (line 100)
        vec_308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 28), l_307, 'vec')
        # Calling vec(args, kwargs) (line 100)
        vec_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 100, 28), vec_308, *[], **kwargs_309)
        
        # Processing the call keyword arguments (line 100)
        kwargs_311 = {}
        # Getting the type of 'self' (line 100)
        self_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 13), 'self', False)
        # Obtaining the member 'plane' of a type (line 100)
        plane_305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), self_304, 'plane')
        # Obtaining the member 'dot' of a type (line 100)
        dot_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 13), plane_305, 'dot')
        # Calling dot(args, kwargs) (line 100)
        dot_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 100, 13), dot_306, *[vec_call_result_310], **kwargs_311)
        
        # Assigning a type to the variable 'vd' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'vd', dot_call_result_312)
        
        # Getting the type of 'vd' (line 101)
        vd_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'vd')
        int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'int')
        # Applying the binary operator '==' (line 101)
        result_eq_315 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), '==', vd_313, int_314)
        
        # Testing if the type of an if condition is none (line 101)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 101, 8), result_eq_315):
            pass
        else:
            
            # Testing the type of an if condition (line 101)
            if_condition_316 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 8), result_eq_315)
            # Assigning a type to the variable 'if_condition_316' (line 101)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'if_condition_316', if_condition_316)
            # SSA begins for if statement (line 101)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'tuple' (line 102)
            tuple_317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 102)
            # Adding element type (line 102)
            str_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 19), 'str', 'none')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 19), tuple_317, str_318)
            # Adding element type (line 102)
            
            # Obtaining an instance of the builtin type 'tuple' (line 102)
            tuple_319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 28), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 102)
            # Adding element type (line 102)
            
            # Call to vec(...): (line 102)
            # Processing the call arguments (line 102)
            float_321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 32), 'float')
            float_322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 37), 'float')
            float_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 42), 'float')
            # Processing the call keyword arguments (line 102)
            kwargs_324 = {}
            # Getting the type of 'vec' (line 102)
            vec_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 28), 'vec', False)
            # Calling vec(args, kwargs) (line 102)
            vec_call_result_325 = invoke(stypy.reporting.localization.Localization(__file__, 102, 28), vec_320, *[float_321, float_322, float_323], **kwargs_324)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 28), tuple_319, vec_call_result_325)
            # Adding element type (line 102)
            
            # Call to vec(...): (line 102)
            # Processing the call arguments (line 102)
            float_327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 52), 'float')
            float_328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 57), 'float')
            float_329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 62), 'float')
            # Processing the call keyword arguments (line 102)
            kwargs_330 = {}
            # Getting the type of 'vec' (line 102)
            vec_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 48), 'vec', False)
            # Calling vec(args, kwargs) (line 102)
            vec_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 102, 48), vec_326, *[float_327, float_328, float_329], **kwargs_330)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 28), tuple_319, vec_call_result_331)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 19), tuple_317, tuple_319)
            
            # Assigning a type to the variable 'stypy_return_type' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'stypy_return_type', tuple_317)
            # SSA join for if statement (line 101)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a UnaryOp to a Name (line 103):
        
        # Assigning a UnaryOp to a Name (line 103):
        
        # Assigning a UnaryOp to a Name (line 103):
        
        
        # Call to dot(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'l' (line 103)
        l_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'l', False)
        # Obtaining the member 'start' of a type (line 103)
        start_336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 30), l_335, 'start')
        # Processing the call keyword arguments (line 103)
        kwargs_337 = {}
        # Getting the type of 'self' (line 103)
        self_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'self', False)
        # Obtaining the member 'plane' of a type (line 103)
        plane_333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), self_332, 'plane')
        # Obtaining the member 'dot' of a type (line 103)
        dot_334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), plane_333, 'dot')
        # Calling dot(args, kwargs) (line 103)
        dot_call_result_338 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), dot_334, *[start_336], **kwargs_337)
        
        # Getting the type of 'self' (line 103)
        self_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 41), 'self')
        # Obtaining the member 'dist' of a type (line 103)
        dist_340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 41), self_339, 'dist')
        # Applying the binary operator '+' (line 103)
        result_add_341 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 15), '+', dot_call_result_338, dist_340)
        
        # Applying the 'usub' unary operator (line 103)
        result___neg___342 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 13), 'usub', result_add_341)
        
        # Assigning a type to the variable 'v0' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'v0', result___neg___342)
        
        # Assigning a BinOp to a Name (line 104):
        
        # Assigning a BinOp to a Name (line 104):
        
        # Assigning a BinOp to a Name (line 104):
        # Getting the type of 'v0' (line 104)
        v0_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'v0')
        # Getting the type of 'vd' (line 104)
        vd_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'vd')
        # Applying the binary operator 'div' (line 104)
        result_div_345 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 12), 'div', v0_343, vd_344)
        
        # Assigning a type to the variable 't' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 't', result_div_345)
        
        # Evaluating a boolean operation
        
        # Getting the type of 't' (line 105)
        t_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 't')
        int_347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 15), 'int')
        # Applying the binary operator '<' (line 105)
        result_lt_348 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 11), '<', t_346, int_347)
        
        
        # Getting the type of 't' (line 105)
        t_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 't')
        int_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 24), 'int')
        # Applying the binary operator '>' (line 105)
        result_gt_351 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 20), '>', t_349, int_350)
        
        # Applying the binary operator 'or' (line 105)
        result_or_keyword_352 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 11), 'or', result_lt_348, result_gt_351)
        
        # Testing if the type of an if condition is none (line 105)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 105, 8), result_or_keyword_352):
            pass
        else:
            
            # Testing the type of an if condition (line 105)
            if_condition_353 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 8), result_or_keyword_352)
            # Assigning a type to the variable 'if_condition_353' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'if_condition_353', if_condition_353)
            # SSA begins for if statement (line 105)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'tuple' (line 106)
            tuple_354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 106)
            # Adding element type (line 106)
            str_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 19), 'str', 'none')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 19), tuple_354, str_355)
            # Adding element type (line 106)
            
            # Obtaining an instance of the builtin type 'tuple' (line 106)
            tuple_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 28), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 106)
            # Adding element type (line 106)
            
            # Call to vec(...): (line 106)
            # Processing the call arguments (line 106)
            float_358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 32), 'float')
            float_359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 37), 'float')
            float_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 42), 'float')
            # Processing the call keyword arguments (line 106)
            kwargs_361 = {}
            # Getting the type of 'vec' (line 106)
            vec_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 28), 'vec', False)
            # Calling vec(args, kwargs) (line 106)
            vec_call_result_362 = invoke(stypy.reporting.localization.Localization(__file__, 106, 28), vec_357, *[float_358, float_359, float_360], **kwargs_361)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), tuple_356, vec_call_result_362)
            # Adding element type (line 106)
            
            # Call to vec(...): (line 106)
            # Processing the call arguments (line 106)
            float_364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 52), 'float')
            float_365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 57), 'float')
            float_366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 62), 'float')
            # Processing the call keyword arguments (line 106)
            kwargs_367 = {}
            # Getting the type of 'vec' (line 106)
            vec_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 48), 'vec', False)
            # Calling vec(args, kwargs) (line 106)
            vec_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 106, 48), vec_363, *[float_364, float_365, float_366], **kwargs_367)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 28), tuple_356, vec_call_result_368)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 19), tuple_354, tuple_356)
            
            # Assigning a type to the variable 'stypy_return_type' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'stypy_return_type', tuple_354)
            # SSA join for if statement (line 105)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'tuple' (line 107)
        tuple_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 107)
        # Adding element type (line 107)
        str_370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 15), 'str', 'one')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 15), tuple_369, str_370)
        # Adding element type (line 107)
        
        # Obtaining an instance of the builtin type 'tuple' (line 107)
        tuple_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 107)
        # Adding element type (line 107)
        # Getting the type of 'l' (line 107)
        l_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'l')
        # Obtaining the member 'start' of a type (line 107)
        start_373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 23), l_372, 'start')
        
        # Call to vec(...): (line 107)
        # Processing the call keyword arguments (line 107)
        kwargs_376 = {}
        # Getting the type of 'l' (line 107)
        l_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'l', False)
        # Obtaining the member 'vec' of a type (line 107)
        vec_375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 34), l_374, 'vec')
        # Calling vec(args, kwargs) (line 107)
        vec_call_result_377 = invoke(stypy.reporting.localization.Localization(__file__, 107, 34), vec_375, *[], **kwargs_376)
        
        # Getting the type of 't' (line 107)
        t_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 44), 't')
        # Applying the binary operator '*' (line 107)
        result_mul_379 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 34), '*', vec_call_result_377, t_378)
        
        # Applying the binary operator '+' (line 107)
        result_add_380 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 23), '+', start_373, result_mul_379)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 23), tuple_371, result_add_380)
        # Adding element type (line 107)
        # Getting the type of 'self' (line 107)
        self_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 48), 'self')
        # Obtaining the member 'plane' of a type (line 107)
        plane_382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 48), self_381, 'plane')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 23), tuple_371, plane_382)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 15), tuple_369, tuple_371)
        
        # Assigning a type to the variable 'stypy_return_type' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'stypy_return_type', tuple_369)
        
        # ################# End of 'intersect(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'intersect' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_383)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'intersect'
        return stypy_return_type_383


# Assigning a type to the variable 'plane' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'plane', plane)
# Declaration of the 'sphere' class
# Getting the type of 'renderobject' (line 110)
renderobject_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 13), 'renderobject')

class sphere(renderobject_384, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 111, 4, False)
        # Assigning a type to the variable 'self' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sphere.__init__', ['pos', 'radius', 'shader'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['pos', 'radius', 'shader'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'self' (line 112)
        self_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'self', False)
        # Getting the type of 'shader' (line 112)
        shader_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 36), 'shader', False)
        # Processing the call keyword arguments (line 112)
        kwargs_389 = {}
        # Getting the type of 'renderobject' (line 112)
        renderobject_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'renderobject', False)
        # Obtaining the member '__init__' of a type (line 112)
        init___386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), renderobject_385, '__init__')
        # Calling __init__(args, kwargs) (line 112)
        init___call_result_390 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), init___386, *[self_387, shader_388], **kwargs_389)
        
        
        # Assigning a Name to a Attribute (line 113):
        
        # Assigning a Name to a Attribute (line 113):
        
        # Assigning a Name to a Attribute (line 113):
        # Getting the type of 'pos' (line 113)
        pos_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'pos')
        # Getting the type of 'self' (line 113)
        self_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self')
        # Setting the type of the member 'pos' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_392, 'pos', pos_391)
        
        # Assigning a Name to a Attribute (line 114):
        
        # Assigning a Name to a Attribute (line 114):
        
        # Assigning a Name to a Attribute (line 114):
        # Getting the type of 'radius' (line 114)
        radius_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'radius')
        # Getting the type of 'self' (line 114)
        self_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self')
        # Setting the type of the member 'radius' of a type (line 114)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_394, 'radius', radius_393)
        
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
        module_type_store = module_type_store.open_function_context('intersect', 116, 4, False)
        # Assigning a type to the variable 'self' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        sphere.intersect.__dict__.__setitem__('stypy_localization', localization)
        sphere.intersect.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        sphere.intersect.__dict__.__setitem__('stypy_type_store', module_type_store)
        sphere.intersect.__dict__.__setitem__('stypy_function_name', 'sphere.intersect')
        sphere.intersect.__dict__.__setitem__('stypy_param_names_list', ['l'])
        sphere.intersect.__dict__.__setitem__('stypy_varargs_param_name', None)
        sphere.intersect.__dict__.__setitem__('stypy_kwargs_param_name', None)
        sphere.intersect.__dict__.__setitem__('stypy_call_defaults', defaults)
        sphere.intersect.__dict__.__setitem__('stypy_call_varargs', varargs)
        sphere.intersect.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        sphere.intersect.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'sphere.intersect', ['l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'intersect', localization, ['l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'intersect(...)' code ##################

        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Assigning a Call to a Name (line 117):
        
        # Call to vec(...): (line 117)
        # Processing the call keyword arguments (line 117)
        kwargs_397 = {}
        # Getting the type of 'l' (line 117)
        l_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'l', False)
        # Obtaining the member 'vec' of a type (line 117)
        vec_396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 15), l_395, 'vec')
        # Calling vec(args, kwargs) (line 117)
        vec_call_result_398 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), vec_396, *[], **kwargs_397)
        
        # Assigning a type to the variable 'lvec' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'lvec', vec_call_result_398)
        
        # Assigning a BinOp to a Name (line 118):
        
        # Assigning a BinOp to a Name (line 118):
        
        # Assigning a BinOp to a Name (line 118):
        
        # Call to sq(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'lvec' (line 118)
        lvec_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'lvec', False)
        # Obtaining the member 'x' of a type (line 118)
        x_401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 15), lvec_400, 'x')
        # Processing the call keyword arguments (line 118)
        kwargs_402 = {}
        # Getting the type of 'sq' (line 118)
        sq_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'sq', False)
        # Calling sq(args, kwargs) (line 118)
        sq_call_result_403 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), sq_399, *[x_401], **kwargs_402)
        
        
        # Call to sq(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'lvec' (line 118)
        lvec_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'lvec', False)
        # Obtaining the member 'y' of a type (line 118)
        y_406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 28), lvec_405, 'y')
        # Processing the call keyword arguments (line 118)
        kwargs_407 = {}
        # Getting the type of 'sq' (line 118)
        sq_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 25), 'sq', False)
        # Calling sq(args, kwargs) (line 118)
        sq_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 118, 25), sq_404, *[y_406], **kwargs_407)
        
        # Applying the binary operator '+' (line 118)
        result_add_409 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 12), '+', sq_call_result_403, sq_call_result_408)
        
        
        # Call to sq(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'lvec' (line 118)
        lvec_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 41), 'lvec', False)
        # Obtaining the member 'z' of a type (line 118)
        z_412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 41), lvec_411, 'z')
        # Processing the call keyword arguments (line 118)
        kwargs_413 = {}
        # Getting the type of 'sq' (line 118)
        sq_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 38), 'sq', False)
        # Calling sq(args, kwargs) (line 118)
        sq_call_result_414 = invoke(stypy.reporting.localization.Localization(__file__, 118, 38), sq_410, *[z_412], **kwargs_413)
        
        # Applying the binary operator '+' (line 118)
        result_add_415 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 36), '+', result_add_409, sq_call_result_414)
        
        # Assigning a type to the variable 'a' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'a', result_add_415)
        
        # Assigning a BinOp to a Name (line 120):
        
        # Assigning a BinOp to a Name (line 120):
        
        # Assigning a BinOp to a Name (line 120):
        int_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 12), 'int')
        # Getting the type of 'lvec' (line 120)
        lvec_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'lvec')
        # Obtaining the member 'x' of a type (line 120)
        x_418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 17), lvec_417, 'x')
        # Getting the type of 'l' (line 120)
        l_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'l')
        # Obtaining the member 'start' of a type (line 120)
        start_420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 27), l_419, 'start')
        # Obtaining the member 'x' of a type (line 120)
        x_421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 27), start_420, 'x')
        # Getting the type of 'self' (line 120)
        self_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 39), 'self')
        # Obtaining the member 'pos' of a type (line 120)
        pos_423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 39), self_422, 'pos')
        # Obtaining the member 'x' of a type (line 120)
        x_424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 39), pos_423, 'x')
        # Applying the binary operator '-' (line 120)
        result_sub_425 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 27), '-', x_421, x_424)
        
        # Applying the binary operator '*' (line 120)
        result_mul_426 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 17), '*', x_418, result_sub_425)
        
        # Getting the type of 'lvec' (line 120)
        lvec_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 53), 'lvec')
        # Obtaining the member 'y' of a type (line 120)
        y_428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 53), lvec_427, 'y')
        # Getting the type of 'l' (line 120)
        l_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 63), 'l')
        # Obtaining the member 'start' of a type (line 120)
        start_430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 63), l_429, 'start')
        # Obtaining the member 'y' of a type (line 120)
        y_431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 63), start_430, 'y')
        # Getting the type of 'self' (line 120)
        self_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 75), 'self')
        # Obtaining the member 'pos' of a type (line 120)
        pos_433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 75), self_432, 'pos')
        # Obtaining the member 'y' of a type (line 120)
        y_434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 75), pos_433, 'y')
        # Applying the binary operator '-' (line 120)
        result_sub_435 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 63), '-', y_431, y_434)
        
        # Applying the binary operator '*' (line 120)
        result_mul_436 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 53), '*', y_428, result_sub_435)
        
        # Applying the binary operator '+' (line 120)
        result_add_437 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 17), '+', result_mul_426, result_mul_436)
        
        # Getting the type of 'lvec' (line 120)
        lvec_438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 89), 'lvec')
        # Obtaining the member 'z' of a type (line 120)
        z_439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 89), lvec_438, 'z')
        # Getting the type of 'l' (line 121)
        l_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'l')
        # Obtaining the member 'start' of a type (line 121)
        start_441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), l_440, 'start')
        # Obtaining the member 'z' of a type (line 121)
        z_442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), start_441, 'z')
        # Getting the type of 'self' (line 121)
        self_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 28), 'self')
        # Obtaining the member 'pos' of a type (line 121)
        pos_444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 28), self_443, 'pos')
        # Obtaining the member 'z' of a type (line 121)
        z_445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 28), pos_444, 'z')
        # Applying the binary operator '-' (line 121)
        result_sub_446 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 16), '-', z_442, z_445)
        
        # Applying the binary operator '*' (line 120)
        result_mul_447 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 89), '*', z_439, result_sub_446)
        
        # Applying the binary operator '+' (line 120)
        result_add_448 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 87), '+', result_add_437, result_mul_447)
        
        # Applying the binary operator '*' (line 120)
        result_mul_449 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 12), '*', int_416, result_add_448)
        
        # Assigning a type to the variable 'b' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'b', result_mul_449)
        
        # Assigning a BinOp to a Name (line 123):
        
        # Assigning a BinOp to a Name (line 123):
        
        # Assigning a BinOp to a Name (line 123):
        
        # Call to sq(...): (line 123)
        # Processing the call keyword arguments (line 123)
        kwargs_453 = {}
        # Getting the type of 'self' (line 123)
        self_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'self', False)
        # Obtaining the member 'pos' of a type (line 123)
        pos_451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), self_450, 'pos')
        # Obtaining the member 'sq' of a type (line 123)
        sq_452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), pos_451, 'sq')
        # Calling sq(args, kwargs) (line 123)
        sq_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), sq_452, *[], **kwargs_453)
        
        
        # Call to sq(...): (line 123)
        # Processing the call keyword arguments (line 123)
        kwargs_458 = {}
        # Getting the type of 'l' (line 123)
        l_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 28), 'l', False)
        # Obtaining the member 'start' of a type (line 123)
        start_456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 28), l_455, 'start')
        # Obtaining the member 'sq' of a type (line 123)
        sq_457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 28), start_456, 'sq')
        # Calling sq(args, kwargs) (line 123)
        sq_call_result_459 = invoke(stypy.reporting.localization.Localization(__file__, 123, 28), sq_457, *[], **kwargs_458)
        
        # Applying the binary operator '+' (line 123)
        result_add_460 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 12), '+', sq_call_result_454, sq_call_result_459)
        
        int_461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 43), 'int')
        # Getting the type of 'self' (line 124)
        self_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'self')
        # Obtaining the member 'pos' of a type (line 124)
        pos_463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), self_462, 'pos')
        # Obtaining the member 'x' of a type (line 124)
        x_464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 16), pos_463, 'x')
        # Getting the type of 'l' (line 124)
        l_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 29), 'l')
        # Obtaining the member 'start' of a type (line 124)
        start_466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 29), l_465, 'start')
        # Obtaining the member 'x' of a type (line 124)
        x_467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 29), start_466, 'x')
        # Applying the binary operator '*' (line 124)
        result_mul_468 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 16), '*', x_464, x_467)
        
        # Getting the type of 'self' (line 124)
        self_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 41), 'self')
        # Obtaining the member 'pos' of a type (line 124)
        pos_470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 41), self_469, 'pos')
        # Obtaining the member 'y' of a type (line 124)
        y_471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 41), pos_470, 'y')
        # Getting the type of 'l' (line 124)
        l_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 54), 'l')
        # Obtaining the member 'start' of a type (line 124)
        start_473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 54), l_472, 'start')
        # Obtaining the member 'y' of a type (line 124)
        y_474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 54), start_473, 'y')
        # Applying the binary operator '*' (line 124)
        result_mul_475 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 41), '*', y_471, y_474)
        
        # Applying the binary operator '+' (line 124)
        result_add_476 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 16), '+', result_mul_468, result_mul_475)
        
        # Getting the type of 'self' (line 124)
        self_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 66), 'self')
        # Obtaining the member 'pos' of a type (line 124)
        pos_478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 66), self_477, 'pos')
        # Obtaining the member 'z' of a type (line 124)
        z_479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 66), pos_478, 'z')
        # Getting the type of 'l' (line 124)
        l_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 79), 'l')
        # Obtaining the member 'start' of a type (line 124)
        start_481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 79), l_480, 'start')
        # Obtaining the member 'z' of a type (line 124)
        z_482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 79), start_481, 'z')
        # Applying the binary operator '*' (line 124)
        result_mul_483 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 66), '*', z_479, z_482)
        
        # Applying the binary operator '+' (line 124)
        result_add_484 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 64), '+', result_add_476, result_mul_483)
        
        # Applying the binary operator '*' (line 123)
        result_mul_485 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 43), '*', int_461, result_add_484)
        
        # Applying the binary operator '-' (line 123)
        result_sub_486 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 41), '-', result_add_460, result_mul_485)
        
        
        # Call to sq(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'self' (line 124)
        self_488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 95), 'self', False)
        # Obtaining the member 'radius' of a type (line 124)
        radius_489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 95), self_488, 'radius')
        # Processing the call keyword arguments (line 124)
        kwargs_490 = {}
        # Getting the type of 'sq' (line 124)
        sq_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 92), 'sq', False)
        # Calling sq(args, kwargs) (line 124)
        sq_call_result_491 = invoke(stypy.reporting.localization.Localization(__file__, 124, 92), sq_487, *[radius_489], **kwargs_490)
        
        # Applying the binary operator '-' (line 124)
        result_sub_492 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 90), '-', result_sub_486, sq_call_result_491)
        
        # Assigning a type to the variable 'c' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'c', result_sub_492)
        
        # Assigning a BinOp to a Name (line 126):
        
        # Assigning a BinOp to a Name (line 126):
        
        # Assigning a BinOp to a Name (line 126):
        # Getting the type of 'b' (line 126)
        b_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'b')
        # Getting the type of 'b' (line 126)
        b_494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'b')
        # Applying the binary operator '*' (line 126)
        result_mul_495 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 12), '*', b_493, b_494)
        
        int_496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 20), 'int')
        # Getting the type of 'a' (line 126)
        a_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'a')
        # Applying the binary operator '*' (line 126)
        result_mul_498 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 20), '*', int_496, a_497)
        
        # Getting the type of 'c' (line 126)
        c_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 28), 'c')
        # Applying the binary operator '*' (line 126)
        result_mul_500 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 26), '*', result_mul_498, c_499)
        
        # Applying the binary operator '-' (line 126)
        result_sub_501 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 12), '-', result_mul_495, result_mul_500)
        
        # Assigning a type to the variable 'i' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'i', result_sub_501)
        
        # Assigning a Str to a Name (line 128):
        
        # Assigning a Str to a Name (line 128):
        
        # Assigning a Str to a Name (line 128):
        str_502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 27), 'str', 'none')
        # Assigning a type to the variable 'intersectiontype' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'intersectiontype', str_502)
        
        # Assigning a Call to a Name (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Assigning a Call to a Name (line 129):
        
        # Call to vec(...): (line 129)
        # Processing the call arguments (line 129)
        float_504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 18), 'float')
        float_505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 23), 'float')
        float_506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 28), 'float')
        # Processing the call keyword arguments (line 129)
        kwargs_507 = {}
        # Getting the type of 'vec' (line 129)
        vec_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 14), 'vec', False)
        # Calling vec(args, kwargs) (line 129)
        vec_call_result_508 = invoke(stypy.reporting.localization.Localization(__file__, 129, 14), vec_503, *[float_504, float_505, float_506], **kwargs_507)
        
        # Assigning a type to the variable 'pos' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'pos', vec_call_result_508)
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Assigning a Call to a Name (line 130):
        
        # Call to vec(...): (line 130)
        # Processing the call arguments (line 130)
        float_510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 19), 'float')
        float_511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 24), 'float')
        float_512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 29), 'float')
        # Processing the call keyword arguments (line 130)
        kwargs_513 = {}
        # Getting the type of 'vec' (line 130)
        vec_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'vec', False)
        # Calling vec(args, kwargs) (line 130)
        vec_call_result_514 = invoke(stypy.reporting.localization.Localization(__file__, 130, 15), vec_509, *[float_510, float_511, float_512], **kwargs_513)
        
        # Assigning a type to the variable 'norm' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'norm', vec_call_result_514)
        
        # Assigning a Num to a Name (line 131):
        
        # Assigning a Num to a Name (line 131):
        
        # Assigning a Num to a Name (line 131):
        float_515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 12), 'float')
        # Assigning a type to the variable 't' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 't', float_515)
        
        # Getting the type of 'i' (line 133)
        i_516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'i')
        int_517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 15), 'int')
        # Applying the binary operator '>' (line 133)
        result_gt_518 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 11), '>', i_516, int_517)
        
        # Testing if the type of an if condition is none (line 133)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 133, 8), result_gt_518):
            pass
        else:
            
            # Testing the type of an if condition (line 133)
            if_condition_519 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 8), result_gt_518)
            # Assigning a type to the variable 'if_condition_519' (line 133)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'if_condition_519', if_condition_519)
            # SSA begins for if statement (line 133)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'i' (line 134)
            i_520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'i')
            int_521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 20), 'int')
            # Applying the binary operator '==' (line 134)
            result_eq_522 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), '==', i_520, int_521)
            
            # Testing if the type of an if condition is none (line 134)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 134, 12), result_eq_522):
                
                # Assigning a Str to a Name (line 138):
                
                # Assigning a Str to a Name (line 138):
                
                # Assigning a Str to a Name (line 138):
                str_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 35), 'str', 'two')
                # Assigning a type to the variable 'intersectiontype' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'intersectiontype', str_531)
                
                # Assigning a BinOp to a Name (line 139):
                
                # Assigning a BinOp to a Name (line 139):
                
                # Assigning a BinOp to a Name (line 139):
                
                # Getting the type of 'b' (line 139)
                b_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'b')
                # Applying the 'usub' unary operator (line 139)
                result___neg___533 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 21), 'usub', b_532)
                
                
                # Call to sqrt(...): (line 139)
                # Processing the call arguments (line 139)
                # Getting the type of 'b' (line 139)
                b_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 31), 'b', False)
                # Getting the type of 'b' (line 139)
                b_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 35), 'b', False)
                # Applying the binary operator '*' (line 139)
                result_mul_537 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 31), '*', b_535, b_536)
                
                int_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 39), 'int')
                # Getting the type of 'a' (line 139)
                a_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 43), 'a', False)
                # Applying the binary operator '*' (line 139)
                result_mul_540 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 39), '*', int_538, a_539)
                
                # Getting the type of 'c' (line 139)
                c_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 47), 'c', False)
                # Applying the binary operator '*' (line 139)
                result_mul_542 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 45), '*', result_mul_540, c_541)
                
                # Applying the binary operator '-' (line 139)
                result_sub_543 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 31), '-', result_mul_537, result_mul_542)
                
                # Processing the call keyword arguments (line 139)
                kwargs_544 = {}
                # Getting the type of 'sqrt' (line 139)
                sqrt_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 26), 'sqrt', False)
                # Calling sqrt(args, kwargs) (line 139)
                sqrt_call_result_545 = invoke(stypy.reporting.localization.Localization(__file__, 139, 26), sqrt_534, *[result_sub_543], **kwargs_544)
                
                # Applying the binary operator '-' (line 139)
                result_sub_546 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 21), '-', result___neg___533, sqrt_call_result_545)
                
                int_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 54), 'int')
                # Getting the type of 'a' (line 139)
                a_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 58), 'a')
                # Applying the binary operator '*' (line 139)
                result_mul_549 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 54), '*', int_547, a_548)
                
                # Applying the binary operator 'div' (line 139)
                result_div_550 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 20), 'div', result_sub_546, result_mul_549)
                
                # Assigning a type to the variable 't' (line 139)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 't', result_div_550)
            else:
                
                # Testing the type of an if condition (line 134)
                if_condition_523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 12), result_eq_522)
                # Assigning a type to the variable 'if_condition_523' (line 134)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'if_condition_523', if_condition_523)
                # SSA begins for if statement (line 134)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Str to a Name (line 135):
                
                # Assigning a Str to a Name (line 135):
                
                # Assigning a Str to a Name (line 135):
                str_524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 35), 'str', 'one')
                # Assigning a type to the variable 'intersectiontype' (line 135)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'intersectiontype', str_524)
                
                # Assigning a BinOp to a Name (line 136):
                
                # Assigning a BinOp to a Name (line 136):
                
                # Assigning a BinOp to a Name (line 136):
                
                # Getting the type of 'b' (line 136)
                b_525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 21), 'b')
                # Applying the 'usub' unary operator (line 136)
                result___neg___526 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 20), 'usub', b_525)
                
                int_527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 26), 'int')
                # Getting the type of 'a' (line 136)
                a_528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 30), 'a')
                # Applying the binary operator '*' (line 136)
                result_mul_529 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 26), '*', int_527, a_528)
                
                # Applying the binary operator 'div' (line 136)
                result_div_530 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 20), 'div', result___neg___526, result_mul_529)
                
                # Assigning a type to the variable 't' (line 136)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 't', result_div_530)
                # SSA branch for the else part of an if statement (line 134)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Str to a Name (line 138):
                
                # Assigning a Str to a Name (line 138):
                
                # Assigning a Str to a Name (line 138):
                str_531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 35), 'str', 'two')
                # Assigning a type to the variable 'intersectiontype' (line 138)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'intersectiontype', str_531)
                
                # Assigning a BinOp to a Name (line 139):
                
                # Assigning a BinOp to a Name (line 139):
                
                # Assigning a BinOp to a Name (line 139):
                
                # Getting the type of 'b' (line 139)
                b_532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'b')
                # Applying the 'usub' unary operator (line 139)
                result___neg___533 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 21), 'usub', b_532)
                
                
                # Call to sqrt(...): (line 139)
                # Processing the call arguments (line 139)
                # Getting the type of 'b' (line 139)
                b_535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 31), 'b', False)
                # Getting the type of 'b' (line 139)
                b_536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 35), 'b', False)
                # Applying the binary operator '*' (line 139)
                result_mul_537 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 31), '*', b_535, b_536)
                
                int_538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 39), 'int')
                # Getting the type of 'a' (line 139)
                a_539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 43), 'a', False)
                # Applying the binary operator '*' (line 139)
                result_mul_540 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 39), '*', int_538, a_539)
                
                # Getting the type of 'c' (line 139)
                c_541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 47), 'c', False)
                # Applying the binary operator '*' (line 139)
                result_mul_542 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 45), '*', result_mul_540, c_541)
                
                # Applying the binary operator '-' (line 139)
                result_sub_543 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 31), '-', result_mul_537, result_mul_542)
                
                # Processing the call keyword arguments (line 139)
                kwargs_544 = {}
                # Getting the type of 'sqrt' (line 139)
                sqrt_534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 26), 'sqrt', False)
                # Calling sqrt(args, kwargs) (line 139)
                sqrt_call_result_545 = invoke(stypy.reporting.localization.Localization(__file__, 139, 26), sqrt_534, *[result_sub_543], **kwargs_544)
                
                # Applying the binary operator '-' (line 139)
                result_sub_546 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 21), '-', result___neg___533, sqrt_call_result_545)
                
                int_547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 54), 'int')
                # Getting the type of 'a' (line 139)
                a_548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 58), 'a')
                # Applying the binary operator '*' (line 139)
                result_mul_549 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 54), '*', int_547, a_548)
                
                # Applying the binary operator 'div' (line 139)
                result_div_550 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 20), 'div', result_sub_546, result_mul_549)
                
                # Assigning a type to the variable 't' (line 139)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 't', result_div_550)
                # SSA join for if statement (line 134)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Evaluating a boolean operation
            
            # Getting the type of 't' (line 141)
            t_551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 't')
            int_552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 19), 'int')
            # Applying the binary operator '>' (line 141)
            result_gt_553 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 15), '>', t_551, int_552)
            
            
            # Getting the type of 't' (line 141)
            t_554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 25), 't')
            int_555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 29), 'int')
            # Applying the binary operator '<' (line 141)
            result_lt_556 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 25), '<', t_554, int_555)
            
            # Applying the binary operator 'and' (line 141)
            result_and_keyword_557 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 15), 'and', result_gt_553, result_lt_556)
            
            # Testing if the type of an if condition is none (line 141)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 141, 12), result_and_keyword_557):
                
                # Assigning a Str to a Name (line 146):
                
                # Assigning a Str to a Name (line 146):
                
                # Assigning a Str to a Name (line 146):
                str_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 35), 'str', 'none')
                # Assigning a type to the variable 'intersectiontype' (line 146)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'intersectiontype', str_573)
            else:
                
                # Testing the type of an if condition (line 141)
                if_condition_558 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 12), result_and_keyword_557)
                # Assigning a type to the variable 'if_condition_558' (line 141)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'if_condition_558', if_condition_558)
                # SSA begins for if statement (line 141)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 142):
                
                # Assigning a BinOp to a Name (line 142):
                
                # Assigning a BinOp to a Name (line 142):
                # Getting the type of 'l' (line 142)
                l_559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 22), 'l')
                # Obtaining the member 'start' of a type (line 142)
                start_560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 22), l_559, 'start')
                # Getting the type of 'lvec' (line 142)
                lvec_561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 32), 'lvec')
                # Getting the type of 't' (line 142)
                t_562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 39), 't')
                # Applying the binary operator '*' (line 142)
                result_mul_563 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 32), '*', lvec_561, t_562)
                
                # Applying the binary operator '+' (line 142)
                result_add_564 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 22), '+', start_560, result_mul_563)
                
                # Assigning a type to the variable 'pos' (line 142)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'pos', result_add_564)
                
                # Assigning a BinOp to a Name (line 143):
                
                # Assigning a BinOp to a Name (line 143):
                
                # Assigning a BinOp to a Name (line 143):
                # Getting the type of 'pos' (line 143)
                pos_565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 23), 'pos')
                # Getting the type of 'self' (line 143)
                self_566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'self')
                # Obtaining the member 'pos' of a type (line 143)
                pos_567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 29), self_566, 'pos')
                # Applying the binary operator '-' (line 143)
                result_sub_568 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 23), '-', pos_565, pos_567)
                
                # Assigning a type to the variable 'norm' (line 143)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'norm', result_sub_568)
                
                # Call to norm(...): (line 144)
                # Processing the call keyword arguments (line 144)
                kwargs_571 = {}
                # Getting the type of 'norm' (line 144)
                norm_569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'norm', False)
                # Obtaining the member 'norm' of a type (line 144)
                norm_570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 16), norm_569, 'norm')
                # Calling norm(args, kwargs) (line 144)
                norm_call_result_572 = invoke(stypy.reporting.localization.Localization(__file__, 144, 16), norm_570, *[], **kwargs_571)
                
                # SSA branch for the else part of an if statement (line 141)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Str to a Name (line 146):
                
                # Assigning a Str to a Name (line 146):
                
                # Assigning a Str to a Name (line 146):
                str_573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 35), 'str', 'none')
                # Assigning a type to the variable 'intersectiontype' (line 146)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 16), 'intersectiontype', str_573)
                # SSA join for if statement (line 141)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 133)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        # Getting the type of 'intersectiontype' (line 148)
        intersectiontype_575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 15), 'intersectiontype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 15), tuple_574, intersectiontype_575)
        # Adding element type (line 148)
        
        # Obtaining an instance of the builtin type 'tuple' (line 148)
        tuple_576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 148)
        # Adding element type (line 148)
        # Getting the type of 'pos' (line 148)
        pos_577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 34), 'pos')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 34), tuple_576, pos_577)
        # Adding element type (line 148)
        # Getting the type of 'norm' (line 148)
        norm_578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 39), 'norm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 34), tuple_576, norm_578)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 15), tuple_574, tuple_576)
        
        # Assigning a type to the variable 'stypy_return_type' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'stypy_return_type', tuple_574)
        
        # ################# End of 'intersect(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'intersect' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_579)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'intersect'
        return stypy_return_type_579


# Assigning a type to the variable 'sphere' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'sphere', sphere)
# Declaration of the 'light' class

class light:

    @norecursion
    def checkshadow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'checkshadow'
        module_type_store = module_type_store.open_function_context('checkshadow', 152, 4, False)
        # Assigning a type to the variable 'self' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        light.checkshadow.__dict__.__setitem__('stypy_localization', localization)
        light.checkshadow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        light.checkshadow.__dict__.__setitem__('stypy_type_store', module_type_store)
        light.checkshadow.__dict__.__setitem__('stypy_function_name', 'light.checkshadow')
        light.checkshadow.__dict__.__setitem__('stypy_param_names_list', ['obj', 'objects', 'l'])
        light.checkshadow.__dict__.__setitem__('stypy_varargs_param_name', None)
        light.checkshadow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        light.checkshadow.__dict__.__setitem__('stypy_call_defaults', defaults)
        light.checkshadow.__dict__.__setitem__('stypy_call_varargs', varargs)
        light.checkshadow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        light.checkshadow.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'light.checkshadow', ['obj', 'objects', 'l'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'checkshadow', localization, ['obj', 'objects', 'l'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'checkshadow(...)' code ##################

        
        # Getting the type of 'objects' (line 153)
        objects_580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'objects')
        # Assigning a type to the variable 'objects_580' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'objects_580', objects_580)
        # Testing if the for loop is going to be iterated (line 153)
        # Testing the type of a for loop iterable (line 153)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 153, 8), objects_580)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 153, 8), objects_580):
            # Getting the type of the for loop variable (line 153)
            for_loop_var_581 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 153, 8), objects_580)
            # Assigning a type to the variable 'ob' (line 153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'ob', for_loop_var_581)
            # SSA begins for a for statement (line 153)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'ob' (line 154)
            ob_582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 15), 'ob')
            # Getting the type of 'obj' (line 154)
            obj_583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), 'obj')
            # Applying the binary operator 'isnot' (line 154)
            result_is_not_584 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 15), 'isnot', ob_582, obj_583)
            
            # Testing if the type of an if condition is none (line 154)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 154, 12), result_is_not_584):
                pass
            else:
                
                # Testing the type of an if condition (line 154)
                if_condition_585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 12), result_is_not_584)
                # Assigning a type to the variable 'if_condition_585' (line 154)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'if_condition_585', if_condition_585)
                # SSA begins for if statement (line 154)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 155):
                
                # Assigning a Call to a Name:
                
                # Assigning a Call to a Name:
                
                # Call to intersect(...): (line 155)
                # Processing the call arguments (line 155)
                # Getting the type of 'l' (line 155)
                l_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 55), 'l', False)
                # Processing the call keyword arguments (line 155)
                kwargs_589 = {}
                # Getting the type of 'ob' (line 155)
                ob_586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 42), 'ob', False)
                # Obtaining the member 'intersect' of a type (line 155)
                intersect_587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 42), ob_586, 'intersect')
                # Calling intersect(args, kwargs) (line 155)
                intersect_call_result_590 = invoke(stypy.reporting.localization.Localization(__file__, 155, 42), intersect_587, *[l_588], **kwargs_589)
                
                # Assigning a type to the variable 'call_assignment_1' (line 155)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'call_assignment_1', intersect_call_result_590)
                
                # Assigning a Call to a Name (line 155):
                
                # Assigning a Call to a Name (line 155):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_1' (line 155)
                call_assignment_1_591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'call_assignment_1', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_592 = stypy_get_value_from_tuple(call_assignment_1_591, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_2' (line 155)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'call_assignment_2', stypy_get_value_from_tuple_call_result_592)
                
                # Assigning a Name to a Name (line 155):
                
                # Assigning a Name to a Name (line 155):
                # Getting the type of 'call_assignment_2' (line 155)
                call_assignment_2_593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'call_assignment_2')
                # Assigning a type to the variable 'intersects' (line 155)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'intersects', call_assignment_2_593)
                
                # Assigning a Call to a Name (line 155):
                
                # Assigning a Call to a Name (line 155):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_1' (line 155)
                call_assignment_1_594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'call_assignment_1', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_595 = stypy_get_value_from_tuple(call_assignment_1_594, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_3' (line 155)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'call_assignment_3', stypy_get_value_from_tuple_call_result_595)
                
                # Assigning a Name to a Tuple (line 155):
                
                # Assigning a Subscript to a Name (line 155):
                
                # Obtaining the type of the subscript
                int_596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 16), 'int')
                # Getting the type of 'call_assignment_3' (line 155)
                call_assignment_3_597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'call_assignment_3')
                # Obtaining the member '__getitem__' of a type (line 155)
                getitem___598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 16), call_assignment_3_597, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 155)
                subscript_call_result_599 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), getitem___598, int_596)
                
                # Assigning a type to the variable 'tuple_var_assignment_15' (line 155)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'tuple_var_assignment_15', subscript_call_result_599)
                
                # Assigning a Subscript to a Name (line 155):
                
                # Obtaining the type of the subscript
                int_600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 16), 'int')
                # Getting the type of 'call_assignment_3' (line 155)
                call_assignment_3_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'call_assignment_3')
                # Obtaining the member '__getitem__' of a type (line 155)
                getitem___602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 16), call_assignment_3_601, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 155)
                subscript_call_result_603 = invoke(stypy.reporting.localization.Localization(__file__, 155, 16), getitem___602, int_600)
                
                # Assigning a type to the variable 'tuple_var_assignment_16' (line 155)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'tuple_var_assignment_16', subscript_call_result_603)
                
                # Assigning a Name to a Name (line 155):
                # Getting the type of 'tuple_var_assignment_15' (line 155)
                tuple_var_assignment_15_604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'tuple_var_assignment_15')
                # Assigning a type to the variable 'pos' (line 155)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 29), 'pos', tuple_var_assignment_15_604)
                
                # Assigning a Name to a Name (line 155):
                # Getting the type of 'tuple_var_assignment_16' (line 155)
                tuple_var_assignment_16_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 16), 'tuple_var_assignment_16')
                # Assigning a type to the variable 'norm' (line 155)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), 'norm', tuple_var_assignment_16_605)
                
                # Getting the type of 'intersects' (line 156)
                intersects_606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 19), 'intersects')
                str_607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 37), 'str', 'none')
                # Applying the binary operator 'isnot' (line 156)
                result_is_not_608 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 19), 'isnot', intersects_606, str_607)
                
                # Testing if the type of an if condition is none (line 156)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 156, 16), result_is_not_608):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 156)
                    if_condition_609 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 16), result_is_not_608)
                    # Assigning a type to the variable 'if_condition_609' (line 156)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'if_condition_609', if_condition_609)
                    # SSA begins for if statement (line 156)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    int_610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 27), 'int')
                    # Assigning a type to the variable 'stypy_return_type' (line 157)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'stypy_return_type', int_610)
                    # SSA join for if statement (line 156)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 154)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        int_611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'stypy_return_type', int_611)
        
        # ################# End of 'checkshadow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'checkshadow' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_612)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'checkshadow'
        return stypy_return_type_612


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 151, 0, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'light.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'light' (line 151)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 0), 'light', light)
# Declaration of the 'parallellight' class
# Getting the type of 'light' (line 161)
light_613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'light')

class parallellight(light_613, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 162, 4, False)
        # Assigning a type to the variable 'self' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'parallellight.__init__', ['direction', 'col'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['direction', 'col'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to norm(...): (line 163)
        # Processing the call keyword arguments (line 163)
        kwargs_616 = {}
        # Getting the type of 'direction' (line 163)
        direction_614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'direction', False)
        # Obtaining the member 'norm' of a type (line 163)
        norm_615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), direction_614, 'norm')
        # Calling norm(args, kwargs) (line 163)
        norm_call_result_617 = invoke(stypy.reporting.localization.Localization(__file__, 163, 8), norm_615, *[], **kwargs_616)
        
        
        # Assigning a Name to a Attribute (line 164):
        
        # Assigning a Name to a Attribute (line 164):
        
        # Assigning a Name to a Attribute (line 164):
        # Getting the type of 'direction' (line 164)
        direction_618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 25), 'direction')
        # Getting the type of 'self' (line 164)
        self_619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self')
        # Setting the type of the member 'direction' of a type (line 164)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_619, 'direction', direction_618)
        
        # Assigning a Name to a Attribute (line 165):
        
        # Assigning a Name to a Attribute (line 165):
        
        # Assigning a Name to a Attribute (line 165):
        # Getting the type of 'col' (line 165)
        col_620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 19), 'col')
        # Getting the type of 'self' (line 165)
        self_621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'self')
        # Setting the type of the member 'col' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), self_621, 'col', col_620)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def inshadow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inshadow'
        module_type_store = module_type_store.open_function_context('inshadow', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        parallellight.inshadow.__dict__.__setitem__('stypy_localization', localization)
        parallellight.inshadow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        parallellight.inshadow.__dict__.__setitem__('stypy_type_store', module_type_store)
        parallellight.inshadow.__dict__.__setitem__('stypy_function_name', 'parallellight.inshadow')
        parallellight.inshadow.__dict__.__setitem__('stypy_param_names_list', ['obj', 'objects', 'pos'])
        parallellight.inshadow.__dict__.__setitem__('stypy_varargs_param_name', None)
        parallellight.inshadow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        parallellight.inshadow.__dict__.__setitem__('stypy_call_defaults', defaults)
        parallellight.inshadow.__dict__.__setitem__('stypy_call_varargs', varargs)
        parallellight.inshadow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        parallellight.inshadow.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'parallellight.inshadow', ['obj', 'objects', 'pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'inshadow', localization, ['obj', 'objects', 'pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'inshadow(...)' code ##################

        
        # Assigning a Call to a Name (line 168):
        
        # Assigning a Call to a Name (line 168):
        
        # Assigning a Call to a Name (line 168):
        
        # Call to line(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'pos' (line 168)
        pos_623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 17), 'pos', False)
        # Getting the type of 'pos' (line 168)
        pos_624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'pos', False)
        # Getting the type of 'self' (line 168)
        self_625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 28), 'self', False)
        # Obtaining the member 'direction' of a type (line 168)
        direction_626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 28), self_625, 'direction')
        float_627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 45), 'float')
        # Applying the binary operator '*' (line 168)
        result_mul_628 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 28), '*', direction_626, float_627)
        
        # Applying the binary operator '+' (line 168)
        result_add_629 = python_operator(stypy.reporting.localization.Localization(__file__, 168, 22), '+', pos_624, result_mul_628)
        
        # Processing the call keyword arguments (line 168)
        kwargs_630 = {}
        # Getting the type of 'line' (line 168)
        line_622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'line', False)
        # Calling line(args, kwargs) (line 168)
        line_call_result_631 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), line_622, *[pos_623, result_add_629], **kwargs_630)
        
        # Assigning a type to the variable 'l' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'l', line_call_result_631)
        
        # Call to checkshadow(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'obj' (line 169)
        obj_634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 32), 'obj', False)
        # Getting the type of 'objects' (line 169)
        objects_635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 37), 'objects', False)
        # Getting the type of 'l' (line 169)
        l_636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 46), 'l', False)
        # Processing the call keyword arguments (line 169)
        kwargs_637 = {}
        # Getting the type of 'self' (line 169)
        self_632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'self', False)
        # Obtaining the member 'checkshadow' of a type (line 169)
        checkshadow_633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 15), self_632, 'checkshadow')
        # Calling checkshadow(args, kwargs) (line 169)
        checkshadow_call_result_638 = invoke(stypy.reporting.localization.Localization(__file__, 169, 15), checkshadow_633, *[obj_634, objects_635, l_636], **kwargs_637)
        
        # Assigning a type to the variable 'stypy_return_type' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stypy_return_type', checkshadow_call_result_638)
        
        # ################# End of 'inshadow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inshadow' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_639)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inshadow'
        return stypy_return_type_639


    @norecursion
    def light(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'light'
        module_type_store = module_type_store.open_function_context('light', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        parallellight.light.__dict__.__setitem__('stypy_localization', localization)
        parallellight.light.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        parallellight.light.__dict__.__setitem__('stypy_type_store', module_type_store)
        parallellight.light.__dict__.__setitem__('stypy_function_name', 'parallellight.light')
        parallellight.light.__dict__.__setitem__('stypy_param_names_list', ['shaderinfo'])
        parallellight.light.__dict__.__setitem__('stypy_varargs_param_name', None)
        parallellight.light.__dict__.__setitem__('stypy_kwargs_param_name', None)
        parallellight.light.__dict__.__setitem__('stypy_call_defaults', defaults)
        parallellight.light.__dict__.__setitem__('stypy_call_varargs', varargs)
        parallellight.light.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        parallellight.light.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'parallellight.light', ['shaderinfo'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'light', localization, ['shaderinfo'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'light(...)' code ##################

        
        # Call to inshadow(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'shaderinfo' (line 172)
        shaderinfo_642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 25), 'shaderinfo', False)
        # Obtaining the member 'thisobj' of a type (line 172)
        thisobj_643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 25), shaderinfo_642, 'thisobj')
        # Getting the type of 'shaderinfo' (line 172)
        shaderinfo_644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 45), 'shaderinfo', False)
        # Obtaining the member 'objects' of a type (line 172)
        objects_645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 45), shaderinfo_644, 'objects')
        # Getting the type of 'shaderinfo' (line 172)
        shaderinfo_646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 65), 'shaderinfo', False)
        # Obtaining the member 'position' of a type (line 172)
        position_647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 65), shaderinfo_646, 'position')
        # Processing the call keyword arguments (line 172)
        kwargs_648 = {}
        # Getting the type of 'self' (line 172)
        self_640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'self', False)
        # Obtaining the member 'inshadow' of a type (line 172)
        inshadow_641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 11), self_640, 'inshadow')
        # Calling inshadow(args, kwargs) (line 172)
        inshadow_call_result_649 = invoke(stypy.reporting.localization.Localization(__file__, 172, 11), inshadow_641, *[thisobj_643, objects_645, position_647], **kwargs_648)
        
        # Testing if the type of an if condition is none (line 172)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 172, 8), inshadow_call_result_649):
            pass
        else:
            
            # Testing the type of an if condition (line 172)
            if_condition_650 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 8), inshadow_call_result_649)
            # Assigning a type to the variable 'if_condition_650' (line 172)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'if_condition_650', if_condition_650)
            # SSA begins for if statement (line 172)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to vec(...): (line 173)
            # Processing the call arguments (line 173)
            float_652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 23), 'float')
            float_653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 28), 'float')
            float_654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 33), 'float')
            # Processing the call keyword arguments (line 173)
            kwargs_655 = {}
            # Getting the type of 'vec' (line 173)
            vec_651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 19), 'vec', False)
            # Calling vec(args, kwargs) (line 173)
            vec_call_result_656 = invoke(stypy.reporting.localization.Localization(__file__, 173, 19), vec_651, *[float_652, float_653, float_654], **kwargs_655)
            
            # Assigning a type to the variable 'stypy_return_type' (line 173)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 12), 'stypy_return_type', vec_call_result_656)
            # SSA join for if statement (line 172)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'self' (line 174)
        self_657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'self')
        # Obtaining the member 'col' of a type (line 174)
        col_658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 15), self_657, 'col')
        
        # Call to dot(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'shaderinfo' (line 174)
        shaderinfo_662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 45), 'shaderinfo', False)
        # Obtaining the member 'normal' of a type (line 174)
        normal_663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 45), shaderinfo_662, 'normal')
        # Processing the call keyword arguments (line 174)
        kwargs_664 = {}
        # Getting the type of 'self' (line 174)
        self_659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 26), 'self', False)
        # Obtaining the member 'direction' of a type (line 174)
        direction_660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 26), self_659, 'direction')
        # Obtaining the member 'dot' of a type (line 174)
        dot_661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 26), direction_660, 'dot')
        # Calling dot(args, kwargs) (line 174)
        dot_call_result_665 = invoke(stypy.reporting.localization.Localization(__file__, 174, 26), dot_661, *[normal_663], **kwargs_664)
        
        # Applying the binary operator '*' (line 174)
        result_mul_666 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 15), '*', col_658, dot_call_result_665)
        
        # Assigning a type to the variable 'stypy_return_type' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'stypy_return_type', result_mul_666)
        
        # ################# End of 'light(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'light' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_667)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'light'
        return stypy_return_type_667


# Assigning a type to the variable 'parallellight' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'parallellight', parallellight)
# Declaration of the 'pointlight' class
# Getting the type of 'light' (line 177)
light_668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'light')

class pointlight(light_668, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'pointlight.__init__', ['position', 'col'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['position', 'col'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 179):
        
        # Assigning a Name to a Attribute (line 179):
        
        # Assigning a Name to a Attribute (line 179):
        # Getting the type of 'position' (line 179)
        position_669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'position')
        # Getting the type of 'self' (line 179)
        self_670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'self')
        # Setting the type of the member 'position' of a type (line 179)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), self_670, 'position', position_669)
        
        # Assigning a Name to a Attribute (line 180):
        
        # Assigning a Name to a Attribute (line 180):
        
        # Assigning a Name to a Attribute (line 180):
        # Getting the type of 'col' (line 180)
        col_671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'col')
        # Getting the type of 'self' (line 180)
        self_672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'self')
        # Setting the type of the member 'col' of a type (line 180)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), self_672, 'col', col_671)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def inshadow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inshadow'
        module_type_store = module_type_store.open_function_context('inshadow', 182, 4, False)
        # Assigning a type to the variable 'self' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        pointlight.inshadow.__dict__.__setitem__('stypy_localization', localization)
        pointlight.inshadow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        pointlight.inshadow.__dict__.__setitem__('stypy_type_store', module_type_store)
        pointlight.inshadow.__dict__.__setitem__('stypy_function_name', 'pointlight.inshadow')
        pointlight.inshadow.__dict__.__setitem__('stypy_param_names_list', ['obj', 'objects', 'pos'])
        pointlight.inshadow.__dict__.__setitem__('stypy_varargs_param_name', None)
        pointlight.inshadow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        pointlight.inshadow.__dict__.__setitem__('stypy_call_defaults', defaults)
        pointlight.inshadow.__dict__.__setitem__('stypy_call_varargs', varargs)
        pointlight.inshadow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        pointlight.inshadow.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'pointlight.inshadow', ['obj', 'objects', 'pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'inshadow', localization, ['obj', 'objects', 'pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'inshadow(...)' code ##################

        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to line(...): (line 183)
        # Processing the call arguments (line 183)
        # Getting the type of 'pos' (line 183)
        pos_674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'pos', False)
        # Getting the type of 'self' (line 183)
        self_675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'self', False)
        # Obtaining the member 'position' of a type (line 183)
        position_676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 22), self_675, 'position')
        # Processing the call keyword arguments (line 183)
        kwargs_677 = {}
        # Getting the type of 'line' (line 183)
        line_673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'line', False)
        # Calling line(args, kwargs) (line 183)
        line_call_result_678 = invoke(stypy.reporting.localization.Localization(__file__, 183, 12), line_673, *[pos_674, position_676], **kwargs_677)
        
        # Assigning a type to the variable 'l' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'l', line_call_result_678)
        
        # Call to checkshadow(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'obj' (line 184)
        obj_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 32), 'obj', False)
        # Getting the type of 'objects' (line 184)
        objects_682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 37), 'objects', False)
        # Getting the type of 'l' (line 184)
        l_683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 46), 'l', False)
        # Processing the call keyword arguments (line 184)
        kwargs_684 = {}
        # Getting the type of 'self' (line 184)
        self_679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'self', False)
        # Obtaining the member 'checkshadow' of a type (line 184)
        checkshadow_680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 15), self_679, 'checkshadow')
        # Calling checkshadow(args, kwargs) (line 184)
        checkshadow_call_result_685 = invoke(stypy.reporting.localization.Localization(__file__, 184, 15), checkshadow_680, *[obj_681, objects_682, l_683], **kwargs_684)
        
        # Assigning a type to the variable 'stypy_return_type' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'stypy_return_type', checkshadow_call_result_685)
        
        # ################# End of 'inshadow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inshadow' in the type store
        # Getting the type of 'stypy_return_type' (line 182)
        stypy_return_type_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_686)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inshadow'
        return stypy_return_type_686


    @norecursion
    def light(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'light'
        module_type_store = module_type_store.open_function_context('light', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        pointlight.light.__dict__.__setitem__('stypy_localization', localization)
        pointlight.light.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        pointlight.light.__dict__.__setitem__('stypy_type_store', module_type_store)
        pointlight.light.__dict__.__setitem__('stypy_function_name', 'pointlight.light')
        pointlight.light.__dict__.__setitem__('stypy_param_names_list', ['shaderinfo'])
        pointlight.light.__dict__.__setitem__('stypy_varargs_param_name', None)
        pointlight.light.__dict__.__setitem__('stypy_kwargs_param_name', None)
        pointlight.light.__dict__.__setitem__('stypy_call_defaults', defaults)
        pointlight.light.__dict__.__setitem__('stypy_call_varargs', varargs)
        pointlight.light.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        pointlight.light.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'pointlight.light', ['shaderinfo'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'light', localization, ['shaderinfo'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'light(...)' code ##################

        
        # Call to inshadow(...): (line 187)
        # Processing the call arguments (line 187)
        # Getting the type of 'shaderinfo' (line 187)
        shaderinfo_689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 25), 'shaderinfo', False)
        # Obtaining the member 'thisobj' of a type (line 187)
        thisobj_690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 25), shaderinfo_689, 'thisobj')
        # Getting the type of 'shaderinfo' (line 187)
        shaderinfo_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 45), 'shaderinfo', False)
        # Obtaining the member 'objects' of a type (line 187)
        objects_692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 45), shaderinfo_691, 'objects')
        # Getting the type of 'shaderinfo' (line 187)
        shaderinfo_693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 65), 'shaderinfo', False)
        # Obtaining the member 'position' of a type (line 187)
        position_694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 65), shaderinfo_693, 'position')
        # Processing the call keyword arguments (line 187)
        kwargs_695 = {}
        # Getting the type of 'self' (line 187)
        self_687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 11), 'self', False)
        # Obtaining the member 'inshadow' of a type (line 187)
        inshadow_688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 11), self_687, 'inshadow')
        # Calling inshadow(args, kwargs) (line 187)
        inshadow_call_result_696 = invoke(stypy.reporting.localization.Localization(__file__, 187, 11), inshadow_688, *[thisobj_690, objects_692, position_694], **kwargs_695)
        
        # Testing if the type of an if condition is none (line 187)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 187, 8), inshadow_call_result_696):
            pass
        else:
            
            # Testing the type of an if condition (line 187)
            if_condition_697 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 187, 8), inshadow_call_result_696)
            # Assigning a type to the variable 'if_condition_697' (line 187)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'if_condition_697', if_condition_697)
            # SSA begins for if statement (line 187)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to vec(...): (line 188)
            # Processing the call arguments (line 188)
            float_699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 23), 'float')
            float_700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 28), 'float')
            float_701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 33), 'float')
            # Processing the call keyword arguments (line 188)
            kwargs_702 = {}
            # Getting the type of 'vec' (line 188)
            vec_698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 19), 'vec', False)
            # Calling vec(args, kwargs) (line 188)
            vec_call_result_703 = invoke(stypy.reporting.localization.Localization(__file__, 188, 19), vec_698, *[float_699, float_700, float_701], **kwargs_702)
            
            # Assigning a type to the variable 'stypy_return_type' (line 188)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'stypy_return_type', vec_call_result_703)
            # SSA join for if statement (line 187)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 189):
        
        # Assigning a BinOp to a Name (line 189):
        
        # Assigning a BinOp to a Name (line 189):
        # Getting the type of 'shaderinfo' (line 189)
        shaderinfo_704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'shaderinfo')
        # Obtaining the member 'position' of a type (line 189)
        position_705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 20), shaderinfo_704, 'position')
        # Getting the type of 'self' (line 189)
        self_706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 42), 'self')
        # Obtaining the member 'position' of a type (line 189)
        position_707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 42), self_706, 'position')
        # Applying the binary operator '-' (line 189)
        result_sub_708 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 20), '-', position_705, position_707)
        
        # Assigning a type to the variable 'direction' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'direction', result_sub_708)
        
        # Call to norm(...): (line 190)
        # Processing the call keyword arguments (line 190)
        kwargs_711 = {}
        # Getting the type of 'direction' (line 190)
        direction_709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'direction', False)
        # Obtaining the member 'norm' of a type (line 190)
        norm_710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 8), direction_709, 'norm')
        # Calling norm(args, kwargs) (line 190)
        norm_call_result_712 = invoke(stypy.reporting.localization.Localization(__file__, 190, 8), norm_710, *[], **kwargs_711)
        
        
        # Assigning a UnaryOp to a Name (line 191):
        
        # Assigning a UnaryOp to a Name (line 191):
        
        # Assigning a UnaryOp to a Name (line 191):
        
        # Getting the type of 'direction' (line 191)
        direction_713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 21), 'direction')
        # Applying the 'usub' unary operator (line 191)
        result___neg___714 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 20), 'usub', direction_713)
        
        # Assigning a type to the variable 'direction' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'direction', result___neg___714)
        # Getting the type of 'self' (line 192)
        self_715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'self')
        # Obtaining the member 'col' of a type (line 192)
        col_716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 15), self_715, 'col')
        
        # Call to dot(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'shaderinfo' (line 192)
        shaderinfo_719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 40), 'shaderinfo', False)
        # Obtaining the member 'normal' of a type (line 192)
        normal_720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 40), shaderinfo_719, 'normal')
        # Processing the call keyword arguments (line 192)
        kwargs_721 = {}
        # Getting the type of 'direction' (line 192)
        direction_717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 26), 'direction', False)
        # Obtaining the member 'dot' of a type (line 192)
        dot_718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 26), direction_717, 'dot')
        # Calling dot(args, kwargs) (line 192)
        dot_call_result_722 = invoke(stypy.reporting.localization.Localization(__file__, 192, 26), dot_718, *[normal_720], **kwargs_721)
        
        # Applying the binary operator '*' (line 192)
        result_mul_723 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 15), '*', col_716, dot_call_result_722)
        
        # Assigning a type to the variable 'stypy_return_type' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'stypy_return_type', result_mul_723)
        
        # ################# End of 'light(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'light' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_724)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'light'
        return stypy_return_type_724


# Assigning a type to the variable 'pointlight' (line 177)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 0), 'pointlight', pointlight)
# Declaration of the 'shader' class

class shader:

    @norecursion
    def getreflected(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getreflected'
        module_type_store = module_type_store.open_function_context('getreflected', 196, 4, False)
        # Assigning a type to the variable 'self' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        shader.getreflected.__dict__.__setitem__('stypy_localization', localization)
        shader.getreflected.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        shader.getreflected.__dict__.__setitem__('stypy_type_store', module_type_store)
        shader.getreflected.__dict__.__setitem__('stypy_function_name', 'shader.getreflected')
        shader.getreflected.__dict__.__setitem__('stypy_param_names_list', ['shaderinfo'])
        shader.getreflected.__dict__.__setitem__('stypy_varargs_param_name', None)
        shader.getreflected.__dict__.__setitem__('stypy_kwargs_param_name', None)
        shader.getreflected.__dict__.__setitem__('stypy_call_defaults', defaults)
        shader.getreflected.__dict__.__setitem__('stypy_call_varargs', varargs)
        shader.getreflected.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        shader.getreflected.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'shader.getreflected', ['shaderinfo'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getreflected', localization, ['shaderinfo'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getreflected(...)' code ##################

        
        # Assigning a Attribute to a Name (line 197):
        
        # Assigning a Attribute to a Name (line 197):
        
        # Assigning a Attribute to a Name (line 197):
        # Getting the type of 'shaderinfo' (line 197)
        shaderinfo_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 16), 'shaderinfo')
        # Obtaining the member 'depth' of a type (line 197)
        depth_726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 16), shaderinfo_725, 'depth')
        # Assigning a type to the variable 'depth' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'depth', depth_726)
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to vec(...): (line 198)
        # Processing the call arguments (line 198)
        float_728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 18), 'float')
        float_729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'float')
        float_730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 28), 'float')
        # Processing the call keyword arguments (line 198)
        kwargs_731 = {}
        # Getting the type of 'vec' (line 198)
        vec_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 14), 'vec', False)
        # Calling vec(args, kwargs) (line 198)
        vec_call_result_732 = invoke(stypy.reporting.localization.Localization(__file__, 198, 14), vec_727, *[float_728, float_729, float_730], **kwargs_731)
        
        # Assigning a type to the variable 'col' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'col', vec_call_result_732)
        
        # Getting the type of 'depth' (line 199)
        depth_733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'depth')
        int_734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 19), 'int')
        # Applying the binary operator '>' (line 199)
        result_gt_735 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 11), '>', depth_733, int_734)
        
        # Testing if the type of an if condition is none (line 199)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 199, 8), result_gt_735):
            pass
        else:
            
            # Testing the type of an if condition (line 199)
            if_condition_736 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 8), result_gt_735)
            # Assigning a type to the variable 'if_condition_736' (line 199)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'if_condition_736', if_condition_736)
            # SSA begins for if statement (line 199)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 200):
            
            # Assigning a Call to a Name (line 200):
            
            # Assigning a Call to a Name (line 200):
            
            # Call to line(...): (line 200)
            # Processing the call arguments (line 200)
            # Getting the type of 'shaderinfo' (line 200)
            shaderinfo_738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 24), 'shaderinfo', False)
            # Obtaining the member 'ray' of a type (line 200)
            ray_739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 24), shaderinfo_738, 'ray')
            # Obtaining the member 'start' of a type (line 200)
            start_740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 24), ray_739, 'start')
            # Getting the type of 'shaderinfo' (line 200)
            shaderinfo_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 46), 'shaderinfo', False)
            # Obtaining the member 'ray' of a type (line 200)
            ray_742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 46), shaderinfo_741, 'ray')
            # Obtaining the member 'end' of a type (line 200)
            end_743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 46), ray_742, 'end')
            # Processing the call keyword arguments (line 200)
            kwargs_744 = {}
            # Getting the type of 'line' (line 200)
            line_737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 19), 'line', False)
            # Calling line(args, kwargs) (line 200)
            line_call_result_745 = invoke(stypy.reporting.localization.Localization(__file__, 200, 19), line_737, *[start_740, end_743], **kwargs_744)
            
            # Assigning a type to the variable 'lray' (line 200)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'lray', line_call_result_745)
            
            # Assigning a Call to a Name (line 201):
            
            # Assigning a Call to a Name (line 201):
            
            # Assigning a Call to a Name (line 201):
            
            # Call to vec(...): (line 201)
            # Processing the call keyword arguments (line 201)
            kwargs_748 = {}
            # Getting the type of 'lray' (line 201)
            lray_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 18), 'lray', False)
            # Obtaining the member 'vec' of a type (line 201)
            vec_747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 18), lray_746, 'vec')
            # Calling vec(args, kwargs) (line 201)
            vec_call_result_749 = invoke(stypy.reporting.localization.Localization(__file__, 201, 18), vec_747, *[], **kwargs_748)
            
            # Assigning a type to the variable 'ray' (line 201)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'ray', vec_call_result_749)
            
            # Assigning a Call to a Name (line 202):
            
            # Assigning a Call to a Name (line 202):
            
            # Assigning a Call to a Name (line 202):
            
            # Call to vec(...): (line 202)
            # Processing the call arguments (line 202)
            # Getting the type of 'shaderinfo' (line 202)
            shaderinfo_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 25), 'shaderinfo', False)
            # Obtaining the member 'normal' of a type (line 202)
            normal_752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 25), shaderinfo_751, 'normal')
            # Obtaining the member 'x' of a type (line 202)
            x_753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 25), normal_752, 'x')
            # Getting the type of 'shaderinfo' (line 202)
            shaderinfo_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 46), 'shaderinfo', False)
            # Obtaining the member 'normal' of a type (line 202)
            normal_755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 46), shaderinfo_754, 'normal')
            # Obtaining the member 'y' of a type (line 202)
            y_756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 46), normal_755, 'y')
            # Getting the type of 'shaderinfo' (line 202)
            shaderinfo_757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 67), 'shaderinfo', False)
            # Obtaining the member 'normal' of a type (line 202)
            normal_758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 67), shaderinfo_757, 'normal')
            # Obtaining the member 'z' of a type (line 202)
            z_759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 67), normal_758, 'z')
            # Processing the call keyword arguments (line 202)
            kwargs_760 = {}
            # Getting the type of 'vec' (line 202)
            vec_750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 21), 'vec', False)
            # Calling vec(args, kwargs) (line 202)
            vec_call_result_761 = invoke(stypy.reporting.localization.Localization(__file__, 202, 21), vec_750, *[x_753, y_756, z_759], **kwargs_760)
            
            # Assigning a type to the variable 'normal' (line 202)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 12), 'normal', vec_call_result_761)
            
            # Assigning a Call to a Name (line 204):
            
            # Assigning a Call to a Name (line 204):
            
            # Assigning a Call to a Name (line 204):
            
            # Call to reflect(...): (line 204)
            # Processing the call arguments (line 204)
            # Getting the type of 'normal' (line 204)
            normal_764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 30), 'normal', False)
            # Processing the call keyword arguments (line 204)
            kwargs_765 = {}
            # Getting the type of 'ray' (line 204)
            ray_762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 18), 'ray', False)
            # Obtaining the member 'reflect' of a type (line 204)
            reflect_763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 18), ray_762, 'reflect')
            # Calling reflect(args, kwargs) (line 204)
            reflect_call_result_766 = invoke(stypy.reporting.localization.Localization(__file__, 204, 18), reflect_763, *[normal_764], **kwargs_765)
            
            # Assigning a type to the variable 'ray' (line 204)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'ray', reflect_call_result_766)
            
            # Assigning a Call to a Name (line 205):
            
            # Assigning a Call to a Name (line 205):
            
            # Assigning a Call to a Name (line 205):
            
            # Call to line(...): (line 205)
            # Processing the call arguments (line 205)
            # Getting the type of 'shaderinfo' (line 205)
            shaderinfo_768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 29), 'shaderinfo', False)
            # Obtaining the member 'position' of a type (line 205)
            position_769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 29), shaderinfo_768, 'position')
            # Getting the type of 'shaderinfo' (line 205)
            shaderinfo_770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 50), 'shaderinfo', False)
            # Obtaining the member 'position' of a type (line 205)
            position_771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 50), shaderinfo_770, 'position')
            # Getting the type of 'ray' (line 205)
            ray_772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 72), 'ray', False)
            # Applying the binary operator '+' (line 205)
            result_add_773 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 50), '+', position_771, ray_772)
            
            # Processing the call keyword arguments (line 205)
            kwargs_774 = {}
            # Getting the type of 'line' (line 205)
            line_767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 24), 'line', False)
            # Calling line(args, kwargs) (line 205)
            line_call_result_775 = invoke(stypy.reporting.localization.Localization(__file__, 205, 24), line_767, *[position_769, result_add_773], **kwargs_774)
            
            # Assigning a type to the variable 'reflected' (line 205)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 12), 'reflected', line_call_result_775)
            
            # Assigning a Attribute to a Name (line 206):
            
            # Assigning a Attribute to a Name (line 206):
            
            # Assigning a Attribute to a Name (line 206):
            # Getting the type of 'shaderinfo' (line 206)
            shaderinfo_776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'shaderinfo')
            # Obtaining the member 'thisobj' of a type (line 206)
            thisobj_777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 18), shaderinfo_776, 'thisobj')
            # Assigning a type to the variable 'obj' (line 206)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 12), 'obj', thisobj_777)
            
            # Assigning a Attribute to a Name (line 207):
            
            # Assigning a Attribute to a Name (line 207):
            
            # Assigning a Attribute to a Name (line 207):
            # Getting the type of 'shaderinfo' (line 207)
            shaderinfo_778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 22), 'shaderinfo')
            # Obtaining the member 'objects' of a type (line 207)
            objects_779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 22), shaderinfo_778, 'objects')
            # Assigning a type to the variable 'objects' (line 207)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'objects', objects_779)
            
            # Assigning a Call to a Name (line 209):
            
            # Assigning a Call to a Name (line 209):
            
            # Assigning a Call to a Name (line 209):
            
            # Call to Shaderinfo(...): (line 209)
            # Processing the call keyword arguments (line 209)
            kwargs_781 = {}
            # Getting the type of 'Shaderinfo' (line 209)
            Shaderinfo_780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 28), 'Shaderinfo', False)
            # Calling Shaderinfo(args, kwargs) (line 209)
            Shaderinfo_call_result_782 = invoke(stypy.reporting.localization.Localization(__file__, 209, 28), Shaderinfo_780, *[], **kwargs_781)
            
            # Assigning a type to the variable 'newshaderinfo' (line 209)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'newshaderinfo', Shaderinfo_call_result_782)
            
            # Assigning a Attribute to a Attribute (line 210):
            
            # Assigning a Attribute to a Attribute (line 210):
            
            # Assigning a Attribute to a Attribute (line 210):
            # Getting the type of 'shaderinfo' (line 210)
            shaderinfo_783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 36), 'shaderinfo')
            # Obtaining the member 'thisobj' of a type (line 210)
            thisobj_784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 36), shaderinfo_783, 'thisobj')
            # Getting the type of 'newshaderinfo' (line 210)
            newshaderinfo_785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'newshaderinfo')
            # Setting the type of the member 'thisobj' of a type (line 210)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), newshaderinfo_785, 'thisobj', thisobj_784)
            
            # Assigning a Attribute to a Attribute (line 211):
            
            # Assigning a Attribute to a Attribute (line 211):
            
            # Assigning a Attribute to a Attribute (line 211):
            # Getting the type of 'shaderinfo' (line 211)
            shaderinfo_786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 36), 'shaderinfo')
            # Obtaining the member 'objects' of a type (line 211)
            objects_787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 36), shaderinfo_786, 'objects')
            # Getting the type of 'newshaderinfo' (line 211)
            newshaderinfo_788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 12), 'newshaderinfo')
            # Setting the type of the member 'objects' of a type (line 211)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 12), newshaderinfo_788, 'objects', objects_787)
            
            # Assigning a Attribute to a Attribute (line 212):
            
            # Assigning a Attribute to a Attribute (line 212):
            
            # Assigning a Attribute to a Attribute (line 212):
            # Getting the type of 'shaderinfo' (line 212)
            shaderinfo_789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 35), 'shaderinfo')
            # Obtaining the member 'lights' of a type (line 212)
            lights_790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 35), shaderinfo_789, 'lights')
            # Getting the type of 'newshaderinfo' (line 212)
            newshaderinfo_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'newshaderinfo')
            # Setting the type of the member 'lights' of a type (line 212)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), newshaderinfo_791, 'lights', lights_790)
            
            # Assigning a Attribute to a Attribute (line 213):
            
            # Assigning a Attribute to a Attribute (line 213):
            
            # Assigning a Attribute to a Attribute (line 213):
            # Getting the type of 'shaderinfo' (line 213)
            shaderinfo_792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 37), 'shaderinfo')
            # Obtaining the member 'position' of a type (line 213)
            position_793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 37), shaderinfo_792, 'position')
            # Getting the type of 'newshaderinfo' (line 213)
            newshaderinfo_794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'newshaderinfo')
            # Setting the type of the member 'position' of a type (line 213)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 12), newshaderinfo_794, 'position', position_793)
            
            # Assigning a Attribute to a Attribute (line 214):
            
            # Assigning a Attribute to a Attribute (line 214):
            
            # Assigning a Attribute to a Attribute (line 214):
            # Getting the type of 'shaderinfo' (line 214)
            shaderinfo_795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 35), 'shaderinfo')
            # Obtaining the member 'normal' of a type (line 214)
            normal_796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 35), shaderinfo_795, 'normal')
            # Getting the type of 'newshaderinfo' (line 214)
            newshaderinfo_797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'newshaderinfo')
            # Setting the type of the member 'normal' of a type (line 214)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 12), newshaderinfo_797, 'normal', normal_796)
            
            # Assigning a Name to a Attribute (line 216):
            
            # Assigning a Name to a Attribute (line 216):
            
            # Assigning a Name to a Attribute (line 216):
            # Getting the type of 'reflected' (line 216)
            reflected_798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 32), 'reflected')
            # Getting the type of 'newshaderinfo' (line 216)
            newshaderinfo_799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'newshaderinfo')
            # Setting the type of the member 'ray' of a type (line 216)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), newshaderinfo_799, 'ray', reflected_798)
            
            # Assigning a BinOp to a Attribute (line 217):
            
            # Assigning a BinOp to a Attribute (line 217):
            
            # Assigning a BinOp to a Attribute (line 217):
            # Getting the type of 'depth' (line 217)
            depth_800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 34), 'depth')
            int_801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 42), 'int')
            # Applying the binary operator '-' (line 217)
            result_sub_802 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 34), '-', depth_800, int_801)
            
            # Getting the type of 'newshaderinfo' (line 217)
            newshaderinfo_803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'newshaderinfo')
            # Setting the type of the member 'depth' of a type (line 217)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), newshaderinfo_803, 'depth', result_sub_802)
            
            # Getting the type of 'objects' (line 220)
            objects_804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), 'objects')
            # Assigning a type to the variable 'objects_804' (line 220)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'objects_804', objects_804)
            # Testing if the for loop is going to be iterated (line 220)
            # Testing the type of a for loop iterable (line 220)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 220, 12), objects_804)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 220, 12), objects_804):
                # Getting the type of the for loop variable (line 220)
                for_loop_var_805 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 220, 12), objects_804)
                # Assigning a type to the variable 'ob' (line 220)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'ob', for_loop_var_805)
                # SSA begins for a for statement (line 220)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'ob' (line 221)
                ob_806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), 'ob')
                # Getting the type of 'obj' (line 221)
                obj_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'obj')
                # Applying the binary operator 'isnot' (line 221)
                result_is_not_808 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 19), 'isnot', ob_806, obj_807)
                
                # Testing if the type of an if condition is none (line 221)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 221, 16), result_is_not_808):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 221)
                    if_condition_809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 16), result_is_not_808)
                    # Assigning a type to the variable 'if_condition_809' (line 221)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'if_condition_809', if_condition_809)
                    # SSA begins for if statement (line 221)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Tuple (line 222):
                    
                    # Assigning a Call to a Name:
                    
                    # Assigning a Call to a Name:
                    
                    # Call to intersect(...): (line 222)
                    # Processing the call arguments (line 222)
                    # Getting the type of 'reflected' (line 222)
                    reflected_812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 66), 'reflected', False)
                    # Processing the call keyword arguments (line 222)
                    kwargs_813 = {}
                    # Getting the type of 'ob' (line 222)
                    ob_810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 53), 'ob', False)
                    # Obtaining the member 'intersect' of a type (line 222)
                    intersect_811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 53), ob_810, 'intersect')
                    # Calling intersect(args, kwargs) (line 222)
                    intersect_call_result_814 = invoke(stypy.reporting.localization.Localization(__file__, 222, 53), intersect_811, *[reflected_812], **kwargs_813)
                    
                    # Assigning a type to the variable 'call_assignment_4' (line 222)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'call_assignment_4', intersect_call_result_814)
                    
                    # Assigning a Call to a Name (line 222):
                    
                    # Assigning a Call to a Name (line 222):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_4' (line 222)
                    call_assignment_4_815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'call_assignment_4', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_816 = stypy_get_value_from_tuple(call_assignment_4_815, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_5' (line 222)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'call_assignment_5', stypy_get_value_from_tuple_call_result_816)
                    
                    # Assigning a Name to a Name (line 222):
                    
                    # Assigning a Name to a Name (line 222):
                    # Getting the type of 'call_assignment_5' (line 222)
                    call_assignment_5_817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'call_assignment_5')
                    # Assigning a type to the variable 'intersects' (line 222)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'intersects', call_assignment_5_817)
                    
                    # Assigning a Call to a Name (line 222):
                    
                    # Assigning a Call to a Name (line 222):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_4' (line 222)
                    call_assignment_4_818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'call_assignment_4', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_819 = stypy_get_value_from_tuple(call_assignment_4_818, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_6' (line 222)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'call_assignment_6', stypy_get_value_from_tuple_call_result_819)
                    
                    # Assigning a Name to a Tuple (line 222):
                    
                    # Assigning a Subscript to a Name (line 222):
                    
                    # Obtaining the type of the subscript
                    int_820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 20), 'int')
                    # Getting the type of 'call_assignment_6' (line 222)
                    call_assignment_6_821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'call_assignment_6')
                    # Obtaining the member '__getitem__' of a type (line 222)
                    getitem___822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 20), call_assignment_6_821, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
                    subscript_call_result_823 = invoke(stypy.reporting.localization.Localization(__file__, 222, 20), getitem___822, int_820)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_17' (line 222)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'tuple_var_assignment_17', subscript_call_result_823)
                    
                    # Assigning a Subscript to a Name (line 222):
                    
                    # Obtaining the type of the subscript
                    int_824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 20), 'int')
                    # Getting the type of 'call_assignment_6' (line 222)
                    call_assignment_6_825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'call_assignment_6')
                    # Obtaining the member '__getitem__' of a type (line 222)
                    getitem___826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 20), call_assignment_6_825, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 222)
                    subscript_call_result_827 = invoke(stypy.reporting.localization.Localization(__file__, 222, 20), getitem___826, int_824)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_18' (line 222)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'tuple_var_assignment_18', subscript_call_result_827)
                    
                    # Assigning a Name to a Name (line 222):
                    # Getting the type of 'tuple_var_assignment_17' (line 222)
                    tuple_var_assignment_17_828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'tuple_var_assignment_17')
                    # Assigning a type to the variable 'position' (line 222)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 33), 'position', tuple_var_assignment_17_828)
                    
                    # Assigning a Name to a Name (line 222):
                    # Getting the type of 'tuple_var_assignment_18' (line 222)
                    tuple_var_assignment_18_829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'tuple_var_assignment_18')
                    # Assigning a type to the variable 'normal' (line 222)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 43), 'normal', tuple_var_assignment_18_829)
                    
                    # Getting the type of 'intersects' (line 223)
                    intersects_830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'intersects')
                    str_831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 41), 'str', 'none')
                    # Applying the binary operator 'isnot' (line 223)
                    result_is_not_832 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 23), 'isnot', intersects_830, str_831)
                    
                    # Testing if the type of an if condition is none (line 223)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 223, 20), result_is_not_832):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 223)
                        if_condition_833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 20), result_is_not_832)
                        # Assigning a type to the variable 'if_condition_833' (line 223)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'if_condition_833', if_condition_833)
                        # SSA begins for if statement (line 223)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a Name to a Attribute (line 224):
                        
                        # Assigning a Name to a Attribute (line 224):
                        
                        # Assigning a Name to a Attribute (line 224):
                        # Getting the type of 'ob' (line 224)
                        ob_834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 48), 'ob')
                        # Getting the type of 'newshaderinfo' (line 224)
                        newshaderinfo_835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 24), 'newshaderinfo')
                        # Setting the type of the member 'thisobj' of a type (line 224)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 24), newshaderinfo_835, 'thisobj', ob_834)
                        
                        # Assigning a Name to a Attribute (line 225):
                        
                        # Assigning a Name to a Attribute (line 225):
                        
                        # Assigning a Name to a Attribute (line 225):
                        # Getting the type of 'position' (line 225)
                        position_836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 49), 'position')
                        # Getting the type of 'newshaderinfo' (line 225)
                        newshaderinfo_837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 24), 'newshaderinfo')
                        # Setting the type of the member 'position' of a type (line 225)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 225, 24), newshaderinfo_837, 'position', position_836)
                        
                        # Assigning a Name to a Attribute (line 226):
                        
                        # Assigning a Name to a Attribute (line 226):
                        
                        # Assigning a Name to a Attribute (line 226):
                        # Getting the type of 'normal' (line 226)
                        normal_838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 47), 'normal')
                        # Getting the type of 'newshaderinfo' (line 226)
                        newshaderinfo_839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'newshaderinfo')
                        # Setting the type of the member 'normal' of a type (line 226)
                        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 24), newshaderinfo_839, 'normal', normal_838)
                        
                        # Assigning a BinOp to a Name (line 227):
                        
                        # Assigning a BinOp to a Name (line 227):
                        
                        # Assigning a BinOp to a Name (line 227):
                        # Getting the type of 'col' (line 227)
                        col_840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 30), 'col')
                        
                        # Call to shade(...): (line 227)
                        # Processing the call arguments (line 227)
                        # Getting the type of 'newshaderinfo' (line 227)
                        newshaderinfo_844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 52), 'newshaderinfo', False)
                        # Processing the call keyword arguments (line 227)
                        kwargs_845 = {}
                        # Getting the type of 'ob' (line 227)
                        ob_841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 36), 'ob', False)
                        # Obtaining the member 'shader' of a type (line 227)
                        shader_842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 36), ob_841, 'shader')
                        # Obtaining the member 'shade' of a type (line 227)
                        shade_843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 36), shader_842, 'shade')
                        # Calling shade(args, kwargs) (line 227)
                        shade_call_result_846 = invoke(stypy.reporting.localization.Localization(__file__, 227, 36), shade_843, *[newshaderinfo_844], **kwargs_845)
                        
                        # Applying the binary operator '+' (line 227)
                        result_add_847 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 30), '+', col_840, shade_call_result_846)
                        
                        # Assigning a type to the variable 'col' (line 227)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 24), 'col', result_add_847)
                        # SSA join for if statement (line 223)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 221)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 199)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'col' (line 228)
        col_848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 15), 'col')
        # Assigning a type to the variable 'stypy_return_type' (line 228)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 8), 'stypy_return_type', col_848)
        
        # ################# End of 'getreflected(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getreflected' in the type store
        # Getting the type of 'stypy_return_type' (line 196)
        stypy_return_type_849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_849)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getreflected'
        return stypy_return_type_849


    @norecursion
    def isoccluded(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'isoccluded'
        module_type_store = module_type_store.open_function_context('isoccluded', 230, 4, False)
        # Assigning a type to the variable 'self' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        shader.isoccluded.__dict__.__setitem__('stypy_localization', localization)
        shader.isoccluded.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        shader.isoccluded.__dict__.__setitem__('stypy_type_store', module_type_store)
        shader.isoccluded.__dict__.__setitem__('stypy_function_name', 'shader.isoccluded')
        shader.isoccluded.__dict__.__setitem__('stypy_param_names_list', ['ray', 'shaderinfo'])
        shader.isoccluded.__dict__.__setitem__('stypy_varargs_param_name', None)
        shader.isoccluded.__dict__.__setitem__('stypy_kwargs_param_name', None)
        shader.isoccluded.__dict__.__setitem__('stypy_call_defaults', defaults)
        shader.isoccluded.__dict__.__setitem__('stypy_call_varargs', varargs)
        shader.isoccluded.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        shader.isoccluded.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'shader.isoccluded', ['ray', 'shaderinfo'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'isoccluded', localization, ['ray', 'shaderinfo'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'isoccluded(...)' code ##################

        
        # Assigning a Call to a Name (line 231):
        
        # Assigning a Call to a Name (line 231):
        
        # Assigning a Call to a Name (line 231):
        
        # Call to mag(...): (line 231)
        # Processing the call keyword arguments (line 231)
        kwargs_852 = {}
        # Getting the type of 'ray' (line 231)
        ray_850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 15), 'ray', False)
        # Obtaining the member 'mag' of a type (line 231)
        mag_851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 15), ray_850, 'mag')
        # Calling mag(args, kwargs) (line 231)
        mag_call_result_853 = invoke(stypy.reporting.localization.Localization(__file__, 231, 15), mag_851, *[], **kwargs_852)
        
        # Assigning a type to the variable 'dist' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 8), 'dist', mag_call_result_853)
        
        # Assigning a Call to a Name (line 232):
        
        # Assigning a Call to a Name (line 232):
        
        # Assigning a Call to a Name (line 232):
        
        # Call to line(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'shaderinfo' (line 232)
        shaderinfo_855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 20), 'shaderinfo', False)
        # Obtaining the member 'position' of a type (line 232)
        position_856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 20), shaderinfo_855, 'position')
        # Getting the type of 'shaderinfo' (line 232)
        shaderinfo_857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 41), 'shaderinfo', False)
        # Obtaining the member 'position' of a type (line 232)
        position_858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 41), shaderinfo_857, 'position')
        # Getting the type of 'ray' (line 232)
        ray_859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 63), 'ray', False)
        # Applying the binary operator '+' (line 232)
        result_add_860 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 41), '+', position_858, ray_859)
        
        # Processing the call keyword arguments (line 232)
        kwargs_861 = {}
        # Getting the type of 'line' (line 232)
        line_854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'line', False)
        # Calling line(args, kwargs) (line 232)
        line_call_result_862 = invoke(stypy.reporting.localization.Localization(__file__, 232, 15), line_854, *[position_856, result_add_860], **kwargs_861)
        
        # Assigning a type to the variable 'test' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'test', line_call_result_862)
        
        # Assigning a Attribute to a Name (line 233):
        
        # Assigning a Attribute to a Name (line 233):
        
        # Assigning a Attribute to a Name (line 233):
        # Getting the type of 'shaderinfo' (line 233)
        shaderinfo_863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 14), 'shaderinfo')
        # Obtaining the member 'thisobj' of a type (line 233)
        thisobj_864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 233, 14), shaderinfo_863, 'thisobj')
        # Assigning a type to the variable 'obj' (line 233)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'obj', thisobj_864)
        
        # Assigning a Attribute to a Name (line 234):
        
        # Assigning a Attribute to a Name (line 234):
        
        # Assigning a Attribute to a Name (line 234):
        # Getting the type of 'shaderinfo' (line 234)
        shaderinfo_865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 18), 'shaderinfo')
        # Obtaining the member 'objects' of a type (line 234)
        objects_866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 234, 18), shaderinfo_865, 'objects')
        # Assigning a type to the variable 'objects' (line 234)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'objects', objects_866)
        
        # Getting the type of 'objects' (line 236)
        objects_867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 18), 'objects')
        # Assigning a type to the variable 'objects_867' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'objects_867', objects_867)
        # Testing if the for loop is going to be iterated (line 236)
        # Testing the type of a for loop iterable (line 236)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 236, 8), objects_867)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 236, 8), objects_867):
            # Getting the type of the for loop variable (line 236)
            for_loop_var_868 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 236, 8), objects_867)
            # Assigning a type to the variable 'ob' (line 236)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'ob', for_loop_var_868)
            # SSA begins for a for statement (line 236)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'ob' (line 237)
            ob_869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 15), 'ob')
            # Getting the type of 'obj' (line 237)
            obj_870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 25), 'obj')
            # Applying the binary operator 'isnot' (line 237)
            result_is_not_871 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 15), 'isnot', ob_869, obj_870)
            
            # Testing if the type of an if condition is none (line 237)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 237, 12), result_is_not_871):
                pass
            else:
                
                # Testing the type of an if condition (line 237)
                if_condition_872 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 237, 12), result_is_not_871)
                # Assigning a type to the variable 'if_condition_872' (line 237)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'if_condition_872', if_condition_872)
                # SSA begins for if statement (line 237)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 238):
                
                # Assigning a Call to a Name:
                
                # Assigning a Call to a Name:
                
                # Call to intersect(...): (line 238)
                # Processing the call arguments (line 238)
                # Getting the type of 'test' (line 238)
                test_875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 62), 'test', False)
                # Processing the call keyword arguments (line 238)
                kwargs_876 = {}
                # Getting the type of 'ob' (line 238)
                ob_873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 49), 'ob', False)
                # Obtaining the member 'intersect' of a type (line 238)
                intersect_874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 49), ob_873, 'intersect')
                # Calling intersect(args, kwargs) (line 238)
                intersect_call_result_877 = invoke(stypy.reporting.localization.Localization(__file__, 238, 49), intersect_874, *[test_875], **kwargs_876)
                
                # Assigning a type to the variable 'call_assignment_7' (line 238)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_7', intersect_call_result_877)
                
                # Assigning a Call to a Name (line 238):
                
                # Assigning a Call to a Name (line 238):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7' (line 238)
                call_assignment_7_878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_7', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_879 = stypy_get_value_from_tuple(call_assignment_7_878, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_8' (line 238)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_8', stypy_get_value_from_tuple_call_result_879)
                
                # Assigning a Name to a Name (line 238):
                
                # Assigning a Name to a Name (line 238):
                # Getting the type of 'call_assignment_8' (line 238)
                call_assignment_8_880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_8')
                # Assigning a type to the variable 'intersects' (line 238)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'intersects', call_assignment_8_880)
                
                # Assigning a Call to a Name (line 238):
                
                # Assigning a Call to a Name (line 238):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7' (line 238)
                call_assignment_7_881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_7', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_882 = stypy_get_value_from_tuple(call_assignment_7_881, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_9' (line 238)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_9', stypy_get_value_from_tuple_call_result_882)
                
                # Assigning a Name to a Tuple (line 238):
                
                # Assigning a Subscript to a Name (line 238):
                
                # Obtaining the type of the subscript
                int_883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 16), 'int')
                # Getting the type of 'call_assignment_9' (line 238)
                call_assignment_9_884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_9')
                # Obtaining the member '__getitem__' of a type (line 238)
                getitem___885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 16), call_assignment_9_884, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 238)
                subscript_call_result_886 = invoke(stypy.reporting.localization.Localization(__file__, 238, 16), getitem___885, int_883)
                
                # Assigning a type to the variable 'tuple_var_assignment_19' (line 238)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'tuple_var_assignment_19', subscript_call_result_886)
                
                # Assigning a Subscript to a Name (line 238):
                
                # Obtaining the type of the subscript
                int_887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 16), 'int')
                # Getting the type of 'call_assignment_9' (line 238)
                call_assignment_9_888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'call_assignment_9')
                # Obtaining the member '__getitem__' of a type (line 238)
                getitem___889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 16), call_assignment_9_888, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 238)
                subscript_call_result_890 = invoke(stypy.reporting.localization.Localization(__file__, 238, 16), getitem___889, int_887)
                
                # Assigning a type to the variable 'tuple_var_assignment_20' (line 238)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'tuple_var_assignment_20', subscript_call_result_890)
                
                # Assigning a Name to a Name (line 238):
                # Getting the type of 'tuple_var_assignment_19' (line 238)
                tuple_var_assignment_19_891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'tuple_var_assignment_19')
                # Assigning a type to the variable 'position' (line 238)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 29), 'position', tuple_var_assignment_19_891)
                
                # Assigning a Name to a Name (line 238):
                # Getting the type of 'tuple_var_assignment_20' (line 238)
                tuple_var_assignment_20_892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'tuple_var_assignment_20')
                # Assigning a type to the variable 'normal' (line 238)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 39), 'normal', tuple_var_assignment_20_892)
                
                # Getting the type of 'intersects' (line 239)
                intersects_893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 19), 'intersects')
                str_894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 37), 'str', 'none')
                # Applying the binary operator 'isnot' (line 239)
                result_is_not_895 = python_operator(stypy.reporting.localization.Localization(__file__, 239, 19), 'isnot', intersects_893, str_894)
                
                # Testing if the type of an if condition is none (line 239)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 239, 16), result_is_not_895):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 239)
                    if_condition_896 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 239, 16), result_is_not_895)
                    # Assigning a type to the variable 'if_condition_896' (line 239)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 16), 'if_condition_896', if_condition_896)
                    # SSA begins for if statement (line 239)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    int_897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 27), 'int')
                    # Assigning a type to the variable 'stypy_return_type' (line 240)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 20), 'stypy_return_type', int_897)
                    # SSA join for if statement (line 239)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 237)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        int_898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 241, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 8), 'stypy_return_type', int_898)
        
        # ################# End of 'isoccluded(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'isoccluded' in the type store
        # Getting the type of 'stypy_return_type' (line 230)
        stypy_return_type_899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_899)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'isoccluded'
        return stypy_return_type_899


    @norecursion
    def doocclusion(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'doocclusion'
        module_type_store = module_type_store.open_function_context('doocclusion', 243, 4, False)
        # Assigning a type to the variable 'self' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        shader.doocclusion.__dict__.__setitem__('stypy_localization', localization)
        shader.doocclusion.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        shader.doocclusion.__dict__.__setitem__('stypy_type_store', module_type_store)
        shader.doocclusion.__dict__.__setitem__('stypy_function_name', 'shader.doocclusion')
        shader.doocclusion.__dict__.__setitem__('stypy_param_names_list', ['samples', 'shaderinfo'])
        shader.doocclusion.__dict__.__setitem__('stypy_varargs_param_name', None)
        shader.doocclusion.__dict__.__setitem__('stypy_kwargs_param_name', None)
        shader.doocclusion.__dict__.__setitem__('stypy_call_defaults', defaults)
        shader.doocclusion.__dict__.__setitem__('stypy_call_varargs', varargs)
        shader.doocclusion.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        shader.doocclusion.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'shader.doocclusion', ['samples', 'shaderinfo'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'doocclusion', localization, ['samples', 'shaderinfo'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'doocclusion(...)' code ##################

        
        # Assigning a Num to a Name (line 245):
        
        # Assigning a Num to a Name (line 245):
        
        # Assigning a Num to a Name (line 245):
        float_900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 245, 13), 'float')
        # Assigning a type to the variable 'oc' (line 245)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 245, 8), 'oc', float_900)
        
        
        # Call to xrange(...): (line 246)
        # Processing the call arguments (line 246)
        # Getting the type of 'samples' (line 246)
        samples_902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 24), 'samples', False)
        # Processing the call keyword arguments (line 246)
        kwargs_903 = {}
        # Getting the type of 'xrange' (line 246)
        xrange_901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 246)
        xrange_call_result_904 = invoke(stypy.reporting.localization.Localization(__file__, 246, 17), xrange_901, *[samples_902], **kwargs_903)
        
        # Assigning a type to the variable 'xrange_call_result_904' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'xrange_call_result_904', xrange_call_result_904)
        # Testing if the for loop is going to be iterated (line 246)
        # Testing the type of a for loop iterable (line 246)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 246, 8), xrange_call_result_904)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 246, 8), xrange_call_result_904):
            # Getting the type of the for loop variable (line 246)
            for_loop_var_905 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 246, 8), xrange_call_result_904)
            # Assigning a type to the variable 'i' (line 246)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'i', for_loop_var_905)
            # SSA begins for a for statement (line 246)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 247):
            
            # Assigning a Call to a Name (line 247):
            
            # Assigning a Call to a Name (line 247):
            
            # Call to vec(...): (line 247)
            # Processing the call arguments (line 247)
            
            # Call to float(...): (line 247)
            # Processing the call arguments (line 247)
            
            # Call to randrange(...): (line 247)
            # Processing the call arguments (line 247)
            int_910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 45), 'int')
            int_911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 51), 'int')
            # Processing the call keyword arguments (line 247)
            kwargs_912 = {}
            # Getting the type of 'random' (line 247)
            random_908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 28), 'random', False)
            # Obtaining the member 'randrange' of a type (line 247)
            randrange_909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 28), random_908, 'randrange')
            # Calling randrange(args, kwargs) (line 247)
            randrange_call_result_913 = invoke(stypy.reporting.localization.Localization(__file__, 247, 28), randrange_909, *[int_910, int_911], **kwargs_912)
            
            # Processing the call keyword arguments (line 247)
            kwargs_914 = {}
            # Getting the type of 'float' (line 247)
            float_907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 22), 'float', False)
            # Calling float(args, kwargs) (line 247)
            float_call_result_915 = invoke(stypy.reporting.localization.Localization(__file__, 247, 22), float_907, *[randrange_call_result_913], **kwargs_914)
            
            
            # Call to float(...): (line 247)
            # Processing the call arguments (line 247)
            
            # Call to randrange(...): (line 247)
            # Processing the call arguments (line 247)
            int_919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 81), 'int')
            int_920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 87), 'int')
            # Processing the call keyword arguments (line 247)
            kwargs_921 = {}
            # Getting the type of 'random' (line 247)
            random_917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 64), 'random', False)
            # Obtaining the member 'randrange' of a type (line 247)
            randrange_918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 64), random_917, 'randrange')
            # Calling randrange(args, kwargs) (line 247)
            randrange_call_result_922 = invoke(stypy.reporting.localization.Localization(__file__, 247, 64), randrange_918, *[int_919, int_920], **kwargs_921)
            
            # Processing the call keyword arguments (line 247)
            kwargs_923 = {}
            # Getting the type of 'float' (line 247)
            float_916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 58), 'float', False)
            # Calling float(args, kwargs) (line 247)
            float_call_result_924 = invoke(stypy.reporting.localization.Localization(__file__, 247, 58), float_916, *[randrange_call_result_922], **kwargs_923)
            
            
            # Call to float(...): (line 248)
            # Processing the call arguments (line 248)
            
            # Call to randrange(...): (line 248)
            # Processing the call arguments (line 248)
            int_928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 45), 'int')
            int_929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 51), 'int')
            # Processing the call keyword arguments (line 248)
            kwargs_930 = {}
            # Getting the type of 'random' (line 248)
            random_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 28), 'random', False)
            # Obtaining the member 'randrange' of a type (line 248)
            randrange_927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 28), random_926, 'randrange')
            # Calling randrange(args, kwargs) (line 248)
            randrange_call_result_931 = invoke(stypy.reporting.localization.Localization(__file__, 248, 28), randrange_927, *[int_928, int_929], **kwargs_930)
            
            # Processing the call keyword arguments (line 248)
            kwargs_932 = {}
            # Getting the type of 'float' (line 248)
            float_925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 22), 'float', False)
            # Calling float(args, kwargs) (line 248)
            float_call_result_933 = invoke(stypy.reporting.localization.Localization(__file__, 248, 22), float_925, *[randrange_call_result_931], **kwargs_932)
            
            # Processing the call keyword arguments (line 247)
            kwargs_934 = {}
            # Getting the type of 'vec' (line 247)
            vec_906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 18), 'vec', False)
            # Calling vec(args, kwargs) (line 247)
            vec_call_result_935 = invoke(stypy.reporting.localization.Localization(__file__, 247, 18), vec_906, *[float_call_result_915, float_call_result_924, float_call_result_933], **kwargs_934)
            
            # Assigning a type to the variable 'ray' (line 247)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'ray', vec_call_result_935)
            
            # Call to norm(...): (line 249)
            # Processing the call keyword arguments (line 249)
            kwargs_938 = {}
            # Getting the type of 'ray' (line 249)
            ray_936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'ray', False)
            # Obtaining the member 'norm' of a type (line 249)
            norm_937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 12), ray_936, 'norm')
            # Calling norm(args, kwargs) (line 249)
            norm_call_result_939 = invoke(stypy.reporting.localization.Localization(__file__, 249, 12), norm_937, *[], **kwargs_938)
            
            
            # Assigning a BinOp to a Name (line 250):
            
            # Assigning a BinOp to a Name (line 250):
            
            # Assigning a BinOp to a Name (line 250):
            # Getting the type of 'ray' (line 250)
            ray_940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 18), 'ray')
            float_941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 24), 'float')
            # Applying the binary operator '*' (line 250)
            result_mul_942 = python_operator(stypy.reporting.localization.Localization(__file__, 250, 18), '*', ray_940, float_941)
            
            # Assigning a type to the variable 'ray' (line 250)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 12), 'ray', result_mul_942)
            
            # Call to isoccluded(...): (line 251)
            # Processing the call arguments (line 251)
            # Getting the type of 'ray' (line 251)
            ray_945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 31), 'ray', False)
            # Getting the type of 'shaderinfo' (line 251)
            shaderinfo_946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 36), 'shaderinfo', False)
            # Processing the call keyword arguments (line 251)
            kwargs_947 = {}
            # Getting the type of 'self' (line 251)
            self_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'self', False)
            # Obtaining the member 'isoccluded' of a type (line 251)
            isoccluded_944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 15), self_943, 'isoccluded')
            # Calling isoccluded(args, kwargs) (line 251)
            isoccluded_call_result_948 = invoke(stypy.reporting.localization.Localization(__file__, 251, 15), isoccluded_944, *[ray_945, shaderinfo_946], **kwargs_947)
            
            # Testing if the type of an if condition is none (line 251)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 251, 12), isoccluded_call_result_948):
                pass
            else:
                
                # Testing the type of an if condition (line 251)
                if_condition_949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 12), isoccluded_call_result_948)
                # Assigning a type to the variable 'if_condition_949' (line 251)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'if_condition_949', if_condition_949)
                # SSA begins for if statement (line 251)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 252):
                
                # Assigning a BinOp to a Name (line 252):
                
                # Assigning a BinOp to a Name (line 252):
                # Getting the type of 'oc' (line 252)
                oc_950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 21), 'oc')
                int_951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 26), 'int')
                # Applying the binary operator '+' (line 252)
                result_add_952 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 21), '+', oc_950, int_951)
                
                # Assigning a type to the variable 'oc' (line 252)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'oc', result_add_952)
                # SSA join for if statement (line 251)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a BinOp to a Name (line 253):
        
        # Assigning a BinOp to a Name (line 253):
        
        # Assigning a BinOp to a Name (line 253):
        # Getting the type of 'oc' (line 253)
        oc_953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 13), 'oc')
        
        # Call to float(...): (line 253)
        # Processing the call arguments (line 253)
        # Getting the type of 'samples' (line 253)
        samples_955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 24), 'samples', False)
        # Processing the call keyword arguments (line 253)
        kwargs_956 = {}
        # Getting the type of 'float' (line 253)
        float_954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 18), 'float', False)
        # Calling float(args, kwargs) (line 253)
        float_call_result_957 = invoke(stypy.reporting.localization.Localization(__file__, 253, 18), float_954, *[samples_955], **kwargs_956)
        
        # Applying the binary operator 'div' (line 253)
        result_div_958 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 13), 'div', oc_953, float_call_result_957)
        
        # Assigning a type to the variable 'oc' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'oc', result_div_958)
        int_959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 15), 'int')
        # Getting the type of 'oc' (line 254)
        oc_960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 19), 'oc')
        # Applying the binary operator '-' (line 254)
        result_sub_961 = python_operator(stypy.reporting.localization.Localization(__file__, 254, 15), '-', int_959, oc_960)
        
        # Assigning a type to the variable 'stypy_return_type' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 8), 'stypy_return_type', result_sub_961)
        
        # ################# End of 'doocclusion(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'doocclusion' in the type store
        # Getting the type of 'stypy_return_type' (line 243)
        stypy_return_type_962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_962)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'doocclusion'
        return stypy_return_type_962


    @norecursion
    def shade(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shade'
        module_type_store = module_type_store.open_function_context('shade', 256, 4, False)
        # Assigning a type to the variable 'self' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        shader.shade.__dict__.__setitem__('stypy_localization', localization)
        shader.shade.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        shader.shade.__dict__.__setitem__('stypy_type_store', module_type_store)
        shader.shade.__dict__.__setitem__('stypy_function_name', 'shader.shade')
        shader.shade.__dict__.__setitem__('stypy_param_names_list', ['shaderinfo'])
        shader.shade.__dict__.__setitem__('stypy_varargs_param_name', None)
        shader.shade.__dict__.__setitem__('stypy_kwargs_param_name', None)
        shader.shade.__dict__.__setitem__('stypy_call_defaults', defaults)
        shader.shade.__dict__.__setitem__('stypy_call_varargs', varargs)
        shader.shade.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        shader.shade.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'shader.shade', ['shaderinfo'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shade', localization, ['shaderinfo'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shade(...)' code ##################

        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Assigning a Call to a Name (line 257):
        
        # Call to vec(...): (line 257)
        # Processing the call arguments (line 257)
        float_964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 18), 'float')
        float_965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 23), 'float')
        float_966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 28), 'float')
        # Processing the call keyword arguments (line 257)
        kwargs_967 = {}
        # Getting the type of 'vec' (line 257)
        vec_963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 14), 'vec', False)
        # Calling vec(args, kwargs) (line 257)
        vec_call_result_968 = invoke(stypy.reporting.localization.Localization(__file__, 257, 14), vec_963, *[float_964, float_965, float_966], **kwargs_967)
        
        # Assigning a type to the variable 'col' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'col', vec_call_result_968)
        
        # Getting the type of 'shaderinfo' (line 258)
        shaderinfo_969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 20), 'shaderinfo')
        # Obtaining the member 'lights' of a type (line 258)
        lights_970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 20), shaderinfo_969, 'lights')
        # Assigning a type to the variable 'lights_970' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'lights_970', lights_970)
        # Testing if the for loop is going to be iterated (line 258)
        # Testing the type of a for loop iterable (line 258)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 258, 8), lights_970)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 258, 8), lights_970):
            # Getting the type of the for loop variable (line 258)
            for_loop_var_971 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 258, 8), lights_970)
            # Assigning a type to the variable 'lite' (line 258)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'lite', for_loop_var_971)
            # SSA begins for a for statement (line 258)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 259):
            
            # Assigning a BinOp to a Name (line 259):
            
            # Assigning a BinOp to a Name (line 259):
            # Getting the type of 'col' (line 259)
            col_972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 18), 'col')
            
            # Call to light(...): (line 259)
            # Processing the call arguments (line 259)
            # Getting the type of 'shaderinfo' (line 259)
            shaderinfo_975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 35), 'shaderinfo', False)
            # Processing the call keyword arguments (line 259)
            kwargs_976 = {}
            # Getting the type of 'lite' (line 259)
            lite_973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 24), 'lite', False)
            # Obtaining the member 'light' of a type (line 259)
            light_974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 259, 24), lite_973, 'light')
            # Calling light(args, kwargs) (line 259)
            light_call_result_977 = invoke(stypy.reporting.localization.Localization(__file__, 259, 24), light_974, *[shaderinfo_975], **kwargs_976)
            
            # Applying the binary operator '+' (line 259)
            result_add_978 = python_operator(stypy.reporting.localization.Localization(__file__, 259, 18), '+', col_972, light_call_result_977)
            
            # Assigning a type to the variable 'col' (line 259)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'col', result_add_978)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'col' (line 260)
        col_979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 15), 'col')
        # Assigning a type to the variable 'stypy_return_type' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'stypy_return_type', col_979)
        
        # ################# End of 'shade(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shade' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_980)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shade'
        return stypy_return_type_980


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 195, 0, False)
        # Assigning a type to the variable 'self' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'shader.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'shader' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'shader', shader)
# Declaration of the 'world' class

class world:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 264, 4, False)
        # Assigning a type to the variable 'self' (line 265)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 265, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'world.__init__', ['width', 'height'], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 265):
        
        # Assigning a List to a Attribute (line 265):
        
        # Assigning a List to a Attribute (line 265):
        
        # Obtaining an instance of the builtin type 'list' (line 265)
        list_981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 265, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 265)
        
        # Getting the type of 'self' (line 265)
        self_982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 8), 'self')
        # Setting the type of the member 'lights' of a type (line 265)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 8), self_982, 'lights', list_981)
        
        # Assigning a List to a Attribute (line 266):
        
        # Assigning a List to a Attribute (line 266):
        
        # Assigning a List to a Attribute (line 266):
        
        # Obtaining an instance of the builtin type 'list' (line 266)
        list_983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 266, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 266)
        
        # Getting the type of 'self' (line 266)
        self_984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 8), 'self')
        # Setting the type of the member 'objects' of a type (line 266)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 8), self_984, 'objects', list_983)
        
        # Assigning a Str to a Attribute (line 267):
        
        # Assigning a Str to a Attribute (line 267):
        
        # Assigning a Str to a Attribute (line 267):
        str_985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 26), 'str', 'persp')
        # Getting the type of 'self' (line 267)
        self_986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 8), 'self')
        # Setting the type of the member 'cameratype' of a type (line 267)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 267, 8), self_986, 'cameratype', str_985)
        
        # Assigning a Name to a Attribute (line 268):
        
        # Assigning a Name to a Attribute (line 268):
        
        # Assigning a Name to a Attribute (line 268):
        # Getting the type of 'width' (line 268)
        width_987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 21), 'width')
        # Getting the type of 'self' (line 268)
        self_988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 8), 'self')
        # Setting the type of the member 'width' of a type (line 268)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 268, 8), self_988, 'width', width_987)
        
        # Assigning a Name to a Attribute (line 269):
        
        # Assigning a Name to a Attribute (line 269):
        
        # Assigning a Name to a Attribute (line 269):
        # Getting the type of 'height' (line 269)
        height_989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 22), 'height')
        # Getting the type of 'self' (line 269)
        self_990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 8), 'self')
        # Setting the type of the member 'height' of a type (line 269)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 269, 8), self_990, 'height', height_989)
        
        # Assigning a Num to a Attribute (line 270):
        
        # Assigning a Num to a Attribute (line 270):
        
        # Assigning a Num to a Attribute (line 270):
        float_991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 25), 'float')
        # Getting the type of 'self' (line 270)
        self_992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'self')
        # Setting the type of the member 'backplane' of a type (line 270)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 270, 8), self_992, 'backplane', float_991)
        
        # Assigning a Num to a Attribute (line 271):
        
        # Assigning a Num to a Attribute (line 271):
        
        # Assigning a Num to a Attribute (line 271):
        float_993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 26), 'float')
        # Getting the type of 'self' (line 271)
        self_994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 8), 'self')
        # Setting the type of the member 'imageplane' of a type (line 271)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 8), self_994, 'imageplane', float_993)
        
        # Assigning a BinOp to a Attribute (line 272):
        
        # Assigning a BinOp to a Attribute (line 272):
        
        # Assigning a BinOp to a Attribute (line 272):
        # Getting the type of 'self' (line 272)
        self_995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 22), 'self')
        # Obtaining the member 'width' of a type (line 272)
        width_996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 22), self_995, 'width')
        
        # Call to float(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'self' (line 272)
        self_998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 41), 'self', False)
        # Obtaining the member 'height' of a type (line 272)
        height_999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 41), self_998, 'height')
        # Processing the call keyword arguments (line 272)
        kwargs_1000 = {}
        # Getting the type of 'float' (line 272)
        float_997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 35), 'float', False)
        # Calling float(args, kwargs) (line 272)
        float_call_result_1001 = invoke(stypy.reporting.localization.Localization(__file__, 272, 35), float_997, *[height_999], **kwargs_1000)
        
        # Applying the binary operator 'div' (line 272)
        result_div_1002 = python_operator(stypy.reporting.localization.Localization(__file__, 272, 22), 'div', width_996, float_call_result_1001)
        
        # Getting the type of 'self' (line 272)
        self_1003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'self')
        # Setting the type of the member 'aspect' of a type (line 272)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 8), self_1003, 'aspect', result_div_1002)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def render(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'render'
        module_type_store = module_type_store.open_function_context('render', 274, 4, False)
        # Assigning a type to the variable 'self' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        world.render.__dict__.__setitem__('stypy_localization', localization)
        world.render.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        world.render.__dict__.__setitem__('stypy_type_store', module_type_store)
        world.render.__dict__.__setitem__('stypy_function_name', 'world.render')
        world.render.__dict__.__setitem__('stypy_param_names_list', ['filename'])
        world.render.__dict__.__setitem__('stypy_varargs_param_name', None)
        world.render.__dict__.__setitem__('stypy_kwargs_param_name', None)
        world.render.__dict__.__setitem__('stypy_call_defaults', defaults)
        world.render.__dict__.__setitem__('stypy_call_varargs', varargs)
        world.render.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        world.render.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'world.render', ['filename'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'render', localization, ['filename'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'render(...)' code ##################

        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Assigning a Call to a Name (line 275):
        
        # Call to file(...): (line 275)
        # Processing the call arguments (line 275)
        # Getting the type of 'filename' (line 275)
        filename_1005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 24), 'filename', False)
        str_1006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 34), 'str', 'w')
        # Processing the call keyword arguments (line 275)
        kwargs_1007 = {}
        # Getting the type of 'file' (line 275)
        file_1004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 19), 'file', False)
        # Calling file(args, kwargs) (line 275)
        file_call_result_1008 = invoke(stypy.reporting.localization.Localization(__file__, 275, 19), file_1004, *[filename_1005, str_1006], **kwargs_1007)
        
        # Assigning a type to the variable 'out_file' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 8), 'out_file', file_call_result_1008)
        str_1009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 277, 27), 'str', 'P3')
        # Getting the type of 'self' (line 278)
        self_1010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 27), 'self')
        # Obtaining the member 'width' of a type (line 278)
        width_1011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 27), self_1010, 'width')
        # Getting the type of 'self' (line 278)
        self_1012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 39), 'self')
        # Obtaining the member 'height' of a type (line 278)
        height_1013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 39), self_1012, 'height')
        str_1014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 279, 27), 'str', '256')
        
        # Assigning a BinOp to a Name (line 280):
        
        # Assigning a BinOp to a Name (line 280):
        
        # Assigning a BinOp to a Name (line 280):
        # Getting the type of 'self' (line 280)
        self_1015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 16), 'self')
        # Obtaining the member 'width' of a type (line 280)
        width_1016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 16), self_1015, 'width')
        # Getting the type of 'self' (line 280)
        self_1017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 29), 'self')
        # Obtaining the member 'height' of a type (line 280)
        height_1018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 280, 29), self_1017, 'height')
        # Applying the binary operator '*' (line 280)
        result_mul_1019 = python_operator(stypy.reporting.localization.Localization(__file__, 280, 16), '*', width_1016, height_1018)
        
        # Assigning a type to the variable 'total' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'total', result_mul_1019)
        
        # Assigning a Num to a Name (line 281):
        
        # Assigning a Num to a Name (line 281):
        
        # Assigning a Num to a Name (line 281):
        int_1020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 281, 16), 'int')
        # Assigning a type to the variable 'count' (line 281)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 281, 8), 'count', int_1020)
        
        
        # Call to xrange(...): (line 283)
        # Processing the call arguments (line 283)
        # Getting the type of 'self' (line 283)
        self_1022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 25), 'self', False)
        # Obtaining the member 'height' of a type (line 283)
        height_1023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 25), self_1022, 'height')
        # Processing the call keyword arguments (line 283)
        kwargs_1024 = {}
        # Getting the type of 'xrange' (line 283)
        xrange_1021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 18), 'xrange', False)
        # Calling xrange(args, kwargs) (line 283)
        xrange_call_result_1025 = invoke(stypy.reporting.localization.Localization(__file__, 283, 18), xrange_1021, *[height_1023], **kwargs_1024)
        
        # Assigning a type to the variable 'xrange_call_result_1025' (line 283)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'xrange_call_result_1025', xrange_call_result_1025)
        # Testing if the for loop is going to be iterated (line 283)
        # Testing the type of a for loop iterable (line 283)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 283, 8), xrange_call_result_1025)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 283, 8), xrange_call_result_1025):
            # Getting the type of the for loop variable (line 283)
            for_loop_var_1026 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 283, 8), xrange_call_result_1025)
            # Assigning a type to the variable 'sy' (line 283)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 283, 8), 'sy', for_loop_var_1026)
            # SSA begins for a for statement (line 283)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a List to a Name (line 284):
            
            # Assigning a List to a Name (line 284):
            
            # Assigning a List to a Name (line 284):
            
            # Obtaining an instance of the builtin type 'list' (line 284)
            list_1027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 284, 25), 'list')
            # Adding type elements to the builtin type 'list' instance (line 284)
            
            # Assigning a type to the variable 'pixel_line' (line 284)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 284, 12), 'pixel_line', list_1027)
            
            
            # Call to xrange(...): (line 285)
            # Processing the call arguments (line 285)
            # Getting the type of 'self' (line 285)
            self_1029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 29), 'self', False)
            # Obtaining the member 'width' of a type (line 285)
            width_1030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 29), self_1029, 'width')
            # Processing the call keyword arguments (line 285)
            kwargs_1031 = {}
            # Getting the type of 'xrange' (line 285)
            xrange_1028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 22), 'xrange', False)
            # Calling xrange(args, kwargs) (line 285)
            xrange_call_result_1032 = invoke(stypy.reporting.localization.Localization(__file__, 285, 22), xrange_1028, *[width_1030], **kwargs_1031)
            
            # Assigning a type to the variable 'xrange_call_result_1032' (line 285)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'xrange_call_result_1032', xrange_call_result_1032)
            # Testing if the for loop is going to be iterated (line 285)
            # Testing the type of a for loop iterable (line 285)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 285, 12), xrange_call_result_1032)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 285, 12), xrange_call_result_1032):
                # Getting the type of the for loop variable (line 285)
                for_loop_var_1033 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 285, 12), xrange_call_result_1032)
                # Assigning a type to the variable 'sx' (line 285)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 12), 'sx', for_loop_var_1033)
                # SSA begins for a for statement (line 285)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a BinOp to a Name (line 286):
                
                # Assigning a BinOp to a Name (line 286):
                
                # Assigning a BinOp to a Name (line 286):
                int_1034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 20), 'int')
                float_1035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 286, 25), 'float')
                # Getting the type of 'sx' (line 286)
                sx_1036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 31), 'sx')
                
                # Call to float(...): (line 286)
                # Processing the call arguments (line 286)
                # Getting the type of 'self' (line 286)
                self_1038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 42), 'self', False)
                # Obtaining the member 'width' of a type (line 286)
                width_1039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 42), self_1038, 'width')
                # Processing the call keyword arguments (line 286)
                kwargs_1040 = {}
                # Getting the type of 'float' (line 286)
                float_1037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 36), 'float', False)
                # Calling float(args, kwargs) (line 286)
                float_call_result_1041 = invoke(stypy.reporting.localization.Localization(__file__, 286, 36), float_1037, *[width_1039], **kwargs_1040)
                
                # Applying the binary operator 'div' (line 286)
                result_div_1042 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 31), 'div', sx_1036, float_call_result_1041)
                
                # Applying the binary operator '-' (line 286)
                result_sub_1043 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 25), '-', float_1035, result_div_1042)
                
                # Applying the binary operator '*' (line 286)
                result_mul_1044 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 20), '*', int_1034, result_sub_1043)
                
                # Getting the type of 'self' (line 286)
                self_1045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 57), 'self')
                # Obtaining the member 'aspect' of a type (line 286)
                aspect_1046 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 286, 57), self_1045, 'aspect')
                # Applying the binary operator '*' (line 286)
                result_mul_1047 = python_operator(stypy.reporting.localization.Localization(__file__, 286, 55), '*', result_mul_1044, aspect_1046)
                
                # Assigning a type to the variable 'x' (line 286)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 16), 'x', result_mul_1047)
                
                # Assigning a BinOp to a Name (line 287):
                
                # Assigning a BinOp to a Name (line 287):
                
                # Assigning a BinOp to a Name (line 287):
                int_1048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 20), 'int')
                float_1049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 25), 'float')
                # Getting the type of 'sy' (line 287)
                sy_1050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 31), 'sy')
                
                # Call to float(...): (line 287)
                # Processing the call arguments (line 287)
                # Getting the type of 'self' (line 287)
                self_1052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 42), 'self', False)
                # Obtaining the member 'height' of a type (line 287)
                height_1053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 287, 42), self_1052, 'height')
                # Processing the call keyword arguments (line 287)
                kwargs_1054 = {}
                # Getting the type of 'float' (line 287)
                float_1051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 287, 36), 'float', False)
                # Calling float(args, kwargs) (line 287)
                float_call_result_1055 = invoke(stypy.reporting.localization.Localization(__file__, 287, 36), float_1051, *[height_1053], **kwargs_1054)
                
                # Applying the binary operator 'div' (line 287)
                result_div_1056 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 31), 'div', sy_1050, float_call_result_1055)
                
                # Applying the binary operator '-' (line 287)
                result_sub_1057 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 25), '-', float_1049, result_div_1056)
                
                # Applying the binary operator '*' (line 287)
                result_mul_1058 = python_operator(stypy.reporting.localization.Localization(__file__, 287, 20), '*', int_1048, result_sub_1057)
                
                # Assigning a type to the variable 'y' (line 287)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 287, 16), 'y', result_mul_1058)
                
                # Getting the type of 'self' (line 288)
                self_1059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 19), 'self')
                # Obtaining the member 'cameratype' of a type (line 288)
                cameratype_1060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 288, 19), self_1059, 'cameratype')
                str_1061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 38), 'str', 'ortho')
                # Applying the binary operator '==' (line 288)
                result_eq_1062 = python_operator(stypy.reporting.localization.Localization(__file__, 288, 19), '==', cameratype_1060, str_1061)
                
                # Testing if the type of an if condition is none (line 288)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 288, 16), result_eq_1062):
                    
                    # Assigning a Call to a Name (line 291):
                    
                    # Assigning a Call to a Name (line 291):
                    
                    # Assigning a Call to a Name (line 291):
                    
                    # Call to line(...): (line 291)
                    # Processing the call arguments (line 291)
                    
                    # Call to vec(...): (line 291)
                    # Processing the call arguments (line 291)
                    float_1082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 35), 'float')
                    float_1083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 40), 'float')
                    float_1084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 45), 'float')
                    # Processing the call keyword arguments (line 291)
                    kwargs_1085 = {}
                    # Getting the type of 'vec' (line 291)
                    vec_1081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 31), 'vec', False)
                    # Calling vec(args, kwargs) (line 291)
                    vec_call_result_1086 = invoke(stypy.reporting.localization.Localization(__file__, 291, 31), vec_1081, *[float_1082, float_1083, float_1084], **kwargs_1085)
                    
                    
                    # Call to vec(...): (line 291)
                    # Processing the call arguments (line 291)
                    # Getting the type of 'x' (line 291)
                    x_1088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 55), 'x', False)
                    # Getting the type of 'y' (line 291)
                    y_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 58), 'y', False)
                    # Getting the type of 'self' (line 291)
                    self_1090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 61), 'self', False)
                    # Obtaining the member 'imageplane' of a type (line 291)
                    imageplane_1091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 61), self_1090, 'imageplane')
                    # Processing the call keyword arguments (line 291)
                    kwargs_1092 = {}
                    # Getting the type of 'vec' (line 291)
                    vec_1087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 51), 'vec', False)
                    # Calling vec(args, kwargs) (line 291)
                    vec_call_result_1093 = invoke(stypy.reporting.localization.Localization(__file__, 291, 51), vec_1087, *[x_1088, y_1089, imageplane_1091], **kwargs_1092)
                    
                    # Processing the call keyword arguments (line 291)
                    kwargs_1094 = {}
                    # Getting the type of 'line' (line 291)
                    line_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 26), 'line', False)
                    # Calling line(args, kwargs) (line 291)
                    line_call_result_1095 = invoke(stypy.reporting.localization.Localization(__file__, 291, 26), line_1080, *[vec_call_result_1086, vec_call_result_1093], **kwargs_1094)
                    
                    # Assigning a type to the variable 'ray' (line 291)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'ray', line_call_result_1095)
                    
                    # Assigning a BinOp to a Attribute (line 292):
                    
                    # Assigning a BinOp to a Attribute (line 292):
                    
                    # Assigning a BinOp to a Attribute (line 292):
                    # Getting the type of 'ray' (line 292)
                    ray_1096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'ray')
                    # Obtaining the member 'end' of a type (line 292)
                    end_1097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 30), ray_1096, 'end')
                    # Getting the type of 'self' (line 292)
                    self_1098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 40), 'self')
                    # Obtaining the member 'backplane' of a type (line 292)
                    backplane_1099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 40), self_1098, 'backplane')
                    # Applying the binary operator '*' (line 292)
                    result_mul_1100 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 30), '*', end_1097, backplane_1099)
                    
                    # Getting the type of 'ray' (line 292)
                    ray_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 20), 'ray')
                    # Setting the type of the member 'end' of a type (line 292)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 20), ray_1101, 'end', result_mul_1100)
                else:
                    
                    # Testing the type of an if condition (line 288)
                    if_condition_1063 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 288, 16), result_eq_1062)
                    # Assigning a type to the variable 'if_condition_1063' (line 288)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 288, 16), 'if_condition_1063', if_condition_1063)
                    # SSA begins for if statement (line 288)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 289):
                    
                    # Assigning a Call to a Name (line 289):
                    
                    # Assigning a Call to a Name (line 289):
                    
                    # Call to line(...): (line 289)
                    # Processing the call arguments (line 289)
                    
                    # Call to vec(...): (line 289)
                    # Processing the call arguments (line 289)
                    # Getting the type of 'x' (line 289)
                    x_1066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 35), 'x', False)
                    # Getting the type of 'y' (line 289)
                    y_1067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 38), 'y', False)
                    float_1068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 41), 'float')
                    # Processing the call keyword arguments (line 289)
                    kwargs_1069 = {}
                    # Getting the type of 'vec' (line 289)
                    vec_1065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 31), 'vec', False)
                    # Calling vec(args, kwargs) (line 289)
                    vec_call_result_1070 = invoke(stypy.reporting.localization.Localization(__file__, 289, 31), vec_1065, *[x_1066, y_1067, float_1068], **kwargs_1069)
                    
                    
                    # Call to vec(...): (line 289)
                    # Processing the call arguments (line 289)
                    # Getting the type of 'x' (line 289)
                    x_1072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 51), 'x', False)
                    # Getting the type of 'y' (line 289)
                    y_1073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 54), 'y', False)
                    # Getting the type of 'self' (line 289)
                    self_1074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 57), 'self', False)
                    # Obtaining the member 'backplane' of a type (line 289)
                    backplane_1075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 57), self_1074, 'backplane')
                    # Processing the call keyword arguments (line 289)
                    kwargs_1076 = {}
                    # Getting the type of 'vec' (line 289)
                    vec_1071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 47), 'vec', False)
                    # Calling vec(args, kwargs) (line 289)
                    vec_call_result_1077 = invoke(stypy.reporting.localization.Localization(__file__, 289, 47), vec_1071, *[x_1072, y_1073, backplane_1075], **kwargs_1076)
                    
                    # Processing the call keyword arguments (line 289)
                    kwargs_1078 = {}
                    # Getting the type of 'line' (line 289)
                    line_1064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'line', False)
                    # Calling line(args, kwargs) (line 289)
                    line_call_result_1079 = invoke(stypy.reporting.localization.Localization(__file__, 289, 26), line_1064, *[vec_call_result_1070, vec_call_result_1077], **kwargs_1078)
                    
                    # Assigning a type to the variable 'ray' (line 289)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 289, 20), 'ray', line_call_result_1079)
                    # SSA branch for the else part of an if statement (line 288)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Name (line 291):
                    
                    # Assigning a Call to a Name (line 291):
                    
                    # Assigning a Call to a Name (line 291):
                    
                    # Call to line(...): (line 291)
                    # Processing the call arguments (line 291)
                    
                    # Call to vec(...): (line 291)
                    # Processing the call arguments (line 291)
                    float_1082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 35), 'float')
                    float_1083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 40), 'float')
                    float_1084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 45), 'float')
                    # Processing the call keyword arguments (line 291)
                    kwargs_1085 = {}
                    # Getting the type of 'vec' (line 291)
                    vec_1081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 31), 'vec', False)
                    # Calling vec(args, kwargs) (line 291)
                    vec_call_result_1086 = invoke(stypy.reporting.localization.Localization(__file__, 291, 31), vec_1081, *[float_1082, float_1083, float_1084], **kwargs_1085)
                    
                    
                    # Call to vec(...): (line 291)
                    # Processing the call arguments (line 291)
                    # Getting the type of 'x' (line 291)
                    x_1088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 55), 'x', False)
                    # Getting the type of 'y' (line 291)
                    y_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 58), 'y', False)
                    # Getting the type of 'self' (line 291)
                    self_1090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 61), 'self', False)
                    # Obtaining the member 'imageplane' of a type (line 291)
                    imageplane_1091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 291, 61), self_1090, 'imageplane')
                    # Processing the call keyword arguments (line 291)
                    kwargs_1092 = {}
                    # Getting the type of 'vec' (line 291)
                    vec_1087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 51), 'vec', False)
                    # Calling vec(args, kwargs) (line 291)
                    vec_call_result_1093 = invoke(stypy.reporting.localization.Localization(__file__, 291, 51), vec_1087, *[x_1088, y_1089, imageplane_1091], **kwargs_1092)
                    
                    # Processing the call keyword arguments (line 291)
                    kwargs_1094 = {}
                    # Getting the type of 'line' (line 291)
                    line_1080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 26), 'line', False)
                    # Calling line(args, kwargs) (line 291)
                    line_call_result_1095 = invoke(stypy.reporting.localization.Localization(__file__, 291, 26), line_1080, *[vec_call_result_1086, vec_call_result_1093], **kwargs_1094)
                    
                    # Assigning a type to the variable 'ray' (line 291)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 291, 20), 'ray', line_call_result_1095)
                    
                    # Assigning a BinOp to a Attribute (line 292):
                    
                    # Assigning a BinOp to a Attribute (line 292):
                    
                    # Assigning a BinOp to a Attribute (line 292):
                    # Getting the type of 'ray' (line 292)
                    ray_1096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 30), 'ray')
                    # Obtaining the member 'end' of a type (line 292)
                    end_1097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 30), ray_1096, 'end')
                    # Getting the type of 'self' (line 292)
                    self_1098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 40), 'self')
                    # Obtaining the member 'backplane' of a type (line 292)
                    backplane_1099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 40), self_1098, 'backplane')
                    # Applying the binary operator '*' (line 292)
                    result_mul_1100 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 30), '*', end_1097, backplane_1099)
                    
                    # Getting the type of 'ray' (line 292)
                    ray_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 20), 'ray')
                    # Setting the type of the member 'end' of a type (line 292)
                    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 292, 20), ray_1101, 'end', result_mul_1100)
                    # SSA join for if statement (line 288)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Call to a Name (line 294):
                
                # Assigning a Call to a Name (line 294):
                
                # Assigning a Call to a Name (line 294):
                
                # Call to vec(...): (line 294)
                # Processing the call arguments (line 294)
                float_1103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 26), 'float')
                float_1104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 31), 'float')
                float_1105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 294, 36), 'float')
                # Processing the call keyword arguments (line 294)
                kwargs_1106 = {}
                # Getting the type of 'vec' (line 294)
                vec_1102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 22), 'vec', False)
                # Calling vec(args, kwargs) (line 294)
                vec_call_result_1107 = invoke(stypy.reporting.localization.Localization(__file__, 294, 22), vec_1102, *[float_1103, float_1104, float_1105], **kwargs_1106)
                
                # Assigning a type to the variable 'col' (line 294)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 294, 16), 'col', vec_call_result_1107)
                
                # Assigning a Attribute to a Name (line 295):
                
                # Assigning a Attribute to a Name (line 295):
                
                # Assigning a Attribute to a Name (line 295):
                # Getting the type of 'self' (line 295)
                self_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 24), 'self')
                # Obtaining the member 'backplane' of a type (line 295)
                backplane_1109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 295, 24), self_1108, 'backplane')
                # Assigning a type to the variable 'depth' (line 295)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 295, 16), 'depth', backplane_1109)
                
                # Assigning a Call to a Name (line 296):
                
                # Assigning a Call to a Name (line 296):
                
                # Assigning a Call to a Name (line 296):
                
                # Call to Shaderinfo(...): (line 296)
                # Processing the call keyword arguments (line 296)
                kwargs_1111 = {}
                # Getting the type of 'Shaderinfo' (line 296)
                Shaderinfo_1110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 29), 'Shaderinfo', False)
                # Calling Shaderinfo(args, kwargs) (line 296)
                Shaderinfo_call_result_1112 = invoke(stypy.reporting.localization.Localization(__file__, 296, 29), Shaderinfo_1110, *[], **kwargs_1111)
                
                # Assigning a type to the variable 'shaderinfo' (line 296)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 16), 'shaderinfo', Shaderinfo_call_result_1112)
                
                # Assigning a Name to a Attribute (line 297):
                
                # Assigning a Name to a Attribute (line 297):
                
                # Assigning a Name to a Attribute (line 297):
                # Getting the type of 'ray' (line 297)
                ray_1113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 33), 'ray')
                # Getting the type of 'shaderinfo' (line 297)
                shaderinfo_1114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 16), 'shaderinfo')
                # Setting the type of the member 'ray' of a type (line 297)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 16), shaderinfo_1114, 'ray', ray_1113)
                
                # Assigning a Attribute to a Attribute (line 298):
                
                # Assigning a Attribute to a Attribute (line 298):
                
                # Assigning a Attribute to a Attribute (line 298):
                # Getting the type of 'self' (line 298)
                self_1115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 36), 'self')
                # Obtaining the member 'lights' of a type (line 298)
                lights_1116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 36), self_1115, 'lights')
                # Getting the type of 'shaderinfo' (line 298)
                shaderinfo_1117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 298, 16), 'shaderinfo')
                # Setting the type of the member 'lights' of a type (line 298)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 298, 16), shaderinfo_1117, 'lights', lights_1116)
                
                # Assigning a Attribute to a Attribute (line 299):
                
                # Assigning a Attribute to a Attribute (line 299):
                
                # Assigning a Attribute to a Attribute (line 299):
                # Getting the type of 'self' (line 299)
                self_1118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 37), 'self')
                # Obtaining the member 'objects' of a type (line 299)
                objects_1119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 37), self_1118, 'objects')
                # Getting the type of 'shaderinfo' (line 299)
                shaderinfo_1120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 299, 16), 'shaderinfo')
                # Setting the type of the member 'objects' of a type (line 299)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 299, 16), shaderinfo_1120, 'objects', objects_1119)
                
                # Assigning a Num to a Attribute (line 300):
                
                # Assigning a Num to a Attribute (line 300):
                
                # Assigning a Num to a Attribute (line 300):
                int_1121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 300, 35), 'int')
                # Getting the type of 'shaderinfo' (line 300)
                shaderinfo_1122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 16), 'shaderinfo')
                # Setting the type of the member 'depth' of a type (line 300)
                module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 16), shaderinfo_1122, 'depth', int_1121)
                
                # Getting the type of 'self' (line 302)
                self_1123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 27), 'self')
                # Obtaining the member 'objects' of a type (line 302)
                objects_1124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 27), self_1123, 'objects')
                # Assigning a type to the variable 'objects_1124' (line 302)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'objects_1124', objects_1124)
                # Testing if the for loop is going to be iterated (line 302)
                # Testing the type of a for loop iterable (line 302)
                is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 302, 16), objects_1124)

                if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 302, 16), objects_1124):
                    # Getting the type of the for loop variable (line 302)
                    for_loop_var_1125 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 302, 16), objects_1124)
                    # Assigning a type to the variable 'obj' (line 302)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 302, 16), 'obj', for_loop_var_1125)
                    # SSA begins for a for statement (line 302)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                    
                    # Assigning a Call to a Tuple (line 303):
                    
                    # Assigning a Call to a Name:
                    
                    # Assigning a Call to a Name:
                    
                    # Call to intersect(...): (line 303)
                    # Processing the call arguments (line 303)
                    # Getting the type of 'ray' (line 303)
                    ray_1128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 67), 'ray', False)
                    # Processing the call keyword arguments (line 303)
                    kwargs_1129 = {}
                    # Getting the type of 'obj' (line 303)
                    obj_1126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 53), 'obj', False)
                    # Obtaining the member 'intersect' of a type (line 303)
                    intersect_1127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 53), obj_1126, 'intersect')
                    # Calling intersect(args, kwargs) (line 303)
                    intersect_call_result_1130 = invoke(stypy.reporting.localization.Localization(__file__, 303, 53), intersect_1127, *[ray_1128], **kwargs_1129)
                    
                    # Assigning a type to the variable 'call_assignment_10' (line 303)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'call_assignment_10', intersect_call_result_1130)
                    
                    # Assigning a Call to a Name (line 303):
                    
                    # Assigning a Call to a Name (line 303):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_10' (line 303)
                    call_assignment_10_1131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'call_assignment_10', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_1132 = stypy_get_value_from_tuple(call_assignment_10_1131, 2, 0)
                    
                    # Assigning a type to the variable 'call_assignment_11' (line 303)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'call_assignment_11', stypy_get_value_from_tuple_call_result_1132)
                    
                    # Assigning a Name to a Name (line 303):
                    
                    # Assigning a Name to a Name (line 303):
                    # Getting the type of 'call_assignment_11' (line 303)
                    call_assignment_11_1133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'call_assignment_11')
                    # Assigning a type to the variable 'intersects' (line 303)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'intersects', call_assignment_11_1133)
                    
                    # Assigning a Call to a Name (line 303):
                    
                    # Assigning a Call to a Name (line 303):
                    
                    # Call to stypy_get_value_from_tuple(...):
                    # Processing the call arguments
                    # Getting the type of 'call_assignment_10' (line 303)
                    call_assignment_10_1134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'call_assignment_10', False)
                    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                    stypy_get_value_from_tuple_call_result_1135 = stypy_get_value_from_tuple(call_assignment_10_1134, 2, 1)
                    
                    # Assigning a type to the variable 'call_assignment_12' (line 303)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'call_assignment_12', stypy_get_value_from_tuple_call_result_1135)
                    
                    # Assigning a Name to a Tuple (line 303):
                    
                    # Assigning a Subscript to a Name (line 303):
                    
                    # Obtaining the type of the subscript
                    int_1136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 20), 'int')
                    # Getting the type of 'call_assignment_12' (line 303)
                    call_assignment_12_1137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'call_assignment_12')
                    # Obtaining the member '__getitem__' of a type (line 303)
                    getitem___1138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 20), call_assignment_12_1137, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
                    subscript_call_result_1139 = invoke(stypy.reporting.localization.Localization(__file__, 303, 20), getitem___1138, int_1136)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_21' (line 303)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'tuple_var_assignment_21', subscript_call_result_1139)
                    
                    # Assigning a Subscript to a Name (line 303):
                    
                    # Obtaining the type of the subscript
                    int_1140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 303, 20), 'int')
                    # Getting the type of 'call_assignment_12' (line 303)
                    call_assignment_12_1141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'call_assignment_12')
                    # Obtaining the member '__getitem__' of a type (line 303)
                    getitem___1142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 20), call_assignment_12_1141, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 303)
                    subscript_call_result_1143 = invoke(stypy.reporting.localization.Localization(__file__, 303, 20), getitem___1142, int_1140)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_22' (line 303)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'tuple_var_assignment_22', subscript_call_result_1143)
                    
                    # Assigning a Name to a Name (line 303):
                    # Getting the type of 'tuple_var_assignment_21' (line 303)
                    tuple_var_assignment_21_1144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'tuple_var_assignment_21')
                    # Assigning a type to the variable 'position' (line 303)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 33), 'position', tuple_var_assignment_21_1144)
                    
                    # Assigning a Name to a Name (line 303):
                    # Getting the type of 'tuple_var_assignment_22' (line 303)
                    tuple_var_assignment_22_1145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 20), 'tuple_var_assignment_22')
                    # Assigning a type to the variable 'normal' (line 303)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 43), 'normal', tuple_var_assignment_22_1145)
                    
                    # Getting the type of 'intersects' (line 304)
                    intersects_1146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 23), 'intersects')
                    str_1147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 41), 'str', 'none')
                    # Applying the binary operator 'isnot' (line 304)
                    result_is_not_1148 = python_operator(stypy.reporting.localization.Localization(__file__, 304, 23), 'isnot', intersects_1146, str_1147)
                    
                    # Testing if the type of an if condition is none (line 304)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 304, 20), result_is_not_1148):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 304)
                        if_condition_1149 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 304, 20), result_is_not_1148)
                        # Assigning a type to the variable 'if_condition_1149' (line 304)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 304, 20), 'if_condition_1149', if_condition_1149)
                        # SSA begins for if statement (line 304)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Evaluating a boolean operation
                        
                        # Getting the type of 'position' (line 305)
                        position_1150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 27), 'position')
                        # Obtaining the member 'z' of a type (line 305)
                        z_1151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 27), position_1150, 'z')
                        # Getting the type of 'depth' (line 305)
                        depth_1152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 40), 'depth')
                        # Applying the binary operator '<' (line 305)
                        result_lt_1153 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 27), '<', z_1151, depth_1152)
                        
                        
                        # Getting the type of 'position' (line 305)
                        position_1154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 50), 'position')
                        # Obtaining the member 'z' of a type (line 305)
                        z_1155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 50), position_1154, 'z')
                        int_1156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 63), 'int')
                        # Applying the binary operator '>' (line 305)
                        result_gt_1157 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 50), '>', z_1155, int_1156)
                        
                        # Applying the binary operator 'and' (line 305)
                        result_and_keyword_1158 = python_operator(stypy.reporting.localization.Localization(__file__, 305, 27), 'and', result_lt_1153, result_gt_1157)
                        
                        # Testing if the type of an if condition is none (line 305)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 305, 24), result_and_keyword_1158):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 305)
                            if_condition_1159 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 305, 24), result_and_keyword_1158)
                            # Assigning a type to the variable 'if_condition_1159' (line 305)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 305, 24), 'if_condition_1159', if_condition_1159)
                            # SSA begins for if statement (line 305)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Attribute to a Name (line 306):
                            
                            # Assigning a Attribute to a Name (line 306):
                            
                            # Assigning a Attribute to a Name (line 306):
                            # Getting the type of 'position' (line 306)
                            position_1160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 36), 'position')
                            # Obtaining the member 'z' of a type (line 306)
                            z_1161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 36), position_1160, 'z')
                            # Assigning a type to the variable 'depth' (line 306)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 28), 'depth', z_1161)
                            
                            # Assigning a Name to a Attribute (line 307):
                            
                            # Assigning a Name to a Attribute (line 307):
                            
                            # Assigning a Name to a Attribute (line 307):
                            # Getting the type of 'obj' (line 307)
                            obj_1162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 49), 'obj')
                            # Getting the type of 'shaderinfo' (line 307)
                            shaderinfo_1163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 28), 'shaderinfo')
                            # Setting the type of the member 'thisobj' of a type (line 307)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 28), shaderinfo_1163, 'thisobj', obj_1162)
                            
                            # Assigning a Name to a Attribute (line 308):
                            
                            # Assigning a Name to a Attribute (line 308):
                            
                            # Assigning a Name to a Attribute (line 308):
                            # Getting the type of 'position' (line 308)
                            position_1164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 50), 'position')
                            # Getting the type of 'shaderinfo' (line 308)
                            shaderinfo_1165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 28), 'shaderinfo')
                            # Setting the type of the member 'position' of a type (line 308)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 28), shaderinfo_1165, 'position', position_1164)
                            
                            # Assigning a Name to a Attribute (line 309):
                            
                            # Assigning a Name to a Attribute (line 309):
                            
                            # Assigning a Name to a Attribute (line 309):
                            # Getting the type of 'normal' (line 309)
                            normal_1166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 48), 'normal')
                            # Getting the type of 'shaderinfo' (line 309)
                            shaderinfo_1167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 309, 28), 'shaderinfo')
                            # Setting the type of the member 'normal' of a type (line 309)
                            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 309, 28), shaderinfo_1167, 'normal', normal_1166)
                            
                            # Assigning a Call to a Name (line 310):
                            
                            # Assigning a Call to a Name (line 310):
                            
                            # Assigning a Call to a Name (line 310):
                            
                            # Call to shade(...): (line 310)
                            # Processing the call arguments (line 310)
                            # Getting the type of 'shaderinfo' (line 310)
                            shaderinfo_1171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 51), 'shaderinfo', False)
                            # Processing the call keyword arguments (line 310)
                            kwargs_1172 = {}
                            # Getting the type of 'obj' (line 310)
                            obj_1168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 310, 34), 'obj', False)
                            # Obtaining the member 'shader' of a type (line 310)
                            shader_1169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 34), obj_1168, 'shader')
                            # Obtaining the member 'shade' of a type (line 310)
                            shade_1170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 310, 34), shader_1169, 'shade')
                            # Calling shade(args, kwargs) (line 310)
                            shade_call_result_1173 = invoke(stypy.reporting.localization.Localization(__file__, 310, 34), shade_1170, *[shaderinfo_1171], **kwargs_1172)
                            
                            # Assigning a type to the variable 'col' (line 310)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 310, 28), 'col', shade_call_result_1173)
                            # SSA join for if statement (line 305)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 304)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for a for statement
                    module_type_store = module_type_store.join_ssa_context()

                
                
                # Call to append(...): (line 312)
                # Processing the call arguments (line 312)
                
                # Call to conv_value(...): (line 312)
                # Processing the call arguments (line 312)
                # Getting the type of 'col' (line 312)
                col_1177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 45), 'col', False)
                # Obtaining the member 'x' of a type (line 312)
                x_1178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 45), col_1177, 'x')
                # Processing the call keyword arguments (line 312)
                kwargs_1179 = {}
                # Getting the type of 'conv_value' (line 312)
                conv_value_1176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 34), 'conv_value', False)
                # Calling conv_value(args, kwargs) (line 312)
                conv_value_call_result_1180 = invoke(stypy.reporting.localization.Localization(__file__, 312, 34), conv_value_1176, *[x_1178], **kwargs_1179)
                
                # Processing the call keyword arguments (line 312)
                kwargs_1181 = {}
                # Getting the type of 'pixel_line' (line 312)
                pixel_line_1174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 312, 16), 'pixel_line', False)
                # Obtaining the member 'append' of a type (line 312)
                append_1175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 312, 16), pixel_line_1174, 'append')
                # Calling append(args, kwargs) (line 312)
                append_call_result_1182 = invoke(stypy.reporting.localization.Localization(__file__, 312, 16), append_1175, *[conv_value_call_result_1180], **kwargs_1181)
                
                
                # Call to append(...): (line 313)
                # Processing the call arguments (line 313)
                
                # Call to conv_value(...): (line 313)
                # Processing the call arguments (line 313)
                # Getting the type of 'col' (line 313)
                col_1186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 45), 'col', False)
                # Obtaining the member 'y' of a type (line 313)
                y_1187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 45), col_1186, 'y')
                # Processing the call keyword arguments (line 313)
                kwargs_1188 = {}
                # Getting the type of 'conv_value' (line 313)
                conv_value_1185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 34), 'conv_value', False)
                # Calling conv_value(args, kwargs) (line 313)
                conv_value_call_result_1189 = invoke(stypy.reporting.localization.Localization(__file__, 313, 34), conv_value_1185, *[y_1187], **kwargs_1188)
                
                # Processing the call keyword arguments (line 313)
                kwargs_1190 = {}
                # Getting the type of 'pixel_line' (line 313)
                pixel_line_1183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 313, 16), 'pixel_line', False)
                # Obtaining the member 'append' of a type (line 313)
                append_1184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 313, 16), pixel_line_1183, 'append')
                # Calling append(args, kwargs) (line 313)
                append_call_result_1191 = invoke(stypy.reporting.localization.Localization(__file__, 313, 16), append_1184, *[conv_value_call_result_1189], **kwargs_1190)
                
                
                # Call to append(...): (line 314)
                # Processing the call arguments (line 314)
                
                # Call to conv_value(...): (line 314)
                # Processing the call arguments (line 314)
                # Getting the type of 'col' (line 314)
                col_1195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 45), 'col', False)
                # Obtaining the member 'z' of a type (line 314)
                z_1196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 45), col_1195, 'z')
                # Processing the call keyword arguments (line 314)
                kwargs_1197 = {}
                # Getting the type of 'conv_value' (line 314)
                conv_value_1194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 34), 'conv_value', False)
                # Calling conv_value(args, kwargs) (line 314)
                conv_value_call_result_1198 = invoke(stypy.reporting.localization.Localization(__file__, 314, 34), conv_value_1194, *[z_1196], **kwargs_1197)
                
                # Processing the call keyword arguments (line 314)
                kwargs_1199 = {}
                # Getting the type of 'pixel_line' (line 314)
                pixel_line_1192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 314, 16), 'pixel_line', False)
                # Obtaining the member 'append' of a type (line 314)
                append_1193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 314, 16), pixel_line_1192, 'append')
                # Calling append(args, kwargs) (line 314)
                append_call_result_1200 = invoke(stypy.reporting.localization.Localization(__file__, 314, 16), append_1193, *[conv_value_call_result_1198], **kwargs_1199)
                
                
                # Assigning a BinOp to a Name (line 315):
                
                # Assigning a BinOp to a Name (line 315):
                
                # Assigning a BinOp to a Name (line 315):
                # Getting the type of 'count' (line 315)
                count_1201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 315, 24), 'count')
                int_1202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 315, 32), 'int')
                # Applying the binary operator '+' (line 315)
                result_add_1203 = python_operator(stypy.reporting.localization.Localization(__file__, 315, 24), '+', count_1201, int_1202)
                
                # Assigning a type to the variable 'count' (line 315)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 315, 16), 'count', result_add_1203)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to join(...): (line 317)
            # Processing the call arguments (line 317)
            # Getting the type of 'pixel_line' (line 317)
            pixel_line_1206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 317, 40), 'pixel_line', False)
            # Processing the call keyword arguments (line 317)
            kwargs_1207 = {}
            str_1204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 317, 31), 'str', ' ')
            # Obtaining the member 'join' of a type (line 317)
            join_1205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 317, 31), str_1204, 'join')
            # Calling join(args, kwargs) (line 317)
            join_call_result_1208 = invoke(stypy.reporting.localization.Localization(__file__, 317, 31), join_1205, *[pixel_line_1206], **kwargs_1207)
            
            
            # Assigning a BinOp to a Name (line 318):
            
            # Assigning a BinOp to a Name (line 318):
            
            # Assigning a BinOp to a Name (line 318):
            
            # Call to str(...): (line 318)
            # Processing the call arguments (line 318)
            
            # Call to int(...): (line 318)
            # Processing the call arguments (line 318)
            # Getting the type of 'count' (line 318)
            count_1211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 34), 'count', False)
            
            # Call to float(...): (line 318)
            # Processing the call arguments (line 318)
            # Getting the type of 'total' (line 318)
            total_1213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 48), 'total', False)
            # Processing the call keyword arguments (line 318)
            kwargs_1214 = {}
            # Getting the type of 'float' (line 318)
            float_1212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 42), 'float', False)
            # Calling float(args, kwargs) (line 318)
            float_call_result_1215 = invoke(stypy.reporting.localization.Localization(__file__, 318, 42), float_1212, *[total_1213], **kwargs_1214)
            
            # Applying the binary operator 'div' (line 318)
            result_div_1216 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 34), 'div', count_1211, float_call_result_1215)
            
            int_1217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 58), 'int')
            # Applying the binary operator '*' (line 318)
            result_mul_1218 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 33), '*', result_div_1216, int_1217)
            
            # Processing the call keyword arguments (line 318)
            kwargs_1219 = {}
            # Getting the type of 'int' (line 318)
            int_1210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 29), 'int', False)
            # Calling int(args, kwargs) (line 318)
            int_call_result_1220 = invoke(stypy.reporting.localization.Localization(__file__, 318, 29), int_1210, *[result_mul_1218], **kwargs_1219)
            
            # Processing the call keyword arguments (line 318)
            kwargs_1221 = {}
            # Getting the type of 'str' (line 318)
            str_1209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 318, 25), 'str', False)
            # Calling str(args, kwargs) (line 318)
            str_call_result_1222 = invoke(stypy.reporting.localization.Localization(__file__, 318, 25), str_1209, *[int_call_result_1220], **kwargs_1221)
            
            str_1223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 318, 66), 'str', '%')
            # Applying the binary operator '+' (line 318)
            result_add_1224 = python_operator(stypy.reporting.localization.Localization(__file__, 318, 25), '+', str_call_result_1222, str_1223)
            
            # Assigning a type to the variable 'percentstr' (line 318)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 318, 12), 'percentstr', result_add_1224)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to close(...): (line 320)
        # Processing the call keyword arguments (line 320)
        kwargs_1227 = {}
        # Getting the type of 'out_file' (line 320)
        out_file_1225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 320, 8), 'out_file', False)
        # Obtaining the member 'close' of a type (line 320)
        close_1226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 320, 8), out_file_1225, 'close')
        # Calling close(args, kwargs) (line 320)
        close_call_result_1228 = invoke(stypy.reporting.localization.Localization(__file__, 320, 8), close_1226, *[], **kwargs_1227)
        
        
        # ################# End of 'render(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'render' in the type store
        # Getting the type of 'stypy_return_type' (line 274)
        stypy_return_type_1229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 274, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1229)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'render'
        return stypy_return_type_1229


# Assigning a type to the variable 'world' (line 263)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 0), 'world', world)
# Declaration of the 'everythingshader' class
# Getting the type of 'shader' (line 323)
shader_1230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 323, 23), 'shader')

class everythingshader(shader_1230, ):

    @norecursion
    def shade(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shade'
        module_type_store = module_type_store.open_function_context('shade', 324, 4, False)
        # Assigning a type to the variable 'self' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        everythingshader.shade.__dict__.__setitem__('stypy_localization', localization)
        everythingshader.shade.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        everythingshader.shade.__dict__.__setitem__('stypy_type_store', module_type_store)
        everythingshader.shade.__dict__.__setitem__('stypy_function_name', 'everythingshader.shade')
        everythingshader.shade.__dict__.__setitem__('stypy_param_names_list', ['shaderinfo'])
        everythingshader.shade.__dict__.__setitem__('stypy_varargs_param_name', None)
        everythingshader.shade.__dict__.__setitem__('stypy_kwargs_param_name', None)
        everythingshader.shade.__dict__.__setitem__('stypy_call_defaults', defaults)
        everythingshader.shade.__dict__.__setitem__('stypy_call_varargs', varargs)
        everythingshader.shade.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        everythingshader.shade.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'everythingshader.shade', ['shaderinfo'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shade', localization, ['shaderinfo'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shade(...)' code ##################

        
        # Assigning a Call to a Name (line 325):
        
        # Assigning a Call to a Name (line 325):
        
        # Assigning a Call to a Name (line 325):
        
        # Call to shade(...): (line 325)
        # Processing the call arguments (line 325)
        # Getting the type of 'self' (line 325)
        self_1233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 27), 'self', False)
        # Getting the type of 'shaderinfo' (line 325)
        shaderinfo_1234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 33), 'shaderinfo', False)
        # Processing the call keyword arguments (line 325)
        kwargs_1235 = {}
        # Getting the type of 'shader' (line 325)
        shader_1231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 325, 14), 'shader', False)
        # Obtaining the member 'shade' of a type (line 325)
        shade_1232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 325, 14), shader_1231, 'shade')
        # Calling shade(args, kwargs) (line 325)
        shade_call_result_1236 = invoke(stypy.reporting.localization.Localization(__file__, 325, 14), shade_1232, *[self_1233, shaderinfo_1234], **kwargs_1235)
        
        # Assigning a type to the variable 'col' (line 325)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 325, 8), 'col', shade_call_result_1236)
        
        # Assigning a Call to a Name (line 326):
        
        # Assigning a Call to a Name (line 326):
        
        # Assigning a Call to a Name (line 326):
        
        # Call to getreflected(...): (line 326)
        # Processing the call arguments (line 326)
        # Getting the type of 'shaderinfo' (line 326)
        shaderinfo_1239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 32), 'shaderinfo', False)
        # Processing the call keyword arguments (line 326)
        kwargs_1240 = {}
        # Getting the type of 'self' (line 326)
        self_1237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 326, 14), 'self', False)
        # Obtaining the member 'getreflected' of a type (line 326)
        getreflected_1238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 326, 14), self_1237, 'getreflected')
        # Calling getreflected(args, kwargs) (line 326)
        getreflected_call_result_1241 = invoke(stypy.reporting.localization.Localization(__file__, 326, 14), getreflected_1238, *[shaderinfo_1239], **kwargs_1240)
        
        # Assigning a type to the variable 'ref' (line 326)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 326, 8), 'ref', getreflected_call_result_1241)
        
        # Assigning a BinOp to a Name (line 327):
        
        # Assigning a BinOp to a Name (line 327):
        
        # Assigning a BinOp to a Name (line 327):
        # Getting the type of 'col' (line 327)
        col_1242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 14), 'col')
        float_1243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 20), 'float')
        # Applying the binary operator '*' (line 327)
        result_mul_1244 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 14), '*', col_1242, float_1243)
        
        # Getting the type of 'ref' (line 327)
        ref_1245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 327, 26), 'ref')
        float_1246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 327, 32), 'float')
        # Applying the binary operator '*' (line 327)
        result_mul_1247 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 26), '*', ref_1245, float_1246)
        
        # Applying the binary operator '+' (line 327)
        result_add_1248 = python_operator(stypy.reporting.localization.Localization(__file__, 327, 14), '+', result_mul_1244, result_mul_1247)
        
        # Assigning a type to the variable 'col' (line 327)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 327, 8), 'col', result_add_1248)
        # Getting the type of 'col' (line 328)
        col_1249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 15), 'col')
        
        # Call to doocclusion(...): (line 328)
        # Processing the call arguments (line 328)
        int_1252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 328, 38), 'int')
        # Getting the type of 'shaderinfo' (line 328)
        shaderinfo_1253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 42), 'shaderinfo', False)
        # Processing the call keyword arguments (line 328)
        kwargs_1254 = {}
        # Getting the type of 'self' (line 328)
        self_1250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 328, 21), 'self', False)
        # Obtaining the member 'doocclusion' of a type (line 328)
        doocclusion_1251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 328, 21), self_1250, 'doocclusion')
        # Calling doocclusion(args, kwargs) (line 328)
        doocclusion_call_result_1255 = invoke(stypy.reporting.localization.Localization(__file__, 328, 21), doocclusion_1251, *[int_1252, shaderinfo_1253], **kwargs_1254)
        
        # Applying the binary operator '*' (line 328)
        result_mul_1256 = python_operator(stypy.reporting.localization.Localization(__file__, 328, 15), '*', col_1249, doocclusion_call_result_1255)
        
        # Assigning a type to the variable 'stypy_return_type' (line 328)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 328, 8), 'stypy_return_type', result_mul_1256)
        
        # ################# End of 'shade(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shade' in the type store
        # Getting the type of 'stypy_return_type' (line 324)
        stypy_return_type_1257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 324, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1257)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shade'
        return stypy_return_type_1257


# Assigning a type to the variable 'everythingshader' (line 323)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 323, 0), 'everythingshader', everythingshader)
# Declaration of the 'spotshader' class
# Getting the type of 'shader' (line 331)
shader_1258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 331, 17), 'shader')

class spotshader(shader_1258, ):

    @norecursion
    def shade(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shade'
        module_type_store = module_type_store.open_function_context('shade', 332, 4, False)
        # Assigning a type to the variable 'self' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        spotshader.shade.__dict__.__setitem__('stypy_localization', localization)
        spotshader.shade.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        spotshader.shade.__dict__.__setitem__('stypy_type_store', module_type_store)
        spotshader.shade.__dict__.__setitem__('stypy_function_name', 'spotshader.shade')
        spotshader.shade.__dict__.__setitem__('stypy_param_names_list', ['shaderinfo'])
        spotshader.shade.__dict__.__setitem__('stypy_varargs_param_name', None)
        spotshader.shade.__dict__.__setitem__('stypy_kwargs_param_name', None)
        spotshader.shade.__dict__.__setitem__('stypy_call_defaults', defaults)
        spotshader.shade.__dict__.__setitem__('stypy_call_varargs', varargs)
        spotshader.shade.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        spotshader.shade.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'spotshader.shade', ['shaderinfo'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shade', localization, ['shaderinfo'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shade(...)' code ##################

        
        # Assigning a Call to a Name (line 333):
        
        # Assigning a Call to a Name (line 333):
        
        # Assigning a Call to a Name (line 333):
        
        # Call to shade(...): (line 333)
        # Processing the call arguments (line 333)
        # Getting the type of 'self' (line 333)
        self_1261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 27), 'self', False)
        # Getting the type of 'shaderinfo' (line 333)
        shaderinfo_1262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 33), 'shaderinfo', False)
        # Processing the call keyword arguments (line 333)
        kwargs_1263 = {}
        # Getting the type of 'shader' (line 333)
        shader_1259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 333, 14), 'shader', False)
        # Obtaining the member 'shade' of a type (line 333)
        shade_1260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 333, 14), shader_1259, 'shade')
        # Calling shade(args, kwargs) (line 333)
        shade_call_result_1264 = invoke(stypy.reporting.localization.Localization(__file__, 333, 14), shade_1260, *[self_1261, shaderinfo_1262], **kwargs_1263)
        
        # Assigning a type to the variable 'col' (line 333)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 333, 8), 'col', shade_call_result_1264)
        
        # Assigning a Attribute to a Name (line 334):
        
        # Assigning a Attribute to a Name (line 334):
        
        # Assigning a Attribute to a Name (line 334):
        # Getting the type of 'shaderinfo' (line 334)
        shaderinfo_1265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 334, 19), 'shaderinfo')
        # Obtaining the member 'position' of a type (line 334)
        position_1266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 334, 19), shaderinfo_1265, 'position')
        # Assigning a type to the variable 'position' (line 334)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 334, 8), 'position', position_1266)
        
        # Assigning a BinOp to a Name (line 335):
        
        # Assigning a BinOp to a Name (line 335):
        
        # Assigning a BinOp to a Name (line 335):
        
        # Call to sin(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'position' (line 335)
        position_1268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 21), 'position', False)
        # Obtaining the member 'x' of a type (line 335)
        x_1269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 21), position_1268, 'x')
        # Processing the call keyword arguments (line 335)
        kwargs_1270 = {}
        # Getting the type of 'sin' (line 335)
        sin_1267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 17), 'sin', False)
        # Calling sin(args, kwargs) (line 335)
        sin_call_result_1271 = invoke(stypy.reporting.localization.Localization(__file__, 335, 17), sin_1267, *[x_1269], **kwargs_1270)
        
        
        # Call to cos(...): (line 335)
        # Processing the call arguments (line 335)
        # Getting the type of 'position' (line 335)
        position_1273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 39), 'position', False)
        # Obtaining the member 'z' of a type (line 335)
        z_1274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 335, 39), position_1273, 'z')
        # Processing the call keyword arguments (line 335)
        kwargs_1275 = {}
        # Getting the type of 'cos' (line 335)
        cos_1272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 335, 35), 'cos', False)
        # Calling cos(args, kwargs) (line 335)
        cos_call_result_1276 = invoke(stypy.reporting.localization.Localization(__file__, 335, 35), cos_1272, *[z_1274], **kwargs_1275)
        
        # Applying the binary operator '+' (line 335)
        result_add_1277 = python_operator(stypy.reporting.localization.Localization(__file__, 335, 17), '+', sin_call_result_1271, cos_call_result_1276)
        
        # Assigning a type to the variable 'jitter' (line 335)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 335, 8), 'jitter', result_add_1277)
        
        # Getting the type of 'jitter' (line 336)
        jitter_1278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 336, 11), 'jitter')
        float_1279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 336, 20), 'float')
        # Applying the binary operator '>' (line 336)
        result_gt_1280 = python_operator(stypy.reporting.localization.Localization(__file__, 336, 11), '>', jitter_1278, float_1279)
        
        # Testing if the type of an if condition is none (line 336)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 336, 8), result_gt_1280):
            pass
        else:
            
            # Testing the type of an if condition (line 336)
            if_condition_1281 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 336, 8), result_gt_1280)
            # Assigning a type to the variable 'if_condition_1281' (line 336)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 336, 8), 'if_condition_1281', if_condition_1281)
            # SSA begins for if statement (line 336)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 337):
            
            # Assigning a BinOp to a Name (line 337):
            
            # Assigning a BinOp to a Name (line 337):
            # Getting the type of 'col' (line 337)
            col_1282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 337, 18), 'col')
            int_1283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 337, 24), 'int')
            # Applying the binary operator 'div' (line 337)
            result_div_1284 = python_operator(stypy.reporting.localization.Localization(__file__, 337, 18), 'div', col_1282, int_1283)
            
            # Assigning a type to the variable 'col' (line 337)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 337, 12), 'col', result_div_1284)
            # SSA join for if statement (line 336)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 338):
        
        # Assigning a Call to a Name (line 338):
        
        # Assigning a Call to a Name (line 338):
        
        # Call to getreflected(...): (line 338)
        # Processing the call arguments (line 338)
        # Getting the type of 'shaderinfo' (line 338)
        shaderinfo_1287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 32), 'shaderinfo', False)
        # Processing the call keyword arguments (line 338)
        kwargs_1288 = {}
        # Getting the type of 'self' (line 338)
        self_1285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 338, 14), 'self', False)
        # Obtaining the member 'getreflected' of a type (line 338)
        getreflected_1286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 338, 14), self_1285, 'getreflected')
        # Calling getreflected(args, kwargs) (line 338)
        getreflected_call_result_1289 = invoke(stypy.reporting.localization.Localization(__file__, 338, 14), getreflected_1286, *[shaderinfo_1287], **kwargs_1288)
        
        # Assigning a type to the variable 'ref' (line 338)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 338, 8), 'ref', getreflected_call_result_1289)
        # Getting the type of 'ref' (line 339)
        ref_1290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 15), 'ref')
        float_1291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 21), 'float')
        # Applying the binary operator '*' (line 339)
        result_mul_1292 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 15), '*', ref_1290, float_1291)
        
        # Getting the type of 'col' (line 339)
        col_1293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 27), 'col')
        float_1294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 33), 'float')
        # Applying the binary operator '*' (line 339)
        result_mul_1295 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 27), '*', col_1293, float_1294)
        
        
        # Call to doocclusion(...): (line 339)
        # Processing the call arguments (line 339)
        int_1298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 339, 56), 'int')
        # Getting the type of 'shaderinfo' (line 339)
        shaderinfo_1299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 60), 'shaderinfo', False)
        # Processing the call keyword arguments (line 339)
        kwargs_1300 = {}
        # Getting the type of 'self' (line 339)
        self_1296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 339, 39), 'self', False)
        # Obtaining the member 'doocclusion' of a type (line 339)
        doocclusion_1297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 339, 39), self_1296, 'doocclusion')
        # Calling doocclusion(args, kwargs) (line 339)
        doocclusion_call_result_1301 = invoke(stypy.reporting.localization.Localization(__file__, 339, 39), doocclusion_1297, *[int_1298, shaderinfo_1299], **kwargs_1300)
        
        # Applying the binary operator '*' (line 339)
        result_mul_1302 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 37), '*', result_mul_1295, doocclusion_call_result_1301)
        
        # Applying the binary operator '+' (line 339)
        result_add_1303 = python_operator(stypy.reporting.localization.Localization(__file__, 339, 15), '+', result_mul_1292, result_mul_1302)
        
        # Assigning a type to the variable 'stypy_return_type' (line 339)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 339, 8), 'stypy_return_type', result_add_1303)
        
        # ################# End of 'shade(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shade' in the type store
        # Getting the type of 'stypy_return_type' (line 332)
        stypy_return_type_1304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 332, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1304)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shade'
        return stypy_return_type_1304


# Assigning a type to the variable 'spotshader' (line 331)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 331, 0), 'spotshader', spotshader)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 342, 0, False)
    
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

    
    # Assigning a Tuple to a Tuple (line 346):
    
    # Assigning a Num to a Name (line 346):
    
    # Assigning a Num to a Name (line 346):
    int_1305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 13), 'int')
    # Assigning a type to the variable 'tuple_assignment_13' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'tuple_assignment_13', int_1305)
    
    # Assigning a Num to a Name (line 346):
    
    # Assigning a Num to a Name (line 346):
    int_1306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 346, 18), 'int')
    # Assigning a type to the variable 'tuple_assignment_14' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'tuple_assignment_14', int_1306)
    
    # Assigning a Name to a Name (line 346):
    
    # Assigning a Name to a Name (line 346):
    # Getting the type of 'tuple_assignment_13' (line 346)
    tuple_assignment_13_1307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'tuple_assignment_13')
    # Assigning a type to the variable 'nx' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'nx', tuple_assignment_13_1307)
    
    # Assigning a Name to a Name (line 346):
    
    # Assigning a Name to a Name (line 346):
    # Getting the type of 'tuple_assignment_14' (line 346)
    tuple_assignment_14_1308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 346, 4), 'tuple_assignment_14')
    # Assigning a type to the variable 'ny' (line 346)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 346, 8), 'ny', tuple_assignment_14_1308)
    
    # Assigning a Call to a Name (line 347):
    
    # Assigning a Call to a Name (line 347):
    
    # Assigning a Call to a Name (line 347):
    
    # Call to world(...): (line 347)
    # Processing the call arguments (line 347)
    # Getting the type of 'nx' (line 347)
    nx_1310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 14), 'nx', False)
    # Getting the type of 'ny' (line 347)
    ny_1311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 18), 'ny', False)
    # Processing the call keyword arguments (line 347)
    kwargs_1312 = {}
    # Getting the type of 'world' (line 347)
    world_1309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 347, 8), 'world', False)
    # Calling world(args, kwargs) (line 347)
    world_call_result_1313 = invoke(stypy.reporting.localization.Localization(__file__, 347, 8), world_1309, *[nx_1310, ny_1311], **kwargs_1312)
    
    # Assigning a type to the variable 'w' (line 347)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 347, 4), 'w', world_call_result_1313)
    
    # Assigning a Num to a Name (line 348):
    
    # Assigning a Num to a Name (line 348):
    
    # Assigning a Num to a Name (line 348):
    float_1314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 348, 15), 'float')
    # Assigning a type to the variable 'numballs' (line 348)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 348, 4), 'numballs', float_1314)
    
    # Assigning a Call to a Name (line 349):
    
    # Assigning a Call to a Name (line 349):
    
    # Assigning a Call to a Name (line 349):
    
    # Call to vec(...): (line 349)
    # Processing the call arguments (line 349)
    float_1316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 17), 'float')
    float_1317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 22), 'float')
    float_1318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 349, 28), 'float')
    # Processing the call keyword arguments (line 349)
    kwargs_1319 = {}
    # Getting the type of 'vec' (line 349)
    vec_1315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 349, 13), 'vec', False)
    # Calling vec(args, kwargs) (line 349)
    vec_call_result_1320 = invoke(stypy.reporting.localization.Localization(__file__, 349, 13), vec_1315, *[float_1316, float_1317, float_1318], **kwargs_1319)
    
    # Assigning a type to the variable 'offset' (line 349)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 349, 4), 'offset', vec_call_result_1320)
    
    # Assigning a Num to a Name (line 350):
    
    # Assigning a Num to a Name (line 350):
    
    # Assigning a Num to a Name (line 350):
    float_1321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 350, 10), 'float')
    # Assigning a type to the variable 'rad' (line 350)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 350, 4), 'rad', float_1321)
    
    # Assigning a BinOp to a Name (line 351):
    
    # Assigning a BinOp to a Name (line 351):
    
    # Assigning a BinOp to a Name (line 351):
    int_1322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 18), 'int')
    float_1323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 351, 22), 'float')
    # Applying the binary operator '*' (line 351)
    result_mul_1324 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 18), '*', int_1322, float_1323)
    
    # Getting the type of 'numballs' (line 351)
    numballs_1325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 351, 34), 'numballs')
    # Applying the binary operator 'div' (line 351)
    result_div_1326 = python_operator(stypy.reporting.localization.Localization(__file__, 351, 17), 'div', result_mul_1324, numballs_1325)
    
    # Assigning a type to the variable 'radperball' (line 351)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 351, 4), 'radperball', result_div_1326)
    
    
    # Call to xrange(...): (line 353)
    # Processing the call arguments (line 353)
    
    # Call to int(...): (line 353)
    # Processing the call arguments (line 353)
    # Getting the type of 'numballs' (line 353)
    numballs_1329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 24), 'numballs', False)
    # Processing the call keyword arguments (line 353)
    kwargs_1330 = {}
    # Getting the type of 'int' (line 353)
    int_1328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 20), 'int', False)
    # Calling int(args, kwargs) (line 353)
    int_call_result_1331 = invoke(stypy.reporting.localization.Localization(__file__, 353, 20), int_1328, *[numballs_1329], **kwargs_1330)
    
    # Processing the call keyword arguments (line 353)
    kwargs_1332 = {}
    # Getting the type of 'xrange' (line 353)
    xrange_1327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 353, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 353)
    xrange_call_result_1333 = invoke(stypy.reporting.localization.Localization(__file__, 353, 13), xrange_1327, *[int_call_result_1331], **kwargs_1332)
    
    # Assigning a type to the variable 'xrange_call_result_1333' (line 353)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'xrange_call_result_1333', xrange_call_result_1333)
    # Testing if the for loop is going to be iterated (line 353)
    # Testing the type of a for loop iterable (line 353)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 353, 4), xrange_call_result_1333)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 353, 4), xrange_call_result_1333):
        # Getting the type of the for loop variable (line 353)
        for_loop_var_1334 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 353, 4), xrange_call_result_1333)
        # Assigning a type to the variable 'i' (line 353)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 353, 4), 'i', for_loop_var_1334)
        # SSA begins for a for statement (line 353)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 354):
        
        # Assigning a BinOp to a Name (line 354):
        
        # Assigning a BinOp to a Name (line 354):
        
        # Call to sin(...): (line 354)
        # Processing the call arguments (line 354)
        float_1336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 354, 16), 'float')
        # Getting the type of 'radperball' (line 354)
        radperball_1337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 22), 'radperball', False)
        
        # Call to float(...): (line 354)
        # Processing the call arguments (line 354)
        # Getting the type of 'i' (line 354)
        i_1339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 41), 'i', False)
        # Processing the call keyword arguments (line 354)
        kwargs_1340 = {}
        # Getting the type of 'float' (line 354)
        float_1338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 35), 'float', False)
        # Calling float(args, kwargs) (line 354)
        float_call_result_1341 = invoke(stypy.reporting.localization.Localization(__file__, 354, 35), float_1338, *[i_1339], **kwargs_1340)
        
        # Applying the binary operator '*' (line 354)
        result_mul_1342 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 22), '*', radperball_1337, float_call_result_1341)
        
        # Applying the binary operator '+' (line 354)
        result_add_1343 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 16), '+', float_1336, result_mul_1342)
        
        # Processing the call keyword arguments (line 354)
        kwargs_1344 = {}
        # Getting the type of 'sin' (line 354)
        sin_1335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 12), 'sin', False)
        # Calling sin(args, kwargs) (line 354)
        sin_call_result_1345 = invoke(stypy.reporting.localization.Localization(__file__, 354, 12), sin_1335, *[result_add_1343], **kwargs_1344)
        
        # Getting the type of 'rad' (line 354)
        rad_1346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 354, 47), 'rad')
        # Applying the binary operator '*' (line 354)
        result_mul_1347 = python_operator(stypy.reporting.localization.Localization(__file__, 354, 12), '*', sin_call_result_1345, rad_1346)
        
        # Assigning a type to the variable 'x' (line 354)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 354, 8), 'x', result_mul_1347)
        
        # Assigning a BinOp to a Name (line 355):
        
        # Assigning a BinOp to a Name (line 355):
        
        # Assigning a BinOp to a Name (line 355):
        
        # Call to cos(...): (line 355)
        # Processing the call arguments (line 355)
        float_1349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 355, 16), 'float')
        # Getting the type of 'radperball' (line 355)
        radperball_1350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 22), 'radperball', False)
        
        # Call to float(...): (line 355)
        # Processing the call arguments (line 355)
        # Getting the type of 'i' (line 355)
        i_1352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 41), 'i', False)
        # Processing the call keyword arguments (line 355)
        kwargs_1353 = {}
        # Getting the type of 'float' (line 355)
        float_1351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 35), 'float', False)
        # Calling float(args, kwargs) (line 355)
        float_call_result_1354 = invoke(stypy.reporting.localization.Localization(__file__, 355, 35), float_1351, *[i_1352], **kwargs_1353)
        
        # Applying the binary operator '*' (line 355)
        result_mul_1355 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 22), '*', radperball_1350, float_call_result_1354)
        
        # Applying the binary operator '+' (line 355)
        result_add_1356 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 16), '+', float_1349, result_mul_1355)
        
        # Processing the call keyword arguments (line 355)
        kwargs_1357 = {}
        # Getting the type of 'cos' (line 355)
        cos_1348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 12), 'cos', False)
        # Calling cos(args, kwargs) (line 355)
        cos_call_result_1358 = invoke(stypy.reporting.localization.Localization(__file__, 355, 12), cos_1348, *[result_add_1356], **kwargs_1357)
        
        # Getting the type of 'rad' (line 355)
        rad_1359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 355, 47), 'rad')
        # Applying the binary operator '*' (line 355)
        result_mul_1360 = python_operator(stypy.reporting.localization.Localization(__file__, 355, 12), '*', cos_call_result_1358, rad_1359)
        
        # Assigning a type to the variable 'y' (line 355)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 355, 8), 'y', result_mul_1360)
        
        # Call to append(...): (line 356)
        # Processing the call arguments (line 356)
        
        # Call to sphere(...): (line 356)
        # Processing the call arguments (line 356)
        
        # Call to vec(...): (line 356)
        # Processing the call arguments (line 356)
        # Getting the type of 'x' (line 356)
        x_1366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 36), 'x', False)
        float_1367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 39), 'float')
        # Getting the type of 'y' (line 356)
        y_1368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 44), 'y', False)
        # Processing the call keyword arguments (line 356)
        kwargs_1369 = {}
        # Getting the type of 'vec' (line 356)
        vec_1365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 32), 'vec', False)
        # Calling vec(args, kwargs) (line 356)
        vec_call_result_1370 = invoke(stypy.reporting.localization.Localization(__file__, 356, 32), vec_1365, *[x_1366, float_1367, y_1368], **kwargs_1369)
        
        # Getting the type of 'offset' (line 356)
        offset_1371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 49), 'offset', False)
        # Applying the binary operator '+' (line 356)
        result_add_1372 = python_operator(stypy.reporting.localization.Localization(__file__, 356, 32), '+', vec_call_result_1370, offset_1371)
        
        float_1373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 356, 57), 'float')
        
        # Call to everythingshader(...): (line 356)
        # Processing the call keyword arguments (line 356)
        kwargs_1375 = {}
        # Getting the type of 'everythingshader' (line 356)
        everythingshader_1374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 62), 'everythingshader', False)
        # Calling everythingshader(args, kwargs) (line 356)
        everythingshader_call_result_1376 = invoke(stypy.reporting.localization.Localization(__file__, 356, 62), everythingshader_1374, *[], **kwargs_1375)
        
        # Processing the call keyword arguments (line 356)
        kwargs_1377 = {}
        # Getting the type of 'sphere' (line 356)
        sphere_1364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 25), 'sphere', False)
        # Calling sphere(args, kwargs) (line 356)
        sphere_call_result_1378 = invoke(stypy.reporting.localization.Localization(__file__, 356, 25), sphere_1364, *[result_add_1372, float_1373, everythingshader_call_result_1376], **kwargs_1377)
        
        # Processing the call keyword arguments (line 356)
        kwargs_1379 = {}
        # Getting the type of 'w' (line 356)
        w_1361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 356, 8), 'w', False)
        # Obtaining the member 'objects' of a type (line 356)
        objects_1362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), w_1361, 'objects')
        # Obtaining the member 'append' of a type (line 356)
        append_1363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 356, 8), objects_1362, 'append')
        # Calling append(args, kwargs) (line 356)
        append_call_result_1380 = invoke(stypy.reporting.localization.Localization(__file__, 356, 8), append_1363, *[sphere_call_result_1378], **kwargs_1379)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to append(...): (line 358)
    # Processing the call arguments (line 358)
    
    # Call to sphere(...): (line 358)
    # Processing the call arguments (line 358)
    
    # Call to vec(...): (line 358)
    # Processing the call arguments (line 358)
    float_1386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 32), 'float')
    float_1387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 37), 'float')
    float_1388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 42), 'float')
    # Processing the call keyword arguments (line 358)
    kwargs_1389 = {}
    # Getting the type of 'vec' (line 358)
    vec_1385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 28), 'vec', False)
    # Calling vec(args, kwargs) (line 358)
    vec_call_result_1390 = invoke(stypy.reporting.localization.Localization(__file__, 358, 28), vec_1385, *[float_1386, float_1387, float_1388], **kwargs_1389)
    
    # Getting the type of 'offset' (line 358)
    offset_1391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 49), 'offset', False)
    # Applying the binary operator '+' (line 358)
    result_add_1392 = python_operator(stypy.reporting.localization.Localization(__file__, 358, 28), '+', vec_call_result_1390, offset_1391)
    
    float_1393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 358, 57), 'float')
    
    # Call to everythingshader(...): (line 358)
    # Processing the call keyword arguments (line 358)
    kwargs_1395 = {}
    # Getting the type of 'everythingshader' (line 358)
    everythingshader_1394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 62), 'everythingshader', False)
    # Calling everythingshader(args, kwargs) (line 358)
    everythingshader_call_result_1396 = invoke(stypy.reporting.localization.Localization(__file__, 358, 62), everythingshader_1394, *[], **kwargs_1395)
    
    # Processing the call keyword arguments (line 358)
    kwargs_1397 = {}
    # Getting the type of 'sphere' (line 358)
    sphere_1384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 21), 'sphere', False)
    # Calling sphere(args, kwargs) (line 358)
    sphere_call_result_1398 = invoke(stypy.reporting.localization.Localization(__file__, 358, 21), sphere_1384, *[result_add_1392, float_1393, everythingshader_call_result_1396], **kwargs_1397)
    
    # Processing the call keyword arguments (line 358)
    kwargs_1399 = {}
    # Getting the type of 'w' (line 358)
    w_1381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 358, 4), 'w', False)
    # Obtaining the member 'objects' of a type (line 358)
    objects_1382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 4), w_1381, 'objects')
    # Obtaining the member 'append' of a type (line 358)
    append_1383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 358, 4), objects_1382, 'append')
    # Calling append(args, kwargs) (line 358)
    append_call_result_1400 = invoke(stypy.reporting.localization.Localization(__file__, 358, 4), append_1383, *[sphere_call_result_1398], **kwargs_1399)
    
    
    # Call to append(...): (line 359)
    # Processing the call arguments (line 359)
    
    # Call to plane(...): (line 359)
    # Processing the call arguments (line 359)
    
    # Call to vec(...): (line 359)
    # Processing the call arguments (line 359)
    float_1406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 31), 'float')
    float_1407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 36), 'float')
    float_1408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 41), 'float')
    # Processing the call keyword arguments (line 359)
    kwargs_1409 = {}
    # Getting the type of 'vec' (line 359)
    vec_1405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 27), 'vec', False)
    # Calling vec(args, kwargs) (line 359)
    vec_call_result_1410 = invoke(stypy.reporting.localization.Localization(__file__, 359, 27), vec_1405, *[float_1406, float_1407, float_1408], **kwargs_1409)
    
    float_1411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 359, 47), 'float')
    
    # Call to spotshader(...): (line 359)
    # Processing the call keyword arguments (line 359)
    kwargs_1413 = {}
    # Getting the type of 'spotshader' (line 359)
    spotshader_1412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 52), 'spotshader', False)
    # Calling spotshader(args, kwargs) (line 359)
    spotshader_call_result_1414 = invoke(stypy.reporting.localization.Localization(__file__, 359, 52), spotshader_1412, *[], **kwargs_1413)
    
    # Processing the call keyword arguments (line 359)
    kwargs_1415 = {}
    # Getting the type of 'plane' (line 359)
    plane_1404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 21), 'plane', False)
    # Calling plane(args, kwargs) (line 359)
    plane_call_result_1416 = invoke(stypy.reporting.localization.Localization(__file__, 359, 21), plane_1404, *[vec_call_result_1410, float_1411, spotshader_call_result_1414], **kwargs_1415)
    
    # Processing the call keyword arguments (line 359)
    kwargs_1417 = {}
    # Getting the type of 'w' (line 359)
    w_1401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 359, 4), 'w', False)
    # Obtaining the member 'objects' of a type (line 359)
    objects_1402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 4), w_1401, 'objects')
    # Obtaining the member 'append' of a type (line 359)
    append_1403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 359, 4), objects_1402, 'append')
    # Calling append(args, kwargs) (line 359)
    append_call_result_1418 = invoke(stypy.reporting.localization.Localization(__file__, 359, 4), append_1403, *[plane_call_result_1416], **kwargs_1417)
    
    
    # Call to append(...): (line 360)
    # Processing the call arguments (line 360)
    
    # Call to parallellight(...): (line 360)
    # Processing the call arguments (line 360)
    
    # Call to vec(...): (line 360)
    # Processing the call arguments (line 360)
    float_1424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 38), 'float')
    float_1425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 43), 'float')
    float_1426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 48), 'float')
    # Processing the call keyword arguments (line 360)
    kwargs_1427 = {}
    # Getting the type of 'vec' (line 360)
    vec_1423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 34), 'vec', False)
    # Calling vec(args, kwargs) (line 360)
    vec_call_result_1428 = invoke(stypy.reporting.localization.Localization(__file__, 360, 34), vec_1423, *[float_1424, float_1425, float_1426], **kwargs_1427)
    
    
    # Call to vec(...): (line 360)
    # Processing the call arguments (line 360)
    float_1430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 59), 'float')
    float_1431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 64), 'float')
    float_1432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 360, 69), 'float')
    # Processing the call keyword arguments (line 360)
    kwargs_1433 = {}
    # Getting the type of 'vec' (line 360)
    vec_1429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 55), 'vec', False)
    # Calling vec(args, kwargs) (line 360)
    vec_call_result_1434 = invoke(stypy.reporting.localization.Localization(__file__, 360, 55), vec_1429, *[float_1430, float_1431, float_1432], **kwargs_1433)
    
    # Processing the call keyword arguments (line 360)
    kwargs_1435 = {}
    # Getting the type of 'parallellight' (line 360)
    parallellight_1422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 20), 'parallellight', False)
    # Calling parallellight(args, kwargs) (line 360)
    parallellight_call_result_1436 = invoke(stypy.reporting.localization.Localization(__file__, 360, 20), parallellight_1422, *[vec_call_result_1428, vec_call_result_1434], **kwargs_1435)
    
    # Processing the call keyword arguments (line 360)
    kwargs_1437 = {}
    # Getting the type of 'w' (line 360)
    w_1419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 360, 4), 'w', False)
    # Obtaining the member 'lights' of a type (line 360)
    lights_1420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 4), w_1419, 'lights')
    # Obtaining the member 'append' of a type (line 360)
    append_1421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 360, 4), lights_1420, 'append')
    # Calling append(args, kwargs) (line 360)
    append_call_result_1438 = invoke(stypy.reporting.localization.Localization(__file__, 360, 4), append_1421, *[parallellight_call_result_1436], **kwargs_1437)
    
    
    # Call to append(...): (line 361)
    # Processing the call arguments (line 361)
    
    # Call to pointlight(...): (line 361)
    # Processing the call arguments (line 361)
    
    # Call to vec(...): (line 361)
    # Processing the call arguments (line 361)
    float_1444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 35), 'float')
    float_1445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 40), 'float')
    float_1446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 47), 'float')
    # Processing the call keyword arguments (line 361)
    kwargs_1447 = {}
    # Getting the type of 'vec' (line 361)
    vec_1443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 31), 'vec', False)
    # Calling vec(args, kwargs) (line 361)
    vec_call_result_1448 = invoke(stypy.reporting.localization.Localization(__file__, 361, 31), vec_1443, *[float_1444, float_1445, float_1446], **kwargs_1447)
    
    
    # Call to vec(...): (line 361)
    # Processing the call arguments (line 361)
    float_1450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 58), 'float')
    float_1451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 63), 'float')
    float_1452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 361, 68), 'float')
    # Processing the call keyword arguments (line 361)
    kwargs_1453 = {}
    # Getting the type of 'vec' (line 361)
    vec_1449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 54), 'vec', False)
    # Calling vec(args, kwargs) (line 361)
    vec_call_result_1454 = invoke(stypy.reporting.localization.Localization(__file__, 361, 54), vec_1449, *[float_1450, float_1451, float_1452], **kwargs_1453)
    
    # Processing the call keyword arguments (line 361)
    kwargs_1455 = {}
    # Getting the type of 'pointlight' (line 361)
    pointlight_1442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 20), 'pointlight', False)
    # Calling pointlight(args, kwargs) (line 361)
    pointlight_call_result_1456 = invoke(stypy.reporting.localization.Localization(__file__, 361, 20), pointlight_1442, *[vec_call_result_1448, vec_call_result_1454], **kwargs_1455)
    
    # Processing the call keyword arguments (line 361)
    kwargs_1457 = {}
    # Getting the type of 'w' (line 361)
    w_1439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 361, 4), 'w', False)
    # Obtaining the member 'lights' of a type (line 361)
    lights_1440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 4), w_1439, 'lights')
    # Obtaining the member 'append' of a type (line 361)
    append_1441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 361, 4), lights_1440, 'append')
    # Calling append(args, kwargs) (line 361)
    append_call_result_1458 = invoke(stypy.reporting.localization.Localization(__file__, 361, 4), append_1441, *[pointlight_call_result_1456], **kwargs_1457)
    
    
    # Call to render(...): (line 363)
    # Processing the call arguments (line 363)
    
    # Call to Relative(...): (line 363)
    # Processing the call arguments (line 363)
    str_1462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 363, 22), 'str', 'test.ppm')
    # Processing the call keyword arguments (line 363)
    kwargs_1463 = {}
    # Getting the type of 'Relative' (line 363)
    Relative_1461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 13), 'Relative', False)
    # Calling Relative(args, kwargs) (line 363)
    Relative_call_result_1464 = invoke(stypy.reporting.localization.Localization(__file__, 363, 13), Relative_1461, *[str_1462], **kwargs_1463)
    
    # Processing the call keyword arguments (line 363)
    kwargs_1465 = {}
    # Getting the type of 'w' (line 363)
    w_1459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 363, 4), 'w', False)
    # Obtaining the member 'render' of a type (line 363)
    render_1460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 363, 4), w_1459, 'render')
    # Calling render(args, kwargs) (line 363)
    render_call_result_1466 = invoke(stypy.reporting.localization.Localization(__file__, 363, 4), render_1460, *[Relative_call_result_1464], **kwargs_1465)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 342)
    stypy_return_type_1467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1467)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_1467

# Assigning a type to the variable 'main' (line 342)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 342, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 366, 0, False)
    
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

    
    # Call to main(...): (line 367)
    # Processing the call keyword arguments (line 367)
    kwargs_1469 = {}
    # Getting the type of 'main' (line 367)
    main_1468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 367, 4), 'main', False)
    # Calling main(args, kwargs) (line 367)
    main_call_result_1470 = invoke(stypy.reporting.localization.Localization(__file__, 367, 4), main_1468, *[], **kwargs_1469)
    
    # Getting the type of 'True' (line 368)
    True_1471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 368, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 368)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 368, 4), 'stypy_return_type', True_1471)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 366)
    stypy_return_type_1472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 366, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1472)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_1472

# Assigning a type to the variable 'run' (line 366)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 366, 0), 'run', run)

# Call to run(...): (line 371)
# Processing the call keyword arguments (line 371)
kwargs_1474 = {}
# Getting the type of 'run' (line 371)
run_1473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 371, 0), 'run', False)
# Calling run(args, kwargs) (line 371)
run_call_result_1475 = invoke(stypy.reporting.localization.Localization(__file__, 371, 0), run_1473, *[], **kwargs_1474)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
