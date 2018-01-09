
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Contains a classes for generating hatch patterns.
3: '''
4: 
5: from __future__ import (absolute_import, division, print_function,
6:                         unicode_literals)
7: 
8: import six
9: from six.moves import xrange
10: 
11: import numpy as np
12: from matplotlib.path import Path
13: 
14: 
15: class HatchPatternBase(object):
16:     '''
17:     The base class for a hatch pattern.
18:     '''
19:     pass
20: 
21: 
22: class HorizontalHatch(HatchPatternBase):
23:     def __init__(self, hatch, density):
24:         self.num_lines = int((hatch.count('-') + hatch.count('+')) * density)
25:         self.num_vertices = self.num_lines * 2
26: 
27:     def set_vertices_and_codes(self, vertices, codes):
28:         steps, stepsize = np.linspace(0.0, 1.0, self.num_lines, False,
29:                                       retstep=True)
30:         steps += stepsize / 2.
31:         vertices[0::2, 0] = 0.0
32:         vertices[0::2, 1] = steps
33:         vertices[1::2, 0] = 1.0
34:         vertices[1::2, 1] = steps
35:         codes[0::2] = Path.MOVETO
36:         codes[1::2] = Path.LINETO
37: 
38: 
39: class VerticalHatch(HatchPatternBase):
40:     def __init__(self, hatch, density):
41:         self.num_lines = int((hatch.count('|') + hatch.count('+')) * density)
42:         self.num_vertices = self.num_lines * 2
43: 
44:     def set_vertices_and_codes(self, vertices, codes):
45:         steps, stepsize = np.linspace(0.0, 1.0, self.num_lines, False,
46:                                       retstep=True)
47:         steps += stepsize / 2.
48:         vertices[0::2, 0] = steps
49:         vertices[0::2, 1] = 0.0
50:         vertices[1::2, 0] = steps
51:         vertices[1::2, 1] = 1.0
52:         codes[0::2] = Path.MOVETO
53:         codes[1::2] = Path.LINETO
54: 
55: 
56: class NorthEastHatch(HatchPatternBase):
57:     def __init__(self, hatch, density):
58:         self.num_lines = int((hatch.count('/') + hatch.count('x') +
59:                           hatch.count('X')) * density)
60:         if self.num_lines:
61:             self.num_vertices = (self.num_lines + 1) * 2
62:         else:
63:             self.num_vertices = 0
64: 
65:     def set_vertices_and_codes(self, vertices, codes):
66:         steps = np.linspace(-0.5, 0.5, self.num_lines + 1, True)
67:         vertices[0::2, 0] = 0.0 + steps
68:         vertices[0::2, 1] = 0.0 - steps
69:         vertices[1::2, 0] = 1.0 + steps
70:         vertices[1::2, 1] = 1.0 - steps
71:         codes[0::2] = Path.MOVETO
72:         codes[1::2] = Path.LINETO
73: 
74: 
75: class SouthEastHatch(HatchPatternBase):
76:     def __init__(self, hatch, density):
77:         self.num_lines = int((hatch.count('\\') + hatch.count('x') +
78:                           hatch.count('X')) * density)
79:         self.num_vertices = (self.num_lines + 1) * 2
80:         if self.num_lines:
81:             self.num_vertices = (self.num_lines + 1) * 2
82:         else:
83:             self.num_vertices = 0
84: 
85:     def set_vertices_and_codes(self, vertices, codes):
86:         steps = np.linspace(-0.5, 0.5, self.num_lines + 1, True)
87:         vertices[0::2, 0] = 0.0 + steps
88:         vertices[0::2, 1] = 1.0 + steps
89:         vertices[1::2, 0] = 1.0 + steps
90:         vertices[1::2, 1] = 0.0 + steps
91:         codes[0::2] = Path.MOVETO
92:         codes[1::2] = Path.LINETO
93: 
94: 
95: class Shapes(HatchPatternBase):
96:     filled = False
97: 
98:     def __init__(self, hatch, density):
99:         if self.num_rows == 0:
100:             self.num_shapes = 0
101:             self.num_vertices = 0
102:         else:
103:             self.num_shapes = ((self.num_rows // 2 + 1) * (self.num_rows + 1) +
104:                                (self.num_rows // 2) * (self.num_rows))
105:             self.num_vertices = (self.num_shapes *
106:                                  len(self.shape_vertices) *
107:                                  (self.filled and 1 or 2))
108: 
109:     def set_vertices_and_codes(self, vertices, codes):
110:         offset = 1.0 / self.num_rows
111:         shape_vertices = self.shape_vertices * offset * self.size
112:         if not self.filled:
113:             inner_vertices = shape_vertices[::-1] * 0.9
114:         shape_codes = self.shape_codes
115:         shape_size = len(shape_vertices)
116: 
117:         cursor = 0
118:         for row in xrange(self.num_rows + 1):
119:             if row % 2 == 0:
120:                 cols = np.linspace(0.0, 1.0, self.num_rows + 1, True)
121:             else:
122:                 cols = np.linspace(offset / 2.0, 1.0 - offset / 2.0,
123:                                    self.num_rows, True)
124:             row_pos = row * offset
125:             for col_pos in cols:
126:                 vertices[cursor:cursor + shape_size] = (shape_vertices +
127:                                                         (col_pos, row_pos))
128:                 codes[cursor:cursor + shape_size] = shape_codes
129:                 cursor += shape_size
130:                 if not self.filled:
131:                     vertices[cursor:cursor + shape_size] = (inner_vertices +
132:                                                             (col_pos, row_pos))
133:                     codes[cursor:cursor + shape_size] = shape_codes
134:                     cursor += shape_size
135: 
136: 
137: class Circles(Shapes):
138:     def __init__(self, hatch, density):
139:         path = Path.unit_circle()
140:         self.shape_vertices = path.vertices
141:         self.shape_codes = path.codes
142:         Shapes.__init__(self, hatch, density)
143: 
144: 
145: class SmallCircles(Circles):
146:     size = 0.2
147: 
148:     def __init__(self, hatch, density):
149:         self.num_rows = (hatch.count('o')) * density
150:         Circles.__init__(self, hatch, density)
151: 
152: 
153: class LargeCircles(Circles):
154:     size = 0.35
155: 
156:     def __init__(self, hatch, density):
157:         self.num_rows = (hatch.count('O')) * density
158:         Circles.__init__(self, hatch, density)
159: 
160: 
161: class SmallFilledCircles(SmallCircles):
162:     size = 0.1
163:     filled = True
164: 
165:     def __init__(self, hatch, density):
166:         self.num_rows = (hatch.count('.')) * density
167:         Circles.__init__(self, hatch, density)
168: 
169: 
170: class Stars(Shapes):
171:     size = 1.0 / 3.0
172:     filled = True
173: 
174:     def __init__(self, hatch, density):
175:         self.num_rows = (hatch.count('*')) * density
176:         path = Path.unit_regular_star(5)
177:         self.shape_vertices = path.vertices
178:         self.shape_codes = np.ones(len(self.shape_vertices)) * Path.LINETO
179:         self.shape_codes[0] = Path.MOVETO
180:         Shapes.__init__(self, hatch, density)
181: 
182: _hatch_types = [
183:     HorizontalHatch,
184:     VerticalHatch,
185:     NorthEastHatch,
186:     SouthEastHatch,
187:     SmallCircles,
188:     LargeCircles,
189:     SmallFilledCircles,
190:     Stars
191:     ]
192: 
193: 
194: def get_path(hatchpattern, density=6):
195:     '''
196:     Given a hatch specifier, *hatchpattern*, generates Path to render
197:     the hatch in a unit square.  *density* is the number of lines per
198:     unit square.
199:     '''
200:     density = int(density)
201: 
202:     patterns = [hatch_type(hatchpattern, density)
203:                 for hatch_type in _hatch_types]
204:     num_vertices = sum([pattern.num_vertices for pattern in patterns])
205: 
206:     if num_vertices == 0:
207:         return Path(np.empty((0, 2)))
208: 
209:     vertices = np.empty((num_vertices, 2))
210:     codes = np.empty((num_vertices,), np.uint8)
211: 
212:     cursor = 0
213:     for pattern in patterns:
214:         if pattern.num_vertices != 0:
215:             vertices_chunk = vertices[cursor:cursor + pattern.num_vertices]
216:             codes_chunk = codes[cursor:cursor + pattern.num_vertices]
217:             pattern.set_vertices_and_codes(vertices_chunk, codes_chunk)
218:             cursor += pattern.num_vertices
219: 
220:     return Path(vertices, codes)
221: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_61530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'unicode', u'\nContains a classes for generating hatch patterns.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import six' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_61531 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six')

if (type(import_61531) is not StypyTypeError):

    if (import_61531 != 'pyd_module'):
        __import__(import_61531)
        sys_modules_61532 = sys.modules[import_61531]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', sys_modules_61532.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'six', import_61531)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from six.moves import xrange' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_61533 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six.moves')

if (type(import_61533) is not StypyTypeError):

    if (import_61533 != 'pyd_module'):
        __import__(import_61533)
        sys_modules_61534 = sys.modules[import_61533]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six.moves', sys_modules_61534.module_type_store, module_type_store, ['xrange'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_61534, sys_modules_61534.module_type_store, module_type_store)
    else:
        from six.moves import xrange

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'six.moves', None, module_type_store, ['xrange'], [xrange])

else:
    # Assigning a type to the variable 'six.moves' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'six.moves', import_61533)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import numpy' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_61535 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy')

if (type(import_61535) is not StypyTypeError):

    if (import_61535 != 'pyd_module'):
        __import__(import_61535)
        sys_modules_61536 = sys.modules[import_61535]
        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', sys_modules_61536.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy', import_61535)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from matplotlib.path import Path' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/')
import_61537 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.path')

if (type(import_61537) is not StypyTypeError):

    if (import_61537 != 'pyd_module'):
        __import__(import_61537)
        sys_modules_61538 = sys.modules[import_61537]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.path', sys_modules_61538.module_type_store, module_type_store, ['Path'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_61538, sys_modules_61538.module_type_store, module_type_store)
    else:
        from matplotlib.path import Path

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.path', None, module_type_store, ['Path'], [Path])

else:
    # Assigning a type to the variable 'matplotlib.path' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'matplotlib.path', import_61537)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/')

# Declaration of the 'HatchPatternBase' class

class HatchPatternBase(object, ):
    unicode_61539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'unicode', u'\n    The base class for a hatch pattern.\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 0, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HatchPatternBase.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'HatchPatternBase' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'HatchPatternBase', HatchPatternBase)
# Declaration of the 'HorizontalHatch' class
# Getting the type of 'HatchPatternBase' (line 22)
HatchPatternBase_61540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'HatchPatternBase')

class HorizontalHatch(HatchPatternBase_61540, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HorizontalHatch.__init__', ['hatch', 'density'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['hatch', 'density'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 24):
        
        # Assigning a Call to a Attribute (line 24):
        
        # Call to int(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Call to count(...): (line 24)
        # Processing the call arguments (line 24)
        unicode_61544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 42), 'unicode', u'-')
        # Processing the call keyword arguments (line 24)
        kwargs_61545 = {}
        # Getting the type of 'hatch' (line 24)
        hatch_61542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'hatch', False)
        # Obtaining the member 'count' of a type (line 24)
        count_61543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 30), hatch_61542, 'count')
        # Calling count(args, kwargs) (line 24)
        count_call_result_61546 = invoke(stypy.reporting.localization.Localization(__file__, 24, 30), count_61543, *[unicode_61544], **kwargs_61545)
        
        
        # Call to count(...): (line 24)
        # Processing the call arguments (line 24)
        unicode_61549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 61), 'unicode', u'+')
        # Processing the call keyword arguments (line 24)
        kwargs_61550 = {}
        # Getting the type of 'hatch' (line 24)
        hatch_61547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 49), 'hatch', False)
        # Obtaining the member 'count' of a type (line 24)
        count_61548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 49), hatch_61547, 'count')
        # Calling count(args, kwargs) (line 24)
        count_call_result_61551 = invoke(stypy.reporting.localization.Localization(__file__, 24, 49), count_61548, *[unicode_61549], **kwargs_61550)
        
        # Applying the binary operator '+' (line 24)
        result_add_61552 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 30), '+', count_call_result_61546, count_call_result_61551)
        
        # Getting the type of 'density' (line 24)
        density_61553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 69), 'density', False)
        # Applying the binary operator '*' (line 24)
        result_mul_61554 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 29), '*', result_add_61552, density_61553)
        
        # Processing the call keyword arguments (line 24)
        kwargs_61555 = {}
        # Getting the type of 'int' (line 24)
        int_61541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'int', False)
        # Calling int(args, kwargs) (line 24)
        int_call_result_61556 = invoke(stypy.reporting.localization.Localization(__file__, 24, 25), int_61541, *[result_mul_61554], **kwargs_61555)
        
        # Getting the type of 'self' (line 24)
        self_61557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Setting the type of the member 'num_lines' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_61557, 'num_lines', int_call_result_61556)
        
        # Assigning a BinOp to a Attribute (line 25):
        
        # Assigning a BinOp to a Attribute (line 25):
        # Getting the type of 'self' (line 25)
        self_61558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 28), 'self')
        # Obtaining the member 'num_lines' of a type (line 25)
        num_lines_61559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 28), self_61558, 'num_lines')
        int_61560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 45), 'int')
        # Applying the binary operator '*' (line 25)
        result_mul_61561 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 28), '*', num_lines_61559, int_61560)
        
        # Getting the type of 'self' (line 25)
        self_61562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'num_vertices' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_61562, 'num_vertices', result_mul_61561)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_vertices_and_codes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_vertices_and_codes'
        module_type_store = module_type_store.open_function_context('set_vertices_and_codes', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HorizontalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_localization', localization)
        HorizontalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HorizontalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_type_store', module_type_store)
        HorizontalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_function_name', 'HorizontalHatch.set_vertices_and_codes')
        HorizontalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_param_names_list', ['vertices', 'codes'])
        HorizontalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_varargs_param_name', None)
        HorizontalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HorizontalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_call_defaults', defaults)
        HorizontalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_call_varargs', varargs)
        HorizontalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HorizontalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HorizontalHatch.set_vertices_and_codes', ['vertices', 'codes'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_vertices_and_codes', localization, ['vertices', 'codes'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_vertices_and_codes(...)' code ##################

        
        # Assigning a Call to a Tuple (line 28):
        
        # Assigning a Call to a Name:
        
        # Call to linspace(...): (line 28)
        # Processing the call arguments (line 28)
        float_61565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 38), 'float')
        float_61566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 43), 'float')
        # Getting the type of 'self' (line 28)
        self_61567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 48), 'self', False)
        # Obtaining the member 'num_lines' of a type (line 28)
        num_lines_61568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 48), self_61567, 'num_lines')
        # Getting the type of 'False' (line 28)
        False_61569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 64), 'False', False)
        # Processing the call keyword arguments (line 28)
        # Getting the type of 'True' (line 29)
        True_61570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 46), 'True', False)
        keyword_61571 = True_61570
        kwargs_61572 = {'retstep': keyword_61571}
        # Getting the type of 'np' (line 28)
        np_61563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'np', False)
        # Obtaining the member 'linspace' of a type (line 28)
        linspace_61564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 26), np_61563, 'linspace')
        # Calling linspace(args, kwargs) (line 28)
        linspace_call_result_61573 = invoke(stypy.reporting.localization.Localization(__file__, 28, 26), linspace_61564, *[float_61565, float_61566, num_lines_61568, False_61569], **kwargs_61572)
        
        # Assigning a type to the variable 'call_assignment_61524' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_61524', linspace_call_result_61573)
        
        # Assigning a Call to a Name (line 28):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61577 = {}
        # Getting the type of 'call_assignment_61524' (line 28)
        call_assignment_61524_61574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_61524', False)
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___61575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), call_assignment_61524_61574, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61578 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61575, *[int_61576], **kwargs_61577)
        
        # Assigning a type to the variable 'call_assignment_61525' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_61525', getitem___call_result_61578)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'call_assignment_61525' (line 28)
        call_assignment_61525_61579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_61525')
        # Assigning a type to the variable 'steps' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'steps', call_assignment_61525_61579)
        
        # Assigning a Call to a Name (line 28):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61583 = {}
        # Getting the type of 'call_assignment_61524' (line 28)
        call_assignment_61524_61580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_61524', False)
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___61581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), call_assignment_61524_61580, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61584 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61581, *[int_61582], **kwargs_61583)
        
        # Assigning a type to the variable 'call_assignment_61526' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_61526', getitem___call_result_61584)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'call_assignment_61526' (line 28)
        call_assignment_61526_61585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'call_assignment_61526')
        # Assigning a type to the variable 'stepsize' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'stepsize', call_assignment_61526_61585)
        
        # Getting the type of 'steps' (line 30)
        steps_61586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'steps')
        # Getting the type of 'stepsize' (line 30)
        stepsize_61587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'stepsize')
        float_61588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 28), 'float')
        # Applying the binary operator 'div' (line 30)
        result_div_61589 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 17), 'div', stepsize_61587, float_61588)
        
        # Applying the binary operator '+=' (line 30)
        result_iadd_61590 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 8), '+=', steps_61586, result_div_61589)
        # Assigning a type to the variable 'steps' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'steps', result_iadd_61590)
        
        
        # Assigning a Num to a Subscript (line 31):
        
        # Assigning a Num to a Subscript (line 31):
        float_61591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'float')
        # Getting the type of 'vertices' (line 31)
        vertices_61592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'vertices')
        int_61593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 17), 'int')
        int_61594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 20), 'int')
        slice_61595 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 31, 8), int_61593, None, int_61594)
        int_61596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 23), 'int')
        # Storing an element on a container (line 31)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 8), vertices_61592, ((slice_61595, int_61596), float_61591))
        
        # Assigning a Name to a Subscript (line 32):
        
        # Assigning a Name to a Subscript (line 32):
        # Getting the type of 'steps' (line 32)
        steps_61597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 28), 'steps')
        # Getting the type of 'vertices' (line 32)
        vertices_61598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'vertices')
        int_61599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 17), 'int')
        int_61600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'int')
        slice_61601 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 32, 8), int_61599, None, int_61600)
        int_61602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'int')
        # Storing an element on a container (line 32)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 8), vertices_61598, ((slice_61601, int_61602), steps_61597))
        
        # Assigning a Num to a Subscript (line 33):
        
        # Assigning a Num to a Subscript (line 33):
        float_61603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'float')
        # Getting the type of 'vertices' (line 33)
        vertices_61604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'vertices')
        int_61605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 17), 'int')
        int_61606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'int')
        slice_61607 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 33, 8), int_61605, None, int_61606)
        int_61608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'int')
        # Storing an element on a container (line 33)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 8), vertices_61604, ((slice_61607, int_61608), float_61603))
        
        # Assigning a Name to a Subscript (line 34):
        
        # Assigning a Name to a Subscript (line 34):
        # Getting the type of 'steps' (line 34)
        steps_61609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'steps')
        # Getting the type of 'vertices' (line 34)
        vertices_61610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'vertices')
        int_61611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'int')
        int_61612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 20), 'int')
        slice_61613 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 34, 8), int_61611, None, int_61612)
        int_61614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'int')
        # Storing an element on a container (line 34)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 8), vertices_61610, ((slice_61613, int_61614), steps_61609))
        
        # Assigning a Attribute to a Subscript (line 35):
        
        # Assigning a Attribute to a Subscript (line 35):
        # Getting the type of 'Path' (line 35)
        Path_61615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'Path')
        # Obtaining the member 'MOVETO' of a type (line 35)
        MOVETO_61616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 22), Path_61615, 'MOVETO')
        # Getting the type of 'codes' (line 35)
        codes_61617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'codes')
        int_61618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 14), 'int')
        int_61619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 17), 'int')
        slice_61620 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 35, 8), int_61618, None, int_61619)
        # Storing an element on a container (line 35)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 8), codes_61617, (slice_61620, MOVETO_61616))
        
        # Assigning a Attribute to a Subscript (line 36):
        
        # Assigning a Attribute to a Subscript (line 36):
        # Getting the type of 'Path' (line 36)
        Path_61621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 22), 'Path')
        # Obtaining the member 'LINETO' of a type (line 36)
        LINETO_61622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 22), Path_61621, 'LINETO')
        # Getting the type of 'codes' (line 36)
        codes_61623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'codes')
        int_61624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'int')
        int_61625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 17), 'int')
        slice_61626 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 36, 8), int_61624, None, int_61625)
        # Storing an element on a container (line 36)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 8), codes_61623, (slice_61626, LINETO_61622))
        
        # ################# End of 'set_vertices_and_codes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_vertices_and_codes' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_61627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61627)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_vertices_and_codes'
        return stypy_return_type_61627


# Assigning a type to the variable 'HorizontalHatch' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'HorizontalHatch', HorizontalHatch)
# Declaration of the 'VerticalHatch' class
# Getting the type of 'HatchPatternBase' (line 39)
HatchPatternBase_61628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'HatchPatternBase')

class VerticalHatch(HatchPatternBase_61628, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VerticalHatch.__init__', ['hatch', 'density'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['hatch', 'density'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 41):
        
        # Assigning a Call to a Attribute (line 41):
        
        # Call to int(...): (line 41)
        # Processing the call arguments (line 41)
        
        # Call to count(...): (line 41)
        # Processing the call arguments (line 41)
        unicode_61632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 42), 'unicode', u'|')
        # Processing the call keyword arguments (line 41)
        kwargs_61633 = {}
        # Getting the type of 'hatch' (line 41)
        hatch_61630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'hatch', False)
        # Obtaining the member 'count' of a type (line 41)
        count_61631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 30), hatch_61630, 'count')
        # Calling count(args, kwargs) (line 41)
        count_call_result_61634 = invoke(stypy.reporting.localization.Localization(__file__, 41, 30), count_61631, *[unicode_61632], **kwargs_61633)
        
        
        # Call to count(...): (line 41)
        # Processing the call arguments (line 41)
        unicode_61637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 61), 'unicode', u'+')
        # Processing the call keyword arguments (line 41)
        kwargs_61638 = {}
        # Getting the type of 'hatch' (line 41)
        hatch_61635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 49), 'hatch', False)
        # Obtaining the member 'count' of a type (line 41)
        count_61636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 49), hatch_61635, 'count')
        # Calling count(args, kwargs) (line 41)
        count_call_result_61639 = invoke(stypy.reporting.localization.Localization(__file__, 41, 49), count_61636, *[unicode_61637], **kwargs_61638)
        
        # Applying the binary operator '+' (line 41)
        result_add_61640 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 30), '+', count_call_result_61634, count_call_result_61639)
        
        # Getting the type of 'density' (line 41)
        density_61641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 69), 'density', False)
        # Applying the binary operator '*' (line 41)
        result_mul_61642 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 29), '*', result_add_61640, density_61641)
        
        # Processing the call keyword arguments (line 41)
        kwargs_61643 = {}
        # Getting the type of 'int' (line 41)
        int_61629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 25), 'int', False)
        # Calling int(args, kwargs) (line 41)
        int_call_result_61644 = invoke(stypy.reporting.localization.Localization(__file__, 41, 25), int_61629, *[result_mul_61642], **kwargs_61643)
        
        # Getting the type of 'self' (line 41)
        self_61645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'num_lines' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_61645, 'num_lines', int_call_result_61644)
        
        # Assigning a BinOp to a Attribute (line 42):
        
        # Assigning a BinOp to a Attribute (line 42):
        # Getting the type of 'self' (line 42)
        self_61646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'self')
        # Obtaining the member 'num_lines' of a type (line 42)
        num_lines_61647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 28), self_61646, 'num_lines')
        int_61648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 45), 'int')
        # Applying the binary operator '*' (line 42)
        result_mul_61649 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 28), '*', num_lines_61647, int_61648)
        
        # Getting the type of 'self' (line 42)
        self_61650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self')
        # Setting the type of the member 'num_vertices' of a type (line 42)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_61650, 'num_vertices', result_mul_61649)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_vertices_and_codes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_vertices_and_codes'
        module_type_store = module_type_store.open_function_context('set_vertices_and_codes', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VerticalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_localization', localization)
        VerticalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VerticalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_type_store', module_type_store)
        VerticalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_function_name', 'VerticalHatch.set_vertices_and_codes')
        VerticalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_param_names_list', ['vertices', 'codes'])
        VerticalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_varargs_param_name', None)
        VerticalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VerticalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_call_defaults', defaults)
        VerticalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_call_varargs', varargs)
        VerticalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VerticalHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VerticalHatch.set_vertices_and_codes', ['vertices', 'codes'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_vertices_and_codes', localization, ['vertices', 'codes'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_vertices_and_codes(...)' code ##################

        
        # Assigning a Call to a Tuple (line 45):
        
        # Assigning a Call to a Name:
        
        # Call to linspace(...): (line 45)
        # Processing the call arguments (line 45)
        float_61653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 38), 'float')
        float_61654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 43), 'float')
        # Getting the type of 'self' (line 45)
        self_61655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 48), 'self', False)
        # Obtaining the member 'num_lines' of a type (line 45)
        num_lines_61656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 48), self_61655, 'num_lines')
        # Getting the type of 'False' (line 45)
        False_61657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 64), 'False', False)
        # Processing the call keyword arguments (line 45)
        # Getting the type of 'True' (line 46)
        True_61658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'True', False)
        keyword_61659 = True_61658
        kwargs_61660 = {'retstep': keyword_61659}
        # Getting the type of 'np' (line 45)
        np_61651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'np', False)
        # Obtaining the member 'linspace' of a type (line 45)
        linspace_61652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 26), np_61651, 'linspace')
        # Calling linspace(args, kwargs) (line 45)
        linspace_call_result_61661 = invoke(stypy.reporting.localization.Localization(__file__, 45, 26), linspace_61652, *[float_61653, float_61654, num_lines_61656, False_61657], **kwargs_61660)
        
        # Assigning a type to the variable 'call_assignment_61527' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'call_assignment_61527', linspace_call_result_61661)
        
        # Assigning a Call to a Name (line 45):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61665 = {}
        # Getting the type of 'call_assignment_61527' (line 45)
        call_assignment_61527_61662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'call_assignment_61527', False)
        # Obtaining the member '__getitem__' of a type (line 45)
        getitem___61663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), call_assignment_61527_61662, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61666 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61663, *[int_61664], **kwargs_61665)
        
        # Assigning a type to the variable 'call_assignment_61528' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'call_assignment_61528', getitem___call_result_61666)
        
        # Assigning a Name to a Name (line 45):
        # Getting the type of 'call_assignment_61528' (line 45)
        call_assignment_61528_61667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'call_assignment_61528')
        # Assigning a type to the variable 'steps' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'steps', call_assignment_61528_61667)
        
        # Assigning a Call to a Name (line 45):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 8), 'int')
        # Processing the call keyword arguments
        kwargs_61671 = {}
        # Getting the type of 'call_assignment_61527' (line 45)
        call_assignment_61527_61668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'call_assignment_61527', False)
        # Obtaining the member '__getitem__' of a type (line 45)
        getitem___61669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), call_assignment_61527_61668, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61672 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61669, *[int_61670], **kwargs_61671)
        
        # Assigning a type to the variable 'call_assignment_61529' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'call_assignment_61529', getitem___call_result_61672)
        
        # Assigning a Name to a Name (line 45):
        # Getting the type of 'call_assignment_61529' (line 45)
        call_assignment_61529_61673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'call_assignment_61529')
        # Assigning a type to the variable 'stepsize' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'stepsize', call_assignment_61529_61673)
        
        # Getting the type of 'steps' (line 47)
        steps_61674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'steps')
        # Getting the type of 'stepsize' (line 47)
        stepsize_61675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'stepsize')
        float_61676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 28), 'float')
        # Applying the binary operator 'div' (line 47)
        result_div_61677 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 17), 'div', stepsize_61675, float_61676)
        
        # Applying the binary operator '+=' (line 47)
        result_iadd_61678 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 8), '+=', steps_61674, result_div_61677)
        # Assigning a type to the variable 'steps' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'steps', result_iadd_61678)
        
        
        # Assigning a Name to a Subscript (line 48):
        
        # Assigning a Name to a Subscript (line 48):
        # Getting the type of 'steps' (line 48)
        steps_61679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 'steps')
        # Getting the type of 'vertices' (line 48)
        vertices_61680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'vertices')
        int_61681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 17), 'int')
        int_61682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 20), 'int')
        slice_61683 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 48, 8), int_61681, None, int_61682)
        int_61684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 23), 'int')
        # Storing an element on a container (line 48)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 8), vertices_61680, ((slice_61683, int_61684), steps_61679))
        
        # Assigning a Num to a Subscript (line 49):
        
        # Assigning a Num to a Subscript (line 49):
        float_61685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'float')
        # Getting the type of 'vertices' (line 49)
        vertices_61686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'vertices')
        int_61687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 17), 'int')
        int_61688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 20), 'int')
        slice_61689 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 49, 8), int_61687, None, int_61688)
        int_61690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'int')
        # Storing an element on a container (line 49)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 8), vertices_61686, ((slice_61689, int_61690), float_61685))
        
        # Assigning a Name to a Subscript (line 50):
        
        # Assigning a Name to a Subscript (line 50):
        # Getting the type of 'steps' (line 50)
        steps_61691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'steps')
        # Getting the type of 'vertices' (line 50)
        vertices_61692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'vertices')
        int_61693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'int')
        int_61694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 20), 'int')
        slice_61695 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 50, 8), int_61693, None, int_61694)
        int_61696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 23), 'int')
        # Storing an element on a container (line 50)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 8), vertices_61692, ((slice_61695, int_61696), steps_61691))
        
        # Assigning a Num to a Subscript (line 51):
        
        # Assigning a Num to a Subscript (line 51):
        float_61697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 28), 'float')
        # Getting the type of 'vertices' (line 51)
        vertices_61698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'vertices')
        int_61699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 17), 'int')
        int_61700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 20), 'int')
        slice_61701 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 51, 8), int_61699, None, int_61700)
        int_61702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'int')
        # Storing an element on a container (line 51)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 8), vertices_61698, ((slice_61701, int_61702), float_61697))
        
        # Assigning a Attribute to a Subscript (line 52):
        
        # Assigning a Attribute to a Subscript (line 52):
        # Getting the type of 'Path' (line 52)
        Path_61703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'Path')
        # Obtaining the member 'MOVETO' of a type (line 52)
        MOVETO_61704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 22), Path_61703, 'MOVETO')
        # Getting the type of 'codes' (line 52)
        codes_61705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'codes')
        int_61706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 14), 'int')
        int_61707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 17), 'int')
        slice_61708 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 52, 8), int_61706, None, int_61707)
        # Storing an element on a container (line 52)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 8), codes_61705, (slice_61708, MOVETO_61704))
        
        # Assigning a Attribute to a Subscript (line 53):
        
        # Assigning a Attribute to a Subscript (line 53):
        # Getting the type of 'Path' (line 53)
        Path_61709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'Path')
        # Obtaining the member 'LINETO' of a type (line 53)
        LINETO_61710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 22), Path_61709, 'LINETO')
        # Getting the type of 'codes' (line 53)
        codes_61711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'codes')
        int_61712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 14), 'int')
        int_61713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 17), 'int')
        slice_61714 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 53, 8), int_61712, None, int_61713)
        # Storing an element on a container (line 53)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 8), codes_61711, (slice_61714, LINETO_61710))
        
        # ################# End of 'set_vertices_and_codes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_vertices_and_codes' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_61715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61715)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_vertices_and_codes'
        return stypy_return_type_61715


# Assigning a type to the variable 'VerticalHatch' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'VerticalHatch', VerticalHatch)
# Declaration of the 'NorthEastHatch' class
# Getting the type of 'HatchPatternBase' (line 56)
HatchPatternBase_61716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'HatchPatternBase')

class NorthEastHatch(HatchPatternBase_61716, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NorthEastHatch.__init__', ['hatch', 'density'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['hatch', 'density'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 58):
        
        # Assigning a Call to a Attribute (line 58):
        
        # Call to int(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to count(...): (line 58)
        # Processing the call arguments (line 58)
        unicode_61720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 42), 'unicode', u'/')
        # Processing the call keyword arguments (line 58)
        kwargs_61721 = {}
        # Getting the type of 'hatch' (line 58)
        hatch_61718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 30), 'hatch', False)
        # Obtaining the member 'count' of a type (line 58)
        count_61719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 30), hatch_61718, 'count')
        # Calling count(args, kwargs) (line 58)
        count_call_result_61722 = invoke(stypy.reporting.localization.Localization(__file__, 58, 30), count_61719, *[unicode_61720], **kwargs_61721)
        
        
        # Call to count(...): (line 58)
        # Processing the call arguments (line 58)
        unicode_61725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 61), 'unicode', u'x')
        # Processing the call keyword arguments (line 58)
        kwargs_61726 = {}
        # Getting the type of 'hatch' (line 58)
        hatch_61723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 49), 'hatch', False)
        # Obtaining the member 'count' of a type (line 58)
        count_61724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 49), hatch_61723, 'count')
        # Calling count(args, kwargs) (line 58)
        count_call_result_61727 = invoke(stypy.reporting.localization.Localization(__file__, 58, 49), count_61724, *[unicode_61725], **kwargs_61726)
        
        # Applying the binary operator '+' (line 58)
        result_add_61728 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 30), '+', count_call_result_61722, count_call_result_61727)
        
        
        # Call to count(...): (line 59)
        # Processing the call arguments (line 59)
        unicode_61731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 38), 'unicode', u'X')
        # Processing the call keyword arguments (line 59)
        kwargs_61732 = {}
        # Getting the type of 'hatch' (line 59)
        hatch_61729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 26), 'hatch', False)
        # Obtaining the member 'count' of a type (line 59)
        count_61730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 26), hatch_61729, 'count')
        # Calling count(args, kwargs) (line 59)
        count_call_result_61733 = invoke(stypy.reporting.localization.Localization(__file__, 59, 26), count_61730, *[unicode_61731], **kwargs_61732)
        
        # Applying the binary operator '+' (line 58)
        result_add_61734 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 66), '+', result_add_61728, count_call_result_61733)
        
        # Getting the type of 'density' (line 59)
        density_61735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 46), 'density', False)
        # Applying the binary operator '*' (line 58)
        result_mul_61736 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 29), '*', result_add_61734, density_61735)
        
        # Processing the call keyword arguments (line 58)
        kwargs_61737 = {}
        # Getting the type of 'int' (line 58)
        int_61717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'int', False)
        # Calling int(args, kwargs) (line 58)
        int_call_result_61738 = invoke(stypy.reporting.localization.Localization(__file__, 58, 25), int_61717, *[result_mul_61736], **kwargs_61737)
        
        # Getting the type of 'self' (line 58)
        self_61739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self')
        # Setting the type of the member 'num_lines' of a type (line 58)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_61739, 'num_lines', int_call_result_61738)
        
        # Getting the type of 'self' (line 60)
        self_61740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'self')
        # Obtaining the member 'num_lines' of a type (line 60)
        num_lines_61741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 11), self_61740, 'num_lines')
        # Testing the type of an if condition (line 60)
        if_condition_61742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 8), num_lines_61741)
        # Assigning a type to the variable 'if_condition_61742' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'if_condition_61742', if_condition_61742)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Attribute (line 61):
        
        # Assigning a BinOp to a Attribute (line 61):
        # Getting the type of 'self' (line 61)
        self_61743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 33), 'self')
        # Obtaining the member 'num_lines' of a type (line 61)
        num_lines_61744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 33), self_61743, 'num_lines')
        int_61745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 50), 'int')
        # Applying the binary operator '+' (line 61)
        result_add_61746 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 33), '+', num_lines_61744, int_61745)
        
        int_61747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 55), 'int')
        # Applying the binary operator '*' (line 61)
        result_mul_61748 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 32), '*', result_add_61746, int_61747)
        
        # Getting the type of 'self' (line 61)
        self_61749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'self')
        # Setting the type of the member 'num_vertices' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 12), self_61749, 'num_vertices', result_mul_61748)
        # SSA branch for the else part of an if statement (line 60)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Attribute (line 63):
        
        # Assigning a Num to a Attribute (line 63):
        int_61750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 32), 'int')
        # Getting the type of 'self' (line 63)
        self_61751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'self')
        # Setting the type of the member 'num_vertices' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), self_61751, 'num_vertices', int_61750)
        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_vertices_and_codes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_vertices_and_codes'
        module_type_store = module_type_store.open_function_context('set_vertices_and_codes', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NorthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_localization', localization)
        NorthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NorthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_type_store', module_type_store)
        NorthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_function_name', 'NorthEastHatch.set_vertices_and_codes')
        NorthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_param_names_list', ['vertices', 'codes'])
        NorthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_varargs_param_name', None)
        NorthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NorthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_call_defaults', defaults)
        NorthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_call_varargs', varargs)
        NorthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NorthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NorthEastHatch.set_vertices_and_codes', ['vertices', 'codes'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_vertices_and_codes', localization, ['vertices', 'codes'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_vertices_and_codes(...)' code ##################

        
        # Assigning a Call to a Name (line 66):
        
        # Assigning a Call to a Name (line 66):
        
        # Call to linspace(...): (line 66)
        # Processing the call arguments (line 66)
        float_61754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 28), 'float')
        float_61755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 34), 'float')
        # Getting the type of 'self' (line 66)
        self_61756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 39), 'self', False)
        # Obtaining the member 'num_lines' of a type (line 66)
        num_lines_61757 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 39), self_61756, 'num_lines')
        int_61758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 56), 'int')
        # Applying the binary operator '+' (line 66)
        result_add_61759 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 39), '+', num_lines_61757, int_61758)
        
        # Getting the type of 'True' (line 66)
        True_61760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 59), 'True', False)
        # Processing the call keyword arguments (line 66)
        kwargs_61761 = {}
        # Getting the type of 'np' (line 66)
        np_61752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'np', False)
        # Obtaining the member 'linspace' of a type (line 66)
        linspace_61753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 16), np_61752, 'linspace')
        # Calling linspace(args, kwargs) (line 66)
        linspace_call_result_61762 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), linspace_61753, *[float_61754, float_61755, result_add_61759, True_61760], **kwargs_61761)
        
        # Assigning a type to the variable 'steps' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'steps', linspace_call_result_61762)
        
        # Assigning a BinOp to a Subscript (line 67):
        
        # Assigning a BinOp to a Subscript (line 67):
        float_61763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 28), 'float')
        # Getting the type of 'steps' (line 67)
        steps_61764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 34), 'steps')
        # Applying the binary operator '+' (line 67)
        result_add_61765 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 28), '+', float_61763, steps_61764)
        
        # Getting the type of 'vertices' (line 67)
        vertices_61766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'vertices')
        int_61767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 17), 'int')
        int_61768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 20), 'int')
        slice_61769 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 67, 8), int_61767, None, int_61768)
        int_61770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 23), 'int')
        # Storing an element on a container (line 67)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 8), vertices_61766, ((slice_61769, int_61770), result_add_61765))
        
        # Assigning a BinOp to a Subscript (line 68):
        
        # Assigning a BinOp to a Subscript (line 68):
        float_61771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'float')
        # Getting the type of 'steps' (line 68)
        steps_61772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'steps')
        # Applying the binary operator '-' (line 68)
        result_sub_61773 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 28), '-', float_61771, steps_61772)
        
        # Getting the type of 'vertices' (line 68)
        vertices_61774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'vertices')
        int_61775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 17), 'int')
        int_61776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'int')
        slice_61777 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 68, 8), int_61775, None, int_61776)
        int_61778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'int')
        # Storing an element on a container (line 68)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 8), vertices_61774, ((slice_61777, int_61778), result_sub_61773))
        
        # Assigning a BinOp to a Subscript (line 69):
        
        # Assigning a BinOp to a Subscript (line 69):
        float_61779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'float')
        # Getting the type of 'steps' (line 69)
        steps_61780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 34), 'steps')
        # Applying the binary operator '+' (line 69)
        result_add_61781 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 28), '+', float_61779, steps_61780)
        
        # Getting the type of 'vertices' (line 69)
        vertices_61782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'vertices')
        int_61783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 17), 'int')
        int_61784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'int')
        slice_61785 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 8), int_61783, None, int_61784)
        int_61786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'int')
        # Storing an element on a container (line 69)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 8), vertices_61782, ((slice_61785, int_61786), result_add_61781))
        
        # Assigning a BinOp to a Subscript (line 70):
        
        # Assigning a BinOp to a Subscript (line 70):
        float_61787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 28), 'float')
        # Getting the type of 'steps' (line 70)
        steps_61788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 34), 'steps')
        # Applying the binary operator '-' (line 70)
        result_sub_61789 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 28), '-', float_61787, steps_61788)
        
        # Getting the type of 'vertices' (line 70)
        vertices_61790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'vertices')
        int_61791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 17), 'int')
        int_61792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 20), 'int')
        slice_61793 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 70, 8), int_61791, None, int_61792)
        int_61794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'int')
        # Storing an element on a container (line 70)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 8), vertices_61790, ((slice_61793, int_61794), result_sub_61789))
        
        # Assigning a Attribute to a Subscript (line 71):
        
        # Assigning a Attribute to a Subscript (line 71):
        # Getting the type of 'Path' (line 71)
        Path_61795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'Path')
        # Obtaining the member 'MOVETO' of a type (line 71)
        MOVETO_61796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 22), Path_61795, 'MOVETO')
        # Getting the type of 'codes' (line 71)
        codes_61797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'codes')
        int_61798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 14), 'int')
        int_61799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 17), 'int')
        slice_61800 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 71, 8), int_61798, None, int_61799)
        # Storing an element on a container (line 71)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 8), codes_61797, (slice_61800, MOVETO_61796))
        
        # Assigning a Attribute to a Subscript (line 72):
        
        # Assigning a Attribute to a Subscript (line 72):
        # Getting the type of 'Path' (line 72)
        Path_61801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'Path')
        # Obtaining the member 'LINETO' of a type (line 72)
        LINETO_61802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 22), Path_61801, 'LINETO')
        # Getting the type of 'codes' (line 72)
        codes_61803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'codes')
        int_61804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 14), 'int')
        int_61805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 17), 'int')
        slice_61806 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 72, 8), int_61804, None, int_61805)
        # Storing an element on a container (line 72)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 8), codes_61803, (slice_61806, LINETO_61802))
        
        # ################# End of 'set_vertices_and_codes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_vertices_and_codes' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_61807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61807)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_vertices_and_codes'
        return stypy_return_type_61807


# Assigning a type to the variable 'NorthEastHatch' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'NorthEastHatch', NorthEastHatch)
# Declaration of the 'SouthEastHatch' class
# Getting the type of 'HatchPatternBase' (line 75)
HatchPatternBase_61808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'HatchPatternBase')

class SouthEastHatch(HatchPatternBase_61808, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 76, 4, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SouthEastHatch.__init__', ['hatch', 'density'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['hatch', 'density'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 77):
        
        # Assigning a Call to a Attribute (line 77):
        
        # Call to int(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to count(...): (line 77)
        # Processing the call arguments (line 77)
        unicode_61812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 42), 'unicode', u'\\')
        # Processing the call keyword arguments (line 77)
        kwargs_61813 = {}
        # Getting the type of 'hatch' (line 77)
        hatch_61810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 30), 'hatch', False)
        # Obtaining the member 'count' of a type (line 77)
        count_61811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 30), hatch_61810, 'count')
        # Calling count(args, kwargs) (line 77)
        count_call_result_61814 = invoke(stypy.reporting.localization.Localization(__file__, 77, 30), count_61811, *[unicode_61812], **kwargs_61813)
        
        
        # Call to count(...): (line 77)
        # Processing the call arguments (line 77)
        unicode_61817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 62), 'unicode', u'x')
        # Processing the call keyword arguments (line 77)
        kwargs_61818 = {}
        # Getting the type of 'hatch' (line 77)
        hatch_61815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 50), 'hatch', False)
        # Obtaining the member 'count' of a type (line 77)
        count_61816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 50), hatch_61815, 'count')
        # Calling count(args, kwargs) (line 77)
        count_call_result_61819 = invoke(stypy.reporting.localization.Localization(__file__, 77, 50), count_61816, *[unicode_61817], **kwargs_61818)
        
        # Applying the binary operator '+' (line 77)
        result_add_61820 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 30), '+', count_call_result_61814, count_call_result_61819)
        
        
        # Call to count(...): (line 78)
        # Processing the call arguments (line 78)
        unicode_61823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 38), 'unicode', u'X')
        # Processing the call keyword arguments (line 78)
        kwargs_61824 = {}
        # Getting the type of 'hatch' (line 78)
        hatch_61821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'hatch', False)
        # Obtaining the member 'count' of a type (line 78)
        count_61822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 26), hatch_61821, 'count')
        # Calling count(args, kwargs) (line 78)
        count_call_result_61825 = invoke(stypy.reporting.localization.Localization(__file__, 78, 26), count_61822, *[unicode_61823], **kwargs_61824)
        
        # Applying the binary operator '+' (line 77)
        result_add_61826 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 67), '+', result_add_61820, count_call_result_61825)
        
        # Getting the type of 'density' (line 78)
        density_61827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 46), 'density', False)
        # Applying the binary operator '*' (line 77)
        result_mul_61828 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 29), '*', result_add_61826, density_61827)
        
        # Processing the call keyword arguments (line 77)
        kwargs_61829 = {}
        # Getting the type of 'int' (line 77)
        int_61809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 'int', False)
        # Calling int(args, kwargs) (line 77)
        int_call_result_61830 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), int_61809, *[result_mul_61828], **kwargs_61829)
        
        # Getting the type of 'self' (line 77)
        self_61831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self')
        # Setting the type of the member 'num_lines' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_61831, 'num_lines', int_call_result_61830)
        
        # Assigning a BinOp to a Attribute (line 79):
        
        # Assigning a BinOp to a Attribute (line 79):
        # Getting the type of 'self' (line 79)
        self_61832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'self')
        # Obtaining the member 'num_lines' of a type (line 79)
        num_lines_61833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 29), self_61832, 'num_lines')
        int_61834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 46), 'int')
        # Applying the binary operator '+' (line 79)
        result_add_61835 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 29), '+', num_lines_61833, int_61834)
        
        int_61836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 51), 'int')
        # Applying the binary operator '*' (line 79)
        result_mul_61837 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 28), '*', result_add_61835, int_61836)
        
        # Getting the type of 'self' (line 79)
        self_61838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'self')
        # Setting the type of the member 'num_vertices' of a type (line 79)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 8), self_61838, 'num_vertices', result_mul_61837)
        
        # Getting the type of 'self' (line 80)
        self_61839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'self')
        # Obtaining the member 'num_lines' of a type (line 80)
        num_lines_61840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), self_61839, 'num_lines')
        # Testing the type of an if condition (line 80)
        if_condition_61841 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), num_lines_61840)
        # Assigning a type to the variable 'if_condition_61841' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_61841', if_condition_61841)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Attribute (line 81):
        
        # Assigning a BinOp to a Attribute (line 81):
        # Getting the type of 'self' (line 81)
        self_61842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), 'self')
        # Obtaining the member 'num_lines' of a type (line 81)
        num_lines_61843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 33), self_61842, 'num_lines')
        int_61844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 50), 'int')
        # Applying the binary operator '+' (line 81)
        result_add_61845 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 33), '+', num_lines_61843, int_61844)
        
        int_61846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 55), 'int')
        # Applying the binary operator '*' (line 81)
        result_mul_61847 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 32), '*', result_add_61845, int_61846)
        
        # Getting the type of 'self' (line 81)
        self_61848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'self')
        # Setting the type of the member 'num_vertices' of a type (line 81)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), self_61848, 'num_vertices', result_mul_61847)
        # SSA branch for the else part of an if statement (line 80)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Attribute (line 83):
        
        # Assigning a Num to a Attribute (line 83):
        int_61849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 32), 'int')
        # Getting the type of 'self' (line 83)
        self_61850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'self')
        # Setting the type of the member 'num_vertices' of a type (line 83)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), self_61850, 'num_vertices', int_61849)
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_vertices_and_codes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_vertices_and_codes'
        module_type_store = module_type_store.open_function_context('set_vertices_and_codes', 85, 4, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SouthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_localization', localization)
        SouthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SouthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_type_store', module_type_store)
        SouthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_function_name', 'SouthEastHatch.set_vertices_and_codes')
        SouthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_param_names_list', ['vertices', 'codes'])
        SouthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_varargs_param_name', None)
        SouthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SouthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_call_defaults', defaults)
        SouthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_call_varargs', varargs)
        SouthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SouthEastHatch.set_vertices_and_codes.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SouthEastHatch.set_vertices_and_codes', ['vertices', 'codes'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_vertices_and_codes', localization, ['vertices', 'codes'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_vertices_and_codes(...)' code ##################

        
        # Assigning a Call to a Name (line 86):
        
        # Assigning a Call to a Name (line 86):
        
        # Call to linspace(...): (line 86)
        # Processing the call arguments (line 86)
        float_61853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 28), 'float')
        float_61854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 34), 'float')
        # Getting the type of 'self' (line 86)
        self_61855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 39), 'self', False)
        # Obtaining the member 'num_lines' of a type (line 86)
        num_lines_61856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 39), self_61855, 'num_lines')
        int_61857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 56), 'int')
        # Applying the binary operator '+' (line 86)
        result_add_61858 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 39), '+', num_lines_61856, int_61857)
        
        # Getting the type of 'True' (line 86)
        True_61859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 59), 'True', False)
        # Processing the call keyword arguments (line 86)
        kwargs_61860 = {}
        # Getting the type of 'np' (line 86)
        np_61851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'np', False)
        # Obtaining the member 'linspace' of a type (line 86)
        linspace_61852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 16), np_61851, 'linspace')
        # Calling linspace(args, kwargs) (line 86)
        linspace_call_result_61861 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), linspace_61852, *[float_61853, float_61854, result_add_61858, True_61859], **kwargs_61860)
        
        # Assigning a type to the variable 'steps' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'steps', linspace_call_result_61861)
        
        # Assigning a BinOp to a Subscript (line 87):
        
        # Assigning a BinOp to a Subscript (line 87):
        float_61862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 28), 'float')
        # Getting the type of 'steps' (line 87)
        steps_61863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 34), 'steps')
        # Applying the binary operator '+' (line 87)
        result_add_61864 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 28), '+', float_61862, steps_61863)
        
        # Getting the type of 'vertices' (line 87)
        vertices_61865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'vertices')
        int_61866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 17), 'int')
        int_61867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 20), 'int')
        slice_61868 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 87, 8), int_61866, None, int_61867)
        int_61869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 23), 'int')
        # Storing an element on a container (line 87)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), vertices_61865, ((slice_61868, int_61869), result_add_61864))
        
        # Assigning a BinOp to a Subscript (line 88):
        
        # Assigning a BinOp to a Subscript (line 88):
        float_61870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 28), 'float')
        # Getting the type of 'steps' (line 88)
        steps_61871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 34), 'steps')
        # Applying the binary operator '+' (line 88)
        result_add_61872 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 28), '+', float_61870, steps_61871)
        
        # Getting the type of 'vertices' (line 88)
        vertices_61873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'vertices')
        int_61874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 17), 'int')
        int_61875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 20), 'int')
        slice_61876 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 88, 8), int_61874, None, int_61875)
        int_61877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 23), 'int')
        # Storing an element on a container (line 88)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 8), vertices_61873, ((slice_61876, int_61877), result_add_61872))
        
        # Assigning a BinOp to a Subscript (line 89):
        
        # Assigning a BinOp to a Subscript (line 89):
        float_61878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 28), 'float')
        # Getting the type of 'steps' (line 89)
        steps_61879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 34), 'steps')
        # Applying the binary operator '+' (line 89)
        result_add_61880 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 28), '+', float_61878, steps_61879)
        
        # Getting the type of 'vertices' (line 89)
        vertices_61881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'vertices')
        int_61882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 17), 'int')
        int_61883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 20), 'int')
        slice_61884 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 8), int_61882, None, int_61883)
        int_61885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 23), 'int')
        # Storing an element on a container (line 89)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), vertices_61881, ((slice_61884, int_61885), result_add_61880))
        
        # Assigning a BinOp to a Subscript (line 90):
        
        # Assigning a BinOp to a Subscript (line 90):
        float_61886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 28), 'float')
        # Getting the type of 'steps' (line 90)
        steps_61887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 34), 'steps')
        # Applying the binary operator '+' (line 90)
        result_add_61888 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 28), '+', float_61886, steps_61887)
        
        # Getting the type of 'vertices' (line 90)
        vertices_61889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'vertices')
        int_61890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 17), 'int')
        int_61891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 20), 'int')
        slice_61892 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 90, 8), int_61890, None, int_61891)
        int_61893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 23), 'int')
        # Storing an element on a container (line 90)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), vertices_61889, ((slice_61892, int_61893), result_add_61888))
        
        # Assigning a Attribute to a Subscript (line 91):
        
        # Assigning a Attribute to a Subscript (line 91):
        # Getting the type of 'Path' (line 91)
        Path_61894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'Path')
        # Obtaining the member 'MOVETO' of a type (line 91)
        MOVETO_61895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 22), Path_61894, 'MOVETO')
        # Getting the type of 'codes' (line 91)
        codes_61896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'codes')
        int_61897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 14), 'int')
        int_61898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 17), 'int')
        slice_61899 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 91, 8), int_61897, None, int_61898)
        # Storing an element on a container (line 91)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 8), codes_61896, (slice_61899, MOVETO_61895))
        
        # Assigning a Attribute to a Subscript (line 92):
        
        # Assigning a Attribute to a Subscript (line 92):
        # Getting the type of 'Path' (line 92)
        Path_61900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'Path')
        # Obtaining the member 'LINETO' of a type (line 92)
        LINETO_61901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 22), Path_61900, 'LINETO')
        # Getting the type of 'codes' (line 92)
        codes_61902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'codes')
        int_61903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 14), 'int')
        int_61904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 17), 'int')
        slice_61905 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 92, 8), int_61903, None, int_61904)
        # Storing an element on a container (line 92)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 8), codes_61902, (slice_61905, LINETO_61901))
        
        # ################# End of 'set_vertices_and_codes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_vertices_and_codes' in the type store
        # Getting the type of 'stypy_return_type' (line 85)
        stypy_return_type_61906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61906)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_vertices_and_codes'
        return stypy_return_type_61906


# Assigning a type to the variable 'SouthEastHatch' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'SouthEastHatch', SouthEastHatch)
# Declaration of the 'Shapes' class
# Getting the type of 'HatchPatternBase' (line 95)
HatchPatternBase_61907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 13), 'HatchPatternBase')

class Shapes(HatchPatternBase_61907, ):
    
    # Assigning a Name to a Name (line 96):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 98, 4, False)
        # Assigning a type to the variable 'self' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Shapes.__init__', ['hatch', 'density'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['hatch', 'density'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # Getting the type of 'self' (line 99)
        self_61908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'self')
        # Obtaining the member 'num_rows' of a type (line 99)
        num_rows_61909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 11), self_61908, 'num_rows')
        int_61910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 28), 'int')
        # Applying the binary operator '==' (line 99)
        result_eq_61911 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 11), '==', num_rows_61909, int_61910)
        
        # Testing the type of an if condition (line 99)
        if_condition_61912 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 8), result_eq_61911)
        # Assigning a type to the variable 'if_condition_61912' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'if_condition_61912', if_condition_61912)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 100):
        
        # Assigning a Num to a Attribute (line 100):
        int_61913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 30), 'int')
        # Getting the type of 'self' (line 100)
        self_61914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'self')
        # Setting the type of the member 'num_shapes' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 12), self_61914, 'num_shapes', int_61913)
        
        # Assigning a Num to a Attribute (line 101):
        
        # Assigning a Num to a Attribute (line 101):
        int_61915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 32), 'int')
        # Getting the type of 'self' (line 101)
        self_61916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'self')
        # Setting the type of the member 'num_vertices' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 12), self_61916, 'num_vertices', int_61915)
        # SSA branch for the else part of an if statement (line 99)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a BinOp to a Attribute (line 103):
        
        # Assigning a BinOp to a Attribute (line 103):
        # Getting the type of 'self' (line 103)
        self_61917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 32), 'self')
        # Obtaining the member 'num_rows' of a type (line 103)
        num_rows_61918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 32), self_61917, 'num_rows')
        int_61919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 49), 'int')
        # Applying the binary operator '//' (line 103)
        result_floordiv_61920 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 32), '//', num_rows_61918, int_61919)
        
        int_61921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 53), 'int')
        # Applying the binary operator '+' (line 103)
        result_add_61922 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 32), '+', result_floordiv_61920, int_61921)
        
        # Getting the type of 'self' (line 103)
        self_61923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 59), 'self')
        # Obtaining the member 'num_rows' of a type (line 103)
        num_rows_61924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 59), self_61923, 'num_rows')
        int_61925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 75), 'int')
        # Applying the binary operator '+' (line 103)
        result_add_61926 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 59), '+', num_rows_61924, int_61925)
        
        # Applying the binary operator '*' (line 103)
        result_mul_61927 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 31), '*', result_add_61922, result_add_61926)
        
        # Getting the type of 'self' (line 104)
        self_61928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 32), 'self')
        # Obtaining the member 'num_rows' of a type (line 104)
        num_rows_61929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 32), self_61928, 'num_rows')
        int_61930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 49), 'int')
        # Applying the binary operator '//' (line 104)
        result_floordiv_61931 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 32), '//', num_rows_61929, int_61930)
        
        # Getting the type of 'self' (line 104)
        self_61932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 55), 'self')
        # Obtaining the member 'num_rows' of a type (line 104)
        num_rows_61933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 55), self_61932, 'num_rows')
        # Applying the binary operator '*' (line 104)
        result_mul_61934 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 31), '*', result_floordiv_61931, num_rows_61933)
        
        # Applying the binary operator '+' (line 103)
        result_add_61935 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 31), '+', result_mul_61927, result_mul_61934)
        
        # Getting the type of 'self' (line 103)
        self_61936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self')
        # Setting the type of the member 'num_shapes' of a type (line 103)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_61936, 'num_shapes', result_add_61935)
        
        # Assigning a BinOp to a Attribute (line 105):
        
        # Assigning a BinOp to a Attribute (line 105):
        # Getting the type of 'self' (line 105)
        self_61937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 33), 'self')
        # Obtaining the member 'num_shapes' of a type (line 105)
        num_shapes_61938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 33), self_61937, 'num_shapes')
        
        # Call to len(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'self' (line 106)
        self_61940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 37), 'self', False)
        # Obtaining the member 'shape_vertices' of a type (line 106)
        shape_vertices_61941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 37), self_61940, 'shape_vertices')
        # Processing the call keyword arguments (line 106)
        kwargs_61942 = {}
        # Getting the type of 'len' (line 106)
        len_61939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'len', False)
        # Calling len(args, kwargs) (line 106)
        len_call_result_61943 = invoke(stypy.reporting.localization.Localization(__file__, 106, 33), len_61939, *[shape_vertices_61941], **kwargs_61942)
        
        # Applying the binary operator '*' (line 105)
        result_mul_61944 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 33), '*', num_shapes_61938, len_call_result_61943)
        
        
        # Evaluating a boolean operation
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 107)
        self_61945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 34), 'self')
        # Obtaining the member 'filled' of a type (line 107)
        filled_61946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 34), self_61945, 'filled')
        int_61947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 50), 'int')
        # Applying the binary operator 'and' (line 107)
        result_and_keyword_61948 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 34), 'and', filled_61946, int_61947)
        
        int_61949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 55), 'int')
        # Applying the binary operator 'or' (line 107)
        result_or_keyword_61950 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 34), 'or', result_and_keyword_61948, int_61949)
        
        # Applying the binary operator '*' (line 106)
        result_mul_61951 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 58), '*', result_mul_61944, result_or_keyword_61950)
        
        # Getting the type of 'self' (line 105)
        self_61952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'self')
        # Setting the type of the member 'num_vertices' of a type (line 105)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), self_61952, 'num_vertices', result_mul_61951)
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def set_vertices_and_codes(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_vertices_and_codes'
        module_type_store = module_type_store.open_function_context('set_vertices_and_codes', 109, 4, False)
        # Assigning a type to the variable 'self' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Shapes.set_vertices_and_codes.__dict__.__setitem__('stypy_localization', localization)
        Shapes.set_vertices_and_codes.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Shapes.set_vertices_and_codes.__dict__.__setitem__('stypy_type_store', module_type_store)
        Shapes.set_vertices_and_codes.__dict__.__setitem__('stypy_function_name', 'Shapes.set_vertices_and_codes')
        Shapes.set_vertices_and_codes.__dict__.__setitem__('stypy_param_names_list', ['vertices', 'codes'])
        Shapes.set_vertices_and_codes.__dict__.__setitem__('stypy_varargs_param_name', None)
        Shapes.set_vertices_and_codes.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Shapes.set_vertices_and_codes.__dict__.__setitem__('stypy_call_defaults', defaults)
        Shapes.set_vertices_and_codes.__dict__.__setitem__('stypy_call_varargs', varargs)
        Shapes.set_vertices_and_codes.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Shapes.set_vertices_and_codes.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Shapes.set_vertices_and_codes', ['vertices', 'codes'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_vertices_and_codes', localization, ['vertices', 'codes'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_vertices_and_codes(...)' code ##################

        
        # Assigning a BinOp to a Name (line 110):
        
        # Assigning a BinOp to a Name (line 110):
        float_61953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 17), 'float')
        # Getting the type of 'self' (line 110)
        self_61954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 23), 'self')
        # Obtaining the member 'num_rows' of a type (line 110)
        num_rows_61955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 23), self_61954, 'num_rows')
        # Applying the binary operator 'div' (line 110)
        result_div_61956 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 17), 'div', float_61953, num_rows_61955)
        
        # Assigning a type to the variable 'offset' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'offset', result_div_61956)
        
        # Assigning a BinOp to a Name (line 111):
        
        # Assigning a BinOp to a Name (line 111):
        # Getting the type of 'self' (line 111)
        self_61957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'self')
        # Obtaining the member 'shape_vertices' of a type (line 111)
        shape_vertices_61958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 25), self_61957, 'shape_vertices')
        # Getting the type of 'offset' (line 111)
        offset_61959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 47), 'offset')
        # Applying the binary operator '*' (line 111)
        result_mul_61960 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 25), '*', shape_vertices_61958, offset_61959)
        
        # Getting the type of 'self' (line 111)
        self_61961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 56), 'self')
        # Obtaining the member 'size' of a type (line 111)
        size_61962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 56), self_61961, 'size')
        # Applying the binary operator '*' (line 111)
        result_mul_61963 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 54), '*', result_mul_61960, size_61962)
        
        # Assigning a type to the variable 'shape_vertices' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'shape_vertices', result_mul_61963)
        
        
        # Getting the type of 'self' (line 112)
        self_61964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'self')
        # Obtaining the member 'filled' of a type (line 112)
        filled_61965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 15), self_61964, 'filled')
        # Applying the 'not' unary operator (line 112)
        result_not__61966 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), 'not', filled_61965)
        
        # Testing the type of an if condition (line 112)
        if_condition_61967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), result_not__61966)
        # Assigning a type to the variable 'if_condition_61967' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_61967', if_condition_61967)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 113):
        
        # Assigning a BinOp to a Name (line 113):
        
        # Obtaining the type of the subscript
        int_61968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 46), 'int')
        slice_61969 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 113, 29), None, None, int_61968)
        # Getting the type of 'shape_vertices' (line 113)
        shape_vertices_61970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'shape_vertices')
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___61971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 29), shape_vertices_61970, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_61972 = invoke(stypy.reporting.localization.Localization(__file__, 113, 29), getitem___61971, slice_61969)
        
        float_61973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 52), 'float')
        # Applying the binary operator '*' (line 113)
        result_mul_61974 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 29), '*', subscript_call_result_61972, float_61973)
        
        # Assigning a type to the variable 'inner_vertices' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'inner_vertices', result_mul_61974)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 114):
        
        # Assigning a Attribute to a Name (line 114):
        # Getting the type of 'self' (line 114)
        self_61975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'self')
        # Obtaining the member 'shape_codes' of a type (line 114)
        shape_codes_61976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 22), self_61975, 'shape_codes')
        # Assigning a type to the variable 'shape_codes' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'shape_codes', shape_codes_61976)
        
        # Assigning a Call to a Name (line 115):
        
        # Assigning a Call to a Name (line 115):
        
        # Call to len(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'shape_vertices' (line 115)
        shape_vertices_61978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'shape_vertices', False)
        # Processing the call keyword arguments (line 115)
        kwargs_61979 = {}
        # Getting the type of 'len' (line 115)
        len_61977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 21), 'len', False)
        # Calling len(args, kwargs) (line 115)
        len_call_result_61980 = invoke(stypy.reporting.localization.Localization(__file__, 115, 21), len_61977, *[shape_vertices_61978], **kwargs_61979)
        
        # Assigning a type to the variable 'shape_size' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'shape_size', len_call_result_61980)
        
        # Assigning a Num to a Name (line 117):
        
        # Assigning a Num to a Name (line 117):
        int_61981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 17), 'int')
        # Assigning a type to the variable 'cursor' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'cursor', int_61981)
        
        
        # Call to xrange(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'self' (line 118)
        self_61983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'self', False)
        # Obtaining the member 'num_rows' of a type (line 118)
        num_rows_61984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 26), self_61983, 'num_rows')
        int_61985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 42), 'int')
        # Applying the binary operator '+' (line 118)
        result_add_61986 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 26), '+', num_rows_61984, int_61985)
        
        # Processing the call keyword arguments (line 118)
        kwargs_61987 = {}
        # Getting the type of 'xrange' (line 118)
        xrange_61982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 19), 'xrange', False)
        # Calling xrange(args, kwargs) (line 118)
        xrange_call_result_61988 = invoke(stypy.reporting.localization.Localization(__file__, 118, 19), xrange_61982, *[result_add_61986], **kwargs_61987)
        
        # Testing the type of a for loop iterable (line 118)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 8), xrange_call_result_61988)
        # Getting the type of the for loop variable (line 118)
        for_loop_var_61989 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 8), xrange_call_result_61988)
        # Assigning a type to the variable 'row' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'row', for_loop_var_61989)
        # SSA begins for a for statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'row' (line 119)
        row_61990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'row')
        int_61991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 21), 'int')
        # Applying the binary operator '%' (line 119)
        result_mod_61992 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 15), '%', row_61990, int_61991)
        
        int_61993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 26), 'int')
        # Applying the binary operator '==' (line 119)
        result_eq_61994 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 15), '==', result_mod_61992, int_61993)
        
        # Testing the type of an if condition (line 119)
        if_condition_61995 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 12), result_eq_61994)
        # Assigning a type to the variable 'if_condition_61995' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'if_condition_61995', if_condition_61995)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 120):
        
        # Assigning a Call to a Name (line 120):
        
        # Call to linspace(...): (line 120)
        # Processing the call arguments (line 120)
        float_61998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 35), 'float')
        float_61999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 40), 'float')
        # Getting the type of 'self' (line 120)
        self_62000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 45), 'self', False)
        # Obtaining the member 'num_rows' of a type (line 120)
        num_rows_62001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 45), self_62000, 'num_rows')
        int_62002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 61), 'int')
        # Applying the binary operator '+' (line 120)
        result_add_62003 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 45), '+', num_rows_62001, int_62002)
        
        # Getting the type of 'True' (line 120)
        True_62004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 64), 'True', False)
        # Processing the call keyword arguments (line 120)
        kwargs_62005 = {}
        # Getting the type of 'np' (line 120)
        np_61996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 23), 'np', False)
        # Obtaining the member 'linspace' of a type (line 120)
        linspace_61997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 23), np_61996, 'linspace')
        # Calling linspace(args, kwargs) (line 120)
        linspace_call_result_62006 = invoke(stypy.reporting.localization.Localization(__file__, 120, 23), linspace_61997, *[float_61998, float_61999, result_add_62003, True_62004], **kwargs_62005)
        
        # Assigning a type to the variable 'cols' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'cols', linspace_call_result_62006)
        # SSA branch for the else part of an if statement (line 119)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 122):
        
        # Assigning a Call to a Name (line 122):
        
        # Call to linspace(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'offset' (line 122)
        offset_62009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 35), 'offset', False)
        float_62010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 44), 'float')
        # Applying the binary operator 'div' (line 122)
        result_div_62011 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 35), 'div', offset_62009, float_62010)
        
        float_62012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 49), 'float')
        # Getting the type of 'offset' (line 122)
        offset_62013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 55), 'offset', False)
        float_62014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 64), 'float')
        # Applying the binary operator 'div' (line 122)
        result_div_62015 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 55), 'div', offset_62013, float_62014)
        
        # Applying the binary operator '-' (line 122)
        result_sub_62016 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 49), '-', float_62012, result_div_62015)
        
        # Getting the type of 'self' (line 123)
        self_62017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 35), 'self', False)
        # Obtaining the member 'num_rows' of a type (line 123)
        num_rows_62018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 35), self_62017, 'num_rows')
        # Getting the type of 'True' (line 123)
        True_62019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 50), 'True', False)
        # Processing the call keyword arguments (line 122)
        kwargs_62020 = {}
        # Getting the type of 'np' (line 122)
        np_62007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 23), 'np', False)
        # Obtaining the member 'linspace' of a type (line 122)
        linspace_62008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 23), np_62007, 'linspace')
        # Calling linspace(args, kwargs) (line 122)
        linspace_call_result_62021 = invoke(stypy.reporting.localization.Localization(__file__, 122, 23), linspace_62008, *[result_div_62011, result_sub_62016, num_rows_62018, True_62019], **kwargs_62020)
        
        # Assigning a type to the variable 'cols' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'cols', linspace_call_result_62021)
        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a BinOp to a Name (line 124):
        
        # Assigning a BinOp to a Name (line 124):
        # Getting the type of 'row' (line 124)
        row_62022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 22), 'row')
        # Getting the type of 'offset' (line 124)
        offset_62023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'offset')
        # Applying the binary operator '*' (line 124)
        result_mul_62024 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 22), '*', row_62022, offset_62023)
        
        # Assigning a type to the variable 'row_pos' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'row_pos', result_mul_62024)
        
        # Getting the type of 'cols' (line 125)
        cols_62025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 27), 'cols')
        # Testing the type of a for loop iterable (line 125)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 125, 12), cols_62025)
        # Getting the type of the for loop variable (line 125)
        for_loop_var_62026 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 125, 12), cols_62025)
        # Assigning a type to the variable 'col_pos' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'col_pos', for_loop_var_62026)
        # SSA begins for a for statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Subscript (line 126):
        
        # Assigning a BinOp to a Subscript (line 126):
        # Getting the type of 'shape_vertices' (line 126)
        shape_vertices_62027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 56), 'shape_vertices')
        
        # Obtaining an instance of the builtin type 'tuple' (line 127)
        tuple_62028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 57), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 127)
        # Adding element type (line 127)
        # Getting the type of 'col_pos' (line 127)
        col_pos_62029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 57), 'col_pos')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 57), tuple_62028, col_pos_62029)
        # Adding element type (line 127)
        # Getting the type of 'row_pos' (line 127)
        row_pos_62030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 66), 'row_pos')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 57), tuple_62028, row_pos_62030)
        
        # Applying the binary operator '+' (line 126)
        result_add_62031 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 56), '+', shape_vertices_62027, tuple_62028)
        
        # Getting the type of 'vertices' (line 126)
        vertices_62032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'vertices')
        # Getting the type of 'cursor' (line 126)
        cursor_62033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 25), 'cursor')
        # Getting the type of 'cursor' (line 126)
        cursor_62034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 32), 'cursor')
        # Getting the type of 'shape_size' (line 126)
        shape_size_62035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 41), 'shape_size')
        # Applying the binary operator '+' (line 126)
        result_add_62036 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 32), '+', cursor_62034, shape_size_62035)
        
        slice_62037 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 126, 16), cursor_62033, result_add_62036, None)
        # Storing an element on a container (line 126)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 16), vertices_62032, (slice_62037, result_add_62031))
        
        # Assigning a Name to a Subscript (line 128):
        
        # Assigning a Name to a Subscript (line 128):
        # Getting the type of 'shape_codes' (line 128)
        shape_codes_62038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 52), 'shape_codes')
        # Getting the type of 'codes' (line 128)
        codes_62039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'codes')
        # Getting the type of 'cursor' (line 128)
        cursor_62040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 22), 'cursor')
        # Getting the type of 'cursor' (line 128)
        cursor_62041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'cursor')
        # Getting the type of 'shape_size' (line 128)
        shape_size_62042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 38), 'shape_size')
        # Applying the binary operator '+' (line 128)
        result_add_62043 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 29), '+', cursor_62041, shape_size_62042)
        
        slice_62044 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 128, 16), cursor_62040, result_add_62043, None)
        # Storing an element on a container (line 128)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 16), codes_62039, (slice_62044, shape_codes_62038))
        
        # Getting the type of 'cursor' (line 129)
        cursor_62045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'cursor')
        # Getting the type of 'shape_size' (line 129)
        shape_size_62046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'shape_size')
        # Applying the binary operator '+=' (line 129)
        result_iadd_62047 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 16), '+=', cursor_62045, shape_size_62046)
        # Assigning a type to the variable 'cursor' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'cursor', result_iadd_62047)
        
        
        
        # Getting the type of 'self' (line 130)
        self_62048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 23), 'self')
        # Obtaining the member 'filled' of a type (line 130)
        filled_62049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 23), self_62048, 'filled')
        # Applying the 'not' unary operator (line 130)
        result_not__62050 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 19), 'not', filled_62049)
        
        # Testing the type of an if condition (line 130)
        if_condition_62051 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 16), result_not__62050)
        # Assigning a type to the variable 'if_condition_62051' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'if_condition_62051', if_condition_62051)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Subscript (line 131):
        
        # Assigning a BinOp to a Subscript (line 131):
        # Getting the type of 'inner_vertices' (line 131)
        inner_vertices_62052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 60), 'inner_vertices')
        
        # Obtaining an instance of the builtin type 'tuple' (line 132)
        tuple_62053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 61), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 132)
        # Adding element type (line 132)
        # Getting the type of 'col_pos' (line 132)
        col_pos_62054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 61), 'col_pos')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 61), tuple_62053, col_pos_62054)
        # Adding element type (line 132)
        # Getting the type of 'row_pos' (line 132)
        row_pos_62055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 70), 'row_pos')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 61), tuple_62053, row_pos_62055)
        
        # Applying the binary operator '+' (line 131)
        result_add_62056 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 60), '+', inner_vertices_62052, tuple_62053)
        
        # Getting the type of 'vertices' (line 131)
        vertices_62057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 20), 'vertices')
        # Getting the type of 'cursor' (line 131)
        cursor_62058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'cursor')
        # Getting the type of 'cursor' (line 131)
        cursor_62059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 36), 'cursor')
        # Getting the type of 'shape_size' (line 131)
        shape_size_62060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 45), 'shape_size')
        # Applying the binary operator '+' (line 131)
        result_add_62061 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 36), '+', cursor_62059, shape_size_62060)
        
        slice_62062 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 131, 20), cursor_62058, result_add_62061, None)
        # Storing an element on a container (line 131)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 20), vertices_62057, (slice_62062, result_add_62056))
        
        # Assigning a Name to a Subscript (line 133):
        
        # Assigning a Name to a Subscript (line 133):
        # Getting the type of 'shape_codes' (line 133)
        shape_codes_62063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 56), 'shape_codes')
        # Getting the type of 'codes' (line 133)
        codes_62064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), 'codes')
        # Getting the type of 'cursor' (line 133)
        cursor_62065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 26), 'cursor')
        # Getting the type of 'cursor' (line 133)
        cursor_62066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 33), 'cursor')
        # Getting the type of 'shape_size' (line 133)
        shape_size_62067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 42), 'shape_size')
        # Applying the binary operator '+' (line 133)
        result_add_62068 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 33), '+', cursor_62066, shape_size_62067)
        
        slice_62069 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 133, 20), cursor_62065, result_add_62068, None)
        # Storing an element on a container (line 133)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 20), codes_62064, (slice_62069, shape_codes_62063))
        
        # Getting the type of 'cursor' (line 134)
        cursor_62070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'cursor')
        # Getting the type of 'shape_size' (line 134)
        shape_size_62071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 30), 'shape_size')
        # Applying the binary operator '+=' (line 134)
        result_iadd_62072 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 20), '+=', cursor_62070, shape_size_62071)
        # Assigning a type to the variable 'cursor' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'cursor', result_iadd_62072)
        
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'set_vertices_and_codes(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_vertices_and_codes' in the type store
        # Getting the type of 'stypy_return_type' (line 109)
        stypy_return_type_62073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62073)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_vertices_and_codes'
        return stypy_return_type_62073


# Assigning a type to the variable 'Shapes' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'Shapes', Shapes)

# Assigning a Name to a Name (line 96):
# Getting the type of 'False' (line 96)
False_62074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 13), 'False')
# Getting the type of 'Shapes'
Shapes_62075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Shapes')
# Setting the type of the member 'filled' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Shapes_62075, 'filled', False_62074)
# Declaration of the 'Circles' class
# Getting the type of 'Shapes' (line 137)
Shapes_62076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 14), 'Shapes')

class Circles(Shapes_62076, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Circles.__init__', ['hatch', 'density'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['hatch', 'density'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Name (line 139):
        
        # Assigning a Call to a Name (line 139):
        
        # Call to unit_circle(...): (line 139)
        # Processing the call keyword arguments (line 139)
        kwargs_62079 = {}
        # Getting the type of 'Path' (line 139)
        Path_62077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'Path', False)
        # Obtaining the member 'unit_circle' of a type (line 139)
        unit_circle_62078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 15), Path_62077, 'unit_circle')
        # Calling unit_circle(args, kwargs) (line 139)
        unit_circle_call_result_62080 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), unit_circle_62078, *[], **kwargs_62079)
        
        # Assigning a type to the variable 'path' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'path', unit_circle_call_result_62080)
        
        # Assigning a Attribute to a Attribute (line 140):
        
        # Assigning a Attribute to a Attribute (line 140):
        # Getting the type of 'path' (line 140)
        path_62081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'path')
        # Obtaining the member 'vertices' of a type (line 140)
        vertices_62082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 30), path_62081, 'vertices')
        # Getting the type of 'self' (line 140)
        self_62083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'self')
        # Setting the type of the member 'shape_vertices' of a type (line 140)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 8), self_62083, 'shape_vertices', vertices_62082)
        
        # Assigning a Attribute to a Attribute (line 141):
        
        # Assigning a Attribute to a Attribute (line 141):
        # Getting the type of 'path' (line 141)
        path_62084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 27), 'path')
        # Obtaining the member 'codes' of a type (line 141)
        codes_62085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 27), path_62084, 'codes')
        # Getting the type of 'self' (line 141)
        self_62086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'self')
        # Setting the type of the member 'shape_codes' of a type (line 141)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 8), self_62086, 'shape_codes', codes_62085)
        
        # Call to __init__(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'self' (line 142)
        self_62089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'self', False)
        # Getting the type of 'hatch' (line 142)
        hatch_62090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'hatch', False)
        # Getting the type of 'density' (line 142)
        density_62091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 37), 'density', False)
        # Processing the call keyword arguments (line 142)
        kwargs_62092 = {}
        # Getting the type of 'Shapes' (line 142)
        Shapes_62087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'Shapes', False)
        # Obtaining the member '__init__' of a type (line 142)
        init___62088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), Shapes_62087, '__init__')
        # Calling __init__(args, kwargs) (line 142)
        init___call_result_62093 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), init___62088, *[self_62089, hatch_62090, density_62091], **kwargs_62092)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Circles' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'Circles', Circles)
# Declaration of the 'SmallCircles' class
# Getting the type of 'Circles' (line 145)
Circles_62094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'Circles')

class SmallCircles(Circles_62094, ):
    
    # Assigning a Num to a Name (line 146):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SmallCircles.__init__', ['hatch', 'density'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['hatch', 'density'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a BinOp to a Attribute (line 149):
        
        # Assigning a BinOp to a Attribute (line 149):
        
        # Call to count(...): (line 149)
        # Processing the call arguments (line 149)
        unicode_62097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 37), 'unicode', u'o')
        # Processing the call keyword arguments (line 149)
        kwargs_62098 = {}
        # Getting the type of 'hatch' (line 149)
        hatch_62095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 25), 'hatch', False)
        # Obtaining the member 'count' of a type (line 149)
        count_62096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 25), hatch_62095, 'count')
        # Calling count(args, kwargs) (line 149)
        count_call_result_62099 = invoke(stypy.reporting.localization.Localization(__file__, 149, 25), count_62096, *[unicode_62097], **kwargs_62098)
        
        # Getting the type of 'density' (line 149)
        density_62100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 45), 'density')
        # Applying the binary operator '*' (line 149)
        result_mul_62101 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 24), '*', count_call_result_62099, density_62100)
        
        # Getting the type of 'self' (line 149)
        self_62102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'self')
        # Setting the type of the member 'num_rows' of a type (line 149)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 8), self_62102, 'num_rows', result_mul_62101)
        
        # Call to __init__(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'self' (line 150)
        self_62105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 25), 'self', False)
        # Getting the type of 'hatch' (line 150)
        hatch_62106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 31), 'hatch', False)
        # Getting the type of 'density' (line 150)
        density_62107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 38), 'density', False)
        # Processing the call keyword arguments (line 150)
        kwargs_62108 = {}
        # Getting the type of 'Circles' (line 150)
        Circles_62103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'Circles', False)
        # Obtaining the member '__init__' of a type (line 150)
        init___62104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 8), Circles_62103, '__init__')
        # Calling __init__(args, kwargs) (line 150)
        init___call_result_62109 = invoke(stypy.reporting.localization.Localization(__file__, 150, 8), init___62104, *[self_62105, hatch_62106, density_62107], **kwargs_62108)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'SmallCircles' (line 145)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'SmallCircles', SmallCircles)

# Assigning a Num to a Name (line 146):
float_62110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 11), 'float')
# Getting the type of 'SmallCircles'
SmallCircles_62111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SmallCircles')
# Setting the type of the member 'size' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SmallCircles_62111, 'size', float_62110)
# Declaration of the 'LargeCircles' class
# Getting the type of 'Circles' (line 153)
Circles_62112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 19), 'Circles')

class LargeCircles(Circles_62112, ):
    
    # Assigning a Num to a Name (line 154):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 156, 4, False)
        # Assigning a type to the variable 'self' (line 157)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LargeCircles.__init__', ['hatch', 'density'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['hatch', 'density'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a BinOp to a Attribute (line 157):
        
        # Assigning a BinOp to a Attribute (line 157):
        
        # Call to count(...): (line 157)
        # Processing the call arguments (line 157)
        unicode_62115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 37), 'unicode', u'O')
        # Processing the call keyword arguments (line 157)
        kwargs_62116 = {}
        # Getting the type of 'hatch' (line 157)
        hatch_62113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 25), 'hatch', False)
        # Obtaining the member 'count' of a type (line 157)
        count_62114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 25), hatch_62113, 'count')
        # Calling count(args, kwargs) (line 157)
        count_call_result_62117 = invoke(stypy.reporting.localization.Localization(__file__, 157, 25), count_62114, *[unicode_62115], **kwargs_62116)
        
        # Getting the type of 'density' (line 157)
        density_62118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 45), 'density')
        # Applying the binary operator '*' (line 157)
        result_mul_62119 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 24), '*', count_call_result_62117, density_62118)
        
        # Getting the type of 'self' (line 157)
        self_62120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 8), 'self')
        # Setting the type of the member 'num_rows' of a type (line 157)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 8), self_62120, 'num_rows', result_mul_62119)
        
        # Call to __init__(...): (line 158)
        # Processing the call arguments (line 158)
        # Getting the type of 'self' (line 158)
        self_62123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 25), 'self', False)
        # Getting the type of 'hatch' (line 158)
        hatch_62124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 31), 'hatch', False)
        # Getting the type of 'density' (line 158)
        density_62125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 38), 'density', False)
        # Processing the call keyword arguments (line 158)
        kwargs_62126 = {}
        # Getting the type of 'Circles' (line 158)
        Circles_62121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'Circles', False)
        # Obtaining the member '__init__' of a type (line 158)
        init___62122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), Circles_62121, '__init__')
        # Calling __init__(args, kwargs) (line 158)
        init___call_result_62127 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), init___62122, *[self_62123, hatch_62124, density_62125], **kwargs_62126)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'LargeCircles' (line 153)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'LargeCircles', LargeCircles)

# Assigning a Num to a Name (line 154):
float_62128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 11), 'float')
# Getting the type of 'LargeCircles'
LargeCircles_62129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LargeCircles')
# Setting the type of the member 'size' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LargeCircles_62129, 'size', float_62128)
# Declaration of the 'SmallFilledCircles' class
# Getting the type of 'SmallCircles' (line 161)
SmallCircles_62130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 25), 'SmallCircles')

class SmallFilledCircles(SmallCircles_62130, ):
    
    # Assigning a Num to a Name (line 162):
    
    # Assigning a Name to a Name (line 163):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SmallFilledCircles.__init__', ['hatch', 'density'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['hatch', 'density'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a BinOp to a Attribute (line 166):
        
        # Assigning a BinOp to a Attribute (line 166):
        
        # Call to count(...): (line 166)
        # Processing the call arguments (line 166)
        unicode_62133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 166, 37), 'unicode', u'.')
        # Processing the call keyword arguments (line 166)
        kwargs_62134 = {}
        # Getting the type of 'hatch' (line 166)
        hatch_62131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 25), 'hatch', False)
        # Obtaining the member 'count' of a type (line 166)
        count_62132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), hatch_62131, 'count')
        # Calling count(args, kwargs) (line 166)
        count_call_result_62135 = invoke(stypy.reporting.localization.Localization(__file__, 166, 25), count_62132, *[unicode_62133], **kwargs_62134)
        
        # Getting the type of 'density' (line 166)
        density_62136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 45), 'density')
        # Applying the binary operator '*' (line 166)
        result_mul_62137 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 24), '*', count_call_result_62135, density_62136)
        
        # Getting the type of 'self' (line 166)
        self_62138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self')
        # Setting the type of the member 'num_rows' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_62138, 'num_rows', result_mul_62137)
        
        # Call to __init__(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'self' (line 167)
        self_62141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'self', False)
        # Getting the type of 'hatch' (line 167)
        hatch_62142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 31), 'hatch', False)
        # Getting the type of 'density' (line 167)
        density_62143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 38), 'density', False)
        # Processing the call keyword arguments (line 167)
        kwargs_62144 = {}
        # Getting the type of 'Circles' (line 167)
        Circles_62139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'Circles', False)
        # Obtaining the member '__init__' of a type (line 167)
        init___62140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), Circles_62139, '__init__')
        # Calling __init__(args, kwargs) (line 167)
        init___call_result_62145 = invoke(stypy.reporting.localization.Localization(__file__, 167, 8), init___62140, *[self_62141, hatch_62142, density_62143], **kwargs_62144)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'SmallFilledCircles' (line 161)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 0), 'SmallFilledCircles', SmallFilledCircles)

# Assigning a Num to a Name (line 162):
float_62146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 11), 'float')
# Getting the type of 'SmallFilledCircles'
SmallFilledCircles_62147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SmallFilledCircles')
# Setting the type of the member 'size' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SmallFilledCircles_62147, 'size', float_62146)

# Assigning a Name to a Name (line 163):
# Getting the type of 'True' (line 163)
True_62148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 13), 'True')
# Getting the type of 'SmallFilledCircles'
SmallFilledCircles_62149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SmallFilledCircles')
# Setting the type of the member 'filled' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SmallFilledCircles_62149, 'filled', True_62148)
# Declaration of the 'Stars' class
# Getting the type of 'Shapes' (line 170)
Shapes_62150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 12), 'Shapes')

class Stars(Shapes_62150, ):
    
    # Assigning a BinOp to a Name (line 171):
    
    # Assigning a Name to a Name (line 172):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 174, 4, False)
        # Assigning a type to the variable 'self' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Stars.__init__', ['hatch', 'density'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['hatch', 'density'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a BinOp to a Attribute (line 175):
        
        # Assigning a BinOp to a Attribute (line 175):
        
        # Call to count(...): (line 175)
        # Processing the call arguments (line 175)
        unicode_62153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 37), 'unicode', u'*')
        # Processing the call keyword arguments (line 175)
        kwargs_62154 = {}
        # Getting the type of 'hatch' (line 175)
        hatch_62151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 25), 'hatch', False)
        # Obtaining the member 'count' of a type (line 175)
        count_62152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 25), hatch_62151, 'count')
        # Calling count(args, kwargs) (line 175)
        count_call_result_62155 = invoke(stypy.reporting.localization.Localization(__file__, 175, 25), count_62152, *[unicode_62153], **kwargs_62154)
        
        # Getting the type of 'density' (line 175)
        density_62156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 45), 'density')
        # Applying the binary operator '*' (line 175)
        result_mul_62157 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 24), '*', count_call_result_62155, density_62156)
        
        # Getting the type of 'self' (line 175)
        self_62158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'self')
        # Setting the type of the member 'num_rows' of a type (line 175)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), self_62158, 'num_rows', result_mul_62157)
        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to unit_regular_star(...): (line 176)
        # Processing the call arguments (line 176)
        int_62161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 38), 'int')
        # Processing the call keyword arguments (line 176)
        kwargs_62162 = {}
        # Getting the type of 'Path' (line 176)
        Path_62159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'Path', False)
        # Obtaining the member 'unit_regular_star' of a type (line 176)
        unit_regular_star_62160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 15), Path_62159, 'unit_regular_star')
        # Calling unit_regular_star(args, kwargs) (line 176)
        unit_regular_star_call_result_62163 = invoke(stypy.reporting.localization.Localization(__file__, 176, 15), unit_regular_star_62160, *[int_62161], **kwargs_62162)
        
        # Assigning a type to the variable 'path' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'path', unit_regular_star_call_result_62163)
        
        # Assigning a Attribute to a Attribute (line 177):
        
        # Assigning a Attribute to a Attribute (line 177):
        # Getting the type of 'path' (line 177)
        path_62164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 30), 'path')
        # Obtaining the member 'vertices' of a type (line 177)
        vertices_62165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 30), path_62164, 'vertices')
        # Getting the type of 'self' (line 177)
        self_62166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'self')
        # Setting the type of the member 'shape_vertices' of a type (line 177)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 8), self_62166, 'shape_vertices', vertices_62165)
        
        # Assigning a BinOp to a Attribute (line 178):
        
        # Assigning a BinOp to a Attribute (line 178):
        
        # Call to ones(...): (line 178)
        # Processing the call arguments (line 178)
        
        # Call to len(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'self' (line 178)
        self_62170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 39), 'self', False)
        # Obtaining the member 'shape_vertices' of a type (line 178)
        shape_vertices_62171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 39), self_62170, 'shape_vertices')
        # Processing the call keyword arguments (line 178)
        kwargs_62172 = {}
        # Getting the type of 'len' (line 178)
        len_62169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 35), 'len', False)
        # Calling len(args, kwargs) (line 178)
        len_call_result_62173 = invoke(stypy.reporting.localization.Localization(__file__, 178, 35), len_62169, *[shape_vertices_62171], **kwargs_62172)
        
        # Processing the call keyword arguments (line 178)
        kwargs_62174 = {}
        # Getting the type of 'np' (line 178)
        np_62167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 27), 'np', False)
        # Obtaining the member 'ones' of a type (line 178)
        ones_62168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 27), np_62167, 'ones')
        # Calling ones(args, kwargs) (line 178)
        ones_call_result_62175 = invoke(stypy.reporting.localization.Localization(__file__, 178, 27), ones_62168, *[len_call_result_62173], **kwargs_62174)
        
        # Getting the type of 'Path' (line 178)
        Path_62176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 63), 'Path')
        # Obtaining the member 'LINETO' of a type (line 178)
        LINETO_62177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 63), Path_62176, 'LINETO')
        # Applying the binary operator '*' (line 178)
        result_mul_62178 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 27), '*', ones_call_result_62175, LINETO_62177)
        
        # Getting the type of 'self' (line 178)
        self_62179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'self')
        # Setting the type of the member 'shape_codes' of a type (line 178)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 8), self_62179, 'shape_codes', result_mul_62178)
        
        # Assigning a Attribute to a Subscript (line 179):
        
        # Assigning a Attribute to a Subscript (line 179):
        # Getting the type of 'Path' (line 179)
        Path_62180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 30), 'Path')
        # Obtaining the member 'MOVETO' of a type (line 179)
        MOVETO_62181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 30), Path_62180, 'MOVETO')
        # Getting the type of 'self' (line 179)
        self_62182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'self')
        # Obtaining the member 'shape_codes' of a type (line 179)
        shape_codes_62183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), self_62182, 'shape_codes')
        int_62184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 25), 'int')
        # Storing an element on a container (line 179)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 8), shape_codes_62183, (int_62184, MOVETO_62181))
        
        # Call to __init__(...): (line 180)
        # Processing the call arguments (line 180)
        # Getting the type of 'self' (line 180)
        self_62187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 24), 'self', False)
        # Getting the type of 'hatch' (line 180)
        hatch_62188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 30), 'hatch', False)
        # Getting the type of 'density' (line 180)
        density_62189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 37), 'density', False)
        # Processing the call keyword arguments (line 180)
        kwargs_62190 = {}
        # Getting the type of 'Shapes' (line 180)
        Shapes_62185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'Shapes', False)
        # Obtaining the member '__init__' of a type (line 180)
        init___62186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), Shapes_62185, '__init__')
        # Calling __init__(args, kwargs) (line 180)
        init___call_result_62191 = invoke(stypy.reporting.localization.Localization(__file__, 180, 8), init___62186, *[self_62187, hatch_62188, density_62189], **kwargs_62190)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Stars' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'Stars', Stars)

# Assigning a BinOp to a Name (line 171):
float_62192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 11), 'float')
float_62193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 17), 'float')
# Applying the binary operator 'div' (line 171)
result_div_62194 = python_operator(stypy.reporting.localization.Localization(__file__, 171, 11), 'div', float_62192, float_62193)

# Getting the type of 'Stars'
Stars_62195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Stars')
# Setting the type of the member 'size' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Stars_62195, 'size', result_div_62194)

# Assigning a Name to a Name (line 172):
# Getting the type of 'True' (line 172)
True_62196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 13), 'True')
# Getting the type of 'Stars'
Stars_62197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Stars')
# Setting the type of the member 'filled' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Stars_62197, 'filled', True_62196)

# Assigning a List to a Name (line 182):

# Assigning a List to a Name (line 182):

# Obtaining an instance of the builtin type 'list' (line 182)
list_62198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 15), 'list')
# Adding type elements to the builtin type 'list' instance (line 182)
# Adding element type (line 182)
# Getting the type of 'HorizontalHatch' (line 183)
HorizontalHatch_62199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'HorizontalHatch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 15), list_62198, HorizontalHatch_62199)
# Adding element type (line 182)
# Getting the type of 'VerticalHatch' (line 184)
VerticalHatch_62200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'VerticalHatch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 15), list_62198, VerticalHatch_62200)
# Adding element type (line 182)
# Getting the type of 'NorthEastHatch' (line 185)
NorthEastHatch_62201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'NorthEastHatch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 15), list_62198, NorthEastHatch_62201)
# Adding element type (line 182)
# Getting the type of 'SouthEastHatch' (line 186)
SouthEastHatch_62202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'SouthEastHatch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 15), list_62198, SouthEastHatch_62202)
# Adding element type (line 182)
# Getting the type of 'SmallCircles' (line 187)
SmallCircles_62203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'SmallCircles')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 15), list_62198, SmallCircles_62203)
# Adding element type (line 182)
# Getting the type of 'LargeCircles' (line 188)
LargeCircles_62204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'LargeCircles')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 15), list_62198, LargeCircles_62204)
# Adding element type (line 182)
# Getting the type of 'SmallFilledCircles' (line 189)
SmallFilledCircles_62205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'SmallFilledCircles')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 15), list_62198, SmallFilledCircles_62205)
# Adding element type (line 182)
# Getting the type of 'Stars' (line 190)
Stars_62206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'Stars')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 15), list_62198, Stars_62206)

# Assigning a type to the variable '_hatch_types' (line 182)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), '_hatch_types', list_62198)

@norecursion
def get_path(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_62207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 35), 'int')
    defaults = [int_62207]
    # Create a new context for function 'get_path'
    module_type_store = module_type_store.open_function_context('get_path', 194, 0, False)
    
    # Passed parameters checking function
    get_path.stypy_localization = localization
    get_path.stypy_type_of_self = None
    get_path.stypy_type_store = module_type_store
    get_path.stypy_function_name = 'get_path'
    get_path.stypy_param_names_list = ['hatchpattern', 'density']
    get_path.stypy_varargs_param_name = None
    get_path.stypy_kwargs_param_name = None
    get_path.stypy_call_defaults = defaults
    get_path.stypy_call_varargs = varargs
    get_path.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_path', ['hatchpattern', 'density'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_path', localization, ['hatchpattern', 'density'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_path(...)' code ##################

    unicode_62208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, (-1)), 'unicode', u'\n    Given a hatch specifier, *hatchpattern*, generates Path to render\n    the hatch in a unit square.  *density* is the number of lines per\n    unit square.\n    ')
    
    # Assigning a Call to a Name (line 200):
    
    # Assigning a Call to a Name (line 200):
    
    # Call to int(...): (line 200)
    # Processing the call arguments (line 200)
    # Getting the type of 'density' (line 200)
    density_62210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'density', False)
    # Processing the call keyword arguments (line 200)
    kwargs_62211 = {}
    # Getting the type of 'int' (line 200)
    int_62209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 14), 'int', False)
    # Calling int(args, kwargs) (line 200)
    int_call_result_62212 = invoke(stypy.reporting.localization.Localization(__file__, 200, 14), int_62209, *[density_62210], **kwargs_62211)
    
    # Assigning a type to the variable 'density' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'density', int_call_result_62212)
    
    # Assigning a ListComp to a Name (line 202):
    
    # Assigning a ListComp to a Name (line 202):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of '_hatch_types' (line 203)
    _hatch_types_62218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 34), '_hatch_types')
    comprehension_62219 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 16), _hatch_types_62218)
    # Assigning a type to the variable 'hatch_type' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'hatch_type', comprehension_62219)
    
    # Call to hatch_type(...): (line 202)
    # Processing the call arguments (line 202)
    # Getting the type of 'hatchpattern' (line 202)
    hatchpattern_62214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 27), 'hatchpattern', False)
    # Getting the type of 'density' (line 202)
    density_62215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 41), 'density', False)
    # Processing the call keyword arguments (line 202)
    kwargs_62216 = {}
    # Getting the type of 'hatch_type' (line 202)
    hatch_type_62213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 16), 'hatch_type', False)
    # Calling hatch_type(args, kwargs) (line 202)
    hatch_type_call_result_62217 = invoke(stypy.reporting.localization.Localization(__file__, 202, 16), hatch_type_62213, *[hatchpattern_62214, density_62215], **kwargs_62216)
    
    list_62220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 16), list_62220, hatch_type_call_result_62217)
    # Assigning a type to the variable 'patterns' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'patterns', list_62220)
    
    # Assigning a Call to a Name (line 204):
    
    # Assigning a Call to a Name (line 204):
    
    # Call to sum(...): (line 204)
    # Processing the call arguments (line 204)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'patterns' (line 204)
    patterns_62224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 60), 'patterns', False)
    comprehension_62225 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 24), patterns_62224)
    # Assigning a type to the variable 'pattern' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 24), 'pattern', comprehension_62225)
    # Getting the type of 'pattern' (line 204)
    pattern_62222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 24), 'pattern', False)
    # Obtaining the member 'num_vertices' of a type (line 204)
    num_vertices_62223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 24), pattern_62222, 'num_vertices')
    list_62226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 24), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 204, 24), list_62226, num_vertices_62223)
    # Processing the call keyword arguments (line 204)
    kwargs_62227 = {}
    # Getting the type of 'sum' (line 204)
    sum_62221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 19), 'sum', False)
    # Calling sum(args, kwargs) (line 204)
    sum_call_result_62228 = invoke(stypy.reporting.localization.Localization(__file__, 204, 19), sum_62221, *[list_62226], **kwargs_62227)
    
    # Assigning a type to the variable 'num_vertices' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'num_vertices', sum_call_result_62228)
    
    
    # Getting the type of 'num_vertices' (line 206)
    num_vertices_62229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 7), 'num_vertices')
    int_62230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, 23), 'int')
    # Applying the binary operator '==' (line 206)
    result_eq_62231 = python_operator(stypy.reporting.localization.Localization(__file__, 206, 7), '==', num_vertices_62229, int_62230)
    
    # Testing the type of an if condition (line 206)
    if_condition_62232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 206, 4), result_eq_62231)
    # Assigning a type to the variable 'if_condition_62232' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'if_condition_62232', if_condition_62232)
    # SSA begins for if statement (line 206)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to Path(...): (line 207)
    # Processing the call arguments (line 207)
    
    # Call to empty(...): (line 207)
    # Processing the call arguments (line 207)
    
    # Obtaining an instance of the builtin type 'tuple' (line 207)
    tuple_62236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 207)
    # Adding element type (line 207)
    int_62237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 30), tuple_62236, int_62237)
    # Adding element type (line 207)
    int_62238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 207, 30), tuple_62236, int_62238)
    
    # Processing the call keyword arguments (line 207)
    kwargs_62239 = {}
    # Getting the type of 'np' (line 207)
    np_62234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 20), 'np', False)
    # Obtaining the member 'empty' of a type (line 207)
    empty_62235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 20), np_62234, 'empty')
    # Calling empty(args, kwargs) (line 207)
    empty_call_result_62240 = invoke(stypy.reporting.localization.Localization(__file__, 207, 20), empty_62235, *[tuple_62236], **kwargs_62239)
    
    # Processing the call keyword arguments (line 207)
    kwargs_62241 = {}
    # Getting the type of 'Path' (line 207)
    Path_62233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 15), 'Path', False)
    # Calling Path(args, kwargs) (line 207)
    Path_call_result_62242 = invoke(stypy.reporting.localization.Localization(__file__, 207, 15), Path_62233, *[empty_call_result_62240], **kwargs_62241)
    
    # Assigning a type to the variable 'stypy_return_type' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'stypy_return_type', Path_call_result_62242)
    # SSA join for if statement (line 206)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to empty(...): (line 209)
    # Processing the call arguments (line 209)
    
    # Obtaining an instance of the builtin type 'tuple' (line 209)
    tuple_62245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 209)
    # Adding element type (line 209)
    # Getting the type of 'num_vertices' (line 209)
    num_vertices_62246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 25), 'num_vertices', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 25), tuple_62245, num_vertices_62246)
    # Adding element type (line 209)
    int_62247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 25), tuple_62245, int_62247)
    
    # Processing the call keyword arguments (line 209)
    kwargs_62248 = {}
    # Getting the type of 'np' (line 209)
    np_62243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'np', False)
    # Obtaining the member 'empty' of a type (line 209)
    empty_62244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 15), np_62243, 'empty')
    # Calling empty(args, kwargs) (line 209)
    empty_call_result_62249 = invoke(stypy.reporting.localization.Localization(__file__, 209, 15), empty_62244, *[tuple_62245], **kwargs_62248)
    
    # Assigning a type to the variable 'vertices' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'vertices', empty_call_result_62249)
    
    # Assigning a Call to a Name (line 210):
    
    # Assigning a Call to a Name (line 210):
    
    # Call to empty(...): (line 210)
    # Processing the call arguments (line 210)
    
    # Obtaining an instance of the builtin type 'tuple' (line 210)
    tuple_62252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 210)
    # Adding element type (line 210)
    # Getting the type of 'num_vertices' (line 210)
    num_vertices_62253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 22), 'num_vertices', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 22), tuple_62252, num_vertices_62253)
    
    # Getting the type of 'np' (line 210)
    np_62254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 38), 'np', False)
    # Obtaining the member 'uint8' of a type (line 210)
    uint8_62255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 38), np_62254, 'uint8')
    # Processing the call keyword arguments (line 210)
    kwargs_62256 = {}
    # Getting the type of 'np' (line 210)
    np_62250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 12), 'np', False)
    # Obtaining the member 'empty' of a type (line 210)
    empty_62251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 12), np_62250, 'empty')
    # Calling empty(args, kwargs) (line 210)
    empty_call_result_62257 = invoke(stypy.reporting.localization.Localization(__file__, 210, 12), empty_62251, *[tuple_62252, uint8_62255], **kwargs_62256)
    
    # Assigning a type to the variable 'codes' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'codes', empty_call_result_62257)
    
    # Assigning a Num to a Name (line 212):
    
    # Assigning a Num to a Name (line 212):
    int_62258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 13), 'int')
    # Assigning a type to the variable 'cursor' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'cursor', int_62258)
    
    # Getting the type of 'patterns' (line 213)
    patterns_62259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 19), 'patterns')
    # Testing the type of a for loop iterable (line 213)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 213, 4), patterns_62259)
    # Getting the type of the for loop variable (line 213)
    for_loop_var_62260 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 213, 4), patterns_62259)
    # Assigning a type to the variable 'pattern' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'pattern', for_loop_var_62260)
    # SSA begins for a for statement (line 213)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'pattern' (line 214)
    pattern_62261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'pattern')
    # Obtaining the member 'num_vertices' of a type (line 214)
    num_vertices_62262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 11), pattern_62261, 'num_vertices')
    int_62263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 35), 'int')
    # Applying the binary operator '!=' (line 214)
    result_ne_62264 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 11), '!=', num_vertices_62262, int_62263)
    
    # Testing the type of an if condition (line 214)
    if_condition_62265 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 8), result_ne_62264)
    # Assigning a type to the variable 'if_condition_62265' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'if_condition_62265', if_condition_62265)
    # SSA begins for if statement (line 214)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 215):
    
    # Assigning a Subscript to a Name (line 215):
    
    # Obtaining the type of the subscript
    # Getting the type of 'cursor' (line 215)
    cursor_62266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 38), 'cursor')
    # Getting the type of 'cursor' (line 215)
    cursor_62267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 45), 'cursor')
    # Getting the type of 'pattern' (line 215)
    pattern_62268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 54), 'pattern')
    # Obtaining the member 'num_vertices' of a type (line 215)
    num_vertices_62269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 54), pattern_62268, 'num_vertices')
    # Applying the binary operator '+' (line 215)
    result_add_62270 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 45), '+', cursor_62267, num_vertices_62269)
    
    slice_62271 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 215, 29), cursor_62266, result_add_62270, None)
    # Getting the type of 'vertices' (line 215)
    vertices_62272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 29), 'vertices')
    # Obtaining the member '__getitem__' of a type (line 215)
    getitem___62273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 29), vertices_62272, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 215)
    subscript_call_result_62274 = invoke(stypy.reporting.localization.Localization(__file__, 215, 29), getitem___62273, slice_62271)
    
    # Assigning a type to the variable 'vertices_chunk' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'vertices_chunk', subscript_call_result_62274)
    
    # Assigning a Subscript to a Name (line 216):
    
    # Assigning a Subscript to a Name (line 216):
    
    # Obtaining the type of the subscript
    # Getting the type of 'cursor' (line 216)
    cursor_62275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 32), 'cursor')
    # Getting the type of 'cursor' (line 216)
    cursor_62276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'cursor')
    # Getting the type of 'pattern' (line 216)
    pattern_62277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 48), 'pattern')
    # Obtaining the member 'num_vertices' of a type (line 216)
    num_vertices_62278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 48), pattern_62277, 'num_vertices')
    # Applying the binary operator '+' (line 216)
    result_add_62279 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 39), '+', cursor_62276, num_vertices_62278)
    
    slice_62280 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 216, 26), cursor_62275, result_add_62279, None)
    # Getting the type of 'codes' (line 216)
    codes_62281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 26), 'codes')
    # Obtaining the member '__getitem__' of a type (line 216)
    getitem___62282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 26), codes_62281, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 216)
    subscript_call_result_62283 = invoke(stypy.reporting.localization.Localization(__file__, 216, 26), getitem___62282, slice_62280)
    
    # Assigning a type to the variable 'codes_chunk' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'codes_chunk', subscript_call_result_62283)
    
    # Call to set_vertices_and_codes(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'vertices_chunk' (line 217)
    vertices_chunk_62286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 43), 'vertices_chunk', False)
    # Getting the type of 'codes_chunk' (line 217)
    codes_chunk_62287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 59), 'codes_chunk', False)
    # Processing the call keyword arguments (line 217)
    kwargs_62288 = {}
    # Getting the type of 'pattern' (line 217)
    pattern_62284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'pattern', False)
    # Obtaining the member 'set_vertices_and_codes' of a type (line 217)
    set_vertices_and_codes_62285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), pattern_62284, 'set_vertices_and_codes')
    # Calling set_vertices_and_codes(args, kwargs) (line 217)
    set_vertices_and_codes_call_result_62289 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), set_vertices_and_codes_62285, *[vertices_chunk_62286, codes_chunk_62287], **kwargs_62288)
    
    
    # Getting the type of 'cursor' (line 218)
    cursor_62290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'cursor')
    # Getting the type of 'pattern' (line 218)
    pattern_62291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 22), 'pattern')
    # Obtaining the member 'num_vertices' of a type (line 218)
    num_vertices_62292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 22), pattern_62291, 'num_vertices')
    # Applying the binary operator '+=' (line 218)
    result_iadd_62293 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 12), '+=', cursor_62290, num_vertices_62292)
    # Assigning a type to the variable 'cursor' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 12), 'cursor', result_iadd_62293)
    
    # SSA join for if statement (line 214)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to Path(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'vertices' (line 220)
    vertices_62295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'vertices', False)
    # Getting the type of 'codes' (line 220)
    codes_62296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 26), 'codes', False)
    # Processing the call keyword arguments (line 220)
    kwargs_62297 = {}
    # Getting the type of 'Path' (line 220)
    Path_62294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 11), 'Path', False)
    # Calling Path(args, kwargs) (line 220)
    Path_call_result_62298 = invoke(stypy.reporting.localization.Localization(__file__, 220, 11), Path_62294, *[vertices_62295, codes_62296], **kwargs_62297)
    
    # Assigning a type to the variable 'stypy_return_type' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'stypy_return_type', Path_call_result_62298)
    
    # ################# End of 'get_path(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_path' in the type store
    # Getting the type of 'stypy_return_type' (line 194)
    stypy_return_type_62299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_62299)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_path'
    return stypy_return_type_62299

# Assigning a type to the variable 'get_path' (line 194)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 0), 'get_path', get_path)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
